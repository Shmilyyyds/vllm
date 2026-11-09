# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.tree import (
    DraftTree,
    expand_tree,
    verify_tree,
)


def _generator(
    device: torch.device, seed: int, position: int, *, drafting: bool
) -> torch.Generator:
    # Use separate streams for proposals and verification for each position.
    key = (seed + (2 * position + drafting) * 0x9E3779B97F4A7C15) % (1 << 64)
    return torch.Generator(device=device).manual_seed(key)


@dataclass
class TreeSparkBatch:
    trees: dict[int, DraftTree]
    # Query offset of the anchor within each request, or -1 for ordinary rows.
    anchor_offsets: torch.Tensor
    visibility: torch.Tensor
    positions: torch.Tensor
    logit_depths: torch.Tensor
    draft_path_rows: torch.Tensor
    accepted_src: list[int] = field(default_factory=list)
    accepted_dst: list[int] = field(default_factory=list)

    def verify(
        self,
        logits: torch.Tensor,
        batch: InputBatch,
        sampler: Sampler,
        max_depth: int,
    ) -> SamplerOutput:
        states = sampler.sampling_states
        logits_pos = self.positions[batch.logits_indices]
        logits_inputs = batch.input_ids[batch.logits_indices]
        processed = sampler.apply_sampling_params(
            logits,
            batch.expanded_idx_mapping,
            batch.idx_mapping,
            batch.idx_mapping_np,
            logits_pos,
            logits_inputs,
            self.logit_depths,
            batch.seq_lens_cpu_upper_bound.numpy(),
            draft_path_rows=self.draft_path_rows,
        )
        # Use the native target sampler for ordinary decode and prefill rows.
        roots = batch.cu_num_logits[:-1].long()
        root_samples = gumbel_sample(
            processed[roots],
            batch.idx_mapping,
            states.temperature.gpu,
            states.seeds.gpu,
            logits_pos[roots],
            apply_temperature=False,
            is_drafting=False,
            use_fp64=sampler.use_fp64_gumbel,
        ).tolist()
        root_positions = logits_pos[roots].tolist()
        anchors = logits_inputs[roots].tolist()
        sampled = []
        lengths = []
        selected_rows = []
        selected_ids = []
        logprob_offsets = [0]
        for req, slot in enumerate(batch.idx_mapping_np):
            start, end = batch.cu_num_logits_np[req : req + 2].tolist()
            tree = self.trees.get(req)
            path: list[int] = []
            if tree is None:
                tokens = [root_samples[req]]
            else:
                path, tokens = verify_tree(
                    tree,
                    end - start - 1,
                    anchors[req],
                    processed[start:end],
                    float(states.temperature.np[slot]),
                    _generator(
                        logits.device,
                        int(states.seeds.np[slot]),
                        root_positions[req],
                        drafting=False,
                    ),
                )
                query_end = int(batch.query_start_loc_np[req + 1])
                anchor_row = query_end - (end - start)
                self.accepted_src.extend(anchor_row + node for node in path)
                self.accepted_dst.extend(
                    range(anchor_row + 1, anchor_row + 1 + len(path))
                )
            lengths.append(len(tokens))
            sampled.append(tokens + [-1] * (max_depth + 1 - len(tokens)))
            selected_rows.extend(start + row for row in [0, *path])
            selected_ids.extend(tokens)
            logprob_offsets.append(len(selected_rows))

        sampled_tensor = torch.tensor(sampled, device=logits.device)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            torch.tensor(lengths, dtype=torch.int32, device=logits.device),
            batch.seq_lens,
            batch.cu_num_logits,
            batch.idx_mapping,
            sampler.req_states.prefill_len.gpu,
        )
        logprobs = None
        dims = sampler.get_logprobs_dims(batch.idx_mapping_np)
        if dims is not None:
            num_logprobs, max_per_req_ids = dims
            rows = torch.tensor(selected_rows, device=logits.device)
            source = (
                processed
                if sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
                else logits
            )
            logprobs = compute_topk_scores(
                source[rows],
                num_logprobs,
                torch.tensor(selected_ids, device=logits.device),
                logprob_offsets,
                logprob_token_ids_state=sampler.logprob_token_ids_state,
                expanded_idx_mapping=batch.expanded_idx_mapping[rows],
                max_per_req_token_ids=max_per_req_ids,
                logits_mode=sampler.logprobs_mode in ("raw_logits", "processed_logits"),
            )
        num_nans = None
        if sampler.compute_nans:
            cumulative_nans = get_num_nans(logits).cumsum(dim=0)
            cumulative_nans = torch.cat((cumulative_nans.new_zeros(1), cumulative_nans))
            num_nans = cumulative_nans[batch.cu_num_logits.long()].diff()
        return SamplerOutput(
            sampled_token_ids=sampled_tensor,
            logprobs_tensors=logprobs,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def commit(
        self,
        hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        caches: dict[str, torch.Tensor],
        slot_mappings: dict[str, torch.Tensor],
    ) -> None:
        if not self.accepted_src:
            return
        src = torch.tensor(self.accepted_src, device=hidden_states.device)
        dst = torch.tensor(self.accepted_dst, device=hidden_states.device)
        # Indexing materializes all sources before any destinations are written.
        for hidden in [hidden_states, *(aux_hidden_states or [])]:
            hidden.index_copy_(0, dst, hidden.index_select(0, src))
        copied = set()
        for name, cache in caches.items():
            if cache.data_ptr() in copied:
                continue
            copied.add(cache.data_ptr())
            slots = slot_mappings[name]
            # FlexAttention stores (blocks, heads, block size, 2 * head size).
            block_size = cache.shape[2]
            src_slots, dst_slots = slots[src], slots[dst]
            cache[dst_slots // block_size, :, dst_slots % block_size] = cache[
                src_slots // block_size, :, src_slots % block_size
            ]


class TreeSparkSpeculator(DSparkSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        assert self.speculative_config.treespark is not None
        self.tree_config = self.speculative_config.treespark
        # TreeSpark saves base logits, so native per-position draft logits are
        # unnecessary even when the target uses stochastic sampling.
        self.use_draft_logits = False
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        self.trees: dict[str, DraftTree] = {}
        self.latest_drafts: DraftTokenIds | None = None
        self._base_logits: torch.Tensor | None = None

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        indices = self.sample_indices[: num_reqs * self.num_speculative_steps]
        self._base_logits = self.model.compute_draft_logits(hidden[indices]).view(
            num_reqs, self.num_speculative_steps, -1
        )

    def _conditional(self, base: torch.Tensor, depth: int, token: int) -> torch.Tensor:
        parent = torch.tensor([token], device=base.device)
        logits = base[depth : depth + 1] + self.model.markov_bias(
            self.model.markov_embed(parent)
        )
        if hasattr(self.model, "apply_output_logit_transform"):
            logits = self.model.apply_output_logit_transform(logits)
        if self.model.draft_id_to_target_id is not None:
            ids = self.model.map_draft_to_target(
                torch.arange(logits.shape[-1], device=logits.device)
            )
            full = logits.new_full((self.target_vocab_size,), -torch.inf)
            return full.scatter_(0, ids.long(), logits[0])
        return logits[0]

    def publish(
        self, batch: InputBatch, sampler: Sampler, num_sampled: torch.Tensor
    ) -> torch.Tensor:
        config = self.tree_config
        result = torch.zeros(
            (batch.num_reqs, config.max_nodes), dtype=torch.int64, device=self.device
        )
        drafts: list[list[int]] = []
        base_logits, self._base_logits = self._base_logits, None
        sampled = num_sampled.tolist()
        anchors = self.input_buffers.input_ids[self._anchor_idx[: batch.num_reqs]]
        anchors = anchors.tolist()
        sample_pos = self.sample_pos[: batch.num_reqs * self.num_speculative_steps]
        sample_pos = sample_pos[:: self.num_speculative_steps].tolist()
        load = sum(n > 0 for n in sampled)
        for i, (req_id, slot) in enumerate(
            zip(batch.req_ids, batch.idx_mapping_np, strict=True)
        ):
            self.trees.pop(req_id, None)
            tree = DraftTree()
            if sampled[i] and base_logits is not None:
                temperature = float(sampler.sampling_states.temperature.np[slot])
                # Retained stochastic proposals must not pin other requests'
                # logits from the same batch.
                base = base_logits[i].clone() if temperature else base_logits[i]
                max_children, theta = config.policy(load, temperature)

                def conditional(
                    depth: int, token: int, base: torch.Tensor = base
                ) -> torch.Tensor:
                    return self._conditional(base, depth, token)

                tree = expand_tree(
                    anchors[i],
                    conditional,
                    temperature,
                    theta,
                    self.num_speculative_steps,
                    config.max_nodes,
                    max_children,
                    config.calibration_a,
                    config.calibration_b,
                    _generator(
                        self.device,
                        int(sampler.sampling_states.seeds.np[slot]),
                        sample_pos[i],
                        drafting=True,
                    ),
                )
            if tree.tokens:
                self.trees[req_id] = tree
                result[i, : len(tree.tokens)] = torch.tensor(
                    tree.tokens, device=self.device
                )
            drafts.append(tree.tokens)
        self.latest_drafts = DraftTokenIds(batch.req_ids, drafts)
        return result

    def prepare_tree_batch(
        self, batch: InputBatch, scheduled: dict[str, list[int]]
    ) -> TreeSparkBatch | None:
        if not scheduled:
            return None
        width = max(map(len, scheduled.values())) + 1
        masks = torch.zeros((batch.num_reqs, width, width), dtype=torch.bool)
        offsets = [-1] * batch.num_reqs
        trees = {}
        positions = batch.positions.clone()
        num_logits = len(batch.logits_indices)
        depths = torch.zeros(num_logits, dtype=torch.int32)
        paths = torch.zeros(
            (num_logits, self.num_speculative_steps), dtype=torch.int64
        )
        for i, req_id in enumerate(batch.req_ids):
            tokens = scheduled.get(req_id)
            if not tokens:
                continue
            tree = self.trees.get(req_id)
            count = len(tokens)
            if tree is None or tokens != tree.tokens[:count]:
                raise RuntimeError(
                    "Scheduled TreeSpark nodes do not match the proposal"
                )
            end = int(batch.query_start_loc_np[i + 1])
            anchor = end - count - 1
            offsets[i] = anchor - int(batch.query_start_loc_np[i])
            positions[anchor + 1 : end] = positions[anchor] + torch.tensor(
                tree.depths[:count], device=positions.device
            )
            visibility = tree.attention_mask(count)
            masks[i, : count + 1, : count + 1] = torch.tensor(visibility)
            start = int(batch.cu_num_logits_np[i])
            depths[start + 1 : start + count + 1] = torch.tensor(
                tree.depths[:count]
            )
            for node in range(1, count + 1):
                path_rows = [
                    start + j for j in range(1, node + 1) if visibility[node][j]
                ]
                paths[start + node, : len(path_rows)] = torch.tensor(path_rows)
            trees[i] = tree
        if not trees:
            return None
        return TreeSparkBatch(
            trees,
            torch.tensor(offsets, device=positions.device),
            masks.to(positions.device),
            positions,
            depths.to(positions.device),
            paths.to(positions.device),
        )
