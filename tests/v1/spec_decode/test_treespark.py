# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.spec_decode.dspark.tree import (
    DraftTree,
    expand_tree,
    verify_tree,
)


def test_tree_verifies_a_noncontiguous_path():
    tree = DraftTree(tokens=[1, 2, 3, 4], parents=[0, 0, 1, 2], depths=[1, 1, 2, 2])
    assert tree.attention_mask(4) == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, False, True, False, False],
        [True, True, False, True, False],
        [True, False, True, False, True],
    ]
    logits = torch.zeros(5, 6)
    logits[0, 2] = logits[2, 4] = logits[4, 5] = 1.0
    path, output = verify_tree(tree, 4, 0, logits, 0.0, torch.Generator())
    assert path == [2, 4]
    assert output == [2, 4, 5]


def test_stochastic_tree_replays_without_replacement_after_rejection():
    # p always emits token 1. If the first proposal is 0, its rejection
    # must leave token 1 as the next proposal's entire support.
    def conditional(depth: int, parent: int) -> torch.Tensor:
        return torch.zeros(2)

    tree = DraftTree(
        tokens=[0, 1], parents=[0, 0], depths=[1, 1], conditional=conditional
    )
    logits = torch.tensor([[-torch.inf, 0.0], [0.0, -torch.inf], [0.0, -torch.inf]])
    path, output = verify_tree(tree, 2, 0, logits, 1.0, torch.Generator())
    assert path == [2]
    assert output == [1, 0]
    path, output = verify_tree(tree, 1, 0, logits, 1.0, torch.Generator())
    assert path == []
    assert output == [1]


def test_stochastic_tree_admission_uses_the_pre_draw_bound():
    def conditional(depth: int, parent: int) -> torch.Tensor:
        return torch.tensor([0.1, 0.0])

    # The calibrated distribution is the identity with a=1 and b=0.
    # Both tokens can be drawn first, but admission uses max q > 0.5.
    first_tokens = set()
    for seed in range(16):
        tree = expand_tree(
            0,
            conditional,
            1.0,
            0.5,
            1,
            2,
            2,
            1.0,
            0.0,
            torch.Generator().manual_seed(seed),
        )
        assert tree.tokens
        first_tokens.add(tree.tokens[0])
    assert first_tokens == {0, 1}
