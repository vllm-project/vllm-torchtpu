# SPDX-License-Identifier: Apache-2.0
"""TPU test for TPU_MOE_EXPERT_PLACEMENT's id lookup in a compiled graph."""

from types import SimpleNamespace

import numpy as np
import torch

from vllm_torchtpu.layers.adapter import expert_permutation
from vllm_torchtpu.layers.core import expert_permutation as core


def test_to_slots_compiled_on_tpu_matches_table():
    rng = np.random.default_rng(0)
    slots = core.balance(rng.gamma(2.0, 10.0, size=512), 32)
    layer = SimpleNamespace(_expert_slots=torch.from_numpy(slots).to("tpu"))
    ids = torch.from_numpy(
        np.stack([rng.choice(512, 10, replace=False) for _ in range(96)])
    ).to(torch.int32)
    ids[5, 2] = -1

    @torch.compile(fullgraph=True)
    def place(topk_ids):
        return expert_permutation.to_slots(layer, topk_ids)

    got = place(ids.to("tpu")).cpu().numpy()
    want = np.where(ids.numpy() >= 0, slots[ids.numpy().clip(0)], -1)
    np.testing.assert_array_equal(got, want)


def test_copy_slots_compiled_on_tpu_match_table():
    rng = np.random.default_rng(1)
    counts = rng.gamma(2.0, 10.0, size=512)
    counts[[7, 300]] = counts.sum() / 32 * np.array([3.0, 1.5])
    slots = core.replicate(counts, 32, 1)  # copies up to slot 543 (> 256)
    copies = (slots >= 0).sum(axis=1)
    table = np.concatenate(
        [copies[:, None], np.maximum(slots, 0) % 256, np.maximum(slots, 0) // 256],
        axis=1,
    )
    layer = SimpleNamespace(
        _expert_copy_digits=torch.from_numpy(table.astype(np.float32)).to("tpu")
    )
    ids = np.stack([rng.choice(512, 10, replace=False) for _ in range(96)])
    ids[:, 0] = 7
    ids[5, 2] = -1
    weights = torch.from_numpy(rng.dirichlet(np.ones(10), 96).astype(np.float32))
    weights = weights.to(torch.bfloat16)

    @torch.compile(fullgraph=True)
    def place(topk_ids, topk_weights):
        return expert_permutation.to_slots(layer, topk_ids, topk_weights)

    got = (
        place(torch.from_numpy(ids).to(torch.int32).to("tpu"), weights.to("tpu"))
        .cpu()
        .numpy()
    )
    bits = (weights.float().numpy().view(np.int32) >> 16) & 0xFFFF
    safe = ids.clip(0)
    want = np.where(ids >= 0, slots[safe, bits % copies[safe]], -1)
    np.testing.assert_array_equal(got, want)
