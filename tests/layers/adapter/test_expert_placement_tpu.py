# SPDX-License-Identifier: Apache-2.0
"""TPU test for TPU_MOE_EXPERT_PLACEMENT's id lookup in a compiled graph."""

from types import SimpleNamespace

import numpy as np
import torch

from vllm_torchtpu.layers.adapter import expert_placement
from vllm_torchtpu.layers.core import expert_placement as core


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
        return expert_placement.to_slots(layer, topk_ids)

    got = place(ids.to("tpu")).cpu().numpy()
    want = np.where(ids.numpy() >= 0, slots[ids.numpy().clip(0)], -1)
    np.testing.assert_array_equal(got, want)
