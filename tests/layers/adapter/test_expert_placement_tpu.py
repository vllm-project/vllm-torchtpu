# SPDX-License-Identifier: Apache-2.0
"""TPU tests for TPU_MOE_EXPERT_PLACEMENT's id lookup and
TPU_MOE_ROUTING_COUNTS_DIR's routing counts in a compiled graph."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import expert_placement, token_padding
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


@pytest.mark.parametrize("dp_size,dp_rank,local", [(1, 0, True), (2, 1, False)])
def test_count_routes_compiled_on_tpu_sums_real_rows(
    monkeypatch, dp_size, dp_rank, local
):
    rows, experts = 96, 512
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(dp_size=dp_size)
    )
    layer.quant_method = SimpleNamespace(supports_internal_mk=local)
    counts = torch.zeros(experts, dtype=torch.int32, device="tpu")
    layer.register_buffer("_tpu_routing_counts", counts, persistent=False)
    if dp_size > 1:
        rank = torch.tensor(dp_rank, dtype=torch.int32, device="tpu")
        layer.register_buffer("_tpu_routing_counts_dp_rank", rank, persistent=False)
    state = token_padding.TokenPaddingState.create(rows, torch.device("tpu"))
    monkeypatch.setattr(token_padding, "_padding_state", state)

    # One static program per input size, as the TPU backend has no dynamic shapes.
    @torch.compile(fullgraph=True, dynamic=False)
    def count(topk_ids):
        expert_placement.count_routes(layer, topk_ids)

    rng = np.random.default_rng(0)
    want = np.zeros(experts, dtype=np.int64)
    for valid in (rows, 37):
        state.update(valid, rows)
        ids = rng.integers(0, experts, size=(dp_size * rows, 10), dtype=np.int32)
        count(torch.from_numpy(ids).to("tpu"))
        mine = ids[dp_rank * rows : dp_rank * rows + valid]
        want += np.bincount(mine.ravel(), minlength=experts)
    np.testing.assert_array_equal(layer._tpu_routing_counts.cpu().numpy(), want)
