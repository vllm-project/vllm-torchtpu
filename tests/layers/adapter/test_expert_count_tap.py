# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU checks for the routed-expert count tap (layers/adapter/expert_count_tap.py)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import expert_count_tap, token_padding
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState

EXPERTS, TOPK = 8, 2


class _Layer(torch.nn.Module):
    def __init__(self, hash_table=None, ep_rank=3, sequence_parallel=False):
        super().__init__()
        self.layer_name = "model.layers.5.ffn.experts"
        self.global_num_experts = EXPERTS
        self.w13_weight = torch.zeros(1)
        self.moe_config = SimpleNamespace(
            experts_per_token=TOPK,
            moe_parallel_config=SimpleNamespace(is_sequence_parallel=sequence_parallel),
        )
        self.expert_map_manager = SimpleNamespace(ep_rank=ep_rank)
        self.renormalize = True
        self.scoring_func = "sqrtsoftplus"
        self.use_grouped_topk = False
        self.routed_scaling_factor = 1.0
        self.custom_routing_function = None
        self.e_score_correction_bias = None
        self.hash_indices_table = hash_table


@pytest.fixture
def tap_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MOE_EXPERT_COUNT_TAP_DIR", str(tmp_path))
    monkeypatch.setenv("MOE_EXPERT_COUNT_TAP_EVERY", "2")
    monkeypatch.setattr(expert_count_tap, "_TAPPED", [])
    monkeypatch.setattr(expert_count_tap, "_STATE", expert_count_tap._fresh_state())
    monkeypatch.setattr(expert_count_tap, "_tp_rank", lambda: 0)
    return tmp_path


def _tapped(layer):
    expert_count_tap.register(layer)
    return getattr(layer, expert_count_tap.COUNTS_ATTR)


def test_counts_squares_and_meta_skip_padding(tap_env):
    layer = _Layer()
    counts = _tapped(layer)
    assert counts.dtype == torch.float32
    logits = torch.full((4, EXPERTS), -5.0)
    logits[:, 3] = 5.0
    logits[:, 6] = 4.0
    logits[3, :] = 0.0  # the padding row selects 1 and another; it must not count
    logits[3, 1] = 9.0
    state = TokenPaddingState(local_padding_mask=torch.zeros(4, dtype=torch.bool))
    state.update(3, 4)
    token_padding.set_padding_state(state)
    try:
        expert_count_tap.record(layer, logits, None)
        expert_count_tap.record(layer, logits, None)
    finally:
        token_padding.set_padding_state(None)
    want = np.zeros(EXPERTS, np.int32)
    want[[3, 6]] = 3
    np.testing.assert_array_equal(counts[0].numpy(), 2 * want)
    np.testing.assert_array_equal(counts[1].numpy(), 2 * want * want)
    assert counts[2, :2].tolist() == [2, 6]


def test_hash_layers_count_the_table_selection(tap_env):
    table = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    layer = _Layer(hash_table=table)
    counts = _tapped(layer)
    expert_count_tap.record(layer, torch.zeros((3, EXPERTS)), torch.tensor([2, 2, 0]))
    assert counts[0].tolist() == [1, 1, 0, 0, 2, 2, 0, 0]


def test_dump_writes_cumulative_snapshots_every_n_steps(tap_env):
    layer = _Layer()
    counts = _tapped(layer)
    counts[0, 2] = 7
    for _ in range(5):
        expert_count_tap.maybe_dump()
    files = sorted(p.name for p in tap_env.iterdir())
    assert files == ["counts_ep3_0000.npz", "counts_ep3_0001.npz"]
    snap = np.load(tap_env / files[-1])
    assert snap["counts"].shape == (1, 3, EXPERTS)
    assert snap["counts"].dtype == np.int64
    assert snap["counts"][0, 0, 2] == 7
    assert snap["layer_names"].tolist() == [layer.layer_name]
    assert int(snap["step"]) == 4


def test_each_dump_drains_the_device_buffer_into_host_totals(tap_env):
    """The device only ever holds one interval, so its sums cannot outgrow
    their type; the snapshots stay cumulative and exact past 2**31."""
    layer = _Layer()
    counts = _tapped(layer)
    counts[1, 4] = 2.0**30
    expert_count_tap.maybe_dump()
    expert_count_tap.maybe_dump()  # snapshot 0
    assert not counts.any()
    counts[1, 4] = 2.0**30
    counts[0, 4] = 5
    expert_count_tap.maybe_dump()
    expert_count_tap.maybe_dump()  # snapshot 1
    snap = np.load(tap_env / "counts_ep3_0001.npz")
    assert snap["counts"][0, 1, 4] == 2**31
    assert snap["counts"][0, 0, 4] == 5


def test_a_restarted_run_numbers_after_the_old_snapshots(tap_env):
    (tap_env / "counts_ep3_0000.npz").write_bytes(b"from an earlier run")
    (tap_env / "counts_ep3_0004.npz").write_bytes(b"from an earlier run")
    _tapped(_Layer())
    for _ in range(2):
        expert_count_tap.maybe_dump()
    assert (tap_env / "counts_ep3_0004.npz").read_bytes() == b"from an earlier run"
    assert int(np.load(tap_env / "counts_ep3_0005.npz")["step"]) == 2


def test_a_tp_replica_does_not_count_its_copy_of_the_rows(tap_env, monkeypatch):
    monkeypatch.setattr(expert_count_tap, "_tp_rank", lambda: 1)
    replica = _Layer(ep_rank=4)
    expert_count_tap.register(replica)
    assert not hasattr(replica, expert_count_tap.COUNTS_ATTR)
    # Sequence parallelism partitions the rows instead, so every rank counts.
    partitioned = _Layer(ep_rank=5, sequence_parallel=True)
    assert _tapped(partitioned) is not None


def test_a_zero_cadence_fails_at_load(tap_env, monkeypatch):
    monkeypatch.setenv("MOE_EXPERT_COUNT_TAP_EVERY", "0")
    with pytest.raises(ValueError, match="MOE_EXPERT_COUNT_TAP_EVERY"):
        expert_count_tap.register(_Layer())


def test_the_directory_expands_environment_variables(tap_env, monkeypatch):
    monkeypatch.setenv("QH_TAP_ROOT", str(tap_env))
    _tapped(_Layer(ep_rank=0))
    monkeypatch.setenv("MOE_EXPERT_COUNT_TAP_DIR", "$QH_TAP_ROOT/sub")
    for _ in range(2):
        expert_count_tap.maybe_dump()
    assert (tap_env / "sub" / "counts_ep0_0000.npz").exists()
