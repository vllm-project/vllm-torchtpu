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

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu import expert_placement
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
    # (rank, size) in DP and in PCP: one rank of each.
    monkeypatch.setattr(expert_count_tap, "_dp_pcp", lambda: ((0, 1), (0, 1)))
    return tmp_path


def _tapped(layer):
    expert_count_tap.register(layer)
    return getattr(layer, expert_count_tap.COUNTS_ATTR)


@contextlib.contextmanager
def _padding(valid, padded):
    """This rank's rows [valid, padded) are padding for the next forward."""
    state = TokenPaddingState(local_padding_mask=torch.zeros(padded, dtype=torch.bool))
    state.update(valid, padded)
    token_padding.set_padding_state(state)
    try:
        yield
    finally:
        token_padding.set_padding_state(None)


def _selecting(*experts):
    """Router logits whose row i selects experts[i] and expert 0."""
    logits = torch.full((len(experts), EXPERTS), -5.0)
    logits[:, 0] = 4.0
    for row, expert in enumerate(experts):
        logits[row, expert] = 5.0
    return logits


def test_counts_squares_and_meta_skip_padding(tap_env):
    layer = _Layer()
    counts = _tapped(layer)
    assert counts.dtype == torch.float32
    logits = torch.full((4, EXPERTS), -5.0)
    logits[:, 3] = 5.0
    logits[:, 6] = 4.0
    logits[3, :] = 0.0  # the padding row selects 1 and another; it must not count
    logits[3, 1] = 9.0
    with _padding(valid=3, padded=4):
        expert_count_tap.record(layer, logits, None)
        expert_count_tap.record(layer, logits, None)
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


def test_a_placed_layer_is_counted_by_checkpoint_expert(tap_env):
    """Its router selects slots; the snapshot names the experts in them."""
    layer = _Layer()
    order = (7, 0, 5, 2, 1, 6, 3, 4)  # slot s holds expert order[s]
    setattr(layer, expert_placement.EXPERT_ORDER_ATTR, order)
    _tapped(layer)
    # Slots 3 and 3 and 6, each with slot 0: experts 2, 2, 3, each with 7.
    expert_count_tap.record(layer, _selecting(3, 3, 6), None)
    for _ in range(2):
        expert_count_tap.maybe_dump()
    snap = np.load(tap_env / "counts_ep3_0000.npz")["counts"][0]
    assert snap[0].tolist() == [0, 0, 2, 1, 0, 0, 0, 3]
    assert snap[1].tolist() == [0, 0, 4, 1, 0, 0, 0, 9]
    assert snap[2, :2].tolist() == [1, 3]


@pytest.mark.parametrize(
    "dp, pcp, over_pcp, mine",
    [
        ((1, 2), (0, 1), False, [2, 3]),  # DP rank 1 of 2: the second half
        ((0, 2), (1, 2), True, [4, 5]),  # PCP outside DP: share 1 * 2 + 0
        ((1, 2), (1, 2), True, [6, 7]),
    ],
)
def test_gathered_rows_count_only_this_ranks_share(
    tap_env, monkeypatch, dp, pcp, over_pcp, mine
):
    monkeypatch.setattr(expert_count_tap, "_dp_pcp", lambda: (dp, pcp))
    layer = _Layer()
    counts = _tapped(layer)
    rows = 2 * dp[1] * (pcp[1] if over_pcp else 1)
    experts = [1 + row % (EXPERTS - 1) for row in range(rows)]
    # This rank's second row is padding; other ranks' rows are not its own.
    with _padding(valid=1, padded=2):
        expert_count_tap.record(
            layer, _selecting(*experts), None, over_dp=True, over_pcp=over_pcp
        )
    want = np.zeros(EXPERTS, np.int64)
    want[0] = 1
    want[experts[mine[0]]] += 1
    np.testing.assert_array_equal(counts[0].numpy(), want)
    assert counts[2, :2].tolist() == [1, 1]


def test_sequence_parallel_rows_gathered_by_the_runner_are_refused(
    tap_env, monkeypatch
):
    monkeypatch.setattr(expert_count_tap, "_dp_pcp", lambda: ((0, 2), (0, 1)))
    layer = _Layer(sequence_parallel=True)
    _tapped(layer)
    with pytest.raises(NotImplementedError, match="sequence-parallel"):
        expert_count_tap.record(layer, _selecting(1, 2), None, over_dp=True)
