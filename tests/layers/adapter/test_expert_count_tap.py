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
from unittest.mock import patch

import numpy as np
import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe.vllm_adapter import (
    FUSED_MOE_EP_OP_ATTR,
)
from vllm_torchtpu.layers.adapter import expert_count_tap, token_padding
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState

EXPERTS, TOPK = 8, 2
# A step of at most this many valid tokens counts as decode.
DECODE_MAX = 4
DECODE, PREFILL = 0, 1


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
        self.quant_method = SimpleNamespace(supports_internal_mk=False)


@pytest.fixture
def tap_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TPU_MOE_EXPERT_COUNT_TAP_DIR", str(tmp_path))
    monkeypatch.setenv("TPU_MOE_EXPERT_COUNT_TAP_EVERY", "2")
    monkeypatch.setattr(expert_count_tap, "_TAPPED", [])
    monkeypatch.setattr(expert_count_tap, "_STATE", expert_count_tap._fresh_state())
    monkeypatch.setattr(expert_count_tap, "_tp_rank", lambda: 0)
    monkeypatch.setattr(expert_count_tap, "_decode_max_tokens", lambda: DECODE_MAX)
    # (rank, size) in DP and in PCP: one rank of each.
    monkeypatch.setattr(expert_count_tap, "_dp_pcp", lambda: ((0, 1), (0, 1)))
    monkeypatch.setattr(expert_count_tap, "_global_rank", lambda: 0)
    # The runner keeps a padding mask whenever the tap is on; no rows padded.
    with _padding(valid=64, padded=64):
        yield tmp_path


def _record(layer, logits, input_ids=None, **kwargs):
    expert_count_tap.record(layer, logits, logits, input_ids, **kwargs)


def _tapped(layer):
    expert_count_tap.register(layer)
    return getattr(layer, expert_count_tap.COUNTS_ATTR)


@contextlib.contextmanager
def _padding(valid, padded):
    """This rank's rows [valid, padded) are padding for the next forward."""
    state = TokenPaddingState(local_padding_mask=torch.zeros(padded, dtype=torch.bool))
    state.update(valid, padded)
    previous = token_padding._padding_state
    token_padding.set_padding_state(state)
    try:
        yield
    finally:
        token_padding.set_padding_state(previous)


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
        _record(layer, logits)
        _record(layer, logits)
    want = np.zeros(EXPERTS, np.int32)
    want[[3, 6]] = 3
    np.testing.assert_array_equal(counts[DECODE, 0].numpy(), 2 * want)
    np.testing.assert_array_equal(counts[DECODE, 1].numpy(), 2 * want * want)
    assert counts[DECODE, 2, :2].tolist() == [2, 6]
    assert not counts[PREFILL].any()


def test_hash_layers_count_the_table_selection(tap_env):
    table = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    layer = _Layer(hash_table=table)
    counts = _tapped(layer)
    _record(layer, torch.zeros((3, EXPERTS)), torch.tensor([2, 2, 0]))
    assert counts[DECODE, 0].tolist() == [1, 1, 0, 0, 2, 2, 0, 0]


def test_dump_writes_cumulative_snapshots_every_n_steps(tap_env):
    layer = _Layer()
    counts = _tapped(layer)
    counts[DECODE, 0, 2] = 7
    counts[PREFILL, 0, 2] = 4
    for _ in range(5):
        expert_count_tap.maybe_dump()
    files = sorted(p.name for p in tap_env.iterdir())
    assert files == ["counts_rank0_ep3_0000.npz", "counts_rank0_ep3_0001.npz"]
    snap = np.load(tap_env / files[-1])
    assert snap["counts"].shape == (1, 3, EXPERTS)
    assert snap["counts"].dtype == np.int64
    assert snap["counts"][0, 0, 2] == 11
    assert snap["counts_by_phase"].shape == (1, 2, 3, EXPERTS)
    assert snap["counts_by_phase"][0, :, 0, 2].tolist() == [7, 4]
    assert snap["phases"].tolist() == ["decode", "prefill"]
    assert int(snap["decode_max_tokens"]) == DECODE_MAX
    assert snap["layer_names"].tolist() == [layer.layer_name]
    assert int(snap["step"]) == 4


def test_each_dump_drains_the_device_buffer_into_host_totals(tap_env):
    """The device only ever holds one interval, so its sums cannot outgrow
    their type; the snapshots stay cumulative and exact past 2**31."""
    layer = _Layer()
    counts = _tapped(layer)
    counts[DECODE, 1, 4] = 2.0**30
    expert_count_tap.maybe_dump()
    expert_count_tap.maybe_dump()  # snapshot 0
    assert not counts.any()
    counts[DECODE, 1, 4] = 2.0**30
    counts[DECODE, 0, 4] = 5
    expert_count_tap.maybe_dump()
    expert_count_tap.maybe_dump()  # snapshot 1
    snap = np.load(tap_env / "counts_rank0_ep3_0001.npz")
    assert snap["counts"][0, 1, 4] == 2**31
    assert snap["counts"][0, 0, 4] == 5


def test_a_restarted_run_numbers_after_the_old_snapshots(tap_env):
    (tap_env / "counts_rank0_ep3_0000.npz").write_bytes(b"from an earlier run")
    (tap_env / "counts_rank0_ep3_0004.npz").write_bytes(b"from an earlier run")
    _tapped(_Layer())
    for _ in range(2):
        expert_count_tap.maybe_dump()
    old = (tap_env / "counts_rank0_ep3_0004.npz").read_bytes()
    assert old == b"from an earlier run"
    assert int(np.load(tap_env / "counts_rank0_ep3_0005.npz")["step"]) == 2


def test_a_tp_replica_does_not_count_its_copy_of_the_rows(tap_env, monkeypatch):
    monkeypatch.setattr(expert_count_tap, "_tp_rank", lambda: 1)
    replica = _Layer(ep_rank=4)
    expert_count_tap.register(replica)
    assert not hasattr(replica, expert_count_tap.COUNTS_ATTR)
    # Sequence parallelism partitions the rows instead, so every rank counts.
    partitioned = _Layer(ep_rank=5, sequence_parallel=True)
    assert _tapped(partitioned) is not None


def test_a_zero_cadence_fails_at_load(tap_env, monkeypatch):
    monkeypatch.setenv("TPU_MOE_EXPERT_COUNT_TAP_EVERY", "0")
    with pytest.raises(ValueError, match="TPU_MOE_EXPERT_COUNT_TAP_EVERY"):
        expert_count_tap.register(_Layer())


def test_the_directory_expands_environment_variables(tap_env, monkeypatch):
    monkeypatch.setenv("QH_TAP_ROOT", str(tap_env))
    _tapped(_Layer(ep_rank=0))
    monkeypatch.setenv("TPU_MOE_EXPERT_COUNT_TAP_DIR", "$QH_TAP_ROOT/sub")
    for _ in range(2):
        expert_count_tap.maybe_dump()
    assert (tap_env / "sub" / "counts_rank0_ep0_0000.npz").exists()


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
        _record(layer, _selecting(*experts), over_dp=True, over_pcp=over_pcp)
    want = np.zeros(EXPERTS, np.int64)
    want[0] = 1
    want[experts[mine[0]]] += 1
    np.testing.assert_array_equal(counts[DECODE, 0].numpy(), want)
    assert counts[DECODE, 2, :2].tolist() == [1, 1]


def test_sequence_parallel_rows_gathered_by_the_runner_are_refused(
    tap_env, monkeypatch
):
    monkeypatch.setattr(expert_count_tap, "_dp_pcp", lambda: ((0, 2), (0, 1)))
    layer = _Layer(sequence_parallel=True)
    _tapped(layer)
    with pytest.raises(NotImplementedError, match="sequence-parallel"):
        _record(layer, _selecting(1, 2), over_dp=True)


def test_steps_are_split_by_valid_tokens_into_decode_and_prefill(tap_env):
    """A step whose valid tokens fit a pure decode step lands in phase 0;
    a larger one in phase 1. Padding does not make a step a prefill."""
    layer = _Layer()
    counts = _tapped(layer)
    with _padding(valid=DECODE_MAX, padded=DECODE_MAX + 2):
        _record(layer, _selecting(*[1] * (DECODE_MAX + 2)))
    _record(layer, _selecting(*[2] * (DECODE_MAX + 1)))
    assert counts[DECODE, 0].tolist() == [DECODE_MAX, DECODE_MAX, 0, 0, 0, 0, 0, 0]
    assert counts[DECODE, 2, :2].tolist() == [1, DECODE_MAX]
    prefill = DECODE_MAX + 1
    assert counts[PREFILL, 0].tolist() == [prefill, 0, prefill, 0, 0, 0, 0, 0]
    assert counts[PREFILL, 1, 2] == prefill * prefill
    assert counts[PREFILL, 2, :2].tolist() == [1, prefill]


def test_a_forward_of_only_padding_is_not_a_step(tap_env):
    layer = _Layer()
    counts = _tapped(layer)
    with _padding(valid=0, padded=4):
        _record(layer, _selecting(1, 2, 3, 4))
    assert not counts.any()


def test_a_forward_without_a_padding_mask_is_refused(tap_env):
    """Without the runner's mask every bucket-padding row would count."""
    layer = _Layer()
    _tapped(layer)
    previous = token_padding._padding_state
    token_padding.set_padding_state(None)
    try:
        with pytest.raises(RuntimeError, match="padding mask"):
            _record(layer, _selecting(1, 2))
    finally:
        token_padding.set_padding_state(previous)


def test_a_custom_routing_function_is_what_gets_counted(tap_env):
    """The tap routes through moe_routing.route, as the quant methods do."""
    layer = _Layer()

    def always_experts_5_and_6(hidden_states, gating_output, topk, renormalize):
        rows = gating_output.shape[0]
        ids = torch.tensor([[5, 6]], dtype=torch.int32).expand(rows, topk)
        return torch.full((rows, topk), 0.5), ids

    layer.custom_routing_function = always_experts_5_and_6
    counts = _tapped(layer)
    _record(layer, _selecting(1, 2, 3))
    assert counts[DECODE, 0].tolist() == [0, 0, 0, 0, 0, 3, 3, 0]


def test_processes_sharing_an_ep_rank_write_apart(tap_env, monkeypatch):
    """DP ranks without expert parallelism, or PP stages, share EP ranks."""
    for rank in (0, 1):
        monkeypatch.setattr(expert_count_tap, "_TAPPED", [])
        monkeypatch.setattr(expert_count_tap, "_STATE", expert_count_tap._fresh_state())
        monkeypatch.setattr(expert_count_tap, "_global_rank", lambda rank=rank: rank)
        _tapped(_Layer(ep_rank=0))
        for _ in range(2):
            expert_count_tap.maybe_dump()
    assert sorted(p.name for p in tap_env.iterdir()) == [
        "counts_rank0_ep0_0000.npz",
        "counts_rank1_ep0_0000.npz",
    ]


def test_record_compiles_fullgraph_with_a_dynamic_token_dimension(tap_env):
    """The phase split is data, so tracing it adds no shape guard or break."""
    layer = _Layer()
    counts = _tapped(layer)
    torch._dynamo.reset()
    compiled = torch.compile(_record, backend="eager", fullgraph=True, dynamic=True)
    compiled(layer, _selecting(1, 2))
    compiled(layer, _selecting(*[3] * (DECODE_MAX + 2)))
    assert counts[DECODE, 2, :2].tolist() == [1, 2]
    assert counts[PREFILL, 2, :2].tolist() == [1, DECODE_MAX + 2]


def _runner(layer, *, dp_size=1, pcp_size=1):
    runner = VllmMoERunner.__new__(VllmMoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = layer
    runner.moe_config = SimpleNamespace(
        dp_size=dp_size,
        pcp_size=pcp_size,
        is_sequence_parallel=False,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    return runner


def _apply(runner, logits):
    upstream = "vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner"
    with patch(f"{upstream}._apply_quant_method", return_value="routed") as apply:
        out = runner._apply_quant_method(
            hidden_states=logits, router_logits=logits, shared_experts_input=None
        )
    assert out == "routed" and apply.call_count == 1


def test_the_runner_counts_only_its_share_of_rows_gathered_over_dp(
    tap_env, monkeypatch
):
    """Naive DP dispatch gathers every DP rank's rows before the quant method;
    the runner's own flags tell the tap to count only this rank's share."""
    monkeypatch.setattr(expert_count_tap, "_dp_pcp", lambda: ((1, 2), (0, 1)))
    layer = _Layer()
    counts = _tapped(layer)
    runner = _runner(layer, dp_size=2)
    assert runner._rows_gathered() == (True, False)
    with _padding(valid=2, padded=2):
        _apply(runner, _selecting(1, 2, 3, 4))  # rows 2-3 are DP rank 1's
    assert counts[DECODE, 0].tolist() == [2, 0, 0, 1, 1, 0, 0, 0]


def test_the_runner_counts_every_row_when_the_fused_kernel_owns_pcp(
    tap_env, monkeypatch
):
    """An armed fused EP layer skips the PCP gather, so its rows are local."""
    monkeypatch.setattr(expert_count_tap, "_dp_pcp", lambda: ((0, 1), (1, 2)))
    layer = _Layer()
    setattr(layer.quant_method, FUSED_MOE_EP_OP_ATTR, "op")
    counts = _tapped(layer)
    runner = _runner(layer, pcp_size=2)
    with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True):
        assert runner._rows_gathered() == (False, False)
        _apply(runner, _selecting(1, 2))
    assert counts[DECODE, 0].tolist() == [2, 1, 1, 0, 0, 0, 0, 0]
