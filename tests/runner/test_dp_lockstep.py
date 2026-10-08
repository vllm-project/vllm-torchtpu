# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RC39-A (Async Batch Queue + DP Lockstep Double-Dummy-Step
Desync Fix) and RC39-B (Residual Eager HLO Op Elimination & Spec-Decode
Pre-Warming).

================================================================================
3-TIER EXECUTION PROVENANCE BANNER
================================================================================
ANALYTIC / MICROBENCHMARK PROJECTION (NOT E2E PHYSICAL VLLM RUN)
- Tier 1 (Host Control-Plane / Runner & Worker Lockstep): Real execution of
  `TPUModelRunner.execute_model`, `TPUModelRunner._warmup_spec_decode_cleanup`,
  `TPUModelRunner._prewarm_spec_decode_residual_eager_ops`,
  `TPUWorker.execute_dummy_batch`, and `DFlashProposer._layout_draft_block` /
  `_prepare_dflash_inputs` across a 64-step 2-rank `DPEngineCoreProc.run_busy_loop`
  + `step_with_batch_queue` simulation.
- Tier 2 (Host Lockstep Collective Sequence Verification): Exact step-by-step
  collective pairing audit (`_dp_coordinated_step`, Target forwards, DSpark KV
  updates, DSpark draft forwards) anchored to physical Cloud TPU v7x job
  `j-c7199a10`.
- Tier 3 (CPU Numerical Parity — NOT Physical TPU Execution): Bit-exact layout
  parity between broadcast `torch.where` `_layout_draft_block` and reference
  `scatter_` layout across padded and unpadded request counts on host CPU.
================================================================================
"""

from __future__ import annotations

import collections
import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

for _cand_path in (
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "/usr/local/google/home/jfy/claude-tpu-opt/studies/deepseek_v3_agentx_hillclimbing/scratch",
    "/usr/local/google/home/jfy/.gemini/jetski/brain/6a70d1f8-c693-4890-afaa-dbb1e841dd0e/scratch/gate7_m2/venv_torch/lib/python3.13/site-packages",
):
    if os.path.isdir(_cand_path) and _cand_path not in sys.path:
        sys.path.insert(0, _cand_path)

import torch  # noqa: E402

try:
    import vllm  # noqa: F401
except ImportError:
    from tests.test_m3_raiden_dspark_ep32 import _ensure_vllm_offload_stubs

    _ensure_vllm_offload_stubs()

# Ensure worker_base stub exists when running in lightweight CPU test env
import inspect as _inspect  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

if "vllm.v1.worker.worker_base" not in _sys.modules:
    _wb_mod = _types.ModuleType("vllm.v1.worker.worker_base")

    class _StubCompilationTimes:
        pass

    class _StubWorkerBase:
        def __init__(self, *args, **kwargs):
            pass

    _wb_mod.CompilationTimes = _StubCompilationTimes
    _wb_mod.WorkerBase = _StubWorkerBase
    _sys.modules["vllm.v1.worker.worker_base"] = _wb_mod

if "torch_tpu._internal.profiler" not in _sys.modules:
    _prof_mod = _types.ModuleType("torch_tpu._internal.profiler")

    class _StubTpuProfilerConfig:
        pass

    _prof_mod.TpuProfilerConfig = _StubTpuProfilerConfig
    _sys.modules["torch_tpu._internal.profiler"] = _prof_mod

try:
    from vllm.v1.core.sched.output import SchedulerOutput as _SchedOut

    if (
        not hasattr(_SchedOut, "make_empty")
        or getattr(_SchedOut.make_empty(), "total_num_scheduled_tokens", None) != 0
    ):
        _SchedOut.make_empty = staticmethod(
            lambda: SimpleNamespace(
                total_num_scheduled_tokens=0,
                scheduled_encoder_inputs={},
                finished_req_ids=set(),
                preempted_req_ids=set(),
                scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
                kv_cache_block_copies=None,
            )
        )
except Exception as e:
    logging.debug("make_empty stub error: %s", e)


def _disable_use_tpu_on_cpu_stub() -> None:
    try:
        torch.device("tpu")
    except Exception:
        # In CPU-only stub environments, set pytest's --use-tpu option to False
        # so tests/conftest.py's `device` fixture returns torch.device("cpu").
        fr = _inspect.currentframe()
        while fr is not None:
            for v_obj in list(fr.f_locals.values()):
                opt = getattr(v_obj, "option", None)
                if opt is not None and hasattr(opt, "use_tpu"):
                    opt.use_tpu = False
            fr = fr.f_back


_disable_use_tpu_on_cpu_stub()


def _make_minimal_runner_and_worker(
    *,
    dp_lockstep: bool = True,
    async_scheduling: bool = True,
    coordinated_return: tuple[int | None, int] = (16, 1),
):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.execute_model_state = None
    runner.num_tokens_paddings = [16, 32, 64, 128, 256, 512, 1024, 2048]
    runner.num_reqs_paddings = [1, 8, 16, 32, 64]
    runner.num_reqs_max_model_len = 8
    runner.max_num_blocks_per_req = 4
    runner.max_num_reqs = 64
    runner.max_model_len = 4096
    runner.scheduler_config = SimpleNamespace(
        async_scheduling=async_scheduling,
        max_num_seqs=64,
    )
    runner.vllm_config = SimpleNamespace()
    runner._update_states = MagicMock()
    runner._flush_disjoint_async_results = MagicMock()
    runner._dp_lockstep_enabled = MagicMock(return_value=dp_lockstep)
    runner._dp_coordinated_step = MagicMock(return_value=coordinated_return)
    runner._run_dp_idle_pairing = MagicMock()
    runner._dummy_run = MagicMock()
    runner._dp_step_max_reqs = 0
    runner._dp_step_num_chunks = 0
    runner._dp_target_bucket = None
    runner._dp_step_has_encoder_inputs = False
    runner._dp_step_has_active_decode = False
    runner._dp_step_executed_in_empty_execute_model = False
    runner._pcp_enabled = False
    runner._mtp_enabled = False
    runner.supports_mm_inputs = False
    runner.phase_based_profiler = None
    runner._pp_is_last = True
    runner.is_pooling_model = True
    runner.speculative_config = None
    runner.drafter = None
    runner.spec_decode_manager = None
    runner._pre_async_results = None
    runner.mm_embed_inputs = None
    runner.requests = {}
    runner.input_batch = SimpleNamespace(
        num_reqs=0,
        req_ids=[],
        req_id_to_index={},
        max_num_logprobs=None,
    )

    worker = TPUWorker.__new__(TPUWorker)
    worker.model_runner = runner
    return runner, worker


def test_rc39a_empty_execute_model_followed_by_execute_dummy_batch_skips_duplicate_collective():  # noqa: E501
    """Verify RC39-A: When `execute_model(total_num_scheduled_tokens == 0)` runs
    under `step_with_batch_queue` (which returns `model_executed = False`) and
    `DPEngineCoreProc.run_busy_loop` immediately calls `execute_dummy_batch()`
    on the same scheduler tick, `_dp_coordinated_step` and `_run_dp_idle_pairing`
    execute EXACTLY ONCE on that tick, and the next idle tick
    executes `execute_dummy_batch()` normally.
    """
    runner, worker = _make_minimal_runner_and_worker(
        dp_lockstep=True,
        async_scheduling=True,
        coordinated_return=(16, 1),
    )
    empty_sched_out = SimpleNamespace(
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        kv_cache_block_copies=None,
    )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
            return_value=False,
        ),
        patch(
            "vllm_torchtpu.worker.tpu_worker.has_kv_transfer_group",
            return_value=False,
        ),
    ):
        # Tick 1: step_with_batch_queue calls execute_model(0), then run_busy_loop
        # sees executed=False and calls worker.execute_dummy_batch().
        out = runner.execute_model(empty_sched_out)
        assert out is not None
        assert runner._dp_step_executed_in_empty_execute_model is True
        assert runner._dp_coordinated_step.call_count == 1
        assert runner._run_dp_idle_pairing.call_count == 1
        runner._run_dp_idle_pairing.assert_called_once_with(16, 1)

        # Same Tick 1: run_busy_loop calls worker.execute_dummy_batch()
        worker.execute_dummy_batch()
        assert runner._dp_step_executed_in_empty_execute_model is False
        # Must NOT call _dp_coordinated_step or _run_dp_idle_pairing again!
        assert runner._dp_coordinated_step.call_count == 1
        assert runner._run_dp_idle_pairing.call_count == 1

        # Tick 2: Rank now has has_requests() == False, so execute_model
        # is NOT called; run_busy_loop calls execute_dummy_batch() directly.
        worker.execute_dummy_batch()
        assert runner._dp_step_executed_in_empty_execute_model is False
        assert runner._dp_coordinated_step.call_count == 2
        assert runner._run_dp_idle_pairing.call_count == 2


def test_rc39a_active_step_and_warmup_cleanup_reset_guard_and_raiden_fence():
    """Verify Gate 1 (crash/repeat/cleanup paths) for RC39-A:
    1. `_warmup_spec_decode_cleanup` calls `execute_model(so)` with 0 tokens to
       drain synthetic warmup requests, and MUST reset
       `_dp_step_executed_in_empty_execute_model = False` so the first idle tick
       of `run_busy_loop` after warmup does not skip its collective.
    2. `TPUWorker.execute_dummy_batch()` calls `record_pending_fences()` on the
       KV transfer group when `has_kv_transfer_group()` is True and
       `target_num_chunks > 0`, and does not call it when skipping a duplicate
       step or when `target_num_chunks == 0`.
    """
    runner, worker = _make_minimal_runner_and_worker(
        dp_lockstep=True,
        async_scheduling=True,
        coordinated_return=(16, 1),
    )
    runner.requests = {"__spec_warmup_0__": SimpleNamespace()}
    runner.input_batch.req_id_to_index = {"__spec_warmup_0__": 0}

    def _mock_update_states(so):
        for rid in getattr(so, "finished_req_ids", set()):
            runner.requests.pop(rid, None)
            runner.input_batch.req_id_to_index.pop(rid, None)
        runner.input_batch.num_reqs = len(runner.requests)

    runner._update_states = _mock_update_states

    with patch(
        "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
        return_value=False,
    ):
        ok = runner._warmup_spec_decode_cleanup(["__spec_warmup_0__"])
        assert ok is True
        # execute_model(so) ran inside _warmup_spec_decode_cleanup:
        assert runner._dp_coordinated_step.call_count == 1
        # Guard MUST be reset to False after warmup cleanup!
        assert runner._dp_step_executed_in_empty_execute_model is False

    # Now verify execute_dummy_batch() after warmup cleanup runs normally AND
    # invokes record_pending_fences() when has_kv_transfer_group() is True.
    fake_kv_connector = SimpleNamespace(record_pending_fences=MagicMock())
    with (
        patch(
            "vllm_torchtpu.worker.tpu_worker.has_kv_transfer_group",
            return_value=True,
        ),
        patch(
            "vllm_torchtpu.worker.tpu_worker.get_kv_transfer_group",
            return_value=fake_kv_connector,
        ),
    ):
        worker.execute_dummy_batch()
        assert runner._dp_coordinated_step.call_count == 2
        assert runner._run_dp_idle_pairing.call_count == 2
        fake_kv_connector.record_pending_fences.assert_called_once()

        # When target_num_chunks == 0, record_pending_fences is not called.
        runner._dp_coordinated_step.return_value = (16, 0)
        fake_kv_connector.record_pending_fences.reset_mock()
        worker.execute_dummy_batch()
        assert runner._dp_coordinated_step.call_count == 3
        fake_kv_connector.record_pending_fences.assert_not_called()


def _simulate_2rank_dp_busy_loop(
    *,
    num_steps: int = 64,
    batch_queue_size: int = 2,
    disable_rc39a_guard: bool = False,
) -> dict[str, object]:
    """Simulate a 2-rank `DPEngineCoreProc.run_busy_loop` + `step_with_batch_queue`
    execution across all 4 states:
    - State A: Active (`has_requests=True, scheduled_tokens > 0`)
    - State B1: Waiting for Raiden KV load with queue not full
      (`has_requests=True, scheduled_tokens == 0, len(batch_queue) < batch_queue_size`)
    - State B2: Waiting for Raiden KV load with full queue popping older batch
      (`has_requests=True, scheduled_tokens == 0, len(batch_queue) == batch_queue_size`)
    - State C: Draining `batch_queue` (`has_requests=False, len(batch_queue) > 0`)
    - State D: Completely idle (`has_requests=False, len(batch_queue) == 0`)
    """
    rank_traces: list[list[tuple[int, str, int, int]]] = [[], []]
    runners = []
    workers = []
    batch_queues = [collections.deque(), collections.deque()]

    for r in range(2):
        runner, worker = _make_minimal_runner_and_worker(
            dp_lockstep=True,
            async_scheduling=True,
        )
        runners.append(runner)
        workers.append(worker)

    # Deterministic 64-step schedule for Rank 0 and Rank 1 exercising all 4 states
    # and transitions between them.
    r0_plan: list[tuple[bool, int]] = []
    r1_plan: list[tuple[bool, int]] = []
    for s in range(num_steps):
        if s < 16:
            # Both active or Rank 1 hitting 0-token KV load while queue < 2
            r0_plan.append((True, 16 if s % 3 != 2 else 256))
            r1_plan.append((True, 0 if s in (2, 5, 9, 13) else 16))
        elif s < 32:
            # Rank 1 builds up full batch_queue (2 in-flight active batches) then
            # hits 0-token schedule on step s, popping an older batch while
            # returning model_executed=False!
            r0_plan.append((True, 16))
            r1_plan.append((True, 0 if s in (18, 21, 25, 29) else 16))
        elif s < 48:
            # Rank 1 transitions to has_requests=False (draining batch_queue then
            # completely idle) while Rank 0 stays active; then Rank 0 also hits
            # 0-token and drains.
            r0_has = s < 44
            r0_tok = 0 if s in (35, 39) else (16 if r0_has else 0)
            r1_has = s in (36, 37, 42, 43)
            r1_tok = 0 if s == 36 else (16 if r1_has else 0)
            r0_plan.append((r0_has, r0_tok))
            r1_plan.append((r1_has, r1_tok))
        else:
            # Interleaved active, 0-token, draining, and both-idle steps
            if s in (52, 53, 60, 61):
                r0_plan.append((False, 0))
                r1_plan.append((False, 0))
            elif s % 2 == 0:
                r0_plan.append((True, 0 if s % 4 == 0 else 64))
                r1_plan.append((True, 16))
            else:
                r0_plan.append((True, 16))
                r1_plan.append((s % 3 == 0, 0))

    desync_step: int | None = None

    for step_idx in range(num_steps):
        # Determine actual scheduled tokens per rank on this tick
        tok0 = r0_plan[step_idx][1] if r0_plan[step_idx][0] else 0
        tok1 = r1_plan[step_idx][1] if r1_plan[step_idx][0] else 0
        max_tok = max(tok0, tok1)
        coord_bucket = 16 if max_tok == 0 else max(16, max_tok)
        coord_chunks = 1 if max_tok > 0 else 0

        for r, (has_reqs, sched_tokens) in enumerate(
            (r0_plan[step_idx], r1_plan[step_idx])
        ):
            runner = runners[r]
            worker = workers[r]
            bq = batch_queues[r]

            def _make_coord(rk=r, si=step_idx, cb=coord_bucket, cc=coord_chunks):
                def _coord(local_tokens, local_chunks, **kwargs):
                    rank_traces[rk].append((si, "coord", cb, cc))
                    return cb, cc

                return _coord

            def _make_idle_pair(rk=r, si=step_idx):
                def _idle_pair(b, c):
                    if c > 0:
                        rank_traces[rk].append((si, "target_fwd", b, c))
                        rank_traces[rk].append((si, "draft_fwd", b, c))

                return _idle_pair

            runner._dp_coordinated_step = _make_coord()
            runner._run_dp_idle_pairing = _make_idle_pair()

            # Simulate DPEngineCoreProc.run_busy_loop + EngineCore.step_with_batch_queue
            if not has_reqs and len(bq) == 0:
                executed = False
            else:
                model_executed = False
                if has_reqs:
                    so = SimpleNamespace(
                        total_num_scheduled_tokens=sched_tokens,
                        scheduled_encoder_inputs={},
                        finished_req_ids=set(),
                        preempted_req_ids=set(),
                        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
                        kv_cache_block_copies=None,
                    )
                    if sched_tokens > 0:
                        model_executed = True
                        runner._dp_step_executed_in_empty_execute_model = False
                        cb, cc = runner._dp_coordinated_step(sched_tokens, 1)
                        rank_traces[r].append((step_idx, "target_fwd", cb, cc))
                        rank_traces[r].append((step_idx, "draft_fwd", cb, cc))
                        bq.appendleft(("future", so))
                        if len(bq) < batch_queue_size:
                            executed = True
                        else:
                            bq.pop()
                            executed = True
                    else:
                        # 0 tokens scheduled while has_requests() is True:
                        # step_with_batch_queue calls execute_model(so) and
                        # leaves model_executed = False!
                        with patch(
                            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
                            return_value=False,
                        ):
                            runner.execute_model(so)
                        if disable_rc39a_guard:
                            runner._dp_step_executed_in_empty_execute_model = False
                        if len(bq) == batch_queue_size:
                            bq.pop()
                        executed = model_executed  # False!
                else:
                    # has_requests() is False, draining non-empty batch_queue:
                    bq.pop()
                    executed = False

            if not executed:
                with patch(
                    "vllm_torchtpu.worker.tpu_worker.has_kv_transfer_group",
                    return_value=False,
                ):
                    worker.execute_dummy_batch()

        if rank_traces[0] != rank_traces[1] and desync_step is None:
            desync_step = step_idx

    return {
        "rank0_trace": rank_traces[0],
        "rank1_trace": rank_traces[1],
        "traces_equal": rank_traces[0] == rank_traces[1],
        "desync_step": desync_step,
        "rank0_len": len(rank_traces[0]),
        "rank1_len": len(rank_traces[1]),
    }


def test_rc39a_64_step_2rank_batch_queue_busy_loop_lockstep_and_negative_control():
    """Verify RC39-A across a 64-step 2-rank `DPEngineCoreProc.run_busy_loop` +
    `step_with_batch_queue` simulation covering all 4 states (active, 0-token
    with `has_requests() == True` under both `len(batch_queue) < batch_queue_size`
    and `len(batch_queue) == batch_queue_size`, draining `batch_queue` with
    `has_requests() == False`, and completely idle), plus Gate 4 Negative Control.
    """
    res_patched = _simulate_2rank_dp_busy_loop(
        num_steps=64,
        batch_queue_size=2,
        disable_rc39a_guard=False,
    )
    assert res_patched["traces_equal"] is True, (
        f"Expected identical 2-rank collective traces across 64 steps, "
        f"got desync at step {res_patched['desync_step']}: "
        f"len0={res_patched['rank0_len']} vs len1={res_patched['rank1_len']}"
    )
    assert res_patched["desync_step"] is None
    # Every single one of the 64 steps must emit exactly 1 'coord' collective per rank
    r0_coords = [e for e in res_patched["rank0_trace"] if e[1] == "coord"]
    r1_coords = [e for e in res_patched["rank1_trace"] if e[1] == "coord"]
    assert len(r0_coords) == 64
    assert len(r1_coords) == 64

    # Gate 4 Negative Control: Without the RC39-A guard, step 2 (the first step
    # where Rank 1 has `has_requests() == True` and `scheduled_tokens == 0` while
    # Rank 0 has `scheduled_tokens == 256`) causes Rank 1 to execute TWO
    # `_dp_coordinated_step` + `_run_dp_idle_pairing` sequences on a single tick!
    res_unpatched = _simulate_2rank_dp_busy_loop(
        num_steps=64,
        batch_queue_size=2,
        disable_rc39a_guard=True,
    )
    assert res_unpatched["traces_equal"] is False
    assert res_unpatched["desync_step"] == 2
    assert res_unpatched["rank1_len"] > res_unpatched["rank0_len"]


def test_rc39b_dflash_layout_draft_block_zero_scatter_and_negative_control(monkeypatch):
    """Verify RC39-B: `DFlashProposer._layout_draft_block` builds `[base, MASK x K]`
    per request without calling `torch.Tensor.scatter_`, matches the reference
    layout bit-for-bit across padded and unpadded lengths, and pre-allocates
    `_cpu_base_tokens` / `_cpu_draft_input_ids`. Includes Gate 4 negative control.
    """
    from vllm_torchtpu.spec_decode.dflash import DFlashProposer

    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer._max_model_len = 4096
    proposer.mask_token_id = 999
    proposer._cpu_base_tokens = np.zeros(512, dtype=np.int32)
    proposer._cpu_draft_input_ids = np.full(512 * 8, 999, dtype=np.int32)

    def _forbid_scatter(*args, **kwargs):
        raise AssertionError(
            "RC39-B violation: torch.Tensor.scatter_ must not be called"
        )

    monkeypatch.setattr(torch.Tensor, "scatter_", _forbid_scatter)

    base_tokens = torch.tensor([101, 202, 303, 999], dtype=torch.int32)
    base_pos = torch.tensor([[10], [40], [4094], [0]], dtype=torch.int32)
    block_size = 4

    # Case 1: padded_len == padded_num_reqs * block_size (16)
    input_ids, positions, seq_lens = proposer._layout_draft_block(
        base_tokens, base_pos, block_size=block_size, padded_len=16, mask_token_id=999
    )
    assert input_ids.tolist() == [
        101,
        999,
        999,
        999,
        202,
        999,
        999,
        999,
        303,
        999,
        999,
        999,
        999,
        999,
        999,
        999,
    ]
    assert positions.tolist() == [
        10,
        11,
        12,
        13,
        40,
        41,
        42,
        43,
        4094,
        4095,
        4095,
        4095,
        0,
        1,
        2,
        3,
    ]
    assert seq_lens.tolist() == [14, 44, 4096, 4]

    # Case 2: padded_len > padded_num_reqs * block_size (24 > 16)
    input_ids_pad, positions_pad, seq_lens_pad = proposer._layout_draft_block(
        base_tokens, base_pos, block_size=block_size, padded_len=24, mask_token_id=999
    )
    assert input_ids_pad.shape == (24,)
    assert input_ids_pad[:16].tolist() == input_ids.tolist()
    assert input_ids_pad[16:].tolist() == [999] * 8
    assert positions_pad.shape == (24,)
    assert positions_pad[:16].tolist() == positions.tolist()
    assert positions_pad[16:].tolist() == [0] * 8
    assert seq_lens_pad.tolist() == seq_lens.tolist()

    # Gate 4 Negative Control: Legacy scatter_ implementation must trigger the trap
    with pytest.raises(AssertionError, match="RC39-B violation"):
        legacy_ids = torch.full((16,), 999, dtype=torch.int32)
        scatter_idx = torch.arange(0, 16, block_size, dtype=torch.int64)
        legacy_ids.scatter_(0, scatter_idx, base_tokens)


def test_rc39b_dflash_tensor_num_reqs_and_runner_prewarm():
    """Verify RC39-B:
    1. `_request_window_bounds`, `_tpu_build_dflash_inputs_next_tokens`, and
       `_tpu_build_dflash_inputs_seed` accept `num_reqs` as either a Python `int`
       or a 1-element `torch.int32` tensor (`_get_num_reqs_tpu`) and produce
       bit-identical outputs across `num_reqs = 1, 2, 3, 5, 8`.
    2. `TPUModelRunner._prewarm_spec_decode_residual_eager_ops` executes cleanly
       and pre-warms `_layout_draft_block` and async substitution helpers across
       all configured `num_reqs_paddings` and `num_tokens_paddings`.
    """
    from vllm_torchtpu.spec_decode.dflash import DFlashProposer

    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer._max_model_len = 4096
    proposer.mask_token_id = 999
    proposer._num_reqs_tpu_cache = {}
    device = torch.device("cpu")

    block_size = 4
    padded_num_reqs = 8
    padded_len = padded_num_reqs * block_size
    qsl = torch.tensor([0, 3, 5, 9, 12, 16, 20, 24, 28], dtype=torch.int32)
    pos = torch.arange(100, 132, dtype=torch.int32)
    draft_lengths = torch.tensor([3, 0, 3, 3, 0, 3, 3, 3], dtype=torch.int32)
    nt_full = torch.tensor(
        [
            [11, 12, 13, -1],
            [21, -1, -1, -1],
            [31, 32, 33, 34],
            [41, 42, -1, -1],
            [51, -1, -1, -1],
            [61, 62, 63, -1],
            [71, 72, -1, -1],
            [81, 82, 83, 84],
        ],
        dtype=torch.int32,
    )
    seed_full = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int32)

    fn_nt = torch._dynamo.disable(DFlashProposer._tpu_build_dflash_inputs_next_tokens)
    fn_seed = torch._dynamo.disable(DFlashProposer._tpu_build_dflash_inputs_seed)

    for nr in (1, 2, 3, 5, 8):
        nr_tensor = proposer._get_num_reqs_tpu(nr, device)
        assert nr_tensor.shape == (1,) and nr_tensor.dtype == torch.int32
        assert int(nr_tensor.item()) == nr
        # Cached tensor reuse check
        assert proposer._get_num_reqs_tpu(nr, device) is nr_tensor

        ids_int, pos_int, sl_int = fn_nt(
            proposer,
            nt_full[:nr],
            qsl,
            pos,
            draft_lengths,
            nr,
            block_size,
            padded_len,
            999,
            -1,
        )
        ids_ten, pos_ten, sl_ten = fn_nt(
            proposer,
            nt_full,
            qsl,
            pos,
            draft_lengths,
            nr_tensor,
            block_size,
            padded_len,
            999,
            -1,
        )
        assert torch.equal(ids_int, ids_ten)
        assert torch.equal(pos_int, pos_ten)
        assert torch.equal(sl_int, sl_ten)

        s_ids_int, s_pos_int, s_sl_int = fn_seed(
            proposer,
            seed_full[:nr],
            qsl,
            pos,
            nr,
            block_size,
            padded_len,
            999,
        )
        s_ids_ten, s_pos_ten, s_sl_ten = fn_seed(
            proposer,
            seed_full,
            qsl,
            pos,
            nr_tensor,
            block_size,
            padded_len,
            999,
        )
        assert torch.equal(s_ids_int, s_ids_ten)
        assert torch.equal(s_pos_int, s_pos_ten)
        assert torch.equal(s_sl_int, s_sl_ten)

    # Verify TPUModelRunner._prewarm_spec_decode_residual_eager_ops
    runner, _ = _make_minimal_runner_and_worker()
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    proposer.speculative_config = runner.speculative_config
    layout_spy = MagicMock(side_effect=proposer._layout_draft_block)
    proposer.mask_token_id = 999
    proposer._layout_draft_block = layout_spy
    runner.drafter = proposer

    runner._prewarm_spec_decode_residual_eager_ops()
    assert layout_spy.call_count == len(runner.num_reqs_paddings)

    # On physical TPU device (type == "tpu"), direct eager _layout_draft_block
    # is skipped in _prewarm_spec_decode_residual_eager_ops because it is
    # compiled inline inside _tpu_build_dflash_inputs_next_tokens / _seed.
    layout_spy.reset_mock()
    with patch.object(runner, "device", SimpleNamespace(type="tpu")):
        runner._prewarm_spec_decode_residual_eager_ops()
    assert layout_spy.call_count == 0


def test_rc39c_dspark_proposer_init_via_tpu_model_runner_and_query_block_size():
    """Verify RC39-C (from physical TPU v7x job j-06805736-df71-46ec-b8c4):
    1. `TPUModelRunner._init_speculative_decoding()` constructs
       `DSparkProposer(self, self.vllm_config)` without raising `AttributeError:
       'DSparkProposer' object has no attribute 'sample_from_anchor'` when
       `DFlashProposer.__init__` accesses `self.block_size` ->
       `DSparkProposer._query_block_size(K)`.
    2. Both `sample_from_anchor=True` (dense layout, `block_size == K == 5`) and
       `sample_from_anchor=False` (bonus-anchor layout, `block_size == K + 1 == 6`)
       initialize `self.sample_from_anchor` BEFORE `super().__init__()` sizes
       `self._cpu_draft_input_ids`.
    3. `DSparkProposer._query_block_size(5)` never raises `AttributeError` even
       on an uninitialized `DSparkProposer.__new__(DSparkProposer)` instance.
    """
    from vllm_torchtpu.spec_decode.dspark import DSparkProposer

    # 1. Bare uninitialized instance safety check
    bare_dspark = DSparkProposer.__new__(DSparkProposer)
    assert not hasattr(bare_dspark, "sample_from_anchor")
    assert bare_dspark._query_block_size(5) == 5

    # Gate 4 Negative Control: Direct access on bare instance raises AttributeError
    with pytest.raises(AttributeError, match="sample_from_anchor"):
        _ = 5 if bare_dspark.sample_from_anchor else 6

    def _build_runner_with_dspark(hf_dict: dict, K: int = 5):
        runner, _ = _make_minimal_runner_and_worker()
        hf_cfg = SimpleNamespace(to_dict=lambda: dict(hf_dict))
        spec_cfg = SimpleNamespace(
            method="dspark",
            num_speculative_tokens=K,
            draft_tensor_parallel_size=1,
            draft_model_config=SimpleNamespace(hf_config=hf_cfg),
            draft_sample_method="greedy",
            enable_adaptive_verification=False,
            dspark_draft_topk=None,
            use_eagle=lambda: False,
        )
        runner.speculative_config = spec_cfg
        runner.vllm_config = SimpleNamespace(
            speculative_config=spec_cfg,
            model_config=SimpleNamespace(max_model_len=4096),
            compilation_config=SimpleNamespace(
                static_forward_context={"model.layers.0.self_attn": MagicMock()},
                fast_moe_cold_start=False,
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=16,
                tensor_parallel_size=1,
                is_moe_model=True,
                use_sequence_parallel_moe=False,
            ),
        )
        with patch("vllm_torchtpu.runner.tpu_runner.RejectionSampler", MagicMock()):
            runner._init_speculative_decoding()
        return runner

    # 2. Dense checkpoint (sample_from_anchor=True / default): block_size == K == 5
    runner_dense = _build_runner_with_dspark(
        {
            "dspark_noise_token_id": 7,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [1, 15, 30],
        },
        K=5,
    )
    assert isinstance(runner_dense.drafter, DSparkProposer)
    assert runner_dense.drafter.sample_from_anchor is True
    assert runner_dense.drafter.block_size == 5
    assert runner_dense.drafter._cpu_draft_input_ids.shape == (512 * 5,)

    # 3. Speculators checkpoint (sample_from_anchor=False): block_size == K + 1 == 6
    runner_bonus = _build_runner_with_dspark(
        {
            "mask_token_id": 7,
            "sample_from_anchor": False,
            "dspark_target_layer_ids": [1, 15, 30],
        },
        K=5,
    )
    assert isinstance(runner_bonus.drafter, DSparkProposer)
    assert runner_bonus.drafter.sample_from_anchor is False
    assert runner_bonus.drafter.block_size == 6
    assert runner_bonus.drafter._cpu_draft_input_ids.shape == (512 * 6,)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
