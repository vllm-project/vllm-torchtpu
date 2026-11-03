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
"""Runner contracts for DSpark adaptive verification: async rollback of
trimmed drafts, confidence bookkeeping, re-admitting drafts under DP lockstep,
and the trimmed-step warmup plan."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID
from vllm_torchtpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    StepCostTable,
)
from vllm_torchtpu.spec_decode.dspark import DraftConfidences, DSparkProposer

K = 3


def _runner(device, req_ids, step_trimmed):
    runner = object.__new__(TPUModelRunner)
    runner.device = device
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=K)
    runner.input_batch = SimpleNamespace(req_ids=req_ids)
    manager = AdaptiveVerificationManager(
        K, draft_block_size=K, async_scheduling=True, calibrate_overhead=False
    )
    manager.step_trimmed = step_trimmed
    runner._adaptive_verification = manager
    runner._unfenced_h2d_sources = []
    runner._free_h2d_sources = {}
    runner._registered_h2d_sources = set()
    return runner


def test_trimmed_drafts_roll_back_as_rejected(device):
    # "a": scheduled K drafts, verified 2, accepted 1. "b": scheduled K,
    # every draft trimmed, so its chunk carries no verify metadata.
    runner = _runner(device, ["a", "b"], {"a": K - 2, "b": K})
    verified = torch.tensor(
        [[11, 12] + [INVALID_TOKEN_ID] * (K - 1)], dtype=torch.int32, device=device
    )
    sampled = torch.tensor(
        [[21] + [INVALID_TOKEN_ID] * K], dtype=torch.int32, device=device
    )
    state = SimpleNamespace(
        num_reqs_list=[1, 1],
        spec_decode_metadata_list=[
            SimpleNamespace(
                draft_lengths=torch.tensor([2], dtype=torch.int32, device=device),
                draft_lengths_cpu=np.array([2], dtype=np.int32),
            ),
            None,
        ],
    )
    drafts = torch.zeros(2, K, dtype=torch.int32, device=device)
    _, _, rejected, num_draft_per_req = runner._assemble_async_spec_substitution(
        drafts, [verified, sampled], state
    )
    # The scheduler advanced both by 1 + K: roll back all but the accepted.
    assert rejected is not None
    assert rejected.cpu().tolist() == [K - 1, K]
    # The runner's own placeholders count only the verified drafts.
    assert num_draft_per_req == {0: 2}


@pytest.mark.parametrize("step_trimmed,expected", [({}, [0, 0]), ({"a": K}, [K, 0])])
def test_fully_trimmed_step_rolls_back_every_scheduled_draft(
    device, step_trimmed, expected
):
    runner = _runner(device, ["a", "b"], step_trimmed)
    drafts = torch.zeros(2, K, dtype=torch.int32, device=device)
    sampled = torch.tensor([[5], [6]], dtype=torch.int32, device=device)
    _, _, rejected, num_draft_per_req = runner._assemble_async_prefill_bootstrap(
        drafts, [sampled], [2]
    )
    assert rejected is not None
    assert rejected.cpu().tolist() == expected
    assert num_draft_per_req == {0: 0, 1: 0}


def test_scheduled_draft_counts_keep_their_host_source_alive(device):
    runner = _runner(device, ["a", "b"], {"a": 1})
    md = mock.Mock(draft_lengths_cpu=np.array([2, 3], dtype=np.int32))
    counts = runner._scheduled_draft_counts(0, 2, 4, md)
    assert counts is not None
    assert counts.cpu().tolist() == [3, 3, 0, 0]
    # TorchTPU does not hold the CPU source of an async copy; the staging
    # fence does.
    [(source, _)] = runner._unfenced_h2d_sources
    assert source.tolist() == [3, 3, 0, 0]


def test_confidences_map_to_their_chunks_requests(device):
    runner = _runner(device, ["x", "a", "b", "c"], {})
    runner.drafter = mock.MagicMock(spec=DSparkProposer)
    # Bucketed chunks of 8 rows: rows past each chunk's requests are padding.
    runner.drafter.take_draft_confidences.return_value = DraftConfidences(
        [torch.full((8, K), 0.5), torch.full((8, K), 0.25)], [(1, 2), (3, 1)]
    )
    runner._record_draft_confidences()
    manager = runner._adaptive_verification
    assert manager is not None
    manager._drain(keep=0)
    assert set(manager._confidences) == {"a", "b", "c"}
    np.testing.assert_allclose(manager._confidences["a"], [0.5] * K)
    np.testing.assert_allclose(manager._confidences["c"], [0.25] * K)


def test_a_step_without_proposal_records_nothing(device):
    runner = _runner(device, ["a"], {})
    runner.drafter = mock.MagicMock(spec=DSparkProposer)
    runner.drafter.take_draft_confidences.return_value = None
    runner._record_draft_confidences()
    manager = runner._adaptive_verification
    assert manager is not None
    assert not manager._pending


class _Batch:
    """The input-batch state the readmit step and the batch reorder touch."""

    def __init__(self, req_ids):
        self.req_ids = list(req_ids)
        self.num_reqs = len(req_ids)
        self.staged_drafts: dict[str, int] = {}

    def swap_states(self, i, j):
        self.req_ids[i], self.req_ids[j] = self.req_ids[j], self.req_ids[i]

    def update_req_spec_token_ids(self, request, scheduled_spec_tokens):
        self.staged_drafts[request.req_id] = len(
            scheduled_spec_tokens.get(request.req_id, ())
        )


def _trimmed_step():
    """A runner laid out for a trimmed decode step, as execute_model leaves it
    before the DP bucket is known: "a" verifies 2 of its drafts and "b",
    whose drafts are worth little, none."""
    manager = AdaptiveVerificationManager(
        K, draft_block_size=K, async_scheduling=False, calibrate_overhead=False
    )
    # A step costs 10 ms up to 4 tokens and 20 ms up to 8.
    manager.cost_table = StepCostTable.from_samples(
        [4, 8],
        [(1, 1, 10.0), (1, 4, 10.0)]
        + [(2, 2, 10.0), (2, 4, 10.0), (2, 5, 20.0), (2, 8, 20.0)],
        [(8, 1.0)],
    )
    manager.record_confidences(["a", "b"], [torch.tensor([[0.9] * K, [0.1] * K])], [2])
    so = SchedulerOutput.make_empty()
    so.num_scheduled_tokens = {"a": 1 + K, "b": 1 + K}
    so.scheduled_spec_decode_tokens = {"a": [-1] * K, "b": [-1] * K}
    so.total_num_scheduled_tokens = 2 * (1 + K)
    trimmed = manager.trim(so)
    assert trimmed.num_scheduled_tokens == {"a": 3, "b": 1}

    runner = object.__new__(TPUModelRunner)
    runner._adaptive_verification = manager
    runner._static_verify_q_len = 1 + K
    runner.requests = {r: SimpleNamespace(req_id=r) for r in ("a", "b")}
    runner.input_batch = _Batch(["a", "b"])
    assert runner._reorder_batch_for_rpa(trimmed) == (1, 2)
    assert runner.input_batch.req_ids == ["b", "a"]
    return runner, trimmed


def test_dp_padding_readmits_drafts_into_the_batch_layout():
    runner, trimmed = _trimmed_step()
    # The DP-wide bucket is 8 tokens; the trimmed step fills 4.
    runner._dp_target_bucket = 8
    readmitted = runner._readmit_trimmed_drafts(trimmed, 1, 1)
    assert readmitted is not None
    scheduler_output, num_decode, num_windowed = readmitted
    assert scheduler_output.num_scheduled_tokens == {"a": 1 + K, "b": 1 + K}
    # "b" goes from no drafts back to all of them, and both reach the batch.
    assert runner.input_batch.staged_drafts == {"a": K, "b": K}
    # "b" left the 1-token decode segment; both windows are full again.
    assert (num_decode, num_windowed) == (0, 2)
    assert runner._num_static_verify_reqs == 2
    manager = runner._adaptive_verification
    assert manager is not None and manager.step_trimmed == {}


@pytest.mark.parametrize(
    "dp_target_bucket,num_chunks",
    [(None, 1), (8, 2), (4, 1)],
    ids=["no_dp_lockstep", "multi_chunk", "no_room"],
)
def test_readmit_leaves_the_step_alone(dp_target_bucket, num_chunks):
    runner, trimmed = _trimmed_step()
    runner._dp_target_bucket = dp_target_bucket
    assert runner._readmit_trimmed_drafts(trimmed, num_chunks, num_chunks) is None
    assert runner.input_batch.staged_drafts == {}
    assert runner.input_batch.req_ids == ["b", "a"]


PADDINGS = [16, 32, 64, 128, 256, 512, 1024]
SMALL_BUDGET = dict(K=5, block=5, max_seqs=8, budget=256, paddings=PADDINGS[:5])
LARGE_BUDGET = dict(K=5, block=5, max_seqs=64, budget=1024, paddings=PADDINGS)
# More requests than full windows fit in the token budget.
SATURATED = dict(K=5, block=5, max_seqs=256, budget=1024, paddings=PADDINGS)
CONFIGS = [SMALL_BUDGET, LARGE_BUDGET, SATURATED]
CONFIG_IDS = ["small_budget", "large_budget", "saturated"]


def _planning_runner(K, block, max_seqs, budget, paddings):
    drafter = mock.MagicMock(spec=DSparkProposer)
    drafter.block_size = block
    return SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        drafter=drafter,
        scheduler_config=SimpleNamespace(max_num_seqs=max_seqs),
        max_num_reqs=max(max_seqs, tpu_runner.MIN_NUM_SEQS),
        num_reqs_max_model_len=max(max_seqs, tpu_runner.MIN_NUM_SEQS),
        max_num_tokens=budget,
        num_tokens_paddings=paddings,
        _static_verify_q_len=K + 1,
    )


def _key(cfg, runner, num_reqs, budget):
    pad = tpu_runner._get_padded_token_len
    return (
        budget > 0,
        tpu_runner._get_padded_num_reqs_with_upper_limit(num_reqs, runner.max_num_reqs),
        pad(cfg["paddings"], num_reqs + budget),
        pad(cfg["paddings"], num_reqs * cfg["block"]),
    )


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_warmup_plan_covers_every_trimmed_step_once(cfg):
    K, runner = cfg["K"], _planning_runner(**cfg)
    plan, max_num_reqs = TPUModelRunner._adaptive_verification_warmup_plan(runner)
    max_window = min(cfg["paddings"][-1], cfg["budget"])
    counts = [n for n in range(1, cfg["max_seqs"] + 1) if n * (1 + K) <= max_window]
    full = {_key(cfg, runner, n, n * K) for n in counts}
    # Every step trimming may give: up to max_num_reqs requests, verifying
    # fewer drafts than the step scheduled.
    trimmed = {
        _key(cfg, runner, n, b)
        for n in range(1, max_num_reqs + 1)
        for b in range(min(n * K, max_window - n))
    }
    planned = [_key(cfg, runner, n, b) for n, b in plan]
    # The full-window warmup already compiled `full`; the plan adds every
    # other bucket a trimmed step reaches, each exactly once, through
    # full-window batches.
    assert len(planned) == len(set(planned))
    assert not set(planned) & full
    assert full | set(planned) == full | trimmed
    assert {n for n, _ in plan} <= set(counts)
    # Every step that verifies nothing takes the non-spec path: warmed too.
    assert any(not k[0] for k in planned)


@pytest.mark.parametrize(
    "K,max_num_reqs",
    [
        # 170 requests fill 1024 tokens with full windows. Up to 204 a step
        # keeps their 256-request pad and 1024-row draft bucket; at 205 its
        # draft rows outgrow the largest bucket.
        (5, 204),
        # 128 requests fill 1024 tokens with windows of 8; at 129 the request
        # pad doubles.
        (7, 128),
    ],
)
def test_trimming_stops_at_the_first_request_count_warmup_leaves_cold(K, max_num_reqs):
    runner = _planning_runner(**{**SATURATED, "K": K, "block": K})
    _, limit = TPUModelRunner._adaptive_verification_warmup_plan(runner)
    assert limit == max_num_reqs


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_profile_plan_samples_both_sides_of_every_bucket_edge(cfg):
    K, pads = cfg["K"], cfg["paddings"]
    runner = _planning_runner(**cfg)
    _, max_num_reqs = TPUModelRunner._adaptive_verification_warmup_plan(runner)
    plan = TPUModelRunner._adaptive_verification_profile_plan(runner, max_num_reqs)
    max_window = min(pads[-1], cfg["budget"])
    grid = sorted({n for n, _, _ in plan})
    # The profile covers exactly the steps trimming may run.
    assert grid[0] == 1 and grid[-1] == max_num_reqs
    # Neighbors' token spans overlap (StepCostTable.verify_cost).
    assert all(b <= a * (1 + K) for a, b in zip(grid, grid[1:]))
    for num_reqs in grid:
        tokens = {t for n, t, _ in plan if n == num_reqs}
        lo, hi = num_reqs, min(num_reqs * (1 + K), max_window)
        assert min(tokens) == lo and max(tokens) == hi
        for b in pads:
            if lo <= b < hi:
                assert {b, b + 1} <= tokens
    for num_reqs, num_tokens, query_lens in plan:
        # Real decode windows, in the step layout's order.
        assert len(query_lens) == num_reqs and sum(query_lens) == num_tokens
        assert all(1 <= q <= 1 + K for q in query_lens)
        segment = [0 if q == 1 else 1 if q == 1 + K else 2 for q in query_lens]
        assert segment == sorted(segment)
