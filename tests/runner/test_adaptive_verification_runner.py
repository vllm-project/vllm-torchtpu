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
trimmed drafts, confidence bookkeeping, and the trimmed-step warmup plan."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID
from vllm_torchtpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
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
    assert rejected.cpu().tolist() == expected
    assert num_draft_per_req == {0: 0, 1: 0}


def test_confidences_map_to_their_chunks_requests(device):
    runner = _runner(device, ["x", "a", "b", "c"], {})
    runner.drafter = mock.MagicMock(spec=DSparkProposer)
    # Bucketed chunks of 8 rows: rows past each chunk's requests are padding.
    runner.drafter.take_draft_confidences.return_value = DraftConfidences(
        [torch.full((8, K), 0.5), torch.full((8, K), 0.25)], [(1, 2), (3, 1)]
    )
    runner._record_draft_confidences()
    manager = runner._adaptive_verification
    manager._drain(keep=0)
    assert set(manager._confidences) == {"a", "b", "c"}
    np.testing.assert_allclose(manager._confidences["a"], [0.5] * K)
    np.testing.assert_allclose(manager._confidences["c"], [0.25] * K)


def test_a_step_without_proposal_records_nothing(device):
    runner = _runner(device, ["a"], {})
    runner.drafter = mock.MagicMock(spec=DSparkProposer)
    runner.drafter.take_draft_confidences.return_value = None
    runner._record_draft_confidences()
    assert not runner._adaptive_verification._pending


AGENTX = dict(K=5, block=5, max_seqs=8, budget=256, paddings=[16, 32, 64, 128, 256])
FLASH = dict(
    K=5, block=5, max_seqs=64, budget=1024, paddings=[16, 32, 64, 128, 256, 512, 1024]
)


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


@pytest.mark.parametrize("cfg", [AGENTX, FLASH], ids=["agentx", "flash"])
def test_warmup_plan_covers_every_trimmed_step_once(cfg):
    runner = _planning_runner(**cfg)
    plan = TPUModelRunner._adaptive_verification_warmup_plan(runner)
    counts = [
        n
        for n in range(1, cfg["max_seqs"] + 1)
        if n * (1 + cfg["K"]) <= min(cfg["paddings"][-1], cfg["budget"])
    ]
    full = {_key(cfg, runner, n, n * cfg["K"]) for n in counts}
    reachable = {
        _key(cfg, runner, n, b) for n in counts for b in range(n * cfg["K"] + 1)
    }
    planned = [_key(cfg, runner, n, b) for n, b in plan]
    # The full-window warmup already compiled `full`; the plan adds every
    # other bucket a trimmed step reaches, each exactly once.
    assert len(planned) == len(set(planned))
    assert not set(planned) & full
    assert full | set(planned) == reachable
    # Every step that verifies nothing takes the non-spec path: warmed too.
    assert any(not k[0] for k in planned)


@pytest.mark.parametrize("cfg", [AGENTX, FLASH], ids=["agentx", "flash"])
def test_profile_plan_samples_both_sides_of_every_bucket_edge(cfg):
    K, pads = cfg["K"], cfg["paddings"]
    runner = _planning_runner(**cfg)
    plan = TPUModelRunner._adaptive_verification_profile_plan(runner)
    max_window = min(pads[-1], cfg["budget"])
    grid = sorted({n for n, _, _ in plan})
    assert grid[0] == 1 and grid[-1] == min(cfg["max_seqs"], max_window)
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
