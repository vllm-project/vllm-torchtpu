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
"""CPU contracts for DSpark adaptive verification's budget, split and trim."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    StepCostTable,
    allocate_draft_budget,
    choose_draft_budget,
    survival_probabilities,
    trim_scheduled_drafts,
    validate_adaptive_verification_config,
)

K = 4


def _table(bucket_ms: dict[int, float], grid=(1, 2)) -> StepCostTable:
    """A table whose steps cost a flat ``bucket_ms`` per token bucket,
    sampled where the profile samples: at both ends of each bucket a decode
    step of each grid request count reaches."""
    buckets = sorted(bucket_ms)
    samples = []
    for num_reqs in grid:
        lo, hi = num_reqs, min(num_reqs * (1 + K), buckets[-1])
        points = {lo, hi}
        points.update(t for b in buckets if lo <= b < hi for t in (b, b + 1))
        for t in sorted(points):
            bucket = next(b for b in buckets if t <= b)
            samples.append((num_reqs, t, bucket_ms[bucket]))
    return StepCostTable.from_samples(buckets, samples, [(16, 1.0)])


# Two token buckets: a step costs 10 ms up to 8 tokens and 20 ms up to 16.
COSTS = _table({8: 10.0, 16: 20.0})


def _decode_step(drafts: dict[str, int], prefill: dict[str, int] | None = None):
    so = SchedulerOutput.make_empty()
    so.num_scheduled_tokens = {r: 1 + n for r, n in drafts.items()}
    so.num_scheduled_tokens.update(prefill or {})
    so.scheduled_spec_decode_tokens = {r: [-1] * n for r, n in drafts.items() if n}
    so.total_num_scheduled_tokens = sum(so.num_scheduled_tokens.values())
    return so


def _manager(async_scheduling=False, costs: StepCostTable | None = COSTS):
    manager = AdaptiveVerificationManager(
        K,
        draft_block_size=K,
        async_scheduling=async_scheduling,
        calibrate_overhead=False,
    )
    manager.cost_table = costs
    return manager


def _record(manager, confidences: dict[str, float], padded_rows=8):
    # A bucketed chunk: rows past the live requests are padding.
    chunk = torch.zeros(padded_rows, K)
    chunk[: len(confidences)] = torch.tensor(list(confidences.values()))[:, None]
    manager.record_confidences(list(confidences), [chunk], [len(confidences)])


def test_survival_masks_unscheduled_slots():
    survival = survival_probabilities(np.full((2, K), 0.5), np.array([K, 2]))
    np.testing.assert_allclose(survival[0], [0.5, 0.25, 0.125, 0.0625])
    assert np.isneginf(survival[1, 2:]).all()


@pytest.mark.parametrize("budget,expected", [(0, [0, 0]), (3, [3, 0]), (5, [4, 1])])
def test_allocation_admits_the_globally_best_slots(budget, expected):
    survival = survival_probabilities(
        np.array([[0.9] * K, [0.5] * K]), np.array([K, K])
    )
    assert allocate_draft_budget(survival, budget).tolist() == expected


@pytest.mark.parametrize(
    "confidence,costs,expected",
    [
        # Crossing into the 16-token bucket doubles the step for two more
        # drafts; stop at the 8-token edge (2 bonus + 6 drafts).
        (0.9, COSTS, 6),
        # One bucket: every draft is free, verify all of them.
        (0.9, _table({16: 10.0}), 2 * K),
        # Drafts that never survive are not worth any slot.
        (0.0, COSTS, 0),
    ],
)
def test_budget_maximizes_expected_tokens_per_ms(confidence, costs, expected):
    survival = survival_probabilities(np.full((2, K), confidence), np.array([K, K]))
    budget = choose_draft_budget(
        survival,
        num_sampling_reqs=2,
        num_non_draft_tokens=2,
        fixed_ms=costs.draft_cost(2 * K),
        costs=costs,
    )
    assert budget == expected


def test_cost_follows_real_tokens_within_a_bucket():
    # Padding rows cost little when the MoE skips them: 2 requests verify
    # 2..8 tokens in the 8-token bucket and 9..10 in the 16-token one.
    table = StepCostTable.from_samples(
        [8, 16], [(2, 2, 10.0), (2, 8, 14.0), (2, 9, 20.0), (2, 10, 21.0)], [(8, 1.0)]
    )
    assert table.max_tokens == 10
    np.testing.assert_allclose(
        table.verify_cost(2, np.array([2, 5, 8, 9, 10])), [10.0, 12.0, 14.0, 20.0, 21.0]
    )


def test_full_windows_may_undercut_a_trimmed_step():
    # Full windows take the static attention pass, partial ones the slower
    # mixed pass: the table keeps the full step's lower cost.
    table = StepCostTable.from_samples(
        [16], [(2, 10, 5.0), (2, 2, 6.0), (2, 6, 7.0)], [(16, 1.0), (8, 2.0)]
    )
    np.testing.assert_allclose(
        table.verify_cost(2, np.array([2, 6, 10])), [6.0, 7.0, 5.0]
    )
    survival = survival_probabilities(np.full((2, K), 0.5), np.array([K, K]))
    budget = choose_draft_budget(
        survival, num_sampling_reqs=2, num_non_draft_tokens=2, fixed_ms=1.0, costs=table
    )
    assert budget == 2 * K
    # Draft costs still flatten noise: a larger bucket never costs less.
    assert table.draft_ms.tolist() == [2.0, 2.0]


def test_cost_interpolates_between_request_counts():
    # 2 requests profile 2..10 tokens, 4 requests 4..20 (K = 4).
    table = StepCostTable.from_samples(
        [32],
        [(2, 2, 10.0), (2, 10, 10.0), (4, 4, 20.0), (4, 20, 20.0)],
        [(16, 1.0)],
    )
    # 3 requests: halfway where both cover the tokens, else the one that does.
    np.testing.assert_allclose(
        table.verify_cost(3, np.array([3, 6, 12])), [10.0, 15.0, 20.0]
    )
    # Past either end of the grid, the nearest entry answers.
    assert table.verify_cost(1, np.array([2])).tolist() == [10.0]
    assert table.verify_cost(8, np.array([20])).tolist() == [20.0]


def test_trim_leaves_the_scheduler_copy_untouched():
    so = _decode_step({"a": K, "b": K})
    trimmed = trim_scheduled_drafts(so, {"a": 2, "b": 0})
    assert trimmed.num_scheduled_tokens == {"a": 3, "b": 1}
    assert trimmed.scheduled_spec_decode_tokens == {"a": [-1, -1]}
    assert trimmed.total_num_scheduled_tokens == 4
    # The scheduler rolls trimmed drafts back from what it scheduled.
    assert so.num_scheduled_tokens == {"a": 1 + K, "b": 1 + K}
    assert so.total_num_scheduled_tokens == 2 * (1 + K)
    assert len(so.scheduled_spec_decode_tokens["b"]) == K


def test_manager_trims_the_least_confident_drafts():
    manager = _manager()
    _record(manager, {"a": 0.9, "b": 0.1})
    so = _decode_step({"a": K, "b": K})
    trimmed = manager.trim(so)
    assert trimmed.num_scheduled_tokens == {"a": 1 + K, "b": 3}
    assert manager.step_trimmed == {"b": 2}


@pytest.mark.parametrize("async_scheduling,trimmed_req", [(False, "b"), (True, "a")])
def test_async_decisions_read_the_previous_proposal(async_scheduling, trimmed_req):
    # Under async scheduling the newest copy holds drafts still on the
    # device; the decision may only read the proposal before it.
    manager = _manager(async_scheduling=async_scheduling)
    _record(manager, {"a": 0.1, "b": 0.9})
    _record(manager, {"a": 0.99, "b": 0.9})
    manager.trim(_decode_step({"a": K, "b": K}))
    assert manager.step_trimmed == {trimmed_req: 2}


@pytest.mark.parametrize(
    "case", ["structured_output", "prefill", "no_cost_table", "multi_chunk"]
)
def test_manager_leaves_steps_it_cannot_trim(case):
    manager = _manager(costs=None if case == "no_cost_table" else COSTS)
    _record(manager, {"a": 0.1, "b": 0.1})
    drafts = {"a": K, "b": K}
    so = _decode_step(drafts, prefill={"p": 5} if case == "prefill" else None)
    if case == "structured_output":
        so.has_structured_output_requests = True
    if case == "multi_chunk":
        manager.cost_table = _table({8: 1.0})
    assert manager.trim(so) is so
    assert manager.step_trimmed == {}


@pytest.mark.parametrize(
    "room,num_tokens,step_trimmed,changed",
    [
        # The DP-wide 16-token bucket has room for both trimmed drafts.
        (16 - 8, {"a": 1 + K, "b": 1 + K}, {}, ["b"]),
        (1, {"a": 1 + K, "b": 4}, {"b": 1}, ["b"]),
    ],
)
def test_expand_readmits_the_next_best_drafts(room, num_tokens, step_trimmed, changed):
    manager = _manager()
    _record(manager, {"a": 0.9, "b": 0.1})
    trimmed = manager.trim(_decode_step({"a": K, "b": K}))
    assert trimmed.total_num_scheduled_tokens == 8
    expanded = manager.expand(room)
    assert expanded is not None
    scheduler_output, changed_req_ids = expanded
    assert scheduler_output.num_scheduled_tokens == num_tokens
    assert manager.step_trimmed == step_trimmed
    assert changed_req_ids == changed


def test_expand_counts_the_readmitted_drafts_toward_the_step():
    # Calibration times the step against the tokens it really verified.
    manager = _manager()
    _record(manager, {"a": 0.9, "b": 0.1})
    manager.trim(_decode_step({"a": K, "b": K}))
    assert manager._last_step is not None and manager._last_step[2] == 8
    manager.expand(1)
    assert manager._last_step[2] == 9
    # Room past the trimmed drafts carries nothing of this rank's.
    manager.expand(8)
    assert manager._last_step[2] == 2 * (1 + K)


@pytest.mark.parametrize("case", ["untrimmed", "no_room", "warmup"])
def test_expand_changes_nothing_it_should_not(case):
    manager = _manager(costs=None if case == "warmup" else COSTS)
    _record(manager, {"a": 0.9, "b": 0.1})
    if case == "warmup":
        manager.forced_budget = 2
    drafts = {"a": 1, "b": 1} if case == "untrimmed" else {"a": K, "b": K}
    manager.trim(_decode_step(drafts))
    assert manager.expand(0 if case == "no_room" else 8) is None


def test_stats_count_only_steps_that_stay_trimmed():
    manager = _manager()
    _record(manager, {"a": 0.9, "b": 0.1})
    manager.trim(_decode_step({"a": K, "b": K}))
    manager.trim(_decode_step({"a": K, "b": K}))
    # The second step's bucket turns out to hold every draft after all.
    manager.expand(8)
    assert manager._num_decisions == 2
    assert manager._num_trimmed_steps == 1
    assert manager._num_verified == (2 * K - 2) + 2 * K


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_back_to_back_decode_steps_calibrate_the_step_overhead():
    manager = _manager()
    manager.calibrate_overhead = True
    clock = manager._clock = _Clock()
    _record(manager, {"a": 0.9, "b": 0.9})
    # 4 tokens: the 8-token bucket's 10 ms plus 1 ms of drafting.
    step = _decode_step({"a": 1, "b": 1})
    manager.trim(step)
    clock.now += 0.031
    manager.trim(step)
    one_sample = 0.05 * 20.0
    assert manager.overhead_ms == pytest.approx(one_sample)
    # A stall is not a decode step.
    clock.now += 10.0
    manager.trim(step)
    assert manager.overhead_ms == pytest.approx(one_sample)
    # Nor is the gap across a step with prefill.
    manager.trim(_decode_step({"a": 1}, prefill={"p": 5}))
    clock.now += 0.031
    manager.trim(step)
    assert manager.overhead_ms == pytest.approx(one_sample)


def test_calibration_keeps_negative_samples_and_skips_catch_up():
    manager = _manager()
    manager.calibrate_overhead = True
    clock = manager._clock = _Clock()
    _record(manager, {"a": 0.9, "b": 0.9})
    step = _decode_step({"a": 1, "b": 1})
    manager.trim(step)
    # Steps cheaper than the table: the table overestimates by 5 ms.
    clock.now += 0.006
    manager.trim(step)
    one_sample = 0.05 * -5.0
    assert manager.overhead_ms == pytest.approx(one_sample)
    # A gap a quarter of the predicted step is the host catching up.
    clock.now += 0.001
    manager.trim(step)
    assert manager.overhead_ms == pytest.approx(one_sample)


def test_without_calibration_the_overhead_stays_put():
    manager = _manager()
    clock = manager._clock = _Clock()
    step = _decode_step({"a": 1, "b": 1})
    manager.trim(step)
    clock.now += 0.031
    manager.trim(step)
    assert manager.overhead_ms == 0.0


def test_step_overhead_makes_trimming_pay_less():
    manager = _manager()
    _record(manager, {"a": 0.9, "b": 0.5})
    manager.trim(_decode_step({"a": K, "b": K}))
    assert manager.step_trimmed == {"b": 2}
    # When every step costs a second anyway, a smaller bucket saves little.
    manager.overhead_ms = 1000.0
    manager.trim(_decode_step({"a": K, "b": K}))
    assert manager.step_trimmed == {}


def test_every_step_drains_copies_and_forgets_unscheduled_requests():
    # Finished requests never come back; their confidences must not linger,
    # even when no step is trimmable.
    manager = _manager()
    _record(manager, {"a": 0.1, "b": 0.9})
    manager.trim(_decode_step({}, prefill={"p": 5}))
    assert not manager._pending
    assert not manager._confidences


def test_forced_budget_overrides_the_cost_model():
    manager = _manager(costs=None)
    manager.forced_budget = 3
    trimmed = manager.trim(_decode_step({"a": K, "b": K}))
    assert sum(len(d) for d in trimmed.scheduled_spec_decode_tokens.values()) == 3
    assert sum(manager.step_trimmed.values()) == 2 * K - 3


def test_reset_drains_in_flight_copies_and_forgets_requests():
    # After the warmup, no copy may stay in flight and no synthetic request
    # may keep steering the split.
    manager = _manager(async_scheduling=True)
    _record(manager, {"a": 0.1})
    _record(manager, {"a": 0.1})
    manager.reset()
    assert not manager._pending
    assert not manager._confidences


def _vllm_config(**overrides):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=True),
        lora_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1
        ),
        model_config=SimpleNamespace(enforce_eager=False),
    )
    for name, value in overrides.items():
        if name == "lora_config":
            config.lora_config = value
        elif name == "enforce_eager":
            config.model_config.enforce_eager = value
        else:
            setattr(config.parallel_config, name, value)
    return config


def test_supported_config_validates():
    validate_adaptive_verification_config(_vllm_config())


@pytest.mark.parametrize(
    "override,error,match",
    [
        ({"lora_config": object()}, ValueError, "LoRA"),
        ({"pipeline_parallel_size": 2}, ValueError, "pipeline"),
        (
            {"prefill_context_parallel_size": 2},
            NotImplementedError,
            "prefill context",
        ),
        ({"enforce_eager": True}, ValueError, "enforce_eager"),
    ],
)
def test_unsupported_config_is_rejected(override, error, match):
    with pytest.raises(error, match=match):
        validate_adaptive_verification_config(_vllm_config(**override))
