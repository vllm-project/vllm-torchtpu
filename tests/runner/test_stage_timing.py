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
"""Stage metrics must retain event time without waiting on the serving thread."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_torchtpu.runner import stage_timing


@pytest.fixture
def events(monkeypatch):
    created = []

    def make_event(*, enable_timing):
        assert enable_timing
        event = MagicMock(spec=["record", "query", "elapsed_time", "synchronize"])
        event.query.return_value = False
        event.synchronize.side_effect = AssertionError("Timing must not synchronize")
        created.append(event)
        return event

    factory = MagicMock(side_effect=make_event)
    factory.events = created
    monkeypatch.setattr(stage_timing, "TpuEvent", factory)
    return factory


@pytest.mark.parametrize("milliseconds", [0.0, 12.5])
def test_completed_event_time_does_not_depend_on_collection_delay(
    events, monkeypatch, milliseconds
):
    clock = iter([100.0, 100.25])
    monkeypatch.setattr(
        stage_timing, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )
    timing = stage_timing.RunnerStageTiming(3)
    events.assert_not_called()
    with timing.stage("main_forward"):
        pass
    # Each stage owns two events; collecting reads their recorded interval,
    # without taking another host clock reading or synchronizing either event.
    start, end = events.events
    end.query.return_value = True
    start.elapsed_time.return_value = milliseconds
    assert timing.take_samples() == (
        [("3", "main_forward", 0.25)],
        [("3", "main_forward", milliseconds / 1000)],
    )
    start.elapsed_time.assert_called_once_with(end)
    assert timing.take_samples() == ([], [])


def test_incomplete_event_is_retained_without_waiting(events):
    timing = stage_timing.RunnerStageTiming(0)
    with timing.stage("sample"):
        pass
    start, end = events.events
    cpu, device = timing.take_samples()
    assert [sample[:2] for sample in cpu] == [("0", "sample")]
    assert device == []
    start.elapsed_time.assert_not_called()
    end.query.return_value = True
    start.elapsed_time.return_value = 2.0
    assert timing.take_samples() == ([], [("0", "sample", 0.002)])


def test_active_nested_stage_counts_toward_capacity(events, monkeypatch):
    monkeypatch.setattr(
        stage_timing.RunnerStageTiming, "MAX_IN_FLIGHT_DEVICE_TIMINGS", 1
    )
    timing = stage_timing.RunnerStageTiming(0)
    with timing.stage("main_forward"):
        with timing.stage("sample"):
            pass
        cpu, device = timing.take_samples()
        assert [sample[:2] for sample in cpu] == [("0", "sample")]
        assert device == []
        events.events[1].query.assert_not_called()
    # The queued outer stage still occupies the slot after its context closes.
    with timing.stage("draft_propose"):
        pass
    assert events.call_count == 2
    start, end = events.events
    start.elapsed_time.return_value = 3.0
    end.query.return_value = True
    cpu, device = timing.take_samples()
    assert [sample[:2] for sample in cpu] == [
        ("0", "main_forward"),
        ("0", "draft_propose"),
    ]
    assert device == [("0", "main_forward", 0.003)]
    with timing.stage("sample"):
        pass
    assert events.call_count == 4


def test_stage_exception_still_records_end_event(events):
    timing = stage_timing.RunnerStageTiming(0)
    with (
        pytest.raises(ValueError, match="stage body"),
        timing.stage("main_forward"),
    ):
        raise ValueError("stage body")
    start, end = events.events
    end.record.assert_called_once_with()
    end.query.return_value = True
    start.elapsed_time.return_value = 4.0
    cpu, device = timing.take_samples()
    assert [sample[:2] for sample in cpu] == [("0", "main_forward")]
    assert device == [("0", "main_forward", 0.004)]


def test_completed_stages_are_drained_in_record_order(events):
    timing = stage_timing.RunnerStageTiming(0)
    for stage in ("main_forward", "sample"):
        with timing.stage(stage):
            pass
    first_start, first_end, second_start, second_end = events.events
    second_end.query.return_value = True
    assert timing.take_samples()[1] == []
    first_end.query.return_value = True
    first_start.elapsed_time.return_value = 1.0
    second_start.elapsed_time.return_value = 2.0
    assert timing.take_samples() == (
        [],
        [("0", "main_forward", 0.001), ("0", "sample", 0.002)],
    )
