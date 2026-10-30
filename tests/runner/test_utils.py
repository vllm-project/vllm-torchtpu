# Copyright 2025 Google LLC
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
from unittest.mock import MagicMock, mock_open, patch

import pytest

from vllm_torchtpu import profiler_trace
from vllm_torchtpu.runner.utils import (
    PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR,
    InferencePhase,
    PhaseBasedProfiler,
    config_list,
    decode_dp_profile_stats,
    determine_dp_phase,
    determine_phase_from_batch_composition_stats,
    encode_dp_profile_stats,
    merge_dp_profile_stats,
)


@pytest.fixture
def profiler_fixture(tmp_path):
    """Fixture to set up a PhaseBasedProfiler with mocked dependencies."""
    target_module = "vllm_torchtpu.runner.utils"
    with (
        patch("torch.profiler.profile") as mock_profile,
        patch("builtins.open", mock_open()) as mock_file,
        patch(f"{target_module}.datetime") as mock_datetime,
        patch(
            f"{target_module}.determine_phase_from_batch_composition_stats"
        ) as mock_determine_phase,
        patch.object(
            PhaseBasedProfiler,
            "_resolve_canonical_dst_ts",
            return_value="2024_01_01_12_00_00",
        ),
        patch.object(PhaseBasedProfiler, "_merge_profile_directories"),
    ):
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2024_01_01_12_00_00"
        mock_datetime.datetime.now.return_value = mock_now

        # Set up mock for the context manager returned by profile()
        mock_context = MagicMock()
        mock_profile.return_value = mock_context

        profiler = PhaseBasedProfiler(
            profile_dir=str(tmp_path),
            standard_opts={
                "host_tracer_level": 2,
                "device_tracer_level": 1,
                "python_tracer_level": 1,
            },
            advanced_opts={
                "tpu_trace_mode": "TRACE_COMPUTE",
                "tpu_num_sparse_cores_to_trace": 1,
                "tpu_num_sparse_core_tiles_to_trace": 1,
            },
        )
        profiler.num_steps_to_profile_for = PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR

        yield {
            "profiler": profiler,
            "mock_profile": mock_profile,
            "mock_context": mock_context,
            "mock_file": mock_file,
            "mock_determine_phase": mock_determine_phase,
        }


def test_phased_profiler_full_cycle(profiler_fixture):
    """Tests a full start-step-stop profiling cycle for one phase."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_context = profiler_fixture["mock_context"]
    mock_file = profiler_fixture["mock_file"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    stats = {"num_reqs": 2, "total_num_scheduled_tokens": 100}

    # 1. Start profiling on PREFILL_HEAVY phase
    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    profiler.step(stats)
    mock_profile.assert_called_once()
    mock_context.__enter__.assert_called_once()
    assert profiler.profiling_n_steps_left == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR
    assert profiler.current_phase == "prefill_heavy"
    assert profiler.inference_phase_seen[InferencePhase.PREFILL_HEAVY]
    assert mock_file().write.call_count == 1  # Wrote stats on start

    # 2. Step profiling (N-1 steps)
    for i in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1):
        profiler.step(stats)
        assert (
            profiler.profiling_n_steps_left
            == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1 - i
        )
        mock_profile.assert_called_once()  # Not called again
        mock_context.__exit__.assert_not_called()

    # 3. Final step stops profiling
    profiler.step(stats)
    mock_context.__exit__.assert_called_once_with(None, None, None)
    assert profiler.profiling_n_steps_left == 0
    assert profiler.current_phase == ""
    assert mock_file().write.call_count == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR + 1


def test_phased_profiler_finish_stops_a_phase_mid_capture(profiler_fixture):
    """finish() force-closes a phase before its step budget runs out; this is
    what stop_phased_profiling() calls when /stop_profile arrives early."""
    profiler = profiler_fixture["profiler"]
    mock_context = profiler_fixture["mock_context"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    profiler.step({"num_reqs": 2, "total_num_scheduled_tokens": 100})
    assert profiler.current_phase == "prefill_heavy"
    mock_context.__exit__.assert_not_called()

    profiler.finish()

    mock_context.__exit__.assert_called_once_with(None, None, None)
    assert profiler.current_phase == ""
    # Otherwise a reused profiler never starts a new phase: step() only calls
    # _start_profiling when profiling_n_steps_left <= 0.
    assert profiler.profiling_n_steps_left == 0


def test_phased_profiler_finish_when_idle_is_a_noop(profiler_fixture):
    """No phase capture in progress; finish() must not touch profile_context."""
    profiler = profiler_fixture["profiler"]
    mock_context = profiler_fixture["mock_context"]

    profiler.finish()

    mock_context.__exit__.assert_not_called()


def test_phased_profiler_ignores_steps_with_no_work(profiler_fixture):
    """A batch that scheduled nothing has no phase to attribute it to.

    determine_phase_from_batch_composition_stats divides by
    total_num_scheduled_tokens, so a zero-token step must never reach it.
    """
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY

    profiler.step({"num_reqs": 0, "total_num_scheduled_tokens": 0})
    mock_profile.assert_not_called()
    assert profiler.current_phase == ""

    # A single scheduled token is real work and must be profiled.
    profiler.step({"num_reqs": 1, "total_num_scheduled_tokens": 1})
    mock_profile.assert_called_once()


def test_determine_phase_treats_single_token_step_as_decode_only():
    """One request in decode schedules exactly one token, all of it decode."""
    assert (
        determine_phase_from_batch_composition_stats(
            {
                "total_num_scheduled_tokens": 1,
                "num_prefill_tokens": 0,
            }
        )
        == InferencePhase.DECODE_ONLY
    )


def test_phased_profiler_full_cycle_at_batch_size_one(profiler_fixture):
    """A max-num-seqs=1 server must still capture decode_only, bounded.

    Regression: the guard on step() required more than one scheduled token, so
    every steady-state step of a batch-size-1 run was dropped before reaching
    the classifier. That made decode_only unreachable, and because the same
    guard covered _step_or_stop_profiling, whichever phase was already
    capturing never counted down -- its trace stayed open until /stop_profile,
    silently ignoring max_iterations.
    """
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_context = profiler_fixture["mock_context"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    # One request in decode: exactly one scheduled token, every step.
    stats = {"num_reqs": 1, "total_num_scheduled_tokens": 1}
    mock_determine_phase.return_value = InferencePhase.DECODE_ONLY

    profiler.step(stats)
    mock_profile.assert_called_once()
    assert profiler.current_phase == "decode_only"
    assert profiler.inference_phase_seen[InferencePhase.DECODE_ONLY]
    assert profiler.profiling_n_steps_left == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR

    # The budget must count down on its own and close the capture.
    for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR):
        profiler.step(stats)

    mock_context.__exit__.assert_called_once_with(None, None, None)
    assert profiler.current_phase == ""
    assert profiler.profiling_n_steps_left == 0


def test_phased_profiler_handles_all_phases(profiler_fixture):
    """Tests that the profiler can profile all defined phases sequentially."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_context = profiler_fixture["mock_context"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    stats = {"num_reqs": 2, "total_num_scheduled_tokens": 100}
    phases_to_profile = [
        InferencePhase.PREFILL_ONLY,
        InferencePhase.PREFILL_HEAVY,
        InferencePhase.DECODE_ONLY,
        InferencePhase.DECODE_HEAVY,
        InferencePhase.BALANCED,
    ]

    for i, phase in enumerate(phases_to_profile):
        # Start profiling for the new phase
        mock_determine_phase.return_value = phase
        profiler.step(stats)
        assert mock_profile.call_count == i + 1
        assert profiler.current_phase == phase.name.lower()
        assert profiler.inference_phase_seen[phase]

        # Step until profiling stops for this phase
        for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR):
            profiler.step(stats)

        assert mock_context.__exit__.call_count == i + 1
        assert profiler.current_phase == ""

    # After all phases seen, should not start again
    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    profiler.step(stats)
    assert mock_profile.call_count == len(phases_to_profile)


def test_phased_profiler_skips_decode_steps_before_profiling(profiler_fixture):
    """Tests that the profiler skips N decode-heavy steps before profiling."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    num_steps_to_skip = 3
    profiler.num_decode_steps_to_skip = num_steps_to_skip

    stats = {"num_reqs": 2, "total_num_scheduled_tokens": 100}
    mock_determine_phase.return_value = InferencePhase.DECODE_HEAVY

    # Each of these steps should be skipped (no profiling started)
    for i in range(num_steps_to_skip):
        profiler.step(stats)
        assert profiler.decode_steps_skipped == i + 1
        mock_profile.assert_not_called()
        assert not profiler.inference_phase_seen[InferencePhase.DECODE_HEAVY]

    # The next step should actually start profiling
    profiler.step(stats)
    mock_profile.assert_called_once()
    assert profiler.inference_phase_seen[InferencePhase.DECODE_HEAVY]
    assert profiler.current_phase == "decode_heavy"
    assert profiler.profiling_n_steps_left == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR


def test_phased_profiler_skip_only_affects_decode_heavy(profiler_fixture):
    """Tests that the skip logic only applies to the DECODE_HEAVY phase."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_context = profiler_fixture["mock_context"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    profiler.num_decode_steps_to_skip = 5  # Large skip count

    stats = {"num_reqs": 2, "total_num_scheduled_tokens": 100}

    # PREFILL_HEAVY should start profiling immediately (no skipping)
    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    profiler.step(stats)
    mock_profile.assert_called_once()
    assert profiler.inference_phase_seen[InferencePhase.PREFILL_HEAVY]
    assert profiler.decode_steps_skipped == 0  # Not incremented

    # Complete the PREFILL_HEAVY profiling cycle
    for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR):
        profiler.step(stats)
    mock_context.__exit__.assert_called_once()

    # BALANCED should also start immediately (no skipping)
    mock_determine_phase.return_value = InferencePhase.BALANCED
    profiler.step(stats)
    assert mock_profile.call_count == 2
    assert profiler.inference_phase_seen[InferencePhase.BALANCED]
    assert profiler.decode_steps_skipped == 0  # Still not incremented

    # Complete the BALANCED profiling cycle
    for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR):
        profiler.step(stats)
    assert mock_context.__exit__.call_count == 2

    # DECODE_HEAVY should be skipped
    mock_determine_phase.return_value = InferencePhase.DECODE_HEAVY
    profiler.step(stats)
    assert mock_profile.call_count == 2  # Not started yet
    assert profiler.decode_steps_skipped == 1


def test_phased_profiler_skips_decode_only_steps_based_on_kv_len(
    profiler_fixture,
):
    """Tests that the profiler skips DECODE_ONLY steps until min KV len
    reaches threshold."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_context = profiler_fixture["mock_context"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    kv_len_threshold = 10
    profiler.decode_kv_len_threshold = kv_len_threshold

    stats = {"num_reqs": 2, "total_num_scheduled_tokens": 100, "min_kv_len": 5}
    mock_determine_phase.return_value = InferencePhase.DECODE_ONLY

    # Should be skipped as min_kv_len (5) < threshold (10)
    profiler.step(stats)
    mock_profile.assert_not_called()
    assert not profiler.inference_phase_seen[InferencePhase.DECODE_ONLY]

    # Should start profiling as min_kv_len (10) >= threshold (10)
    stats["min_kv_len"] = 10
    profiler.step(stats)
    mock_profile.assert_called_once()
    assert profiler.inference_phase_seen[InferencePhase.DECODE_ONLY]
    assert profiler.current_phase == "decode_only"
    assert profiler.profiling_n_steps_left == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR

    # Profiling continues
    stats["min_kv_len"] = 11
    profiler.step(stats)
    mock_profile.assert_called_once()

    # Complete the profiling cycle (1 more step already done in the step above)
    for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1):
        profiler.step(stats)
    mock_context.__exit__.assert_called_once()
    assert profiler.current_phase == ""


def test_phased_profiler_skips_prefill_only_steps_based_on_kv_len(
    profiler_fixture,
):
    """Tests that the profiler skips PREFILL_ONLY steps until min KV len
    reaches threshold."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_context = profiler_fixture["mock_context"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    kv_len_threshold = 10
    profiler.prefill_kv_len_threshold = kv_len_threshold

    stats = {"num_reqs": 1, "total_num_scheduled_tokens": 100, "min_kv_len": 0}
    mock_determine_phase.return_value = InferencePhase.PREFILL_ONLY

    # The first chunk of a chunked prefill sits at min_kv_len 0, which is
    # exactly the step the threshold exists to skip.
    profiler.step(stats)
    mock_profile.assert_not_called()
    assert not profiler.inference_phase_seen[InferencePhase.PREFILL_ONLY]

    stats["min_kv_len"] = 5
    profiler.step(stats)
    mock_profile.assert_not_called()
    assert not profiler.inference_phase_seen[InferencePhase.PREFILL_ONLY]

    # Should start profiling as min_kv_len (10) >= threshold (10)
    stats["min_kv_len"] = 10
    profiler.step(stats)
    mock_profile.assert_called_once()
    assert profiler.inference_phase_seen[InferencePhase.PREFILL_ONLY]
    assert profiler.current_phase == "prefill_only"
    assert profiler.profiling_n_steps_left == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR

    # Profiling continues
    stats["min_kv_len"] = 11
    profiler.step(stats)
    mock_profile.assert_called_once()

    # Complete the profiling cycle (1 more step already done in the step above)
    for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1):
        profiler.step(stats)
    mock_context.__exit__.assert_called_once()
    assert profiler.current_phase == ""


def test_phased_profiler_prefill_kv_len_threshold_defaults_to_off(
    profiler_fixture,
):
    """Without a threshold the capture must still arm on the first prefill step."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    assert profiler.prefill_kv_len_threshold < 0

    stats = {"num_reqs": 1, "total_num_scheduled_tokens": 100, "min_kv_len": 0}
    mock_determine_phase.return_value = InferencePhase.PREFILL_ONLY

    profiler.step(stats)
    mock_profile.assert_called_once()
    assert profiler.inference_phase_seen[InferencePhase.PREFILL_ONLY]


def test_phased_profiler_prefill_threshold_does_not_gate_prefill_heavy(
    profiler_fixture,
):
    """PREFILL_HEAVY always carries a freshly-admitted request at kv len 0, so a
    min-based threshold must not be applied to it or it would never arm."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    profiler.prefill_kv_len_threshold = 10

    stats = {"num_reqs": 2, "total_num_scheduled_tokens": 100, "min_kv_len": 0}
    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY

    profiler.step(stats)
    mock_profile.assert_called_once()
    assert profiler.inference_phase_seen[InferencePhase.PREFILL_HEAVY]


def _run_steps(profiler, num_reqs, steps):
    for _ in range(steps):
        profiler.step({"num_reqs": num_reqs, "total_num_scheduled_tokens": num_reqs})


def test_phased_profiler_target_concurrencies(profiler_fixture):
    """A target concurrency list: one capture per listed num_reqs, named
    <phase>_concur_<num_reqs>; other request counts are skipped."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]
    profiler.only_phases = frozenset({"decode_only"})
    profiler.target_concurrencies = frozenset({93, 47})
    profiler.track_concurrency = True
    mock_determine_phase.return_value = InferencePhase.DECODE_ONLY

    _run_steps(profiler, 90, 1)  # not a target
    mock_profile.assert_not_called()
    _run_steps(profiler, 93, PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR + 1)
    assert mock_profile.call_count == 1
    _run_steps(profiler, 93, 2)  # already captured
    assert mock_profile.call_count == 1
    _run_steps(profiler, 47, 1)
    assert mock_profile.call_count == 2
    assert profiler.current_phase == "decode_only_concur_47"


def test_phased_profiler_only_phases_skips_other_phases(profiler_fixture):
    """Only the listed phases are captured; once they all are, steps stop
    looking for new captures."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]
    profiler.only_phases = frozenset({"prefill_heavy"})

    mock_determine_phase.return_value = InferencePhase.DECODE_ONLY
    _run_steps(profiler, 4, 1)
    mock_profile.assert_not_called()
    assert not profiler.inference_phase_seen[InferencePhase.DECODE_ONLY]

    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    _run_steps(profiler, 4, PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR + 1)
    assert mock_profile.call_count == 1
    assert profiler.current_phase == ""
    assert profiler._all_captures_done()


def test_phased_profiler_filter_does_not_consume_the_step(profiler_fixture):
    """A step at a non-target num_reqs leaves the decode KV-len gate and the
    phase untouched for a later matching step."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]
    profiler.target_concurrencies = frozenset({8})
    profiler.track_concurrency = True
    profiler.decode_kv_len_threshold = 100
    mock_determine_phase.return_value = InferencePhase.DECODE_ONLY

    profiler.step({"num_reqs": 7, "total_num_scheduled_tokens": 7, "min_kv_len": 500})
    profiler.step({"num_reqs": 8, "total_num_scheduled_tokens": 8, "min_kv_len": 50})
    mock_profile.assert_not_called()
    profiler.step({"num_reqs": 8, "total_num_scheduled_tokens": 8, "min_kv_len": 500})
    mock_profile.assert_called_once()
    assert profiler.current_phase == "decode_only_concur_8"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"only_phases": ["decode"]}, "unknown phases"),
        ({"target_concurrencies": [0]}, "positive"),
    ],
)
def test_phased_profiler_rejects_bad_filters(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        PhaseBasedProfiler(profile_dir=str(tmp_path), **kwargs)


def test_phased_profiler_target_implies_tracking(tmp_path):
    profiler = PhaseBasedProfiler(
        profile_dir=str(tmp_path),
        only_phases=["Decode_Only ", "prefill_heavy"],
        target_concurrencies=[93],
    )
    assert profiler.track_concurrency
    assert profiler.only_phases == {"decode_only", "prefill_heavy"}


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ("", []),
        ("decode_only, prefill_heavy", ["decode_only", "prefill_heavy"]),
        (["decode_only"], ["decode_only"]),
        ([93, 47], ["93", "47"]),
    ],
)
def test_config_list(value, expected):
    assert config_list(value) == expected


def _stage_rank_capture(rank_dir, ts_name, filename, content):
    """Helper: simulate writing one xplane file under rank_N/plugins/profile/<ts>/."""
    ts_dir = rank_dir / "plugins" / "profile" / ts_name
    ts_dir.mkdir(parents=True)
    (ts_dir / filename).write_text(content)
    return ts_dir


def test_resolve_canonical_dst_ts_rank_zero_writes_marker(tmp_path):
    """Rank 0 picks the canonical ts and publishes it for the other ranks."""
    profiler = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=0)
    phase_dir = tmp_path / "prefill_heavy"
    phase_dir.mkdir()

    ts = profiler._resolve_canonical_dst_ts(str(phase_dir))

    marker = phase_dir / f".canonical_ts_{profiler_trace.profile_session_id()}"
    assert marker.exists()
    assert marker.read_text().strip() == ts


def test_resolve_canonical_dst_ts_non_zero_rank_reads_marker(tmp_path):
    """Non-zero rank reads whatever rank 0 already published."""
    phase_dir = tmp_path / "prefill_heavy"
    phase_dir.mkdir()
    marker = phase_dir / f".canonical_ts_{profiler_trace.profile_session_id()}"
    marker.write_text("2026_05_06_04_47_36")

    profiler = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=2)
    ts = profiler._resolve_canonical_dst_ts(str(phase_dir))

    assert ts == "2026_05_06_04_47_36"


def test_resolve_canonical_dst_ts_non_zero_rank_falls_back_on_timeout(
    tmp_path, monkeypatch
):
    """When rank 0 never publishes, a non-zero rank falls back to own ts."""
    phase_dir = tmp_path / "prefill_heavy"
    phase_dir.mkdir()

    profiler = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=1)
    # Speed the test up by shrinking the timeout.
    monkeypatch.setattr(profiler_trace, "CANONICAL_TS_POLL_TIMEOUT_S", 0.1)
    monkeypatch.setattr(profiler_trace, "CANONICAL_TS_POLL_INTERVAL_S", 0.02)

    ts = profiler._resolve_canonical_dst_ts(str(phase_dir))

    # Format check: looks like a strftime("%Y_%m_%d_%H_%M_%S") string.
    import datetime as _dt

    _dt.datetime.strptime(ts, "%Y_%m_%d_%H_%M_%S")


def test_merge_profile_directories_single_rank(tmp_path):
    """Single rank, non-MPMD: capture moves from sandbox to
    <phase>/plugins/profile/<canonical_ts>/."""
    profiler = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=0)
    phase_dir = tmp_path / "prefill_heavy"
    rank_dir = phase_dir / "rank_0"
    rank_dir.mkdir(parents=True)
    profiler.profile_dir_with_phase_suffix = str(rank_dir)
    profiler._canonical_dst_ts = "2026_05_06_04_47_36"

    # We need to create the actual directory structure for listdir to work in test
    _stage_rank_capture(
        rank_dir, "2026_05_06_04_47_36_pt", "t1v-n-host-w-0.xplane.pb", "data_0"
    )
    # A phase always writes stats into the sandbox before it captures.
    (rank_dir / "batch_composition_stats_1.json").write_text("{}")

    profiler._merge_profile_directories()

    dst = phase_dir / "plugins" / "profile" / "2026_05_06_04_47_36"
    assert (dst / "t1v-n-host-w-0.xplane.pb").read_text() == "data_0"
    # Sandbox plugins/ subtree cleaned up; rank_0/ itself remains for stats.
    assert not (rank_dir / "plugins").exists()
    assert rank_dir.exists()


def test_merge_profile_directories_clears_marker_on_rank_zero(tmp_path):
    """Rank 0 drops the marker on merge, so a later run cannot inherit it."""
    phase_dir = tmp_path / "prefill_heavy"
    phase_dir.mkdir()

    profilers = {}
    # Rank 0 resolves first so it publishes the marker the other rank reads.
    for rank in (0, 1):
        profiler = PhaseBasedProfiler(
            profile_dir=str(tmp_path), worker_rank=rank, world_size=2
        )
        rank_dir = phase_dir / f"rank_{rank}"
        rank_dir.mkdir()
        profiler.profile_dir_with_phase_suffix = str(rank_dir)
        profiler._canonical_dst_ts = profiler._resolve_canonical_dst_ts(str(phase_dir))
        _stage_rank_capture(
            rank_dir, f"pt_ts_{rank}", "t1v-n-host-w-0.xplane.pb", f"rank_{rank}"
        )
        profilers[rank] = profiler

    assert profilers[1]._canonical_dst_ts == profilers[0]._canonical_dst_ts

    # A non-zero rank merging first must not pull the marker out from under
    # ranks that have not read it yet.
    profilers[1]._merge_profile_directories()
    assert list(phase_dir.glob(".canonical_ts_*"))

    profilers[0]._merge_profile_directories()
    assert not list(phase_dir.glob(".canonical_ts_*"))


def test_merge_profile_directories_multiworker(tmp_path):
    """4 ranks captured to their own sandboxes with identical filenames; each
    moves to the SAME canonical ts dir with a rank{N}_ prefix.

    DP=4/TP=1 and TP=4/DP=1 are the same case here: the worker hands the
    profiler the slice-global world size either way, so there is one path to
    cover rather than one per parallelism mode.
    """
    phase_dir = tmp_path / "prefill_heavy"

    canonical_ts = "2026_05_06_04_47_36"
    profilers = []
    for rank in range(4):
        profiler = PhaseBasedProfiler(
            profile_dir=str(tmp_path), worker_rank=rank, world_size=4
        )
        rank_dir = phase_dir / f"rank_{rank}"
        rank_dir.mkdir(parents=True)
        profiler.profile_dir_with_phase_suffix = str(rank_dir)
        profiler._canonical_dst_ts = canonical_ts
        _stage_rank_capture(
            rank_dir,
            f"pt_ts_{rank}",
            "t1v-n-host-w-0.xplane.pb",
            f"rank_{rank}_xplane",
        )
        profilers.append(profiler)

    for rank, profiler in enumerate(profilers):
        profiler._merge_profile_directories()

    dst = phase_dir / "plugins" / "profile" / canonical_ts
    for rank in range(4):
        assert (
            dst / f"rank{rank}_t1v-n-host-w-0.xplane.pb"
        ).read_text() == f"rank_{rank}_xplane"
        rank_dir = phase_dir / f"rank_{rank}"
        assert not (rank_dir / "plugins").exists()


def _local(num_reqs, prefill, total, min_kv):
    return {
        "num_reqs": num_reqs,
        "num_prefill_tokens": prefill,
        "total_num_scheduled_tokens": total,
        "min_kv_len": min_kv,
    }


def _dp_reduce(*per_rank):
    """What the runner's MAX all-reduce makes of each rank's encoding."""
    stats = decode_dp_profile_stats([max(col) for col in zip(*per_rank)])
    stats["dp_num_tokens_max"] = 93
    stats["dp_padded_num_tokens"] = 128
    return stats


def test_dp_profile_stats_decode_only_at_the_fullest_engine():
    """Decode-only on every busy rank: the step is decode_only, keyed on the
    largest engine's num_reqs; an idle rank changes neither phase nor KV."""
    dp = _dp_reduce(
        encode_dp_profile_stats(_local(93, 0, 93, 8300), armed=True),
        encode_dp_profile_stats(_local(92, 0, 92, 8260), armed=True),
        encode_dp_profile_stats(None, armed=True),
    )
    assert not dp["dp_trigger_blocked"]
    assert dp["dp_num_reqs_max"] == 93 and dp["dp_num_reqs_min"] == 0
    assert dp["dp_min_kv_len"] == 8260
    stats = merge_dp_profile_stats({"batch_id": 7, "num_reqs": 92}, dp)
    assert stats["phase"] == "DECODE_ONLY"
    assert stats["num_reqs"] == 93 and stats["min_kv_len"] == 8260
    assert stats["padded_total_num_scheduled_tokens"] == 128
    assert stats["local"] == {"batch_id": 7, "num_reqs": 92}


def test_dp_profile_stats_another_ranks_prefill_spoils_decode_only():
    """A rank decoding 93 requests is not in a decode_only step when another
    rank prefills in it: every rank runs that rank's padded step."""
    dp = _dp_reduce(
        encode_dp_profile_stats(_local(93, 0, 93, 8300), armed=True),
        encode_dp_profile_stats(_local(93, 931, 1023, 0), armed=True),
    )
    assert (
        determine_dp_phase(dp["dp_prefill_permille_min"], dp["dp_prefill_permille_max"])
        is InferencePhase.AMBIGUOUS
    )


@pytest.mark.parametrize(
    "ranks, phase",
    [
        ([(93, 931, 1024), (93, 960, 1024)], InferencePhase.PREFILL_HEAVY),
        ([(93, 931, 1024), (93, 0, 93)], InferencePhase.AMBIGUOUS),
        ([(8, 1024, 1024), (8, 1024, 1024)], InferencePhase.PREFILL_ONLY),
        ([(93, 10, 103), (93, 0, 93)], InferencePhase.DECODE_HEAVY),
        ([(50, 500, 1000), (50, 450, 1000)], InferencePhase.BALANCED),
    ],
)
def test_dp_phase_holds_only_on_every_busy_rank(ranks, phase):
    dp = _dp_reduce(
        *(encode_dp_profile_stats(_local(n, p, t, 0), armed=True) for n, p, t in ranks)
    )
    assert determine_phase_from_batch_composition_stats(dp) is phase


def test_dp_profile_stats_unarmed_rank_blocks_the_start(profiler_fixture):
    """Arming reaches ranks at different steps; until every rank is armed no
    capture may start, or the ranks would capture on different steps."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    profiler_fixture["mock_determine_phase"].return_value = InferencePhase.DECODE_ONLY
    dp = _dp_reduce(
        encode_dp_profile_stats(_local(93, 0, 93, 8300), armed=True),
        encode_dp_profile_stats(_local(93, 0, 93, 8300), armed=False),
    )
    assert dp["dp_trigger_blocked"]
    profiler.step(merge_dp_profile_stats({"batch_id": 1}, dp))
    mock_profile.assert_not_called()
    assert not profiler.inference_phase_seen[InferencePhase.DECODE_ONLY]


@pytest.mark.parametrize("rank, captures", [(0, True), (8, False)])
def test_phased_profiler_capture_ranks(profiler_fixture, rank, captures):
    """Every rank decides and records the phase as seen; only the listed ranks
    start the profiler."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    profiler_fixture["mock_determine_phase"].return_value = InferencePhase.DECODE_ONLY
    profiler.worker_rank = rank
    profiler.capture_ranks = frozenset({0})
    profiler.captures_here = rank in profiler.capture_ranks

    profiler.step({"num_reqs": 93, "total_num_scheduled_tokens": 93})
    assert profiler.inference_phase_seen[InferencePhase.DECODE_ONLY]
    assert mock_profile.called is captures
    assert (profiler.current_phase == "decode_only") is captures
    profiler.step({"num_reqs": 93, "total_num_scheduled_tokens": 93})
    assert mock_profile.call_count == int(captures)


def test_phased_profiler_per_phase_target_concurrencies(tmp_path):
    """The map form: decode_only only at 93 requests; prefill_only, left out of
    the map, once at any request count (one request per engine when a wave's
    first prompts prefill)."""
    with (
        patch("torch.profiler.profile") as mock_profile,
        patch("builtins.open", mock_open()),
        patch.object(
            PhaseBasedProfiler, "_resolve_canonical_dst_ts", return_value="ts"
        ),
        patch.object(PhaseBasedProfiler, "_merge_profile_directories"),
    ):
        profiler = PhaseBasedProfiler(
            profile_dir=str(tmp_path),
            num_steps_to_profile_for=2,
            standard_opts={
                "host_tracer_level": 2,
                "device_tracer_level": 1,
                "python_tracer_level": 1,
            },
            only_phases=["decode_only", "prefill_only"],
            target_concurrencies={"decode_only": [93]},
        )
        assert profiler._targets(InferencePhase.PREFILL_ONLY) == frozenset()
        prefill = {
            "num_reqs": 1,
            "num_prefill_tokens": 1024,
            "total_num_scheduled_tokens": 1024,
            "min_kv_len": 0,
        }
        decode = {
            "num_reqs": 92,
            "num_prefill_tokens": 0,
            "total_num_scheduled_tokens": 92,
            "min_kv_len": 9000,
        }

        profiler.step(prefill)
        assert profiler.current_phase == "prefill_only"
        for _ in range(2):
            profiler.step(prefill)
        assert profiler.current_phase == ""
        profiler.step(dict(prefill, num_reqs=2))  # once, not once per num_reqs
        assert mock_profile.call_count == 1

        profiler.step(decode)  # not the target
        assert mock_profile.call_count == 1
        assert not profiler._all_captures_done()
        profiler.step(dict(decode, num_reqs=93, total_num_scheduled_tokens=93))
        assert profiler.current_phase == "decode_only_concur_93"
        for _ in range(2):
            profiler.step(dict(decode, num_reqs=93, total_num_scheduled_tokens=93))
        assert profiler._all_captures_done()
        assert mock_profile.call_count == 2


def test_phased_profiler_rejects_unknown_phase_in_target_map(tmp_path):
    with pytest.raises(ValueError, match="unknown phase"):
        PhaseBasedProfiler(
            profile_dir=str(tmp_path), target_concurrencies={"decode": [93]}
        )


def test_phased_profiler_stops_a_capture_when_the_phase_ends(profiler_fixture):
    """With stop_on_phase_change, the capture ends before the first step that
    left the phase, so every captured step is the phase's."""
    profiler = profiler_fixture["profiler"]
    mock_context = profiler_fixture["mock_context"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]
    profiler.stop_on_phase_change = True
    stats = {"num_reqs": 1, "total_num_scheduled_tokens": 1024}

    mock_determine_phase.return_value = InferencePhase.PREFILL_ONLY
    profiler.step(stats)
    profiler.step(stats)
    assert profiler.current_phase == "prefill_only"
    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    profiler.step(stats)
    mock_context.__exit__.assert_called_once_with(None, None, None)
    assert profiler.current_phase == ""


def test_phased_profiler_stop_on_concurrency_change(profiler_fixture):
    """A capture keyed on num_reqs ends when num_reqs changes."""
    profiler = profiler_fixture["profiler"]
    mock_context = profiler_fixture["mock_context"]
    profiler_fixture["mock_determine_phase"].return_value = InferencePhase.DECODE_ONLY
    profiler.stop_on_phase_change = True
    profiler.target_concurrencies = frozenset({93})
    profiler.track_concurrency = True

    profiler.step({"num_reqs": 93, "total_num_scheduled_tokens": 93})
    profiler.step({"num_reqs": 93, "total_num_scheduled_tokens": 93})
    mock_context.__exit__.assert_not_called()
    profiler.step({"num_reqs": 92, "total_num_scheduled_tokens": 92})
    mock_context.__exit__.assert_called_once()


def test_phased_profiler_no_start_while_an_engine_is_idle(profiler_fixture):
    """Under DP a step with an idle engine (padded dummy work there) does not
    start a capture."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    profiler_fixture["mock_determine_phase"].return_value = InferencePhase.PREFILL_ONLY
    profiler.step(
        {"num_reqs": 1, "total_num_scheduled_tokens": 1024, "dp_num_reqs_min": 0}
    )
    mock_profile.assert_not_called()
    profiler.step(
        {"num_reqs": 1, "total_num_scheduled_tokens": 1024, "dp_num_reqs_min": 1}
    )
    mock_profile.assert_called_once()
