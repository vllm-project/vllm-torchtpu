# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, mock_open, patch

import pytest

from vllm_torchtpu import profiler_trace
from vllm_torchtpu.runner.utils import (
    PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR, InferencePhase,
    PhaseBasedProfiler, determine_phase_from_batch_composition_stats)


@pytest.fixture
def profiler_fixture(tmp_path):
    """Fixture to set up a PhaseBasedProfiler with mocked dependencies."""
    target_module = "vllm_torchtpu.runner.utils"
    with patch("torch.profiler.profile") as mock_profile, patch(
            "builtins.open", mock_open()
    ) as mock_file, patch(f"{target_module}.datetime") as mock_datetime, patch(
            f"{target_module}.determine_phase_from_batch_composition_stats"
    ) as mock_determine_phase, patch.object(
            PhaseBasedProfiler,
            "_resolve_canonical_dst_ts",
            return_value="2024_01_01_12_00_00",
    ), patch.object(PhaseBasedProfiler, "_merge_profile_directories"):

        mock_now = MagicMock()
        mock_now.strftime.return_value = "2024_01_01_12_00_00"
        mock_datetime.datetime.now.return_value = mock_now

        # Set up mock for the context manager returned by profile()
        mock_context = MagicMock()
        mock_profile.return_value = mock_context

        profiler = PhaseBasedProfiler(profile_dir=str(tmp_path))
        profiler.num_steps_to_profile_for = (
            PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR)

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
    assert (profiler.profiling_n_steps_left ==
            PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR)
    assert profiler.current_phase == "prefill_heavy"
    assert profiler.inference_phase_seen[InferencePhase.PREFILL_HEAVY]
    assert mock_file().write.call_count == 1  # Wrote stats on start

    # 2. Step profiling (N-1 steps)
    for i in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1):
        profiler.step(stats)
        assert (profiler.profiling_n_steps_left ==
                PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1 - i)
        mock_profile.assert_called_once()  # Not called again
        mock_context.__exit__.assert_not_called()

    # 3. Final step stops profiling
    profiler.step(stats)
    mock_context.__exit__.assert_called_once_with(None, None, None)
    assert profiler.profiling_n_steps_left == 0
    assert profiler.current_phase == ""
    assert (mock_file().write.call_count ==
            PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR + 1)


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
    assert determine_phase_from_batch_composition_stats({
        "total_num_scheduled_tokens":
        1,
        "num_prefill_tokens":
        0,
    }) == InferencePhase.DECODE_ONLY


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
    assert (profiler.profiling_n_steps_left ==
            PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR)

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
    assert (profiler.profiling_n_steps_left ==
            PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR)


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
    profiler_fixture, ):
    """Tests that the profiler skips DECODE_ONLY steps until min KV len reaches threshold."""
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
    assert (profiler.profiling_n_steps_left ==
            PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR)

    # Profiling continues
    stats["min_kv_len"] = 11
    profiler.step(stats)
    mock_profile.assert_called_once()

    # Complete the profiling cycle (1 more step already done in the step above)
    for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1):
        profiler.step(stats)
    mock_context.__exit__.assert_called_once()
    assert profiler.current_phase == ""


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
        tmp_path, monkeypatch):
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
    """Single rank, non-MPMD: capture moves from sandbox to <phase>/plugins/profile/<canonical_ts>/."""
    profiler = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=0)
    phase_dir = tmp_path / "prefill_heavy"
    rank_dir = phase_dir / "rank_0"
    rank_dir.mkdir(parents=True)
    profiler.profile_dir_with_phase_suffix = str(rank_dir)
    profiler._canonical_dst_ts = "2026_05_06_04_47_36"

    # We need to create the actual directory structure for listdir to work in test
    _stage_rank_capture(rank_dir, "2026_05_06_04_47_36_pt",
                        "t1v-n-host-w-0.xplane.pb", "data_0")
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
        profiler = PhaseBasedProfiler(profile_dir=str(tmp_path),
                                      worker_rank=rank,
                                      world_size=2)
        rank_dir = phase_dir / f"rank_{rank}"
        rank_dir.mkdir()
        profiler.profile_dir_with_phase_suffix = str(rank_dir)
        profiler._canonical_dst_ts = profiler._resolve_canonical_dst_ts(
            str(phase_dir))
        _stage_rank_capture(rank_dir, f"pt_ts_{rank}",
                            "t1v-n-host-w-0.xplane.pb", f"rank_{rank}")
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
        profiler = PhaseBasedProfiler(profile_dir=str(tmp_path),
                                      worker_rank=rank,
                                      world_size=4)
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
        assert (dst / f"rank{rank}_t1v-n-host-w-0.xplane.pb"
                ).read_text() == f"rank_{rank}_xplane"
        rank_dir = phase_dir / f"rank_{rank}"
        assert not (rank_dir / "plugins").exists()
