# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from vllm_torchtpu.runner.utils import (
    PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR, InferencePhase,
    PhaseBasedProfiler)


@pytest.fixture
def profiler_fixture(tmp_path):
    """Fixture to set up a PhaseBasedProfiler with mocked dependencies."""
    target_module = "vllm_torchtpu.runner.utils"
    with patch(f"{target_module}.profiler_api.profile") as mock_profile, patch(
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


def test_phased_profiler_ignores_initial_request(profiler_fixture):
    """Tests that profiling is not triggered for initial single-token requests."""
    profiler = profiler_fixture["profiler"]
    mock_profile = profiler_fixture["mock_profile"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY

    profiler.step({"num_reqs": 1, "total_num_scheduled_tokens": 1})
    mock_profile.assert_not_called()

    profiler.step({"num_reqs": 2, "total_num_scheduled_tokens": 1})
    mock_profile.assert_not_called()

    profiler.step({"num_reqs": 2, "total_num_scheduled_tokens": 2})
    mock_profile.assert_called_once()


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


def _stage_dp_rank_capture(rank_dir, ts_name, filename, content):
    """Helper: simulate writing one xplane file under dp_rank_N/plugins/profile/<ts>/."""
    ts_dir = rank_dir / "plugins" / "profile" / ts_name
    ts_dir.mkdir(parents=True)
    (ts_dir / filename).write_text(content)
    return ts_dir


def test_resolve_canonical_dst_ts_rank_zero_writes_marker(tmp_path):
    """Rank 0 picks the canonical ts and writes a marker keyed by ppid."""
    profiler = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=0)
    phase_dir = tmp_path / "prefill_heavy"
    phase_dir.mkdir()

    ts = profiler._resolve_canonical_dst_ts(str(phase_dir))

    marker = phase_dir / f".canonical_ts_{os.getppid()}"
    assert marker.exists()
    assert marker.read_text().strip() == ts


def test_resolve_canonical_dst_ts_non_zero_rank_reads_marker(tmp_path):
    """Non-zero rank reads whatever rank 0 already published."""
    phase_dir = tmp_path / "prefill_heavy"
    phase_dir.mkdir()
    marker = phase_dir / f".canonical_ts_{os.getppid()}"
    marker.write_text("2026_05_06_04_47_36")

    profiler = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=2)
    ts = profiler._resolve_canonical_dst_ts(str(phase_dir))

    assert ts == "2026_05_06_04_47_36"


def test_resolve_canonical_dst_ts_non_zero_rank_falls_back_on_timeout(
        tmp_path):
    """When rank 0 never publishes, a non-zero rank falls back to own ts."""
    phase_dir = tmp_path / "prefill_heavy"
    phase_dir.mkdir()

    profiler = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=1)
    # Speed the test up by shrinking the timeout.
    profiler._CANONICAL_TS_POLL_TIMEOUT_S = 0.1
    profiler._CANONICAL_TS_POLL_INTERVAL_S = 0.02

    ts = profiler._resolve_canonical_dst_ts(str(phase_dir))

    # Format check: looks like a strftime("%Y_%m_%d_%H_%M_%S") string.
    import datetime as _dt

    _dt.datetime.strptime(ts, "%Y_%m_%d_%H_%M_%S")


def test_merge_profile_directories_single_rank(tmp_path, monkeypatch):
    """Single rank, non-MPMD: capture moves from sandbox to <phase>/plugins/profile/<canonical_ts>/."""
    profiler = PhaseBasedProfiler(profile_dir=str(tmp_path), worker_rank=0)
    phase_dir = tmp_path / "prefill_heavy"
    rank_dir = phase_dir / "dp_rank_0"
    rank_dir.mkdir(parents=True)
    profiler.profile_dir_with_phase_suffix = str(rank_dir)
    profiler._canonical_dst_ts = "2026_05_06_04_47_36"

    # We need to create the actual directory structure for listdir to work in test
    _stage_dp_rank_capture(rank_dir, "2026_05_06_04_47_36_pt",
                           "t1v-n-host-w-0.xplane.pb", "data_0")

    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "1")
    monkeypatch.delenv("TPU_MULTIPROCESS_DP", raising=False)
    profiler._merge_profile_directories()

    dst = phase_dir / "plugins" / "profile" / "2026_05_06_04_47_36"
    assert (dst / "t1v-n-host-w-0.xplane.pb").read_text() == "data_0"
    # Sandbox plugins/ subtree cleaned up; dp_rank_0/ itself remains for stats.
    assert not (rank_dir / "plugins").exists()
    assert rank_dir.exists()


def test_merge_profile_directories_mpmd(tmp_path, monkeypatch):
    """MPMD (DP=4, TP=1): 4 DP replicas captured to their own sandboxes with identical filenames; each moves to the SAME canonical ts dir with dp{D}_rank{R}_ prefix."""
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "4")
    phase_dir = tmp_path / "prefill_heavy"

    canonical_ts = "2026_05_06_04_47_36"
    profilers = []
    for dp_rank in range(4):
        # TP=1, so every DP replica's own local (TP*PP) rank is 0.
        profiler = PhaseBasedProfiler(profile_dir=str(tmp_path),
                                      worker_rank=0,
                                      dp_rank=dp_rank)
        rank_dir = phase_dir / f"dp{dp_rank}_rank0"
        rank_dir.mkdir(parents=True)
        profiler.profile_dir_with_phase_suffix = str(rank_dir)
        profiler._canonical_dst_ts = canonical_ts
        _stage_dp_rank_capture(
            rank_dir,
            f"pt_ts_{dp_rank}",
            "t1v-n-host-w-0.xplane.pb",
            f"rank_{dp_rank}_xplane",
        )
        profilers.append(profiler)

    for profiler in profilers:
        profiler._merge_profile_directories()

    dst = phase_dir / "plugins" / "profile" / canonical_ts
    for dp_rank in range(4):
        assert (dst / f"dp{dp_rank}_rank0_t1v-n-host-w-0.xplane.pb"
                ).read_text() == f"rank_{dp_rank}_xplane"
        rank_dir = phase_dir / f"dp{dp_rank}_rank0"
        assert not (rank_dir / "plugins").exists()


def test_merge_profile_directories_tp_multiworker(tmp_path, monkeypatch):
    """TP multi-worker (TP=4, DP=1): world_size=4 with no DP env vars; each local rank moves to canonical ts dir with dp0_rank{R}_ prefix."""
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "1")
    monkeypatch.delenv("TPU_MULTIPROCESS_DP", raising=False)
    phase_dir = tmp_path / "prefill_heavy"

    canonical_ts = "2026_05_06_04_47_36"
    profilers = []
    for rank in range(4):
        # DP=1, so every worker's dp_rank is 0; TP*PP-local rank varies.
        profiler = PhaseBasedProfiler(profile_dir=str(tmp_path),
                                      worker_rank=rank,
                                      dp_rank=0,
                                      world_size=4)
        rank_dir = phase_dir / f"dp0_rank{rank}"
        rank_dir.mkdir(parents=True)
        profiler.profile_dir_with_phase_suffix = str(rank_dir)
        profiler._canonical_dst_ts = canonical_ts
        _stage_dp_rank_capture(
            rank_dir,
            f"pt_ts_{rank}",
            "t1v-n-host-w-0.xplane.pb",
            f"rank_{rank}_xplane",
        )
        profilers.append(profiler)

    for profiler in profilers:
        profiler._merge_profile_directories()

    dst = phase_dir / "plugins" / "profile" / canonical_ts
    for rank in range(4):
        assert (dst / f"dp0_rank{rank}_t1v-n-host-w-0.xplane.pb"
                ).read_text() == f"rank_{rank}_xplane"
        rank_dir = phase_dir / f"dp0_rank{rank}"
        assert not (rank_dir / "plugins").exists()


def test_merge_profile_directories_dp_and_tp_do_not_collide(
        tmp_path, monkeypatch):
    """DP=2, TP=2: this is the exact bug scenario -- workers at the same
    dp_rank with different local ranks (or vice versa) must not collide."""
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "2")
    phase_dir = tmp_path / "prefill_heavy"
    canonical_ts = "2026_05_06_04_47_36"

    profilers = []
    for dp_rank in range(2):
        for local_rank in range(2):
            profiler = PhaseBasedProfiler(profile_dir=str(tmp_path),
                                          worker_rank=local_rank,
                                          dp_rank=dp_rank,
                                          world_size=2)
            rank_dir = phase_dir / f"dp{dp_rank}_rank{local_rank}"
            rank_dir.mkdir(parents=True)
            profiler.profile_dir_with_phase_suffix = str(rank_dir)
            profiler._canonical_dst_ts = canonical_ts
            _stage_dp_rank_capture(
                rank_dir,
                f"pt_ts_{dp_rank}_{local_rank}",
                "t1v-n-host-w-0.xplane.pb",
                f"data_dp{dp_rank}_local{local_rank}",
            )
            profilers.append(profiler)

    for profiler in profilers:
        profiler._merge_profile_directories()

    dst = phase_dir / "plugins" / "profile" / canonical_ts
    seen_files = sorted(p.name for p in dst.iterdir())
    assert seen_files == [
        "dp0_rank0_t1v-n-host-w-0.xplane.pb",
        "dp0_rank1_t1v-n-host-w-0.xplane.pb",
        "dp1_rank0_t1v-n-host-w-0.xplane.pb",
        "dp1_rank1_t1v-n-host-w-0.xplane.pb",
    ]
    for dp_rank in range(2):
        for local_rank in range(2):
            assert (dst /
                    f"dp{dp_rank}_rank{local_rank}_t1v-n-host-w-0.xplane.pb"
                    ).read_text() == f"data_dp{dp_rank}_local{local_rank}"
