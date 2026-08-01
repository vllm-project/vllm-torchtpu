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
"""Tests for the shared multi-rank trace capture helpers."""

import datetime
import os

from vllm_torchtpu import profiler_trace


def _stage_capture(capture_dir, ts_name, filename, content):
    """Simulate a profiler capture under <capture_dir>/plugins/profile/<ts>/."""
    ts_dir = capture_dir / "plugins" / "profile" / ts_name
    ts_dir.mkdir(parents=True)
    (ts_dir / filename).write_text(content)
    return ts_dir


def test_profile_session_id_uses_dp_master_port(monkeypatch):
    """DP replicas have different parents, but share the DP master port."""
    monkeypatch.setenv("TORCH_TPU_DP_MASTER_PORT", "12345")

    assert profiler_trace.profile_session_id() == "dp12345"


def test_profile_session_id_falls_back_to_parent_pid(monkeypatch):
    monkeypatch.delenv("TORCH_TPU_DP_MASTER_PORT", raising=False)

    assert profiler_trace.profile_session_id() == str(os.getppid())


def test_rank_capture_dir_is_not_dp_specific(tmp_path):
    """The rank is slice-global, so the sandbox name must not say "dp"."""
    assert profiler_trace.rank_capture_dir(str(tmp_path), 3) == os.path.join(
        str(tmp_path), "rank_3")


def test_resolve_canonical_dst_ts_publishes_and_is_read_by_other_ranks(
        tmp_path):
    """Rank 0 publishes a ts; the other ranks pick up the same one."""
    ts = profiler_trace.resolve_canonical_dst_ts(str(tmp_path),
                                                 0,
                                                 session_key="key")

    marker = tmp_path / ".canonical_ts_key"
    assert marker.read_text().strip() == ts
    datetime.datetime.strptime(ts, profiler_trace.CANONICAL_TS_FORMAT)

    for rank in (1, 7):
        assert profiler_trace.resolve_canonical_dst_ts(str(tmp_path),
                                                       rank,
                                                       session_key="key") == ts


def test_resolve_canonical_dst_ts_is_scoped_by_session_key(
        tmp_path, monkeypatch):
    """A marker from an earlier start/stop cycle is not reused."""
    (tmp_path / ".canonical_ts_key_0").write_text("1999_01_01_00_00_00")
    # This rank is meant to time out; do not wait the production window.
    monkeypatch.setattr(profiler_trace, "CANONICAL_TS_POLL_TIMEOUT_S", 0.1)
    monkeypatch.setattr(profiler_trace, "CANONICAL_TS_POLL_INTERVAL_S", 0.02)

    ts = profiler_trace.resolve_canonical_dst_ts(str(tmp_path),
                                                 1,
                                                 session_key="key_1")

    # No marker for this cycle -> own fallback ts, not the stale one.
    assert ts != "1999_01_01_00_00_00"
    datetime.datetime.strptime(ts, profiler_trace.CANONICAL_TS_FORMAT)


def test_clear_canonical_ts_marker_is_idempotent(tmp_path):
    profiler_trace.resolve_canonical_dst_ts(str(tmp_path),
                                            0,
                                            session_key="key")

    profiler_trace.clear_canonical_ts_marker(str(tmp_path), "key")
    profiler_trace.clear_canonical_ts_marker(str(tmp_path), "key")

    assert not (tmp_path / ".canonical_ts_key").exists()


def test_merge_rank_capture_ignores_dp_env_vars(tmp_path, monkeypatch):
    """Only world_size decides the prefix; a stray DP env var must not.

    Both of these once fed the decision. Neither is set by anything in the
    tree today, and `TORCH_TPU_DP_SIZE` is popped before workers spawn.
    """
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "4")
    monkeypatch.setenv("TPU_MULTIPROCESS_DP", "1")
    capture_dir = tmp_path / "rank_0"
    capture_dir.mkdir()
    _stage_capture(capture_dir, "pt_ts", "t1v-n-host-w-0.xplane.pb", "data_0")

    profiler_trace.merge_rank_capture(str(capture_dir),
                                      str(tmp_path),
                                      "ts",
                                      0,
                                      world_size=1)

    dst = tmp_path / "plugins" / "profile" / "ts"
    assert (dst / "t1v-n-host-w-0.xplane.pb").read_text() == "data_0"


def test_merge_rank_capture_single_worker_keeps_filenames(tmp_path):
    capture_dir = tmp_path / "rank_0"
    capture_dir.mkdir()
    _stage_capture(capture_dir, "2026_05_06_04_47_36_pt",
                   "t1v-n-host-w-0.xplane.pb", "data_0")

    profiler_trace.merge_rank_capture(str(capture_dir), str(tmp_path),
                                      "2026_05_06_04_47_36", 0)

    dst = tmp_path / "plugins" / "profile" / "2026_05_06_04_47_36"
    assert (dst / "t1v-n-host-w-0.xplane.pb").read_text() == "data_0"
    # The trace was all the sandbox held, so it is not left behind empty.
    assert not capture_dir.exists()


def test_merge_rank_capture_keeps_sandbox_holding_other_artifacts(tmp_path):
    """The phased profiler parks batch stats next to the capture."""
    capture_dir = tmp_path / "rank_0"
    capture_dir.mkdir()
    _stage_capture(capture_dir, "pt_ts", "t1v-n-host-w-0.xplane.pb", "data_0")
    (capture_dir / "batch_composition_stats_1.json").write_text("{}")

    profiler_trace.merge_rank_capture(str(capture_dir), str(tmp_path), "ts", 0)

    assert not (capture_dir / "plugins").exists()
    assert (capture_dir / "batch_composition_stats_1.json").exists()


def test_merge_rank_capture_multi_worker_prefixes_and_unifies(tmp_path):
    """All ranks land in one run dir; same-named files stay distinct."""
    canonical_ts = "2026_05_06_04_47_36"

    for rank in range(4):
        capture_dir = tmp_path / f"rank_{rank}"
        capture_dir.mkdir()
        _stage_capture(capture_dir, f"pt_ts_{rank}",
                       "t1v-n-host-w-0.xplane.pb", f"rank_{rank}_xplane")
        profiler_trace.merge_rank_capture(str(capture_dir),
                                          str(tmp_path),
                                          canonical_ts,
                                          rank,
                                          world_size=4)

    dst = tmp_path / "plugins" / "profile" / canonical_ts
    for rank in range(4):
        assert (dst / f"rank{rank}_t1v-n-host-w-0.xplane.pb"
                ).read_text() == f"rank_{rank}_xplane"


def test_merge_rank_capture_without_capture_is_noop(tmp_path):
    capture_dir = tmp_path / "rank_0"
    capture_dir.mkdir()

    profiler_trace.merge_rank_capture(str(capture_dir), str(tmp_path), "ts", 0)

    assert not (tmp_path / "plugins").exists()
    assert capture_dir.exists()
