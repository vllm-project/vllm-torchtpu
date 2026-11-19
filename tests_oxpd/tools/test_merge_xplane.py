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
"""Tests for the offline per-rank xplane.pb merge tool."""

import pytest

from vllm_torchtpu.tools import merge_xplane, xplane_pb2

# The five planes a real Qwen3 TP=4 capture on v6e contains, in capture order.
# All four ranks name their own chip `/device:TPU:0` and carry their own
# `/host:metadata`, which is the collision the merge exists to resolve.
_REAL_PLANES = (
    "/device:TPU:0",
    "/host:metadata",
    "/device:CUSTOM:Megascale Trace",
    "Task Environment",
    "/host:CPU",
)


def _write_trace(session_dir, rank, *, host="host-0", planes=_REAL_PLANES):
    """Write one rank's capture. `rank=None` omits the rank filename prefix."""
    session_dir.mkdir(parents=True, exist_ok=True)
    space = xplane_pb2.XSpace()
    space.hostnames.append(host)
    for name in planes:
        plane = space.planes.add(id=1, name=name)
        if name == "/host:CPU":
            # Sparse, non-zero-based keys: `len(stat_metadata) + 1` would
            # collide with an existing one when injecting `process_id`.
            for key in (2, 3):
                plane.stat_metadata[key].id = key
                plane.stat_metadata[key].name = f"existing_{key}"
        line = plane.lines.add(id=100 + (rank or 0), name="")
        line.events.add(metadata_id=1, offset_ps=1, duration_ps=10)
    prefix = "" if rank is None else f"rank{rank}_"
    path = session_dir / f"{prefix}{host}.xplane.pb"
    path.write_bytes(space.SerializeToString())
    return path


def _load(path):
    space = xplane_pb2.XSpace()
    space.ParseFromString(path.read_bytes())
    return space


def _session_dir(root, phase="decode_only", timestamp="2026_08_20_10_00_00"):
    return root / phase / "plugins" / "profile" / timestamp


def _merge(tmp_path):
    """Merge the one session under `tmp_path`, writing outside the scanned tree."""
    session = merge_xplane.discover_sessions([str(tmp_path)])[0]
    out = tmp_path.parent / f"{tmp_path.name}-merged.xplane.pb"
    result = merge_xplane.merge_session(session, str(out))
    return _load(out), result.planes, session


def test_merge_combines_ranks_into_one_xspace(tmp_path):
    """The whole structural contract of a merge, on a realistic capture."""
    for rank in range(4):
        _write_trace(_session_dir(tmp_path), rank)

    merged, plane_count, _ = _merge(tmp_path)
    names = [p.name for p in merged.planes]

    # 4 ranks x 5 planes, less the 3 duplicate metadata planes.
    assert plane_count == 17
    assert [p.id for p in merged.planes] == list(range(1, 18))
    # Each rank's chip is renumbered globally.
    assert [n for n in names if n.startswith("/device:TPU")] == [
        "/device:TPU:0",
        "/device:TPU:1",
        "/device:TPU:2",
        "/device:TPU:3",
    ]
    # A plane whose name carries no `:<id>` is tagged instead of renumbered.
    assert names.count("/device:CUSTOM:Megascale Trace [rank 2]") == 1
    assert names.count("/host:metadata") == 1
    assert [n for n in names if n.startswith("/host:CPU")] == [
        "/host:CPU",
        "/host:CPU [1]",
        "/host:CPU [2]",
        "/host:CPU [3]",
    ]
    assert "Task Environment [rank 2]" in names

    for expected, plane in enumerate(
        p for p in merged.planes if p.name.startswith("/host:CPU")
    ):
        # Internal XProf keys host traces on process_id, so it has to be
        # present, distinct, and must not disturb existing stat metadata.
        (stat_id,) = [
            key
            for key, meta in plane.stat_metadata.items()
            if meta.name == "process_id"
        ]
        assert [s.int64_value for s in plane.stats if s.metadata_id == stat_id] == [
            expected
        ]
        assert plane.stat_metadata[2].name == "existing_2"
        # Thread lines are tagged so ranks stay apart, with no leading space
        # for lines that have no name of their own.
        assert plane.lines[0].name == f"{100 + expected} [rank {expected}]"


def test_merge_preserves_every_event(tmp_path):
    """Deduplicated metadata planes are the only thing allowed to disappear."""
    sources = [_write_trace(_session_dir(tmp_path), rank) for rank in range(3)]

    merged, _, _ = _merge(tmp_path)

    def events(space):
        return sum(len(line.events) for p in space.planes for line in p.lines)

    dropped = sum(
        len(line.events)
        for path in sources[1:]
        for p in _load(path).planes
        if p.name == "/host:metadata"
        for line in p.lines
    )
    assert events(merged) == sum(events(_load(p)) for p in sources) - dropped


def test_ranks_sort_numerically(tmp_path):
    """`rank10_` precedes `rank2_` as text, which would mislabel every rank."""
    for rank in (0, 2, 10):
        _write_trace(_session_dir(tmp_path), rank)

    session = merge_xplane.discover_sessions([str(tmp_path)])[0]

    assert [c.rank for c in session.captures] == [0, 2, 10]


def test_devices_are_renumbered_across_hosts(tmp_path):
    """Numbering per host would give two hosts the same `/device:TPU:0`.

    SparseCore planes name their chip with a trailing segment, which has to
    follow the chip it belongs to rather than be renumbered on its own.
    """
    chip = ["/device:TPU:0", "/device:TPU:0 SparseCore 0"]
    session_dir = _session_dir(tmp_path)
    _write_trace(session_dir, 0, host="host-a", planes=chip)
    _write_trace(session_dir, 1, host="host-b", planes=chip)

    merged, _, _ = _merge(tmp_path)

    assert [p.name for p in merged.planes] == [
        "/device:TPU:0",
        "/device:TPU:0 SparseCore 0",
        "/device:TPU:1",
        "/device:TPU:1 SparseCore 0",
    ]
    assert list(merged.hostnames) == ["host-a", "host-b"]


def test_unranked_captures_do_not_collide(tmp_path):
    """Files with no rank prefix still need distinct host planes."""
    session_dir = _session_dir(tmp_path)
    _write_trace(session_dir, 0, planes=["/host:CPU"])
    _write_trace(session_dir, None, host="other", planes=["/host:CPU"])

    merged, _, _ = _merge(tmp_path)

    assert [p.name for p in merged.planes] == ["/host:CPU", "/host:CPU [1]"]


def test_cli_merges_each_phase_into_its_own_phase_directory(tmp_path):
    """Phases are independent capture windows and must not be fused.

    Each result lands in its phase directory -- above `plugins/`, where XProf
    does not look for hosts.
    """
    for phase in ("decode_only", "prefill_only"):
        for rank in range(2):
            _write_trace(_session_dir(tmp_path, phase), rank)

    assert merge_xplane.main([str(tmp_path)]) == 0

    assert [
        str(p.relative_to(tmp_path)) for p in sorted(tmp_path.glob("*/*.xplane.pb"))
    ] == [
        "decode_only/decode_only_2026_08_20_10_00_00.xplane.pb",
        "prefill_only/prefill_only_2026_08_20_10_00_00.xplane.pb",
    ]


@pytest.mark.parametrize(
    "start, argument",
    [
        (".", "."),
        (".", "decode_only"),
        (".", "decode_only/plugins"),
        (".", "decode_only/plugins/profile"),
        (".", "decode_only/plugins/profile/2026_08_20_10_00_00"),
        ("decode_only/plugins/profile/2026_08_20_10_00_00", "."),
    ],
)
def test_destination_follows_the_capture_not_the_invocation(
    tmp_path, monkeypatch, start, argument
):
    """Any way of naming a phase writes the merged file to the same place.

    Relative arguments included: `_phase_dir` walks up with `dirname`, which
    sees nothing above a bare `.`.
    """
    for rank in range(2):
        _write_trace(_session_dir(tmp_path), rank)
    monkeypatch.chdir(tmp_path / start)

    assert merge_xplane.main([argument]) == 0

    assert [
        str(p.relative_to(tmp_path))
        for p in tmp_path.glob("**/*.xplane.pb")
        if not p.name.startswith("rank")
    ] == ["decode_only/decode_only_2026_08_20_10_00_00.xplane.pb"]


def test_cli_refuses_to_write_inside_a_session_directory(tmp_path, capsys):
    """XProf would load the merged file as an extra host holding every rank."""
    session_dir = _session_dir(tmp_path)
    _write_trace(session_dir, 0)

    code = merge_xplane.main([str(tmp_path), "-d", str(session_dir)])

    assert code == 2
    assert "double-counting" in capsys.readouterr().err


def test_missing_rank_warns_and_keeps_surviving_rank_labels(tmp_path):
    """A gap must not shift the labels of the ranks that did report."""
    for rank in (0, 1, 3):
        _write_trace(_session_dir(tmp_path), rank)

    merged, _, session = _merge(tmp_path)

    assert session.missing_ranks == [2]
    names = [p.name for p in merged.planes]
    assert "Task Environment [rank 3]" in names
    assert "Task Environment [rank 2]" not in names
    assert "/host:CPU [3]" in names


def test_dry_run_reports_without_writing(tmp_path, capsys):
    for rank in (0, 2):
        _write_trace(_session_dir(tmp_path), rank)

    assert merge_xplane.main([str(tmp_path), "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "rank(s) 1" in out
    # The destination is the point of the report, so it has to be the full path.
    assert (
        str(tmp_path / "decode_only" / "decode_only_2026_08_20_10_00_00.xplane.pb")
        in out
    )
    assert not list(tmp_path.glob("*/*.xplane.pb"))


def test_rerunning_ignores_previously_merged_output(tmp_path):
    """Its own output must not be picked up as another capture to merge.

    The merged file lands in the phase directory, which the second run walks
    straight through, so discovery has to drop it on sight.
    """
    for rank in range(2):
        _write_trace(_session_dir(tmp_path), rank)
    assert merge_xplane.main([str(tmp_path)]) == 0

    assert merge_xplane.main([str(tmp_path)]) == 0

    assert [str(p.relative_to(tmp_path)) for p in tmp_path.glob("*/*.xplane.pb")] == [
        "decode_only/decode_only_2026_08_20_10_00_00.xplane.pb"
    ]
