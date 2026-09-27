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
"""Combine a run's per-rank ``.xplane.pb`` captures into one XSpace file.

A multi-chip capture writes one trace per worker (see `profiler_trace`), all
landing in a single ``plugins/profile/<timestamp>/`` directory. OSS XProf reads
that whole directory as one distributed session, so **this tool is not needed
for the `xprof --logdir` workflow**. It exists for consumers that take exactly
one file — notably Google-internal XProf's offline upload — and for attaching a
single artifact to a bug.

Merging is not just concatenation. Every rank names its own chip
``/device:TPU:0`` and every rank ships a full copy of the ``/host:metadata``
plane, so the planes have to be renumbered globally and the metadata collapsed
to one master. That dedup is also why the output is typically about half the
size of the inputs combined.

Usage::

    # every capture session under a run directory. Each merged file lands in
    # its phase directory, e.g.
    #   /tmp/vllm_phased_profile/decode_only/decode_only_<timestamp>.xplane.pb
    python -m vllm_torchtpu.tools.merge_xplane /tmp/vllm_phased_profile

    # naming any level of one phase writes to that same place
    python -m vllm_torchtpu.tools.merge_xplane \\
        /tmp/vllm_phased_profile/decode_only/plugins/profile/2026_08_20_22_14_43

    # one session, explicit destination
    python -m vllm_torchtpu.tools.merge_xplane \\
        /tmp/vllm_phased_profile/decode_only -o /tmp/decode.xplane.pb

    # see what would happen
    python -m vllm_torchtpu.tools.merge_xplane /tmp/vllm_profile --dry-run
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass

from vllm_torchtpu.tools import xplane_pb2

# Plane names of the form `<prefix>:<id>`, e.g. `/device:TPU:0`, optionally
# followed by a whitespace-led trailing segment such as ` SparseCore 0`. The
# id is rewritten; the trailing segment is preserved verbatim.
_PREFIX_ID_RE = re.compile(r"^(\S+):(\d+)(\s+.*)?$")

# `profiler_trace.merge_rank_capture` prefixes each file with its slice-global
# rank whenever more than one worker captured.
_RANK_PREFIX_RE = re.compile(r"^rank(\d+)_")

_TRACE_SUFFIX = ".xplane.pb"

# XSpace field numbers, from xplane.proto. Used to append planes to the output
# without holding the merged message in memory; see `merge_session`.
_XSPACE_PLANES_FIELD = 1
_XSPACE_ERRORS_FIELD = 2
_XSPACE_WARNINGS_FIELD = 3
_XSPACE_HOSTNAMES_FIELD = 4

_WIRE_TYPE_LENGTH_DELIMITED = 2

_METADATA_PLANE = "/host:metadata"
_HOST_CPU_PLANE = "/host:CPU"


@dataclass(frozen=True)
class Capture:
    """One worker's trace file."""

    path: str
    rank: int | None
    # Unique small integer within the session, used where a value has to be
    # distinct per capture rather than merely descriptive -- the `/host:CPU`
    # suffix and the injected `process_id`. Equal to `rank` whenever the file
    # carries a rank prefix, which is every multi-worker capture.
    slot: int

    @property
    def label(self) -> str:
        """Human-readable tag used in merged plane and line names."""
        return f"rank {self.rank}" if self.rank is not None else (f"file {self.slot}")


@dataclass(frozen=True)
class Session:
    """One capture session: every rank's trace from a single profile window."""

    label: str
    timestamp: str
    directory: str
    captures: tuple[Capture, ...]

    @property
    def output_name(self) -> str:
        stem = f"{self.label}_{self.timestamp}" if self.timestamp else self.label
        return f"{stem}{_TRACE_SUFFIX}"

    @property
    def default_output_dir(self) -> str:
        """Where the merged file goes when the caller named no destination.

        The phase directory -- one level above `plugins/` -- so the result sits
        beside the capture it summarises without joining it. XProf enumerates
        runs only at `*/plugins/profile/*`, so a file there is invisible to the
        session it came from. Anchoring on the session rather than on whatever
        the caller happened to name means every way of pointing at a phase
        (the timestamp directory, `plugins/profile`, `plugins`, or the phase
        directory itself) writes to the same place.
        """
        phase_dir = _phase_dir(self.directory)
        return phase_dir or os.path.join(self.directory, "merged")

    @property
    def missing_ranks(self) -> list[int]:
        """Ranks absent from an otherwise contiguous 0..N-1 set.

        A capture that lost a rank still merges, but the result silently
        describes less of the slice than it appears to, so callers warn.
        """
        ranks = sorted(c.rank for c in self.captures if c.rank is not None)
        if not ranks:
            return []
        return [r for r in range(ranks[-1]) if r not in set(ranks)]


def _varint(value: int) -> bytes:
    """Encode `value` as a protobuf base-128 varint."""
    out = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return bytes(out)


def _delimited(field_number: int, payload: bytes) -> bytes:
    """Encode one length-delimited field, ready to append to a message.

    A serialized protobuf message is the concatenation of its serialized
    fields, and a repeated field is simply each element in turn. So writing
    `_delimited(_XSPACE_PLANES_FIELD, plane.SerializeToString())` once per
    plane builds a valid XSpace without ever materialising one.
    """
    tag = _varint((field_number << 3) | _WIRE_TYPE_LENGTH_DELIMITED)
    return tag + _varint(len(payload)) + payload


def _rank_of(path: str) -> int | None:
    match = _RANK_PREFIX_RE.match(os.path.basename(path))
    return int(match.group(1)) if match else None


def _phase_dir(session_dir: str) -> str | None:
    """The directory holding `plugins/`, or None if this is not a session dir.

    Captures live at ``<phase>/plugins/profile/<timestamp>/``, where `<phase>`
    is the phase name for a phased run and the trace-directory name otherwise.
    """
    profile_dir = os.path.dirname(session_dir)
    plugins_dir = os.path.dirname(profile_dir)
    if (
        os.path.basename(profile_dir) == "profile"
        and os.path.basename(plugins_dir) == "plugins"
    ):
        return os.path.dirname(plugins_dir)
    return None


def _session_identity(directory: str) -> tuple[str, str]:
    """Derive a (label, timestamp) pair from a session directory path.

    Anything not shaped like a capture session falls back to the directory's
    own name and no timestamp.
    """
    timestamp = os.path.basename(directory)
    phase_dir = _phase_dir(directory)
    if phase_dir is None:
        return timestamp, ""
    return os.path.basename(phase_dir) or timestamp, timestamp


def _make_session(directory: str, paths: list[str]) -> Session:
    # Sort numerically by rank, not by filename: lexicographic order puts
    # `rank10_` before `rank2_`, which would mislabel every rank above 9.
    ranked = sorted(
        ((path, _rank_of(path)) for path in paths),
        key=lambda pair: (pair[1] is None, pair[1] or 0, os.path.basename(pair[0])),
    )

    # A file with no rank prefix (a single-worker capture) still needs a slot
    # that cannot collide with a real rank, or two of them would produce the
    # same `/host:CPU` plane and the same process_id.
    taken = {rank for _, rank in ranked if rank is not None}
    next_free = 0
    captures = []
    for path, rank in ranked:
        if rank is None:
            while next_free in taken:
                next_free += 1
            taken.add(next_free)
            slot = next_free
        else:
            slot = rank
        captures.append(Capture(path=path, rank=rank, slot=slot))

    # Absolute and normalized: `_phase_dir` walks up with `dirname`, which
    # cannot see past a relative path (`dirname(".")` is `""`), and a trailing
    # separator would otherwise shift every component by one.
    directory = os.path.normpath(os.path.abspath(directory))
    label, timestamp = _session_identity(directory)
    return Session(
        label=label, timestamp=timestamp, directory=directory, captures=tuple(captures)
    )


def _is_within(path: str, ancestor: str) -> bool:
    path = os.path.normpath(os.path.abspath(path))
    ancestor = os.path.normpath(os.path.abspath(ancestor))
    return path == ancestor or path.startswith(ancestor + os.sep)


def discover_sessions(paths: list[str], *, skip: tuple[str, ...] = ()) -> list[Session]:
    """Group the trace files under `paths` into capture sessions.

    One session per directory holding `.xplane.pb` files. That keeps a phased
    run's phases apart: each is an independent capture window anchored at its
    own zero, so fusing them would stack unrelated timelines at t=0.

    A phase directory holds this tool's own output rather than a capture, so
    any that turn up in the walk are dropped: re-running over a tree already
    merged must not merge the merged files, fusing every phase together.
    `skip` excludes destinations that are not phase directories.

    Explicit file arguments are treated as a single session, on the assumption
    that someone naming files by hand means them to go together.
    """
    by_directory: dict[str, list[str]] = {}
    explicit: list[str] = []

    for path in paths:
        if os.path.isfile(path):
            if not any(_is_within(path, s) for s in skip):
                explicit.append(path)
        elif os.path.isdir(path):
            for root, _, filenames in os.walk(path):
                if any(_is_within(root, s) for s in skip):
                    continue
                found = [
                    os.path.join(root, f)
                    for f in filenames
                    if f.endswith(_TRACE_SUFFIX)
                ]
                if found:
                    by_directory.setdefault(root, []).extend(found)
        else:
            raise FileNotFoundError(path)

    sessions = [
        _make_session(directory, found)
        for directory, found in sorted(by_directory.items())
    ]
    if explicit:
        sessions.insert(0, _make_session(os.path.dirname(explicit[0]), explicit))

    def normalized(path: str) -> str:
        return os.path.normpath(os.path.abspath(path))

    phase_dirs = {
        normalized(session.default_output_dir)
        for session in sessions
        if _phase_dir(session.directory)
    }
    return [
        session
        for session in sessions
        if _phase_dir(session.directory)
        or normalized(session.directory) not in phase_dirs
    ]


def _next_stat_metadata_id(plane) -> int:
    """Return an unused key for `plane.stat_metadata`.

    Keys are XStatMetadata ids, which are arbitrary and need not be dense, so
    the next key has to come from the maximum rather than the count.
    """
    return max(plane.stat_metadata, default=0) + 1


def _tag_host_cpu_plane(plane, capture: Capture) -> None:
    """Make one rank's host-CPU plane distinguishable in the merged file.

    Google-internal XProf discovers host traces by process id, so each rank's
    plane needs both a distinct name and a `process_id` stat; without them the
    ranks collapse into a single host track. Thread lines are tagged too, since
    their names are otherwise identical across ranks.
    """
    slot = capture.slot
    if slot:
        plane.name = f"{_HOST_CPU_PLANE} [{slot}]"

    stat_id = next(
        (key for key, meta in plane.stat_metadata.items() if meta.name == "process_id"),
        None,
    )
    if stat_id is None:
        stat_id = _next_stat_metadata_id(plane)
        plane.stat_metadata[stat_id].id = stat_id
        plane.stat_metadata[stat_id].name = "process_id"

    stat = next((s for s in plane.stats if s.metadata_id == stat_id), None)
    if stat is None:
        stat = plane.stats.add()
        stat.metadata_id = stat_id
    stat.int64_value = slot

    for line in plane.lines:
        if not line.events:
            continue
        original = line.name or ""
        subsystem, _, thread = original.rpartition("/")
        thread = thread or str(line.id)
        if "pjrt-tpu-tasks" in original.lower():
            # Keep the subsystem leading so the viewer still groups these
            # lines together, with the rank as part of the group name.
            line.name = f"pjrt-tpu-tasks [{capture.label}]/{thread}"
        elif subsystem:
            line.name = f"{subsystem} {thread} [{capture.label}]"
        else:
            line.name = f"{thread} [{capture.label}]"


def _rewrite_plane_name(
    plane,
    capture: Capture,
    local_to_global: dict[tuple[str, int], int],
    next_id_per_prefix: dict[str, int],
) -> None:
    """Rename one plane so it stays distinct once ranks are combined.

    Device planes are renumbered from a single global counter per prefix, which
    also keeps ranks on different hosts apart -- numbering per host would give
    two hosts the same `/device:TPU:0`.
    """
    match = _PREFIX_ID_RE.match(plane.name)
    if match:
        prefix, local_id, trailing = (
            match.group(1),
            int(match.group(2)),
            match.group(3) or "",
        )
        key = (prefix, local_id)
        if key not in local_to_global:
            local_to_global[key] = next_id_per_prefix.get(prefix, 0)
            next_id_per_prefix[prefix] = local_to_global[key] + 1
        plane.name = f"{prefix}:{local_to_global[key]}{trailing}"
    elif plane.name == _HOST_CPU_PLANE:
        _tag_host_cpu_plane(plane, capture)
    else:
        plane.name = f"{plane.name} [{capture.label}]"


@dataclass(frozen=True)
class MergeResult:
    """What one merge produced.

    `planes_read` and `deduplicated` exist so callers can explain the output
    plane count instead of just stating it: a reader who knows there were four
    input *files* has no way to guess where "17 planes" came from.
    """

    path: str
    size: int
    planes: int
    planes_read: int
    deduplicated: int


def merge_session(session: Session, output_path: str) -> MergeResult:
    """Merge `session` into `output_path`.

    Ranks are read one at a time and appended to the open file, so peak memory
    tracks a single rank's trace rather than the whole session -- which matters
    because a single phase can be hundreds of megabytes per rank.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    next_id_per_prefix: dict[str, int] = {}
    hostnames: list[str] = []
    plane_count = 0
    planes_read = 0
    deduplicated = 0
    seen_metadata = False
    plane_id = 1

    temporary_path = f"{output_path}.tmp"
    try:
        with open(temporary_path, "wb") as out:
            for capture in session.captures:
                space = xplane_pb2.XSpace()
                with open(capture.path, "rb") as handle:
                    space.ParseFromString(handle.read())

                for hostname in space.hostnames:
                    if hostname not in hostnames:
                        hostnames.append(hostname)

                local_to_global: dict[tuple[str, int], int] = {}
                for plane in space.planes:
                    planes_read += 1
                    if plane.name == _METADATA_PLANE:
                        # Every rank carries a full copy, and it holds the HLO
                        # protos, so keeping one is both correct for the
                        # XProf tools and most of the size reduction.
                        if seen_metadata:
                            deduplicated += 1
                            continue
                        seen_metadata = True
                    else:
                        _rewrite_plane_name(
                            plane, capture, local_to_global, next_id_per_prefix
                        )
                    plane.id = plane_id
                    plane_id += 1
                    plane_count += 1
                    out.write(
                        _delimited(_XSPACE_PLANES_FIELD, plane.SerializeToString())
                    )

                for error in space.errors:
                    out.write(_delimited(_XSPACE_ERRORS_FIELD, error.encode()))
                for warning in space.warnings:
                    out.write(_delimited(_XSPACE_WARNINGS_FIELD, warning.encode()))
                del space

            for hostname in hostnames:
                out.write(_delimited(_XSPACE_HOSTNAMES_FIELD, hostname.encode()))

        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    return MergeResult(
        path=output_path,
        size=os.path.getsize(output_path),
        planes=plane_count,
        planes_read=planes_read,
        deduplicated=deduplicated,
    )


def _is_inside_session_dir(directory: str) -> bool:
    """True if `directory` sits within a `plugins/profile/` capture session.

    XProf builds its host list from the filenames in a session directory, so a
    merged file dropped beside the per-rank ones shows up as an extra host
    holding a second copy of every rank.
    """
    normalized = os.path.normpath(os.path.abspath(directory))
    return f"{os.sep}plugins{os.sep}profile{os.sep}" in f"{normalized}{os.sep}"


def _format_size(num_bytes: int) -> str:
    """Human-readable size, staying in MB so totals stay comparable.

    Small captures still have to read as something other than `0 MB`, which
    makes a successful merge look like it produced nothing.
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes / (1024 * 1024):.0f} MB"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_torchtpu.tools.merge_xplane",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Trace files, or directories to search for them. A directory "
        "holding several capture sessions yields one merged file each.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "-o",
        "--output",
        help="Write to this exact path. Only valid when the inputs resolve "
        "to a single capture session.",
    )
    destination.add_argument(
        "-d",
        "--output-dir",
        help="Write one <label>_<timestamp>.xplane.pb per session here. "
        "Defaults to each session's phase directory, the one holding "
        "plugins/.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Report what would be merged without writing anything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # An explicit destination is resolved before scanning so the scan can
    # exclude it. Without one, each session writes to its own phase directory,
    # which discovery drops on sight rather than skipping by path.
    explicit_dir = (
        os.path.dirname(os.path.abspath(args.output))
        if args.output
        else args.output_dir
    )

    if explicit_dir and _is_inside_session_dir(explicit_dir):
        print(
            f"Refusing to write into {explicit_dir}: XProf would load the "
            "merged file as an extra host alongside the per-rank files, "
            "double-counting every rank. Choose a directory outside "
            "plugins/profile/.",
            file=sys.stderr,
        )
        return 2

    if args.output:
        skip: tuple[str, ...] = (os.path.abspath(args.output),)
    elif explicit_dir:
        skip = (explicit_dir,)
    else:
        # Captures that are not laid out as `<phase>/plugins/profile/<ts>/`
        # have no phase directory and fall back to `<input>/merged`.
        skip = tuple(
            os.path.join(path, "merged") for path in args.inputs if os.path.isdir(path)
        )

    sessions = discover_sessions(args.inputs, skip=skip)

    if not sessions:
        print(
            f"No {_TRACE_SUFFIX} files found in: {' '.join(args.inputs)}",
            file=sys.stderr,
        )
        return 2

    if args.output and len(sessions) > 1:
        print(
            f"--output takes a single file but {len(sessions)} capture "
            "sessions were found. Use --output-dir, or name one session "
            "directory.",
            file=sys.stderr,
        )
        return 2

    noun = "session" if len(sessions) == 1 else "sessions"
    print(f"Found {len(sessions)} capture {noun}")
    if not args.dry_run:
        # Without this, the plane totals below invite the reading that four
        # input files somehow became seventeen of something.
        print(
            "A plane is one timeline inside a capture: one per TPU chip, one per host,"
        )
        print("plus a shared metadata block. Merging combines planes, not whole files.")

    total_in = total_out = 0
    written: list[str] = []

    for session in sessions:
        source_bytes = sum(os.path.getsize(c.path) for c in session.captures)
        total_in += source_bytes
        count = len(session.captures)
        ranks = [c.rank for c in session.captures if c.rank is not None]
        rank_note = f" (ranks {min(ranks)}-{max(ranks)})" if len(ranks) > 1 else ""

        print(f"\n  {session.label}  {session.timestamp}")
        print(
            f"    input   {count} {'capture' if count == 1 else 'captures'}"
            f"{rank_note}, {_format_size(source_bytes)}"
        )

        if session.missing_ranks:
            gaps = ", ".join(str(r) for r in session.missing_ranks)
            print(
                f"    WARNING: no trace for rank(s) {gaps}. The merged file "
                "will describe only part of the slice."
            )

        destination = args.output or os.path.join(
            explicit_dir or session.default_output_dir, session.output_name
        )
        written.append(destination)

        if args.dry_run:
            print(f"    would write {destination}")
            continue

        result = merge_session(session, destination)
        total_out += result.size
        dropped = (
            (f" ({result.deduplicated} duplicate {_METADATA_PLANE} dropped)")
            if result.deduplicated
            else ""
        )
        print(
            f"    planes  {result.planes_read} read -> {result.planes} written{dropped}"
        )
        print(f"    output  {_format_size(result.size)}")
        print(f"    -> {result.path}")

    # Where the files went, collected in one place: the per-session blocks
    # scroll away, and that is the one thing every caller came for.
    print("\nSummary")
    if args.dry_run:
        print(
            f"  {len(sessions)} {noun} to merge, {_format_size(total_in)} of captures"
        )
    else:
        # Usually a large reduction, from collapsing the duplicated
        # `/host:metadata` planes. Captures small enough that the per-rank
        # tagging outweighs that do grow, and saying so beats a negative
        # "smaller".
        change = ""
        if total_in and total_out != total_in:
            percent = 100 * abs(total_out - total_in) // total_in
            change = (
                f", {percent}% smaller"
                if total_out < total_in
                else f", {percent}% larger"
            )
        print(
            f"  {len(sessions)} {noun} merged, {_format_size(total_in)} -> "
            f"{_format_size(total_out)}{change}"
        )
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
