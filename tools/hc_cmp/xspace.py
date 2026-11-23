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
"""Reads an xprof capture (tensorflow.profiler.XSpace, the .xplane.pb file) from its
protobuf wire format, so the capture needs no protobuf package; planes, lines and event
metadata decode on first use.

Only the fields the capture tools read are decoded: XSpace.planes and hostnames; XPlane
id, name, lines, event_metadata and stat_metadata; XLine name, display_name,
timestamp_ns and events; XEvent metadata_id, offset_ps and duration_ps; XEventMetadata
id, name, display_name and stats; XStat's value fields.
"""

from __future__ import annotations

import gzip
import struct


def _varint(b, i):
    c = b[i]
    if c < 0x80:
        return c, i + 1
    val, shift = c & 0x7F, 7
    i += 1
    while True:
        c = b[i]
        i += 1
        val |= (c & 0x7F) << shift
        if c < 0x80:
            return val, i
        shift += 7


def fields(b, i=0, end=None):
    """(field number, wire type, value) of a message: an int for varints, (start, end)
    for length-delimited fields, the raw bytes for fixed 64 and 32 bit fields."""
    end = len(b) if end is None else end
    while i < end:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            yield fn, 0, v
        elif wt == 2:
            n, i = _varint(b, i)
            yield fn, 2, (i, i + n)
            i += n
        elif wt == 1:
            yield fn, 1, bytes(b[i : i + 8])
            i += 8
        elif wt == 5:
            yield fn, 5, bytes(b[i : i + 4])
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt} at byte {i}")


def first_field(b, want, i=0, end=None):
    """The (start, end) of the first length-delimited field ``want``, or None."""
    for fn, wt, v in fields(b, i, end):
        if fn == want and wt == 2:
            return v
    return None


def _str(b, span):
    return bytes(b[span[0] : span[1]]).decode("utf-8", "replace")


def _signed(v):
    return v - (1 << 64) if v >= 1 << 63 else v


class EventMetadata:
    __slots__ = ("id", "name", "display_name", "stats")

    def __init__(self, id_, name, display_name, stats):
        self.id = id_
        self.name = name
        self.display_name = display_name
        self.stats = stats  # stat name -> value (bytes values as bytes)


class Line:
    """One timeline of a plane; ``events`` are (metadata id, offset ps, duration ps)."""

    def __init__(self, space, span):
        self._b = space._b
        self._span = span
        self.name = self.display_name = ""
        self.timestamp_ns = 0
        self._events = []
        for fn, wt, v in fields(self._b, *span):
            if fn == 2:
                self.name = _str(self._b, v)
            elif fn == 11:
                self.display_name = _str(self._b, v)
            elif fn == 3:
                self.timestamp_ns = _signed(v)
            elif fn == 4:
                self._events.append(v)
        self._decoded = None

    def __len__(self):
        return len(self._events)

    @property
    def events(self) -> list:
        if self._decoded is None:
            b = self._b
            out = []
            ap = out.append
            for i, j in self._events:
                mid = off = dur = 0
                while i < j:
                    key = b[i]
                    i += 1
                    if key & 7 == 0:
                        c = b[i]
                        if c < 0x80:
                            val = c
                            i += 1
                        else:
                            val, i = _varint(b, i)
                        fn = key >> 3
                        if fn == 1:
                            mid = val
                        elif fn == 2:
                            off = val
                        elif fn == 3:
                            dur = val
                    elif key & 7 == 2:
                        n, i = _varint(b, i)
                        i += n
                    elif key & 7 == 1:
                        i += 8
                    elif key & 7 == 5:
                        i += 4
                    else:
                        raise ValueError(
                            f"unsupported wire type in an event at byte {i}"
                        )
                ap((mid, off, dur))
            self._decoded = out
            self._events = None
        return self._decoded

    def timed(self):
        """(start ps since the epoch, duration ps, metadata id) of each event."""
        t0 = self.timestamp_ns * 1000
        return [(t0 + off, dur, mid) for mid, off, dur in self.events]


class Plane:
    def __init__(self, space, span):
        self._space = space
        b = space._b
        self.id = 0
        self.name = ""
        self._lines, self._evmd, self._stmd, self._stats = [], [], [], []
        for fn, wt, v in fields(b, *span):
            if fn == 1:
                self.id = _signed(v)
            elif fn == 2:
                self.name = _str(b, v)
            elif fn == 3:
                self._lines.append(v)
            elif fn == 4:
                self._evmd.append(v)
            elif fn == 5:
                self._stmd.append(v)
            elif fn == 6:
                self._stats.append(v)
        self._lines_obj = None
        self._event_md = None
        self._stat_md = None

    @property
    def lines(self) -> list:
        if self._lines_obj is None:
            self._lines_obj = [Line(self._space, s) for s in self._lines]
        return self._lines_obj

    def line(self, name):
        return next((ln for ln in self.lines if ln.name == name), None)

    @property
    def stats(self) -> dict:
        """The plane's own stats, stat name -> value."""
        smd = self.stat_metadata
        return dict(_stat(self._space._b, v, smd) for v in self._stats)

    @property
    def stat_metadata(self) -> dict:
        """Stat metadata id -> stat name."""
        if self._stat_md is None:
            b = self._space._b
            out = {}
            for s, e in self._stmd:
                val = first_field(b, 2, s, e)
                if val is None:
                    continue
                sid, name = 0, ""
                for fn, wt, v in fields(b, *val):
                    if fn == 1:
                        sid = _signed(v)
                    elif fn == 2:
                        name = _str(b, v)
                out[sid] = name
            self._stat_md = out
        return self._stat_md

    @property
    def event_metadata(self) -> dict:
        """Event metadata id -> EventMetadata."""
        if self._event_md is None:
            b = self._space._b
            smd = self.stat_metadata
            out = {}
            for s, e in self._evmd:
                val = first_field(b, 2, s, e)
                if val is None:
                    continue
                eid, name, disp, stats = 0, "", "", {}
                for fn, wt, v in fields(b, *val):
                    if fn == 1:
                        eid = _signed(v)
                    elif fn == 2:
                        name = _str(b, v)
                    elif fn == 4:
                        disp = _str(b, v)
                    elif fn == 5:
                        k, x = _stat(b, v, smd)
                        stats[k] = x
                out[eid] = EventMetadata(eid, name, disp, stats)
            self._event_md = out
        return self._event_md


def _stat(b, span, smd):
    sid, val = 0, None
    for fn, wt, v in fields(b, *span):
        if fn == 1:
            sid = _signed(v)
        elif fn == 2:
            val = struct.unpack("<d", v)[0]
        elif fn == 3:
            val = v
        elif fn == 4:
            val = _signed(v)
        elif fn == 5:
            val = _str(b, v)
        elif fn == 6:
            val = bytes(b[v[0] : v[1]])
        elif fn == 7:
            val = smd.get(v, v)
    return smd.get(sid, str(sid)), val


class XSpace:
    def __init__(self, data: bytes):
        self._b = memoryview(data)
        self._planes = []
        self.hostnames = []
        for fn, wt, v in fields(self._b):
            if fn == 1:
                self._planes.append(v)
            elif fn == 4:
                self.hostnames.append(_str(self._b, v))
        self._plane_obj = None

    @property
    def planes(self) -> list:
        if self._plane_obj is None:
            self._plane_obj = [Plane(self, s) for s in self._planes]
        return self._plane_obj

    def plane(self, name):
        return next((p for p in self.planes if p.name == name), None)


def load(path: str) -> XSpace:
    """The capture at ``path`` (gzip-compressed or not)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return XSpace(data)
