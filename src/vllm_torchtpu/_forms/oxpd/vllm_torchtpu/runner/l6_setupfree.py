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
"""q38 L6 lever ``setupfree`` (T-5(b) SETUP-FREE): ping-pong metadata sets for STEADY.

Lever ``setupfree`` (``VLLM_TPU_L6_SETUP_FREE``), read once per worker step
through ``core/l6_ab.py`` (0 / 1 / ab:<classes>; never part of the compile key;
no compiled program and no upload changes). It acts only on STEADY's fast path
(runner/l6_steady.py ``am_try``); off, STEADY's ``_clone`` runs unchanged.

On a steady step STEADY clones one fresh ``AttentionMetadata`` per KV group
(59 at Qwen3.8 TP8) and the per-layer dict from its templates, and the previous
step's set dies when its DraftChunkInputs is released at the next
sample_setup. On, the runner keeps two such sets and hands them out in turns,
rewriting the per-step fields of the one whose turn it is in place (and its
template fields too after STEADY re-recorded its cache with the same group
structure). A set is rewritten only when nothing but the store references it
or any of its objects: the reference counts must equal those of a probe held
only by a store (measured once through the same code), else the step takes
STEADY's clone (``setupfree.busy``). Served, a step's DraftChunkInputs holds
its set until the next sample_setup, so the set of step N is busy at N + 1 and
free at N + 2. The values handed out equal a fresh ``_clone`` of the same
cache: the same layer names in the same order, the same object sharing, and per
object the same keys in the same order holding the same value objects; only the
object identities repeat every two steps. A set whose group structure or
template keys differ from the new cache's is dropped (``setupfree.reset``).

Sizing (tests/q38_l6/bench_l6_seed.py, CPU pod, 59 groups / 37 mamba / 61
layer names): ``_clone`` 40 us, the release of a cloned set 4 us, the in-place
step rewrite 10 us, the template + step rewrite 21 us, the reference-count
guard 1.4 us. The saving is the clone's allocation; the release at
sample_setup is small on the host.

``VLLM_TPU_L6_SETUP_FREE_CHECK=N``: every N rewritten sets STEADY's clone is
built as well and compared on the host (the structure above, value identity).
A mismatch logs one ERROR, counts ``setupfree.mismatch``, hands out the clone
and turns the lever off for the process. ``VLLM_TPU_L6_CHECK_STEPS`` bounds it,
as every L6 CHECK mode.

Engagement (``setupfree.<key>=n`` on the L6AB window line): ``fill`` (a set
built by STEADY's clone and kept), ``reuse`` (a set rewritten in place),
``retemplate`` (of which the templates were rewritten too), ``busy``, ``reset``,
``check``, ``mismatch``.
"""

from __future__ import annotations

import itertools
import os
import sys
from typing import Any, Callable

from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LEVER = "setupfree"

# VLLM_TPU_L6_SETUP_FREE_CHECK, resolved on first use (tests reset it).
_CHECK: list[int | None] = [None]


def check_every() -> int:
    v = _CHECK[0]
    if v is None:
        raw = os.environ.get("VLLM_TPU_L6_SETUP_FREE_CHECK", "").strip()
        v = int(raw) if raw else 0
        if v < 0:
            raise ValueError(f"VLLM_TPU_L6_SETUP_FREE_CHECK must be >= 0, got {v}")
        _CHECK[0] = v
    # VLLM_TPU_L6_CHECK_STEPS bounds every L6 CHECK mode to the first N worker steps.
    return v if _l6_ab.check_allowed() else 0


def on(runner: Any) -> bool:
    """The lever's state for this step; off, any kept sets are dropped."""
    if _l6_ab.on(LEVER) and not runner.__dict__.get("_l6_sf_off", False):
        return True
    runner.__dict__.pop("_l6_sf", None)
    return False


def drop(runner: Any) -> None:
    """Forget the kept sets."""
    runner.__dict__.pop("_l6_sf", None)


_getrc = sys.getrefcount


class _Set:
    __slots__ = ("layer", "objs", "md", "ad", "dicts", "want", "cache")


class _Store:
    __slots__ = ("sig", "cache", "sets", "nxt", "since_check")


def _counts(s: _Set) -> tuple:
    """The per-layer dict's and each object's reference count, read the same
    way for the probe and for a kept set."""
    return _getrc(s.layer), list(map(_getrc, s.objs))


_PROBE: list = []


def _probe() -> tuple[int, int]:
    """(dict count, object count) of a probe dict held by one ``_Set`` slot and
    a probe object held by one list entry: what a kept set that nothing else
    references reads (an object adds one per layer name mapping to it)."""
    if not _PROBE:
        p = _Set()
        p.layer = {}
        p.objs = [object()]
        lc, oc = _counts(p)
        _PROBE.append((lc, oc[0]))
    return _PROBE[0]


def _sig(c: Any) -> tuple:
    return (tuple(c.name_idx), tuple((tuple(t), m) for t, m in c.groups))


def _steps(runner: Any, ctx: Any) -> tuple[dict, dict]:
    """STEADY ``_clone``'s per-step fields (the same keys and objects)."""
    step_all = {
        "input_positions": runner.position_ids,
        "seq_lens": ctx.seq_lens,
        "query_start_loc": ctx.query_start_loc,
        "request_distribution": ctx.request_distribution,
    }
    step_m = dict(
        step_all,
        mamba_slot_read_offsets=ctx.mamba_slot_read_offsets,
        mamba_request_distribution=ctx.mamba_request_distribution,
    )
    return step_all, step_m


def _keep(per_layer: dict, c: Any) -> _Set:
    """A kept set over STEADY's clone `per_layer` of cache `c`."""
    objs: list = [None] * len(c.groups)
    refs = [0] * len(c.groups)
    for name, i in c.name_idx:
        objs[i] = per_layer[name]
        refs[i] += 1
    s = _Set()
    s.layer = per_layer
    s.objs = objs
    s.dicts = [o.__dict__ for o in objs]
    s.md = [d for d, (_t, m) in zip(s.dicts, c.groups) if m]
    s.ad = [d for d, (_t, m) in zip(s.dicts, c.groups) if not m]
    lc, oc = _probe()
    s.want = (lc, [oc + k for k in refs])
    s.cache = c
    return s


def clone(c: Any, runner: Any, ctx: Any, stock: Callable[[Any, Any, Any], dict]) -> dict:
    """STEADY's per-layer metadata for a fast step (cache `c`): the kept set
    whose turn it is, rewritten in place when it is free; else
    ``stock(c, runner, ctx)`` (STEADY's ``_clone``), kept when that set does
    not exist yet."""
    st = runner.__dict__.get("_l6_sf")
    if st is None or st.cache is not c:
        sig = _sig(c)
        if st is None or st.sig != sig:
            if st is not None:
                _l6_ab.note(LEVER, "reset")
            since = st.since_check if st is not None else 0
            st = _Store()
            st.sig = sig
            st.sets = [None, None]
            st.nxt = 0
            st.since_check = since
            runner._l6_sf = st
        st.cache = c
    k = st.nxt
    st.nxt = 1 - k
    s = st.sets[k]
    if s is None:
        per_layer = stock(c, runner, ctx)
        st.sets[k] = _keep(per_layer, c)
        _l6_ab.note(LEVER, "fill")
        return per_layer
    if _counts(s) != s.want:
        # Something still holds this set or one of its objects: never rewrite it.
        _l6_ab.note(LEVER, "busy")
        return stock(c, runner, ctx)
    step_all, step_m = _steps(runner, ctx)
    if s.cache is not c:
        # STEADY re-recorded its cache with the same structure: new templates.
        list(map(dict.update, s.dicts, [t for t, _m in c.groups]))
        s.cache = c
        _l6_ab.note(LEVER, "retemplate")
    list(map(dict.update, s.md, itertools.repeat(step_m)))
    list(map(dict.update, s.ad, itertools.repeat(step_all)))
    _l6_ab.note(LEVER, "reuse")
    out = s.layer
    if _check_due(st):
        _l6_ab.note(LEVER, "check")
        ref = stock(c, runner, ctx)
        why = _differ(ref, out)
        if why is not None:
            _disable(runner, why)
            return ref
    return out


def _check_due(st: _Store) -> bool:
    n = check_every()
    if n <= 0:
        return False
    st.since_check += 1
    if st.since_check >= n:
        st.since_check = 0
        return True
    return False


def _differ(ref: dict, out: dict) -> str | None:
    """None when `out` equals the fresh clone `ref`: the same layer names in the
    same order, the same object sharing, and per object the same type and the
    same keys in the same order holding the same value objects."""
    if list(ref) != list(out):
        return "layer names"
    first_r: dict[int, str] = {}
    first_o: dict[int, str] = {}
    for name in ref:
        a, b = ref[name], out[name]
        if first_r.setdefault(id(a), name) != first_o.setdefault(id(b), name):
            return f"{name}: object sharing"
        if type(a) is not type(b):
            return f"{name}: type"
        da, db = a.__dict__, b.__dict__
        if list(da) != list(db):
            return f"{name}: keys"
        for key, v in da.items():
            if db[key] is not v:
                return f"{name}.{key}: value"
    return None


def _disable(runner: Any, detail: str) -> None:
    runner._l6_sf_off = True
    drop(runner)
    _l6_ab.note(LEVER, "mismatch")
    logger.error(
        "q38 L6 SETUPFREE: CHECK mismatch in the metadata set (%s); the lever is off "
        "for the rest of this process",
        detail,
    )
