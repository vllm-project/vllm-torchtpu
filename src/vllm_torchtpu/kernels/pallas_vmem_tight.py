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
"""Size each Pallas call's scoped-VMEM reservation to its working set.

A Pallas call reserves its ``vmem_limit_bytes`` of scoped VMEM for the whole
call (default 32 MiB; many DSV4 kernels pass 100 MiB, which libtpu clamps to
64 MiB - 64 KiB, i.e. all of a v7x TensorCore's VMEM). XLA's memory-space
assignment must keep that much free across the call, so at decode sizes a
64 MiB reservation evicts the producer's VMEM output to HBM right before the
call (a serial ``copy-start``/``copy-done`` the kernel then reads back) and
stops XLA prefetching the next op's operands across it. The kernel itself
needs a few MiB at decode.

``TPU_PALLAS_VMEM_TIGHT=1`` wraps ``pl.pallas_call`` and, per call (i.e. per
traced shape), traces the kernel once to read what Pallas will allocate in
VMEM: pipelined in/out blocks times their buffer count, VMEM scratch, and
every ``run_scoped`` VMEM allocation in the kernel body, and every
array-valued intermediate the body computes (Mosaic's VMEM stack), all
tile-padded and summed as if live at once (an upper bound). The reservation
becomes::

    min(original, max(MIN, ceil_MiB(SLACK * working_set + MARGIN)))

so it never rises and never drops below the traced need (Mosaic's own
internal scratch and spills live in the MARGIN; an undersized limit is a
compile error, not a wrong answer). Same kernels, same tiles, same
arithmetic: outputs are bit-identical. Calls whose kernel cannot be traced
standalone (e.g. collectives that need a mesh axis) are left alone.

``TPU_PALLAS_VMEM_TIGHT=2`` measures instead of bounding: per call shape it
compiles the kernel alone at rungs of a ladder (4 .. 48 MiB, binary search)
and keeps the lowest rung Mosaic accepts (E1001 CompileTimeScopedVmemOom on
the others). The body-value bound of mode 1 is far above the real stack for
unrolled kernels (rope_quant: bound 39 MiB at T=8), which leaves the limits
that matter at 64 MiB; the probe costs a few standalone kernel compiles per
distinct shape at model compile time (memoized).

Knobs: ``TPU_PALLAS_VMEM_TIGHT_MIN_MIB`` (default 8),
``TPU_PALLAS_VMEM_TIGHT_SLACK`` (1.5), ``TPU_PALLAS_VMEM_TIGHT_MARGIN_MIB``
(8), ``TPU_PALLAS_VMEM_TIGHT_SKIP`` (regex on the pallas_call ``name=`` or
kernel function name; default ``_DEFAULT_SKIP``: the mHC seam, owned by
``VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS``, and the fused EP MoE family, which
sizes its own tiles from its budget), ``TPU_PALLAS_VMEM_TIGHT_ONLY`` (regex,
default all), ``TPU_PALLAS_VMEM_TIGHT_DEFAULTED`` (1: also tighten calls that
pass no limit, i.e. the 32 MiB libtpu default). Default off.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import re
import sys
import time

logger = logging.getLogger(__name__)

_MARK = "_torchtpu_vmem_tight_patch"
_MIB = 1 << 20
_LIBTPU_DEFAULT = 32 * _MIB
_V7X_CAP = 64 * _MIB - 64 * 1024
_DEFAULT_SKIP = r"^(_fused_kernel$|fused_ep_moe_v2_|moe_v2_)"
_logged: set = set()
_memo: dict = {}
# Probe ladder (MiB) for TPU_PALLAS_VMEM_TIGHT=2.
_LADDER_MIB = (4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48)


def _mode() -> int:
    try:
        return int(os.environ.get("TPU_PALLAS_VMEM_TIGHT", "0") or 0)
    except ValueError:
        return 0


def _on() -> bool:
    return _mode() > 0


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _space(ms) -> str:
    """'vmem' | 'other' for a ref's memory space (None = the VMEM default)."""
    if ms is None:
        return "vmem"
    s = str(ms).lower()
    if "vmem" in s or s.split(".")[-1] in ("default", "none"):
        return "vmem"
    return "other"


def _nbytes(aval) -> int:
    """Bytes of an array as laid out in VMEM: the minor dim padded to 128 lanes,
    the second-minor to the dtype's sublane tile (8 rows of 32 bits)."""
    inner = getattr(aval, "inner_aval", aval)
    shape = getattr(inner, "shape", None)
    dtype = getattr(inner, "dtype", None)
    if shape is None or dtype is None:
        return 0
    try:
        import numpy as np

        item = np.dtype(dtype).itemsize
    except TypeError:
        return 0  # semaphores and other opaque types
    dims = [int(d) if isinstance(d, int) else 1 for d in shape]
    if not dims:
        return 0  # scalars live in sregs / SMEM
    dims[-1] = -(-dims[-1] // 128) * 128
    sub = 8 * max(1, 4 // max(item, 1))
    if len(dims) >= 2:
        dims[-2] = -(-dims[-2] // sub) * sub
    else:
        dims[-1] = -(-dims[-1] // (128 * sub)) * (128 * sub)
    n = 1
    for d in dims:
        n *= d
    return n * item


def _subjaxprs(eqn):
    for p in eqn.params.values():
        for sub in p if isinstance(p, (tuple, list)) else (p,):
            j = getattr(sub, "jaxpr", sub)
            if hasattr(j, "eqns"):
                yield j


def _body_bytes(jaxpr) -> int:
    """Upper bound on Mosaic's VMEM stack for values computed in the kernel body:
    every array-valued intermediate counted once, as if all were live at once
    (loop and branch bodies once each). Refs are excluded (counted elsewhere)."""
    total = 0
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            av = getattr(v, "aval", None)
            if av is not None and not hasattr(av, "inner_aval"):
                total += _nbytes(av)
        for j in _subjaxprs(eqn):
            total += _body_bytes(j)
    return total


def _run_scoped_bytes(jaxpr) -> int:
    """VMEM bytes of every run_scoped allocation in a kernel jaxpr (recursive)."""
    total = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "run_scoped":
            inner = eqn.params["jaxpr"]
            for v in inner.invars:
                av = v.aval
                if _space(getattr(av, "memory_space", None)) == "vmem":
                    total += _nbytes(av)
        for p in eqn.params.values():
            for sub in p if isinstance(p, (tuple, list)) else (p,):
                j = getattr(sub, "jaxpr", sub)
                if hasattr(j, "eqns"):
                    total += _run_scoped_bytes(j)
    return total


def _working_set(eqn) -> int:
    gm = eqn.params["grid_mapping"]
    grid = tuple(getattr(gm, "grid", ()) or ())
    pipelined = any((g if isinstance(g, int) else 2) > 1 for g in grid) or bool(grid)
    ws = 0
    for bm in gm.block_mappings:
        av = getattr(bm, "transformed_block_aval", None) or bm.block_aval
        if _space(getattr(av, "memory_space", None)) != "vmem":
            continue
        nbuf = 1
        if pipelined:
            nbuf = 2
            pm = getattr(bm, "pipeline_mode", None)
            cnt = getattr(pm, "buffer_count", None)
            if isinstance(cnt, int) and cnt > 0:
                nbuf = cnt
        ws += nbuf * _nbytes(av)
    for av in getattr(gm, "scratch_avals", ()) or ():
        if _space(getattr(av, "memory_space", None)) == "vmem":
            ws += _nbytes(av)
    ws += _run_scoped_bytes(eqn.params["jaxpr"])
    ws += _body_bytes(eqn.params["jaxpr"])
    return ws


def _find_pallas_eqn(jaxpr):
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            return eqn
        for p in eqn.params.values():
            for sub in p if isinstance(p, (tuple, list)) else (p,):
                j = getattr(sub, "jaxpr", sub)
                if hasattr(j, "eqns"):
                    e = _find_pallas_eqn(j)
                    if e is not None:
                        return e
    return None


def tight_limit(ws: int, orig: int) -> int:
    lo = int(_f("TPU_PALLAS_VMEM_TIGHT_MIN_MIB", 8) * _MIB)
    slack = _f("TPU_PALLAS_VMEM_TIGHT_SLACK", 1.5)
    margin = _f("TPU_PALLAS_VMEM_TIGHT_MARGIN_MIB", 8) * _MIB
    want = int(math.ceil((slack * ws + margin) / _MIB)) * _MIB
    return min(orig, max(lo, want))


def install_pallas_vmem_tight() -> None:
    if not _on():
        return
    import dataclasses

    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    orig = pl.pallas_call
    if getattr(orig, _MARK, False):
        return
    skip = os.environ.get("TPU_PALLAS_VMEM_TIGHT_SKIP", _DEFAULT_SKIP)
    only = os.environ.get("TPU_PALLAS_VMEM_TIGHT_ONLY", "")
    skip_re = re.compile(skip) if skip else None
    only_re = re.compile(only) if only else None
    defaulted = os.environ.get("TPU_PALLAS_VMEM_TIGHT_DEFAULTED", "1") not in ("0", "")

    @functools.wraps(orig)
    def pallas_call(kernel, out_shape, *, grid_spec=None, in_specs=None, **kw):
        def build(cp):
            k2 = dict(kw)
            if cp is not None:
                k2["compiler_params"] = cp
            if in_specs is None:
                return orig(kernel, out_shape, grid_spec=grid_spec, **k2)
            return orig(kernel, out_shape, grid_spec=grid_spec, in_specs=in_specs, **k2)

        call = build(None)
        fn = kernel
        while isinstance(fn, functools.partial):
            fn = fn.func
        name = kw.get("name") or getattr(fn, "__name__", "") or ""
        fname = getattr(fn, "__name__", "") or ""
        if skip_re is not None and (skip_re.search(name) or skip_re.search(fname)):
            return call
        if only_re is not None and not (only_re.search(name) or only_re.search(fname)):
            return call
        cp = kw.get("compiler_params")
        if isinstance(cp, dict):  # legacy {"mosaic": {...}} form: leave alone
            return call
        if cp is not None and not isinstance(cp, pltpu.CompilerParams):
            return call
        cur = getattr(cp, "vmem_limit_bytes", None) if cp is not None else None
        if cur is None and not defaulted:
            return call
        orig_lim = min(cur, _V7X_CAP) if cur is not None else _LIBTPU_DEFAULT

        @functools.wraps(call)
        def with_limit(lim):
            if cp is None:
                return pltpu.CompilerParams(vmem_limit_bytes=lim)
            return dataclasses.replace(cp, vmem_limit_bytes=lim)

        def compiles(sds, lim) -> bool | None:
            """True/False = the kernel alone does/does not fit `lim`; None = cannot tell."""
            try:
                jax.jit(build(with_limit(lim))).lower(*sds).compile()
                return True
            except Exception as e:  # noqa: BLE001
                m = str(e)
                if "E1001" in m or "ScopedVmem" in m or "scoped vmem" in m.lower():
                    return False
                return None

        def probe(sds, ub):
            """Lowest rung of the ladder (below the original limit) that compiles."""
            rungs = [r * _MIB for r in _LADDER_MIB if r * _MIB < orig_lim]
            rungs = [r for r in rungs if r >= _f("TPU_PALLAS_VMEM_TIGHT_MIN_MIB", 4) * _MIB]
            if not rungs:
                return orig_lim, 0
            lo, hi, best, n = 0, len(rungs) - 1, orig_lim, 0
            # the upper-bound estimate, when it is a rung, is the first guess
            while lo <= hi:
                mid = (lo + hi) // 2
                ok = compiles(sds, rungs[mid]); n += 1
                if ok is None:
                    return orig_lim, n
                if ok:
                    best, hi = rungs[mid], mid - 1
                else:
                    lo = mid + 1
            return best, n

        def wrapped(*args):
            try:
                sds = [
                    jax.ShapeDtypeStruct(a.shape, a.dtype)
                    if hasattr(a, "shape") and hasattr(a, "dtype") else a
                    for a in args
                ]
                eqn = _find_pallas_eqn(jax.make_jaxpr(call)(*sds).jaxpr)
                ws = _working_set(eqn)
            except Exception as e:  # noqa: BLE001 - untraceable standalone: keep as is
                key = (name, "err")
                if key not in _logged:
                    _logged.add(key)
                    msg = f"TPU_PALLAS_VMEM_TIGHT: {name} left alone ({type(e).__name__}: {str(e)[:120]})"
                    logger.info(msg)
                    print(msg, file=sys.stderr, flush=True)
                return call(*args)
            mkey = (name, tuple((getattr(a, "shape", None), str(getattr(a, "dtype", ""))) for a in args), orig_lim)
            lim = _memo.get(mkey)
            if lim is None:
                ub = tight_limit(ws, orig_lim)
                n, t0 = 0, time.perf_counter()
                if _mode() >= 2:
                    lim, n = probe(sds, ub)
                    if lim < orig_lim:
                        # The kernel inside a program can need more than alone (XLA
                        # may place its operands in VMEM): wo_a_projection 4.09 MiB
                        # in the twin vs a 4 MiB rung alone; mhc post/pre-mixes at
                        # T=128..2048 up to 25 MiB vs a 16 MiB rung in a unit test.
                        # So keep 2x + 4 MiB headroom above the measured rung.
                        want = lim * _f("TPU_PALLAS_VMEM_TIGHT_PROBE_SLACK", 2.0) + _f(
                            "TPU_PALLAS_VMEM_TIGHT_PROBE_MARGIN_MIB", 4) * _MIB
                        up = [r * _MIB for r in _LADDER_MIB if r * _MIB >= want]
                        lim = min(orig_lim, up[0] if up else orig_lim)
                else:
                    lim = ub
                _memo[mkey] = lim
                msg = "TPU_PALLAS_VMEM_TIGHT: %s shapes=%s ws=%.2f MiB ub=%.0f limit %.1f -> %.1f MiB (%d probes, %.2f s)" % (
                    name, tuple(getattr(a, "shape", ()) for a in args[:4]), ws / _MIB, ub / _MIB,
                    orig_lim / _MIB, lim / _MIB, n, time.perf_counter() - t0)
                logger.info(msg)
                print(msg, file=sys.stderr, flush=True)
            if lim >= orig_lim:
                return call(*args)
            return build(with_limit(lim))(*args)

        return wrapped

    setattr(pallas_call, _MARK, True)
    pl.pallas_call = pallas_call
    logger.info(
        "TPU_PALLAS_VMEM_TIGHT=1: per-call scoped VMEM = working set (skip=%r only=%r)",
        skip, only,
    )
