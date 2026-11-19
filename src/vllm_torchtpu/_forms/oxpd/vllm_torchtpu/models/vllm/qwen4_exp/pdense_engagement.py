# SPDX-License-Identifier: Apache-2.0
"""P-DENSE engagement counters (VLLM_TPU_PDENSE_COUNT; debug, log-only, outside the compile key).

The P-DENSE paths are the one-hot PLE conv read (VLLM_TPU_PLE_CONV_ONEHOT), the fused PLE layer
(VLLM_TPU_PLE_FUSED) and the HC deferral (TPU_QWEN4_HC_PREFILL_DEFER_WHOLE / ..._DEFER_COMBINE). Each stage process
registers what it built (outside every traced forward), and the runner counts every executed prefill step on the host
with the same row rules the traced forward uses (see "Host-side engagement" below), logging one INFO line every N
steps:

  P-DENSE engagement: PLE one-hot conv steps A, PLE fused steps B, HC deferred steps C (hand-offs H) of N
  prefill steps on this stage process

A path's count is > 0 only on a stage process that holds the layer and only if the path was compiled for
the row counts that actually ran. The accuracy bundle's gate (q38_pdense_acc1) reads these lines.
"""

from vllm_torchtpu import envs

# Trace-time records are opt-in (VLLM_TPU_PDENSE_TRACE_RECORDS, default off), decided once at import, outside every
# traced forward: unless it is on, the call sites skip the record_* calls (and record_* return before touching their
# arguments), so the counters are never traced. Traced with a dynamic token dimension (vLLM's profile run) they failed
# engine init with ConstraintViolationError (q38 p75 j-8eb772a8, every arm): _rows' int() of a SymInt specializes the
# marked-dynamic size. FR10b's PP8 env sets VLLM_TPU_PDENSE_COUNT=200, so COUNT no longer controls trace-time
# recording (it keeps only the host-side logging).
ENABLED = envs.VLLM_TPU_PDENSE_TRACE_RECORDS

_TRACED = {"ple_onehot": set(), "ple_fused": set(), "hc_defer": {}}  # rows -> path (hc: hand-offs per forward)
_COUNTS = {"steps": 0, "ple_onehot": 0, "ple_fused": 0, "hc_defer": 0, "hc_handoffs": 0}


def _rows(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):  # a symbolic size: no static record
        return None


def record_ple(path: str, rows) -> None:
    """Trace time: the PLE layer took ``path`` ("ple_onehot" or "ple_fused") for steps of ``rows`` rows."""
    if not ENABLED:  # before touching any argument (a SymInt while tracing)
        return
    r = _rows(rows)
    if r is not None:
        _TRACED[path].add(r)


def record_hc_defer(rows, handoffs: int) -> None:
    """Trace time: a stage forward of ``rows`` rows handed ``handoffs`` mlp combines to the next layer."""
    if not ENABLED:  # before touching any argument (a SymInt while tracing)
        return
    r = _rows(rows)
    if r is not None and handoffs:
        _TRACED["hc_defer"][r] = int(handoffs)


# Host-side engagement (VLLM_TPU_PDENSE_COUNT). Serving's Dynamo traces are all dynamic in the token dimension
# (compilation/shape_variants.py: "vLLM traces the model once with the token dimension dynamic"; retrace keeps it),
# so a trace-time record keyed by row count can never fire in serving. Instead each stage process registers what it
# BUILT, outside every traced forward (Qwen4ExpPooledPLE.bind_pool, Qwen4ExpBackbone.__init__), and the runner counts
# every executed prefill forward on the host with the SAME row rules the traced forward uses (fused_rows,
# defer_whole_rows: one function each, called by the model and by count_step, so they cannot drift). Log-only,
# outside the compile key; read by the accuracy bundle's engagement rule (AG-1 clauses section 3).
_BUILT = {"ple": {}, "hc": None}  # ple: id(module) -> (onehot conv, fused tile or 0); hc: (whole tile, hand-offs, any)


def fused_rows(rows, tile):
    """The fused PLE's row rule (whole fused tiles): Qwen4ExpPooledPLE._fused_applies (traced) and count_step."""
    return rows >= tile and rows % tile == 0


def defer_whole_rows(rows, tile):
    """TPU_QWEN4_HC_PREFILL_DEFER_WHOLE's row rule (at least two whole HC prefill tiles): Qwen4ExpBackbone's stage
    loop (traced) and count_step."""
    return rows >= 2 * tile and rows % tile == 0


def register_ple(module, onehot: bool, fused_tile: int) -> None:
    """Cache bind (Qwen4ExpPooledPLE.bind_pool, before Dynamo traces): this PLE layer reads its history taps by the
    one-hot conv (VLLM_TPU_PLE_CONV_ONEHOT) and/or runs the fused PLE on whole ``fused_tile``-row steps (0 = never)."""
    _BUILT["ple"][id(module)] = (bool(onehot), int(fused_tile))


def register_hc_defer(whole_tile: int, handoffs: int, every_step: bool = False) -> None:
    """Model build (Qwen4ExpBackbone.__init__): this stage hands ``handoffs`` mlp combines per deferred forward, on
    steps whose rows meet defer_whole_rows(rows, whole_tile) (DEFER_WHOLE; 0 = off) or on every step (the older
    DEFER_COMBINE, ``every_step``)."""
    on = bool(handoffs) and (bool(whole_tile) or bool(every_step))
    _BUILT["hc"] = (int(whole_tile), int(handoffs), bool(every_step)) if on else None


def count_step(tokens: int, every: int) -> dict:
    """Host, once per executed prefill forward on this stage process (no speculative input, no sequence
    parallelism: the PP8 prefill stages); logs every ``every`` steps. Counts from the registered builds and the
    shared row rules, never from trace-time records."""
    t = int(tokens)
    _COUNTS["steps"] += 1
    fused = onehot = False
    for oh, tile in _BUILT["ple"].values():
        f = bool(tile) and bool(fused_rows(t, tile))
        fused = fused or f
        onehot = onehot or (oh and not f)   # the fused PLE replaces the conv on its steps
    hc = _BUILT["hc"]
    defer = hc is not None and (hc[2] or (bool(hc[0]) and bool(defer_whole_rows(t, hc[0]))))
    hit = {"ple_onehot": onehot, "ple_fused": fused, "hc_defer": defer}
    for k, v in hit.items():
        if v:
            _COUNTS[k] += 1
    if defer:
        _COUNTS["hc_handoffs"] += hc[1]
    if every and _COUNTS["steps"] % every == 0:
        from vllm_torchtpu.logger import init_logger

        init_logger(__name__).info(
            "P-DENSE engagement: PLE one-hot conv steps %d, PLE fused steps %d, HC deferred steps %d "
            "(hand-offs %d) of %d prefill steps on this stage process",
            _COUNTS["ple_onehot"], _COUNTS["ple_fused"], _COUNTS["hc_defer"], _COUNTS["hc_handoffs"],
            _COUNTS["steps"],
        )
    return hit


def reset_counts() -> None:
    """Tests only: zero the host counts, keep the registered builds."""
    for k in _COUNTS:
        _COUNTS[k] = 0


def reset_traced() -> None:
    """Tests only: clear the trace-time records, keep the builds and counts."""
    for v in _TRACED.values():
        v.clear()


def reset() -> None:
    """Tests only."""
    for v in _TRACED.values():
        v.clear()
    _BUILT["ple"].clear()
    _BUILT["hc"] = None
    for k in _COUNTS:
        _COUNTS[k] = 0
