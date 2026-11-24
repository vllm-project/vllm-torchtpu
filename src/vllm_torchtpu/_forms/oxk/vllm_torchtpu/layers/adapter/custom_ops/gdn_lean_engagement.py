# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL GDN engagement counters for VLLM_TPU_GDN_LEAN_GLUE3 (VLLM_TPU_GDN_LEAN_GLUE3_COUNT=N; debug,
log-only, host only, outside the compile key).

VLLM_TPU_GDN_LEAN_GLUE3 (ported from Benliang Wang aa8e6b237, three of his four lean-glue pieces) changes every
pooled GDN call of a stock forward: the step's stacked slot rows (one read-offset gather per step instead of one
per layer), the zero fill of the output written by the BATCHED kernel's DMAs (no XLA zeros_like per layer) and b / a
packed into one operand. Whether a forward runs it is decided by things the host can see (C):

* at model build, each GDN layer whose pooled op is built with the flag (``lean_glue3_applies``: the flag and the
  per-region unified pool) records itself (the stacked slot rows then run on every call with read offsets, i.e.
  every speculative-decode step);
* on the host, the runner counts every executed target forward, and those the megakernel did not take (the
  stock path, where every pooled GDN layer runs the lean op), and logs one INFO line every N forwards::

    q38 GDN-LEAN3 engagement: lean pooled GDN layers L; stock forwards S of N on this rank (zero fill Z) |
    rows R: s/f ...

S > 0 with L > 0 means those forwards' compiled programs contain the lean GDN calls. The zero fill runs on the
same forwards unless VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT skips the fill on prefill-sized buckets (more rows than
request windows), where the PER_SEQ call has nothing to fill (Z counts the forwards whose pooled calls fill). The
device-side proof is the xprof kernel names (``fused_conv1d_gdn_batched_..._zf_ba``) on an annotated run.
"""

from __future__ import annotations

_STATE = {"layers": 0, "steps": 0, "stock": 0, "zero_fill": 0}
_ROWS: dict[int, list[int]] = {}  # rows -> [forwards, stock forwards]


def record_built() -> None:
    """Model build: one GDN layer's pooled op runs VLLM_TPU_GDN_LEAN_GLUE3."""
    _STATE["layers"] += 1


def zero_fill_applies(rows: int, slots: int | None, num_spec_tokens: int) -> bool | None:
    """The pooled op's own fill predicate (layers/core/gdn_attention.pooled_gdn_zero_initialize_out); None when
    it needs the request slots and they are unknown (only under VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT)."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT:
        return True
    if slots is None:
        return None
    from vllm_torchtpu.layers.core.gdn_attention import pooled_gdn_zero_initialize_out

    return pooled_gdn_zero_initialize_out(int(rows), int(slots), int(num_spec_tokens))


def count_step(
    rows: int, every: int, mk: bool = False, slots: int | None = None, num_spec_tokens: int = 0
) -> bool:
    """Host, once per executed target forward (after dispatch). Returns whether this forward ran the lean GDN
    op (a stock forward with at least one lean layer)."""
    rows = int(rows)
    c = _ROWS.setdefault(rows, [0, 0])
    c[0] += 1
    _STATE["steps"] += 1
    stock = not mk and _STATE["layers"] > 0
    if stock:
        c[1] += 1
        _STATE["stock"] += 1
        fill = zero_fill_applies(rows, slots, num_spec_tokens)
        if fill is None:
            _STATE["zero_fill"] = -1
        elif fill and _STATE["zero_fill"] >= 0:
            _STATE["zero_fill"] += 1
    if every and _STATE["steps"] % every == 0:
        from vllm_torchtpu.logger import init_logger

        init_logger(__name__).info("%s", format_line())
    return stock


def format_line() -> str:
    parts = " | ".join("rows %d: %d/%d" % (r, _ROWS[r][1], _ROWS[r][0]) for r in sorted(_ROWS))
    fill = "n/a" if _STATE["zero_fill"] < 0 else str(_STATE["zero_fill"])
    return "q38 GDN-LEAN3 engagement: lean pooled GDN layers %d; stock forwards %d of %d on this rank " \
        "(zero fill %s) | %s" % (_STATE["layers"], _STATE["stock"], _STATE["steps"], fill, parts)


def snapshot() -> dict:
    return dict(_STATE, rows={r: list(v) for r, v in _ROWS.items()})


def reset() -> None:
    """Tests only."""
    for k in _STATE:
        _STATE[k] = 0
    _ROWS.clear()
