# SPDX-License-Identifier: Apache-2.0
"""DEC-HC engagement counters for the fused decode HC mix (VLLM_TPU_HC_DECODE_COUNT; debug, log-only,
outside the compile key).

Whether a decode step runs the fused HC kernel is decided by these things only (C,
``GatedResidual._decode_fused``): the module built the fused ops (TPU_QWEN4_HC_DECODE_FUSED, and the
module was not handed to the MTP draft, which calls ``disable_decode_fused``), and the step's row count
has a decode tile (``hc_decode.decode_token_tile``: at most 512 rows) and, under HCGATE
(VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS = N), is at most N. All are observable without a
trace, so these counts hold on engines that load every compiled program from a cache:

* at model build, each GatedResidual that builds the fused ops records itself (and whether it reads the
  up weight transposed, TPU_QWEN4_HC_DECODE_UP_T); ``disable_decode_fused`` takes it back out;
* on the host, the runner counts every executed target forward and those whose padded row count has a
  decode tile, and logs one INFO line every N forwards:

  DEC-HC engagement: fused decode HC on M HC modules (UP_T u); fused-eligible forwards E of N on this rank

E > 0 with M > 0 means the compiled programs of those forwards contain the fused kernel (one call per
module per forward). The device-side proof is the xprof kernel names (qwen4exp_hc_decode[_combine],
with _upt under UP_T) on every TensorCore plane of an annotated run.

q38 PORT-BL HC, TPU_QWEN4_HC_DECODE_FOLD: the backbone registers at build the hand-offs one folded forward
makes (``register_fold``: one per layer whose next layer has no PLE, plus the final mixer). A stock forward
folds exactly when its rows take the fused decode HC (``fold_rows``, the same predicate as
GatedResidual.decode_fold_rows); forwards the megakernel took (``mk``: D14, forward_segments / whole step)
never run the stock loop and are not counted as folded. With a fold registered, a second INFO line follows
the DEC-HC line:

  DEC-HC fold engagement: folded forwards F of N target forwards on this rank (megakernel K); hand-offs
  per folded forward H (L layer + X final mixer); folded hand-offs T; folded forwards by rows {R: n, ...}

so F / N is the folded share of target forwards and the by-rows map gives it per bucket. Nothing is
registered, counted or logged with the key unset.
"""

from vllm_torchtpu.kernels.qwen4_exp.hc_decode import (
    decode_fused_allows,
    decode_fused_max_rows,
    decode_token_tile,
)

_STATE = {"modules": 0, "up_t": 0, "steps": 0, "eligible": 0}
# q38 PORT-BL HC (TPU_QWEN4_HC_DECODE_FOLD): registered hand-offs per folded forward, host counts.
_FOLD = {"layers": 0, "final": 0}
_FOLD_COUNTS = {"steps": 0, "mk": 0, "folded": 0, "handoffs": 0, "by_rows": {}}


def record_built(up_t: bool) -> None:
    """Model build: one GatedResidual built the fused decode ops."""
    _STATE["modules"] += 1
    _STATE["up_t"] += int(bool(up_t))


def record_disabled(up_t: bool) -> None:
    """``disable_decode_fused``: that module keeps the unfused mix at every token count."""
    _STATE["modules"] -= 1
    _STATE["up_t"] -= int(bool(up_t))


def would_fuse(tokens: int) -> bool:
    """Whether a target forward of ``tokens`` padded rows runs the fused decode HC mix (host-observable; also
    read by q38 STACK-MARK's per-bucket route counter, stack_route.py)."""
    return (
        _STATE["modules"] > 0
        and decode_fused_allows(int(tokens), decode_fused_max_rows())
        and decode_token_tile(int(tokens)) is not None
    )


def register_fold(layers: int, final: int) -> None:
    """Model build (Qwen4ExpBackbone._register_hc_fold, TPU_QWEN4_HC_DECODE_FOLD): one folded forward of
    this stage hands ``layers`` mlp combines to the next layer's attn mix and ``final`` (0 / 1) to the
    final mixer."""
    _FOLD["layers"] = int(layers)
    _FOLD["final"] = int(final)


def fold_handoffs() -> int:
    """Hand-offs per folded forward (0: no fold registered on this rank)."""
    return _FOLD["layers"] + _FOLD["final"]


def fold_rows(tokens: int) -> bool:
    """Whether a stock target forward of ``tokens`` padded rows folds: a fold is registered and the rows take
    the fused decode HC (HCGATE limit and decode tile; GatedResidual._decode_rows, which
    GatedResidual.decode_fold_rows applies)."""
    return (
        fold_handoffs() > 0
        and decode_fused_allows(int(tokens), decode_fused_max_rows())
        and decode_token_tile(int(tokens)) is not None
    )


def _count_fold(tokens: int, mk: bool) -> None:
    c = _FOLD_COUNTS
    c["steps"] += 1
    if mk:
        c["mk"] += 1
        return
    if fold_rows(tokens):
        c["folded"] += 1
        c["handoffs"] += fold_handoffs()
        rows = int(tokens)
        c["by_rows"][rows] = c["by_rows"].get(rows, 0) + 1


def count_step(tokens: int, every: int, mk: bool = False) -> bool:
    """Host, once per executed target forward; logs every ``every`` forwards. Returns whether this
    forward ran the fused decode HC mix. ``mk``: the megakernel took this forward (only read for the
    TPU_QWEN4_HC_DECODE_FOLD count, when a fold is registered)."""
    _STATE["steps"] += 1
    fused = would_fuse(tokens)
    if fused:
        _STATE["eligible"] += 1
    fold = fold_handoffs() > 0
    if fold:
        _count_fold(tokens, mk)
    if every and _STATE["steps"] % every == 0:
        from vllm_torchtpu.logger import init_logger

        log = init_logger(__name__)
        log.info(
            "DEC-HC engagement: fused decode HC on %d HC modules (UP_T %d); fused-eligible forwards %d "
            "of %d on this rank",
            _STATE["modules"], _STATE["up_t"], _STATE["eligible"], _STATE["steps"],
        )
        if fold:
            c = _FOLD_COUNTS
            log.info(
                "DEC-HC fold engagement: folded forwards %d of %d target forwards on this rank (megakernel "
                "%d); hand-offs per folded forward %d (%d layer + %d final mixer); folded hand-offs %d; "
                "folded forwards by rows %s",
                c["folded"], c["steps"], c["mk"], fold_handoffs(), _FOLD["layers"], _FOLD["final"],
                c["handoffs"], "{" + ", ".join(f"{r}: {n}" for r, n in sorted(c["by_rows"].items())) + "}",
            )
    return fused


def fold_snapshot() -> dict:
    """TPU_QWEN4_HC_DECODE_FOLD registration and host counts (tests, readers)."""
    return dict(_FOLD, **{k: (dict(v) if isinstance(v, dict) else v) for k, v in _FOLD_COUNTS.items()})


def snapshot() -> dict:
    return dict(_STATE)


def reset() -> None:
    """Tests only."""
    for k in _STATE:
        _STATE[k] = 0
    for k in _FOLD:
        _FOLD[k] = 0
    for k in _FOLD_COUNTS:
        _FOLD_COUNTS[k] = {} if k == "by_rows" else 0
