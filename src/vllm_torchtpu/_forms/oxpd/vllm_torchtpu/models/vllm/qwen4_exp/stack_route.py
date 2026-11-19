# SPDX-License-Identifier: Apache-2.0
"""q38 L2 STACK-MARK: per-bucket route markers for the c64 / c128 served stack car
(VLLM_TPU_Q38_STACK_ROUTE_COUNT=N; debug, log-only, host only, outside the compile key).

The runner calls ``count_step`` once per executed target forward, after dispatch, on the host (never inside a
Dynamo-traced region; ``count_step`` returns at once if it finds itself compiling). Keyed by the forward's padded
rows R it counts, from the code's own predicates (nothing is re-derived here):

* ``fwd``     forwards of R rows;
* ``mk``      of those, forwards the runner's megakernel took (``try_forward`` hit): no stock QSA / HC / GDN route
              runs on them, so they add to no other column;
* ``o1s``     forwards whose target QSA layers take the O1s request-sharded selection: the bound target layer's
              own ``Qwen4ExpQSA._request_shard_applies`` (which needs the layer's request-shard op, i.e. its
              ``_decode_request_shard_mode`` at build, and calls ``qsa_op.request_shard_applies`` with R, the
              forward's request slots ``context_lens.shape[0]``, the layer's TP size, max_query_len and
              VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS as read at build). ``n/a`` when the model has no target
              QSA layer or the forward passes no ``context_lens``;
* ``hcfused`` forwards whose HC mixes take the fused decode kernel: ``hc_engagement.would_fuse(R)`` (fused modules
              recorded at build, ``decode_fused_allows(R, decode_fused_max_rows())``, ``decode_token_tile(R)``);
* ``gdnwt``   forwards run with VLLM_TPU_GDN_WINDOW_TILE on. Labelled ``gdnwt(env)``: the wrapper's other
              condition, ``window_tile.supported(cfg, block, dtype)`` on the BATCHED (decode / verify window) path,
              is decided at trace time from the GDN config and the pool's block shape and is NOT observable on the
              host without a trace, so this column says only that the flag was on for that forward.

Every N forwards each rank logs one line (rows ascending)::

  q38 STACK route rank K: rows R: fwd F mk K o1s A hcfused B gdnwt(env) C | rows R2: ...

and once per process a setup line with the predicates' static inputs::

  q38 STACK route rank K setup: o1s layer L (mode M, tp T, max_query_len Q, min_rows N) | hc modules H (UP_T U),
  HCGATE limits (lo, hi) | gdn window_tile env E (supported() is decided at trace time; not host-observable)
"""

from __future__ import annotations

from types import SimpleNamespace

_COUNTS: dict[int, list[int]] = {}  # rows -> [fwd, o1s, hcfused, gdnwt, mk]
_STATE = {"steps": 0, "layer": None, "layer_name": None, "searched": False, "setup_logged": False}
O1S_NA = -1


def _logger():
    from vllm_torchtpu.logger import init_logger

    return init_logger(__name__)


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - log-only: no TP group in a unit test
        return -1


def _compiling() -> bool:
    try:
        import torch

        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def target_qsa_layer(model):
    """The first target (non-MTP) module that owns the O1s request-shard predicate, or (None, None)."""
    if model is None or not hasattr(model, "named_modules"):
        return None, None
    for name, module in model.named_modules():
        if "mtp" in name:
            continue
        if hasattr(module, "_request_shard_applies") and hasattr(module, "decode_request_shard"):
            return name, module
    return None, None


def o1s_applies(layer, rows: int, slots: int | None) -> int:
    """1 / 0 from the layer's own ``_request_shard_applies``; O1S_NA when it cannot be evaluated on the host."""
    if layer is None or slots is None:
        return O1S_NA
    hidden = SimpleNamespace(shape=(int(rows),))
    metadata = SimpleNamespace(context_lens=SimpleNamespace(shape=(int(slots),)))
    return int(bool(layer._request_shard_applies(hidden, metadata)))


def _setup_line(rank: int) -> str:
    import os

    from vllm_torchtpu.kernels.qwen4_exp.hc_decode import decode_fused_max_rows
    from vllm_torchtpu.models.vllm.qwen4_exp import hc_engagement

    layer, hc = _STATE["layer"], hc_engagement.snapshot()
    o1s = "none" if layer is None else "%s (mode %s, tp %s, max_query_len %s, min_rows %s, op %s)" % (
        _STATE["layer_name"], getattr(layer, "decode_request_shard", None), getattr(layer, "tp_size", None),
        getattr(layer, "max_query_len", None), getattr(layer, "request_shard_min_rows", None),
        getattr(layer, "_request_shard_dcp_op", None) is not None)
    try:
        limits = decode_fused_max_rows()
    except ValueError as e:
        limits = "invalid (%s)" % e
    return ("q38 STACK route rank %d setup: o1s layer %s | hc modules %d (UP_T %d), HCGATE limits %s | gdn "
            "window_tile env %s (supported() is decided at trace time; not host-observable)"
            % (rank, o1s, hc["modules"], hc["up_t"], limits,
               os.environ.get("VLLM_TPU_GDN_WINDOW_TILE", "unset") or "unset"))


def format_line(rank: int) -> str:
    parts = []
    for rows in sorted(_COUNTS):
        fwd, o1s, hc, wt, mk = _COUNTS[rows]
        parts.append("rows %d: fwd %d mk %d o1s %s hcfused %d gdnwt(env) %d"
                     % (rows, fwd, mk, "n/a" if o1s < 0 else str(o1s), hc, wt))
    return "q38 STACK route rank %d: %s" % (rank, " | ".join(parts))


def count_step(model, model_kwargs, rows: int, every: int, mk: bool = False) -> tuple[int, int, int] | None:
    """Host, once per executed target forward (after dispatch). Returns this forward's (o1s, hcfused, gdnwt).
    ``mk``: the runner's megakernel took this forward (``try_forward`` hit), so no stock QSA / HC / GDN route ran:
    it counts in ``fwd`` and ``mk`` only, and returns (0, 0, 0)."""
    if not every or _compiling():
        return None
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp import hc_engagement

    if not _STATE["searched"]:
        _STATE["searched"] = True
        _STATE["layer_name"], _STATE["layer"] = target_qsa_layer(model)
    rows = int(rows)
    c = _COUNTS.setdefault(rows, [0, 0, 0, 0, 0])
    if mk:
        o1s, hc, wt = 0, 0, 0
        c[0] += 1
        c[4] += 1
        _STATE["steps"] += 1
        _maybe_log(every)
        return o1s, hc, wt
    lens = (model_kwargs or {}).get("context_lens") if isinstance(model_kwargs, dict) else None
    slots = int(lens.shape[0]) if lens is not None and hasattr(lens, "shape") else None
    o1s = o1s_applies(_STATE["layer"], rows, slots)
    hc = int(hc_engagement.would_fuse(rows))
    wt = int(bool(envs.VLLM_TPU_GDN_WINDOW_TILE))
    c[0] += 1
    c[1] = O1S_NA if o1s == O1S_NA else (c[1] if c[1] != O1S_NA else 0) + o1s
    c[2] += hc
    c[3] += wt
    _STATE["steps"] += 1
    _maybe_log(every)
    return o1s, hc, wt


def _maybe_log(every: int) -> None:
    if _STATE["steps"] % every:
        return
    rank = _rank()
    log = _logger()
    if not _STATE["setup_logged"]:
        _STATE["setup_logged"] = True
        log.info("%s", _setup_line(rank))
    log.info("%s", format_line(rank))


def snapshot() -> dict:
    return {rows: list(v) for rows, v in _COUNTS.items()}


def reset() -> None:
    """Tests only."""
    _COUNTS.clear()
    _STATE.update(steps=0, layer=None, layer_name=None, searched=False, setup_logged=False)
