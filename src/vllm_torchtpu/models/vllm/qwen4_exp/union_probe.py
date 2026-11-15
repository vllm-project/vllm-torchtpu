# SPDX-License-Identifier: Apache-2.0
"""Debug-only DCP prefill union probe (``VLLM_TPU_QSA_PREFILL_UNION_PROBE=<N>``, 0 = off).

Torch side only, no host callback, no op change. Under main-KV DCP, the probed
QSA layers (``VLLM_TPU_QSA_PREFILL_UNION_PROBE_LAYERS``, default the first and
the middle QSA layer) copy the DCP attend op's existing ``selected`` output
(token ids, -1 padded) and the rows' positions into persistent device buffers
on every all-token step (``update_rows_``, in the compiled step). The runner
calls ``poll`` at the start of each ``execute_model``, outside the compiled
program: for every N-th step with at least ``MIN_TOKENS`` scheduled tokens it
reads the buffers the previous step wrote (one device read, that step only)
and, on global rank 0, logs per 128- and 256-row tile of the step's rows:
the owned union of selected 4-token groups (partial tails included) for DCP
rank 0, the owned visible groups, their ratio, the rows' owned selected
groups, and the same union over both ranks (``global``) for comparison with
replicated benches.
"""

import re

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MIN_TOKENS = 4096
RATIO = 4
_PROBES: list = []  # (layer name, layer) of every probed layer
_STATE = {"long_steps": 0, "pending": False, "page_size": 512}


def set_page_size(page_size: int) -> None:
    """Main-KV pool page rows (B); a DCP table entry covers 2 * B tokens."""
    _STATE["page_size"] = int(page_size)


def probe_layer_indices(layer_types) -> set[int]:
    """Decoder layer indices to probe: the env list, else the first and the middle QSA layer."""
    raw = envs.VLLM_TPU_QSA_PREFILL_UNION_PROBE_LAYERS.strip()
    if raw:
        return {int(x) for x in raw.split(",") if x.strip()}
    from vllm_torchtpu.models.vllm.qwen4_exp.config import QSA_LAYER_TYPES

    qsa = [i for i, kind in enumerate(layer_types) if kind in QSA_LAYER_TYPES]
    return {qsa[0], qsa[len(qsa) // 2]} if qsa else set()


def maybe_attach(layer, *, max_tokens: int, width: int, device, layer_types) -> bool:
    """Register the probe buffers on a DCP QSA layer (``bind_pool``, before any compile)."""
    if not envs.VLLM_TPU_QSA_PREFILL_UNION_PROBE:
        return False
    match = re.search(r"layers\.(\d+)\.", getattr(layer, "prefix", ""))
    if match is None or "mtp" in layer.prefix or int(match.group(1)) not in probe_layer_indices(layer_types):
        return False
    if getattr(layer, "_union_probe_selected", None) is None:
        layer.register_buffer(
            "_union_probe_selected",
            torch.full((max_tokens, width), -1, dtype=torch.int32, device=device), persistent=False)
        layer.register_buffer(
            "_union_probe_positions",
            torch.full((max_tokens, 1), -1, dtype=torch.int32, device=device), persistent=False)
        _PROBES.append((layer.prefix, layer))
        logger.info("QSA DCP union probe on %s (%d rows x %d)", layer.prefix, max_tokens, width)
    return True


def stash(layer, selected: torch.Tensor, positions: torch.Tensor) -> None:
    """In the compiled step: this step's ``selected`` / positions into the layer's buffers."""
    from vllm_torchtpu.layers.adapter.custom_ops.rows_update_op import update_rows_

    rows = min(selected.shape[0], layer._union_probe_selected.shape[0])
    update_rows_(layer._union_probe_selected, selected[:rows].to(torch.int32))
    update_rows_(layer._union_probe_positions, positions[:rows].to(torch.int32).reshape(rows, 1))


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:  # noqa: BLE001
        return True


def poll(scheduled_tokens: int) -> None:
    """Host, start of ``execute_model``: log the previous step if it was a sampled long prefill."""
    every = envs.VLLM_TPU_QSA_PREFILL_UNION_PROBE
    if not every or not _PROBES:
        return
    if _STATE["pending"]:
        _STATE["pending"] = False
        if _is_rank0():
            for name, layer in _PROBES:
                sel = layer._union_probe_selected.cpu().numpy()
                pos = layer._union_probe_positions.cpu().numpy()[:, 0]
                logger.info("QSA DCP union probe long-prefill %d %s: %s", _STATE["long_steps"], name,
                            summarize(sel, pos, _STATE["page_size"]))
    if scheduled_tokens >= MIN_TOKENS:
        _STATE["long_steps"] += 1
        if (_STATE["long_steps"] - 1) % every == 0:
            _STATE["pending"] = True


def tile_stats(sel, pos, page_size, *, rank=0, world=2, tile=128):
    """Per tile of live rows: (owned union, owned visible, owned selected per row mean, global union,
    global visible). ``sel``: [rows, width] token ids (-1 none); ``pos``: [rows] positions."""
    live = (sel >= 0).any(axis=1)
    rows = int(np.nonzero(live)[0].max()) + 1 if live.any() else 0
    groups_per_page = page_size // RATIO
    out = []
    for t0 in range(0, rows, tile):
        s, p, lv = sel[t0:t0 + tile], pos[t0:t0 + tile], live[t0:t0 + tile]
        if not lv.any():
            continue
        g = np.unique(s[s >= 0] // RATIO)
        own = g[(g // groups_per_page) % world == rank]
        length = p[lv].astype(np.int64) + 1
        tail = length // RATIO
        tail_page = tail // groups_per_page
        tail_owned = tail_page % world == rank
        below = np.maximum(tail_page - rank + world - 1, 0) // world * groups_per_page + np.where(
            tail_owned, tail % groups_per_page, 0)
        visible_own = int((below + (tail_owned & (length % RATIO != 0))).max())
        per_row = [np.count_nonzero(((np.unique(r[r >= 0] // RATIO) // groups_per_page) % world) == rank)
                   for r in s[lv]]
        out.append((len(own), visible_own, float(np.mean(per_row)), len(g), int(-(-length.max() // RATIO))))
    return np.asarray(out, np.float64).reshape(-1, 5)


def summarize(sel, pos, page_size) -> str:
    parts = []
    for tile in (128, 256):
        st = tile_stats(sel, pos, page_size, tile=tile)
        if not len(st):
            parts.append(f"T{tile}: no live rows")
            continue
        frac = st[:, 0] / np.maximum(st[:, 1], 1)
        gfrac = st[:, 3] / np.maximum(st[:, 4], 1)
        q = lambda x: np.percentile(x, [10, 50, 90]).round(3).tolist()  # noqa: E731
        parts.append(
            f"T{tile}: tiles={len(st)} owned_union p10/50/90={q(st[:, 0])} owned_visible p50={np.median(st[:, 1]):.0f} "
            f"owned_union/visible p10/50/90={q(frac)} rows_owned_sel mean={st[:, 2].mean():.1f} | "
            f"global_union p50={np.median(st[:, 3]):.0f} global_union/visible p10/50/90={q(gfrac)}")
    return " || ".join(parts)
