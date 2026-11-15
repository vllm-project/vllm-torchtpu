# SPDX-License-Identifier: Apache-2.0
"""Debug-only dump of the target's QSA selections on sampled verify steps.

``VLLM_TPU_QSA_SEL_DUMP=N``: every target QSA layer copies the first N rows of
its ``selected`` output (``qsa_rpa.selected_tokens`` layout, one row per step
token) into a module buffer, written in the forward with the same donating row
update as the MTP draft's selection sink. ``VLLM_TPU_QSA_SEL_DUMP_AT="t0:dt:n"``
picks the steps: the first sample ``t0`` seconds after the first verify step
that has at least ``MIN_REQUESTS`` requests of ``K + 1`` rows, then one every
``dt`` seconds after the previous sample, ``n`` samples. On a sampled step the runner reads the
buffers back (tensor-parallel rank 0 only), saves them with the step's query
start locations and positions to ``$CDK_OUTPUT_DIR/seldump`` and logs a summary
line. With the flag unset nothing is registered and the forward is unchanged.

The statistics follow the served decode geometry: 4-token compression groups,
512-token main-KV pages (128 groups, the DCP ownership unit), DCP size 4 and a
512-token RPA KV tile of 4-token pages.
"""

import json
import os
import threading
import time

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

RATIO = 4
UNIT_GROUPS = 128
DCP = 4
TILE_TOKENS = 512
MIN_REQUESTS = 16
BUCKETS = ((0, 16384), (16384, 65536), (65536, 131072), (131072, 1 << 40))

_state = {"first": None, "due": None, "taken": 0, "busy": False}


def _attention_layers(model):
    # The target's QSA layers, found as MTP bind_shared_selection finds them
    # (Qwen4ExpBackbone has no `norm`, so the generic backbone lookup misses it).
    backbone = getattr(model, "model", None)
    return [
        layer.self_attn
        for layer in getattr(backbone, "layers", ())
        if getattr(layer, "layer_type", None) == "full_attention"
    ]


def bind(target_model) -> int:
    """Registers the dump buffer on every target QSA layer; returns the count."""
    rows = envs.VLLM_TPU_QSA_SEL_DUMP
    if not rows:
        return 0
    layers = _attention_layers(target_model)
    for attention in layers:
        buf = torch.full(
            (rows, attention.max_selected_tokens),
            -1,
            dtype=torch.int32,
            device=attention.rope_cache.device,
        )
        attention.register_buffer("_selection_dump", buf, persistent=False)
    logger.info("QSA selection dump: %d target layers, %d rows each", len(layers), rows)
    return len(layers)


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return int(os.environ.get("RANK", "0"))


def _schedule() -> tuple[float, float, int]:
    # ":" separates the fields so the value fits a comma-separated env list.
    t0, dt, n = (envs.VLLM_TPU_QSA_SEL_DUMP_AT or "240:60:6").split(":")
    return float(t0), float(dt), int(n)


def after_forward(runner, num_reqs: int, chunk_index: int) -> None:
    """Saves the selection buffers when this step is a sampled verify step."""
    if not envs.VLLM_TPU_QSA_SEL_DUMP or chunk_index != 0:
        return
    t0, dt, n = _schedule()
    if _state["taken"] >= n or _state["busy"]:
        return
    qsl = runner.query_start_loc_np[: num_reqs + 1].astype(np.int64)
    rows_per_req = np.diff(qsl)
    k1 = int(np.bincount(rows_per_req).argmax()) if len(rows_per_req) else 0
    verify = k1 > 1 and int(np.sum(rows_per_req == k1)) >= MIN_REQUESTS
    if not verify:
        return
    now = time.monotonic()
    if _state["first"] is None:
        _state["first"] = now
        _state["due"] = now + t0
        logger.info("QSA selection dump: first verify step (%d requests)", num_reqs)
    if now < _state["due"]:
        return
    sample = _state["taken"]
    _state["taken"] += 1
    # Counted from this sample, so a pause in traffic does not bunch the rest.
    _state["due"] = now + dt
    if _tp_rank() != 0:
        return
    layers = _attention_layers(runner.model)
    if not layers or getattr(layers[0], "_selection_dump", None) is None:
        return
    tokens = int(qsl[-1])
    rows = min(tokens, envs.VLLM_TPU_QSA_SEL_DUMP)
    selected = np.stack(
        [layer._selection_dump[:rows].cpu().numpy() for layer in layers]
    )
    positions = runner.positions_np[:tokens].astype(np.int64).copy()
    seq_lens = runner.seq_lens_np[:num_reqs].astype(np.int64).copy()
    _state["busy"] = True
    threading.Thread(
        target=_save,
        args=(sample, now - _state["first"], selected, qsl, positions, seq_lens),
        daemon=True,
    ).start()


def _save(sample, elapsed, selected, qsl, positions, seq_lens) -> None:
    try:
        out = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "seldump")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"sample{sample:02d}.npz")
        tmp = f"/tmp/seldump_sample{sample:02d}.npz"
        np.savez_compressed(
            tmp,
            selected=selected,
            query_start_loc=qsl,
            positions=positions,
            seq_lens=seq_lens,
            elapsed=np.float64(elapsed),
        )
        with open(tmp, "rb") as src, open(path, "wb") as dst:
            dst.write(src.read())
        os.remove(tmp)
        stats = summarize(step_records(selected, qsl, positions))
        logger.info(
            "SELDUMP sample=%d t=%.0fs requests=%d layers=%d file=%s %s",
            sample,
            elapsed,
            len(qsl) - 1,
            len(selected),
            path,
            json.dumps(stats, sort_keys=True),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("QSA selection dump sample %d failed: %r", sample, e)
    finally:
        _state["busy"] = False


def row_tokens(row: np.ndarray) -> np.ndarray:
    """The distinct selected token positions of one ``selected_tokens`` row."""
    return np.unique(row[row >= 0])


def request_record(rows: list[np.ndarray], context: int | None = None) -> dict:
    """Pages, union, runs and RPA descriptors of one request's rows in one layer.

    ``rows``: the ``selected_tokens`` rows of a request's K + 1 verify tokens;
    ``context``: the first row's causal length (default: the largest selected
    token + 1).
    A page is a 4-token compression group; a run is a stretch of consecutive
    groups inside one 128-group ownership unit (contiguous in the pool).
    """
    toks = [row_tokens(r) for r in rows]
    groups = [np.unique(t // RATIO) for t in toks]
    union = np.unique(np.concatenate(groups)) if groups else np.empty(0, np.int64)
    if context is None:
        context = int(max((t.max() + 1 for t in toks if len(t)), default=0))
    rec = {
        "context": context,
        "pages": [int(len(g)) for g in groups],
        "union": int(len(union)),
        "ranks": [],
    }
    for rank in range(DCP):
        own = [g[(g // UNIT_GROUPS) % DCP == rank] for g in groups]
        own_union = union[(union // UNIT_GROUPS) % DCP == rank]
        own_tokens = [
            int(np.sum(((t // RATIO) // UNIT_GROUPS) % DCP == rank)) for t in toks
        ]
        tiles = [-(-x // TILE_TOKENS) for x in own_tokens]
        real = [-(-x // RATIO) for x in own_tokens]
        rec["ranks"].append(
            {
                "pages": [int(len(o)) for o in own],
                "union": int(len(own_union)),
                "union_runs": _runs(own_union),
                "row_runs": [_runs(o) for o in own],
                "tiles": tiles,
                "zero_size": int(
                    sum(t * (TILE_TOKENS // RATIO) - r for t, r in zip(tiles, real))
                ),
                "descriptors": int(sum(t * (TILE_TOKENS // RATIO + 1) for t in tiles)),
            }
        )
    return rec


def _runs(groups: np.ndarray) -> int:
    if not len(groups):
        return 0
    breaks = (np.diff(groups) != 1) | (
        groups[1:] // UNIT_GROUPS != groups[:-1] // UNIT_GROUPS
    )
    return int(1 + np.count_nonzero(breaks))


def step_records(
    selected: np.ndarray, qsl: np.ndarray, positions: np.ndarray | None = None
) -> list[dict]:
    """``request_record`` for every request with the step's modal row count."""
    rows_per_req = np.diff(qsl)
    k1 = int(np.bincount(rows_per_req).argmax())
    recs = []
    for layer in range(selected.shape[0]):
        for req in range(len(qsl) - 1):
            lo, hi = int(qsl[req]), int(qsl[req + 1])
            if hi - lo != k1 or hi > selected.shape[1]:
                continue
            context = None if positions is None else int(positions[lo]) + 1
            rec = request_record([selected[layer, i] for i in range(lo, hi)], context)
            if rec["union"]:
                rec.update(layer=layer, request=req)
                recs.append(rec)
    return recs


def summarize(recs: list[dict]) -> dict:
    """Page-weighted union factors overall and on DCP rank 0, by context bucket.

    rho = distinct pages over a request's rows / one row's pages (the mean of
    its rows); weighted by pages: sum of unions over sum of mean row pages.
    """
    out = {}
    for lo, hi in ((0, 1 << 40), *BUCKETS):
        sel = [r for r in recs if lo <= r["context"] < hi]
        if not sel:
            continue
        if lo == 0 and hi == 1 << 40:
            key = "all"
        elif hi == 1 << 40:
            key = f">{lo // 1024}k"
        else:
            key = f"{lo // 1024}k-{hi // 1024}k"
        one = sum(np.mean(r["pages"]) for r in sel)
        own = [r["ranks"][0] for r in sel]
        own_one = sum(np.mean(o["pages"]) for o in own)
        out[key] = {
            "n": len(sel),
            "rho": round(sum(r["union"] for r in sel) / one, 3),
            "rho_r0": round(sum(o["union"] for o in own) / max(own_one, 1e-9), 3),
            "union_runs_r0": sum(o["union_runs"] for o in own),
            "union_r0": sum(o["union"] for o in own),
            "row_runs_r0": sum(sum(o["row_runs"]) for o in own),
            "row_pages_r0": sum(sum(o["pages"]) for o in own),
            "zero_size_r0": sum(o["zero_size"] for o in own),
            "descriptors_r0": sum(o["descriptors"] for o in own),
        }
    return out
