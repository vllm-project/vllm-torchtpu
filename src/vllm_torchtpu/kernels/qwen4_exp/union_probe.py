# SPDX-License-Identifier: Apache-2.0
"""Debug-only DCP prefill union probe (``VLLM_TPU_QSA_PREFILL_UNION_PROBE=<N>``, default 0 = off).

Under main-KV DCP, each QSA prefill attention call of a step with at least
``MIN_TOKENS`` live prefill rows computes, in the graph, per 128-row and per
256-row query tile (rows aligned as the step arrives, no per-request
alignment):

- the owned union: distinct owned 4-token groups the tile's rows select,
  partial tail groups included (what the DCP union walk loads per tile);
- the owned visible groups: the tile's longest row's owned causal groups
  (``ceil(local length / 4)``, the dense extent of the tile on this rank);
- the rows' owned selected counts (mean and max over the tile's live rows)
  and their global selected counts (mean).

The few-KB per-tile tensor leaves the program through one unordered
``jax.debug.callback`` per call (no outputs, nothing waits on the host).
The host keeps the first ``N`` such steps (a step = consecutive calls with the
same row lengths; one call per QSA layer) and logs one line per step with
percentiles over the step's tiles and layers, plus a per-layer line for layer
0 and the middle layer. Off: nothing is traced (same jaxpr).
"""

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MIN_TOKENS = 4096
_TILES = (128, 256)

_state = {"steps": 0, "key": None, "layers": [], "done": False}


def _tile_stats(ubits128, live_rows, local_lengths, owned_sel, global_sel, ratio, tile):
    """int32 [tiles, 6]: union, visible, owned-sel mean*16, owned-sel max, global-sel mean*16, live rows."""
    per = tile // 128
    words = ubits128.shape[1]
    ubits = jax.lax.bitcast_convert_type(ubits128, jnp.uint32).reshape(-1, per, words)
    ubits = jax.lax.reduce(ubits, np.uint32(0), jax.lax.bitwise_or, (1,))
    union = jnp.sum(jax.lax.population_count(ubits).astype(jnp.int32), axis=1)
    live = live_rows.reshape(-1, tile)
    count = jnp.maximum(jnp.sum(live, axis=1), 1)
    visible = jnp.max(((local_lengths + ratio - 1) // ratio).reshape(-1, tile), axis=1)
    own = jnp.where(live, owned_sel.reshape(-1, tile), 0)
    glob = jnp.where(live, global_sel.reshape(-1, tile), 0)
    return jnp.stack(
        [
            union,
            visible,
            jnp.sum(own, axis=1) * 16 // count,
            jnp.max(own, axis=1),
            jnp.sum(glob, axis=1) * 16 // count,
            jnp.sum(live, axis=1),
        ],
        axis=1,
    ).astype(jnp.int32)


def record(blocks, lengths, starts, table_columns, *, rank, world, page_size, ratio, steps):
    """Trace the probe into the DCP prefill attention call (``qsa_rpa._prefill_attention``).

    ``table_columns``: the DCP table's entries per row (static); the owned group reach is
    ``table_columns * page_size // ratio``, as the DCP union walk's."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_dense import dcp_local_selection
    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import union_bits_plan

    tokens = blocks.shape[0]
    padded = -(-tokens // 256) * 256
    if padded != tokens:
        blocks = jnp.pad(blocks, ((0, padded - tokens), (0, 0)), constant_values=-1)
        lengths = jnp.pad(lengths, (0, padded - tokens))
    live_rows = lengths > 0
    live_tokens = jnp.sum(live_rows, dtype=jnp.int32)
    local_blocks, local_lengths = dcp_local_selection(
        blocks, lengths, rank=rank, world=world, page_size=page_size, ratio=ratio
    )
    groups_cap = int(table_columns) * int(page_size) // ratio

    def compute():
        _, ubits, _, _ = union_bits_plan(
            local_blocks, local_lengths, group=128, union_cap=groups_cap, groups_cap=groups_cap, ratio=ratio
        )
        owned_sel = jnp.sum(local_blocks >= 0, axis=1, dtype=jnp.int32)
        global_sel = jnp.sum((blocks >= 0) & live_rows[:, None], axis=1, dtype=jnp.int32)
        return [
            _tile_stats(ubits, live_rows, local_lengths, owned_sel, global_sel, ratio, t) for t in _TILES
        ]

    def skip():
        return [jnp.zeros((padded // t, 6), jnp.int32) for t in _TILES]

    stats = jax.lax.cond(live_tokens >= MIN_TOKENS, compute, skip)
    fingerprint = jnp.stack([live_tokens, jnp.sum(lengths, dtype=jnp.int32), starts[-1]])
    jax.debug.callback(
        lambda f, a, b: _host(np.asarray(f), [np.asarray(a), np.asarray(b)], steps), fingerprint, *stats
    )


def _summary(arrays, tile):
    rows = np.concatenate([a for a in arrays if len(a)]) if arrays else np.zeros((0, 6), np.int64)
    rows = rows[rows[:, 5] > 0] if len(rows) else rows
    if not len(rows):
        return f"T{tile}: no live tiles"
    union, visible = rows[:, 0].astype(np.float64), rows[:, 1].astype(np.float64)
    frac = union / np.maximum(visible, 1)
    q = lambda x: np.percentile(x, [10, 50, 90]).round(2).tolist()  # noqa: E731
    return (
        f"T{tile}: tiles={len(rows)} union p10/50/90={q(union)} max={int(union.max())} "
        f"visible p50={float(np.median(visible)):.0f} union/visible p10/50/90={q(frac)} "
        f"rows_owned_sel mean={rows[:, 2].mean() / 16:.1f} max={int(rows[:, 3].max())} "
        f"rows_global_sel mean={rows[:, 4].mean() / 16:.1f}"
    )


def _flush():
    s = _state
    layers = s["layers"]
    if not layers:
        return
    s["steps"] += 1
    tokens = int(s["key"][0])
    parts = [_summary([l[i] for l in layers], t) for i, t in enumerate(_TILES)]
    logger.info("QSA DCP union probe step %d (%d prefill rows, %d layer calls): %s",
                s["steps"], tokens, len(layers), " | ".join(parts))
    for index in sorted({0, len(layers) // 2}):
        logger.info("QSA DCP union probe step %d layer-call %d: %s", s["steps"], index,
                    " | ".join(_summary([layers[index][i]], t) for i, t in enumerate(_TILES)))
    s["layers"] = []


def _host(fingerprint, stats, steps):
    s = _state
    if s["done"]:
        return
    key = tuple(int(x) for x in fingerprint)
    if key[0] < MIN_TOKENS:
        return
    if key != s["key"]:
        _flush()
        if s["steps"] >= steps:
            s["done"] = True
            return
        s["key"] = key
    s["layers"].append([np.asarray(x, np.int64) for x in stats])
