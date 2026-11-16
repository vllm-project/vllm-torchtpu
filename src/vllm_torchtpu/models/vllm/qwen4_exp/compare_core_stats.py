# SPDX-License-Identifier: Apache-2.0
"""Debug-only counters of the QSA prefill compare core on served steps.

``VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS=N`` (with VLLM_TPU_QSA_PREFILL_COMPARE_CORE): every pooled QSA layer
keeps an f32 module buffer of ``qsa_compare_core.STATS_SHAPE`` that its op folds each step's counters into
(``qsa_op.build_pooled_qsa_op``): per range of the step's longest row, calls, calls that took the core, live
16-row union groups, 128-slot tiles, rows, the largest union, the largest relative error against the flag-off
call and non-finite calls. Every N seconds the runner reads the buffers, logs one ``QSA_CMP_STATS`` line per
layer and range with calls, and zeroes them, so each line covers the steps since the previous one.
"""

import time

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The work order's k: tiles a live union group over the ~4.03 of its interpolated selections (about 980 tile
# trips over 243 union groups).
INTERPOLATED_TILES = 4.03

_layers: list = []
_state = {"last": None, "index": 0}


def enabled() -> bool:
    return bool(envs.VLLM_TPU_QSA_PREFILL_COMPARE_CORE and envs.VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS)


def register(attention, device) -> torch.Tensor:
    """Gives a pooled QSA layer its counter buffer, lists it for the runner and returns the buffer."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_compare_core import STATS_SHAPE

    if getattr(attention, "_cmp_stats", None) is None:
        attention.register_buffer(
            "_cmp_stats", torch.zeros(STATS_SHAPE, dtype=torch.float32, device=device), persistent=False
        )
        _layers.append(attention)
    return attention._cmp_stats


def after_forward(runner) -> None:
    """Logs and zeroes the counter buffers once every VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS seconds."""
    del runner
    if not _layers:
        return
    now = time.monotonic()
    if _state["last"] is None:
        _state["last"] = now
        return
    if now - _state["last"] < envs.VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS:
        return
    _state["last"] = now
    index = _state["index"]
    _state["index"] += 1
    from vllm_torchtpu.kernels.qwen4_exp.qsa_compare_core import STATS_EDGES

    edges = (0, *STATS_EDGES, 1 << 31)
    for layer in _layers:
        stats = layer._cmp_stats.cpu()
        layer._cmp_stats.zero_()
        for which, row in enumerate(stats.tolist()):
            calls, core, groups, tiles, rows, union_max, max_rel, nonfinite = row
            if not calls:
                continue
            per_group = tiles / groups if groups else 0.0
            logger.info(
                "QSA_CMP_STATS interval=%d layer=%s longest=%d-%d calls=%d core=%d rows=%d live_groups=%d "
                "tiles=%d tiles_per_group=%.3f k=%.3f union_max=%d max_rel=%.5f nonfinite=%d",
                index,
                layer.prefix,
                edges[which],
                edges[which + 1],
                int(calls),
                int(core),
                int(rows),
                int(groups),
                int(tiles),
                per_group,
                per_group / INTERPOLATED_TILES,
                int(union_max),
                max_rel,
                int(nonfinite),
            )
