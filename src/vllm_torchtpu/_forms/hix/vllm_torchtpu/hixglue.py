# SPDX-License-Identifier: Apache-2.0
"""q38 HIX-GLUE (HIX-LEAD-8 lane, 2026-11-24): switches and helpers of the HIX 16-row outside-K glue levers (ox form).

Every lever is behind its own registered, hashed, strict env key (envs.py: "0" default, "1" on, anything else raises):

  VLLM_TPU_HXG_SINK_LAYOUT  SL  the MTP draft's shared-selection buffers (layer 47's `_selection_sink`, aliased as the
                                draft's `_selection_source`, and the draft's carried `_selection_rows`) are allocated with
                                their width padded to a multiple of 128 lanes (2051 -> 2176 at the served budget). At the
                                logical width the device layout of these s32[max_tokens, 2051] buffers is column-major
                                ({0,1}, no lane padding), so the draft program transposes each 16.8 MB buffer to gather /
                                update a few rows (T62z j-0bb8a641: copy.550 / copy.551 / remat pair / copy.616, about
                                58 us per step, M). Every reader slices the logical width back; the padding columns are -1
                                and are never read. Values bitwise.
  VLLM_TPU_HXG_GDN_META     GM  the whole step's GDN metadata in ONE vectorised pass at every bucket, the 16-row bucket included:
                                the D07-b1 shared program (`call_shared`, `gdn_blocks_shared`) regardless of
                                VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS (requires VLLM_TPU_MK_GDN_META_DEDUP=1, refused otherwise), and
                                the GDN-SIDECAR accepted-index operand from one gather over the stacked state ids instead of one
                                chain per GDN layer. Replaces about 262 small XLA ops (88 us per step at 16 rows, T62z, M). Integer
                                ops only: bitwise.
  VLLM_TPU_HXG_PLE_ONE      PF0 the resident PLE table lookup of a megakernel bucket (<= 64 rows) in ONE gather of the step's tokens:
                                VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS (80 on the served decode) is treated as 0, so the lookup takes
                                VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS (512; required >= 64) and a 16-token step runs one gather instead
                                of a two-trip lax.map of 8-token gathers (T62z u c3: while.11 22.9 us + 2 gathers 20.3 us per step,
                                kernels/qwen4_exp/ple.py:78 / :28, M). Bitwise (the gather is per token; envs.py's own note).

Each key is read when the code that uses it runs or is traced. Engagement is a per-rank line at bind time
('q38 HIX-GLUE <KEY> bound rank <r>: ...'), printed by every rank process that binds.
"""

from __future__ import annotations

import logging
from typing import Any

KEYS = (
    "VLLM_TPU_HXG_SINK_LAYOUT",
    "VLLM_TPU_HXG_GDN_META",
    "VLLM_TPU_HXG_PLE_ONE",
)
BIND_PREFIX = "q38 HIX-GLUE "
LANES = 128

logger = logging.getLogger("vllm.vllm_torchtpu.hixglue")


def on(key: str) -> bool:
    """True iff the registered strict key `key` is "1" (the registry getter raises on any value but "0" / "1")."""
    assert key in KEYS, key
    from vllm_torchtpu import envs

    return envs.environment_variables[key]() == "1"


def bind_line(key: str, rank: Any, detail: str) -> str:
    """The per-rank bind-time engagement line of `key` (printed on every call: binds run once per rank process)."""
    assert key in KEYS, key
    msg = "%s%s bound rank %s: %s" % (BIND_PREFIX, key, rank, detail)
    logger.info(msg)
    return msg


def rank_of(obj: Any) -> Any:
    """This process's TP rank for a bind line: `obj.tp_rank` when set, else vLLM's TP rank, else '?' (the reader voids '?')."""
    r = getattr(obj, "tp_rank", None)
    if r is not None:
        return r
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001  never stop a bind for a log field
        return "?"


# ---------------------------------------------------------------------------------------------------------- SL
def sink_alloc_width(width: int) -> int:
    """Allocated width of the shared-selection buffers: `width` with SL off, the next multiple of 128 with SL on."""
    if not on("VLLM_TPU_HXG_SINK_LAYOUT"):
        return int(width)
    return -(-int(width) // LANES) * LANES


def pad_cols(t: Any, width: int, fill: int = -1) -> Any:
    """`t [..., w]` -> `[..., width]` padded with `fill` (a no-op, the same tensor, when w == width)."""
    w = int(t.shape[-1])
    if w == int(width):
        return t
    assert w < int(width), (w, width)
    import torch.nn.functional as F

    return F.pad(t, (0, int(width) - w), value=fill)


def ple_min_rows(min_rows: int, chunk_tokens: int, rank: Any) -> int:
    """PF0: the PLE lookup's min-rows gate (0 with VLLM_TPU_HXG_PLE_ONE on, after its bind line; `min_rows` unchanged off)."""
    if not on("VLLM_TPU_HXG_PLE_ONE"):
        return int(min_rows)
    if int(chunk_tokens) < 64:
        raise ValueError("VLLM_TPU_HXG_PLE_ONE=1 needs VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS >= 64 (one gather per megakernel bucket); "
                         "it is %d" % int(chunk_tokens))
    bind_line("VLLM_TPU_HXG_PLE_ONE", rank, "resident PLE lookup: one gather of the step's tokens at every bucket <= %d rows "
              "(VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS %d -> 0, chunk tokens %d)" % (int(chunk_tokens), int(min_rows), int(chunk_tokens)))
    return 0


def logical_cols(t: Any, width: int) -> Any:
    """The first `width` columns of `t` (a no-op, the same tensor, when t is already `width` wide)."""
    if int(t.shape[-1]) == int(width):
        return t
    return t[..., : int(width)]
