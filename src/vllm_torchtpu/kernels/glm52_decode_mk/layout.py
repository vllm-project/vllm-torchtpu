# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Host-side layouts shared by the kernel, its wrapper and the adapter: the per-layer smalls row, the operand plan
(stacked dense weight kinds, expert kinds, cache kinds) and the static dense-tile stream (csq's DenseWeightRing
stream table `dtab`, explore/clean-sheet-qwen/probe/kw/csq_proto.py `tables`)."""

import jax.numpy as jnp
import numpy as np

from . import pool_alias
from .config import rup

FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32
U32 = jnp.uint32
U8 = jnp.uint8
I32 = jnp.int32


def smalls_layout(cfg):
    """name -> (offset, length) of the per-layer f32 row (every field 128-lane aligned); row L holds the final
    norm in `in_ln`. Per-rank fields: q_b_s (this rank's heads), o_proj_s / sw2_s / w2_s (scales over the local K),
    sw13_s / w13_s (this rank's gate | up columns)."""
    fields = [
        ("in_ln", cfg.H),
        ("post_ln", cfg.H),
        ("q_a_ln", cfg.QL),
        ("kv_a_ln", cfg.KL),
        ("qkv_a_s", cfg.QKVP),
        ("q_b_s", cfg.HR * cfg.DQ),
        ("o_proj_s", cfg.H),
        ("idx_wq_s", cfg.IH * cfg.IDIM),
        ("idx_kn_w", cfg.IDIM),
        ("idx_kn_b", cfg.IDIM),
        ("gate_bias", cfg.EPAD),
        ("sw13_s", 2 * cfg.ISR),
        ("sw2_s", cfg.H),
        ("w13_s", 2 * cfg.IDR),
        ("w2_s", cfg.H),
        ("k_scale", 128),
    ]
    off, lay = 0, {}
    for name, n in fields:
        lay[name] = (off, n)
        off += rup(n)
    return lay, off


# dense stream kinds, in a layer's consumption order
DKINDS = (
    "qkva",
    "qb",
    "wuk",
    "iwq",
    "iwk",
    "wuv",
    "wo",
    "gate",
    "sh13",
    "sh2",
    "dw13",
    "dw2",
)
WHOLE = ("wuk", "wuv")  # one tile = the whole per-layer tensor


def layers_of(cfg, kind):
    all_l = list(range(cfg.L))
    if kind in ("iwq", "iwk"):
        return [l for l in all_l if cfg.full_indexer(l)]
    if kind in ("gate", "sh13", "sh2"):
        return [l for l in all_l if cfg.is_moe(l)]
    if kind in ("dw13", "dw2"):
        return [l for l in all_l if not cfg.is_moe(l)]
    return all_l


def dense_shapes(cfg):
    """kind -> (per-layer shape, dtype) of the stacked HBM operand [n_layers_of_kind, *shape]."""
    H = cfg.H
    return {
        "qkva": ((H, cfg.QKVP), FP8),  # replicated, N padded
        "qb": ((cfg.QL, cfg.HR * cfg.DQ), FP8),  # this rank's HR heads (qrep 0)
        "wuk": ((cfg.HR, cfg.DN, cfg.KL), BF16),  # W_UK_T of this rank's heads
        "iwq": ((cfg.QL, cfg.IH * cfg.IDIM), FP8),  # replicated
        "iwk": ((H, cfg.IKW), BF16),  # replicated, N padded
        "wuv": ((cfg.HR, cfg.KL, cfg.DV), BF16),  # this rank's heads
        "wo": ((cfg.KR, H), FP8),  # local K rows
        "gate": ((H, cfg.EPAD), BF16),  # replicated, E padded
        "sh13": ((H, 2 * cfg.ISR), FP8),  # TP slice [gate | up]
        "sh2": ((cfg.ISR, H), FP8),
        "dw13": ((H, 2 * cfg.IDR), FP8),
        "dw2": ((cfg.IDR, H), FP8),
    }


def expert_shapes(cfg):
    """per-layer expert tensors (main's served TP-sharded layout, EP off)."""
    return {
        "ew13": ((cfg.E, cfg.H, 2 * cfg.IR), FP8),
        "ew2": ((cfg.E, cfg.IR, cfg.H), FP8),
        "es13": ((cfg.E, 1, 1, 2 * cfg.IR), F32),
        "es2": ((cfg.E, 1, 1, cfg.H), F32),
    }


def cache_kinds(cfg, nb, nbi):
    all_l = list(range(cfg.L))
    full = [l for l in all_l if cfg.full_indexer(l)]
    pg = cfg.page
    return {
        "nope": (all_l, (nb, pg, cfg.KL // 4), U32),
        "rope": (all_l, (nb, pg // 4, 128), U32),
        "idx": (full, (nbi, cfg.IREC // 4, 4, pg), U8),
    }


def tile_rows(K, N, dtype, bank):
    """Largest 128-multiple divisor of K whose (tk, N) tile fits one bank (glm5_mk_dcp Tiles.tk)."""
    best = None
    for tk in range(128, K + 1, 128):
        if K % tk == 0 and pool_alias.storage_size((tk, N), jnp.dtype(dtype)) <= bank:
            best = tk
    assert best is not None, ("no 128-row tile fits a bank", K, N, dtype, bank)
    return best


def tile_plan(cfg, bank):
    """kind -> (tile shape, dtype, n tiles, tk)."""
    out = {}
    for k, (shp, dt) in dense_shapes(cfg).items():
        if k in WHOLE:
            out[k] = (shp, dt, 1, None)
        else:
            K, N = shp
            tk = tile_rows(K, N, dt, bank)
            out[k] = ((tk, N), dt, K // tk, tk)
    return out


def bank_bytes(cfg, bank):
    """Bank size: the configured bank, raised to the largest tensor consumed as one tile (W_UK / W_UV / router gate /
    shared expert) if needed (FULL: no change)."""
    need = max(
        pool_alias.storage_size(shp, jnp.dtype(dt))
        for k, (shp, dt) in dense_shapes(cfg).items()
        if k in WHOLE + ("gate", "sh13", "sh2")
    )
    return max(bank, rup(need, 4096))


def layer_tiles(cfg, l, plan):
    """Static tile order of layer l: [(kind, tile index)]."""
    seq = (
        [("qkva", t) for t in range(plan["qkva"][2])]
        + [("qb", t) for t in range(plan["qb"][2])]
        + [("wuk", 0)]
    )
    if cfg.full_indexer(l):
        seq += [("iwq", t) for t in range(plan["iwq"][2])] + [
            ("iwk", t) for t in range(plan["iwk"][2])
        ]
    seq += [("wuv", 0)] + [("wo", t) for t in range(plan["wo"][2])]
    if cfg.is_moe(l):
        seq += (
            [("gate", t) for t in range(plan["gate"][2])]
            + [("sh13", t) for t in range(plan["sh13"][2])]
            + [("sh2", t) for t in range(plan["sh2"][2])]
        )
    else:
        seq += [("dw13", t) for t in range(plan["dw13"][2])] + [
            ("dw2", t) for t in range(plan["dw2"][2])
        ]
    return seq


def stream_tables(cfg, plan):
    """Host: ltab i32[L, 4] = (first stream index g0, fresh, moe, kind position of the layer's moe weights) and
    dtab i32[T] = kind << 20 | layer position in the kind's stack << 8 | tile (csq_proto.tables, generalized to
    per-layer weights)."""
    ltab = np.zeros((cfg.L, 4), np.int32)
    dt = []
    for l in range(cfg.L):
        ltab[l] = (
            len(dt),
            int(cfg.full_indexer(l)),
            int(cfg.is_moe(l)),
            layers_of(cfg, "gate").index(l) if cfg.is_moe(l) else 0,
        )
        for k, t in layer_tiles(cfg, l, plan):
            pos = layers_of(cfg, k).index(l)
            assert t < 256 and pos < 4096
            dt.append(DKINDS.index(k) << 20 | pos << 8 | t)
    return ltab, np.asarray(dt, np.int32)
