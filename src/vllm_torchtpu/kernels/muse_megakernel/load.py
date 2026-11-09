# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Muse Spark checkpoint loading, NVFP4 slicing, and TP8 weight layouts."""

from __future__ import annotations

import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .config import (
    TP_AXIS_NAME,
    MuseMegakernelConfig,
    dequantize_nvfp4,
    residual_coefficients,
)

_SAFETENSORS_DTYPES = {
    "BF16": (np.uint16, 2),
    "F32": (np.float32, 4),
    "U8": (np.uint8, 1),
    "F8_E4M3": (np.uint8, 1),
}

KERNEL_WEIGHT_NAMES = (
    "qwkv",
    "ow_group8",
    "pool_local",
    "unpool_group8",
    "router_ordered",
    "router_bias_ordered",
    "scalars",
    "expert_norms",
    "gu_pack",
    "dw_pack",
    "gu_dense",
    "dw_dense",
    "gu_global",
    "dw_global",
    "lmw",
    "fnorm",
    "emb",
)


def lm_pad(c: MuseMegakernelConfig, tp: int) -> int:
    # The decode head streams 640 wide double buffered tiles, so the shard
    # width must be a multiple of 1280 or trailing vocab columns are skipped.
    shard = c.vocab // tp
    return (-shard) % 1280


def lm_width(c: MuseMegakernelConfig, tp: int) -> int:
    return c.vocab // tp + lm_pad(c, tp)


def tile_hw(
    k: int, n: int, max_k: int | None = None, max_n: int | None = None
) -> tuple[int, int]:
    """(bk, bn) tile for a [k, n] matrix: largest 128 multiples within caps."""
    max_k = max_k or 1024
    max_n = max_n or 1280

    def pick(d: int, cap: int) -> int:
        divisors = [b for b in range(128, min(d, cap) + 1, 128) if d % b == 0]
        if not divisors:
            divisors = [b for b in range(min(d, cap), 0, -1) if d % b == 0]
        return max(divisors)

    return pick(k, max_k), pick(n, max_n)


def router_order_indices(experts: int) -> np.ndarray:
    columns = np.arange(experts, dtype=np.int32)
    if experts in (128, 256):
        return (127 - columns % 128) * (experts // 128) + columns // 128
    return columns


def permute_sublane_axis_np(x: np.ndarray, axis: int) -> np.ndarray:
    """Permute an axis of length K (multiple of 16) from (K // 16, 16) to (16, K // 16)
    for
    zero-shuffle TPU VREG scale broadcast.
    """
    sh = list(x.shape)
    ax = axis % len(sh)
    k = sh[ax]
    new_sh = sh[:ax] + [k // 16, 16] + sh[ax + 1 :]
    return np.swapaxes(x.reshape(new_sh), ax, ax + 1).reshape(sh)


def repack_nvfp4_k_sublane_jax(v_u8: jax.Array) -> jax.Array:
    """Repack packed E2M1 uint8 [..., K // 2, N] along K from (K // 16, 16) to
    sublane-aligned (16, K // 16).
    """
    *lead, k_half, n = v_u8.shape
    k = k_half * 2
    lo = v_u8 & jnp.uint8(0x0F)
    hi = (v_u8 >> jnp.uint8(4)) & jnp.uint8(0x0F)
    nibs = jnp.stack([lo, hi], axis=-2).reshape(*lead, k // 16, 16, n)
    nibs_t = jnp.swapaxes(nibs, -3, -2).reshape(*lead, k_half, 2, n)
    return (nibs_t[..., 0, :] | (nibs_t[..., 1, :] << jnp.uint8(4))).astype(jnp.uint8)


def permute_gu_cols_sublane_jax(x: jax.Array) -> jax.Array:
    """Permute gate and up output columns [..., K_rows, 2 * width] within each half from
    (width // 16, 16) to (16, width // 16).
    """
    *lead, k_rows, two_w = x.shape
    w = two_w // 2
    x_r = x.reshape(*lead, k_rows, 2, w // 16, 16)
    return jnp.swapaxes(x_r, -2, -1).reshape(*lead, k_rows, two_w)


def permute_k_sublane_jax(x: jax.Array) -> jax.Array:
    """Permute K rows [..., K, N] from (K // 16, 16) to (16, K // 16)."""
    *lead, k, n = x.shape
    x_r = x.reshape(*lead, k // 16, 16, n)
    return jnp.swapaxes(x_r, -3, -2).reshape(*lead, k, n)


class Checkpoint:
    """Raw byte safetensors reader preserving BF16 and F8_E4M3 bit patterns."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        idx_path = path / "model.safetensors.index.json"
        if idx_path.exists():
            index = json.loads(idx_path.read_text())
            self.weight_map = index["weight_map"]
        else:
            shard = "model.safetensors"
            with open(path / shard, "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(size))
            self.weight_map = {k: shard for k in header if k != "__metadata__"}
        self.root = path
        self._headers: dict[str, tuple[dict[str, Any], int]] = {}

    def _header(self, shard: str) -> tuple[dict[str, Any], int]:
        if shard not in self._headers:
            with open(self.root / shard, "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(size))
            self._headers[shard] = (header, 8 + size)
        return self._headers[shard]

    def info(self, name: str) -> tuple[str, list[int], list[int]]:
        header, _ = self._header(self.weight_map[name])
        entry = header[name]
        return entry["dtype"], entry["shape"], entry["data_offsets"]

    def get(self, name: str) -> np.ndarray:
        dtype, shape, (start, end) = self.info(name)
        np_dtype, itemsize = _SAFETENSORS_DTYPES[dtype]
        _, base = self._header(self.weight_map[name])
        count = (end - start) // itemsize
        raw = np.fromfile(
            self.root / self.weight_map[name],
            dtype=np_dtype,
            count=count,
            offset=base + start,
        ).reshape(shape)
        if dtype == "BF16":
            return raw.view(ml_dtypes.bfloat16)
        if dtype == "F8_E4M3":
            return raw.view(ml_dtypes.float8_e4m3fn)
        return raw

    def memmap(self, name: str) -> np.ndarray:
        dtype, shape, (start, _end) = self.info(name)
        np_dtype, _ = _SAFETENSORS_DTYPES[dtype]
        _, base = self._header(self.weight_map[name])
        raw = np.memmap(
            self.root / self.weight_map[name],
            dtype=np_dtype,
            mode="r",
            offset=base + start,
            shape=tuple(shape),
        )
        if dtype == "BF16":
            return raw.view(ml_dtypes.bfloat16)
        if dtype == "F8_E4M3":
            return raw.view(ml_dtypes.float8_e4m3fn)
        return raw


def prepare_tp8_weights(
    mesh: jax.sharding.Mesh, weights: dict[str, Any]
) -> dict[str, Any]:
    """Prepare output column projections and Mosaic router ordering for TP8 decode."""
    if mesh.size != 8:
        raise ValueError("TP8 preparation requires eight devices")
    if "ow_group8" in weights:
        return weights
    weights = dict(weights)
    hidden = weights["scalars"].shape[-1]
    experts = weights["router_bias"].shape[-1]
    shard = NamedSharding(mesh, P(TP_AXIS_NAME))

    @jax.jit(out_shardings=shard)
    def columns(value):
        _, layers, width, _ = value.shape
        return (
            value.reshape(8, layers, width, 8, hidden // 8)
            .transpose(3, 1, 0, 2, 4)
            .reshape(8, layers, 8 * width, hidden // 8)
        )

    @jax.jit(out_shardings=(shard, shard))
    def router_order(value, bias):
        _, layers, width, _ = value.shape
        order = jnp.arange(experts, dtype=jnp.int32)
        if experts in (128, 256):
            order = (127 - order % 128) * (experts // 128) + order // 128
        full = value.transpose(1, 2, 0, 3).reshape(layers, width, experts)
        return (
            full[..., order]
            .reshape(layers, width, 8, experts // 8)
            .transpose(2, 0, 3, 1),
            bias[..., order],
        )

    router, bias = router_order(weights["router"], weights["router_bias"])
    return jax.block_until_ready(
        {
            **weights,
            "ow_group8": columns(weights["ow"]),
            "pool_local": weights["pool"],
            "unpool_group8": columns(weights["unpool"]),
            "router_ordered": router,
            "router_bias_ordered": bias,
        }
    )


def load_sharded_weights(
    mesh: jax.sharding.Mesh,
    ckpt_dir: str | Path,
    c: MuseMegakernelConfig | None = None,
    *,
    prepare_decode: bool = True,
) -> dict[str, jax.Array]:
    """Stream checkpoint weights to 8 TPU devices with in-place donated sharding."""
    if c is None:
        c = MuseMegakernelConfig.from_checkpoint(ckpt_dir)
    tp = mesh.size
    if tp != 8:
        raise ValueError(f"Expected 8 TPU devices in mesh, got {tp}")
    ckpt = Checkpoint(ckpt_dir)
    sharding = NamedSharding(mesh, P(TP_AXIS_NAME))

    @jax.jit(out_shardings=sharding)
    def shard_gu(x):
        e, two_i, k = x.shape
        half = two_i // 2
        w = half // tp
        g = x[:, :half, :].reshape(e, tp, w, k)
        u = x[:, half:, :].reshape(e, tp, w, k)
        return jnp.concatenate([g, u], axis=2).transpose(1, 0, 3, 2)

    @jax.jit(out_shardings=sharding)
    def shard_dw(x):
        e, k, ic = x.shape
        w = ic // tp
        return x.reshape(e, k, tp, w).transpose(2, 0, 3, 1)

    @jax.jit(donate_argnums=(0,), out_shardings=sharding)
    def put_gu_pack(buf, idx, gu, gus):
        gu_v = repack_nvfp4_k_sublane_jax(permute_gu_cols_sublane_jax(shard_gu(gu)))
        gu_s = permute_gu_cols_sublane_jax(shard_gu(gus))
        v = jnp.concatenate([gu_v, gu_s], axis=2)
        return jax.lax.dynamic_update_index_in_dim(buf, v, idx, axis=1)

    @jax.jit(donate_argnums=(0,), out_shardings=sharding)
    def put_dw_pack(buf, idx, dw, dws):
        dw_v = repack_nvfp4_k_sublane_jax(shard_dw(dw))
        dw_s = shard_dw(dws)
        v = jnp.concatenate([dw_v, dw_s], axis=2)
        return jax.lax.dynamic_update_index_in_dim(buf, v, idx, axis=1)

    @jax.jit(donate_argnums=(0,), out_shardings=sharding)
    def put_gu_dense(buf, idx, gu):
        v = permute_k_sublane_jax(permute_gu_cols_sublane_jax(shard_gu(gu)))
        return jax.lax.dynamic_update_index_in_dim(buf, v, idx, axis=1)

    @jax.jit(donate_argnums=(0,), out_shardings=sharding)
    def put_dw_dense(buf, idx, dw):
        v = permute_k_sublane_jax(shard_dw(dw))
        return jax.lax.dynamic_update_index_in_dim(buf, v, idx, axis=1)

    n_packed = max(1, c.layers - len(c.bf16_expert_layers))
    n_dense = max(1, len(c.bf16_expert_layers))
    width = c.moe_intermediate // tp
    gu_pack = jax.jit(
        lambda: jnp.zeros(
            (tp, n_packed, c.experts, c.moe_hidden * 9 // 16, 2 * width), jnp.uint8
        ),
        out_shardings=sharding,
    )()
    dw_pack = jax.jit(
        lambda: jnp.zeros(
            (tp, n_packed, c.experts, width * 9 // 16, c.moe_hidden), jnp.uint8
        ),
        out_shardings=sharding,
    )()
    gu_dense = jax.jit(
        lambda: jnp.zeros(
            (tp, n_dense, c.experts, c.moe_hidden, 2 * width), jnp.bfloat16
        ),
        out_shardings=sharding,
    )()
    dw_dense = jax.jit(
        lambda: jnp.zeros((tp, n_dense, c.experts, width, c.moe_hidden), jnp.bfloat16),
        out_shardings=sharding,
    )()

    gu_glob_l, dw_glob_l = [], []
    qwkv_l, ow_l, pool_l, unpool_l, router_l, rbias_l, scalars_l, xnorms_l = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    qh = c.heads // tp
    kvh = c.kv_heads // tp
    eshard = c.experts // tp
    mh_r = c.moe_hidden // tp
    pi, bi = 0, 0

    def read_layer_host(i: int):
        p = f"model.language_model.layers.{i}."
        a, m = p + "self_attn.", p + "mlp."
        q_w, g_w, k_w, v_w, o_w = [
            ckpt.get(a + k + ".weight")
            for k in ("q_proj", "gate_proj", "k_proj", "v_proj", "o_proj")
        ]
        norm_in = np.asarray(ckpt.get(p + "input_layernorm.weight"), np.float32) + 1.0
        alpha_a, beta_a = residual_coefficients(
            ckpt.get(p + "post_attention_residual_gate.gate"), c.gate_temperature
        )
        norm_pre = (
            np.asarray(ckpt.get(p + "pre_feedforward_layernorm.weight"), np.float32)
            + 1.0
        )
        norm_post = (
            np.asarray(ckpt.get(p + "post_feedforward_layernorm.weight"), np.float32)
            + 1.0
        )
        alpha_f, beta_f = residual_coefficients(
            ckpt.get(p + "post_feedforward_residual_gate.gate"), c.gate_temperature
        )
        row = np.zeros((8, c.dim), np.float32)
        row[0], row[1], row[2] = norm_in, norm_pre, norm_post
        row[3], row[4], row[5], row[6] = alpha_a, beta_a, alpha_f, beta_f
        rt = np.asarray(ckpt.get(m + "gate.weight"), np.float32)
        rb = np.asarray(ckpt.get(m + "gate.e_score_correction_bias"), np.float32)
        pl_w = permute_sublane_axis_np(ckpt.get(m + "pre_expert_proj.weight"), 0)
        up_w = ckpt.get(m + "post_expert_proj.weight")
        xn = np.stack(
            (
                permute_sublane_axis_np(
                    np.asarray(ckpt.get(m + "pre_expert_norm.weight"), np.float32)
                    + 1.0,
                    0,
                ),
                np.asarray(ckpt.get(m + "experts.post_expert_norm.weight"), np.float32)
                + 1.0,
            )
        )
        qwkv_r, ow_r = [], []
        for r in range(tp):
            qs = r * qh * c.head_dim
            ks = r * kvh * c.head_dim
            qwkv_r.append(
                np.concatenate(
                    (
                        q_w[qs : qs + qh * c.head_dim].T,
                        g_w[qs : qs + qh * c.head_dim].T,
                        k_w[ks : ks + kvh * c.head_dim].T,
                        v_w[ks : ks + kvh * c.head_dim].T,
                    ),
                    axis=1,
                )
            )
            ow_r.append(o_w[:, qs : qs + qh * c.head_dim].T)
        if i in c.bf16_expert_layers:
            ex: Any = (
                ckpt.get(m + "experts.gate_up_proj"),
                ckpt.get(m + "experts.down_proj"),
            )
        else:
            ex = (
                ckpt.get(m + "experts.gate_up_proj"),
                ckpt.get(m + "experts.gate_up_proj_weight_scale").view(np.uint8),
                ckpt.get(m + "experts.down_proj"),
                ckpt.get(m + "experts.down_proj_weight_scale").view(np.uint8),
                np.asarray(
                    ckpt.get(m + "experts.gate_up_proj_weight_scale_2"), np.float32
                ),
                np.asarray(
                    ckpt.get(m + "experts.down_proj_weight_scale_2"), np.float32
                ),
            )
        return row, rt, rb, pl_w, up_w, xn, np.stack(qwkv_r), np.stack(ow_r), ex

    with ThreadPoolExecutor(2) as pool_ex:
        futs = {0: pool_ex.submit(read_layer_host, 0)}
        if c.layers > 1:
            futs[1] = pool_ex.submit(read_layer_host, 1)
        for i in range(c.layers):
            if i + 2 < c.layers:
                futs[i + 2] = pool_ex.submit(read_layer_host, i + 2)
            row, rt, rb, pl_w, up_w, xn, qwkv_s, ow_s, ex = futs.pop(i).result()
            scalars_l.append(row)
            router_l.append(rt)
            rbias_l.append(rb)
            pool_l.append(pl_w)
            unpool_l.append(up_w)
            xnorms_l.append(xn)
            qwkv_l.append(qwkv_s)
            ow_l.append(ow_s)
            if i in c.bf16_expert_layers:
                gu_dense = put_gu_dense(
                    gu_dense, jnp.int32(bi), ex[0]
                ).block_until_ready()
                dw_dense = put_dw_dense(
                    dw_dense, jnp.int32(bi), ex[1]
                ).block_until_ready()
                bi += 1
            else:
                gu_pack = put_gu_pack(
                    gu_pack, jnp.int32(pi), ex[0], ex[1]
                ).block_until_ready()
                dw_pack = put_dw_pack(
                    dw_pack, jnp.int32(pi), ex[2], ex[3]
                ).block_until_ready()
                gu_glob_l.append(ex[4])
                dw_glob_l.append(ex[5])
                pi += 1
            del ex

    if not gu_glob_l:
        gu_glob_l.append(np.zeros((c.experts, 2), np.float32))
        dw_glob_l.append(np.zeros((c.experts,), np.float32))

    w = {
        "qwkv": jax.device_put(np.stack(qwkv_l, axis=1), sharding),
        "ow": jax.device_put(np.stack(ow_l, axis=1), sharding),
        "pool": jax.device_put(
            np.stack(
                [
                    np.stack([p[r * mh_r : (r + 1) * mh_r].T for r in range(tp)])
                    for p in pool_l
                ],
                axis=1,
            ),
            sharding,
        ),
        "unpool": jax.device_put(
            np.stack(
                [
                    np.stack([u[:, r * mh_r : (r + 1) * mh_r].T for r in range(tp)])
                    for u in unpool_l
                ],
                axis=1,
            ),
            sharding,
        ),
        "router": jax.device_put(
            np.stack(
                [
                    np.stack([rt[r * eshard : (r + 1) * eshard].T for r in range(tp)])
                    for rt in router_l
                ],
                axis=1,
            ),
            sharding,
        ),
        "router_bias": jax.device_put(
            np.broadcast_to(np.stack(rbias_l)[None], (tp, c.layers, c.experts)),
            sharding,
        ),
        "scalars": jax.device_put(
            np.broadcast_to(np.stack(scalars_l)[None], (tp, c.layers, 8, c.dim)),
            sharding,
        ),
        "expert_norms": jax.device_put(
            np.broadcast_to(np.stack(xnorms_l)[None], (tp, c.layers, 2, c.moe_hidden)),
            sharding,
        ),
        "gu_pack": gu_pack,
        "dw_pack": dw_pack,
        "gu_dense": gu_dense,
        "dw_dense": dw_dense,
        "gu_global": jax.device_put(
            np.broadcast_to(
                np.stack(gu_glob_l)[None], (tp, len(gu_glob_l), c.experts, 2)
            ),
            sharding,
        ),
        "dw_global": jax.device_put(
            np.broadcast_to(np.stack(dw_glob_l)[None], (tp, len(dw_glob_l), c.experts)),
            sharding,
        ),
    }
    del qwkv_l, ow_l, pool_l, unpool_l, router_l
    vshard = c.vocab // tp
    emb = ckpt.get("model.language_model.embed_tokens.weight")
    w["emb"] = jax.device_put(
        np.stack([emb[r * vshard : (r + 1) * vshard] for r in range(tp)]), sharding
    )
    del emb
    lm = ckpt.get("lm_head.weight")
    lmw_np = np.zeros((tp, c.dim, lm_width(c, tp)), ml_dtypes.bfloat16)
    for r in range(tp):
        lmw_np[r, :, :vshard] = lm[r * vshard : (r + 1) * vshard].T
    del lm
    w["lmw"] = jax.device_put(lmw_np, sharding)
    del lmw_np
    fnorm_np = np.asarray(ckpt.get("model.language_model.norm.weight"), np.float32)
    w["fnorm"] = jax.device_put(
        np.broadcast_to(fnorm_np[None, None], (tp, 1, c.dim)), sharding
    )
    if prepare_decode:
        w = prepare_tp8_weights(mesh, w)
    return w


def canonical_test_weights(
    mesh: jax.sharding.Mesh, c: MuseMegakernelConfig, seed: int = 0
) -> dict[str, jax.Array]:
    """Generate small synthetic TP8 weights on device for unit tests."""
    tp = mesh.size
    sharding = NamedSharding(mesh, P(TP_AXIS_NAME))
    rng = np.random.default_rng(seed)
    qh = c.heads // tp
    kvh = max(c.kv_heads // tp, 1)
    eshard = c.experts // tp
    mh_r = c.moe_hidden // tp
    width = c.moe_intermediate // tp
    n_packed = max(1, c.layers - len(c.bf16_expert_layers))
    n_dense = max(1, len(c.bf16_expert_layers))
    vshard = c.vocab // tp

    def b16(*shape, scale=0.02):
        if int(np.prod(shape)) > (1 << 22):
            base = (rng.standard_normal(shape[-2:], dtype=np.float32) * scale).astype(
                ml_dtypes.bfloat16
            )
            return jax.jit(
                lambda x: jnp.broadcast_to(x, shape), out_shardings=sharding
            )(base)
        return jax.device_put(
            (rng.standard_normal(shape, dtype=np.float32) * scale).astype(
                ml_dtypes.bfloat16
            ),
            sharding,
        )

    def nvfp4_pack(*shape):
        # Rows [0, K/2) are packed e2m1 nibbles; rows [K/2, 9K/16) are e4m3 block
        # scales. Random bytes there can be 0x7F/0xFF (e4m3 NaN), so pin them to 1.0.
        n_val = shape[-2] * 8 // 9
        if int(np.prod(shape)) > (1 << 22):
            base = rng.integers(0, 256, size=shape[-2:], dtype=np.uint8)
            base[n_val:, :] = 0x38
            return jax.jit(
                lambda x: jnp.broadcast_to(x, shape), out_shardings=sharding
            )(base)
        packed = rng.integers(0, 256, size=shape, dtype=np.uint8)
        packed[..., n_val:, :] = 0x38
        return jax.device_put(packed, sharding)

    scalars = np.zeros((tp, c.layers, 8, c.dim), np.float32)
    scalars[:, :, :7, :] = 1.0
    w = {
        "qwkv": b16(tp, c.layers, c.dim, 2 * qh * c.head_dim + 2 * kvh * c.head_dim),
        "ow": b16(tp, c.layers, qh * c.head_dim, c.dim),
        "pool": b16(tp, c.layers, c.dim, mh_r),
        "unpool": b16(tp, c.layers, mh_r, c.dim),
        "router": jax.device_put(
            rng.standard_normal((tp, c.layers, c.dim, eshard), dtype=np.float32) * 0.02,
            sharding,
        ),
        "router_bias": jax.device_put(
            np.zeros((tp, c.layers, c.experts), np.float32), sharding
        ),
        "scalars": jax.device_put(scalars, sharding),
        "expert_norms": jax.device_put(
            np.ones((tp, c.layers, 2, c.moe_hidden), np.float32), sharding
        ),
        "gu_pack": nvfp4_pack(
            tp, n_packed, c.experts, c.moe_hidden * 9 // 16, 2 * width
        ),
        "dw_pack": nvfp4_pack(tp, n_packed, c.experts, width * 9 // 16, c.moe_hidden),
        "gu_dense": b16(tp, n_dense, c.experts, c.moe_hidden, 2 * width),
        "dw_dense": b16(tp, n_dense, c.experts, width, c.moe_hidden),
        "gu_global": jax.device_put(
            np.full((tp, n_packed, c.experts, 2), 0.01, np.float32), sharding
        ),
        "dw_global": jax.device_put(
            np.full((tp, n_packed, c.experts), 0.01, np.float32), sharding
        ),
        "emb": b16(tp, vshard, c.dim),
        "lmw": b16(tp, c.dim, lm_width(c, tp)),
        "fnorm": jax.device_put(np.ones((tp, 1, c.dim), np.float32), sharding),
    }
    return prepare_tp8_weights(mesh, w)


__all__ = [
    "Checkpoint",
    "KERNEL_WEIGHT_NAMES",
    "canonical_test_weights",
    "dequantize_nvfp4",
    "lm_pad",
    "lm_width",
    "load_sharded_weights",
    "prepare_tp8_weights",
    "router_order_indices",
]
