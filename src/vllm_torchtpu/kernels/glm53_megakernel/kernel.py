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
"""GLM-5.3-Flash single-dispatch Pallas decode megakernel.

One `pl.pallas_call` per decode step covers: the vocab-parallel embedding and
its TP all-reduce, every decoder layer (mHC, KDA or absorbed NoPE MLA, dense
MLP or routed MoE plus the TP shared expert, all collectives in kernel), the
final norm, the vocab-sharded lm head, the global top-k logprobs and the
greedy argmax. `make_spec_decode_step` adds a q=4 verify window, exact greedy
acceptance, a lossless KDA state rollback and a k=3 MTP draft chain.

The kernel reproduces the rounding points of a plain JAX decode step (bf16
activations, f32 accumulation, per-row fp8 activations for the fp8 linears),
so on the CPU interpreter it matches a JAX reference to f32 reduction-order
error.

Per-request decode state lives in arrays indexed by a slot:

- KDA conv state `[Lk, slots, K - 1, 3 * H * D]` bf16 and recurrent state
  `[Lk, slots, H, D, D]` f32 (per rank, H is the local head count);
- the MLA latent cache, sequence sharded across ranks: page `lp` (of
  `cfg.page_size` tokens) lives on rank `lp % world`, and `KV_GROUP`
  consecutive local pages form one superpage. Superpage rows hold the two fp8
  codes of the 2-component latent quantization packed four rows per int32
  word (`kv_words`), plus their two f32 scales. `kpt [slots, superpages]`
  maps a slot's superpage to a physical superpage of the pool.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .config import Glm53MegakernelConfig

F8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32
FP8_MAX = float(jnp.finfo(F8).max)
HIGHEST = jax.lax.Precision.HIGHEST
# Activation row tile (the native bf16 MXU tile); batch rows >= B are zero
# padding.
ROWS = 16
TOPK_OUT = 64
# Per-lane top-k candidate pool. A row whose top 64 put more than 8 entries in
# one lane (about 4e-7 per row for uniform ids) falls back to the exact merge.
LANE_TOPK = 8
NEG = -1e30
TILE_BYTES = 2 << 20
COLLECTIVE_ID = 31
# Scalar prefetch operands: tokens, positions, slots, valid, ltab, kpt, rank,
# peer device table. The rank arrives as data because torch_tpu's jax.export
# bridge rejects partition-id (lax.axis_index) when it recompiles the module.
N_PREFETCH = 8

_M_IN_LN, _M_POST_LN, _M_HCA_BASE, _M_HCA_SCALE, _M_HCF_BASE, _M_HCF_SCALE = (
    0,
    1,
    2,
    3,
    4,
    5,
)
_M_QA_NORM, _M_KVA_NORM, _M_O_NORM, _M_RBIAS, _M_HEADS = 6, 7, 8, 9, 10


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _nt(a: jax.Array, b: jax.Array, prec: Any = None) -> jax.Array:
    """a [M, K] x b [N, K] -> [M, N] float32."""
    return jax.lax.dot_general(
        a, b, (((1,), (1,)), ((), ())), preferred_element_type=F32, precision=prec
    )


def _nn(a: jax.Array, b: jax.Array, prec: Any = None) -> jax.Array:
    """a [M, K] x b [K, N] -> [M, N] float32."""
    return jax.lax.dot_general(
        a, b, (((1,), (0,)), ((), ())), preferred_element_type=F32, precision=prec
    )


def _split3(a: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """f32 -> three bf16 parts whose f32 sum is a.

    This is the bf16x3 split Precision.HIGHEST uses on the MXU.
    """
    hi = a.astype(BF16)
    r1 = a - hi.astype(F32)
    mid = r1.astype(BF16)
    lo = (r1 - mid.astype(F32)).astype(BF16)
    return hi, mid, lo


def _lane_chunk(n: int) -> int:
    for c in (2048, 1024, 512, 256, 128):
        if n % c == 0:
            return c
    raise ValueError(f"width {n} is not a multiple of 128")


# Weight tiles in flight per streamed linear: 2 MiB tiles need at least 3 in
# flight to reach the DMA peak.
STREAM_DEPTH = 4


def _stream_tiles(n: int, k: int, itemsize: int) -> tuple[int, int, int]:
    """(tn, n_full, tail): a 128-multiple row tile of about TILE_BYTES.

    The tail rows (fewer than tn, a multiple of 128) are peeled.
    """
    if n % 128:
        raise ValueError(f"streamed width {n} is not a multiple of 128")
    tn = max(128, min(n, TILE_BYTES // (k * itemsize)) // 128 * 128)
    return tn, n // tn, n % tn


def _quant_rows(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-row fp8 quantization with a float32 scale."""
    amax = jnp.max(jnp.abs(a), axis=-1, keepdims=True)
    scale = amax.astype(F32) / jnp.float32(FP8_MAX)
    scale = jnp.where(scale == 0, 1.0, scale)
    aq = jnp.clip(a.astype(F32) / scale, -FP8_MAX, FP8_MAX).astype(F8)
    return aq, scale


def _bmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """bf16 x bf16 product rounded to bf16, explicitly.

    XLA rounds each bf16 elementwise op to bf16. The Pallas TPU CPU
    interpreter keeps the f32 product, so the rounding is spelled out to give
    identical values in interpret mode and on hardware.
    """
    return (a.astype(F32) * b.astype(F32)).astype(BF16)


def _bfadd(a: jax.Array, b: jax.Array) -> jax.Array:
    """bf16 + bf16 rounded to bf16 explicitly (see _bmul)."""
    return (a.astype(F32) + b.astype(F32)).astype(BF16)


def _silu_bf16(x: jax.Array) -> jax.Array:
    """jax.nn.silu on bf16 as XLA lowers it.

    x * (1 / (1 + exp(-x))) with every op rounded to bf16.
    """
    e = jnp.exp(-x.astype(F32)).astype(BF16)
    den = (1.0 + e.astype(F32)).astype(BF16)
    sig = (1.0 / den.astype(F32)).astype(BF16)
    return _bmul(x, sig)


def _rms(x: jax.Array, w: jax.Array, eps: float) -> jax.Array:
    """Weighted RMSNorm with a float32 copy of a bf16 weight row.

    The normalized value is rounded to bf16 before the weight multiply, as
    vLLM does.
    """
    xf = x.astype(F32)
    xf = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return _bmul(xf.astype(BF16), w.astype(BF16)).astype(x.dtype)


def _clamped_swiglu(gate: jax.Array, up: jax.Array, lim: float) -> jax.Array:
    g_c = jnp.minimum(gate, lim)
    u_c = jnp.clip(up, -lim, lim)
    return _bmul(jax.nn.silu(g_c.astype(F32)).astype(gate.dtype), u_c)


# Routed expert weight tiles in flight (2 MiB gate_up and 1 MiB down slices at
# the released model size).
EXPERT_RING_DEPTH = 6
# Local pages per kernel superpage.
KV_GROUP = 8
# Superpages in flight in the MLA latent DMA ring.
KV_DEPTH = 4


def lane_topk(
    load_block: Any, nblk: int, sr: int, live_rows: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Top TOPK_OUT per row (value desc, ties by smaller id).

    Built from per-lane top LANE_TOPK candidate pools. load_block(j) returns
    (values [sr, 128] f32, global ids [sr, 128] f32) for lane block j. Each
    lane keeps its best LANE_TOPK (elementwise compares only), then TOPK_OUT
    extraction rounds run on the pool. Returns (values [sr, 128], ids
    [sr, 128], unsafe [sr, 128] bool). The result equals the exact top-k
    unless some lane was full of candidates ranking at or ahead of the last
    selection; `unsafe` is then set somewhere in a live row and the caller
    must use an exact method.
    """
    big = jnp.float32(3e38)

    def insert(j: Any, carry: Any) -> Any:
        vs, gs = carry
        x, g = load_block(j)
        # Strict: equal values arrive in increasing id order and rank later.
        gt = [x > v for v in vs]
        nv = [jnp.where(gt[0], x, vs[0])]
        ng = [jnp.where(gt[0], g, gs[0])]
        for k in range(1, LANE_TOPK):
            nv.append(jnp.where(gt[k - 1], vs[k - 1], jnp.where(gt[k], x, vs[k])))
            ng.append(jnp.where(gt[k - 1], gs[k - 1], jnp.where(gt[k], g, gs[k])))
        return tuple(nv), tuple(ng)

    pv, pg = jax.lax.fori_loop(
        0,
        nblk,
        insert,
        (
            tuple(jnp.full((sr, 128), -jnp.inf, F32) for _ in range(LANE_TOPK)),
            tuple(jnp.full((sr, 128), big, F32) for _ in range(LANE_TOPK)),
        ),
    )
    last_v, last_g = pv[-1], pg[-1]
    pv = list(pv)
    k_iota = jax.lax.broadcasted_iota(jnp.int32, (sr, 128), 1)
    fv = jnp.full((sr, 128), NEG, F32)
    fi = jnp.zeros((sr, 128), F32)
    m = gsel = None
    for j in range(TOPK_OUT):
        mv = pv[0]
        for k in range(1, LANE_TOPK):
            mv = jnp.maximum(mv, pv[k])
        m = jnp.max(mv, axis=1, keepdims=True)
        cid = jnp.where(pv[0] == m, pg[0], big)
        for k in range(1, LANE_TOPK):
            cid = jnp.minimum(cid, jnp.where(pv[k] == m, pg[k], big))
        gsel = jnp.min(cid, axis=1, keepdims=True)
        for k in range(LANE_TOPK):
            pv[k] = jnp.where(pg[k] == gsel, -jnp.inf, pv[k])
        fv = jnp.where(k_iota == j, m, fv)
        fi = jnp.where(k_iota == j, gsel, fi)
    # A lane can have dropped a qualifying element only if its last kept entry
    # ranks at or ahead of the last (TOPK_OUT-th) selection.
    unsafe = (last_v > m) | ((last_v == m) & (last_g <= gsel))
    unsafe = unsafe & (jax.lax.broadcasted_iota(jnp.int32, (sr, 128), 0) < live_rows)
    return fv, fi, unsafe


class _Spec:
    """Static geometry for one compiled (config, world size, batch, cache)."""

    def __init__(
        self,
        cfg: Glm53MegakernelConfig,
        world_size: int,
        batch: int,
        max_batch: int,
        n_pages: int,
        *,
        speculative: bool = False,
        experts_kmajor: bool = False,
    ) -> None:
        self.cfg = cfg
        self.w = world_size
        self.spec = speculative
        # True: routed experts in the vLLM TP layout, gate_up [E, H, 2I/W] and
        # down [E, I/W, H] (K-major), so the decode reads the serving engine's
        # single expert copy. False: the N-major TP layout [E, 2I/W, H] and
        # [E, H/W, I] (down sharded on its output).
        self.experts_kmajor = experts_kmajor
        self.q = 4 if speculative else 1
        self.b_seq = batch
        self.b = batch * self.q
        self.rows = _round_up(max(self.b, ROWS), 16)
        self.bmax = max_batch
        self.n_pages = n_pages
        self.h = cfg.hidden_size
        self.l = cfg.num_hidden_layers
        self.l_dense = cfg.first_k_dense_replace
        self.hk = cfg.kda_num_heads // world_size
        self.dk = cfg.kda_head_dim
        self.ik = self.hk * self.dk
        self.hm = cfg.mla_num_heads // world_size
        self.dqk = cfg.qk_nope_head_dim
        self.dv = cfg.v_head_dim
        self.kv = cfg.kv_lora_rank
        self.ql = cfg.q_lora_rank
        self.ps = cfg.page_size
        self.prow = _round_up(cfg.page_size, 32)
        self.g_kv = KV_GROUP
        self.kprow = _round_up(self.g_kv * self.prow, 128)
        self.n_kpages = max(
            1, _round_up(n_pages, world_size * self.g_kv) // (world_size * self.g_kv)
        )
        self.hm_pad = _round_up(self.hm, 8)
        self.qh_loc = self.q * self.hm
        self.qh_loc_pad = _round_up(self.qh_loc, 8)
        self.h_mla_pad = _round_up(world_size * self.hm, 8)
        self.lk = len(cfg.kda_layers)
        self.lm = len(cfg.mla_layers)
        self.e_loc = cfg.n_routed_experts // world_size
        self.e_pad = _round_up(cfg.n_routed_experts, 128)
        self.topk = cfg.num_experts_per_tok
        self.i_moe = cfg.moe_intermediate_size
        self.i_dense = cfg.intermediate_size // world_size
        self.i_sh = cfg.n_shared_experts * cfg.moe_intermediate_size // world_size
        self.v_loc = cfg.vocab_size // world_size
        self.v_pad = _round_up(self.v_loc, 128)
        self.mix = (2 + cfg.hc_mult) * cfg.hc_mult
        # Sublane dim of the mHC fn weight (a bf16 tile is 16 rows).
        self.mix_pad = _round_up(self.mix, 32)
        self.meta_rows = _round_up(_M_HEADS + 2 * self.hk, 8)
        self.meta_w = max(
            self.h, self.ql, self.kv, self.dk, self.e_pad, self.mix_pad, 128
        )
        self.meta_w = _round_up(self.meta_w, 128)
        self.ar_rows = _round_up(self.b * (self.h // 128), 16)
        # Fused q | k | v | f_a | g_a | b input projection (b padded to 128).
        self.kda_in = 3 * self.ik + 2 * self.dk + 128
        # Routed expert intermediate slice and output slice per rank.
        self.il = self.i_moe // world_size
        self.hl = self.h // world_size
        # Rows carried by the activation all-gather.
        self.sr = 8 if self.b <= 8 else (16 if self.b <= 16 else self.rows)
        # Bound on the number of distinct routed experts of one step.
        self.nd_pad = _round_up(min(self.b * self.topk, cfg.n_routed_experts), 8)
        self.act_slots = min(self.nd_pad, 32 if speculative else 64)
        if (
            self.il % 128
            or self.hl % 128
            or self.hl > 2 * self.il
            or self.i_moe > self.h
        ):
            raise ValueError(
                "TP experts need 128-aligned I/W and H/W, H/W <= 2I/W and I <= H "
                "(ring reuse)"
            )
        if self.h % 128 or self.ik % 128 or self.dk % 128 or self.kv % 128:
            raise ValueError("kernel needs 128-lane aligned hidden/KDA/latent widths")
        if cfg.kda_conv_kernel_size != 4:
            raise ValueError(
                "kernel implements a 4-tap KDA short conv, got "
                f"kda_conv_kernel_size={cfg.kda_conv_kernel_size}"
            )
        if batch > ROWS or batch > max_batch:
            raise ValueError(
                f"batch {batch} exceeds row tile {ROWS} or max_batch {max_batch}"
            )
        if self.v_loc % 16:
            raise ValueError(
                "per-rank vocab must be a multiple of 16 (embedding row DMA)"
            )
        if TOPK_OUT * 2 > 128 or self.topk > 128:
            raise ValueError("top-k packing assumes <= 64 outputs and <= 128 routes")


# ----------------------------------------------------------------------------
# Weight and cache layout (pure layout transforms, run once at load time)
# ----------------------------------------------------------------------------

KERNEL_WEIGHT_NAMES = (
    "meta",
    "embed_tokens",
    "lm_head",
    "hc_attn_fn",
    "hc_ffn_fn",
    "kda_in_proj",
    "kda_conv",
    "kda_f_b_proj",
    "kda_g_b_proj",
    "kda_o_proj",
    "mla_a_proj_q",
    "mla_a_proj_s",
    "mla_q_b_proj_q",
    "mla_q_b_proj_s",
    "mla_w_uk",
    "mla_w_uv",
    "mla_o_proj_q",
    "mla_o_proj_s",
    "dense_gu_q",
    "dense_gu_s",
    "dense_down_q",
    "dense_down_s",
    "moe_router_w",
    "moe_shared_gu_q",
    "moe_shared_gu_s",
    "moe_shared_down_q",
    "moe_shared_down_s",
    "moe_expert_gate_up_q",
    "moe_expert_gate_up_s",
    "moe_expert_down_q",
    "moe_expert_down_s",
)

KERNEL_MTP_WEIGHT_NAMES = (
    "mtp_eh_proj",
    "mtp_mla_a_proj_q",
    "mtp_mla_a_proj_s",
    "mtp_mla_q_b_proj_q",
    "mtp_mla_q_b_proj_s",
    "mtp_mla_w_uk",
    "mtp_mla_w_uv",
    "mtp_mla_o_proj_q",
    "mtp_mla_o_proj_s",
    "mtp_moe_router_w",
    "mtp_moe_shared_gu_q",
    "mtp_moe_shared_gu_s",
    "mtp_moe_shared_down_q",
    "mtp_moe_shared_down_s",
    "mtp_moe_expert_gate_up_q",
    "mtp_moe_expert_gate_up_s",
    "mtp_moe_expert_down_q",
    "mtp_moe_expert_down_s",
)

EXPERT_NAMES = (
    "moe_expert_gate_up_q",
    "moe_expert_gate_up_s",
    "moe_expert_down_q",
    "moe_expert_down_s",
)
MTP_EXPERT_NAMES = (
    "mtp_moe_expert_gate_up_q",
    "mtp_moe_expert_gate_up_s",
    "mtp_moe_expert_down_q",
    "mtp_moe_expert_down_s",
)


def _pad_axis(x: jax.Array, axis: int, size: int, value: float = 0.0) -> jax.Array:
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, size - x.shape[axis])
    return jnp.pad(x, pad, constant_values=value)


def expert_ep_to_tp(
    w: dict[str, jax.Array], cfg: Glm53MegakernelConfig, world_size: int
) -> dict[str, jax.Array]:
    """Expert-parallel whole experts -> TP slices of every expert (per rank).

    Runs inside a shard_map over 'tp'. Input (per-channel fp8, the
    `load.to_per_channel_fp8` layout): gate_up_q [L, E/W, 2I, H], gate_up_s
    [L, E/W, 2I, 1], down_q [L, E/W, H, I], down_s [L, E/W, H, 1]. Output:
    gate_up [L, E, 2I/W, H] holding gate rows [rI/W, (r+1)I/W) then the
    matching up rows; down [L, E, H/W, I] holding output rows
    [rH/W, (r+1)H/W). Values are moved, never changed. One all_to_all per
    tensor at load time.
    """
    out = dict(w)
    for name in EXPERT_NAMES + MTP_EXPERT_NAMES:
        if name in w:
            out[name] = _ep_to_tp_one(name, w[name], cfg, world_size)
    return out


def experts_tp_to_kmajor(
    cfg: Glm53MegakernelConfig,
    mesh: Any,
    w: dict[str, jax.Array],
    donate: bool = True,
) -> dict[str, jax.Array]:
    """N-major TP routed experts -> the K-major layout of `experts_kmajor`.

    gate_up [L, E, 2I/W, H] -> [L, E, H, 2I/W] (a local transpose; the same
    [gate | up] halves per rank). down [L, E, H/W, I] (output sharded) ->
    [L, E, I/W, H] (contraction sharded): an all-to-all moves the I slices.
    Scales are already [L, E, 1, N] in the kernel weights: gate_up_s is
    unchanged and down_s is all-gathered to [L, E, 1, H].
    """
    world = int(mesh.size)
    p = jax.sharding.PartitionSpec

    def conv(name: str, x: jax.Array) -> jax.Array:
        def local(xr: jax.Array) -> jax.Array:
            xr = xr[0]
            if name.endswith("gate_up_q"):
                return jnp.swapaxes(xr, -1, -2)[None]
            if name.endswith("gate_up_s"):
                return xr[None]  # already [L, E, 1, 2I/W]
            if name.endswith("down_s"):  # [L, E, 1, H/W] -> [L, E, 1, H]
                gathered = (
                    jax.lax.all_gather(xr, "tp", axis=3, tiled=True)
                    if world > 1
                    else xr
                )
                return gathered[None]
            # down_q [L, E, H/W, I] -> [L, E, I/W, H]
            l_, e_, hl, i_ = xr.shape
            y = xr.reshape(l_, e_, hl, world, i_ // world)
            if world > 1:
                # -> [L, E, H, 1, I/W]
                y = jax.lax.all_to_all(y, "tp", split_axis=3, concat_axis=2, tiled=True)
            y = y.reshape(l_, e_, hl * world, i_ // world)
            return jnp.swapaxes(y, -1, -2)[None]

        return jax.jit(
            jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(p("tp"),),
                out_specs=p("tp"),
                check_vma=False,
            ),
            donate_argnums=(0,) if donate else (),
        )(x)

    out = dict(w)
    for name in EXPERT_NAMES + MTP_EXPERT_NAMES:
        if name in out:
            out[name] = conv(name, out.pop(name))
    return out


def _ep_to_tp_one(
    name: str, x: jax.Array, cfg: Glm53MegakernelConfig, world: int
) -> jax.Array:
    """`expert_ep_to_tp` for one tensor."""
    i_moe, h = cfg.moe_intermediate_size, cfg.hidden_size
    lead = x.shape[:2]
    if "gate_up" in name:
        il = i_moe // world
        g = x[:, :, :i_moe].reshape(*lead, world, il, *x.shape[3:])
        u = x[:, :, i_moe:].reshape(*lead, world, il, *x.shape[3:])
        x = jnp.concatenate([g, u], axis=3)
    else:
        x = x.reshape(*lead, world, h // world, *x.shape[3:])
    if world == 1:
        return x[:, :, 0]
    return jax.lax.all_to_all(x, "tp", split_axis=2, concat_axis=1, tiled=True)[:, :, 0]


def prepare_kernel_weights_local(
    cfg: Glm53MegakernelConfig,
    w: dict[str, jax.Array],
    world_size: int,
    experts_are_tp: bool = False,
) -> dict[str, jax.Array]:
    """Per-rank layout repack of per-channel fp8 weights.

    `w` is `load.to_per_channel_fp8` output for one rank. No value changes:
    transposes per-channel scales to lane vectors, zero-pads lane dims to 128,
    and packs every small per-layer vector into one float32 `meta` block per
    layer. `experts_are_tp=True` means the routed experts are already in a TP
    layout (or absent); otherwise they are converted from the expert-parallel
    layout with `expert_ep_to_tp`.
    """
    sp = _Spec(cfg, world_size, 1, 1, 1)
    if not experts_are_tp:
        w = expert_ep_to_tp(w, cfg, world_size)
    out: dict[str, jax.Array] = {}
    for name, val in w.items():
        if name.endswith("_s") and val.shape[-1] == 1:
            # [..., N, 1] -> [..., 1, N]
            out[name] = jnp.swapaxes(val, -1, -2).astype(F32)
        else:
            out[name] = val
    out["lm_head"] = _pad_axis(w["lm_head"], 0, sp.v_pad)
    out["hc_attn_fn"] = _pad_axis(w["hc_attn_fn"], 1, sp.mix_pad)
    out["hc_ffn_fn"] = _pad_axis(w["hc_ffn_fn"], 1, sp.mix_pad)
    # Fused projections sharing one input: one weight stream each (layout
    # only; the per-column math is unchanged).
    out["kda_in_proj"] = jnp.concatenate(
        [
            w["kda_q_proj"],
            w["kda_k_proj"],
            w["kda_v_proj"],
            w["kda_f_a_proj"],
            w["kda_g_a_proj"],
            _pad_axis(w["kda_b_proj"], 1, 128),
        ],
        axis=1,
    )

    def cat_s(*names: str) -> jax.Array:
        return jnp.concatenate([out[n] for n in names], axis=-1)

    out["mla_a_proj_q"] = jnp.concatenate(
        [w["mla_q_a_proj_q"], w["mla_kv_a_proj_q"]], axis=1
    )
    out["mla_a_proj_s"] = cat_s("mla_q_a_proj_s", "mla_kv_a_proj_s")
    out["dense_gu_q"] = jnp.concatenate([w["dense_gate_q"], w["dense_up_q"]], axis=1)
    out["dense_gu_s"] = cat_s("dense_gate_s", "dense_up_s")
    out["moe_shared_gu_q"] = jnp.concatenate(
        [w["moe_shared_gate_q"], w["moe_shared_up_q"]], axis=1
    )
    out["moe_shared_gu_s"] = cat_s("moe_shared_gate_s", "moe_shared_up_s")
    out["kda_conv"] = w["kda_conv"].reshape(
        w["kda_conv"].shape[0], w["kda_conv"].shape[1], -1
    )
    out["moe_router_w"] = _pad_axis(w["moe_router_w"], 1, sp.e_pad)

    has_mtp = "mtp_eh_proj" in w
    if has_mtp:
        out["mtp_mla_a_proj_q"] = jnp.concatenate(
            [w["mtp_mla_q_a_proj_q"], w["mtp_mla_kv_a_proj_q"]], axis=1
        )
        out["mtp_mla_a_proj_s"] = cat_s("mtp_mla_q_a_proj_s", "mtp_mla_kv_a_proj_s")
        out["mtp_moe_shared_gu_q"] = jnp.concatenate(
            [w["mtp_moe_shared_gate_q"], w["mtp_moe_shared_up_q"]], axis=1
        )
        out["mtp_moe_shared_gu_s"] = cat_s(
            "mtp_moe_shared_gate_s", "mtp_moe_shared_up_s"
        )
        out["mtp_moe_router_w"] = _pad_axis(w["mtp_moe_router_w"], 1, sp.e_pad)

    l_all, mw = sp.l, sp.meta_w
    meta = jnp.zeros((l_all + (2 if has_mtp else 1), sp.meta_rows, mw), F32)

    def put(m: jax.Array, rows: Any, row: int, vals: jax.Array) -> jax.Array:
        vals = vals.astype(F32)
        return m.at[rows, row, : vals.shape[-1]].set(vals)

    every = jnp.arange(l_all)
    meta = put(meta, every, _M_IN_LN, w["input_layernorm"])
    meta = put(meta, every, _M_POST_LN, w["post_attention_layernorm"])
    meta = put(meta, every, _M_HCA_BASE, w["hc_attn_base"])
    meta = put(meta, every, _M_HCA_SCALE, w["hc_attn_scale"])
    meta = put(meta, every, _M_HCF_BASE, w["hc_ffn_base"])
    meta = put(meta, every, _M_HCF_SCALE, w["hc_ffn_scale"])
    mla = jnp.asarray(cfg.mla_layers, jnp.int32)
    kda = jnp.asarray(cfg.kda_layers, jnp.int32)
    moe = jnp.arange(sp.l_dense, l_all, dtype=jnp.int32)
    if len(cfg.mla_layers):
        meta = put(meta, mla, _M_QA_NORM, w["mla_q_a_norm"])
        meta = put(meta, mla, _M_KVA_NORM, w["mla_kv_a_norm"])
    if len(cfg.kda_layers):
        meta = put(meta, kda, _M_O_NORM, w["kda_o_norm"])
        for h in range(sp.hk):
            a_row = jnp.broadcast_to(
                w["kda_a_log"][:, h : h + 1].astype(F32), (len(cfg.kda_layers), sp.dk)
            )
            meta = put(meta, kda, _M_HEADS + h, a_row)
            meta = put(meta, kda, _M_HEADS + sp.hk + h, w["kda_dt_bias"][:, h, :])
    if l_all > sp.l_dense:
        rbias = _pad_axis(w["moe_router_bias"].astype(F32), 1, sp.e_pad, NEG)
        meta = put(meta, moe, _M_RBIAS, rbias)
    meta = meta.at[l_all, 0, : sp.h].set(w["final_norm"].astype(F32))
    if has_mtp:
        m_mtp = l_all + 1
        meta = put(meta, m_mtp, _M_IN_LN, w["mtp_input_layernorm"])
        meta = put(meta, m_mtp, _M_POST_LN, w["mtp_post_attention_layernorm"])
        meta = put(meta, m_mtp, _M_HCA_BASE, w["mtp_enorm"])
        meta = put(meta, m_mtp, _M_HCA_SCALE, w["mtp_hnorm"])
        meta = put(meta, m_mtp, _M_HCF_BASE, w["mtp_shared_head_norm"])
        meta = put(meta, m_mtp, _M_QA_NORM, w["mtp_mla_q_a_norm"][0])
        meta = put(meta, m_mtp, _M_KVA_NORM, w["mtp_mla_kv_a_norm"][0])
        mtp_rbias = _pad_axis(w["mtp_moe_router_bias"][0].astype(F32), 0, sp.e_pad, NEG)
        meta = put(meta, m_mtp, _M_RBIAS, mtp_rbias)
    out["meta"] = meta
    all_names = KERNEL_WEIGHT_NAMES + KERNEL_MTP_WEIGHT_NAMES
    return {k: out[k] for k in all_names if k in out}


def layer_tables(cfg: Glm53MegakernelConfig) -> np.ndarray:
    """int32 [3 * L]: attention kind (0 KDA, 1 MLA), attention and FFN index."""
    kind = np.zeros(cfg.num_hidden_layers, np.int32)
    sub = np.zeros(cfg.num_hidden_layers, np.int32)
    ffn = np.zeros(cfg.num_hidden_layers, np.int32)
    for lid in range(cfg.num_hidden_layers):
        if lid in cfg.mla_layers:
            kind[lid], sub[lid] = 1, cfg.mla_layers.index(lid)
        else:
            sub[lid] = cfg.kda_layers.index(lid)
        ffn[lid] = (
            lid if lid < cfg.first_k_dense_replace else lid - cfg.first_k_dense_replace
        )
    return np.concatenate([kind, sub, ffn])


def _kv_tile(wd: jax.Array) -> jax.Array:
    """In kernel: int32 tile [prow/4, L] -> fp8 [prow, L] (inverse of kv_words)."""
    if wd.dtype != jnp.int32:
        raise TypeError(wd.dtype)
    return pltpu.bitcast(wd, F8)


def kv_words(q: jax.Array) -> jax.Array:
    """fp8 [..., R, L] -> int32 [..., R/4, L].

    Byte k of word (r, l) is q[4r + k, l]: the sublane packing that
    `pltpu.bitcast(words, fp8)` undoes inside the kernel.
    """
    *lead, r, lanes = q.shape
    q = jnp.swapaxes(q.reshape(*lead, r // 4, 4, lanes), -1, -2)
    return jax.lax.bitcast_convert_type(q, jnp.int32)


def kv_bytes(wd: jax.Array) -> jax.Array:
    """Inverse of kv_words: int32 [..., R/4, L] -> fp8 [..., R, L]."""
    *lead, r4, lanes = wd.shape
    return jnp.swapaxes(jax.lax.bitcast_convert_type(wd, F8), -1, -2).reshape(
        *lead, r4 * 4, lanes
    )


def kernel_cache_token_loc(
    pos: Any, page_size: int, world_size: int, kv_group: int | None = None
) -> tuple[Any, Any, Any]:
    """Token position -> (owner rank, superpage index, row in superpage).

    The latent cache is sequence sharded: page `lp = pos // page_size` lives
    on rank `lp % world_size`, and that rank's local pages are grouped by
    `kv_group` into superpages whose pages are padded to 32-row multiples.
    """
    if kv_group is None:
        kv_group = KV_GROUP
    prow = _round_up(page_size, 32)
    lp, off = pos // page_size, pos % page_size
    owner = lp % world_size
    k = lp // world_size
    grp, g = k // kv_group, k % kv_group
    return owner, grp, g * prow + off


def latent_pool_shapes(
    cfg: Glm53MegakernelConfig, num_layers: int, num_kpages: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Per-rank shapes of the latent superpage pool (codes int32, scales f32).

    `num_layers` counts the cached MLA layers (the target MLA layers, plus the
    MTP layer for the speculative step); `num_kpages` counts physical
    superpages including the null superpage 0.
    """
    prow = _round_up(cfg.page_size, 32)
    kprow = _round_up(KV_GROUP * prow, 128)
    kv = cfg.kv_lora_rank
    return (
        (num_layers, num_kpages, kprow // 4, 2 * kv),
        (num_layers, num_kpages, 8, kprow),
    )


def superpages_per_slot(
    page_size: int, max_model_len: int, world_size: int, *, lookahead: int = 0
) -> int:
    """Superpages one slot needs to hold `max_model_len + lookahead` tokens."""
    n_pages = -(-(max_model_len + lookahead) // page_size)
    return max(1, -(-n_pages // (world_size * KV_GROUP)))


# ----------------------------------------------------------------------------
# Kernel
# ----------------------------------------------------------------------------


def _build_kernel(  # noqa: C901
    sp: _Spec,
    *,
    fp8_native: bool,
    return_logits: bool,
    profile_scopes: bool = False,
    ignore_eos: bool = False,
):
    cfg = sp.cfg

    def scope(name: str) -> Any:
        """Device trace region (tpu.trace_start/stop) for XProf phase attribution; off
        by
        default.
        """
        return jax.named_scope(name) if profile_scopes else contextlib.nullcontext()

    eps = cfg.rms_norm_eps
    names = KERNEL_WEIGHT_NAMES + (KERNEL_MTP_WEIGHT_NAMES if sp.spec else ())
    n_w = len(names)
    fp8_dt = F8 if fp8_native else F32
    dot_dt = (
        BF16 if fp8_native else F32
    )  # CPU interpret: bf16 values, exact f32 products
    tpu = fp8_native
    r_v = sp.rows

    def hdot_nt(a_f32: jax.Array, b_bf16: jax.Array) -> jax.Array:
        """HIGHEST-precision a [M,K] f32 x b [N,K] (b exact in bf16), without f32 MXU
        operand copies on TPU.
        """
        if not tpu:
            return _nt(a_f32, b_bf16.astype(F32), HIGHEST)
        hi, mid, lo = _split3(a_f32)
        return _nt(hi, b_bf16) + _nt(mid, b_bf16) + _nt(lo, b_bf16)

    b_rows = sp.b

    def kernel(*refs):  # noqa: C901
        tok_ref, pos_ref, slot_ref, valid_ref, ltab_ref, kpt_ref, rank_ref, dev_ref = (
            refs[:N_PREFETCH]
        )
        w = dict(zip(names, refs[N_PREFETCH : N_PREFETCH + n_w]))
        conv_in, rec_in, kvq_in, kvs_in = refs[N_PREFETCH + n_w : N_PREFETCH + 4 + n_w]
        outs = refs[N_PREFETCH + 4 + n_w :]
        top_lp_ref, top_id_ref, next_ref = outs[0], outs[1], outs[2]
        # States are read-only inputs; the kernel emits the new KDA states per row and
        # the new MLA latent rows; XLA writes them into the donated states (no full
        # state copies).
        conv_hbm, rec_hbm, kvq_hbm, kvs_hbm = conv_in, rec_in, kvq_in, kvs_in
        conv_ckpt: Any = None
        rec_ckpt: Any = None
        if sp.spec:
            conv_ckpt, rec_ckpt, conv_new, rec_new, kvrow_q, kvrow_s = outs[3:9]
            k = 9
        else:
            conv_new, rec_new, kvrow_q, kvrow_s = outs[3:7]
            k = 7
        logits_out: Any = None
        if return_logits:
            logits_out = outs[k]
            k += 1
        (
            meta_v,
            meta_sem,
            streams_v,
            x_v,
            attn_v,
            ar_buf,
            ar_send,
            ar_recv,
            gather_buf,
            gather_send,
            gather_recv,
            logits_v,
        ) = outs[k : k + 12]
        fn_bufs, fn_sems = outs[k + 12], outs[k + 13]
        # Routed expert activation all-gather and MLA sequence-shard buffers: top level
        # scratch because peers write into them remotely (a run_scoped buffer could
        # alias another region on a peer that is behind).
        act_buf, act_send, act_recv = outs[k + 14], outs[k + 15], outs[k + 16]
        mla_part, mla_send, mla_recv = outs[k + 17], outs[k + 18], outs[k + 19]
        kda_conv_buf, kda_rec_buf, kda_sems = outs[k + 20], outs[k + 21], outs[k + 22]
        del conv_in, rec_in, kvq_in, kvs_in  # aliased to the HBM outputs
        rank = rank_ref[0]

        def peer(q: Any) -> Any:
            # `DeviceIdType.MESH` coordinate of logical rank q (identity when the
            # caller's mesh orders `mesh.devices` by TP rank).
            return dev_ref[q]

        row_iota = jax.lax.broadcasted_iota(jnp.int32, (r_v, 1), 0)
        live_rows = row_iota < b_rows

        def sync_copy(src: Any, dst: Any, sem: Any) -> None:
            c = pltpu.make_async_copy(src, dst, sem)
            c.start()
            c.wait()

        # ---------------- collectives ----------------
        if sp.w > 1:
            barrier = pltpu.get_barrier_semaphore()
            for off in range(1, sp.w):
                pl.semaphore_signal(
                    barrier,
                    1,
                    device_id=(peer(rank ^ off),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, sp.w - 1)

        n_lane_blocks = sp.h // 128

        def all_reduce(part: jax.Array, slot: int, n_rows: int = b_rows) -> jax.Array:
            """Sum bf16 partials [R, H] over ranks in fixed rank order; returns bf16 [R,
            H].
            """
            r_out = part.shape[0]
            ar_used = _round_up(n_rows * n_lane_blocks, 16)
            dst = ar_buf.at[slot, rank]
            for c in range(n_lane_blocks):
                dst[pl.ds(c * n_rows, n_rows), :] = part[
                    :n_rows, c * 128 : (c + 1) * 128
                ].astype(BF16)
            copies = []
            for off in range(1, sp.w):
                cp = pltpu.make_async_remote_copy(
                    ar_buf.at[slot, rank, pl.ds(0, ar_used)],
                    ar_buf.at[slot, rank, pl.ds(0, ar_used)],
                    ar_send.at[slot, off - 1],
                    ar_recv.at[slot, off - 1],
                    device_id=(peer(rank ^ off),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                cp.start()
                copies.append(cp)
            for cp in copies:
                cp.wait_send()
                cp.wait_recv()
            if n_rows >= 8 and n_rows % 8 == 0:
                step_c = min(
                    n_lane_blocks, 2 if n_rows >= 32 else (4 if n_rows >= 16 else 8)
                )
                for c0 in range(0, n_lane_blocks, step_c):
                    sl = pl.ds(c0 * n_rows, step_c * n_rows)
                    tot_c = ar_buf[slot, 0, sl].astype(F32)
                    for r in range(1, sp.w):
                        tot_c = tot_c + ar_buf[slot, r, sl].astype(F32)
                    ar_buf[slot, 0, sl] = tot_c.astype(BF16)
                pieces = [
                    ar_buf[slot, 0, pl.ds(c * n_rows, n_rows)]
                    for c in range(n_lane_blocks)
                ]
                res = jnp.concatenate(pieces, axis=1)
            else:
                tot = ar_buf[slot, 0, :ar_used].astype(F32)
                for r in range(1, sp.w):
                    tot = tot + ar_buf[slot, r, :ar_used].astype(F32)
                pieces = [
                    tot[c * n_rows : (c + 1) * n_rows, :] for c in range(n_lane_blocks)
                ]
                res = jnp.concatenate(pieces, axis=1).astype(BF16)
            if n_rows < r_out:
                res = jnp.concatenate(
                    [res, jnp.zeros((r_out - n_rows, sp.h), BF16)], axis=0
                )
            return res

        # ---------------- streamed linear layers ----------------
        def stream_rows(
            src_at: Any, n: int, kdim: int, dtype: Any, consume: Any
        ) -> None:
            """Ring-buffered DMA of row tiles [tn, K] of an [N, K] HBM matrix;
            consume(off, rows, tile_ref).

            Up to STREAM_DEPTH tiles are in flight; a ragged last tile (n not a multiple
            of tn) is peeled
            into a static tail so large tiles can be used for any 128-multiple N.
            """
            itemsize = jnp.dtype(dtype).itemsize
            tn, nfull, tail = _stream_tiles(n, kdim, itemsize)
            nt = nfull + (1 if tail else 0)
            depth = min(STREAM_DEPTH, nt)

            def scoped(ring: Any, sems: Any) -> None:
                def cp(t: Any, s: Any, rows: int = tn) -> Any:
                    off = t * tn if isinstance(t, int) else pl.multiple_of(t * tn, 128)
                    return pltpu.make_async_copy(
                        src_at(off, rows), ring.at[s, pl.ds(0, rows)], sems.at[s]
                    )

                def start(t: int) -> None:
                    if t < nfull:
                        cp(t, t % depth).start()
                    else:
                        cp(t, t % depth, tail).start()

                for t0 in range(depth):
                    start(t0)

                def body(t: Any, carry: Any) -> Any:
                    s = t % depth
                    cp(t, s).wait()
                    consume(pl.multiple_of(t * tn, 128), tn, ring.at[s])

                    @pl.when(t + depth < nfull)
                    def _():
                        cp(t + depth, s).start()

                    if tail:

                        @pl.when(t + depth == nfull)
                        def _():
                            cp(nfull, s, tail).start()

                    return carry

                jax.lax.fori_loop(0, nfull, body, 0)
                if tail:
                    s = nfull % depth
                    cp(nfull, s, tail).wait()
                    consume(nfull * tn, tail, ring.at[s, pl.ds(0, tail)])

            pl.run_scoped(
                scoped,
                pltpu.VMEM((depth, tn, kdim), dtype),
                pltpu.SemaphoreType.DMA((depth,)),
            )

        def bf16_linear(x: jax.Array, w_at: Any, n: int, out_ref: Any) -> None:
            """reference.bf16_linear: out_ref[:, :n] = bf16(x @ W^T)."""
            xb = x.astype(BF16).astype(dot_dt)

            def consume(off: Any, tn: int, tile: Any) -> None:
                out_ref[:, pl.ds(off, tn)] = _nt(xb, tile[...].astype(dot_dt)).astype(
                    out_ref.dtype
                )

            stream_rows(w_at, n, x.shape[-1], BF16, consume)

        def with_scale(s_at: Any, n: int, fn: Any) -> None:
            def scoped(sv: Any, sem: Any) -> None:
                sync_copy(s_at, sv, sem.at[0])
                fn(sv)

            pl.run_scoped(
                scoped, pltpu.VMEM((1, n), F32), pltpu.SemaphoreType.DMA((1,))
            )

        def fp8_linear(
            x: jax.Array, q_at: Any, s_at: Any, n: int, out_ref: Any
        ) -> None:
            """Per-channel FP8 block matmul with per-row FP8 LHS."""
            xq, xs = _quant_rows(x)
            xq = xq.astype(fp8_dt)

            def run(sv: Any) -> None:
                def consume(off: Any, tn: int, tile: Any) -> None:
                    acc = _nt(xq, tile[...].astype(fp8_dt))
                    ws = sv[:, pl.ds(off, tn)]
                    out_ref[:, pl.ds(off, tn)] = (acc * xs * ws).astype(out_ref.dtype)

                stream_rows(q_at, n, x.shape[-1], F8, consume)

            with_scale(s_at, n, run)

        def scoped_out(shape: tuple[int, ...], dtype: Any, fn: Any) -> jax.Array:
            """Run fn(out_ref) into a scoped VMEM buffer and return its value."""

            def scoped(o: Any) -> jax.Array:
                fn(o)
                return o[...]

            return pl.run_scoped(scoped, pltpu.VMEM(shape, dtype))

        def meta_row(i: int, n: int) -> jax.Array:
            return meta_v[i : i + 1, :n]

        # ---------------- mHC ----------------
        def fn_copy(lid: Any, which: int) -> Any:
            src = (w["hc_attn_fn"] if which == 0 else w["hc_ffn_fn"]).at[lid]
            return pltpu.make_async_copy(src, fn_bufs.at[which], fn_sems.at[which])

        def mhc_collapse(fn_at: Any, base_row: int, scale_row: int, norm_row: int):
            hc = cfg.hc_mult
            kc = _lane_chunk(sp.h)
            nkc = sp.h // kc

            lid_, which = fn_at
            fn_v = fn_bufs.at[which]
            fn_copy(
                lid_, which
            ).wait()  # issued at layer start (overlaps the previous sublayer)

            def proj() -> tuple[jax.Array, jax.Array]:
                def body(i: Any, carry: Any) -> Any:
                    acc, ss = carry
                    si = i // nkc
                    off = pl.multiple_of((i % nkc) * kc, kc)
                    st_c = streams_v[si, :, pl.ds(off, kc)]
                    stf = st_c.astype(F32)
                    ss = ss + jnp.sum(stf * stf, axis=-1, keepdims=True)
                    fn_c = fn_v[:, pl.ds(pl.multiple_of(si * sp.h + off, kc), kc)]
                    # streams and fn are exact in bf16, so a bf16 MXU pass with f32
                    # accumulation equals HIGHEST.
                    return acc + _nt(st_c.astype(dot_dt), fn_c.astype(dot_dt)), ss

                init = (jnp.zeros((r_v, sp.mix_pad), F32), jnp.zeros((r_v, 1), F32))
                return jax.lax.fori_loop(0, hc * nkc, body, init)

            acc, ss = proj()
            inv_rms = jax.lax.rsqrt(ss / (hc * sp.h) + eps)
            sr = (
                8 if b_rows <= 8 else r_v
            )  # rows carried through the Sinkhorn (one f32 vreg at B <= 8)

            def lanes128(m: jax.Array) -> jax.Array:
                def scoped(buf: Any) -> jax.Array:
                    buf[...] = jnp.zeros((r_v, 128), F32)
                    buf[:, : sp.mix_pad] = m
                    return buf[:sr, :]

                return pl.run_scoped(scoped, pltpu.VMEM((r_v, 128), F32))

            # Lane layout (reference order): [pre 0..hc) [post hc..2hc) [comb (i, j) at
            # 2hc + hc*i + j].
            mix = lanes128(acc * inv_rms)
            lane = jax.lax.broadcasted_iota(jnp.int32, (sr, 128), 1)
            scale = meta_row(scale_row, 128)
            scale_v = jnp.where(
                lane < hc,
                scale[:, 0:1],
                jnp.where(lane < 2 * hc, scale[:, 1:2], scale[:, 2:3]),
            )
            z = mix * scale_v + meta_row(base_row, 128)
            sig = jax.nn.sigmoid(z)
            # A comb row group (fixed i) is hc consecutive lanes; a column group (fixed
            # j) has lane stride hc. Members are gathered with lane rotations and
            # combined in the reference's sequential order (member 0, 1, ...), so every
            # lane holds exactly the scalar loop's value.
            rel = jnp.maximum(lane - 2 * hc, 0)
            eq_row = [rel % hc == v for v in range(hc)]
            eq_col = [(rel // hc) % hc == v for v in range(hc)]

            def roll(c: jax.Array, k: int) -> jax.Array:
                return pltpu.roll(c, k, 1) if tpu else jnp.roll(c, k, 1)

            def group_reduce(c: jax.Array, eq: list, stride: int, op: Any) -> jax.Array:
                shifted = {
                    d: c if d == 0 else roll(c, (-d * stride) % 128)
                    for d in range(1 - hc, hc)
                }

                def member(m: int) -> jax.Array:
                    out = shifted[m]
                    for v in range(1, hc):
                        out = jnp.where(eq[v], shifted[m - v], out)
                    return out

                r = member(0)
                for m in range(1, hc):
                    r = op(r, member(m))
                return r

            e = jnp.exp(z - group_reduce(z, eq_row, 1, jnp.maximum))
            comb = e / group_reduce(e, eq_row, 1, jnp.add) + cfg.hc_eps
            comb = comb / (group_reduce(comb, eq_col, hc, jnp.add) + cfg.hc_eps)
            for _ in range(cfg.hc_sinkhorn_iters - 1):
                comb = comb / (group_reduce(comb, eq_row, 1, jnp.add) + cfg.hc_eps)
                comb = comb / (group_reduce(comb, eq_col, hc, jnp.add) + cfg.hc_eps)

            def colv(v: jax.Array, i: int) -> jax.Array:
                c = v[:, i : i + 1]
                if sr < r_v:
                    c = jnp.concatenate([c, jnp.zeros((r_v - sr, 1), F32)], axis=0)
                return c

            pre = [colv(sig, i) + cfg.hc_eps for i in range(hc)]
            post = [2.0 * colv(sig, hc + i) for i in range(hc)]
            comb = [
                [colv(comb, 2 * hc + hc * i + j) for j in range(hc)] for i in range(hc)
            ]
            collapsed = pre[0] * streams_v[0].astype(F32)
            for s in range(1, hc):
                collapsed = collapsed + pre[s] * streams_v[s].astype(F32)
            normed = _rms(collapsed.astype(BF16), meta_row(norm_row, sp.h), eps)
            return post, comb, normed

        def mhc_combine(sub: jax.Array, post: list, comb: list) -> None:
            hc = cfg.hc_mult

            def scoped(tmp: Any) -> None:
                subf = sub.astype(F32)
                for i in range(hc):
                    mixed = comb[0][i] * streams_v[0].astype(F32)
                    for kk in range(1, hc):
                        mixed = mixed + comb[kk][i] * streams_v[kk].astype(F32)
                    tmp[i] = jnp.where(
                        live_rows, (post[i] * subf + mixed).astype(BF16), 0
                    )
                for i in range(hc):
                    streams_v[i] = tmp[i]

            pl.run_scoped(scoped, pltpu.VMEM((hc, r_v, sp.h), BF16))

        # ---------------- KDA ----------------
        def kda_prefetch_start(ki: Any, ks: Any) -> None:
            kda_conv_buf[ks] = jnp.zeros(
                (ROWS, cfg.kda_conv_kernel_size - 1, 3 * sp.ik), BF16
            )
            for b in range(sp.b_seq):
                slot = slot_ref[b]
                pltpu.make_async_copy(
                    conv_hbm.at[ki, slot],
                    kda_conv_buf.at[ks, b],
                    kda_sems.at[ks, 2 * b],
                ).start()
                pltpu.make_async_copy(
                    rec_hbm.at[ki, slot],
                    kda_rec_buf.at[ks, b],
                    kda_sems.at[ks, 2 * b + 1],
                ).start()

        def kda_prefetch_wait(ki: Any, ks: Any) -> None:
            for b in range(sp.b_seq):
                slot = slot_ref[b]
                pltpu.make_async_copy(
                    conv_hbm.at[ki, slot],
                    kda_conv_buf.at[ks, b],
                    kda_sems.at[ks, 2 * b],
                ).wait()
                pltpu.make_async_copy(
                    rec_hbm.at[ki, slot],
                    kda_rec_buf.at[ks, b],
                    kda_sems.at[ks, 2 * b + 1],
                ).wait()

        def kda_body(x: jax.Array, ki: Any) -> jax.Array:
            hk, dk, ik = sp.hk, sp.dk, sp.ik
            n_in = sp.kda_in
            ks = ki % 2
            ns = (ki + 1) % 2

            @pl.when(ki + 1 < sp.lk)
            def _():
                kda_prefetch_start(ki + 1, ns)

            with scope("kda_in_proj"):
                fused = scoped_out(
                    (r_v, n_in),
                    BF16,
                    lambda o: bf16_linear(
                        x, lambda a, n: w["kda_in_proj"].at[ki, pl.ds(a, n)], n_in, o
                    ),
                )
            mixed = fused[:, : 3 * ik]
            f_a = fused[:, 3 * ik : 3 * ik + dk]
            g_a = fused[:, 3 * ik + dk : 3 * ik + 2 * dk]
            raw_beta = fused[:, 3 * ik + 2 * dk :]
            with scope("kda_fg_proj"):
                raw_gate = scoped_out(
                    (r_v, ik),
                    BF16,
                    lambda o: bf16_linear(
                        f_a, lambda a, n: w["kda_f_b_proj"].at[ki, pl.ds(a, n)], ik, o
                    ),
                )
                out_gate = scoped_out(
                    (r_v, ik),
                    BF16,
                    lambda o: bf16_linear(
                        g_a, lambda a, n: w["kda_g_b_proj"].at[ki, pl.ds(a, n)], ik, o
                    ),
                )

            def scoped(*s_refs: Any) -> jax.Array:
                conv_w, qkvg, out_v, sems = s_refs[:4]
                conv_step_v: Any = None
                rec_step_v: Any = None
                mix_v: Any = None
                taps_c_v: Any = None
                if sp.spec:
                    conv_step_v, rec_step_v, mix_v, taps_c_v = s_refs[4:8]
                sync_copy(w["kda_conv"].at[ki], conv_w, sems.at[0])
                kda_prefetch_wait(ki, ks)
                cw = conv_w[...].astype(F32)
                if not sp.spec:
                    acc = kda_conv_buf[ks, :, 0, :].astype(F32) * cw[0:1]
                    acc = acc + kda_conv_buf[ks, :, 1, :].astype(F32) * cw[1:2]
                    acc = acc + kda_conv_buf[ks, :, 2, :].astype(F32) * cw[2:3]
                    acc = acc + mixed.astype(F32) * cw[3:4]
                    conv_out = jax.nn.silu(acc)
                    valid_rows = jnp.zeros((r_v, 1), jnp.bool_)
                    for b in range(b_rows):
                        valid_rows = valid_rows | (
                            (row_iota == b) & (valid_ref[b] != 0)
                        )
                    new_c = [
                        kda_conv_buf[ks, :, 1, :],
                        kda_conv_buf[ks, :, 2, :],
                        mixed,
                    ]
                    for t in range(3):
                        kda_conv_buf[ks, :, t, :] = jnp.where(
                            valid_rows, new_c[t], kda_conv_buf[ks, :, t, :]
                        )
                else:
                    mix_v[...] = mixed
                    taps_c_v[...] = jnp.zeros((3, r_v, 3 * ik), BF16)
                    for b in range(sp.b_seq):
                        ok_b = valid_ref[b] != 0
                        c0 = kda_conv_buf[ks, b : b + 1, 0, :]
                        c1 = kda_conv_buf[ks, b : b + 1, 1, :]
                        c2 = kda_conv_buf[ks, b : b + 1, 2, :]
                        m0 = mix_v[4 * b + 0 : 4 * b + 1, :]
                        m1 = mix_v[4 * b + 1 : 4 * b + 2, :]
                        m2 = mix_v[4 * b + 2 : 4 * b + 3, :]
                        m3 = mix_v[4 * b + 3 : 4 * b + 4, :]
                        t0_r = (c0, c1, c2, m0)
                        t1_r = (c1, c2, m0, m1)
                        t2_r = (c2, m0, m1, m2)
                        m_r = (m0, m1, m2, m3)
                        for j in range(4):
                            r_idx = 4 * b + j
                            taps_c_v[0, r_idx : r_idx + 1, :] = t0_r[j]
                            taps_c_v[1, r_idx : r_idx + 1, :] = t1_r[j]
                            taps_c_v[2, r_idx : r_idx + 1, :] = t2_r[j]
                            conv_step_v[b, j, 0:1, :] = jnp.where(ok_b, t1_r[j], c0)
                            conv_step_v[b, j, 1:2, :] = jnp.where(ok_b, t2_r[j], c1)
                            conv_step_v[b, j, 2:3, :] = jnp.where(ok_b, m_r[j], c2)
                    acc = (
                        taps_c_v[0].astype(F32) * cw[0:1]
                        + taps_c_v[1].astype(F32) * cw[1:2]
                        + taps_c_v[2].astype(F32) * cw[2:3]
                        + mixed.astype(F32) * cw[3:4]
                    )
                    conv_out = jax.nn.silu(acc)

                qk_scale = jnp.float32(dk**-0.5)

                def heads(a: jax.Array) -> jax.Array:
                    return a.reshape(r_v, hk, dk)

                qf = heads(conv_out[:, :ik])
                kf = heads(conv_out[:, ik : 2 * ik])
                vf = heads(conv_out[:, 2 * ik :])
                qn = (
                    qf
                    * jax.lax.rsqrt(jnp.sum(qf * qf, axis=-1, keepdims=True) + 1e-6)
                    * qk_scale
                )
                kn = kf * jax.lax.rsqrt(jnp.sum(kf * kf, axis=-1, keepdims=True) + 1e-6)
                a_log = meta_v[_M_HEADS : _M_HEADS + hk, :dk]
                dt_b = meta_v[_M_HEADS + hk : _M_HEADS + 2 * hk, :dk]
                g_arg = jnp.exp(a_log)[None] * (
                    heads(raw_gate.astype(F32)) + dt_b[None]
                )
                g = jnp.float32(cfg.kda_gate_lower_bound) * jax.nn.sigmoid(g_arg)
                beta = jax.nn.sigmoid(raw_beta[:, :hk].astype(F32))
                qkvg[0] = qn
                qkvg[1] = kn
                qkvg[2] = vf
                qkvg[3] = g
                qkvg[4] = jnp.broadcast_to(beta[:, :, None], (r_v, hk, dk))
                out_v[...] = jnp.zeros(out_v.shape, F32)
                if not sp.spec:

                    def row_plain(i: Any, carry: Any) -> Any:
                        b = i // hk
                        hh = pl.ds(i % hk, 1)
                        s = kda_rec_buf[ks, b, hh]
                        ok = valid_ref[b] != 0
                        q_t, k_t, v_t, g_t, bt = (qkvg[m, b, hh] for m in range(5))
                        s_dec = s * jnp.exp(g_t)[:, :, None]
                        pred = jnp.sum(k_t[:, :, None] * s_dec, axis=1)
                        resid = v_t - pred
                        s_next = s_dec + (bt * k_t)[:, :, None] * resid[:, None, :]
                        o_t = jnp.sum(q_t[:, :, None] * s_next, axis=1)
                        kda_rec_buf[ks, b, hh] = jnp.where(ok, s_next, s)
                        out_v[b, hh] = jnp.where(ok, o_t, 0.0)
                        return carry

                    jax.lax.fori_loop(0, sp.b_seq * hk, row_plain, 0)
                    copies = []
                    for b in range(b_rows):
                        c1 = pltpu.make_async_copy(
                            kda_conv_buf.at[ks, b],
                            conv_new.at[b, ki],
                            sems.at[1 + 2 * b],
                        )
                        c2 = pltpu.make_async_copy(
                            kda_rec_buf.at[ks, b], rec_new.at[b, ki], sems.at[2 + 2 * b]
                        )
                        c1.start()
                        c2.start()
                        copies += [c1, c2]
                    for c in copies:
                        c.wait()
                else:
                    # Conv windows are tiny: one DMA for all sequences, overlapped with
                    # the recurrence.
                    c_conv = pltpu.make_async_copy(
                        conv_step_v, conv_ckpt.at[ki], sems.at[1]
                    )
                    c_conv.start()

                    # Recurrent checkpoints: double buffered per sequence so the 2 MiB
                    # write of sequence b overlaps the recurrence of sequence b + 1.
                    def rec_cp(b: Any) -> Any:
                        return pltpu.make_async_copy(
                            rec_step_v.at[b % 2], rec_ckpt.at[ki, b], sems.at[2 + b % 2]
                        )

                    def rec_seq(b: Any, carry: Any) -> Any:
                        ok = valid_ref[b] != 0
                        slot_b = b % 2

                        @pl.when(b >= 2)
                        def _():
                            rec_cp(b - 2).wait()

                        def head_body(hi: Any, h_carry: Any) -> Any:
                            hh = pl.ds(hi, 1)
                            s = kda_rec_buf[ks, b, hh]
                            for j in range(4):
                                r_idx = b * 4 + j
                                q_t, k_t, v_t, g_t, bt = (
                                    qkvg[m, r_idx, hh] for m in range(5)
                                )
                                s_dec = s * jnp.exp(g_t)[:, :, None]
                                pred = jnp.sum(k_t[:, :, None] * s_dec, axis=1)
                                resid = v_t - pred
                                s_next = (
                                    s_dec + (bt * k_t)[:, :, None] * resid[:, None, :]
                                )
                                o_t = jnp.sum(q_t[:, :, None] * s_next, axis=1)
                                s = jnp.where(ok, s_next, s)
                                out_v[r_idx, hh] = jnp.where(ok, o_t, 0.0)
                                rec_step_v[slot_b, j, hh] = s
                            return h_carry

                        jax.lax.fori_loop(0, hk, head_body, 0)
                        rec_cp(b).start()
                        return carry

                    jax.lax.fori_loop(0, sp.b_seq, rec_seq, 0)
                    for b_tail in range(max(0, sp.b_seq - 2), sp.b_seq):
                        rec_cp(b_tail).wait()
                    c_conv.wait()
                return out_v[...]

            kda_scoped_specs = [
                pltpu.VMEM((cfg.kda_conv_kernel_size, 3 * ik), BF16),
                pltpu.VMEM((5, r_v, hk, dk), F32),
                pltpu.VMEM((r_v, hk, dk), F32),
                pltpu.SemaphoreType.DMA((1 + 2 * max(b_rows, 2),)),
            ]
            if sp.spec:
                kda_scoped_specs += [
                    pltpu.VMEM(
                        (sp.b_seq, 4, cfg.kda_conv_kernel_size - 1, 3 * ik), BF16
                    ),
                    pltpu.VMEM((2, 4, hk, dk, dk), F32),
                    pltpu.VMEM((r_v, 3 * ik), BF16),
                    pltpu.VMEM((3, r_v, 3 * ik), BF16),
                ]
            with scope("kda_core"):
                attn = pl.run_scoped(scoped, *kda_scoped_specs)
            out_f = attn.astype(BF16).astype(F32)
            normed = out_f * jax.lax.rsqrt(
                jnp.mean(out_f * out_f, axis=-1, keepdims=True) + eps
            )
            normed = (normed * meta_v[_M_O_NORM : _M_O_NORM + 1, :dk][None]).astype(
                BF16
            )
            gated = normed * jax.nn.sigmoid(out_gate.reshape(r_v, hk, dk))
            gated = gated.reshape(r_v, ik)
            with scope("kda_o_proj"):
                proj = scoped_out(
                    (r_v, sp.h),
                    BF16,
                    lambda o: bf16_linear(
                        gated, lambda a, n: w["kda_o_proj"].at[ki, pl.ds(a, n)], sp.h, o
                    ),
                )
            return proj

        # ---------------- absorbed NoPE MLA ----------------
        def mla_body(x: jax.Array, mi: Any) -> jax.Array:
            hm, dqk, dv, kv, ps, prow = sp.hm, sp.dqk, sp.dv, sp.kv, sp.ps, sp.prow
            g_kv, kprow, hm_pad, h_pad = sp.g_kv, sp.kprow, sp.hm_pad, sp.h_mla_pad
            qh_pad = sp.q * h_pad
            with scope("mla_a_proj"):
                qkv_a = scoped_out(
                    (r_v, sp.ql + kv),
                    BF16,
                    lambda o: fp8_linear(
                        x,
                        lambda a, n: w["mla_a_proj_q"].at[mi, pl.ds(a, n)],
                        w["mla_a_proj_s"].at[mi],
                        sp.ql + kv,
                        o,
                    ),
                )
            q_lora = _rms(qkv_a[:, : sp.ql], meta_row(_M_QA_NORM, sp.ql), eps)
            kv_a = qkv_a[:, sp.ql :]
            lat = jnp.nan_to_num(
                _rms(kv_a, meta_row(_M_KVA_NORM, kv), eps).astype(F32),
                nan=0.0,
                posinf=448.0,
                neginf=-448.0,
            )
            s_hi = jnp.maximum(
                jnp.max(jnp.abs(lat), axis=-1, keepdims=True) / 448.0,
                jnp.float32(2.0**-14),
            )
            q_hi = jnp.clip(lat / s_hi, -448.0, 448.0).astype(F8).astype(F32)
            rem = lat - q_hi * s_hi
            s_lo = jnp.maximum(
                jnp.max(jnp.abs(rem), axis=-1, keepdims=True) / 448.0,
                jnp.float32(2.0**-24),
            )
            q_lo = jnp.clip(rem / s_lo, -448.0, 448.0).astype(F8).astype(F32)
            lane = jax.lax.broadcasted_iota(jnp.int32, (r_v, 128), 1)
            new_s = jnp.where(lane == 0, s_hi, jnp.where(lane == 1, s_lo, 0.0))
            new_q = jnp.concatenate([q_hi, q_lo], axis=1)
            with scope("mla_q_b_proj"):
                q_nope = scoped_out(
                    (r_v, hm * dqk),
                    BF16,
                    lambda o: fp8_linear(
                        q_lora,
                        lambda a, n: w["mla_q_b_proj_q"].at[mi, pl.ds(a, n)],
                        w["mla_q_b_proj_s"].at[mi],
                        hm * dqk,
                        o,
                    ),
                )
            sm_scale = float(dqk**-0.5)

            def scoped(
                wuk: Any,
                wuv: Any,
                qabs_loc: Any,
                u_v: Any,
                newq_v: Any,
                news_v: Any,
                ringq: Any,
                rings: Any,
                sems: Any,
                acc_v: Any,
                m_v: Any,
                l_v: Any,
            ) -> jax.Array:
                sync_copy(w["mla_w_uk"].at[mi], wuk, sems.at[0])
                sync_copy(w["mla_w_uv"].at[mi], wuv, sems.at[0])
                newq_v[...] = new_q
                news_v[...] = new_s
                ringq[...] = jnp.zeros(ringq.shape, jnp.int32)
                rings[...] = jnp.zeros(rings.shape, F32)
                hm_iota = jax.lax.broadcasted_iota(jnp.int32, (hm_pad, 1), 0)
                h_all_iota = jax.lax.broadcasted_iota(jnp.int32, (h_pad, 1), 0)
                stat_lane = jax.lax.broadcasted_iota(jnp.int32, (qh_pad, 128), 1)
                c_iota = jax.lax.broadcasted_iota(jnp.int32, (1, kprow), 1)
                sub_g, sub_o = c_iota // prow, c_iota % prow
                base_k_pos = jnp.where(
                    (sub_g < g_kv) & (sub_o < ps),
                    sub_g * (sp.w * ps) + sub_o,
                    jnp.int32(2**29),
                )
                q_by_h = [
                    _nn(
                        q_nope[:, h * dqk : (h + 1) * dqk].astype(dot_dt),
                        wuk[h].astype(dot_dt),
                    )
                    for h in range(hm)
                ]
                for b in range(b_rows):
                    qb = jnp.zeros((hm_pad, kv), F32)
                    for h in range(hm):
                        row_b = jnp.sum(
                            jnp.where(row_iota == b, q_by_h[h], 0.0),
                            axis=0,
                            keepdims=True,
                        )
                        qb = jnp.where(hm_iota == h, row_b, qb)
                    qabs_loc[b] = qb
                    mla_part[0, rank, b] = jnp.concatenate(
                        [qb, jnp.zeros((hm_pad, 128), F32)], axis=1
                    )

                if sp.w > 1:
                    q_copies = []
                    for off in range(1, sp.w):
                        cp_q = pltpu.make_async_remote_copy(
                            mla_part.at[0, rank],
                            mla_part.at[0, rank],
                            mla_send.at[0, off - 1],
                            mla_recv.at[0, off - 1],
                            device_id=(peer(rank ^ off),),
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                        cp_q.start()
                        q_copies.append(cp_q)
                    for cp_q in q_copies:
                        cp_q.wait_send()
                        cp_q.wait_recv()

                c128_iota = jax.lax.broadcasted_iota(jnp.int32, (128, 1), 0)
                l128_iota = jax.lax.broadcasted_iota(jnp.int32, (hm_pad, 128), 1)
                q_j_iota = jax.lax.broadcasted_iota(jnp.int32, (qh_pad, 1), 0) // h_pad

                def row(b: Any, carry: Any) -> Any:
                    pos = pos_ref[b]
                    slot = slot_ref[b]
                    q_parts = []
                    for j in range(sp.q):
                        r_idx = b * sp.q + j
                        if hm == hm_pad:
                            q_j = jnp.concatenate(
                                [mla_part[0, r, r_idx, :, :kv] for r in range(sp.w)],
                                axis=0,
                            )
                        else:
                            q_j = jnp.zeros((h_pad, kv), F32)
                            for r in range(sp.w):
                                part_r = mla_part[0, r, r_idx, :, :kv]
                                for h in range(hm):
                                    rh = jnp.sum(
                                        jnp.where(hm_iota == h, part_r, 0.0),
                                        axis=0,
                                        keepdims=True,
                                    )
                                    q_j = jnp.where(h_all_iota == r * hm + h, rh, q_j)
                        q_parts.append(q_j)
                    q_all = (
                        q_parts[0] if sp.q == 1 else jnp.concatenate(q_parts, axis=0)
                    )
                    q_0 = q_all.astype(BF16).astype(dot_dt)
                    m_v[...] = jnp.full((qh_pad, 1), NEG, F32)
                    l_v[...] = jnp.zeros((qh_pad, 1), F32)
                    acc_v[...] = jnp.zeros((qh_pad, kv), F32)
                    # In step verify keys (pos .. pos + 2) are not in the cache yet.
                    # Patch each one into the VMEM page tile at its cache row so verify
                    # row j runs the exact plain-step page math at pos + j.
                    patches = []
                    if sp.q > 1:
                        r32_iota = jax.lax.broadcasted_iota(jnp.int32, (32, 1), 0)
                        r8_iota = jax.lax.broadcasted_iota(jnp.int32, (8, 1), 0)
                        for k_v in range(3):
                            r_k = 4 * b + k_v
                            nq_k = jnp.sum(
                                jnp.where(row_iota == r_k, new_q, 0.0),
                                axis=0,
                                keepdims=True,
                            )
                            ns_k = jnp.sum(
                                jnp.where(row_iota == r_k, new_s, 0.0),
                                axis=0,
                                keepdims=True,
                            )
                            o_kv, g_kv_idx, row_kv = kernel_cache_token_loc(
                                pos + k_v, ps, sp.w, g_kv
                            )
                            patches.append(
                                (
                                    rank == o_kv,
                                    g_kv_idx,
                                    row_kv,
                                    nq_k.astype(F8),
                                    ns_k[:, 0:1],
                                    ns_k[:, 1:2],
                                )
                            )

                    def patch_slot(grp: Any, s: Any) -> None:
                        for on_rank, g_k, row_k, nq8, s_hi_k, s_lo_k in patches:

                            @pl.when(on_rank & (grp == g_k))
                            def _(row_k=row_k, nq8=nq8, s_hi_k=s_hi_k, s_lo_k=s_lo_k):
                                w8 = pl.multiple_of((row_k // 32) * 8, 8)
                                blk = pltpu.bitcast(ringq[s, pl.ds(w8, 8), :], F8)
                                blk = jnp.where(r32_iota == row_k - 4 * w8, nq8, blk)
                                ringq[s, pl.ds(w8, 8), :] = pltpu.bitcast(
                                    blk, jnp.int32
                                )
                                at_k = c_iota == row_k
                                sc_t = rings[s]
                                sc_t = jnp.where(
                                    at_k & (r8_iota == 0),
                                    s_hi_k,
                                    jnp.where(at_k & (r8_iota == 1), s_lo_k, sc_t),
                                )
                                rings[s] = sc_t

                    @pl.when((valid_ref[b] != 0) & (pos + sp.q - 1 > 0))
                    def _():
                        n_grp = (pos + sp.q - 2) // (g_kv * sp.w * ps) + 1

                        def cp(grp: Any, s: Any) -> tuple[Any, Any]:
                            phys = kpt_ref[slot, grp]
                            return (
                                pltpu.make_async_copy(
                                    kvq_hbm.at[mi, phys], ringq.at[s], sems.at[4 + s]
                                ),
                                pltpu.make_async_copy(
                                    kvs_hbm.at[mi, phys],
                                    rings.at[s],
                                    sems.at[4 + KV_DEPTH + s],
                                ),
                            )

                        for t0 in range(KV_DEPTH):

                            @pl.when(t0 < n_grp)
                            def _(t0=t0):
                                for c in cp(t0, t0):
                                    c.start()

                        def tile_sc(s: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
                            tile = _kv_tile(ringq[s]).astype(fp8_dt)
                            return tile[:, :kv], tile[:, kv:], rings[s]

                        def qk_slot(s: Any) -> jax.Array:
                            hi, lo, sc = tile_sc(s)
                            return (
                                _nt(q_0, hi) * sc[0:1, :] + _nt(q_0, lo) * sc[1:2, :]
                            ) * sm_scale

                        for c in cp(0, 0):
                            c.wait()
                        patch_slot(0, 0)
                        scores0 = qk_slot(0)

                        def page_body(grp: Any, scores: jax.Array) -> jax.Array:
                            s = grp % KV_DEPTH
                            sn = (grp + 1) % KV_DEPTH

                            @pl.when(grp + 1 < n_grp)
                            def _():
                                for c in cp(grp + 1, sn):
                                    c.wait()
                                patch_slot(grp + 1, sn)

                            nxt = qk_slot(sn)
                            hi, lo, sc = tile_sc(s)
                            k_pos = (grp * (g_kv * sp.w * ps) + rank * ps) + base_k_pos
                            mask = k_pos < (pos + q_j_iota if sp.q > 1 else pos)
                            scores = jnp.where(mask, scores, NEG)
                            m_prev = m_v[...]
                            m_new = jnp.maximum(
                                m_prev, jnp.max(scores, axis=-1, keepdims=True)
                            )
                            alpha = jnp.exp(m_prev - m_new)
                            p = jnp.where(mask, jnp.exp(scores - m_new), 0.0)
                            l_new = l_v[...] * alpha + jnp.sum(
                                p, axis=-1, keepdims=True
                            )
                            p_hi = (
                                jnp.where(mask, p * sc[0:1, :], 0.0)
                                .astype(BF16)
                                .astype(dot_dt)
                            )
                            p_lo = (
                                jnp.where(mask, p * sc[1:2, :], 0.0)
                                .astype(BF16)
                                .astype(dot_dt)
                            )
                            pv = _nn(p_hi, hi) + _nn(p_lo, lo)
                            l_v[...] = l_new
                            acc_v[...] = acc_v[...] * alpha + pv
                            m_v[...] = m_new

                            @pl.when(grp + KV_DEPTH < n_grp)
                            def _():
                                for c in cp(grp + KV_DEPTH, s):
                                    c.start()

                            return nxt

                        jax.lax.fori_loop(0, n_grp, page_body, scores0)

                    stats = jnp.where(
                        stat_lane == 0,
                        m_v[...],
                        jnp.where(stat_lane == 1, l_v[...], 0.0),
                    )
                    pkt = jnp.concatenate([acc_v[...], stats], axis=1)
                    for j in range(sp.q):
                        pkt_j = pkt if sp.q == 1 else pkt[j * h_pad : (j + 1) * h_pad]
                        r_idx = b * sp.q + j
                        if hm == hm_pad:
                            for dst in range(sp.w):
                                mla_part[0, dst, r_idx] = pkt_j[
                                    dst * hm_pad : (dst + 1) * hm_pad
                                ]
                        else:
                            for dst in range(sp.w):
                                sb = jnp.zeros((hm_pad, kv + 128), F32)
                                for h in range(hm):
                                    rh = jnp.sum(
                                        jnp.where(
                                            h_all_iota == dst * hm + h, pkt_j, 0.0
                                        ),
                                        axis=0,
                                        keepdims=True,
                                    )
                                    sb = jnp.where(hm_iota == h, rh, sb)
                                mla_part[0, dst, r_idx] = sb
                    return carry

                jax.lax.fori_loop(0, sp.b_seq, row, 0)
                mla_part[1, rank] = mla_part[0, rank]
                if sp.w > 1:
                    p_copies = []
                    for off in range(1, sp.w):
                        dst = rank ^ off
                        cp_p = pltpu.make_async_remote_copy(
                            mla_part.at[0, dst],
                            mla_part.at[1, rank],
                            mla_send.at[1, off - 1],
                            mla_recv.at[1, off - 1],
                            device_id=(peer(dst),),
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                        cp_p.start()
                        p_copies.append(cp_p)
                    for cp_p in p_copies:
                        cp_p.wait_send()
                        cp_p.wait_recv()

                for b in range(b_rows):
                    acc_list = [mla_part[1, r, b, :, :kv] for r in range(sp.w)]
                    st_list = [mla_part[1, r, b, :, kv:] for r in range(sp.w)]
                    m_list = [st[:, 0:1] for st in st_list]
                    l_list = [st[:, 1:2] for st in st_list]
                    m_hist = m_list[0]
                    for r in range(1, sp.w):
                        m_hist = jnp.maximum(m_hist, m_list[r])
                    sc0 = jnp.exp(m_list[0] - m_hist)
                    l_hist = l_list[0] * sc0
                    acc_hist = acc_list[0] * sc0
                    for r in range(1, sp.w):
                        scr = jnp.exp(m_list[r] - m_hist)
                        l_hist = l_hist + l_list[r] * scr
                        acc_hist = acc_hist + acc_list[r] * scr
                    nq_b = jnp.sum(
                        jnp.where(row_iota == b, new_q, 0.0), axis=0, keepdims=True
                    )
                    ns_b = jnp.sum(
                        jnp.where(row_iota == b, new_s, 0.0), axis=0, keepdims=True
                    )
                    cur_hi = nq_b[:, :kv].astype(fp8_dt)
                    cur_lo = nq_b[:, kv:].astype(fp8_dt)
                    cur_sh = ns_b[:, 0:1]
                    cur_sl = ns_b[:, 1:2]
                    cur_tile = jnp.where(
                        c128_iota == 0,
                        cur_hi,
                        jnp.where(c128_iota == 1, cur_lo, jnp.zeros((128, kv), fp8_dt)),
                    )
                    qb = qabs_loc[b]
                    qb0 = qb.astype(BF16).astype(dot_dt)
                    ab_cur = _nt(qb0, cur_tile)
                    score_c = (
                        ab_cur[:, 0:1] * cur_sh + ab_cur[:, 1:2] * cur_sl
                    ) * sm_scale
                    m_f = jnp.maximum(m_hist, score_c)
                    alpha = jnp.exp(m_hist - m_f)
                    p_c = jnp.exp(score_c - m_f)
                    l_f = l_hist * alpha + p_c
                    p_hi_c = (p_c * cur_sh).astype(BF16).astype(dot_dt)
                    p_lo_c = (p_c * cur_sl).astype(BF16).astype(dot_dt)
                    p_tile = jnp.where(
                        l128_iota == 0,
                        p_hi_c,
                        jnp.where(
                            l128_iota == 1, p_lo_c, jnp.zeros((hm_pad, 128), dot_dt)
                        ),
                    )
                    acc_f = acc_hist * alpha + _nn(p_tile, cur_tile)
                    u_v[b] = jnp.where(
                        valid_ref[b // sp.q] != 0, acc_f / jnp.maximum(l_f, 1e-30), 0.0
                    )

                wq = pltpu.make_async_copy(newq_v, kvrow_q.at[mi], sems.at[2])
                ws = pltpu.make_async_copy(news_v, kvrow_s.at[mi], sems.at[3])
                wq.start()
                ws.start()
                wq.wait()
                ws.wait()
                outs_h = []
                for h in range(hm):
                    u_h = jnp.zeros((r_v, kv), F32)
                    for b in range(b_rows):
                        ub_h = jnp.sum(
                            jnp.where(hm_iota == h, u_v[b], 0.0), axis=0, keepdims=True
                        )
                        u_h = jnp.where(row_iota == b, ub_h, u_h)
                    outs_h.append(hdot_nt(u_h, wuv[h]).astype(BF16))
                return jnp.concatenate(outs_h, axis=1)

            with scope("mla_core"):
                attn_out = pl.run_scoped(
                    scoped,
                    pltpu.VMEM((hm, dqk, kv), BF16),
                    pltpu.VMEM((hm, dv, kv), BF16),
                    pltpu.VMEM((b_rows, hm_pad, kv), F32),
                    pltpu.VMEM((b_rows, hm_pad, kv), F32),
                    pltpu.VMEM((r_v, 2 * kv), F32),
                    pltpu.VMEM((r_v, 128), F32),
                    pltpu.VMEM((KV_DEPTH, kprow // 4, 2 * kv), jnp.int32),
                    pltpu.VMEM((KV_DEPTH, 8, kprow), F32),
                    pltpu.SemaphoreType.DMA((4 + 2 * KV_DEPTH,)),
                    pltpu.VMEM((qh_pad, kv), F32),
                    pltpu.VMEM((qh_pad, 1), F32),
                    pltpu.VMEM((qh_pad, 1), F32),
                )
            with scope("mla_o_proj"):
                return scoped_out(
                    (r_v, sp.h),
                    BF16,
                    lambda o: fp8_linear(
                        attn_out,
                        lambda a, n: w["mla_o_proj_q"].at[mi, pl.ds(a, n)],
                        w["mla_o_proj_s"].at[mi],
                        sp.h,
                        o,
                    ),
                )

        # ---------------- FFN ----------------
        def dense_body(x: jax.Array, di: Any) -> jax.Array:
            n = sp.i_dense
            gu = scoped_out(
                (r_v, 2 * n),
                BF16,
                lambda o: fp8_linear(
                    x,
                    lambda a, m: w["dense_gu_q"].at[di, pl.ds(a, m)],
                    w["dense_gu_s"].at[di],
                    2 * n,
                    o,
                ),
            )
            act = _clamped_swiglu(gu[:, :n], gu[:, n:], cfg.swiglu_limit)
            return scoped_out(
                (r_v, sp.h),
                BF16,
                lambda o: fp8_linear(
                    act,
                    lambda a, m: w["dense_down_q"].at[di, pl.ds(a, m)],
                    w["dense_down_s"].at[di],
                    sp.h,
                    o,
                ),
            )

        def quant_one_block_bf16(bx: jax.Array) -> tuple[jax.Array, jax.Array]:
            """reference._moe_fp8_gmm_jax LHS: per 512-block bf16 amax scale, fp8 codes
            in
            registers.
            """
            amax = jnp.max(jnp.abs(bx), axis=1, keepdims=True)
            sc = (amax.astype(F32) / FP8_MAX).astype(BF16)
            sc_inv = jnp.where(sc == 0, 0, (1.0 / sc.astype(F32))).astype(BF16)
            return _bmul(bx, sc_inv).astype(F8), sc

        def routed_experts_kmajor(
            x: jax.Array,
            ei: Any,
            top_ids: jax.Array,
            top_w: jax.Array,
            *,
            pfx: str = "moe",
            n_rows: int = b_rows,
        ) -> jax.Array:
            """TP routed experts on the vLLM layout (`sp.experts_kmajor`).

            gate_up [E, H, 2I/W] and down [E, I/W, H] are K-major. Each rank computes
            its I/W slice of every distinct
            active expert and contracts it against its down slice, which yields a
            partial sum over the full H; the
            MoE layer's all-reduce adds the partials. No activation all-gather is
            needed.
            """
            h, il = sp.h, sp.il
            depth = EXPERT_RING_DEPTH
            nslot = depth + 1
            blk_h = min(512, h)
            er = 8 if n_rows <= 8 else (16 if n_rows <= 16 else r_v)
            nd_local = _round_up(min(n_rows * sp.topk, cfg.n_routed_experts), 8)
            lane = jax.lax.broadcasted_iota(jnp.int32, (er, 128), 1)
            colk = [
                jnp.sum(jnp.where(lane == k, top_ids[:er], 0), axis=1, keepdims=True)
                for k in range(sp.topk)
            ]
            colw = [
                jnp.sum(jnp.where(lane == k, top_w[:er], 0.0), axis=1, keepdims=True)
                for k in range(sp.topk)
            ]

            def scoped(
                ids_v,
                ids_s,
                dist,
                seen_s,
                gring,
                dring,
                rsem,
                gus,
                dns,
                ssem,
                xq,
                xs,
                slot_out,
                isem,
            ):
                ids_v[...] = top_ids[:er]
                sync_copy(ids_v, ids_s, isem.at[0])

                def zero(e: Any, c: Any) -> Any:
                    seen_s[e] = 0
                    return c

                jax.lax.fori_loop(0, sp.e_pad, zero, 0, unroll=8)

                def mark(g: Any, c: Any) -> Any:
                    seen_s[ids_s[g % n_rows, g // n_rows]] = 1
                    return c

                def collect(e: Any, n: Any) -> Any:
                    hit = seen_s[e] == 1

                    @pl.when(hit)
                    def _():
                        dist[n] = e

                    return n + hit.astype(jnp.int32)

                with scope("rt_setup"):
                    jax.lax.fori_loop(
                        0, sp.topk * n_rows, mark, 0, unroll=min(8, sp.topk * n_rows)
                    )
                    n = jax.lax.fori_loop(
                        0,
                        cfg.n_routed_experts,
                        collect,
                        jnp.int32(0),
                        unroll=8 if cfg.n_routed_experts % 8 == 0 else 1,
                    )

                xb0 = x[:er].astype(BF16)
                for bi in range(h // blk_h):
                    q_bi, sc_bi = quant_one_block_bf16(
                        xb0[:, bi * blk_h : (bi + 1) * blk_h]
                    )
                    xq[:, bi * blk_h : (bi + 1) * blk_h] = q_bi
                    xs[bi] = jnp.broadcast_to(sc_bi, (er, 128))
                slot_out[...] = jnp.zeros(slot_out.shape, slot_out.dtype)

                def copies(t: Any) -> tuple[Any, ...]:
                    s_ = t % nslot
                    e = dist[t]
                    return (
                        pltpu.make_async_copy(
                            w[f"{pfx}_expert_gate_up_q"].at[ei, e],
                            gring.at[s_],
                            rsem.at[0, s_],
                        ),
                        pltpu.make_async_copy(
                            w[f"{pfx}_expert_down_q"].at[ei, e],
                            dring.at[s_],
                            rsem.at[1, s_],
                        ),
                        pltpu.make_async_copy(
                            w[f"{pfx}_expert_gate_up_s"].at[ei, e],
                            gus.at[s_],
                            ssem.at[0, s_],
                        ),
                        pltpu.make_async_copy(
                            w[f"{pfx}_expert_down_s"].at[ei, e],
                            dns.at[s_],
                            ssem.at[1, s_],
                        ),
                    )

                for t0 in range(depth):

                    @pl.when(t0 < n)
                    def _(t0=t0):
                        for c in copies(jnp.int32(t0)):
                            c.start()

                def body(t: Any, carry: Any) -> Any:
                    s_ = t % nslot
                    for c in copies(t):
                        c.wait()
                    gw = gus[s_].astype(BF16)  # [1, 2I/W]
                    acc = jnp.zeros((er, 2 * il), BF16)
                    for bi in range(h // blk_h):
                        wt = gring[s_, bi * blk_h : (bi + 1) * blk_h, :].astype(fp8_dt)
                        part = _nn(
                            xq[:, bi * blk_h : (bi + 1) * blk_h].astype(fp8_dt), wt
                        ).astype(BF16)
                        acc = _bfadd(acc, _bmul(_bmul(part, xs[bi][:, :1]), gw))
                    gate = jnp.clip(acc[:, :il], max=cfg.swiglu_limit)
                    up = jnp.clip(
                        acc[:, il:], min=-cfg.swiglu_limit, max=cfg.swiglu_limit
                    )
                    act = _bmul(_silu_bf16(gate), up)
                    q_a, sc_a = quant_one_block_bf16(act)
                    dw = dns[s_].astype(BF16)  # [1, H]
                    e = dist[t]
                    w_e = jnp.where(colk[0] == e, colw[0], 0.0)
                    for k in range(1, sp.topk):
                        w_e = jnp.where(colk[k] == e, colw[k], w_e)
                    for bo in range(h // blk_h):
                        wt = dring[s_, :, bo * blk_h : (bo + 1) * blk_h].astype(fp8_dt)
                        part = _nn(q_a.astype(fp8_dt), wt).astype(BF16)
                        y = _bmul(
                            _bmul(part, sc_a), dw[:, bo * blk_h : (bo + 1) * blk_h]
                        )
                        slot_out[:, bo * blk_h : (bo + 1) * blk_h] = (
                            slot_out[:, bo * blk_h : (bo + 1) * blk_h]
                            + y.astype(F32) * w_e
                        )

                    @pl.when(t + depth < n)
                    def _():
                        for c in copies(t + depth):
                            c.start()

                    return carry

                with scope("rt_kmajor"):
                    jax.lax.fori_loop(0, n, body, 0)
                routed = slot_out[...]
                if er < r_v:
                    routed = jnp.concatenate(
                        [routed, jnp.zeros((r_v - er, h), F32)], axis=0
                    )
                return routed

            return pl.run_scoped(
                scoped,
                pltpu.VMEM((er, 128), jnp.int32),
                pltpu.SMEM((er, 128), jnp.int32),
                pltpu.SMEM((nd_local,), jnp.int32),
                pltpu.SMEM((sp.e_pad,), jnp.int32),
                pltpu.VMEM((nslot, h, 2 * il), F8),
                pltpu.VMEM((nslot, il, h), F8),
                pltpu.SemaphoreType.DMA((2, nslot)),
                pltpu.VMEM((nslot, 1, 2 * il), F32),
                pltpu.VMEM((nslot, 1, h), F32),
                pltpu.SemaphoreType.DMA((2, nslot)),
                pltpu.VMEM((er, h), F8),
                pltpu.VMEM((h // blk_h, er, 128), BF16),
                pltpu.VMEM((er, h), F32),
                pltpu.SemaphoreType.DMA((1,)),
            )

        def routed_experts(
            x: jax.Array,
            ei: Any,
            top_ids: jax.Array,
            top_w: jax.Array,
            *,
            pfx: str = "moe",
            n_rows: int = b_rows,
        ) -> jax.Array:
            """TP routed experts: each rank streams its slice of every distinct active
            expert.
            """
            h, i_moe, il, hl = sp.h, sp.i_moe, sp.il, sp.hl
            depth = EXPERT_RING_DEPTH
            nslot = depth + 1
            blk_h = min(512, h)
            blk_i = min(512, i_moe)
            er = 8 if n_rows <= 8 else (16 if n_rows <= 16 else r_v)
            nd_local = _round_up(min(n_rows * sp.topk, cfg.n_routed_experts), 8)
            act_cap = sp.act_slots
            lane = jax.lax.broadcasted_iota(jnp.int32, (er, 128), 1)
            colk = [
                jnp.sum(jnp.where(lane == k, top_ids[:er], 0), axis=1, keepdims=True)
                for k in range(sp.topk)
            ]
            colw = [
                jnp.sum(jnp.where(lane == k, top_w[:er], 0.0), axis=1, keepdims=True)
                for k in range(sp.topk)
            ]

            asend, arecv = act_send, act_recv

            def scoped(
                ids_v,
                ids_s,
                dist,
                seen_s,
                ring,
                rsem,
                gus,
                dns,
                ssem,
                xq,
                xs,
                slot_out,
                isem,
            ):
                ids_v[...] = top_ids[:er]
                sync_copy(ids_v, ids_s, isem.at[0])

                def zero(e: Any, c: Any) -> Any:
                    seen_s[e] = 0
                    return c

                jax.lax.fori_loop(0, sp.e_pad, zero, 0, unroll=8)

                def mark(g: Any, c: Any) -> Any:
                    seen_s[ids_s[g % n_rows, g // n_rows]] = 1
                    return c

                # Ascending expert id order: the TPU accumulator sums each row's experts
                # in dist order, so dist must not depend on what the other rows route to
                # (batch invariance).
                def collect(e: Any, n: Any) -> Any:
                    hit = seen_s[e] == 1

                    @pl.when(hit)
                    def _():
                        dist[n] = e

                    return n + hit.astype(jnp.int32)

                with scope("rt_setup"):
                    jax.lax.fori_loop(
                        0, sp.topk * n_rows, mark, 0, unroll=min(8, sp.topk * n_rows)
                    )
                    n = jax.lax.fori_loop(
                        0,
                        cfg.n_routed_experts,
                        collect,
                        jnp.int32(0),
                        unroll=8 if cfg.n_routed_experts % 8 == 0 else 1,
                    )

                xb0 = x[:er].astype(BF16)
                for bi in range(h // blk_h):
                    q_bi, sc_bi = quant_one_block_bf16(
                        xb0[:, bi * blk_h : (bi + 1) * blk_h]
                    )
                    xq[:, bi * blk_h : (bi + 1) * blk_h] = q_bi
                    xs[bi] = jnp.broadcast_to(sc_bi, (er, 128))
                slot_out[...] = jnp.zeros(slot_out.shape, slot_out.dtype)

                def run_chunk(ci: Any, c_carry: Any) -> Any:
                    c_base = ci * act_cap
                    cn = jnp.minimum(n - c_base, act_cap)
                    total = 2 * cn

                    def copies(t: Any, down: bool) -> tuple[Any, Any]:
                        s = t % nslot
                        if down:
                            e = dist[c_base + t - cn]
                            return (
                                pltpu.make_async_copy(
                                    w[f"{pfx}_expert_down_q"].at[ei, e],
                                    ring.at[s, pl.ds(0, hl), pl.ds(0, i_moe)],
                                    rsem.at[s],
                                ),
                                pltpu.make_async_copy(
                                    w[f"{pfx}_expert_down_s"].at[ei, e],
                                    dns.at[s],
                                    ssem.at[s],
                                ),
                            )
                        e = dist[c_base + t]
                        return (
                            pltpu.make_async_copy(
                                w[f"{pfx}_expert_gate_up_q"].at[ei, e],
                                ring.at[s],
                                rsem.at[s],
                            ),
                            pltpu.make_async_copy(
                                w[f"{pfx}_expert_gate_up_s"].at[ei, e],
                                gus.at[s],
                                ssem.at[s],
                            ),
                        )

                    def issue(t: Any) -> None:
                        @pl.when(t < cn)
                        def _():
                            for c in copies(t, False):
                                c.start()

                        @pl.when(t >= cn)
                        def _():
                            for c in copies(t, True):
                                c.start()

                    for t0 in range(depth):

                        @pl.when(t0 < total)
                        def _(*, t0: int = t0) -> None:
                            issue(jnp.int32(t0))

                    def act_copy(t: Any, off: int) -> Any:
                        return pltpu.make_async_remote_copy(
                            act_buf.at[rank, t, pl.ds(0, er)],
                            act_buf.at[rank, t, pl.ds(0, er)],
                            asend.at[off - 1],
                            arecv.at[off - 1],
                            device_id=(peer(rank ^ off),),
                            device_id_type=pl.DeviceIdType.MESH,
                        )

                    def gather_wait(t: Any, carry: Any) -> Any:
                        for off in range(1, sp.w):
                            cp = act_copy(t, off)
                            cp.wait_send()
                            cp.wait_recv()
                        return carry

                    def body(t: Any, carry: Any) -> Any:
                        s = t % nslot
                        tile = ring.at[s]

                        @pl.when(t < cn)
                        def _():
                            for c in copies(t, False):
                                c.wait()
                            ws_b = gus[s].astype(BF16)
                            acc = jnp.zeros((er, 2 * il), BF16)
                            for bi in range(h // blk_h):
                                wt = tile[:, bi * blk_h : (bi + 1) * blk_h].astype(
                                    fp8_dt
                                )
                                part = _nt(
                                    xq[:, bi * blk_h : (bi + 1) * blk_h].astype(fp8_dt),
                                    wt,
                                ).astype(BF16)
                                acc = _bfadd(
                                    acc, _bmul(_bmul(part, xs[bi][:, :1]), ws_b)
                                )
                            gate = jnp.clip(acc[:, :il], max=cfg.swiglu_limit)
                            up = jnp.clip(
                                acc[:, il:], min=-cfg.swiglu_limit, max=cfg.swiglu_limit
                            )
                            act_buf[rank, t, :er] = _bmul(_silu_bf16(gate), up)
                            for off in range(1, sp.w):
                                act_copy(t, off).start()

                        @pl.when(t >= cn)
                        def _():
                            d = t - cn
                            for c in copies(t, True):
                                c.wait()
                            ws_b = dns[s].astype(BF16)
                            acc = jnp.zeros((er, hl), BF16)
                            if blk_i % il == 0:
                                r_per_blk = blk_i // il
                                for bi in range(i_moe // blk_i):
                                    bx = jnp.concatenate(
                                        [
                                            act_buf[r, d, :er]
                                            for r in range(
                                                bi * r_per_blk, (bi + 1) * r_per_blk
                                            )
                                        ],
                                        axis=1,
                                    )
                                    q_bi, sc_bi = quant_one_block_bf16(bx)
                                    wt = tile[
                                        pl.ds(0, hl), pl.ds(bi * blk_i, blk_i)
                                    ].astype(fp8_dt)
                                    part = _nt(q_bi.astype(fp8_dt), wt).astype(BF16)
                                    acc = _bfadd(acc, _bmul(_bmul(part, sc_bi), ws_b))
                            else:
                                full = jnp.concatenate(
                                    [act_buf[r, d, :er] for r in range(sp.w)], axis=1
                                )
                                for bi in range(i_moe // blk_i):
                                    bx = full[:, bi * blk_i : (bi + 1) * blk_i]
                                    q_bi, sc_bi = quant_one_block_bf16(bx)
                                    wt = tile[
                                        pl.ds(0, hl), pl.ds(bi * blk_i, blk_i)
                                    ].astype(fp8_dt)
                                    part = _nt(q_bi.astype(fp8_dt), wt).astype(BF16)
                                    acc = _bfadd(acc, _bmul(_bmul(part, sc_bi), ws_b))
                            e = dist[c_base + d]
                            if tpu:
                                w_e = jnp.where(colk[0] == e, colw[0], 0.0)
                                for k in range(1, sp.topk):
                                    w_e = jnp.where(colk[k] == e, colw[k], w_e)
                                slot_out[...] = slot_out[...] + acc.astype(F32) * w_e
                            else:
                                for k in range(sp.topk):
                                    slot_out[k] = jnp.where(
                                        colk[k] == e, acc, slot_out[k]
                                    )

                        @pl.when(t + depth < total)
                        def _():
                            issue(t + depth)

                        return carry

                    with scope("rt_gu"):
                        jax.lax.fori_loop(0, cn, body, 0)
                    if sp.w > 1:
                        with scope("rt_gather"):
                            jax.lax.fori_loop(0, cn, gather_wait, 0)
                    with scope("rt_dn"):
                        jax.lax.fori_loop(cn, total, body, 0)
                    return c_carry

                if nd_local <= act_cap:
                    run_chunk(jnp.int32(0), 0)
                else:
                    jax.lax.fori_loop(0, (n + act_cap - 1) // act_cap, run_chunk, 0)
                if tpu:
                    routed = slot_out[...]
                else:
                    routed = jnp.zeros((er, hl), F32)
                    for k in range(sp.topk):
                        routed = routed + slot_out[k].astype(F32) * colw[k]
                if er < r_v:
                    routed = jnp.concatenate(
                        [routed, jnp.zeros((r_v - er, hl), F32)], axis=0
                    )
                if sp.w == 1:
                    return routed
                tiled = jnp.concatenate([routed] * sp.w, axis=1)
                hl_iota = jax.lax.broadcasted_iota(jnp.int32, (r_v, h), 1) // hl
                return jnp.where(hl_iota == rank, tiled, 0.0)

            return pl.run_scoped(
                scoped,
                pltpu.VMEM((er, 128), jnp.int32),
                pltpu.SMEM((er, 128), jnp.int32),
                pltpu.SMEM((nd_local,), jnp.int32),
                pltpu.SMEM((sp.e_pad,), jnp.int32),
                pltpu.VMEM((nslot, 2 * il, h), F8),
                pltpu.SemaphoreType.DMA((nslot,)),
                pltpu.VMEM((nslot, 1, 2 * il), F32),
                pltpu.VMEM((nslot, 1, hl), F32),
                pltpu.SemaphoreType.DMA((nslot,)),
                pltpu.VMEM((er, h), F8),
                pltpu.VMEM((h // blk_h, er, 128), BF16),
                pltpu.VMEM((er, hl), F32)
                if tpu
                else pltpu.VMEM((sp.topk, er, hl), BF16),
                pltpu.SemaphoreType.DMA((1,)),
            )

        def moe_body(
            x: jax.Array, ei: Any, *, pfx: str = "moe", n_rows: int = b_rows
        ) -> jax.Array:
            n = sp.i_sh
            mr = 8 if n_rows <= 8 else (16 if n_rows <= 16 else r_v)
            xm = x[:mr]
            sh_scope = scope("moe_shared")
            sh_scope.__enter__()
            sh_gu = scoped_out(
                (mr, 2 * n),
                BF16,
                lambda o: fp8_linear(
                    xm,
                    lambda a, m: w[f"{pfx}_shared_gu_q"].at[ei, pl.ds(a, m)],
                    w[f"{pfx}_shared_gu_s"].at[ei],
                    2 * n,
                    o,
                ),
            )
            sh_act = _clamped_swiglu(sh_gu[:, :n], sh_gu[:, n:], cfg.swiglu_limit)
            sh_local = scoped_out(
                (mr, sp.h),
                BF16,
                lambda o: fp8_linear(
                    sh_act,
                    lambda a, m: w[f"{pfx}_shared_down_q"].at[ei, pl.ds(a, m)],
                    w[f"{pfx}_shared_down_s"].at[ei],
                    sp.h,
                    o,
                ),
            )
            if mr < r_v:
                sh_local = jnp.concatenate(
                    [sh_local, jnp.zeros((r_v - mr, sp.h), BF16)], axis=0
                )

            sh_scope.__exit__(None, None, None)
            rt_scope = scope("moe_router")
            rt_scope.__enter__()
            logits = scoped_out(
                (mr, sp.e_pad),
                BF16,
                lambda o: bf16_linear(
                    xm,
                    lambda a, m: w[f"{pfx}_router_w"].at[ei, pl.ds(a, m)],
                    sp.e_pad,
                    o,
                ),
            ).astype(F32)
            scores = jax.nn.sigmoid(logits)
            biased = scores + meta_row(_M_RBIAS, sp.e_pad)
            e_iota = jax.lax.broadcasted_iota(jnp.int32, (mr, sp.e_pad), 1)
            k_iota = jax.lax.broadcasted_iota(jnp.int32, (mr, 128), 1)
            m_row_iota = jax.lax.broadcasted_iota(jnp.int32, (mr, 1), 0)
            top_ids = jnp.full((mr, 128), -1, jnp.int32)
            top_w = jnp.zeros((mr, 128), F32)
            for j in range(sp.topk):
                win = jnp.argmax(biased, axis=1).astype(jnp.int32)[:, None]
                hit = e_iota == win
                wsel = jnp.sum(jnp.where(hit, scores, 0.0), axis=1, keepdims=True)
                biased = jnp.where(hit, -jnp.inf, biased)
                top_ids = jnp.where(k_iota == j, win, top_ids)
                top_w = jnp.where(k_iota == j, wsel, top_w)
            if cfg.norm_topk_prob:
                top_w = top_w / jnp.maximum(
                    jnp.sum(top_w[:, : sp.topk], axis=1, keepdims=True), 1e-20
                )
            top_w = (
                (top_w * jnp.float32(cfg.routed_scaling_factor))
                .astype(BF16)
                .astype(F32)
            )
            top_ids = jnp.where(m_row_iota < n_rows, top_ids, -1)

            rt_scope.__exit__(None, None, None)
            with scope("moe_routed"):
                routed_fn = (
                    routed_experts_kmajor if sp.experts_kmajor else routed_experts
                )
                routed = routed_fn(xm, ei, top_ids, top_w, pfx=pfx, n_rows=n_rows)
            return _bfadd(sh_local, routed.astype(BF16))

        # ---------------- embedding ----------------
        def embed_from(src_tok: Any, n_rows: int = b_rows) -> jax.Array:
            r16_iota = jax.lax.broadcasted_iota(jnp.int32, (16, 1), 0)

            def scoped(rowbuf: Any, sem: Any) -> jax.Array:
                emb = jnp.zeros((r_v, sp.h), BF16)
                for b in range(n_rows):
                    local = src_tok(b) - rank * sp.v_loc
                    ok = (local >= 0) & (local < sp.v_loc)
                    safe = jnp.clip(local, 0, sp.v_loc - 1)
                    base = (safe // 16) * 16
                    sync_copy(w["embed_tokens"].at[pl.ds(base, 16)], rowbuf, sem.at[0])
                    sel = jnp.sum(
                        jnp.where(
                            r16_iota == safe - base, rowbuf[...].astype(F32), 0.0
                        ),
                        axis=0,
                        keepdims=True,
                    )
                    sel = jnp.where(ok, sel, 0.0).astype(BF16)
                    emb = jnp.where(row_iota == b, sel, emb)
                return emb

            emb = pl.run_scoped(
                scoped, pltpu.VMEM((16, sp.h), BF16), pltpu.SemaphoreType.DMA((1,))
            )
            return all_reduce(emb, 1, n_rows)

        # ---------------- layer loop ----------------
        if sp.lk > 0:
            kda_prefetch_start(0, 0)
        with scope("embed"):
            emb = embed_from(lambda b: tok_ref[b], b_rows)
        for s in range(cfg.hc_mult):
            streams_v[s] = emb

        def layer(lid: Any) -> None:
            fn_copy(lid, 0).start()
            fn_copy(lid, 1).start()
            with scope("meta"):
                sync_copy(w["meta"].at[lid], meta_v, meta_sem.at[0])
            kind = ltab_ref[lid]
            sub = ltab_ref[sp.l + lid]
            fsub = ltab_ref[2 * sp.l + lid]
            with scope("mhc_collapse_a"):
                post_a, comb_a, norm_a = mhc_collapse(
                    (lid, 0), _M_HCA_BASE, _M_HCA_SCALE, _M_IN_LN
                )
            x_v[...] = norm_a

            def run_mla() -> None:
                with scope("mla"):
                    attn_v[...] = mla_body(x_v[...], sub)

            def run_kda() -> None:
                with scope("kda"):
                    attn_v[...] = kda_body(x_v[...], sub)

            jax.lax.cond(kind == 1, run_mla, run_kda)
            with scope("ar_attn"):
                attn_red = all_reduce(attn_v[...], 0)
            with scope("mhc_combine_a"):
                mhc_combine(attn_red, post_a, comb_a)
            with scope("mhc_collapse_f"):
                post_f, comb_f, norm_f = mhc_collapse(
                    (lid, 1), _M_HCF_BASE, _M_HCF_SCALE, _M_POST_LN
                )
            x_v[...] = norm_f

            def run_dense() -> None:
                with scope("dense"):
                    attn_v[...] = dense_body(x_v[...], fsub)

            def run_moe() -> None:
                with scope("moe"):
                    attn_v[...] = moe_body(x_v[...], fsub)

            jax.lax.cond(lid < sp.l_dense, run_dense, run_moe)
            part = attn_v[...]
            with scope("ar_ffn"):
                ffn_red = all_reduce(part, 1)
            with scope("mhc_combine_f"):
                mhc_combine(ffn_red, post_f, comb_f)

        def any_layer(lid: Any, c: Any) -> Any:
            layer(lid)  # one traced layer body; FFN kind chosen by lax.cond
            return c

        jax.lax.fori_loop(0, sp.l, any_layer, 0)

        # ---------------- final norm, lm head, top-k ----------------
        sync_copy(w["meta"].at[sp.l], meta_v, meta_sem.at[0])
        coll = streams_v[0].astype(F32)
        for s in range(1, cfg.hc_mult):
            coll = coll + streams_v[s].astype(F32)
        coll = (coll / cfg.hc_mult).astype(BF16)
        normed = _rms(coll, meta_row(0, sp.h), eps)
        with scope("lm_head"):
            bf16_linear(
                normed, lambda a, n: w["lm_head"].at[pl.ds(a, n)], sp.v_pad, logits_v
            )
        vc = _lane_chunk(sp.v_pad)
        nvc = sp.v_pad // vc
        vc_iota = jax.lax.broadcasted_iota(jnp.int32, (r_v, vc), 1)
        k_iota = jax.lax.broadcasted_iota(jnp.int32, (r_v, 128), 1)

        def mask_chunk(c: Any) -> jax.Array:
            off = pl.multiple_of(c * vc, vc)
            return jnp.where(vc_iota + off < sp.v_loc, logits_v[:, pl.ds(off, vc)], NEG)

        if return_logits:
            for c in range(nvc):
                logits_out[:, c * vc : (c + 1) * vc] = mask_chunk(c)

        def stats_body(c: Any, carry: Any) -> Any:
            m, ssum = carry
            x = mask_chunk(c)
            m_new = jnp.maximum(m, jnp.max(x, axis=1, keepdims=True))
            ssum = ssum * jnp.exp(m - m_new) + jnp.sum(
                jnp.exp(x - m_new), axis=1, keepdims=True
            )
            return m_new, ssum

        tk_scope = scope("topk")
        tk_scope.__enter__()
        m_loc, s_loc = jax.lax.fori_loop(
            0, nvc, stats_body, (jnp.full((r_v, 1), NEG, F32), jnp.zeros((r_v, 1), F32))
        )

        def topk_body(c: Any, carry: Any) -> Any:
            """Merge chunk c into running top-64 (values desc, ties by smaller id)."""
            vals, ids = carry
            off = c * vc
            work = mask_chunk(c)
            cand_v = jnp.concatenate([vals, jnp.full((r_v, 128), NEG, F32)], axis=1)[
                :, :128
            ]
            new_v = jnp.full((r_v, 128), NEG, F32)
            new_i = jnp.zeros((r_v, 128), F32)
            for j in range(TOPK_OUT):
                m_chunk = jnp.max(work, axis=1, keepdims=True)
                m_run = jnp.max(cand_v, axis=1, keepdims=True)
                id_chunk = jnp.min(
                    jnp.where(work == m_chunk, vc_iota, vc), axis=1, keepdims=True
                )
                id_run = jnp.min(
                    jnp.where(cand_v == m_run, ids, jnp.float32(3e38)),
                    axis=1,
                    keepdims=True,
                )
                gid_chunk = (id_chunk + off + rank * sp.v_loc).astype(F32)
                take_chunk = (m_chunk > m_run) | (
                    (m_chunk == m_run) & (gid_chunk < id_run)
                )
                val = jnp.where(take_chunk, m_chunk, m_run)
                gid = jnp.where(take_chunk, gid_chunk, id_run)
                work = jnp.where(take_chunk & (vc_iota == id_chunk), -jnp.inf, work)
                cand_v = jnp.where(
                    ~take_chunk & (cand_v == m_run) & (ids == id_run), -jnp.inf, cand_v
                )
                new_v = jnp.where(k_iota == j, val, new_v)
                new_i = jnp.where(k_iota == j, gid, new_i)
            return new_v, new_i

        def exact_topk() -> tuple[jax.Array, jax.Array]:
            return jax.lax.fori_loop(
                0,
                nvc,
                topk_body,
                (jnp.full((r_v, 128), NEG, F32), jnp.full((r_v, 128), 3e38, F32)),
            )

        sr = 8 if b_rows <= 8 else r_v
        l_iota = jax.lax.broadcasted_iota(jnp.int32, (sr, 128), 1)

        def load_block(j: Any) -> tuple[jax.Array, jax.Array]:
            off = pl.multiple_of(j * 128, 128)
            gl = l_iota + off
            return (
                jnp.where(gl < sp.v_loc, logits_v[:sr, pl.ds(off, 128)], NEG),
                (gl + rank * sp.v_loc).astype(F32),
            )

        fv, fi, unsafe = lane_topk(load_block, sp.v_pad // 128, sr, b_rows)
        n_unsafe = jnp.max(jnp.where(unsafe, 1, 0))

        def pad_rows(v: jax.Array, fill: float) -> jax.Array:
            if sr == r_v:
                return v
            return jnp.concatenate([v, jnp.full((r_v - sr, 128), fill, F32)], axis=0)

        vals, ids = jax.lax.cond(
            n_unsafe > 0, exact_topk, lambda: (pad_rows(fv, NEG), pad_rows(fi, 3e38))
        )
        stats = jnp.where(k_iota == 0, m_loc, jnp.where(k_iota == 1, s_loc, 0.0))
        packet = jnp.concatenate([vals[:, :TOPK_OUT], ids[:, :TOPK_OUT], stats], axis=1)
        gather_buf[rank] = packet
        copies = []
        for off in range(1, sp.w):
            cp = pltpu.make_async_remote_copy(
                gather_buf.at[rank],
                gather_buf.at[rank],
                gather_send.at[off - 1],
                gather_recv.at[off - 1],
                device_id=(peer(rank ^ off),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            cp.start()
            copies.append(cp)
        for cp in copies:
            cp.wait_send()
            cp.wait_recv()
        all_vals = jnp.concatenate(
            [gather_buf[r][:, :TOPK_OUT] for r in range(sp.w)], axis=1
        )
        all_ids = jnp.concatenate(
            [gather_buf[r][:, TOPK_OUT : 2 * TOPK_OUT] for r in range(sp.w)], axis=1
        )
        ms = [gather_buf[r][:, 128:129] for r in range(sp.w)]
        ss_ = [gather_buf[r][:, 129:130] for r in range(sp.w)]
        m_g = ms[0]
        for r in range(1, sp.w):
            m_g = jnp.maximum(m_g, ms[r])
        s_g = ss_[0] * jnp.exp(ms[0] - m_g)
        for r in range(1, sp.w):
            s_g = s_g + ss_[r] * jnp.exp(ms[r] - m_g)
        lse = m_g + jnp.log(s_g)
        out_lp = jnp.full((r_v, 128), NEG, F32)
        out_id = jnp.zeros((r_v, 128), jnp.int32)
        work = all_vals
        big = jnp.float32(3e38)
        for j in range(TOPK_OUT):
            mx = jnp.max(work, axis=1, keepdims=True)
            idc = jnp.min(jnp.where(work == mx, all_ids, big), axis=1, keepdims=True)
            take = (work == mx) & (all_ids == idc)
            work = jnp.where(take, -jnp.inf, work)
            out_lp = jnp.where(k_iota == j, mx - lse, out_lp)
            out_id = jnp.where(k_iota == j, idc.astype(jnp.int32), out_id)
        tk_scope.__exit__(None, None, None)
        top_lp_ref[...] = out_lp
        top_id_ref[...] = out_id
        if not sp.spec:
            next_ref[...] = jnp.broadcast_to(out_id[:, 0:1], next_ref.shape)
            return

        # ---------------- Speculative acceptance, KDA rollback, and k=3 MTP draft chain
        # ----------------
        streams_v[0] = normed

        def not_eos(tok_val: Any) -> Any:
            if ignore_eos or not cfg.eos_token_ids:
                return jnp.bool_(True)
            ok_e = jnp.bool_(True)
            for eid in cfg.eos_token_ids:
                ok_e = ok_e & (tok_val != int(eid))
            return ok_e

        def slice_hl(a: jax.Array) -> jax.Array:
            def _s(v: Any) -> jax.Array:
                v[...] = a
                return v[:, pl.ds(pl.multiple_of(rank * sp.hl, 128), sp.hl)]

            return pl.run_scoped(_s, pltpu.VMEM(a.shape, BF16))

        def mtp_eh_proj_fn(e_in: jax.Array, h_in: jax.Array, n_rows: int) -> jax.Array:
            er = 8 if n_rows <= 8 else (16 if n_rows <= 16 else r_v)
            e_n = _rms(e_in[:er], meta_row(_M_HCA_BASE, sp.h), eps)
            h_n = _rms(h_in[:er], meta_row(_M_HCA_SCALE, sp.h), eps)
            eh_loc = jnp.concatenate([slice_hl(e_n), slice_hl(h_n)], axis=1)
            part = scoped_out(
                (er, sp.h),
                BF16,
                lambda o: bf16_linear(
                    eh_loc, lambda a, n: w["mtp_eh_proj"].at[pl.ds(a, n)], sp.h, o
                ),
            )
            if er < r_v:
                part = jnp.concatenate(
                    [part, jnp.zeros((r_v - er, sp.h), BF16)], axis=0
                )
            return all_reduce(part, 0, n_rows)

        def run_spec_tail(
            t_id_v, t_id_s, acc_s, acc_v, mtp_lat_q, mtp_lat_s, d_qkva_v, d_sem
        ):
            r8_iota = jax.lax.broadcasted_iota(jnp.int32, (8, 1), 0)
            t_id_v[...] = out_id
            sync_copy(t_id_v, t_id_s, d_sem.at[0])
            acc_v[...] = jnp.zeros((ROWS, 128), jnp.int32)
            sync_copy(acc_v, acc_s, d_sem.at[0])
            for b in range(sp.b_seq):
                t0 = t_id_s[4 * b + 0, 0]
                t1 = t_id_s[4 * b + 1, 0]
                t2 = t_id_s[4 * b + 2, 0]
                t3 = t_id_s[4 * b + 3, 0]
                d1 = tok_ref[4 * b + 1]
                d2 = tok_ref[4 * b + 2]
                d3 = tok_ref[4 * b + 3]
                acc1 = (d1 == t0) & not_eos(t0)
                acc2 = acc1 & (d2 == t1) & not_eos(t1)
                acc3 = acc2 & (d3 == t2) & not_eos(t2)
                l_acc = (
                    acc1.astype(jnp.int32)
                    + acc2.astype(jnp.int32)
                    + acc3.astype(jnp.int32)
                )
                ok_b = valid_ref[b] != 0
                n_acc = jnp.where(ok_b, l_acc + 1, 0)
                s_tok = jnp.where(
                    l_acc == 3,
                    t3,
                    jnp.where(l_acc == 2, t2, jnp.where(l_acc == 1, t1, t0)),
                )
                acc_s[b, 0] = n_acc
                acc_s[b, 1] = l_acc
                acc_s[b, 2] = t0
                acc_s[b, 3] = t1
                acc_s[b, 4] = t2
                acc_s[b, 5] = t3
                acc_s[b, 6] = s_tok
                acc_s[b, 10] = pos_ref[b] + n_acc

            def rb_read(ki: Any, s: Any) -> list[Any]:
                rc = []
                s_off = s * 2 * sp.b_seq
                for b in range(sp.b_seq):
                    la = acc_s[b, 1]
                    c1 = pltpu.make_async_copy(
                        conv_ckpt.at[ki, b, la],
                        kda_conv_buf.at[s, b],
                        kda_sems.at[0, s_off + 2 * b],
                    )
                    c2 = pltpu.make_async_copy(
                        rec_ckpt.at[ki, b, la],
                        kda_rec_buf.at[s, b],
                        kda_sems.at[0, s_off + 2 * b + 1],
                    )
                    rc += [c1, c2]
                return rc

            def rb_write(ki: Any, s: Any) -> list[Any]:
                wc = []
                s_off = s * 2 * sp.b_seq
                for b in range(sp.b_seq):
                    c1 = pltpu.make_async_copy(
                        kda_conv_buf.at[s, b],
                        conv_new.at[b, ki],
                        kda_sems.at[1, s_off + 2 * b],
                    )
                    c2 = pltpu.make_async_copy(
                        kda_rec_buf.at[s, b],
                        rec_new.at[b, ki],
                        kda_sems.at[1, s_off + 2 * b + 1],
                    )
                    wc += [c1, c2]
                return wc

            def copy_kda_rollback(ki: Any, carry: Any) -> Any:
                s = ki % 2
                ns = (ki + 1) % 2
                for c in rb_read(ki, s):
                    c.wait()

                @pl.when(ki + 1 < sp.lk)
                def _():
                    for c in rb_read(ki + 1, ns):
                        c.start()

                @pl.when(ki > 0)
                def _():
                    for c in rb_write(ki - 1, ns):
                        c.wait()

                for c in rb_write(ki, s):
                    c.start()
                return carry

            if sp.lk > 0:
                for c in rb_read(0, 0):
                    c.start()
                jax.lax.fori_loop(0, sp.lk, copy_kda_rollback, 0)
                for c in rb_write(sp.lk - 1, (sp.lk - 1) % 2):
                    c.wait()

            sync_copy(w["meta"].at[sp.l + 1], meta_v, meta_sem.at[0])
            v_emb = embed_from(lambda r_idx: t_id_s[r_idx, 0], b_rows)
            for r_idx in range(b_rows):
                pos_r = pos_ref[r_idx // 4] + (r_idx % 4)
                v_emb = jnp.where((row_iota == r_idx) & (pos_r == 0), 0, v_emb)
            v_eh = mtp_eh_proj_fn(v_emb, streams_v[0], b_rows)
            v_norm_a = _rms(v_eh, meta_row(_M_IN_LN, sp.h), eps)
            v_qkv_a = scoped_out(
                (r_v, sp.ql + sp.kv),
                BF16,
                lambda o: fp8_linear(
                    v_norm_a,
                    lambda a, n: w["mtp_mla_a_proj_q"].at[0, pl.ds(a, n)],
                    w["mtp_mla_a_proj_s"].at[0],
                    sp.ql + sp.kv,
                    o,
                ),
            )
            v_lat = jnp.nan_to_num(
                _rms(v_qkv_a[:, sp.ql :], meta_row(_M_KVA_NORM, sp.kv), eps).astype(
                    F32
                ),
                nan=0.0,
                posinf=448.0,
                neginf=-448.0,
            )
            v_shi = jnp.maximum(
                jnp.max(jnp.abs(v_lat), axis=-1, keepdims=True) / 448.0,
                jnp.float32(2.0**-14),
            )
            v_qhi = jnp.clip(v_lat / v_shi, -448.0, 448.0).astype(F8).astype(F32)
            v_rem = v_lat - v_qhi * v_shi
            v_slo = jnp.maximum(
                jnp.max(jnp.abs(v_rem), axis=-1, keepdims=True) / 448.0,
                jnp.float32(2.0**-24),
            )
            v_qlo = jnp.clip(v_rem / v_slo, -448.0, 448.0).astype(F8).astype(F32)
            v_lane = jax.lax.broadcasted_iota(jnp.int32, (r_v, 128), 1)
            v_ns = jnp.where(v_lane == 0, v_shi, jnp.where(v_lane == 1, v_slo, 0.0))
            v_nq = jnp.concatenate([v_qhi, v_qlo], axis=1)

            def write_v_kv(vq_v: Any, vs_v: Any) -> None:
                vq_v[...] = v_nq
                vs_v[...] = v_ns
                sync_copy(vq_v, kvrow_q.at[sp.lm], d_sem.at[0])
                sync_copy(vs_v, kvrow_s.at[sp.lm], d_sem.at[1])

            pl.run_scoped(
                write_v_kv,
                pltpu.VMEM((r_v, 2 * sp.kv), F32),
                pltpu.VMEM((r_v, 128), F32),
            )
            for j in range(4):
                jq = jnp.zeros((8, 2 * sp.kv), F32)
                js = jnp.zeros((8, 128), F32)
                for b in range(sp.b_seq):
                    r_idx = 4 * b + j
                    rq = jnp.sum(
                        jnp.where(row_iota == r_idx, v_nq, 0.0), axis=0, keepdims=True
                    )
                    rs = jnp.sum(
                        jnp.where(row_iota == r_idx, v_ns, 0.0), axis=0, keepdims=True
                    )
                    jq = jnp.where(r8_iota == b, rq, jq)
                    js = jnp.where(r8_iota == b, rs, js)
                mtp_lat_q[j] = jq
                mtp_lat_s[j] = js
            mtp_lat_q[4] = jnp.zeros((8, 2 * sp.kv), F32)
            mtp_lat_s[4] = jnp.zeros((8, 128), F32)
            mtp_lat_q[5] = jnp.zeros((8, 2 * sp.kv), F32)
            mtp_lat_s[5] = jnp.zeros((8, 128), F32)

            eh_s0 = jnp.zeros((r_v, sp.h), BF16)
            qkva_s0 = jnp.zeros((8, sp.ql + sp.kv), BF16)
            veh_f = v_eh.astype(F32)
            vqkva_f = v_qkv_a.astype(F32)
            for b in range(sp.b_seq):
                r_acc = 4 * b + acc_s[b, 1]
                sel_eh = jnp.sum(
                    jnp.where(row_iota == r_acc, veh_f, 0.0), axis=0, keepdims=True
                ).astype(BF16)
                sel_qa = jnp.sum(
                    jnp.where(row_iota == r_acc, vqkva_f, 0.0), axis=0, keepdims=True
                ).astype(BF16)
                eh_s0 = jnp.where(row_iota == b, sel_eh, eh_s0)
                qkva_s0 = jnp.where(r8_iota == b, sel_qa, qkva_s0)
            streams_v[2] = eh_s0
            d_qkva_v[...] = qkva_s0

            def mtp_mla_body(step_s: Any) -> jax.Array:
                hm, dqk, dv, kv, ps, prow = sp.hm, sp.dqk, sp.dv, sp.kv, sp.ps, sp.prow
                g_kv, kprow, hm_pad, h_pad = sp.g_kv, sp.kprow, sp.hm_pad, sp.h_mla_pad
                qkv_a = d_qkva_v[...]
                q_lora = _rms(qkv_a[:, : sp.ql], meta_row(_M_QA_NORM, sp.ql), eps)
                q_nope = scoped_out(
                    (8, hm * dqk),
                    BF16,
                    lambda o: fp8_linear(
                        q_lora,
                        lambda a, n: w["mtp_mla_q_b_proj_q"].at[0, pl.ds(a, n)],
                        w["mtp_mla_q_b_proj_s"].at[0],
                        hm * dqk,
                        o,
                    ),
                )
                sm_scale = float(dqk**-0.5)

                def m_scoped(
                    wuk, wuv, qabs_loc, u_v, ringq, rings, sems, acc_v, m_v, l_v
                ) -> jax.Array:
                    sync_copy(w["mtp_mla_w_uk"].at[0], wuk, sems.at[0])
                    sync_copy(w["mtp_mla_w_uv"].at[0], wuv, sems.at[0])
                    ringq[...] = jnp.zeros(ringq.shape, jnp.int32)
                    rings[...] = jnp.zeros(rings.shape, F32)
                    hm_iota = jax.lax.broadcasted_iota(jnp.int32, (hm_pad, 1), 0)
                    h_all_iota = jax.lax.broadcasted_iota(jnp.int32, (h_pad, 1), 0)
                    stat_lane = jax.lax.broadcasted_iota(jnp.int32, (h_pad, 128), 1)
                    c_iota = jax.lax.broadcasted_iota(jnp.int32, (1, kprow), 1)
                    sub_g, sub_o = c_iota // prow, c_iota % prow
                    base_k_pos = jnp.where(
                        (sub_g < g_kv) & (sub_o < ps),
                        sub_g * (sp.w * ps) + sub_o,
                        jnp.int32(2**29),
                    )
                    q_by_h = [
                        _nn(
                            q_nope[:, h * dqk : (h + 1) * dqk].astype(dot_dt),
                            wuk[h].astype(dot_dt),
                        )
                        for h in range(hm)
                    ]
                    for b in range(sp.b_seq):
                        qb = jnp.zeros((hm_pad, kv), F32)
                        for h in range(hm):
                            row_b = jnp.sum(
                                jnp.where(r8_iota == b, q_by_h[h], 0.0),
                                axis=0,
                                keepdims=True,
                            )
                            qb = jnp.where(hm_iota == h, row_b, qb)
                        qabs_loc[b] = qb
                        mla_part[0, rank, b] = jnp.concatenate(
                            [qb, jnp.zeros((hm_pad, 128), F32)], axis=1
                        )

                    if sp.w > 1:
                        q_copies = []
                        for off in range(1, sp.w):
                            cp_q = pltpu.make_async_remote_copy(
                                mla_part.at[0, rank, pl.ds(0, sp.b_seq)],
                                mla_part.at[0, rank, pl.ds(0, sp.b_seq)],
                                mla_send.at[0, off - 1],
                                mla_recv.at[0, off - 1],
                                device_id=(peer(rank ^ off),),
                                device_id_type=pl.DeviceIdType.MESH,
                            )
                            cp_q.start()
                            q_copies.append(cp_q)
                        for cp_q in q_copies:
                            cp_q.wait_send()
                            cp_q.wait_recv()

                    c128_i = jax.lax.broadcasted_iota(jnp.int32, (128, 1), 0)

                    def d_row(b: Any, carry: Any) -> Any:
                        pos = pos_ref[b]
                        slot = slot_ref[b]
                        l_acc = acc_s[b, 1]
                        q_pos = pos + l_acc + step_s
                        if hm == hm_pad:
                            q_all = jnp.concatenate(
                                [mla_part[0, r, b, :, :kv] for r in range(sp.w)], axis=0
                            )
                        else:
                            q_all = jnp.zeros((h_pad, kv), F32)
                            for r in range(sp.w):
                                part_r = mla_part[0, r, b, :, :kv]
                                for h in range(hm):
                                    rh = jnp.sum(
                                        jnp.where(hm_iota == h, part_r, 0.0),
                                        axis=0,
                                        keepdims=True,
                                    )
                                    q_all = jnp.where(
                                        h_all_iota == r * hm + h, rh, q_all
                                    )
                        q_0 = q_all.astype(BF16).astype(dot_dt)
                        m_v[...] = jnp.full((h_pad, 1), NEG, F32)
                        l_v[...] = jnp.zeros((h_pad, 1), F32)
                        acc_v[...] = jnp.zeros((h_pad, kv), F32)
                        cur_tile_b = jnp.zeros((128, kv), fp8_dt)
                        sh_b = []
                        sl_b = []
                        in_rank_kv = []
                        grp_kv_list = []
                        for k_v in range(5):
                            nq_k = jnp.sum(
                                jnp.where(r8_iota == b, mtp_lat_q[k_v], 0.0),
                                axis=0,
                                keepdims=True,
                            )
                            ns_k = jnp.sum(
                                jnp.where(r8_iota == b, mtp_lat_s[k_v], 0.0),
                                axis=0,
                                keepdims=True,
                            )
                            cur_tile_b = jnp.where(
                                c128_i == k_v,
                                nq_k[:, :kv].astype(fp8_dt),
                                jnp.where(
                                    c128_i == 6 + k_v,
                                    nq_k[:, kv:].astype(fp8_dt),
                                    cur_tile_b,
                                ),
                            )
                            sh_b.append(ns_k[:, 0:1])
                            sl_b.append(ns_k[:, 1:2])
                            pos_kv = (pos + k_v) if k_v < 4 else (pos + l_acc + 1)
                            is_hist_kv = (
                                ((k_v <= l_acc) & (pos_kv < q_pos))
                                if k_v < 4
                                else (step_s == 2)
                            )
                            o_kv, g_kv_idx, _ = kernel_cache_token_loc(
                                pos_kv, ps, sp.w, g_kv
                            )
                            in_rank_kv.append(is_hist_kv & (rank == o_kv))
                            grp_kv_list.append(g_kv_idx)
                        ab_cur_all = _nt(q_0, cur_tile_b)
                        sc_kv_list = [
                            (
                                ab_cur_all[:, k_v : k_v + 1] * sh_b[k_v]
                                + ab_cur_all[:, 6 + k_v : 7 + k_v] * sl_b[k_v]
                            )
                            * sm_scale
                            for k_v in range(5)
                        ]

                        @pl.when((valid_ref[b] != 0) & (q_pos > 0))
                        def _():
                            n_grp = (q_pos - 1) // (g_kv * sp.w * ps) + 1

                            def cp(grp: Any, s: Any) -> tuple[Any, Any]:
                                phys = kpt_ref[slot, grp]
                                return (
                                    pltpu.make_async_copy(
                                        kvq_hbm.at[sp.lm, phys],
                                        ringq.at[s],
                                        sems.at[4 + s],
                                    ),
                                    pltpu.make_async_copy(
                                        kvs_hbm.at[sp.lm, phys],
                                        rings.at[s],
                                        sems.at[4 + KV_DEPTH + s],
                                    ),
                                )

                            for t0 in range(KV_DEPTH):

                                @pl.when(t0 < n_grp)
                                def _(t0=t0):
                                    for c in cp(t0, t0):
                                        c.start()

                            def tile_sc(
                                s: Any,
                            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                                tile = _kv_tile(ringq[s]).astype(fp8_dt)
                                return tile[:, :kv], tile[:, kv:], rings[s]

                            def qk_slot(s: Any) -> jax.Array:
                                hi, lo, sc = tile_sc(s)
                                return (
                                    _nt(q_0, hi) * sc[0:1, :]
                                    + _nt(q_0, lo) * sc[1:2, :]
                                ) * sm_scale

                            for c in cp(0, 0):
                                c.wait()
                            scores0 = qk_slot(0)

                            def page_body(grp: Any, scores: jax.Array) -> jax.Array:
                                s = grp % KV_DEPTH
                                sn = (grp + 1) % KV_DEPTH

                                @pl.when(grp + 1 < n_grp)
                                def _():
                                    for c in cp(grp + 1, sn):
                                        c.wait()

                                nxt = qk_slot(sn)
                                hi, lo, sc = tile_sc(s)
                                k_pos = (
                                    grp * (g_kv * sp.w * ps) + rank * ps
                                ) + base_k_pos
                                mask = k_pos < pos
                                scores = jnp.where(mask, scores, NEG)
                                m_prev = m_v[...]
                                m_new = jnp.maximum(
                                    m_prev, jnp.max(scores, axis=-1, keepdims=True)
                                )
                                for k_v in range(5):
                                    act_kv = in_rank_kv[k_v] & (grp == grp_kv_list[k_v])
                                    m_new = jnp.maximum(
                                        m_new, jnp.where(act_kv, sc_kv_list[k_v], NEG)
                                    )
                                alpha = jnp.exp(m_prev - m_new)
                                p = jnp.where(mask, jnp.exp(scores - m_new), 0.0)
                                l_new = l_v[...] * alpha + jnp.sum(
                                    p, axis=-1, keepdims=True
                                )
                                p_hi = (
                                    jnp.where(mask, p * sc[0:1, :], 0.0)
                                    .astype(BF16)
                                    .astype(dot_dt)
                                )
                                p_lo = (
                                    jnp.where(mask, p * sc[1:2, :], 0.0)
                                    .astype(BF16)
                                    .astype(dot_dt)
                                )
                                pv = _nn(p_hi, hi) + _nn(p_lo, lo)
                                p_cur_g = jnp.zeros((h_pad, 128), dot_dt)
                                for k_v in range(5):
                                    act_kv = in_rank_kv[k_v] & (grp == grp_kv_list[k_v])
                                    p_kv = jnp.where(
                                        act_kv, jnp.exp(sc_kv_list[k_v] - m_new), 0.0
                                    )
                                    l_new = l_new + p_kv
                                    p_hi_kv = (
                                        jnp.where(act_kv, p_kv * sh_b[k_v], 0.0)
                                        .astype(BF16)
                                        .astype(dot_dt)
                                    )
                                    p_lo_kv = (
                                        jnp.where(act_kv, p_kv * sl_b[k_v], 0.0)
                                        .astype(BF16)
                                        .astype(dot_dt)
                                    )
                                    p_cur_g = jnp.where(
                                        stat_lane == k_v,
                                        p_hi_kv,
                                        jnp.where(
                                            stat_lane == 6 + k_v, p_lo_kv, p_cur_g
                                        ),
                                    )
                                pv = pv + _nn(p_cur_g, cur_tile_b)
                                l_v[...] = l_new
                                acc_v[...] = acc_v[...] * alpha + pv
                                m_v[...] = m_new

                                @pl.when(grp + KV_DEPTH < n_grp)
                                def _():
                                    for c in cp(grp + KV_DEPTH, s):
                                        c.start()

                                return nxt

                            jax.lax.fori_loop(0, n_grp, page_body, scores0)

                        stats = jnp.where(
                            stat_lane == 0,
                            m_v[...],
                            jnp.where(stat_lane == 1, l_v[...], 0.0),
                        )
                        pkt = jnp.concatenate([acc_v[...], stats], axis=1)
                        if hm == hm_pad:
                            for dst in range(sp.w):
                                mla_part[0, dst, b] = pkt[
                                    dst * hm_pad : (dst + 1) * hm_pad
                                ]
                        else:
                            for dst in range(sp.w):
                                sb = jnp.zeros((hm_pad, kv + 128), F32)
                                for h in range(hm):
                                    rh = jnp.sum(
                                        jnp.where(h_all_iota == dst * hm + h, pkt, 0.0),
                                        axis=0,
                                        keepdims=True,
                                    )
                                    sb = jnp.where(hm_iota == h, rh, sb)
                                mla_part[0, dst, b] = sb
                        return carry

                    jax.lax.fori_loop(0, sp.b_seq, d_row, 0)
                    mla_part[1, rank, : sp.b_seq] = mla_part[0, rank, : sp.b_seq]
                    if sp.w > 1:
                        p_copies = []
                        for off in range(1, sp.w):
                            dst = rank ^ off
                            cp_p = pltpu.make_async_remote_copy(
                                mla_part.at[0, dst, pl.ds(0, sp.b_seq)],
                                mla_part.at[1, rank, pl.ds(0, sp.b_seq)],
                                mla_send.at[1, off - 1],
                                mla_recv.at[1, off - 1],
                                device_id=(peer(dst),),
                                device_id_type=pl.DeviceIdType.MESH,
                            )
                            cp_p.start()
                            p_copies.append(cp_p)
                        for cp_p in p_copies:
                            cp_p.wait_send()
                            cp_p.wait_recv()

                    l128_iota = jax.lax.broadcasted_iota(jnp.int32, (hm_pad, 128), 1)
                    for b in range(sp.b_seq):
                        acc_list = [mla_part[1, r, b, :, :kv] for r in range(sp.w)]
                        st_list = [mla_part[1, r, b, :, kv:] for r in range(sp.w)]
                        m_list = [st[:, 0:1] for st in st_list]
                        l_list = [st[:, 1:2] for st in st_list]
                        m_hist = m_list[0]
                        for r in range(1, sp.w):
                            m_hist = jnp.maximum(m_hist, m_list[r])
                        sc0 = jnp.exp(m_list[0] - m_hist)
                        l_hist = l_list[0] * sc0
                        acc_hist = acc_list[0] * sc0
                        for r in range(1, sp.w):
                            scr = jnp.exp(m_list[r] - m_hist)
                            l_hist = l_hist + l_list[r] * scr
                            acc_hist = acc_hist + acc_list[r] * scr
                        l_acc_b = acc_s[b, 1]
                        cur_kv = jnp.where(
                            step_s == 0, l_acc_b, jnp.where(step_s == 1, 4, 5)
                        )
                        nq_b = jnp.zeros((1, 2 * kv), F32)
                        ns_b = jnp.zeros((1, 128), F32)
                        for k_v in range(6):
                            rq_k = jnp.sum(
                                jnp.where(r8_iota == b, mtp_lat_q[k_v], 0.0),
                                axis=0,
                                keepdims=True,
                            )
                            rs_k = jnp.sum(
                                jnp.where(r8_iota == b, mtp_lat_s[k_v], 0.0),
                                axis=0,
                                keepdims=True,
                            )
                            nq_b = jnp.where(cur_kv == k_v, rq_k, nq_b)
                            ns_b = jnp.where(cur_kv == k_v, rs_k, ns_b)
                        cur_hi = nq_b[:, :kv].astype(fp8_dt)
                        cur_lo = nq_b[:, kv:].astype(fp8_dt)
                        cur_sh = ns_b[:, 0:1]
                        cur_sl = ns_b[:, 1:2]
                        cur_tile = jnp.where(
                            c128_i == 0,
                            cur_hi,
                            jnp.where(
                                c128_i == 1, cur_lo, jnp.zeros((128, kv), fp8_dt)
                            ),
                        )
                        qb0 = qabs_loc[b].astype(BF16).astype(dot_dt)
                        ab_cur = _nt(qb0, cur_tile)
                        score_c = (
                            ab_cur[:, 0:1] * cur_sh + ab_cur[:, 1:2] * cur_sl
                        ) * sm_scale
                        m_f = jnp.maximum(m_hist, score_c)
                        alpha = jnp.exp(m_hist - m_f)
                        p_c = jnp.exp(score_c - m_f)
                        l_f = l_hist * alpha + p_c
                        p_hi_c = (p_c * cur_sh).astype(BF16).astype(dot_dt)
                        p_lo_c = (p_c * cur_sl).astype(BF16).astype(dot_dt)
                        p_tile = jnp.where(
                            l128_iota == 0,
                            p_hi_c,
                            jnp.where(
                                l128_iota == 1, p_lo_c, jnp.zeros((hm_pad, 128), dot_dt)
                            ),
                        )
                        acc_f = acc_hist * alpha + _nn(p_tile, cur_tile)
                        u_v[b] = jnp.where(
                            valid_ref[b] != 0, acc_f / jnp.maximum(l_f, 1e-30), 0.0
                        )

                    outs_h = []
                    for h in range(hm):
                        u_h = jnp.zeros((8, kv), F32)
                        for b in range(sp.b_seq):
                            ub_h = jnp.sum(
                                jnp.where(hm_iota == h, u_v[b], 0.0),
                                axis=0,
                                keepdims=True,
                            )
                            u_h = jnp.where(r8_iota == b, ub_h, u_h)
                        outs_h.append(hdot_nt(u_h, wuv[h]).astype(BF16))
                    return jnp.concatenate(outs_h, axis=1)

                attn_out = pl.run_scoped(
                    m_scoped,
                    pltpu.VMEM((hm, dqk, kv), BF16),
                    pltpu.VMEM((hm, dv, kv), BF16),
                    pltpu.VMEM((8, hm_pad, kv), F32),
                    pltpu.VMEM((8, hm_pad, kv), F32),
                    pltpu.VMEM((KV_DEPTH, kprow // 4, 2 * kv), jnp.int32),
                    pltpu.VMEM((KV_DEPTH, 8, kprow), F32),
                    pltpu.SemaphoreType.DMA((4 + 2 * KV_DEPTH,)),
                    pltpu.VMEM((h_pad, kv), F32),
                    pltpu.VMEM((h_pad, 1), F32),
                    pltpu.VMEM((h_pad, 1), F32),
                )
                proj_8 = scoped_out(
                    (8, sp.h),
                    BF16,
                    lambda o: fp8_linear(
                        attn_out,
                        lambda a, n: w["mtp_mla_o_proj_q"].at[0, pl.ds(a, n)],
                        w["mtp_mla_o_proj_s"].at[0],
                        sp.h,
                        o,
                    ),
                )
                return jnp.concatenate(
                    [proj_8, jnp.zeros((r_v - 8, sp.h), BF16)], axis=0
                )

            def draft_argmax(h_d: jax.Array, step_s: Any) -> None:
                sr_d = 8
                bf16_linear(
                    h_d[:sr_d],
                    lambda a, n: w["lm_head"].at[pl.ds(a, n)],
                    sp.v_pad,
                    logits_v.at[:sr_d],
                )
                l_d = jax.lax.broadcasted_iota(jnp.int32, (sr_d, 128), 1)

                def best_blk(j: Any, carry: Any) -> Any:
                    bv, bi = carry
                    off = pl.multiple_of(j * 128, 128)
                    gl = l_d + off
                    x = jnp.where(gl < sp.v_loc, logits_v[:sr_d, pl.ds(off, 128)], NEG)
                    g = (gl + rank * sp.v_loc).astype(F32)
                    take = x > bv
                    return jnp.where(take, x, bv), jnp.where(take, g, bi)

                bv, bi = jax.lax.fori_loop(
                    0,
                    sp.v_pad // 128,
                    best_blk,
                    (
                        jnp.full((sr_d, 128), -jnp.inf, F32),
                        jnp.full((sr_d, 128), 3e38, F32),
                    ),
                )
                m_d = jnp.max(bv, axis=1, keepdims=True)
                id_d = jnp.min(
                    jnp.where(bv == m_d, bi, jnp.float32(3e38)), axis=1, keepdims=True
                )
                k256 = jax.lax.broadcasted_iota(jnp.int32, (sr_d, 256), 1)
                gather_buf[rank, :sr_d] = jnp.where(
                    k256 == 0, m_d, jnp.where(k256 == 1, id_d, 0.0)
                )
                if sp.w > 1:
                    d_cps = []
                    for off in range(1, sp.w):
                        cp = pltpu.make_async_remote_copy(
                            gather_buf.at[rank, pl.ds(0, sr_d)],
                            gather_buf.at[rank, pl.ds(0, sr_d)],
                            gather_send.at[off - 1],
                            gather_recv.at[off - 1],
                            device_id=(peer(rank ^ off),),
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                        cp.start()
                        d_cps.append(cp)
                    for cp in d_cps:
                        cp.wait_send()
                        cp.wait_recv()
                m_g = gather_buf[0, :sr_d, 0:1]
                id_g = gather_buf[0, :sr_d, 1:2]
                for r in range(1, sp.w):
                    mr = gather_buf[r, :sr_d, 0:1]
                    idr = gather_buf[r, :sr_d, 1:2]
                    take = (mr > m_g) | ((mr == m_g) & (idr < id_g))
                    m_g = jnp.where(take, mr, m_g)
                    id_g = jnp.where(take, idr, id_g)
                if r_v > sr_d:
                    id_g = jnp.concatenate(
                        [id_g, jnp.zeros((r_v - sr_d, 1), F32)], axis=0
                    )
                t_id_v[...] = jnp.broadcast_to(id_g.astype(jnp.int32), (r_v, 128))
                sync_copy(t_id_v, t_id_s, d_sem.at[0])
                for b in range(sp.b_seq):
                    acc_s[b, 7 + step_s] = t_id_s[b, 0]

            def draft_step(step_s: Any, d_carry: Any) -> Any:
                @pl.when(step_s > 0)
                def _():
                    d_emb = embed_from(lambda b: acc_s[b, 6 + step_s], sp.b_seq)
                    for b in range(sp.b_seq):
                        pos_b = pos_ref[b] + acc_s[b, 1] + step_s
                        d_emb = jnp.where((row_iota == b) & (pos_b == 0), 0, d_emb)
                    streams_v[2] = mtp_eh_proj_fn(d_emb, streams_v[1], sp.b_seq)
                    norm_a_8 = _rms(streams_v[2][:8], meta_row(_M_IN_LN, sp.h), eps)
                    fp8_linear(
                        norm_a_8,
                        lambda a, n: w["mtp_mla_a_proj_q"].at[0, pl.ds(a, n)],
                        w["mtp_mla_a_proj_s"].at[0],
                        sp.ql + sp.kv,
                        d_qkva_v,
                    )
                    lat = jnp.nan_to_num(
                        _rms(
                            d_qkva_v[:, sp.ql :], meta_row(_M_KVA_NORM, sp.kv), eps
                        ).astype(F32),
                        nan=0.0,
                        posinf=448.0,
                        neginf=-448.0,
                    )
                    s_hi = jnp.maximum(
                        jnp.max(jnp.abs(lat), axis=-1, keepdims=True) / 448.0,
                        jnp.float32(2.0**-14),
                    )
                    q_hi = jnp.clip(lat / s_hi, -448.0, 448.0).astype(F8).astype(F32)
                    rem = lat - q_hi * s_hi
                    s_lo = jnp.maximum(
                        jnp.max(jnp.abs(rem), axis=-1, keepdims=True) / 448.0,
                        jnp.float32(2.0**-24),
                    )
                    q_lo = jnp.clip(rem / s_lo, -448.0, 448.0).astype(F8).astype(F32)
                    lane8 = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 1)
                    new_s = jnp.where(
                        lane8 == 0, s_hi, jnp.where(lane8 == 1, s_lo, 0.0)
                    )
                    new_q = jnp.concatenate([q_hi, q_lo], axis=1)
                    mtp_lat_q[3 + step_s] = new_q
                    mtp_lat_s[3 + step_s] = new_s

                attn_r = all_reduce(mtp_mla_body(step_s), 0, sp.b_seq)
                streams_v[2] = _bfadd(streams_v[2], attn_r)
                norm_f = _rms(streams_v[2], meta_row(_M_POST_LN, sp.h), eps)
                ffn_p = moe_body(norm_f, 0, pfx="mtp_moe", n_rows=sp.b_seq)
                ffn_r = all_reduce(ffn_p, 1, sp.b_seq)
                streams_v[2] = _bfadd(streams_v[2], ffn_r)
                streams_v[1] = _rms(streams_v[2], meta_row(_M_HCF_BASE, sp.h), eps)
                draft_argmax(streams_v[1], step_s)
                return d_carry

            with scope("mtp_draft"):
                jax.lax.fori_loop(0, 3, draft_step, 0)
            sync_copy(acc_s, acc_v, d_sem.at[0])
            next_ref[...] = acc_v[...]

        pl.run_scoped(
            run_spec_tail,
            pltpu.VMEM((r_v, 128), jnp.int32),
            pltpu.SMEM((r_v, 128), jnp.int32),
            pltpu.SMEM((ROWS, 128), jnp.int32),
            pltpu.VMEM((ROWS, 128), jnp.int32),
            pltpu.VMEM((6, 8, 2 * sp.kv), F32),
            pltpu.VMEM((6, 8, 128), F32),
            pltpu.VMEM((8, sp.ql + sp.kv), BF16),
            pltpu.SemaphoreType.DMA((2,)),
        )

    return kernel


def _writeback_kv_step(
    kvq: jax.Array,
    kvs: jax.Array,
    row_q: jax.Array,
    row_s: jax.Array,
    *,
    positions: jax.Array,
    slots: jax.Array,
    valid: jax.Array,
    kpt: jax.Array,
    rank: jax.Array,
    batch: int,
    n_q_per_seq: int,
    n_layers: int,
    page_size: int,
    world: int,
    g_kv: int,
) -> tuple[jax.Array, jax.Array]:
    """Pack and scatter `n_q_per_seq` new KV rows per sequence into `kvq` and `kvs`."""
    z = jnp.int32(0)
    total_rows = batch * n_q_per_seq
    row_b = jax.lax.bitcast_convert_type(
        row_q[:, :total_rows].astype(F8), jnp.uint8
    ).astype(jnp.uint32)
    col_iota = jnp.arange(128, dtype=jnp.int32)[None, None, None, :]
    for j in range(n_q_per_seq):
        pos_j = positions if n_q_per_seq == 1 else positions + j
        owner, grp, row_in_kpage = kernel_cache_token_loc(pos_j, page_size, world, g_kv)
        for b in range(batch):
            r_idx = b if n_q_per_seq == 1 else n_q_per_seq * b + j
            phys = kpt[slots[b], grp[b]]
            r_in = row_in_kpage[b]
            do_write = (valid[b] != 0) & (owner[b] == rank)
            idx = (z, phys, r_in // 4, z)
            cur = jax.lax.dynamic_slice(kvq, idx, (n_layers, 1, 1, kvq.shape[-1]))
            sh = (8 * (r_in % 4)).astype(jnp.uint32)
            cu = jax.lax.bitcast_convert_type(cur, jnp.uint32)
            nb = row_b[:, r_idx].reshape(cu.shape) << sh
            nu = (cu & ~(jnp.uint32(0xFF) << sh)) | nb
            new_q_word = jnp.where(
                do_write, jax.lax.bitcast_convert_type(nu, jnp.int32), cur
            )
            kvq = jax.lax.dynamic_update_slice(kvq, new_q_word, idx)
            col_blk = (r_in // 128) * 128
            col_off = r_in % 128
            sidx = (z, phys, z, col_blk)
            cs = jax.lax.dynamic_slice(kvs, sidx, (n_layers, 1, 8, 128))
            s_col = row_s[:, r_idx, :8][:, None, :, None]
            kvs = jax.lax.dynamic_update_slice(
                kvs, jnp.where(do_write & (col_iota == col_off), s_col, cs), sidx
            )
    return kvq, kvs


def make_decode_step(
    cfg: Glm53MegakernelConfig,
    mesh: jax.sharding.Mesh,
    *,
    batch: int,
    max_batch: int,
    n_pages: int,
    interpret: bool | pltpu.InterpretParams = False,
    return_logits: bool = False,
    profile_scopes: bool = False,
    experts_kmajor: bool = False,
    keep_rank_axis: bool = False,
    side_effects: bool = True,
) -> Any:
    """Build the jitted single-dispatch decode step.

    Signature: step(kweights, tokens[B], positions[B], slots[B], valid[B], conv, rec,
    kv_q, kv_s)
    -> (top_logprobs [B, 64], top_ids [B, 64], next_token [B], conv, rec, kv_q, kv_s[,
    logits]).
    kweights / states carry a leading P("tp") rank axis; caches are updated in place
    (aliased).
    `keep_rank_axis` keeps the rank axis on the token outputs too (every rank holds the
    same values), so a caller
    running one process per rank (vLLM workers via `pallas.jax_op`) reads its own copy
    with no cross rank gather.
    `side_effects=False` drops the custom call's side effect flag: torch_tpu
    re-partitions the exported module and
    XLA's SPMD partitioner rejects a side effecting custom call without sharding. Every
    output is consumed there, so
    nothing can be eliminated.
    """
    world = int(mesh.size)
    sp = _Spec(cfg, world, batch, max_batch, n_pages, experts_kmajor=experts_kmajor)
    kern = _build_kernel(
        sp,
        fp8_native=not bool(interpret),
        return_logits=return_logits,
        profile_scopes=profile_scopes,
    )
    ltab = jnp.asarray(layer_tables(cfg))
    vmem = pltpu.VMEM
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    scratch = [
        vmem((sp.meta_rows, sp.meta_w), F32),
        pltpu.SemaphoreType.DMA((2,)),
        vmem((cfg.hc_mult, ROWS, sp.h), BF16),
        vmem((ROWS, sp.h), BF16),
        vmem((ROWS, sp.h), BF16),
        vmem((2, world, sp.ar_rows, 128), BF16),
        pltpu.SemaphoreType.DMA((2, max(world - 1, 1))),
        pltpu.SemaphoreType.DMA((2, max(world - 1, 1))),
        vmem((world, ROWS, 256), F32),
        pltpu.SemaphoreType.DMA((max(world - 1, 1),)),
        pltpu.SemaphoreType.DMA((max(world - 1, 1),)),
        vmem((ROWS, sp.v_pad), F32),
        vmem((2, sp.mix_pad, cfg.hc_mult * sp.h), BF16),
        pltpu.SemaphoreType.DMA((2,)),
        *(
            [None, None, None]
            if sp.experts_kmajor
            else [
                vmem((world, sp.nd_pad, sp.sr, sp.il), BF16),
                pltpu.SemaphoreType.DMA((max(world - 1, 1),)),
                pltpu.SemaphoreType.DMA((max(world - 1, 1),)),
            ]
        ),
        vmem((2, world, sp.b, sp.hm_pad, sp.kv + 128), F32),
        pltpu.SemaphoreType.DMA((2, max(world - 1, 1))),
        pltpu.SemaphoreType.DMA((2, max(world - 1, 1))),
        vmem((2, ROWS, cfg.kda_conv_kernel_size - 1, 3 * sp.ik), BF16),
        vmem((2, sp.b, sp.hk, sp.dk, sp.dk), F32),
        pltpu.SemaphoreType.DMA((2, 2 * sp.b)),
    ]

    def local(
        kw: dict[str, jax.Array],
        tokens: Any,
        positions: Any,
        slots: Any,
        valid: Any,
        conv: Any,
        rec: Any,
        kvq: Any,
        kvs: Any,
        kpt: Any,
        rank_arr: Any,
        dev_arr: Any,
    ) -> tuple[Any, ...]:
        kw = {k: v[0] for k, v in kw.items()}
        conv, rec, kvq, kvs = conv[0], rec[0], kvq[0], kvs[0]
        kvq_orig_shape, kvs_orig_shape = kvq.shape, kvs.shape
        if kvq.ndim == 5:
            kvq = kvq.reshape(kvq.shape[0], -1, kvq.shape[-2], kvq.shape[-1])
            kvs = kvs.reshape(kvs.shape[0], -1, kvs.shape[-2], kvs.shape[-1])
        out_shape = [
            jax.ShapeDtypeStruct((ROWS, 128), F32),
            jax.ShapeDtypeStruct((ROWS, 128), jnp.int32),
            jax.ShapeDtypeStruct((ROWS, 128), jnp.int32),
            jax.ShapeDtypeStruct(
                (batch,) + conv.shape[:1] + conv.shape[2:], conv.dtype
            ),  # [B, Lk, ...]
            jax.ShapeDtypeStruct((batch,) + rec.shape[:1] + rec.shape[2:], rec.dtype),
            jax.ShapeDtypeStruct((sp.lm, ROWS, kvq.shape[-1]), F32),
            jax.ShapeDtypeStruct((sp.lm, ROWS, 128), kvs.dtype),
        ]
        out_specs = [vmem_spec] * 3 + [any_spec] * 4
        if return_logits:
            out_shape.append(jax.ShapeDtypeStruct((ROWS, sp.v_pad), F32))
            out_specs.append(vmem_spec)
        args = [kw[n] for n in KERNEL_WEIGHT_NAMES] + [conv, rec, kvq, kvs]
        call = pl.pallas_call(
            kern,
            out_shape=tuple(out_shape),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=N_PREFETCH,
                grid=(),
                in_specs=[any_spec] * len(args),
                out_specs=tuple(out_specs),
                scratch_shapes=scratch,
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=COLLECTIVE_ID,
                vmem_limit_bytes=60 * 2**20,
                disable_bounds_checks=True,
                has_side_effects=side_effects,
            ),
            interpret=(
                interpret
                if isinstance(interpret, pltpu.InterpretParams)
                else (pltpu.InterpretParams() if interpret else False)
            ),
            name="glm53_decode_megakernel",
        )
        res = call(tokens, positions, slots, valid, ltab, kpt, rank_arr, dev_arr, *args)
        top_lp, top_id, nxt = (
            res[0][:batch, :TOPK_OUT],
            res[1][:batch, :TOPK_OUT],
            res[2][:batch, 0],
        )
        conv_n, rec_n, row_q, row_s = res[3:7]
        # Write back only what changed. The kernel returns unchanged KDA states for
        # invalid rows; their KV write is a no-op (current values written back). Slots
        # of the batch are distinct.
        conv = conv.at[:, slots].set(jnp.swapaxes(conv_n, 0, 1), unique_indices=True)
        rec = rec.at[:, slots].set(jnp.swapaxes(rec_n, 0, 1), unique_indices=True)
        kvq, kvs = _writeback_kv_step(
            kvq,
            kvs,
            row_q,
            row_s,
            positions=positions,
            slots=slots,
            valid=valid,
            kpt=kpt,
            rank=rank_arr[0],
            batch=batch,
            n_q_per_seq=1,
            n_layers=sp.lm,
            page_size=cfg.page_size,
            world=world,
            g_kv=sp.g_kv,
        )
        if len(kvq_orig_shape) == 5:
            kvq = kvq.reshape(kvq_orig_shape)
            kvs = kvs.reshape(kvs_orig_shape)
        outs = (top_lp[None], top_id[None], nxt[None]) + tuple(
            r[None] for r in (conv, rec, kvq, kvs)
        )
        if return_logits:
            outs = outs + (res[7][None, :batch, : sp.v_loc],)
        return outs

    tp = jax.sharding.PartitionSpec("tp")
    rep = jax.sharding.PartitionSpec()
    w_specs = {k: tp for k in KERNEL_WEIGHT_NAMES}
    out_specs = (tp,) * (7 + int(return_logits))
    fn = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(w_specs, rep, rep, rep, rep, tp, tp, tp, tp, rep, tp, rep),
        out_specs=out_specs,
        check_vma=False,
    )

    @functools.partial(jax.jit, donate_argnums=(5, 6, 7, 8))
    def step(
        kw: Any,
        tokens: Any,
        positions: Any,
        slots: Any,
        valid: Any,
        conv: Any,
        rec: Any,
        kvq: Any,
        kvs: Any,
        kpt: Any = None,
        rank: Any = None,
        dev: Any = None,
    ) -> tuple[Any, ...]:
        if kpt is None:
            kpt = jnp.arange(sp.bmax * sp.n_kpages, dtype=jnp.int32).reshape(
                sp.bmax, sp.n_kpages
            )
        if rank is None:
            rank = jnp.arange(world, dtype=jnp.int32)
        if dev is None:
            dev = jnp.arange(world, dtype=jnp.int32)
        kw_sub = {k: kw[k] for k in KERNEL_WEIGHT_NAMES}
        res = fn(
            kw_sub, tokens, positions, slots, valid, conv, rec, kvq, kvs, kpt, rank, dev
        )
        if keep_rank_axis:
            return tuple(res)
        head = (res[0][0], res[1][0], res[2][0])
        return head + tuple(res[3:])

    return step


def make_spec_decode_step(
    cfg: Glm53MegakernelConfig,
    mesh: jax.sharding.Mesh,
    *,
    batch: int,
    max_batch: int,
    n_pages: int,
    interpret: bool | pltpu.InterpretParams = False,
    return_logits: bool = False,
    ignore_eos: bool = False,
    profile_scopes: bool = False,
    experts_kmajor: bool = False,
    keep_rank_axis: bool = False,
    side_effects: bool = True,
) -> Any:
    """Build the jitted single-dispatch speculative decode step (q=4 verify + rollback +
    k=3 MTP draft).

    Signature: spec_step(kweights, verify_tokens[B, 4], positions[B], slots[B],
    valid[B], conv, rec, kv_q, kv_s, kpt)
    -> (top_lp [B, 4, 64], top_id [B, 4, 64], target_toks [B, 4], num_accepted [B],
    next_verify_tokens [B, 4],
        next_positions [B], conv, rec, kv_q, kv_s[, verify_logits]).
    """
    world = int(mesh.size)
    sp = _Spec(
        cfg,
        world,
        batch,
        max_batch,
        n_pages,
        speculative=True,
        experts_kmajor=experts_kmajor,
    )
    kern = _build_kernel(
        sp,
        fp8_native=not bool(interpret),
        return_logits=return_logits,
        profile_scopes=profile_scopes,
        ignore_eos=ignore_eos,
    )
    ltab = jnp.asarray(layer_tables(cfg))
    spec_names = KERNEL_WEIGHT_NAMES + KERNEL_MTP_WEIGHT_NAMES
    r_v = sp.rows
    vmem = pltpu.VMEM
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    scratch = [
        vmem((sp.meta_rows, sp.meta_w), F32),
        pltpu.SemaphoreType.DMA((2,)),
        vmem((cfg.hc_mult, r_v, sp.h), BF16),
        vmem((r_v, sp.h), BF16),
        vmem((r_v, sp.h), BF16),
        vmem((2, world, sp.ar_rows, 128), BF16),
        pltpu.SemaphoreType.DMA((2, max(world - 1, 1))),
        pltpu.SemaphoreType.DMA((2, max(world - 1, 1))),
        vmem((world, r_v, 256), F32),
        pltpu.SemaphoreType.DMA((max(world - 1, 1),)),
        pltpu.SemaphoreType.DMA((max(world - 1, 1),)),
        vmem((r_v, sp.v_pad), F32),
        vmem((2, sp.mix_pad, cfg.hc_mult * sp.h), BF16),
        pltpu.SemaphoreType.DMA((2,)),
        *(
            [None, None, None]
            if sp.experts_kmajor
            else [
                vmem((world, sp.act_slots, sp.sr, sp.il), BF16),
                pltpu.SemaphoreType.DMA((max(world - 1, 1),)),
                pltpu.SemaphoreType.DMA((max(world - 1, 1),)),
            ]
        ),
        vmem((2, world, sp.b, sp.hm_pad, sp.kv + 128), F32),
        pltpu.SemaphoreType.DMA((2, max(world - 1, 1))),
        pltpu.SemaphoreType.DMA((2, max(world - 1, 1))),
        vmem((2, ROWS, cfg.kda_conv_kernel_size - 1, 3 * sp.ik), BF16),
        vmem((2, sp.b_seq, sp.hk, sp.dk, sp.dk), F32),
        pltpu.SemaphoreType.DMA((2, 4 * sp.b_seq)),
    ]

    def local(
        kw: dict[str, jax.Array],
        verify_tokens: Any,
        positions: Any,
        slots: Any,
        valid: Any,
        conv: Any,
        rec: Any,
        kvq: Any,
        kvs: Any,
        kpt: Any,
        rank_arr: Any,
        dev_arr: Any,
    ) -> tuple[Any, ...]:
        kw = {k: v[0] for k, v in kw.items()}
        conv, rec, kvq, kvs = conv[0], rec[0], kvq[0], kvs[0]
        kvq_orig_shape, kvs_orig_shape = kvq.shape, kvs.shape
        if kvq.ndim == 5:
            kvq = kvq.reshape(kvq.shape[0], -1, kvq.shape[-2], kvq.shape[-1])
            kvs = kvs.reshape(kvs.shape[0], -1, kvs.shape[-2], kvs.shape[-1])
        lm_tot = sp.lm + 1
        out_shape = [
            jax.ShapeDtypeStruct((r_v, 128), F32),
            jax.ShapeDtypeStruct((r_v, 128), jnp.int32),
            jax.ShapeDtypeStruct((ROWS, 128), jnp.int32),
            jax.ShapeDtypeStruct(
                conv.shape[:1] + (batch, 4) + conv.shape[2:], conv.dtype
            ),
            jax.ShapeDtypeStruct(rec.shape[:1] + (batch, 4) + rec.shape[2:], rec.dtype),
            jax.ShapeDtypeStruct(
                (batch,) + conv.shape[:1] + conv.shape[2:], conv.dtype
            ),
            jax.ShapeDtypeStruct((batch,) + rec.shape[:1] + rec.shape[2:], rec.dtype),
            jax.ShapeDtypeStruct((lm_tot, r_v, kvq.shape[-1]), F32),
            jax.ShapeDtypeStruct((lm_tot, r_v, 128), kvs.dtype),
        ]
        out_specs = [vmem_spec] * 3 + [any_spec] * 6
        if return_logits:
            out_shape.append(jax.ShapeDtypeStruct((r_v, sp.v_pad), F32))
            out_specs.append(vmem_spec)
        args = [kw[n] for n in spec_names] + [conv, rec, kvq, kvs]
        call = pl.pallas_call(
            kern,
            out_shape=tuple(out_shape),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=N_PREFETCH,
                grid=(),
                in_specs=[any_spec] * len(args),
                out_specs=tuple(out_specs),
                scratch_shapes=scratch,
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=COLLECTIVE_ID,
                vmem_limit_bytes=60 * 2**20,
                disable_bounds_checks=True,
                has_side_effects=side_effects,
            ),
            interpret=(
                interpret
                if isinstance(interpret, pltpu.InterpretParams)
                else (pltpu.InterpretParams() if interpret else False)
            ),
            name="glm53_spec_decode_megakernel",
        )
        tok_flat = verify_tokens.reshape(-1)
        res = call(
            tok_flat, positions, slots, valid, ltab, kpt, rank_arr, dev_arr, *args
        )
        top_lp = res[0][: sp.b, :TOPK_OUT].reshape(batch, 4, TOPK_OUT)
        top_id = res[1][: sp.b, :TOPK_OUT].reshape(batch, 4, TOPK_OUT)
        spec_meta = res[2][:batch]
        num_acc = spec_meta[:, 0]
        target_toks = spec_meta[:, 2:6]
        next_verify_tokens = spec_meta[:, 6:10]
        next_positions = spec_meta[:, 10]
        conv_n, rec_n, row_q, row_s = res[5:9]
        if sp.lk > 0:
            conv = conv.at[:, slots].set(
                jnp.swapaxes(conv_n, 0, 1), unique_indices=True
            )
            rec = rec.at[:, slots].set(jnp.swapaxes(rec_n, 0, 1), unique_indices=True)
        kvq, kvs = _writeback_kv_step(
            kvq,
            kvs,
            row_q,
            row_s,
            positions=positions,
            slots=slots,
            valid=valid,
            kpt=kpt,
            rank=rank_arr[0],
            batch=batch,
            n_q_per_seq=4,
            n_layers=lm_tot,
            page_size=cfg.page_size,
            world=world,
            g_kv=sp.g_kv,
        )
        if len(kvq_orig_shape) == 5:
            kvq = kvq.reshape(kvq_orig_shape)
            kvs = kvs.reshape(kvs_orig_shape)
        outs = (
            top_lp[None],
            top_id[None],
            target_toks[None],
            num_acc[None],
            next_verify_tokens[None],
            next_positions[None],
        ) + tuple(r[None] for r in (conv, rec, kvq, kvs))
        if return_logits:
            outs = outs + (
                res[9][None, : sp.b, : sp.v_loc].reshape(1, batch, 4, sp.v_loc),
            )
        return outs

    tp = jax.sharding.PartitionSpec("tp")
    rep = jax.sharding.PartitionSpec()
    w_specs = {k: tp for k in spec_names}
    out_specs = (tp,) * (10 + int(return_logits))
    fn = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(w_specs, rep, rep, rep, rep, tp, tp, tp, tp, rep, tp, rep),
        out_specs=out_specs,
        check_vma=False,
    )

    @functools.partial(jax.jit, donate_argnums=(5, 6, 7, 8))
    def spec_step(
        kw: Any,
        verify_tokens: Any,
        positions: Any,
        slots: Any,
        valid: Any,
        conv: Any,
        rec: Any,
        kvq: Any,
        kvs: Any,
        kpt: Any = None,
        rank: Any = None,
        dev: Any = None,
    ) -> tuple[Any, ...]:
        if kpt is None:
            kpt = jnp.arange(sp.bmax * sp.n_kpages, dtype=jnp.int32).reshape(
                sp.bmax, sp.n_kpages
            )
        if rank is None:
            rank = jnp.arange(world, dtype=jnp.int32)
        if dev is None:
            dev = jnp.arange(world, dtype=jnp.int32)
        kw_sub = {k: kw[k] for k in spec_names}
        res = fn(
            kw_sub,
            verify_tokens,
            positions,
            slots,
            valid,
            conv,
            rec,
            kvq,
            kvs,
            kpt,
            rank,
            dev,
        )
        if keep_rank_axis:
            return tuple(res)
        head = tuple(res[i][0] for i in range(6))
        return head + tuple(res[6:])

    return spec_step
