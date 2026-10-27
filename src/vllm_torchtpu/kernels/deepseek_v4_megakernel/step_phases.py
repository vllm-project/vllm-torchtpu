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
"""Interfaces of the per-layer phases of the DSv4 whole-step kernel (L4e).

One grid-less pallas_call per rank on the R-rank mesh runs every decoder layer
in a fori over layers. The step skeleton (``step_kernel.py``, owner: B6) runs
the dense chain on all T global rows (mHC, norms, every dense projection from
the cross-layer weight ring), both all-to-alls, the attention all-reduce and
the final collapse. Two phases are separate bodies with the contracts below:

* `owner_attention_phase` -- this rank's own requests only: KV / compressor /
  indexer writes, indexer scores + TC top-k (CSA), SWA / HCA / CSA attention.
* `moe_phase` -- routed experts (EP: this rank's local experts, served fp4 in
  place) on all T rows, added onto the shared-expert partial the skeleton
  leaves, then the fused all-reduce.

Both are called once per layer on EVERY rank (idle ranks included: an idle
owner has no active request and must return quickly; the MoE phase's
collectives must pair across ranks every layer). Neither may use the kernel
barrier semaphore (the skeleton's start barrier owns it).

Row conventions (all static shapes; padding rows are zero and inactive):
  T       global rows of the step (bucket 16 / 32 / 64), replicated activations.
  q       rows per request (1 + num_speculative_tokens; 7 for K=6 verify).
  B       this rank's request slots; own rows r * q + t (t < q).
  RP      cdiv(q * 128, 128) * 128: rows of one request's all-head block.

Layer-indexed HBM operands: the served KV caches and MoE weights are separate
arrays per layer, and the layer index is dynamic. `LayerRefs` issues a DMA
from the array of a traced layer index with one branch per layer; waits use
the first array (a wait only needs the transfer shape). Keep the number of
DMA *start* sites on layer-indexed arrays small: each costs ~n_layers copies
of one descriptor issue in the program.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

Ref = Any


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclasses.dataclass(frozen=True)
class StepGeometry:
    """Static shape of one compiled step (one bucket)."""

    R: int  # ranks on the mesh (16 two-host)
    T: int  # global rows (multiple of 16)
    q: int  # rows per request
    B: int  # this rank's request slots
    layer_kinds: tuple[str, ...]  # "csa" / "hca" / "swa" per decoder layer
    hidden: int = 7168
    hc_mult: int = 4
    n_heads: int = 128
    head_dim: int = 512
    rope_dim: int = 64
    q_lora: int = 1536
    o_groups: int = 16
    o_lora: int = 1024
    idx_heads: int = 64
    idx_head_dim: int = 128
    n_experts: int = 384
    topk: int = 6
    moe_inter: int = 3072
    num_hash_layers: int = 3
    # Max tokens per request this bucket serves: sizes the indexer score buffer
    # and the top-k scratch (round_up(max_ctx / 4, 16384) columns).
    max_ctx: int = 131072
    # VMEM bytes of the shared pool the phases may borrow (see PoolLease)
    lease_bytes: int = 0
    # query heads the attention phase attends per query row (q_in / out row
    # t * attn_heads + h); 128 = all (DP owner), 16 under group TP8
    attn_heads: int = 128
    # ranks per attention group, which split each group request's attention
    # (StepKernelConfig.attn_tp); group rank jj = me % attn_tp attends heads
    # [jj * attn_heads, (jj + 1) * attn_heads)
    attn_tp: int = 1
    # CSA indexer rows see (pos + 1) // 4 compressed rows, not pos // 4 + 1
    # (stock TPU_STREAMIDX_STRICT_CAUSAL; attn_csa.visible_rows)
    strict_causal: bool = False

    @property
    def heads_per_rank(self) -> int:
        return self.n_heads // self.R

    @property
    def RP(self) -> int:
        return cdiv(self.q * self.attn_heads, 128) * 128

    @property
    def own_rows(self) -> int:
        return self.B * self.q

    @property
    def n_layers(self) -> int:
        return len(self.layer_kinds)


@dataclasses.dataclass(frozen=True)
class PoolLease:
    """Bytes of the skeleton's VMEM weight pool lent to a phase for one call.

    ``pool`` is the u8 ``[rows, 128]`` pool ref; the phase owns rows
    ``[row0, row0 + rows)`` during the call and nothing else of it. No DMA is
    in flight into those rows when the phase starts, and the skeleton writes
    nothing there until the phase returns. Contents are garbage at entry and
    are not preserved. Carve typed views with ``vmem_alias.view(pool.at[
    pl.ds(row0 + off, n)], shape, dtype)`` (offsets in 128-byte rows; each view
    ``storage_size`` bytes). In interpret mode (typed scratch only) the
    skeleton passes ``pool=None`` and the phase falls back to its own
    dedicated scratch for the same buffers -- declare both through
    ``*_scratch_shapes(geom, leased)``.
    """

    pool: Ref | None
    row0: int
    rows: int


class LayerRefs:
    """Per-layer HBM arrays of one role, addressed by a traced layer index."""

    def __init__(self, refs: Sequence[Ref], layers: Sequence[int] | None = None):
        # layers[i]: decoder layer of refs[i] (default: refs[i] is layer i).
        self.refs = list(refs)
        self.layers = list(range(len(refs))) if layers is None else list(layers)

    def dispatch(self, layer, fn: Callable[[Ref], None]):
        """fn(refs[k]) for the entry whose layer id equals the traced `layer`,
        by binary search over the sorted ids (log2(n) branches per issue; a
        linear 61-way chain cost ~0.45 us per issue)."""
        cases = sorted(zip(self.layers, self.refs), key=lambda c: c[0])

        def rec(cs):
            if len(cs) == 1:
                lid, ref = cs[0]
                pl.when(layer == lid)(lambda: fn(ref))
                return
            mid = len(cs) // 2
            pivot = cs[mid][0]
            pl.when(layer < pivot)(lambda: rec(cs[:mid]))
            pl.when(layer >= pivot)(lambda: rec(cs[mid:]))

        rec(cases)

    def start(self, layer, src: Callable[[Ref], Ref], dst: Ref, sem: Ref):
        """Start ``copy(src(refs[layer]) -> dst)`` for a traced ``layer``."""
        self.dispatch(
            layer, lambda ref: pltpu.make_async_copy(src(ref), dst, sem).start()
        )

    def start_to(self, layer, src: Ref, dst: Callable[[Ref], Ref], sem: Ref):
        """Start ``copy(src -> dst(refs[layer]))`` (a cache write)."""
        self.dispatch(
            layer, lambda ref: pltpu.make_async_copy(src, dst(ref), sem).start()
        )

    def wait(self, src: Callable[[Ref], Ref], dst: Ref, sem: Ref):
        pltpu.make_async_copy(src(self.refs[0]), dst, sem).wait()

    def wait_to(self, src: Ref, dst: Callable[[Ref], Ref], sem: Ref):
        pltpu.make_async_copy(src, dst(self.refs[0]), sem).wait()


# ---------------------------------------------------------------------------
# Owner attention phase (B3 + B1 + kv_write; owner: magic-badger)
# ---------------------------------------------------------------------------
class AttnMeta(NamedTuple):
    """Metadata of this rank's requests (int32), fixed for the step: SMEM,
    except bt_flat (HBM; the phase stages what it reads)."""

    num_req: Ref  # [1]: active request slots (prefix 0 .. num_req-1)
    seq_lens: Ref  # [B]: tokens incl. this window (0 = inactive)
    positions: Ref  # [B * q]: position of own row r*q+t
    # All block-table groups concatenated (XLA, before the kernel). For role
    # (keys of AttnCaches.group / "swa_pages") and decoder layer l, request
    # r's table is bt_flat[off + r * w : off + (r + 1) * w] with
    # off = bt_off[role][l], w = bt_w[role][l] (both SMEM int32 [L]; 0 for
    # layers without the role).
    bt_flat: Ref
    bt_off: dict[str, Ref]
    bt_w: dict[str, Ref]
    # attn_core.swa_window_pages per request ([B * 3] per SWA group), stored
    # in bt_flat too under role "swa_pages" (w = 3).
    # Kind "swa" only (the DSpark draft): [B * 8] position of context row c of
    # request r at r * 8 + c, consecutive from c = 0, -1 past the last.
    ctx_positions: Ref | None = None


class AttnInputs(NamedTuple):
    """VMEM inputs the skeleton writes before the call (read-only here)."""

    # f32 [B, RP, 512] holding bf16 values (f32 so the all-to-all's 8-head
    # blocks are tile-aligned): row t*128 + h = query head h of own row t, after the
    # stock deepseek_v4_qnorm_rope (per-head RMSNorm without weight, eps
    # 1e-6, RoPE on the last 64 dims at the row's position). Zero rows for
    # inactive requests and for t >= q.
    q: Ref
    # bf16 [B, 8, 512]: kv_norm'd latent of own row t, after the stock
    # deepseek_v4_kv_rope (row t < q valid).
    kv: Ref
    # f32 [B, 8, W_c]: compressor projection x_attn @ fused_wkv_wgate for own
    # row t, columns [kv | gate] as served (CSA W_c = 2048: overlap halves,
    # HCA 1024). The ape bias is NOT added (the phase adds it, as
    # compressor_forward's epilogue).
    comp: Ref
    # CSA only (None for HCA layers):
    idx_q: Ref | None  # bf16 [B, 8, 64 * 128]: indexer wq_b output, pre-rope
    idx_w: Ref | None  # f32 [B, 8, 128]: weights_proj output in lanes 0..63
    idx_comp: Ref | None  # f32 [B, 8, 512]: indexer compressor projection
    # Kind "swa" only: bf16 [B, 8, 512]: the context rows' (the target's verify
    # rows') kv after kv_norm + kv_rope, row c at AttnMeta.ctx_positions.
    ctx_kv: Ref | None = None


class AttnCaches(NamedTuple):
    """Layer-indexed served caches (HBM, updated in place; aliased outputs).

    Roles as kernel_interface.LayerCaches (block tables via AttnMeta.bt_*).
    CSA-only roles hold only the CSA layers' arrays (LayerRefs.layers lists
    them). Interpret mode passes int32 views of the u8 arrays."""

    swa: LayerRefs
    kv: LayerRefs
    state: LayerRefs
    kv_rope: LayerRefs
    indexer_k: LayerRefs
    indexer_state: LayerRefs


class AttnWeights(NamedTuple):
    """Per-layer attention parameters (HBM, read-only)."""

    attn_sink: Ref  # f32 [L, 128]
    # f32 [L, 128, 1024]: row = position % ratio; HCA uses cols < 512 of rows
    # < 128 (its ape is [128, 512]), CSA cols < 1024 of rows < 4.
    comp_ape: Ref
    comp_norm: Ref  # f32 [L, 512]
    idx_comp_ape: Ref  # f32 [L_csa, 4, >= 256]: cols < 256 used (indexer ape [4, 256])
    idx_comp_norm: Ref  # f32 [L_csa, 128]
    # rotary tables as pack_cos_sin gives them: f32 [ceil(max_pos / 2), 128]
    cos_sin: Ref  # main rotary (q, kv, wo_a inverse)
    comp_cos_sin: Ref  # compress_rope_theta rotary
    idx_cos_sin: Ref  # indexer rotary


def owner_attention_phase(
    geom: StepGeometry,
    kind: str,  # "csa" | "hca", static (the skeleton branches on it)
    layer,  # traced int32: decoder layer index
    csa_ordinal,  # traced int32: index among CSA layers (CSA only)
    meta: AttnMeta,
    inputs: AttnInputs,
    caches: AttnCaches,
    weights: AttnWeights,
    out: Ref,  # VMEM f32 [B, RP, 512]: attention output (bf16-rounded), row t*128 + h
    lease: PoolLease,
    scratch: Sequence[Ref],  # attention_scratch_shapes(geom, leased) order
    me=None,  # traced int32 mesh index (needed when geom.attn_tp > 1)
) -> None:
    """One layer of attention for this rank's requests.

    Must do, for every active request r (rows t < q):
      * write the q new KV latents into the SWA pages, the compressor states,
        and every compressed row that closes in this window (kv_write.py);
        CSA also the indexer compressor states / keys;
      * CSA: indexer scores over the request's compressed keys, exact top-1024
        with sparsecore_topk semantics (tc_topk.topk_group), union of the q
        rows' lists, sparse attention over it; HCA: compressed-cache pass;
        both continued from the SWA window pass with the attention sink;
      * write ``out[r]`` (bf16, row t*128 + h); rows t >= q may hold anything.
    An inactive request reads and writes nothing. Uses only its own DMA
    semaphores (declared in its scratch), no collectives. Expected cost per
    layer at q=7, 128k context: HCA ~15 us/request + ~4 us each extra,
    CSA (scores + top-k + gather) ~60-80 us/request (B3/B1 isolated numbers).
    """
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase

    attn_phase.owner_attention_phase(
        geom,
        kind,
        layer,
        csa_ordinal,
        meta,
        inputs,
        caches,
        weights,
        out,
        lease,
        scratch,
        me=me,
    )


def pack_cos_sin(cs):
    """An AttnWeights rotary table from f32 [max_pos, 64] (once, not per step)."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase

    return attn_phase.pack_cos_sin(cs)


def pad_tables(bt_flat):
    """AttnMeta.bt_flat as the phase reads it (whole tiles; XLA side)."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase

    return attn_phase.pad_tables(bt_flat)


def reset_table_stage(geom: StepGeometry, scratch: Sequence[Ref], leased: bool) -> None:
    """Once per kernel, before the first owner_attention_phase call (scratch:
    attention_scratch_shapes(geom, leased) order)."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase

    attn_phase.reset_table_stage(geom, scratch, leased)


def attention_scratch_shapes(geom: StepGeometry, leased: bool) -> list[Any]:
    """Dedicated scratch (VMEM / SMEM / semaphores) of owner_attention_phase;
    with ``leased`` the large buffers live in the PoolLease instead."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase

    return attn_phase.owner_attention_scratch_shapes(geom, leased)


# ---------------------------------------------------------------------------
# MoE phase (B4; owner: lively-heron)
# ---------------------------------------------------------------------------
class MoEInputs(NamedTuple):
    """VMEM inputs written by the skeleton (read-only here unless noted)."""

    x: Ref  # bf16 [T, 7168]: ffn_norm output, all global rows
    logits: Ref  # f32 [T, 384]: router gate logits (x @ gate, bf16 weight)
    # f32 [T, 7168]: shared-expert partial of this rank (its 1/R
    # intermediate slice, already through w2 and its scale); the phase adds
    # its routed sum (x routed_scaling_factor) to it and all-reduces the sum
    # into `out` (partial itself need not be updated).
    partial: Ref
    hash_ids: Ref  # SMEM int32 [num_hash_layers, T * 8]: tid2eid rows (6 used)
    row_valid: Ref  # SMEM int32 [T]: 1 for live global rows
    amax: Any = None  # optional f32 [T, 1]: precomputed row max |x| from ffn_norm


class MoEWeights(NamedTuple):
    """Served fp4 experts of this rank (EP: E/R local experts), per layer."""

    # Each of the four: LayerRefs of per-layer arrays, or ONE stacked ref
    # [L, ...] of them (preferred: indexed in the DMA, no per-layer branch).
    w13: LayerRefs  # as MoEConfig(layout="stock"): w1 [EL, H, 2I] packed
    w13_scale: LayerRefs  # pad_scale_rows(w1 scale) f32
    w2: LayerRefs  # w2 [EL, I, H] packed
    w2_scale: LayerRefs  # pad_scale_rows(w2 scale) f32
    bias: Ref  # f32 [L, 384] e_score_correction_bias (hash layers: zeros)


def moe_phase(
    geom: StepGeometry,
    layer,  # traced int32
    is_hash,  # traced bool: layer < num_hash_layers (ids from hash_ids)
    me,  # traced int32: this rank's mesh index (ascending device id)
    inputs: MoEInputs,
    weights: MoEWeights,
    out: Ref,  # VMEM bf16 [T, 7168]: all-reduced layer output (all ranks)
    lease: PoolLease,
    scratch: Sequence[Ref],  # moe_scratch_shapes(geom, leased, own_ar) order
    ar: Any = None,  # allreduce.AllReduce over the skeleton's scratch, or None
    ar_slot: Any = None,  # slot of `ar` this call uses (with ar)
    expert_block: Any = None,  # traced int32: EP block of the local experts
    before_ar: Any = None,  # called once the lease is free, before the AR
    expert_held: Any = None,  # SMEM int32 [L, slots]: placement (moe_phase.py)
    expert_copy: Any = None,  # SMEM int32 [L, slots] or None: replica split
    shared_expert: Any = None,  # optional callback computing inputs.partial
) -> None:
    """Router (moe_router semantics, noaux_tc / hash), this rank's routed
    experts on all T rows (W4A8 as stock), sum x routed_scaling_factor added
    onto ``inputs.partial``, then the fused all-reduce over the R ranks
    (hrsag, bf16 payload, rank-independent f32 order) into ``out``.
    Collectives use only semaphores declared in its scratch; every rank's
    signal/wait counts balance within the call. Expected ~76/161/257 us per
    layer at T = 7/28/56 live rows (B4 R=16, incl. its AR)."""
    from . import moe_phase as impl

    impl.moe_phase(
        geom, layer, is_hash, me, inputs, weights, out, lease, scratch, ar, ar_slot,
        expert_block, before_ar, expert_held, expert_copy, shared_expert,
    )  # fmt: skip


def moe_scratch_shapes(
    geom: StepGeometry, leased: bool, own_ar: bool = True
) -> list[Any]:
    from . import moe_phase as impl

    return impl.moe_scratch_shapes(geom, leased, own_ar)


# Skeleton-side buffer ownership, for reference:
#   pool (u8): dense weight ring banks; banks [lease_bank0, n_banks) are lent
#     to whichever phase runs (attention, then MoE), banks below keep
#     prefetching the next stream tiles.
#   VMEM owned by the skeleton: residual [T, 4*7168] bf16, mHC mixes, dense
#     activations, AttnInputs.* (own-row gathers), attention `out`, the MoE
#     inputs and `out`.
# Phase owners may refine their own *Weights / scratch fields (e.g. the ape
# layout); the skeleton passes exactly what the phase declares.
