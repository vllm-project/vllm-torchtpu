# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 whole-backbone decode megakernel, stage A (campaign glm52-decode-mk, WHITEBOARD.v1 md5 6e3abc12).

ONE gridless TensorCore pallas_call per decode step per rank ('glm52_decode_mk'): a fori_loop over all L layers,
then a separate barrier-free cache writer call (writer.py, 'glm52_decode_mk_writer'). Composition (§4):

  host (csq: explore/clean-sheet-qwen/probe/kw/csq_proto.py, the Qwen-MK scaffold)
    * the global static dense-tile stream (DenseWeightRing): tile g of the step lands in bank g % NBD, bank g is
      refilled with tile g + NBD once g is consumed (`dgive`); the router / shared tiles of a MoE layer are given
      back only after the expert stream (lazy flush), so the next layer's prefetch never competes with experts;
    * the 6-slot TP16 expert ring, unit = (expert, this rank's I/R slice), moe_tp_decode numerics
      (main kernels/experimental/moe_tp_decode/kernel.py @ 9ffdb1eb4: _route, _quantize_blocks, the gate_up /
      finish math, _combine), dot1(e+1) issued before the VPU tail of e;
    * replicated f32 residual, two in-kernel all-reduces per layer (collectives.all_reduce = moe_tp_decode's).
  DSA core (G1: dsa.py, from hning/glm52-mk-g1-dsa-spike @ 0a73f68bc), behind the csq DSA interface:
    qall [R, T, HR*QA/2] u32 (after the q AG) + iqv / wts / rec_v (fresh layers) + caches -> osend (partial + lse).
  DCP pieces (glm5_mk_dcp @ d41e70f6d kernel.py): attention front (norms, rope, q_b, W_UK absorb, q packing),
    KV staging rows, indexer front (I_proj), q head all-gather + partial/lse all-to-all (`exchange`), LSE merge.

Per layer: fused add-RMSNorm -> qkv_a -> q_a / kv_a norms, k_pe rope -> KV staging -> q_b + rope + absorb -> q AG
start -> [fresh: indexer front + batched index path] -> batched attention (alias window on the expert ring) ->
A2A + LSE merge -> W_UV, o_proj -> AR#1 -> post norm -> dense MLP | router, shared, TP16 experts, combine -> AR#2.
Trace build (`tiles.trace`): one 'L:layer' scope per layer iteration (layer i = the i-th occurrence in a step; a
traced layer index cannot name a static jax.named_scope, so 'L{i}:{kind}' is recovered from the occurrence order and
the inner kind scopes 'L_index' / 'D_mlp' / 'M_route'), phase scopes inside, per-step work in 'S_prologue' /
'S_epilogue' outside every layer scope.

Numerics follow the instruments' reference (kernel_work/glm52-decode-mk/instruments/glm_ref.py, native mode, bq4
f32 indexer head reduce = the bar T path). `tiles.dot = "fp8"` is the production fp8 x fp8 MXU path;
`"mixed"` (bf16 activation x fp8 weight) is exact in products and serves the oracle on hardware.
"""

import contextlib
import dataclasses
import math
import types

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as mtk
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_block

from . import collectives as CO
from . import dsa as DSA
from . import layout as LY
from . import pool_alias

FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
U32 = jnp.uint32
U8 = jnp.uint8
FP8_MAX = 448.0
NT = (((1,), (1,)), ((), ()))
NN = (((1,), (0,)), ((), ()))
MB_SH = 27  # own_u entries: gather source | row bits << MB_SH
SRC_MASK = (1 << MB_SH) - 1
COLLECTIVE_ID = 47  # not 0 (moe_tp_decode, fused_moe), 37/39 (glm5 MKs), 43 (probes/01), 45 (csq), 61 (Qwen)
KERNEL_NAME = "glm52_decode_mk"
VMEM_LIMIT = 67_043_328  # XLA clamps the request to this on v7x [measured: j-5b5bf421]

# kernel-side fault controls (FAULTS of dsa.py are forwarded to the core)
FAULTS = (
    DSA.FAULTS
    + (
        "stale_accumulator",
        "dropped_tail_row",
        "missing_rank_partial",
        "router_tiebreak",
        "rope_off_by_one",
        "dcp_owner",
        "stale_indexshare",
        "f_ichunk",
        "f_slotreuse",
        "f_hitcap",
        "f_xrace",
        "f_nope",
        "f_alias_poison",  # alias layout only (Mosaic, or interpret via interp_alias)
        "f_alias_poison_red",
        "f_stale_is_alias",  # own_uv placed in the alias (+ poison)
        "f_ar_earlyread",  # AR#1 sums its slots before waiting them (delay-exposed race)
        "f_alias_poison_nan",  # F-ALIAS-POISON with 0xFF bytes (e4m3fn NaN): must stay identical
        "f_alias_poison_nan_noguard",  # ... and the DSA's tail-row zeroing off: must go RED
        "f_a2a_early",  # L4 C2: an A2A chunk is sent one request before its rows are emitted
        "f_qag_pair_drop",  # stage-B L2-Q: qag="pair" forwards one same-parity block too few
        "f_leaf_shift",
        "f_lv_short",  # LV: AR#1 moves one 4-row block too few (that block's rows lose the peers' partials)
        "f_replica_nowrite",  # V16: only replica 0 of each DCP shard writes its cache copy (writer)
    )
)  # L3: the expert-operand leaf / index is one MoE layer off -> RED
# WHITEBOARD.v1's F-ALIAS-EARLY is not built (needs a hook between a
# granule's gather wait and its read)
ABLATIONS = ("weights", "experts", "coll", "index", "gather", "attn")


@dataclasses.dataclass(frozen=True)
class Tiles:
    bank: int = 3 << 20  # dense ring bank (csq: 8 x 3 MiB)
    nbd: int = 8  # dense ring banks
    nbe: int = 6  # expert ring slots (moe_tp_decode _NUM_BUFFERS)
    dot: str = (
        "fp8"  # "fp8" (production) | "mixed" (bf16 act x fp8 weight: exact products)
    )
    trace: bool = False  # named scopes (per kind, per phase)
    # batched DSA core (G1 tunables, unchanged defaults)
    pool: int = 1344
    gring: int = 32
    scan_sd: int = 4
    scan_np: int = 16
    pv: str = "f32"  # WHITEBOARD.v1 A5: f32 PV (served formula)
    recip: str = "exact"  # attention 1 / l: "exact" (production since the fix pass: the approximate EUP form
    # failed the TINY_tpu core-vs-f64 gate in interpret, 4.441e-3 > 4.404e-3, and costs
    # one (QN, 1) reciprocal per request per layer either way) | "approx" (the served
    # sparse kernel's form; an A1 arm)
    thr_switch: int = 512  # Sigma np above which the threshold uses 2-bit digits (G1 design §3.3, §11.2)
    gunroll: int = 16
    xkc: int = 512
    fault: str = ""  # oracle fault control (FAULTS)
    alias: bool = True  # Mosaic: expert ring + DSA scratch in the u8 pool `epool` (WHITEBOARD.v1 §7)
    alias_dsa: bool = True  # Mosaic: DSA layer-transient scratch as epool views (False: native scratch, the
    # AOT negative control -- WHITEBOARD.v1 §7.2 says it cannot fit)
    cid: int = COLLECTIVE_ID
    scan_unroll: int = 4  # DSA scan inner unroll (was the MKD_SCAN_UNROLL env var: outside every cache key)
    interp_alias: bool = True  # interpret mode: run the Mosaic alias layout (dpool / epool views) through
    # interp_alias.install() instead of separate typed buffers (alias=False: typed)
    # ---- stage B lane L4 (collectives); every default is the stage-A program ----
    a2a_lse: str = "lanes"  # "lanes": 128 lse lanes in every A2A payload row (stage A) | "tile": one [8, 128] f32
    # lse tile per peer (sublane = head within the owner, lane = row), payload = OH lanes
    a2a_rw: int = 0  # > 0: the A2A goes in chunks of a2a_rw rows, each started inside the attention loop
    # once the requests covering its rows have emitted (requires a2a_lse == "tile")
    replay: str = ""  # TIMING-ONLY backing arms (need recorded operands, see build()):
    # "a2a": the A2A + LSE merge are replaced by an HBM load of the recorded merged `oh`;
    # "ar": both all-reduces are replaced by HBM loads of their recorded outputs
    shadow_a2a: bool = False  # TIMING-ONLY addition arm: a second, unconsumed A2A of the same bytes after the real one
    # ---- stage-B lane L1 (WHITEBOARD.v3 §6.1); every default is the stage-A program ----
    thr_unroll: int = 1  # L1-U: threshold count loop, tiles per iteration
    thr_early: bool = False  # L1-E: a threshold group stops as soon as #(key >= prefix) == KT on every row (exact)
    thr_dsend: bool = False  # L1-D: count-exchange send completions waited two rounds later (pipelined form)
    thr_swar: bool = (
        False  # L1-S: 4-bit round counts from a packed 8-bit digit histogram (exact)
    )
    thr_merge: int = 0  # L1-M: one merged count exchange per round when the wave's Sigma np <= thr_merge
    # (0: never; >= pool: always, and the per-group form is not compiled)
    rec_thr: bool = False  # timing harness only: export b_row[2:5] / own_uv / b_req / b_greq per fresh layer
    thr_given: bool = False  # timing-only backing arm: load the recorded T / P / allowance, skip threshold()
    union_given: bool = False  # timing-only backing arm: load the recorded union table, skip threshold() + unions
    # ---- stage B, lane L2 (WHITEBOARD.v3 §6.2); every default reproduces stage A (e852e35b7) ----
    gearly: tuple = ()  # (g1, g2): the next IndexShare layer's OLD gather entries of granules [0, g1) are
    # issued inside this MoE layer's AR#2 reduce-scatter wait and [g1, g2) inside its
    # all-gather wait (the TC is idle there; the alias window is free once the expert
    # loop has drained); () = stage A
    qag: str = "mk"  # q head all-gather transport: "mk" = stage A's one-round all-peer `exchange`;
    # "pair" = chip-sibling dedupe (P-TX j-ae1c975b: 22.7 vs 35.4 us at T = 32): own
    # block -> the R/2 - 1 same-parity ranks + the sibling me^1 (same chip); at q_wait
    # the same-parity blocks are forwarded to the sibling over the on-chip path
    qmid: int = (
        0  # qag="pair": forward the same-parity blocks to the sibling right after the
    )
    # attention pre-issue has issued `qmid` granules (0 = at q_wait, after the whole
    # pre-issue), so the on-chip forward overlaps the remaining issue
    # timing-only backing arms (WHITEBOARD.v3 §5.3 record / replay; compose: L2 `replay` renamed `l2replay`; outputs must stay bit-identical):
    record: bool = False  # extra HBM outputs: qall per layer [L, R, TMAX, QH] and the first `grec` gather
    # granules' rows per layer [L, grec * 128, 128] (nope, rope) as gathered
    l2replay: str = ""  # "q": qall per layer DMA'd from a recorded HBM input instead of the q AG; "g": the
    # first `grec` granules' rows DMA'd from recorded inputs instead of the row gather
    # (n rows per granule, the waits unchanged); "qg": both
    grec: int = 40
    # stage-B lane L3 (served operand form; WHITEBOARD.v3 §6.3). Defaults = stage A, program unchanged.
    ecas: str = "when2"  # expert-operand cascade over the Lm per-layer operands (stack_e=False, or edummy):
    # "when2" (stage A: two pl.when per level) | "cond" (L3-a: one lax.cond if/else
    # per level, half the compares)
    ebatch: int = 1  # L3-b: expert-ring refill batch k. 1 = stage A (one issue, one cascade evaluation
    # per unit). k > 1: the ring primes in one evaluation and refills once >= k slots
    # are free (or the layer's last units are issuable): ~U / k evaluations per layer
    edummy: bool = False  # L3 mechanism arm (stack_e=True only): route every issue through the served
    # form's Lm-leaf cascade, leaf i reading the stacked operands at the static `.at[i]`
    esingle: bool = False  # L3-s: ONE issue site per MoE layer (bundles): the expert loop runs from step -1,
    # whose iteration primes the ring through the same refill site and then runs the
    # shared expert; uses the L3-b refill rule with k = ebatch (k = 1 included)
    # LV (stageB/dcp8-design §3.4; stageB/lv-probe, lv-prod): live-row collectives. The residual all-reduces move only
    # n = ceil(T / 4) * 4 rows (collectives.lv_ar_rows / ar_pieces: Mosaic-exact row pieces), the "pair" q all-gather
    # 4 or ceil(T / 8) * 8 rows, and the L4 A2A only the ceil(T / 4) live 4-row chunks (padded chunks are not sent at
    # layer end). Exact: every row of every collective is independent, and rows >= T are masked before every
    # consumer. Default off = the 03dbe3617 program.
    lv: bool = False
    lv_eight: str = "split"  # an all-reduce of 8 (mod 16) rows: "split" = two 4-row pieces | "round" = 12 rows
    lv_floor: int = 0  # rows below which nothing is trimmed (the next-smaller row bucket: T <= floor never runs here)
    lv_loop: int = 0  # LV all-reduce issue sites (collectives.all_reduce_lv): 0 = per row count, peers unrolled |
    # 1 = per row count, peer fori_loops (bundles) | 2 = 1 except the all-rows transfer (unrolled)
    lv_sites: str = "aqx"  # which collectives LV trims: a = all-reduces, q = q all-gather, x = A2A (arms)
    lv_dead: bool = False  # TIMING-ONLY control: the LV all-reduce / q AG code is compiled under a never-true
    # predicate and the stage-B code runs (prices code size / branch structure with no row saved)


def bf(x):
    return x.astype(BF16).astype(F32)


def quant_rows(x):
    """Per-token fp8 quant (quantized_matmul/util.py quantize_block): scale = amax / 448 (0 -> 1)."""
    s = jnp.max(jnp.abs(x), axis=1, keepdims=True) / FP8_MAX
    s = jnp.where(s == 0, 1.0, s)
    return jnp.clip(x / s, -FP8_MAX, FP8_MAX).astype(FP8), s


def rmsnorm(x, w, eps):
    """vLLM rms_norm: bf16(x * rsqrt(mean x^2 + eps)) * w -> bf16 (x f32, w bf16-valued f32)."""
    return bf(bf(x * lax.rsqrt(jnp.mean(x * x, axis=1, keepdims=True) + eps)) * w)


def lane_swap(x):
    n = x.shape[-1]
    lane = lax.broadcasted_iota(I32, x.shape, x.ndim - 1)
    return jnp.where(
        lane % 2 == 0, pltpu.roll(x, n - 1, x.ndim - 1), pltpu.roll(x, 1, x.ndim - 1)
    )


def half_swap(x):
    """NeoX (half-split) rope partner: lane l <-> l +- 32 inside each 64-lane rope segment (DR = 64)."""
    n = x.shape[-1]
    lane = lax.broadcasted_iota(I32, x.shape, x.ndim - 1)
    return jnp.where(
        lane % 64 < 32, pltpu.roll(x, n - 32, x.ndim - 1), pltpu.roll(x, 32, x.ndim - 1)
    )


def rope(x, cos, sin_signed, neox=False):
    """Rope with torch bf16 rounding: bf16(bf16(x*cos) + bf16(swap(x)*sin)). GPT-J interleaved pairs (lane_swap), or
    with `neox` (MKConfig.rope_neox) half-split pairs (half_swap; the table from wrapper.rope_table in that order)."""
    return bf(bf(x * cos) + bf((half_swap(x) if neox else lane_swap(x)) * sin_signed))


def fp8_bytes(x8):
    return lax.bitcast_convert_type(x8, U8).astype(I32)


def fp8_decode_hw(b):
    b = jnp.where((b & 0x7F) == 0x7F, b ^ 1, b)
    return lax.bitcast_convert_type(b.astype(U8), FP8).astype(BF16)


def shl(x, n):
    return lax.shift_left(x, jnp.asarray(n, x.dtype))


def shr(x, n):
    return lax.shift_right_logical(x, jnp.asarray(n, x.dtype))


def pack2(lo, hi):
    a = shr(lax.bitcast_convert_type(lo, U32), 16)
    b = lax.bitcast_convert_type(hi, U32) & jnp.uint32(0xFFFF0000)
    return a | b


def unpack_lo(u):
    return lax.bitcast_convert_type(shl(u, 16), F32)


def unpack_hi(u):
    return lax.bitcast_convert_type(u & jnp.uint32(0xFFFF0000), F32)


def mono_key(score):
    b = lax.bitcast_convert_type(score, I32)
    return jnp.where(b >= 0, b, b ^ jnp.int32(0x7FFFFFFF))


def mxu(a, b, tiles):
    if tiles.dot == "fp8" or (a.dtype != FP8 and b.dtype != FP8):
        return a, b
    if tiles.dot == "mixed":
        return a.astype(BF16), b
    return a.astype(BF16), b.astype(BF16)


def bswitch(idx, fns):
    """Run fns[k]() for k == idx (traced scalar), keys sorted ints: binary cascade of whens."""
    keys = sorted(fns)

    def rec(ks):
        if len(ks) == 1:
            fns[ks[0]]()
            return
        mid = len(ks) // 2

        @pl.when(idx < ks[mid])
        def _():
            rec(ks[:mid])

        @pl.when(idx >= ks[mid])
        def _():
            rec(ks[mid:])

    rec(keys)


def bswitch_cond(idx, fns):
    """bswitch with one lax.cond (scf.if with an else region) per level: one compare per level instead of two
    [source: jax/_src/pallas/mosaic/lowering.py:4719-4752@7f6fe90 (cond -> scf.IfOp has_else=True); pl.when =
    cond(c, f, noop), jax/_src/pallas/helpers.py:68-87@7f6fe90]."""
    keys = sorted(fns)

    def rec(ks):
        if len(ks) == 1:
            fns[ks[0]]()
            return
        mid = len(ks) // 2
        lax.cond(idx < ks[mid], lambda: rec(ks[:mid]), lambda: rec(ks[mid:]))

    rec(keys)


def ring_issue_evals(count, nbe, k):
    """Host mirror of the expert ring's issue schedule (M6 driver): cascade evaluations for one MoE layer with
    `count` units, ring depth nbe, refill batch k (k = 1: stage A, one per unit)."""
    if count <= 0:
        return 0
    if k == 1:
        return count
    ev, iss = 1, min(count, nbe)
    for s_ in range(count):
        avail = min(s_ + nbe, count - 1) + 1
        pend = avail - iss
        if pend >= k or (avail == count and pend > 0):
            ev, iss = ev + 1, avail
    assert iss == count, (count, nbe, k, iss)
    return ev


class Phases:
    def __init__(self, on):
        self.on, self.cur = on, None

    def mark(self, name):
        self.end()
        if self.on:
            self.cur = jax.named_scope(name)
            self.cur.__enter__()

    def end(self):
        if self.cur is not None:
            self.cur.__exit__(None, None, None)
            self.cur = None


def cache_groups(groups, ck):
    """{kind: n_groups} (default one stacked operand per kind; n = number of layers of the kind = per-layer)."""
    g = {k: 1 for k in ck}
    if groups:
        g.update({k: int(n) for k, n in groups.items()})
    for k, n in g.items():
        assert len(ck[k][0]) % n == 0, (k, n)
    return g


def cache_names(ck, G):
    return [(k, gi) for k in ck for gi in range(G[k])]


def aliased_names(tiles):
    """The DSA buffers that are epool views. Fault builds F-STALE-IS-ALIAS / F-ALIAS-POISON-RED also place the
    IndexShare table `own_uv` (persistent by design) in the alias, after the layer-transient set."""
    if not tiles.alias_dsa:
        return ()
    extra = ("own_uv",) if tiles.fault in DSA.UV_ALIAS_FAULTS else ()
    return DSA.ALIASED + extra


def epool_layout(cfg, tiles, max_pages):
    """Byte offsets of the expert-slot views and the aliased DSA views in the u8 pool `epool` (WHITEBOARD.v1 §7.1):
    every view's pool_alias storage equals its raw bytes, offsets are 4096-B aligned and non-overlapping within each
    class (expert slots / DSA views); the DSA views overlap the slots by design (the alias window).
    Returns (pool rows of 128 B, {name: (row offset, shape, dtype)}, slot rows, DSA bytes)."""
    w13 = ((cfg.H, 2 * cfg.IR), FP8)
    w2 = ((cfg.IR, cfg.H), FP8)
    views = {}
    off = 0
    for i in range(tiles.nbe):
        for nm, (s, d) in (("w13", w13), ("w2", w2)):
            sz = pool_alias.storage_size(s, jnp.dtype(d))
            assert sz == math.prod(s) * jnp.dtype(d).itemsize, (nm, s, sz)
            views[f"{nm}_{i}"] = (off // 128, s, d)
            off += sz
    ring_b = off
    dsa_specs = DSA.scratch(cfg, tiles, max_pages, cfg.BMAX)
    off = 0
    for nm in aliased_names(tiles):
        spec = dsa_specs[nm]
        s, d = tuple(spec.shape), spec.dtype
        if (
            nm not in DSA.ALIASED
        ):  # fault-only view (own_uv): round its rows up to the dtype's row tile,
            rt = 8 * max(
                1, 4 // jnp.dtype(d).itemsize
            )  # since a pool_alias view cannot pad
            s = s[:-2] + (-(-s[-2] // rt) * rt, s[-1])
        sz = pool_alias.storage_size(s, jnp.dtype(d))
        assert sz == math.prod(s) * jnp.dtype(d).itemsize, (
            "alias view pads",
            nm,
            s,
            sz,
        )
        assert off % 4096 == 0
        views[nm] = (off // 128, s, d)
        off += sz
    dsa_b = off
    total = max(ring_b, dsa_b)
    assert total % 128 == 0
    return total // 128, views, ring_b, dsa_b


DEFAULT_TILES = Tiles()


def build(
    cfg,
    *,
    max_pages,
    nb,
    nbi,
    tiles=DEFAULT_TILES,
    interpret=False,
    debug=False,
    skip=frozenset(),
    stack_e=True,
    groups=None,
):
    """Returns (call, info). call(*fixed, *dense weights, *expert weights, *caches) -> (out [M, H] bf16,
    nrow [L, T, 128] u32, rrow [L, T, 128] u32, irec [L, T, 256] i32[, debug outputs]) for ONE rank (wrap in
    shard_map over the R-rank axis; the rank comes in as scal[2], never axis_index).

    fixed (11): scal SMEM i32[16] (0 live rows T, 1 requests B, 2 rank, 3 F-DELAY rank (-1 off), 4 F-DELAY spin
      count, 5 F-DELAY layer), rmeta SMEM i32[T, 8] (0 valid, 1 pos, 2 req, 3 seq_len), qmeta SMEM i32[BM, 4]
      (0 first row, 1 rows, 2 seq_len, 3 valid), bt / bti SMEM i32[BM, MAXP] (local latent / index-K pages),
      ltab SMEM i32[L, 4], dtab SMEM i32[T_stream], rvec VMEM i32[M, 128] (lane 0 valid, 1 pos),
      h0 VMEM f32[M, H] (embedding, bf16 values), rope_tab VMEM f32[4, M, 128], smalls HBM f32[L + 1, 1, S].
    dense weights: one stacked HBM operand per kind in layout.DKINDS order (kinds with layers only).
    experts (main's served TP-sharded layout, EP off): stack_e -> ew13 [Lm, E, H, 2IR], ew2 [Lm, E, IR, H],
      es13 [Lm, E, 1, 1, 2IR], es2 [Lm, E, 1, 1, H]; else per MoE layer the four served tensors themselves
      (w13 [E, H, 2IR], w2 [E, IR, H], s13 [E, 1, 1, 2IR], s2 [E, 1, 1, H]; zero-copy; Lm x 4 operands, bswitch).
    caches: per kind (nope, rope, idx) `groups[kind]` stacked operands [n_layers / groups, ...]."""
    cfg.check()
    assert tiles.fault in FAULTS, tiles.fault
    GEARLY = bool(tiles.gearly)
    assert set(tiles.l2replay) <= {"q", "g"}, tiles.l2replay
    REC_IN = (["qrec"] if "q" in tiles.l2replay else []) + (
        ["gnrec", "grrec"] if "g" in tiles.l2replay else []
    )
    REC_OUT = ["qrec_o", "gnrec_o", "grrec_o"] if tiles.record else []
    assert tiles.qag in ("mk", "pair"), tiles.qag
    assert (
        tiles.qag == "mk"
        or cfg.R == 1
        or (cfg.R % 2 == 0 and (cfg.R & (cfg.R - 1)) == 0)
    ), (tiles.qag, cfg.R)
    POISON_FAULTS = (
        "f_alias_poison",
        "f_alias_poison_red",
        "f_alias_poison_nan",
        "f_alias_poison_nan_noguard",
    )
    assert set(skip) <= set(ABLATIONS), skip
    fault = tiles.fault
    R, L, H, M, TMAX, BMAX = cfg.R, cfg.L, cfg.H, cfg.M, cfg.TMAX, cfg.BMAX
    HR, KL, DQ, DN, DV, QL = cfg.HR, cfg.KL, cfg.DQ, cfg.DN, cfg.DV, cfg.QL
    IH, IDIM, E, EPAD, TOPK = cfg.IH, cfg.IDIM, cfg.E, cfg.EPAD, cfg.TOPK
    IR, ISR, IDR, KR, QB = cfg.IR, cfg.ISR, cfg.IDR, cfg.KR, cfg.QB
    QA = KL + 128
    QH = HR * QA // 2
    OH = HR * KL // 2
    OW = OH + 128
    # ---- L4 (stage B) collectives variants; defaults = the stage-A program ----
    assert tiles.a2a_lse in ("lanes", "tile"), tiles.a2a_lse
    LSE_TILE = tiles.a2a_lse == "tile"
    RW = int(tiles.a2a_rw)
    if RW >= TMAX:  # one chunk = the unchunked exchange
        RW = 0
    assert RW == 0 or (LSE_TILE and RW % 4 == 0 and TMAX % RW == 0), (
        "a2a_rw needs a2a_lse='tile', 4 | rw | TMAX",
        RW,
        TMAX,
    )
    NCH = TMAX // RW if RW else 1
    HPAD = max(
        8, -(-HR // 8) * 8
    )  # lse tile rows per owner (heads padded to the f32 sublane tile)
    if LSE_TILE:
        assert R * HPAD <= 128 and TMAX <= 128, (
            "lse tile: R * HPAD rows, TMAX lanes",
            R,
            HPAD,
            TMAX,
        )
    OWX = OH if LSE_TILE else OW  # A2A payload row (u32 lanes)
    LV = bool(tiles.lv) and cfg.R > 1 and "coll" not in skip
    LVD = bool(tiles.lv_dead) and cfg.R > 1 and "coll" not in skip
    assert not (LV and LVD), "lv_dead is the no-LV control"
    if LV or LVD:
        assert LSE_TILE and RW == 4, ("LV rides on L4 C3: lse tile + 4-row A2A chunks", tiles.a2a_lse, RW)
        assert tiles.qag == "pair", ("LV trims the 'pair' q all-gather only", tiles.qag)
        assert not tiles.l2replay and not tiles.record and tiles.replay == "" and not tiles.shadow_a2a, (
            "LV: no timing-only record / replay / shadow arms"
        )
        assert tiles.fault != "f_ar_earlyread", "LV's all-reduce has no early-read fault variant"
        assert set(tiles.lv_sites) <= set("aqx") and 0 <= tiles.lv_floor < TMAX and tiles.lv_floor % 4 == 0, (
            tiles.lv_sites,
            tiles.lv_floor,
        )
    LV_A = (LV or LVD) and "a" in tiles.lv_sites
    LV_Q = (LV or LVD) and "q" in tiles.lv_sites
    LV_X = LV and "x" in tiles.lv_sites
    LV_FLOOR = int(tiles.lv_floor)
    AR_OPTS = CO.lv_ar_options(M, LV_FLOOR, tiles.lv_eight) if LV_A else ()
    # LV q AG row counts: u32 DMA rows must be < 8 or a multiple of 8 (12 does not compile), clamped to >= floor
    Q_OPTS = tuple(sorted({min(max(n, LV_FLOOR), TMAX) for n in (4,) + tuple(range(8, TMAX + 1, 8))}))
    assert tiles.replay in ("", "a2a", "ar"), tiles.replay
    n_r = {"": 0, "a2a": 1, "ar": 2}[tiles.replay]
    NQB = HR * DQ
    NBLK = H // QB
    QB8 = 8
    NBD, NBE = tiles.nbd, tiles.nbe
    KB = int(tiles.ebatch)
    ES = bool(tiles.esingle)
    assert tiles.ecas in ("when2", "cond"), tiles.ecas
    assert KB >= 1 and NBE - KB >= 3, (
        "refill batch keeps >= 3 slots (>= 6 MiB) in flight",
        NBE,
        KB,
    )
    assert not (tiles.edummy and not stack_e), (
        "edummy is the stacked-form mechanism arm"
    )
    # typed: separate typed buffers per dense kind / expert slot / DSA buffer. The Mosaic path (and, through
    # interp_alias, the interpret path) instead runs the shared-storage layout: dense banks shared across kinds in
    # `dpool`, expert slots and the aliased DSA set as views of `epool`.
    typed = not tiles.alias or (bool(interpret) and not tiles.interp_alias)
    if interpret and not typed:
        from . import interp_alias

        interp_alias.install()
    assert bool(interpret) or (H // R) % 128 == 0, (
        "all-reduce column chunks must be lane aligned on the Mosaic path"
    )
    assert nb < (1 << (MB_SH - 8)), nb
    lay, S = LY.smalls_layout(cfg)
    BANK = LY.bank_bytes(cfg, tiles.bank)
    plan = LY.tile_plan(cfg, BANK)
    for k_ in ("gate", "sh13", "sh2", "wuk", "wuv"):
        assert plan[k_][2] == 1, ("consumed as one tile", k_, plan[k_])
    dkinds = [k for k in LY.DKINDS if LY.layers_of(cfg, k)]
    ltab_np, dtab_np = LY.stream_tables(cfg, plan)
    T_STREAM = len(dtab_np)
    eshapes = LY.expert_shapes(cfg)
    moe_layers = LY.layers_of(cfg, "gate")
    Lm = len(moe_layers)
    ekinds = ("ew13", "ew2", "es13", "es2")
    e_names = (
        [(k, "stk") for k in ekinds]
        if stack_e
        else [(k, i) for i in range(Lm) for k in ekinds]
    )
    ck = LY.cache_kinds(cfg, nb, nbi)
    G = cache_groups(groups, ck)
    c_names = cache_names(ck, G)
    full_l = LY.layers_of(cfg, "iwq")
    Lf = len(full_l)

    def cpos_fn(kind):
        ls = ck[kind][0]
        if kind != "idx":
            return lambda l: l
        fr, sk = cfg.idx_freq, cfg.idx_skip
        for i, l in enumerate(ls):
            assert (l if l < sk else sk + (l - sk) // fr) == i, (ls, l)
        return lambda l: jnp.clip(
            jnp.where(l < sk, l, sk + (l - sk) // fr), 0, len(ls) - 1
        )

    CPOS = {k: cpos_fn(k) for k in ck}

    NACC = LY.rup(max(cfg.QKVP, NQB, H, 2 * IDR, IH * IDIM, cfg.IKW, EPAD))
    KX = max(H, QL, KR, IDR, ISR)

    # ---------------- scratch ----------------
    dsa_specs = DSA.scratch(cfg, tiles, max_pages, BMAX)
    scratch = {
        "sm": pltpu.VMEM((2, 1, S), F32),
        "sm_sem": pltpu.SemaphoreType.DMA((2,)),
        "res": pltpu.VMEM((M, H), F32),
        "hcur": pltpu.VMEM((M, H), F32),
        "xbf": pltpu.VMEM((M, H), BF16),
        "xq": pltpu.VMEM((M, KX), FP8),
        "sxr": pltpu.VMEM((M, 128), F32),
        "acc": pltpu.VMEM((M, NACC), F32),
        "qc": pltpu.VMEM((M, QL), F32),
        "kvc": pltpu.VMEM((M, KL), F32),
        "kpe": pltpu.VMEM((M, 128), F32),
        "wrow": pltpu.VMEM((TMAX, 128), U32),
        "rwrow": pltpu.VMEM((TMAX, 128), U32),
        "rec_v": pltpu.VMEM((TMAX, 256), I32),
        "kvw_sem": pltpu.SemaphoreType.DMA((3,)),
        "qsend": pltpu.VMEM((TMAX, QH), U32),
        "qall": pltpu.VMEM((R, TMAX, QH), U32),
        "iqv": pltpu.VMEM((IH, M + QB8, 128), F32),
        "wts": pltpu.VMEM((M + QB8, 128), F32),
        "osend": pltpu.VMEM((R, TMAX, OWX), U32),
        "orecv": pltpu.VMEM((R, TMAX, OWX), U32),
        "orow": pltpu.VMEM((M, KR), F32),
        "oh": pltpu.VMEM((HR, M, KL), F32),
        "mvalid": pltpu.VMEM((M, 128), F32),
        "part": pltpu.VMEM((M, H), F32),
        "attn": pltpu.VMEM((M, H), F32),
        # MoE (csq / moe_tp_decode)
        "owner": pltpu.VMEM((M, H), F32),
        "shared": pltpu.VMEM((M, H), F32),
        "xqb": pltpu.VMEM((M, H), FP8),
        "xsb": pltpu.VMEM((NBLK, M, 128), BF16),
        "wtab": pltpu.VMEM((M, EPAD), F32),
        "hits_v": pltpu.VMEM((1, EPAD), I32),
        "hits_s": pltpu.SMEM((1, EPAD), I32),
        "active": pltpu.SMEM((E + 8,), I32),
        "hit_sem": pltpu.SemaphoreType.DMA((1,)),
        "es13": pltpu.VMEM((NBE, 2 * IR), F32),
        "es2": pltpu.VMEM((NBE, H), F32),
        "ring_sem": pltpu.SemaphoreType.DMA((NBE, 4)),
        # dense ring
        "dsem": pltpu.SemaphoreType.DMA((NBD,)),
        "misc_sem": pltpu.SemaphoreType.DMA(()),
        "q_send": pltpu.SemaphoreType.DMA(()),
        "q_recv": pltpu.SemaphoreType.DMA(()),
        "a_send": pltpu.SemaphoreType.DMA(()),
        "a_recv": pltpu.SemaphoreType.DMA(()),
        "spin": pltpu.VMEM((8, 128), F32),
        "arbuf": pltpu.VMEM((M, H), BF16),
    }
    if REC_IN or REC_OUT:
        scratch["rec_sem"] = pltpu.SemaphoreType.DMA((2,))
    if tiles.qag == "pair":
        scratch["q_send2"] = pltpu.SemaphoreType.DMA(())
        scratch["q_recv2"] = pltpu.SemaphoreType.DMA(())
    for i, sp in enumerate(CO.ar_scratch(R, M, H) if R > 1 else []):
        scratch[f"ar1_{CO.AR_NAMES[i]}"] = sp
    if LSE_TILE:  # rows d * HPAD + hh (owner d, head hh), lanes = rows t
        scratch["lsend"] = pltpu.VMEM((128, 128), F32)
        scratch["lrecv"] = pltpu.VMEM((128, 128), F32)
        scratch["lse_t"] = pltpu.VMEM(
            (128, 128), F32
        )  # lrecv transposed: row t, lane s * HPAD + hh
    if RW:
        scratch["a2a_n"] = pltpu.SMEM(
            (1,), I32
        )  # chunks of this layer's A2A already started
    if tiles.shadow_a2a:
        scratch["a2_send"] = pltpu.SemaphoreType.DMA(())
        scratch["a2_recv"] = pltpu.SemaphoreType.DMA(())
    if n_r:
        scratch["rep_sem"] = pltpu.SemaphoreType.DMA(())
    if tiles.replay == "ar":
        scratch["rbuf"] = pltpu.VMEM((M, H), F32)
    if (
        fault != "f_xrace"
    ):  # F-XRACE: one slot set shared by both all-reduces of a layer
        for i, sp in enumerate(CO.ar_scratch(R, M, H) if R > 1 else []):
            scratch[f"ar2_{CO.AR_NAMES[i]}"] = sp
    if typed:
        for k in dkinds:
            shp, dt, _, _ = plan[k]
            scratch["dring_" + k] = pltpu.VMEM((NBD,) + tuple(shp), dt)
        scratch["er13"] = pltpu.VMEM((NBE, H, 2 * IR), FP8)
        scratch["er2"] = pltpu.VMEM((NBE, IR, H), FP8)
        for nm, sp in dsa_specs.items():
            scratch[nm] = sp
        EPOOL = None
    else:
        BROWS = BANK // 128
        scratch["dpool"] = pltpu.VMEM((NBD * BROWS, 128), U8)
        epool_rows, eviews, ring_b, dsa_b = epool_layout(cfg, tiles, max_pages)
        scratch["epool"] = pltpu.VMEM((epool_rows, 128), U8)
        for nm, sp in dsa_specs.items():
            if nm not in aliased_names(tiles):
                scratch[nm] = sp
        EPOOL = dict(rows=epool_rows, views=eviews, ring_bytes=ring_b, dsa_bytes=dsa_b)
    if fault == "f_nope":
        scratch["fnope"] = pltpu.VMEM((M, KL), F32)
    if fault == "stale_indexshare":
        g_ = DSA.geometry(cfg, tiles, max_pages, BMAX)
        scratch["sv_uv"] = pltpu.VMEM((g_["UR2"], 128), I32)
        scratch["sv_u"] = pltpu.SMEM((g_["UR2"], 128), I32)
        scratch["sv_req"] = pltpu.SMEM((BMAX, 16), I32)
        scratch["sv_greq"] = pltpu.SMEM((g_["UR2"],), I32)
        scratch["sv_st"] = pltpu.SMEM((2,), I32)
    if debug:
        g_ = DSA.geometry(cfg, tiles, max_pages, BMAX)
        scratch["dbg_uv"] = pltpu.VMEM((g_["UR2"], 128), I32)
        scratch["dbg_sem"] = pltpu.SemaphoreType.DMA((1,))
        scratch["dbg_qc"] = pltpu.VMEM((M, QL), F32)
        scratch["dbg_x"] = pltpu.VMEM((M, H), F32)
    s_names = list(scratch)

    n_fixed = 11
    n_d = len(dkinds)
    n_e = len(e_names)
    n_c = len(c_names)
    GEO = DSA.geometry(cfg, tiles, max_pages, BMAX)
    # L1 backing arms: records exported by a rec_thr build and loaded by a thr_given / union_given build (HBM)
    LfX = max(Lf, 1)
    rec_specs = {
        "thr": ((LfX, 3, GEO["TR"], 128), I32),
        "uv": ((LfX, GEO["UR2"], 128), I32),
        "req": ((LfX, BMAX, 16), I32),
        "greq": ((LfX, GEO["UR2"]), I32),
    }
    xin_names = (
        ["uv", "req", "greq"]
        if tiles.union_given
        else ["thr"]
        if tiles.thr_given
        else []
    )
    xout_names = ["thr", "uv", "req", "greq"] if tiles.rec_thr else []
    n_x = len(xin_names)

    def kernel(*refs):
        (scal, rmeta, qmeta, bt, bti, ltab, dtab, rvec, h0, rope_tab, smalls) = refs[
            :n_fixed
        ]
        Wd = dict(zip(dkinds, refs[n_fixed : n_fixed + n_d]))
        e_in = refs[n_fixed + n_d : n_fixed + n_d + n_e]
        r0_ = n_fixed + n_d + n_e
        RI = dict(
            zip(REC_IN, refs[r0_ : r0_ + len(REC_IN)])
        )  # L2 records: before the caches
        c_in = refs[r0_ + len(REC_IN) : r0_ + len(REC_IN) + n_c]
        # extra HBM inputs after the caches: L4 replay records (n_r), then L1 backing-arm records (n_x)
        x0_ = r0_ + len(REC_IN) + n_c
        recs = refs[x0_ : x0_ + n_r]
        x_in = dict(zip(xin_names, refs[x0_ + n_r : x0_ + n_r + n_x]))
        o0 = x0_ + n_r + n_x
        out_ref, nrow_out, rrow_out, irec_out = refs[o0 : o0 + 4]
        RO = dict(zip(REC_OUT, refs[o0 + 4 : o0 + 4 + len(REC_OUT)]))
        rest = refs[o0 + 4 + len(REC_OUT) :]
        if debug:
            (
                d_state,
                d_attn,
                d_xin,
                d_xffn,
                d_qc,
                d_kvcn,
                d_qall,
                d_oloc,
                d_wtab,
                d_uv,
                d_ureq,
            ) = rest[:11]
            rest = rest[11:]
        x_out = dict(zip(xout_names, rest[: len(xout_names)]))
        rest = rest[len(xout_names) :]
        scr = dict(zip(s_names, rest))
        sc = (
            (lambda nm: jax.named_scope(nm))
            if tiles.trace
            else (lambda nm: contextlib.nullcontext())
        )

        # ---- views (Mosaic: dense banks and expert slots / DSA scratch as pool_alias views) ----
        if not typed:
            BROWS_ = BANK // 128

            def dview(kind, bank):
                shp, dt, _, _ = plan[kind]
                return pool_alias.view(
                    scr["dpool"].at[
                        pl.ds(
                            bank * BROWS_,
                            pool_alias.storage_size(shp, jnp.dtype(dt)) // 128,
                        )
                    ],
                    shp,
                    dt,
                )

            def eview(name):
                r0, shp, dt = EPOOL["views"][name]
                return pool_alias.view(
                    scr["epool"].at[
                        pl.ds(r0, pool_alias.storage_size(shp, jnp.dtype(dt)) // 128)
                    ],
                    shp,
                    dt,
                )

            for nm in aliased_names(tiles):
                scr[nm] = eview(nm)
            _, shp13, _ = EPOOL["views"]["w13_0"]
            r2_0, shp2, _ = EPOOL["views"]["w2_0"]
            R13 = r2_0  # rows of a slot's w13 view (= its byte size / 128)
            SLOTR = (
                EPOOL["views"]["w13_1"][0]
                if NBE > 1
                else R13 + pool_alias.storage_size(shp2, jnp.dtype(FP8)) // 128
            )

            def er13v(slot):
                """slot's w13 view (traced slot): a dynamic row window of epool, as csq's dense ring views a bank."""
                return pool_alias.view(
                    scr["epool"].at[pl.ds(slot * SLOTR, R13)], shp13, FP8
                )

            def er2v(slot):
                return pool_alias.view(
                    scr["epool"].at[
                        pl.ds(
                            slot * SLOTR + R13,
                            pool_alias.storage_size(shp2, jnp.dtype(FP8)) // 128,
                        )
                    ],
                    shp2,
                    FP8,
                )
        else:

            def dview(kind, bank):
                return scr["dring_" + kind].at[bank]

        E_REFS = {}
        if stack_e:
            for (k, _), r in zip(e_names, e_in):
                E_REFS[k] = r
        else:
            for (k, i), r in zip(e_names, e_in):
                E_REFS.setdefault(k, {})[i] = r

        CS = {}
        for (k, gi), r in zip(c_names, c_in):
            CS.setdefault(k, []).append(r)

        def csel(kinds, l, fn):
            ng = len(CS[kinds[0]])
            assert all(len(CS[k]) == ng for k in kinds), kinds
            pos = CPOS[kinds[0]](l)
            if ng == 1:
                fn(*[CS[k][0].at[pos] for k in kinds])
            else:
                gs = len(ck[kinds[0]][0]) // ng
                bswitch(
                    pos // gs,
                    {
                        gi: (
                            lambda gi=gi: fn(
                                *[CS[k][gi].at[pos - gi * gs] for k in kinds]
                            )
                        )
                        for gi in range(ng)
                    },
                )

        me = scal[2]
        T = scal[0]
        if LV or LVD:  # rank-uniform (T is replicated metadata), so every byte-counting wait stays balanced
            lv_live = T <= TMAX  # LV: always true (T <= TMAX by contract); lv_dead runs the LV code under its negation
            lv_dead_p = jnp.logical_not(lv_live)
            n_ar = CO.lv_ar_rows(T, M, LV_FLOOR, tiles.lv_eight)
            nq_lv = jnp.minimum(jnp.maximum(jnp.where(T <= 4, 4, (T + 7) // 8 * 8), LV_FLOOR), TMAX)
            nch_lv = (T + (RW - 1)) // RW
        T_eff = (T - 1) if fault == "dropped_tail_row" else T
        row_m = lax.broadcasted_iota(I32, (M, 1), 0)
        rvalid = jnp.where(
            jnp.logical_and(rvec[:, 0:1] > 0, row_m < T_eff), 1.0, 0.0
        )  # [M, 1] f32
        lane128 = lax.broadcasted_iota(I32, (1, 128), 1)

        def sm(slot, name):
            o, n = lay[name]
            return scr["sm"][slot, :, o : o + n]

        # ---------------- dense ring (csq DenseWeightRing, SMEM stream table) ----------------
        def dsrc(kind, pos, t):
            r = Wd[kind]
            if kind in LY.WHOLE:
                return r.at[pos]
            tk = plan[kind][3]
            return r.at[pos, pl.ds(pl.multiple_of(t * tk, tk), tk)]

        def dcopy(kind, src, bank):
            return pltpu.make_async_copy(src, dview(kind, bank), scr["dsem"].at[bank])

        def fetch(g):
            if "weights" in skip:
                return

            @pl.when(g < T_STREAM)
            def _():
                d = dtab[jnp.minimum(g, T_STREAM - 1)]
                kc = lax.shift_right_logical(d, 20)
                pos = lax.shift_right_logical(d, 8) & 4095
                t = d & 255
                bank = lax.rem(g, NBD)
                for code, kind in enumerate(LY.DKINDS):
                    if kind not in Wd:
                        continue

                    @pl.when(kc == code)
                    def _(kind=kind):
                        dcopy(kind, dsrc(kind, pos, t), bank).start()

        def dtake(kind, g):
            bank = lax.rem(g, NBD)
            if "weights" not in skip:
                with sc("W_wait"):
                    dcopy(kind, dsrc(kind, 0, 0), bank).wait()
            return dview(kind, bank)

        def dgive(g):
            fetch(g + NBD)

        def dot8(a, w):
            a_, w_ = mxu(a, w, tiles)
            return lax.dot_general(a_, w_, NN, preferred_element_type=F32)

        def matmul(kind, g0, xref, nt=None):
            """acc[:, :N] = xref[:, :K] @ W_l (tiles g0 .. g0 + nt - 1 off the stream); returns acc[:, :N]."""
            shp, dt, ntl, tk = plan[kind]
            N = shp[-1]
            acc = scr["acc"]
            if ntl == 1:
                w_ = dtake(kind, g0)
                acc[:, :N] = dot8(xref[:, 0 : shp[0]], w_[...])
                dgive(g0)
                return acc[:, :N]
            acc[:, :N] = jnp.zeros((M, N), F32)

            def body(t, c):
                w_ = dtake(kind, g0 + t)
                k0 = pl.multiple_of(t * tk, tk)
                acc[:, :N] += dot8(xref[:, pl.ds(k0, tk)], w_[...])
                dgive(g0 + t)
                return c

            lax.fori_loop(0, ntl, body, 0)
            return acc[:, :N]

        # ---------------- collectives ----------------
        def ar_set(which):
            pre = "ar1_" if (which == 1 or fault == "f_xrace") else "ar2_"
            return {n: scr[pre + n] for n in CO.AR_NAMES} if R > 1 else {}

        def all_reduce(partial_f32, which, drop_last=False, early_read=False, mids=()):
            """bf16 partial [M, H] (rows >= T zeroed, moe_tp_decode all_reduce._masked) -> rank-order f32 sum ->
            bf16 values as f32."""
            if (
                tiles.replay == "ar"
            ):  # TIMING-ONLY: the recorded all-reduce output of this layer
                cp = pltpu.make_async_copy(
                    recs[0 if which == 1 else 1].at[cur_layer[0]],
                    scr["rbuf"],
                    scr["rep_sem"],
                )
                cp.start()
                cp.wait()
                return scr["rbuf"][...]
            buf = scr["arbuf"]
            buf[...] = jnp.where(row_m < T_eff, bf(partial_f32), 0.0).astype(BF16)
            if "coll" not in skip and LV_A and LV:
                with sc(f"X_ar{which}"):
                    # FAULT f_lv_short: AR#1 moves one 4-row block too few (n - 4 rows; when that is not a compiled
                    # option -- n = 4, or the bucket floor -- no option fires and AR#1 moves nothing, rank-uniformly)
                    short = fault == "f_lv_short" and which == 1
                    CO.all_reduce_lv(
                        R,
                        me,
                        buf,
                        M,
                        H,
                        ar_set(which),
                        n_ar - 4 if short else n_ar,
                        AR_OPTS,
                        drop_last=drop_last,
                        mids=mids,
                        loop=int(tiles.lv_loop),
                    )
            elif "coll" not in skip:
                if LV_A and LVD:  # TIMING-ONLY control: the LV all-reduce compiled, never run

                    @pl.when(lv_dead_p)
                    def _():
                        CO.all_reduce_lv(
                            R, me, buf, M, H, ar_set(which), n_ar, AR_OPTS, drop_last=drop_last, mids=mids,
                            loop=int(tiles.lv_loop),
                        )

                with sc(f"X_ar{which}"):
                    if mids:
                        CO.all_reduce(
                            R,
                            me,
                            buf,
                            M,
                            H,
                            ar_set(which),
                            drop_last=drop_last,
                            early_read=early_read,
                            mids=mids,
                        )
                    else:
                        CO.all_reduce(
                            R,
                            me,
                            buf,
                            M,
                            H,
                            ar_set(which),
                            drop_last=drop_last,
                            early_read=early_read,
                        )
            return buf[...].astype(F32)

        def exchange(src_fn, dst_fn, ss, rs, wait=True):
            if "coll" in skip:
                return None if wait else (lambda: None)
            return CO.exchange(
                R,
                me,
                src_fn,
                dst_fn,
                scr[ss],
                scr[rs],
                wait=wait,
                scope=sc if tiles.trace else None,
            )

        def a2a_chunk(c_):
            """L4 C2: rows [c_ * RW, (c_ + 1) * RW) of every owner's payload to that owner (static chunk index)."""
            r0_ = c_ * RW
            CO.start_all(
                R,
                me,
                lambda d: scr["osend"].at[d, pl.ds(r0_, RW)],
                lambda i: scr["orecv"].at[i, pl.ds(r0_, RW)],
                scr["a_send"],
                scr["a_recv"],
            )

        def qag_pair_start():
            """q AG, transport "pair" (stage-B L2-Q): start the R/2 sends (sibling + same-parity ranks); returns
            finish(): wait the same-parity blocks, forward them to the sibling, wait the sibling's own block and
            its forwards, then every send. Exact (copies); correctness does not depend on the chip mapping."""
            if "coll" in skip or R == 1:
                return lambda: None
            sib = me ^ 1
            par = me & 1
            peers = [((me >> 1) ^ k) * 2 + par for k in range(1, R // 2)]

            def rc(src, dst, ss, rs, d):
                pltpu.make_async_remote_copy(
                    src_ref=src,
                    dst_ref=dst,
                    send_sem=ss,
                    recv_sem=rs,
                    device_id=(d,),
                    device_id_type=CO.MESH,
                ).start()

            def qs(n):  # LV: the first n rows of the block (n None: the whole [TMAX, QH] block)
                return scr["qsend"] if n is None else scr["qsend"].at[pl.ds(0, n)]

            def qa(i, n):
                return scr["qall"].at[i] if n is None else scr["qall"].at[i, pl.ds(0, n)]

            def lvs(fn):
                if LV_Q and LV:
                    CO.row_switch(nq_lv, Q_OPTS, fn)
                elif LV_Q and LVD:  # TIMING-ONLY control: the LV sites compiled, never run

                    @pl.when(lv_dead_p)
                    def _():
                        CO.row_switch(nq_lv, Q_OPTS, fn)

                    fn(None)
                else:
                    fn(None)

            def start(n):
                rc(qs(n), qa(me, n), scr["q_send"], scr["q_recv"], sib)
                for p_ in peers:
                    rc(qs(n), qa(me, n), scr["q_send2"], scr["q_recv2"], p_)

            lvs(start)

            def desc(ss, rs, n):
                return pltpu.make_async_remote_copy(
                    src_ref=qs(n),
                    dst_ref=qa(0, n),
                    send_sem=ss,
                    recv_sem=rs,
                    device_id=(me,),
                    device_id_type=CO.MESH,
                )

            nf = len(peers) - (1 if fault == "f_qag_pair_drop" else 0)

            def forward_n(n):
                d2 = desc(scr["q_send2"], scr["q_recv2"], n)
                with sc("X_qpar"):
                    for _ in peers:
                        d2.wait_recv()
                for i, p_ in enumerate(peers):
                    if (
                        fault == "f_qag_pair_drop" and i == 0
                    ):  # FAULT: one forward omitted (sibling waits one less)
                        continue
                    rc(
                        qa(p_, n),
                        qa(p_, n),
                        scr["q_send"],
                        scr["q_recv"],
                        sib,
                    )

            def forward():
                lvs(forward_n)

            def finish_n(n):
                d2 = desc(scr["q_send2"], scr["q_recv2"], n)
                d1 = desc(scr["q_send"], scr["q_recv"], n)
                with sc("X_qsib"):
                    for _ in range(1 + nf):
                        d1.wait_recv()
                for _ in range(1 + nf):
                    d1.wait_send()
                for _ in peers:
                    d2.wait_send()

            def finish():
                if not tiles.qmid:
                    forward()
                lvs(finish_n)

            if tiles.qmid:
                finish.mid = forward  # called by the DSA pre-issue loop (exactly once)
            return finish

        cur_layer = [None]

        def delay_here(l, extra=True):
            """F-DELAY: rank scal[3] spins scal[4] iterations at layer scal[5] (< 0: every layer); runtime operands,
            so the clean program is the timed program (scal[3] = -1: never)."""

            @pl.when(
                jnp.logical_and(
                    jnp.logical_and(me == scal[3], extra),
                    jnp.logical_or(scal[5] < 0, scal[5] == l),
                )
            )
            def _():
                CO.spin(scal[4], scr["spin"])

        def delay_count(gi, r):
            """F-DELAY hook of the DSA count exchange: the delayed rank stalls before its round-1 send of group 0,
            so its peers run ahead into the next parity (the `sharedsem` hazard)."""
            delay_here(cur_layer[0], extra=jnp.logical_and(gi == 0, r == 1))

        # ---------------- expert ring (csq / moe_tp_decode) ----------------
        lshift = (
            1 if fault == "f_leaf_shift" else 0
        )  # FAULT: the operand leaf / index is one MoE layer off

        def esel(mpos, fn):
            if stack_e and not tiles.edummy:
                mp = lax.rem(mpos + lshift, Lm) if lshift else mpos
                fn(
                    E_REFS["ew13"].at[mp],
                    E_REFS["ew2"].at[mp],
                    E_REFS["es13"].at[mp],
                    E_REFS["es2"].at[mp],
                )
            else:

                def leaf(i):
                    j = (i + lshift) % Lm
                    if stack_e:  # edummy: the served cascade, static leaves over the stacked operands
                        fn(
                            E_REFS["ew13"].at[j],
                            E_REFS["ew2"].at[j],
                            E_REFS["es13"].at[j],
                            E_REFS["es2"].at[j],
                        )
                    else:
                        fn(
                            E_REFS["ew13"][j],
                            E_REFS["ew2"][j],
                            E_REFS["es13"][j],
                            E_REFS["es2"][j],
                        )

                sw = bswitch if tiles.ecas == "when2" else bswitch_cond
                sw(mpos, {i: (lambda i=i: leaf(i)) for i in range(Lm)})

        def ecopies(w13r, w2r, s13r, s2r, w, slot):
            d13, d2 = (
                (scr["er13"].at[slot], scr["er2"].at[slot])
                if typed
                else (er13v(slot), er2v(slot))
            )
            return [
                pltpu.make_async_copy(w13r.at[w], d13, scr["ring_sem"].at[slot, 0]),
                pltpu.make_async_copy(w2r.at[w], d2, scr["ring_sem"].at[slot, 1]),
                pltpu.make_async_copy(
                    s13r.at[w, 0],
                    scr["es13"].at[pl.ds(slot, 1)],
                    scr["ring_sem"].at[slot, 2],
                ),
                pltpu.make_async_copy(
                    s2r.at[w, 0],
                    scr["es2"].at[pl.ds(slot, 1)],
                    scr["ring_sem"].at[slot, 3],
                ),
            ]

        def estart(step, mpos):
            w = scr["active"][step]
            if fault == "f_ichunk":  # FAULT: the unit loads another expert's slice
                w = lax.rem(w + 1, E)
            slot = lax.rem(step, NBE)

            def arm(w13r, w2r, s13r, s2r):
                for cp in ecopies(w13r, w2r, s13r, s2r, w, slot):
                    cp.start()

            esel(mpos, arm)

        def estart_range(lo, hi, mpos):
            """L3-b: issue units [lo, hi) of the current MoE layer under ONE operand selection (cascade)."""

            def arm(w13r, w2r, s13r, s2r):
                def one(u, c):
                    w = scr["active"][u]
                    if (
                        fault == "f_ichunk"
                    ):  # FAULT: the unit loads another expert's slice
                        w = lax.rem(w + 1, E)
                    for cp in ecopies(w13r, w2r, s13r, s2r, w, lax.rem(u, NBE)):
                        cp.start()
                    return c

                lax.fori_loop(lo, hi, one, 0)

            esel(mpos, arm)

        def ewait(step):
            slot = lax.rem(step, NBE)
            r13, r2, r13s, r2s = (
                E_REFS["ew13"],
                E_REFS["ew2"],
                E_REFS["es13"],
                E_REFS["es2"],
            )
            if not stack_e:
                r13, r2, r13s, r2s = r13[0], r2[0], r13s[0], r2s[0]
            else:
                r13, r2, r13s, r2s = r13.at[0], r2.at[0], r13s.at[0], r2s.at[0]
            for cp in ecopies(r13, r2, r13s, r2s, 0, slot):
                cp.wait()

        def w13_of(slot, k0, k1):
            """rows [k0, k1) of the slot's w13 tile (traced slot)."""
            if typed:
                return scr["er13"][slot, k0:k1, :]
            return er13v(slot)[k0:k1, :]

        def w2_of(slot, n0, n1):
            if typed:
                return scr["er2"][slot, :, n0:n1]
            return er2v(slot)[:, n0:n1]

        # ---------------- KV staging (glm5_mk_dcp stage_rows; static k_scale divides as main writes) -------------
        def stage_rows(l, slot):
            ks = sm(slot, "k_scale")[:, 0:1]
            kvsrc = scr["kvc"][0:TMAX]
            if (
                fault == "f_nope"
            ):  # FAULT F-NOPE: the f32 norm product without its bf16 rounding
                kvsrc = scr["fnope"][0:TMAX]
            nb8 = fp8_bytes(jnp.clip(kvsrc / ks, -FP8_MAX, FP8_MAX).astype(FP8))
            word = (
                nb8[:, 0:128]
                | shl(nb8[:, 128:256], 8)
                | shl(nb8[:, 256:384], 16)
                | shl(nb8[:, 384:512], 24)
            )
            scr["wrow"][...] = lax.bitcast_convert_type(word, U32)
            rb = fp8_bytes(
                jnp.clip(scr["kpe"][0:TMAX] / ks, -FP8_MAX, FP8_MAX).astype(FP8)
            )
            lane_t = lax.broadcasted_iota(I32, (TMAX, 128), 1)
            rw = jnp.where(lane_t < 32, rb | shl(pltpu.roll(rb, 96, 1), 8), 0)
            seg = rvec[0:TMAX, 1:2] & 3
            rq = jnp.zeros((TMAX, 128), I32)
            for sg in range(4):
                rq = jnp.where(
                    (seg == sg) & (lane_t // 32 == sg),
                    pltpu.roll(rw, 32 * sg, 1) if sg else rw,
                    rq,
                )
            scr["rwrow"][...] = lax.bitcast_convert_type(rq, U32)
            pltpu.make_async_copy(
                scr["wrow"], nrow_out.at[l], scr["kvw_sem"].at[0]
            ).start()
            pltpu.make_async_copy(
                scr["rwrow"], rrow_out.at[l], scr["kvw_sem"].at[1]
            ).start()

        def stage_wait(l):
            pltpu.make_async_copy(
                scr["wrow"], nrow_out.at[l], scr["kvw_sem"].at[0]
            ).wait()
            pltpu.make_async_copy(
                scr["rwrow"], rrow_out.at[l], scr["kvw_sem"].at[1]
            ).wait()

        # ---------------- indexer front (glm5_mk_dcp indexer_proj) ----------------
        def indexer_proj(l, slot, g0):
            xq, sx = quant_rows(scr["qc"][...])
            scr["xq"][:, :QL] = xq
            a = matmul("iwq", g0, scr["xq"])
            iq = bf(a * sx * sm(slot, "idx_wq_s"))
            a2 = matmul("iwk", g0 + plan["iwq"][2], scr["xbf"])
            kw = bf(a2)
            wts = jnp.zeros((M, 128), F32)
            for h_ in range(IH):
                v = rope(
                    iq[:, h_ * IDIM : (h_ + 1) * IDIM],
                    rope_tab[0],
                    rope_tab[1],
                    cfg.rope_neox,
                )
                s = bf(jnp.max(jnp.abs(v), axis=1, keepdims=True) * (1.0 / FP8_MAX))
                inv = jnp.where(s == 0, 0.0, bf(1.0 / jnp.where(s == 0, 1.0, s)))
                qv = jnp.clip(bf(v * inv), -FP8_MAX, FP8_MAX).astype(FP8).astype(F32)
                scr["iqv"][h_, 0:M, :] = qv
                wcol = bf(
                    ((kw[:, IDIM + h_ : IDIM + h_ + 1] * s) * (IDIM**-0.5)) * (IH**-0.5)
                )
                wts = jnp.where(lane128 == h_, wcol, wts)
            scr["iqv"][:, M : M + QB8, :] = jnp.zeros((IH, QB8, 128), F32)
            scr["wts"][0:M, :] = wts
            scr["wts"][M : M + QB8, :] = jnp.zeros((QB8, 128), F32)
            kx = kw[:, 0:IDIM]
            mu = jnp.mean(kx, axis=1, keepdims=True)
            var = jnp.mean((kx - mu) ** 2, axis=1, keepdims=True)
            kn = bf(
                (kx - mu) * lax.rsqrt(var + cfg.idx_eps) * sm(slot, "idx_kn_w")
                + sm(slot, "idx_kn_b")
            )
            kr = rope(kn, rope_tab[0], rope_tab[1], cfg.rope_neox)
            s = jnp.max(jnp.abs(kr), axis=1, keepdims=True) * (1.0 / FP8_MAX)
            sb = lax.bitcast_convert_type(s, I32)
            e = (shr(sb, 23) & 0xFF) - 127
            e = jnp.where((sb & 0x7FFFFF) != 0, e + 1, e)
            e = jnp.where(s == 0, -127, e)
            e = jnp.clip(e, -127, 127)
            scl = lax.bitcast_convert_type(shl(jnp.clip(e, -126, 127) + 127, 23), F32)
            scl = jnp.where(e == -127, 2.0**-127, scl)
            k8 = fp8_bytes(jnp.clip(kr / scl, -FP8_MAX, FP8_MAX).astype(FP8))
            rec = jnp.concatenate(
                [
                    k8,
                    jnp.where(lax.broadcasted_iota(I32, (M, 128), 1) == 0, e + 127, 0),
                ],
                axis=1,
            )
            scr["rec_v"][...] = rec[0:TMAX]
            pltpu.make_async_copy(
                scr["rec_v"], irec_out.at[l], scr["kvw_sem"].at[2]
            ).start()

        # ---------------- LSE merge (glm5_mk_dcp merge: main cp_mla_attention.py:148-186) ----------------
        def merge():
            MCH = min(TMAX, 16)
            scr["mvalid"][...] = jnp.broadcast_to(rvalid, (M, 128))
            if LSE_TILE:
                scr["lse_t"][...] = jnp.transpose(scr["lrecv"][...])

            def body(c, cc):
                r0 = pl.multiple_of(c * MCH, MCH)

                def lse_of(s_):
                    if (
                        LSE_TILE
                    ):  # lanes hh < HR = source s_'s heads (lanes >= HR are unused)
                        x_ = scr["lse_t"][pl.ds(r0, MCH), :]
                        sh_ = (128 - HPAD * s_) % 128
                        return pltpu.roll(x_, sh_, 1) if sh_ else x_
                    return lax.bitcast_convert_type(
                        scr["orecv"][s_, pl.ds(r0, MCH), OH : OH + 128], F32
                    )

                shift = jnp.full((MCH, 128), -jnp.inf, F32)
                for s_ in range(R):
                    x_ = lse_of(s_)
                    shift = jnp.maximum(
                        shift, jnp.where(jnp.isfinite(x_), x_, -jnp.inf)
                    )
                shift = jnp.where(jnp.isfinite(shift), shift, 0.0)

                def w_of(s_):
                    x_ = lse_of(s_)
                    return jnp.where(jnp.isfinite(x_), jnp.exp(x_ - shift), 0.0)

                tot = w_of(0)
                for s_ in range(1, R):
                    tot = tot + w_of(s_)
                tot = jnp.where(tot > 0, tot, 1.0)
                hi = [jnp.zeros((MCH, KL), F32) for _ in range(HR)]
                lo = [jnp.zeros((MCH, KL), F32) for _ in range(HR)]
                for s_ in range(R):
                    blk = scr["orecv"][s_, pl.ds(r0, MCH), 0:OH]
                    o_all = jnp.concatenate([unpack_lo(blk), unpack_hi(blk)], axis=1)
                    w_s = w_of(s_) / tot
                    for hh in range(HR):
                        o_ = o_all[:, hh * KL : (hh + 1) * KL]
                        w_ = w_s[:, hh : hh + 1]
                        wh = bf(w_)
                        hi[hh] = hi[hh] + o_ * wh
                        lo[hh] = lo[hh] + o_ * (w_ - wh)
                valid_t = scr["mvalid"][pl.ds(r0, MCH), 0:1]
                for hh in range(HR):
                    scr["oh"][hh, pl.ds(r0, MCH), :] = bf(hi[hh] + lo[hh]) * valid_t
                return cc

            lax.fori_loop(0, TMAX // MCH, body, 0)

        # ---------------- debug export ----------------
        def dput(dst, src):
            if not debug:
                return
            cp = pltpu.make_async_copy(src, dst, scr["dbg_sem"].at[0])
            cp.start()
            cp.wait()

        def dput_val(dst, val, stage):
            if not debug:
                return
            stage[...] = val
            dput(dst, stage)

        # ---------------- the batched DSA core ----------------
        core = {}
        cur_slot = [None]
        Ens = types.SimpleNamespace(
            cfg=cfg,
            tiles=tiles,
            scr=scr,
            max_pages=max_pages,
            BMAX=BMAX,
            me=me,
            scal=scal,
            rmeta=rmeta,
            qmeta=qmeta,
            bt=bt,
            bti=bti,
            rvec=rvec,
            skip=frozenset(x for x in skip if x in ("index", "gather", "attn", "coll")),
            MB_SH=MB_SH,
            SRC_MASK=SRC_MASK,
            QA=QA,
            QH=QH,
            OH=OH,
            sm_scale=cfg.sm_scale,
            Phases=Phases,
            csel=csel,
            mxu=mxu,
            fp8_decode_hw=fp8_decode_hw,
            mono_key=mono_key,
            pack2=pack2,
            unpack_lo=unpack_lo,
            unpack_hi=unpack_hi,
            bf=bf,
            misc_sem=scr["misc_sem"],
            wrow=scr["wrow"],
            rwrow=scr["rwrow"],
            M=M,
            debug=False,
            nb=nb,
            ks=lambda: sm(cur_slot[0], "k_scale")[:, 0:1],
            grec_in=(RI["gnrec"], RI["grrec"]) if "g" in tiles.l2replay else None,
            grec_out=(RO["gnrec_o"], RO["grrec_o"]) if tiles.record else None,
            rec_sem=scr["rec_sem"] if (REC_IN or REC_OUT) else None,
            dbg_uv=scr["dbg_uv"] if debug else None,
            delay=delay_count,
            Lf=Lf,
            xin=x_in,
            xout=x_out if xout_names else None,
        )
        core.update(DSA.make(Ens))

        def save_table():
            """FAULT stale_indexshare: snapshot the IndexShare table before a fresh layer replaces it."""
            g_ = GEO
            scr["sv_uv"][...] = scr["own_uv"][...]

            def cu(i, c):
                for j in range(128):
                    scr["sv_u"][i, j] = scr["own_u"][i, j]
                scr["sv_greq"][i] = scr["b_greq"][i]
                return c

            lax.fori_loop(0, g_["UR2"], cu, 0)
            for b_ in range(BMAX):
                for k_ in range(16):
                    scr["sv_req"][b_, k_] = scr["b_req"][b_, k_]
            scr["sv_st"][0] = scr["b_st"][0]

        def restore_table():
            g_ = GEO
            scr["own_uv"][...] = scr["sv_uv"][...]

            def cu(i, c):
                for j in range(128):
                    scr["own_u"][i, j] = scr["sv_u"][i, j]
                scr["b_greq"][i] = scr["sv_greq"][i]
                return c

            lax.fori_loop(0, g_["UR2"], cu, 0)
            for b_ in range(BMAX):
                for k_ in range(16):
                    scr["b_req"][b_, k_] = scr["sv_req"][b_, k_]
            scr["b_st"][0] = scr["sv_st"][0]

        # ---------------- one layer ----------------
        def layer(l, carry):
            """One layer under the per-layer scope 'L:layer' (trace build). A traced layer index cannot name a
            static scope, so layer i is the i-th 'L:layer' occurrence of a step; the kind shows as the inner
            'L_index' (fresh), 'D_mlp' (dense) or 'M_route' (MoE) scope."""
            with sc("L:layer"):
                return layer_body(l, carry)

        def layer_body(l, carry):
            g0 = ltab[l, 0]
            fresh = ltab[l, 1] > 0
            is_moe = ltab[l, 2] > 0
            mpos = ltab[l, 3]
            slot = l % 2
            cur_slot[0] = slot
            cur_layer[0] = l
            pltpu.make_async_copy(
                smalls.at[0], scr["sm"].at[slot], scr["sm_sem"].at[slot]
            ).wait()

            @pl.when(l + 1 < L)
            def _():
                pltpu.make_async_copy(
                    smalls.at[l + 1], scr["sm"].at[1 - slot], scr["sm_sem"].at[1 - slot]
                ).start()

            if debug:
                dput(d_state.at[l, 0], scr["hcur"])
                dput(d_state.at[l, 1], scr["res"])
            # ===== fused add-RMSNorm (vLLM fused_add_rms_norm; layer 0: res = 0, so x = h) =====
            with sc("L_norm"):
                x = scr["hcur"][...] + scr["res"][...]
                scr["res"][...] = bf(x)
                xn = rmsnorm(x, sm(slot, "in_ln"), cfg.eps) * rvalid
                scr["xbf"][...] = xn.astype(BF16)
                if debug:
                    dput_val(d_xin.at[l], xn, scr["dbg_x"])

            # ===== qkv_a, q_a / kv_a norms, k_pe rope =====
            with sc("L_qkva"):
                xq, sx = quant_rows(xn)
                scr["xq"][:, :H] = xq
                a = matmul("qkva", g0, scr["xq"])
                o_s, _ = lay["qkv_a_s"]
                y = bf(a * sx * scr["sm"][slot, :, o_s : o_s + cfg.QKVP])
                qc = rmsnorm(y[:, :QL], sm(slot, "q_a_ln"), cfg.eps) * rvalid
                kvn = y[:, QL : QL + KL]
                kvc = rmsnorm(kvn, sm(slot, "kv_a_ln"), cfg.eps) * rvalid
                if fault == "f_nope":  # the norm product before its last bf16 rounding
                    yk = bf(
                        kvn
                        * lax.rsqrt(
                            jnp.mean(kvn * kvn, axis=1, keepdims=True) + cfg.eps
                        )
                    )
                    scr["fnope"][...] = yk * sm(slot, "kv_a_ln") * rvalid
                o_pe = QL + KL
                kpe_raw = y[:, o_pe : o_pe + 128]
                kpe_raw = jnp.where(
                    lax.broadcasted_iota(I32, (M, 128), 1) < cfg.DR, kpe_raw, 0.0
                )
                kpe = rope(kpe_raw, rope_tab[0], rope_tab[1], cfg.rope_neox) * rvalid
                scr["qc"][...] = qc
                scr["kvc"][...] = kvc
                scr["kpe"][...] = kpe
                if debug:
                    dput(d_qc.at[l], scr["qc"])
                    dput(d_kvcn.at[l], scr["kvc"])

            # ===== q_b on this rank's heads (qrep 0), rope, W_UK absorb -> qsend; q AG start =====
            with sc("L_qb"):
                xq2, sx2 = quant_rows(qc)
                scr["xq"][:, :QL] = xq2
                a = matmul("qb", g0 + plan["qkva"][2], scr["xq"])
                o_qb, _ = lay["q_b_s"]
                q = bf(a * sx2 * scr["sm"][slot, :, o_qb : o_qb + NQB])
                g_uk = g0 + plan["qkva"][2] + plan["qb"][2]
                wb = dtake("wuk", g_uk)
                parts = []
                for hh in range(HR):
                    qn = q[:, hh * DQ : hh * DQ + DN].astype(BF16)
                    ql = bf(lax.dot_general(qn, wb[hh], NN, preferred_element_type=F32))
                    win = q[:, hh * DQ + DQ - 128 : hh * DQ + DQ]
                    wr = rope(win, rope_tab[2], rope_tab[3], cfg.rope_neox)
                    qpe = jnp.where(
                        lax.broadcasted_iota(I32, (M, 128), 1) < cfg.DR,
                        pltpu.roll(wr, cfg.DR, 1),
                        0.0,
                    )
                    parts += [ql[0:TMAX], qpe[0:TMAX]]
                dgive(g_uk)
                qrow = jnp.concatenate(parts, axis=1)
                scr["qsend"][...] = pack2(qrow[:, :QH], qrow[:, QH:])
                scr["qall"][me] = scr["qsend"][...]
                if (
                    "q" in tiles.l2replay
                ):  # backing arm q_replay: qall from the record (bit-equal)
                    qcp = pltpu.make_async_copy(
                        RI["qrec"].at[l], scr["qall"], scr["rec_sem"].at[0]
                    )
                    qcp.start()
                    q_wait = qcp.wait
                elif tiles.qag == "pair":
                    q_wait = qag_pair_start()
                else:
                    q_wait = exchange(
                        lambda d: scr["qsend"],
                        lambda i: scr["qall"].at[i],
                        "q_send",
                        "q_recv",
                        wait=False,
                    )

            # ===== this layer's new KV rows -> staging (the q AG flies meanwhile) =====
            with sc("L_kvw"):
                stage_rows(l, slot)

            g_idx = g_uk + 1
            n_idx = plan["iwq"][2] + plan["iwk"][2]

            # ===== indexer front + batched index path (fresh layers) =====
            if fault == "stale_indexshare":

                @pl.when(jnp.logical_and(fresh, l > full_l[0]))
                def _():
                    save_table()

            @pl.when(fresh)
            def _():
                with sc("L_index"):
                    indexer_proj(l, slot, g_idx)
                    core["indexer_tail"](l)
                    pltpu.make_async_copy(
                        scr["rec_v"], irec_out.at[l], scr["kvw_sem"].at[2]
                    ).wait()
                if debug:
                    fp = jnp.clip(
                        jnp.where(
                            l < cfg.idx_skip,
                            l,
                            cfg.idx_skip + (l - cfg.idx_skip) // cfg.idx_freq,
                        ),
                        0,
                        Lf - 1,
                    )
                    dput(d_uv.at[fp], scr["dbg_uv"])
                    dput(d_ureq.at[fp], scr["b_req"])

            # ===== batched attention (IndexShare layers issue the gather ring before the q AG wait) =====
            after_req = None
            if RW and "coll" not in skip and tiles.replay != "a2a":
                scr["a2a_n"][0] = jnp.int32(0)

                def after_req(b):  # L4 C2: start every chunk whose rows are all emitted
                    done = qmeta[b, 0] + qmeta[b, 1]
                    if (
                        fault == "f_a2a_early"
                    ):  # FAULT: one request early (its rows not yet emitted)
                        done = done + cfg.QMAX
                    for c_ in range(NCH - 1):

                        @pl.when(
                            jnp.logical_and(
                                scr["a2a_n"][0] == c_, done >= (c_ + 1) * RW
                            )
                        )
                        def _(c_=c_):
                            a2a_chunk(c_)
                            scr["a2a_n"][0] = jnp.int32(c_ + 1)

            with sc("L_attn"):
                if GEARLY:
                    prev_moe = jnp.logical_and(
                        l >= 1, ltab[jnp.maximum(l - 1, 0), 2] > 0
                    )
                    core["attention"](
                        l,
                        fresh,
                        q_wait,
                        after_req=after_req,
                        early=jnp.logical_and(prev_moe, jnp.logical_not(fresh)),
                    )
                else:
                    core["attention"](l, fresh, q_wait, after_req=after_req)
            if fault == "stale_indexshare":

                @pl.when(jnp.logical_and(fresh, l > full_l[0]))
                def _():
                    restore_table()

            if debug:
                dput(d_qall.at[l], scr["qall"])
            if tiles.record:
                rcp = pltpu.make_async_copy(
                    scr["qall"], RO["qrec_o"].at[l], scr["rec_sem"].at[0]
                )
                rcp.start()
                rcp.wait()

            # ===== A2A partials + lse to head owners, LSE merge =====
            with sc("L_merge"):
                if (
                    tiles.replay == "a2a"
                ):  # TIMING-ONLY: the recorded merged oh of this layer
                    cp = pltpu.make_async_copy(recs[0].at[l], scr["oh"], scr["rep_sem"])
                    cp.start()
                    cp.wait()
                else:
                    scr["orecv"][me] = scr["osend"][me]
                    if not LSE_TILE:  # the stage-A exchange, verbatim
                        exchange(
                            lambda d: scr["osend"].at[d],
                            lambda i: scr["orecv"].at[i],
                            "a_send",
                            "a_recv",
                        )
                    else:
                        r8 = pl.multiple_of(me * HPAD, 8)
                        scr["lrecv"][pl.ds(r8, HPAD), :] = scr["lsend"][
                            pl.ds(r8, HPAD), :
                        ]
                        if "coll" not in skip:
                            if RW:
                                for c_ in range(
                                    NCH
                                ):  # the chunks not started inside the attention loop

                                    tail = scr["a2a_n"][0] <= c_
                                    if LV_X:  # padded chunks (no live row) are not sent
                                        tail = jnp.logical_and(tail, c_ < nch_lv)

                                    @pl.when(tail)
                                    def _(c_=c_):
                                        a2a_chunk(c_)
                            else:
                                CO.start_all(
                                    R,
                                    me,
                                    lambda d: scr["osend"].at[d],
                                    lambda i: scr["orecv"].at[i],
                                    scr["a_send"],
                                    scr["a_recv"],
                                )
                            CO.start_all(
                                R,
                                me,
                                lambda d: scr["lsend"].at[pl.ds(d * HPAD, HPAD)],
                                lambda i: scr["lrecv"].at[
                                    pl.ds(pl.multiple_of(i * HPAD, 8), HPAD)
                                ],
                                scr["a_send"],
                                scr["a_recv"],
                            )
                            rw_ = RW or TMAX
                            CO.wait_pieces(
                                R,
                                me,
                                [
                                    (
                                        scr["osend"].at[0, pl.ds(0, rw_)],
                                        scr["orecv"].at[0, pl.ds(0, rw_)],
                                        nch_lv if LV_X else NCH,
                                    ),
                                    (
                                        scr["lsend"].at[pl.ds(0, HPAD)],
                                        scr["lrecv"].at[pl.ds(0, HPAD)],
                                        1,
                                    ),
                                ],
                                scr["a_send"],
                                scr["a_recv"],
                                scope=sc if tiles.trace else None,
                            )
                    if (
                        tiles.shadow_a2a and "coll" not in skip
                    ):  # TIMING-ONLY: the same bytes again, unconsumed
                        CO.exchange(
                            R,
                            me,
                            lambda d: scr["osend"].at[d],
                            lambda i: scr["orecv"].at[i],
                            scr["a2_send"],
                            scr["a2_recv"],
                            scope=sc if tiles.trace else None,
                        )
                    merge()
                if debug:
                    dput(d_oloc.at[l], scr["oh"])

            # ===== W_UV + o_proj (local K, per-rank scale) -> AR#1 =====
            g_uv = g_idx + jnp.where(fresh, n_idx, 0)
            with sc("L_oproj"):
                wv = dtake("wuv", g_uv)
                for hh in range(HR):
                    y_ = lax.dot_general(
                        scr["oh"][hh].astype(BF16),
                        wv[hh],
                        NN,
                        preferred_element_type=F32,
                    )
                    scr["orow"][:, hh * DV : (hh + 1) * DV] = bf(y_)
                dgive(g_uv)
                xq3, sx3 = quant_rows(scr["orow"][...])
                scr["xq"][:, :KR] = xq3
                a = matmul("wo", g_uv + 1, scr["xq"])
                part1 = bf(a * sx3 * sm(slot, "o_proj_s"))
            delay_here(l)  # F-DELAY: skew into the layer's AR#1
            attn = all_reduce(
                part1,
                1,
                drop_last=(fault == "missing_rank_partial"),
                early_read=(fault == "f_ar_earlyread"),
            )
            scr["attn"][...] = attn
            if debug:
                dput(d_attn.at[l], scr["attn"])

            # ===== post-attention fused add-RMSNorm =====
            with sc("L_postnorm"):
                x2 = scr["attn"][...] + scr["res"][...]
                scr["res"][...] = bf(x2)
                xn2 = rmsnorm(x2, sm(slot, "post_ln"), cfg.eps) * rvalid
                scr["xbf"][...] = xn2.astype(BF16)
                if debug:
                    dput_val(d_xffn.at[l], xn2, scr["dbg_x"])

            g_ffn = g_uv + 1 + plan["wo"][2]

            # ===== dense MLP (layers < first_dense; TP slice) =====
            if cfg.first_dense > 0:

                @pl.when(jnp.logical_not(is_moe))
                def _dense():
                    with sc("D_mlp"):
                        xn_ = scr["xbf"][...].astype(F32)
                        xqd, sxd = quant_rows(xn_)
                        scr["xq"][:, :H] = xqd
                        a_ = matmul("dw13", g_ffn, scr["xq"])
                        gu = bf(a_ * sxd * sm(slot, "w13_s"))
                        mid = bf(gu[:, :IDR] * jax.nn.sigmoid(gu[:, :IDR]))
                        mid = bf(mid * gu[:, IDR:])
                        xq4, sx4 = quant_rows(mid)
                        scr["xq"][:, :IDR] = xq4
                        a_ = matmul("dw2", g_ffn + plan["dw13"][2], scr["xq"])
                        scr["part"][...] = bf(a_ * sx4 * sm(slot, "w2_s"))
                    scr["hcur"][...] = all_reduce(scr["part"][...], 2)

            # ===== MoE: router, hit list, shared expert, TP16 routed experts (moe_tp_decode), combine, AR#2 =====
            if cfg.first_dense < L:

                @pl.when(is_moe)
                def _moe():
                    moe(l, slot, g_ffn, mpos)

            stage_wait(l)
            if fault in POISON_FAULTS:
                if GEARLY:  # MoE layers poisoned before their AR#2 (moe())

                    @pl.when(jnp.logical_not(is_moe))
                    def _():
                        poison_alias()
                else:
                    poison_alias()
            return carry

        def poison_alias():
            """F-ALIAS-POISON (alias layout only): after each layer's MoE, the whole alias window holds 0x7E bytes
            (finite fp8 448 / f32 8.4e37), so a buffer wrongly assumed persistent reads garbage. Must stay
            bit-identical to clean; its pairing F-ALIAS-POISON-RED also places own_uv in the window -> RED.
            F-ALIAS-POISON-NAN writes 0xFF (the e4m3fn NaN code; f32 NaN) instead: a partial gather granule's tail
            rows then hold NaN codes, which the DSA's tail-row zeroing must keep out of P.V (identical to clean);
            its pairing ..._NOGUARD also drops that zeroing -> RED."""
            if typed:
                return
            n_ = EPOOL["dsa_bytes"] // 128
            byte = 0xFF if fault.startswith("f_alias_poison_nan") else 0x7E
            scr["epool"][0:n_, :] = jnp.full((n_, 128), byte, U8)

        def moe(l, slot, g_ffn, mpos):
            with sc("M_route"):
                xb2 = scr["xbf"][...]
                g_gate = g_ffn
                wg = dtake("gate", g_gate)
                logits = bf(
                    lax.dot_general(xb2, wg[...], NN, preferred_element_type=F32)
                )
                scores = jax.nn.sigmoid(logits)
                bias = sm(slot, "gate_bias")
                scr["wtab"][...] = route(scores, bias, T_eff)
                if debug:
                    dput(d_wtab.at[l], scr["wtab"])
                hit = jnp.any(scr["wtab"][...] != 0, axis=0, keepdims=True)
                scr["hits_v"][...] = jnp.where(hit, 1, 0).astype(I32)
                hc = pltpu.make_async_copy(
                    scr["hits_v"], scr["hits_s"], scr["hit_sem"].at[0]
                )
                hc.start()
                hc.wait()

                def compact(e, cnt):
                    h_ = scr["hits_s"][0, e] != 0
                    scr["active"][cnt] = e
                    return cnt + h_.astype(I32)

                count = lax.fori_loop(0, E, compact, jnp.int32(0))
                if (
                    fault == "f_hitcap"
                ):  # FAULT: hit list truncated one short of all experts
                    count = jnp.minimum(count, E - 1)
                if "experts" in skip:
                    count = jnp.int32(0)

                if ES:
                    pass  # L3-s: primed by the loop's step -1 (one site)
                elif KB == 1:

                    def prime(st0, c):
                        estart(st0, mpos)
                        return c

                    lax.fori_loop(0, jnp.minimum(count, NBE), prime, 0)
                else:  # L3-b: the whole prime under one selection

                    @pl.when(count > 0)
                    def _():
                        estart_range(jnp.int32(0), jnp.minimum(count, NBE), mpos)

            def shared_part():
                with sc("M_shared"):
                    w13s = dtake("sh13", g_ffn + plan["gate"][2])
                    w2s = dtake("sh2", g_ffn + plan["gate"][2] + plan["sh13"][2])
                    if fault != "stale_accumulator":
                        scr["owner"][...] = jnp.zeros((M, H), F32)
                    else:  # FAULT: the routed accumulator is never cleared

                        @pl.when(mpos == 0)
                        def _():
                            scr["owner"][...] = jnp.zeros((M, H), F32)

                    xb2f = xb2.astype(F32)
                    sxq, sxs = quantize_block(xb2f, axis=1, target_dtype=FP8)
                    gu = dot8(sxq, w13s[...])
                    gu = (gu * sxs * sm(slot, "sw13_s")).astype(BF16).astype(F32)
                    act = (
                        jax.nn.silu(gu[:, :ISR]).astype(BF16).astype(F32) * gu[:, ISR:]
                    )
                    aq, asc = quantize_block(act.astype(BF16), axis=1, target_dtype=FP8)
                    for n0 in range(0, H, 512):
                        n1 = min(H, n0 + 512)
                        yv = dot8(aq, w2s[:, n0:n1])
                        scr["shared"][:, n0:n1] = yv * asc * sm(slot, "sw2_s")[:, n0:n1]
                    mtk._quantize_blocks(xb2, scr["xqb"], scr["xsb"])

            if not ES:
                shared_part()

            lane_e = lax.broadcasted_iota(I32, (M, EPAD), 1)
            with sc("M_exp"):

                def gate_up(eslot):
                    w_scale = scr["es13"][pl.ds(eslot, 1), :].astype(BF16)
                    a1 = jnp.zeros((M, 2 * IR), BF16)
                    for k0 in range(0, H, QB):
                        part = dot8(
                            scr["xqb"][:, k0 : k0 + QB], w13_of(eslot, k0, k0 + QB)
                        ).astype(BF16)
                        part *= jnp.concatenate(
                            [scr["xsb"][k0 // QB]] * (2 * IR // 128), axis=1
                        )
                        part *= w_scale
                        a1 += part
                    return a1

                def finish(step, eslot, a1):
                    e = scr["active"][step]
                    coeff = jnp.sum(
                        jnp.where(lane_e == e, scr["wtab"][...], 0.0),
                        axis=1,
                        keepdims=True,
                    )
                    act_ = jax.nn.silu(a1[:, :IR]) * a1[:, IR:]
                    mid = jnp.where(coeff != 0, act_.astype(F32) * coeff, 0.0)
                    ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / FP8_MAX
                    mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(FP8)
                    for n0 in range(0, H, 512):
                        n1 = min(H, n0 + 512)
                        a2 = dot8(mq, w2_of(eslot, n0, n1))
                        scr["owner"][:, n0:n1] += (
                            a2 * scr["es2"][pl.ds(eslot, 1), n0:n1] * ms
                        )

                def refill(step, iss):
                    """L3-b: units < avail may be issued (their slots' previous units have finished); issue
                    them once >= KB are pending, or when the layer's last unit is issuable (flush). With
                    NBE - KB >= 3 every unit is issued >= NBE - KB steps before its wait."""
                    avail = jnp.minimum(step + NBE, count - 1) + 1
                    pend = avail - iss
                    go = jnp.logical_or(
                        pend >= KB, jnp.logical_and(avail == count, pend > 0)
                    )

                    @pl.when(go)
                    def _():
                        estart_range(iss, avail, mpos)

                    return jnp.where(go, avail, iss)

                def when_stage_a(c):  # L3-s replaces the whole block below
                    return (lambda f: None) if ES else pl.when(c)

                @when_stage_a(count > 0)
                def _():
                    ewait(jnp.int32(0))

                    def body(step, a1):
                        nxt = step + 1

                        @pl.when(nxt < count)
                        def _():
                            ewait(nxt)

                        # the last unit has no successor: skip its gate_up (was one wasted unit per MoE layer,
                        # reading a stale slot)
                        a1n = lax.cond(
                            nxt < count,
                            lambda: gate_up(lax.rem(nxt, NBE)),
                            lambda: jnp.zeros((M, 2 * IR), BF16),
                        )
                        if (
                            fault == "f_slotreuse"
                        ):  # FAULT: refill before this unit's down-proj read

                            @pl.when(step + NBE < count)
                            def _():
                                estart(step + NBE, mpos)

                        finish(step, lax.rem(step, NBE), a1)
                        if fault != "f_slotreuse":

                            @pl.when(step + NBE < count)
                            def _():
                                estart(step + NBE, mpos)

                        return a1n

                    def body_b(step, carry):
                        a1, iss = carry
                        nxt = step + 1

                        @pl.when(nxt < count)
                        def _():
                            ewait(nxt)

                        a1n = lax.cond(
                            nxt < count,
                            lambda: gate_up(lax.rem(nxt, NBE)),
                            lambda: jnp.zeros((M, 2 * IR), BF16),
                        )
                        if (
                            fault == "f_slotreuse"
                        ):  # FAULT: refill before this unit's down-proj read
                            iss = refill(step, iss)
                        finish(step, lax.rem(step, NBE), a1)
                        if fault != "f_slotreuse":
                            iss = refill(step, iss)
                        return a1n, iss

                    if KB == 1:
                        lax.fori_loop(0, count, body, gate_up(0))
                    else:
                        lax.fori_loop(
                            0, count, body_b, (gate_up(0), jnp.minimum(count, NBE))
                        )

                if ES:
                    z1 = jnp.zeros((M, 2 * IR), BF16)

                    def body_s(step, carry):
                        """L3-s: step -1 = prime (through the one refill site) + shared expert + unit 0's wait /
                        gate_up; steps >= 0 = the stage-A unit body with the refill at the same site."""
                        a1, iss = carry

                        def unit():
                            nxt = step + 1

                            @pl.when(nxt < count)
                            def _():
                                ewait(nxt)

                            a1n = lax.cond(
                                nxt < count,
                                lambda: gate_up(lax.rem(nxt, NBE)),
                                lambda: z1,
                            )
                            finish(step, lax.rem(step, NBE), a1)
                            return a1n

                        def first():
                            ewait(jnp.int32(0))
                            return gate_up(0)

                        def pro():
                            shared_part()
                            return lax.cond(count > 0, first, lambda: z1)

                        if (
                            fault == "f_slotreuse"
                        ):  # FAULT: refill before this unit's down-proj read
                            iss = refill(step, iss)
                        a1x = lax.cond(step >= 0, unit, lambda: a1)
                        if fault != "f_slotreuse":
                            iss = refill(step, iss)
                        a1y = lax.cond(step < 0, pro, lambda: a1x)
                        return a1y, iss

                    lax.fori_loop(-1, count, body_s, (z1, jnp.int32(0)))
                # lazy flush: the router / shared tiles refill only now (Qwen flush_rel)
                dgive(g_ffn)
                dgive(g_ffn + plan["gate"][2])
                dgive(g_ffn + plan["gate"][2] + plan["sh13"][2])
            if (
                GEARLY and fault in POISON_FAULTS
            ):  # the alias window is idle from here: poison it BEFORE the
                poison_alias()  # early gathers land in it (not at the end of the layer)
            with sc("M_combine"):
                comb = mtk._combine(
                    scr["owner"][...], scr["shared"][...], cfg.routed_scale
                )
            if GEARLY:
                nxt_is = jnp.logical_and(
                    l + 1 < L, ltab[jnp.minimum(l + 1, L - 1), 1] == 0
                )

                def early(which):
                    def f():
                        @pl.when(nxt_is)
                        def _():
                            with sc("G_early"):
                                core["early_issue"](l + 1, which)

                    return f

                scr["hcur"][...] = all_reduce(
                    comb.astype(F32), 2, mids=(early(0), early(1))
                )
            else:
                scr["hcur"][...] = all_reduce(comb.astype(F32), 2)

        def route(scores, bias, n_valid):
            """moe_tp_decode._route (kernel.py:171-193 @ 9ffdb1eb4); FAULT router_tiebreak picks the highest
            attaining column instead of the lowest."""
            if fault != "router_tiebreak":
                return mtk._route(scores, bias, n_valid, topk=TOPK)
            rows, n = scores.shape
            lane = lax.broadcasted_iota(I32, (rows, n), 1)
            masked = float(jnp.finfo(F32).min)
            x = scores + bias
            x = jnp.where(x > mtk.ROUTER_NEG, x, mtk.ROUTER_NEG)
            wsum = jnp.zeros((rows, 1), F32)
            for _ in range(TOPK):
                m = jnp.max(x, axis=1, keepdims=True)
                idx = jnp.max(jnp.where(x == m, lane, 0), axis=1, keepdims=True)
                pick = lane == idx
                wsum = wsum + jnp.sum(
                    jnp.where(pick, scores, 0.0), axis=1, keepdims=True
                )
                x = jnp.where(pick, masked, x)
            chosen = x == masked
            denom = jnp.where(wsum < mtk.SUM_FLOOR, mtk.SUM_FLOOR, wsum)
            w = (jnp.where(chosen, scores, 0.0) / denom).astype(BF16)
            valid = lax.broadcasted_iota(I32, (rows, n), 0) < n_valid
            return jnp.where(jnp.logical_and(chosen, valid), w.astype(F32), 0.0)

        # ---------------- step ----------------
        def prefetch(b0, c):  # the stream starts before anything else
            fetch(b0)
            return c

        with sc("S_prologue"):  # per-step work, outside every layer scope
            lax.fori_loop(0, NBD, prefetch, 0)
            pltpu.make_async_copy(
                smalls.at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
            ).start()
            if "coll" not in skip:
                CO.entry_barrier(R)
            scr["hcur"][...] = h0[...]
            scr["res"][...] = jnp.zeros((M, H), F32)
            scr["osend"][...] = jnp.zeros(scr["osend"].shape, U32)
            if LSE_TILE:  # lanes >= T and rows hh >= HR are never written
                scr["lsend"][...] = jnp.zeros((128, 128), F32)
                scr["lrecv"][...] = jnp.zeros((128, 128), F32)
            scr["spin"][...] = jnp.zeros((8, 128), F32)
            if fault == "stale_accumulator":
                scr["owner"][...] = jnp.zeros((M, H), F32)
            core["step_init"]()
        lax.fori_loop(0, L, layer, 0)
        if debug:
            dput(d_state.at[L, 0], scr["hcur"])
            dput(d_state.at[L, 1], scr["res"])
        with sc("S_epilogue"):
            x = scr["hcur"][...] + scr["res"][...]
            pltpu.make_async_copy(
                smalls.at[L], scr["sm"].at[0], scr["sm_sem"].at[0]
            ).start()
            pltpu.make_async_copy(
                smalls.at[L], scr["sm"].at[0], scr["sm_sem"].at[0]
            ).wait()
            o, n = lay["in_ln"]
            out_ref[...] = rmsnorm(x, scr["sm"][0, :, o : o + n], cfg.eps).astype(BF16)

    # ---------------- pallas_call ----------------
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = (
        [smem] * 7
        + [vmem] * 3
        + [hbm]
        + [hbm] * (n_d + n_e + len(REC_IN) + n_c + n_r + n_x)
    )
    rec_shapes = {
        "qrec": ((L, R, TMAX, QH), U32),
        "gnrec": ((L, tiles.grec * 128, 128), U32),
        "grrec": ((L, tiles.grec * 128, 128), U32),
    }
    out_shape = [
        jax.ShapeDtypeStruct((M, H), BF16),
        jax.ShapeDtypeStruct((L, TMAX, 128), U32),
        jax.ShapeDtypeStruct((L, TMAX, 128), U32),
        jax.ShapeDtypeStruct((L, TMAX, 256), I32),
    ]
    out_specs = [vmem, hbm, hbm, hbm]
    for nm in REC_OUT:
        out_shape.append(jax.ShapeDtypeStruct(*rec_shapes[nm[:-2]]))
        out_specs.append(hbm)
    if debug:
        dshapes = [
            ((L + 1, 2, M, H), F32),
            ((L, M, H), F32),
            ((L, M, H), F32),
            ((L, M, H), F32),
            ((L, M, QL), F32),
            ((L, M, KL), F32),
            ((L, R, TMAX, QH), U32),
            ((L, HR, M, KL), F32),
            ((L, M, EPAD), F32),
            ((max(Lf, 1), GEO["UR2"], 128), I32),
            ((max(Lf, 1), BMAX, 16), I32),
        ]
        out_shape += [jax.ShapeDtypeStruct(s, d) for s, d in dshapes]
        out_specs += [hbm] * len(dshapes)
    out_shape += [jax.ShapeDtypeStruct(*rec_specs[k]) for k in xout_names]
    out_specs += [hbm] * len(xout_names)
    call = pl.pallas_call(
        kernel,
        name=KERNEL_NAME + ("_dbg" if debug else ""),
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(
            collective_id=tiles.cid if R > 1 else None,
            vmem_limit_bytes=VMEM_LIMIT,
            disable_bounds_checks=True,
        ),
        interpret=(
            interpret
            if isinstance(interpret, pltpu.InterpretParams)
            else pltpu.InterpretParams()
        )
        if interpret
        else False,
    )
    vmem_by = {
        k: int(np.prod(v.shape)) * jnp.dtype(v.dtype).itemsize
        for k, v in scratch.items()
        if str(getattr(v, "memory_space", "")).lower().endswith("vmem")
    }
    info = dict(
        s_names=s_names,
        dkinds=dkinds,
        e_names=e_names,
        c_names=c_names,
        plan=plan,
        BANK=BANK,
        S=S,
        lay=lay,
        ltab=ltab_np,
        dtab=dtab_np,
        M=M,
        ck=ck,
        groups=G,
        OW=OW,
        OH=OH,
        QH=QH,
        QA=QA,
        MB_SH=MB_SH,
        geometry=GEO,
        epool=EPOOL,
        vmem_scratch_bytes=sum(vmem_by.values()),
        vmem_by=vmem_by,
        moe_layers=moe_layers,
        full_layers=full_l,
        eshapes=eshapes,
        n_fixed=n_fixed,
        # L4 drivers (M6) and the timing-only replay operands (appended after the caches)
        a2a=dict(
            lse=tiles.a2a_lse,
            rw=RW,
            chunks=NCH,
            payload_lanes=OWX,
            bytes_per_peer=TMAX * OWX * 4 + (128 * HPAD * 4 if LSE_TILE else 0),
            starts_per_peer=NCH + (1 if LSE_TILE else 0),
            shadow=bool(tiles.shadow_a2a),
        ),
        n_rec=n_r,
        rec_shapes=(
            {
                "a2a": [((L, HR, M, KL), "float32")],
                "ar": [((L, M, H), "float32"), ((L, M, H), "float32")],
            }.get(tiles.replay, [])
        ),
        xin=[(k,) + rec_specs[k] for k in xin_names],
        xout=[(k,) + rec_specs[k] for k in xout_names],
        rec_in=[(nm,) + rec_shapes[nm] for nm in REC_IN],
        rec_out=[(nm,) + rec_shapes[nm[:-2]] for nm in REC_OUT],
    )
    return call, info
