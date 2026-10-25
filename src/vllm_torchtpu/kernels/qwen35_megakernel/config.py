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
"""Configuration, tiling policy, weight/state shapes, and math primitives for Qwen3.5 megakernel."""

from __future__ import annotations

import dataclasses
import os

import jax
import jax._src.state.discharge as _discharge
import jax.numpy as jnp
from jax import lax

from . import pool_alias

if not getattr(_discharge.transform_swap_array, "_pool_io_patched", False):
    _orig_swap = _discharge.transform_swap_array

    def _patched_swap(x, transforms, val):
        if not transforms or all(
            isinstance(t, _discharge.indexing.NDIndexer) for t in transforms
        ):
            return _orig_swap(x, transforms, val)
        intermediates = [x]
        new_val = x
        for t in transforms:
            new_val = _discharge.transform_array(new_val, [t])
            intermediates.append(new_val)
        if val is None:
            return new_val, x
        new_x = val
        for inter, t in reversed(list(zip(intermediates[:-1], transforms))):
            if isinstance(t, _discharge.indexing.NDIndexer):
                _, new_x = _orig_swap(inter, [t], new_x)
            elif isinstance(t, _discharge.BitcastTransform):
                new_x = _discharge.bitcast(new_x, inter.dtype)
            elif isinstance(t, _discharge.ReshapeTransform):
                new_x = new_x.reshape(inter.shape)
            else:
                _, new_x = _orig_swap(inter, [t], new_x)
        return new_val, new_x

    _patched_swap._pool_io_patched = True
    _discharge.transform_swap_array = _patched_swap

FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
U8 = jnp.uint8
U32 = jnp.uint32
NN = (((1,), (0,)), ((), ()))
NT = (((1,), (1,)), ((), ()))
COLLECTIVE_ID = 61


def rup(x, m=128):
    return (x + m - 1) // m * m


# route() top-k with rows on lanes (topk_rows_on_lanes) instead of TOPK serial cross-lane
# max-reduce picks (each ~141 cycles of XLU reduce latency in the dependent chain).
TOPK_LANES = True
TOPK_NL = 16  # experts per presorted group (power of 2 dividing 128)


def _batcher_net(n):
    """Batcher odd-even mergesort comparators (i, j), i < j, i takes the max; n a power of 2."""
    out = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        out.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return out


def topk_rows_on_lanes(key, topk, em, need_ids=True, nl=TOPK_NL):
    """Exact top-k of distinct f32 keys [MR, E] (MR <= 128, E % 128 == 0), same result as the
    serial max-and-remove loop: (ids [MR, 128] (lane k = em - low bits of the k-th largest key,
    0 beyond topk; None unless need_ids), sel [MR, E] = the topk largest keys).

    The keys are transposed once (rows on lanes, experts on sublanes / vregs). Each 128-expert
    block is cut into nl slabs; slab i of every block is level i of E / nl groups of nl experts
    (one expert per slab), sorted descending once with a compare-exchange network (elementwise
    VPU max/min). Pick k is the max over the group heads (elementwise max over E / (8 nl)
    vregs + a sublane reduce), after which the picked group shifts up one level. The dependent
    chain holds no cross-lane reduction and no lane move; two transposes bring the picks back
    to rows."""
    mr, e = key.shape
    kp = key if mr == 128 else jnp.concatenate([key, jnp.zeros((128 - mr, e), F32)], axis=0)
    kt = kp.T  # [E, 128]
    sl = 128 // nl
    lev = [jnp.concatenate([kt[bk * 128 + i * sl:bk * 128 + (i + 1) * sl]
                            for bk in range(e // 128)], axis=0) for i in range(nl)]
    for i, j in _batcher_net(nl):
        lev[i], lev[j] = jnp.maximum(lev[i], lev[j]), jnp.minimum(lev[i], lev[j])
    ks = []
    for k in range(topk):
        m = jnp.max(lev[0], axis=0, keepdims=True)  # [1, 128]: k-th largest key of each row
        ks.append(m)
        nup = min(nl, topk - 1 - k)  # levels that can still reach the head
        if nup == 0:
            break
        hit = lev[0] == m
        for j in range(nup):
            lev[j] = jnp.where(hit, lev[j + 1] if j + 1 < nl else -1.0, lev[j])
    thr = jnp.broadcast_to(ks[topk - 1], (128, 128)).T[0:mr, :]  # [MR, 128]: row's topk-th key
    sel = key >= jnp.concatenate([thr] * (e // 128), axis=1)
    ids = None
    if need_ids:
        rows = jnp.concatenate(ks + [jnp.zeros((128 - topk, 128), F32)], axis=0)
        kk = rows.T[0:mr, :]  # [MR, 128]: lane k = k-th largest key of the row
        lane = lax.broadcasted_iota(I32, (mr, 128), 1)
        ids = jnp.where(lane < topk, em - (lax.bitcast_convert_type(kk, I32) & em), 0)
    return ids, sel


def rnd(x):
    """Round an f32 value to bf16 precision (the served model's activation dtype)."""
    return x.astype(BF16).astype(F32)


def bswitch(idx, fns):
    """Binary-tree lax.cond dispatch over fns[0..len(fns)-1] (depth ceil(log2(len(fns))))."""
    if len(fns) == 1:
        return fns[0]()
    mid = len(fns) // 2
    return lax.cond(idx < mid, lambda: bswitch(idx, fns[:mid]),
                    lambda: bswitch(idx - mid, fns[mid:]))


class Spec(jax.ShapeDtypeStruct):
    """ShapeDtypeStruct that also unpacks and compares as a (shape, dtype) 2-tuple."""

    def __iter__(self):
        yield self.shape
        yield self.dtype

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return (self.shape, self.dtype)[i]

    def __eq__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            return self.shape == other[0] and self.dtype == jnp.dtype(other[1])
        return super().__eq__(other)


@dataclasses.dataclass(frozen=True)
class MKConfig:
    """Model geometry. Derived per-rank sizes assume TP = EP = R."""

    R: int = 8
    B: int = 4
    L: int = 60
    H: int = 4096
    NH: int = 32
    NKV: int = 2
    HD: int = 256
    ROT: int = 64
    theta: float = 1e7
    KH: int = 16
    VH: int = 64
    KD: int = 128
    VD: int = 128
    E: int = 512
    TOPK: int = 10
    I: int = 1024
    IS: int = 1024
    eps: float = 1e-6
    ctx: int = 10240
    moe_tp: bool = True  # routed experts TP-sliced over the intermediate dim (else EP)
    att_every: int = 4  # layer l is full attention iff l % att_every == att_every - 1
    # tokens per sequence. T > 1 is the MTP verify pass: rows s*T .. s*T+T-1 are tokens 0..T-1 of
    # sequence s (at positions pos_s .. pos_s+T-1), which share one KV cache and one GDN state
    T: int = 1
    # Unstacked per-layer routed expert weights (L-tuple of 4D [EW, n_tiles, tk, N] arrays) and
    # optional physical expert slot count (e.g. e_slots=72 for 64 primary + 8 redundant EPLB slots)
    stack_e: bool = True
    e_slots: int = 0
    # Unified paged FP8 KV cache in raw[0..NATT-1] of shape [num_pages, 2, 64, 4, page_size] fp8
    paged_kv: bool = False
    num_pages: int = 37705
    page_size: int = 128
    max_pages_per_seq: int = 2050
    # Kernel pages between consecutive GDN checkpoint slots in the pool (0 -> gdn_pages). vLLM and
    # Raiden place manager block b's GDN state at page b * (block_size // page_size), so a vLLM
    # block larger than gdn_pages * page_size tokens (e.g. 1280 at TP8) leaves a gap per slot.
    gdn_stride_pages: int = 0
    fused_pool_io: bool = False  # P1+P2: gather/scatter GDN & KV states directly in raw[0..14]
    act_mode: str = "from_ckpt"  # "from_ckpt", "all", or "ref" when fused_pool_io=True
    ckpt_2d: bool = False        # True when ckpt_indices is 2D [NS, K] instead of 3D [G_PER_ATT, NS, K]
    ckpt_k: int = 0              # K dimension of ckpt_indices (0 -> T)
    roff_1d: bool = False        # True when read_offsets is 1D [NS] instead of 2D [G_PER_ATT, NS]
    # Wide step (A2): B rows run as microbatches of mb_rows rows inside one call, the attention
    # and GatedDeltaNet phases per microbatch and the MoE phase once over all rows, so each hit
    # expert's weights stream once per layer per step. 0: one microbatch of B rows.
    mb_rows: int = 0
    # Decode context parallelism: the dcp ranks of a KV-head group hold one shard of the pool
    # each. Global kernel page p of a sequence sits on DCP rank p % dcp at page-table column
    # p // dcp (vLLM's DCP layout, the batched RPA kernel's and Raiden's tp{T}dcp{D}_decode rule);
    # the page table is the rank-local one. 1: the pool is replicated over the group.
    dcp: int = 1

    @property
    def wide(self) -> bool:
        return self.mb_rows > 0

    @property
    def n_mb(self):  # microbatches of a wide step (the last one padded with inactive rows)
        return -(-self.B // self.mb_rows) if self.wide else 1

    def microbatch(self) -> "MKConfig":
        """The per-microbatch geometry of a wide step (mb_rows rows, itself when not wide)."""
        if not self.wide:
            return self
        return dataclasses.replace(self, B=self.mb_rows, mb_rows=0)

    @property
    def NS(self):  # sequences (KV caches, GDN input states)
        return self.B // self.T

    @property
    def HR(self):  # verify conv history rows per sequence: 3 cached taps, then the T new inputs
        return rup(3 + self.T, 8)

    @property
    def M(self):  # padded bf16 rows carried through the kernel: 16 up to 16 live rows
        return max(16, rup(self.B, 16))

    @property
    def MR(self):  # routing / expert-output rows: the B live rows rounded up to a sublane tile
        return rup(self.B, 8)

    @property
    def G(self):  # ranks sharing one KV head
        return self.R // self.NKV

    @property
    def nh(self):
        return self.NH // self.R

    @property
    def kh(self):
        return self.KH // self.R

    @property
    def vh(self):
        return self.VH // self.R

    @property
    def QKV(self):
        return 2 * self.kh * self.KD + self.vh * self.VD

    @property
    def Z(self):
        return self.vh * self.VD

    @property
    def BA(self):  # lane offset of the b|a block inside the GDN in_proj output
        return self.QKV + self.Z

    @property
    def NLI(self):  # GDN in_proj width: qkv | z | b (lanes 0..vh) a (lanes 64..64+vh)
        return self.QKV + self.Z + 128

    @property
    def NAI(self):  # attention in_proj width: per head [q | gate], then k, v of one KV head
        return self.nh * 2 * self.HD + 2 * self.HD

    @property
    def ER(self):
        return self.E // self.R

    @property
    def IS_SLICE(self):
        return self.IS // self.R

    @property
    def ISR(self):
        return rup(self.IS // self.R, 128)

    @property
    def I_SLICE(self):
        return self.I // self.R if self.moe_tp else self.I

    @property
    def IR(self):  # routed-expert intermediate width per rank (128-lane padded)
        return rup(self.I // self.R if self.moe_tp else self.I, 128)

    @property
    def EW(self):  # routed experts with weights on each rank (or physical e_slots when > 0)
        return self.e_slots if self.e_slots > 0 else (self.E if self.moe_tp else self.ER)

    @property
    def NGDN(self):
        return self.L - self.L // self.att_every

    @property
    def NATT(self):
        return self.L // self.att_every

    @property
    def ngdn(self):
        return self.NGDN

    @property
    def natt(self):
        return self.NATT

    @property
    def rec_pages(self) -> int:  # 128-token (64 KiB) kernel pages per GDN recurrent state checkpoint
        return max(1, ((self.vh * self.KD) // 2 + 127) // 128)

    @property
    def gdn_pages(self) -> int:  # total kernel pages per GDN state checkpoint (recurrent + 1 conv)
        return self.rec_pages + 1

    @property
    def gdn_stride(self) -> int:  # kernel pages from one GDN checkpoint slot to the next
        stride = self.gdn_stride_pages or self.gdn_pages
        assert stride >= self.gdn_pages, (stride, self.gdn_pages)
        return stride

    @property
    def CR(self):  # conv state rows: three taps x B rows
        return rup(3 * self.B, 8)

    @property
    def PC(self):  # all-reduce payload columns
        return min(1024, self.H)

    @property
    def PR(self):  # all-reduce payload rows (the B live rows folded to PC-wide rows, bf16 tile)
        return rup(self.B * self.H // self.PC, 16)

    def check(self):
        assert self.L % self.att_every == 0 and self.NH % self.R == 0 and self.R % self.NKV == 0
        assert self.KH % self.R == 0 and self.VH % self.R == 0 and self.E % self.R == 0
        assert self.IS % self.R == 0 and (not self.moe_tp or self.I % self.R == 0)
        b_max = 1024 if self.wide else 128
        assert self.vh <= 64 and 1 <= self.B <= self.M <= b_max, (self.B, self.M)
        assert self.wide or self.B * self.TOPK <= 1280, (self.B, self.TOPK)
        if self.wide:
            assert self.mb_rows % self.T == 0 and 16 <= self.mb_rows <= 128, (self.mb_rows, self.T)
        assert self.E % 128 == 0 and self.E <= 1024 and self.H % self.PC == 0 and self.nh <= 8
        assert self.ROT % 2 == 0 and self.R & (self.R - 1) == 0 and self.VH % self.KH == 0
        assert self.T >= 1 and self.B % self.T == 0, (self.B, self.T)
        assert self.e_slots == 0 or self.e_slots >= (self.E if self.moe_tp else self.ER)
        if self.paged_kv:
            assert self.page_size == 128 and self.num_pages >= 1 and self.max_pages_per_seq >= 1
        assert self.dcp >= 1 and self.G % self.dcp == 0, (self.dcp, self.G)
        assert self.dcp == 1 or self.paged_kv, "dcp > 1 shards the paged pool"
        for n in (self.H, self.QKV, self.Z, self.KD, self.VD, self.HD, self.IR, self.ISR):
            assert n % 128 == 0, n


FULL = MKConfig()
FULL_EP = dataclasses.replace(FULL, moe_tp=False)
# Every structural constraint of FULL at a size the CPU interpreter runs in seconds.
TINY = MKConfig(L=4, H=1024, NH=8, KH=8, VH=16, E=128, TOPK=4, I=1024, IS=1024, ctx=2048)
TINY_EP = dataclasses.replace(TINY, moe_tp=False)


@dataclasses.dataclass(frozen=True)
class Tiles:
    bank: int = 2 << 20  # bytes per dense ring slot
    # ring depths tuned on v7x-8 (decode 2,788 -> 2,735 us vs nbd=10, nbe=16 on one node)
    nbd: int = 14  # dense ring slots
    nbe: int = 10  # expert ring slots
    ebank: int = 1 << 20  # bytes per expert ring slot
    # KV stream: a ring of nkv stages, each holding one (row, block) item of this rank's work
    # list (listed once per step from pos), so VMEM does not grow with B
    kv_blk: int = 2048  # KV tokens per block
    nkv: int = 3
    vmem_mib: int = 62
    bounds_checks: bool = False
    lazy: bool = True  # refill the attention / MoE blocks' dense slots after their streams
    pipe: bool = True  # software-pipeline single-tile experts (gate|up of i+1 beside down of i)
    topk: bool = True  # one-reduction top-k picks on packed (value, lane) keys (route())
    # alias the GDN state inputs to their outputs. Off: XLA copies aliased operands in and all
    # outputs out of collective (collective_id) Pallas calls, so aliasing adds a state copy.
    inplace: bool = False
    # sequence-split attention: the G ranks sharing a KV head split each row's KV blocks (row b's
    # block t goes to group rank (t + b) % G) for all G * nh query heads of the group (q
    # all-gathered in the group), then exchange the partial softmax (m, l, acc) of each rank's
    # own heads over remote DMA and merge. Cuts the per-rank KV stream G-fold. Off: each rank
    # streams its group's whole cache.
    split: bool = True
    # two-round all-reduces: reduce-scatter (rank j sums lane chunk j of every partial), then
    # all-gather of the bf16 chunk sums. Sends 2 / R of the one-shot push's bytes for one more
    # round of latency, and rounds the sums to bf16 (the one-shot keeps them f32). Off: every rank
    # pushes its whole partial to every peer (latency-bound at 4 rows, bandwidth-bound at 16).
    rsag: bool = False
    # hierarchical all-reduce on v7x-8 (2x2x1: me ^ 1 on-chip, me ^ 2/4/6 same core across chips):
    # 0: flat R-1 peer rounds (one-shot or rsag); 1: on-chip push + 4-chip push; 2: on-chip
    # reduce-scatter + 4-chip push + on-chip all-gather; 3: log2(R)-step butterfly (off = 1, 2, ...).
    hier: int = 0
    # hierarchical reduce-scatter + all-gather for the flat rsag buckets (hier == 0, rsag, R = 8 on
    # 2x2x1): on-chip RS of half the lane chunks, 4-chip RS of one chunk, 4-chip AG, on-chip AG.
    # Each core sends 0.75 x payload to other chips instead of rsag's 1.5 x (remote writes run at
    # ~53 GB/s per core whatever the peer count, j-9406c8ab), in the same VMEM. Rounds the
    # chip-partial sums to bf16 (as hier=2 does), so it is not bit-identical to flat rsag.
    hr: int = 0
    # hier=2 with its 4-chip push (each core sends its half-sum to 3 chips: 1.5 x payload off-chip)
    # replaced by a 4-chip reduce-scatter + all-gather of that half (0.75 x payload): one more
    # round, same sums in the same order (bit-identical to hier=2).
    hx: int = 0
    # unfolded [MR, H] all-reduce payload (-1: auto when WIDE or MR * H <= PR * PC; 0: folded
    # [PR, PC]; 1: unfolded [MR, H])
    war: int = -1
    # pack PB = 8 // nh rows per 8-row tile in split-attention qg / pso / pri slabs (-1: auto)
    pack: int = -1
    # internal activation row count MB (0: M = max(16, rup(B, 16)); 1: max(16, MR); 2: MR)
    mb8: int = 0
    # above 16 rows the GatedDeltaNet states (256 KiB per row and layer) stream through a ring of
    # gdn_bufs buffers of gdn_rows rows each (row chunks, one DMA in and one out per chunk), so
    # their VMEM no longer grows with B. Up to 16 rows (or gdn_rows=0) each layer's states move
    # as one block.
    gdn_rows: int = 8
    gdn_bufs: int = 3
    # KV tokens per online-softmax update inside a block (0: the whole block). The verify pass
    # (T > 1) scores T * QR query rows per block; sub-chunks keep the score tile small.
    kv_sub: int = 0
    # attention mask mode: 0 = legacy (TQ, SUB) + (HD, SUB) iotas on every block;
    # 1 = (1, SUB) iota + U32 v-mask; 2 = (1, SUB) iota + skip mask on full blocks (off+SUB <= plen);
    # 3 = (1, SUB) iota + skip scores mask on full blocks, no v-mask;
    # 4 = (1, SUB) iota branchless scores mask, no v-mask.
    # 3 and 4 are UNSAFE in production: KV columns beyond plen (tail pages of the last block, which
    # the block table pads with page 0 = the unified pool's GDN null block) can hold fp8 NaN bit
    # patterns (0x7F / 0xFF), and p=0 times NaN V poisons the row (hidden NaN -> garbage token, NaN
    # K/V written back, drafts rejected). 1 keeps the (1, SUB) iota and masks V on the U32 view.
    amask: int = 1
    # ECOPY dedup optimization: 1 = deduplicate unique experts in B*TOPK loop first, then classify
    # only the n_tot unique experts.
    ecopy_opt: int = 1
    # autotuned-derived kernel changes. CPU-interpret bit-exact vs v15.3 and measured on v7x-8
    # (L=60, T=4, seq_len=32768). On by default (the v21 values); each can be switched with
    # VLLM_TPU_QWEN35_MK_{R2_DIRECT,ROUTE_VEC,DENSE_ACC}=0 / 1 (vllm_adapter).
    # autotuned R2: direct tile-0 store (= instead of zero-fill + +=) in experts_piped_ep (PIPE_EP).
    r2_direct_store: bool = True
    # autotuned cand-cc85bca1283c: VPU/MXU vectorized route_dedup (eliminates B*TOPK scalar SMEM loop).
    # -12/-36/-90/-87/-115/-172/-209 us at B=4/8/12/16/20/24/32.
    route_vec: bool = True
    # autotuned cand-aeab2123d449: 2-tile dmatmul register pairing + direct step-0 store + nt<=2 unroll.
    # -39/-47/-63/-29/-41/-46/-55 us at B=4/8/12/16/20/24/32.
    dense_acc_pair: bool = True
    # paged-KV ring stages aliased onto the (idle during attention) expert pool slots: NBE
    # stages in flight instead of nkv, and no separate kvbuf scratch (build: KVE)
    kv_ering: bool = True
    # Two KV work items per online-softmax loop iteration (item_front / item_back split): item
    # i+1's QK matmul, mask and row max overlap item i's exp / PV / state update. Pairs consume
    # two ring stages at once, so the ring is raised to >= 4 stages (double-buffered pairs; with
    # kv_ering the ring already has NBE >= 12 stages).
    pair_kv: bool = True

    def tk(self, K, N, dtype, bank):
        """Largest 128-multiple divisor of K whose [tk, N] tile fits a `bank`-byte slot."""
        best = None
        for tk in range(128, K + 1, 128):
            if K % tk == 0 and pool_alias.storage_size((tk, N), jnp.dtype(dtype)) <= bank:
                best = tk
        assert best is not None, ("no 128-row tile fits a bank", K, N, dtype, bank)
        return best


DEFAULT_TILES = Tiles(hier=1, pack=1, mb8=2, kv_sub=1024)  # pair_kv + 1024-token sub-blocks
# 9..16 rows (MTP verify, 3..4 sequences x 4 tokens): 3-stage hierarchical all-reduces (on-chip
# reduce-scatter + 4-chip push + on-chip all-gather), making room for a 4th KV stage
VERIFY_TILES = dataclasses.replace(DEFAULT_TILES, nkv=4, rsag=True, hier=2, ecopy_opt=1)
# 17..64 rows (MTP verify of 5..16 sequences): VERIFY_TILES with chunked GatedDeltaNet states and
# shallower rings, to fit the per-row buffers (activations, attention state, all-reduce payloads)
WIDE_TILES = dataclasses.replace(VERIFY_TILES, nbd=10, nkv=3, vmem_mib=62, hier=0)
TINY_TILES = Tiles(bank=128 << 10, nbd=10, nbe=6, ebank=256 << 10, kv_blk=256, nkv=3,
                   vmem_mib=32)


# Fused-path conv checkpoint staging (3 live rows only) from this many verify rows up.
CONV_STAGE_MIN_B = 8
# dmatmul full-step unroll (port of r4 DMM_UNROLL): ops of <= this many ring tiles run as one
# barrier-free step (all ring waits, per-chunk register sums over all tiles, one acc store, then
# all refills), so each chunk's MXU drain overlaps the next chunk's weight pushes.
DMM_UNROLL = 8
DMM_UNROLL_OFF_B = ()
MXU_DR_BUCKETS = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128)
GDN_LCV = True
GDN_ZDEFER = True
ATT_GDEFER = True
ATT_KVDEFER = True
ATT_QSWAP = True
FLUSH_IN_AR1 = True
SH2_PRE_ROUTE = True
# Where the shared expert (SH_PRE path) runs relative to routing. 0: before route() (default,
# unchanged). 1: inside route(), between starting the hit-list VMEM->SMEM copy and waiting on it.
# 2: after route() has issued the first expert DMAs (under the first tile's latency). Same ops,
# same order within the shared expert. Ported from the hostov stack (a83d4ad9f); serving knob
# VLLM_TPU_QWEN35_MK_SH_AT (see MKBuildContext).
SH_AT = 0
ARPIPE_ON = True
ARPIPE_OFF_B = (20, 24, 32)
ARPIPE_H1_ON = True
AR0S_ON = True
# Dense ring depth of the fused verify buckets: the deepest ring that fits the 62 MiB scoped
# VMEM limit (44 / 52 / 60 / 68 / 76: compile-searched on v7x-8, DCP=4).
FUSED_WIDE_NBD = {
    32: 12, 40: 12, 44: 12, 48: 12, 52: 11, 56: 11, 60: 10, 64: 9, 68: 9, 72: 8, 76: 8,
    80: 8, 96: 5, 112: 2,
}
# Fused verify buckets of 72..112 rows (18..28 sequences at K=3), each ONE kernel (see
# vllm_adapter._MAX_SINGLE_KERNEL_B) instead of a 2-microbatch split that re-streams every hit
# expert's weights per microbatch. They use the flat reduce-scatter + all-gather all-reduce
# (hier=0, rsag): its send / recv / gather buffers take half the VMEM of hier=2's (10.5 MiB at
# 112 rows), which buys back dense-ring depth under the 62 MiB scoped limit. Value: the dense
# ring depth (nbd) that fits. Microbench ms (v7x-8, DCP=4, ctx 80K; rsag vs hier=2 at its
# deepest fitting ring): 72: 16.64 vs 16.98, 76: 17.39 vs 17.59, 80: 17.63 vs 18.25,
# 84: 19.59 vs 19.53, 88: 19.97 vs 20.24, 96: 21.07 vs 21.51, 104: 23.86 vs 24.01,
# 112: 24.35 vs 25.22 (68 rows and below: no gain, keep hier=2). Ported from the final tree
# (6fc248fae, 236115c9c); VLLM_TPU_QWEN35_MK_RSAG_WIDE=0 turns the override off.
FUSED_WIDE_RSAG_NBD = {72: 11, 76: 10, 80: 10, 84: 9, 88: 9, 96: 8, 104: 6, 112: 6}
RSAG_WIDE_ON = os.environ.get("VLLM_TPU_QWEN35_MK_RSAG_WIDE", "1") != "0"
# Hierarchical rsag (Tiles.hr) for every rsag bucket on R = 8: VLLM_TPU_QWEN35_MK_HRSAG=1.
HRSAG_ON = os.environ.get("VLLM_TPU_QWEN35_MK_HRSAG", "1") == "1"
# 4-chip RS + AG inside hier=2 (Tiles.hx) on R = 8: VLLM_TPU_QWEN35_MK_HX=1.
HX_ON = os.environ.get("VLLM_TPU_QWEN35_MK_HX", "1") == "1"
# hx pays from ~48 rows (j-dfcd1c82, MK step p50: 64 rows -4.9%, 48 -2.8%, 32 0%, 24 +2.7%,
# 16 +5.7%: the extra round's latency outweighs the halved bytes on small payloads).
HX_MIN_ROWS = int(os.environ.get("VLLM_TPU_QWEN35_MK_HX_MIN_ROWS", "48"))
# EP expert ring refills with static banks and one layer-dispatch walk per macro-step
# (moe_block.experts_piped_ep). Bit-exact; microbench -1.5% at 64 / 80 rows, -6% at 96 / 112
# (final tree e25fc0d38). VLLM_TPU_QWEN35_MK_EP_FAST_REFILL=0 turns it off.
EP_FAST_REFILL = os.environ.get("VLLM_TPU_QWEN35_MK_EP_FAST_REFILL", "1") != "0"
# Wide step (mb_rows > 0): dense ring slots, gathered rows per routed-expert tile, and the row
# count above which the all-rows MoE buffers no longer fit VMEM beside the microbatch buffers.
WIDE_NBD = 8
WIDE_CAP = 32
WIDE_MAX_ROWS = 256

# ROUTE_SHRINK (VEC route): shorter serial scalar chain from the EARLY_FILL issue to the first
# expert compute. The copy-candidate scan packs each candidate's copy slot (its meta lane) above
# the expert id in cplist, so the po2c passes decode both holders with shifts / masks instead of a
# dependent ecopy SMEM load + f32 convert + two divisions per copy hit. Same decisions (exact).
ROUTE_SHRINK = True
CP_SH = 16

# PRIM_NOW: the fixed-hit compaction (prim) stores no wlist entries (slot = e - me * ER is
# derived in fetch_e / _unit_meta for list index < st[5]); halves prim's scalar stores before the first fill.
PRIM_NOW = True

# WGROUP (with GDEFER): att_in's ring tile waits through dtake_group (one bank rem, then the waits
# back to back) instead of one rem + wait per tile (~23 scalar cycles each, all before the first q
# MXU push). Same tiles, same views.
ATT_WGROUP = True

# Dense ring wait-group hoist: a group of consecutive dense tiles computes all its ring banks with ONE runtime
# rem BEFORE its first DMA wait, and each wait uses a static bank-0 dst view (dma.done.wait reads only the semaphore).
DWAIT_HOIST = True

# GDN_ZDEFER_ROLL: deferred z-column chunks inside rolled GDN sequence loop
GDN_ZDEFER_ROLL = True
GDN_ZDEFER_ROLL_BUCKETS = (32,)
GDN_SEQ_ROLL_MIN_B = 24


def default_tiles(c: MKConfig) -> Tiles:
    """Tuned tiles for c's row count: 2-stage hierarchical push up to 4 rows, 3-stage hierarchical
    RS+push+AG at 5..8 rows (nkv=3), 9..16 rows (VERIFY_TILES, nkv=4), 17..32 rows, 33..48 rows
    (WIDE_TILES with nbd=8, hier=2) and 49..64 rows (nbd=8 up to 56 rows, nbd=7 above).

    The hierarchical all-reduces (hier=1 / 2) are built for the 2x2x1 v7x-8 topology. On other
    meshes (e.g. R=16) they become the flat all-reduce (hier=0): reduce-scatter + all-gather where
    the bucket has rsag, the one-shot push at 1..4 rows. At R=16 on 2x2x2 flat rsag beats the
    log2(R)-step butterfly (hier=3) by 10-21% of the step at 8..32 rows."""
    if c.wide:
        # the microbatch's tiles; the dense ring keeps every tile of one input projection (the
        # attention and GatedDeltaNet blocks take them at once)
        t = default_tiles(c.microbatch())
        return dataclasses.replace(t, nbd=min(t.nbd, WIDE_NBD))
    # EP mode has 12 tiles per expert (T13=8, T2=4); nbe=12 fits 1 whole expert in the ring and
    # unlocks PIPE_EP (overlapping 1 w2 tile of expert i with 2 w13 tiles of expert i+1).
    nbe_ep = 12 if not c.moe_tp else 10
    # 1..8 rows: a 20-slot expert ring (1.67 EP experts in flight; NBE % TE != 0 drops PIPE_EP for
    # the plain expert loop). v7x-8 (L=60, T=4, seq_len=32768): -46 us at B=4, -19 us at B=8 over
    # nbe=12 on v15.3; ~0 at B >= 12 (kept at 12 there). VLLM_TPU_QWEN35_MK_NBE_SMALL overrides.
    nbe_small = 20 if not c.moe_tp else nbe_ep
    if c.B > 48:  # serving v20 B=56 / 64 buckets (adapter picks fused_pool_io for these)
        t = dataclasses.replace(WIDE_TILES, nbd=8 if c.B <= 56 else 7, nbe=nbe_ep, hier=2)
    elif c.B > 32:
        # nbd 9 -> 8: room for the pool-gather (GPR) staging under the scoped VMEM limit with the
        # forced-routing operand (the B=48 forced build overflowed by 1.34 MB at nbd=9)
        t = dataclasses.replace(WIDE_TILES, nbd=8, nbe=nbe_ep, hier=2)
    elif c.B > 24:
        t = dataclasses.replace(WIDE_TILES, nbe=nbe_ep, hier=2)
    elif c.B > 16:
        t = dataclasses.replace(WIDE_TILES, nbd=12, nbe=nbe_ep, hier=2)
    elif c.B > 8:
        t = dataclasses.replace(VERIFY_TILES, nbe=nbe_ep)
    elif c.B > 4:
        t = dataclasses.replace(DEFAULT_TILES, nbe=nbe_small, rsag=True, hier=2, ecopy_opt=1)
    else:
        t = dataclasses.replace(DEFAULT_TILES, nbe=nbe_small, ecopy_opt=1)
    if c.R != 8 and t.hier in (1, 2):
        t = dataclasses.replace(t, hier=0)
    if c.T > 1 and c.paged_kv and c.fused_pool_io and c.B in FUSED_WIDE_NBD:
        t = dataclasses.replace(t, nbd=FUSED_WIDE_NBD[c.B])
    if RSAG_WIDE_ON and c.T > 1 and c.paged_kv and c.fused_pool_io and c.B in FUSED_WIDE_RSAG_NBD:
        t = dataclasses.replace(t, hier=0, rsag=True, nbd=FUSED_WIDE_RSAG_NBD[c.B])
    if HRSAG_ON and c.R == 8 and t.rsag and t.hier == 0:
        t = dataclasses.replace(t, hr=1)
    if HX_ON and c.R == 8 and t.hier == 2 and c.B >= HX_MIN_ROWS:
        t = dataclasses.replace(t, hx=1)
    if c.NGDN == 0:
        t = dataclasses.replace(t, nbd=14)
    return t


def stream_kinds(c: MKConfig):
    """Streamed weight kinds: (K, N, dtype) of the per-rank matrix."""
    return {
        "lin_in": (c.H, c.NLI, FP8),
        "lin_out": (c.Z, c.H, FP8),
        "att_in": (c.H, c.NAI, FP8),
        "att_out": (c.nh * c.HD, c.H, FP8),
        "router": (c.H, c.E, BF16),
        "sh13": (c.H, 2 * c.ISR, FP8),
        "sh2": (c.ISR, c.H, FP8),
        "e13": (c.H, 2 * c.IR, FP8),
        "e2": (c.IR, c.H, FP8),
    }


DENSE = ("lin_in", "lin_out", "att_in", "att_out", "router", "sh13", "sh2")
EXPERT = ("e13", "e2")


def smalls_layout(c: MKConfig):
    """name -> (lane offset, width) in the per-layer f32 row of small parameters."""
    segs = [
        ("in_norm", c.H),
        ("post_norm", c.H),
        ("in_s", max(c.NLI, c.NAI)),
        ("out_s", c.H),
        ("sh13_s", 2 * c.ISR),
        ("sh2_s", c.H),
        ("sh_gate", c.H),
        ("conv", 4 * c.QKV),
        ("alog", 128),
        ("dtb", 128),
        ("gnorm", c.VD),
        ("qn", c.HD),
        ("kn", c.HD),
    ]
    if not c.moe_tp and c.e_slots > c.ER:
        segs.append(("ecopy", c.E))
    lay, o = {}, 0
    for name, n in segs:
        lay[name] = (o, n)
        o += rup(n)
    return lay, o


def encode_ecopy(c: MKConfig, cores) -> list[float]:
    """Pack per-layer redundant expert copy descriptors into an [E] f32 row for smalls['ecopy'].

    For each virtual expert v = r0 * ER + s0:
      - 0.0 if v has no secondary copy
      - float(1 + r1 * (EW - ER) + (s1 - ER)) if v is duplicated on rank r1 at slot s1 >= ER
    """
    er, ew = c.ER, c.EW
    extra = ew - er
    out = [0.0] * c.E
    if c.moe_tp or extra <= 0:
        return out
    vid = {int(g): r0 * er + s0 for r0, lst in enumerate(cores) for s0, g in enumerate(lst[:er])}
    for r1, lst in enumerate(cores):
        for k, g in enumerate(lst[er:ew]):
            v = vid.get(int(g))
            if v is not None and v // er != r1:
                out[v] = float(1 + r1 * extra + k)
    return out


def plan(c: MKConfig, tiles: Tiles):
    kinds = stream_kinds(c)
    tk = {k: tiles.tk(K, N, dt, tiles.ebank if k in EXPERT else tiles.bank)
          for k, (K, N, dt) in kinds.items()}
    nt = {k: kinds[k][0] // tk[k] for k in kinds}
    t_gdn = nt["lin_in"] + nt["lin_out"] + nt["router"] + nt["sh13"] + nt["sh2"]
    t_att = nt["att_in"] + nt["att_out"] + nt["router"] + nt["sh13"] + nt["sh2"]
    return {"kinds": kinds, "tk": tk, "nt": nt, "t_gdn": t_gdn, "t_att": t_att,
            "t_block": (c.att_every - 1) * t_gdn + t_att}


def weight_shapes(c: MKConfig, tiles: Tiles):
    """Per-rank HBM operand shapes (streamed weights pre-tiled [.., n_tiles, tk, N])."""
    p = plan(c, tiles)
    tk, nt = p["tk"], p["nt"]
    S = smalls_layout(c)[1]
    ngdn = max(1, c.NGDN)  # attention-only stacks keep one (unread) GatedDeltaNet layer

    def tiled(kind, lead):
        _, N, dt = p["kinds"][kind]
        return ((*lead, nt[kind], tk[kind], N), dt)

    if c.stack_e:
        e13 = tiled("e13", (c.L, c.EW))
        e2 = tiled("e2", (c.L, c.EW))
    else:
        e13 = tuple(Spec(*tiled("e13", (c.EW,))) for _ in range(c.L))
        e2 = tuple(Spec(*tiled("e2", (c.EW,))) for _ in range(c.L))
    return {
        "lin_in": tiled("lin_in", (ngdn,)),
        "lin_out": tiled("lin_out", (ngdn,)),
        "att_in": tiled("att_in", (c.NATT,)),
        "att_out": tiled("att_out", (c.NATT,)),
        "router": tiled("router", (c.L,)),
        "sh13": tiled("sh13", (c.L,)),
        "sh2": tiled("sh2", (c.L,)),
        "e13": e13,
        "e2": e2,
        # per-channel scales, gate|up then down. One [1, N] row per expert: a one-row slice of a
        # 3-D [L, EW, N] array DMAs wrong data on TPU (interpret mode cannot catch it).
        "es": ((c.L, c.EW, 1, 2 * c.IR + c.H), F32),
        "smalls": ((c.L, 1, S), F32),
        "final_norm": ((1, c.H), F32),
    }


def state_shapes(c: MKConfig):
    """Per-rank input states. T > 1 (verify): one KV cache and one recurrent state per sequence,
    and the conv state as a per-sequence history [NS, HR, QKV] whose rows 0..2 are the cached taps
    (oldest first; rows >= 3 are scratch)."""
    ngdn = max(1, c.NGDN)
    conv = (ngdn, c.CR, c.QKV) if c.T == 1 else (ngdn, c.NS, c.HR, c.QKV)
    if c.paged_kv and c.fused_pool_io:
        G_PER_ATT = max(1, c.att_every - 1)
        K_ckpt = c.ckpt_k if c.ckpt_k > 0 else c.T
        ckpt_shape = (c.NS, K_ckpt) if c.ckpt_2d else (G_PER_ATT, c.NS, K_ckpt)
        roff_shape = (c.NS,) if c.roff_1d else (G_PER_ATT, c.NS)
        st = {
            "raw": tuple(Spec((c.num_pages, 2, 64, 4, c.page_size), FP8) for _ in range(c.NATT)),
            "page_table": Spec((c.NS, c.max_pages_per_seq), I32),
            "ckpt_indices": Spec(ckpt_shape, I32),
            "read_offsets": Spec(roff_shape, I32),
        }
        if c.act_mode == "ref":
            st["active_seqs"] = Spec((c.NS,), I32)
        return st
    if c.paged_kv:
        return {
            "conv": (conv, F32),
            "rec": ((ngdn, c.NS * c.vh, c.KD, c.VD), BF16),
            "raw": tuple(Spec((c.num_pages, 2, 64, 4, c.page_size), FP8) for _ in range(c.NATT)),
            "page_table": Spec((c.NS, c.max_pages_per_seq), I32),
        }
    return {
        "conv": (conv, F32),
        "rec": ((ngdn, c.NS * c.vh, c.KD, c.VD), BF16),  # key-major S[k, v]
        "kc": ((c.NATT, c.NS, c.ctx, c.HD), BF16),
        "vc": ((c.NATT, c.NS, c.ctx, c.HD), BF16),
    }


def kernel_out_shapes(c: MKConfig):
    """hidden, new GDN states, and the new K/V rows (row b of [a] is layer a's new key of row b).

    T > 1: conv is the history [NS, HR, QKV]: rows 0..2 the cached taps, row 3 + t token t's conv
    input, so the conv state after token t is rows t+1 .. t+3. rec holds the state after every
    token (row b = s * T + t: after token t of sequence s). See accept_states."""
    if c.paged_kv and c.fused_pool_io:
        return {
            "hidden": ((c.M, c.H), BF16),
            "raw": tuple(Spec((c.num_pages, 2, 64, 4, c.page_size), FP8) for _ in range(c.NATT)),
        }
    sh = state_shapes(c)
    ngdn = max(1, c.NGDN)
    kv_dt = FP8 if c.paged_kv else F32
    return {
        "hidden": ((c.M, c.H), BF16),
        "conv": sh["conv"],
        "rec": ((ngdn, c.B * c.vh, c.KD, c.VD), BF16),
        "knew": ((c.NATT, c.MR, c.HD), kv_dt),
        "vnew": ((c.NATT, c.MR, c.HD), kv_dt),
    }


def accept_states(c: MKConfig, st, acc):
    """Verify outputs -> the next step's GDN input states when sequence s accepted acc[s] of its
    T tokens (1 <= acc[s] <= T): conv history rows acc .. acc+2 become the cached taps, and the
    recurrent state after token acc-1 is kept. st: rank-unfolded {"conv": [R, ngdn, NS, HR, QKV],
    "rec": [R, ngdn, B*vh, KD, VD], ...}; other entries pass through."""
    conv, rec, vh = st["conv"], st["rec"], c.vh
    keep = []
    for s in range(c.NS):
        a = int(acc[s])
        assert 1 <= a <= c.T, acc
        conv = conv.at[:, :, s, 0:3].set(st["conv"][:, :, s, a:a + 3])
        b = s * c.T + a - 1
        keep.append(rec[:, :, b * vh:(b + 1) * vh])
    return {**st, "conv": conv, "rec": jnp.concatenate(keep, axis=2)}


def fold(x):
    """[R, N0, ...] -> [R * N0, ...]: the rank-folded layout make_step takes for states."""
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def unfold(x, R):
    return x.reshape((R, x.shape[0] // R) + x.shape[1:])


W_NAMES = tuple(weight_shapes(TINY, TINY_TILES).keys())
S_NAMES = ("conv", "rec", "kc", "vc")
O_NAMES = ("hidden", "conv", "rec", "knew", "vnew")
ABLATIONS = ("experts", "ar", "attn", "gdn", "dense", "kv")
# GQA query-group packing of the split-attention q all-gather / partial-softmax exchange (see GQ
# in build)
GQ_PACK = True
