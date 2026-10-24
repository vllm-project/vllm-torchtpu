"""Qwen3.5-397B-A17B decode megakernel for TPU v7x: one pallas_call per decode step.

Per rank r of R: TP for GatedDeltaNet / attention / shared expert; routed experts TP-sliced along
their intermediate dimension (moe_tp, default: every rank streams 1/R of every hit expert, so the
per-rank load does not depend on how routing spreads over the ranks) or EP (each rank streams its
own E/R whole experts). B <= 48 live decode rows carried as M = max(16, rup(B, 16)) padded bf16
rows (above 16 rows: GatedDeltaNet states streamed in row chunks, unfolded two-round all-reduce;
see WIDE_TILES). W8A16 numerics: fp8 weights with per-output-channel f32 scales (the served
checkpoint is requantized to per-channel fp8 at load time) against bf16 activations, f32
accumulation.

  for each layer l (fori_loop):
    zero-centered RMSNorm
    GDN layers:  in_proj -> causal conv + SiLU -> l2norm q/k -> gated delta rule (bf16 state in
                 HBM) -> gated RMSNorm -> out_proj
    attn layers: q|gate, k, v proj -> q/k RMSNorm -> partial RoPE -> online-softmax attention over
                 the KV cache plus the new token -> sigmoid gate -> o_proj. The KV blocks stream
                 through a ring as a flat (row, block) work list; the ranks sharing a KV head split
                 each row's blocks and merge partial softmaxes over remote DMA (Tiles.split). The
                 new K/V rows are returned and scattered into the cache after the call.
    all-reduce (one-shot remote DMA, bf16 payload, f32 sum; or reduce-scatter + all-gather,
      Tiles.rsag) -> residual
    RMSNorm -> router (bf16, replicated) -> softmax top-k renorm -> hit list (SMEM, deduplicated
      picks; EP keeps the local ones) -> routed expert tiles start streaming
    shared expert (TP slice) x sigmoid gate, then every listed expert (slice) on all rows
    all-reduce -> residual
  final RMSNorm

Weights reach VMEM through two rings: a dense ring of `bank`-byte slots whose static tile schedule
runs ahead across ops and layers, and an expert ring of `ebank`-byte slots filled once routing
publishes the hit list.
Ring slots are typed views of one u8 pool per ring (pool_alias), or typed per-kind rings in
interpret mode.

The kernel communicates (remote DMA), and XLA TPU never places a communicating custom call's
outputs in live-out buffers: an aliased state would be copied in and out every step. So the KV
caches are read-only operands (the new rows come back as small outputs, scattered by XLA after the
call), and the GatedDeltaNet states are written to fresh outputs (XLA copies those, ~50 MB/rank).

MTP verify (MKConfig.T > 1): the B rows are T tokens of each of B / T sequences (row s*T + t at
position pos_s + t), with one KV cache and one GatedDeltaNet state per sequence. The KV work list
runs over (sequence, block) and scores each block for the sequence's T * QR query rows; the T new
K/V rows are applied in-kernel with a causal mask; the GatedDeltaNet runs the T-step recurrence
from the loaded state and returns the conv input history and the state after every token
(accept_states turns them into the next step's inputs for a given number of accepted tokens).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math

import jax
import jax._src.state.discharge as _discharge
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

try:
    from . import pool_alias
except ImportError:
    import pool_alias

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
    fused_pool_io: bool = False  # P1+P2: gather/scatter GDN & KV states directly in raw[0..14]
    act_mode: str = "from_ckpt"  # "from_ckpt", "all", or "ref" when fused_pool_io=True
    ckpt_2d: bool = False        # True when ckpt_indices is 2D [NS, K] instead of 3D [G_PER_ATT, NS, K]
    ckpt_k: int = 0              # K dimension of ckpt_indices (0 -> T)
    roff_1d: bool = False        # True when read_offsets is 1D [NS] instead of 2D [G_PER_ATT, NS]

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
    def ISR(self):
        return self.IS // self.R

    @property
    def IR(self):  # routed-expert intermediate width per rank
        return self.I // self.R if self.moe_tp else self.I

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
        assert self.vh <= 64 and self.B * self.TOPK <= 640 and 1 <= self.B <= self.M <= 64
        assert self.E % 128 == 0 and self.E <= 1024 and self.H % self.PC == 0 and self.nh <= 8
        assert self.ROT % 2 == 0 and self.R & (self.R - 1) == 0 and self.VH % self.KH == 0
        assert self.T >= 1 and self.B % self.T == 0, (self.B, self.T)
        assert self.e_slots == 0 or self.e_slots >= (self.E if self.moe_tp else self.ER)
        if self.paged_kv:
            assert self.page_size == 128 and self.num_pages >= 1 and self.max_pages_per_seq >= 1
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
    # reduce-scatter + 4-chip push + on-chip all-gather; 3: 3-step butterfly (off = 1, 2, 4).
    hier: int = 0
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
    amask: int = 1
    # ECOPY dedup optimization: 1 = deduplicate unique experts in B*TOPK loop first, then classify
    # only the n_tot unique experts.
    ecopy_opt: int = 0

    def tk(self, K, N, dtype, bank):
        """Largest 128-multiple divisor of K whose [tk, N] tile fits a `bank`-byte slot."""
        best = None
        for tk in range(128, K + 1, 128):
            if K % tk == 0 and pool_alias.storage_size((tk, N), jnp.dtype(dtype)) <= bank:
                best = tk
        assert best is not None, ("no 128-row tile fits a bank", K, N, dtype, bank)
        return best


DEFAULT_TILES = Tiles(hier=1, pack=1, mb8=2)
# 9..16 rows (MTP verify, 3..4 sequences x 4 tokens): 3-stage hierarchical all-reduces (on-chip
# reduce-scatter + 4-chip push + on-chip all-gather), making room for a 4th KV stage
VERIFY_TILES = dataclasses.replace(DEFAULT_TILES, nkv=4, rsag=True, hier=2, ecopy_opt=1)
# 17..48 rows (MTP verify of 5..12 sequences): VERIFY_TILES with chunked GatedDeltaNet states and
# shallower rings, to fit the per-row buffers (activations, attention state, all-reduce payloads)
WIDE_TILES = dataclasses.replace(VERIFY_TILES, nbd=10, nkv=3, vmem_mib=62, hier=0)
TINY_TILES = Tiles(bank=128 << 10, nbd=10, nbe=6, ebank=256 << 10, kv_blk=256, nkv=3,
                   vmem_mib=32)


def default_tiles(c: MKConfig) -> Tiles:
    """Tuned tiles for c's row count: 2-stage hierarchical push up to 4 rows, 3-stage hierarchical
    RS+push+AG at 5..8 rows (nkv=3), 9..16 rows (VERIFY_TILES, nkv=4), and 17..32 rows, else
    WIDE_TILES."""
    if c.B > 32:
        return WIDE_TILES
    if c.B > 24:
        return dataclasses.replace(WIDE_TILES, hier=2)
    if c.B > 16:
        return dataclasses.replace(WIDE_TILES, nbd=12, hier=2)
    if c.B > 8:
        return VERIFY_TILES
    if c.B > 4:
        return dataclasses.replace(DEFAULT_TILES, rsag=True, hier=2, ecopy_opt=1)
    return DEFAULT_TILES


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


W_NAMES = tuple(weight_shapes(TINY, TINY_TILES).keys())
S_NAMES = ("conv", "rec", "kc", "vc")
O_NAMES = ("hidden", "conv", "rec", "knew", "vnew")
ABLATIONS = ("experts", "ar", "attn", "gdn", "dense", "kv")


def build(c: MKConfig, tiles: Tiles | None = None, *, interpret=False, forced_routing=False,
          skip=frozenset(), scopes=False, row_mask=False):
    """Returns the per-rank kernel call (pos, hidden, *weights, *states, [rbias], [moe_rows]) ->
    (hidden_out, conv, rec, knew, vnew), plus an info dict. Wrap in shard_map over a ("tp",) mesh
    of R devices (see make_step, which also scatters knew / vnew into the caches).

    pos [B] int32: cached tokens per row (the new token is written at index pos).
    hidden [M, H] bf16: the new tokens' embeddings in rows 0..B-1 (rows >= B zero).
    rbias [L, M, E] f32 (forced_routing): routing selects the TOPK largest rbias entries per row
    instead of the TOPK largest router probabilities (benchmarks with synthetic weights).
    moe_rows [B] int32 (row_mask): rows with moe_rows[b] == 0 add no routed experts to the
    layer's hit list (rows whose outputs are discarded: padding, or MTP verify rows whose draft
    is not used). Their outputs only carry the experts that the other rows hit.
    c.T > 1 (MTP verify): row b = s * T + t is token t of sequence s and pos[b] must be
    pos[s * T] + t (pos[s * T]: tokens in sequence s's cache). Token t attends to the cache and
    to new tokens 0..t; the GatedDeltaNet runs the T-step recurrence from the sequence's state
    and returns the conv history and the state after every token (kernel_out_shapes).
    tiles: default_tiles(c) when None.
    skip: ablations for timing breakdowns (results are wrong when non-empty): experts / attn / gdn
    skip that compute, ar the remote copies, dense / kv the dense weight / KV DMAs.
    scopes: emit per-phase device trace scopes (profiling); 2 also scopes the weight-ring waits.
    """
    c.check()
    assert set(skip) <= set(ABLATIONS), skip
    tiles = default_tiles(c) if tiles is None else tiles

    def sc(name, level=1):
        return jax.named_scope(name) if int(scopes) >= level else contextlib.nullcontext()

    R, B, M, H, L, E = c.R, c.B, c.M, c.H, c.L, c.E
    AE, MR, G = c.att_every, c.MR, c.G
    MB = MR if tiles.mb8 == 2 else (max(16, MR) if tiles.mb8 else M)
    p = plan(c, tiles)
    tk, nt, kinds = p["tk"], p["nt"], p["kinds"]
    t_gdn, t_att, t_block = p["t_gdn"], p["t_att"], p["t_block"]
    T_TOTAL = (L // AE) * t_block
    lay, S = smalls_layout(c)
    NBD, NBE = tiles.nbd, tiles.nbe
    BROWS, EBROWS = tiles.bank // 128, tiles.ebank // 128
    T13, T2 = nt["e13"], nt["e2"]
    TE = T13 + T2
    NSS = 3 + (NBE - 1) // TE  # expert scale-row slots: in flight, plus the one being used
    PIPE = tiles.pipe and T13 == 1 and T2 == 1
    BLK, NKV = tiles.kv_blk, tiles.nkv
    assert (c.paged_kv or c.ctx % BLK == 0) and BLK % 128 == 0
    if c.paged_kv:
        assert BLK % c.page_size == 0, (BLK, c.page_size)
    nh, kh, vh, HD, KD, VD = c.nh, c.kh, c.vh, c.HD, c.KD, c.VD
    QKV, Z, BA, IR, ER = c.QKV, c.Z, c.BA, c.IR, c.ER
    PR, PC = c.PR, c.PC
    HPC = H // PC
    CH = min(1024, H)  # output lanes per down-projection chunk (single-tile experts)
    assert H % CH == 0
    NACC = max(c.NLI, c.NAI, H, E, 2 * c.ISR, 2 * IR)
    NY = max(c.NLI, c.NAI)
    KX2 = max(Z, nh * HD, c.ISR, IR)
    NMIX = max(Z, nh * HD)
    typed = interpret
    # Lazy dense refills: the attention block's and the MoE block's dense tiles are refilled after
    # the KV / expert stream instead of as each is consumed, so that the refills overlap the
    # all-reduces rather than compete for HBM. Needs the block's tiles to fit the ring at once.
    N_MOE = nt["router"] + nt["sh13"] + nt["sh2"]
    LAZY_ATT = tiles.lazy and nt["att_in"] + nt["att_out"] <= NBD
    LAZY_MOE = tiles.lazy and N_MOE <= NBD
    # sequence-split attention: row b's KV block t is streamed by group rank (t + b) % G (TS = G)
    SPLIT = tiles.split and G > 1
    assert not SPLIT or G & (G - 1) == 0  # groups are aligned rank blocks: peers are me ^ off
    TS = G if SPLIT else 1
    QR = rup(G * nh, 8) if SPLIT else 8  # query rows per sequence row in the KV loop
    # verify (T > 1): the KV rows are the NS sequences; each KV block is scored for the T * QR
    # query rows of its sequence (token t's heads in rows t * QR ..). T = 1: NS = B, TQ = QR.
    T, NS = c.T, c.NS
    VER = T > 1
    TQ = T * QR
    assert not (VER and tiles.inplace), "verify returns per-token states: no aliasing"
    SUB = tiles.kv_sub or BLK  # KV tokens per online-softmax update
    assert BLK % SUB == 0 and SUB % 128 == 0, (BLK, SUB)
    PW = HD + 128  # partial-softmax slab lanes: acc | m (64 lanes) | l (64 lanes)
    eff_ctx = max(c.ctx, c.max_pages_per_seq * c.page_size) if c.paged_kv else c.ctx
    max_nb = -(-eff_ctx // BLK)
    NWL = rup(NS * -(-max_nb // TS), 8)  # KV work items per rank, at most
    # above 16 rows ("wide"; up to 16 rows the kernel is unchanged):
    # - GCH: each GatedDeltaNet layer's states stream through NRB buffers of CB rows (chunk j of
    #   the layer in buffer j % NRB; chunk j + NRB - 1 is fetched once chunk j - 1 is written back;
    #   NF full chunks, then a tail chunk NF of the TR = B % CB remaining rows)
    # - WAR: all-reduce on unfolded [MR, H] rows when WIDE or MR * H <= PR * PC (no row fold)
    FUSED = c.paged_kv and c.fused_pool_io
    WIDE = B > 16
    GCH = WIDE and tiles.gdn_rows > 0 and c.NGDN > 0 and not FUSED
    CB = tiles.gdn_rows if GCH else B
    assert not GCH or CB % 8 == 0, ("gdn_rows must be a multiple of 8", tiles.gdn_rows)
    assert not (GCH and VER) or CB % T == 0, ("gdn_rows must be a multiple of T", CB, T)
    NF, TR = divmod(B, CB)
    NCH = NF + (TR > 0)
    NRB = min(tiles.gdn_bufs, NCH)
    assert not GCH or NCH == 1 or NRB >= 2, ("gdn_bufs >= 2", tiles.gdn_bufs)
    CBV = CB * vh  # state heads per full chunk
    NGI = min(NS, 8)
    NGO = min(NS, 2)
    G_PER_ATT = max(1, AE - 1)
    qkv_128 = QKV // 128
    A_conv = 16
    conv8_rows = A_conv * qkv_128
    ssm_u32_rows = (vh * KD) // 2
    ckpt_is_3d = not c.ckpt_2d
    roff_is_2d = not c.roff_1d
    K_ckpt = c.ckpt_k if c.ckpt_k > 0 else T
    ckpt_t_off = 0 if K_ckpt == T else 1
    HIER = tiles.hier if R > 1 else 0
    RSAG = tiles.rsag and R > 1 and HIER == 0
    WAR = (WIDE or MR * H <= PR * PC) if tiles.war < 0 else bool(tiles.war)
    AR_R = MR if WAR else PR
    AR_C = H if WAR else PC
    CW = AR_C // R  # rsag: payload lanes summed by each rank
    HC2 = AR_C // 2  # hier=2: payload lanes per on-chip core
    assert not RSAG or CW % 128 == 0
    assert HIER in (0, 1, 2, 3) and (HIER == 0 or R == 8)
    assert HIER != 2 or HC2 % 128 == 0
    AR_ROUNDS = (0, 1, 2) if HIER in (2, 3) else ((0, 1) if (RSAG or HIER == 1) else (0,))
    AR_PEERS = 1 if HIER == 3 else (3 if HIER in (1, 2) else max(R - 1, 1))
    # - PACK: the split attention's per-row slabs (q of this rank's nh heads; partial softmax of a
    #   group rank's heads) pack PB rows per 8-row tile: row b at tile b // PB, rows
    #   (b % PB) * nh .. (instead of one zero-padded tile per row), halving the q all-gather and
    #   the partial-softmax exchange for nh = 4
    PACK = (WIDE if tiles.pack < 0 else bool(tiles.pack)) and SPLIT and nh < 8
    PB = 8 // nh if PACK else 1
    NBP = -(-B // PB)

    def qrow(b):  # row b's tile and first sublane in the per-row slabs (qg, pso, pri)
        return b // PB, (b % PB) * nh

    scratch = {
        "dsem": pltpu.SemaphoreType.DMA((NBD,)),
        "esem": pltpu.SemaphoreType.DMA((NBE,)),
        "sm": pltpu.VMEM((2, 1, S), F32),
        "sm_sem": pltpu.SemaphoreType.DMA((2,)),
        "res": pltpu.VMEM((MB, H), F32),
        "xb": pltpu.VMEM((MB, H), BF16),  # normed rows: mixer / router / expert input
        "xb2": pltpu.VMEM((MB, KX2), BF16),  # intermediate rows: out / down projection input
        "acc": pltpu.VMEM((MB, NACC), F32),
        "ybuf": pltpu.VMEM((MB, NY), F32),
        "part": pltpu.VMEM((MB, H), F32),
        "mix": pltpu.VMEM((MB, NMIX), F32),
        # all-reduce [slot, ...]; rsag: lane chunks [slot, rank, AR_R, CW] (round 1 send / receive)
        "send": pltpu.VMEM((3, 2, AR_R, AR_C) if HIER == 3 else
                           (2, 2, AR_R, HC2) if HIER == 2 else
                           (2, R, AR_R, CW) if RSAG else (2, AR_R, AR_C), BF16),
        "recv": pltpu.VMEM((3, 2, 2, AR_R, AR_C) if HIER == 3 else
                           (2, 2, AR_R, HC2) if HIER == 2 else
                           (2, 2, AR_R, AR_C) if HIER == 1 else
                           (2, R, AR_R, CW) if RSAG else (2, R, AR_R, AR_C), BF16),
        "ar_ssem": pltpu.SemaphoreType.DMA((len(AR_ROUNDS), 2, AR_PEERS)),
        "ar_rsem": pltpu.SemaphoreType.DMA((len(AR_ROUNDS), 2, AR_PEERS)),
        # GatedDeltaNet
        "recb": pltpu.VMEM((B * vh, KD, VD), BF16),  # key-major states
        "sbuf": pltpu.VMEM((vh, KD, VD), F32),
        # T = 1: conv taps [3 x B rows], updated in place. T > 1: per-sequence history (3 cached
        # taps, then the T new inputs), DMA'd in and out whole
        "convb": pltpu.VMEM((NS, c.HR, QKV) if VER else (c.CR, QKV), F32),
        "st_sem": pltpu.SemaphoreType.DMA((4,)),  # conv in, rec in, conv out, rec out
        # attention: KV ring stage s holds one work item; item i is block wl_t[i] of KV row
        # (sequence) wl_b[i] (item count in st[1])
        "kbuf": pltpu.VMEM((NKV, BLK, HD), BF16),
        "vbuf": pltpu.VMEM((NKV, BLK, HD), BF16),
        "kv_sem": pltpu.SemaphoreType.DMA((2, NKV)),
        "wl_b": pltpu.SMEM((NWL,), I32),
        "wl_t": pltpu.SMEM((NWL,), I32),
        "qb": pltpu.VMEM((NS, TQ, HD), BF16),
        # online-softmax state per KV row: running max (lane-broadcast), sum, unnormalized output
        "am": pltpu.VMEM((NS, TQ, 128), F32),
        "al": pltpu.VMEM((NS, TQ, 128), F32),
        "aacc": pltpu.VMEM((NS, TQ, HD), F32),
        "knew": pltpu.VMEM((MR, HD), F32),
        "vnew": pltpu.VMEM((MR, HD), F32),
        "gate": pltpu.VMEM((MR, nh * HD), F32),
        "kv_out_sem": pltpu.SemaphoreType.DMA((2,)),
        # MoE
        "wn": pltpu.VMEM((MR, E), F32),  # renormalized routing weights of rows 0..MR-1
        "sg": pltpu.VMEM((MB, 128), F32),
        "hitv": pltpu.VMEM((MR, 128), I32),  # row b lanes 0..TOPK-1: row b's picks
        "hits": pltpu.SMEM((MR, 128), I32),
        "elist": pltpu.SMEM((rup(B * c.TOPK),), I32),  # deduplicated hit experts, pick order
        "seen": pltpu.SMEM((8, 128), I32),  # [e // 128, e % 128]: layer + 1 of e's last listing
        "misc_sem": pltpu.SemaphoreType.DMA(()),
        "es": pltpu.VMEM((NSS, 1, 2 * IR + H), F32),
        "es_sem": pltpu.SemaphoreType.DMA((NSS,)),
        "st": pltpu.SMEM((8,), I32),
        "fn": pltpu.VMEM((1, H), F32),
    }
    if c.paged_kv:
        del scratch["kbuf"], scratch["vbuf"]
        scratch["kvbuf"] = pltpu.VMEM((NKV, 2, 64, 4, BLK), FP8)
        scratch["kv_sem"] = pltpu.SemaphoreType.DMA((NKV,))
        scratch["knew_o"] = pltpu.VMEM((MR, HD), FP8)
        scratch["vnew_o"] = pltpu.VMEM((MR, HD), FP8)
    if FUSED:
        for k in ("recb", "convb", "st_sem", "kv_out_sem"):
            del scratch[k]
        if c.NGDN > 0:
            scratch.update({
                "gdn_in_page": pltpu.VMEM((NGI, 5, 2, 64, 4, c.page_size), FP8),
                "gdn_out_page": pltpu.VMEM((NGO, 5 * T, 2, 64, 4, c.page_size), FP8),
                "gdn_in_sem": pltpu.SemaphoreType.DMA((NGI,)),
                "gdn_out_sem": pltpu.SemaphoreType.DMA((NGO, T)),
                "gdn_in_pend": pltpu.SMEM((NGI,), I32),
                "gdn_out_mode": pltpu.SMEM((NGO,), I32),
            })
        else:
            del scratch["sbuf"]
        scratch.update({
            "kv_scat_page": pltpu.VMEM((NS, 2, 64, 4, c.page_size), FP8),
            "kv_rd_sem": pltpu.SemaphoreType.DMA((NS,)),
            "kv_wr_sem": pltpu.SemaphoreType.DMA((NS,)),
            "kv_rd_pend": pltpu.SMEM((NS,), I32),
            "kv_wr_pend": pltpu.SMEM((NS,), I32),
        })
    ECOPY = not c.moe_tp and c.e_slots > ER
    assert not (row_mask and ECOPY), "row_mask: identity expert placement only"
    EXTRA = c.EW - ER
    if ECOPY:
        nb_picks = rup(B * c.TOPK)
        scratch.update({
            "ecopy": pltpu.SMEM((2, 1, E), F32),
            "ec_sem": pltpu.SemaphoreType.DMA((2,)),
            "wlist": pltpu.SMEM((nb_picks,), I32),
            "cplist": pltpu.SMEM((nb_picks,), I32),
            "cpchoice": pltpu.SMEM((nb_picks,), I32),
            "rload": pltpu.SMEM((R,), I32),
        })
        if tiles.ecopy_opt == 1:
            scratch["ulist"] = pltpu.SMEM((nb_picks,), I32)
    if SPLIT:
        scratch.update({
            "qg": pltpu.VMEM((G, B, 8, HD), F32),  # [g]: group rank g's query rows (bf16-exact)
            "pso": pltpu.VMEM((G, B, 8, PW), F32),  # [g]: partials of group rank g's heads
            "pri": pltpu.VMEM((G, B, 8, PW), F32),  # [g]: partials of own heads from rank g
            "sq_ssem": pltpu.SemaphoreType.DMA((G - 1,)),
            "sq_rsem": pltpu.SemaphoreType.DMA((G - 1,)),
            "sp_ssem": pltpu.SemaphoreType.DMA((G - 1,)),
            "sp_rsem": pltpu.SemaphoreType.DMA((G - 1,)),
        })
    elif VER:
        scratch["qg"] = pltpu.VMEM((1, B, 8, HD), F32)  # own query rows (bf16-exact)
    if VER and not GCH and not FUSED:
        scratch.update({
            "ub": pltpu.VMEM((NS, rup(T, 8), QKV), F32),  # [s, t]: token t's conv output
            "rin_sem": pltpu.SemaphoreType.DMA((NS,)),  # recurrent state in, per sequence
        })
    if RSAG:  # round 2: this rank's chunk sum, and the chunk sums gathered from every rank
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, CW), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, R, AR_R, CW), BF16)
    elif HIER == 1:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, AR_C), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, 4, AR_R, AR_C), BF16)
    elif HIER == 2:
        scratch["ag_send"] = pltpu.VMEM((2, AR_R, HC2), BF16)
        scratch["ag_recv"] = pltpu.VMEM((2, 4, AR_R, HC2), BF16)
        scratch["s2_send"] = pltpu.VMEM((2, AR_R, HC2), BF16)
        scratch["s2_recv"] = pltpu.VMEM((2, 2, AR_R, HC2), BF16)
    if forced_routing:
        scratch["rb"] = pltpu.VMEM((2, M, E), F32)
        scratch["rb_sem"] = pltpu.SemaphoreType.DMA((2,))
    if GCH:  # [buffer, CB * vh, KD, VD]: one row chunk of a layer's key-major states
        scratch["recb"] = pltpu.VMEM((NRB, CBV, KD, VD), BF16)
        scratch["rec_sem"] = pltpu.SemaphoreType.DMA((2, NRB))  # [in / out, buffer]
    elif WIDE and c.NGDN == 0 and not FUSED:  # attention-only stack: no GatedDeltaNet buffers
        for k in ("recb", "sbuf", "convb"):
            del scratch[k]
    if PACK:  # [g, tile]: PB rows' slabs per tile (qrow)
        scratch["qg"] = pltpu.VMEM((G, NBP, 8, HD), F32)
        scratch["pso"] = pltpu.VMEM((G, NBP, 8, PW), F32)
        scratch["pri"] = pltpu.VMEM((G, NBP, 8, PW), F32)
    vshape = {k: (tk[k], kinds[k][1]) for k in kinds}
    vdtype = {k: kinds[k][2] for k in kinds}
    for k in kinds:
        assert pool_alias.storage_size(vshape[k], jnp.dtype(vdtype[k])) <= (
            tiles.ebank if k in EXPERT else tiles.bank), k
    if typed:
        for k in DENSE:
            scratch["dring_" + k] = pltpu.VMEM((NBD,) + vshape[k], vdtype[k])
        for k in EXPERT:
            scratch["ering_" + k] = pltpu.VMEM((NBE,) + vshape[k], vdtype[k])
    else:
        scratch["dpool"] = pltpu.VMEM((NBD * BROWS, 128), U8)
        scratch["epool"] = pltpu.VMEM((NBE * EBROWS, 128), U8)
    s_names = list(scratch)
    if FUSED:
        st_names = ("raw", "page_table", "ckpt_indices", "read_offsets") + (
            ("active_seqs",) if c.act_mode == "ref" else ()
        )
        o_names = ("hidden", "raw") if bool(interpret) else ("hidden",)
    elif c.paged_kv:
        st_names = ("conv", "rec", "raw", "page_table")
        o_names = O_NAMES
    else:
        st_names = S_NAMES
        o_names = O_NAMES
    n_w, n_s = len(W_NAMES), len(st_names)

    def mm(x, w, dims=NN):
        if interpret and x.dtype != w.dtype:
            w = w.astype(x.dtype)
        return lax.dot_general(x, w, dims, preferred_element_type=F32)

    def kernel(*refs):
        pos_ref, hid_ref = refs[:2]
        W = dict(zip(W_NAMES, refs[2:2 + n_w]))
        ST = dict(zip(st_names, refs[2 + n_w:2 + n_w + n_s]))
        k_ = 2 + n_w + n_s
        rb_hbm = None
        if forced_routing:
            rb_hbm = refs[k_]
            k_ += 1
        mr_ref = None
        if row_mask:
            mr_ref = refs[k_]
            k_ += 1
        O = dict(zip(o_names, refs[k_:k_ + len(o_names)]))
        out_ref = O["hidden"]
        raw_refs = (
            (O["raw"] if bool(interpret) else ST["raw"])
            if FUSED
            else (ST["raw"] if c.paged_kv else None)
        )
        scr = dict(zip(s_names, refs[k_ + len(o_names):]))
        me = lax.axis_index("tp").astype(I32)
        gi = lax.rem(me, G) if SPLIT else jnp.int32(0)  # place in the KV-head group
        lane_e8 = lax.broadcasted_iota(I32, (MR, E), 1)
        lane128 = lax.broadcasted_iota(I32, (MR, 128), 1)
        row8 = lax.broadcasted_iota(I32, (MR, 1), 0)
        row_m = lax.broadcasted_iota(I32, (MB, 1), 0)
        live = row_m < B
        EM = (1 << max(1, (E - 1).bit_length())) - 1  # top-k key: low mantissa bits = lane
        FLT_MIN = float(jnp.finfo(F32).tiny)

        def is_seq_active(s):
            if c.act_mode == "ref":
                return ST["active_seqs"][s] != 0
            if c.act_mode == "from_ckpt":
                return (
                    (ST["ckpt_indices"][0, s, 0] > 0)
                    if ckpt_is_3d
                    else (ST["ckpt_indices"][s, 0] > 0)
                )
            if c.act_mode == "pos":
                return pos_ref[s * T] >= 0
            return jnp.bool_(True)

        def gdn_roff(sub, s):
            return ST["read_offsets"][sub, s] if roff_is_2d else ST["read_offsets"][s]

        def gdn_pb_in(sub, s):
            roff = gdn_roff(sub, s)
            if ckpt_is_3d:
                return ST["ckpt_indices"][sub, s, roff] * 5
            return ST["ckpt_indices"][s, roff] * (5 * G_PER_ATT) + sub * 5

        def gdn_pb_out(sub, s, t):
            if ckpt_is_3d:
                return ST["ckpt_indices"][sub, s, t + ckpt_t_off] * 5
            return ST["ckpt_indices"][s, t + ckpt_t_off] * (5 * G_PER_ATT) + sub * 5

        def gdn_contig_ckpt(sub, s):
            if not ckpt_is_3d or T <= 1:
                return jnp.bool_(T == 1)
            s0 = ST["ckpt_indices"][sub, s, ckpt_t_off]
            ok = jnp.bool_(True)
            for t in range(1, T):
                ok = jnp.logical_and(
                    ok, ST["ckpt_indices"][sub, s, t + ckpt_t_off] == s0 + t
                )
            return ok

        def wait_gdn_out(slot_o):
            out_page = scr["gdn_out_page"]
            out_sem = scr["gdn_out_sem"]
            out_mode = scr["gdn_out_mode"]
            mode = out_mode[slot_o]

            @pl.when(mode == 1)
            def _():
                pltpu.make_async_copy(
                    out_page.at[slot_o],
                    raw_refs[0].at[pl.ds(0, 5 * T)],
                    out_sem.at[slot_o, 0],
                ).wait()

            if T > 1:
                @pl.when(mode == 2)
                def _():
                    for t in range(T):
                        pltpu.make_async_copy(
                            out_page.at[slot_o, pl.ds(t * 5, 5)],
                            raw_refs[0].at[pl.ds(0, 5)],
                            out_sem.at[slot_o, t],
                        ).wait()

            out_mode[slot_o] = jnp.int32(0)

        def wait_kv_wr(s):
            wr_pend = scr["kv_wr_pend"]
            pend = wr_pend[s]

            @pl.when(pend != 0)
            def _():
                pltpu.make_async_copy(
                    scr["kv_scat_page"].at[s], raw_refs[0].at[0], scr["kv_wr_sem"].at[s]
                ).wait()

            wr_pend[s] = jnp.int32(0)

        def batch_to_i32_cols(tok_mr_hd):
            rows_bf16 = tok_mr_hd.astype(BF16)
            if MR < c.page_size:
                rows_bf16 = jnp.pad(rows_bf16, ((0, c.page_size - MR), (0, 0)))
            cols_fp8 = jnp.swapaxes(rows_bf16, 0, 1).astype(FP8)
            return pltpu.bitcast(cols_fp8, I32)

        # ------------------------------ weight rings ------------------------------
        def dview(kind, bank):
            if typed:
                return scr["dring_" + kind].at[bank]
            shp, dt = vshape[kind], jnp.dtype(vdtype[kind])
            return pool_alias.view(
                scr["dpool"].at[pl.ds(bank * BROWS, pool_alias.storage_size(shp, dt) // 128)],
                shp, dt)

        def eview(kind, bank):
            if typed:
                return scr["ering_" + kind].at[bank]
            shp, dt = vshape[kind], jnp.dtype(vdtype[kind])
            return pool_alias.view(
                scr["epool"].at[pl.ds(bank * EBROWS, pool_alias.storage_size(shp, dt) // 128)],
                shp, dt)

        def dcopy(kind, src, bank):
            return pltpu.make_async_copy(src, dview(kind, bank), scr["dsem"].at[bank])

        def ecopy(kind, src, bank):
            return pltpu.make_async_copy(src, eview(kind, bank), scr["esem"].at[bank])

        def seg_issue(segs, r, ords, bank):
            for kind, lo, hi, ordi in segs:
                @pl.when(jnp.logical_and(r >= lo, r < hi))
                def _(kind=kind, lo=lo, ordi=ordi):
                    dcopy(kind, W[kind].at[ords[ordi], r - lo], bank).start()

        def fetch(g):
            """Issue dense stream tile g into bank g % NBD (static schedule)."""
            @pl.when(jnp.logical_and(g < T_TOTAL, "dense" not in skip))
            def _():
                blk = g // t_block
                sub = g - blk * t_block
                lt = jnp.minimum(sub // t_gdn, AE - 1)
                r = sub - lt * t_gdn
                layer = blk * AE + lt
                bank = g % NBD

                if AE > 1:
                    @pl.when(lt < AE - 1)
                    def _gdn():
                        a0 = nt["lin_in"]
                        a1 = a0 + nt["lin_out"]
                        a2 = a1 + nt["router"]
                        a3 = a2 + nt["sh13"]
                        seg_issue(
                            [("lin_in", 0, a0, 0), ("lin_out", a0, a1, 0),
                             ("router", a1, a2, 1), ("sh13", a2, a3, 1), ("sh2", a3, t_gdn, 1)],
                            r, (blk * (AE - 1) + lt, layer), bank)

                @pl.when(lt == AE - 1)
                def _att():
                    a0 = nt["att_in"]
                    a1 = a0 + nt["att_out"]
                    a2 = a1 + nt["router"]
                    a3 = a2 + nt["sh13"]
                    seg_issue(
                        [("att_in", 0, a0, 0), ("att_out", a0, a1, 0), ("router", a1, a2, 1),
                         ("sh13", a2, a3, 1), ("sh2", a3, t_att, 1)],
                        r, (blk, layer), bank)

        def dtake(kind, g):
            bank = g % NBD
            if "dense" not in skip:
                with sc("dwait", 2):
                    dcopy(kind, W[kind].at[0, 0], bank).wait()
            return dview(kind, bank)

        def dgive(g):
            fetch(g + NBD)

        def flush(g0, n):
            """Refill the slots of tiles g0 .. g0+n-1, consumed lazily (no refill yet)."""
            for t in range(n):
                fetch(g0 + t + NBD)

        def dmatmul(kind, g0, xsrc, N, lazy=False):
            """acc[:, :N] = xsrc[:, :K] @ W(kind), tiles g0 .. g0+nt-1 of the dense stream. lazy:
            leave the consumed slots for a later flush."""
            n_t, t_k = nt[kind], tk[kind]
            acc = scr["acc"]
            acc[:, :N] = jnp.zeros((MB, N), F32)

            def body(t, carry):
                g = g0 + t
                w = dtake(kind, g)
                k0 = pl.multiple_of(t * t_k, t_k)
                acc[:, :N] += mm(xsrc[:, pl.ds(k0, t_k)], w[...])
                if not lazy:
                    dgive(g)
                return carry

            lax.fori_loop(0, n_t, body, 0)
            return acc[:, :N]

        # expert stream: flat unit u within the layer, i = u // TE listed expert, t = u % TE tile
        def fetch_e(u, l, t=None):
            """Issue unit u into bank u % NBE. t: u's tile index when known at trace time (static
            u, or a pipelined refill), which drops the runtime tile dispatch."""
            if t is None and isinstance(u, int):
                t = u % TE
            i = u // TE if t is None else (u - t) // TE
            if ECOPY:
                w = scr["wlist"][i]
            else:
                e = scr["elist"][i]
                w = e if c.moe_tp else e - me * ER
            bank = u % NBE

            def es_copy():
                s = i % NSS
                pltpu.make_async_copy(W["es"].at[l, w], scr["es"].at[s],
                                      scr["es_sem"].at[s]).start()

            def e13_start(tt):
                if c.stack_e:
                    ecopy("e13", W["e13"].at[l, w, tt], bank).start()
                else:
                    bswitch(l, [lambda r=r: ecopy("e13", r.at[w, tt], bank).start()
                                for r in W["e13"]])

            def e2_start(tt):
                if c.stack_e:
                    ecopy("e2", W["e2"].at[l, w, tt], bank).start()
                else:
                    bswitch(l, [lambda r=r: ecopy("e2", r.at[w, tt], bank).start()
                                for r in W["e2"]])

            if t is not None:
                if t < T13:
                    e13_start(t)
                else:
                    e2_start(t - T13)
                if t == 0:
                    es_copy()
                return
            t = u - i * TE

            @pl.when(t < T13)
            def _():
                e13_start(t)

            @pl.when(t >= T13)
            def _():
                e2_start(t - T13)

            pl.when(t == 0)(es_copy)

        def etake(kind, u):
            bank = u % NBE
            with sc("ewait", 2):
                w0 = W[kind].at[0, 0, 0] if c.stack_e else W[kind][0].at[0, 0]
                ecopy(kind, w0, bank).wait()
            return eview(kind, bank)

        def egive(u, l, t=None):
            nu = u + NBE
            nt_ = None if t is None or NBE % TE else t

            @pl.when(nu < scr["st"][0])
            def _():
                fetch_e(nu, l, nt_)

        # ------------------------------ small helpers ------------------------------
        def sm(slot, name, n=None):
            o, w = lay[name]
            return scr["sm"][slot, :, o:o + (w if n is None else n)]

        def rms_zc(x, w):
            v = x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + c.eps)
            return rnd(v * (1.0 + w))

        def ar_copies(slot, rnd_i=0):
            """Remote copies of all-reduce round rnd_i of this slot to the peers me ^ off. A peer's
            copy to this rank uses the same off, so it lands on the same semaphore and size."""
            if HIER == 3:
                off = 1 << rnd_i
                bit = lax.rem(me // off, 2)
                return [pltpu.make_async_remote_copy(
                    src_ref=scr["send"].at[rnd_i, slot], dst_ref=scr["recv"].at[rnd_i, slot, bit],
                    send_sem=scr["ar_ssem"].at[rnd_i, slot, 0],
                    recv_sem=scr["ar_rsem"].at[rnd_i, slot, 0],
                    device_id=(me ^ off,), device_id_type=pl.DeviceIdType.MESH)]
            if HIER in (1, 2):
                c_idx = lax.rem(me, 2)
                if rnd_i == 0:
                    src = scr["send"].at[slot, 1 - c_idx] if HIER == 2 else scr["send"].at[slot]
                    peers = [(0, 1)]
                    dst = scr["recv"].at[slot, c_idx]
                elif rnd_i == 1:
                    src = scr["ag_send"].at[slot]
                    peers = list(enumerate((2, 4, 6)))
                    dst = scr["ag_recv"].at[slot, me // 2]
                else:
                    src = scr["s2_send"].at[slot]
                    peers = [(0, 1)]
                    dst = scr["s2_recv"].at[slot, c_idx]
                return [pltpu.make_async_remote_copy(
                    src_ref=src, dst_ref=dst,
                    send_sem=scr["ar_ssem"].at[rnd_i, slot, i],
                    recv_sem=scr["ar_rsem"].at[rnd_i, slot, i],
                    device_id=(me ^ off,), device_id_type=pl.DeviceIdType.MESH)
                    for i, off in peers]
            if RSAG and rnd_i == 0:  # lane chunk p of this rank's partial -> rank p
                src, dst = (lambda off: scr["send"].at[slot, me ^ off]), scr["recv"].at[slot, me]
            elif RSAG:  # this rank's chunk sum -> every peer
                src, dst = (lambda off: scr["ag_send"].at[slot]), scr["ag_recv"].at[slot, me]
            else:  # the whole partial -> every peer
                src, dst = (lambda off: scr["send"].at[slot]), scr["recv"].at[slot, me]
            return [pltpu.make_async_remote_copy(
                src_ref=src(off), dst_ref=dst,
                send_sem=scr["ar_ssem"].at[rnd_i, slot, off - 1],
                recv_sem=scr["ar_rsem"].at[rnd_i, slot, off - 1],
                device_id=(me ^ off,), device_id_type=pl.DeviceIdType.MESH)
                for off in range(1, R)]

        comm = R > 1 and "ar" not in skip

        def ar_round(slot, rnd_i):
            descs = ar_copies(slot, rnd_i)
            for d in descs:
                d.start()
            for d in descs:
                d.wait_recv()

        def allreduce(slot, x, first):
            """x [M, H] f32 partial -> the sum over ranks of its B live rows ([MR, H] when WAR,
            else folded [PR, PC]). Only the incoming payloads are waited for; this slot's sends are
            waited for when its send buffers are next rewritten (not first) and at kernel exit."""
            send, recv = scr["send"], scr["recv"]
            with sc("ar_pack"):
                if WAR:
                    val = x[:MR]
                else:
                    chunks = [x[0:B, j * PC:(j + 1) * PC] for j in range(HPC)]
                    if B * HPC < PR:
                        chunks.append(jnp.zeros((PR - B * HPC, PC), F32))
                    val = jnp.concatenate(chunks, axis=0)
                if comm:
                    @pl.when(jnp.logical_not(first))
                    def _():
                        for rnd_i in AR_ROUNDS:
                            for d in ar_copies(slot, rnd_i):
                                d.wait_send()

                c_idx = lax.rem(me, 2)
                if HIER == 3:
                    send[0, slot] = val.astype(BF16)
                    recv[0, slot, c_idx] = send[0, slot]
                elif HIER == 2:
                    send[slot, 0] = val[:, :HC2].astype(BF16)
                    send[slot, 1] = val[:, HC2:].astype(BF16)
                    recv[slot, c_idx] = send[slot, c_idx]
                elif HIER == 1:
                    send[slot] = val.astype(BF16)
                    recv[slot, c_idx] = send[slot]
                elif RSAG:
                    for j in range(R):
                        send[slot, j] = val[:, j * CW:(j + 1) * CW].astype(BF16)
                    recv[slot, me] = send[slot, me]
                else:
                    send[slot] = val.astype(BF16)
                    recv[slot, me] = send[slot]
            if not comm:
                return rnd(val)
            with sc("ar_comm"):
                ar_round(slot, 0)
            if HIER == 3:
                with sc("ar_sum"):
                    b1 = lax.rem(me // 2, 2)
                    send[1, slot] = (recv[0, slot, 0].astype(F32)
                                     + recv[0, slot, 1].astype(F32)).astype(BF16)
                    recv[1, slot, b1] = send[1, slot]
                with sc("ar_comm"):
                    ar_round(slot, 1)
                with sc("ar_sum"):
                    b2 = lax.rem(me // 4, 2)
                    send[2, slot] = (recv[1, slot, 0].astype(F32)
                                     + recv[1, slot, 1].astype(F32)).astype(BF16)
                    recv[2, slot, b2] = send[2, slot]
                with sc("ar_comm"):
                    ar_round(slot, 2)
                with sc("ar_sum"):
                    return recv[2, slot, 0].astype(F32) + recv[2, slot, 1].astype(F32)
            if HIER in (1, 2):
                ag_send, ag_recv = scr["ag_send"], scr["ag_recv"]
                with sc("ar_sum"):
                    ag_send[slot] = (recv[slot, 0].astype(F32)
                                     + recv[slot, 1].astype(F32)).astype(BF16)
                    ag_recv[slot, me // 2] = ag_send[slot]
                with sc("ar_comm"):
                    ar_round(slot, 1)
                if HIER == 1:
                    with sc("ar_sum"):
                        return jnp.sum(ag_recv[slot].astype(F32), axis=0)
                s2_send, s2_recv = scr["s2_send"], scr["s2_recv"]
                with sc("ar_sum"):
                    s2_send[slot] = jnp.sum(ag_recv[slot].astype(F32), axis=0).astype(BF16)
                    s2_recv[slot, c_idx] = s2_send[slot]
                with sc("ar_comm"):
                    ar_round(slot, 2)
                with sc("ar_sum"):
                    return jnp.concatenate([s2_recv[slot, 0], s2_recv[slot, 1]],
                                           axis=-1).astype(F32)
            if not RSAG:
                with sc("ar_sum"):
                    return jnp.sum(recv[slot].astype(F32), axis=0)
            ag_send, ag_recv = scr["ag_send"], scr["ag_recv"]
            with sc("ar_sum"):
                ag_send[slot] = jnp.sum(recv[slot].astype(F32), axis=0).astype(BF16)
                ag_recv[slot, me] = ag_send[slot]
            with sc("ar_comm"):
                ar_round(slot, 1)
            with sc("ar_sum"):
                return jnp.concatenate([ag_recv[slot, j] for j in range(R)], axis=-1).astype(F32)

        def add_residual(tot):
            res = scr["res"]
            with sc("ar_res"):
                if not WAR:
                    tot = jnp.concatenate([tot[j * B:(j + 1) * B] for j in range(HPC)], axis=-1)
                res[0:B, :] = rnd(res[0:B, :] + tot[0:B])

        # ------------------------------ GatedDeltaNet ------------------------------
        def rec_chunk(o, j, s, out):
            """GCH: chunk j (state heads j * CBV ..; static j == NF: the TR-row tail) of
            GatedDeltaNet layer o's states, HBM -> ring buffer s (out: buffer s -> the output
            states). T > 1: a chunk holds n // (T * vh) sequences; their input states are placed
            in the last n // T head slots of buffer s, where each sequence's token 0 reads its
            initial state before those slots are overwritten by the last sequence's tokens."""
            n = TR * vh if TR and isinstance(j, int) and j == NF else CBV
            r0 = j * CBV if isinstance(j, int) else pl.multiple_of(j * CBV, CBV)
            buf = scr["recb"].at[s] if n == CBV else scr["recb"].at[s, pl.ds(0, n)]
            if out:
                return pltpu.make_async_copy(buf, O["rec"].at[o, pl.ds(r0, n)],
                                             scr["rec_sem"].at[1, s])
            if VER:
                ni = n // T
                ri = j * (CBV // T) if isinstance(j, int) else pl.multiple_of(
                    j * (CBV // T), CBV // T)
                return pltpu.make_async_copy(ST["rec"].at[o, pl.ds(ri, ni)],
                                             scr["recb"].at[s, pl.ds(n - ni, ni)],
                                             scr["rec_sem"].at[0, s])
            return pltpu.make_async_copy(ST["rec"].at[o, pl.ds(r0, n)], buf,
                                         scr["rec_sem"].at[0, s])

        def rec_drain(o):
            """GCH: wait for layer o's write-backs still in flight (its last NRB chunks)."""
            for j_ in range(NCH - NRB, NCH):
                rec_chunk(o, j_, j_ % NRB, True).wait()

        def gdn(l, slot, g0):
            o = l - (l + 1) // AE
            if FUSED:
                in_page = scr["gdn_in_page"]
                out_page = scr["gdn_out_page"]
                in_sem = scr["gdn_in_sem"]
                out_sem = scr["gdn_out_sem"]
                in_pend = scr["gdn_in_pend"]
                out_mode = scr["gdn_out_mode"]
                sb = scr["sbuf"]

                def start_gdn_in(o_, s):
                    slot_i = s % NGI
                    sub = lax.rem(o_, G_PER_ATT)
                    roff = gdn_roff(sub, s)
                    act_in = jnp.logical_and(is_seq_active(s), roff >= 0)
                    in_pend[slot_i] = jnp.where(act_in, 1, 0).astype(I32)

                    @pl.when(act_in)
                    def _():
                        g_pool = lax.div(o_, G_PER_ATT)
                        pb = gdn_pb_in(sub, s)

                        def do_in(r):
                            pltpu.make_async_copy(
                                r.at[pl.ds(pb, 5)], in_page.at[slot_i], in_sem.at[slot_i]
                            ).start()

                        bswitch(g_pool, [lambda r=r: do_in(r) for r in raw_refs])

                def wait_gdn_in(slot_i):
                    pend = in_pend[slot_i]

                    @pl.when(pend != 0)
                    def _():
                        pltpu.make_async_copy(
                            raw_refs[0].at[pl.ds(0, 5)],
                            in_page.at[slot_i],
                            in_sem.at[slot_i],
                        ).wait()

                    in_pend[slot_i] = jnp.int32(0)
                    return pend != 0

                def start_gdn_out(o_, s):
                    slot_o = s % NGO
                    act_out = is_seq_active(s)
                    sub = lax.rem(o_, G_PER_ATT)
                    g_pool = lax.div(o_, G_PER_ATT)
                    if T == 1:
                        out_mode[slot_o] = jnp.where(act_out, 1, 0).astype(I32)

                        @pl.when(act_out)
                        def _():
                            pb0 = gdn_pb_out(sub, s, 0)

                            def do_out(r):
                                pltpu.make_async_copy(
                                    out_page.at[slot_o],
                                    r.at[pl.ds(pb0, 5)],
                                    out_sem.at[slot_o, 0],
                                ).start()

                            bswitch(g_pool, [lambda r=r: do_out(r) for r in raw_refs])
                    else:
                        contig = gdn_contig_ckpt(sub, s)
                        out_mode[slot_o] = jnp.where(
                            act_out, jnp.where(contig, 1, 2), 0
                        ).astype(I32)

                        @pl.when(jnp.logical_and(act_out, contig))
                        def _():
                            pb0 = gdn_pb_out(sub, s, 0)

                            def do_out_c(r):
                                pltpu.make_async_copy(
                                    out_page.at[slot_o],
                                    r.at[pl.ds(pb0, 5 * T)],
                                    out_sem.at[slot_o, 0],
                                ).start()

                            bswitch(g_pool, [lambda r=r: do_out_c(r) for r in raw_refs])

                        @pl.when(jnp.logical_and(act_out, jnp.logical_not(contig)))
                        def _():
                            def do_out_nc(r):
                                for t in range(T):
                                    pb_t = gdn_pb_out(sub, s, t)
                                    pltpu.make_async_copy(
                                        out_page.at[slot_o, pl.ds(t * 5, 5)],
                                        r.at[pl.ds(pb_t, 5)],
                                        out_sem.at[slot_o, t],
                                    ).start()

                            bswitch(g_pool, [lambda r=r: do_out_nc(r) for r in raw_refs])

                with sc("gdn_in"):
                    for s_ in range(NGI):
                        start_gdn_in(o, s_)
                    y = dmatmul("lin_in", g0, scr["xb"], c.NLI)
                    ybuf = scr["ybuf"]
                    ybuf[:, :c.NLI] = rnd(y * sm(slot, "in_s", c.NLI))

                ow, _ = lay["conv"]
                taps = [scr["sm"][slot, :, ow + t * QKV:ow + (t + 1) * QKV] for t in range(4)]
                alog = sm(slot, "alog")[0, :vh]
                dtb = sm(slot, "dtb")[0, :vh]
                gnw = sm(slot, "gnorm")[0, :].astype(BF16)
                expand = vh // kh
                row_iota_conv = lax.broadcasted_iota(I32, (A_conv, QKV), 0)

                def cols(x):
                    xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
                    return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(vh, KD, VD)

                for s in range(NS):
                    slot_i = s % NGI
                    slot_o = s % NGO
                    with sc("gdn_rec_wait"):
                        act_in = wait_gdn_in(slot_i)
                    if typed:
                        u32_in = pltpu.bitcast(
                            in_page[slot_i, 0:4], U32
                        ).reshape(512, 128)[:ssm_u32_rows, :]
                        raw_conv = (
                            pltpu.bitcast(in_page[slot_i, 4], BF16)
                            .reshape(256, 128)[:conv8_rows, :]
                            .reshape(A_conv, QKV)
                            .astype(F32)
                        )
                    else:
                        u32_in = (
                            in_page.at[slot_i, pl.ds(0, 4)]
                            .bitcast(U32)
                            .reshape(512, 128)[:ssm_u32_rows, :]
                        )
                        raw_conv = (
                            in_page.at[slot_i, 4]
                            .bitcast(BF16)
                            .reshape(256, 128)[:conv8_rows, :]
                            .reshape(A_conv, QKV)
                            .astype(F32)
                        )
                    u32_in = jnp.where(act_in, u32_in, jnp.uint32(0))
                    s_bf16 = pltpu.bitcast(u32_in, BF16).reshape(vh, KD, VD)
                    raw_conv = jnp.where(act_in, raw_conv, 0.0)
                    c0 = raw_conv[0:1, :]
                    c1 = raw_conv[1:2, :]
                    c2 = raw_conv[2:3, :]
                    if s + NGI < NS:
                        start_gdn_in(o, s + NGI)
                    wait_gdn_out(slot_o)
                    if "gdn" not in skip:
                        with sc("gdn_rec"):
                            for t in range(T):
                                b = s * T + t
                                x_t = ybuf[b:b + 1, 0:QKV]
                                u_t = rnd(
                                    jax.nn.silu(
                                        c0 * taps[0]
                                        + c1 * taps[1]
                                        + c2 * taps[2]
                                        + x_t * taps[3]
                                    )
                                )
                                c0, c1, c2 = c1, c2, x_t
                                q = u_t[0, :kh * KD].reshape(kh, KD)
                                k = u_t[0, kh * KD:2 * kh * KD].reshape(kh, KD)
                                v = u_t[0, 2 * kh * KD:].reshape(vh, VD)
                                q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
                                q = q * KD ** -0.5
                                k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                                qc, kc_ = cols(q), cols(k)
                                beta = rnd(jax.nn.sigmoid(ybuf[b, BA:BA + vh]))
                                a = ybuf[b, BA + 64:BA + 64 + vh]
                                g = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
                                sb[...] = s_bf16.astype(F32) * jnp.exp(g)[:, None, None]
                                pred = jnp.sum(sb[...] * kc_, axis=1)
                                delta = (v - pred) * beta[:, None]
                                sb[...] = sb[...] + kc_ * delta[:, None, :]
                                out = rnd(jnp.sum(sb[...] * qc, axis=1))
                                s_bf16 = sb[...].astype(BF16)
                                val = out * lax.rsqrt(
                                    jnp.mean(out * out, axis=-1, keepdims=True) + c.eps
                                )
                                val = (gnw[None, :] * val.astype(BF16)).astype(F32)
                                zz = ybuf[b, QKV:QKV + Z].reshape(vh, VD)
                                scr["mix"][b, :Z] = rnd(val * jax.nn.silu(zz)).reshape(Z)

                                u32_out = pltpu.bitcast(s_bf16.reshape(vh * KD, VD), U32)
                                c0_f = jnp.broadcast_to(c0, (A_conv, QKV))
                                c1_f = jnp.broadcast_to(c1, (A_conv, QKV))
                                c2_f = jnp.broadcast_to(c2, (A_conv, QKV))
                                w_conv = jnp.where(
                                    row_iota_conv == 0,
                                    c0_f,
                                    jnp.where(
                                        row_iota_conv == 1,
                                        c1_f,
                                        jnp.where(row_iota_conv == 2, c2_f, 0.0),
                                    ),
                                ).astype(BF16)
                                if typed:
                                    u32_p = (
                                        jnp.pad(u32_out, ((0, 512 - ssm_u32_rows), (0, 0)))
                                        if ssm_u32_rows < 512
                                        else u32_out
                                    )
                                    out_page[slot_o, t * 5:t * 5 + 4] = pltpu.bitcast(
                                        u32_p.reshape(4, 2, 64, 1, 128), FP8
                                    )
                                    conv_p = w_conv.reshape(conv8_rows, 128)
                                    if conv8_rows < 256:
                                        conv_p = jnp.pad(conv_p, ((0, 256 - conv8_rows), (0, 0)))
                                    out_page[slot_o, t * 5 + 4] = pltpu.bitcast(
                                        conv_p.reshape(2, 64, 2, 128), FP8
                                    )
                                else:
                                    out_page.at[slot_o, pl.ds(t * 5, 4)].bitcast(U32).reshape(
                                        512, 128
                                    )[:ssm_u32_rows, :] = u32_out
                                    out_page.at[slot_o, t * 5 + 4].bitcast(BF16).reshape(
                                        256, 128
                                    )[:conv8_rows, :] = w_conv.reshape(conv8_rows, 128)
                    start_gdn_out(o, s)

                with sc("gdn_out"):
                    scr["xb2"][:, :Z] = scr["mix"][:, :Z].astype(BF16)
                    yo = dmatmul("lin_out", g0 + nt["lin_in"], scr["xb2"], H)
                    return yo * sm(slot, "out_s")

            convb, recb, sem = scr["convb"], scr["recb"], scr["st_sem"]

            def rec_in():
                """Recurrent state loads. T > 1: sequence s's state goes to the slot of its token 0
                (row s * T), where its recurrence starts."""
                if not VER:
                    return [pltpu.make_async_copy(ST["rec"].at[o], recb, sem.at[1])]
                return [pltpu.make_async_copy(ST["rec"].at[o, pl.ds(s * vh, vh)],
                                              recb.at[pl.ds(s * T * vh, vh)],
                                              scr["rin_sem"].at[s]) for s in range(NS)]

            with sc("gdn_in"):
                @pl.when(o > 0)
                def _():
                    pltpu.make_async_copy(convb, O["conv"].at[0], sem.at[2]).wait()
                    if GCH:  # one write-back per buffer is still in flight
                        rec_drain(0)
                    else:
                        pltpu.make_async_copy(recb, O["rec"].at[0], sem.at[3]).wait()

                pltpu.make_async_copy(ST["conv"].at[o], convb, sem.at[0]).start()
                if GCH:
                    for s_ in range(NRB):
                        rec_chunk(o, s_, s_, False).start()
                else:
                    for d in rec_in():
                        d.start()
                y = dmatmul("lin_in", g0, scr["xb"], c.NLI)
                ybuf = scr["ybuf"]
                ybuf[:, :c.NLI] = rnd(y * sm(slot, "in_s", c.NLI))
            with sc("gdn_conv"):
                pltpu.make_async_copy(ST["conv"].at[o], convb, sem.at[0]).wait()
                ow, _ = lay["conv"]
                taps = [scr["sm"][slot, :, ow + t * QKV:ow + (t + 1) * QKV] for t in range(4)]
                u = None
                if VER:
                    # history rows 3 + t <- token t's input; token t's window is rows t .. t+3
                    # (the cached taps, then new inputs 0..t)
                    us = []
                    for s in range(NS):
                        convb[s, 3:3 + T, :] = ybuf[s * T:(s + 1) * T, 0:QKV]
                        w = [convb[s, k:k + T, :] for k in range(4)]
                        u_ = rnd(jax.nn.silu(
                            w[0] * taps[0] + w[1] * taps[1] + w[2] * taps[2] + w[3] * taps[3]))
                        if GCH:
                            us.append(u_)
                        else:
                            scr["ub"][s, 0:T, :] = u_
                    if GCH:
                        u = jnp.concatenate(us, axis=0)
                else:
                    x = ybuf[0:B, 0:QKV]
                    s0, s1, s2 = convb[0:B, :], convb[B:2 * B, :], convb[2 * B:3 * B, :]
                    u = rnd(jax.nn.silu(s0 * taps[0] + s1 * taps[1] + s2 * taps[2] + x * taps[3]))
                    convb[0:B, :] = s1
                    convb[B:2 * B, :] = s2
                    convb[2 * B:3 * B, :] = x
                pltpu.make_async_copy(convb, O["conv"].at[o], sem.at[2]).start()
                if GCH:  # the chunk loop reads u rows from ybuf (x is consumed)
                    ybuf[0:B, 0:QKV] = u
            if GCH:
                with sc("gdn_rec"):
                    gdn_rec_chunked(o, slot)
            else:
                with sc("gdn_rec_wait"):
                    for d in rec_in():
                        d.wait()
                if "gdn" not in skip:
                    with sc("gdn_rec"):
                        gdn_rec(u, ybuf, slot)
            with sc("gdn_out"):
                if not GCH:
                    pltpu.make_async_copy(recb, O["rec"].at[o], sem.at[3]).start()
                scr["xb2"][:, :Z] = scr["mix"][:, :Z].astype(BF16)
                yo = dmatmul("lin_out", g0 + nt["lin_in"], scr["xb2"], H)
                return yo * sm(slot, "out_s")

        def gdn_rec(u, ybuf, slot):
            """Gated delta rule on the key-major bf16 states recb [B * vh, KD, VD] (in place), then
            the gated RMSNorm and SiLU(z) gate into mix[:B, :Z]. Key-major makes both state
            contractions sublane reductions and broadcasts delta along sublanes; q / k become
            lane-broadcast columns through one [VD, KD] transpose per key head.
            T > 1: token t of a sequence starts from token t-1's state (slot b - 1, just written
            as bf16) and token 0 from the loaded state (slot b); conv outputs come from ub."""
            recb, sb = scr["recb"], scr["sbuf"]
            alog = sm(slot, "alog")[0, :vh]
            dtb = sm(slot, "dtb")[0, :vh]
            gnw = sm(slot, "gnorm")[0, :].astype(BF16)
            expand = vh // kh

            def cols(x):  # [kh, KD] -> [vh, KD, VD] holding x[h // expand, i] at [h, i, :]
                xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
                return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(vh, KD, VD)

            for b in range(B):
                src = b
                if VER:
                    s_, t_ = divmod(b, T)
                    urow_ = scr["ub"][s_, t_:t_ + 1, :]
                    q = urow_[0, :kh * KD].reshape(kh, KD)
                    k = urow_[0, kh * KD:2 * kh * KD].reshape(kh, KD)
                    v = urow_[0, 2 * kh * KD:].reshape(vh, VD)
                    src = b if t_ == 0 else b - 1
                else:
                    q = u[b, :kh * KD].reshape(kh, KD)
                    k = u[b, kh * KD:2 * kh * KD].reshape(kh, KD)
                    v = u[b, 2 * kh * KD:].reshape(vh, VD)
                q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
                q = q * KD ** -0.5
                k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                qc, kc_ = cols(q), cols(k)
                beta = rnd(jax.nn.sigmoid(ybuf[b, BA:BA + vh]))
                a = ybuf[b, BA + 64:BA + 64 + vh]
                g = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
                sb[...] = recb[src * vh:(src + 1) * vh].astype(F32) * jnp.exp(g)[:, None, None]
                pred = jnp.sum(sb[...] * kc_, axis=1)
                delta = (v - pred) * beta[:, None]
                sb[...] = sb[...] + kc_ * delta[:, None, :]
                out = rnd(jnp.sum(sb[...] * qc, axis=1))
                recb[b * vh:(b + 1) * vh] = sb[...].astype(BF16)
                val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + c.eps)
                val = (gnw[None, :] * val.astype(BF16)).astype(F32)
                zz = ybuf[b, QKV:QKV + Z].reshape(vh, VD)
                scr["mix"][b, :Z] = rnd(val * jax.nn.silu(zz)).reshape(Z)

        def gdn_rec_chunked(o, slot):
            """GCH: gdn_rec over row chunks. Chunk j of layer o lives in ring buffer j % NRB; the
            prologue (gdn) fetched chunks 0 .. NRB-1. Iteration j waits for chunk j, updates its
            rows, writes it back, then refills the buffer of chunk j - 1 (written back one
            iteration earlier) with chunk j - 1 + NRB. A fori_loop runs the NF full chunks, the
            TR-row tail chunk follows statically. The last NRB write-backs stay in flight until
            the next GatedDeltaNet layer (or the end of the kernel)."""
            recb, sb, ybuf = scr["recb"], scr["sbuf"], scr["ybuf"]
            alog = sm(slot, "alog")[0, :vh]
            dtb = sm(slot, "dtb")[0, :vh]
            gnw = sm(slot, "gnorm")[0, :].astype(BF16)
            expand = vh // kh

            def cols(x):  # [kh, KD] -> [vh, KD, VD] holding x[h // expand, i] at [h, i, :]
                xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
                return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(vh, KD, VD)

            def rows(r0, n, s):  # rows r0 .. r0 + n - 1 (r0 8-aligned), states in buffer s
                uc = ybuf[pl.ds(r0, n), 0:QKV]
                bac = ybuf[pl.ds(r0, n), BA:BA + 128]
                zc = ybuf[pl.ds(r0, n), QKV:QKV + Z]
                st = recb.at[s]
                row_i = lax.broadcasted_iota(I32, (n, 1), 0)
                blk = jnp.zeros((n, Z), F32)
                for bb in range(n):
                    h0 = bb * vh
                    q = uc[bb, :kh * KD].reshape(kh, KD)
                    k = uc[bb, kh * KD:2 * kh * KD].reshape(kh, KD)
                    v = uc[bb, 2 * kh * KD:].reshape(vh, VD)
                    q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
                    q = q * KD ** -0.5
                    k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                    qc, kc_ = cols(q), cols(k)
                    beta = rnd(jax.nn.sigmoid(bac[bb, 0:vh]))
                    a = bac[bb, 64:64 + vh]
                    g = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
                    src = ((n - n // T + bb // T) if bb % T == 0 else bb - 1) * vh if VER else h0
                    sb[...] = st[src:src + vh].astype(F32) * jnp.exp(g)[:, None, None]
                    pred = jnp.sum(sb[...] * kc_, axis=1)
                    delta = (v - pred) * beta[:, None]
                    sb[...] = sb[...] + kc_ * delta[:, None, :]
                    out = rnd(jnp.sum(sb[...] * qc, axis=1))
                    st[h0:h0 + vh] = sb[...].astype(BF16)
                    val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + c.eps)
                    val = (gnw[None, :] * val.astype(BF16)).astype(F32)
                    zz = zc[bb].reshape(vh, VD)
                    row = rnd(val * jax.nn.silu(zz)).reshape(1, Z)
                    blk = jnp.where(row_i == bb, row, blk)
                scr["mix"][pl.ds(r0, n), 0:Z] = blk

            def chunk(j, carry):
                s = lax.rem(j, NRB)
                with sc("gdn_rec_wait"):
                    rec_chunk(o, j, s, False).wait()
                if "gdn" not in skip:
                    rows(pl.multiple_of(j * CB, CB), CB, s)
                rec_chunk(o, j, s, True).start()
                jp, jn = j - 1, j - 1 + NRB

                @pl.when(jnp.logical_and(j >= 1, jn < NCH))
                def _():
                    sp = lax.rem(jp, NRB)
                    rec_chunk(o, jp, sp, True).wait()
                    if TR:  # the refill may be the tail chunk (its own DMA size)
                        @pl.when(jn == NF)
                        def _():
                            rec_chunk(o, NF, sp, False).start()

                        @pl.when(jn < NF)
                        def _():
                            rec_chunk(o, jn, sp, False).start()
                    else:
                        rec_chunk(o, jn, sp, False).start()

                return carry

            lax.fori_loop(0, NF, chunk, 0)
            if TR:  # tail chunk NF: no refill follows (NRB >= 2)
                s = NF % NRB
                with sc("gdn_rec_wait"):
                    rec_chunk(o, NF, s, False).wait()
                if "gdn" not in skip:
                    rows(NF * CB, TR, s)
                rec_chunk(o, NF, s, True).start()

        # ------------------------------ full attention ------------------------------
        def group_copies(src, dst, ssem, rsem):
            """Remote copies src(off) -> dst on group peer me ^ off, off = 1 .. G-1. A peer's copy
            to this rank uses the same off, so it lands on rsem[off - 1] with the same size."""
            return [pltpu.make_async_remote_copy(
                src_ref=src(off), dst_ref=dst, send_sem=ssem.at[off - 1],
                recv_sem=rsem.at[off - 1], device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH)
                for off in range(1, G)]

        def exchange(descs):
            for d in descs:
                d.start()
            for d in descs:
                d.wait_recv()
            for d in descs:
                d.wait_send()

        def attention(l, slot, g0):
            a = l // AE
            ksem = scr["kv_sem"]
            am, al, aacc = scr["am"], scr["al"], scr["aacc"]
            # KV stream: ring stage i % NKV holds work item i, block wl_t[i] of row wl_b[i] (listed
            # in the step prologue). Items depend only on pos: start before att_in.
            n_items = scr["st"][1]
            if c.paged_kv:
                pt_ref = ST["page_table"]
                kvbuf = scr["kvbuf"]
                P_BLK = BLK // c.page_size

                def kv_start(i):
                    @pl.when(jnp.logical_and(i < n_items, "kv" not in skip))
                    def _():
                        s = i % NKV
                        b, t = scr["wl_b"][i], scr["wl_t"][i]

                        def start_for_ref(r):
                            def issue_p(p, _):
                                kp = jnp.clip(pt_ref[b, t * P_BLK + p], 0, c.num_pages - 1)
                                dst = kvbuf.at[s, :, :, :, pl.ds(p * c.page_size, c.page_size)]
                                pltpu.make_async_copy(r.at[kp], dst, ksem.at[s]).start()
                                return 0

                            lax.fori_loop(0, P_BLK, issue_p, 0)

                        bswitch(a, [lambda r=r: start_for_ref(r) for r in raw_refs])

                def kv_wait(i):
                    if "kv" not in skip:
                        s = i % NKV

                        def wait_p(p, _):
                            pltpu.make_async_copy(
                                raw_refs[0].at[0],
                                kvbuf.at[s, :, :, :, pl.ds(p * c.page_size, c.page_size)],
                                ksem.at[s],
                            ).wait()
                            return 0

                        lax.fori_loop(0, P_BLK, wait_p, 0)

                lax.fori_loop(0, NKV, lambda i, _: (kv_start(i), 0)[1], 0)
            else:
                kc, vc = ST["kc"], ST["vc"]
                kbuf, vbuf = scr["kbuf"], scr["vbuf"]

                def kv_item(i, b, t):
                    s = i % NKV
                    t0 = pl.multiple_of(t * BLK, BLK)
                    return (
                        pltpu.make_async_copy(kc.at[a, b, pl.ds(t0, BLK)], kbuf.at[s],
                                              ksem.at[0, s]),
                        pltpu.make_async_copy(vc.at[a, b, pl.ds(t0, BLK)], vbuf.at[s],
                                              ksem.at[1, s]),
                    )

                def kv_start(i):
                    @pl.when(jnp.logical_and(i < n_items, "kv" not in skip))
                    def _():
                        for d in kv_item(i, scr["wl_b"][i], scr["wl_t"][i]):
                            d.start()

                def kv_wait(i):
                    if "kv" not in skip:
                        zero = jnp.int32(0)  # waits need only the stage's sems / size
                        for d in kv_item(i, zero, zero):
                            d.wait()

                for i in range(NKV):
                    kv_start(jnp.int32(i))

            with sc("att_in"):
                y = dmatmul("att_in", g0, scr["xb"], c.NAI, lazy=LAZY_ATT)
                ybuf = scr["ybuf"]
                ybuf[:, :c.NAI] = rnd(y * sm(slot, "in_s", c.NAI))
            qn = sm(slot, "qn")
            kn = sm(slot, "kn")
            half = c.ROT // 2
            inv = 1.0 / (c.theta ** (lax.broadcasted_iota(I32, (1, half), 1).astype(F32)
                                     * 2.0 / c.ROT))

            def rope(x, cos, sin):
                x1, x2, rest = x[:, :half], x[:, half:c.ROT], x[:, c.ROT:]
                return jnp.concatenate(
                    (rnd(rnd(x1 * cos) - rnd(x2 * sin)), rnd(rnd(x2 * cos) + rnd(x1 * sin)),
                     rest), axis=-1)

            koff = nh * 2 * HD
            with sc("att_qk"):
                for b in range(B):
                    pb = pos_ref[b]
                    ang = pb.astype(F32) * inv
                    cos, sin = rnd(jnp.cos(ang)), rnd(jnp.sin(ang))
                    qg = ybuf[b, :koff].reshape(nh, 2 * HD)
                    scr["gate"][b, :] = qg[:, HD:].reshape(nh * HD)
                    q = rms_zc(qg[:, :HD], qn)
                    k = rms_zc(ybuf[b:b + 1, koff:koff + HD], kn)
                    v = ybuf[b:b + 1, koff + HD:koff + 2 * HD]
                    q = rnd(rope(q, cos, sin) * HD ** -0.5)
                    k = rope(k, cos, sin)
                    if PACK:
                        t_, o_ = qrow(b)
                        scr["qg"][gi, t_, o_:o_ + nh] = q
                    else:
                        if nh < 8:
                            q = jnp.concatenate((q, jnp.zeros((8 - nh, HD), F32)), axis=0)
                        if SPLIT or VER:
                            scr["qg"][gi, b] = q
                        else:
                            scr["qb"][b] = q.astype(BF16)
                    if c.paged_kv:
                        k_fp8 = jnp.clip(k, -448.0, 448.0).astype(FP8)
                        v_fp8 = jnp.clip(v, -448.0, 448.0).astype(FP8)
                        scr["knew_o"][b:b + 1, :] = k_fp8
                        scr["vnew_o"][b:b + 1, :] = v_fp8
                        scr["knew"][b:b + 1, :] = k_fp8.astype(F32)
                        scr["vnew"][b:b + 1, :] = v_fp8.astype(F32)
                    else:
                        scr["knew"][b:b + 1, :] = k
                        scr["vnew"][b:b + 1, :] = v
                if not FUSED:
                    k_src = scr["knew_o"] if c.paged_kv else scr["knew"]
                    v_src = scr["vnew_o"] if c.paged_kv else scr["vnew"]
                    kv_out = (
                        pltpu.make_async_copy(k_src, O["knew"].at[a], scr["kv_out_sem"].at[0]),
                        pltpu.make_async_copy(v_src, O["vnew"].at[a], scr["kv_out_sem"].at[1]),
                    )
                    for d in kv_out:
                        d.start()
            if SPLIT or VER:
                with sc("att_qgather"):
                    qgb = scr["qg"]
                    if SPLIT:
                        exchange(group_copies(lambda off: qgb.at[gi], qgb.at[gi],
                                              scr["sq_ssem"], scr["sq_rsem"]))

                    def group_q(b):  # row b's G * nh group query heads, zero-padded to QR rows
                        if not SPLIT:
                            return qgb[0, b]
                        t_, o_ = qrow(b)
                        qa = jnp.concatenate([qgb[g, t_, o_:o_ + nh, :] for g in range(G)],
                                             axis=0)
                        if G * nh < QR:
                            qa = jnp.concatenate((qa, jnp.zeros((QR - G * nh, HD), F32)), 0)
                        return qa

                    for sq in range(NS):  # token t of KV row sq: query rows t * QR .. of qb[sq]
                        qs = [group_q(sq * T + t) for t in range(T)]
                        qs = qs[0] if T == 1 else jnp.concatenate(qs, axis=0)
                        scr["qb"][sq] = qs.astype(BF16)

            with sc("att_core"):
                AMASK = tiles.amask
                col = lax.broadcasted_iota(I32, (1 if AMASK else TQ, SUB), 1)
                col_v = (
                    lax.broadcasted_iota(I32, (HD, SUB), 1)
                    if (c.paged_kv and AMASK == 0)
                    else None
                )
                am[...] = jnp.full(am.shape, -1e30, F32)
                al[...] = jnp.zeros(al.shape, F32)
                aacc[...] = jnp.zeros(aacc.shape, F32)

                def item(i, carry):
                    """Online-softmax step of KV row (sequence) b = wl_b[i] over its KV block
                    t = wl_t[i], for the TQ query rows of its T tokens (cached positions only: the
                    new tokens are added in att_new)."""
                    s = i % NKV
                    kv_wait(i)
                    if "attn" not in skip:
                        b, t = scr["wl_b"][i], scr["wl_t"][i]
                        plen = pos_ref[b * T] if VER else pos_ref[b]
                        qrows = scr["qb"][b]
                        if c.paged_kv:
                            if interpret:
                                k_full = kvbuf[s, 0].reshape(HD, BLK)
                                v_full = kvbuf[s, 1].reshape(HD, BLK)
                            else:
                                k_u32 = kvbuf.at[s, 0].bitcast(U32).reshape(64, BLK)[...]
                                v_u32 = kvbuf.at[s, 1].bitcast(U32).reshape(64, BLK)[...]
                                k_full = pltpu.bitcast(k_u32, FP8)
                                if AMASK in (0, 3, 4):
                                    v_full = pltpu.bitcast(v_u32, FP8)
                        for j in range(BLK // SUB):
                            off = t * BLK + j * SUB
                            rem = plen - off
                            full_blk = off + SUB <= plen
                            if c.paged_kv:
                                kj = k_full if SUB == BLK else k_full[:, j * SUB:(j + 1) * SUB]
                                if AMASK == 0:
                                    vj = v_full if SUB == BLK else v_full[:, j * SUB:(j + 1) * SUB]
                                    vj = jnp.where(col_v + off < plen, vj, jnp.zeros_like(vj))
                                elif interpret:
                                    vj = v_full if SUB == BLK else v_full[:, j * SUB:(j + 1) * SUB]
                                    if AMASK not in (3, 4):
                                        vj = jnp.where(col < rem, vj, jnp.zeros_like(vj))
                                elif AMASK == 1:
                                    vu = v_u32 if SUB == BLK else v_u32[:, j * SUB:(j + 1) * SUB]
                                    vu = jnp.where(col < rem, vu, jnp.uint32(0))
                                    vj = pltpu.bitcast(vu, FP8)
                                elif AMASK == 2:
                                    vu = v_u32 if SUB == BLK else v_u32[:, j * SUB:(j + 1) * SUB]
                                    vu = lax.cond(
                                        full_blk,
                                        lambda: vu,
                                        lambda: jnp.where(col < rem, vu, jnp.uint32(0)),
                                    )
                                    vj = pltpu.bitcast(vu, FP8)
                                else:
                                    vj = v_full if SUB == BLK else v_full[:, j * SUB:(j + 1) * SUB]
                                scores = mm(qrows, kj, NN)
                            else:
                                kj = kbuf[s] if SUB == BLK else kbuf[s, j * SUB:(j + 1) * SUB]
                                vj = vbuf[s] if SUB == BLK else vbuf[s, j * SUB:(j + 1) * SUB]
                                scores = mm(qrows, kj, NT)
                            if AMASK == 0:
                                scores = jnp.where(col + off < plen, scores, -jnp.inf)
                            elif AMASK in (1, 4):
                                scores = jnp.where(col < rem, scores, -jnp.inf)
                            else:
                                scores = lax.cond(
                                    full_blk,
                                    lambda: scores,
                                    lambda: jnp.where(col < rem, scores, -jnp.inf),
                                )
                            m_old = am[b]
                            m_new = jnp.maximum(m_old, jnp.max(scores, axis=-1, keepdims=True))
                            alpha = jnp.exp(m_old - m_new)
                            pr = jnp.exp(scores - m_new[:, 0:1])
                            pv = mm(pr.astype(BF16), vj, NT if c.paged_kv else NN)
                            al[b] = al[b] * alpha + jnp.sum(pr, axis=-1, keepdims=True)
                            aacc[b] = aacc[b] * alpha[:, 0:1] + pv
                            am[b] = m_new
                    kv_start(i + NKV)
                    return carry

                lax.fori_loop(0, n_items, item, 0)

            def seq_same_page(s):
                if T == 1:
                    return jnp.bool_(True)
                b0 = s * T
                lp0 = lax.div(pos_ref[b0], c.page_size)
                ok = jnp.bool_(True)
                for t in range(1, T):
                    ok = jnp.logical_and(ok, lax.div(pos_ref[b0 + t], c.page_size) == lp0)
                return ok

            if FUSED:
                kv_scat = scr["kv_scat_page"]
                rd_sem, wr_sem = scr["kv_rd_sem"], scr["kv_wr_sem"]
                rd_pend, wr_pend = scr["kv_rd_pend"], scr["kv_wr_pend"]
                for s in range(NS):
                    wait_kv_wr(s)
                    act_s = is_seq_active(s)
                    same_s = seq_same_page(s)
                    do_rd = jnp.logical_and(act_s, same_s)
                    rd_pend[s] = jnp.where(do_rd, 1, 0).astype(I32)

                    @pl.when(do_rd)
                    def _(s=s):
                        lp0 = lax.div(pos_ref[s * T], c.page_size)
                        kp0 = jnp.clip(pt_ref[s, lp0], 0, c.num_pages - 1)

                        def issue_rd(r):
                            pltpu.make_async_copy(
                                r.at[kp0], kv_scat.at[s], rd_sem.at[s]
                            ).start()

                        bswitch(a, [lambda r=r: issue_rd(r) for r in raw_refs])

            if SPLIT:
                with sc("att_merge"):
                    # rows g * nh .. of a token's group query rows are group rank g's heads: send
                    # them (with m in lanes HD.., l in HD + 64..) to rank g, merge what arrives
                    pso, pri = scr["pso"], scr["pri"]
                    lane_pw = lax.broadcasted_iota(I32, (nh, 128), 1)
                    for b in range(B):
                        t_, o_ = qrow(b)
                        sq, q0 = b // T, (b % T) * QR  # row b's query rows in KV row sq
                        mb, lb = am[sq, q0:q0 + QR], al[sq, q0:q0 + QR]
                        ab = aacc[sq, q0:q0 + QR]
                        for g in range(G):
                            r0 = g * nh
                            pso[g, t_, o_:o_ + nh, 0:HD] = ab[r0:r0 + nh]
                            pso[g, t_, o_:o_ + nh, HD:PW] = jnp.where(
                                lane_pw < 64, mb[r0:r0 + nh], lb[r0:r0 + nh])
                    pri[gi] = pso[gi]
                    exchange(group_copies(lambda off: pso.at[gi ^ off], pri.at[gi],
                                          scr["sp_ssem"], scr["sp_rsem"]))
                    for t in range(NBP):  # PACK: one merge per tile of PB rows
                        parts = [pri[g, t] for g in range(G)]
                        mg = [x[:, HD:HD + 1] for x in parts]
                        m = mg[0]
                        for x in mg[1:]:
                            m = jnp.maximum(m, x)
                        wg = [jnp.exp(x - m) for x in mg]
                        lsum = sum(w * x[:, HD + 64:HD + 65] for w, x in zip(wg, parts))
                        asum = sum(w * x[:, :HD] for w, x in zip(wg, parts))
                        if not PACK:  # tile t is row t
                            sq, q0 = t // T, (t % T) * QR
                            am[sq, q0:q0 + 8] = jnp.broadcast_to(m, (8, 128))
                            al[sq, q0:q0 + 8] = jnp.broadcast_to(lsum, (8, 128))
                            aacc[sq, q0:q0 + 8] = asum
                            continue
                        for b in range(t * PB, min(B, (t + 1) * PB)):
                            o_ = qrow(b)[1]
                            sq, q0 = b // T, (b % T) * QR
                            am[sq, q0:q0 + nh] = jnp.broadcast_to(m[o_:o_ + nh], (nh, 128))
                            al[sq, q0:q0 + nh] = jnp.broadcast_to(lsum[o_:o_ + nh], (nh, 128))
                            aacc[sq, q0:q0 + nh] = asum[o_:o_ + nh]

            with sc("att_new"):
                NR = nh if PACK else 8  # softmax-state rows: rows 0..nh-1 are own heads
                for b in range(B):
                    # the new tokens' columns; rows q0 .. q0+NR-1 of KV row sq's softmax state are
                    # row b's (own heads in rows q0 .. q0+nh-1)
                    sq, tb = divmod(b, T)
                    q0 = tb * QR
                    m_old = am[sq, q0:q0 + NR][:, 0:1]
                    l_old = al[sq, q0:q0 + NR][:, 0:1]
                    a_old = aacc[sq, q0:q0 + NR]
                    if PACK:
                        t_, o_ = qrow(b)
                        qf = scr["qg"][gi, t_, o_:o_ + nh]
                    else:
                        qf = scr["qg"][gi, b] if SPLIT or VER else scr["qb"][b].astype(F32)
                    if VER:  # token tb sees new tokens 0..tb of its sequence
                        for j in range(tb + 1):
                            bj = sq * T + j
                            s_new = jnp.sum(qf * scr["knew"][bj:bj + 1, :], axis=-1,
                                            keepdims=True)
                            m_new = jnp.maximum(m_old, s_new)
                            alpha = jnp.exp(m_old - m_new)
                            p_new = jnp.exp(s_new - m_new)
                            l_old = l_old * alpha + p_new
                            a_old = a_old * alpha + p_new * scr["vnew"][bj:bj + 1, :]
                            m_old = m_new
                        outb = a_old / l_old
                    else:
                        s_new = jnp.sum(qf * scr["knew"][b:b + 1, :], axis=-1, keepdims=True)
                        m_new = jnp.maximum(m_old, s_new)
                        alpha = jnp.exp(m_old - m_new)
                        p_new = jnp.exp(s_new - m_new)
                        l_tot = l_old * alpha + p_new
                        outb = (a_old * alpha + p_new * scr["vnew"][b:b + 1, :]) / l_tot
                    for h in range(nh):
                        oh = rnd(outb[h:h + 1, :])
                        gh = rnd(jax.nn.sigmoid(scr["gate"][b:b + 1, h * HD:(h + 1) * HD]))
                        scr["mix"][b:b + 1, h * HD:(h + 1) * HD] = rnd(oh * gh)

            with sc("att_out"):
                if FUSED:
                    lane_iota_kv = lax.broadcasted_iota(I32, (64, c.page_size), 1)
                    k_cols = batch_to_i32_cols(scr["knew_o"][...])
                    v_cols = batch_to_i32_cols(scr["vnew_o"][...])
                    for s in range(NS):
                        act_s = is_seq_active(s)
                        same_s = seq_same_page(s)
                        do_fast = rd_pend[s] != 0
                        rd_pend[s] = jnp.int32(0)

                        @pl.when(do_fast)
                        def _(s=s):
                            pltpu.make_async_copy(
                                raw_refs[0].at[0], kv_scat.at[s], rd_sem.at[s]
                            ).wait()
                            b0 = s * T
                            if typed:
                                k_cur = pltpu.bitcast(kv_scat[s, 0], I32).reshape(64, c.page_size)
                                v_cur = pltpu.bitcast(kv_scat[s, 1], I32).reshape(64, c.page_size)
                            else:
                                k_view = kv_scat.at[s, 0].bitcast(I32).reshape(64, c.page_size)
                                v_view = kv_scat.at[s, 1].bitcast(I32).reshape(64, c.page_size)
                                k_cur = k_view[...]
                                v_cur = v_view[...]
                            for t in range(T):
                                mask_b = lane_iota_kv == (b0 + t)
                                mask_ln = lane_iota_kv == lax.rem(pos_ref[b0 + t], c.page_size)
                                k_col = jnp.sum(
                                    jnp.where(mask_b, k_cols, jnp.int32(0)), axis=1, keepdims=True
                                )
                                v_col = jnp.sum(
                                    jnp.where(mask_b, v_cols, jnp.int32(0)), axis=1, keepdims=True
                                )
                                k_cur = jnp.where(mask_ln, k_col, k_cur)
                                v_cur = jnp.where(mask_ln, v_col, v_cur)
                            if typed:
                                kv_scat[s, 0] = pltpu.bitcast(
                                    k_cur.reshape(64, 1, c.page_size), FP8
                                )
                                kv_scat[s, 1] = pltpu.bitcast(
                                    v_cur.reshape(64, 1, c.page_size), FP8
                                )
                            else:
                                k_view[...] = k_cur
                                v_view[...] = v_cur
                            lp0 = lax.div(pos_ref[b0], c.page_size)
                            kp0 = jnp.clip(pt_ref[s, lp0], 0, c.num_pages - 1)

                            def issue_wr(r):
                                pltpu.make_async_copy(
                                    kv_scat.at[s], r.at[kp0], wr_sem.at[s]
                                ).start()

                            bswitch(a, [lambda r=r: issue_wr(r) for r in raw_refs])

                        wr_pend[s] = jnp.where(do_fast, 1, 0).astype(I32)
                        if T > 1:
                            @pl.when(jnp.logical_and(act_s, jnp.logical_not(same_s)))
                            def _(s=s):
                                b0 = s * T
                                for t in range(T):
                                    p_t = pos_ref[b0 + t]
                                    lp_t = lax.div(p_t, c.page_size)
                                    ln_t = lax.rem(p_t, c.page_size)
                                    kp_t = jnp.clip(pt_ref[s, lp_t], 0, c.num_pages - 1)

                                    def rd_one(r, kp_t=kp_t):
                                        pltpu.make_async_copy(
                                            r.at[kp_t], kv_scat.at[s], rd_sem.at[s]
                                        ).start()

                                    bswitch(a, [lambda r=r: rd_one(r) for r in raw_refs])
                                    pltpu.make_async_copy(
                                        raw_refs[0].at[0], kv_scat.at[s], rd_sem.at[s]
                                    ).wait()
                                    mask_b = lane_iota_kv == (b0 + t)
                                    mask_ln = lane_iota_kv == ln_t
                                    k_col = jnp.sum(
                                        jnp.where(mask_b, k_cols, jnp.int32(0)),
                                        axis=1,
                                        keepdims=True,
                                    )
                                    v_col = jnp.sum(
                                        jnp.where(mask_b, v_cols, jnp.int32(0)),
                                        axis=1,
                                        keepdims=True,
                                    )
                                    if typed:
                                        k_cur = pltpu.bitcast(kv_scat[s, 0], I32).reshape(
                                            64, c.page_size
                                        )
                                        v_cur = pltpu.bitcast(kv_scat[s, 1], I32).reshape(
                                            64, c.page_size
                                        )
                                        k_cur = jnp.where(mask_ln, k_col, k_cur)
                                        v_cur = jnp.where(mask_ln, v_col, v_cur)
                                        kv_scat[s, 0] = pltpu.bitcast(
                                            k_cur.reshape(64, 1, c.page_size), FP8
                                        )
                                        kv_scat[s, 1] = pltpu.bitcast(
                                            v_cur.reshape(64, 1, c.page_size), FP8
                                        )
                                    else:
                                        k_view = (
                                            kv_scat.at[s, 0].bitcast(I32).reshape(64, c.page_size)
                                        )
                                        v_view = (
                                            kv_scat.at[s, 1].bitcast(I32).reshape(64, c.page_size)
                                        )
                                        k_view[...] = jnp.where(mask_ln, k_col, k_view[...])
                                        v_view[...] = jnp.where(mask_ln, v_col, v_view[...])

                                    def wr_one(r, kp_t=kp_t):
                                        pltpu.make_async_copy(
                                            kv_scat.at[s], r.at[kp_t], wr_sem.at[s]
                                        ).start()

                                    bswitch(a, [lambda r=r: wr_one(r) for r in raw_refs])
                                    pltpu.make_async_copy(
                                        kv_scat.at[s], raw_refs[0].at[0], wr_sem.at[s]
                                    ).wait()
                else:
                    for d in kv_out:
                        d.wait()
                NO = nh * HD
                scr["xb2"][:, :NO] = scr["mix"][:, :NO].astype(BF16)
                yo = dmatmul("att_out", g0 + nt["att_in"], scr["xb2"], H, lazy=LAZY_ATT)
            if LAZY_ATT:
                with sc("refill"):
                    flush(g0, nt["att_in"] + nt["att_out"])
            return yo * sm(slot, "out_s")

        # ------------------------------ MoE ------------------------------
        def moe(l, slot, g0, is_att):
            seg_r = g0 + jnp.where(is_att, nt["att_in"] + nt["att_out"],
                                   nt["lin_in"] + nt["lin_out"])
            with sc("moe_router"):
                x = rms_zc(scr["res"][...], sm(slot, "post_norm"))
                scr["xb"][...] = x.astype(BF16)
                logits = dmatmul("router", seg_r, scr["xb"], E, lazy=LAZY_MOE)[0:MR]
                glog = jnp.sum(x * sm(slot, "sh_gate"), axis=-1, keepdims=True)
                scr["sg"][...] = jnp.broadcast_to(jax.nn.sigmoid(glog), (MB, 128))
                mx = jnp.max(logits, axis=-1, keepdims=True)
                pexp = jnp.exp(logits - mx)
            with sc("moe_route"):
                n_hit = route(l, pexp)
            with sc("moe_shared"):
                # shared expert (TP slice of the intermediate), under the first expert tiles
                g13 = dmatmul("sh13", seg_r + nt["router"], scr["xb"], 2 * c.ISR, lazy=LAZY_MOE)
                g13 = g13 * sm(slot, "sh13_s")
                mid = rnd(rnd(jax.nn.silu(rnd(g13[:, :c.ISR]))) * rnd(g13[:, c.ISR:]))
                scr["xb2"][:, :c.ISR] = mid.astype(BF16)
                ysh = dmatmul("sh2", seg_r + nt["router"] + nt["sh13"], scr["xb2"], H,
                              lazy=LAZY_MOE)
                scr["part"][...] = ysh * sm(slot, "sh2_s") * scr["sg"][:, 0:1]
            with sc("moe_experts"):
                if PIPE:
                    experts_piped(l, n_hit)
                else:
                    lax.fori_loop(0, n_hit, functools.partial(expert, l), 0)
            if LAZY_MOE:
                with sc("refill"):
                    flush(seg_r, N_MOE)
            return scr["part"][...]

        def route(l, p8):
            """Top-k of rows 0..MR-1 (live rows 0..B-1; p8 = their unnormalized probabilities) ->
            renormalized weights scr["wn"]; the deduplicated hit experts (EP: this rank's) ->
            scr["elist"][:n_hit] in pick order, and their first tiles start. Returns n_hit."""
            choice = p8
            if forced_routing:
                rs = l % 2
                pltpu.make_async_copy(rb_hbm.at[0], scr["rb"].at[rs], scr["rb_sem"].at[rs]).wait()

                @pl.when(l + 1 < L)
                def _():
                    pltpu.make_async_copy(rb_hbm.at[l + 1], scr["rb"].at[1 - rs],
                                          scr["rb_sem"].at[1 - rs]).start()

                choice = scr["rb"][rs, 0:MR, :]
            with sc("route_topk", 2):
                ids = jnp.zeros((MR, 128), I32)
                if tiles.topk:
                    # one max-reduction per pick: the key keeps choice's high bits and holds
                    # EM - lane in the low log2(E) mantissa bits, so keys are distinct and the
                    # max names its lane. Ties (and values equal in all but those bits, 2^-14
                    # relative for E=512) go to the lowest lane. choice >= 0; the FLT_MIN clamp
                    # keeps keys normal floats (no denormal flushing).
                    kb = lax.bitcast_convert_type(jnp.maximum(choice, FLT_MIN), I32)
                    key = lax.bitcast_convert_type((kb & ~EM) | (EM - lane_e8), F32)
                    for k in range(c.TOPK):
                        kmax = jnp.max(key, axis=-1, keepdims=True)
                        am_ = EM - (lax.bitcast_convert_type(kmax, I32) & EM)
                        ids = jnp.where(lane128 == k, am_, ids)
                        key = jnp.where(key == kmax, -1.0, key)
                    sel = key < 0.0
                else:
                    sel = jnp.zeros((MR, E), jnp.bool_)
                    for k in range(c.TOPK):
                        cmax = jnp.max(choice, axis=-1, keepdims=True)
                        am_ = jnp.min(jnp.where(choice == cmax, lane_e8, E), axis=-1,
                                      keepdims=True)
                        ids = jnp.where(lane128 == k, am_, ids)
                        hit = lane_e8 == am_
                        sel = jnp.logical_or(sel, hit)
                        choice = jnp.where(hit, -jnp.inf, choice)
                scr["hitv"][...] = ids
                wsel = jnp.where(sel, p8, 0.0)
                scr["wn"][...] = jnp.where(row8 < B, wsel / jnp.sum(wsel, axis=-1, keepdims=True),
                                           0.0)
            with sc("route_dma", 2):
                cp = pltpu.make_async_copy(scr["hitv"], scr["hits"], scr["misc_sem"])
                cp.start()
                cp.wait()
            stamp = l + 1
            slot = l % 2
            if ECOPY:
                for r in range(R):
                    scr["rload"][r] = jnp.int32(0)

                if tiles.ecopy_opt == 1:
                    def pick_uniq(b, k, n_tot):
                        e = scr["hits"][b, k]
                        hi, lo = lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
                        new = scr["seen"][hi, lo] != stamp
                        scr["seen"][hi, lo] = stamp
                        scr["ulist"][n_tot] = e
                        return n_tot + jnp.where(new, 1, 0).astype(I32)

                    def classify_uniq(i, carry_c):
                        n, n_cp = carry_c
                        e = scr["ulist"][i]
                        code = scr["ecopy"][slot, 0, e].astype(I32)
                        has_cp = code > 0
                        r0 = lax.div(e, ER)
                        s0 = e - r0 * ER
                        scr["rload"][r0] = scr["rload"][r0] + jnp.int32(1)
                        mine_fixed = jnp.logical_and(jnp.logical_not(has_cp), r0 == me)
                        scr["elist"][n] = e
                        scr["wlist"][n] = s0
                        scr["cplist"][n_cp] = e
                        return (
                            n + jnp.where(mine_fixed, 1, 0).astype(I32),
                            n_cp + jnp.where(has_cp, 1, 0).astype(I32),
                        )

                    n_tot = jnp.int32(0)
                    with sc("route_dedup", 2):
                        if WIDE:
                            def row_uniq(b, nt_in):
                                return lax.fori_loop(
                                    0, c.TOPK, functools.partial(pick_uniq, b), nt_in, unroll=True
                                )

                            n_tot = lax.fori_loop(0, B, row_uniq, n_tot, unroll=2)
                        else:
                            for b in range(B):
                                n_tot = lax.fori_loop(
                                    0, c.TOPK, functools.partial(pick_uniq, b), n_tot, unroll=True
                                )
                        n_hit, n_cp = lax.fori_loop(
                            0, n_tot, classify_uniq, (jnp.int32(0), jnp.int32(0))
                        )
                else:
                    def pick_ecopy(b, k, carry):
                        n, n_cp, n_tot = carry
                        e = scr["hits"][b, k]
                        hi, lo = lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
                        new = scr["seen"][hi, lo] != stamp
                        scr["seen"][hi, lo] = stamp
                        code = scr["ecopy"][slot, 0, e].astype(I32)
                        has_cp = code > 0
                        r0 = lax.div(e, ER)
                        s0 = e - r0 * ER
                        fixed_new = jnp.logical_and(new, jnp.logical_not(has_cp))
                        cp_new = jnp.logical_and(new, has_cp)
                        scr["rload"][r0] = scr["rload"][r0] + jnp.where(new, 1, 0).astype(I32)
                        mine_fixed = jnp.logical_and(fixed_new, r0 == me)
                        scr["elist"][n] = e
                        scr["wlist"][n] = s0
                        scr["cplist"][n_cp] = e
                        return (
                            n + jnp.where(mine_fixed, 1, 0).astype(I32),
                            n_cp + jnp.where(cp_new, 1, 0).astype(I32),
                            n_tot + jnp.where(new, 1, 0).astype(I32),
                        )

                    carry = (jnp.int32(0), jnp.int32(0), jnp.int32(0))
                    with sc("route_dedup", 2):
                        if WIDE:
                            def row_ecopy(b, c_in):
                                return lax.fori_loop(0, c.TOPK, functools.partial(pick_ecopy, b),
                                                     c_in, unroll=True)

                            n_hit, n_cp, n_tot = lax.fori_loop(0, B, row_ecopy, carry, unroll=2)
                        else:
                            for b in range(B):
                                carry = lax.fori_loop(0, c.TOPK, functools.partial(pick_ecopy, b),
                                                      carry, unroll=True)
                            n_hit, n_cp, n_tot = carry

                with sc("route_dedup", 2):
                    def po2c_pass1(j, _):
                        e = scr["cplist"][j]
                        r0 = lax.div(e, ER)
                        c0 = scr["ecopy"][slot, 0, e].astype(I32) - 1
                        r1 = lax.div(c0, EXTRA)
                        l0 = scr["rload"][r0]
                        l1 = scr["rload"][r1]
                        use_c = jnp.logical_or(
                            l1 + 1 < l0,
                            jnp.logical_and(
                                l1 < l0, jnp.logical_or(l0 * R > n_tot, l1 * R < n_tot)
                            ),
                        )
                        d = jnp.where(use_c, 1, 0).astype(I32)
                        scr["rload"][r0] = l0 - d
                        scr["rload"][r1] = l1 + d
                        scr["cpchoice"][j] = d
                        return 0

                    def po2c_pass2(j, n):
                        e = scr["cplist"][j]
                        r0 = lax.div(e, ER)
                        s0 = e - r0 * ER
                        c0 = scr["ecopy"][slot, 0, e].astype(I32) - 1
                        r1 = lax.div(c0, EXTRA)
                        s1 = ER + (c0 - r1 * EXTRA)
                        use_c = scr["cpchoice"][j] != 0
                        cur = jnp.where(use_c, r1, r0)
                        alt = jnp.where(use_c, r0, r1)
                        l_cur = scr["rload"][cur]
                        l_alt = scr["rload"][alt]
                        flip = jnp.logical_or(
                            l_alt + 1 < l_cur,
                            jnp.logical_and(
                                l_alt < l_cur,
                                jnp.logical_and(l_cur * R > n_tot, l_alt * R < n_tot),
                            ),
                        )
                        d = jnp.where(flip, 1, 0).astype(I32)
                        scr["rload"][cur] = l_cur - d
                        scr["rload"][alt] = l_alt + d
                        final_c = jnp.not_equal(use_c, flip)
                        pick_r = jnp.where(final_c, r1, r0)
                        pick_w = jnp.where(final_c, s1, s0)
                        mine = pick_r == me
                        scr["elist"][n] = e
                        scr["wlist"][n] = pick_w
                        return n + jnp.where(mine, 1, 0).astype(I32)

                    lax.fori_loop(0, n_cp, po2c_pass1, 0)
                    n_hit = lax.fori_loop(0, n_cp, po2c_pass2, n_hit)

            else:
                def pick(b, k, n):
                    # branch-free: a repeated (or, for EP, remote) pick rewrites slot n and stamps
                    # again without advancing n
                    e = scr["hits"][b, k]
                    hi, lo = lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
                    seen = scr["seen"][hi, lo]
                    new = seen != stamp
                    if not c.moe_tp:
                        new = jnp.logical_and(new, lax.div(e, ER) == me)
                    if row_mask:  # a masked row's pick neither counts nor stamps
                        on = mr_ref[b] != 0
                        new = jnp.logical_and(new, on)
                        scr["seen"][hi, lo] = jnp.where(on, stamp, seen)
                    else:
                        scr["seen"][hi, lo] = stamp
                    scr["elist"][n] = e
                    return n + jnp.where(new, 1, 0).astype(I32)

                n_hit = jnp.int32(0)
                with sc("route_dedup", 2):
                    if WIDE:  # up to 640 picks: a row loop (TOPK picks unrolled per row)
                        def row(b, n):
                            return lax.fori_loop(0, c.TOPK, functools.partial(pick, b), n,
                                                 unroll=True)

                        n_hit = lax.fori_loop(0, B, row, n_hit, unroll=2)
                    else:
                        for b in range(B):
                            n_hit = lax.fori_loop(0, c.TOPK, functools.partial(pick, b), n_hit,
                                                  unroll=True)
            if "experts" in skip:
                n_hit = jnp.int32(0)
            scr["st"][0] = n_hit * TE
            with sc("route_fetch", 2):
                if c.stack_e:
                    for k in range(NBE):
                        @pl.when(k < n_hit * TE)
                        def _(k=k):
                            fetch_e(k, l)
                else:
                    lax.fori_loop(0, jnp.minimum(NBE, n_hit * TE),
                                  lambda k, _: (fetch_e(k, l), 0)[1], 0)

            return n_hit

        def expert(l, i, carry):
            """Listed expert i on all rows: part += wn[:, e] * expert_e(xb). EP: the whole expert;
            TP: this rank's slice of the intermediate dim (the all-reduce sums the slices)."""
            e = scr["elist"][i]
            s = i % NSS
            base = i * TE
            acc, xb, xb2, part = scr["acc"], scr["xb"], scr["xb2"], scr["part"]
            if T13 == 1:
                h = mm(xb[...], etake("e13", base)[...])
                egive(base, l)
            else:
                acc[:, :2 * IR] = jnp.zeros((MB, 2 * IR), F32)

                def b13(t, cc):
                    u = base + t
                    w = etake("e13", u)
                    k0 = pl.multiple_of(t * tk["e13"], tk["e13"])
                    acc[:, :2 * IR] += mm(xb[:, pl.ds(k0, tk["e13"])], w[...])
                    egive(u, l)
                    return cc

                lax.fori_loop(0, T13, b13, 0)
                h = acc[:, :2 * IR]
            es_wait(i)
            h = h * scr["es"][s, :, :2 * IR]
            mid_ = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:]))
            xb2[:, :IR] = mid_.astype(BF16)
            wcol = jnp.sum(jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True)
            u2 = base + T13
            if T2 == 1:
                # one [IR, H] tile: CH-lane chunks keep each result in a few vregs
                w2 = etake("e2", u2)[...]
                xm = xb2[:, :IR]
                for j in range(H // CH):
                    lo = 2 * IR + j * CH
                    y = mm(xm, w2[:, j * CH:(j + 1) * CH])[0:MR]
                    y = rnd(y * scr["es"][s, :, lo:lo + CH])
                    part[0:MR, j * CH:(j + 1) * CH] += wcol * y
                egive(u2, l)
            else:
                acc[:, :H] = jnp.zeros((MB, H), F32)

                def b2(t, cc):
                    u = u2 + t
                    w = etake("e2", u)
                    k0 = pl.multiple_of(t * tk["e2"], tk["e2"])
                    acc[:, :H] += mm(xb2[:, pl.ds(k0, tk["e2"])], w[...])
                    egive(u, l)
                    return cc

                lax.fori_loop(0, T2, b2, 0)
                y = rnd(acc[0:MR, :H] * scr["es"][s, :, 2 * IR:])
                part[0:MR, :] += wcol * y
            return carry

        def es_wait(i):
            s = i % NSS
            with sc("swait", 2):
                pltpu.make_async_copy(W["es"].at[0, 0], scr["es"].at[s],
                                      scr["es_sem"].at[s]).wait()

        def gate_up(i, w13):
            """Single-tile expert i's activated gate|up rows [MB, IR] (bf16)."""
            h = mm(scr["xb"][...], w13[...]) * scr["es"][i % NSS, :, :2 * IR]
            return rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)

        def down(i, mid, w2):
            """part[0:MR] += wn[:, e] * (mid @ e2) for single-tile expert i, in CH-lane chunks."""
            e, s, part = scr["elist"][i], i % NSS, scr["part"]
            wcol = jnp.sum(jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True)
            for j in range(H // CH):
                lo = 2 * IR + j * CH
                y = mm(mid, w2[:, j * CH:(j + 1) * CH])[0:MR]
                y = rnd(y * scr["es"][s, :, lo:lo + CH])
                part[0:MR, j * CH:(j + 1) * CH] += wcol * y

        def experts_piped(l, n_hit):
            """Single-tile experts, software-pipelined: iteration i runs expert i+1's gate|up
            next to expert i's down projection (independent MXU chains). Ring waits come first
            and refills last, so neither splits the compute."""
            xb2 = scr["xb2"]

            @pl.when(n_hit > 0)
            def _():
                w13 = etake("e13", 0)
                es_wait(0)
                xb2[:, :IR] = gate_up(0, w13)
                egive(0, l, 0)

                def body(i, carry):
                    u13, u2 = (i + 1) * TE, i * TE + T13
                    w13 = etake("e13", u13)
                    es_wait(i + 1)
                    w2 = etake("e2", u2)[...]
                    mid = xb2[:, :IR]
                    nxt = gate_up(i + 1, w13)
                    down(i, mid, w2)
                    xb2[:, :IR] = nxt
                    egive(u13, l, 0)
                    egive(u2, l, T13)
                    return carry

                lax.fori_loop(0, n_hit - 1, body, 0)
                last = n_hit - 1
                u2 = last * TE + T13
                down(last, xb2[:, :IR], etake("e2", u2)[...])
                egive(u2, l, T13)

        # ------------------------------ layer loop ------------------------------
        def layer(l, carry):
            slot = l % 2
            with sc("norm_in"):
                pltpu.make_async_copy(W["smalls"].at[0], scr["sm"].at[slot],
                                      scr["sm_sem"].at[slot]).wait()
                if ECOPY:
                    o_ec = lay["ecopy"][0]
                    pltpu.make_async_copy(W["smalls"].at[0, :, pl.ds(o_ec, E)],
                                          scr["ecopy"].at[slot], scr["ec_sem"].at[slot]).wait()

                @pl.when(l + 1 < L)
                def _():
                    pltpu.make_async_copy(W["smalls"].at[l + 1], scr["sm"].at[1 - slot],
                                          scr["sm_sem"].at[1 - slot]).start()
                    if ECOPY:
                        o_ec = lay["ecopy"][0]
                        pltpu.make_async_copy(W["smalls"].at[l + 1, :, pl.ds(o_ec, E)],
                                              scr["ecopy"].at[1 - slot],
                                              scr["ec_sem"].at[1 - slot]).start()

                is_att = l % AE == AE - 1
                g0 = (l // AE) * t_block + (l % AE) * t_gdn
                scr["xb"][...] = rms_zc(scr["res"][...], sm(slot, "in_norm")).astype(BF16)

            first = l == 0

            def att_branch():
                add_residual(allreduce(0, attention(l, slot, g0), first))

            def gdn_branch():
                add_residual(allreduce(0, gdn(l, slot, g0), first))

            if AE == 1:
                att_branch()
            else:
                lax.cond(is_att, att_branch, gdn_branch)
            add_residual(allreduce(1, moe(l, slot, g0, is_att), first))
            return carry

        # ------------------------------ step ------------------------------
        scr["hitv"][...] = jnp.zeros((MR, 128), I32)
        zero_seen = pltpu.make_async_copy(scr["hitv"].at[pl.ds(0, 8)], scr["seen"],
                                          scr["misc_sem"])
        zero_seen.start()
        for i in range(NBD):
            fetch(jnp.int32(i))
        pltpu.make_async_copy(W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]).start()
        if ECOPY:
            o_ec = lay["ecopy"][0]
            pltpu.make_async_copy(W["smalls"].at[0, :, pl.ds(o_ec, E)], scr["ecopy"].at[0],
                                  scr["ec_sem"].at[0]).start()
        if forced_routing:
            pltpu.make_async_copy(rb_hbm.at[0], scr["rb"].at[0], scr["rb_sem"].at[0]).start()
        # this rank's KV work items, row-major over the KV rows (T > 1: sequences, whose caches
        # hold pos[s * T] tokens). Split: KV row b's block t is streamed by group rank
        # (t + b) % G, which spreads the rows' partial last blocks over the group.
        n = jnp.int32(0)
        for b in range(NS):
            nb = jnp.minimum(jnp.maximum((pos_ref[b * T] + BLK - 1) // BLK, 0), max_nb)
            t0 = lax.rem(gi + (TS - b % TS), TS) if SPLIT else jnp.int32(0)
            cnt = jnp.maximum(nb - t0 + TS - 1, 0) // TS

            def put(j, carry, b=b, t0=t0, n=n):
                scr["wl_b"][n + j] = jnp.int32(b)
                scr["wl_t"][n + j] = t0 + j * TS
                return carry

            lax.fori_loop(0, cnt, put, 0)
            n = n + cnt
        scr["st"][1] = n
        if R > 1:
            barrier = pltpu.get_barrier_semaphore()
            for off in range(1, R):
                pl.semaphore_signal(barrier, 1, device_id=(me ^ off,),
                                    device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(barrier, R - 1)
        scr["res"][...] = jnp.where(live, hid_ref[:MB].astype(F32), 0.0)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        if c.paged_kv:
            scr["knew_o"][...] = jnp.zeros((MR, HD), FP8)
            scr["vnew_o"][...] = jnp.zeros((MR, HD), FP8)
        if FUSED:
            for s in range(NS):
                scr["kv_rd_pend"][s] = jnp.int32(0)
                scr["kv_wr_pend"][s] = jnp.int32(0)
            if c.NGDN > 0:
                for slot_i in range(NGI):
                    scr["gdn_in_pend"][slot_i] = jnp.int32(0)
                for slot_o in range(NGO):
                    scr["gdn_out_mode"][slot_o] = jnp.int32(0)
                scr["gdn_out_page"][...] = jnp.zeros(scr["gdn_out_page"].shape, FP8)
        if SPLIT:
            scr["pso"][...] = jnp.zeros(scr["pso"].shape, F32)
        if PACK:  # a partial last tile's spare rows are sent (never read): keep them defined
            scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)
        zero_seen.wait()
        lax.fori_loop(0, L, layer, 0)
        if comm:
            for ar_slot in range(2):
                for rnd_i in AR_ROUNDS:
                    for d in ar_copies(ar_slot, rnd_i):
                        d.wait_send()
        if FUSED:
            if c.NGDN > 0:
                for slot_o in range(NGO):
                    wait_gdn_out(slot_o)
            for s in range(NS):
                wait_kv_wr(s)
        elif c.NGDN > 0:
            o_last = c.NGDN - 1
            pltpu.make_async_copy(scr["convb"], O["conv"].at[o_last], scr["st_sem"].at[2]).wait()
            if GCH:
                rec_drain(o_last)
            else:
                pltpu.make_async_copy(scr["recb"], O["rec"].at[o_last],
                                      scr["st_sem"].at[3]).wait()
        cp = pltpu.make_async_copy(W["final_norm"], scr["fn"], scr["misc_sem"])
        cp.start()
        cp.wait()
        out_ref[:MB] = rms_zc(scr["res"][...], scr["fn"][...]).astype(BF16)
        if MB < M:
            out_ref[MB:] = jnp.zeros((M - MB, H), BF16)

    oshapes = kernel_out_shapes(c)
    w_specs = [
        tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(c.L))
        if (not c.stack_e and k in ("e13", "e2"))
        else pl.BlockSpec(memory_space=pltpu.HBM)
        for k in W_NAMES
    ]
    s_specs = [
        tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(c.NATT))
        if k == "raw"
        else pl.BlockSpec(memory_space=pltpu.SMEM)
        if k in ("page_table", "ckpt_indices", "read_offsets", "active_seqs")
        else pl.BlockSpec(memory_space=pltpu.HBM)
        for k in st_names
    ]
    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.VMEM)]
        + w_specs
        + s_specs
        + ([pl.BlockSpec(memory_space=pltpu.HBM)] if forced_routing else [])
        + ([pl.BlockSpec(memory_space=pltpu.SMEM)] if row_mask else [])
    )
    n_w_leaves = sum(c.L if (not c.stack_e and k in ("e13", "e2")) else 1 for k in W_NAMES)
    out_shape = [
        tuple(jax.ShapeDtypeStruct(*s) for s in oshapes[k])
        if k == "raw"
        else jax.ShapeDtypeStruct(*oshapes[k])
        for k in o_names
    ]
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM)
        if k == "hidden"
        else tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(c.NATT))
        if k == "raw"
        else pl.BlockSpec(memory_space=pltpu.HBM)
        for k in o_names
    ]
    if FUSED:
        aliases = (
            {2 + n_w_leaves + a: 1 + a for a in range(c.NATT)}
            if bool(interpret)
            else {}
        )
    else:
        aliases = (
            {2 + n_w_leaves + st_names.index(k): o_names.index(k) for k in ("conv", "rec")}
            if tiles.inplace
            else {}
        )
    call = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        # GDN states update in place: layer o's slice is read (and waited) before it is written
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID if R > 1 else None,
            vmem_limit_bytes=tiles.vmem_mib * 1024 * 1024,
            disable_bounds_checks=not tiles.bounds_checks,
        ),
        interpret=(interpret if isinstance(interpret, pltpu.InterpretParams)
                   else pltpu.InterpretParams() if interpret else False),
        name="qwen3_5_decode_megakernel",
    )
    vmem = 0
    for k in s_names:
        v = scratch[k]
        if hasattr(v, "shape") and "VMEM" in str(getattr(v, "memory_space", "")).upper():
            vmem += math.prod(v.shape) * jnp.dtype(v.dtype).itemsize
    info = {"plan": p, "T_TOTAL": T_TOTAL, "vmem_scratch_bytes": vmem, "s_names": s_names,
            "wshape": weight_shapes(c, tiles), "NSS": NSS, "TE": TE}
    return call, info


def fold(x):
    """[R, N0, ...] -> [R * N0, ...]: the rank-folded layout make_step takes for states."""
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def unfold(x, R):
    return x.reshape((R, x.shape[0] // R) + x.shape[1:])


def make_step(mesh, c: MKConfig, tiles: Tiles | None = None, *, interpret=False,
              forced_routing=False, skip=frozenset(), scopes=False):
    """shard_map decode step: (pos [B], hidden [R*M, H] (rank-replicated rows), weights, states,
    [rbias [R*L, M, E]]) -> (hidden_out [R*M, H], states). Weights carry a leading rank axis
    [R, ...]; states are rank-folded [R * N0, ...] (see `fold`) so that each shard is exactly the
    kernel operand and the in-place state update needs no copy. Both are sharded over "tp".
    T > 1 (verify): the returned conv / rec are the per-token outputs of kernel_out_shapes (feed
    accept_states' result back), and every row's new K/V lands in its sequence's cache at pos."""
    call, info = build(c, tiles, interpret=interpret, forced_routing=forced_routing, skip=skip,
                       scopes=scopes)

    def local(pos, hidden, weights, states, *rb):
        w = [tuple(x[0] for x in weights[k]) if isinstance(weights[k], tuple) else weights[k][0]
             for k in W_NAMES]
        s = [states[k] for k in S_NAMES]
        hid, conv, rec, knew, vnew = call(pos, hidden, *w, *s, *rb)
        kc, vc = states["kc"], states["vc"]
        if c.NS > 16:  # one scatter per cache (2 * NS dynamic-update-slices cost ~1.6 us each)
            bi = jnp.arange(c.B) if c.T == 1 else jnp.arange(c.B) // c.T
            kw = dict(indices_are_sorted=True, unique_indices=True, mode="promise_in_bounds")
            kc = kc.at[:, bi, pos].set(knew[:, :c.B].astype(BF16), **kw)
            vc = vc.at[:, bi, pos].set(vnew[:, :c.B].astype(BF16), **kw)
            return hid, {"conv": conv, "rec": rec, "kc": kc, "vc": vc}
        for s in range(c.NS):
            b0 = s * c.T
            upd = (0, s, pos[b0], 0)
            kc = lax.dynamic_update_slice(kc, knew[:, None, b0:b0 + c.T].astype(BF16), upd)
            vc = lax.dynamic_update_slice(vc, vnew[:, None, b0:b0 + c.T].astype(BF16), upd)
        return hid, {"conv": conv, "rec": rec, "kc": kc, "vc": vc}

    def step(pos, hidden, weights, states, *rb):
        ws = {k: tuple(P("tp") for _ in weights[k]) if isinstance(weights[k], tuple) else P("tp")
              for k in weights}
        ss = {k: P("tp") for k in states}
        in_specs = (P(), P("tp"), ws, ss) + ((P("tp"),) if forced_routing else ())
        return jax.shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=(P("tp"), ss),
                             check_vma=False)(pos, hidden, weights, states, *rb)

    return step, info
