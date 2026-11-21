"""Geometry of the GLM-5.2 decode megakernel for TP x DCP (x EP) deployments.

One DCP group spans all R ranks (TP = DCP = EP = R). Every rank sees every one of the step's rows; row
t's residual is owned by rank t % R (RPR rows per rank, TMAX = R * RPR rows per step). The KV of a
request is position-sharded: owner(p) = (p % (ILV * R)) // ILV, local index (p // (ILV * R)) * ILV + p % ILV,
held at (block_table[req, local // page], local % page) on the owner
(kernel_work/glm52-mk-tp16dcp16/incumbent/main-dcp-semantics.md §A).
`FULL` is GLM-5.2-FP8 at TP16 x DCP16; `TINY` keeps every structural constraint at an interpretable size."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class MKDConfig:
    R: int                 # ranks = TP = DCP = EP
    L: int                 # decoder layers
    first_dense: int       # layers [0, first_dense) use the dense MLP
    H: int                 # hidden
    NH: int                # attention heads (NH % R == 0)
    QL: int                # q_lora
    KL: int = 512          # kv_lora (512 B nope cache token)
    DN: int = 192          # qk_nope
    DR: int = 64           # rope
    DV: int = 256          # v head
    E: int = 256           # routed experts (E % R == 0)
    TOPK: int = 8
    I: int = 2048          # moe intermediate
    IS: int = 2048         # shared expert intermediate (TP-sliced: IS / R per rank)
    ID: int = 12288        # dense intermediate (TP-sliced: ID / R per rank)
    IH: int = 32           # indexer heads
    IDIM: int = 128        # indexer head dim
    KT: int = 2048         # indexer top-k (global and local)
    idx_freq: int = 4
    idx_skip: int = 3
    page: int = 256        # tokens per local page
    ILV: int = 4           # cp_kv_cache_interleave_size
    RPR: int = 1           # rows per rank (TMAX = R * RPR)
    QMAX: int = 4          # rows per request (1 + MTP drafts)
    rope_theta: float = 8e6
    eps: float = 1e-5
    routed_scale: float = 2.5
    QB: int = 512          # expert lhs quant block (gmm_v2 quant_block_size)
    KREP: int = 0          # decode replica slots per rank per MoE layer (0: off, program unchanged)
    V: int = 0             # vocab (> 0: the MTP drafter build; tied to the target's TP vocab shards)

    @property
    def TMAX(self):
        return self.R * self.RPR

    @property
    def HR(self):          # heads per rank
        return self.NH // self.R

    @property
    def ER(self):          # experts per rank
        return self.E // self.R

    @property
    def ISR(self):         # shared intermediate per rank
        return self.IS // self.R

    @property
    def IDR(self):         # dense intermediate per rank
        return self.ID // self.R

    @property
    def DQ(self):          # q head width
        return self.DN + self.DR

    @property
    def KR(self):          # o_proj local K
        return self.HR * self.DV

    @property
    def QKVA(self):
        return self.QL + self.KL + self.DR

    @property
    def IREC(self):        # indexer record bytes (IDIM fp8 + 1 ue8m0 byte, padded to 4)
        return (self.IDIM + 1 + 3) // 4 * 4

    def full_indexer(self, l):
        """Layers that compute a fresh top-k (0, 1, 2, 6, 10, ... for GLM-5.2)."""
        return l < self.idx_skip or (l - self.idx_skip) % self.idx_freq == self.idx_freq - 1

    def is_moe(self, l):
        return l >= self.first_dense

    # --- DCP position map (main-dcp-semantics.md §A) ---
    def owner(self, p):
        return (p % (self.ILV * self.R)) // self.ILV

    def local(self, p):
        return (p // (self.ILV * self.R)) * self.ILV + p % self.ILV

    def n_local(self, L, r):
        """Local tokens rank r holds for context length L (cp_local_length)."""
        g = self.ILV * self.R
        return self.ILV * (L // g) + min(max(L % g - self.ILV * r, 0), self.ILV)


FULL = MKDConfig(R=16, L=78, first_dense=3, H=6144, NH=64, QL=2048)

TINY = MKDConfig(R=4, L=5, first_dense=1, H=512, NH=8, QL=256, DN=64, DV=128,
                 E=8, TOPK=2, I=256, IS=512, ID=512, IH=4, KT=256, idx_freq=2,
                 idx_skip=1, page=128, RPR=2, QB=128)

# MTP drafter (WHITEBOARD.v4): 3 passes of the one MTP layer as 3 "layers" (fresh top-k on pass 0 only)
FULL_DRAFT = dataclasses.replace(FULL, L=3, first_dense=0, idx_skip=1, idx_freq=64, V=154880)
TINY_DRAFT = dataclasses.replace(TINY, L=3, first_dense=0, idx_skip=1, idx_freq=64, V=800)
