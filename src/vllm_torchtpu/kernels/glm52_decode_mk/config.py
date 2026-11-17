# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Geometry of the GLM-5.2 whole-backbone decode megakernel (stage A, WHITEBOARD.v1 of campaign glm52-decode-mk).

Deployment: TP = DCP = R ranks on one mesh axis, qrep 0 (each rank computes its HR = NH / R query heads and all-gathers
them), TP16 MoE with EP off (every rank streams the hit experts' 1/R intermediate slice, moe_tp_decode numerics),
replicated residual, interleave-ILV DCP KV ownership. One step = rows of up to TMAX // QMAX requests, each of
2..QMAX verify rows (MTP K = QMAX - 1).

`MKConfig` carries the field names the batched DSA core (dsa.py, ported from glm5_mk_dcp's G1 spike) reads
(R, TMAX, KT, page, KL, IREC, IH, IDIM, NH, HR, QMAX, ILV, DR)."""

import dataclasses


def rup(x, a=128):
    return (x + a - 1) // a * a


@dataclasses.dataclass(frozen=True)
class MKConfig:
    R: int  # ranks = TP = DCP
    L: int  # decoder layers
    first_dense: int  # layers [0, first_dense) use the dense MLP
    H: int  # hidden
    NH: int  # attention heads (global)
    QL: int  # q_lora
    KL: int = 512  # kv_lora (512 B nope cache token)
    DN: int = 192  # qk_nope
    DR: int = 64  # rope
    DV: int = 256  # v head
    E: int = 256  # routed experts
    TOPK: int = 8
    I: int = 2048  # moe intermediate (global, per expert)
    IS: int = 2048  # shared expert intermediate (global)
    ID: int = 12288  # dense intermediate (global)
    IH: int = 32  # indexer heads
    IDIM: int = 128  # indexer head dim
    KT: int = 2048  # indexer top-k
    idx_freq: int = 4
    idx_skip: int = 3
    page: int = 256  # tokens per local page
    ILV: int = 4  # cp_kv_cache_interleave_size
    TMAX: int = 32  # row bucket of the step (16 or 32)
    QMAX: int = 4  # rows per request (1 + MTP drafts)
    BM: int = 8  # request slots (--max-num-seqs 8)
    eps: float = 1e-5
    idx_eps: float = 1e-6
    routed_scale: float = 2.5
    QB: int = 512  # moe_tp_decode / gmm_v2 activation quant block
    # Rope pair layout of the served weights and caches. False: GPT-J interleaved pairs (2i, 2i+1), the checkpoint's
    # layout. True: half-split (i, i + DR/2) -- glm-agentx's TPU_MLA_ROPE_NEOX_PERMUTE / TPU_INDEXER_ROPE_NEOX_PERMUTE
    # permute the rope output columns of q_b / fused_qkv_a / indexer wq_b / wk (and k_norm) at load, so q, k_pe and
    # the indexer q / k arrive (and the caches hold k_pe / k) in that order. Same per-element rope arithmetic on the
    # same operand pairs (bit-identical rope values, permuted); only the partner lane differs (kernel.rope).
    rope_neox: bool = False
    # DCP size of the KV cache (0 = R: TP = DCP, the incumbent). D = R / 2 is V16 (stageB/v16/DESIGN-ADDENDUM.md):
    # the cache is DCP-D (shard k = r % D held by the VR = R / D replicas k, k + D, ...); the replicas split shard k by
    # page parity (request b's local page j belongs to replica (j + b) % VR), so every rank does a DCP-R share of the
    # work and every collective stays R-way.
    DCP: int = 0

    # ---- per rank ----
    @property
    def D(self):  # KV-cache DCP size
        return self.DCP or self.R

    @property
    def VR(self):  # replicas per DCP shard (1: incumbent; 2: V16)
        return self.R // self.D

    @property
    def HR(self):
        return self.NH // self.R

    @property
    def IR(self):  # routed expert intermediate slice per rank (TP16 MoE)
        return self.I // self.R

    @property
    def ISR(self):
        return self.IS // self.R

    @property
    def IDR(self):
        return self.ID // self.R

    @property
    def DQ(self):
        return self.DN + self.DR

    @property
    def KR(self):  # o_proj local K
        return self.HR * self.DV

    @property
    def QKVA(self):
        return self.QL + self.KL + self.DR

    @property
    def QKVP(self):  # padded N of fused_qkv_a (rule 5: 2624 -> 2688)
        return rup(self.QKVA)

    @property
    def IKW(self):  # padded N of the indexer wk | weights_proj (rule 5: 160 -> 256)
        return rup(self.IDIM + self.IH)

    @property
    def IREC(self):  # indexer SAL record bytes (IDIM fp8 + 1 ue8m0 byte, padded to 4)
        return (self.IDIM + 1 + 3) // 4 * 4

    @property
    def EPAD(self):
        return rup(self.E)

    @property
    def M(self):  # activation rows
        return self.TMAX

    @property
    def BMAX(self):  # request slots
        return self.BM

    @property
    def sm_scale(self):
        return self.DQ**-0.5

    def full_indexer(self, l):
        """Fresh top-k layers: vllm deepseek_v2.py:1103-1106 max(l - skip + 1, 0) % freq == 0."""
        return max(l - self.idx_skip + 1, 0) % self.idx_freq == 0

    def is_moe(self, l):
        return l >= self.first_dense

    def owner(self, p):
        """DCP shard (= the rank at VR == 1) of global position p."""
        return (p % (self.ILV * self.D)) // self.ILV

    def local(self, p):
        return (p // (self.ILV * self.D)) * self.ILV + p % self.ILV

    def check(self):
        """Admission asserts for the couplings the kernel hard-codes (WHITEBOARD.v1 §18 / task: ILV % 4 == 0,
        block <= 256, QMAX == 4). Raises AssertionError on an unsupported configuration."""
        assert self.ILV % 4 == 0, (
            "rope staging packs 4-token rows: ILV % 4 == 0",
            self.ILV,
        )
        assert self.page == 256, (
            "the batched DSA core's (8, 128) key tile is 2 x 128 positions of a 256 page",
            self.page,
        )
        assert self.page % self.ILV == 0
        assert self.QMAX == 4, (
            "the DSA core's 4-row key tile is built for MTP K = 3 (dsa.make asserts QMAX == 4); "
            "other K decline here (adapter: Unsupported)",
            self.QMAX,
        )
        assert self.TMAX % 16 == 0 and self.TMAX <= 64, self.TMAX
        assert (
            self.NH % self.R == 0
            and self.I % self.R == 0
            and self.IS % self.R == 0
            and self.ID % self.R == 0
        )
        assert self.H % self.QB == 0 and self.H % 512 == 0 and self.H % 128 == 0, self.H
        assert self.KL == 512 and self.DR == 64 and self.IDIM == 128, (
            "production cache byte layouts"
        )
        assert self.DQ >= 128 and self.DN % 64 == 0, self.DQ
        assert self.KT % 128 == 0 and self.IH * 8 <= 512 and self.NH <= 128
        assert self.IR % 128 == 0 and self.ISR % 128 == 0 and self.IDR % 128 == 0, (
            self.IR,
            self.ISR,
            self.IDR,
        )
        assert (self.HR * (self.KL + 128)) % 256 == 0 and (
            self.HR * self.KL
        ) % 256 == 0, self.HR
        assert self.E >= self.TOPK and 1 <= self.BM <= self.TMAX
        assert self.D >= 1 and self.R % self.D == 0 and self.VR in (1, 2), (
            "V16: DCP = R or R / 2",
            self.R,
            self.DCP,
        )
        return self


FULL = MKConfig(R=16, L=78, first_dense=3, H=6144, NH=64, QL=2048)
FULL_B16 = dataclasses.replace(FULL, TMAX=16)


def from_dims(dims, R, TMAX):
    """MKConfig from an instruments spec.ModelDims (kernel_work/glm52-decode-mk/instruments/spec.py)."""
    return MKConfig(
        R=R,
        L=dims.n_layers,
        first_dense=dims.first_k_dense,
        H=dims.hidden,
        NH=dims.n_heads,
        QL=dims.q_lora,
        KL=dims.kv_lora,
        DN=dims.qk_nope,
        DR=dims.qk_rope,
        DV=dims.v_head,
        E=dims.n_experts,
        TOPK=dims.moe_topk,
        I=dims.moe_inter,
        IS=dims.shared_inter,
        ID=dims.dense_inter,
        IH=dims.idx_heads,
        IDIM=dims.idx_head_dim,
        KT=dims.idx_topk,
        idx_freq=dims.index_topk_freq,
        idx_skip=dims.index_skip_topk_offset,
        page=dims.page_size,
        ILV=dims.interleave,
        TMAX=TMAX,
        eps=dims.rms_eps,
        idx_eps=dims.idx_ln_eps,
        routed_scale=dims.routed_scale,
    )
