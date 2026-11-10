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

"""Configuration dataclasses and hardware tiling constants for Gated DeltaNet v3."""

import dataclasses
import enum
import os
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs


class AttentionMode(enum.StrEnum):
    # Gated Delta Net: one scalar gate per value head.
    GDN = enum.auto()
    # Kimi Delta Attention: the gate is per-channel, so it carries
    # `kq_head_dim` values per value head instead of one.
    KDA = enum.auto()


class GDNMode(enum.StrEnum):
    # Multiple sequences per tile, one tile per sequence. Each sequence
    # carries `window_size` tokens: a single decoded token, or a speculative
    # verify window. See `GDNConfig.window_size`.
    BATCHED = enum.auto()
    PER_SEQ = enum.auto()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dtypes:
    act_in: jnp.dtype
    act_out: jnp.dtype
    compute: jnp.dtype
    recurrent_state: jnp.dtype
    conv_state: jnp.dtype


@dataclasses.dataclass(frozen=True)
class StateRegion:
    """Copy-plan for one state region of an indexed external state source.

    A slot's region spans ``nblocks`` consecutive source blocks starting
    ``kb0`` blocks into the slot's window, rows ``[row0, row0 + nrows)``
    of each — one contiguous DMA per slot. The raw bytes are accessed
    through a typed view (``view_dtype`` elements, ``lane_split``-way
    128-lane split); the leading ``rows_used`` typed rows carry the
    state, the rest is zero padding.
    """

    kb0: int
    nblocks: int
    row0: int
    nrows: int
    view_dtype: jnp.dtype
    lane_split: int
    rows_used: int
    # Optional static typed-row permutation between the pool's stored row
    # order and the kernel's logical row order: logical_rows[i] is stored
    # at typed row rows_perm[i].  None or the identity means the stored
    # order IS the logical order.  Used by the QK pair-blocked conv layout,
    # where full-width states interleave Q and K row-pairs per tap so that
    # one whole pool token holds one head-pair's Q and K rows.
    rows_perm: tuple[int, ...] | None = None
    pack: int = 1
    block_row0: int = 0

    def __post_init__(self):
        if self.pack not in (1, 2, 4, 8):
            raise ValueError(f"unsupported state pack factor: {self.pack}")
        if self.pack != 1 and self.rows_perm is not None:
            raise ValueError(
                "a packed region's rows_perm would permute packed rows, not state rows"
            )
        if self.block_row0 and self.nblocks != 1:
            raise ValueError("a co-located region spans one block")


@dataclasses.dataclass(frozen=True)
class StateSourcePlan:
    """States live in an indexed external source instead of dense per-slot
    tensors: each state index owns a window of ``stride`` consecutive
    source blocks holding both state regions.

    The plan describes one checkpoint's geometry. With speculative
    decoding every checkpoint is an independent source block named by
    ``MetadataRef.s_idx_to_ckpt_indices`` and reuses this same geometry,
    so nothing here varies with the verify window."""

    stride: int
    conv: StateRegion
    recurrent: StateRegion
    whole_block_dma: bool = False
    scale: StateRegion | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNConfig:
    mode: GDNMode
    dtypes: Dtypes
    batch_size: int
    dim_size: int
    kernel_size: int
    tile_size: int
    num_kq_heads: int
    num_v_heads: int
    kq_head_dim: int
    v_head_dim: int
    attention_mode: AttentionMode = AttentionMode.GDN
    # KDA only, and must be negative. Selects the bounded gate form over the
    # unbounded one; see `fused_conv1d_gdn` for the two and their validation.
    gate_lower_bound: float | None = None
    num_buffers: int = 2
    state_plan: StateSourcePlan | None = None
    # Max tokens per speculative verify window (= num_speculative_tokens + 1),
    # which is also the number of state checkpoints kept per sequence. The
    # kernel reads a sequence's initial state from
    # `state_indices[s] + read_offset[s]` and writes one checkpoint per window
    # position to `state_indices[s] + t`, which is how rejected draft tokens
    # are rolled back (by checkpoint selection). It is 1 without speculative
    # decoding, where only the state after the last real token is kept, so
    # shapes and loops can be sized off it unconditionally and the extra axis
    # / iterations fold away in the non-speculative paths.
    window_size: int = 1
    qkv_2d: bool = False
    # Pooled BATCHED GDN only: the recurrent checkpoints bypass the
    # pipeline's output buffering and are DMA'd to the source by the kernel
    # body itself, one checkpoint at a time as the token loop produces it,
    # from a two-slot staging tile (`ckpt_stage_ref`). See ckpt_manual_dma().
    manual_ckpt_dma: bool = False

    @property
    def chunk_size(self) -> int:
        if self.mode == GDNMode.PER_SEQ:
            return self.tile_size
        # One tile per sequence, holding its whole verify window. BATCHED is
        # the single-token case of that (window_size == 1).
        return self.window_size

    @property
    def seq_tile_size(self) -> int:
        if self.mode == GDNMode.PER_SEQ:
            return 1
        return self.tile_size

    @property
    def prev_kernel_size(self) -> int:
        return self.kernel_size - 1

    @property
    def is_kda(self) -> bool:
        return self.attention_mode == AttentionMode.KDA

    @property
    def gate_dim(self) -> int:
        """Width of the gate activation `a`, in elements per token."""
        if self.is_kda:
            return self.num_v_heads * self.kq_head_dim
        return self.num_v_heads

    @property
    def aligned_gate_dim(self) -> int:
        num_lanes = pltpu.get_tpu_info().num_lanes
        return pl.cdiv(self.gate_dim, num_lanes) * num_lanes

    @property
    def triangular_block_size(self) -> int:
        """Sub-block size of the chunked KDA intra-chunk solve.

        The diagonal sub-blocks are evaluated on the VPU/XLU with exact
        pairwise gate differences and the off-diagonal ones as MXU GEMMs,
        so this trades the two units off against each other: raising it
        moves work from MXU to XLU. More value heads already means more
        XLU work, hence the smaller blocks there.
        """
        # Within-block forward substitution issues head_groups * B / 2 broadcasts
        # per token. At 1 head group (<= 8 heads), B = 16 is optimal.
        # - GDN issues O(1 / B) between-block solver matmuls per token, so scale
        #   B inversely with head_groups (head_groups * B <= 16).
        # - KDA also tiles Aqk and L into O(1 / B^2) sub-block matmuls per token,
        #   so scale B^2 inversely with head_groups (head_groups * B**2 <= 16**2).
        head_groups = pl.cdiv(self.num_v_heads, 8)
        if self.is_kda:
            for b in (16, 8, 4):
                if head_groups * b * b <= 256:
                    return b
            return 2
        # GDN: between-block updates scale as 1 / B.
        override = prefill_tinv_block()
        if override > 0 and self.mode == GDNMode.PER_SEQ:
            return max(1, min(override, self.chunk_size))
        return max(2, min(16, 16 // head_groups))

    @property
    def use_recurrent(self) -> bool:
        """Whether GDN runs the token-recurrent scan instead of the chunked one.

        Keeping more than one state checkpoint mandates the recurrent scan,
        since the chunked path only ever produces the final state.
        """
        return self.chunk_size == 1 or self.window_size > 1

    @property
    def v_dim_size(self) -> int:
        return self.num_v_heads * self.v_head_dim

    @property
    def kq_dim_size(self) -> int:
        return self.num_kq_heads * self.kq_head_dim

    @property
    def v_per_kq_head(self) -> int:
        return self.num_v_heads // self.num_kq_heads

    @property
    def aligned_num_v_heads(self) -> int:
        tpu_info = pltpu.get_tpu_info()
        num_lanes = tpu_info.num_lanes
        return pl.cdiv(self.num_v_heads, num_lanes) * num_lanes

    def get_kernel_name(self) -> str:
        # Windows of different sizes compile to different kernels; keep them
        # distinguishable in profiles.
        suffix = f"_w{self.window_size}" if self.window_size > 1 else ""
        if self.is_kda and self.gate_lower_bound is not None:
            suffix += "_bounded"
        return (
            f"fused_conv1d_{self.attention_mode.value}_{self.mode.value}"
            f"_b{self.seq_tile_size}_c{self.chunk_size}{suffix}"
        )

    def get_metadata(self) -> dict[str, str | int | float]:
        cfgs_dict = dataclasses.asdict(self)
        ret = {}
        for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
            key = jax.tree_util.keystr(path, simple=True, separator=".")
            if not isinstance(val, str | int | float):
                val = str(val)
            ret[key] = val
        return ret

    def get_out_shape(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            (self.batch_size, self.num_v_heads, self.v_head_dim),
            self.dtypes.act_out,
        )

    # Fraction of VMEM the kernel may use. A multi-token window holds one
    # state checkpoint per position in VMEM, so for large-head models the
    # default 0.80 budget is not enough; those get a higher limit and the
    # wrapper sizes the tile against the same factor.
    DEFAULT_VMEM_FRACTION: ClassVar[float] = 0.80
    WINDOWED_VMEM_FRACTION: ClassVar[float] = 0.90

    @classmethod
    def get_vmem_limit_bytes(cls, window_size: int = 1) -> int:
        """Returns the VMEM limit for a kernel keeping window_size checkpoints."""
        fraction = (
            cls.WINDOWED_VMEM_FRACTION if window_size > 1 else cls.DEFAULT_VMEM_FRACTION
        )
        return int(fraction * pltpu.get_tpu_info().vmem_capacity_bytes)

    def get_scratch_shape_dict(self) -> dict[str, Any]:
        conv_shape = (self.seq_tile_size, self.prev_kernel_size, 1, self.dim_size)
        recurrent_shape = (
            self.seq_tile_size,
            self.num_v_heads,
            self.kq_head_dim,
            self.v_head_dim,
        )

        carry_conv_scratch = carry_recurrent_scratch = None
        # NOTE: In batched mode 1 seq = 1 tile, so inter-tile carry is not
        # needed.
        if self.mode == GDNMode.PER_SEQ:
            carry_conv_scratch = pltpu.VMEM(conv_shape, jnp.float32)
            carry_recurrent_scratch = pltpu.VMEM(recurrent_shape, jnp.float32)

        act_scratch = None
        dense_act_scratch = None
        if self.dense_conv:
            # Token-dense f32 rows of the unpacked bf16 qkv tile.
            # Rows [8 - prev_kernel_size, 8) hold the carried conv rows, rows
            # [8, 8 + chunk) the chunk, so every tap is a sublane shift of one
            # 8-row-aligned load.
            scratch_chunk = 8 + pl.cdiv(self.chunk_size, 16) * 16
            dense_act_scratch = pltpu.VMEM((scratch_chunk, self.dim_size), jnp.float32)
        elif self.dtypes.act_in == jnp.bfloat16.dtype:
            scratch_chunk = (
                pl.cdiv(self.chunk_size, 16) * 16 if self.qkv_2d else self.chunk_size
            )
            act_scratch = pltpu.VMEM(
                (self.seq_tile_size, scratch_chunk, 1, self.dim_size), jnp.float32
            )

        return dict(
            carry_conv_scratch_ref=carry_conv_scratch,
            carry_recurrent_scratch_ref=carry_recurrent_scratch,
            act_scratch_ref=act_scratch,
            dense_act_scratch_ref=dense_act_scratch,
        )

    @property
    def dense_conv(self) -> bool:
        """Whether this tile runs the token-dense conv path (PER_SEQ chunked
        GDN over 2D bf16 qkv rows, TPU_GDN_PREFILL_DENSE_CONV)."""
        return (
            prefill_dense_conv()
            and self.mode == GDNMode.PER_SEQ
            and self.qkv_2d
            and not self.use_recurrent
            and not self.is_kda
            and self.chunk_size % 8 == 0
        )


def bench_ckpt_writes(window_size: int) -> int:
    """Window positions whose checkpoints are written back: the last n.

    Benchmark attribution knob (TPU_GDN_BENCH_CKPT_WRITES); unset writes
    every position, which is the only correct setting for serving.
    """
    raw = os.environ.get("TPU_GDN_BENCH_CKPT_WRITES", "").strip()
    if not raw:
        return window_size
    return max(0, min(int(raw), window_size))


def bench_stub() -> str:
    """Part of the token chain replaced by a stand-in of wrong numerics.

    Benchmark attribution knob (TPU_GDN_BENCH_STUB): "recurrence" skips the
    state update and the dots, "dots" replaces the MXU dots with VPU
    products of the same shape. Unset runs the real kernel.
    """
    return os.environ.get("TPU_GDN_BENCH_STUB", "").strip().lower()


def ckpt_store_inline() -> bool:
    """Whether BATCHED pooled tiles encode each recurrent checkpoint as the
    token loop produces it (TPU_GDN_CKPT_STORE_INLINE), instead of stacking
    the window's f32 states and encoding them after the loop."""
    return os.environ.get("TPU_GDN_CKPT_STORE_INLINE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def ckpt_manual_dma() -> bool:
    """Whether pooled BATCHED GDN tiles issue the recurrent checkpoints'
    write-back DMAs from the kernel body (TPU_GDN_CKPT_MANUAL_DMA) rather than
    through the pipeline's end-of-step copy_out. Implies ckpt_store_inline.
    "1" starts each copy as the token loop encodes its checkpoint; "next"
    starts a step's copies at the following step, after that step's setup
    and before its token chain (see ckpt_manual_dma_deferred)."""
    return os.environ.get("TPU_GDN_CKPT_MANUAL_DMA", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
        "next",
    )


def ckpt_manual_dma_deferred() -> bool:
    """Whether the manual checkpoint DMAs of a step are started at the next
    step (TPU_GDN_CKPT_MANUAL_DMA=next) rather than inside the token loop."""
    return os.environ.get("TPU_GDN_CKPT_MANUAL_DMA", "0").strip().lower() == "next"


def head_group() -> int:
    """Value heads per token-chain group in pooled BATCHED GDN tiles
    (TPU_GDN_HEAD_GROUP). With a positive value the recurrence runs head
    groups outer and window tokens inner, so a group's state is decoded from
    the input tile once, stays live across the window, and each checkpoint
    is encoded straight into the output tile; the whole-tile f32 state copies
    disappear. 0 (the default) runs tokens outer over all heads at once."""
    raw = os.environ.get("TPU_GDN_HEAD_GROUP", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def kt_xpose() -> bool:
    """Whether the recurrence builds k^T's lane broadcast with a sublane
    broadcast and an XLU transpose (TPU_GDN_KT_XPOSE), shared by the value
    heads of one kq head, instead of the masked lane reduction and a
    per-head broadcast. Applies to the production token order and to the
    grouped one."""
    return os.environ.get("TPU_GDN_KT_XPOSE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def dec_ckpt_fast() -> bool:
    """DGDN L1 (TPU_GDN_DEC_CKPT_FAST): pooled batched decode tiles store the
    conv checkpoints as static row slices instead of the per-row select chain.
    Checkpoints below the real size equal the chain's (no select fires there);
    those at or past it are never copied back to HBM. Bitwise identical
    (decgdn6/decgdn7)."""
    return bool(envs.TPU_GDN_DEC_CKPT_FAST)


def dec_rowpick() -> bool:
    """DGDN L6 (TPU_GDN_DEC_ROWPICK): the fused-1dot chain reads k.S and q.S
    as rows 0 / 1 of the dot plus +0.0 instead of a masked sublane sum, which
    equals row + 0.0 exactly. Bitwise identical (decgdn6/decgdn7)."""
    return bool(envs.TPU_GDN_DEC_ROWPICK)


def dec_conv_perm_lane() -> bool:
    """DGDN P1 (TPU_GDN_DEC_CONV_PERM_LANE): apply a tap-uniform QK-pair conv
    row permutation on whole 128-lane tiles of the reshaped (taps, 1, dim)
    state instead of on single typed rows. Pure data movement."""
    return bool(envs.TPU_GDN_DEC_CONV_PERM_LANE)


def dec_conv_perm_lane_mutant() -> bool:
    """Test only: reverse P1's per-tap tile pattern (must change results)."""
    return bool(envs.TPU_GDN_DEC_CONV_PERM_LANE_MUTANT)


def trace_scopes() -> bool:
    """Whether the kernel body wraps its phases in named scopes
    (TPU_GDN_TRACE_SCOPES), which the TPU profiler records in TRACE_COMPUTE
    mode. Benchmark attribution knob."""
    return os.environ.get("TPU_GDN_TRACE_SCOPES", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def prefill_dot_precision() -> str:
    """Operand precision of the chunked GDN dots (TPU_GDN_PREFILL_DOT_PRECISION):
    "" for f32 operands at Mosaic's default contraction, "bf16" for bf16
    operands with f32 accumulation, "fp32" for Precision.HIGHEST."""
    mode = envs.TPU_GDN_PREFILL_DOT_PRECISION.lower()
    if mode not in ("", "bf16", "fp32"):
        raise ValueError(f"TPU_GDN_PREFILL_DOT_PRECISION={mode!r}: expected '', bf16 or fp32")
    return mode


def prefill_chunk() -> int:
    """PER_SEQ chunk override in tokens (TPU_GDN_PREFILL_CHUNK), 0 for the
    VMEM-fit pick."""
    return envs.TPU_GDN_PREFILL_CHUNK


def prefill_head_group() -> int:
    """Value heads per chunked-GDN head group (TPU_GDN_PREFILL_HEAD_GROUP),
    0 for all heads at once."""
    return envs.TPU_GDN_PREFILL_HEAD_GROUP


def prefill_tinv_block() -> int:
    """Triangular-solve sub-block override (TPU_GDN_PREFILL_TINV_BLOCK), 0
    for the head-count-derived default."""
    return envs.TPU_GDN_PREFILL_TINV_BLOCK


def prefill_dense_conv() -> bool:
    """Whether PER_SEQ tiles run the token-dense conv path
    (TPU_GDN_PREFILL_DENSE_CONV)."""
    return envs.TPU_GDN_PREFILL_DENSE_CONV


def prefill_vmem_fraction() -> float:
    """Scoped-VMEM fraction override of the PER_SEQ pallas_call
    (TPU_GDN_PREFILL_VMEM_FRACTION), 0 for the GDNConfig default."""
    return envs.TPU_GDN_PREFILL_VMEM_FRACTION


def prefill_tinv_mode() -> str:
    """Triangular-solve algorithm of the chunked GDN (TPU_GDN_PREFILL_TINV_MODE):
    "" for the blocked forward substitution, "doubling" for the MXU series."""
    mode = envs.TPU_GDN_PREFILL_TINV_MODE.lower()
    if mode not in ("", "doubling"):
        raise ValueError(f"TPU_GDN_PREFILL_TINV_MODE={mode!r}: expected '' or doubling")
    return mode


def prefill_shared_kkt() -> bool:
    """Whether chunked GDN shares k k^T across the value heads of a kq head
    (TPU_GDN_PREFILL_SHARED_KKT)."""
    return envs.TPU_GDN_PREFILL_SHARED_KKT


def prefill_lazy_conv_state() -> bool:
    """Whether PER_SEQ chunked tiles defer the conv-state row select to a
    partial last tile (TPU_GDN_PREFILL_LAZY_CONV_STATE)."""
    return envs.TPU_GDN_PREFILL_LAZY_CONV_STATE
