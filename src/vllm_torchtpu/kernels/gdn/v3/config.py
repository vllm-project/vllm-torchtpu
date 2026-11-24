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

import dataclasses
import enum
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs


# Row granule of a 2D bf16 qkv tile (see `GDNConfig.qkv_2d`): one packed bf16
# sublane tile, so every DMA offset and size is aligned to it.
QKV_2D_ROW_ALIGN = 16
# Output rows a zero-fill DMA moves (GDNConfig.zero_fill_rows).
ZERO_FILL_ROWS = 32


class AttentionMode(enum.StrEnum):
    # Gated Delta Net: one scalar gate per value head.
    GDN = enum.auto()
    # Kimi Delta Attention: the gate is per-channel, so it carries
    # `kq_head_dim` values per value head instead of one.
    KDA = enum.auto()


class CheckpointMode(enum.StrEnum):
    # Today's rollback: one full state checkpoint per verify window position,
    # each in its own source block; a step resumes from checkpoint[read_offset].
    SELECT = enum.auto()
    # Checkpoint block 0 holds the state after window position 0, and its
    # spare rows hold the draft log: the operands (gate, key, update vector
    # and conv input) of window positions 1..K. A step resumes from
    # checkpoint 0 with the first read_offset logged updates applied, which
    # gives the state checkpoint[read_offset] holds in SELECT mode.
    LOG = enum.auto()


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
    # CheckpointMode.LOG only (the wrapper runs that mode exactly when `log`
    # is set): the draft log in checkpoint block 0's spare rows. `log` holds
    # the gates, keys and update vectors of the K draft positions as d_v-lane
    # f32 rows, `log_conv` their conv inputs in the conv region's view. Row
    # layout: `log_layout_rows`.
    log: StateRegion | None = None
    log_conv: StateRegion | None = None


def log_layout_rows(num_log_positions: int, num_kq_heads: int, num_v_heads: int):
    """Typed-row offsets of the gate, key and update rows in a `log` region.

    Draft position j + 1 (j < num_log_positions) has its gate row at row
    `j` (lane h: the decay of value head h after exp), key head h at row
    `keys + j * num_kq_heads + h` and the update vector of value head h at
    row `updates + j * num_v_heads + h`.

    Returns:
        (keys, updates, rows_used) typed-row offsets and total.
    """
    keys = num_log_positions
    updates = keys + num_log_positions * num_kq_heads
    rows_used = updates + num_log_positions * num_v_heads
    return keys, updates, rows_used


def validate_log_plan(
    plan: StateSourcePlan,
    *,
    log_positions: int,
    num_kq_heads: int,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    dim_size: int,
    block_rows: int,
    raw_lanes: int,
    raw_row_bytes: int,
) -> None:
    """Checks that a plan's draft log regions match the LOG mode's layout.

    Args:
        plan: The state-source plan carrying `log` and `log_conv`.
        log_positions: Draft positions logged (num_speculative_tokens).
        num_kq_heads, num_v_heads, kq_head_dim, v_head_dim, dim_size: The
            kernel geometry (dim_size is the conv channel count).
        block_rows: Rows of one source kernel block.
        raw_lanes: Minormost dim of the source.
        raw_row_bytes: Bytes of one source row.
    """
    log, log_conv, conv = plan.log, plan.log_conv, plan.conv
    if log is None or log_conv is None:
        raise ValueError("LOG mode needs both state_plan.log and state_plan.log_conv")
    if log_positions < 1:
        raise ValueError(f"LOG mode needs speculative positions, got {log_positions}")
    if plan.whole_block_dma:
        raise NotImplementedError("LOG mode does not support whole-block DMA plans")
    if kq_head_dim != v_head_dim:
        raise NotImplementedError(
            f"LOG rows hold keys and updates alike: d_k={kq_head_dim} != d_v={v_head_dim}"
        )
    if num_v_heads > v_head_dim:
        raise NotImplementedError(
            f"a gate row holds one lane per value head: {num_v_heads} > {v_head_dim}"
        )
    if jnp.dtype(log.view_dtype) != jnp.dtype(jnp.float32):
        raise ValueError(f"log rows are float32, got {log.view_dtype}")
    if raw_lanes // log.lane_split != v_head_dim or log.rows_perm is not None:
        raise ValueError(
            f"log rows must be {v_head_dim} lanes wide: lanes={raw_lanes}, "
            f"lane_split={log.lane_split}"
        )
    _, _, rows_used = log_layout_rows(log_positions, num_kq_heads, num_v_heads)
    if log.rows_used != rows_used:
        raise ValueError(f"log.rows_used={log.rows_used}, expected {rows_used}")
    if (
        jnp.dtype(log_conv.view_dtype) != jnp.dtype(conv.view_dtype)
        or log_conv.lane_split != conv.lane_split
        or log_conv.rows_perm is not None
        or conv.rows_perm is not None
    ):
        raise ValueError("log_conv must share the conv region's plain view")
    conv_lanes = raw_lanes // log_conv.lane_split
    if dim_size % conv_lanes or log_conv.rows_used != (
        log_positions * dim_size // conv_lanes
    ):
        raise ValueError(
            f"log_conv.rows_used={log_conv.rows_used}, expected "
            f"{log_positions} * {dim_size} / {conv_lanes}"
        )
    for region, lanes in ((log, raw_lanes // log.lane_split), (log_conv, conv_lanes)):
        typed_bytes = region.rows_used * lanes * jnp.dtype(region.view_dtype).itemsize
        if region.nblocks != 1 or typed_bytes > region.nrows * raw_row_bytes:
            raise ValueError(f"log region does not fit its rows: {region}")
    if log_conv.kb0 != log.kb0 or log_conv.row0 != log.row0 + log.nrows:
        raise ValueError("log_conv must follow log in the same kernel block")
    if log_conv.row0 + log_conv.nrows > block_rows:
        raise ValueError("log regions overrun the kernel block")
    lo, hi = log.row0, log_conv.row0 + log_conv.nrows
    for other in (plan.recurrent, conv):
        for kb in range(other.kb0, other.kb0 + other.nblocks):
            if kb == log.kb0 and other.row0 < hi and lo < other.row0 + other.nrows:
                raise ValueError(f"log rows [{lo}, {hi}) overlap {other}")


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
    # CheckpointMode.LOG: the state lives in checkpoint block 0 plus the draft
    # log of `log_positions` (= num_speculative_tokens) positions, in every
    # kernel mode; see `CheckpointMode` and `StateSourcePlan.log`.
    checkpoint_mode: CheckpointMode = CheckpointMode.SELECT
    log_positions: int = 0
    # LOG mode, BATCHED: the window runs in factored form
    # (`compute_gdn.factored_gdn`): one product of the window's keys and
    # queries with the loaded base state, the logged and window updates as
    # vector work, and one update pass for the state stored to checkpoint 0.
    # Left out of the kernel metadata while False.
    factored_window: bool = False
    # Token-dense conv path of a PER_SEQ chunked tile over 2D bf16 rows
    # (VLLM_TPU_GDN_PREFILL_DENSE_CONV); resolved by the wrapper. Left out of
    # the kernel metadata while False, so other kernels' metadata is unchanged.
    dense_conv: bool = False
    # BATCHED: the kernel itself writes zeros to output rows
    # [zero_from, batch_size) (zero_from from SMEM), the rows no sequence of
    # the call owns, instead of reading a zero-filled output buffer. Left out
    # of the kernel metadata while False.
    zero_fill_rows: bool = False
    # b and a stream from one f32 [tokens, 1, aligned_num_v_heads +
    # aligned_gate_dim] array (b first). Left out of the kernel metadata while
    # False.
    ba_packed: bool = False

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
        if envs.VLLM_TPU_GDN_DIRECT_SOLVE:
            # Direct RHS solve (`solve_triangular_rhs`) updates the whole [B, n]
            # block per step; below 8 sublanes it burns the same VALU work while
            # issuing more matmuls and padding to 8 sublanes in VMEM.
            return 16 if head_groups <= 1 else 8
        # GDN: between-block updates scale as 1 / B. B must divide the chunk
        # (a power of two): 3 head groups (24 value heads, e.g. Qwen3.8 at
        # TP2) would otherwise get B = 5, which no chunk splits into.
        block = max(2, min(16, 16 // head_groups))
        return 1 << (block.bit_length() - 1)

    @property
    def qkv_2d(self) -> bool:
        """Whether qkv streams as 2D bf16 [tokens, dim] rows.

        Only the PER_SEQ tile takes them: each sequence tile DMAs the
        `QKV_2D_ROW_ALIGN`-aligned rows around its chunk and the kernel casts
        its rows to f32 in VMEM, instead of reading an f32 copy made in HBM.
        Derived from the input dtype rather than stored, so f32 configs keep
        their exact kernel metadata.
        """
        return self.dtypes.act_in == jnp.bfloat16

    @property
    def qkv_2d_aligned_chunk(self) -> int:
        """Chunk rows rounded up to whole 2D bf16 row tiles."""
        return pl.cdiv(self.chunk_size, QKV_2D_ROW_ALIGN) * QKV_2D_ROW_ALIGN

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

    @property
    def is_window_log(self) -> bool:
        return self.checkpoint_mode == CheckpointMode.LOG

    @property
    def writes_window_log(self) -> bool:
        """Whether the kernel writes the draft log: the BATCHED kernel in LOG
        mode. Every LOG-mode kernel reads it."""
        return self.is_window_log and self.mode == GDNMode.BATCHED

    @property
    def uses_factored_window(self) -> bool:
        """Whether this kernel runs the factored window body (BATCHED, LOG)."""
        return self.factored_window and self.writes_window_log

    @property
    def state_out_positions(self) -> int:
        """State checkpoints a sequence writes: one per window position, or
        checkpoint 0 alone in LOG mode (positions 1..K go to the log)."""
        return 1 if self.is_window_log else self.window_size

    def get_kernel_name(self) -> str:
        # Windows of different sizes compile to different kernels; keep them
        # distinguishable in profiles.
        suffix = f"_w{self.window_size}" if self.window_size > 1 else ""
        if self.is_kda and self.gate_lower_bound is not None:
            suffix += "_bounded"
        if self.is_window_log:
            suffix += "_log"
        if self.uses_factored_window:
            suffix += "_fact"
        if (
            envs.VLLM_TPU_GDN_DIRECT_SOLVE
            and not self.is_kda
            and not self.use_recurrent
        ):
            suffix += "_ds"
        return (
            f"fused_conv1d_{self.attention_mode.value}_{self.mode.value}"
            f"_b{self.seq_tile_size}_c{self.chunk_size}{suffix}"
        )

    def get_metadata(self) -> dict[str, str | int | float]:
        cfgs_dict = dataclasses.asdict(self)
        if not cfgs_dict.get("dense_conv"):
            cfgs_dict.pop("dense_conv", None)
        # Defaults stay out of the metadata, so other kernels' metadata is
        # unchanged.
        if not self.is_window_log:
            cfgs_dict.pop("checkpoint_mode", None)
            cfgs_dict.pop("log_positions", None)
        if not self.factored_window:
            cfgs_dict.pop("factored_window", None)
        if not self.zero_fill_rows:
            cfgs_dict.pop("zero_fill_rows", None)
        if not self.ba_packed:
            cfgs_dict.pop("ba_packed", None)
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

        scratches = dict(
            carry_conv_scratch_ref=carry_conv_scratch,
            carry_recurrent_scratch_ref=carry_recurrent_scratch,
        )
        if self.dense_conv:
            # Token-dense f32 rows of the tile: rows [8 - prev_kernel_size, 8)
            # hold the carried conv rows, rows [8, 8 + chunk) the chunk, so
            # every tap is a sublane shift of one 8-row-aligned load.
            scratches["act_scratch_ref"] = pltpu.VMEM(
                (8 + self.qkv_2d_aligned_chunk, self.dim_size), jnp.float32
            )
        elif self.qkv_2d:
            # f32 rows of the chunk, cast from the 2D bf16 tile; also where
            # the post-conv activations are staged for the head loads.
            scratches["act_scratch_ref"] = pltpu.VMEM(
                (self.seq_tile_size, self.qkv_2d_aligned_chunk, 1, self.dim_size),
                jnp.float32,
            )
        if self.zero_fill_rows:
            scratches["zero_scratch_ref"] = pltpu.VMEM(
                (ZERO_FILL_ROWS, self.num_v_heads, self.v_head_dim), self.dtypes.act_out
            )
            scratches["zero_sem_ref"] = pltpu.SemaphoreType.DMA(())
        return scratches
