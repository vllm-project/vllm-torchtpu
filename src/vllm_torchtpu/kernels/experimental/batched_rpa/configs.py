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
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import utils
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def get_tpu_info() -> pltpu.TpuInfo:
    """The local chip's TPU info; TPU7x's when JAX runs without a TPU backend
    (CPU tests and tracing), so the configs can still be computed. A TPU that
    JAX does not recognize raises rather than take TPU7x's SMEM and VMEM."""
    try:
        return pltpu.get_tpu_info()
    except ValueError:
        if jax.default_backend() == "tpu":
            raise
        logger.warning_once(
            "Batched RPA: no TPU backend (JAX backend %s); sizing tiles and "
            "schedules for TPU7x.",
            jax.default_backend(),
        )
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tuning parameters for the RPA kernel.

    Attributes:
        bq_sz: Query block size.
        bq_c_sz: Query chunk block size for prefill.
        bkv_sz: Key/Value block size.
        batch_size: Batch size.
    """

    bq_sz: int
    bq_c_sz: int
    bkv_sz: int
    batch_size: int


@dataclasses.dataclass(frozen=True)
class ModelConfigs:
    """Model config that will always stay constant.

    Attributes:
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads.
        head_dim: Dimensionality of each attention head.
        mask_value: Attention mask value for masked positions.
        sm_scale: Softmax scaling factor.
        soft_cap: Soft-capping threshold for attention logits.
        sliding_window: Size of the sliding attention window.
    """

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    mask_value: float
    sm_scale: float = 1.0
    soft_cap: float | None = None
    sliding_window: int | None = None

    @property
    def num_q_heads_per_kv_head(self) -> int:
        return self.num_q_heads // self.num_kv_heads


class AttentionScope(enum.StrEnum):
    """Which KV positions to attend to.

    FULL:            attend all positions (default).
    CACHE_ONLY:      attend only cached tokens, skip new tokens.
    NEW_TOKENS_ONLY: attend only new tokens, skip cached tokens.
    """

    FULL = enum.auto()
    CACHE_ONLY = enum.auto()
    NEW_TOKENS_ONLY = enum.auto()


class KVLayout(enum.StrEnum):
    """Represents the different layouts for KV cache.

    - HEAD_ALONG_SUBLANE: Number of heads on sublane, head_dim on lane.
    - SEQ_ALONG_LANE: Sequence is packed along the lane, head_dim on sublane.
    """

    HEAD_ALONG_SUBLANE = enum.auto()
    SEQ_ALONG_LANE = enum.auto()

    @property
    def symbol(self) -> str:
        match self:
            case KVLayout.HEAD_ALONG_SUBLANE:
                return "snh"
            case KVLayout.SEQ_ALONG_LANE:
                return "nhs"


class RpaCase(enum.StrEnum):
    """Represents the different cases for Ragged Paged Attention.

    - DECODE: Sequences are in decode-only mode (q_len = 1).
    - PREFILL: Sequences are in prefill-only mode (q_len > 1, static). Note that
        Batched RPA folds PREFILL into MIXED, but the enum value is kept because
        distribution still has three entries and RPA v3 / MLA use it.
    - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
    """

    DECODE = enum.auto()
    PREFILL = enum.auto()
    MIXED = enum.auto()

    @property
    def symbol(self) -> str:
        match self:
            case RpaCase.DECODE:
                return "d"
            case RpaCase.PREFILL:
                return "p"
            case RpaCase.MIXED:
                return "m"

    def get_range(
        self, distribution: jax.Array
    ) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
        assert distribution.shape == (3,)
        match self:
            case RpaCase.DECODE:
                return 0, distribution[0]
            case RpaCase.PREFILL:
                return distribution[0], distribution[1]
            case RpaCase.MIXED:
                return distribution[1], distribution[2]


class Fp8Operand(enum.StrEnum):
    """How a native fp8 matmul brings one operand into the fp8 KV dtype.

    - OFF: the matmul runs on dequantized K / V, as without native fp8.
    - ROW: one dynamic scale per row of the operand.
    - SPLIT: ROW, with the rounding residual quantized as a second fp8 half
      stacked under the first along the matmul rows: about bf16 accuracy for
      twice the LHS rows. DECODE only.
    - STATIC: a fixed scale: q is cast as is (after q_norm and RoPE it sits
      well inside the e4m3 range), the softmax weights are cast times
      `flash_attention.P_SCALE`, which the division by l removes.
    """

    OFF = enum.auto()
    ROW = enum.auto()
    SPLIT = enum.auto()
    STATIC = enum.auto()


_FP8_OPERAND_SUFFIX = {
    Fp8Operand.ROW: "",
    Fp8Operand.SPLIT: "2",
    Fp8Operand.STATIC: "s",
}


@dataclasses.dataclass(frozen=True)
class NativeFp8:
    """Which matmuls of a kernel run natively in the fp8 KV dtype.

    `qk` quantizes q per (token, head) row; `pv` quantizes the softmax weights
    per row. Every operand OFF (the default) is the dequantizing kernel.
    """

    qk: Fp8Operand = Fp8Operand.OFF
    pv: Fp8Operand = Fp8Operand.OFF

    def any(self) -> bool:
        return self.qk is not Fp8Operand.OFF or self.pv is not Fp8Operand.OFF

    def __str__(self) -> str:
        """Short form for logs: "qk" / "pv" with "2" for SPLIT and "s" for
        STATIC (e.g. "pvs", "qk2pv2"), "off" when no operand is on."""
        parts = [
            name + _FP8_OPERAND_SUFFIX[op]
            for name, op in (("qk", self.qk), ("pv", self.pv))
            if op is not Fp8Operand.OFF
        ]
        return "".join(parts) or "off"


NATIVE_FP8_OFF = NativeFp8()


# Inputs need three slots. The fetch for a tile is issued throughout the
# preceding tile rather than all at once at its start, so it cannot be waited
# on right away -- it needs one extra tile in flight for the DMA to land.
NUM_INPUT_SLOTS = 3

# Outputs only need two. The writeback issued at step N is waited at the top of
# step N + 2, which is exactly when slot N % 2 comes back around, so the DMA
# still gets all of step N + 1 to land.
NUM_OUTPUT_SLOTS = 2


@dataclasses.dataclass(frozen=True)
class ServingConfigs:
    """Serving config that can change depending on use cases."""

    num_seqs: int
    page_size: int
    total_q_tokens: int
    num_page_indices: int
    dtype_q: jnp.dtype
    dtype_kv: jnp.dtype
    dtype_out: jnp.dtype
    per_token_scale: bool | None = None
    per_token_scale_dtype: jnp.dtype | None = None
    scale_q: int | None = None
    scale_k: int | None = None
    scale_v: int | None = None
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE
    smem_fraction_limit_for_schedule_generation: float = 0.33
    max_schedule_size_multiplier: int = 16
    page_indices_window_size: int = 2048
    decode_query_size: int = 1
    cp_group_size: int | None = None
    attention_scope: AttentionScope = AttentionScope.FULL
    return_lse: bool = False
    skip_kv_update: bool = False
    mask_invalid_v: bool | None = None
    # Native fp8 matmuls of the DECODE kernel, and of the PREFILL and MIXED
    # kernels (no SPLIT operands there).
    native_fp8: NativeFp8 = NATIVE_FP8_OFF
    native_fp8_prefill: NativeFp8 = NATIVE_FP8_OFF
    # Run the DECODE softmax in base 2: log2(e) is folded into the score
    # scale and the exponentials are exp2. Equal in exact arithmetic, but
    # where the scale is folded into q (`flash_attention.q_folded_scale`) it
    # is cast to q's dtype: for a bf16 q at head_dim 256, log2(e) / 16 =
    # 0.09016844 rounds to 0.09033203, so the scores are scaled 0.18% more
    # than on the base-e path, and q * scale takes one more bf16 rounding.
    # Not applied when the kernel returns the LSE (return_lse, as the DCP
    # passes do), since callers merge LSEs in natural log, nor with soft_cap,
    # which caps the base-e scores. The wrapper logs either once.
    decode_exp2: bool = False

    def __post_init__(self):
        if Fp8Operand.SPLIT in (self.native_fp8.qk, self.native_fp8.pv) and (
            jnp.dtype(self.dtype_kv) == jnp.dtype(jnp.float8_e5m2)
        ):
            raise ValueError(
                f"native_fp8={self.native_fp8} with an fp8_e5m2 KV cache: e5m2 "
                "keeps 2 mantissa bits, so a SPLIT operand's residual half can "
                "overflow to inf; use ROW, or an fp8_e4m3 KV cache."
            )
        if self.mask_invalid_v is None:
            object.__setattr__(
                self,
                "mask_invalid_v",
                self.kv_layout == KVLayout.SEQ_ALONG_LANE,
            )
        for name in ("native_fp8", "native_fp8_prefill"):
            self._check_native_fp8(name, getattr(self, name))

    def _check_native_fp8(self, name: str, modes: NativeFp8) -> None:
        if not modes.any():
            return
        if jnp.dtype(self.dtype_kv) not in (
            jnp.dtype(jnp.float8_e4m3fn),
            jnp.dtype(jnp.float8_e5m2),
        ):
            raise ValueError(
                f"{name}={modes} runs matmuls in the fp8 KV dtype, but the KV "
                f"cache is {jnp.dtype(self.dtype_kv)}; use an fp8 KV cache or "
                f"leave {name} off."
            )
        if modes.qk is not Fp8Operand.OFF and self.scale_q is not None:
            raise ValueError(
                f"{name}.qk with a static q scale: the q-scale path already "
                "casts q to the KV dtype; pass one or the other."
            )
        if name == "native_fp8_prefill" and Fp8Operand.SPLIT in (modes.qk, modes.pv):
            raise ValueError(
                f"{name}={modes}: SPLIT operands are DECODE only; use ROW or STATIC."
            )
        if modes.pv is Fp8Operand.STATIC and self.per_token_scale:
            raise ValueError(
                f"{name}.pv STATIC with per-token V scales: the scales multiply "
                "the softmax weights before the cast and can leave the fp8 "
                "range; use ROW."
            )

    @property
    def max_decode_bkv_p_new(self) -> int:
        return 1 + pl.cdiv(self.decode_query_size - 1, self.page_size)

    @property
    def pages_per_seq(self) -> int:
        return self.num_page_indices // self.num_seqs

    @property
    def page_size_log2(self) -> int:
        return (self.page_size - 1).bit_length()

    @property
    def page_size_mask(self) -> int:
        return self.page_size - 1

    @property
    def packing_q(self) -> int:
        return utils.get_dtype_packing(self.dtype_q)

    @property
    def packing_kv(self) -> int:
        return utils.get_dtype_packing(self.dtype_kv)

    @property
    def scale_channels(self) -> int:
        if self.per_token_scale:
            assert self.per_token_scale_dtype is not None
            scale_bits = jax.dtypes.itemsize_bits(self.per_token_scale_dtype)
            kv_bits = jax.dtypes.itemsize_bits(self.dtype_kv)
            return max(1, scale_bits // kv_bits)
        return 0


# Most chunks of max_steps_ub steps the HBM schedule holds.
MAX_SCHEDULE_CHUNKS = 128


def worst_case_schedule_chunks(cfgs: "RpaConfigs") -> int:
    """Chunks of `cfgs.max_steps_ub` steps the shape's worst case fills: each
    q block of each sequence paired with every KV block it can visit. That is
    every block of a full block table, except in the new-token pass
    (NEW_TOKENS_ONLY), which visits only the blocks of the sequence's new
    tokens; under context parallelism those are global positions, so they
    can span more blocks than the rank-local block table holds."""
    # DECODE: one q block per sequence. Its range holds only 1-token decodes
    # and speculative verify windows of at most decode_query_size tokens,
    # which is the DECODE bq.
    extra_q_blocks = 0
    if cfgs.mode != RpaCase.DECODE:
        extra_q_blocks = cfgs.serve.total_q_tokens // cfgs.bq_sz
    if cfgs.serve.attention_scope == AttentionScope.NEW_TOKENS_ONLY:
        # The j-th q block of a sequence visits the KV blocks from the one
        # holding the sequence's first new token to the one holding the q
        # block's last token (schedule.py's q_loop), at most visits(j) of
        # them. visits grows with j, so the worst case puts every q block
        # past the sequences' first ones in one sequence.
        def visits(j: int) -> int:
            return pl.cdiv((j + 1) * cfgs.bq_sz - 1, cfgs.bkv_sz) + 1

        pairs = cfgs.serve.num_seqs * visits(0)
        pairs += sum(visits(j) for j in range(1, extra_q_blocks + 1))
    else:
        kv_tokens = cfgs.serve.pages_per_seq * cfgs.serve.page_size
        q_blocks = cfgs.serve.num_seqs + extra_q_blocks
        pairs = q_blocks * pl.cdiv(kv_tokens, cfgs.bkv_sz)
    steps = pl.cdiv(pairs, cfgs.batch_size)
    return pl.cdiv(steps, cfgs.max_steps_ub)


@dataclasses.dataclass(frozen=True, eq=True)
class RpaConfigs:
    block: BlockSizes
    model: ModelConfigs
    serve: ServingConfigs
    mode: RpaCase
    vmem_limit_bytes: int

    # Expose block sizes for ease of use.

    @property
    def bq_sz(self) -> int:
        return self.block.bq_sz

    @property
    def bq_c_sz(self) -> int:
        return self.block.bq_c_sz

    @property
    def bkv_sz(self) -> int:
        return self.block.bkv_sz

    @property
    def batch_size(self) -> int:
        return self.block.batch_size

    def seq_range(
        self, distribution: jax.Array | jax.Ref
    ) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
        """Sequences this kernel call schedules.

        Batched RPA has no specialised prefill-only path, so MIXED also covers the
        prefill-only segment [distribution[0], distribution[1]).
        """
        if self.mode == RpaCase.DECODE:
            return 0, distribution[0]
        if self.mode == RpaCase.MIXED:
            return distribution[0], distribution[2]
        raise ValueError(f"Batched RPA does not dispatch {self.mode}; use MIXED.")

    # Define derived values.

    @property
    def max_steps_ub(self) -> int:
        """Get maximum upper bound of kernel steps based on SMEM limit."""

        fixed_bytes = 0
        fixed_bytes += self.serve.num_seqs  # kv_lens
        fixed_bytes += self.serve.num_seqs + 1  # cu_q_lens
        fixed_bytes += 2 * self.page_window_slot_len  # page_indices windows
        fixed_bytes += 3  # distribution
        fixed_bytes += self.block.batch_size  # lane_lengths
        fixed_bytes += 1  # actual_steps

        word_size_bytes = 4
        fixed_bytes *= word_size_bytes

        tpu_info = get_tpu_info()
        smem_limit_bytes = (
            tpu_info.smem_capacity_bytes - 32 * 1024
        ) * self.serve.smem_fraction_limit_for_schedule_generation
        available_bytes = smem_limit_bytes - fixed_bytes

        # Per step per batch item:
        # s_idx, q_idx, k_idx, is_last_k, do_writeback: 5 * 4 = 20
        # dma_q: 2 * 4 = 8
        # dma_kv_cache: bkv_p_cache * 3 * 4 = 12 * bkv_p_cache
        # dma_kv_new: bkv_p_new * self.dma_kv_new_size * 4
        bytes_per_step = (
            28 + 12 * self.bkv_p_cache + 4 * self.dma_kv_new_size * self.bkv_p_new
        )
        bytes_per_step *= self.block.batch_size
        # Add 20 bytes for the 5 total_wait fields (total_wait_kv_in,
        # total_wait_kv_out, total_wait_q_in, total_wait_o_out, total_wait_lse_out)
        # which are 1D arrays (not multiplied by batch_size).
        bytes_per_step += 20

        max_steps_ub = available_bytes // bytes_per_step

        num_lanes = tpu_info.num_lanes
        max_steps_ub = max(1, max_steps_ub // num_lanes) * num_lanes
        return int(max_steps_ub)

    @property
    def page_window_seqs(self) -> int:
        """Sequences per page_indices DMA window: a power of two dividing num_seqs."""
        max_rows = max(
            1, self.serve.page_indices_window_size // self.serve.pages_per_seq
        )
        rows = 1 << (max_rows.bit_length() - 1)
        return math.gcd(rows, self.serve.num_seqs)

    @property
    def page_window_entries(self) -> int:
        return self.page_window_seqs * self.serve.pages_per_seq

    @property
    def page_window_slot_len(self) -> int:
        """SMEM entries per window slot: lane-aligned, plus a lane group of slack
        when the window itself is not."""
        num_lanes = get_tpu_info().num_lanes
        entries = self.page_window_entries
        return utils.align_to(entries, num_lanes) + (
            num_lanes if entries % num_lanes else 0
        )

    @property
    def bkv_p(self) -> int:
        return pl.cdiv(self.block.bkv_sz, self.serve.page_size)

    @property
    def bkv_p_cache(self) -> int:
        if self.mode == RpaCase.PREFILL:
            return 0
        return self.bkv_p

    @property
    def bkv_p_new(self) -> int:
        if self.mode == RpaCase.DECODE or self.block.bq_sz == 1:
            return min(self.serve.max_decode_bkv_p_new, self.bkv_p)
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return self.bkv_p + 1
        return self.bkv_p

    @property
    def bkv_stride(self) -> int:
        bkv_stride = pl.cdiv(self.model.num_kv_heads * 2, self.serve.packing_kv)

        if utils.has_bank_conflicts(bkv_stride):
            bkv_stride += 1
        return bkv_stride

    @property
    def aligned_q_head_dim(self) -> int:
        num_lanes = get_tpu_info().num_lanes
        return utils.align_to(self.model.head_dim, num_lanes)

    # Actual head dimension used for compute, does not include scale factors.
    @property
    def compute_kv_head_dim(self) -> int:
        tpu_info = get_tpu_info()
        num_lanes = tpu_info.num_lanes
        num_sublanes = tpu_info.num_sublanes
        kv_packing = utils.get_dtype_packing(self.serve.dtype_kv)
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return utils.align_to(self.model.head_dim, num_sublanes * kv_packing)
        return utils.align_to(self.model.head_dim, num_lanes)

    # Head dimension used for storing kv in VMEM, includes scale factors if
    # per_token_scale is enabled.
    @property
    def aligned_kv_head_dim(self) -> int:
        tpu_info = get_tpu_info()
        num_lanes = tpu_info.num_lanes
        num_sublanes = tpu_info.num_sublanes
        kv_packing = utils.get_dtype_packing(self.serve.dtype_kv)

        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            total_head_dim = self.model.head_dim
            if self.serve.dtype_kv == jnp.uint8:
                total_head_dim = total_head_dim // 2
            # If per-token scale is enabled, we need to add num_scale_channels to the
            # head dimension to account for the scale factor. Note that if per-token
            # scale is not enabled, self.serve.scale_channels is 0.
            total_head_dim += self.serve.scale_channels

            return utils.align_to(total_head_dim, num_sublanes * kv_packing)

        return utils.align_to(self.model.head_dim, num_lanes)

    @property
    def aligned_num_kv_heads_x2(self) -> int:
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return self.model.num_kv_heads * 2
        packing_kv = self.serve.packing_kv
        return utils.align_to(self.model.num_kv_heads * 2, packing_kv)

    @property
    def aligned_num_q_heads_per_kv_head(self) -> int:
        packing_q = self.serve.packing_q
        return utils.align_to(self.model.num_q_heads_per_kv_head, packing_q)

    @property
    def kv_hbm_stride(self) -> int:
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return self.model.num_kv_heads * 2
        kv_packing = utils.get_dtype_packing(self.serve.dtype_kv)
        return utils.align_to(self.model.num_kv_heads * 2, kv_packing) // kv_packing

    @property
    def fuse_accum(self) -> bool:
        return self.mode == RpaCase.DECODE

    @property
    def kv_bytes_per_token(self) -> int:
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            num_elements = self.model.num_kv_heads * 2 * self.aligned_kv_head_dim
        else:
            num_elements = self.aligned_num_kv_heads_x2 * self.aligned_kv_head_dim
        # Calculate byte size using bits to account for subbyte dtypes like fp4.
        kv_dtype_bytes = utils.get_dtype_bits(self.serve.dtype_kv)
        return int(num_elements * kv_dtype_bytes) >> 3  # 8 bits per byte

    @property
    def q_bytes_per_token(self) -> int:
        q_dtype_bytes = utils.get_dtype_bits(self.serve.dtype_q)
        return (
            int(
                self.model.num_kv_heads
                * self.aligned_num_q_heads_per_kv_head
                * self.aligned_q_head_dim
                * q_dtype_bytes
            )
            >> 3
        )

    @property
    def o_bytes_per_token(self) -> int:
        out_dtype_bytes = utils.get_dtype_bits(self.serve.dtype_out)
        return (
            int(
                self.model.num_kv_heads
                * self.aligned_num_q_heads_per_kv_head
                * self.aligned_q_head_dim
                * out_dtype_bytes
            )
            >> 3
        )

    @property
    def q_vmem_shape(self):
        q_per_kv_packing = self.aligned_num_q_heads_per_kv_head // self.serve.packing_q
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz,
            q_per_kv_packing,
            self.serve.packing_q,
            self.aligned_q_head_dim,
        )

    @property
    def kv_token_align(self) -> int:
        """DMA token granularity: the 128-lane tile for SEQ_ALONG_LANE, else 1.

        SEQ_ALONG_LANE packs tokens along the lane dim, so every DMA token count
        must be a multiple of the 128-lane tile. HEAD_ALONG_SUBLANE puts tokens on
        the sublane dim, where single tokens are addressable.
        """
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return get_tpu_info().num_lanes
        return 1

    @property
    def kv_vmem_lanes(self) -> int:
        """Lane-dim size of the KV VMEM buffer; SEQ_ALONG_LANE only."""
        # o = kv from cache, n = kv from new, - = pad. Each [] is a 128-lane cell:
        #
        #   want  [onnn][nnnn]
        #   got   [o---][---n][nnnn][nn--]
        #
        # because we fetch at 128-lane boundary, we need 2 extra padding slots
        return self.block.bkv_sz + 2 * self.kv_token_align

    @property
    def kv_vmem_shape(self):
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return (
                self.block.batch_size,
                self.model.num_kv_heads * 2,
                self.aligned_kv_head_dim // self.serve.packing_kv,
                self.serve.packing_kv,
                self.kv_vmem_lanes,
            )
        return (
            self.block.batch_size,
            self.block.bkv_sz,
            self.bkv_stride,
            self.serve.packing_kv,
            self.aligned_kv_head_dim,
        )

    @property
    def dma_kv_new_size(self) -> int:
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return 5
        if (
            self.serve.kv_layout == KVLayout.HEAD_ALONG_SUBLANE
            and self.serve.cp_group_size is not None
        ):
            return 5
        return 4

    @property
    def lm_scratch_shape(self):
        num_lanes = get_tpu_info().num_lanes
        return (
            self.model.num_kv_heads,
            self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
            num_lanes,
        )

    @property
    def lse_vmem_shape(self):
        num_lanes = get_tpu_info().num_lanes
        q_per_kv_packing = self.aligned_num_q_heads_per_kv_head // self.serve.packing_q
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz,
            q_per_kv_packing,
            self.serve.packing_q,
            num_lanes,
        )

    @property
    def acc_scratch_shape(self):
        return (
            self.model.num_kv_heads,
            self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
            self.compute_kv_head_dim,
        )

    @property
    def max_schedule_size_multiplier(self) -> int:
        """HBM schedule size in chunks of max_steps_ub steps: the serving
        value, raised (up to MAX_SCHEDULE_CHUNKS) to this shape's worst case
        (`worst_case_schedule_chunks`). Past it the schedule kernel drops steps
        rather than write out of bounds (see rpa_metadata_schedule_kernel)."""
        need = min(worst_case_schedule_chunks(self), MAX_SCHEDULE_CHUNKS)
        return max(self.serve.max_schedule_size_multiplier, need)

    def validate_inputs(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        kv_cache: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
    ):
        """Validate inputs to the RPA kernel statically."""

        if not q.ndim == k.ndim == v.ndim == 3:
            raise ValueError(
                f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}"
            )
        if k.shape != v.shape:
            raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
        if not (q.shape[0] == k.shape[0] == v.shape[0]):
            raise ValueError(
                "Expected number of sequences in Q, K, and V to be the same, but got"
                f" {q.shape[0]=}, {k.shape[0]=}, and {v.shape[0]=}"
            )
        expected_kv_head_dim = q.shape[2]
        if self.serve.dtype_kv == jnp.uint8:
            expected_kv_head_dim = q.shape[2] // 2

        if not (k.shape[2] == v.shape[2] == expected_kv_head_dim):
            raise ValueError(
                "Expected number of head dimensions in K and V to be"
                f" {expected_kv_head_dim} (matching Q shape {q.shape[2]} with"
                f" packing for {self.serve.dtype_kv}), but got {k.shape[2]=} and"
                f" {v.shape[2]=}"
            )

        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            if self.serve.page_size % 128 != 0:
                raise ValueError(
                    "Expected page_size to be a multiple of 128 for SEQ_ALONG_LANE"
                    f" tile alignment, but got {self.serve.page_size=}"
                )
            expected_kv_cache_shape = (
                kv_cache.shape[0],
                self.model.num_kv_heads * 2,
                self.aligned_kv_head_dim // self.serve.packing_kv,
                self.serve.packing_kv,
                self.serve.page_size,
            )
        else:
            expected_kv_cache_shape = (
                kv_cache.shape[0],
                self.serve.page_size,
                self.aligned_num_kv_heads_x2 // self.serve.packing_kv,
                self.serve.packing_kv,
                self.aligned_kv_head_dim,
            )

        if kv_cache.shape != expected_kv_cache_shape:
            raise ValueError(
                f"Expected {kv_cache.shape=} to be equal to {expected_kv_cache_shape=}"
            )

        # If we are using per-token quantization, we expect the kv-cache scale to be
        # in VMEM, and so it will not be in the configs.
        if self.serve.per_token_scale:
            if self.serve.kv_layout != KVLayout.SEQ_ALONG_LANE:
                raise ValueError(
                    "Expected kv_layout=KVLayout.SEQ_ALONG_LANE when per_token_scale is"
                    f" True, but got {self.serve.kv_layout=}."
                )
            if self.serve.scale_k is not None or self.serve.scale_v is not None:
                raise ValueError(
                    "Expected scale_k and scale_v to be None when per_token_scale is"
                    " True."
                )

        if not (
            jnp.int32
            == kv_lens.dtype
            == page_indices.dtype
            == cu_q_lens.dtype
            == distribution.dtype
        ):
            raise ValueError(
                f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
                f" {cu_q_lens.dtype=}, {distribution.dtype=}"
            )

        if not (kv_lens.ndim == page_indices.ndim == cu_q_lens.ndim == 1):
            raise ValueError(
                f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
                f" {cu_q_lens.shape=}"
            )

        max_num_seqs = kv_lens.shape[0]
        num_page_indices = page_indices.shape[0]
        if num_page_indices % max_num_seqs != 0:
            raise ValueError(
                f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
            )
        if cu_q_lens.shape != (max_num_seqs + 1,):
            raise ValueError(
                f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)."
            )
        if distribution.shape != (3,):
            raise ValueError(f"Expected {distribution.shape=} to be (3,).")

        # Context Parallel Support
        if self.serve.cp_group_size is not None:
            if self.serve.attention_scope == AttentionScope.FULL:
                raise ValueError(
                    "Context Parallel does not support AttentionScope.FULL"
                    " where cache is sharded but current tokens is sequential"
                )
            if self.model.sliding_window is not None:
                raise ValueError(
                    "Context Parallel does not support sliding window right now"
                )

        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            bkv_sz = self.block.bkv_sz
            page_size = self.serve.page_size
            if bkv_sz % page_size != 0:
                raise NotImplementedError(
                    f"SEQ_ALONG_LANE expects {bkv_sz=} to be aligned to {page_size=}."
                )
