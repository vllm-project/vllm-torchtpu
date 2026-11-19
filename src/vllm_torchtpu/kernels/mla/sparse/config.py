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
"""Typed configuration, constants, and geometry predicates for SparseMLA."""

from __future__ import annotations

import dataclasses
from enum import Enum, StrEnum
from typing import Any, Literal, overload

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.mla.sparse import native_sc_gather

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE = 64
DEFAULT_ATTENTION_KERNEL_BATCH_SIZE = 16
# Native SparseCore gather unit; ``topk_cap`` must be a multiple of this.
GATHER_ATOM_ROWS = native_sc_gather.OCCURRENCES_PER_ATOM

# Decode buckets served by fused window prep (whole top-k block fits in VMEM).
FUSED_PREP_MAX_TOKENS = 32
# One (8, 128) i32 tile appended after window rows for the native gather live vec.
FUSED_PREP_META_ROWS = 8

# In-kernel q rope layout: raw q head is nope (192) | rope (64) = 256 lanes.
Q_ROPE_DIM = 64
Q_ROPE_HEAD_DIM = 256

DECODE_STEP8_TOKENS = 8
MIXED_BODY_MAX_SKIP_TOKENS = 64

# One 16-bit sublane tile of padding ahead of each destination's partial rows;
# row SEND_PAD_ROWS - 1 holds the packed LSE words.
SEND_PAD_ROWS = 16
SEND_SLAB_ROW0 = SEND_PAD_ROWS // 2
# Distinct barrier semaphore group for the fused DCP all-to-all exchange.
SEND_EXCHANGE_COLLECTIVE_ID = 13
SEND_EXCHANGE_VMEM_LIMIT_BYTES = 16 * 1024 * 1024


class _Unset(Enum):
    """Hashable sentinel distinguishing omitted legacy kwargs from explicit ones."""

    UNSET = "UNSET"


UNSET = _Unset.UNSET


class CacheLayout(StrEnum):
    """Physical KV cache layout consumed by SparseMLA."""

    TENSORCORE = "tensorcore"
    SPARSECORE = "sparsecore"


TENSORCORE_CACHE_LAYOUT = CacheLayout.TENSORCORE
SPARSECORE_CACHE_LAYOUT = CacheLayout.SPARSECORE


class GatherWaitMode(StrEnum):
    """Synchronization mode for native SparseCore NOPE/ROPE gather DMAs."""

    OVERLAP_ROPE = "overlap_rope"
    BEFORE_ROPE = "before_rope"


class BodyExecutionStrategy(StrEnum):
    """Inner-loop attention execution strategy within a grid step."""

    TOKEN_PIPELINED = "token_pipelined"
    PAIR = "pair"
    QUAD = "quad"
    TRANSPOSED_QK = "xqk"


class HbmPinRole(StrEnum):
    """Operand roles eligible for explicit HBM pinning in fused exchange mode."""

    KV = "kv"
    Q = "q"
    Q_PE = "q_pe"
    TAB = "tab"
    RECV = "recv"


SEND_EXCHANGE_PIN_ROLES = tuple(r.value for r in HbmPinRole)

MIXED_BODY_LEVERS = (
    "tr",
    "zu",
    "pv512",
    "noinf",
    "pbf16",
    "fxskip",
    "q128",
    "pair",
    "xqk",
)
MIXED_BODY_DEFAULT = ("tr", "zu", "pv512", "noinf")

DECODE_BODY_LEVERS = (
    "tr",
    "zu",
    "pv512",
    "noinf",
    "pair",
    "fxskip",
    "q128",
    "nofill",
    "quad",
    "lrows",
    "step8",
)
DECODE_BODY_DEFAULT = (
    "tr",
    "zu",
    "pv512",
    "noinf",
    "quad",
    "fxskip",
    "lrows",
    "step8",
)
MIXED_KERNEL_V2_LEVERS = ("fxskip", "pair")

_MIXED_BODY_LEVER_SET = frozenset(MIXED_BODY_LEVERS)
_DECODE_BODY_LEVER_SET = frozenset(DECODE_BODY_LEVERS)
_ALL_BODY_LEVER_SET = _MIXED_BODY_LEVER_SET | _DECODE_BODY_LEVER_SET


@overload
def cdiv(a: int, b: int) -> int: ...


@overload
def cdiv(a: jax.Array, b: int) -> jax.Array: ...


def cdiv(a: int | jax.Array, b: int) -> int | jax.Array:
    """Ceiling integer division ``ceil(a / b)`` for non-negative ``a``."""
    assert b != 0
    return (a + b - 1) // b


@overload
def align_to(x: int, a: int) -> int: ...


@overload
def align_to(x: jax.Array, a: int) -> jax.Array: ...


def align_to(x: int | jax.Array, a: int) -> int | jax.Array:
    """Rounds ``x`` up to the next multiple of ``a``."""
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype: Any) -> int:
    """Returns the bit width of a JAX/NumPy dtype."""
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype: Any) -> int:
    """Returns the number of elements of ``dtype`` packed per 32-bit word."""
    return 32 // get_dtype_bitwidth(dtype)


def largest_divisor(x: int, cap: int) -> int:
    """Returns the largest divisor of ``x`` that is <= ``cap``."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def live_group_supported(
    chunk_size: int, kernel_batch_size: int, topk_cap: int
) -> bool:
    """Whether the live-group capped path serves a chunk.

    Requires whole grid steps per chunk, a 64-slot atom block per grid step
    aligned to ``(8, 128)`` i32 tiles, and a window of whole 128-lane rows.
    """
    atom = GATHER_ATOM_ROWS
    tile = kernel_batch_size * atom
    return (
        chunk_size % kernel_batch_size == 0
        and tile % (8 * 128) == 0
        and topk_cap % 128 == 0
        and topk_cap % atom == 0
        and tile % topk_cap == 0
    )


def mixed_fused_prep_supported(
    num_tokens: int,
    topk: int,
    chunk_size: int,
    kernel_batch_size: int,
    topk_cap: int,
) -> bool:
    """Whether fused live-group window prep serves a mixed bucket."""
    return (
        live_group_supported(chunk_size, kernel_batch_size, topk_cap)
        and num_tokens % chunk_size == 0
        and chunk_size % 128 == 0
        and topk % topk_cap == 0
        and topk % 128 == 0
    )


def merge_pieces_supported(num_tokens: int, chunk_size: int, pieces: int) -> bool:
    """Whether ``out_pieces=pieces`` applies to a multi-chunk bucket."""
    if pieces <= 1 or num_tokens <= chunk_size or num_tokens % chunk_size:
        return False
    if num_tokens % pieces:
        return False
    rows = num_tokens // pieces
    return rows % chunk_size == 0 or chunk_size % rows == 0


def chunk_kernel_batch_size(
    chunk_size: int,
    attention_kernel_batch_size: int,
    small_chunk_tokens: int,
    small_chunk_batch_size: int | None,
) -> tuple[bool, int]:
    """Returns ``(is_small_chunk, attention_grid_step)`` for a chunk."""
    small_chunk = small_chunk_tokens > 0 and chunk_size <= small_chunk_tokens
    batch_cap = attention_kernel_batch_size
    if small_chunk and small_chunk_batch_size is not None:
        batch_cap = small_chunk_batch_size
    return small_chunk, largest_divisor(chunk_size, batch_cap)


def send_layout_lanes(num_tokens: int, num_heads: int, world: int) -> int:
    """Lane width of the in-kernel LSE word build for DCP send layout."""
    rows = num_tokens * (num_heads // world)
    return align_to(max(2 * rows, rows + num_heads), 128)


def send_layout_supported(
    num_tokens: int, num_heads: int, world: int, vdim: int, batch_size: int
) -> bool:
    """Whether the kernel can write the packed DCP send buffer directly."""
    if world < 2 or num_heads % world or vdim % 128:
        return False
    per = num_heads // world
    rows = num_tokens * per
    lanes = send_layout_lanes(num_tokens, num_heads, world)
    return (
        per & (per - 1) == 0
        and rows & (rows - 1) == 0
        and lanes & (lanes - 1) == 0
        and 2 * rows <= vdim
        and lanes <= vdim
        # The word build's strided roll shifts sublane d by d*L < a vreg.
        and (world - 1) * per < 128
        and num_tokens % batch_size == 0
        and (batch_size * per) % SEND_PAD_ROWS == 0
    )


def send_layout_shape(
    num_tokens: int, num_heads: int, world: int, vdim: int
) -> tuple[int, int, int]:
    """Returns the shape of one rank's int16 DCP send buffer."""
    return (world, SEND_PAD_ROWS + num_tokens * (num_heads // world), vdim)


@dataclasses.dataclass(frozen=True)
class BodyOptimizationConfig:
    """Attention-body execution strategy and kernel optimization levers."""

    strategy: BodyExecutionStrategy = BodyExecutionStrategy.TOKEN_PIPELINED
    transpose_nope_unpack: bool = False  # "tr"
    zero_nope_words_only: bool = False  # "zu"
    pv_nope_only: bool = False  # "pv512"
    drop_runahead_dep: bool = False  # "noinf"
    bf16_pv_weights: bool = False  # "pbf16"
    skip_clean_overflow_fixup: bool = False  # "fxskip"
    narrow_q_rope_dma: bool = False  # "q128"
    skip_redundant_lse_fill: bool = False  # "nofill"
    fetch_live_rows_only: bool = False  # "lrows"
    step8_attention_grid: bool = False  # "step8"

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", BodyExecutionStrategy(self.strategy))

    @property
    def uses_kv_stage(self) -> bool:
        """Whether the strategy stages unpacked KV tiles in a VMEM ring."""
        return self.strategy in (
            BodyExecutionStrategy.PAIR,
            BodyExecutionStrategy.QUAD,
        )

    @property
    def has_any_lever(self) -> bool:
        """True iff any body optimization differs from the baseline default."""
        return (
            self.strategy != BodyExecutionStrategy.TOKEN_PIPELINED
            or self.transpose_nope_unpack
            or self.zero_nope_words_only
            or self.pv_nope_only
            or self.drop_runahead_dep
            or self.bf16_pv_weights
            or self.skip_clean_overflow_fixup
            or self.narrow_q_rope_dma
            or self.skip_redundant_lse_fill
            or self.fetch_live_rows_only
            or self.step8_attention_grid
        )

    @property
    def use_lever_loader(self) -> bool:
        """Whether ``load_bkv`` dispatches to the lever KV unpack path.

        Matches legacy ``if levers: return load_bkv_levers(...)`` semantics:
        any active lever routes through the SparseCore lever loader.
        """
        return self.has_any_lever

    def without_narrow_q_rope(self) -> BodyOptimizationConfig:
        """Returns a copy with ``narrow_q_rope_dma`` disabled."""
        if not self.narrow_q_rope_dma:
            return self
        return dataclasses.replace(self, narrow_q_rope_dma=False)

    def to_levers(self) -> tuple[str, ...]:
        """Serializes enabled optimizations back to legacy lever strings."""
        levers: list[str] = []
        if self.transpose_nope_unpack:
            levers.append("tr")
        if self.zero_nope_words_only:
            levers.append("zu")
        if self.pv_nope_only:
            levers.append("pv512")
        if self.drop_runahead_dep:
            levers.append("noinf")
        if self.bf16_pv_weights:
            levers.append("pbf16")
        if self.strategy == BodyExecutionStrategy.PAIR:
            levers.append("pair")
        elif self.strategy == BodyExecutionStrategy.QUAD:
            levers.append("quad")
        elif self.strategy == BodyExecutionStrategy.TRANSPOSED_QK:
            levers.append("xqk")
        if self.skip_clean_overflow_fixup:
            levers.append("fxskip")
        if self.narrow_q_rope_dma:
            levers.append("q128")
        if self.skip_redundant_lse_fill:
            levers.append("nofill")
        if self.fetch_live_rows_only:
            levers.append("lrows")
        if self.step8_attention_grid:
            levers.append("step8")
        return tuple(levers)

    def validate_for_mode(self, mode: Literal["mixed", "decode"]) -> None:
        """Validates this config against mixed or decode mode constraints."""
        if mode == "mixed":
            if (
                self.strategy == BodyExecutionStrategy.QUAD
                or self.skip_redundant_lse_fill
                or self.fetch_live_rows_only
                or self.step8_attention_grid
            ):
                raise ValueError(
                    f"mixed_body_levers {self.to_levers()} not in {MIXED_BODY_LEVERS}"
                )
        elif mode == "decode":
            if (
                self.strategy == BodyExecutionStrategy.TRANSPOSED_QK
                or self.bf16_pv_weights
            ):
                raise ValueError(
                    f"decode_body_levers {self.to_levers()} not in {DECODE_BODY_LEVERS}"
                )
            if self.zero_nope_words_only and not self.pv_nope_only:
                # "zu" leaves ROPE rows past kv_len as gathered; only a NOPE-only
                # PV keeps them out of the output.
                raise ValueError("decode_body_levers: 'zu' needs 'pv512'")
        else:
            raise ValueError(f"unknown body lever mode {mode!r}")

    @classmethod
    def from_levers(
        cls,
        levers: tuple[str, ...] | BodyOptimizationConfig,
        *,
        mode: Literal["mixed", "decode", "any"] = "any",
    ) -> BodyOptimizationConfig:
        """Parses and validates a legacy lever tuple into a typed config."""
        if isinstance(levers, cls):
            if mode != "any":
                levers.validate_for_mode(mode)
            return levers

        lever_tuple = tuple(levers)
        lever_set = frozenset(lever_tuple)
        if mode == "mixed":
            if not lever_set <= _MIXED_BODY_LEVER_SET:
                raise ValueError(
                    f"mixed_body_levers {lever_tuple} not in {MIXED_BODY_LEVERS}"
                )
            if "pair" in lever_set and "xqk" in lever_set:
                raise ValueError("mixed_body_levers: 'pair' and 'xqk' are alternatives")
        elif mode == "decode":
            if not lever_set <= _DECODE_BODY_LEVER_SET:
                raise ValueError(
                    f"decode_body_levers {lever_tuple} not in {DECODE_BODY_LEVERS}"
                )
            if "quad" in lever_set and "pair" in lever_set:
                raise ValueError(
                    "decode_body_levers: 'pair' and 'quad' are alternatives"
                )
            if "zu" in lever_set and "pv512" not in lever_set:
                raise ValueError("decode_body_levers: 'zu' needs 'pv512'")
        elif mode == "any":
            if not lever_set <= _ALL_BODY_LEVER_SET:
                allowed = tuple(sorted(_ALL_BODY_LEVER_SET))
                raise ValueError(f"body_levers {lever_tuple} not in {allowed}")
            strategies = [s for s in ("pair", "quad", "xqk") if s in lever_set]
            if len(strategies) > 1:
                raise ValueError(f"body_levers: {strategies} are alternatives")
        else:
            raise ValueError(f"unknown body lever mode {mode!r}")

        if "pair" in lever_set:
            strategy = BodyExecutionStrategy.PAIR
        elif "quad" in lever_set:
            strategy = BodyExecutionStrategy.QUAD
        elif "xqk" in lever_set:
            strategy = BodyExecutionStrategy.TRANSPOSED_QK
        else:
            strategy = BodyExecutionStrategy.TOKEN_PIPELINED

        return cls(
            strategy=strategy,
            transpose_nope_unpack="tr" in lever_set,
            zero_nope_words_only="zu" in lever_set,
            pv_nope_only="pv512" in lever_set,
            drop_runahead_dep="noinf" in lever_set,
            bf16_pv_weights="pbf16" in lever_set,
            skip_clean_overflow_fixup="fxskip" in lever_set,
            narrow_q_rope_dma="q128" in lever_set,
            skip_redundant_lse_fill="nofill" in lever_set,
            fetch_live_rows_only="lrows" in lever_set,
            step8_attention_grid="step8" in lever_set,
        )


@dataclasses.dataclass(frozen=True)
class SendExchangeConfig:
    """Configuration for the fused DCP small all-to-all exchange."""

    axis: str
    lane_axis: str | None = None
    vmem_limit_bytes: int = SEND_EXCHANGE_VMEM_LIMIT_BYTES
    hbm_pin: bool | tuple[HbmPinRole, ...] = True
    in_kernel_merge: bool = False

    def __post_init__(self) -> None:
        if self.vmem_limit_bytes <= 0:
            raise ValueError(
                f"vmem_limit_bytes must be positive, got {self.vmem_limit_bytes}"
            )
        if isinstance(self.hbm_pin, bool):
            return
        if isinstance(self.hbm_pin, tuple):
            normalized = tuple(HbmPinRole(role) for role in self.hbm_pin)
            object.__setattr__(self, "hbm_pin", normalized)
            return
        raise ValueError(
            f"hbm_pin must be a bool or tuple of HbmPinRole, got {self.hbm_pin!r}"
        )

    def should_pin(self, role: HbmPinRole | str | None) -> bool:
        """Whether an operand with ``role`` should be pinned to HBM.

        Matches legacy ``pin is True or (pin and r in pin)``: when ``hbm_pin``
        is ``True``, every operand (including ``role=None``) is pinned; when it
        is a role tuple, only matching ``HbmPinRole`` members are pinned.
        """
        if self.hbm_pin is True:
            return True
        if isinstance(self.hbm_pin, tuple) and role is not None:
            return role in self.hbm_pin
        return False

    @classmethod
    def from_raw(
        cls, raw: tuple[Any, ...] | SendExchangeConfig | None
    ) -> SendExchangeConfig | None:
        """Parses a legacy 2-to-5-element ``send_exchange`` tuple or config."""
        if raw is None or isinstance(raw, cls):
            return raw
        if not isinstance(raw, tuple) or len(raw) not in (2, 3, 4, 5):
            raise ValueError(
                "send_exchange must be a 2-to-5-element tuple "
                "(axis, lane_axis[, vmem_limit_bytes[, hbm_pin[, in_kernel_merge]]]), "
                f"got {raw!r}"
            )
        axis = raw[0]
        lane_axis = raw[1]
        vmem_limit = (
            SEND_EXCHANGE_VMEM_LIMIT_BYTES
            if len(raw) <= 2 or raw[2] is None
            else raw[2]
        )
        raw_pin = raw[3] if len(raw) > 3 else True
        hbm_pin: bool | tuple[HbmPinRole, ...]
        if raw_pin is None:
            hbm_pin = False
        elif isinstance(raw_pin, bool):
            hbm_pin = raw_pin
        elif isinstance(raw_pin, tuple):
            hbm_pin = tuple(HbmPinRole(r) for r in raw_pin)
        else:
            raise ValueError(
                f"hbm_pin must be a bool, None, or tuple of HbmPinRole, "
                f"got {raw_pin!r}"
            )
        in_kernel_merge = bool(raw[4]) if len(raw) > 4 else False
        return cls(
            axis=axis,
            lane_axis=lane_axis,
            vmem_limit_bytes=vmem_limit,
            hbm_pin=hbm_pin,
            in_kernel_merge=in_kernel_merge,
        )


@dataclasses.dataclass(frozen=True)
class GatherConfig:
    """KV cache layout and gather chunking configuration."""

    cache_layout: CacheLayout = CacheLayout.TENSORCORE
    packed_rope: bool = False
    chunk_size: int = DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE
    sparsecore_atoms_per_batch: int = 16
    sparsecore_wait_mode: GatherWaitMode = GatherWaitMode.OVERLAP_ROPE
    sparsecore_pipelined: bool = True
    small_chunk_tokens: int = 16
    small_chunk_batch_size: int | None = 1
    small_chunk_live_gather: bool = True
    small_chunk_gather_atoms: int = 1
    live_gather_all_chunks: bool = True

    def __post_init__(self) -> None:
        try:
            layout = CacheLayout(self.cache_layout)
        except ValueError:
            raise ValueError(
                f"cache_layout must be {TENSORCORE_CACHE_LAYOUT.value!r} or "
                f"{SPARSECORE_CACHE_LAYOUT.value!r}, got {self.cache_layout!r}"
            ) from None
        object.__setattr__(self, "cache_layout", layout)

        wait_mode = GatherWaitMode(self.sparsecore_wait_mode)
        object.__setattr__(self, "sparsecore_wait_mode", wait_mode)

        if self.packed_rope and layout != CacheLayout.SPARSECORE:
            raise ValueError("packed_rope needs the sparsecore cache layout")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.sparsecore_atoms_per_batch <= 0:
            raise ValueError(
                "sparsecore_atoms_per_batch must be positive, "
                f"got {self.sparsecore_atoms_per_batch}"
            )
        if self.small_chunk_tokens < 0:
            raise ValueError(
                "small_chunk_tokens must be non-negative, "
                f"got {self.small_chunk_tokens}"
            )
        if self.small_chunk_batch_size is not None and self.small_chunk_batch_size <= 0:
            raise ValueError(
                "small_chunk_batch_size must be positive when set, "
                f"got {self.small_chunk_batch_size}"
            )
        if self.small_chunk_gather_atoms <= 0:
            raise ValueError(
                "small_chunk_gather_atoms must be positive, "
                f"got {self.small_chunk_gather_atoms}"
            )

    @property
    def is_sparsecore(self) -> bool:
        """Whether the physical KV cache uses the native SparseCore layout."""
        return self.cache_layout == CacheLayout.SPARSECORE

    @property
    def rope_tokens_per_row(self) -> int:
        """Tokens stored per 128-word ROPE cache row."""
        return 8 if self.packed_rope else 4


def _validate_width_ladder(
    widths: tuple[int, ...], name: str, topk_cap: int | None
) -> None:
    if widths and (
        list(widths) != sorted(set(widths)) or any(w <= 0 or w % 128 for w in widths)
    ):
        raise ValueError(
            f"{name} {widths} must be ascending multiples of 128 "
            f"below topk_cap ({topk_cap})"
        )


@dataclasses.dataclass(frozen=True)
class CappedWindowConfig:
    """Capped top-k windowing, live-group, and body-optimization settings."""

    topk_cap: int | None = None
    capped_live_rows: bool = False
    fused_window_prep: bool = False
    mixed_live_gather: bool = False
    mixed_fused_prep: bool = False
    mixed_live_widths: tuple[int, ...] = ()
    decode_row_widths: tuple[int, ...] = ()
    mixed_body: BodyOptimizationConfig = dataclasses.field(
        default_factory=BodyOptimizationConfig
    )
    decode_body: BodyOptimizationConfig = dataclasses.field(
        default_factory=BodyOptimizationConfig
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "mixed_live_widths", tuple(self.mixed_live_widths))
        object.__setattr__(self, "decode_row_widths", tuple(self.decode_row_widths))
        if self.topk_cap is not None and self.topk_cap < 0:
            raise ValueError(f"topk_cap must be non-negative, got {self.topk_cap}")
        _validate_width_ladder(
            self.mixed_live_widths, "mixed_live_widths", self.topk_cap
        )
        _validate_width_ladder(
            self.decode_row_widths, "decode_row_widths", self.topk_cap
        )
        self.mixed_body.validate_for_mode("mixed")
        self.decode_body.validate_for_mode("decode")

    def resolve_for_inputs(
        self,
        sparsecore_layout: bool,
        topk: int,
        has_q_rope_table: bool,
    ) -> CappedWindowConfig:
        """Resolves ``topk_cap`` and body flags against concrete input geometry."""
        topk_cap = self.topk_cap
        if not sparsecore_layout or (topk_cap is not None and topk_cap >= topk):
            topk_cap = None
        if topk_cap is not None and (
            topk_cap <= 0 or topk_cap % GATHER_ATOM_ROWS or topk % topk_cap
        ):
            raise ValueError(
                f"topk_cap ({topk_cap}) must be a multiple of {GATHER_ATOM_ROWS} "
                f"dividing topk ({topk})"
            )
        if self.mixed_live_widths and (
            topk_cap is None
            or any(not 0 < w < topk_cap for w in self.mixed_live_widths)
        ):
            raise ValueError(
                f"mixed_live_widths {self.mixed_live_widths} must be ascending "
                f"multiples of 128 below topk_cap ({topk_cap})"
            )
        if self.decode_row_widths and (
            topk_cap is None
            or any(not 0 < w < topk_cap for w in self.decode_row_widths)
        ):
            raise ValueError(
                f"decode_row_widths {self.decode_row_widths} must be ascending "
                f"multiples of 128 below topk_cap ({topk_cap})"
            )
        mixed_body = (
            self.mixed_body
            if has_q_rope_table
            else self.mixed_body.without_narrow_q_rope()
        )
        decode_body = (
            self.decode_body
            if has_q_rope_table
            else self.decode_body.without_narrow_q_rope()
        )
        if (
            topk_cap == self.topk_cap
            and mixed_body == self.mixed_body
            and decode_body == self.decode_body
        ):
            return self
        return dataclasses.replace(
            self,
            topk_cap=topk_cap,
            mixed_body=mixed_body,
            decode_body=decode_body,
        )


@dataclasses.dataclass(frozen=True)
class LseConfig:
    """Log-sum-exp output layout and ownership settings."""

    return_lse: bool = False
    lane_dense: bool = False
    mask_empty: bool = False
    in_kernel: bool = False

    def effective_lse_fill(self, num_chunks: int) -> bool:
        """Whether the kernel owns and initializes the LSE output buffer."""
        return bool(
            self.in_kernel and self.return_lse and self.lane_dense and num_chunks == 1
        )


@dataclasses.dataclass(frozen=True)
class DcpConfig:
    """Decode Context Parallelism (DCP) and fused exchange settings."""

    is_dcp_mode: bool = False
    send_world: int = 0
    send_exchange: SendExchangeConfig | None = None
    out_pieces: int = 0
    out_pieces_gather_first: bool = False

    def __post_init__(self) -> None:
        if self.send_world < 0:
            raise ValueError(f"send_world must be non-negative, got {self.send_world}")
        if self.out_pieces < 0:
            raise ValueError(f"out_pieces must be non-negative, got {self.out_pieces}")
        if self.send_exchange is not None and self.send_world <= 0:
            raise ValueError("send_exchange requires send_world > 0")


def _pop_alias(
    remaining: dict[str, Any], primary: str, alias: str, default: Any
) -> Any:
    has_primary = primary in remaining
    has_alias = alias in remaining
    if has_primary and has_alias:
        raise ValueError(f"cannot specify both {primary!r} and {alias!r}")
    if has_primary:
        return remaining.pop(primary)
    if has_alias:
        return remaining.pop(alias)
    return default


@dataclasses.dataclass(frozen=True)
class SparseMlaConfig:
    """Top-level configuration for ``sparse_ragged_paged_attention``."""

    sm_scale: float = 1.0
    k_scale: float = 1.0
    attention_kernel_batch_size: int = DEFAULT_ATTENTION_KERNEL_BATCH_SIZE
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES
    gather: GatherConfig = dataclasses.field(default_factory=GatherConfig)
    capped: CappedWindowConfig = dataclasses.field(default_factory=CappedWindowConfig)
    lse: LseConfig = dataclasses.field(default_factory=LseConfig)
    dcp: DcpConfig = dataclasses.field(default_factory=DcpConfig)

    def __post_init__(self) -> None:
        if self.attention_kernel_batch_size <= 0:
            raise ValueError(
                "attention_kernel_batch_size must be positive, "
                f"got {self.attention_kernel_batch_size}"
            )
        if self.vmem_limit_bytes <= 0:
            raise ValueError(
                f"vmem_limit_bytes must be positive, got {self.vmem_limit_bytes}"
            )

    @classmethod
    def from_legacy_kwargs(cls, **kwargs: Any) -> SparseMlaConfig:
        """Builds a ``SparseMlaConfig`` from legacy entry-point keyword args.

        Supports both ``sparse_ragged_paged_attention`` kwargs and the
        ``sparse_attention_chunk`` aliases (``group_widths``, ``row_widths``,
        ``group_levers``, ``row_levers``, ``lse_fill``, ``chunk_size``).
        """
        rem = dict(kwargs)
        sm_scale = rem.pop("sm_scale", 1.0)
        k_scale = rem.pop("k_scale", 1.0)
        attention_kernel_batch_size = rem.pop(
            "attention_kernel_batch_size", DEFAULT_ATTENTION_KERNEL_BATCH_SIZE
        )
        vmem_limit_bytes = rem.pop("vmem_limit_bytes", DEFAULT_VMEM_LIMIT_BYTES)

        gather = GatherConfig(
            cache_layout=rem.pop("cache_layout", TENSORCORE_CACHE_LAYOUT),
            packed_rope=rem.pop("packed_rope", False),
            chunk_size=_pop_alias(
                rem,
                "gather_and_attention_chunk_size",
                "chunk_size",
                DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE,
            ),
            sparsecore_atoms_per_batch=rem.pop("sparsecore_atoms_per_batch", 16),
            sparsecore_wait_mode=rem.pop(
                "sparsecore_gather_wait_mode", GatherWaitMode.OVERLAP_ROPE
            ),
            sparsecore_pipelined=rem.pop("sparsecore_gather_pipelined", True),
            small_chunk_tokens=rem.pop("small_chunk_tokens", 16),
            small_chunk_batch_size=rem.pop("small_chunk_batch_size", 1),
            small_chunk_live_gather=rem.pop("small_chunk_live_gather", True),
            small_chunk_gather_atoms=rem.pop("small_chunk_gather_atoms", 1),
            live_gather_all_chunks=rem.pop("live_gather_all_chunks", True),
        )

        raw_mixed_body = _pop_alias(rem, "mixed_body_levers", "group_levers", ())
        raw_decode_body = _pop_alias(rem, "decode_body_levers", "row_levers", ())
        capped = CappedWindowConfig(
            topk_cap=rem.pop("topk_cap", None),
            capped_live_rows=rem.pop("capped_live_rows", False),
            fused_window_prep=rem.pop("fused_window_prep", False),
            mixed_live_gather=rem.pop("mixed_live_gather", False),
            mixed_fused_prep=rem.pop("mixed_fused_prep", False),
            mixed_live_widths=tuple(
                _pop_alias(rem, "mixed_live_widths", "group_widths", ())
            ),
            decode_row_widths=tuple(
                _pop_alias(rem, "decode_row_widths", "row_widths", ())
            ),
            mixed_body=BodyOptimizationConfig.from_levers(raw_mixed_body, mode="mixed"),
            decode_body=BodyOptimizationConfig.from_levers(
                raw_decode_body, mode="decode"
            ),
        )

        lse = LseConfig(
            return_lse=rem.pop("return_lse", False),
            lane_dense=rem.pop("lse_lane_dense", False),
            mask_empty=rem.pop("lse_mask_empty", False),
            in_kernel=_pop_alias(rem, "lse_in_kernel", "lse_fill", False),
        )

        dcp = DcpConfig(
            is_dcp_mode=rem.pop("is_dcp_mode", False),
            send_world=rem.pop("send_world", 0),
            send_exchange=SendExchangeConfig.from_raw(rem.pop("send_exchange", None)),
            out_pieces=rem.pop("out_pieces", 0),
            out_pieces_gather_first=rem.pop("out_pieces_gather_first", False),
        )

        if rem:
            unexpected = ", ".join(sorted(rem))
            raise TypeError(f"unexpected keyword arguments: {unexpected}")

        return cls(
            sm_scale=sm_scale,
            k_scale=k_scale,
            attention_kernel_batch_size=attention_kernel_batch_size,
            vmem_limit_bytes=vmem_limit_bytes,
            gather=gather,
            capped=capped,
            lse=lse,
            dcp=dcp,
        )

    def validate_caches(
        self, cache_kv_nope: jax.Array, cache_kv_rope: jax.Array
    ) -> int:
        """Validates KV cache dtypes/shapes and returns ``page_size``."""
        rope_tokens_per_row = self.gather.rope_tokens_per_row
        if self.gather.is_sparsecore:
            if (
                cache_kv_nope.dtype != jnp.uint32
                or cache_kv_nope.ndim != 3
                or cache_kv_nope.shape[-1] != 128
            ):
                raise ValueError(
                    "sparsecore NOPE cache must be uint32[pages,page_size,128]"
                )
            if (
                cache_kv_rope.dtype != jnp.uint32
                or cache_kv_rope.ndim != 3
                or cache_kv_rope.shape[-1] != 128
            ):
                raise ValueError(
                    "sparsecore ROPE cache must be uint32"
                    f"[pages,page_size/{rope_tokens_per_row},128]"
                )
        else:
            if self.gather.packed_rope:
                raise ValueError("packed_rope needs the sparsecore cache layout")
            if (
                cache_kv_nope.dtype != jnp.uint8
                or cache_kv_nope.ndim != 4
                or cache_kv_nope.shape[-2:] != (4, 128)
            ):
                raise ValueError(
                    "tensorcore NOPE cache must be uint8[pages,page_size,4,128]"
                )
            if (
                cache_kv_rope.dtype != jnp.uint8
                or cache_kv_rope.ndim != 4
                or cache_kv_rope.shape[-2:] != (4, 128)
            ):
                raise ValueError(
                    "tensorcore ROPE cache must be uint8[pages,page_size/4,4,128]"
                )
        if cache_kv_nope.size != rope_tokens_per_row * cache_kv_rope.size:
            raise ValueError(
                f"NOPE cache size ({cache_kv_nope.size}) must be exactly "
                f"{rope_tokens_per_row} * ROPE cache size ({cache_kv_rope.size})"
            )
        if (
            cache_kv_nope.shape[0] != cache_kv_rope.shape[0]
            or cache_kv_nope.shape[1] != rope_tokens_per_row * cache_kv_rope.shape[1]
        ):
            raise ValueError(
                "NOPE and ROPE caches must have matching page counts and page geometry"
            )
        return cache_kv_nope.shape[1]

    def validate_queries(
        self,
        q: jax.Array,
        q_pe: jax.Array | None,
        q_rope_table: jax.Array | None,
        topk_indices: jax.Array,
    ) -> None:
        """Validates query, split-q/rope-table, and top-k index arrays."""
        _, actual_num_q_heads, actual_head_dim = q.shape
        if q_rope_table is not None:
            if q_pe is None or q_pe.shape[-1] != Q_ROPE_HEAD_DIM:
                raise ValueError(
                    "q_rope_table needs the raw q head view as q_pe, "
                    f"[tokens, heads, {Q_ROPE_HEAD_DIM}]"
                )
            if (
                q_rope_table.shape != (q.shape[0], 2, 128)
                or q_rope_table.dtype != jnp.float32
            ):
                raise ValueError(
                    f"q_rope_table {q_rope_table.shape}/{q_rope_table.dtype} must be "
                    f"f32[{q.shape[0]}, 2, 128]"
                )
        if q_pe is not None:
            if q_pe.shape[:2] != q.shape[:2] or q_pe.dtype != q.dtype:
                raise ValueError(
                    f"q_pe {q_pe.shape}/{q_pe.dtype} must match q_nope "
                    f"{q.shape}/{q.dtype} in tokens, heads and dtype"
                )
            if actual_head_dim % 128 or actual_num_q_heads % get_dtype_packing(q.dtype):
                raise ValueError(
                    "split q needs a 128-multiple nope dim and packed heads, "
                    f"got {q.shape}"
                )
        if topk_indices.dtype != jnp.int32 or topk_indices.ndim != 2:
            raise ValueError("topk_indices must be int32[num_tokens, topk]")
        if q.shape[0] != topk_indices.shape[0]:
            raise ValueError(
                f"q tokens ({q.shape[0]}) must match topk_indices tokens "
                f"({topk_indices.shape[0]})"
            )
        topk = topk_indices.shape[-1]
        if self.gather.is_sparsecore and topk % native_sc_gather.ROPE_PACKING != 0:
            raise ValueError(
                f"topk ({topk}) must be a multiple of "
                f"{native_sc_gather.ROPE_PACKING} for sparsecore ROPE packing"
            )


def resolve_entry_config(
    config: SparseMlaConfig | None,
    legacy_kwargs: dict[str, Any],
    *,
    defaults: dict[str, Any] | None = None,
    required: tuple[str, ...] = (),
) -> SparseMlaConfig:
    """Resolves ``config`` vs legacy kwargs, failing fast on mixed usage."""
    provided = {k: v for k, v in legacy_kwargs.items() if v is not UNSET}
    if config is not None:
        if provided:
            conflicts = ", ".join(sorted(provided))
            raise ValueError(
                f"cannot pass both 'config' and legacy config kwargs: {conflicts}"
            )
        return config
    missing = [name for name in required if name not in provided]
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise TypeError(
            f"missing required keyword-only argument(s) when 'config' is None: {names}"
        )
    merged = {**(defaults or {}), **provided}
    return SparseMlaConfig.from_legacy_kwargs(**merged)


@dataclasses.dataclass(frozen=True)
class AttentionKernelConfig:
    """Static configuration for one Pallas ``_attention_kernel`` invocation."""

    sm_scale: float
    k_scale: float = 1.0
    batch_size: int = 1
    preserve_suffix: bool = False
    return_lse: bool = False
    sparsecore_layout: bool = False
    capped_windows: int | None = None
    is_dcp_mode: bool = False
    split_q: bool = False
    q_rope: bool = False
    lse_lane_dense: bool = False
    lse_mask_empty: bool = False
    lse_fill: bool = False
    has_group_tab: bool = False
    live_widths: tuple[int, ...] = ()
    row_widths: tuple[int, ...] = ()
    body: BodyOptimizationConfig = dataclasses.field(
        default_factory=BodyOptimizationConfig
    )
    send_world: int = 0
    out_row_offset: int = 0
    out_in: bool = False
    send_exchange: SendExchangeConfig | None = None
    recv_in: bool = False

    @property
    def send_enabled(self) -> bool:
        """Whether the kernel emits the packed DCP send layout."""
        return self.send_world > 0

    @property
    def xchg_enabled(self) -> bool:
        """Whether the kernel runs the fused DCP small all-to-all exchange."""
        return self.send_exchange is not None

    @property
    def xchg_merge(self) -> bool:
        """Whether fused exchange merges received slabs in VMEM."""
        return bool(self.send_exchange and self.send_exchange.in_kernel_merge)

    @property
    def slab_row0(self) -> int:
        """First partial-output row inside each destination's VMEM slab."""
        return SEND_SLAB_ROW0 if self.xchg_enabled else 0

    def _validate_lse_and_output_invariants(self) -> None:
        if self.preserve_suffix and (self.return_lse or self.sparsecore_layout):
            raise ValueError("preserve_suffix requires TensorCore layout without LSE")
        if self.lse_fill and not (self.return_lse and self.lse_lane_dense):
            raise ValueError("lse_fill requires return_lse and lse_lane_dense")
        if self.out_in and not (self.return_lse and not self.lse_fill):
            raise ValueError("out_in requires return_lse and lse_fill=False")

    def _validate_exchange_invariants(self) -> None:
        if self.xchg_enabled and not self.send_enabled:
            raise ValueError("send_exchange requires send_world > 0")
        if self.recv_in and not self.xchg_enabled:
            raise ValueError("recv_in requires send_exchange")
        if self.xchg_merge and self.recv_in:
            raise ValueError("in_kernel_merge cannot be combined with recv_in")
        if self.send_enabled and not (
            self.lse_fill and self.split_q and not self.preserve_suffix
        ):
            raise ValueError(
                "send_world requires lse_fill, split_q, and preserve_suffix=False"
            )

    def _validate_body_invariants(self) -> None:
        if self.body.has_any_lever:
            if not self.sparsecore_layout:
                raise ValueError("body optimizations require sparsecore_layout")
            if not (self.has_group_tab or self.row_widths):
                raise ValueError(
                    "body optimizations require has_group_tab or row_widths"
                )
            self.body.validate_for_mode("mixed" if self.has_group_tab else "decode")
        if self.body.narrow_q_rope_dma and not self.q_rope:
            raise ValueError("narrow_q_rope_dma ('q128') requires q_rope")
        if (
            self.body.strategy == BodyExecutionStrategy.PAIR
            and self.batch_size % 2 != 0
        ):
            raise ValueError(
                f"PAIR body strategy requires even batch_size, got {self.batch_size}"
            )

    def _validate_window_and_body_invariants(self) -> None:
        if self.has_group_tab and self.capped_windows is None:
            raise ValueError("has_group_tab requires capped_windows")
        if self.live_widths and not self.has_group_tab:
            raise ValueError("live_widths requires has_group_tab")
        if self.row_widths and self.has_group_tab:
            raise ValueError("row_widths and has_group_tab are mutually exclusive")
        if self.row_widths and (
            self.capped_windows is None or not self.sparsecore_layout
        ):
            raise ValueError("row_widths requires capped_windows and sparsecore_layout")
        if self.capped_windows is not None:
            if not self.sparsecore_layout:
                raise ValueError("capped_windows requires sparsecore_layout")
            if self.capped_windows <= 0:
                raise ValueError(
                    f"capped_windows must be positive, got {self.capped_windows}"
                )
        self._validate_body_invariants()

    def __post_init__(self) -> None:
        object.__setattr__(self, "live_widths", tuple(self.live_widths))
        object.__setattr__(self, "row_widths", tuple(self.row_widths))
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.send_world < 0:
            raise ValueError(f"send_world must be non-negative, got {self.send_world}")
        if self.out_row_offset < 0:
            raise ValueError(
                f"out_row_offset must be non-negative, got {self.out_row_offset}"
            )
        if self.q_rope and not self.split_q:
            raise ValueError("q_rope requires split_q")
        self._validate_lse_and_output_invariants()
        self._validate_exchange_invariants()
        self._validate_window_and_body_invariants()
