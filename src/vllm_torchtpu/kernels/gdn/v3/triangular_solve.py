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
"""Unit lower-triangular solvers for chunked GDN."""

import dataclasses
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v3 import fp8_matmul

# Base diagonal block inverted on the vector unit (VPU, 8 rows) and target block
# size reached via matrix-unit (MXU) Schur block doubling (adapting Bao et al.,
# FlyWheel, https://arxiv.org/abs/2611.01426).
BASE_BLOCK_SIZE = 8
TARGET_BLOCK_SIZE = 64

# Fixed power-of-two scales keeping bounded coupling entries L and block
# inverses M inside the float8_e4m3fn normal range without per-element divides:
#   S_L scales strictly lower coupling L before casting to FP8/BF16,
#   S_D scales diagonal block inverses M and solved outputs u,
#   LY_INV = 1 / (S_L * S_D) removes both factors after (S_L * L) @ (S_D * ...).
S_L = 32.0
S_D = 8.0
LY_INV = 1.0 / (S_L * S_D)

_MM_DIMS = (((1,), (0,)), ((), ()))
_SUPPORTED_FP8_DTYPES = frozenset(
    {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }
)


def supports_block_doubling_chunk(
    chunk: int, target_block_size: int = TARGET_BLOCK_SIZE
) -> bool:
    """Returns whether `BlockDoublingSolver` supports `chunk`."""
    if chunk < 32 or chunk % 32 or (chunk > 128 and chunk % 128):
        return False
    if target_block_size < 16 or (target_block_size & (target_block_size - 1)):
        return False
    tb = min(target_block_size, chunk)
    return chunk % tb == 0


def solve_triangular_rhs(
    strict_lower: jax.Array, rhs: jax.Array, block_size: int
) -> jax.Array:
    """Solves the unit lower-triangular system (I + L) @ x = rhs in blocks.

    Args:
        strict_lower: L, the strictly lower-triangular part of the system
            matrix, [num_heads, chunk, chunk]. Its diagonal and upper triangle
            must be zero; the unit diagonal is implicit.
        rhs: Right-hand side [num_heads, chunk, n].
        block_size: Diagonal block size B. Blocks are solved in order; each
            block first removes the coupling to the already-solved rows with
            one matmul and then runs a dense in-block forward substitution.

    Returns:
        x with the dtype of `strict_lower`.
    """
    out_dtype = strict_lower.dtype
    chunk = strict_lower.shape[-1]
    block_size = min(block_size, chunk)
    num_blocks = (chunk + block_size - 1) // block_size
    rhs_f32 = rhs.astype(jnp.float32)

    def local_forward_sub(l_block: jax.Array, b_block: jax.Array) -> jax.Array:
        # Column-oriented (right-looking) substitution over the whole dense
        # [num_heads, B, n] block: after step j, row j is final and the rows
        # below it have column j's contribution removed. Rows <= j are
        # untouched because their coefficient in the strictly lower block is
        # zero, so each step is one multiply-subtract over B rows held densely
        # in sublanes. A row-by-row chain costs one vreg op per head per
        # (row, column) pair instead, which made the solve VALU-bound at 8-
        # and 16-row blocks (few V heads per core).
        x = b_block
        for j in range(l_block.shape[1] - 1):
            x = x - l_block[:, :, j : j + 1] * x[:, j : j + 1, :]
        return x

    x_blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, chunk)
        target_b = rhs_f32[:, start:end, :]
        if i > 0:
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jax.lax.dot(
                strict_lower[:, start:end, :start],
                solved_x,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            target_b = target_b - prev_sum

        l_block = strict_lower[:, start:end, start:end].astype(jnp.float32)
        x_blocks.append(local_forward_sub(l_block, target_b).astype(out_dtype))

    return jnp.concatenate(x_blocks, axis=1)


@dataclasses.dataclass(frozen=True)
class _PackedBitMask:
    """Packed uint8/uint16 bitmask for fast bitwise-AND zeroing.

    Attributes:
      u_mask: All-ones/all-zeros uint8 or uint16 bitmask for 1- or 2-byte
        dtypes.
    """

    u_mask: jax.Array

    def apply(self, x: jax.Array) -> jax.Array:
        """Zeros false positions of `x` via bitwise AND."""
        u = jax.lax.bitcast_convert_type(x, self.u_mask.dtype)
        return jax.lax.bitcast_convert_type(jnp.bitwise_and(u, self.u_mask), x.dtype)


def _pack_bitmask(mask: jax.Array, dtype: jnp.dtype) -> _PackedBitMask:
    """Builds a `_PackedBitMask` for 1- or 2-byte `dtype`."""
    itemsize = jnp.dtype(dtype).itemsize
    if itemsize == 1:
        u_mask = jnp.where(mask, jnp.uint8(0xFF), jnp.uint8(0))
    elif itemsize == 2:
        u_mask = jnp.where(mask, jnp.uint16(0xFFFF), jnp.uint16(0))
    else:
        raise ValueError(
            f"_pack_bitmask requires a 1- or 2-byte dtype; got {jnp.dtype(dtype)}."
        )
    return _PackedBitMask(u_mask=u_mask)


def _interleave_odd_blocks(
    x: jax.Array, zero_b: jax.Array, b: int, m: int
) -> jax.Array:
    """Interleaves `m` zero `[b, N]` blocks ahead of the `m` strips of `x`."""
    # Aligns odd-row coupling strips with odd columns of M_odd in [chunk, chunk].
    if m == 1:
        return jnp.concat([zero_b, x], axis=0)
    parts = []
    for p in range(m):
        parts.append(zero_b)
        parts.append(x[p * b : (p + 1) * b])
    return jnp.concat(parts, axis=0)


def _residual_high_low(
    rhs_bf16: jax.Array, ly_raw: jax.Array, fp8_dtype: jnp.dtype | None
) -> list[jax.Array]:
    """Subtracts scaled lower-block coupling and splits into high/low operands."""
    ly_s = ly_raw * LY_INV
    # FP8 mode subtracts in bfloat16 to cut float32 vector register pressure;
    # BF16 mode keeps float32 subtraction before the final cast.
    if fp8_dtype is not None:
        return fp8_matmul.split_high_low(
            rhs_bf16 - ly_s.astype(jnp.bfloat16), fp8_dtype, truncate_e4m3=True
        )
    return fp8_matmul.split_high_low(
        rhs_bf16.astype(jnp.float32) - ly_s, fp8_dtype, truncate_e4m3=True
    )


def _pad_block_rows(
    y_limbs: list[jax.Array], bi: int, num_tb: int, tb: int
) -> list[jax.Array]:
    """Zero-pads block `bi` into `[num_tb * tb, v_dim]` for a full-width dot."""
    dtype = y_limbs[0].dtype
    v_dim = y_limbs[0].shape[1]
    z_head = jnp.zeros((bi * tb, v_dim), dtype=dtype) if bi > 0 else None
    z_tail = (
        jnp.zeros(((num_tb - 1 - bi) * tb, v_dim), dtype=dtype)
        if bi + 1 < num_tb
        else None
    )
    return [
        jnp.concat([p for p in (z_head, yl, z_tail) if p is not None], axis=0)
        for yl in y_limbs
    ]


class DoubledSolveState:
    """Per-head solved `S_D * u` state with deferred final-block splitting.

    Holding the final block as float32 until `pop_u_limbs` is called keeps fewer
    8-bit arrays live in on-chip memory (VMEM) during the per-head tail loop.
    """

    def __init__(
        self,
        fp8_dtype: jnp.dtype | None,
        last_du_map: dict[int, jax.Array],
        prefix_u_map: dict[int, list[list[jax.Array]]] | None = None,
    ):
        """Stores solved prefix blocks and the unsplit final block per head.

        Args:
          fp8_dtype: 8-bit FP8 operand dtype, or `None` for 16-bit bfloat16 mode.
          last_du_map: Per-head final-block `S_D * u` values in float32.
          prefix_u_map: Optional per-head already-split `[high, low]` prefix blocks
            preceding the final block.
        """
        self._fp8_dtype = fp8_dtype
        self._last_du_map = last_du_map
        self._prefix_u_map = prefix_u_map

    def pop_u_limbs(self, v: int) -> list[jax.Array]:
        """Pops and returns the `[chunk, v_dim]` high/low operands of `S_D * u`."""
        last_u = fp8_matmul.split_high_low(
            self._last_du_map.pop(v), self._fp8_dtype, truncate_e4m3=True
        )
        if self._prefix_u_map is None:
            return last_u
        parts = self._prefix_u_map.pop(v) + [last_u]
        return [jnp.concat([ub[t] for ub in parts], axis=0) for t in range(len(last_u))]


class BlockDoublingSolver:
    """Stage-major matrix-unit Schur block-doubling triangular solver.

    Inverts 8-row diagonal blocks on the vector unit (VPU), doubles them to
    `min(target_block_size, chunk)` rows on the matrix unit (MXU) via Schur
    complements, and solves `(I + L) @ u = rhs` across a group of heads.

    Attributes:
      chunk: Chunk length in tokens.
      tb: Effective doubled diagonal block size (`min(target_block_size, chunk)`).
      fp8_dtype: 8-bit FP8 operand dtype, or `None` for bfloat16 mode.
      lm_dtype: Storage dtype for `L` and doubled diagonal inverses (`fp8_dtype`
        or `bfloat16`).
    """

    def __init__(
        self,
        chunk: int,
        fp8_dtype: jnp.dtype | None,
        target_block_size: int = TARGET_BLOCK_SIZE,
    ):
        """Initializes block-size geometry and 8-row diagonal-inverse constants.

        Args:
          chunk: Chunk length in tokens (must be >= 32, divisible by 32, divisible
            by 128 when > 128, and divisible by `min(target_block_size, chunk)`).
          fp8_dtype: 8-bit FP8 operand dtype, or `None` for 16-bit bfloat16 mode.
          target_block_size: Maximum diagonal block size reached by doubling (must
            be a power of two >= 16).
        """
        if chunk < 32 or chunk % 32 or (chunk > 128 and chunk % 128):
            raise ValueError(
                "Block-doubling GDN requires chunk_size >= 32, divisible by 32, and"
                f" divisible by 128 when > 128; got chunk_size={chunk}."
            )
        if target_block_size < 16 or (target_block_size & (target_block_size - 1)):
            raise ValueError(
                "target_block_size must be a power of two >= 16; got"
                f" target_block_size={target_block_size}."
            )
        tb = min(target_block_size, chunk)
        if chunk % tb:
            raise ValueError(
                f"chunk_size={chunk} must be divisible by effective target block size"
                f" {tb}."
            )
        if fp8_dtype is not None:
            fp8_dtype = jnp.dtype(fp8_dtype)
            if fp8_dtype not in _SUPPORTED_FP8_DTYPES:
                raise ValueError(
                    "fp8_dtype must be None, float8_e4m3fn, or float8_e5m2; got"
                    f" {fp8_dtype}."
                )
        self.chunk = chunk
        self.tb = tb
        self.fp8_dtype = fp8_dtype
        self.lm_dtype = fp8_dtype if fp8_dtype is not None else jnp.dtype(jnp.bfloat16)
        self._iota_r = jax.lax.broadcasted_iota(jnp.int32, (chunk, chunk), 0)
        self._iota_c = jax.lax.broadcasted_iota(jnp.int32, (chunk, chunk), 1)

        # Pack strip // 8 independent 8x8 diagonal blocks side-by-side into
        # [8, strip] (strip <= 128 lanes) so _gather_idx[j] broadcasts column j
        # of each block across its 8 lanes in parallel.
        strip = min(chunk, 128)
        self._strip = strip
        shape = (BASE_BLOCK_SIZE, strip)
        lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        rows = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
        lane_block = lanes // BASE_BLOCK_SIZE
        self._block_masks = tuple(
            lane_block == i for i in range(strip // BASE_BLOCK_SIZE)
        )
        self._identity = jnp.where(
            lanes % BASE_BLOCK_SIZE == rows, jnp.float32(S_D), jnp.float32(0.0)
        )
        self._gather_idx = tuple(
            lane_block * BASE_BLOCK_SIZE + j for j in range(BASE_BLOCK_SIZE - 1)
        )

    def _stage_sizes(self) -> list[int]:
        sizes = []
        b = BASE_BLOCK_SIZE
        while b < self.tb:
            sizes.append(b)
            b *= 2
        return sizes

    def _pack_mask(self, mask: jax.Array) -> _PackedBitMask:
        return _pack_bitmask(mask, self.lm_dtype)

    def _select_block_diag_strip(self, strict_lower: jax.Array) -> jax.Array:
        coef = strict_lower[:BASE_BLOCK_SIZE, :]
        for i in range(1, len(self._block_masks)):
            coef = jnp.where(
                self._block_masks[i],
                strict_lower[i * BASE_BLOCK_SIZE : (i + 1) * BASE_BLOCK_SIZE, :],
                coef,
            )
        return coef

    def select_block_diag(self, mat: jax.Array) -> jax.Array:
        """Extracts side-by-side `[8, chunk]` diagonal blocks across all strips."""
        if self.chunk == self._strip:
            return self._select_block_diag_strip(mat)
        return jnp.concat(
            [
                self._select_block_diag_strip(
                    mat[s : s + self._strip, s : s + self._strip]
                )
                for s in range(0, self.chunk, self._strip)
            ],
            axis=1,
        )

    def _invert_strip(self, coef: jax.Array) -> jax.Array:
        # Forward substitution on all (strip // 8) 8-row diagonal blocks in parallel
        # across TPU lanes, starting from S_D * I so the output is S_D * (I + L)^-1.
        x = self._identity
        for j in range(BASE_BLOCK_SIZE - 1):
            coef_j = jnp.take_along_axis(coef, self._gather_idx[j], axis=1)
            x = x - coef_j * x[j : j + 1, :]
        return x

    def invert_base_blocks(self, coef_full: jax.Array) -> jax.Array:
        """Returns `S_D * M_8` (`[8, chunk]`) from pre-selected diagonal blocks."""
        if self.chunk == self._strip:
            return self._invert_strip(coef_full)
        return jnp.concat(
            [
                self._invert_strip(coef_full[:, s : s + self._strip])
                for s in range(0, self.chunk, self._strip)
            ],
            axis=1,
        )

    def slice_coupling(
        self, l_s: jax.Array
    ) -> tuple[dict[int, jax.Array], jax.Array | None]:
        """Pre-slices scaled `S_L * L` (`bfloat16`) for doubling and block solve.

        Args:
          l_s: Scaled strictly lower-triangular matrix `S_L * L` `[chunk, chunk]` in
            bfloat16.

        Returns:
          `(cs_map, lw)` in `self.lm_dtype`, where `cs_map[b]` holds the
          off-diagonal coupling strips for doubling stage `b` and `lw` holds the
          lower-block coupling matrix for `solve_blocks` (or `None` if
          `chunk == tb`).
        """
        chunk, tb, lm_dtype = self.chunk, self.tb, self.lm_dtype
        cs_map: dict[int, jax.Array] = {}
        lw: jax.Array | None = None
        hb = tb // 2
        if chunk == 2 * tb and hb == 32:
            # Single conversion feeds both the 32 -> 64 doubling strip and the
            # lower-left [64, 64] block of L used in the 2-block solve.
            ltail = l_s[hb:, :].astype(lm_dtype)
            cs_map[hb] = jnp.concat([ltail[:hb, :], ltail[2 * hb :, :]], axis=0)
            lw = ltail[hb:, :tb]
        for b in self._stage_sizes():
            if b not in cs_map:
                m = chunk // (2 * b)
                if m == 1:
                    cs_s = l_s[b : 2 * b, :]
                else:
                    cs_s = jnp.concat(
                        [l_s[(2 * p + 1) * b : (2 * p + 2) * b, :] for p in range(m)],
                        axis=0,
                    )
                cs_map[b] = cs_s.astype(lm_dtype)
        if lw is None and chunk > tb:
            lw_src = l_s[tb:, :tb] if chunk == 2 * tb else l_s[tb:, :]
            lw = lw_src.astype(lm_dtype)
        return cs_map, lw

    def _prepare_d_and_rhs_even(
        self,
        d_v: jax.Array,
        b: int,
        z_b: jax.Array | None,
        rhs_even_mask: _PackedBitMask | None,
        rhs_strip_masks: tuple[_PackedBitMask, ...] | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Converts `d_v` to `lm_dtype` and tiles even-block RHS for `p1`."""
        chunk, lm_dtype = self.chunk, self.lm_dtype
        use_fp8 = self.fp8_dtype is not None
        if b < 32:
            assert z_b is not None
            # Stage narrow tiles through bfloat16 before FP8 conversion to avoid extra
            # float32-to-FP8 packing shuffles on sub-32-row blocks.
            if use_fp8 and b == 16:
                d_arr = jnp.concat([d_v.astype(jnp.bfloat16), z_b], axis=0).astype(
                    lm_dtype
                )
            elif use_fp8 and b == 8:
                pair_bf16 = jnp.concat([d_v, z_b], axis=0).astype(jnp.bfloat16)
                d_arr = jnp.concat([pair_bf16] * 2, axis=0).astype(lm_dtype)
            else:
                pair = jnp.concat([d_v, z_b], axis=0)
                tile32 = (
                    pair
                    if 2 * b == 32
                    else jnp.concat([pair] * (32 // (2 * b)), axis=0)
                )
                d_arr = tile32.astype(lm_dtype)
            rep = d_arr if chunk == 32 else jnp.concat([d_arr] * (chunk // 32), axis=0)
            rhs_even = rep if rhs_even_mask is None else rhs_even_mask.apply(rep)
            return d_arr, rhs_even

        d_arr = d_v.astype(lm_dtype)
        zb = jnp.zeros_like(d_arr)
        if rhs_strip_masks is None:
            return d_arr, jnp.concat([d_arr, zb], axis=0)
        parts = []
        for sm in rhs_strip_masks:
            parts.append(sm.apply(d_arr))
            parts.append(zb)
        return d_arr, jnp.concat(parts, axis=0)

    def _prepare_rhs_z(
        self,
        z_raw: jax.Array,
        b: int,
        m: int,
        z_b: jax.Array | None,
        z_rhs_mask: _PackedBitMask | None,
        even_mask: _PackedBitMask | None,
    ) -> jax.Array:
        """Converts and masks `p1` output into the `[chunk, chunk]` RHS for `p2`."""
        lm_dtype = self.lm_dtype
        if z_rhs_mask is not None:
            assert z_b is not None
            z_src = (
                z_raw.astype(jnp.bfloat16)
                if (self.fp8_dtype is not None and b == 16)
                else z_raw
            )
            stacked = _interleave_odd_blocks(z_src, z_b, b, m).astype(lm_dtype)
            return z_rhs_mask.apply(stacked)

        assert even_mask is not None
        z_arr = even_mask.apply(z_raw.astype(lm_dtype))
        zb = jnp.zeros((b, z_raw.shape[1]), dtype=lm_dtype)
        return _interleave_odd_blocks(z_arr, zb, b, m)

    def double_inverses(
        self,
        d_map: Mapping[int, jax.Array],
        cs_map: dict[int, dict[int, jax.Array]],
        heads: Sequence[int],
    ) -> dict[int, jax.Array]:
        """Stage-major Schur block doubling of `S_D * M` from 8 rows up to `tb`.

        Args:
          d_map: Per-head base 8-row block inverses `S_D * M_8` (`[8, chunk]`).
          cs_map: Per-head pre-sliced coupling strips from `slice_coupling`; popped
            per stage to free live buffers early.
          heads: Head indices to process stage-major.

        Returns:
          Per-head doubled block inverse `[tb, chunk]` in `self.lm_dtype`.
        """
        chunk, tb, lm_dtype = self.chunk, self.tb, self.lm_dtype
        iota_r, iota_c = self._iota_r, self._iota_c
        neg_ly_inv = -LY_INV
        cur = dict(d_map)
        d_inv_out: dict[int, jax.Array] = {}
        stages = self._stage_sizes()

        # Each doubling stage merges adjacent b-row diagonal inverses (M_even,
        # M_odd) coupled by lower-left block C into a 2b-row inverse whose
        # off-diagonal block is -M_odd @ C @ M_even. Two MXU dots compute this as:
        #   p1 = (S_L * C) @ (S_D * M_even)
        #   p2 = (S_D * M_odd) @ p1
        # then scaling by -LY_INV = -1 / (S_L * S_D) gives
        # S_D * (-M_odd @ C @ M_even).
        for b in stages:
            m = chunk // (2 * b)
            is_last = 2 * b == tb
            even_col = ((iota_c[:b, :] // b) % 2) == 0

            # Blocks smaller than 32 rows are tiled to 32-row strips and masked in
            # place (rhs_even_mask keeps even diagonal (2p, 2p) blocks; z_rhs_mask
            # keeps subdiagonal (2p + 1, 2p) blocks); 32-row blocks already fill
            # full matrix tiles without tiling.
            if b < 32:
                rb, cb = iota_r // b, iota_c // b
                rhs_even_mask = (
                    self._pack_mask(((rb % 2) == 0) & (cb == rb)) if m > 1 else None
                )
                z_rhs_mask = self._pack_mask(((rb % 2) == 1) & (cb == rb - 1))
                even_mask = None
                rhs_strip_masks = None
                z_dtype = (
                    jnp.bfloat16
                    if (self.fp8_dtype is not None and b == 16)
                    else jnp.float32
                )
                z_b = jnp.zeros((b, chunk), dtype=z_dtype)
            else:
                half_r, half_c = iota_r[: chunk // 2, :], iota_c[: chunk // 2, :]
                rhs_even_mask = None
                z_rhs_mask = None
                even_mask = self._pack_mask((half_c // b) == 2 * (half_r // b))
                if m > 1:
                    rhs_strip_masks = tuple(
                        self._pack_mask((iota_c[:b, :] // b) == 2 * p) for p in range(m)
                    )
                else:
                    rhs_strip_masks = None
                z_b = None

            # Process all heads through p1, then p2, then combine so the compiler can
            # overlap independent matrix-unit dots across heads.
            dl_map: dict[int, jax.Array] = {}
            p1_map: dict[int, jax.Array] = {}
            for v in heads:
                d_arr, rhs_even = self._prepare_d_and_rhs_even(
                    cur[v], b, z_b, rhs_even_mask, rhs_strip_masks
                )
                dl_map[v] = d_arr
                p1_map[v] = fp8_matmul.dot_general(cs_map[v].pop(b), rhs_even, _MM_DIMS)

            p2_map: dict[int, jax.Array] = {}
            for v in heads:
                rhs_z = self._prepare_rhs_z(
                    p1_map.pop(v), b, m, z_b, z_rhs_mask, even_mask
                )
                p2_map[v] = fp8_matmul.dot_general(dl_map.pop(v), rhs_z, _MM_DIMS)

            # On the final stage, mask out the upper-right odd columns after casting
            # to lm_dtype so the final [tb, chunk] inverse is strictly block-lower.
            if is_last:
                if b >= 32:
                    fuse_mask = self._pack_mask(even_col)
                else:
                    fuse_mask = self._pack_mask(
                        (iota_r[:tb, :] >= b) | (((iota_c[:tb, :] // b) % 2) == 0)
                    )
            else:
                fuse_mask = None

            # Assemble 2b-row inverse [[M_even, 0], [-M_odd @ C @ M_even, M_odd]].
            for v in heads:
                d_v = cur[v]
                p2 = p2_map[v][:b, :] if b < 32 else p2_map[v]
                bl_packed = p2 * neg_ly_inv
                if fuse_mask is not None:
                    d_bot = jnp.where(even_col, bl_packed, d_v)
                    raw = jnp.concat([d_v, d_bot], axis=0).astype(lm_dtype)
                    d_inv_out[v] = (
                        jnp.concat([fuse_mask.apply(raw[:b, :]), raw[b:, :]], axis=0)
                        if b >= 32
                        else fuse_mask.apply(raw)
                    )
                    del cur[v]
                else:
                    d_top = jnp.where(even_col, d_v, jnp.float32(0.0))
                    d_bot = jnp.where(even_col, bl_packed, d_v)
                    cur[v] = jnp.concat([d_top, d_bot], axis=0)

        return d_inv_out

    def _solve_single_block(
        self,
        d_inv_map: dict[int, jax.Array],
        rhs_bf16_map: dict[int, jax.Array],
        heads: Sequence[int],
    ) -> DoubledSolveState:
        """Applies the full `[chunk, chunk]` inverse in one matmul per head."""
        fp8_dtype = self.fp8_dtype
        single_du_map = {
            v: fp8_matmul.dot_general(
                d_inv_map.pop(v),
                fp8_matmul.split_high_low(
                    rhs_bf16_map.pop(v), fp8_dtype, truncate_e4m3=True
                ),
                _MM_DIMS,
            )
            for v in heads
        }
        return DoubledSolveState(fp8_dtype, single_du_map)

    def _solve_two_blocks(
        self,
        d_inv_map: dict[int, jax.Array],
        lw_map: dict[int, jax.Array | None],
        rhs_bf16_map: dict[int, jax.Array],
        heads: Sequence[int],
        *,
        num_heads: int,
    ) -> DoubledSolveState:
        """Solves two `[tb, tb]` diagonal blocks with one coupling subtraction."""
        tb, fp8_dtype = self.tb, self.fp8_dtype
        du0_map: dict[int, jax.Array] = {}
        rhs_tail_map: dict[int, jax.Array] = {}
        for v in heads:
            vd = rhs_bf16_map.pop(v)
            v0, rhs_tail_map[v] = vd[:tb, :], vd[tb:, :]
            du0_map[v] = fp8_matmul.dot_general(
                d_inv_map[v][:, :tb],
                fp8_matmul.split_high_low(v0, fp8_dtype, truncate_e4m3=True),
                _MM_DIMS,
            )
        prefix_u_map: dict[int, list[list[jax.Array]]] = {}
        ly0_map: dict[int, jax.Array] = {}
        for v in heads:
            u0_l = fp8_matmul.split_high_low(
                du0_map.pop(v), fp8_dtype, truncate_e4m3=True
            )
            prefix_u_map[v] = [u0_l]
            lw = lw_map.pop(v)
            assert lw is not None
            ly0_map[v] = fp8_matmul.dot_general(lw, u0_l, _MM_DIMS)
        du1_map: dict[int, jax.Array] = {}
        # Tuned on TPU v7x: slicing d_inv[:, tb:] halves matmul width at small
        # head counts (<= 16) and avoids padding registers at high BF16 head
        # counts (>= 64), whereas mid head counts (17..63) and high-head FP8
        # run faster by zero-padding y1 to full chunk width to avoid sub-tile
        # column slicing on the matrix-unit path.
        slice_d1 = num_heads <= 16 or (fp8_dtype is None and num_heads >= 64)
        for v in heads:
            y1_l = _residual_high_low(rhs_tail_map.pop(v), ly0_map.pop(v), fp8_dtype)
            d_inv = d_inv_map.pop(v)
            if slice_d1:
                du1_map[v] = fp8_matmul.dot_general(d_inv[:, tb:], y1_l, _MM_DIMS)
            else:
                du1_map[v] = fp8_matmul.dot_general(
                    d_inv, _pad_block_rows(y1_l, 1, 2, tb), _MM_DIMS
                )
        return DoubledSolveState(fp8_dtype, du1_map, prefix_u_map)

    def _solve_multi_block(
        self,
        d_inv_map: dict[int, jax.Array],
        lw_map: dict[int, jax.Array | None],
        rhs_bf16_map: dict[int, jax.Array],
        heads: Sequence[int],
        *,
        num_tb: int,
        v_dim: int,
    ) -> DoubledSolveState:
        """Steps block-by-block when `chunk > 2 * tb`, subtracting prior coupling."""
        tb, fp8_dtype = self.tb, self.fp8_dtype
        prefix_u_map: dict[int, list[list[jax.Array]]] = {v: [] for v in heads}
        rhs_rem = {v: rhs_bf16_map.pop(v) for v in heads}
        last_du_map: dict[int, jax.Array] = {}
        for bi in range(num_tb):
            ly_map: dict[int, jax.Array] = {}
            if bi > 0:
                for v in heads:
                    u_prev = prefix_u_map[v]
                    z_tail = jnp.zeros(
                        ((num_tb - bi) * tb, v_dim), dtype=u_prev[0][0].dtype
                    )
                    rhs_u = [
                        jnp.concat([ub[t] for ub in u_prev] + [z_tail], axis=0)
                        for t in range(len(u_prev[0]))
                    ]
                    lw = lw_map[v]
                    assert lw is not None
                    ly_map[v] = fp8_matmul.dot_general(
                        lw[(bi - 1) * tb : bi * tb, :], rhs_u, _MM_DIMS
                    )
            du_map: dict[int, jax.Array] = {}
            for v in heads:
                vd = rhs_rem.pop(v)
                y_cur = vd[:tb, :]
                if bi + 1 < num_tb:
                    rhs_rem[v] = vd[tb:, :]
                if bi > 0:
                    y_l = _residual_high_low(y_cur, ly_map.pop(v), fp8_dtype)
                else:
                    y_l = fp8_matmul.split_high_low(
                        y_cur, fp8_dtype, truncate_e4m3=True
                    )
                du_map[v] = fp8_matmul.dot_general(
                    d_inv_map[v], _pad_block_rows(y_l, bi, num_tb, tb), _MM_DIMS
                )
            if bi + 1 == num_tb:
                last_du_map = du_map
                for v in heads:
                    d_inv_map.pop(v)
                    lw_map.pop(v)
            else:
                for v in heads:
                    prefix_u_map[v].append(
                        fp8_matmul.split_high_low(
                            du_map[v], fp8_dtype, truncate_e4m3=True
                        )
                    )
        return DoubledSolveState(fp8_dtype, last_du_map, prefix_u_map)

    def solve_blocks(
        self,
        d_inv_map: dict[int, jax.Array],
        lw_map: dict[int, jax.Array | None],
        rhs_bf16_map: dict[int, jax.Array],
        heads: Sequence[int],
        *,
        num_heads: int,
        v_dim: int,
    ) -> DoubledSolveState:
        """Solves `(I + L) @ u = rhs` across `heads` using doubled block inverses.

        Consumes entries from `d_inv_map`, `lw_map`, and `rhs_bf16_map` as stages
        finish to keep live VMEM buffers minimal.

        Args:
          d_inv_map: Per-head doubled block inverse `[tb, chunk]` from
            `double_inverses`.
          lw_map: Per-head lower-block coupling matrix from `slice_coupling`.
          rhs_bf16_map: Per-head `[chunk, v_dim]` bfloat16 right-hand sides.
          heads: Head indices to solve stage-major.
          num_heads: Total value heads per core (selects slicing vs zero-padding on
            the 2-block second-half multiply).
          v_dim: Right-hand side column dimension (`v_head_dim`).

        Returns:
          `DoubledSolveState` yielding per-head `S_D * u` high/low operands via
          `pop_u_limbs(v)`.
        """
        num_tb = self.chunk // self.tb
        if num_tb == 1:
            return self._solve_single_block(d_inv_map, rhs_bf16_map, heads)
        if num_tb == 2:
            return self._solve_two_blocks(
                d_inv_map, lw_map, rhs_bf16_map, heads, num_heads=num_heads
            )
        return self._solve_multi_block(
            d_inv_map, lw_map, rhs_bf16_map, heads, num_tb=num_tb, v_dim=v_dim
        )
