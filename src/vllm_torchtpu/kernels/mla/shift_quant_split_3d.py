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
"""Zero-transpose 3D head-major shift, FP8 row quantization, and pair-half split."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8_MAX = 448.0
SCALE_PAD_COLS = 128
DEFAULT_VMEM_LIMIT_BYTES = 12 * 1024 * 1024
_VMEM_HEADROOM_BYTES = 2 * 1024 * 1024
_CANDIDATE_ROW_BLOCKS = (64, 32)


def _vmem_scratch_bytes(num_heads: int, row_block: int, head_dim: int) -> int:
    in_bytes = num_heads * row_block * head_dim * 2
    out_q_bytes = row_block * num_heads * head_dim
    out_scale_bytes = row_block * SCALE_PAD_COLS * 4
    return 2 * (in_bytes + out_q_bytes + out_scale_bytes)


def select_row_block(
    num_heads: int,
    num_rows: int,
    split_rows: int,
    head_dim: int,
    *,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> int | None:
    """Return largest supported row block (`64` or `32`) fitting VMEM, else `None`."""
    if num_heads <= 0 or head_dim <= 0 or split_rows <= 0 or split_rows > num_rows:
        return None
    for row_block in _CANDIDATE_ROW_BLOCKS:
        if (
            num_rows % row_block == 0
            and split_rows % row_block == 0
            and num_rows // row_block >= 2
            and _vmem_scratch_bytes(num_heads, row_block, head_dim)
            + _VMEM_HEADROOM_BYTES
            <= vmem_limit_bytes
        ):
            return row_block
    return None


def supports_shift_quant_split_3d(
    num_heads: int,
    num_rows: int,
    split_rows: int,
    head_dim: int,
    dtype: jnp.dtype,
    *,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    allow_non_tpu: bool = False,
) -> bool:
    """Return True when the 3D head-major Pallas shift+quant+split kernel applies."""
    if not allow_non_tpu and jax.default_backend() != "tpu":
        return False
    if jnp.dtype(dtype) != jnp.bfloat16:
        return False
    if num_heads < 2 or num_heads % 2 != 0:
        return False
    if head_dim < 128 or head_dim % 128 != 0:
        return False
    return (
        select_row_block(
            num_heads,
            num_rows,
            split_rows,
            head_dim,
            vmem_limit_bytes=vmem_limit_bytes,
        )
        is not None
    )


def shift_quant_split_3d_ref(
    lead_hrv: jax.Array,
    carried_hrv: jax.Array,
    own_hrv: jax.Array,
    rank_arr: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Shift `[H, S/R, V]` to sequence rows, FP8-quantize per row, and split halves."""
    num_heads, num_rows, head_dim = own_hrv.shape
    split_rows = lead_hrv.shape[1]
    half_cols = (num_heads * head_dim) // 2
    rank = rank_arr[0]
    lead = jnp.where(rank == 0, lead_hrv.astype(own_hrv.dtype), carried_hrv)
    shifted_hrv = jnp.concatenate(
        [lead, own_hrv[:, : num_rows - split_rows, :]], axis=1
    )
    shifted_2d = jnp.swapaxes(shifted_hrv, 0, 1).reshape(num_rows, num_heads * head_dim)
    x32 = shifted_2d.astype(jnp.float32)
    scale = jnp.max(jnp.abs(x32), axis=-1, keepdims=True) * jnp.float32(1.0 / FP8_MAX)
    scale = jnp.where(scale == 0.0, jnp.float32(1.0), scale)
    x_q = jnp.clip(x32 * (jnp.float32(1.0) / scale), -FP8_MAX, FP8_MAX).astype(
        jnp.float8_e4m3fn
    )
    parity = lax.rem(rank, 2)
    mine_q = lax.dynamic_slice_in_dim(x_q, parity * half_cols, half_cols, axis=1)
    peer_q = lax.dynamic_slice_in_dim(x_q, (1 - parity) * half_cols, half_cols, axis=1)
    return mine_q, peer_q, scale


def shift_quant_split_3d_fp8(
    lead_hrv: jax.Array,
    carried_hrv: jax.Array,
    own_hrv: jax.Array,
    rank_arr: jax.Array,
    *,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    interpret: bool | Any = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Shift `[H, S/R, V]` in VMEM into FP8 pair halves `[R, K/2]` and scale `[R, 1]`.

    Requires bf16 3D head-major inputs (`lead_hrv`, `carried_hrv` `[H, S, V]`,
    `own_hrv` `[H, R, V]`) with even `H`, `V % 128 == 0`, `0 < S <= R`, and
    `R, S` aligned to a supported row block (`64` or `32`) with `R // row_block >= 2`.
    """
    if own_hrv.ndim != 3 or lead_hrv.ndim != 3 or carried_hrv.ndim != 3:
        raise ValueError(
            "lead_hrv, carried_hrv, and own_hrv must be 3D [H, rows, V] arrays"
        )
    if lead_hrv.shape != carried_hrv.shape:
        raise ValueError(
            f"lead_hrv {lead_hrv.shape} must match carried_hrv {carried_hrv.shape}"
        )
    num_heads, num_rows, head_dim = own_hrv.shape
    split_rows = lead_hrv.shape[1]
    if lead_hrv.shape[0] != num_heads or lead_hrv.shape[2] != head_dim:
        raise ValueError(
            f"lead_hrv {lead_hrv.shape} incompatible with own_hrv {own_hrv.shape}"
        )
    if (
        jnp.dtype(own_hrv.dtype) != jnp.bfloat16
        or jnp.dtype(lead_hrv.dtype) != jnp.bfloat16
        or jnp.dtype(carried_hrv.dtype) != jnp.bfloat16
    ):
        raise ValueError("shift_quant_split_3d_fp8 requires bfloat16 inputs")
    if num_heads < 2 or num_heads % 2 != 0:
        raise ValueError(f"num_heads must be positive and even, got {num_heads}")
    if head_dim < 128 or head_dim % 128 != 0:
        raise ValueError(f"head_dim must be a positive multiple of 128, got {head_dim}")
    row_block = select_row_block(
        num_heads,
        num_rows,
        split_rows,
        head_dim,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    if row_block is None:
        raise ValueError(
            f"Unsupported 3D shift_quant_split shape H={num_heads}, R={num_rows}, "
            f"S={split_rows}, V={head_dim} for vmem_limit_bytes={vmem_limit_bytes}"
        )

    half_heads = num_heads // 2
    half_cols = half_heads * head_dim
    num_lead_blocks = split_rows // row_block
    num_blocks = num_rows // row_block

    def _kernel(
        lead_hbm,
        carried_hbm,
        own_hbm,
        rank_smem,
        mine_hbm,
        peer_hbm,
        scale_hbm,
        in_vm,
        out_mine_vm,
        out_peer_vm,
        out_scale_vm,
        in_sems,
        out_sems,
    ):
        rank = rank_smem[0]
        is_rank0 = rank == 0
        is_odd = (rank & 1) != 0

        # Avoid BlockSpec auto-windowing across lead/carried/own: explicit
        # pl.ANY async DMAs read only the single active source per row block
        # and avoid materializing a [R, H*V] transpose in HBM.
        def _in_copy(block_idx, slot, *, is_start: bool):
            sem = in_sems.at[slot]
            dst = in_vm.at[slot]

            @pl.when(block_idx < num_lead_blocks)
            def _from_lead():
                lead_row = jnp.minimum(block_idx, num_lead_blocks - 1) * row_block

                @pl.when(is_rank0)
                def _from_rank0():
                    op = pltpu.make_async_copy(
                        lead_hbm.at[:, pl.ds(lead_row, row_block), :], dst, sem
                    )
                    if is_start:
                        op.start()
                    else:
                        op.wait()

                @pl.when(~is_rank0)
                def _from_carried():
                    op = pltpu.make_async_copy(
                        carried_hbm.at[:, pl.ds(lead_row, row_block), :], dst, sem
                    )
                    if is_start:
                        op.start()
                    else:
                        op.wait()

            @pl.when(block_idx >= num_lead_blocks)
            def _from_own():
                own_row = jnp.maximum(block_idx - num_lead_blocks, 0) * row_block
                op = pltpu.make_async_copy(
                    own_hbm.at[:, pl.ds(own_row, row_block), :], dst, sem
                )
                if is_start:
                    op.start()
                else:
                    op.wait()

        def _out_copy(block_idx, slot, *, is_start: bool):
            row0 = block_idx * row_block
            c_mine = pltpu.make_async_copy(
                out_mine_vm.at[slot],
                mine_hbm.at[pl.ds(row0, row_block), :],
                out_sems.at[slot, 0],
            )
            c_peer = pltpu.make_async_copy(
                out_peer_vm.at[slot],
                peer_hbm.at[pl.ds(row0, row_block), :],
                out_sems.at[slot, 1],
            )
            c_scale = pltpu.make_async_copy(
                out_scale_vm.at[slot],
                scale_hbm.at[pl.ds(row0, row_block), :],
                out_sems.at[slot, 2],
            )
            if is_start:
                c_mine.start()
                c_peer.start()
                c_scale.start()
            else:
                c_mine.wait()
                c_peer.wait()
                c_scale.wait()

        _in_copy(0, 0, is_start=True)

        def _body(block_idx, _):
            slot = block_idx & 1
            next_slot = 1 - slot

            @pl.when(block_idx + 1 < num_blocks)
            def _prefetch_next():
                _in_copy(block_idx + 1, next_slot, is_start=True)

            @pl.when(block_idx >= 2)
            def _drain_prev_out():
                _out_copy(block_idx - 2, slot, is_start=False)

            _in_copy(block_idx, slot, is_start=False)
            x_ref = in_vm.at[slot]

            row_max = jnp.zeros((row_block, 1), jnp.float32)
            for head_idx in range(num_heads):
                x_head = x_ref[head_idx].astype(jnp.float32)
                row_max = jnp.maximum(
                    row_max, jnp.max(jnp.abs(x_head), axis=-1, keepdims=True)
                )
            scale = row_max * jnp.float32(1.0 / FP8_MAX)
            scale = jnp.where(scale == 0.0, jnp.float32(1.0), scale)
            inv_scale = jnp.float32(1.0) / scale

            mine_dst = out_mine_vm.at[slot]
            peer_dst = out_peer_vm.at[slot]

            def _quantize_half(head_offset: int, dst_ref) -> None:
                for local_head in range(half_heads):
                    x_head = x_ref[head_offset + local_head].astype(jnp.float32)
                    q_head = jnp.clip(x_head * inv_scale, -FP8_MAX, FP8_MAX).astype(
                        jnp.float8_e4m3fn
                    )
                    dst_ref[:, pl.ds(local_head * head_dim, head_dim)] = q_head

            @pl.when(is_odd)
            def _write_odd():
                _quantize_half(half_heads, mine_dst)
                _quantize_half(0, peer_dst)

            @pl.when(~is_odd)
            def _write_even():
                _quantize_half(0, mine_dst)
                _quantize_half(half_heads, peer_dst)

            out_scale_vm.at[slot][...] = jnp.broadcast_to(
                scale, (row_block, SCALE_PAD_COLS)
            )
            _out_copy(block_idx, slot, is_start=True)

        lax.fori_loop(0, num_blocks, _body, None)
        _out_copy(num_blocks - 2, (num_blocks - 2) & 1, is_start=False)
        _out_copy(num_blocks - 1, (num_blocks - 1) & 1, is_start=False)

    mine_q, peer_q, scale_padded = pl.pallas_call(
        _kernel,
        name=f"dcp_shift_quant_split_3d_fp8_b{row_block}",
        grid=(),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        scratch_shapes=(
            pltpu.VMEM((2, num_heads, row_block, head_dim), jnp.bfloat16),
            pltpu.VMEM((2, row_block, half_cols), jnp.float8_e4m3fn),
            pltpu.VMEM((2, row_block, half_cols), jnp.float8_e4m3fn),
            pltpu.VMEM((2, row_block, SCALE_PAD_COLS), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2, 3)),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_rows, half_cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((num_rows, half_cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((num_rows, SCALE_PAD_COLS), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret,
    )(lead_hrv, carried_hrv, own_hrv, rank_arr)
    return mine_q, peer_q, scale_padded[:, :1]
