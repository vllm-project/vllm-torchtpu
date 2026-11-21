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
DEFAULT_REMOTE_VMEM_LIMIT_BYTES = 16 * 1024 * 1024
REMOTE_COLLECTIVE_ID = 17
_VMEM_HEADROOM_BYTES = 2 * 1024 * 1024
_CANDIDATE_ROW_BLOCKS = (64, 32)


def _vmem_scratch_bytes(num_heads: int, row_block: int, head_dim: int) -> int:
    in_bytes = num_heads * row_block * head_dim * 2
    out_q_bytes = row_block * num_heads * head_dim
    out_scale_bytes = row_block * SCALE_PAD_COLS * 4
    return 2 * (in_bytes + out_q_bytes + out_scale_bytes)


def _remote_vmem_scratch_bytes(num_heads: int, row_block: int, head_dim: int) -> int:
    in_bytes = num_heads * row_block * head_dim * 2
    out_q_bytes = row_block * num_heads * head_dim
    out_scale_bytes = row_block * SCALE_PAD_COLS * 4
    return 2 * in_bytes + 3 * (out_q_bytes + out_scale_bytes)


def _quantize_tile_to_halves(
    x_ref,
    mine_dst,
    peer_dst,
    scale_dst,
    parity_odd,
    *,
    num_heads: int,
    row_block: int,
    head_dim: int,
) -> None:
    half_heads = num_heads // 2
    row_max = jnp.zeros((row_block, 1), jnp.float32)
    for head_idx in range(num_heads):
        x_head = x_ref[head_idx].astype(jnp.float32)
        row_max = jnp.maximum(row_max, jnp.max(jnp.abs(x_head), axis=-1, keepdims=True))
    scale = row_max * jnp.float32(1.0 / FP8_MAX)
    scale = jnp.where(scale == 0.0, jnp.float32(1.0), scale)
    inv_scale = jnp.float32(1.0) / scale

    def _quantize_half(head_offset: int, dst_ref) -> None:
        for local_head in range(half_heads):
            x_head = x_ref[head_offset + local_head].astype(jnp.float32)
            q_head = jnp.clip(x_head * inv_scale, -FP8_MAX, FP8_MAX).astype(
                jnp.float8_e4m3fn
            )
            dst_ref[:, pl.ds(local_head * head_dim, head_dim)] = q_head

    @pl.when(parity_odd)
    def _write_odd():
        _quantize_half(half_heads, mine_dst)
        _quantize_half(0, peer_dst)

    @pl.when(~parity_odd)
    def _write_even():
        _quantize_half(0, mine_dst)
        _quantize_half(half_heads, peer_dst)

    scale_dst[...] = jnp.broadcast_to(scale, (row_block, SCALE_PAD_COLS))


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
            _quantize_tile_to_halves(
                in_vm.at[slot],
                out_mine_vm.at[slot],
                out_peer_vm.at[slot],
                out_scale_vm.at[slot],
                is_odd,
                num_heads=num_heads,
                row_block=row_block,
                head_dim=head_dim,
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


def supports_shift_quant_split_3d_remote(
    num_heads: int,
    num_rows: int,
    split_rows: int,
    head_dim: int,
    dtype: jnp.dtype,
    dcp_size: int,
    *,
    vmem_limit_bytes: int = DEFAULT_REMOTE_VMEM_LIMIT_BYTES,
    allow_non_tpu: bool = False,
) -> bool:
    if not allow_non_tpu and jax.default_backend() != "tpu":
        return False
    if dcp_size < 2 or split_rows not in _CANDIDATE_ROW_BLOCKS:
        return False
    if jnp.dtype(dtype) != jnp.bfloat16:
        return False
    if num_heads < 2 or num_heads % 2 != 0:
        return False
    if head_dim < 128 or head_dim % 128 != 0:
        return False
    if num_rows <= 0 or num_rows % split_rows != 0 or num_rows // split_rows < 3:
        return False
    return (
        _remote_vmem_scratch_bytes(num_heads, split_rows, head_dim)
        + _VMEM_HEADROOM_BYTES
        <= vmem_limit_bytes
    )


def shift_quant_split_3d_fp8_remote(
    lead_hrv: jax.Array,
    own_hrv: jax.Array,
    rank_arr: jax.Array,
    *,
    dcp_size: int,
    axis_name: str,
    vmem_limit_bytes: int = DEFAULT_REMOTE_VMEM_LIMIT_BYTES,
    collective_id: int = REMOTE_COLLECTIVE_ID,
    interpret: bool | Any = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Quantize `[H, R, V]` to FP8 pair halves `[R, K/2]` with fused 1-hop tail push.

    Replaces an external `bf16[H, S, V]` `lax.ppermute` (`i -> i + 1`) by
    quantizing `own_hrv[:, R - S:, :]` in rank `i + 1`'s pair parity and pushing
    the quantized FP8 halves and f32 scale directly to rank `i + 1`'s output
    slice `[0:S]`, overlapped with the local body `own_hrv[:, :R - S, :]`.
    """
    if own_hrv.ndim != 3 or lead_hrv.ndim != 3:
        raise ValueError("lead_hrv and own_hrv must be 3D [H, rows, V] arrays")
    num_heads, num_rows, head_dim = own_hrv.shape
    split_rows = lead_hrv.shape[1]
    if lead_hrv.shape[0] != num_heads or lead_hrv.shape[2] != head_dim:
        raise ValueError(
            f"lead_hrv {lead_hrv.shape} incompatible with own_hrv {own_hrv.shape}"
        )
    if (
        jnp.dtype(own_hrv.dtype) != jnp.bfloat16
        or jnp.dtype(lead_hrv.dtype) != jnp.bfloat16
    ):
        raise ValueError("shift_quant_split_3d_fp8_remote requires bfloat16 inputs")
    if dcp_size < 2:
        raise ValueError(f"dcp_size must be >= 2, got {dcp_size}")
    if not supports_shift_quant_split_3d_remote(
        num_heads,
        num_rows,
        split_rows,
        head_dim,
        own_hrv.dtype,
        dcp_size,
        vmem_limit_bytes=vmem_limit_bytes,
        allow_non_tpu=True,
    ):
        raise ValueError(
            f"Unsupported remote shift_quant_split shape H={num_heads}, R={num_rows}, "
            f"S={split_rows}, V={head_dim}, dcp={dcp_size} for "
            f"vmem_limit_bytes={vmem_limit_bytes}"
        )

    row_block = split_rows
    half_cols = (num_heads // 2) * head_dim
    num_blocks = num_rows // row_block
    n_body = num_blocks - 1
    mesh_id_type = pl.DeviceIdType.MESH

    def _kernel(
        lead_hbm,
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
        rem_send_sems,
        rem_recv_sems,
    ):
        rank = rank_smem[0]
        is_rank0 = rank == 0
        has_prev = rank > 0
        has_next = rank < (dcp_size - 1)
        is_odd = (rank & 1) != 0
        next_rank = jnp.minimum(rank + 1, dcp_size - 1)
        prev_rank = jnp.maximum(rank - 1, 0)
        next_dev = {axis_name: next_rank}
        barrier_sem = pltpu.get_barrier_semaphore()

        # Avoid racing remote writes before rank+1 enters the kernel and exposes
        # its destination HBM slices: each rank > 0 signals rank - 1 on entry.
        @pl.when(has_prev)
        def _signal_prev():
            pl.semaphore_signal(
                barrier_sem,
                1,
                device_id={axis_name: prev_rank},
                device_id_type=mesh_id_type,
            )

        rem_mine = pltpu.make_async_remote_copy(
            out_mine_vm.at[2],
            mine_hbm.at[pl.ds(0, row_block), :],
            rem_send_sems.at[0],
            rem_recv_sems.at[0],
            device_id=next_dev,
            device_id_type=mesh_id_type,
        )
        rem_peer = pltpu.make_async_remote_copy(
            out_peer_vm.at[2],
            peer_hbm.at[pl.ds(0, row_block), :],
            rem_send_sems.at[1],
            rem_recv_sems.at[1],
            device_id=next_dev,
            device_id_type=mesh_id_type,
        )
        rem_scale = pltpu.make_async_remote_copy(
            out_scale_vm.at[2],
            scale_hbm.at[pl.ds(0, row_block), :],
            rem_send_sems.at[2],
            rem_recv_sems.at[2],
            device_id=next_dev,
            device_id_type=mesh_id_type,
        )

        # Prefetch this rank's tail block into slot 0 for remote push to rank+1,
        # and overlap the first local body block fetch into slot 1.
        @pl.when(has_next)
        def _start_tail():
            pltpu.make_async_copy(
                own_hbm.at[:, pl.ds(num_rows - row_block, row_block), :],
                in_vm.at[0],
                in_sems.at[0],
            ).start()

        pltpu.make_async_copy(
            own_hbm.at[:, pl.ds(0, row_block), :], in_vm.at[1], in_sems.at[1]
        ).start()

        @pl.when(has_next)
        def _push_tail():
            pltpu.make_async_copy(
                own_hbm.at[:, pl.ds(num_rows - row_block, row_block), :],
                in_vm.at[0],
                in_sems.at[0],
            ).wait()
            # Quantize in destination rank+1's parity (~is_odd) and hold in
            # dedicated slot 2 until wait_send() so body double-buffering stays
            # confined to slots 0 and 1.
            _quantize_tile_to_halves(
                in_vm.at[0],
                out_mine_vm.at[2],
                out_peer_vm.at[2],
                out_scale_vm.at[2],
                ~is_odd,
                num_heads=num_heads,
                row_block=row_block,
                head_dim=head_dim,
            )
            pl.semaphore_wait(barrier_sem, 1)
            rem_mine.start()
            rem_peer.start()
            rem_scale.start()

        @pl.when(is_rank0)
        def _write_rank0_lead():
            lead_in = pltpu.make_async_copy(
                lead_hbm.at[:, pl.ds(0, row_block), :], in_vm.at[0], in_sems.at[0]
            )
            lead_in.start()
            lead_in.wait()
            _quantize_tile_to_halves(
                in_vm.at[0],
                out_mine_vm.at[0],
                out_peer_vm.at[0],
                out_scale_vm.at[0],
                is_odd,
                num_heads=num_heads,
                row_block=row_block,
                head_dim=head_dim,
            )
            c_mine = pltpu.make_async_copy(
                out_mine_vm.at[0],
                mine_hbm.at[pl.ds(0, row_block), :],
                out_sems.at[0, 0],
            )
            c_peer = pltpu.make_async_copy(
                out_peer_vm.at[0],
                peer_hbm.at[pl.ds(0, row_block), :],
                out_sems.at[0, 1],
            )
            c_scale = pltpu.make_async_copy(
                out_scale_vm.at[0],
                scale_hbm.at[pl.ds(0, row_block), :],
                out_sems.at[0, 2],
            )
            c_mine.start()
            c_peer.start()
            c_scale.start()
            c_mine.wait()
            c_peer.wait()
            c_scale.wait()

        def _out_copy(body_idx, slot, *, is_start: bool):
            row0 = (body_idx + 1) * row_block
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

        def _body(body_idx, _):
            slot = (body_idx + 1) & 1
            next_slot = 1 - slot

            @pl.when(body_idx + 1 < n_body)
            def _prefetch_next():
                pltpu.make_async_copy(
                    own_hbm.at[:, pl.ds((body_idx + 1) * row_block, row_block), :],
                    in_vm.at[next_slot],
                    in_sems.at[next_slot],
                ).start()

            @pl.when(body_idx >= 2)
            def _drain_prev_out():
                _out_copy(body_idx - 2, slot, is_start=False)

            pltpu.make_async_copy(
                own_hbm.at[:, pl.ds(body_idx * row_block, row_block), :],
                in_vm.at[slot],
                in_sems.at[slot],
            ).wait()
            _quantize_tile_to_halves(
                in_vm.at[slot],
                out_mine_vm.at[slot],
                out_peer_vm.at[slot],
                out_scale_vm.at[slot],
                is_odd,
                num_heads=num_heads,
                row_block=row_block,
                head_dim=head_dim,
            )
            _out_copy(body_idx, slot, is_start=True)

        lax.fori_loop(0, n_body, _body, None)
        _out_copy(n_body - 2, ((n_body - 2) + 1) & 1, is_start=False)
        _out_copy(n_body - 1, ((n_body - 1) + 1) & 1, is_start=False)

        @pl.when(has_next)
        def _wait_send():
            rem_mine.wait_send()
            rem_peer.wait_send()
            rem_scale.wait_send()

        @pl.when(has_prev)
        def _wait_recv():
            rem_mine.wait_recv()
            rem_peer.wait_recv()
            rem_scale.wait_recv()

    mine_q, peer_q, scale_padded = pl.pallas_call(
        _kernel,
        name=f"dcp_shift_quant_remote_3d_fp8_b{row_block}",
        grid=(),
        in_specs=[
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
            pltpu.VMEM((3, row_block, half_cols), jnp.float8_e4m3fn),
            pltpu.VMEM((3, row_block, half_cols), jnp.float8_e4m3fn),
            pltpu.VMEM((3, row_block, SCALE_PAD_COLS), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2, 3)),
            pltpu.SemaphoreType.DMA((3,)),
            pltpu.SemaphoreType.DMA((3,)),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_rows, half_cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((num_rows, half_cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((num_rows, SCALE_PAD_COLS), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes, collective_id=collective_id
        ),
        interpret=interpret,
    )(lead_hrv, own_hrv, rank_arr)
    return mine_q, peer_q, scale_padded[:, :1]


def shift_quant_split_3d_local_ref(
    lead_hrv: jax.Array,
    own_hrv: jax.Array,
    rank_arr: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Assemble local `[H, R, V]` (`rank 0` prepends `lead_hrv`), quantize, split."""
    num_heads, num_rows, head_dim = own_hrv.shape
    split_rows = lead_hrv.shape[1]
    half_cols = (num_heads * head_dim) // 2
    rank = rank_arr[0]
    r0_hrv = jnp.concatenate(
        [lead_hrv.astype(own_hrv.dtype), own_hrv[:, : num_rows - split_rows, :]],
        axis=1,
    )
    seq_hrv = jnp.where(rank == 0, r0_hrv, own_hrv)
    seq_2d = jnp.swapaxes(seq_hrv, 0, 1).reshape(num_rows, num_heads * head_dim)
    x32 = seq_2d.astype(jnp.float32)
    scale = jnp.max(jnp.abs(x32), axis=-1, keepdims=True) * jnp.float32(1.0 / FP8_MAX)
    scale = jnp.where(scale == 0.0, jnp.float32(1.0), scale)
    x_q = jnp.clip(x32 * (jnp.float32(1.0) / scale), -FP8_MAX, FP8_MAX).astype(
        jnp.float8_e4m3fn
    )
    parity = lax.rem(rank, 2)
    mine_q = lax.dynamic_slice_in_dim(x_q, parity * half_cols, half_cols, axis=1)
    peer_q = lax.dynamic_slice_in_dim(x_q, (1 - parity) * half_cols, half_cols, axis=1)
    return mine_q, peer_q, scale


def shift_quant_split_3d_fp8_local(
    lead_hrv: jax.Array,
    own_hrv: jax.Array,
    rank_arr: jax.Array,
    *,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    interpret: bool | Any = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Quantize and split `[H, R, V]` with zero cross-rank transfer.

    Rank 0 consumes `lead_hrv` for `[0, S)` and `own_hrv[:, :R - S, :]` for
    `[S, R)`; nonzero ranks consume `own_hrv` directly at `[0, R)`.
    """
    if own_hrv.ndim != 3 or lead_hrv.ndim != 3:
        raise ValueError("lead_hrv and own_hrv must be 3D [H, rows, V] arrays")
    num_heads, num_rows, head_dim = own_hrv.shape
    split_rows = lead_hrv.shape[1]
    if lead_hrv.shape[0] != num_heads or lead_hrv.shape[2] != head_dim:
        raise ValueError(
            f"lead_hrv {lead_hrv.shape} incompatible with own_hrv {own_hrv.shape}"
        )
    if (
        jnp.dtype(own_hrv.dtype) != jnp.bfloat16
        or jnp.dtype(lead_hrv.dtype) != jnp.bfloat16
    ):
        raise ValueError("shift_quant_split_3d_fp8_local requires bfloat16 inputs")
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

        def _in_copy(block_idx, slot, *, is_start: bool):
            sem = in_sems.at[slot]
            dst = in_vm.at[slot]
            take_lead = is_rank0 & (block_idx < num_lead_blocks)

            @pl.when(take_lead)
            def _from_lead():
                lead_row = jnp.minimum(block_idx, num_lead_blocks - 1) * row_block
                op = pltpu.make_async_copy(
                    lead_hbm.at[:, pl.ds(lead_row, row_block), :], dst, sem
                )
                if is_start:
                    op.start()
                else:
                    op.wait()

            @pl.when(~take_lead)
            def _from_own():
                own_blk = jnp.where(
                    is_rank0,
                    jnp.maximum(block_idx - num_lead_blocks, 0),
                    block_idx,
                )
                own_row = own_blk * row_block
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
            _quantize_tile_to_halves(
                in_vm.at[slot],
                out_mine_vm.at[slot],
                out_peer_vm.at[slot],
                out_scale_vm.at[slot],
                is_odd,
                num_heads=num_heads,
                row_block=row_block,
                head_dim=head_dim,
            )
            _out_copy(block_idx, slot, is_start=True)

        lax.fori_loop(0, num_blocks, _body, None)
        _out_copy(num_blocks - 2, (num_blocks - 2) & 1, is_start=False)
        _out_copy(num_blocks - 1, (num_blocks - 1) & 1, is_start=False)

    mine_q, peer_q, scale_padded = pl.pallas_call(
        _kernel,
        name=f"dcp_shift_quant_local_3d_fp8_b{row_block}",
        grid=(),
        in_specs=[
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
    )(lead_hrv, own_hrv, rank_arr)
    return mine_q, peer_q, scale_padded[:, :1]
