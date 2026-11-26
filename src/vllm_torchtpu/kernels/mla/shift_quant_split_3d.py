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
        next_dev: Any = {axis_name: next_rank}
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


DEFAULT_WUV_VMEM_LIMIT_BYTES = 56 * 1024 * 1024
_WUV_CANDIDATE_ROW_BLOCKS = (128, 64, 32)


def _wuv_vmem_scratch_bytes(
    num_heads: int,
    row_block: int,
    lkv_dim: int,
    head_dim: int,
    *,
    overlap_quant: bool,
) -> int:
    w_bytes = num_heads * lkv_dim * head_dim * 2
    lat_bytes = 2 * num_heads * row_block * lkv_dim * 2
    out_bufs = 2 if overlap_quant else 1
    out_bytes = out_bufs * num_heads * row_block * head_dim * 2
    q_bytes = 2 * row_block * num_heads * head_dim
    scale_bytes = 2 * row_block * SCALE_PAD_COLS * 4
    return w_bytes + lat_bytes + out_bytes + q_bytes + scale_bytes


def select_wuv_row_block(
    num_heads: int,
    chunk_rows: int,
    split_rows: int,
    lkv_dim: int,
    head_dim: int,
    *,
    overlap_quant: bool = True,
    vmem_limit_bytes: int = DEFAULT_WUV_VMEM_LIMIT_BYTES,
) -> int | None:
    if (
        num_heads <= 0
        or num_heads % 2 != 0
        or lkv_dim <= 0
        or lkv_dim % 128 != 0
        or head_dim <= 0
        or head_dim % 128 != 0
        or split_rows <= 0
        or split_rows % 8 != 0
        or chunk_rows <= 0
    ):
        return None
    budget = vmem_limit_bytes - _VMEM_HEADROOM_BYTES
    for rb in _WUV_CANDIDATE_ROW_BLOCKS:
        if (
            chunk_rows % rb == 0
            and split_rows <= rb
            and (rb - split_rows) % 8 == 0
            and _wuv_vmem_scratch_bytes(
                num_heads, rb, lkv_dim, head_dim, overlap_quant=overlap_quant
            )
            <= budget
        ):
            return rb
    return None


def supports_wuv_quant_split_3d_local(
    num_heads: int,
    num_rows: int,
    chunk_rows: int,
    split_rows: int,
    lkv_dim: int,
    head_dim: int,
    dtype: Any,
    *,
    overlap_quant: bool = True,
    vmem_limit_bytes: int = DEFAULT_WUV_VMEM_LIMIT_BYTES,
) -> bool:
    if jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
        return False
    if num_rows <= 0 or chunk_rows <= 0 or num_rows % chunk_rows != 0:
        return False
    if split_rows > chunk_rows:
        return False
    if (
        select_wuv_row_block(
            num_heads,
            chunk_rows,
            split_rows,
            lkv_dim,
            head_dim,
            overlap_quant=overlap_quant,
            vmem_limit_bytes=vmem_limit_bytes,
        )
        is not None
    ):
        return True
    if overlap_quant:
        return (
            select_wuv_row_block(
                num_heads,
                chunk_rows,
                split_rows,
                lkv_dim,
                head_dim,
                overlap_quant=False,
                vmem_limit_bytes=vmem_limit_bytes,
            )
            is not None
        )
    return False


def wuv_quant_split_3d_local_ref(
    lead_hrl: jax.Array,
    own_hrl_or_chunks: jax.Array | tuple[jax.Array, ...],
    w_uv: jax.Array,
    rank: jax.Array,
    *,
    num_valid_own_rows: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reference for fused head-major W_UV + FP8 row-quant + pair-half split."""
    own_hrl = (
        jnp.concatenate(own_hrl_or_chunks, axis=1)
        if isinstance(own_hrl_or_chunks, tuple)
        else own_hrl_or_chunks
    )
    if num_valid_own_rows is not None:
        n_valid = jnp.reshape(num_valid_own_rows, ()).astype(jnp.int32)
        own_mask = (
            jnp.arange(own_hrl.shape[1], dtype=jnp.int32)[None, :, None] < n_valid
        )
        own_hrl = jnp.where(own_mask, own_hrl, jnp.zeros((), own_hrl.dtype))
    lead_hrv = jnp.einsum(
        "hrl,hlv->hrv", lead_hrl, w_uv, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    own_hrv = jnp.einsum(
        "hrl,hlv->hrv", own_hrl, w_uv, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    return shift_quant_split_3d_local_ref(lead_hrv, own_hrv, rank)


def _start_chunk_slice(
    chunk_refs: tuple[Any, ...],
    chunk_idx: jax.Array | int,
    row_off: jax.Array | int,
    length: int,
    dst_ref: Any,
    sem: Any,
) -> None:
    aligned_off = row_off if isinstance(row_off, int) else pl.multiple_of(row_off, 8)
    if len(chunk_refs) == 1:
        pltpu.make_async_copy(
            chunk_refs[0].at[:, pl.ds(aligned_off, length), :], dst_ref, sem
        ).start()
        return

    for ci, ref in enumerate(chunk_refs):

        def _branch(ref_=ref) -> None:
            pltpu.make_async_copy(
                ref_.at[:, pl.ds(aligned_off, length), :], dst_ref, sem
            ).start()

        pl.when(chunk_idx == ci)(_branch)


def wuv_quant_split_3d_fp8_local(
    lead_hrl: jax.Array,
    own_hrl_or_chunks: jax.Array | tuple[jax.Array, ...],
    w_uv: jax.Array,
    rank: jax.Array,
    *,
    num_valid_own_rows: jax.Array | None = None,
    row_block: int | None = None,
    overlap_quant: bool = True,
    vmem_limit_bytes: int = DEFAULT_WUV_VMEM_LIMIT_BYTES,
    interpret: bool | Any = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fuse `[H, S/R, L] @ [H, L, V]` + FP8 row-quant + pair-half split.

    Consumes head-major MLA latent output `lead_hrl` (`bf16[H, S, L]`) and
    either a single `own_hrl` (`bf16[H, R, L]`) or a tuple of equal chunks
    (`bf16[H, R//C, L]`) directly with stationary VMEM `w_uv`.
    """
    chunks = (
        tuple(own_hrl_or_chunks)
        if isinstance(own_hrl_or_chunks, tuple)
        else (own_hrl_or_chunks,)
    )
    num_chunks = len(chunks)
    if num_chunks == 0:
        raise ValueError("own_hrl_or_chunks must be non-empty")
    num_heads, split_rows, lkv_dim = lead_hrl.shape
    chunk_rows = int(chunks[0].shape[1])
    num_rows = chunk_rows * num_chunks
    for c in chunks:
        if c.shape != (num_heads, chunk_rows, lkv_dim):
            raise ValueError(
                f"Chunk shape mismatch: {c.shape} vs {(num_heads, chunk_rows, lkv_dim)}"
            )
        if c.dtype != jnp.bfloat16:
            raise ValueError("wuv_quant_split_3d_fp8_local requires bfloat16 inputs")
    if lead_hrl.dtype != jnp.bfloat16 or w_uv.dtype != jnp.bfloat16:
        raise ValueError("wuv_quant_split_3d_fp8_local requires bfloat16 inputs")
    if w_uv.shape[:2] != (num_heads, lkv_dim):
        raise ValueError(
            f"w_uv shape mismatch: {w_uv.shape} vs ({num_heads}, {lkv_dim}, V)"
        )
    head_dim = int(w_uv.shape[2])

    if row_block is None:
        rb = select_wuv_row_block(
            num_heads,
            chunk_rows,
            split_rows,
            lkv_dim,
            head_dim,
            overlap_quant=overlap_quant,
            vmem_limit_bytes=vmem_limit_bytes,
        )
        if rb is None and overlap_quant:
            overlap_quant = False
            rb = select_wuv_row_block(
                num_heads,
                chunk_rows,
                split_rows,
                lkv_dim,
                head_dim,
                overlap_quant=False,
                vmem_limit_bytes=vmem_limit_bytes,
            )
        if rb is None:
            raise ValueError(
                f"Unsupported shape for wuv_quant_split_3d_fp8_local: "
                f"H={num_heads}, R={num_rows}, chunk={chunk_rows}, S={split_rows}, "
                f"L={lkv_dim}, V={head_dim}"
            )
        row_block = rb

    if chunk_rows % row_block != 0 or split_rows > row_block:
        raise ValueError(
            f"chunk_rows={chunk_rows} must be a multiple of row_block={row_block} "
            f"and split_rows={split_rows} <= row_block"
        )
    n_rb = num_rows // row_block
    use_overlap = bool(overlap_quant and n_rb >= 2)
    half_heads = num_heads // 2
    out_cols = half_heads * head_dim
    w_chunks = 4 if num_heads % 4 == 0 else (2 if num_heads % 2 == 0 else 1)
    hw = num_heads // w_chunks
    tail_rows = row_block - split_rows
    has_mask = num_valid_own_rows is not None
    rank_vec = jnp.reshape(rank, (1,)).astype(jnp.int32)
    if has_mask:
        assert num_valid_own_rows is not None
        meta_vec = jnp.stack(
            [rank_vec[0], jnp.reshape(num_valid_own_rows, ()).astype(jnp.int32)]
        )
    else:
        meta_vec = rank_vec

    def _kernel(*refs):
        lead_hbm = refs[0]
        own_refs = tuple(refs[1 : 1 + num_chunks])
        w_hbm = refs[1 + num_chunks]
        meta_sm = refs[2 + num_chunks]
        (
            mine_hbm,
            peer_hbm,
            scale_hbm,
            w_vm,
            lat_vm,
            out_vm,
            mine_vm,
            peer_vm,
            scl_vm,
            w_sems,
            lat_sems,
            out_sems,
        ) = refs[3 + num_chunks :]

        rank_i = meta_sm[0]
        is_rank0 = rank_i == jnp.int32(0)
        parity_odd = (rank_i & jnp.int32(1)) != jnp.int32(0)
        n_valid = (
            jnp.maximum(meta_sm[1], jnp.int32(0)) if has_mask else jnp.int32(num_rows)
        )

        def _start_lat(rb_i, slot):
            sem = lat_sems.at[slot]

            @pl.when(rb_i == 0)
            def _rb0():
                @pl.when(is_rank0)
                def _r0():
                    pltpu.make_async_copy(
                        lead_hbm, lat_vm.at[slot, :, pl.ds(0, split_rows), :], sem
                    ).start()
                    if tail_rows > 0:
                        pltpu.make_async_copy(
                            own_refs[0].at[:, pl.ds(0, tail_rows), :],
                            lat_vm.at[slot, :, pl.ds(split_rows, tail_rows), :],
                            sem,
                        ).start()

                @pl.when(~is_rank0)
                def _rn():
                    pltpu.make_async_copy(
                        own_refs[0].at[:, pl.ds(0, row_block), :],
                        lat_vm.at[slot],
                        sem,
                    ).start()

            @pl.when(rb_i > 0)
            def _rbk():
                r_n = rb_i * jnp.int32(row_block)
                r_0 = r_n - jnp.int32(split_rows)

                @pl.when(is_rank0)
                def _r0k():
                    if num_chunks == 1:
                        pltpu.make_async_copy(
                            own_refs[0].at[:, pl.ds(r_0, row_block), :],
                            lat_vm.at[slot],
                            sem,
                        ).start()
                    else:
                        c_lo = lax.div(r_0, jnp.int32(chunk_rows))
                        off_lo = lax.rem(r_0, jnp.int32(chunk_rows))
                        crosses = off_lo > jnp.int32(chunk_rows - row_block)

                        @pl.when(~crosses)
                        def _no_cross():
                            _start_chunk_slice(
                                own_refs, c_lo, off_lo, row_block, lat_vm.at[slot], sem
                            )

                        @pl.when(crosses)
                        def _cross():
                            # Every rank-0 straddle starts at chunk_rows - split_rows
                            # because chunk_rows % row_block == 0 and split <= rb.
                            _start_chunk_slice(
                                own_refs,
                                c_lo,
                                chunk_rows - split_rows,
                                split_rows,
                                lat_vm.at[slot, :, pl.ds(0, split_rows), :],
                                sem,
                            )
                            if tail_rows > 0:
                                _start_chunk_slice(
                                    own_refs,
                                    c_lo + jnp.int32(1),
                                    0,
                                    tail_rows,
                                    lat_vm.at[slot, :, pl.ds(split_rows, tail_rows), :],
                                    sem,
                                )

                @pl.when(~is_rank0)
                def _rnk():
                    if num_chunks == 1:
                        pltpu.make_async_copy(
                            own_refs[0].at[:, pl.ds(r_n, row_block), :],
                            lat_vm.at[slot],
                            sem,
                        ).start()
                    else:
                        c_n = lax.div(r_n, jnp.int32(chunk_rows))
                        off_n = lax.rem(r_n, jnp.int32(chunk_rows))
                        _start_chunk_slice(
                            own_refs, c_n, off_n, row_block, lat_vm.at[slot], sem
                        )

        def _wait_out(slot):
            pltpu.make_async_copy(
                mine_vm.at[slot], mine_vm.at[slot], out_sems.at[slot]
            ).wait()
            pltpu.make_async_copy(
                peer_vm.at[slot], peer_vm.at[slot], out_sems.at[slot]
            ).wait()
            pltpu.make_async_copy(
                scl_vm.at[slot], scl_vm.at[slot], out_sems.at[slot]
            ).wait()

        def _start_out(rb_out, slot):
            r_out = rb_out * jnp.int32(row_block)
            pltpu.make_async_copy(
                mine_vm.at[slot],
                mine_hbm.at[pl.ds(r_out, row_block)],
                out_sems.at[slot],
            ).start()
            pltpu.make_async_copy(
                peer_vm.at[slot],
                peer_hbm.at[pl.ds(r_out, row_block)],
                out_sems.at[slot],
            ).start()
            pltpu.make_async_copy(
                scl_vm.at[slot],
                scale_hbm.at[pl.ds(r_out, row_block)],
                out_sems.at[slot],
            ).start()

        def _row_valid_mask(rb_i):
            # Avoid (row_block, 1) i1 mask inside Mosaic select_n lowering.
            row_idx = lax.broadcasted_iota(jnp.int32, (row_block, lkv_dim), 0)
            g_row = rb_i * jnp.int32(row_block) + row_idx
            own_limit = jnp.where(is_rank0, n_valid + jnp.int32(split_rows), n_valid)
            return g_row < own_limit

        def _quant_pair(out_slot, inv_scale, is_odd: bool, dst_slot):
            def _qh(h_off: int, dst_ref) -> None:
                for lh in range(half_heads):
                    xh = out_vm[out_slot, h_off + lh].astype(jnp.float32)
                    dst_ref[:, pl.ds(lh * head_dim, head_dim)] = jnp.clip(
                        xh * inv_scale, -FP8_MAX, FP8_MAX
                    ).astype(jnp.float8_e4m3fn)

            if is_odd:
                _qh(half_heads, mine_vm.at[dst_slot])
                _qh(0, peer_vm.at[dst_slot])
            else:
                _qh(0, mine_vm.at[dst_slot])
                _qh(half_heads, peer_vm.at[dst_slot])

        # Issue all weight chunks + block 0 latent concurrently.
        for wc in range(w_chunks):
            pltpu.make_async_copy(
                w_hbm.at[pl.ds(wc * hw, hw)],
                w_vm.at[pl.ds(wc * hw, hw)],
                w_sems.at[wc],
            ).start()
        _start_lat(jnp.int32(0), jnp.int32(0))

        def _read_lat(slot, h: int, mask):
            x = lat_vm[slot, h]
            if has_mask:
                return jnp.where(mask, x, jnp.zeros((), jnp.bfloat16))
            return x

        if not use_overlap:

            def _body(rb_i, _):
                slot = rb_i & jnp.int32(1)
                nxt = rb_i + jnp.int32(1)

                @pl.when(nxt < n_rb)
                def _pref():
                    _start_lat(nxt, nxt & jnp.int32(1))

                @pl.when(rb_i >= 2)
                def _wo():
                    _wait_out(slot)

                pltpu.make_async_copy(
                    lat_vm.at[slot], lat_vm.at[slot], lat_sems.at[slot]
                ).wait()
                mask = _row_valid_mask(rb_i) if has_mask else None
                # Avoid per-head cross-lane max reductions inside the head loop:
                # accumulate non-negative bf16 max values elementwise (exact
                # top-16 bits match f32), reducing across lanes once per block.
                rm_mat = jnp.zeros((row_block, head_dim), jnp.bfloat16)
                for wc in range(w_chunks):
                    h0 = wc * hw

                    def _wait_w(h0_=h0, wc_=wc):
                        pltpu.make_async_copy(
                            w_vm.at[pl.ds(h0_, hw)],
                            w_vm.at[pl.ds(h0_, hw)],
                            w_sems.at[wc_],
                        ).wait()

                    pl.when(rb_i == 0)(_wait_w)
                    for hi in range(hw):
                        h = h0 + hi
                        acc = jnp.dot(
                            _read_lat(slot, h, mask),
                            w_vm[h],
                            preferred_element_type=jnp.float32,
                        )
                        val = acc.astype(jnp.bfloat16)
                        out_vm[0, h] = val
                        rm_mat = jnp.maximum(rm_mat, jnp.abs(val))
                row_max = jnp.max(rm_mat.astype(jnp.float32), axis=-1, keepdims=True)
                scale = row_max * jnp.float32(1.0 / FP8_MAX)
                scale = jnp.where(scale == 0.0, jnp.float32(1.0), scale)
                inv_scale = jnp.float32(1.0) / scale

                @pl.when(parity_odd)
                def _odd():
                    _quant_pair(0, inv_scale, True, slot)

                @pl.when(~parity_odd)
                def _even():
                    _quant_pair(0, inv_scale, False, slot)

                scl_vm[slot] = jnp.broadcast_to(scale, (row_block, SCALE_PAD_COLS))
                _start_out(rb_i, slot)
                return None

            lax.fori_loop(0, n_rb, _body, None)
            for tail in range(min(2, n_rb)):
                _wait_out((n_rb - 1 - tail) & 1)
        else:
            # Avoid dynamic parity checks inside the per-head loop: branch once outside.
            _start_lat(jnp.int32(1), jnp.int32(1))
            pltpu.make_async_copy(lat_vm.at[0], lat_vm.at[0], lat_sems.at[0]).wait()
            mask0 = _row_valid_mask(jnp.int32(0)) if has_mask else None
            rm0_mat = jnp.zeros((row_block, head_dim), jnp.bfloat16)
            for wc in range(w_chunks):
                h0 = wc * hw
                pltpu.make_async_copy(
                    w_vm.at[pl.ds(h0, hw)],
                    w_vm.at[pl.ds(h0, hw)],
                    w_sems.at[wc],
                ).wait()
                for hi in range(hw):
                    h = h0 + hi
                    acc = jnp.dot(
                        _read_lat(jnp.int32(0), h, mask0),
                        w_vm[h],
                        preferred_element_type=jnp.float32,
                    )
                    val = acc.astype(jnp.bfloat16)
                    out_vm[0, h] = val
                    rm0_mat = jnp.maximum(rm0_mat, jnp.abs(val))
            rm0 = jnp.max(rm0_mat.astype(jnp.float32), axis=-1, keepdims=True)
            s0 = rm0 * jnp.float32(1.0 / FP8_MAX)
            s0 = jnp.where(s0 == 0.0, jnp.float32(1.0), s0)

            def _run_pipe(is_odd: bool) -> None:
                def _step(rb_i, prev_scale):
                    slot = rb_i & jnp.int32(1)
                    prev_slot = slot ^ jnp.int32(1)
                    nxt = rb_i + jnp.int32(1)

                    @pl.when(nxt < n_rb)
                    def _pref():
                        _start_lat(nxt, nxt & jnp.int32(1))

                    @pl.when(rb_i >= 3)
                    def _wo():
                        _wait_out(prev_slot)

                    pltpu.make_async_copy(
                        lat_vm.at[slot], lat_vm.at[slot], lat_sems.at[slot]
                    ).wait()
                    mask = _row_valid_mask(rb_i) if has_mask else None
                    prev_inv = jnp.float32(1.0) / prev_scale
                    rm_mat = jnp.zeros((row_block, head_dim), jnp.bfloat16)
                    for h in range(num_heads):
                        acc = jnp.dot(
                            _read_lat(slot, h, mask),
                            w_vm[h],
                            preferred_element_type=jnp.float32,
                        )
                        val = acc.astype(jnp.bfloat16)
                        out_vm[slot, h] = val
                        rm_mat = jnp.maximum(rm_mat, jnp.abs(val))
                        xh = out_vm[prev_slot, h].astype(jnp.float32)
                        qh = jnp.clip(xh * prev_inv, -FP8_MAX, FP8_MAX).astype(
                            jnp.float8_e4m3fn
                        )
                        if h < half_heads:
                            col = pl.ds(h * head_dim, head_dim)
                            if is_odd:
                                peer_vm[prev_slot, :, col] = qh
                            else:
                                mine_vm[prev_slot, :, col] = qh
                        else:
                            col = pl.ds((h - half_heads) * head_dim, head_dim)
                            if is_odd:
                                mine_vm[prev_slot, :, col] = qh
                            else:
                                peer_vm[prev_slot, :, col] = qh
                    scl_vm[prev_slot] = jnp.broadcast_to(
                        prev_scale, (row_block, SCALE_PAD_COLS)
                    )
                    _start_out(rb_i - jnp.int32(1), prev_slot)
                    rm = jnp.max(rm_mat.astype(jnp.float32), axis=-1, keepdims=True)
                    sc = rm * jnp.float32(1.0 / FP8_MAX)
                    return jnp.where(sc == 0.0, jnp.float32(1.0), sc)

                last_sc = lax.fori_loop(1, n_rb, _step, s0)
                lslot = (n_rb - 1) & 1
                if n_rb >= 3:
                    _wait_out(lslot)
                linv = jnp.float32(1.0) / last_sc
                _quant_pair(lslot, linv, is_odd, lslot)
                scl_vm[lslot] = jnp.broadcast_to(last_sc, (row_block, SCALE_PAD_COLS))
                _start_out(jnp.int32(n_rb - 1), jnp.int32(lslot))

            @pl.when(parity_odd)
            def _ro():
                _run_pipe(True)

            @pl.when(~parity_odd)
            def _re():
                _run_pipe(False)

            for tail in range(min(2, n_rb)):
                _wait_out((n_rb - 1 - tail) & 1)

    n_out_bufs = 2 if use_overlap else 1
    mine_q, peer_q, scale_padded = pl.pallas_call(
        _kernel,
        name=f"dcp_wuv_quant_local_3d_fp8_c{num_chunks}_b{row_block}",
        grid=(),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            *([pl.BlockSpec(memory_space=pl.ANY)] * num_chunks),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        scratch_shapes=(
            pltpu.VMEM((num_heads, lkv_dim, head_dim), jnp.bfloat16),
            pltpu.VMEM((2, num_heads, row_block, lkv_dim), jnp.bfloat16),
            pltpu.VMEM((n_out_bufs, num_heads, row_block, head_dim), jnp.bfloat16),
            pltpu.VMEM((2, row_block, out_cols), jnp.float8_e4m3fn),
            pltpu.VMEM((2, row_block, out_cols), jnp.float8_e4m3fn),
            pltpu.VMEM((2, row_block, SCALE_PAD_COLS), jnp.float32),
            pltpu.SemaphoreType.DMA((w_chunks,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_rows, out_cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((num_rows, out_cols), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((num_rows, SCALE_PAD_COLS), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        interpret=interpret,
    )(lead_hrl, *chunks, w_uv, meta_vec)
    return mine_q, peer_q, scale_padded[:, :1]
