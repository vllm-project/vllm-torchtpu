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
"""Micro-batched hierarchical reduce-scatter on the TensorCores of tpu7x."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.hierrs_tc.config import (
    _MIN_SAFE_MICRO_BATCHES,
    Config,
    next_multiple_of,
    pick_num_micro_batches,
    splits_hidden_evenly,
)
from vllm_torchtpu.kernels.collectives.hierrs_tc.kernel_microbatch import (
    hier_rs_kernel,
    make_unified_scratch_shapes,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_SUPPORTED_NUM_DEVICES = 8
_MIN_OUTPUT_ROWS = 8
# Every dtype's sublane tile on tpu7x divides 32, so per-device chunks padded
# to it are legal DMA slice and block sizes.
_SEQ_TILE_ROWS = 32


def supports_num_devices(num_devices: int) -> bool:
    return num_devices == _SUPPORTED_NUM_DEVICES


def is_supported_by_hierrs_tc(num_devices: int, rows: int, hidden: int) -> bool:
    return (
        supports_num_devices(num_devices)
        and rows % num_devices == 0
        and rows // num_devices >= _MIN_OUTPUT_ROWS
        and splits_hidden_evenly(hidden, _MIN_SAFE_MICRO_BATCHES, num_devices)
    )


_NUM_WORK_BUFFERS = 3  # running_sum, recv_buf, phase-2 landing buffer


def _work_set_bytes(local_seq_len, hidden_dim_size, itemsize):
    return (
        _NUM_WORK_BUFFERS * _packed_seq_len(local_seq_len) * hidden_dim_size * itemsize
    )


def _packed_seq_len(local_seq_len):
    # Only this device's chunk parity is ever touched (ChunkLocator.pack).
    return local_seq_len // 2


@functools.cache
def _log_work_set_in_hbm(work_bytes: int, capacity: int) -> None:
    logger.info(
        "hierrs_tc: the working set (%.2f MiB) plus the operand does not fit "
        "in %.0f MiB of VMEM; the working buffers are placed in HBM.",
        work_bytes / 2**20,
        capacity / 2**20,
    )


def _plan_work_scratch(
    local_seq_len,
    hidden_dim_size,
    itemsize,
    num_micro_batches,
    vmem_frac,
):
    """Returns (work set in VMEM scratch?, scoped VMEM claim as a fraction).

    The claim is sized to need: whatever it leaves free is where XLA can keep
    the operand and output, and an underestimate fails at compile time
    (CompileTimeScopedVmemOom), never silently.
    """
    capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    operand = local_seq_len * hidden_dim_size * itemsize
    work = _work_set_bytes(local_seq_len, hidden_dim_size, itemsize)
    scoped = _scoped_need(operand, num_micro_batches)
    need = scoped + work
    if need + operand > capacity * _VMEM_TOTAL_SAFETY:
        _log_work_set_in_hbm(work, capacity)
        return False, min(vmem_frac, scoped * _HBM_FORM_SCOPED_HEADROOM / capacity)
    return True, need / capacity


_VMEM_FRAC_DEFAULT = 0.95
_VMEM_SCOPED_SLACK = 1.30
_HBM_FORM_SCOPED_HEADROOM = 1.5
# Left for the other buffers XLA places in VMEM.
_VMEM_TOTAL_SAFETY = 0.92


def _scoped_need(operand_bytes, num_micro_batches):
    # Each BufferedRef block is (seq_chunk_size, hidden / num_micro_batches).
    return int(operand_bytes / max(1, num_micro_batches) * _VMEM_SCOPED_SLACK)


def _pick_scoped_claim(local_seq_len, hidden_dim_size, itemsize, num_micro_batches):
    """Scoped VMEM claim, as a fraction, capped so the operand fits beside it."""
    capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    operand = local_seq_len * hidden_dim_size * itemsize
    scoped_need = _scoped_need(operand, num_micro_batches)

    if operand + scoped_need > capacity * _VMEM_TOTAL_SAFETY:
        return _VMEM_FRAC_DEFAULT

    lo = scoped_need / capacity
    hi = (capacity - operand) / capacity
    if lo > hi:
        return _VMEM_FRAC_DEFAULT
    return min(max(lo, 0.05), hi)


def hierarchical_reduce_scatter_local(
    local_x: jax.Array,
    num_devices: int,
    num_micro_batches: int | None = None,
    axis_name: str | tuple[str, ...] = "x",
) -> jax.Array:
    num_chips = num_devices // 2
    num_hcube_dims = int(math.log2(num_chips))
    local_seq_len, hidden_dim_size = local_x.shape

    seq_chunk_size_orig = local_seq_len // num_devices
    seq_chunk_size_padded = next_multiple_of(
        max(seq_chunk_size_orig, 1), _SEQ_TILE_ROWS
    )
    needs_padding = seq_chunk_size_padded != seq_chunk_size_orig

    if needs_padding:
        reshaped_x = local_x.reshape(num_devices, -1, hidden_dim_size)
        padded_x = jnp.pad(
            reshaped_x,
            ((0, 0), (0, seq_chunk_size_padded - seq_chunk_size_orig), (0, 0)),
        )
        local_x = padded_x.reshape(-1, hidden_dim_size)
        local_seq_len = local_x.shape[0]

    if num_micro_batches is None:
        num_micro_batches = pick_num_micro_batches(
            local_seq_len,
            hidden_dim_size,
            local_x.dtype.itemsize,
            num_devices=num_devices,
        )

    vmem_frac = _pick_scoped_claim(
        local_seq_len, hidden_dim_size, local_x.dtype.itemsize, num_micro_batches
    )

    seq_chunk_size = local_seq_len // num_devices

    out_shape = jax.ShapeDtypeStruct((seq_chunk_size, hidden_dim_size), local_x.dtype)
    packed_seq_len = _packed_seq_len(local_seq_len)
    work_buffer_shape = jax.ShapeDtypeStruct(
        (packed_seq_len, hidden_dim_size), local_x.dtype
    )

    work_set_in_vmem, vmem_frac = _plan_work_scratch(
        local_seq_len,
        hidden_dim_size,
        local_x.dtype.itemsize,
        num_micro_batches,
        vmem_frac,
    )

    # Pallas cannot allocate HBM scratch, so in HBM the work buffers are
    # unread pl.ANY outputs. Pallas passes inputs, outputs, then scratch: the
    # buffers are all trailing outputs or all leading scratch, so the kernel
    # sees them at the same positions either way.
    out_shapes = [out_shape]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY)]
    work_scratch = []
    for _ in range(_NUM_WORK_BUFFERS):
        if work_set_in_vmem:
            work_scratch.append(pltpu.VMEM(work_buffer_shape.shape, local_x.dtype))
        else:
            out_shapes.append(work_buffer_shape)
            out_specs.append(pl.BlockSpec(memory_space=pl.ANY))

    config = Config(
        num_devices=num_devices,
        hidden_dim_size=hidden_dim_size,
        local_seq_len=local_seq_len,
        dtype=local_x.dtype,
        _num_micro_batches=num_micro_batches,
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        # A VMEM BlockSpec would copy the operand into scoped scratch rather
        # than place it in VMEM.
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=tuple(out_specs),
        # pyrefly: ignore[bad-argument-type]
        scratch_shapes=tuple(work_scratch)
        + tuple(
            make_unified_scratch_shapes(
                seq_chunk_size,
                config.mb_size,
                local_x.dtype,
                num_chips,
                num_hcube_dims,
                num_micro_batches,
            )
        ),
        grid=(1,),
    )

    hier_rs = pl.pallas_call(
        jax.tree_util.Partial(
            hier_rs_kernel,
            config=config,
            axis_name=axis_name,
        ),
        out_shape=tuple(out_shapes),
        grid_spec=grid_spec,
        name=f"hier_rs_kernel.mb{num_micro_batches}",
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * vmem_frac),
            disable_bounds_checks=True,
        ),
    )
    out = hier_rs(local_x)[0]
    if needs_padding:
        out = out[:seq_chunk_size_orig, :]
    return out
