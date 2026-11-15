# SPDX-License-Identifier: Apache-2.0
"""Row gathers from the PLE short-conv history, read from HBM by a Pallas kernel.

With VLLM_TPU_PLE_HISTORY_ROW_KERNEL every gather of the [R, S, C] history
(the short conv's taps and final window, the checkpoint windows) goes through
`gather_history_rows` instead of an XLA gather. The history is pinned to HBM
as the kernel's input, so no XLA gather reads it from VMEM: XLA's TPU gather
over a VMEM-resident bf16[128,9,10240] history halts some compiled programs
with a TensorCore vector program error (upstream report pending).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs

LANES = 128
# Rows a grid step copies; 32 rows of 10240 bf16 make a 640 KiB output block.
BLOCK_ROWS = 32


def history_row_kernel_enabled() -> bool:
    return envs.VLLM_TPU_PLE_HISTORY_ROW_KERNEL


def gather_history_rows(history: jax.Array, rows: jax.Array) -> jax.Array:
    """``history.reshape(R * S, C)[rows]``, copied by DMA from an HBM history.

    ``rows`` holds flat row ids in [0, R * S) of any shape; the result has
    shape ``rows.shape + (C,)``. Ids must be in range: the copy does no
    clamping (every caller builds them from clamped request and position
    indices).
    """
    count, state_len, channels = history.shape
    assert channels % LANES == 0, channels
    flat = rows.reshape(-1).astype(jnp.int32)
    n = flat.shape[0]
    if n == 0 or count * state_len == 0:
        return jnp.zeros(rows.shape + (channels,), history.dtype)
    padded = pl.cdiv(n, BLOCK_ROWS) * BLOCK_ROWS
    # Padding ids copy row 0 into output rows that are sliced away below.
    flat = jnp.pad(flat, (0, padded - n))
    # One leading index per row: each row is whole (16, 128) tiles, so a row
    # copy is one tile-aligned DMA.
    source = history.reshape(count * state_len, channels // LANES, LANES)
    source = pltpu.with_memory_space_constraint(source, pltpu.HBM)

    def kernel(ids, src, out, sem):
        base = pl.program_id(0) * BLOCK_ROWS
        for j in range(BLOCK_ROWS):
            pltpu.make_async_copy(src.at[ids[base + j]], out.at[j], sem).start()
        for j in range(BLOCK_ROWS):
            pltpu.make_async_copy(src.at[0], out.at[j], sem).wait()

    out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(padded // BLOCK_ROWS,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(
                (BLOCK_ROWS, channels // LANES, LANES), lambda i, ids: (i, 0, 0)
            ),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(
            (padded, channels // LANES, LANES), history.dtype
        ),
        name="ple_history_rows",
    )(flat, source)
    return out[:n].reshape(rows.shape + (channels,))
