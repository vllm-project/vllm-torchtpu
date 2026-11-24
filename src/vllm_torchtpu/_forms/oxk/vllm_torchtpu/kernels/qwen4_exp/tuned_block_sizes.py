# SPDX-License-Identifier: Apache-2.0
"""Measured QSA attention and cache I/O tile choices."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# (chip, physical page shape, carrier dtype, ring elements, tag elements).
# B8/B128 exact-byte and native XProf checks validate eight outstanding DMAs
# per tile for these BF16 ring/int32 tag regions. B128 state I/O is 27 us
# on v7, versus 236 us for the original four serial region kernels.
RING_TAG_DMA_BATCHES = {
    (
        pltpu.ChipVersion.TPU_7X,
        (1664, 1, 4, 256),
        jnp.dtype(jnp.float8_e4m3fn),
        1024,
        16,
    ): 8,
    (pltpu.ChipVersion.TPU_7X, (896, 1, 2, 256), jnp.dtype(jnp.bfloat16), 1024, 16): 8,
}

# (chip, physical page shape, carrier dtype, packed row shape) -> DMA batch.
# Eight concurrent decode-row DMAs preserve ordered writes across packets.
# Native FP8 B128 writer latency falls from 72.39 us to 10.48 us; contiguous
# prefill tiles retain one bulk DMA. Duplicate destinations use serial DMAs.
MAIN_KV_DMA_BATCHES = {
    (
        pltpu.ChipVersion.TPU_7X,
        (1664, 1, 4, 256),
        jnp.dtype(jnp.float8_e4m3fn),
        (1, 4, 256),
    ): 8,
    (
        pltpu.ChipVersion.TPU_7X,
        (896, 1, 2, 256),
        jnp.dtype(jnp.bfloat16),
        (1, 2, 256),
    ): 8,
}


def get_main_kv_dma_batch(pool: jax.Array, rows: jax.Array) -> int:
    """Select measured row concurrency, preserving serial writes on misses."""
    key = (
        pltpu.get_tpu_info().chip_version,
        pool.shape[1:],
        jnp.dtype(pool.dtype),
        rows.shape[1:],
    )
    if key not in MAIN_KV_DMA_BATCHES:
        logger.warning_once("QSA uses serial main KV DMA tiles for key: %s", key)
        return 1
    return MAIN_KV_DMA_BATCHES[key] if rows.dtype == pool.dtype else 1


def get_ring_tag_dma_batch(
    pool: jax.Array, requests: int, *, ring_elements: int, tag_elements: int
) -> int:
    """Use a measured DMA batch that divides the request count."""
    key = (
        pltpu.get_tpu_info().chip_version,
        pool.shape[1:],
        jnp.dtype(pool.dtype),
        ring_elements,
        tag_elements,
    )
    override = envs.VLLM_TPU_QSA_RING_DMA_BATCH
    if override:
        # The largest power-of-two divisor of the request rows up to it.
        batch = 1 << (override.bit_length() - 1)
        while requests % batch:
            batch //= 2
        return batch
    if key not in RING_TAG_DMA_BATCHES:
        logger.warning_once("QSA uses single-request state DMA tiles for key: %s", key)
        return 1
    batch = RING_TAG_DMA_BATCHES[key]
    return batch if requests % batch == 0 else 1
