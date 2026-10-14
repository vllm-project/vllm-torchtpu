# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured tiles for attention over a full pre-gathered prefill stage."""

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# (chip, query heads, head dim, top-k): tokens per DMA/compute batch.
# On v7x, 32 hides more input DMA latency than 16. A 64-token batch needs
# 66.23 MiB of VMEM, exceeding the measured 63.94 MiB available to the kernel.
PREFILL_BATCHES: dict[tuple[str, int, int, int], int] = {("7x", 64, 512, 512): 32}


def prefill_batch_size(chip: str, heads: int, dim: int, topk: int) -> int:
    key = (chip, heads, dim, topk)
    if key not in PREFILL_BATCHES:
        logger.warning_once("No measured pre-gathered SparseMLA tile for %s", key)
        return 16
    return PREFILL_BATCHES[key]
