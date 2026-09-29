# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""TPU input padding and token bucketing helpers."""

import bisect

from vllm_torchtpu.llm_tpu_core.logger import init_logger

logger = init_logger(__name__)

MIN_NUM_SEQS = 8


def get_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    """Calculates power-of-two request padding ladder."""
    logger.info("Preparing request paddings:")
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: list[int] = []
    num = max(MIN_NUM_SEQS, min_req_size)
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        logger.info("    %d", num)
        num = get_padded_num_reqs_with_upper_limit(num + 1, max_req_size)
    return paddings


def get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    """Rounds up to next power-of-two bounded by upper limit."""
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def get_padded_token_len(paddings: list[int], x: int) -> int:
    """Returns the first element in paddings list greater or equal to x."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


def get_padded_num_kv_cache_update_slices(
    num_tokens: int, max_num_reqs: int, page_size: int
) -> int:
    """Calculates the padded number of KV cache update slices."""
    padded_num_slices = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = min(padded_num_slices, num_tokens)
    return padded_num_slices


def prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)."""
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


def get_num_slices_per_kv_cache_update_block(page_size_bytes: int) -> int:
    """Finds the optimum number of slices to copy per Pallas program instance."""
    vmem_limit = 16 * 1024 * 1024
    num_slices_per_block = vmem_limit // page_size_bytes
    assert num_slices_per_block > 0, "Number of slices should be positive"
    num_slices_per_block = prev_power_of_2(num_slices_per_block)
    if num_slices_per_block > 64:
        num_slices_per_block = 64
    return num_slices_per_block
