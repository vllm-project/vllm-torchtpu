# SPDX-License-Identifier: Apache-2.0
import bisect

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Smallest output size
MIN_NUM_SEQS = 8


def _get_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    logger.info("Preparing request paddings:")
    # assert min_req_size is power of 2
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: list = []
    num = max(MIN_NUM_SEQS, min_req_size)
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        logger.info("    %d", num)
        num = _get_padded_num_reqs_with_upper_limit(num + 1, max_req_size)
    return paddings


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """Return the first element in paddings list greater or equal to x."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


def _get_padded_num_kv_cache_update_slices(num_tokens: int, max_num_reqs: int,
                                           page_size: int) -> int:
    """Calculates the padded number of KV cache update slices to avoid
    recompilation."""
    # NOTE(chengjiyao): let's say R_i is the token num for i-th request,
    # so it occupies most 2 + R_i // page_size pages. The total maximum
    # possible number of pages needed is sum(2 + R_i // page_size), which
    # is <= 2 * max_num_reqs + sum(R_i) // page_size
    # = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = min(padded_num_slices, num_tokens)
    return padded_num_slices


def _prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


def _get_num_slices_per_kv_cache_update_block(page_size_bytes: int) -> int:
    """Find the optimum number of slices to copy per Pallas program instance.

    Increasing the number of slices copied in one instance of the kernel program
    will increase HBM bandwidth utilization via more in-flight DMAs.

    However, it will also use more VMEM, and experimentally, we observed
    performance regression at 128 slices on v6e, likely due to running
    out of scalar registers. Thus this function will limit the number of
    slices to 64.
    """
    # The default vmem_limit_bytes of a pallas kernel is 32MB. Here we
    # calculate num_slices_per_block based on 16MB in case any register spills.
    vmem_limit = 16 * 1024 * 1024
    num_slices_per_block = vmem_limit // page_size_bytes
    assert num_slices_per_block > 0, "Number of slices should be positive"
    num_slices_per_block = _prev_power_of_2(num_slices_per_block)
    if num_slices_per_block > 64:
        num_slices_per_block = 64
    return num_slices_per_block
