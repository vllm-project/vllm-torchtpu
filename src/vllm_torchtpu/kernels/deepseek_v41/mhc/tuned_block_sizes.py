# SPDX-License-Identifier: Apache-2.0

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# (tokens, streams, hidden size). On the kernel's TPU7x envelope, keeping the
# BF16 normalized value in VMEM saves its store/reload before FP8 quantization.
# Paired 288-row mHC + input-projection profiles measured 27.7 -> 26.6 us.
VMEM_NORMALIZED_VALUE_SHAPES = frozenset({(288, 4, 5120)})


def normalized_value_in_vmem(tokens: int, streams: int, hidden_size: int) -> bool:
    key = (tokens, streams, hidden_size)
    if key in VMEM_NORMALIZED_VALUE_SHAPES:
        return True
    logger.warning_once(
        "mHC normalized-value output tuning miss for TPU7x/BF16/HBM-gates "
        "(tokens, streams, hidden_size)=%s; retaining HBM output",
        key,
    )
    return False
