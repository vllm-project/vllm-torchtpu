# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for the packed suffix layout boundary."""

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.packed_suffix import extract_packed_suffix

_suffix_op = pallas.jax_op("pallas::extract_packed_suffix", extract_packed_suffix)


@_suffix_op.register_fake
def _fake_suffix(packed: torch.Tensor, num_words: int) -> torch.Tensor:
    return packed.new_empty((1, num_words))


def packed_suffix(packed: torch.Tensor, num_words: int) -> torch.Tensor:
    """Return the trailing int32 words without fusing consumers into the input."""
    if packed.device.type != "tpu" or num_words == 0:
        return packed.flatten()[packed.numel() - num_words :].view(1, num_words)
    return _suffix_op(packed, num_words)
