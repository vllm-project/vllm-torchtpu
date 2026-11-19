# SPDX-License-Identifier: Apache-2.0
"""Decode-prefix routing preserves ragged row order, padding and empty requests."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.request_types import decode_prefix


@pytest.mark.parametrize(
    "lengths,count",
    [
        ([1, 1, 128, 0], 2),
        ([1, 4, 1, 0], 1),
        ([128, 1], 0),
        ([0, 1], 0),
        ([0, 0], 0),
        ([1, 1], 2),
        ([128, 129], 0),
        ([], 0),
    ],
)
def test_decode_prefix_counts_only_leading_one_token_requests(lengths, count):
    starts = jnp.asarray(np.r_[0, np.cumsum(lengths)], jnp.int32)
    assert int(decode_prefix(starts)) == count
