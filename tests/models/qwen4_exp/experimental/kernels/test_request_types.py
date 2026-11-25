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
"""Decode-prefix routing preserves ragged row order, padding and empty requests."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.kernels.request_types import (
    decode_prefix,
)


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
