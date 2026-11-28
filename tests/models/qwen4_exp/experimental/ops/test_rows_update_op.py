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
"""The model-owned rows update replaces the leading rows, keeps the rest and
casts the rows to the buffer dtype; its fake returns the buffer's metadata."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.qwen4_exp.experimental.ops.rows_update_op import (
    _fake_rows_update,
    rows_update_jax,
)


@pytest.mark.parametrize(
    "rows,width,count,buf_dtype,rows_dtype",
    [
        (64, 32, 16, np.int32, np.int32),  # leading rows only
        (64, 48, 64, np.int32, np.int32),  # the whole buffer
        (32, 128, 5, np.float32, np.int32),  # rows cast to the buffer dtype
    ],
)
def test_rows_update_replaces_leading_rows(rows, width, count, buf_dtype, rows_dtype):
    rng = np.random.default_rng(rows + width + count)
    buf = rng.integers(0, 1000, (rows, width)).astype(buf_dtype)
    new = rng.integers(-1, 5000, (count, width)).astype(rows_dtype)
    got = rows_update_jax(jnp.asarray(buf), jnp.asarray(new))
    want = buf.copy()
    want[:count] = new.astype(buf_dtype)
    assert got.dtype == jnp.dtype(buf_dtype) and got.shape == buf.shape
    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("dtype", [torch.int32, torch.bfloat16])
def test_fake_rows_update_matches_buffer(dtype):
    buf = torch.zeros(64, 32, dtype=dtype)
    out = _fake_rows_update(buf, torch.zeros(8, 32, dtype=torch.int64))
    assert out.shape == buf.shape and out.dtype == buf.dtype
