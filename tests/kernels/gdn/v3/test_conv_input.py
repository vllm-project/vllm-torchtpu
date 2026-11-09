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
"""Independent cast/padding reference, including the Pallas interpreter."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3.conv_input import cast_conv_input


@pytest.mark.parametrize(
    "tokens,width",
    [
        (128, 128),
        (256, 256),
        (8, 384),
        (16, 384),
        (9, 384),
        (63, 257),
        (64, 512),
        (96, 512),
        (17, 128),
        (1, 129),
        (0, 128),
        (129, 129),
        (127, 255),
        (17, 0),
    ],
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
def test_cast_layout_and_fallback(monkeypatch, tokens, width, dtype):
    if jax.default_backend() == "cpu":
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
        monkeypatch.setattr(
            pl, "pallas_call", functools.partial(pl.pallas_call, interpret=True)
        )
    data = np.resize(
        np.array([-1.5, -0.0, 0.0, 0.5, 1.75], np.float32), (tokens, width)
    )
    raw = jnp.asarray(data, dtype)
    before = np.asarray(raw).copy()
    expected = np.asarray(raw, np.float32)[:, None, :]
    actual = cast_conv_input(raw)
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), expected.view(np.uint32)
    )
    np.testing.assert_array_equal(
        np.asarray(raw).view(np.uint16), before.view(np.uint16)
    )


def test_float16_uses_original_cast(monkeypatch):
    if jax.default_backend() == "cpu":
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)

    def unexpected_kernel(*args, **kwargs):
        pytest.fail("FP16 must use the original cast and layout")

    monkeypatch.setattr(pl, "pallas_call", unexpected_kernel)
    raw = jnp.ones((128, 128), dtype=jnp.float16)
    actual = cast_conv_input(raw)
    np.testing.assert_array_equal(np.asarray(actual), np.ones((128, 1, 128)))


@pytest.mark.parametrize("tokens", [8, 9, 16])
def test_packed_prefix_keeps_row_stride(monkeypatch, tokens):
    if jax.default_backend() == "cpu":
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
        monkeypatch.setattr(
            pl, "pallas_call", functools.partial(pl.pallas_call, interpret=True)
        )
    width, prefix = 640, 384
    data = np.arange(tokens * width, dtype=np.float32).reshape(tokens, width)
    data[:, prefix:] = np.nan
    raw = jnp.asarray(data, jnp.bfloat16)
    before = np.asarray(raw).copy()
    expected = np.asarray(raw[:, :prefix], np.float32)[:, None, :]
    actual = cast_conv_input(raw, qkv_width=prefix)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), expected.view(np.uint32)
    )
    np.testing.assert_array_equal(
        np.asarray(raw).view(np.uint16), before.view(np.uint16)
    )
