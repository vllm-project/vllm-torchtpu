# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Metadata-only checks for sorted FP8 preparation eligibility."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import prepare


def descriptors(tokens=288, width=128, key_dtype=jnp.bfloat16):
    banks = (jax.ShapeDtypeStruct((512, 128), jnp.int32),) * 2
    records = jax.ShapeDtypeStruct((tokens,), jnp.int32)
    keys = jax.ShapeDtypeStruct((tokens, width), key_dtype)
    return banks, records, keys


@pytest.mark.parametrize("tokens", [128, 256, 288, 384, 512])
def test_verified_shapes_supported(monkeypatch, tokens):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert prepare.supports_sorted_fp8(*descriptors(tokens), 256)


@pytest.mark.parametrize("tokens", [0, 1, 127, 129, 255, 287, 289, 513, 1024])
def test_unverified_shapes_fallback(monkeypatch, tokens):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert not prepare.supports_sorted_fp8(*descriptors(tokens), 256)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_non_tpu_fallback(monkeypatch, backend):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    assert not prepare.supports_sorted_fp8(*descriptors(), 256)


@pytest.mark.parametrize(
    "change", ["width", "dtype", "records", "page", "banks", "bank_dtype"]
)
def test_layout_contract_fallback(monkeypatch, change):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    banks, records, keys = descriptors()
    page = 256
    if change == "width":
        keys = jax.ShapeDtypeStruct((288, 64), jnp.bfloat16)
    elif change == "dtype":
        keys = jax.ShapeDtypeStruct((288, 128), jnp.float32)
    elif change == "records":
        records = jax.ShapeDtypeStruct((287,), jnp.int32)
    elif change == "page":
        page = 128
    elif change == "banks":
        banks = banks[:1]
    elif change == "bank_dtype":
        banks = (jax.ShapeDtypeStruct((512, 128), jnp.float32),) * 2
    assert not prepare.supports_sorted_fp8(banks, records, keys, page)
