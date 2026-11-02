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
"""Admission and dtype boundary regression tests for retained decode preparation."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import retained_preparation as prep


def test_retained_preparation_is_limited_to_validated_envelope(monkeypatch):
    arr = jax.ShapeDtypeStruct
    args = [
        arr((16, 12288), jnp.bfloat16),
        arr((16, 512), jnp.bfloat16),
        arr((16, 640), jnp.bfloat16),
        arr((256,), jnp.bfloat16),
        arr((256,), jnp.bfloat16),
        arr((128,), jnp.bfloat16),
        arr((256,), jnp.float32),
        arr((67584, 128), jnp.bfloat16),
        arr((32,), jnp.int32),
        arr((32, 41), jnp.int32),
        arr((20, 1664, 1, 4, 256), jnp.float8_e4m3fn),
    ]
    options = dict(ratio=4, topk=2048, rotary_dim=64)
    monkeypatch.setattr(
        prep,
        "_hardware",
        lambda: SimpleNamespace(chip_version=pltpu.ChipVersion.TPU_7X),
    )
    assert prep.use_retained_preparation(*args, **options)
    for index, value in [
        (0, arr((32, 12288), jnp.bfloat16)),
        (6, None),
        (6, arr((256,), jnp.bfloat16)),
        (8, arr((16,), jnp.int32)),
        (9, arr((32, 2), jnp.int32)),
        (10, arr((20, 1664, 1, 4, 256), jnp.bfloat16)),
    ]:
        changed = args.copy()
        changed[index] = value
        assert not prep.use_retained_preparation(*changed, **options)
    monkeypatch.setattr(prep, "_hardware", lambda: None)
    assert not prep.use_retained_preparation(*args, **options)
