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
"""gmm_v2 (the fused MoE GMM) traces and lowers for TPU on the installed JAX.

kernel_main asks emit_pipeline for DMA `priority` 1, which only an internal
Pallas build accepts; it goes through kernels/pallas_compat.py. The cases
cover a single k tile (no accumulator scratch) and several, quantized
weights and a fused activation.
"""
import contextlib
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
from jax._src.pallas.mosaic import tpu_info

from vllm_torchtpu.kernels import pallas_compat
from vllm_torchtpu.kernels.megablox import gmm_v2

pytestmark = pytest.mark.cpu_test

_BF16 = jnp.bfloat16
_TILES_K256 = gmm_v2.TileSizes(tile_m=128,
                               tile_k=256,
                               tile_n=256,
                               bucket_base=64)


@contextlib.contextmanager
def _tpu_info_on_cpu():
    """Answer TPU7x for the CPU device (as tests/kernels/test_pallas_compat.py
    does), so gmm_v2 can size its tiles."""

    def tpu7x():
        return tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)

    tpu_info.get_tpu_info.cache_clear()
    jax.clear_caches()
    try:
        with mock.patch.dict(tpu_info.registry, {"cpu": tpu7x}):
            yield
    finally:
        tpu_info.get_tpu_info.cache_clear()
        jax.clear_caches()


def _sds(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


# name -> (size_m, size_k, size_n, rhs dtype, gmm_v2 keyword arguments)
_CASES = {
    "bf16_single_k": (128, 512, 256, _BF16, {}),
    "bf16_multi_k": (128, 1024, 256, _BF16, dict(tile_info=_TILES_K256)),
    "bf16_multi_k_tail": (128, 640, 256, _BF16, dict(tile_info=_TILES_K256)),
    "fp8_rhs_scale": (256, 2048, 512, jnp.float8_e4m3fn, dict(rhs_scale=1)),
    "int8_rhs_scale_multi_k":
    (128, 1024, 256, jnp.int8, dict(rhs_scale=4, tile_info=_TILES_K256)),
    "fused_silu_bias":
    (128, 512, 512, _BF16, dict(fuse_act="silu", rhs_bias=True)),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_gmm_v2_lowers_for_tpu(case, monkeypatch):
    size_m, size_k, size_n, rhs_dtype, kwargs = _CASES[case]
    num_groups = 8
    kwargs = dict(kwargs)
    if "rhs_scale" in kwargs:
        kwargs["rhs_scale"] = _sds(
            (num_groups, kwargs["rhs_scale"], 1, size_n), jnp.float32)
    if kwargs.pop("rhs_bias", False):
        kwargs["rhs_bias"] = _sds((num_groups, 1, size_n), _BF16)
    priorities = []
    emit_pipeline = pallas_compat.emit_pipeline

    def recording_emit_pipeline(*args, priority=0, **kw):
        priorities.append(priority)
        return emit_pipeline(*args, priority=priority, **kw)

    monkeypatch.setattr(pallas_compat, "emit_pipeline",
                        recording_emit_pipeline)
    with _tpu_info_on_cpu():
        traced = gmm_v2.gmm_v2.trace(_sds((size_m, size_k), _BF16),
                                     _sds((num_groups, size_k, size_n),
                                          rhs_dtype),
                                     _sds((num_groups, ), jnp.int32),
                                     maybe_quantize_lhs=False,
                                     **kwargs)
        lowered = traced.lower(lowering_platforms=("tpu", ))
    assert "tpu_custom_call" in lowered.as_text()
    # The pipeline still asks for priority 1 (forwarded where supported).
    assert priorities == [1], priorities
