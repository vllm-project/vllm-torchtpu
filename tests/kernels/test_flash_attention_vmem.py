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

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.flash_attention.kernel import (SegmentIds,
                                                          flash_attention)

_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def test_k3_vision_shape_compiles_with_default_block_selection() -> None:
    """K3's large vision-attention shape must use the tiled kernel."""
    if not any(device.platform == "tpu" for device in jax.devices()):
        pytest.skip("requires a TPU compiler")

    batch_size = 1
    num_heads = 12
    seq_len = 44032
    head_dim = 128
    qkv = jax.ShapeDtypeStruct(
        (batch_size, num_heads, seq_len, head_dim),
        jnp.bfloat16,
    )
    segment = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)
    flash_attention.lower(
        qkv,
        qkv,
        qkv,
        segment_ids=SegmentIds(q=segment, kv=segment),
        causal=False,
        sm_scale=1.0,
        vmem_limit_bytes=_VMEM_LIMIT_BYTES,
    ).compile()
