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
"""Unit tests for head-major-to-token-major RPA output quantization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import output_quant
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

_FP8 = jnp.float8_e4m3fn


def _head_major(
    kv_heads: int, tokens: int, groups: int, head_dim: int, seed: int = 0
) -> jax.Array:
    pairs = -(-groups // 2)
    k0, k1 = jax.random.split(jax.random.PRNGKey(seed))
    mag = jax.random.uniform(k0, (1, tokens, 1, 1, 1), minval=0.01, maxval=4.0)
    o = jax.random.normal(k1, (kv_heads, tokens, pairs, 2, head_dim)) * mag
    return o.astype(jnp.bfloat16).at[:, 1].set(0)


def _reference(o: jax.Array, groups: int, head_dim: int) -> tuple[jax.Array, jax.Array]:
    kv_heads, tokens, pairs, packing, _ = o.shape
    rows = o.reshape(kv_heads, tokens, pairs * packing, -1)[:, :, :groups, :head_dim]
    return quantize_tensor(rows.swapaxes(0, 1).reshape(tokens, -1), _FP8)


_CASES = [
    pytest.param(8, 64, 2, 256, id="even-groups"),
    pytest.param(2, 32, 8, 256, id="wide-groups"),
    pytest.param(4, 20, 2, 128, id="padded-tokens"),
    pytest.param(3, 32, 3, 128, id="padding-head"),
]


@pytest.mark.parametrize("kv_heads,tokens,groups,head_dim", _CASES)
def test_lowers_for_tpu7x(kv_heads: int, tokens: int, groups: int, head_dim: int):
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    pairs = -(-groups // 2)
    o = jax.ShapeDtypeStruct((kv_heads, tokens, pairs, 2, head_dim), jnp.bfloat16)
    with jax.sharding.use_abstract_mesh(mesh):
        traced = jax.jit(
            lambda x: output_quant.quantize_head_major_output(
                x,
                quant_dtype=_FP8,
                num_q_heads_per_kv_head=groups,
                head_dim=head_dim,
                interpret=False,
            )
        ).trace(o)
        hlo = traced.lower(lowering_platforms=("tpu",)).as_text()
    q_info, scale_info = traced.out_info
    assert q_info.shape == (tokens, kv_heads * groups * head_dim)
    assert q_info.dtype == _FP8
    assert scale_info.shape == (tokens, 1)
    assert scale_info.dtype == jnp.float32
    assert "tpu_custom_call" in hlo or "custom_call" in hlo


def test_block_tokens_scaling():
    assert output_quant.block_tokens(16384, 16, vmem_limit_bytes=32 << 20) == 256
    assert output_quant.block_tokens(16384, 32, vmem_limit_bytes=32 << 20) == 128
    assert output_quant.block_tokens(48, 16, vmem_limit_bytes=32 << 20) == 16
    assert output_quant.block_tokens(64, 4, vmem_limit_bytes=32 << 20) == 64


def test_rejects_unaligned_heads():
    with jax.default_device(jax.devices("cpu")[0]):
        o = jnp.zeros((2, 16, 1, 2, 192), jnp.bfloat16)
    with pytest.raises(ValueError, match="lane-aligned"):
        output_quant.quantize_head_major_output(
            o,
            quant_dtype=_FP8,
            num_q_heads_per_kv_head=2,
            head_dim=192,
            interpret=False,
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "kv_heads,tokens,groups,head_dim",
    [*_CASES, pytest.param(4, 128, 4, 256, id="gemma4-full")],
)
def test_matches_quantize_tensor_tpu(
    kv_heads: int, tokens: int, groups: int, head_dim: int
):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU runtime")
    o = _head_major(kv_heads, tokens, groups, head_dim)
    q, scale = output_quant.quantize_head_major_output(
        o,
        quant_dtype=_FP8,
        num_q_heads_per_kv_head=groups,
        head_dim=head_dim,
        interpret=False,
    )
    ref_q, ref_scale = _reference(o, groups, head_dim)
    assert q.dtype == _FP8 and scale.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(scale), np.asarray(ref_scale), rtol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(q.view(jnp.uint8)), np.asarray(ref_q.view(jnp.uint8))
    )
