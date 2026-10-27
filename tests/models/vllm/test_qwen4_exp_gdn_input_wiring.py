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
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput
from vllm_torchtpu.models.vllm.qwen4_exp import gdn


@pytest.mark.parametrize("tokens", [1, 8, 127, 257])
def test_small_or_unaligned_qwen_input_is_unchanged(tokens):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    qkv = jnp.zeros((tokens, 384), jnp.bfloat16)
    assert gdn.prepare_gdn_input(qkv, 1, mesh) is qkv


@pytest.mark.parametrize(
    "tokens,capacity", [(128, 128), (256, 256), (128, 1), (2048, 9), (128, 2)]
)
def test_shape_policy_prepares_chunked_buckets_only(monkeypatch, tokens, capacity):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    qkv = jnp.ones((tokens, 384), jnp.bfloat16)
    monkeypatch.setattr(
        gdn, "compact_qkv_input", lambda x: x.astype(jnp.float32)[:, None, :]
    )
    prepared = gdn.prepare_gdn_input(qkv, capacity, mesh)
    if tokens <= capacity:
        assert prepared is qkv
    else:
        assert isinstance(prepared, Conv1DInput)
        assert prepared.output_dtype == qkv.dtype
        assert prepared.values.shape == (tokens, 1, 384)
        np.testing.assert_array_equal(prepared.values, 1)


@pytest.mark.tpu_test
def test_qwen_prepared_input_uses_real_torch_bridge_and_preserves_state():
    from types import SimpleNamespace as NS

    import torch
    from torch_tpu._internal import compile as tpu_compile
    from torch_tpu._internal import sync

    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    config = NS(cache_config=NS(block_size=1664))
    operators = []
    with set_vllm_model_wrapper_context(mesh=mesh, vllm_config=config):
        for cls in (VllmGatedDeltaNetAttention, gdn.Qwen4ExpGatedDeltaNetAttention):
            # The constructor and checkpoint loader are outside this bridge test.
            attn = cls.__new__(cls)
            attn.num_k_heads, attn.num_v_heads, attn.tp_size = 16, 48, 1
            attn.gqa_interleaved_layout = False
            attn.head_k_dim = attn.head_v_dim = 128
            attn.conv_kernel_size, attn.num_spec = 4, 0
            attn.model_config = NS(
                dtype=torch.bfloat16, architecture="Qwen4ExpForCausalLM"
            )
            attn.cache_config = NS(
                mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="bfloat16"
            )
            operators.append(
                torch.compile(
                    attn._build_pooled_gdn_op(),
                    backend=tpu_compile.TpuBackend(),
                    fullgraph=True,
                    dynamic=False,
                )
            )

    rng = torch.Generator(device="cpu").manual_seed(54)

    def normal(shape, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=rng) * 0.02).to(dtype).to("tpu")

    weight, bias = normal((10240, 1, 4)), normal((10240,))
    a_log, dt_bias = normal((48,), torch.float32), normal((48,), torch.float32)
    pools = [
        torch.zeros((2, 1664, 1, 4, 256), dtype=torch.float8_e4m3fn).to("tpu")
        for _ in operators
    ]
    context = 0
    for tokens in (128, 1):
        qkv, b, a = normal((tokens, 10240)), normal((tokens, 48)), normal((tokens, 48))
        metadata = [
            torch.tensor(v, dtype=torch.int32).to("tpu")
            for v in (
                [1],
                [0, tokens],
                [0, 1, 1] if tokens > 1 else [1, 1, 1],
                [context + tokens],
            )
        ]
        outputs = [
            op(qkv, b, a, pool, weight, bias, a_log, dt_bias, *metadata)
            for op, pool in zip(operators, pools)
        ]
        sync.synchronize([*outputs, *pools], wait=True)
        actual, expected = [x.cpu() for x in outputs]
        assert actual.dtype == expected.dtype == torch.bfloat16
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            pools[0].cpu().view(torch.uint8),
            pools[1].cpu().view(torch.uint8),
            rtol=0,
            atol=0,
        )
        context += tokens
