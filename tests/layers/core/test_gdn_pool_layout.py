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

import pytest

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout


@pytest.mark.parametrize(
    ("ssm_bytes", "conv_bytes", "ssm_tokens", "conv_tokens"),
    [
        pytest.param(512 * 1024, 12 * 1024, 512, 16, id="p4"),
        pytest.param(1024 * 1024, 24 * 1024, 1024, 32, id="d2"),
    ],
)
def test_qwen35_layout_includes_conv_tile_padding(
    ssm_bytes: int,
    conv_bytes: int,
    ssm_tokens: int,
    conv_tokens: int,
):
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=1024,
    )

    assert layout.ssm_tokens == ssm_tokens
    assert layout.conv_tokens == conv_tokens
    assert layout.required_tokens == ssm_tokens + conv_tokens
    assert layout.required_bytes == (ssm_tokens + conv_tokens) * 1024


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("adapter_bridge", [False, True])
def test_pooled_prepares_inside_real_shard_map(
    monkeypatch, prepared, tp_size, adapter_bridge
):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput
    from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op
    from vllm_torchtpu.layers.core import gdn_attention

    devices = jax.devices("cpu")[:tp_size]
    if len(devices) < tp_size:
        pytest.skip("requires two CPU devices")
    mesh = jax.sharding.Mesh(np.array(devices), ("model",))
    projection = jnp.arange(4 * 512 * tp_size, dtype=jnp.bfloat16).reshape(
        4, 512 * tp_size
    )
    prepared_shapes = []

    def prepare_local(values):
        prepared_shapes.append(values.shape)
        return Conv1DInput(values.astype(jnp.float32)[:, None, :], values.dtype)

    pool = jnp.zeros((3, 256, 1, 4, 256), jnp.int8)
    specs, plans = [], []
    shard_map = jax.shard_map

    def record_map(*args, **kwargs):
        specs.append(kwargs["in_specs"][0])
        return shard_map(*args, **kwargs)

    def fused(inputs, b, a, conv, recurrent, *args, **kwargs):
        assert conv is None and recurrent is None
        assert kwargs["state_source"].shape == pool.shape
        plan = kwargs["state_plan"]
        plans.append(plan)
        assert plan.stride == 1
        assert plan.recurrent.nrows == 128
        assert plan.conv.row0 == 128 and plan.conv.nrows == 4
        assert plan.conv.view_dtype == jnp.bfloat16
        values = inputs.values[:, 0] if isinstance(inputs, Conv1DInput) else inputs
        dtype = inputs.output_dtype if isinstance(inputs, Conv1DInput) else inputs.dtype
        return kwargs["state_source"], values[:, :256].astype(dtype)

    monkeypatch.setattr(jax, "shard_map", record_map)
    monkeypatch.setattr(gdn_attention.gdn_v3_wrapper, "fused_conv1d_gdn", fused)
    entry = (
        gdn_attention_op.gdn_attention_pooled_core_tpu
        if adapter_bridge
        else gdn_attention.run_jax_gdn_attention_pooled
    )
    new_pool, output = entry(
        projection,
        jnp.zeros((4, 2 * tp_size)),
        jnp.zeros((4, 2 * tp_size)),
        pool,
        jnp.zeros((512 * tp_size, 1, 4)),
        None,
        jnp.zeros(2 * tp_size),
        jnp.zeros(2 * tp_size),
        jnp.array([1], jnp.int32),
        jnp.array([0, 4], jnp.int32),
        jnp.array([0, 1, 1], jnp.int32),
        jnp.array([4], jnp.int32),
        n_kq=tp_size,
        n_v=2 * tp_size,
        d_k=128,
        d_v=128,
        kernel_size=4,
        pool_block_tokens=256,
        mesh=mesh,
        prepare_input=prepare_local if prepared else None,
        recurrent_state_dtype=jnp.float32,
    )
    assert plans and len(specs) == 1
    assert specs[0] == P(None, "model")
    assert prepared_shapes == ([(4, 512)] if prepared else [])
    assert output.dtype == projection.dtype
    expected = (
        np.asarray(projection).reshape(4, tp_size, 512)[:, :, :256].reshape(4, -1)
    )
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(new_pool, pool)


@pytest.mark.parametrize(
    ("ssm_bytes", "conv_bytes", "token_bytes", "ssm_tokens", "conv_tokens"),
    [
        # Kimi-Linear TP8: the f32 KDA state (2**18 B) does not divide the
        # padded MLA token row (2**9 * 5 B); the last SSM row is padding.
        pytest.param(262144, 9216, 2560, 103, 4, id="kimi-linear-tp8"),
        # The conv tile no longer needs to divide the SSM region, so it may
        # exceed it in token count.
        pytest.param(3 * 1024, 12 * 1024, 1024, 3, 16, id="conv-wider-than-ssm"),
    ],
)
def test_layout_pads_states_to_token_rows(
    ssm_bytes: int,
    conv_bytes: int,
    token_bytes: int,
    ssm_tokens: int,
    conv_tokens: int,
):
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=token_bytes,
    )

    assert layout.ssm_tokens == ssm_tokens
    assert layout.conv_tokens == conv_tokens
    assert layout.ssm_tokens * token_bytes >= ssm_bytes
    assert layout.conv_tokens * token_bytes >= conv_bytes
    assert layout.required_tokens == ssm_tokens + conv_tokens
