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
"""CPU coverage for zero-tail trimming and its layer cache identity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2_prefill import host, kernel, layer

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("logical,inter", [(256, 512), (2304, 2560)])
@pytest.mark.parametrize("height", [32, 64, 96, 128, 256])
@pytest.mark.parametrize("act_fn", ["silu", "silu_and_mul_with_clamp"])
def test_trim_preserves_full_row_quantization(height, act_fn, logical, inter):
    rng = np.random.default_rng(28)
    hidden, qb = 512, 512
    x = jnp.asarray(rng.integers(-3, 4, (height, hidden)), dtype=host.FP8)
    w1 = rng.integers(-3, 4, (hidden, 2 * inter)).astype(np.float32)
    w2 = rng.integers(-3, 4, (inter, hidden)).astype(np.float32)
    w1[:, logical:inter] = 0
    w1[:, inter + logical :] = 0
    w2[logical:] = 0
    w1, w2 = jnp.asarray(w1, host.FP8), jnp.asarray(w2, host.FP8)
    s1 = jnp.full((1, 2 * inter), 0.01)
    s2 = jnp.broadcast_to(
        0.01 * jnp.arange(1, inter // qb + 1)[:, None],
        (inter // qb, hidden),
    )
    row_scale = jnp.full((height, 1), 0.1)
    reads = []

    def read1(b, lo, hi):
        reads.append(("w1", lo, hi))
        return w1[b * qb : (b + 1) * qb, lo:hi]

    def read2(b, lo, hi, *, block_rows=qb):
        reads.append(("w2", block_rows))
        return w2[b * qb : b * qb + block_rows, lo:hi]

    args = (x, row_scale, read1, read2, s1, s2)
    full = kernel.expert_ffn_blockscale(*args, qb=qb, act_fn=act_fn)
    reads.clear()
    events = []
    trimmed = kernel.expert_ffn_blockscale(
        *args,
        qb=qb,
        act_fn=act_fn,
        logical_intermediate_size=logical,
        after_gate_up=lambda: events.append("gate_done"),
        after_down_dot=lambda n: events.append(n),
    )
    for actual, expected in zip(trimmed, full, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert events == ["gate_done", 0, 1]
    assert ("w1", logical, inter) not in reads
    assert ("w1", inter + logical, 2 * inter) not in reads
    assert [r[1] for r in reads if r[0] == "w2"] == (
        [qb] * (inter // qb - 1) + [logical % qb]
    ) * (hidden // kernel.FP4_OUTPUT_CHUNK)


@pytest.mark.parametrize("first,second", [(256, None), (None, 256)])
def test_layer_cache_separates_logical_intermediate(monkeypatch, first, second):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("d",))
    builds, maps = [], []

    def build(**kwargs):
        builds.append(kwargs["logical_intermediate_size"])
        return object()

    def shard_map(body, **kwargs):
        maps.append(body)
        return lambda *args: args[0]

    monkeypatch.setattr(layer, "build_fused_ep_moe_kernel", build)
    monkeypatch.setattr(jax, "shard_map", shard_map)
    monkeypatch.setattr(layer, "_LAYER_SM_CACHE", {})
    x = jnp.zeros((32, 512), jnp.bfloat16)
    w1 = jnp.zeros((1, 512, 1024), host.FP4)
    w2 = jnp.zeros((1, 512, 512), host.FP4)
    scales1 = jnp.ones((1, 1, 1024))
    scales2 = jnp.ones((1, 1, 512))
    gate = jnp.zeros((32, 1))
    for logical in (first, second, first):
        layer.fused_ep_moe_v2(
            x,
            w1,
            w2,
            scales1,
            scales2,
            gate,
            mesh=mesh,
            renormalize=True,
            capacity=128,
            topk=1,
            weight_format=host.WeightFormat.FP4,
            rhs_qb=512,
            logical_intermediate_size=logical,
        )
    assert builds == [first, second, first]
    assert len(maps) == 2


@pytest.mark.parametrize("logical", [0, -256, 255, 2048, 2816, 2304.5])
def test_invalid_trim_contract_rejected_before_device_setup(logical):
    with pytest.raises(ValueError, match="logical intermediate"):
        kernel._build_fused_ep_moe_kernel(
            g_local=48,
            capacity=128,
            hidden=5120,
            inter=2560,
            ep=8,
            weight_format=host.WeightFormat.FP4,
            rhs_qb=512,
            logical_intermediate_size=logical,
        )
