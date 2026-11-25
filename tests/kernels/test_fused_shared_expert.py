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

"""Fused FP8 shared expert: dispatch envelope, bitwise parity with the
unfused torch path (two quantized matmuls around SiluAndMulWithClamp) at the
decode shape, and agreement within FP8 noise at the prefill buckets."""

import jax
import jax.numpy as jnp
import pytest
import torch

from vllm_torchtpu.kernels.quantized_matmul import shared_expert as se

HIDDEN, INTER, LIMIT = 5120, 2304, 10.0


def _abstract(tokens=288, hidden=HIDDEN, inter=INTER):
    return (
        jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
        jax.ShapeDtypeStruct((hidden, 2 * inter), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((2 * inter,), jnp.float32),
        jax.ShapeDtypeStruct((inter, hidden), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((hidden,), jnp.float32),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens,expected",
    [(288, True), (512, True), (1024, True), (240, False), (2048, False)],
)
@pytest.mark.parametrize("generation", [-1, 6, 7])
def test_envelope(monkeypatch, tokens, expected, generation):
    monkeypatch.setattr(se, "get_tpu_version", lambda: generation)
    assert se.is_supported(*_abstract(tokens)) is (expected and generation == 7)


@pytest.mark.cpu_test
def test_envelope_rejects_other_dims(monkeypatch):
    monkeypatch.setattr(se, "get_tpu_version", lambda: 7)
    assert not se.is_supported(*_abstract(inter=2560))
    x, w13, s13, w2, s2 = _abstract()
    blockwise = jax.ShapeDtypeStruct((1, 40, 1, 2 * INTER), jnp.float32)
    assert not se.is_supported(x, w13, blockwise, w2, s2)


@pytest.fixture(scope="module")
def compiled():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    from vllm_torchtpu.layers.adapter.linear_common import (
        _get_quantized_matmul_op,
        fp8_swiglu_mlp,
        get_fp8_swiglu_mlp_op,
        quantized_matmul,
    )

    _get_quantized_matmul_op()
    get_fp8_swiglu_mlp_op(LIMIT)

    with set_current_vllm_config(VllmConfig()):
        act = SiluAndMulWithClamp(LIMIT)

    def both(x, w13, s13, w2, s2, routed, mask):
        # The served layer adds the masked, scaled routed output to the shared
        # expert's result, and XLA fuses that add with the down projection.
        routed = torch.where(mask[:, None], routed, 0) * 1.5
        reference = quantized_matmul(
            act.forward_native(quantized_matmul(x, w13, s13)), w2, s2
        )
        actual = fp8_swiglu_mlp(x, w13, s13, w2, s2, LIMIT)
        return reference, actual, reference + routed, actual + routed

    return torch.compile(both, backend="tpu", fullgraph=True)


def _weights(seed):
    gen = torch.Generator().manual_seed(seed)
    w13 = torch.randn(HIDDEN, 2 * INTER, generator=gen).mul(64).clamp(-448, 448)
    w2 = torch.randn(INTER, HIDDEN, generator=gen).mul(64).clamp(-448, 448)
    s13 = torch.rand(2 * INTER, generator=gen).add(0.5).mul(1e-3)
    s2 = torch.rand(HIDDEN, generator=gen).add(0.5).mul(1e-3)
    return [
        t.to(torch.float8_e4m3fn).to("tpu") if t.dim() == 2 else t.to("tpu")
        for t in (w13, s13, w2, s2)
    ]


@pytest.mark.parametrize(
    "tokens,pattern,seed",
    [
        (288, "random", 0),
        (288, "random", 1),
        (288, "mixed", 2),
        (288, "zero", 3),
        (256, "random", 4),
    ],
)
def test_fused_shared_expert_is_bitwise_equal(compiled, tokens, pattern, seed):
    graph = str(
        jax.make_jaxpr(lambda *a: se.fp8_swiglu_mlp(*a, limit=LIMIT))(
            *_abstract(tokens)
        )
    )
    assert ("fused_shared_expert_fp8" in graph) is (tokens == 288)
    gen = torch.Generator().manual_seed(100 + seed)
    x = torch.randn(tokens, HIDDEN, generator=gen)
    if pattern == "zero":
        x.zero_()
    elif pattern == "mixed":
        x *= torch.rand(tokens, 1, generator=gen).mul(8).sub(4).exp()
        x[:16] = 0
    routed = torch.randn(tokens, HIDDEN, generator=gen).mul(3).to(torch.bfloat16)
    mask = torch.rand(tokens, generator=gen) > 0.05
    outs = compiled(
        x.to(torch.bfloat16).to("tpu"),
        *_weights(seed),
        routed.to("tpu"),
        mask.to("tpu"),
    )
    reference, actual, ref_layer, actual_layer = (t.cpu() for t in outs)
    assert torch.isfinite(actual).all()
    assert torch.equal(reference.view(torch.int16), actual.view(torch.int16))
    assert torch.equal(ref_layer.view(torch.int16), actual_layer.view(torch.int16))


@pytest.mark.parametrize("tokens,seed", [(512, 5), (1024, 6)])
def test_fused_shared_expert_prefill_within_tolerance(tokens, seed):
    """At the prefill buckets the kernel accumulates its GEMMs in its own
    order, so it matches the unfused form the push MoE op otherwise runs
    (fp8_swiglu_gate_up + fp8_swiglu_down) to FP8 re-quantization noise
    (rel L2 0.9-1.6% measured), not bitwise."""
    if se.get_tpu_version() != 7:
        pytest.skip("the prefill rows of SHARED_EXPERT_TUNING are v7 only")
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)

    def fp8_weight(key, shape):
        w = jnp.clip(jax.random.normal(key, shape) * 64, -448, 448)
        return w.astype(jnp.float8_e4m3fn)

    w13 = fp8_weight(keys[0], (HIDDEN, 2 * INTER))
    w2 = fp8_weight(keys[1], (INTER, HIDDEN))
    s13 = (jax.random.uniform(keys[2], (2 * INTER,)) + 0.5) * 1e-3
    s2 = (jax.random.uniform(keys[3], (HIDDEN,)) + 0.5) * 1e-3
    x = jax.random.normal(keys[4], (tokens, HIDDEN)).astype(jnp.bfloat16)
    assert se.is_supported(x, w13, s13, w2, s2)
    fused = jax.jit(lambda *a: se.fused_shared_expert(*a, limit=LIMIT))
    unfused = jax.jit(
        lambda x, w13, s13, w2, s2: se.fp8_swiglu_down(
            se.fp8_swiglu_gate_up(x, w13, s13, limit=LIMIT), w2, s2
        )
    )
    actual = fused(x, w13, s13, w2, s2).astype(jnp.float32)
    reference = unfused(x, w13, s13, w2, s2).astype(jnp.float32)
    assert bool(jnp.isfinite(actual).all())
    rel = jnp.linalg.norm(actual - reference) / jnp.linalg.norm(reference)
    assert float(rel) < 3e-2
