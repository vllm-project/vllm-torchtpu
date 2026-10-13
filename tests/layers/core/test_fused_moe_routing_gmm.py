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
"""fused_moe_func on the routing_moe kernel (TPU_MOE_ROUTING_GMM)."""

import functools
import importlib.metadata
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.core import fused_moe_gmm

# Qwen3-30B-A3B MoE shape: hidden 2048, moe_intermediate 768, top-8.
HIDDEN = 2048
INTER = 768
TOPK = 8

_FAKE_TPU_INFO = types.SimpleNamespace(
    smem_capacity_bytes=1 << 20, vmem_capacity_bytes=64 << 20
)


def _weights(num_experts, dtype=jnp.bfloat16, hidden=HIDDEN, inter=INTER):
    w1 = jax.ShapeDtypeStruct((num_experts, hidden, 2 * inter), dtype)
    w2 = jax.ShapeDtypeStruct((num_experts, inter, hidden), dtype)
    return w1, w2


def _can_use(w1, w2, w1_scale=None, w2_scale=None, **overrides):
    kwargs = dict(
        hidden_states=jax.ShapeDtypeStruct((64, HIDDEN), jnp.bfloat16),
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=None,
        w2_bias=None,
        activation="silu",
        topk=TOPK,
        rhs_quant_dtype=None,
    )
    kwargs.update(overrides)
    return fused_moe_gmm.can_use_routing_gmm(**kwargs)


@pytest.fixture
def fake_tpu(monkeypatch):
    monkeypatch.setattr(fused_moe_gmm.pltpu, "get_tpu_info", lambda: _FAKE_TPU_INFO)


def test_routing_gmm_env_defaults_on(monkeypatch):
    monkeypatch.delenv("TPU_MOE_ROUTING_GMM", raising=False)
    assert envs.environment_variables["TPU_MOE_ROUTING_GMM"]() is True


@pytest.mark.parametrize(
    ("jax_version", "libtpu_version", "supported"),
    [
        ("0.12.0.dev20261007", "0.0.50", True),
        ("0.12.1", "0.0.51", True),
        ("0.11.2", "0.0.50", False),
        ("0.12.0.dev20261006", "0.0.50", False),
        ("0.12.0.dev20261007", "0.0.47", False),
        ("0.12.0.dev20261007", None, False),
    ],
)
def test_routing_gmm_runtime_gate(monkeypatch, jax_version, libtpu_version, supported):
    def version(name):
        assert name == "libtpu"
        if libtpu_version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return libtpu_version

    monkeypatch.setattr(jax, "__version__", jax_version)
    monkeypatch.setattr(importlib.metadata, "version", version)
    assert fused_moe_gmm.routing_gmm_runtime_supported() is supported
    monkeypatch.setenv("TPU_MOE_ROUTING_GMM", "1")
    assert fused_moe_gmm._resolve_routing_gmm() is supported


def test_routing_gmm_env_off_forces_gmm_v2(monkeypatch):
    monkeypatch.setattr(fused_moe_gmm, "routing_gmm_runtime_supported", lambda: True)
    monkeypatch.setenv("TPU_MOE_ROUTING_GMM", "0")
    assert fused_moe_gmm._resolve_routing_gmm() is False


def test_routing_gmm_admits_bf16_and_per_channel_fp8(fake_tpu):
    assert _can_use(*_weights(16))
    w1, w2 = _weights(16, jnp.float8_e4m3fn)
    w1_scale = jax.ShapeDtypeStruct((16, 1, 1, 2 * INTER), jnp.float32)
    w2_scale = jax.ShapeDtypeStruct((16, 1, 1, HIDDEN), jnp.float32)
    assert _can_use(w1, w2, w1_scale, w2_scale)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(w1_bias=jax.ShapeDtypeStruct((16, 1, 2 * INTER), jnp.bfloat16)),
        dict(activation="situ:1.0:none"),
        dict(rhs_quant_dtype=jnp.int4),
        dict(hidden_states=jax.ShapeDtypeStruct((64, HIDDEN), jnp.float32)),
        # 64K tokens x top-8 overflows the SMEM routing metadata.
        dict(hidden_states=jax.ShapeDtypeStruct((1 << 16, HIDDEN), jnp.bfloat16)),
    ],
    ids=["bias", "situ", "packed", "f32_activations", "too_many_rows"],
)
def test_routing_gmm_rejects_unsupported_layer(fake_tpu, overrides):
    assert not _can_use(*_weights(16), **overrides)


def test_routing_gmm_rejects_unsupported_weights(fake_tpu):
    # Block-scaled fp8: two K blocks per scale row.
    w1, w2 = _weights(16, jnp.float8_e4m3fn)
    w1_scale = jax.ShapeDtypeStruct((16, 2, 1, 2 * INTER), jnp.float32)
    w2_scale = jax.ShapeDtypeStruct((16, 1, 1, HIDDEN), jnp.float32)
    assert not _can_use(w1, w2, w1_scale, w2_scale)
    # fp4 weights would need fp4 activations.
    assert not _can_use(*_weights(16, jnp.float4_e2m1fn))
    # The gate/up output must feed w2 unpadded.
    w1, _ = _weights(16)
    _, w2 = _weights(16, inter=INTER - 256)
    assert not _can_use(w1, w2)
    # A whole K x N bf16 slab of a 7168 x 2x4096 expert does not fit VMEM.
    assert not _can_use(*_weights(16, hidden=7168, inter=4096))


def test_routing_gmm_rejected_off_tpu(monkeypatch):
    def no_tpu():
        raise ValueError("not a TPU")

    monkeypatch.setattr(fused_moe_gmm.pltpu, "get_tpu_info", no_tpu)
    assert not _can_use(*_weights(16))


@pytest.mark.parametrize("enabled", [False, True])
def test_fused_moe_func_dispatches_on_flag(monkeypatch, enabled):
    calls = []

    def fake_routing_moe_gmm(hidden_states, *args, **kwargs):
        calls.append(kwargs)
        return jnp.full_like(hidden_states, 7)

    monkeypatch.setattr(fused_moe_gmm, "_ROUTING_GMM", enabled)
    monkeypatch.setattr(fused_moe_gmm, "can_use_routing_gmm", lambda *a, **k: True)
    monkeypatch.setattr(fused_moe_gmm, "routing_moe_gmm", fake_routing_moe_gmm)
    sentinel = jnp.full((4, 256), 3, jnp.bfloat16)
    monkeypatch.setattr(
        fused_moe_gmm, "prepare_routed_gmm_inputs", lambda *a, **k: (sentinel,) * 6
    )
    monkeypatch.setattr(fused_moe_gmm, "moe_gmm", lambda x, *a, **k: x)

    out = fused_moe_gmm.fused_moe_func.__wrapped__(
        jnp.zeros((4, 256), jnp.bfloat16),
        jnp.zeros((2, 256, 512), jnp.bfloat16),
        jnp.zeros((2, 256, 256), jnp.bfloat16),
        None,
        None,
        None,
        None,
        jnp.ones((4, 2), jnp.bfloat16),
        jnp.zeros((4, 2), jnp.int32),
        topk=2,
    )
    assert bool((out == (7 if enabled else 3)).all())
    assert len(calls) == int(enabled)


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _per_channel_fp8(w):
    """Quantize [E, K, N] to fp8 with one scale per output channel."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    scale = jnp.max(jnp.abs(w.astype(jnp.float32)), axis=1, keepdims=True) / fp8_max
    w_q = (w.astype(jnp.float32) / scale).astype(jnp.float8_e4m3fn)
    return w_q, scale  # scale: [E, 1, N]


def _run_both_paths(monkeypatch, args, static):
    """fused_moe_func on gmm_v2 and on routing_gmm, each traced fresh."""
    routed = []
    real_routing_moe_gmm = fused_moe_gmm.routing_moe_gmm

    def spy(*a, **k):
        routed.append(True)
        return real_routing_moe_gmm(*a, **k)

    monkeypatch.setattr(fused_moe_gmm, "routing_moe_gmm", spy)
    outputs = {}
    for enabled in (False, True):
        monkeypatch.setattr(fused_moe_gmm, "_ROUTING_GMM", enabled)
        fn = jax.jit(functools.partial(fused_moe_gmm.fused_moe_func.__wrapped__, **static))
        outputs[enabled] = np.asarray(fn(*args).astype(jnp.float32))
        # Traced once per arm: only the enabled arm may reach routing_gmm.
        assert len(routed) == int(enabled)
    return outputs[False], outputs[True]


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_tokens", [64, 512])
@pytest.mark.parametrize(
    ("use_ep", "skip_padded_tokens", "fp8"),
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
    ],
    ids=["bf16", "bf16_ep", "bf16_ep_skip_padded", "fp8_per_channel"],
)
def test_routing_gmm_matches_gmm_v2(
    monkeypatch, num_tokens, use_ep, skip_padded_tokens, fp8
):
    _require_tpu()
    if not fused_moe_gmm.routing_gmm_runtime_supported():
        pytest.skip("routing_gmm needs jax >= 0.12.0.dev20261007, libtpu >= 0.0.50")
    local_experts = 16
    # Under EP this shard owns global experts [16, 32); the rest are remote.
    global_experts = 2 * local_experts if use_ep else local_experts
    k = jax.random.split(jax.random.key(0), 5)
    hidden_states = (
        jax.random.normal(k[0], (num_tokens, HIDDEN), jnp.float32) / 10
    ).astype(jnp.bfloat16)
    w1 = (
        jax.random.normal(k[1], (local_experts, HIDDEN, 2 * INTER), jnp.float32) / 10
    ).astype(jnp.bfloat16)
    w2 = (
        jax.random.normal(k[2], (local_experts, INTER, HIDDEN), jnp.float32) / 10
    ).astype(jnp.bfloat16)
    w1_scale = w2_scale = None
    if fp8:
        w1, w1_scale = _per_channel_fp8(w1)
        w2, w2_scale = _per_channel_fp8(w2)
    logits = jax.random.normal(k[3], (num_tokens, global_experts), jnp.float32)
    topk_weights, topk_ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), TOPK)
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    if skip_padded_tokens:
        # The last eight tokens are padding: zero routing weight.
        topk_weights = topk_weights.at[-8:].set(0)
    experts_start = jnp.array(local_experts if use_ep else 0, jnp.int32)

    args = (
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        None,
        None,
        topk_weights,
        topk_ids.astype(jnp.int32),
        experts_start,
    )
    static = dict(
        topk=TOPK,
        activation="silu",
        use_ep=use_ep,
        skip_padded_tokens=skip_padded_tokens,
    )
    assert fused_moe_gmm.can_use_routing_gmm(
        hidden_states,
        w1,
        w2,
        None if w1_scale is None else w1_scale[:, :, None],
        None if w2_scale is None else w2_scale[:, :, None],
        None,
        None,
        activation="silu",
        topk=TOPK,
        rhs_quant_dtype=None,
    )
    expected, actual = _run_both_paths(monkeypatch, args, static)

    assert np.abs(expected).max() > 0
    # Both paths quantize fp8 activations, but not identically.
    rtol = 5e-2 if fp8 else 2e-2
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=rtol * float(np.abs(expected).max())
    )
    if skip_padded_tokens:
        np.testing.assert_array_equal(actual[-8:], 0)


def test_routing_gmm_pad_intermediate_defaults_on(monkeypatch):
    monkeypatch.delenv("TPU_MOE_ROUTING_GMM_PAD_INTERMEDIATE", raising=False)
    assert envs.environment_variables["TPU_MOE_ROUTING_GMM_PAD_INTERMEDIATE"]() is True


@pytest.mark.parametrize(
    ("routing", "pad", "expected"),
    [(False, False, 128), (False, True, 128), (True, False, 128), (True, True, 256)],
)
def test_moe_intermediate_alignment(monkeypatch, routing, pad, expected):
    monkeypatch.setattr(fused_moe_gmm, "_ROUTING_GMM", routing)
    monkeypatch.setenv("TPU_MOE_ROUTING_GMM_PAD_INTERMEDIATE", "1" if pad else "0")
    assert fused_moe_gmm.moe_intermediate_alignment() == expected


def _fp8_loader_weights(monkeypatch, pad, num_experts, inter, block_size=None):
    """Qwen3.5-397B TP=8 shard through the fp8 MoE loader: torch layout
    w13 [E, 2*inter, H] and w2 [E, H, inter] -> kernel layout jax arrays."""
    import torch

    from vllm_torchtpu.layers.adapter.quantization import fp8

    monkeypatch.setattr(fused_moe_gmm, "_ROUTING_GMM", True)
    monkeypatch.setenv("TPU_MOE_ROUTING_GMM_PAD_INTERMEDIATE", "1" if pad else "0")
    gen = torch.Generator().manual_seed(0)
    w13 = torch.randn(num_experts, 2 * inter, 4096, generator=gen) / 10
    w2 = torch.randn(num_experts, 4096, inter, generator=gen) / 10

    def fmt(w, name):
        w, s = fp8._quantize_and_format_single_moe_weight(
            w, name=name, quant_dtype=torch.float8_e4m3fn, block_size=block_size
        )
        w = jnp.asarray(w.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
        return w, jnp.asarray(s.numpy())

    return (*fmt(w13, "w13"), *fmt(w2, "w2"))


def test_fp8_loader_pads_tp8_shard_to_256(monkeypatch):
    w1, s1, w2, s2 = _fp8_loader_weights(monkeypatch, pad=False, num_experts=2, inter=128)
    assert w1.shape == (2, 4096, 256) and w2.shape == (2, 128, 4096)
    w1, s1, w2, s2 = _fp8_loader_weights(monkeypatch, pad=True, num_experts=2, inter=128)
    assert w1.shape == (2, 4096, 512) and s1.shape == (2, 1, 1, 512)
    assert w2.shape == (2, 256, 4096) and s2.shape == (2, 1, 1, 4096)
    # [gate | 0 | up | 0] and zero rows at the end of w2.
    zeros = np.asarray(w1.astype(jnp.float32))
    assert not zeros[..., 128:256].any() and not zeros[..., 384:].any()
    assert zeros[..., :128].any() and zeros[..., 256:384].any()
    assert not np.asarray(w2.astype(jnp.float32))[:, 128:].any()


def test_fp8_loader_keeps_128_for_block_scales(monkeypatch):
    w1, _, w2, _ = _fp8_loader_weights(
        monkeypatch, pad=True, num_experts=2, inter=128, block_size=128
    )
    assert w1.shape == (2, 4096, 256) and w2.shape == (2, 128, 4096)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_padded_tp8_shard_routes_and_matches_gmm_v2(monkeypatch, fp8):
    """A 1024-wide expert at TP=8 (128 per shard) padded to 256 must take
    routing_gmm and match gmm_v2 on the unpadded weights."""
    _require_tpu()
    if not fused_moe_gmm.routing_gmm_runtime_supported():
        pytest.skip("routing_gmm needs jax >= 0.12.0.dev20261007, libtpu >= 0.0.50")
    num_experts, inter, num_tokens, topk = 64, 128, 256, 10
    if fp8:
        w1, s1, w2, s2 = _fp8_loader_weights(monkeypatch, False, num_experts, inter)
        pw1, ps1, pw2, ps2 = _fp8_loader_weights(monkeypatch, True, num_experts, inter)
    else:
        k = jax.random.split(jax.random.key(0), 2)
        w1 = (jax.random.normal(k[0], (num_experts, 4096, 2 * inter)) / 10).astype(
            jnp.bfloat16
        )
        w2 = (jax.random.normal(k[1], (num_experts, inter, 4096)) / 10).astype(
            jnp.bfloat16
        )
        s1 = s2 = ps1 = ps2 = None
        # The unquantized loader's layout at 256: [gate | 0 | up | 0].
        zeros = jnp.zeros((num_experts, 4096, 256 - inter), jnp.bfloat16)
        pw1 = jnp.concatenate([w1[..., :inter], zeros, w1[..., inter:], zeros], -1)
        pw2 = jnp.pad(w2, ((0, 0), (0, 256 - inter), (0, 0)))
    assert not fused_moe_gmm.can_use_routing_gmm(
        jax.ShapeDtypeStruct((num_tokens, 4096), jnp.bfloat16), w1, w2, s1, s2,
        None, None, activation="silu", topk=topk, rhs_quant_dtype=None,
    )
    assert fused_moe_gmm.can_use_routing_gmm(
        jax.ShapeDtypeStruct((num_tokens, 4096), jnp.bfloat16), pw1, pw2, ps1, ps2,
        None, None, activation="silu", topk=topk, rhs_quant_dtype=None,
    )

    k = jax.random.split(jax.random.key(1), 2)
    x = (jax.random.normal(k[0], (num_tokens, 4096)) / 10).astype(jnp.bfloat16)
    logits = jax.random.normal(k[1], (num_tokens, num_experts))
    topk_weights, topk_ids = jax.lax.top_k(jax.nn.softmax(logits, -1), topk)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdims=True)
    routed = []
    real = fused_moe_gmm.routing_moe_gmm
    monkeypatch.setattr(
        fused_moe_gmm,
        "routing_moe_gmm",
        lambda *a, **kw: (routed.append(True), real(*a, **kw))[1],
    )
    outputs = []
    for on, (a, sa, b, sb) in ((False, (w1, s1, w2, s2)), (True, (pw1, ps1, pw2, ps2))):
        monkeypatch.setattr(fused_moe_gmm, "_ROUTING_GMM", on)
        fn = jax.jit(
            functools.partial(
                fused_moe_gmm.fused_moe_func.__wrapped__, topk=topk, activation="silu"
            )
        )
        out = fn(x, a, b, sa, sb, None, None, topk_weights, topk_ids.astype(jnp.int32))
        outputs.append(np.asarray(out.astype(jnp.float32)))
        assert len(routed) == int(on)
    expected, actual = outputs
    assert np.abs(expected).max() > 0
    rtol = 5e-2 if fp8 else 2e-2
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=rtol * float(np.abs(expected).max())
    )


def test_fp8_loader_leaves_ep8_shard_unpadded(monkeypatch):
    """Qwen3.5-397B at EP=8 keeps whole 1024-wide experts: already a multiple
    of 256, so the pad flag changes nothing."""
    for pad in (False, True):
        w1, s1, w2, s2 = _fp8_loader_weights(monkeypatch, pad, num_experts=2, inter=1024)
        assert w1.shape == (2, 4096, 2048) and s1.shape == (2, 1, 1, 2048)
        assert w2.shape == (2, 1024, 4096) and s2.shape == (2, 1, 1, 4096)
