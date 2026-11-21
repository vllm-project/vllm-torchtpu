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
"""Unit and parity tests for the post_norm_moe_op bridge."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from torch_tpu._internal import pallas
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from vllm.models.muse_spark.common.moe import MuseSpark1xRouter

from tests.layers.core.post_norm_moe._ref_gmm import ref_gmm
from vllm_torchtpu.layers.adapter import expert_placement
from vllm_torchtpu.layers.adapter.custom_ops import post_norm_moe_op
from vllm_torchtpu.layers.adapter.custom_ops.post_norm_moe_op import (
    PostNormMoeStatic,
    build_post_norm_moe,
    get_post_norm_moe_op,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
)
from vllm_torchtpu.layers.core.fused_moe_gmm import quantize_to_native_fp4_kmajor
from vllm_torchtpu.layers.core.post_norm_moe import post_norm_moe_jax


def _torch_to_jax_bf16(t: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(t.detach().float().numpy()).astype(jnp.bfloat16)


def _upstream_moe_reference(
    h: torch.Tensor,
    router_w_kmajor: torch.Tensor,
    router_bias: torch.Tensor,
    pooler_w_kmajor: torch.Tensor,
    pre_norm_w: torch.Tensor,
    w13_kmajor: torch.Tensor,
    w2_kmajor: torch.Tensor,
    post_norm_w: torch.Tensor,
    *,
    topk: int,
    output_multiplier: float,
    routed_scaling_factor: float,
    rms_norm_eps: float,
    post_norm_eps: float,
    router_eps: float = 1e-15,
) -> torch.Tensor:
    """Reference MoE composed from upstream MuseSpark1xRouter and norms."""
    num_tokens = h.shape[0]
    num_experts = router_bias.numel()
    moe_hidden = pre_norm_w.numel()

    router = MuseSpark1xRouter(
        top_k=topk,
        num_experts=num_experts,
        expert_bias=router_bias.float(),
        route_scale=routed_scaling_factor,
        router_output_multiplier=output_multiplier,
        router_score_normalization="sigmoid",
        router_score_division_eps=router_eps,
    )
    logits = h.float() @ router_w_kmajor.float()
    topk_weights, topk_ids = router._compute_routing_native(logits, torch.int64)

    pre_norm = MuseSpark1xRMSNorm(
        hidden_size=moe_hidden,
        eps=rms_norm_eps,
        has_weight=True,
        is_post_norm=False,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=2,
    )
    pre_norm.weight.data.copy_(pre_norm_w.float())

    post_norm = MuseSpark1xRMSNormWithInputScale(
        hidden_size=moe_hidden,
        eps=post_norm_eps,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None,
    )
    post_norm.weight.data.copy_(post_norm_w.float())

    z = (h.float() @ pooler_w_kmajor.float()).to(h.dtype)
    z = pre_norm.forward_native(z)

    out = torch.zeros((num_tokens, moe_hidden), dtype=torch.float32)
    for tok in range(num_tokens):
        for slot in range(topk):
            eid = int(topk_ids[tok, slot].item())
            weight = float(topk_weights[tok, slot].item())
            gate_up = z[tok : tok + 1].float() @ w13_kmajor[eid].float()
            gate, up = gate_up.chunk(2, dim=-1)
            inter = (F.silu(gate) * up).to(h.dtype)
            down = (inter.float() @ w2_kmajor[eid].float()).to(h.dtype)
            normed = post_norm.forward_native(down)
            out[tok] += normed[0].float() * weight
    return out


def test_op_name_is_schema_safe_and_injective(monkeypatch: pytest.MonkeyPatch) -> None:
    """_format_op_name emits valid PyTorch custom-op names without float collisions."""
    recorded: list[str] = []

    class _StubOp:
        def __init__(self, name: str) -> None:
            self.name = name

        def register_fake(self, fn: Any) -> None:
            self.fake = fn

    def _record_jax_op(name: str, fn: Any) -> _StubOp:
        del fn
        recorded.append(name)
        return _StubOp(name)

    monkeypatch.setattr(post_norm_moe_op, "_POST_NORM_MOE_OPS", {})
    monkeypatch.setattr(pallas, "jax_op", _record_jax_op)

    vals = [1e-20, 1e20, 0.17677669529663687, 1.0, 1.0000000000000002]
    for v in vals:
        static = PostNormMoeStatic(
            topk=8,
            output_multiplier=v,
            routed_scaling_factor=v,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            router_eps=1e-15,
            activation="silu",
            use_ep=True,
            pre_projected=False,
        )
        get_post_norm_moe_op(static)

    assert len(recorded) == len(vals)
    assert len(set(recorded)) == len(vals)
    for name in recorded:
        assert re.fullmatch(r"tpu::[A-Za-z0-9_]+", name), name


def test_op_raises_on_cpu_without_injected_gmm() -> None:
    """post_norm_moe_jax refuses to silently fall back on non-TPU backends."""
    if jax.default_backend() == "tpu":
        pytest.skip("CPU-only negative test")
    h = jnp.zeros((2, 32), dtype=jnp.bfloat16)
    rw = jnp.zeros((32, 4), dtype=jnp.float32)
    rb = jnp.zeros((4,), dtype=jnp.float32)
    pw = jnp.zeros((32, 16), dtype=jnp.bfloat16)
    pnw = jnp.zeros((16,), dtype=jnp.float32)
    w13 = jnp.zeros((4, 16, 32), dtype=jnp.bfloat16)
    w2 = jnp.zeros((4, 16, 16), dtype=jnp.bfloat16)
    postw = jnp.zeros((16,), dtype=jnp.float32)
    es = jnp.array(0, dtype=jnp.int32)

    with pytest.raises(NotImplementedError, match="requires a TPU backend"):
        post_norm_moe_jax(
            h, rw, rb, pw, pnw, w13, None, w2, None, postw, es, topk=2, use_ep=False
        )


@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("pre_projected", [False, True])
def test_moe_matches_reference_dense(num_tokens: int, pre_projected: bool) -> None:
    """post_norm_moe_jax(gmm=ref_gmm) matches the upstream PyTorch reference."""
    torch.manual_seed(100 + num_tokens)
    hidden, moe_hidden, inter, experts, topk = 64, 32, 24, 8, 4
    output_multiplier = 0.25

    h = torch.randn(num_tokens, hidden, dtype=torch.bfloat16) * 0.5
    router_w = torch.randn(hidden, experts, dtype=torch.float32) * 0.1
    router_b = torch.linspace(-0.2, 0.2, experts, dtype=torch.float32)
    pooler_w = torch.randn(hidden, moe_hidden, dtype=torch.bfloat16) * 0.2
    pre_norm_w = torch.randn(moe_hidden, dtype=torch.float32) * 0.1
    w13 = torch.randn(experts, moe_hidden, 2 * inter, dtype=torch.bfloat16) * 0.2
    w2 = torch.randn(experts, inter, moe_hidden, dtype=torch.bfloat16) * 0.2
    post_norm_w = torch.randn(moe_hidden, dtype=torch.float32) * 0.1

    ref = _upstream_moe_reference(
        h,
        router_w,
        router_b,
        pooler_w,
        pre_norm_w,
        w13,
        w2,
        post_norm_w,
        topk=topk,
        output_multiplier=output_multiplier,
        routed_scaling_factor=1.25,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
    )

    pre_arg = (
        _torch_to_jax_bf16((h.float() @ pooler_w.float()).to(torch.bfloat16))
        if pre_projected
        else _torch_to_jax_bf16(pooler_w)
    )
    got = post_norm_moe_jax(
        _torch_to_jax_bf16(h),
        jnp.asarray(router_w.numpy()),
        jnp.asarray(router_b.numpy()),
        pre_arg,
        jnp.asarray(pre_norm_w.numpy()),
        _torch_to_jax_bf16(w13),
        None,
        _torch_to_jax_bf16(w2),
        None,
        jnp.asarray(post_norm_w.numpy()),
        jnp.array(0, dtype=jnp.int32),
        topk=topk,
        output_multiplier=output_multiplier,
        routed_scaling_factor=1.25,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        use_ep=False,
        pre_projected=pre_projected,
        router_topk_impl="lax",
        gmm=ref_gmm,
    )
    np.testing.assert_allclose(np.asarray(got), ref.numpy(), rtol=2e-2, atol=2e-2)


def test_moe_matches_reference_fp4() -> None:
    """Native float4_e2m1fn weights + block scales match dequantized reference."""
    torch.manual_seed(202)
    num_tokens, hidden, moe_hidden, inter, experts, topk = 6, 64, 32, 32, 8, 2

    h = torch.randn(num_tokens, hidden, dtype=torch.bfloat16) * 0.5
    router_w = torch.randn(hidden, experts, dtype=torch.float32) * 0.1
    router_b = torch.linspace(-0.2, 0.2, experts, dtype=torch.float32)
    pooler_w = torch.randn(hidden, moe_hidden, dtype=torch.bfloat16) * 0.2
    pre_norm_w = torch.randn(moe_hidden, dtype=torch.float32) * 0.1
    w13_raw = jnp.asarray(
        torch.randn(experts, moe_hidden, 2 * inter, dtype=torch.float32).numpy() * 0.2
    )
    w2_raw = jnp.asarray(
        torch.randn(experts, inter, moe_hidden, dtype=torch.float32).numpy() * 0.2
    )
    post_norm_w = torch.randn(moe_hidden, dtype=torch.float32) * 0.1

    w13_fp4, w13_scale = quantize_to_native_fp4_kmajor(w13_raw, block=16, pack=False)
    w2_fp4, w2_scale = quantize_to_native_fp4_kmajor(w2_raw, block=16, pack=False)

    def _dequant(q: jnp.ndarray, s: jnp.ndarray) -> torch.Tensor:
        qf = q.astype(jnp.float32).reshape(q.shape[0], s.shape[1], 16, q.shape[2])
        dq = (qf * s).reshape(q.shape)
        return torch.from_numpy(np.asarray(dq).copy())

    ref = _upstream_moe_reference(
        h,
        router_w,
        router_b,
        pooler_w,
        pre_norm_w,
        _dequant(w13_fp4, w13_scale),
        _dequant(w2_fp4, w2_scale),
        post_norm_w,
        topk=topk,
        output_multiplier=1.0,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
    )

    got = post_norm_moe_jax(
        _torch_to_jax_bf16(h),
        jnp.asarray(router_w.numpy()),
        jnp.asarray(router_b.numpy()),
        _torch_to_jax_bf16(pooler_w),
        jnp.asarray(pre_norm_w.numpy()),
        w13_fp4,
        w13_scale.squeeze(2),
        w2_fp4,
        w2_scale.squeeze(2),
        jnp.asarray(post_norm_w.numpy()),
        jnp.array(0, dtype=jnp.int32),
        topk=topk,
        use_ep=False,
        router_topk_impl="lax",
        gmm=ref_gmm,
    )
    np.testing.assert_allclose(np.asarray(got), ref.numpy(), rtol=2e-2, atol=2e-2)


def test_ep_shards_sum_to_full() -> None:
    """Sum of 8 EP rank partials equals the single-rank full-expert result."""
    rng = np.random.default_rng(77)
    num_tokens, hidden, moe_hidden, inter, experts, ep_size, topk = (
        10,
        64,
        32,
        16,
        16,
        8,
        4,
    )
    local_e = experts // ep_size

    h = jnp.asarray(rng.standard_normal((num_tokens, hidden), dtype=np.float32)).astype(
        jnp.bfloat16
    )
    rw = jnp.asarray(rng.standard_normal((hidden, experts), dtype=np.float32) * 0.1)
    rb = jnp.linspace(-0.2, 0.2, experts, dtype=jnp.float32)
    pw = jnp.asarray(
        rng.standard_normal((hidden, moe_hidden), dtype=np.float32) * 0.2
    ).astype(jnp.bfloat16)
    pnw = jnp.asarray(rng.standard_normal((moe_hidden,), dtype=np.float32) * 0.1)
    w13 = jnp.asarray(
        rng.standard_normal((experts, moe_hidden, 2 * inter), dtype=np.float32) * 0.2
    ).astype(jnp.bfloat16)
    w2 = jnp.asarray(
        rng.standard_normal((experts, inter, moe_hidden), dtype=np.float32) * 0.2
    ).astype(jnp.bfloat16)
    postw = jnp.asarray(rng.standard_normal((moe_hidden,), dtype=np.float32) * 0.1)

    full = post_norm_moe_jax(
        h,
        rw,
        rb,
        pw,
        pnw,
        w13,
        None,
        w2,
        None,
        postw,
        jnp.array(0, dtype=jnp.int32),
        topk=topk,
        use_ep=False,
        router_topk_impl="lax",
        gmm=ref_gmm,
    )

    partials = []
    for rank in range(ep_size):
        start = rank * local_e
        part = post_norm_moe_jax(
            h,
            rw,
            rb,
            pw,
            pnw,
            w13[start : start + local_e],
            None,
            w2[start : start + local_e],
            None,
            postw,
            jnp.array(start, dtype=jnp.int32),
            topk=topk,
            use_ep=True,
            router_topk_impl="lax",
            gmm=ref_gmm,
        )
        partials.append(part)

    summed = sum(partials)
    np.testing.assert_allclose(
        np.asarray(summed), np.asarray(full), rtol=1e-5, atol=1e-5
    )


def test_padded_rows_do_not_nan() -> None:
    """Rows past group_sizes.sum() containing NaNs from GMM are masked cleanly."""
    rng = np.random.default_rng(91)
    h = jnp.asarray(rng.standard_normal((4, 32), dtype=np.float32)).astype(jnp.bfloat16)
    rw = jnp.asarray(rng.standard_normal((32, 8), dtype=np.float32))
    rb = jnp.zeros((8,), dtype=jnp.float32)
    pw = jnp.ones((32, 16), dtype=jnp.bfloat16) * 0.1
    pnw = jnp.zeros((16,), dtype=jnp.float32)
    w13 = jnp.ones((2, 16, 16), dtype=jnp.bfloat16) * 0.1
    w2 = jnp.ones((2, 8, 16), dtype=jnp.bfloat16) * 0.1
    postw = jnp.zeros((16,), dtype=jnp.float32)

    def _nan_tail_gmm(
        lhs: jax.Array,
        rhs: jax.Array,
        rhs_scale: jax.Array | None,
        rhs_bias: jax.Array | None,
        group_sizes: jax.Array,
        **kwargs: Any,
    ) -> jax.Array:
        out = ref_gmm(lhs, rhs, rhs_scale, rhs_bias, group_sizes, **kwargs)
        valid = group_sizes.sum(dtype=jnp.int32)
        mask = jnp.arange(out.shape[0], dtype=jnp.int32) < valid
        return jnp.where(mask[:, None], out, jnp.nan)

    # Local experts [0, 1] out of 8 ensures some routed slots miss this rank,
    # making group_sizes.sum() < num_tokens * topk.
    got = post_norm_moe_jax(
        h,
        rw,
        rb,
        pw,
        pnw,
        w13,
        None,
        w2,
        None,
        postw,
        jnp.array(0, dtype=jnp.int32),
        topk=4,
        use_ep=True,
        router_topk_impl="lax",
        gmm=_nan_tail_gmm,
    )
    assert bool(jnp.all(jnp.isfinite(got)))


def test_fake_shape_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """register_fake produces float32 [T, moe_hidden] output under FakeTensorMode."""

    def _stub_jax_op(name: str, fn: Any) -> Any:
        def _impl(
            h: torch.Tensor,
            rw: torch.Tensor,
            rb: torch.Tensor,
            pre: torch.Tensor,
            pnw: torch.Tensor,
            w13: torch.Tensor,
            w13s: torch.Tensor | None,
            w2: torch.Tensor,
            w2s: torch.Tensor | None,
            postw: torch.Tensor,
            es: torch.Tensor,
        ) -> torch.Tensor:
            del rw, rb, pre, w13, w13s, w2, w2s, postw, es
            return torch.zeros(
                (h.shape[0], pnw.shape[0]), dtype=torch.float32, device=h.device
            )

        custom = torch.library.custom_op(name, _impl, mutates_args=())
        del fn
        return custom

    monkeypatch.setattr(post_norm_moe_op, "_POST_NORM_MOE_OPS", {})
    monkeypatch.setattr(pallas, "jax_op", _stub_jax_op)

    static = PostNormMoeStatic(
        topk=2,
        output_multiplier=1.0,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        router_eps=1e-15,
        activation="silu",
        use_ep=False,
        pre_projected=False,
    )
    op = get_post_norm_moe_op(static)
    with FakeTensorMode() as fake_mode:
        h = fake_mode.from_tensor(torch.zeros((5, 32), dtype=torch.bfloat16))
        rw = fake_mode.from_tensor(torch.zeros((32, 4), dtype=torch.float32))
        rb = fake_mode.from_tensor(torch.zeros((4,), dtype=torch.float32))
        pw = fake_mode.from_tensor(torch.zeros((32, 16), dtype=torch.bfloat16))
        pnw = fake_mode.from_tensor(torch.zeros((16,), dtype=torch.float32))
        w13 = fake_mode.from_tensor(torch.zeros((4, 16, 16), dtype=torch.bfloat16))
        w2 = fake_mode.from_tensor(torch.zeros((4, 8, 16), dtype=torch.bfloat16))
        postw = fake_mode.from_tensor(torch.zeros((16,), dtype=torch.float32))
        es = fake_mode.from_tensor(torch.zeros((), dtype=torch.int32))
        out = op(h, rw, rb, pw, pnw, w13, None, w2, None, postw, es)
        assert out.shape == (5, 16)
        assert out.dtype == torch.float32


def test_real_pallas_jax_op_registration_and_fake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unmocked pallas.jax_op registration accepts bound gmm=None and FakeTensorMode."""
    monkeypatch.setattr(post_norm_moe_op, "_POST_NORM_MOE_OPS", {})
    static = PostNormMoeStatic(
        topk=2,
        output_multiplier=0.5,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        router_eps=1e-15,
        activation="silu",
        use_ep=False,
        pre_projected=False,
    )
    op = get_post_norm_moe_op(static)
    with FakeTensorMode() as fake_mode:
        h = fake_mode.from_tensor(torch.zeros((4, 32), dtype=torch.bfloat16))
        rw = fake_mode.from_tensor(torch.zeros((32, 4), dtype=torch.float32))
        rb = fake_mode.from_tensor(torch.zeros((4,), dtype=torch.float32))
        pw = fake_mode.from_tensor(torch.zeros((32, 16), dtype=torch.bfloat16))
        pnw = fake_mode.from_tensor(torch.zeros((16,), dtype=torch.float32))
        w13 = fake_mode.from_tensor(torch.zeros((4, 16, 16), dtype=torch.bfloat16))
        w2 = fake_mode.from_tensor(torch.zeros((4, 8, 16), dtype=torch.bfloat16))
        postw = fake_mode.from_tensor(torch.zeros((16,), dtype=torch.float32))
        es = fake_mode.from_tensor(torch.zeros((), dtype=torch.int32))
        out = op(h, rw, rb, pw, pnw, w13, None, w2, None, postw, es)
        assert out.shape == (4, 16)
        assert out.dtype == torch.float32


def test_onehot_ep_nan_hygiene_isolates_bad_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NaN in one padded/corrupt row does not poison finite tokens under one-hot EP."""
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    rng = np.random.default_rng(123)
    h_np = rng.standard_normal((4, 32), dtype=np.float32) * 0.2
    h_np[3] = np.nan
    h = jnp.asarray(h_np).astype(jnp.bfloat16)
    rw = jnp.asarray(rng.standard_normal((32, 4), dtype=np.float32) * 0.1)
    rb = jnp.zeros((4,), dtype=jnp.float32)
    pw = jnp.asarray(rng.standard_normal((32, 16), dtype=np.float32) * 0.2).astype(
        jnp.bfloat16
    )
    pnw = jnp.zeros((16,), dtype=jnp.float32)
    w13 = jnp.ones((2, 16, 16), dtype=jnp.bfloat16) * 0.1
    w2 = jnp.ones((2, 8, 16), dtype=jnp.bfloat16) * 0.1
    postw = jnp.zeros((16,), dtype=jnp.float32)

    got = post_norm_moe_jax(
        h,
        rw,
        rb,
        pw,
        pnw,
        w13,
        None,
        w2,
        None,
        postw,
        jnp.array(0, dtype=jnp.int32),
        topk=2,
        use_ep=True,
        use_sparse_core=True,
        onehot_threshold=64,
        router_topk_impl="lax",
        gmm=ref_gmm,
    )
    assert bool(jnp.all(jnp.isfinite(got[:3])))


def _make_stub_routed(
    *,
    activation: str | None = "silu",
    tp_size: int = 1,
    use_ep: bool = False,
) -> Any:
    method = object.__new__(VllmUnquantizedFusedMoEMethod)
    method._tpu_activation_str = activation
    mpc = SimpleNamespace(tp_size=tp_size, use_ep=use_ep)
    return SimpleNamespace(
        w13_weight=torch.ones((4, 16, 16), dtype=torch.bfloat16),
        w2_weight=torch.ones((4, 8, 16), dtype=torch.bfloat16),
        quant_method=method,
        moe_config=SimpleNamespace(moe_parallel_config=mpc),
        _experts_start=torch.tensor(0, dtype=torch.int32),
    )


def test_build_post_norm_moe_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_post_norm_moe enforces finalization, placement, TP, and layout guards."""
    monkeypatch.setattr(
        post_norm_moe_op, "get_post_norm_moe_op", lambda s: lambda *a: a[0]
    )

    static = PostNormMoeStatic(
        topk=2,
        output_multiplier=1.0,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        router_eps=1e-15,
        activation="silu",
        use_ep=False,
        pre_projected=False,
    )
    rw = torch.zeros((32, 4), dtype=torch.float32)
    rb = torch.zeros((4,), dtype=torch.float32)
    pw = torch.zeros((32, 16), dtype=torch.bfloat16)
    pnw = torch.zeros((16,), dtype=torch.float32)
    postw = torch.zeros((16,), dtype=torch.float32)

    # 1. Unfinalized quant method (_tpu_activation_str is None) -> RuntimeError (F014).
    with pytest.raises(RuntimeError, match="initialized _tpu_activation_str"):
        build_post_norm_moe(
            _make_stub_routed(activation=None),
            static=static,
            router_w_kmajor=rw,
            router_bias=rb,
            pre_w=pw,
            pre_norm_w=pnw,
            post_norm_w=postw,
        )

    # 2. TP-sharded experts -> NotImplementedError (F012).
    with pytest.raises(NotImplementedError, match="moe_parallel_config.tp_size=2"):
        build_post_norm_moe(
            _make_stub_routed(tp_size=2),
            static=static,
            router_w_kmajor=rw,
            router_bias=rb,
            pre_w=pw,
            pre_norm_w=pnw,
            post_norm_w=postw,
        )

    # 3. Transposed router weight [E, H] when H != E -> ValueError (F009).
    with pytest.raises(ValueError, match="K-major shape"):
        build_post_norm_moe(
            _make_stub_routed(),
            static=static,
            router_w_kmajor=torch.zeros((4, 32), dtype=torch.float32),
            router_bias=rb,
            pre_w=pw,
            pre_norm_w=pnw,
            post_norm_w=postw,
        )

    # 4. Mismatched static.use_ep vs mpc.use_ep -> ValueError.
    with pytest.raises(ValueError, match="PostNormMoeStatic.use_ep"):
        build_post_norm_moe(
            _make_stub_routed(use_ep=True),
            static=static,
            router_w_kmajor=rw,
            router_bias=rb,
            pre_w=pw,
            pre_norm_w=pnw,
            post_norm_w=postw,
        )

    # 5. BF16 experts with biases -> NotImplementedError (no bias operands).
    for bias_attr in ("w13_bias", "w2_bias"):
        biased = _make_stub_routed()
        setattr(biased, bias_attr, torch.zeros((4, 16), dtype=torch.bfloat16))
        with pytest.raises(NotImplementedError, match="expert biases"):
            build_post_norm_moe(
                biased,
                static=static,
                router_w_kmajor=rw,
                router_bias=rb,
                pre_w=pw,
                pre_norm_w=pnw,
                post_norm_w=postw,
            )

    # 6. Placed layer under TPU_EXPERT_PLACEMENT -> NotImplementedError (B4).
    monkeypatch.setattr(expert_placement, "expert_order", lambda _r: (3, 2, 1, 0))
    with pytest.raises(NotImplementedError, match="TPU_EXPERT_PLACEMENT"):
        build_post_norm_moe(
            _make_stub_routed(),
            static=static,
            router_w_kmajor=rw,
            router_bias=rb,
            pre_w=pw,
            pre_norm_w=pnw,
            post_norm_w=postw,
        )


@pytest.mark.tpu_test
def test_post_norm_moe_tpu_parity() -> None:
    """Real gmm_wrapper on TPU matches ref_gmm within bf16 tolerance."""
    rng = np.random.default_rng(303)
    h = jnp.asarray(rng.standard_normal((8, 128), dtype=np.float32) * 0.5).astype(
        jnp.bfloat16
    )
    rw = jnp.asarray(rng.standard_normal((128, 16), dtype=np.float32) * 0.1)
    rb = jnp.linspace(-0.1, 0.1, 16, dtype=jnp.float32)
    pw = jnp.asarray(rng.standard_normal((128, 128), dtype=np.float32) * 0.2).astype(
        jnp.bfloat16
    )
    pnw = jnp.asarray(rng.standard_normal((128,), dtype=np.float32) * 0.1)
    w13 = jnp.asarray(
        rng.standard_normal((16, 128, 256), dtype=np.float32) * 0.2
    ).astype(jnp.bfloat16)
    w2 = jnp.asarray(
        rng.standard_normal((16, 128, 128), dtype=np.float32) * 0.2
    ).astype(jnp.bfloat16)
    postw = jnp.asarray(rng.standard_normal((128,), dtype=np.float32) * 0.1)
    es = jnp.array(0, dtype=jnp.int32)

    got = post_norm_moe_jax(
        h, rw, rb, pw, pnw, w13, None, w2, None, postw, es, topk=4, use_ep=False
    )
    ref = post_norm_moe_jax(
        h,
        rw,
        rb,
        pw,
        pnw,
        w13,
        None,
        w2,
        None,
        postw,
        es,
        topk=4,
        use_ep=False,
        router_topk_impl="lax",
        gmm=ref_gmm,
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=3e-2, atol=3e-2)
