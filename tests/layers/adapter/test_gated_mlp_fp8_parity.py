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
"""Parity, tile-selection, TP/bias, and patch tests for fused gated-MLP FP8."""

from types import SimpleNamespace
from unittest import mock

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest
import torch


def _apply_ref_act(gate_f32: jax.Array, act_kind: int) -> jax.Array:
    if act_kind == 0:
        return jax.nn.gelu(gate_f32, approximate=True)
    return jax.nn.silu(gate_f32)


def _reference(gate_up, w_q, w_s, act_kind=0):
    from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

    d = gate_up.shape[-1] // 2
    gate, up = gate_up[:, :d], gate_up[:, d:]
    act = _apply_ref_act(gate.astype(jnp.float32), act_kind).astype(gate_up.dtype) * up
    return xla_quantized_matmul(act, w_q, w_s), act


def _fp32_oracle(act, w_q, w_s):
    w_deq = w_q.astype(jnp.float32) * w_s[None, :]
    return (act.astype(jnp.float32) @ w_deq).astype(act.dtype)


def test_select_block_m_and_ast_matcher_cpu():
    """CPU check of VMEM-derived pipelined tile selection, fake op, and AST matcher."""
    from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul

    from vllm_torchtpu.kernels.gated_mlp_fp8 import (
        GatedActivationKind,
        select_block_m,
        vmem_limit_bytes,
    )
    from vllm_torchtpu.layers.adapter import gated_mlp_fp8 as adapter

    v7x_vmem = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )
    assert vmem_limit_bytes(v7x_vmem) == 56 * 1024 * 1024

    # Gemma 4 31B TP=2 (d=10752): pipelined 4-step bm=64 at m=256, bm=128 at large m.
    assert select_block_m(128, 10752, v7x_vmem) == 32
    assert select_block_m(216, 10752, v7x_vmem) == 216
    assert select_block_m(256, 10752, v7x_vmem) == 64
    assert select_block_m(512, 10752, v7x_vmem) == 128
    assert select_block_m(16384, 10752, v7x_vmem) == 128
    assert select_block_m(130, 10752, v7x_vmem) is None
    assert select_block_m(600, 10752, v7x_vmem) is None

    # Gemma 4 31B TP=4 (d=5376), d=5632, and near-56-MiB boundary (d=9728 -> 52.25 MiB):
    assert select_block_m(128, 5376, v7x_vmem) == 64
    assert select_block_m(256, 5376, v7x_vmem) == 128
    assert select_block_m(16384, 5376, v7x_vmem) == 256
    assert select_block_m(1024, 5632, v7x_vmem) == 256
    assert select_block_m(512, 9728, v7x_vmem) == 256

    gelu_tanh = GeluAndMul.__new__(GeluAndMul)
    gelu_tanh.approximate = "tanh"
    gelu_none = GeluAndMul.__new__(GeluAndMul)
    gelu_none.approximate = "none"
    silu_mul = SiluAndMul.__new__(SiluAndMul)
    assert adapter._resolve_act_kind(gelu_tanh) == GatedActivationKind.GELU_TANH
    assert adapter._resolve_act_kind(gelu_none) is None
    assert adapter._resolve_act_kind(silu_mul) == GatedActivationKind.SILU

    fake_out = adapter._fake_fused_gated_act_fp8_matmul(
        torch.empty((256, 21504), dtype=torch.bfloat16),
        torch.empty((10752, 5376), dtype=torch.float8_e4m3fn),
        torch.empty((5376,), dtype=torch.float32),
        int(GatedActivationKind.GELU_TANH),
        64,
        v7x_vmem,
    )
    assert fake_out.shape == (256, 5376)
    assert fake_out.dtype == torch.bfloat16

    class _StandardGatedMLP(torch.nn.Module):
        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)
            return x

    class _NonStandardMLP(torch.nn.Module):
        def forward(self, x):
            y, _ = self.fc1(x)
            return self.act(y)

    assert adapter._matches_standard_gated_mlp_forward(_StandardGatedMLP.forward)
    assert not adapter._matches_standard_gated_mlp_forward(_NonStandardMLP.forward)


@pytest.fixture
def gemma4_patch_env(monkeypatch):
    """Restore Gemma4MLP.forward and reset TPU_7X + fallback log state per test."""
    import vllm.model_executor.models.gemma4 as gemma4_mod
    import vllm.model_executor.models.llama as llama_mod

    from vllm_torchtpu.layers.adapter import gated_mlp_fp8 as adapter

    orig_gemma4_fwd = gemma4_mod.Gemma4MLP.__dict__["forward"]
    orig_llama_fwd = llama_mod.LlamaMLP.__dict__["forward"]
    monkeypatch.setattr(gemma4_mod.Gemma4MLP, "forward", orig_gemma4_fwd)
    monkeypatch.setattr(adapter, "get_chip_version", lambda: pltpu.ChipVersion.TPU_7X)
    monkeypatch.setattr(adapter, "_LOGGED_FALLBACK_REASONS", set())
    monkeypatch.setenv("TPU_FUSED_GATED_MLP_FP8", "1")
    monkeypatch.delenv("QUANTIZE_ON_LOAD_PREFIXES", raising=False)
    return SimpleNamespace(
        adapter=adapter,
        gemma4_mod=gemma4_mod,
        llama_mod=llama_mod,
        orig_llama_fwd=orig_llama_fwd,
    )


class _FakeTpuTensor(torch.Tensor):
    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu")


def _make_stub_gemma4_mlp(
    gemma4_mod,
    prefix: str,
    *,
    valid_gu: bool = True,
    valid_qm: bool = True,
    tpu_gu_out: bool = False,
):
    from vllm.model_executor.layers.activation import GeluAndMul
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    mlp = gemma4_mod.Gemma4MLP.__new__(gemma4_mod.Gemma4MLP)
    torch.nn.Module.__init__(mlp)

    gu_tracker = SimpleNamespace(call_count=0)

    def _gu_out(x):
        if not torch.compiler.is_compiling():
            gu_tracker.call_count += 1
        gu = torch.ones((x.shape[0], 256), dtype=torch.bfloat16)
        return (gu.as_subclass(_FakeTpuTensor) if tpu_gu_out else gu), None

    if valid_gu:
        gu_layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
        torch.nn.Module.__init__(gu_layer)
    else:
        gu_layer = torch.nn.Identity()  # type: ignore[assignment]
    gu_layer.forward = _gu_out  # type: ignore[method-assign]
    mlp.gate_up_proj = gu_layer

    act_fn = GeluAndMul.__new__(GeluAndMul)
    torch.nn.Module.__init__(act_fn)
    act_fn.approximate = "tanh"
    act_fn.forward = (  # type: ignore[method-assign]
        lambda gu: (gu.as_subclass(torch.Tensor) if type(gu) is _FakeTpuTensor else gu)[
            :, :128
        ]
    )
    mlp.act_fn = act_fn

    dp = RowParallelLinear.__new__(RowParallelLinear)
    torch.nn.Module.__init__(dp)
    dp.prefix = prefix
    if valid_qm:
        qm = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
        qm.w8a16_max_rows = 0
        dp.quant_method = qm
    else:
        dp.quant_method = torch.nn.Identity()
    dp.input_is_parallel = True
    dp.return_bias = True
    dp.skip_bias_add = False
    dp.tp_size = 1
    dp.tp_rank = 0
    dp.reduce_results = False
    dp.register_parameter("bias", None)
    dp.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.zeros((128, 64), dtype=torch.float8_e4m3fn), requires_grad=False
        ),
    )
    dp.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones((64,), dtype=torch.float32), requires_grad=False),
    )
    dp.forward = lambda y: (y[:, :64], None)  # type: ignore[method-assign]
    mlp.down_proj = dp
    return mlp, gu_tracker


def test_patch_requires_model_config_cpu(gemma4_patch_env):
    """model_config=None returns False so patch_registry retries at model_load."""
    assert gemma4_patch_env.adapter.maybe_patch_gated_mlp(None) is False


def test_patch_noop_off_v7x_cpu(gemma4_patch_env, monkeypatch):
    """Non-TPU_7X chips permanently skip patching and return None."""
    env = gemma4_patch_env
    monkeypatch.setattr(
        env.adapter, "get_chip_version", lambda: pltpu.ChipVersion.TPU_V6E
    )
    cfg = SimpleNamespace(architectures=["Gemma4ForCausalLM"], quantization="fp8")
    assert env.adapter.maybe_patch_gated_mlp(cfg) is None
    assert not getattr(env.gemma4_mod.Gemma4MLP.forward, "_tpu_fused_gated_mlp", False)


def test_patch_kill_switch_cpu(gemma4_patch_env, monkeypatch):
    """TPU_FUSED_GATED_MLP_FP8=0 disables patching without code changes."""
    from vllm_torchtpu import envs

    env = gemma4_patch_env
    monkeypatch.setenv("TPU_FUSED_GATED_MLP_FP8", "0")
    assert not envs.TPU_FUSED_GATED_MLP_FP8
    cfg = SimpleNamespace(architectures=["Gemma4ForCausalLM"], quantization="fp8")
    assert env.adapter.maybe_patch_gated_mlp(cfg) is None
    assert not getattr(env.gemma4_mod.Gemma4MLP.forward, "_tpu_fused_gated_mlp", False)


def test_patch_unquantized_skipped_cpu(gemma4_patch_env):
    """Unquantized checkpoints without QUANTIZE_ON_LOAD_PREFIXES skip patching."""
    env = gemma4_patch_env
    cfg = SimpleNamespace(architectures=["Gemma4ForCausalLM"], quantization=None)
    assert env.adapter.maybe_patch_gated_mlp(cfg) is None
    assert not getattr(env.gemma4_mod.Gemma4MLP.forward, "_tpu_fused_gated_mlp", False)


def test_patch_non_allowlisted_untouched_cpu(gemma4_patch_env):
    """Non-allowlisted models leave both LlamaMLP and Gemma4MLP untouched."""
    env = gemma4_patch_env
    cfg = SimpleNamespace(architectures=["LlamaForCausalLM"], quantization="fp8")
    assert env.adapter.maybe_patch_gated_mlp(cfg) is None
    assert env.llama_mod.LlamaMLP.__dict__["forward"] is env.orig_llama_fwd
    assert not getattr(env.gemma4_mod.Gemma4MLP.forward, "_tpu_fused_gated_mlp", False)


def test_patch_gemma4_dispatches_cpu(gemma4_patch_env, monkeypatch):
    """Quantized Gemma4ForCausalLM patches Gemma4MLP.forward and invokes _fp8_op."""
    env = gemma4_patch_env
    cfg = SimpleNamespace(
        architectures=["Gemma4ForCausalLM"], quantization="compressed-tensors"
    )
    assert env.adapter.maybe_patch_gated_mlp(cfg) is None
    assert getattr(env.gemma4_mod.Gemma4MLP.forward, "_tpu_fused_gated_mlp", False)
    assert env.llama_mod.LlamaMLP.__dict__["forward"] is env.orig_llama_fwd

    fp8_spy = mock.Mock(
        side_effect=lambda gu, w_q, w_s, ak, w8a16, vmem: torch.zeros(
            (gu.shape[0], w_q.shape[1]), dtype=torch.bfloat16
        )
    )
    monkeypatch.setattr(env.adapter, "_fp8_op", fp8_spy)
    mlp, gu_mock = _make_stub_gemma4_mlp(
        env.gemma4_mod, "model.layers.0.mlp.down_proj", tpu_gu_out=True
    )
    out = mlp(torch.zeros((5, 64), dtype=torch.bfloat16))
    assert out.shape == (5, 64)
    assert gu_mock.call_count == 1
    assert fp8_spy.call_count == 1


def test_drafter_prefix_excluded_cpu(gemma4_patch_env, monkeypatch):
    """Gemma4MTP draft_model.* layers fall back once before gate_up_proj and compile."""
    env = gemma4_patch_env
    cfg = SimpleNamespace(architectures=["Gemma4ForCausalLM"], quantization="fp8")
    env.adapter.maybe_patch_gated_mlp(cfg)

    fp8_spy = mock.Mock()
    monkeypatch.setattr(env.adapter, "_fp8_op", fp8_spy)
    logged: list[str] = []
    monkeypatch.setattr(
        env.adapter.logger,
        "info_once",
        lambda msg, *args, **kwargs: logged.append(msg % args if args else msg),
    )

    mtp_mlp, gu_mock = _make_stub_gemma4_mlp(
        env.gemma4_mod, "draft_model.layers.0.mlp.down_proj", tpu_gu_out=False
    )
    out = mtp_mlp(torch.zeros((5, 64), dtype=torch.bfloat16))
    assert out.shape == (5, 64)
    assert gu_mock.call_count == 1
    assert fp8_spy.call_count == 0
    assert any(
        "draft_model.layers.0.mlp.down_proj" in m and "drafter prefix" in m
        for m in logged
    )

    out_compiled = torch.compile(mtp_mlp, backend="eager", fullgraph=True)(
        torch.zeros((5, 64), dtype=torch.bfloat16)
    )
    assert out_compiled.shape == (5, 64)


def test_fallback_logs_cpu(gemma4_patch_env, monkeypatch):
    """Ineligible gate_up_proj and quant_method emit deduplicated info_once logs."""
    env = gemma4_patch_env
    cfg = SimpleNamespace(architectures=["Gemma4ForCausalLM"], quantization="fp8")
    env.adapter.maybe_patch_gated_mlp(cfg)

    logged: list[str] = []
    monkeypatch.setattr(
        env.adapter.logger,
        "info_once",
        lambda msg, *args, **kwargs: logged.append(msg % args if args else msg),
    )

    bad_gu, _ = _make_stub_gemma4_mlp(
        env.gemma4_mod, "model.layers.1.mlp.down_proj", valid_gu=False
    )
    _ = bad_gu(torch.zeros((5, 64), dtype=torch.bfloat16))
    _ = bad_gu(torch.zeros((5, 64), dtype=torch.bfloat16))
    assert len(logged) == 1 and "gate_up_proj is Identity" in logged[0]

    logged.clear()
    bad_qm, _ = _make_stub_gemma4_mlp(
        env.gemma4_mod, "model.layers.2.mlp.down_proj", valid_qm=False
    )
    _ = bad_qm(torch.zeros((5, 64), dtype=torch.bfloat16))
    assert len(logged) == 1 and "unsupported quant_method Identity" in logged[0]


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("bias_present", [True, False])
@pytest.mark.parametrize("reduce_results", [True, False])
@pytest.mark.parametrize("per_tensor_scale", [False, True])
@pytest.mark.parametrize("use_tpu_cls", [True, False])
def test_apply_gated_mlp_tp_bias_and_all_reduce_cpu(
    monkeypatch, tp_size, bias_present, reduce_results, per_tensor_scale, use_tpu_cls
):
    """Check bias placement and TP all-reduce match TpuRowParallelLinear."""
    import vllm.model_executor.layers.linear as linear_mod
    from vllm.model_executor.layers.activation import GeluAndMul
    from vllm.model_executor.layers.linear import RowParallelLinear

    import vllm_torchtpu.layers.adapter.linear as tpu_linear_mod
    from vllm_torchtpu.layers.adapter import gated_mlp_fp8 as adapter
    from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    rng = torch.Generator().manual_seed(7)
    m, d, n = 128, 128, 64
    v7x_vmem = 64 * 1024 * 1024
    bias = torch.randn(n, generator=rng, dtype=torch.bfloat16) if bias_present else None
    w_blocks = [
        (torch.randn((d, n), generator=rng) * 0.05).to(torch.float8_e4m3fn)
        for _ in range(tp_size)
    ]
    w_scales = [
        torch.full((1 if per_tensor_scale else n,), 0.05, dtype=torch.float32)
        for _ in range(tp_size)
    ]
    gu_blocks = [
        torch.randn((m, 2 * d), generator=rng, dtype=torch.bfloat16)
        for _ in range(tp_size)
    ]

    act_fn = GeluAndMul.__new__(GeluAndMul)
    torch.nn.Module.__init__(act_fn)
    act_fn.approximate = "tanh"
    act_fn.forward = (  # type: ignore[method-assign]
        lambda gu: (
            torch.nn.functional.gelu(gu[:, :d].float(), approximate="tanh").bfloat16()
            * gu[:, d:]
        )
    )

    def _fake_kernel(gu, w_q, w_s, act_kind, w8a16_max_rows, vmem):
        del act_kind
        assert w8a16_max_rows == 0 and vmem == v7x_vmem
        act = act_fn.forward(gu).float()
        w_deq = w_q.float() * w_s.unsqueeze(0)
        return (act @ w_deq).bfloat16()

    class _FakeTpuTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="tpu")

    monkeypatch.setattr(adapter, "_fp8_op", _fake_kernel)

    def _make_down_proj(rank: int, w_q: torch.Tensor, w_s: torch.Tensor):
        cls = TpuRowParallelLinear if use_tpu_cls else RowParallelLinear
        layer = cls.__new__(cls)
        torch.nn.Module.__init__(layer)
        qm = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
        qm.w8a16_max_rows = 0

        def _qm_apply(lyr, inp, bias_):
            out = (
                inp.float() @ (lyr.weight.float() * lyr.weight_scale.unsqueeze(0))
            ).bfloat16()
            if bias_ is not None:
                out = out + bias_
            return out

        qm.apply = _qm_apply  # type: ignore[method-assign]
        layer.quant_method = qm
        layer.input_is_parallel = True
        layer.return_bias = True
        layer.skip_bias_add = False
        layer.tp_size = tp_size
        layer.tp_rank = rank
        layer.reduce_results = reduce_results
        layer.register_parameter(
            "bias",
            torch.nn.Parameter(bias, requires_grad=False) if bias is not None else None,
        )
        layer.register_parameter("weight", torch.nn.Parameter(w_q, requires_grad=False))
        layer.register_parameter(
            "weight_scale", torch.nn.Parameter(w_s, requires_grad=False)
        )
        return layer

    # Simulate 2-pass TP all-reduce and verify pre-reduce tensors contain NO bias.
    fused_pre: list[torch.Tensor] = []
    monkeypatch.setattr(
        adapter,
        "tensor_model_parallel_all_reduce",
        lambda t: (fused_pre.append(t.clone()), t)[1],
    )
    for r in range(tp_size):
        dp = _make_down_proj(r, w_blocks[r], w_scales[r])
        gu_fake = gu_blocks[r].as_subclass(_FakeTpuTensor)
        _ = adapter.apply_gated_mlp_act_down(gu_fake, act_fn, dp, v7x_vmem)

    if reduce_results and tp_size > 1:
        assert len(fused_pre) == tp_size
        for r in range(tp_size):
            raw_matmul = _fake_kernel(
                gu_blocks[r], w_blocks[r], w_scales[r], 0, 0, v7x_vmem
            )
            torch.testing.assert_close(torch.Tensor(fused_pre[r]), raw_matmul)
        reduced_sum = sum(torch.Tensor(t) for t in fused_pre)
        monkeypatch.setattr(
            adapter,
            "tensor_model_parallel_all_reduce",
            lambda t: reduced_sum.as_subclass(_FakeTpuTensor),
        )

    fused_outs = [
        torch.Tensor(
            adapter.apply_gated_mlp_act_down(
                gu_blocks[r].as_subclass(_FakeTpuTensor),
                act_fn,
                _make_down_proj(r, w_blocks[r], w_scales[r]),
                v7x_vmem,
            )
        )
        for r in range(tp_size)
    ]

    # Compare against TpuRowParallelLinear.forward on every rank.
    ref_pre: list[torch.Tensor] = []

    def _cap_ref(t):
        ref_pre.append(t.clone())
        return t

    monkeypatch.setattr(linear_mod, "tensor_model_parallel_all_reduce", _cap_ref)
    monkeypatch.setattr(tpu_linear_mod, "tensor_model_parallel_all_reduce", _cap_ref)
    for r in range(tp_size):
        dp_ref = _make_down_proj(r, w_blocks[r], w_scales[r])
        _ = TpuRowParallelLinear.forward(dp_ref, act_fn.forward(gu_blocks[r]))
    if ref_pre:
        ref_sum = sum(ref_pre)
        monkeypatch.setattr(
            linear_mod, "tensor_model_parallel_all_reduce", lambda t: ref_sum
        )
        monkeypatch.setattr(
            tpu_linear_mod, "tensor_model_parallel_all_reduce", lambda t: ref_sum
        )
    ref_outs = [
        TpuRowParallelLinear.forward(
            _make_down_proj(r, w_blocks[r], w_scales[r]),
            act_fn.forward(gu_blocks[r]),
        )[0]
        for r in range(tp_size)
    ]

    for r in range(tp_size):
        torch.testing.assert_close(fused_outs[r], ref_outs[r], msg=f"rank {r}")


def test_jax_op_static_argnums_and_fallback_cpu(monkeypatch):
    """Check static_argnums=(3, 4, 5) and JAX w8a16 / small-m / unaligned fallback."""
    import inspect

    from torch_tpu._internal import pallas as pallas_mod
    from vllm.model_executor.layers.activation import GeluAndMul

    from vllm_torchtpu.kernels import gated_mlp_fp8 as kernel
    from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
    from vllm_torchtpu.layers.adapter import gated_mlp_fp8 as adapter
    from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    sig = inspect.signature(kernel.fused_gated_act_fp8_matmul, follow_wrapped=False)
    params = list(sig.parameters.values())
    assert [i for i, p in enumerate(params) if p.annotation is int] == [3, 4, 5]
    if hasattr(pallas_mod, "_infer_static_argnums"):
        assert pallas_mod._infer_static_argnums(sig) == (3, 4, 5)

    # Verify w8a16_max_rows is forwarded as static arg 4 by apply_gated_mlp_act_down.
    seen_args: list[tuple[int, int, int]] = []

    def _capture_fp8_op(gu, w_q, w_s, act_kind, w8a16_max_rows, vmem):
        seen_args.append((act_kind, w8a16_max_rows, vmem))
        return torch.zeros((gu.shape[0], w_q.shape[1]), dtype=gu.dtype)

    monkeypatch.setattr(adapter, "_fp8_op", _capture_fp8_op)

    class _FakeTpuTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="tpu")

    d, n = 128, 64
    act_fn = GeluAndMul.__new__(GeluAndMul)
    torch.nn.Module.__init__(act_fn)
    act_fn.approximate = "tanh"
    qm = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
    qm.w8a16_max_rows = 256

    dp = TpuRowParallelLinear.__new__(TpuRowParallelLinear)
    torch.nn.Module.__init__(dp)
    dp.quant_method = qm
    dp.input_is_parallel = True
    dp.return_bias = True
    dp.skip_bias_add = False
    dp.tp_size = 1
    dp.tp_rank = 0
    dp.reduce_results = True
    dp.register_parameter("bias", None)
    dp.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.zeros((d, n), dtype=torch.float8_e4m3fn), requires_grad=False
        ),
    )
    dp.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones((n,), dtype=torch.float32), requires_grad=False),
    )

    v7x_vmem = 64 * 1024 * 1024
    for m in (32, 256):
        gu = torch.ones((m, 2 * d), dtype=torch.bfloat16).as_subclass(_FakeTpuTensor)
        out = adapter.apply_gated_mlp_act_down(gu, act_fn, dp, v7x_vmem)
        assert out.shape == (m, n)
    assert seen_args == [(0, 256, v7x_vmem), (0, 256, v7x_vmem)]

    # Verify JAX-level fallback skips fused_gated_act_fp8_quant when m < 128,
    # unaligned (m=130), or m <= w8a16_max_rows.
    quant_calls = 0

    def _boom(*args, **kwargs):
        nonlocal quant_calls
        quant_calls += 1
        raise AssertionError("Pallas kernel should not run on fallback buckets")

    monkeypatch.setattr(kernel, "fused_gated_act_fp8_quant", _boom)

    w_q_j = jnp.ones((d, n), dtype=jnp.float8_e4m3fn)
    w_s_j = jnp.full((n,), 0.05, dtype=jnp.float32)
    for m, w8a16_rows in ((32, 0), (130, 0), (256, 256)):
        gu_j = jax.random.normal(jax.random.key(m), (m, 2 * d), jnp.float32).astype(
            jnp.bfloat16
        )
        out_j = kernel.fused_gated_act_fp8_matmul(
            gu_j,
            w_q_j,
            w_s_j,
            act_kind=0,
            w8a16_max_rows=w8a16_rows,
            vmem_capacity_bytes=v7x_vmem,
        )
        act_j = (
            jax.nn.gelu(gu_j[:, :d].astype(jnp.float32), approximate=True).astype(
                jnp.bfloat16
            )
            * gu_j[:, d:]
        )
        ref_j = xla_quantized_matmul(
            act_j, w_q_j, w_s_j, quantize_activation=not (0 < m <= w8a16_rows)
        )
        np.testing.assert_array_equal(np.asarray(out_j), np.asarray(ref_j))

    # d < 128 or d % 128 != 0 also falls back cleanly to xla_quantized_matmul.
    for d_unaligned in (64, 192):
        gu_u = jax.random.normal(
            jax.random.key(d_unaligned), (128, 2 * d_unaligned), jnp.float32
        ).astype(jnp.bfloat16)
        wq_u = jnp.ones((d_unaligned, n), dtype=jnp.float8_e4m3fn)
        out_u = kernel.fused_gated_act_fp8_matmul(
            gu_u, wq_u, w_s_j, act_kind=0, vmem_capacity_bytes=v7x_vmem
        )
        act_u = (
            jax.nn.gelu(
                gu_u[:, :d_unaligned].astype(jnp.float32), approximate=True
            ).astype(jnp.bfloat16)
            * gu_u[:, d_unaligned:]
        )
        ref_u = xla_quantized_matmul(act_u, wq_u, w_s_j, quantize_activation=True)
        np.testing.assert_array_equal(np.asarray(out_u), np.asarray(ref_u))
    assert quant_calls == 0


def test_kernel_input_and_tile_validation_cpu():
    """Verify standalone kernel guards on dtype, rank, d/bm alignment, VMEM, and w_s."""
    from vllm_torchtpu.kernels import gated_mlp_fp8 as kernel

    vmem = 64 * 1024 * 1024
    assert kernel.select_block_m(128, 64, vmem) is None
    assert kernel.select_block_m(128, 192, vmem) is None
    assert kernel.select_block_m(0, 128, vmem) is None
    assert kernel.select_block_m(128, 128, 0) is None

    gu = jnp.zeros((128, 256), dtype=jnp.bfloat16)
    w_q = jnp.zeros((128, 64), dtype=jnp.float8_e4m3fn)
    w_s = jnp.ones((64,), dtype=jnp.float32)

    # 1-byte FP8 dtype check rejects bf16, fp32, and int8.
    for bad_dt in (jnp.bfloat16, jnp.float32, jnp.int8):
        with pytest.raises(ValueError, match="1-byte floating FP8"):
            kernel.fused_gated_act_fp8_quant(
                gu, bad_dt, bm=64, vmem_capacity_bytes=vmem
            )
        with pytest.raises(ValueError, match="1-byte floating FP8"):
            kernel.fused_gated_act_fp8_matmul(
                gu, jnp.zeros((128, 64), dtype=bad_dt), w_s, vmem_capacity_bytes=vmem
            )

    # gate_up rank, dtype, and positive-evenness checks.
    for bad_gu in (
        jnp.zeros((256,), dtype=jnp.bfloat16),
        jnp.zeros((1, 128, 256), dtype=jnp.bfloat16),
        jnp.zeros((128, 256), dtype=jnp.float32),
        jnp.zeros((128, 255), dtype=jnp.bfloat16),
        jnp.zeros((128, 0), dtype=jnp.bfloat16),
    ):
        with pytest.raises(ValueError):
            kernel.fused_gated_act_fp8_matmul(
                bad_gu, w_q, w_s, vmem_capacity_bytes=vmem
            )

    # Direct quant entry enforces d >= 128, d % 128 == 0, bm alignment, and VMEM.
    with pytest.raises(ValueError, match="d >= 128"):
        kernel.fused_gated_act_fp8_quant(
            jnp.zeros((128, 128), dtype=jnp.bfloat16),
            jnp.float8_e4m3fn,
            bm=64,
            vmem_capacity_bytes=vmem,
        )
    for m_bad, bm_bad, vmem_bad in (
        (12, 12, vmem),  # bm % 8 != 0
        (128, 16, vmem),  # m > bm but bm % 32 != 0
        (128, 64, 1024),  # exceeds scoped VMEM limit
    ):
        with pytest.raises(ValueError, match="Invalid tile config"):
            kernel.fused_gated_act_fp8_quant(
                jnp.zeros((m_bad, 256), dtype=jnp.bfloat16),
                jnp.float8_e4m3fn,
                bm=bm_bad,
                vmem_capacity_bytes=vmem_bad,
            )

    # w_q 2D and w_s 1D (N,) or (1,) checks reject 2D or mismatched scales.
    with pytest.raises(ValueError, match="Expected 2D w_q"):
        kernel.fused_gated_act_fp8_matmul(
            gu,
            jnp.zeros((128, 64, 1), dtype=jnp.float8_e4m3fn),
            w_s,
            vmem_capacity_bytes=vmem,
        )
    for bad_ws in (
        jnp.ones((64, 1), dtype=jnp.float32),
        jnp.ones((1, 64), dtype=jnp.float32),
        jnp.ones((65,), dtype=jnp.float32),
        jnp.ones((64,), dtype=jnp.int32),
    ):
        with pytest.raises(ValueError, match="Expected 1D"):
            kernel.fused_gated_act_fp8_matmul(gu, w_q, bad_ws, vmem_capacity_bytes=vmem)

    for bad_act in (99, True):
        with pytest.raises(ValueError, match="Unsupported gated activation"):
            kernel.fused_gated_act_fp8_matmul(
                gu, w_q, w_s, act_kind=bad_act, vmem_capacity_bytes=vmem
            )


@pytest.mark.parametrize("act_kind", [0, 1])
@pytest.mark.parametrize("per_tensor_scale", [False, True])
def test_kernel_pallas_interpret_parity_cpu(monkeypatch, act_kind, per_tensor_scale):
    """Run _gated_act_fp8_kernel + fused_gated_act_fp8_matmul via Pallas interpret."""
    import jax.experimental.pallas as pl

    from vllm_torchtpu.kernels import gated_mlp_fp8 as kernel
    from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

    orig_pallas_call = pl.pallas_call

    def _interpret_pallas_call(
        f, *, out_shape, grid_spec, compiler_params=None, name=None
    ):
        del compiler_params
        return orig_pallas_call(
            f,
            out_shape=out_shape,
            grid_spec=pl.GridSpec(
                grid=grid_spec.grid,
                in_specs=grid_spec.in_specs,
                out_specs=grid_spec.out_specs,
            ),
            interpret=True,
            name=name,
        )

    monkeypatch.setattr(pl, "pallas_call", _interpret_pallas_call)

    m, d, n = 128, 256, 64
    vmem = 64 * 1024 * 1024
    k1, k2 = jax.random.split(jax.random.key(19 + act_kind))
    gu = jax.random.normal(k1, (m, 2 * d), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.normal(k2, (d, n), jnp.float32) * 0.05
    w_q, w_s_ch = quantize_tensor(w.T, jnp.float8_e4m3fn)
    w_q = w_q.T
    w_s = (
        jnp.array([0.05], dtype=jnp.float32) if per_tensor_scale else w_s_ch.reshape(-1)
    )

    out = kernel.fused_gated_act_fp8_matmul(
        gu, w_q, w_s, act_kind=act_kind, w8a16_max_rows=0, vmem_capacity_bytes=vmem
    )
    ref, _ = _reference(gu, w_q, w_s, act_kind=act_kind)
    assert out.shape == (m, n) and out.dtype == jnp.bfloat16
    f = np.asarray(out.astype(jnp.float32))
    r = np.asarray(ref.astype(jnp.float32))
    cos = float(np.sum(f * r) / (np.linalg.norm(f) * np.linalg.norm(r)))
    assert cos > 0.9999


@pytest.mark.tpu_test
@pytest.mark.parametrize("m", [128, 256, 2048])
@pytest.mark.parametrize("d", [10752, 2688])
@pytest.mark.parametrize("act_kind", [0, 1])
def test_fused_quant_intermediate_bit_accuracy_tpu(m, d, act_kind):
    """Compare fused (x_q, x_s) directly against quantize_tensor(act): <=1 FP8 bin."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("requires TPU")
    from vllm_torchtpu.kernels import gated_mlp_fp8 as kernel
    from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

    vmem = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    bm = kernel.select_block_m(m, d, vmem)
    assert bm is not None

    gu = jax.random.normal(jax.random.key(11), (m, 2 * d), jnp.float32).astype(
        jnp.bfloat16
    )
    x_q, x_s = jax.jit(
        lambda x: kernel.fused_gated_act_fp8_quant(
            x, jnp.float8_e4m3fn, act_kind, bm=bm, vmem_capacity_bytes=vmem
        )
    )(gu)

    @jax.jit
    def _ref_quant(x):
        act = (
            _apply_ref_act(x[:, :d].astype(jnp.float32), act_kind).astype(x.dtype)
            * x[:, d:]
        )
        return quantize_tensor(act, jnp.float8_e4m3fn)

    ref_q, ref_s = _ref_quant(gu)

    u8_fused = np.asarray(jax.lax.bitcast_convert_type(x_q, jnp.uint8))
    u8_ref = np.asarray(jax.lax.bitcast_convert_type(ref_q, jnp.uint8))
    exact_frac = float(np.mean(u8_fused == u8_ref))
    # Compare magnitude bins (low 7 bits; sign bit matches everywhere away from +/-0).
    same_sign = (u8_fused >> 7) == (u8_ref >> 7)
    bin_diff = np.where(
        same_sign,
        np.abs((u8_fused & 0x7F).astype(np.int16) - (u8_ref & 0x7F).astype(np.int16)),
        (u8_fused & 0x7F).astype(np.int16) + (u8_ref & 0x7F).astype(np.int16),
    )
    s_fused = np.asarray(x_s)
    s_ref = np.asarray(ref_s)
    max_scale_rel = float(np.max(np.abs(s_fused - s_ref) / np.maximum(s_ref, 1e-12)))

    # Pallas VPU tanh/sigmoid vs XLA tanh/sigmoid differs by <=1 bf16 ULP (~1/128) on
    # ~2.8% of elements; every element stays within <=1 FP8 bin (0 elements >1 bin).
    assert exact_frac >= 0.965, (
        f"Expected >=96.5% bit-exact FP8 elements, got {exact_frac:.4f}"
    )
    assert int(np.max(bin_diff)) <= 1, f"Max FP8 bin diff {int(np.max(bin_diff))} > 1"
    assert max_scale_rel < 1e-2, f"Max scale rel diff too high: {max_scale_rel:.6e}"


# (10752, 5376) is Gemma 4 31B at TP=2 (intermediate 21504 / 2 = 10752 per rank);
# (9728, 4096) exercises bm=256 at 52.25 MiB right below the 56 MiB VMEM cap;
# (5376, 5376) is TP=4; (5632, 4096) checks VMEM headroom above d=5376.
@pytest.mark.tpu_test
@pytest.mark.parametrize("m", [128, 216, 256, 512, 2048, 16384])
@pytest.mark.parametrize(
    "d,n",
    [(10752, 5376), (9728, 4096), (5376, 5376), (5632, 4096), (1024, 512)],
)
@pytest.mark.parametrize("act_kind", [0, 1])
def test_fused_matches_unfused_tpu(m, d, n, act_kind):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("requires TPU")
    if m == 16384 and d not in (10752, 5376):
        pytest.skip("16k covered on Gemma 4 TP=2 and TP=4 shapes")
    from vllm_torchtpu.kernels import gated_mlp_fp8 as kernel
    from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

    vmem = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    bm = kernel.select_block_m(m, d, vmem)
    assert bm is not None, f"Expected valid block_m for (m={m}, d={d})"

    k1, k2 = jax.random.split(jax.random.key(42))
    gate_up = jax.random.normal(k1, (m, 2 * d), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.normal(k2, (d, n), jnp.float32) * 0.02
    w_q, w_s = quantize_tensor(w.T, jnp.float8_e4m3fn)
    w_q, w_s = w_q.T, w_s.reshape(-1)

    fused = jax.jit(
        lambda gu, wq, ws: kernel.fused_gated_act_fp8_matmul(
            gu, wq, ws, act_kind=act_kind, w8a16_max_rows=0, vmem_capacity_bytes=vmem
        )
    )(gate_up, w_q, w_s)
    ref, act = jax.jit(lambda gu, wq, ws: _reference(gu, wq, ws, act_kind=act_kind))(
        gate_up, w_q, w_s
    )
    oracle = jax.jit(_fp32_oracle)(act, w_q, w_s)

    assert fused.shape == ref.shape and fused.dtype == ref.dtype
    f = np.asarray(fused.astype(jnp.float32))
    r = np.asarray(ref.astype(jnp.float32))
    o = np.asarray(oracle.astype(jnp.float32))

    cos_fr = float(np.sum(f * r) / (np.linalg.norm(f) * np.linalg.norm(r)))
    rms_rel_fr = float(np.linalg.norm(f - r) / (np.linalg.norm(r)))
    cos_fo = float(np.sum(f * o) / (np.linalg.norm(f) * np.linalg.norm(o)))
    cos_ro = float(np.sum(r * o) / (np.linalg.norm(r) * np.linalg.norm(o)))

    assert cos_fr > 0.9997, f"fused vs ref cosine too low: {cos_fr:.6f}"
    assert rms_rel_fr < 0.025, f"fused vs ref RMS rel diff too high: {rms_rel_fr:.6f}"
    assert cos_fo > 0.9995, f"fused vs fp32 oracle cosine too low: {cos_fo:.6f}"
    assert abs(cos_fo - cos_ro) < 2e-4, (
        f"fused vs oracle ({cos_fo:.6f}) diverged from ref vs oracle ({cos_ro:.6f})"
    )


@pytest.mark.tpu_test
def test_apply_gated_mlp_act_down_torch_compile_tpu_backend(monkeypatch):
    """Compile m=256 (fused) and m=64/32 (JAX fallback) via backend='tpu' fullgraph."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("requires TPU")
    from vllm.model_executor.layers.activation import GeluAndMul
    from vllm.model_executor.layers.linear import RowParallelLinear

    import vllm_torchtpu.compilation.tpu_compiler  # noqa: F401
    from vllm_torchtpu.kernels import gated_mlp_fp8 as kernel
    from vllm_torchtpu.layers.adapter import gated_mlp_fp8 as adapter
    from vllm_torchtpu.layers.adapter.linear_common import quantized_matmul
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    d, n = 1024, 512
    vmem = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )

    real_quant = kernel.fused_gated_act_fp8_quant
    quant_traced_m: list[int] = []

    def _spy_quant(gate_up, quant_dtype, act_kind=0, *, bm, vmem_capacity_bytes):
        quant_traced_m.append(int(gate_up.shape[0]))
        return real_quant(
            gate_up,
            quant_dtype,
            act_kind,
            bm=bm,
            vmem_capacity_bytes=vmem_capacity_bytes,
        )

    monkeypatch.setattr(kernel, "fused_gated_act_fp8_quant", _spy_quant)

    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        act_fn = GeluAndMul(approximate="tanh")

    qm = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
    qm.w8a16_max_rows = 0

    down_proj = RowParallelLinear.__new__(RowParallelLinear)
    torch.nn.Module.__init__(down_proj)
    down_proj.prefix = "model.layers.0.mlp.down_proj"
    down_proj.quant_method = qm
    down_proj.input_is_parallel = True
    down_proj.return_bias = True
    down_proj.skip_bias_add = False
    down_proj.tp_size = 1
    down_proj.tp_rank = 0
    down_proj.reduce_results = False
    down_proj.register_parameter("bias", None)
    w_fp8 = (torch.randn((d, n), dtype=torch.float32) * 0.02).to(torch.float8_e4m3fn)
    w_scale = torch.full((n,), 0.02, dtype=torch.float32)
    down_proj.register_parameter(
        "weight", torch.nn.Parameter(w_fp8.to("tpu"), requires_grad=False)
    )
    down_proj.register_parameter(
        "weight_scale", torch.nn.Parameter(w_scale.to("tpu"), requires_grad=False)
    )
    down_proj.forward = (  # type: ignore[method-assign]
        lambda x: (quantized_matmul(x, down_proj.weight, down_proj.weight_scale), None)
    )

    for m, expected_quant_m in ((256, [256]), (64, []), (32, [])):
        quant_traced_m.clear()
        compiled_bucket = torch.compile(
            lambda gu: adapter.apply_gated_mlp_act_down(gu, act_fn, down_proj, vmem),
            backend="tpu",
            dynamic=False,
            fullgraph=True,
        )
        gu_cpu = torch.randn((m, 2 * d), dtype=torch.bfloat16)
        gu_tpu = gu_cpu.to("tpu")
        out_compiled = compiled_bucket(gu_tpu).cpu().float()
        assert quant_traced_m == expected_quant_m, (
            f"m={m}: expected Pallas trace {expected_quant_m}, got {quant_traced_m}"
        )
        ref = (
            quantized_matmul(act_fn(gu_tpu), down_proj.weight, down_proj.weight_scale)
            .cpu()
            .float()
        )
        assert out_compiled.shape == (m, n)
        cos = float(
            torch.sum(out_compiled * ref)
            / (torch.linalg.norm(out_compiled) * torch.linalg.norm(ref))
        )
        assert cos > 0.9997, f"m={m} compiled fullgraph cos={cos:.6f}"
