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
"""AST-discovery, binding, and fullgraph tests for the RMSNorm+FP8 adapter."""

import contextlib
from types import SimpleNamespace
from unittest import mock

import jax
import jax.experimental.pallas.tpu as pltpu
import pytest
import torch
from vllm import envs as vllm_envs
from vllm import ir
from vllm.compilation.backends import set_model_tag
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.config.lora import LoRAConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models import gemma4 as gemma4_mod
from vllm.model_executor.models import llama as llama_mod
from vllm.model_executor.models import qwen2 as qwen2_mod
from vllm.model_executor.models.registry import ModelRegistry

from vllm_torchtpu.layers.adapter import rmsnorm_fp8 as adapter
from vllm_torchtpu.layers.adapter.ir_ops import TPU_IR_PROVIDER
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
    quantized_matmul,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


class _FakeTpuTensor(torch.Tensor):
    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu", index=0)


def _is_fused_op_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and "fused_rmsnorm_fp8_matmul" in str(node.target)


def _fused_op_node_count(gm: torch.fx.GraphModule) -> int:
    return sum(_is_fused_op_node(node) for node in gm.graph.nodes)


def _make_fp8_linear(
    cls: type[torch.nn.Module],
    k: int,
    n_out: int,
    *,
    w8a16_max_rows: int = 0,
    prefix: str = "model.layers.0.mlp.gate_up_proj",
    flipped: bool = True,
    bias_tensor: torch.Tensor | None = None,
    skip_bias_add: bool = False,
    return_bias: bool = True,
    weight_tensor: torch.Tensor | None = None,
    scale_tensor: torch.Tensor | None = None,
):
    lin = cls.__new__(cls)
    torch.nn.Module.__init__(lin)
    qm = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
    qm.w8a16_max_rows = w8a16_max_rows
    lin.quant_method = qm
    lin.prefix = prefix
    lin.gather_output = False
    lin.skip_bias_add = skip_bias_add
    lin.return_bias = return_bias
    if flipped:
        setattr(lin, WEIGHT_FLIPPED_ATTR, True)
    if bias_tensor is None:
        lin.register_parameter("bias", None)
    else:
        lin.register_parameter(
            "bias", torch.nn.Parameter(bias_tensor, requires_grad=False)
        )
    w = (
        torch.empty((k, n_out), dtype=torch.float8_e4m3fn, device="meta")
        if weight_tensor is None
        else weight_tensor
    )
    s = (
        torch.empty((n_out,), dtype=torch.float32, device="meta")
        if scale_tensor is None
        else scale_tensor
    )
    lin.register_parameter("weight", torch.nn.Parameter(w, requires_grad=False))
    lin.register_parameter("weight_scale", torch.nn.Parameter(s, requires_grad=False))
    return lin


def test_can_fuse_controls_and_ast_discovery_cpu(monkeypatch):
    """Verify MergedColumnParallelLinear scoping, must-fail guards, and AST matching."""
    v7x_vmem = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        rms = RMSNorm(5376, eps=1e-6)
        rms_no_w = RMSNorm(5376, eps=1e-6, has_weight=False)
        rms_var_ovr = RMSNorm(5376, eps=1e-6, var_hidden_size=2048)
        gemma_rms = GemmaRMSNorm(5376, eps=1e-6)

    assert adapter._resolve_gamma_offset(rms) == 0.0
    assert adapter._resolve_gamma_offset(gemma_rms) == 1.0
    assert adapter._resolve_gamma_offset(rms_no_w) is None
    assert adapter._resolve_gamma_offset(rms_var_ovr) is None

    x_256 = torch.zeros((256, 5376), dtype=torch.bfloat16).as_subclass(_FakeTpuTensor)
    x_64 = torch.zeros((64, 5376), dtype=torch.bfloat16).as_subclass(_FakeTpuTensor)

    lin_merged = _make_fp8_linear(MergedColumnParallelLinear, 5376, 21504)
    assert adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_merged, v7x_vmem)
    assert adapter.can_fuse_rmsnorm_fp8_linear(x_256, gemma_rms, lin_merged, v7x_vmem)
    # Dynamo-level check stays guard-free across M < 512 and w8a16_max_rows > 0:
    assert adapter.can_fuse_rmsnorm_fp8_linear(x_64, rms, lin_merged, v7x_vmem)
    lin_w8a16 = _make_fp8_linear(
        MergedColumnParallelLinear, 5376, 21504, w8a16_max_rows=256
    )
    assert adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_w8a16, v7x_vmem)

    # Attention QKVParallelLinear, plain ColumnParallelLinear, RowParallelLinear,
    # and subclasses of MergedColumnParallelLinear must stay unfused:
    class _SubMergedLinear(MergedColumnParallelLinear):
        pass

    for rejected_cls in (
        QKVParallelLinear,
        ColumnParallelLinear,
        RowParallelLinear,
        _SubMergedLinear,
    ):
        lin_rej = _make_fp8_linear(rejected_cls, 5376, 8192)
        assert not adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_rej, v7x_vmem)

    # Norm, layout, unflipped square weight, drafter prefix, and batch-invariant guards:
    assert not adapter.can_fuse_rmsnorm_fp8_linear(
        x_256, rms_no_w, lin_merged, v7x_vmem
    )
    assert not adapter.can_fuse_rmsnorm_fp8_linear(
        x_256, rms_var_ovr, lin_merged, v7x_vmem
    )

    setattr(lin_merged, KEEP_VLLM_LAYOUT_ATTR, True)
    assert not adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_merged, v7x_vmem)
    delattr(lin_merged, KEEP_VLLM_LAYOUT_ATTR)

    lin_unflipped_sq = _make_fp8_linear(
        MergedColumnParallelLinear, 5376, 5376, flipped=False
    )
    assert not adapter.can_fuse_rmsnorm_fp8_linear(
        x_256, rms, lin_unflipped_sq, v7x_vmem
    )

    lin_drafter = _make_fp8_linear(
        MergedColumnParallelLinear,
        5376,
        21504,
        prefix="draft_model.layers.0.mlp.gate_up_proj",
    )
    assert not adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_drafter, v7x_vmem)

    lin_block_scale = _make_fp8_linear(
        MergedColumnParallelLinear,
        5376,
        21504,
        scale_tensor=torch.empty((1, 42, 1, 21504), dtype=torch.float32, device="meta"),
    )
    assert not adapter.can_fuse_rmsnorm_fp8_linear(
        x_256, rms, lin_block_scale, v7x_vmem
    )

    monkeypatch.setattr(vllm_envs, "VLLM_BATCH_INVARIANT", True)
    assert not adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_merged, v7x_vmem)
    monkeypatch.setattr(vllm_envs, "VLLM_BATCH_INVARIANT", False)

    # Real upstream vLLM class AST discovery checks:
    assert (
        "pre_feedforward_layernorm",
        "mlp",
        0,
    ) in adapter._find_fusible_norm_submod_pairs(gemma4_mod.Gemma4DecoderLayer.forward)
    assert adapter._find_fusible_norm_submod_pairs(
        llama_mod.LlamaDecoderLayer.forward
    ) == [("post_attention_layernorm", "mlp", 0)]
    assert adapter._find_fusible_norm_submod_pairs(
        qwen2_mod.Qwen2DecoderLayer.forward
    ) == [("post_attention_layernorm", "mlp", 0)]
    assert adapter._entry_projection_attr(gemma4_mod.Gemma4MLP.forward, 0) == (
        "gate_up_proj"
    )
    assert adapter._entry_projection_attr(qwen2_mod.Qwen2MLP.forward, 0) == (
        "gate_up_proj"
    )
    # Gemma4Attention branching (`q_proj` vs `qkv_proj`) is intentionally rejected:
    assert (
        adapter._entry_projection_attr(
            gemma4_mod.Gemma4Attention.forward, "hidden_states"
        )
        is None
    )

    class _MultiUseNormLayer(torch.nn.Module):
        def forward(self, hidden_states):
            normed = self.pre_feedforward_layernorm(hidden_states)
            out = self.mlp(normed)
            return out + normed

    class _LeakyRebindMLP(torch.nn.Module):
        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up) + x
            return x

    assert adapter._find_fusible_norm_submod_pairs(_MultiUseNormLayer.forward) == []
    assert adapter._entry_projection_attr(_LeakyRebindMLP.forward, 0) is None

    # Binding changes the module instances, so a norm, submodule or projection
    # that the forward references twice must stay unbound.
    class _NormCalledTwiceLayer(torch.nn.Module):
        def forward(self, hidden_states):
            normed = self.norm(hidden_states)
            normed = self.mlp(normed)
            return normed + self.norm(hidden_states)

    class _SubmodCalledTwiceLayer(torch.nn.Module):
        def forward(self, hidden_states):
            normed = self.norm(hidden_states)
            normed = self.mlp(normed)
            return self.mlp(normed)

    class _ProjCalledTwiceMLP(torch.nn.Module):
        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            out, _ = self.gate_up_proj(gate_up)
            return out

    for twice_cls in (_NormCalledTwiceLayer, _SubmodCalledTwiceLayer):
        assert adapter._find_fusible_norm_submod_pairs(twice_cls.forward) == []
    assert adapter._entry_projection_attr(_ProjCalledTwiceMLP.forward, 0) is None

    # One-step scan: an adjacent pair (`norm1 -> norm2`) must not skip (`norm2 -> mlp`).
    class _ChainedNormsLayer(torch.nn.Module):
        def forward(self, hidden_states):
            h = self.norm1(hidden_states)
            h = self.norm2(h)
            h = self.mlp(h)
            return h

    assert ("norm2", "mlp", 0) in adapter._find_fusible_norm_submod_pairs(
        _ChainedNormsLayer.forward
    )

    # Dtypes that fused_rmsnorm_fp8_matmul rejects fall back here instead:
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        rms_f64 = RMSNorm(5376, eps=1e-6).double()
    lin_int_scale = _make_fp8_linear(
        MergedColumnParallelLinear,
        5376,
        21504,
        scale_tensor=torch.empty((21504,), dtype=torch.int32, device="meta"),
    )
    assert not adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms_f64, lin_merged, v7x_vmem)
    assert not adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_int_scale, v7x_vmem)

    assert adapter.maybe_patch_rmsnorm_fp8(None) is False
    gemma4_config = SimpleNamespace(
        architectures=["Gemma4ForCausalLM"], model_impl="auto"
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        adapter.logger,
        "warning_once",
        lambda msg, *args, **kwargs: warnings.append(msg % args if args else msg),
    )
    # Flag unset: returns before the chip query or any class scan.
    monkeypatch.delenv("TPU_FUSED_RMSNORM_FP8", raising=False)
    scan = mock.Mock(side_effect=AssertionError("scanned with the flag off"))
    monkeypatch.setattr(adapter, "get_chip_version", scan)
    monkeypatch.setattr(adapter, "_all_subclasses", scan)
    assert adapter.maybe_patch_rmsnorm_fp8(gemma4_config) is None
    assert warnings == []
    # Flag set where it cannot apply: no scan, and one warning naming the cause.
    monkeypatch.setenv("TPU_FUSED_RMSNORM_FP8", "1")
    monkeypatch.setattr(adapter, "get_chip_version", lambda: pltpu.ChipVersion.TPU_V6E)
    assert adapter.maybe_patch_rmsnorm_fp8(gemma4_config) is None
    assert len(warnings) == 1
    assert "TPU_FUSED_RMSNORM_FP8=1 is ignored on chip" in warnings[0]
    monkeypatch.setattr(adapter, "get_chip_version", lambda: pltpu.ChipVersion.TPU_7X)
    monkeypatch.setattr(vllm_envs, "VLLM_BATCH_INVARIANT", True)
    assert adapter.maybe_patch_rmsnorm_fp8(gemma4_config) is None
    assert len(warnings) == 2
    assert (
        "TPU_FUSED_RMSNORM_FP8=1 is ignored because VLLM_BATCH_INVARIANT=1"
        in warnings[1]
    )
    scan.assert_not_called()

    # Ray copies only VLLM_* and additional_env_vars to remote workers, where
    # maybe_patch_rmsnorm_fp8 reads the flag at model load.
    assert "TPU_FUSED_RMSNORM_FP8" in TpuPlatform.additional_env_vars


def test_maybe_patch_installs_decoder_init_hooks_once_cpu(monkeypatch):
    """With the flag on for v7x, the class scan patches real decoders once."""
    decoder_classes = (
        gemma4_mod.Gemma4DecoderLayer,
        llama_mod.LlamaDecoderLayer,
        qwen2_mod.Qwen2DecoderLayer,
    )

    def _own_inits() -> dict[type, object]:
        return {
            cls: cls.__dict__["__init__"]
            for cls in adapter._all_subclasses(torch.nn.Module)
            if cls.__module__.startswith("vllm.model_executor.models.")
            and "__init__" in cls.__dict__
        }

    gemma4_config = SimpleNamespace(
        architectures=["Gemma4ForCausalLM"], model_impl="auto"
    )
    # Import the model modules first, so the snapshot below covers every class
    # the scan can patch and teardown restores all of them.
    ModelRegistry.resolve_model_cls(
        gemma4_config.architectures, model_config=gemma4_config
    )

    # Registers every candidate __init__ for restore at teardown.
    orig_inits = _own_inits()
    for cls, init_fn in orig_inits.items():
        monkeypatch.setattr(cls, "__init__", init_fn)
    for cls in decoder_classes:
        assert not getattr(orig_inits[cls], "_tpu_fused_rmsnorm_fp8", False)

    monkeypatch.setenv("TPU_FUSED_RMSNORM_FP8", "1")
    monkeypatch.setattr(vllm_envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(adapter, "get_chip_version", lambda: pltpu.ChipVersion.TPU_7X)

    assert adapter.maybe_patch_rmsnorm_fp8(gemma4_config) is None
    first_inits = _own_inits()
    assert first_inits.keys() == orig_inits.keys()
    for cls in decoder_classes:
        patched = first_inits[cls]
        assert getattr(patched, "_tpu_fused_rmsnorm_fp8", False)
        assert patched.__wrapped__ is orig_inits[cls]
    # A class whose forward has no norm -> submodule pair is left alone.
    assert first_inits[gemma4_mod.Gemma4MLP] is orig_inits[gemma4_mod.Gemma4MLP]

    # A second call (model_load after platform_activation) must not wrap again.
    assert adapter.maybe_patch_rmsnorm_fp8(gemma4_config) is None
    second_inits = _own_inits()
    assert all(second_inits[cls] is first_inits[cls] for cls in first_inits)


@contextlib.contextmanager
def _rmsnorm_priority(first: str):
    priority = [first] if first == "native" else [first, "native"]
    with (
        ir.ops.rms_norm.set_priority(priority),
        ir.ops.fused_add_rms_norm.set_priority(priority),
    ):
        yield


def test_binding_requires_tpu_provider_and_scanned_forward_cpu():
    """A native RMSNorm provider or a subclass's own forward stays unfused."""
    k = 128
    vmem = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )

    class _MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = _make_fp8_linear(MergedColumnParallelLinear, k, 512)

        def forward(self, x):
            x, _ = self.gate_up_proj(x)
            return x

    class _PreNormLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RMSNorm(k, eps=1e-6)
            self.mlp = _MLP()

        def forward(self, hidden_states):
            hidden_states = self.norm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            return hidden_states

    class _NormAfterLayer(_PreNormLayer):
        def forward(self, hidden_states):
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.norm(hidden_states)
            return hidden_states

    pairs = adapter._find_fusible_norm_submod_pairs(_PreNormLayer.forward)
    assert pairs == [("norm", "mlp", 0)]
    _PreNormLayer.__init__ = adapter._make_patched_decoder_init(  # type: ignore[method-assign]
        _PreNormLayer.__init__, _PreNormLayer.forward, pairs, vmem
    )

    lora_vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    lora_vllm_config.lora_config = LoRAConfig()
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        with _rmsnorm_priority(TPU_IR_PROVIDER):
            pre_norm = _PreNormLayer()
            norm_after = _NormAfterLayer()
            # Drafters are built under their own model tag.
            with set_model_tag("eagle_head"):
                drafter = _PreNormLayer()
        with _rmsnorm_priority("native"):
            native_provider = _PreNormLayer()
    # LoRA wraps gate_up_proj later and bypasses its forward, so a deferred
    # norm would never run.
    with set_current_vllm_config(lora_vllm_config), _rmsnorm_priority(TPU_IR_PROVIDER):
        with_lora = _PreNormLayer()

    assert "forward" in pre_norm.norm.__dict__
    assert "forward" in pre_norm.mlp.gate_up_proj.__dict__
    for unfused in (norm_after, native_provider, drafter, with_lora):
        assert "forward" not in unfused.norm.__dict__
        assert "forward" not in unfused.mlp.gate_up_proj.__dict__


def test_decoder_layer_binding_and_torch_compile_fullgraph_cpu(monkeypatch):
    """Verify Gemma4 (1-arg) and Llama/Qwen (2-arg) MLP binding under fullgraph=True."""
    k, n_qkv, n_gu = 128, 256, 512
    vmem = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )
    rng = torch.Generator().manual_seed(17)

    def _cpu_fp8_proj(cls: type[torch.nn.Module], n_out: int, prefix: str):
        w_q = (torch.randn((k, n_out), generator=rng) * 0.05).to(torch.float8_e4m3fn)
        w_s = torch.full((n_out,), 0.04, dtype=torch.float32)
        bias = (torch.randn((n_out,), generator=rng) * 0.01).to(torch.bfloat16)
        proj = _make_fp8_linear(
            cls,
            k,
            n_out,
            prefix=prefix,
            bias_tensor=bias,
            weight_tensor=w_q,
            scale_tensor=w_s,
        )
        return proj

    def _cpu_quant_mm(
        x: torch.Tensor,
        gamma: torch.Tensor,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        eps: float = 1e-6,
        gamma_offset: float = 0.0,
        w8a16_max_rows: int = 0,
        vmem_capacity_bytes: int = 0,
    ) -> torch.Tensor:
        del w8a16_max_rows, vmem_capacity_bytes
        raw_x = x.as_subclass(torch.Tensor) if type(x) is _FakeTpuTensor else x
        xf = raw_x.float()
        inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = (xf * inv_rms * (gamma.float() + gamma_offset)).to(raw_x.dtype)
        w_deq = w_q.float() * w_s.unsqueeze(0)
        out = (normed.float() @ w_deq).to(raw_x.dtype)
        return out.as_subclass(_FakeTpuTensor) if type(x) is _FakeTpuTensor else out

    real_mm_op = adapter._rmsnorm_fp8_mm_op
    op_spy = mock.Mock(side_effect=_cpu_quant_mm)
    monkeypatch.setattr(adapter, "_rmsnorm_fp8_mm_op", op_spy)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantized_matmul",
        lambda x, w, ws, w8a16_max_rows=0: (
            (
                (
                    x.as_subclass(torch.Tensor) if type(x) is _FakeTpuTensor else x
                ).float()
                @ (w.float() * ws.unsqueeze(0))
            )
            .to(x.dtype)
            .as_subclass(_FakeTpuTensor if type(x) is _FakeTpuTensor else torch.Tensor)
        ),
    )

    class _StubAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = _cpu_fp8_proj(
                QKVParallelLinear, n_qkv, "model.layers.0.self_attn.qkv_proj"
            )

        def forward(self, positions, hidden_states):
            del positions
            qkv, _ = self.qkv_proj(hidden_states)
            return qkv[..., :k]

    class _StubMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = _cpu_fp8_proj(
                MergedColumnParallelLinear, n_gu, "model.layers.0.mlp.gate_up_proj"
            )

        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            x = gate_up[..., :k]
            return x

    class _Gemma4StyleDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = GemmaRMSNorm(k, eps=1e-6)
            self.self_attn = _StubAttn()
            self.pre_feedforward_layernorm = GemmaRMSNorm(k, eps=1e-6)
            self.mlp = _StubMLP()

        def forward(self, positions, hidden_states):
            residual = hidden_states
            hidden_states = self.input_layernorm(residual)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states = hidden_states + residual
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            return hidden_states

    class _LlamaStyleDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = RMSNorm(k, eps=1e-6)
            self.self_attn = _StubAttn()
            self.post_attention_layernorm = RMSNorm(k, eps=1e-6)
            self.mlp = _StubMLP()

        def forward(self, positions, hidden_states, residual):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

    for cls in (_Gemma4StyleDecoder, _LlamaStyleDecoder):
        pairs = adapter._find_fusible_norm_submod_pairs(cls.forward)
        cls.__init__ = adapter._make_patched_decoder_init(  # type: ignore[method-assign]
            cls.__init__, cls.forward, pairs, vmem
        )

    with (
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))),
        _rmsnorm_priority(TPU_IR_PROVIDER),
    ):
        gemma_layer = _Gemma4StyleDecoder()
        llama_layer = _LlamaStyleDecoder()

    # Attention QKVParallelLinear stays unfused; only pre-MLP
    # MergedColumnParallelLinear is bound.
    assert not getattr(gemma_layer.input_layernorm, "_tpu_rmsnorm_fp8_bound", False)
    assert not getattr(gemma_layer.self_attn.qkv_proj, "_tpu_rmsnorm_fp8_bound", False)
    assert getattr(
        gemma_layer.pre_feedforward_layernorm, "_tpu_rmsnorm_fp8_bound", False
    )
    assert getattr(gemma_layer.mlp.gate_up_proj, "_tpu_rmsnorm_fp8_bound", False)

    assert not getattr(llama_layer.input_layernorm, "_tpu_rmsnorm_fp8_bound", False)
    assert not getattr(llama_layer.self_attn.qkv_proj, "_tpu_rmsnorm_fp8_bound", False)
    assert getattr(
        llama_layer.post_attention_layernorm, "_tpu_rmsnorm_fp8_bound", False
    )
    assert getattr(llama_layer.mlp.gate_up_proj, "_tpu_rmsnorm_fp8_bound", False)

    # Runtime CPU-tensor fallback check (verifies single normalization, no double norm,
    # and info_once diagnostic against an unpatched reference layer):
    unpatched_ref = _LlamaStyleDecoder.__new__(_LlamaStyleDecoder)
    torch.nn.Module.__init__(unpatched_ref)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        unpatched_ref.input_layernorm = RMSNorm(k, eps=1e-6)
        unpatched_ref.self_attn = _StubAttn()
        unpatched_ref.post_attention_layernorm = RMSNorm(k, eps=1e-6)
        unpatched_ref.mlp = _StubMLP()
    unpatched_ref.load_state_dict(llama_layer.state_dict())

    logged: list[str] = []
    monkeypatch.setattr(
        adapter.logger,
        "info_once",
        lambda msg, *args, **kwargs: logged.append(msg % args if args else msg),
    )
    h_cpu = torch.randn((8, k), generator=rng, dtype=torch.bfloat16)
    r_cpu = torch.randn((8, k), generator=rng, dtype=torch.bfloat16)
    out_fb, res_fb = llama_layer(torch.arange(8), h_cpu.clone(), r_cpu.clone())
    out_ref, res_ref = unpatched_ref(torch.arange(8), h_cpu.clone(), r_cpu.clone())
    assert op_spy.call_count == 0
    assert any("Falling back to unfused pre-MLP" in m for m in logged)
    torch.testing.assert_close(res_fb, res_ref, rtol=0, atol=0)
    torch.testing.assert_close(out_fb, out_ref, rtol=1e-2, atol=2e-2)

    # 1. Eager _FakeTpuTensor check: bound MLP dispatches to _rmsnorm_fp8_mm_op
    # while unbound QKVParallelLinear stays on quantized_matmul:
    for m_eager in (16, 256):
        pos_e = torch.arange(m_eager)
        h_e = torch.randn(
            (m_eager, k), generator=rng, dtype=torch.bfloat16
        ).as_subclass(_FakeTpuTensor)
        r_e = torch.randn(
            (m_eager, k), generator=rng, dtype=torch.bfloat16
        ).as_subclass(_FakeTpuTensor)

        before = op_spy.call_count
        g_e = gemma_layer(pos_e, h_e)
        assert op_spy.call_count == before + 1
        assert op_spy.call_args.args[5] == 1.0  # GemmaRMSNorm gamma_offset=1.0
        assert g_e.shape == (m_eager, k)

        before = op_spy.call_count
        l_e, lr_e = llama_layer(pos_e, h_e, r_e)
        assert op_spy.call_count == before + 1
        assert op_spy.call_args.args[5] == 0.0  # RMSNorm gamma_offset=0.0
        assert l_e.shape == (m_eager, k) and lr_e.shape == (m_eager, k)

    # 2. Fullgraph dynamic compilation of both bound decoder layers across M in
    # {16, 128, 256, 512, 1024}, on the fallback branch and then on the fused
    # branch, plus 3D [B, S, K] on the bound gate_up_proj. The real op is
    # traced, so can_fuse_rmsnorm_fp8_linear and the registered fake impl run
    # under Dynamo. CPU cannot execute the Pallas op, so the backend counts its
    # nodes and swaps in CPU math before running the graph.
    fused_node_counts: list[int] = []

    def _traceable_mm(
        x: torch.Tensor,
        gamma: torch.Tensor,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        eps: float = 1e-6,
        gamma_offset: float = 0.0,
        w8a16_max_rows: int = 0,
        vmem_capacity_bytes: int = 0,
    ) -> torch.Tensor:
        del w8a16_max_rows, vmem_capacity_bytes
        xf = x.float()
        inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = (xf * inv_rms * (gamma.float() + gamma_offset)).to(x.dtype)
        return (normed.float() @ (w_q.float() * w_s.unsqueeze(0))).to(x.dtype)

    def _counting_backend(gm: torch.fx.GraphModule, _example_inputs):
        fused_node_counts.append(_fused_op_node_count(gm))
        for node in gm.graph.nodes:
            if _is_fused_op_node(node):
                node.target = _traceable_mm
        gm.recompile()
        return gm.forward

    monkeypatch.setattr(adapter, "_rmsnorm_fp8_mm_op", real_mm_op)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantized_matmul",
        lambda x, w, ws, w8a16_max_rows=0: (
            x.float() @ (w.float() * ws.unsqueeze(0))
        ).to(x.dtype),
    )

    def _run_token_buckets() -> None:
        gemma_compiled = torch.compile(
            gemma_layer, backend=_counting_backend, fullgraph=True, dynamic=True
        )
        llama_compiled = torch.compile(
            llama_layer, backend=_counting_backend, fullgraph=True, dynamic=True
        )
        for m in (16, 128, 256, 512, 1024):
            pos = torch.arange(m)
            h_in = torch.randn((m, k), generator=rng, dtype=torch.bfloat16)
            r_in = torch.randn((m, k), generator=rng, dtype=torch.bfloat16)

            g_out = gemma_compiled(pos, h_in)
            assert g_out.shape == (m, k)

            for res_arg in (None, r_in):
                l_out, l_res = llama_compiled(pos, h_in.clone(), res_arg)
                assert l_out.shape == (m, k) and l_res.shape == (m, k)

    with torch.no_grad():
        # CPU tensors keep both layers on the unfused fallback: one graph for
        # Gemma and one per residual form for Llama, none with the fused op.
        _run_token_buckets()
        assert fused_node_counts == [0, 0, 0]
        fused_node_counts.clear()

        # Same layers on the fused branch. One dynamic FX graph per layer and
        # residual form still covers all five token buckets, and each calls the
        # fused op once. A SymInt guard on M in can_fuse_rmsnorm_fp8_linear
        # would add graphs here.
        class _TpuDevice:
            type = "tpu"

        class _TpuTensorMeta:
            device = _TpuDevice()

            def __init__(self, t: torch.Tensor) -> None:
                self.dtype = t.dtype
                self.ndim = t.ndim
                self.shape = t.shape

        orig_can_fuse = adapter.can_fuse_rmsnorm_fp8_linear
        monkeypatch.setattr(
            adapter,
            "can_fuse_rmsnorm_fp8_linear",
            lambda x, norm, linear, vmem: orig_can_fuse(
                _TpuTensorMeta(x),  # type: ignore[arg-type]
                norm,
                linear,
                vmem,
            ),
        )
        _run_token_buckets()
        assert fused_node_counts == [1, 1, 1]

        proj_compiled = torch.compile(
            lambda x: gemma_layer.mlp.gate_up_proj(x),
            backend=_counting_backend,
            fullgraph=True,
            dynamic=True,
        )
        h_3d = torch.randn((2, 64, k), generator=rng, dtype=torch.bfloat16)
        gu_3d, gu_bias = proj_compiled(h_3d)
        assert gu_3d.shape == (2, 64, n_gu) and gu_bias is None
        assert fused_node_counts == [1, 1, 1, 1]


@pytest.mark.tpu_test
def test_decoder_layer_rmsnorm_fp8_torch_compile_fullgraph_tpu():
    """Verify fullgraph=True on TPU leaves QKVParallelLinear unfused and fuses MLP."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")

    k, n_qkv, n_gu = 1024, 2048, 4096
    vmem = int(pltpu.get_tpu_info().vmem_capacity_bytes)

    def _make_tpu_proj(cls: type[torch.nn.Module], n_out: int, prefix: str):
        w_q = (torch.randn((k, n_out), dtype=torch.float32) * 0.02).to(
            torch.float8_e4m3fn
        )
        w_s = torch.full((n_out,), 0.02, dtype=torch.float32)
        bias = (torch.randn((n_out,), dtype=torch.bfloat16) * 0.01).to("tpu")
        return _make_fp8_linear(
            cls,
            k,
            n_out,
            prefix=prefix,
            bias_tensor=bias,
            weight_tensor=w_q.to("tpu"),
            scale_tensor=w_s.to("tpu"),
        )

    class _ToyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = _make_tpu_proj(
                QKVParallelLinear, n_qkv, "model.layers.0.self_attn.qkv_proj"
            )

        def forward(self, positions, hidden_states):
            del positions
            qkv, _ = self.qkv_proj(hidden_states)
            return qkv[..., :k]

    class _ToyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = _make_tpu_proj(
                MergedColumnParallelLinear, n_gu, "model.layers.0.mlp.gate_up_proj"
            )

        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            x = gate_up[..., :k]
            return x

    class _ToyDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = RMSNorm(k, eps=1e-6).to("tpu")
            self.self_attn = _ToyAttn()
            self.post_attention_layernorm = RMSNorm(k, eps=1e-6).to("tpu")
            self.mlp = _ToyMLP()

        def forward(self, positions, hidden_states, residual):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

    pairs = adapter._find_fusible_norm_submod_pairs(_ToyDecoderLayer.forward)
    assert pairs == [("post_attention_layernorm", "mlp", 0)]
    _ToyDecoderLayer.__init__ = adapter._make_patched_decoder_init(  # type: ignore[method-assign]
        _ToyDecoderLayer.__init__, _ToyDecoderLayer.forward, pairs, vmem
    )

    with (
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))),
        _rmsnorm_priority(TPU_IR_PROVIDER),
    ):
        layer = _ToyDecoderLayer()

    assert not getattr(layer.input_layernorm, "_tpu_rmsnorm_fp8_bound", False)
    assert not getattr(layer.self_attn.qkv_proj, "_tpu_rmsnorm_fp8_bound", False)
    assert getattr(layer.post_attention_layernorm, "_tpu_rmsnorm_fp8_bound", False)
    assert getattr(layer.mlp.gate_up_proj, "_tpu_rmsnorm_fp8_bound", False)

    def _manual_ref(h: torch.Tensor, res: torch.Tensor | None):
        def _rn(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            tf = t.float()
            inv_rms = torch.rsqrt(tf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            return (tf * inv_rms * w.float()).to(t.dtype)

        r1 = h if res is None else (h.float() + res.float()).to(h.dtype)
        n1 = _rn(r1, layer.input_layernorm.weight)
        qkv_proj = layer.self_attn.qkv_proj
        qkv = (
            quantized_matmul(n1, qkv_proj.weight, qkv_proj.weight_scale) + qkv_proj.bias
        )
        a_out = qkv[..., :k]
        r2 = (a_out.float() + r1.float()).to(h.dtype)
        n2 = _rn(r2, layer.post_attention_layernorm.weight)
        gu_proj = layer.mlp.gate_up_proj
        gu = quantized_matmul(n2, gu_proj.weight, gu_proj.weight_scale) + gu_proj.bias
        return gu[..., :k], r2

    traced_graphs: list[torch.fx.GraphModule] = []

    def _recording_backend(gm: torch.fx.GraphModule, _example_inputs):
        traced_graphs.append(gm)
        return gm.forward

    compiled = torch.compile(
        layer, backend=_recording_backend, fullgraph=True, dynamic=True
    )
    with torch.no_grad(), _rmsnorm_priority(TPU_IR_PROVIDER):
        for m in (16, 128, 256, 512, 1024):
            pos = torch.arange(m, device="tpu")
            h_in = torch.randn((m, k), dtype=torch.bfloat16).to("tpu")
            r_in = torch.randn((m, k), dtype=torch.bfloat16).to("tpu")

            for res_arg in (None, r_in):
                # Run _manual_ref first so linear_common initializes its cached
                # quantized_matmul op outside fullgraph=True Dynamo tracing.
                out_r, res_r = _manual_ref(h_in, res_arg)
                out_c, res_c = compiled(pos, h_in, res_arg)
                torch.testing.assert_close(res_c.cpu(), res_r.cpu(), rtol=0, atol=0)
                oc = out_c.cpu().float()
                orr = out_r.cpu().float()
                cos = float(
                    torch.sum(oc * orr)
                    / (torch.linalg.norm(oc) * torch.linalg.norm(orr))
                )
                assert cos > 0.9999, f"m={m} res={res_arg is not None} cos={cos:.6f}"

    # One graph per residual form (None, tensor) covers all five token counts,
    # and each calls pallas::fused_rmsnorm_fp8_matmul once for gate_up_proj.
    assert [_fused_op_node_count(gm) for gm in traced_graphs] == [1, 1]
