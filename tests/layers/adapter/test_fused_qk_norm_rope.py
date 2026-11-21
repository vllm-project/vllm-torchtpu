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
"""TPU_FUSED_QK_NORM_ROPE: the patched Qwen3NextAttention._project_qkv_gate
gives the same q, k, v and gate as vLLM's eager path.

The eager side is vLLM's own method with a real GemmaRMSNorm and a real
get_rope rotary embedding. The fused side runs the Pallas kernel in interpret
mode through a numpy bridge in place of the torch_tpu op, so this runs on CPU.
"""

import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models import qwen3_next

import vllm_torchtpu
from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qk_norm_rope_gate import qk_norm_rope_gate_rows
from vllm_torchtpu.layers.adapter import fused_qk_norm_rope_tpu as fused

# Qwen3.5-397B at TP8: 32 q heads and 2 kv heads over 8 cores (kv replicated),
# head_dim 256, partial rotary factor 0.25.
_NUM_HEADS, _NUM_KV_HEADS, _HEAD_DIM = 4, 1, 256
_EPS = 1e-6
_TEXT_ROPE = {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25}
_MROPE = {
    **_TEXT_ROPE,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}


def _interpret_mode_op(calls):
    """Stands in for the torch_tpu op: the same kernel, in interpret mode."""

    def get_op(num_heads, num_kv_heads, head_dim, rotary_dim, eps):
        def op(q_gate, k, q_weight, k_weight, cos_sin):
            calls.append((q_weight, k_weight))
            outputs = qk_norm_rope_gate_rows(
                jnp.asarray(q_gate.detach().float().numpy(), jnp.bfloat16),
                jnp.asarray(k.detach().float().numpy(), jnp.bfloat16),
                jnp.asarray(q_weight.detach().float().numpy()),
                jnp.asarray(k_weight.detach().float().numpy()),
                jnp.asarray(cos_sin.detach().float().numpy()),
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                eps=eps,
                interpret=True,
            )
            return tuple(
                torch.from_numpy(np.asarray(out, np.float32)).to(q_gate.dtype)
                for out in outputs
            )

        return op

    return get_op


@pytest.fixture
def patched(monkeypatch):
    """Applies the patch with the env on, undone after the test."""
    cls = qwen3_next.Qwen3NextAttention
    monkeypatch.setattr(envs, "TPU_FUSED_QK_NORM_ROPE", True)
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "_project_qkv_gate", cls._project_qkv_gate)
    monkeypatch.setattr(cls, "_tpu_fused_qk_norm_rope_patch", False, raising=False)
    original_project_qkv_gate = cls._project_qkv_gate
    calls = []
    monkeypatch.setattr(fused, "_get_op", _interpret_mode_op(calls))
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()
    return SimpleNamespace(
        calls=calls, original_project_qkv_gate=original_project_qkv_gate
    )


def _attention(rope_parameters, seed=0):
    """The attributes _project_qkv_gate reads, built from vLLM's own layers."""
    torch.manual_seed(seed)
    with set_current_vllm_config(VllmConfig()):
        attention = SimpleNamespace(
            num_heads=_NUM_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            q_size=_NUM_HEADS * _HEAD_DIM,
            kv_size=_NUM_KV_HEADS * _HEAD_DIM,
            attn_output_gate=True,
            # vLLM's own fused kernel is CUDA/XPU only; off on TPU.
            use_fused_qk_norm_rope_gate=False,
            q_norm=GemmaRMSNorm(_HEAD_DIM, eps=_EPS),
            k_norm=GemmaRMSNorm(_HEAD_DIM, eps=_EPS),
            rotary_emb=get_rope(
                head_size=_HEAD_DIM,
                max_position=4096,
                rope_parameters=rope_parameters,
                dtype=torch.bfloat16,
            ),
            _tpu_fused_qk_norm_rope=True,
        )
    with torch.no_grad():
        attention.q_norm.weight.normal_(0.0, 0.1)
        attention.k_norm.weight.normal_(0.0, 0.1)
    return attention


def _qkv(rows, seed=1):
    generator = torch.Generator().manual_seed(seed)
    width = (2 * _NUM_HEADS + 2 * _NUM_KV_HEADS) * _HEAD_DIM
    return torch.randn(rows, width, generator=generator).to(torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 8, 13, 64])
@pytest.mark.parametrize("rope", ["text", "mrope"])
def test_fused_matches_eager(patched, rows, rope):
    attention = _attention(_TEXT_ROPE if rope == "text" else _MROPE)
    qkv = _qkv(rows)
    positions = torch.randint(0, 4096, (rows,))
    if rope == "mrope":
        # Text-only M-RoPE positions: three equal rows.
        positions = positions.expand(3, rows).contiguous()
    with set_current_vllm_config(VllmConfig()):
        want = patched.original_project_qkv_gate(attention, qkv, positions)
        got = qwen3_next.Qwen3NextAttention._project_qkv_gate(attention, qkv, positions)
    assert len(patched.calls) == 1
    for name, actual, expected in zip(("q", "k", "v", "gate"), got, want):
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected.dtype, name
        # The fused kernel ropes in f32 before one cast; eager casts the norm
        # output to bf16 first. Allow a couple of bf16 roundings.
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=3e-2, rtol=2e-2, msg=name
        )


def test_rows_above_the_cap_keep_the_eager_path(patched):
    attention = _attention(_TEXT_ROPE)
    rows = vllm_torchtpu._FUSED_QK_NORM_ROPE_MAX_ROWS + 1
    qkv = _qkv(rows)
    positions = torch.arange(rows)
    with set_current_vllm_config(VllmConfig()):
        got = qwen3_next.Qwen3NextAttention._project_qkv_gate(attention, qkv, positions)
        want = patched.original_project_qkv_gate(attention, qkv, positions)
    assert patched.calls == []
    for actual, expected in zip(got, want):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_reads_the_hoisted_norm_weight_when_present(patched):
    """With TPU_NORM_WEIGHT_HOIST's f32 (1 + w) buffer on the norm, the kernel
    gets that buffer; without it, (1 + w) computed from the weight."""
    attention = _attention(_TEXT_ROPE)
    qkv, positions = _qkv(4), torch.arange(4)
    project = qwen3_next.Qwen3NextAttention._project_qkv_gate
    with set_current_vllm_config(VllmConfig()):
        project(attention, qkv, positions)
        hoisted = attention.q_norm.weight.detach().float() + 1.0
        attention.q_norm.register_buffer(
            "_tpu_weight_plus_one", hoisted, persistent=False
        )
        project(attention, qkv, positions)
    (computed_q, computed_k), (hoisted_q, hoisted_k) = patched.calls
    torch.testing.assert_close(computed_q, attention.q_norm.weight.float() + 1.0)
    torch.testing.assert_close(computed_k, attention.k_norm.weight.float() + 1.0)
    assert hoisted_q is hoisted
    torch.testing.assert_close(hoisted_k, computed_k)


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("language_model_only", [None, False, True])
def test_enabled_only_for_text_only_models(
    monkeypatch, positional, language_model_only
):
    class StubAttention:
        def __init__(self, config, model_config=None):
            self.attn_output_gate = True
            self.rotary_emb = SimpleNamespace(
                is_neox_style=True, rotary_dim=64, cos_sin_cache=torch.empty(1)
            )
            self.num_heads = self.num_kv_heads = 1
            self.head_dim = 256
            self.q_norm = SimpleNamespace(variance_epsilon=_EPS)

        def _project_qkv_gate(self, qkv, positions):
            return "eager"

    monkeypatch.setattr(envs, "TPU_FUSED_QK_NORM_ROPE", True)
    monkeypatch.setattr(qwen3_next, "Qwen3NextAttention", StubAttention)
    prebuilt = []
    monkeypatch.setattr(fused, "prebuild", lambda *args: prebuilt.append(args))
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()
    mm_config = (
        None
        if language_model_only is None
        else SimpleNamespace(language_model_only=language_model_only)
    )
    model_config = SimpleNamespace(multimodal_config=mm_config)
    attention = (
        StubAttention(None, model_config)
        if positional
        else StubAttention(None, model_config=model_config)
    )
    expected = language_model_only is not False
    assert attention._tpu_fused_qk_norm_rope == expected
    assert prebuilt == ([(1, 1, 256, 64, _EPS)] if expected else [])
    if not expected:
        assert attention._project_qkv_gate(torch.empty(1, 1024), None) == "eager"


def test_env_off_leaves_vllm_untouched(monkeypatch):
    cls = qwen3_next.Qwen3NextAttention
    monkeypatch.setattr(envs, "TPU_FUSED_QK_NORM_ROPE", False)
    project_qkv_gate = cls._project_qkv_gate
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()
    assert cls._project_qkv_gate is project_qkv_gate
    assert not getattr(cls, "_tpu_fused_qk_norm_rope_patch", False)


def test_fake_matches_the_call(monkeypatch):
    """Dynamo traces the forward against the op's fake; its shapes and dtypes
    must match the real call's."""

    class FakeJaxOp:
        def __init__(self, name, fn):
            self.fake = None

        def register_fake(self, fake):
            self.fake = fake

        def __call__(self, *args):
            return self.fake(*args)

    monkeypatch.setitem(
        sys.modules,
        "torch_tpu._internal",
        SimpleNamespace(pallas=SimpleNamespace(jax_op=FakeJaxOp)),
    )
    monkeypatch.setitem(sys.modules, "torch_tpu", SimpleNamespace())
    monkeypatch.setattr(fused, "_ops_by_geometry", {})
    num_heads, num_kv_heads, head_dim, rotary_dim = 8, 1, 256, 64
    fused.prebuild(num_heads, num_kv_heads, head_dim, rotary_dim, _EPS)
    with FakeTensorMode():
        q, k, gate = fused.fused_qk_rmsnorm_rope_gate(
            torch.empty(4, num_heads * head_dim * 2, dtype=torch.bfloat16),
            torch.empty(4, num_kv_heads * head_dim, dtype=torch.bfloat16),
            torch.empty(head_dim, dtype=torch.float32),
            torch.empty(head_dim, dtype=torch.float32),
            torch.empty(4096, rotary_dim, dtype=torch.bfloat16),
            torch.zeros(4, dtype=torch.int64),
            _EPS,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
        )
    assert tuple(q.shape) == (4, num_heads * head_dim)
    assert tuple(k.shape) == (4, num_kv_heads * head_dim)
    assert tuple(gate.shape) == (4, num_heads * head_dim)
    assert q.dtype == k.dtype == gate.dtype == torch.bfloat16
