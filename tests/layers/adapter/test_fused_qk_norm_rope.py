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
"""TPU_FUSED_QK_NORM_ROPE: the fused projection against vLLM's eager
``Qwen3NextAttention._project_qkv_gate`` on real norm and rope modules, and the
patch's row fork. The torch<->JAX op bridge runs the kernel in the Pallas
interpreter on the host."""

import functools
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import vllm.model_executor.custom_op as custom_op_mod
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from vllm_torchtpu.layers.adapter import fused_qk_norm_rope as fused

pytestmark = pytest.mark.cpu_test

HEADS, KV_HEADS, HEAD_DIM, ROT = 4, 1, 256, 64

_TO_JAX = {torch.bfloat16: jnp.bfloat16, torch.float32: jnp.float32}
_TO_TORCH = {jnp.dtype(v): k for k, v in _TO_JAX.items()}


class _HostOp:
    """`pallas.jax_op` stand-in that runs the JAX function on CPU arrays."""

    calls: list = []

    def __init__(self, name, fn):
        self.fn = fn

    def register_fake(self, fake):
        pass

    def __call__(self, *args):
        _HostOp.calls.append(args)
        arrays = [
            jnp.asarray(a.detach().float().numpy(), dtype=_TO_JAX[a.dtype])
            for a in args
        ]
        return tuple(
            torch.from_numpy(np.asarray(a, np.float32)).to(_TO_TORCH[a.dtype])
            for a in self.fn(*arrays)
        )


@pytest.fixture(autouse=True)
def host_ops(monkeypatch):
    monkeypatch.setattr(fused.pallas, "jax_op", _HostOp)
    monkeypatch.setattr(fused, "_ops", {})
    monkeypatch.setattr(
        fused,
        "qk_norm_rope_gate",
        functools.partial(fused.qk_norm_rope_gate, interpret=True),
    )
    _HostOp.calls = []
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _custom_ops_native():
    compilation_config = CompilationConfig(
        mode=CompilationMode.NONE, custom_ops=["none"]
    )
    return patch.object(
        custom_op_mod, "get_cached_compilation_config", lambda: compilation_config
    )


def _layer(fused_qk_norm_rope=True, is_neox_style=True):
    """The attributes `_project_qkv_gate` reads, on real norm and rope modules."""
    generator = torch.Generator().manual_seed(0)
    with _custom_ops_native():
        q_norm = GemmaRMSNorm(HEAD_DIM, eps=1e-6)
        k_norm = GemmaRMSNorm(HEAD_DIM, eps=1e-6)
        rotary_emb = RotaryEmbedding(
            head_size=HEAD_DIM,
            rotary_dim=ROT,
            max_position_embeddings=4096,
            base=10000.0,
            is_neox_style=is_neox_style,
            dtype=torch.bfloat16,
        )
    for norm in (q_norm, k_norm):
        norm.weight.data.copy_(0.1 * torch.randn(HEAD_DIM, generator=generator))
    layer = SimpleNamespace(
        attn_output_gate=True,
        use_fused_qk_norm_rope_gate=False,
        fused_qk_norm_rope=fused_qk_norm_rope,
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        q_size=HEADS * HEAD_DIM,
        kv_size=KV_HEADS * HEAD_DIM,
        q_norm=q_norm,
        k_norm=k_norm,
        rotary_emb=rotary_emb,
    )
    if fused_qk_norm_rope:
        fused._build_op(fused._op_key(layer))
    return layer


def _inputs(rows, seed=1):
    generator = torch.Generator().manual_seed(seed)
    width = (2 * HEADS + 2 * KV_HEADS) * HEAD_DIM
    qkv = torch.randn(rows, width, generator=generator).to(torch.bfloat16)
    positions = torch.randint(0, 4096, (rows,), generator=generator)
    return qkv, positions


@pytest.mark.parametrize("rows", [1, 13, 64])
def test_fused_projection_matches_the_eager_path(rows):
    layer = _layer()
    qkv, positions = _inputs(rows)
    want = Qwen3NextAttention._project_qkv_gate(layer, qkv, positions)
    got = fused.project_qkv_gate(layer, qkv, positions)
    assert got[2] is not None
    for name, a, b in zip(("q", "k", "v", "gate"), got, want):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        # The eager path rounds the normed q/k to bf16 before the rope.
        torch.testing.assert_close(a, b, atol=3e-2, rtol=3e-2, msg=name)


def test_fused_projection_reads_the_hoisted_norm_weight():
    layer = _layer()
    hoisted = layer.q_norm.weight.float() + 1.0
    layer.q_norm._tpu_weight_plus_one = hoisted
    fused.project_qkv_gate(layer, *_inputs(8))
    (args,) = _HostOp.calls
    assert args[2] is hoisted
    torch.testing.assert_close(args[3], layer.k_norm.weight.float() + 1.0)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setenv("TPU_FUSED_QK_NORM_ROPE", "1")
    monkeypatch.setattr(Qwen3NextAttention, "__init__", Qwen3NextAttention.__init__)
    monkeypatch.setattr(
        Qwen3NextAttention,
        "_project_qkv_gate",
        Qwen3NextAttention._project_qkv_gate,
    )
    fused.install_qk_norm_rope_patch()


@pytest.mark.parametrize(
    "rows, enabled, uses_kernel",
    [
        (fused.MAX_ROWS, True, True),
        (2 * fused.MAX_ROWS, True, False),
        (8, False, False),
    ],
)
def test_patch_forks_on_the_token_bucket(patched, rows, enabled, uses_kernel):
    layer = _layer(fused_qk_norm_rope=enabled)
    q, k, v, gate = Qwen3NextAttention._project_qkv_gate(layer, *_inputs(rows))
    assert q.shape == (rows, HEADS * HEAD_DIM) and gate.shape == q.shape
    assert len(_HostOp.calls) == int(uses_kernel)


def test_patch_is_inert_with_the_flag_off(monkeypatch):
    monkeypatch.delenv("TPU_FUSED_QK_NORM_ROPE", raising=False)
    original = Qwen3NextAttention._project_qkv_gate
    fused.install_qk_norm_rope_patch()
    assert Qwen3NextAttention._project_qkv_gate is original


def _vllm_config(max_num_seqs, num_speculative_tokens):
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.scheduler_config.max_num_seqs = max_num_seqs
    if num_speculative_tokens:
        config.speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens
        )
    return config


@pytest.mark.parametrize(
    "max_num_seqs, k, armed", [(16, 3, "fused"), (32, 3, "eager"), (64, 0, "fused")]
)
def test_prebuild_logs_which_path_decode_takes(monkeypatch, max_num_seqs, k, armed):
    messages = []
    monkeypatch.setattr(
        fused.logger, "info_once", lambda msg, *args: messages.append(msg % args)
    )
    layer = _layer(fused_qk_norm_rope=False)
    with set_current_vllm_config(_vllm_config(max_num_seqs, k)):
        assert fused._prebuild(layer, model_config=None)
    assert len(fused._ops) == 1
    (message,) = messages
    assert f"runs the {armed} path" in message
    assert f"<= {fused.MAX_ROWS // (1 + k)} requests" in message


@pytest.mark.parametrize(
    "is_neox_style, multimodal, reason",
    [
        (False, None, "not NeoX"),
        (True, SimpleNamespace(language_model_only=False), "--language-model-only"),
    ],
)
def test_prebuild_declines_what_the_kernel_cannot_compute(
    monkeypatch, is_neox_style, multimodal, reason
):
    warnings = []
    monkeypatch.setattr(
        fused.logger, "warning_once", lambda msg, *args: warnings.append(msg % args)
    )
    model_config = SimpleNamespace(multimodal_config=multimodal)
    layer = _layer(fused_qk_norm_rope=False, is_neox_style=is_neox_style)
    assert not fused._prebuild(layer, model_config)
    assert not fused._ops
    assert reason in warnings[0]


def test_prebuild_declines_a_layer_outside_the_backbone_graph(monkeypatch):
    """The MTP draft is built under the "eagle_head" tag and compiled from one
    trace, which cannot carry both sides of the row fork."""
    from vllm.compilation import backends

    warnings = []
    monkeypatch.setattr(
        fused.logger, "warning_once", lambda msg, *args: warnings.append(msg % args)
    )
    layer = _layer(fused_qk_norm_rope=False)
    with backends.set_model_tag("eagle_head"):
        assert not fused._prebuild(layer, model_config=None)
    assert not fused._ops
    assert "'eagle_head' graph" in warnings[0]
    with set_current_vllm_config(_vllm_config(16, 3)):
        assert fused._prebuild(layer, model_config=None)
