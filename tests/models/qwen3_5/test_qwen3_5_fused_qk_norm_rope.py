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
"""The Qwen3.5 fused q/k norm + rope: the fused projection against vLLM's eager
``Qwen3NextAttention._project_qkv_gate`` on real norm and rope modules, and the
op's row fork. The torch<->JAX op bridge runs the kernel in the Pallas
interpreter on the host."""

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

from vllm_torchtpu.models.qwen3_5.kernels import fused_qk_norm_rope as fused

pytestmark = pytest.mark.cpu_test

HEADS, KV_HEADS, HEAD_DIM, ROT = 4, 1, 256, 64

_TO_JAX = {torch.bfloat16: jnp.bfloat16, torch.float32: jnp.float32}
_TO_TORCH = {jnp.dtype(v): k for k, v in _TO_JAX.items()}


class _HostOp:
    """`pallas.jax_op` stand-in that runs the JAX function on CPU arrays."""

    calls: list = []
    kernel_calls: list = []
    reference_calls: list = []

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
    # Only this module's binding: a module first imported during a test (one
    # that builds a pallas op at import) must keep the real jax_op.
    monkeypatch.setattr(fused, "pallas", SimpleNamespace(jax_op=_HostOp))
    monkeypatch.setattr(fused, "_ops", {})
    orig_kernel = fused.qk_norm_rope_gate
    orig_ref = fused.qk_norm_rope_gate_reference

    def _tracked_kernel(*args, **kwargs):
        _HostOp.kernel_calls.append(args[0].shape[0])
        return orig_kernel(*args, **kwargs, interpret=True)

    def _tracked_ref(*args, **kwargs):
        _HostOp.reference_calls.append(args[0].shape[0])
        return orig_ref(*args, **kwargs)

    monkeypatch.setattr(fused, "qk_norm_rope_gate", _tracked_kernel)
    monkeypatch.setattr(fused, "qk_norm_rope_gate_reference", _tracked_ref)
    _HostOp.calls = []
    _HostOp.kernel_calls = []
    _HostOp.reference_calls = []
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
    monkeypatch.setattr(Qwen3NextAttention, "__init__", Qwen3NextAttention.__init__)
    monkeypatch.setattr(
        Qwen3NextAttention,
        "_project_qkv_gate",
        Qwen3NextAttention._project_qkv_gate,
    )
    fused.install_qk_norm_rope_patch()


@pytest.mark.parametrize(
    "rows, min_rows, max_rows, enabled, uses_op, uses_kernel",
    [
        (64, 0, 64, True, True, True),
        (128, 0, 64, True, True, False),
        (4, 16, 64, True, True, False),
        (8, 0, 64, False, False, False),
    ],
)
def test_patch_forks_on_the_token_bucket(
    patched, monkeypatch, rows, min_rows, max_rows, enabled, uses_op, uses_kernel
):
    monkeypatch.setattr(fused, "MIN_ROWS", min_rows)
    monkeypatch.setattr(fused, "MAX_ROWS", max_rows)
    layer = _layer(fused_qk_norm_rope=enabled)
    q, k, v, gate = Qwen3NextAttention._project_qkv_gate(layer, *_inputs(rows))
    assert q.shape == (rows, HEADS * HEAD_DIM) and gate.shape == q.shape
    assert len(_HostOp.calls) == int(uses_op)
    assert len(_HostOp.kernel_calls) == int(uses_kernel)
    assert len(_HostOp.reference_calls) == int(uses_op and not uses_kernel)


@pytest.mark.parametrize("rows, uses_kernel", [(16384, True), (16385, False)])
def test_buckets_up_to_16384_rows_take_the_kernel(rows, uses_kernel):
    """A 16384-row bucket (MAX_ROWS) runs the kernel and a 16385-row bucket
    the JAX reference. The op is traced for its shapes only."""
    layer = _layer()
    fn = fused._ops[fused._op_key(layer)].fn
    jax.eval_shape(
        fn,
        jax.ShapeDtypeStruct((rows, 2 * HEADS * HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, KV_HEADS * HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((HEAD_DIM,), jnp.float32),
        jax.ShapeDtypeStruct((HEAD_DIM,), jnp.float32),
        jax.ShapeDtypeStruct((rows, ROT), jnp.float32),
    )
    assert _HostOp.kernel_calls == ([rows] if uses_kernel else [])
    assert _HostOp.reference_calls == ([] if uses_kernel else [rows])


@pytest.fixture
def restore_attention(monkeypatch):
    """Undo whatever a test installs on Qwen3NextAttention."""
    monkeypatch.setattr(Qwen3NextAttention, "__init__", Qwen3NextAttention.__init__)
    monkeypatch.setattr(
        Qwen3NextAttention,
        "_project_qkv_gate",
        Qwen3NextAttention._project_qkv_gate,
    )


@pytest.mark.parametrize(
    "architectures, installs",
    [
        (None, False),
        # A stand-in without architectures, as the patch registry's tests pass.
        ("no-architectures", False),
        (["Qwen3NextForCausalLM"], False),
        (["Qwen3_5MoeForConditionalGeneration"], True),
    ],
)
def test_the_rule_installs_the_patch_for_qwen3_5_only(
    restore_attention, architectures, installs
):
    original_init = Qwen3NextAttention.__init__
    original_project = Qwen3NextAttention._project_qkv_gate
    if architectures == "no-architectures":
        model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="llama"))
    else:
        model_config = (
            None
            if architectures is None
            else SimpleNamespace(architectures=architectures)
        )
    result = fused.maybe_install_qk_norm_rope_patch(model_config)
    # False keeps the patch eligible for the next lifecycle stage.
    assert result is (None if installs else False)
    assert (Qwen3NextAttention._project_qkv_gate is not original_project) is installs
    assert (Qwen3NextAttention.__init__ is not original_init) is installs


def test_install_wraps_the_attention_once(restore_attention):
    fused.install_qk_norm_rope_patch()
    project = Qwen3NextAttention._project_qkv_gate
    init = Qwen3NextAttention.__init__
    fused.install_qk_norm_rope_patch()
    assert Qwen3NextAttention._project_qkv_gate is project
    assert Qwen3NextAttention.__init__ is init


def _vllm_config(max_num_seqs, num_speculative_tokens):
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.scheduler_config.max_num_seqs = max_num_seqs
    if num_speculative_tokens:
        config.speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens
        )
    return config


@pytest.mark.parametrize(
    "max_num_seqs, k, armed",
    [(16, 3, "fused"), (fused.MAX_ROWS // 2, 3, "eager"), (64, 0, "fused")],
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


def test_prebuild_arms_both_backbone_and_eagle_head_graphs():
    """Because the row-count fork lives inside the lowered JAX op rather than
    in Python during Dynamo tracing, both 'backbone' and 'eagle_head' arm."""
    from vllm.compilation import backends

    layer = _layer(fused_qk_norm_rope=False)
    assert backends.model_tag == "backbone"
    with set_current_vllm_config(_vllm_config(16, 3)):
        assert fused._prebuild(layer, model_config=None)
        with backends.set_model_tag("eagle_head"):
            assert fused._prebuild(layer, model_config=None)
    assert len(fused._ops) == 1
