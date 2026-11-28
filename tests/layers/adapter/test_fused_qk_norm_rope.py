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
matches vLLM's eager path across unaligned, odd-8, and multi-tile batches."""

import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import vllm.distributed as distributed_mod
from jax.experimental.pallas import tpu as pltpu
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter as parameter_mod
from vllm.model_executor.layers import linear as linear_mod
from vllm.model_executor.layers import rotary_embedding as rope_mod
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models import qwen3_next
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

import vllm_torchtpu
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.kernels.qk_norm_rope_gate import qk_norm_rope_gate_rows
from vllm_torchtpu.layers.adapter import fused_qk_norm_rope_tpu as fused
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.tpu_info import get_chip_version

# Qwen3.5-397B at TP8: 32 q heads and 2 kv heads over 8 cores (kv replicated),
# head_dim 256, partial rotary factor 0.25.
_NUM_HEADS, _NUM_KV_HEADS, _HEAD_DIM = 4, 1, 256
_EPS = 1e-6
_VMEM_64M = 64 * 1024 * 1024
_TEXT_ROPE = {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25}
_MROPE = {
    **_TEXT_ROPE,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}


def _cpu_vllm_config() -> VllmConfig:
    return VllmConfig(device_config=DeviceConfig(device="cpu"))


def _interpret_mode_op(calls):
    """Stand in for the torch_tpu op: run the real Pallas kernel in interpret mode."""

    def get_op(
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
        vmem_capacity_bytes=None,
        num_sublanes=None,
        chip_version=None,
        min_pallas_qkv_bytes=fused.MIN_PALLAS_QKV_BYTES,
    ):
        def op(qkv, q_weight, k_weight, cos_sin):
            calls.append((tuple(qkv.shape), q_weight, k_weight))
            jax_dtype = jnp.bfloat16 if qkv.dtype == torch.bfloat16 else jnp.float32
            outputs = qk_norm_rope_gate_rows(
                jnp.asarray(qkv.detach().float().numpy(), jax_dtype),
                jnp.asarray(q_weight.detach().float().numpy()),
                jnp.asarray(k_weight.detach().float().numpy()),
                jnp.asarray(cos_sin.detach().float().numpy()),
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                eps=eps,
                vmem_capacity_bytes=vmem_capacity_bytes,
                num_sublanes=num_sublanes,
                chip_version=chip_version,
                min_pallas_qkv_bytes=min_pallas_qkv_bytes,
                interpret=True,
            )
            return tuple(
                torch.from_numpy(np.asarray(out, np.float32)).to(qkv.dtype)
                for out in outputs
            )

        return op

    return get_op


@pytest.fixture
def patched(monkeypatch):
    """Apply the patch with the env on, undone cleanly after the test."""
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    get_chip_version.cache_clear()
    cls = qwen3_next.Qwen3NextAttention
    monkeypatch.setattr(envs, "TPU_FUSED_QK_NORM_ROPE", True)
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "_project_qkv_gate", cls._project_qkv_gate)
    monkeypatch.setattr(cls, "_tpu_fused_qk_norm_rope_patch", False, raising=False)
    original_project_qkv_gate = cls._project_qkv_gate
    calls = []
    monkeypatch.setattr(fused, "_get_op", _interpret_mode_op(calls))
    monkeypatch.setattr(fused, "_is_tpu_tensor", lambda _tensor: True)
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()
    try:
        yield SimpleNamespace(
            calls=calls, original_project_qkv_gate=original_project_qkv_gate
        )
    finally:
        get_chip_version.cache_clear()
        torch._dynamo.reset()


def _attention(rope_parameters, seed=0, vllm_config: VllmConfig | None = None):
    """The attributes _project_qkv_gate reads, built from vLLM's own layers."""
    torch.manual_seed(seed)
    cfg = _cpu_vllm_config() if vllm_config is None else vllm_config
    with set_current_vllm_config(cfg):
        attention = SimpleNamespace(
            num_heads=_NUM_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            q_size=_NUM_HEADS * _HEAD_DIM,
            kv_size=_NUM_KV_HEADS * _HEAD_DIM,
            attn_output_gate=True,
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
            _tpu_compilation_config=cfg.compilation_config,
            _tpu_chip_vmem_bytes=_VMEM_64M,
            _tpu_chip_num_sublanes=8,
            _tpu_chip_version=pltpu.ChipVersion.TPU_7X,
        )
    with torch.no_grad():
        attention.q_norm.weight.normal_(0.0, 0.1)
        attention.k_norm.weight.normal_(0.0, 0.1)
    return attention


def _qkv(rows, seed=1):
    generator = torch.Generator().manual_seed(seed)
    width = (2 * _NUM_HEADS + 2 * _NUM_KV_HEADS) * _HEAD_DIM
    return torch.randn(rows, width, generator=generator).to(torch.bfloat16)


def test_env_propagated_to_workers():
    assert "TPU_FUSED_QK_NORM_ROPE" in TpuPlatform.additional_env_vars


@pytest.mark.parametrize("rows", [1, 8, 13, 24, 64, 128])
@pytest.mark.parametrize("rope", ["text", "mrope"])
def test_fused_matches_eager(patched, rows, rope):
    attention = _attention(_TEXT_ROPE if rope == "text" else _MROPE)
    qkv = _qkv(rows)
    positions = torch.randint(0, 4096, (rows,), dtype=torch.int32)
    if rope == "mrope":
        positions = positions.expand(3, rows).contiguous()
    with set_current_vllm_config(_cpu_vllm_config()):
        want = patched.original_project_qkv_gate(attention, qkv, positions)
        got = qwen3_next.Qwen3NextAttention._project_qkv_gate(attention, qkv, positions)
    assert len(patched.calls) == 1
    for name, actual, expected in zip(("q", "k", "v", "gate"), got, want):
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected.dtype, name
        if name == "v":
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        else:
            torch.testing.assert_close(
                actual.float(), expected.float(), atol=3e-2, rtol=2e-2, msg=name
            )


def test_fallbacks_for_vmem_unaligned_noncontiguous_and_rope_patch(patched):
    attention = _attention(_TEXT_ROPE)
    # 208-row batch (>= 1 MiB at TP=8) with tiny 32 KiB VMEM falls back cleanly.
    attention._tpu_chip_vmem_bytes = 32 * 1024
    qkv = _qkv(208)
    positions = torch.arange(208, dtype=torch.int32)
    with set_current_vllm_config(_cpu_vllm_config()):
        got = qwen3_next.Qwen3NextAttention._project_qkv_gate(attention, qkv, positions)
        want = patched.original_project_qkv_gate(attention, qkv, positions)
    assert patched.calls == []
    for actual, expected in zip(got, want):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    # Large unaligned prompt (2049 rows at TP=8) also falls back cleanly to eager.
    attention._tpu_chip_vmem_bytes = _VMEM_64M
    qkv_2049 = _qkv(2049)
    pos_2049 = torch.arange(2049, dtype=torch.int32) % 4096
    with set_current_vllm_config(_cpu_vllm_config()):
        got_2049 = qwen3_next.Qwen3NextAttention._project_qkv_gate(
            attention, qkv_2049, pos_2049
        )
        want_2049 = patched.original_project_qkv_gate(attention, qkv_2049, pos_2049)
    assert patched.calls == []
    for actual, expected in zip(got_2049, want_2049):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    # Non-contiguous qkv view falls back cleanly to eager.
    padded = torch.randn(16, qkv.shape[1] + 128, dtype=torch.bfloat16)
    non_contig = padded[:, : qkv.shape[1]]
    assert not non_contig.is_contiguous()
    with set_current_vllm_config(_cpu_vllm_config()):
        got_nc = qwen3_next.Qwen3NextAttention._project_qkv_gate(
            attention, non_contig, torch.arange(16, dtype=torch.int32)
        )
        want_nc = patched.original_project_qkv_gate(
            attention, non_contig, torch.arange(16, dtype=torch.int32)
        )
    assert patched.calls == []
    for actual, expected in zip(got_nc, want_nc):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    # Post-init rope_patch deleting or replacing cos_sin_cache falls back via
    # Qwen3NextAttention._project_qkv_gate without raising AttributeError.
    fallback_calls = []
    cls = qwen3_next.Qwen3NextAttention
    saved_cache = attention.rotary_emb.cos_sin_cache
    saved_env = envs.TPU_FUSED_QK_NORM_ROPE
    saved_flag = getattr(cls, "_tpu_fused_qk_norm_rope_patch", False)
    saved_proj = cls._project_qkv_gate
    try:
        cls._project_qkv_gate = lambda self, x, p: (
            fallback_calls.append(x.shape[0]) or ("q", "k", "v", "g")
        )
        cls._tpu_fused_qk_norm_rope_patch = False
        vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()
        delattr(attention.rotary_emb, "cos_sin_cache")
        got_del = cls._project_qkv_gate(
            attention, _qkv(16), torch.arange(16, dtype=torch.int32)
        )
        assert got_del == ("q", "k", "v", "g")
        attention.rotary_emb.cos_sin_cache = SimpleNamespace(tag="in_graph_rope")
        got_obj = cls._project_qkv_gate(
            attention, _qkv(16), torch.arange(16, dtype=torch.int32)
        )
        assert got_obj == ("q", "k", "v", "g")
        assert fallback_calls == [16, 16]
        assert patched.calls == []
    finally:
        attention.rotary_emb.cos_sin_cache = saved_cache
        envs.TPU_FUSED_QK_NORM_ROPE = saved_env
        cls._tpu_fused_qk_norm_rope_patch = saved_flag
        cls._project_qkv_gate = saved_proj


_DYN_SEEN_SHAPES: list[int] = []


@torch.library.custom_op("test_qknorm_dyn::fused", mutates_args=())
def _stub_qknorm_dyn(
    qkv: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del q_w, k_w, cos_sin
    rows = qkv.shape[0]
    _DYN_SEEN_SHAPES.append(int(rows))
    return (
        torch.zeros((rows, _NUM_HEADS * _HEAD_DIM), dtype=qkv.dtype),
        torch.zeros((rows, _NUM_KV_HEADS * _HEAD_DIM), dtype=qkv.dtype),
        torch.zeros((rows, _NUM_KV_HEADS * _HEAD_DIM), dtype=qkv.dtype),
        torch.zeros((rows, _NUM_HEADS * _HEAD_DIM), dtype=qkv.dtype),
    )


@_stub_qknorm_dyn.register_fake
def _(
    qkv: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del q_w, k_w, cos_sin
    rows = qkv.shape[0]
    return (
        qkv.new_empty((rows, _NUM_HEADS * _HEAD_DIM)),
        qkv.new_empty((rows, _NUM_KV_HEADS * _HEAD_DIM)),
        qkv.new_empty((rows, _NUM_KV_HEADS * _HEAD_DIM)),
        qkv.new_empty((rows, _NUM_HEADS * _HEAD_DIM)),
    )


def test_compile_dynamic_buckets_without_shape_guards(patched, monkeypatch):
    """When compile_sizes are all supported, dynamic SymInt tracing emits one
    reusable graph across < 8 and >= 8 buckets without ShapeEnv row guards,
    even when get_current_vllm_config() is unset during forward()."""
    attention = _attention(_TEXT_ROPE)
    _DYN_SEEN_SHAPES.clear()

    monkeypatch.setattr(fused, "_get_op", lambda *_a, **_kw: _stub_qknorm_dyn)
    cfg = _cpu_vllm_config()
    cfg.compilation_config.compile_sizes = [1, 4, 8, 16, 64, 2048]
    attention._tpu_compilation_config = cfg.compilation_config

    trace_count = 0

    def counting_backend(gm, example_inputs):
        del example_inputs
        nonlocal trace_count
        trace_count += 1
        return gm.forward

    def call_fn(x, pos):
        return qwen3_next.Qwen3NextAttention._project_qkv_gate(attention, x, pos)

    # Deliberately run OUTSIDE set_current_vllm_config(cfg), matching vLLM
    # TPUModelRunner._precompile_backbone where no global vLLM config is active.
    compiled = torch.compile(call_fn, backend=counting_backend, fullgraph=True)
    for bucket in (4, 16, 64):
        qkv = _qkv(bucket)
        pos = torch.arange(bucket, dtype=torch.int32)
        torch._dynamo.mark_dynamic(qkv, 0)
        torch._dynamo.mark_dynamic(pos, 0)
        out = compiled(qkv, pos)
        assert len(out) == 4
    assert _DYN_SEEN_SHAPES == [4, 16, 64]
    assert trace_count == 1

    # Static-shape torch.compile with unset compile_sizes also dispatches fused op.
    _DYN_SEEN_SHAPES.clear()
    attention._tpu_compilation_config = None
    static_compiled = torch.compile(
        call_fn, backend=counting_backend, fullgraph=True, dynamic=False
    )
    out_static = static_compiled(_qkv(24), torch.arange(24, dtype=torch.int32))
    assert len(out_static) == 4
    assert _DYN_SEEN_SHAPES == [24]


def test_cpu_tensor_falls_back_and_direct_call_raises(monkeypatch):
    """Unpatched _is_tpu_tensor rejects CPU tensors cleanly without invoking pallas."""
    cls = qwen3_next.Qwen3NextAttention
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    get_chip_version.cache_clear()
    monkeypatch.setattr(envs, "TPU_FUSED_QK_NORM_ROPE", True)
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "_project_qkv_gate", cls._project_qkv_gate)
    monkeypatch.setattr(cls, "_tpu_fused_qk_norm_rope_patch", False, raising=False)
    orig = cls._project_qkv_gate
    monkeypatch.setattr(
        fused,
        "_get_op",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("should not build op")),
    )
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()
    try:
        attention = _attention(_TEXT_ROPE)
        qkv = _qkv(8)
        positions = torch.arange(8, dtype=torch.int32)
        assert not fused.can_dispatch_fused_qk_norm_rope(
            qkv,
            positions,
            attention.rotary_emb.cos_sin_cache,
            _NUM_HEADS,
            _NUM_KV_HEADS,
            _HEAD_DIM,
            64,
            _VMEM_64M,
        )
        with set_current_vllm_config(_cpu_vllm_config()):
            got = cls._project_qkv_gate(attention, qkv, positions)
            want = orig(attention, qkv, positions)
        for actual, expected in zip(got, want):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with pytest.raises(ValueError, match="must be on a TPU device"):
            fused.fused_qk_rmsnorm_rope_gate(
                qkv,
                torch.ones(_HEAD_DIM, dtype=torch.float32),
                torch.ones(_HEAD_DIM, dtype=torch.float32),
                attention.rotary_emb.cos_sin_cache,
                positions,
                _EPS,
                _NUM_HEADS,
                _NUM_KV_HEADS,
                _HEAD_DIM,
                64,
                vmem_capacity_bytes=_VMEM_64M,
            )
    finally:
        get_chip_version.cache_clear()


_SPLIT_CALLS: list[int] = []


_SPLIT_MIN_BYTES = 28 * (2 * _NUM_HEADS + 2 * _NUM_KV_HEADS) * _HEAD_DIM * 2


@torch.library.custom_op("test_qknorm_split::fused32", mutates_args=())
def _split_fused32(
    qkv: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _SPLIT_CALLS.append(int(qkv.shape[0]))
    jax_dtype = jnp.bfloat16 if qkv.dtype == torch.bfloat16 else jnp.float32
    outputs = qk_norm_rope_gate_rows(
        jnp.asarray(qkv.detach().float().numpy(), jax_dtype),
        jnp.asarray(q_w.detach().float().numpy()),
        jnp.asarray(k_w.detach().float().numpy()),
        jnp.asarray(cos_sin.detach().float().numpy()),
        num_heads=_NUM_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        rotary_dim=64,
        eps=_EPS,
        vmem_capacity_bytes=_VMEM_64M,
        num_sublanes=8,
        chip_version=pltpu.ChipVersion.TPU_7X,
        min_pallas_qkv_bytes=_SPLIT_MIN_BYTES,
        interpret=True,
    )
    return tuple(  # type: ignore[return-value]
        torch.from_numpy(np.asarray(out, np.float32)).to(qkv.dtype) for out in outputs
    )


@_split_fused32.register_fake
def _(
    qkv: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del q_w, k_w, cos_sin
    rows = qkv.shape[0]
    return (
        qkv.new_empty((rows, _NUM_HEADS * _HEAD_DIM)),
        qkv.new_empty((rows, _NUM_KV_HEADS * _HEAD_DIM)),
        qkv.new_empty((rows, _NUM_KV_HEADS * _HEAD_DIM)),
        qkv.new_empty((rows, _NUM_HEADS * _HEAD_DIM)),
    )


_SPLIT_OP_LIB = torch.library.Library("test_qknorm_split", "FRAGMENT")
_SPLIT_OP_LIB.define("attn_barrier(Tensor q, Tensor k, Tensor v) -> Tensor")
_SPLIT_OP_LIB.impl("attn_barrier", lambda q, _k, _v: q.clone(), "CPU")
torch.library.register_fake(
    "test_qknorm_split::attn_barrier", lambda q, _k, _v: q.new_empty(q.shape)
)


def test_mixed_compile_sizes_falls_back_inside_shape_variants_warmup(
    patched, monkeypatch
):
    """When any compile_sizes bucket >= MIN_PALLAS_QKV_BYTES is unsupported
    (such as 209 rows = 1,070,080 B unaligned > 32 at TP=8), _project_qkv_gate
    falls back unconditionally even inside shape_variants.warmup() so all
    buckets share the eager FX placeholder signature."""
    attention = _attention(_TEXT_ROPE)
    _DYN_SEEN_SHAPES.clear()
    monkeypatch.setattr(fused, "_get_op", lambda *_a, **_kw: _stub_qknorm_dyn)
    cfg = _cpu_vllm_config()
    cfg.compilation_config.compile_sizes = [4, 16, 209, 256]
    attention._tpu_compilation_config = cfg.compilation_config

    def call_fn(x, pos):
        return qwen3_next.Qwen3NextAttention._project_qkv_gate(attention, x, pos)

    with shape_variants.warmup() as session:
        assert shape_variants.aot_cache_tag() == "trace0"

        def checking_backend(gm, example_inputs):
            del example_inputs
            if session.get("cache", "bkt", gm) is None:
                session.put("cache", "bkt", gm, gm.forward)
            return gm.forward

        compiled = torch.compile(call_fn, backend=checking_backend, fullgraph=True)
        with set_current_vllm_config(cfg):
            for bucket in (4, 16, 209, 256):
                qkv = _qkv(bucket)
                pos = torch.arange(bucket, dtype=torch.int32)
                torch._dynamo.mark_dynamic(qkv, 0)
                torch._dynamo.mark_dynamic(pos, 0)
                got = compiled(qkv, pos)
                want = patched.original_project_qkv_gate(attention, qkv, pos)
                for actual, expected in zip(got, want):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert _DYN_SEEN_SHAPES == []


@pytest.mark.parametrize("rope_kind", ["text_1d", "mrope_2d"])
def test_piecewise_split_graph_single_trace_across_inline_and_pallas_buckets(
    patched, monkeypatch, rope_kind
):
    """A 2-layer Qwen3NextAttention graph partitioned by vLLM's real
    split_graph keeps a single FX trace and identical submod_* signatures
    across buckets below MIN_PALLAS_QKV_BYTES (8, 16, 24 rows -> inline JAX)
    and at or above MIN_PALLAS_QKV_BYTES (32, 64 rows -> Pallas) for both
    1D TpuRotaryEmbedding and 2D MRotaryEmbedding."""
    from vllm.compilation.backends import split_graph

    import vllm_torchtpu.layers.adapter.rotary_embedding  # noqa: F401

    monkeypatch.setattr(fused, "_get_op", lambda *_a, **_kw: _split_fused32)
    monkeypatch.setattr(fused, "MIN_PALLAS_QKV_BYTES", _SPLIT_MIN_BYTES)
    num_heads, num_kv_heads, head_dim = _NUM_HEADS, _NUM_KV_HEADS, _HEAD_DIM
    qkv_width = (2 * num_heads + 2 * num_kv_heads) * head_dim
    rope_params = _TEXT_ROPE if rope_kind == "text_1d" else _MROPE
    cfg = _cpu_vllm_config()
    cfg.compilation_config.compile_sizes = [8, 16, 24, 32, 64]

    def _make_layer(seed: int):
        torch.manual_seed(seed)
        with set_current_vllm_config(cfg):
            layer = SimpleNamespace(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_size=num_heads * head_dim,
                kv_size=num_kv_heads * head_dim,
                attn_output_gate=True,
                use_fused_qk_norm_rope_gate=False,
                q_norm=GemmaRMSNorm(head_dim, eps=_EPS),
                k_norm=GemmaRMSNorm(head_dim, eps=_EPS),
                rotary_emb=get_rope(
                    head_size=head_dim,
                    max_position=4096,
                    rope_parameters=rope_params,
                    dtype=torch.bfloat16,
                ),
                _tpu_fused_qk_norm_rope=True,
                _tpu_compilation_config=cfg.compilation_config,
                _tpu_chip_vmem_bytes=_VMEM_64M,
                _tpu_chip_num_sublanes=8,
                _tpu_chip_version=pltpu.ChipVersion.TPU_7X,
            )
        with torch.no_grad():
            layer.q_norm.weight.normal_(0.0, 0.1)
            layer.k_norm.weight.normal_(0.0, 0.1)
        return layer

    layer0 = _make_layer(10)
    layer1 = _make_layer(20)
    proj1 = torch.randn(num_heads * head_dim, qkv_width, dtype=torch.bfloat16) * 0.02

    def two_layer_fn(qkv0, pos):
        q0, k0, v0, g0 = qwen3_next.Qwen3NextAttention._project_qkv_gate(
            layer0, qkv0, pos
        )
        attn0 = torch.ops.test_qknorm_split.attn_barrier(q0, k0, v0) * torch.sigmoid(g0)
        qkv1 = attn0 @ proj1
        q1, k1, v1, g1 = qwen3_next.Qwen3NextAttention._project_qkv_gate(
            layer1, qkv1, pos
        )
        attn1 = torch.ops.test_qknorm_split.attn_barrier(q1, k1, v1) * torch.sigmoid(g1)
        return attn1

    def ref_two_layer_fn(qkv0, pos):
        q0, k0, v0, g0 = patched.original_project_qkv_gate(layer0, qkv0, pos)
        attn0 = q0 * torch.sigmoid(g0)
        qkv1 = attn0 @ proj1
        q1, _k1, _v1, g1 = patched.original_project_qkv_gate(layer1, qkv1, pos)
        return q1 * torch.sigmoid(g1)

    split_submod_counts: list[int] = []
    with shape_variants.warmup() as session:

        def piecewise_checking_backend(gm, example_inputs):
            del example_inputs
            split_gm, items = split_graph(gm, ["test_qknorm_split::attn_barrier"])
            non_split = [it for it in items if not it.is_splitting_graph]
            assert len(non_split) >= 2
            split_submod_counts.append(len(non_split))
            for it in non_split:
                cached = session.get("cache", it.submod_name, it.graph)
                if cached is None:
                    session.put("cache", it.submod_name, it.graph, it.graph.forward)
            return split_gm.forward

        torch._dynamo.reset()
        compiled = torch.compile(
            two_layer_fn, backend=piecewise_checking_backend, fullgraph=True
        )
        _SPLIT_CALLS.clear()
        for bucket in (8, 16, 24, 32, 64):
            gen = torch.Generator().manual_seed(100 + bucket)
            qkv0 = torch.randn(bucket, qkv_width, generator=gen, dtype=torch.bfloat16)
            pos_1d = torch.arange(bucket, dtype=torch.int32)
            pos = (
                pos_1d
                if rope_kind == "text_1d"
                else pos_1d.expand(3, bucket).contiguous()
            )
            torch._dynamo.mark_dynamic(qkv0, 0)
            torch._dynamo.mark_dynamic(pos, 0 if rope_kind == "text_1d" else 1)
            with set_current_vllm_config(cfg):
                got = compiled(qkv0, pos)
                want = ref_two_layer_fn(qkv0, pos)
            torch.testing.assert_close(got.float(), want.float(), atol=4e-2, rtol=3e-2)
        assert session.traces == 0
        assert len(split_submod_counts) == 1
        assert _SPLIT_CALLS == [8, 8, 16, 16, 24, 24, 32, 32, 64, 64]


def _patch_single_rank_tp(monkeypatch):
    monkeypatch.setattr(current_platform, "current_device", lambda: torch.device("cpu"))
    for mod in (qwen3_next, linear_mod, parameter_mod, distributed_mod):
        if hasattr(mod, "get_tensor_model_parallel_rank"):
            monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)
        if hasattr(mod, "get_tensor_model_parallel_world_size"):
            monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)


def _real_qwen3_next_config(head_dim: int, rope_parameters: dict) -> Qwen3NextConfig:
    return Qwen3NextConfig(
        hidden_size=2 * head_dim,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        max_position_embeddings=256,
        rms_norm_eps=_EPS,
        rope_parameters=rope_parameters,
    )


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize(
    "mm_mode,expected_mm_ok",
    [
        ("none", True),
        ("lang_only", True),
        ("lang_only_mm_embeds", False),
        ("limit_zero", True),
        ("multimodal_active", False),
        ("draft_multimodal_active", False),
        ("draft_lang_only", True),
    ],
)
@pytest.mark.parametrize("head_dim", [256, 64])
def test_enabled_only_for_true_text_only_and_aligned_head_dim(
    monkeypatch, positional, mm_mode, expected_mm_ok, head_dim
):
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    get_chip_version.cache_clear()
    _patch_single_rank_tp(monkeypatch)

    cls = qwen3_next.Qwen3NextAttention
    monkeypatch.setattr(envs, "TPU_FUSED_QK_NORM_ROPE", True)
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "_project_qkv_gate", cls._project_qkv_gate)
    monkeypatch.setattr(cls, "_tpu_fused_qk_norm_rope_patch", False, raising=False)
    prebuilt = []
    monkeypatch.setattr(
        fused,
        "prebuild",
        lambda *args, **kwargs: prebuilt.append((args, kwargs)),
    )
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()

    parent_vllm_cfg = _cpu_vllm_config()
    resolve_kv_cache_layout(parent_vllm_cfg, [["LBNHC", "LBHNC"]])
    # Simulate TPU device_config for prebuild while allocating weights on CPU.
    parent_vllm_cfg.device_config.device_type = "tpu"
    if mm_mode == "none":
        model_config = SimpleNamespace(
            runner_type="generate",
            multimodal_config=None,
            supports_multimodal_inputs=False,
        )
    elif mm_mode == "lang_only":
        model_config = SimpleNamespace(
            runner_type="generate",
            multimodal_config=SimpleNamespace(
                language_model_only=True, enable_mm_embeds=False
            ),
            supports_multimodal_inputs=False,
        )
    elif mm_mode == "lang_only_mm_embeds":
        model_config = SimpleNamespace(
            runner_type="generate",
            multimodal_config=SimpleNamespace(
                language_model_only=True, enable_mm_embeds=True
            ),
            supports_multimodal_inputs=False,
        )
    elif mm_mode == "limit_zero":
        model_config = SimpleNamespace(
            runner_type="generate",
            multimodal_config=SimpleNamespace(
                language_model_only=False, enable_mm_embeds=False
            ),
            supports_multimodal_inputs=False,
        )
    elif mm_mode == "draft_multimodal_active":
        parent_vllm_cfg.model_config = SimpleNamespace(
            dtype=torch.bfloat16,
            is_mm_prefix_lm=False,
            is_attention_free=False,
            is_hybrid=False,
            max_model_len=256,
            rswa_window=None,
            supports_multimodal_inputs=True,
        )
        model_config = SimpleNamespace(
            runner_type="draft",
            multimodal_config=SimpleNamespace(
                language_model_only=False, enable_mm_embeds=False
            ),
            supports_multimodal_inputs=False,
        )
    elif mm_mode == "draft_lang_only":
        parent_vllm_cfg.model_config = SimpleNamespace(
            dtype=torch.bfloat16,
            is_mm_prefix_lm=False,
            is_attention_free=False,
            is_hybrid=False,
            max_model_len=256,
            rswa_window=None,
            supports_multimodal_inputs=False,
        )
        model_config = SimpleNamespace(
            runner_type="draft",
            multimodal_config=SimpleNamespace(
                language_model_only=True, enable_mm_embeds=False
            ),
            supports_multimodal_inputs=False,
        )
    else:
        model_config = SimpleNamespace(
            runner_type="generate",
            multimodal_config=SimpleNamespace(
                language_model_only=False, enable_mm_embeds=False
            ),
            supports_multimodal_inputs=True,
        )

    rope_params = (
        _MROPE
        if head_dim == 256
        else {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 1.0}
    )
    hf_cfg = _real_qwen3_next_config(head_dim, rope_params)
    try:
        with torch.device("cpu"), set_current_vllm_config(parent_vllm_cfg):
            attention = (
                cls(hf_cfg, model_config, prefix="model.layers.0.self_attn")
                if positional
                else cls(
                    hf_cfg,
                    model_config=model_config,
                    prefix="model.layers.0.self_attn",
                )
            )
        expected = expected_mm_ok and (head_dim % 128 == 0)
        assert attention._tpu_fused_qk_norm_rope == expected
        if expected:
            assert len(prebuilt) == 1
            assert prebuilt[0][0] == (2, 1, head_dim, 64, _EPS)
            assert prebuilt[0][1]["vmem_capacity_bytes"] == _VMEM_64M
            assert prebuilt[0][1]["num_sublanes"] == 8
            assert prebuilt[0][1]["chip_version"] == pltpu.ChipVersion.TPU_7X
        else:
            assert prebuilt == []
    finally:
        get_chip_version.cache_clear()


def test_incompatible_rope_subclasses_and_cpu_device_config(monkeypatch):
    """Real Qwen3NextAttention rejects non-standard RoPE subclasses and skips
    prebuild when device_config.device.type != 'tpu'."""
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    get_chip_version.cache_clear()
    _patch_single_rank_tp(monkeypatch)

    cls = qwen3_next.Qwen3NextAttention
    unpatched_init = cls.__init__
    monkeypatch.setattr(envs, "TPU_FUSED_QK_NORM_ROPE", True)
    monkeypatch.setattr(cls, "__init__", unpatched_init)
    monkeypatch.setattr(cls, "_project_qkv_gate", cls._project_qkv_gate)
    monkeypatch.setattr(cls, "_tpu_fused_qk_norm_rope_patch", False, raising=False)
    prebuilt = []
    monkeypatch.setattr(
        fused,
        "prebuild",
        lambda *args, **kwargs: prebuilt.append((args, kwargs)),
    )

    rope_holder: dict[str, object | None] = {"override": None}

    def init_with_rope_override(self, *args, **kwargs):
        unpatched_init(self, *args, **kwargs)
        if rope_holder["override"] is not None:
            self.rotary_emb = rope_holder["override"]

    cls.__init__ = init_with_rope_override
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()

    cpu_cfg = _cpu_vllm_config()
    resolve_kv_cache_layout(cpu_cfg, [["LBNHC", "LBHNC"]])
    hf_cfg = _real_qwen3_next_config(256, _TEXT_ROPE)
    text_mc = SimpleNamespace(
        runner_type="generate",
        multimodal_config=None,
        supports_multimodal_inputs=False,
    )

    try:
        with torch.device("cpu"), set_current_vllm_config(cpu_cfg):
            rejected_ropes = [
                rope_mod.FourierRotaryEmbedding(
                    head_size=256,
                    rotary_dim=64,
                    max_position_embeddings=256,
                    base=10000.0,
                    is_neox_style=True,
                    dtype=torch.bfloat16,
                    init_cache=False,
                    num_key_value_heads=1,
                    num_inv_freq=16,
                    fope_sep_head=True,
                    fope_init_factor=0.1,
                ),
                rope_mod.BailingMRotaryEmbedding(
                    head_size=256,
                    rotary_dim=64,
                    max_position_embeddings=256,
                    base=10000.0,
                    is_neox_style=True,
                    dtype=torch.bfloat16,
                    mrope_section=[16, 8, 8],
                ),
                rope_mod.DeepseekV4ScalingRotaryEmbedding(
                    head_size=256,
                    rotary_dim=64,
                    max_position_embeddings=256,
                    base=10000.0,
                    is_neox_style=True,
                    scaling_factor=2.0,
                    dtype=torch.bfloat16,
                ),
                rope_mod.LinearScalingRotaryEmbedding(
                    head_size=256,
                    rotary_dim=64,
                    max_position_embeddings=256,
                    base=10000.0,
                    is_neox_style=True,
                    scaling_factors=[1.0, 2.0],
                    dtype=torch.bfloat16,
                ),
            ]
            accepted_linear = rope_mod.LinearScalingRotaryEmbedding(
                head_size=256,
                rotary_dim=64,
                max_position_embeddings=256,
                base=10000.0,
                is_neox_style=True,
                scaling_factors=[2.0],
                dtype=torch.bfloat16,
            )
            for idx, bad_rope in enumerate(rejected_ropes):
                rope_holder["override"] = bad_rope
                attn_bad = cls(
                    hf_cfg,
                    model_config=text_mc,
                    prefix=f"model.layers.{idx}.self_attn",
                )
                assert attn_bad._tpu_fused_qk_norm_rope is False, type(bad_rope)

            # Default TpuRotaryEmbedding and single-factor LinearScalingRotaryEmbedding
            # are both eligible, and under CPU DeviceConfig prebuild is skipped cleanly.
            rope_holder["override"] = None
            attn_default = cls(
                hf_cfg, model_config=text_mc, prefix="model.layers.9.self_attn"
            )
            assert attn_default._tpu_fused_qk_norm_rope is True
            rope_holder["override"] = accepted_linear
            attn_ok = cls(
                hf_cfg, model_config=text_mc, prefix="model.layers.10.self_attn"
            )
            assert attn_ok._tpu_fused_qk_norm_rope is True
            assert prebuilt == []
    finally:
        get_chip_version.cache_clear()


def test_distinct_eps_and_chip_info_register_distinct_op_names(monkeypatch):
    registered_names = []

    class RecordingJaxOp:
        def __init__(self, name, fn):
            registered_names.append(name)
            self.name = name
            self.fn = fn

        def register_fake(self, fake):
            self.fake = fake

    monkeypatch.setitem(
        sys.modules,
        "torch_tpu._internal",
        SimpleNamespace(pallas=SimpleNamespace(jax_op=RecordingJaxOp)),
    )
    monkeypatch.setitem(sys.modules, "torch_tpu", SimpleNamespace())
    monkeypatch.setattr(fused, "_ops_by_geometry", {})

    op_a = fused._get_op(
        4,
        1,
        256,
        64,
        1e-6,
        vmem_capacity_bytes=_VMEM_64M,
        num_sublanes=8,
        chip_version=pltpu.ChipVersion.TPU_7X,
    )
    op_b = fused._get_op(
        4,
        1,
        256,
        64,
        1e-5,
        vmem_capacity_bytes=_VMEM_64M,
        num_sublanes=8,
        chip_version=pltpu.ChipVersion.TPU_7X,
    )
    op_c = fused._get_op(
        4,
        1,
        256,
        64,
        1e-6,
        vmem_capacity_bytes=_VMEM_64M,
        num_sublanes=8,
        chip_version=pltpu.ChipVersion.TPU_V6E,
    )
    assert op_a is not op_b
    assert op_a is not op_c
    assert len(set(registered_names)) == 3


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
            assert self.fake is not None
            return self.fake(*args)

    monkeypatch.setitem(
        sys.modules,
        "torch_tpu._internal",
        SimpleNamespace(pallas=SimpleNamespace(jax_op=FakeJaxOp)),
    )
    monkeypatch.setitem(sys.modules, "torch_tpu", SimpleNamespace())
    monkeypatch.setattr(fused, "_ops_by_geometry", {})
    monkeypatch.setattr(fused, "_is_tpu_tensor", lambda _tensor: True)
    num_heads, num_kv_heads, head_dim, rotary_dim = 8, 1, 256, 64
    total_cols = (2 * num_heads + 2 * num_kv_heads) * head_dim
    fused.prebuild(
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        _EPS,
        vmem_capacity_bytes=_VMEM_64M,
    )
    with FakeTensorMode():
        q, k, v, gate = fused.fused_qk_rmsnorm_rope_gate(
            torch.empty(16, total_cols, dtype=torch.bfloat16),
            torch.empty(head_dim, dtype=torch.float32),
            torch.empty(head_dim, dtype=torch.float32),
            torch.empty(4096, rotary_dim, dtype=torch.bfloat16),
            torch.zeros(16, dtype=torch.int32),
            _EPS,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            vmem_capacity_bytes=_VMEM_64M,
        )
    assert tuple(q.shape) == (16, num_heads * head_dim)
    assert tuple(k.shape) == (16, num_kv_heads * head_dim)
    assert tuple(v.shape) == (16, num_kv_heads * head_dim)
    assert tuple(gate.shape) == (16, num_heads * head_dim)
    assert q.dtype == k.dtype == v.dtype == gate.dtype == torch.bfloat16


@pytest.mark.tpu_test
@pytest.mark.parametrize("rows", [4, 24, 32, 64])
def test_tpu_pallas_jax_op_eager_and_compiled(monkeypatch, rows):
    """End-to-end TPU execution through real torch_tpu pallas.jax_op across
    both inline (< MIN_PALLAS_QKV_BYTES: 4, 24) and Pallas (>= 1 MiB: 32, 64)
    buckets at DP=8 / TP=1 Qwen3.5-397B head geometry."""
    if not pltpu.is_tpu_device():
        pytest.skip("requires TPU")
    cls = qwen3_next.Qwen3NextAttention
    monkeypatch.setattr(envs, "TPU_FUSED_QK_NORM_ROPE", True)
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "_project_qkv_gate", cls._project_qkv_gate)
    monkeypatch.setattr(cls, "_tpu_fused_qk_norm_rope_patch", False, raising=False)
    monkeypatch.setattr(fused, "_ops_by_geometry", {})
    orig = cls._project_qkv_gate
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()
    try:
        device = torch.device("tpu:0")
        num_heads, num_kv_heads, head_dim = 32, 2, 256
        qkv_width = (2 * num_heads + 2 * num_kv_heads) * head_dim
        vcfg = VllmConfig()
        with set_current_vllm_config(vcfg):
            attention = SimpleNamespace(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_size=num_heads * head_dim,
                kv_size=num_kv_heads * head_dim,
                attn_output_gate=True,
                use_fused_qk_norm_rope_gate=False,
                q_norm=GemmaRMSNorm(head_dim, eps=_EPS).to(device),
                k_norm=GemmaRMSNorm(head_dim, eps=_EPS).to(device),
                rotary_emb=get_rope(
                    head_size=head_dim,
                    max_position=4096,
                    rope_parameters=_TEXT_ROPE,
                    dtype=torch.bfloat16,
                ).to(device),
                _tpu_fused_qk_norm_rope=True,
                _tpu_compilation_config=vcfg.compilation_config,
                _tpu_chip_vmem_bytes=_VMEM_64M,
                _tpu_chip_num_sublanes=8,
                _tpu_chip_version=pltpu.ChipVersion.TPU_7X,
            )
        fused.prebuild(
            num_heads,
            num_kv_heads,
            head_dim,
            64,
            _EPS,
            vmem_capacity_bytes=_VMEM_64M,
            num_sublanes=8,
            chip_version=pltpu.ChipVersion.TPU_7X,
        )
        gen = torch.Generator().manual_seed(rows)
        qkv = torch.randn(rows, qkv_width, generator=gen, dtype=torch.bfloat16).to(
            device
        )
        pos = torch.arange(rows, dtype=torch.int32, device=device)
        with torch.no_grad():
            want = orig(attention, qkv, pos)
            got_eager = cls._project_qkv_gate(attention, qkv, pos)
            compiled = torch.compile(
                lambda x, p: cls._project_qkv_gate(attention, x, p),
                backend="tpu",
                fullgraph=True,
                dynamic=False,
            )
            got_compiled = compiled(qkv, pos)
        for got in (got_eager, got_compiled):
            for name, actual, expected in zip(("q", "k", "v", "gate"), got, want):
                torch.testing.assert_close(
                    actual.cpu().float(),
                    expected.cpu().float(),
                    atol=3e-2,
                    rtol=2e-2,
                    msg=name,
                )
    finally:
        torch._dynamo.reset()
