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
"""Tests for the GLM-5.3-Flash TPU KDA layer (``glm53_kda_tpu``).

The CPU tests cover the glue around the Kimi KDA ops: the state shape and
dtype contract, the conv-weight fold, the ``o_norm`` swap, how the layer builds
and calls the pooled and dispatched ops, and, with the Pallas pieces replaced
by pure-JAX stand-ins (byte-level pool gather/scatter and a reference
recurrence in place of the fused kernel), multi-step batches whose state
round-trips through a ragged-paged-attention-shaped unified pool. The real
kernels run in the ``tpu_test`` cases.
"""

from __future__ import annotations

import importlib
import math
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.custom_op import op_registry_oot

from vllm_torchtpu import gdn_pool_layout
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.layers.adapter.custom_ops import glm53_kda_tpu as L
from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op

H, D, K = 2, 16, 4  # Local heads, head dim, conv taps.
PROJ = H * D
DIM = 3 * PROJ
T = 6  # Tokens in a routing step.
LB = -5.0
EPS = 1e-5
HIDDEN = 12
PREFIX = "model.layers.0.self_attn"
FLA_KDA_MODULES = (
    "vllm.third_party.flash_linear_attention.ops.kda",
    "vllm.model_executor.layers.fla.ops.kda",
)


class _Conv(nn.Module):
    """``conv1d`` as the base class builds it: weight ``[3 * proj, 1, K]``."""

    def __init__(self, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(DIM, 1, K))
        self.bias = nn.Parameter(torch.randn(DIM)) if bias else None


class _BaseNorm(nn.Module):
    """The attributes of vLLM's ``FusedRMSNormGated`` the layer reads."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(D, dtype=torch.bfloat16))
        self.eps = EPS
        self.activation = "sigmoid"


def _base_init(conv_bias: bool, num_spec: int):
    """Stands in for ``KimiGatedDeltaNetAttention.__init__``, which needs a
    full vLLM config and TP groups. Sets only what the TPU subclass reads."""

    def init(self, config, vllm_config, prefix=""):
        del config, vllm_config
        nn.Module.__init__(self)
        self.prefix = prefix
        self.conv_size = K
        self.local_num_heads = H
        self.head_dim = D
        self.local_projection_size = PROJ
        self.num_spec = num_spec
        self.gate_lower_bound = LB
        self.use_full_rank_gate = True
        self.in_proj_padding = 0
        self.conv1d = _Conv(conv_bias)
        self.o_norm = _BaseNorm()
        self.A_log = nn.Parameter(torch.randn(H) * 0.3)
        self.dt_bias = nn.Parameter(torch.randn(PROJ) * 0.3)

    return init


def _tripwire(name: str):
    """An op stand-in that fails the test if the layer calls it."""

    def op(*args, **kwargs):
        raise AssertionError(f"{name} must not run on this path")

    return op


class _RecordingOp:
    """Records its arguments and returns a distinctive ``[T, H, D]`` output."""

    def __init__(self, num_tokens: int = T) -> None:
        self.calls: list[tuple] = []
        self.output = torch.randn(num_tokens, H, D)

    def __call__(self, *args):
        self.calls.append(args)
        return self.output.clone()


def _vllm_config(block_size: int = 16):
    return SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size))


@pytest.fixture
def builders(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replaces the Kimi op builders and records how the layer calls them."""
    calls: dict = {}

    def recorder(kind: str):
        def build(prefix, **kwargs):
            calls[kind] = (prefix, kwargs)
            return _tripwire(f"{kind}_kda_op")

        return build

    monkeypatch.setattr(L, "build_kimi_dispatched_kda_op", recorder("dispatched"))
    monkeypatch.setattr(L, "build_kimi_pooled_kda_op", recorder("pooled"))
    monkeypatch.setattr(L, "is_conv_state_dim_first", lambda: False)
    return calls


def _make_layer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    conv_bias: bool = False,
    num_spec: int = 0,
    vllm_config=None,
):
    monkeypatch.setattr(
        L.KimiGatedDeltaNetAttention, "__init__", _base_init(conv_bias, num_spec)
    )
    if vllm_config is None:
        vllm_config = _vllm_config()
    return L.Glm53TpuKimiDeltaAttention(SimpleNamespace(), vllm_config, PREFIX)


def _set_context(monkeypatch: pytest.MonkeyPatch, metadata) -> None:
    context = SimpleNamespace(
        attn_metadata=None if metadata is None else {PREFIX: metadata}
    )
    monkeypatch.setattr(L, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(L, "get_forward_context", lambda: context)


# Import hygiene.


def test_import_has_no_global_side_effects() -> None:
    """Re-executing the module leaves sys.modules and vLLM's pluggable-layer
    registry untouched (no FLA shim, no out-of-tree registration)."""
    before_modules = {name: sys.modules.get(name) for name in FLA_KDA_MODULES}
    before_registry = dict(op_registry_oot)

    importlib.reload(L)

    assert {name: sys.modules.get(name) for name in FLA_KDA_MODULES} == (before_modules)
    assert dict(op_registry_oot) == before_registry
    assert "Glm53TpuKimiDeltaAttention" not in op_registry_oot
    assert "KimiGatedDeltaNetAttention" not in op_registry_oot


def test_layer_instantiates_as_itself(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    """``PluggableLayer.__new__`` swaps only registered class names."""
    layer = _make_layer(monkeypatch)
    assert type(layer) is L.Glm53TpuKimiDeltaAttention
    assert isinstance(layer, L.KimiGatedDeltaNetAttention)


# The state contract.


def test_state_shape() -> None:
    assert L.glm53_kda_state_shape(4, 8, 128) == ((3, 1, 3 * 8 * 128), (8, 128, 128))


def test_state_dtype_is_the_pool_state_dtype() -> None:
    conv, recurrent = L.glm53_kda_state_dtype()
    assert (conv, recurrent) == (torch.float32, torch.float32)
    assert str(conv) == gdn_pool_layout.DEFAULT_POOLED_GDN_CONV_STATE_DTYPE
    assert str(recurrent) == gdn_pool_layout.DEFAULT_POOLED_GDN_SSM_STATE_DTYPE


@pytest.mark.parametrize(("heads", "head_dim"), [(2, 16), (8, 128), (4, 64)])
@pytest.mark.parametrize("token_bytes", [1024, 2048])
def test_pool_state_layout_sizes_the_conv_region_in_float32(
    heads: int, head_dim: int, token_bytes: int
) -> None:
    layout = L.glm53_kda_pool_state_layout(
        conv_kernel_size=K,
        local_num_heads=heads,
        head_dim=head_dim,
        token_bytes=token_bytes,
    )
    conv_bytes = (K - 1) * 3 * heads * head_dim * 4
    ssm_bytes = heads * head_dim * head_dim * 4
    assert layout.conv_bytes == conv_bytes
    assert layout.ssm_bytes == ssm_bytes
    assert layout.conv_tokens * token_bytes >= conv_bytes
    assert layout.ssm_tokens == math.ceil(ssm_bytes / token_bytes)
    # The same layout the pooled op derives for its gather and scatter.
    assert layout == gdn_pool_layout.derive_pooled_gdn_state_layout(
        ssm_bytes=gdn_pool_layout.pooled_gdn_ssm_state_bytes(
            num_v_heads=heads, head_k_dim=head_dim, head_v_dim=head_dim
        ),
        conv_bytes=gdn_pool_layout.pooled_gdn_conv_state_bytes(
            kernel_size=K, conv_dim=3 * heads * head_dim
        ),
        token_bytes=token_bytes,
    )


def test_layer_declares_the_helper_state(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _make_layer(monkeypatch)
    assert layer.get_state_shape() == L.glm53_kda_state_shape(K, H, D)
    assert layer.get_state_dtype() == L.glm53_kda_state_dtype()


# Construction.


@pytest.mark.parametrize("num_spec", [0, 2])
def test_construction_builds_both_ops(
    monkeypatch: pytest.MonkeyPatch, builders: dict, num_spec: int
) -> None:
    config = _vllm_config()
    layer = _make_layer(monkeypatch, num_spec=num_spec, vllm_config=config)

    assert not layer._conv_weight_folded
    assert builders["dispatched"] == (
        PREFIX,
        dict(lower_bound=LB, eps=EPS, state_dim_first=False, num_spec_tokens=num_spec),
    )
    prefix, kwargs = builders["pooled"]
    assert prefix == PREFIX
    assert kwargs.pop("vllm_config") is config
    assert kwargs == dict(
        lower_bound=LB, eps=EPS, num_spec_tokens=num_spec, pool_tokens_per_row=1
    )
    buffer = dict(layer.named_buffers())["conv_weight_fused"]
    assert buffer.shape == (K, 3, H, D)
    assert buffer.dtype == torch.float32
    assert "conv_weight_fused" not in layer.state_dict()


def test_construction_uses_mla_row_packing_when_deepseek_mla_enabled(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    config = _vllm_config()
    config.model_config = SimpleNamespace(
        model_arch_config=SimpleNamespace(is_deepseek_mla=True)
    )
    _make_layer(monkeypatch, vllm_config=config)
    _, kwargs = builders["pooled"]
    assert kwargs["pool_tokens_per_row"] is None


def test_construction_replaces_the_triton_norm(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _make_layer(monkeypatch)
    norm = layer.o_norm
    assert isinstance(norm, L.Glm53GatedRMSNorm)
    assert norm.eps == EPS
    assert norm.activation == "sigmoid"
    # Same parameter name and dtype, so checkpoints load unchanged.
    assert norm.weight.dtype == torch.bfloat16
    assert norm.weight.shape == (D,)
    assert "o_norm.weight" in layer.state_dict()


def test_gated_norm_matches_the_reference() -> None:
    norm = L.Glm53GatedRMSNorm(D, eps=EPS)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(D))
    x, g = torch.randn(T, H, D), torch.randn(T, H, D)
    expected = (
        x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * norm.weight
    ) * torch.sigmoid(g)
    torch.testing.assert_close(norm(x, g), expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="activation"):
        L.Glm53GatedRMSNorm(D, activation="relu")


def test_construction_rejects_a_conv_bias(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    with pytest.raises(NotImplementedError, match="bias-free"):
        _make_layer(monkeypatch, conv_bias=True)


class _FakeJaxOp:
    """``pallas.jax_op`` on CPU: converts torch <-> JAX around the function."""

    instances: list[_FakeJaxOp] = []

    def __init__(self, name, function, donate_argnums=()):
        self.name = name
        self.function = function
        self.donate_argnums = donate_argnums
        _FakeJaxOp.instances.append(self)

    def register_fake(self, fake):
        self.fake = fake

    def __call__(self, *args):
        out = self.function(*(_to_jax(a) for a in args))
        return tuple(_to_torch(o) for o in out)


def _to_jax(t):
    if t is None:
        return None
    if t.dtype == torch.bfloat16:
        bits = jnp.asarray(t.detach().contiguous().view(torch.int16).numpy())
        return jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
    return jnp.asarray(t.detach().numpy())


def _to_torch(a) -> torch.Tensor:
    if a.dtype == jnp.bfloat16:
        bits = np.asarray(jax.lax.bitcast_convert_type(a, jnp.int16))
        return torch.from_numpy(bits.copy()).view(torch.bfloat16)
    return torch.from_numpy(np.array(a))


@pytest.mark.parametrize("num_spec", [0, 2])
def test_real_pooled_builder_compiles_the_row_packing_and_conv_dtype(
    monkeypatch: pytest.MonkeyPatch, num_spec: int
) -> None:
    """The layer's pooled op is the shared Kimi op, built with one token per
    pool row and the fp32 conv region; both are part of the program name."""
    monkeypatch.setattr(kda_attention_op.pallas, "jax_op", _FakeJaxOp)
    monkeypatch.setattr(L, "is_conv_state_dim_first", lambda: False)
    _FakeJaxOp.instances.clear()
    _make_layer(monkeypatch, num_spec=num_spec)
    names = [op.name for op in _FakeJaxOp.instances]
    (pooled,) = [n for n in names if "kimi_pooled_kda" in n]
    assert pooled.endswith("_rows1")
    assert "_conv_float32" in pooled
    assert ("_window" in pooled) == (num_spec > 0)
    assert any("kimi_dispatched_kda" in n for n in names)


# The conv-weight fold.


def test_fold_conv_weight_layout(monkeypatch: pytest.MonkeyPatch, builders: dict):
    layer = _make_layer(monkeypatch)
    channel = torch.arange(DIM, dtype=torch.float32)[:, None, None]
    tap = torch.arange(K, dtype=torch.float32)[None, None, :]
    with torch.no_grad():
        layer.conv1d.weight.copy_(100 * channel + tap)

    layer.fold_conv_weight()

    fused = layer.conv_weight_fused
    assert layer._conv_weight_folded
    assert fused.shape == (K, 3, H, D)
    assert fused.is_contiguous()
    j, h, d = torch.meshgrid(
        torch.arange(3), torch.arange(H), torch.arange(D), indexing="ij"
    )
    c = (j * PROJ + h * D + d).float()
    for k in range(K):
        torch.testing.assert_close(fused[k], 100 * c + k, rtol=0, atol=0)
    assert dict(layer.named_buffers())["conv_weight_fused"] is fused
    assert "conv_weight_fused" not in layer.state_dict()


def test_folded_weight_reproduces_the_causal_depthwise_conv(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _make_layer(monkeypatch)
    layer.fold_conv_weight()
    x = torch.randn(T, DIM)
    expected = F.conv1d(
        F.pad(x.t()[None], (K - 1, 0)), layer.conv1d.weight.detach(), groups=DIM
    )[0].t()
    padded = F.pad(x, (0, 0, K - 1, 0))
    fused = layer.conv_weight_fused.reshape(K, DIM)
    actual = sum(fused[k] * padded[k : k + T] for k in range(K))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_process_weights_after_loading_folds_idempotently(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _make_layer(monkeypatch)
    layer.fold_conv_weight()
    first = layer.conv_weight_fused.clone()
    layer.process_weights_after_loading(torch.bfloat16)
    torch.testing.assert_close(layer.conv_weight_fused, first, rtol=0, atol=0)


# Routing.


def _metadata(**overrides) -> SimpleNamespace:
    """vllm_torchtpu per-layer metadata: one decode, then one prefill."""
    fields = dict(
        query_start_loc=torch.tensor([0, 1, T], dtype=torch.int32),
        seq_lens=torch.tensor([9, T - 1], dtype=torch.int32),
        request_distribution=torch.tensor([1, 1, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([3, 5], dtype=torch.int32),
        mamba_request_distribution=None,
        mamba_slot_read_offsets=None,
        mamba_ckpt_indices=None,
        mamba_prefill_ckpt_offsets=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _inputs():
    """What the base class hands ``_forward``: a split view of the fused
    projection, g1 and beta with a batch dim, g2 without one."""
    mixed_qkv = torch.randn(T, DIM + 5).split([DIM, 5], dim=-1)[0]
    return (
        mixed_qkv,
        torch.randn(1, T, H, D),
        torch.randn(T, H, D),
        torch.randn(1, T, H),
    )


def _bound_layer(monkeypatch: pytest.MonkeyPatch, kv_cache, num_spec: int = 0):
    layer = _make_layer(monkeypatch, num_spec=num_spec)
    layer.fold_conv_weight()
    layer.kv_cache = kv_cache
    layer.pooled_kda_op = _tripwire("pooled_kda_op")
    layer.dispatched_kda_op = _tripwire("dispatched_kda_op")
    return layer


def _pool() -> torch.Tensor:
    # A ragged paged attention pool: (blocks, block_size, kv_heads_x2 //
    # packing, packing, head_dim).
    return torch.zeros(4, 16, 2, 2, 128, dtype=torch.bfloat16)


def _run_forward(monkeypatch, layer, metadata, inputs, out_ndim: int = 4):
    _set_context(monkeypatch, metadata)
    mixed_qkv, g1, g2, beta = inputs
    core = torch.full((1, T, H, D)[4 - out_ndim :], float("nan"))
    layer._forward(mixed_qkv, g1, g2, beta, core)
    return core


@pytest.mark.parametrize("out_ndim", [3, 4])
def test_forward_sends_the_pool_to_the_pooled_op(
    monkeypatch: pytest.MonkeyPatch, builders: dict, out_ndim: int
) -> None:
    pool = _pool()
    layer = _bound_layer(monkeypatch, [pool])
    op = layer.pooled_kda_op = _RecordingOp()
    metadata = _metadata()
    inputs = mixed_qkv, g1, g2, beta = _inputs()

    core = _run_forward(monkeypatch, layer, metadata, inputs, out_ndim)

    (args,) = op.calls
    assert len(args) == 16
    assert args[0].is_contiguous()
    torch.testing.assert_close(args[0], mixed_qkv, rtol=0, atol=0)
    torch.testing.assert_close(args[1], g1.reshape(T, PROJ), rtol=0, atol=0)
    torch.testing.assert_close(args[2], beta.reshape(T, H), rtol=0, atol=0)
    torch.testing.assert_close(args[3], g2.reshape(T, PROJ), rtol=0, atol=0)
    expected = [
        pool,
        layer.conv_weight_fused,
        layer.A_log,
        layer.dt_bias,
        layer.o_norm.weight,
        metadata.query_start_loc,
        metadata.mamba_state_indices,
        metadata.seq_lens,
        metadata.request_distribution,
    ]
    for got, want in zip(args[4:13], expected, strict=True):
        assert got is want
    assert args[13:] == (None, None, None)
    want_out = op.output if out_ndim == 3 else op.output[None]
    torch.testing.assert_close(core, want_out, rtol=0, atol=0)


def test_pooled_forward_forwards_speculative_metadata(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _bound_layer(monkeypatch, [_pool()], num_spec=2)
    op = layer.pooled_kda_op = _RecordingOp()
    metadata = _metadata(
        mamba_request_distribution=torch.tensor([1, 2, 2], dtype=torch.int32),
        mamba_slot_read_offsets=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
        mamba_ckpt_indices=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
    )
    _run_forward(monkeypatch, layer, metadata, _inputs())
    (args,) = op.calls
    assert args[13] is metadata.mamba_request_distribution
    assert args[14] is metadata.mamba_slot_read_offsets
    assert args[15] is metadata.mamba_ckpt_indices


@pytest.mark.parametrize("missing", ["mamba_slot_read_offsets", "mamba_ckpt_indices"])
def test_pooled_speculative_forward_requires_checkpoint_metadata(
    monkeypatch: pytest.MonkeyPatch, builders: dict, missing: str
) -> None:
    layer = _bound_layer(monkeypatch, [_pool()], num_spec=1)
    fields = dict(
        mamba_slot_read_offsets=torch.zeros(4, dtype=torch.int32),
        mamba_ckpt_indices=torch.zeros(2, 2, dtype=torch.int32),
    )
    fields[missing] = None
    with pytest.raises(RuntimeError, match="mamba_ckpt_indices"):
        _run_forward(monkeypatch, layer, _metadata(**fields), _inputs())


@pytest.mark.parametrize("speculative", [False, True])
def test_forward_sends_per_layer_caches_to_the_dispatched_op(
    monkeypatch: pytest.MonkeyPatch, builders: dict, speculative: bool
) -> None:
    conv, recurrent = torch.zeros(8, K - 1, 1, DIM), torch.zeros(8, H, D, D)
    layer = _bound_layer(monkeypatch, [conv, recurrent])
    op = layer.dispatched_kda_op = _RecordingOp()
    overrides = {}
    if speculative:
        overrides = dict(
            mamba_request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
            mamba_slot_read_offsets=torch.tensor([0, 1], dtype=torch.int32),
        )
    metadata = _metadata(**overrides)

    core = _run_forward(monkeypatch, layer, metadata, _inputs())

    (args,) = op.calls
    assert len(args) == 16
    expected = [
        conv,
        recurrent,
        layer.conv_weight_fused,
        layer.A_log,
        layer.dt_bias,
        layer.o_norm.weight,
        metadata.query_start_loc,
        metadata.mamba_state_indices,
        metadata.seq_lens,
        metadata.request_distribution,
    ]
    for got, want in zip(args[4:14], expected, strict=True):
        assert got is want
    if speculative:
        assert args[14] is metadata.mamba_request_distribution
        assert args[15] is metadata.mamba_slot_read_offsets
    else:
        assert args[14] is metadata.request_distribution
        assert args[15] is metadata.mamba_state_indices
    torch.testing.assert_close(core, op.output[None], rtol=0, atol=0)


def test_forward_requires_the_folded_conv_weight(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _bound_layer(monkeypatch, [_pool()])
    layer._conv_weight_folded = False
    with pytest.raises(RuntimeError, match="never folded"):
        _run_forward(monkeypatch, layer, _metadata(), _inputs())


def test_forward_requires_mamba_state_indices(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _bound_layer(monkeypatch, [_pool()])
    with pytest.raises(RuntimeError, match="mamba_state_indices"):
        _run_forward(monkeypatch, layer, _metadata(mamba_state_indices=None), _inputs())


def test_forward_rejects_prefill_checkpoints(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _bound_layer(monkeypatch, [_pool()])
    metadata = _metadata(mamba_prefill_ckpt_offsets=torch.zeros(2, dtype=torch.int32))
    with pytest.raises(NotImplementedError, match="prefill checkpoints"):
        _run_forward(monkeypatch, layer, metadata, _inputs())


def test_forward_rejects_pcp_streaming_on_the_pool(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _bound_layer(monkeypatch, [_pool()])
    monkeypatch.setattr(L, "is_pcp_streaming_attention_metadata", lambda m: True)
    with pytest.raises(NotImplementedError, match="PCP"):
        _run_forward(monkeypatch, layer, _metadata(), _inputs())


def test_forward_rejects_gdn_metadata(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _bound_layer(monkeypatch, [_pool()])
    metadata = object.__new__(L.GDNAttentionMetadata)
    with pytest.raises(NotImplementedError, match="GDNAttentionMetadata"):
        _run_forward(monkeypatch, layer, metadata, _inputs())


def test_forward_reads_the_first_entry_for_an_unknown_prefix(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _bound_layer(monkeypatch, [_pool()])
    op = layer.pooled_kda_op = _RecordingOp()
    metadata = _metadata()
    context = SimpleNamespace(attn_metadata={"model.layers.9.self_attn": metadata})
    monkeypatch.setattr(L, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(L, "get_forward_context", lambda: context)
    mixed_qkv, g1, g2, beta = _inputs()
    layer._forward(mixed_qkv, g1, g2, beta, torch.empty(1, T, H, D))
    (args,) = op.calls
    assert args[10] is metadata.mamba_state_indices


@pytest.mark.parametrize(
    "setup", ["no_metadata", "no_cache", "empty_cache_list", "empty_cache"]
)
def test_forward_runs_no_op_before_the_cache_exists(
    monkeypatch: pytest.MonkeyPatch, builders: dict, setup: str
) -> None:
    kv_cache = {
        "no_metadata": [_pool()],
        "no_cache": None,
        "empty_cache_list": [],
        "empty_cache": [torch.empty(0)],
    }[setup]
    layer = _bound_layer(monkeypatch, kv_cache)
    metadata = None if setup == "no_metadata" else _metadata()
    core = _run_forward(monkeypatch, layer, metadata, _inputs())
    assert torch.all(core == 0)


def _with_projections(layer) -> SimpleNamespace:
    """Full-rank-gate projections, as GLM-5.3 configures them."""
    torch.manual_seed(1)
    in_proj = nn.Linear(HIDDEN, DIM + PROJ + D + H, bias=False)
    f_b = nn.Linear(D, PROJ, bias=False)
    o_proj = nn.Linear(PROJ, HIDDEN, bias=False)
    layer.in_proj_qkvgfab = lambda x: (in_proj(x),)
    layer.f_b_proj = lambda x: (f_b(x),)
    quant_method = SimpleNamespace(apply=lambda module, x, bias=None: o_proj(x) * 10.0)
    layer.o_proj = SimpleNamespace(
        __call__=None, quant_method=quant_method, linear=o_proj
    )
    layer.o_proj = _OProj(o_proj, quant_method)
    return SimpleNamespace(in_proj=in_proj, f_b=f_b, o_proj=o_proj)


class _OProj:
    def __init__(self, linear, quant_method) -> None:
        self.linear = linear
        self.quant_method = quant_method

    def __call__(self, x):
        return (self.linear(x),)


@torch.no_grad()
def test_forward_direct_matches_the_kernel_seam(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _bound_layer(monkeypatch, [_pool()])
    mods = _with_projections(layer)
    op = layer.pooled_kda_op = _RecordingOp()
    _set_context(monkeypatch, _metadata())
    hidden = torch.randn(T, HIDDEN)

    got = layer.forward_direct(hidden, None)
    assert torch.equal(layer.forward(hidden, None), got)

    projected = mods.in_proj(hidden)
    mixed_qkv, output_gate, f_a, beta = projected.split([DIM, PROJ, D, H], dim=-1)
    g1 = mods.f_b(f_a).reshape(1, T, H, D)
    core = torch.empty(1, T, H, D)
    layer._forward(mixed_qkv, g1, output_gate.reshape(T, H, D), beta[None], core)
    want = mods.o_proj(core.reshape(T, PROJ))
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    first, second, third = op.calls
    for a, b in zip(first[:4], third[:4], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(first[4:], third[4:], strict=True):
        assert a is b
    del second


@torch.no_grad()
def test_forward_direct_profiles_with_zeros_before_the_cache_exists(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _bound_layer(monkeypatch, [torch.empty(0)])
    mods = _with_projections(layer)
    _set_context(monkeypatch, _metadata())
    got = layer.forward_direct(torch.randn(T, HIDDEN), None)
    torch.testing.assert_close(got, mods.o_proj(torch.zeros(T, PROJ)))


# Multi-step batches through the unified pool, with the Pallas pieces replaced
# by pure-JAX stand-ins.


def _pool_by_manager_block(pool, split: int):
    return pool.reshape(pool.shape[0] // split, pool.shape[1] * split, *pool.shape[2:])


def _fake_gather_region(
    pool,
    state_indices,
    *,
    tok0,
    ntok,
    out_dtype,
    out_lanes=None,
    split=1,
    num_valid=None,
    row_offset=0,
):
    """Row-major byte view of a pool token range (the real op bitcasts in
    VMEM; the round trip with ``_fake_scatter_region`` is what matters).
    Rows the real op leaves uninitialized (the leading ``row_offset`` rows and
    the requests past ``num_valid``) are NaN poison."""
    region = _pool_by_manager_block(pool, split)[state_indices, tok0 : tok0 + ntok]
    num = region.shape[0]
    raw = jax.lax.bitcast_convert_type(region, jnp.uint8).reshape(num, -1)
    size = jnp.dtype(out_dtype).itemsize
    vals = jax.lax.bitcast_convert_type(raw.reshape(num, -1, size), out_dtype)
    vals = vals.reshape(num, -1, out_lanes or pool.shape[-1])
    if num_valid is not None:
        live = jnp.arange(num) < num_valid
        vals = jnp.where(live[:, None, None], vals, jnp.nan)
    poison = jnp.full((row_offset, *vals.shape[1:]), jnp.nan, vals.dtype)
    return jnp.concatenate([poison, vals], axis=0)


def _fake_scatter_region(
    pool, vals, state_indices, *, tok0, ntok, split=1, num_valid=None, row_offset=0
):
    by_block = _pool_by_manager_block(pool, split)
    vals = vals[row_offset:]
    num = vals.shape[0]
    raw = jax.lax.bitcast_convert_type(vals, jnp.uint8).reshape(num, -1)
    size = jnp.dtype(pool.dtype).itemsize
    new = jax.lax.bitcast_convert_type(raw.reshape(num, -1, size), pool.dtype)
    new = new.reshape(num, ntok, *pool.shape[2:])
    if num_valid is not None:
        live = jnp.arange(num) < num_valid
        keep = by_block[state_indices, tok0 : tok0 + ntok]
        new = jnp.where(live.reshape(-1, *([1] * (new.ndim - 1))), new, keep)
    by_block = by_block.at[state_indices, tok0 : tok0 + ntok].set(new)
    return by_block.reshape(pool.shape)


def _kda_reference_token(window, state, x, a, b, conv_w, a_log, dt_bias, lb):
    """One token of short conv + KDA. window [K - 1, dim], state [H, K, V],
    conv_w [dim, K]; returns (window, state, out [H, V])."""
    full = np.concatenate([window, x[None]], axis=0)
    conv = (full * conv_w.T).sum(0)
    conv = conv / (1.0 + np.exp(-conv))
    q, k, v = (t.reshape(H, D) for t in np.split(conv, 3))
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * D**-0.5
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    g = a.reshape(H, D) + dt_bias
    gate = 1.0 / (1.0 + np.exp(-np.exp(a_log)[:, None] * g))
    decay = np.exp(lb * gate)
    beta = 1.0 / (1.0 + np.exp(-b))
    state = state * decay[:, :, None]
    pred = np.einsum("hk,hkv->hv", k, state)
    state = state + np.einsum("hk,hv->hkv", beta[:, None] * k, v - pred)
    out = np.einsum("hk,hkv->hv", q, state)
    return full[1:], state, out


def _fake_fused_conv1d_gdn(**kw):
    """The fused kernel's contract, by a per-token reference recurrence."""
    conv_state, rec_state = kw["states"]
    assert kw.get("conv_dtype") in (None, jnp.float32)
    assert kw.get("num_spec_tokens", 0) == 0
    conv_shape = conv_state.shape
    conv = np.array(conv_state, np.float32).reshape(conv_shape[0], K - 1, -1)
    rec = np.array(rec_state, np.float32)
    qkv = np.array(kw["qkv"], np.float32)
    a = np.array(kw["a"], np.float32)
    b = np.array(kw["b"], np.float32)
    conv_w = np.array(kw["conv_weight"], np.float32)[:, 0, :]
    a_log = np.array(kw["a_log"], np.float32)
    dt_bias = np.array(kw["dt_bias"], np.float32)
    qsl = np.array(kw["query_start_loc"])
    indices = np.array(kw["state_indices"])
    seq_lens = np.array(kw["seq_lens"])
    num_tokens = qkv.shape[0]
    out = np.zeros((num_tokens, H * D), np.float32)
    for i, slot in enumerate(indices):
        start, end = int(qsl[i]), min(int(qsl[i + 1]), num_tokens)
        if end <= start:
            continue
        if seq_lens[i] > end - start:
            window, state = conv[slot], rec[slot]
        else:
            window, state = np.zeros_like(conv[slot]), np.zeros_like(rec[slot])
        for t in range(start, end):
            window, state, o = _kda_reference_token(
                window,
                state,
                qkv[t],
                a[t],
                b[t],
                conv_w,
                a_log,
                dt_bias,
                kw["gate_lower_bound"],
            )
            out[t] = o.reshape(-1)
        conv[slot], rec[slot] = window, state
    new_conv = jnp.asarray(conv.reshape(conv_shape), conv_state.dtype)
    new_rec = jnp.asarray(rec, rec_state.dtype)
    return (new_conv, new_rec), jnp.asarray(out, kw["qkv"].dtype)


class _Request:
    def __init__(self, rng: np.random.Generator, length: int) -> None:
        self.x = rng.standard_normal((length, DIM)).astype(np.float32)
        self.g1 = rng.standard_normal((length, PROJ)).astype(np.float32)
        self.beta = rng.standard_normal((length, H)).astype(np.float32)
        self.g2 = rng.standard_normal((length, PROJ)).astype(np.float32)
        self.done = 0
        self.window = np.zeros((K - 1, DIM), np.float32)
        self.state = np.zeros((H, D, D), np.float32)
        self.slot = 0


def _reference_outputs(layer, req: _Request, n: int) -> np.ndarray:
    conv_w = layer.conv1d.weight.detach().numpy()[:, 0, :]
    a_log = layer.A_log.detach().numpy()
    dt_bias = layer.dt_bias.detach().numpy().reshape(H, D)
    norm_w = layer.o_norm.weight.detach().float().numpy()
    outs = []
    for t in range(req.done, req.done + n):
        req.window, req.state, o = _kda_reference_token(
            req.window,
            req.state,
            req.x[t],
            req.g1[t],
            req.beta[t],
            conv_w,
            a_log,
            dt_bias,
            LB,
        )
        o = o * (1.0 / np.sqrt((o * o).mean(-1, keepdims=True) + EPS)) * norm_w
        outs.append(o / (1.0 + np.exp(-req.g2[t].reshape(H, D))))
    return np.stack(outs)


# Each step: (request id, scheduled tokens) in batch order, decodes first.
_SCHEDULE = (
    ((0, 12), (1, 7)),  # Two fresh prefills, the first chunked.
    ((1, 1), (0, 8), (2, 1)),  # A decode, the second chunk, a 1-token prompt.
    ((0, 1), (1, 1), (2, 1)),  # Pure decode.
    ((0, 1), (1, 1), (2, 1), (3, 5)),  # Decodes and a prompt in a reused slot.
    ((0, 1), (1, 1), (2, 1), (3, 1)),
)
_MAX_REQS = 8
# Must be a multiple of GDN v3's `triangular_block_size` (16 at <= 8 heads),
# matching vLLM's power-of-two token compile buckets (16, 32, 64, ...).
_PADDED_TOKENS = 32


@torch.no_grad()
def _run_schedule(monkeypatch, layer, rng, slots, *, check_reference=True):
    """Drive ``_forward`` through ``_SCHEDULE``; return the per-step outputs."""
    requests = [_Request(rng, 24) for _ in range(4)]
    for req, slot in zip(requests, slots, strict=True):
        req.slot = slot
    outputs = []
    for step in _SCHEDULE:
        lens = [n for _, n in step]
        qsl = np.zeros(_MAX_REQS + 1, np.int32)
        qsl[1 : len(step) + 1] = np.cumsum(lens)
        qsl[len(step) + 1 :] = qsl[len(step)]
        seq_lens = rng.integers(0, 10**6, _MAX_REQS).astype(np.int32)  # Stale.
        indices = np.zeros(_MAX_REQS, np.int32)
        num_decodes = 0
        x = rng.standard_normal((_PADDED_TOKENS, DIM)).astype(np.float32) * 50
        g1 = rng.standard_normal((_PADDED_TOKENS, PROJ)).astype(np.float32)
        beta = rng.standard_normal((_PADDED_TOKENS, H)).astype(np.float32)
        g2 = rng.standard_normal((_PADDED_TOKENS, PROJ)).astype(np.float32)
        for i, (rid, n) in enumerate(step):
            req = requests[rid]
            seq_lens[i] = req.done + n
            indices[i] = req.slot
            num_decodes += int(n == 1 and req.done > 0 and num_decodes == i)
            rows = slice(int(qsl[i]), int(qsl[i]) + n)
            x[rows] = req.x[req.done : req.done + n]
            g1[rows] = req.g1[req.done : req.done + n]
            beta[rows] = req.beta[req.done : req.done + n]
            g2[rows] = req.g2[req.done : req.done + n]
        metadata = _metadata(
            query_start_loc=torch.from_numpy(qsl),
            seq_lens=torch.from_numpy(seq_lens),
            request_distribution=torch.tensor(
                [num_decodes, num_decodes, len(step)], dtype=torch.int32
            ),
            mamba_state_indices=torch.from_numpy(indices),
        )
        _set_context(monkeypatch, metadata)
        core = torch.full((1, _PADDED_TOKENS, H, D), float("nan"))
        layer._forward(
            torch.from_numpy(x),
            torch.from_numpy(g1).reshape(1, -1, H, D),
            torch.from_numpy(g2).reshape(-1, H, D),
            torch.from_numpy(beta)[None],
            core,
        )
        outputs.append(core[0, : int(qsl[len(step)])].numpy().copy())
        for i, (rid, n) in enumerate(step):
            req = requests[rid]
            if check_reference:
                want = _reference_outputs(layer, req, n)
                got = core[0, int(qsl[i]) : int(qsl[i]) + n].numpy()
                np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
            req.done += n
        assert torch.isfinite(core[0, int(qsl[len(step)]) :]).all()
    return outputs, requests


def _cpu_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kda_attention_op.pallas, "jax_op", _FakeJaxOp)
    monkeypatch.setattr(pool_adapters, "gather_region", _fake_gather_region)
    monkeypatch.setattr(pool_adapters, "scatter_region", _fake_scatter_region)
    monkeypatch.setattr(
        kda_attention_op.gdn_wrapper, "fused_conv1d_gdn", _fake_fused_conv1d_gdn
    )
    monkeypatch.setattr(L, "is_conv_state_dim_first", lambda: False)


def _real_op_layer(monkeypatch, block_size: int):
    torch.manual_seed(0)
    layer = _make_layer(monkeypatch, vllm_config=_vllm_config(block_size))
    with torch.no_grad():
        layer.conv1d.weight.mul_(0.5)
        layer.o_norm.weight.copy_(1 + 0.1 * torch.randn(D))
    layer.fold_conv_weight()
    return layer


@pytest.mark.parametrize("split", [1, 2])
def test_pooled_batches_round_trip_state_through_the_pool(
    monkeypatch: pytest.MonkeyPatch, split: int
) -> None:
    """Chunked prefills, decodes and a reused slot over five steps match a
    per-request reference, with every state byte living in a 5-D RPA pool
    whose manager block is ``split`` pool blocks."""
    _cpu_kernels(monkeypatch)
    block_size = 16
    layer = _real_op_layer(monkeypatch, block_size)
    pool = torch.randn(12 * split, block_size // split, 2, 2, 128).to(torch.bfloat16)
    layer.kv_cache = [pool]
    slots = [3, 7, 5, 9]
    # Request 3 starts in a slot full of garbage; it must start from zeros.
    _run_schedule(monkeypatch, layer, np.random.default_rng(0), slots)

    # The state of the other blocks is untouched.
    untouched = [b for b in range(12) if b not in slots]
    ref = torch.randn(12 * split, block_size // split, 2, 2, 128)  # Unused seed.
    del ref
    by_block = pool.reshape(12, block_size, 2, 2, 128)
    assert torch.isfinite(by_block[untouched].float()).all()


def test_pooled_conv_region_holds_float32_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conv window is stored as fp32 bytes right after the SSM region,
    where ``glm53_kda_pool_state_layout`` says it is."""
    _cpu_kernels(monkeypatch)
    layer = _real_op_layer(monkeypatch, 16)
    pool = torch.zeros(12, 16, 2, 2, 128, dtype=torch.bfloat16)
    layer.kv_cache = [pool]
    _, requests = _run_schedule(
        monkeypatch, layer, np.random.default_rng(1), [3, 7, 5, 9]
    )
    token_bytes = 2 * 2 * 128 * 2
    layout = L.glm53_kda_pool_state_layout(
        conv_kernel_size=K, local_num_heads=H, head_dim=D, token_bytes=token_bytes
    )
    pool_j = _to_jax(pool)
    for req in requests:
        conv = _fake_gather_region(
            pool_j,
            jnp.array([req.slot], jnp.int32),
            tok0=layout.ssm_tokens,
            ntok=layout.conv_tokens,
            out_dtype=jnp.float32,
        ).reshape(-1)[: (K - 1) * DIM]
        np.testing.assert_allclose(
            np.asarray(conv).reshape(K - 1, DIM), req.window, rtol=0, atol=0
        )
        ssm = _fake_gather_region(
            pool_j,
            jnp.array([req.slot], jnp.int32),
            tok0=0,
            ntok=layout.ssm_tokens,
            out_dtype=jnp.float32,
        ).reshape(-1)[: H * D * D]
        np.testing.assert_allclose(
            np.asarray(ssm).reshape(H, D, D), req.state, rtol=1e-5, atol=1e-5
        )


def test_per_layer_caches_match_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dispatched op on per-layer caches produces the pooled outputs."""
    _cpu_kernels(monkeypatch)
    layer = _real_op_layer(monkeypatch, 16)
    layer.kv_cache = [torch.randn(12, 16, 2, 2, 128).to(torch.bfloat16)]
    pooled, _ = _run_schedule(
        monkeypatch, layer, np.random.default_rng(2), [3, 7, 5, 9]
    )
    layer.kv_cache = [torch.randn(12, K - 1, 1, DIM), torch.randn(12, H, D, D)]
    dispatched, requests = _run_schedule(
        monkeypatch, layer, np.random.default_rng(2), [3, 7, 5, 9]
    )
    for a, b in zip(pooled, dispatched, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    conv_cache, recurrent_cache = layer.kv_cache
    for req in requests:
        np.testing.assert_allclose(
            conv_cache[req.slot, :, 0].numpy(), req.window, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            recurrent_cache[req.slot].numpy(), req.state, rtol=1e-5, atol=1e-5
        )


# Real kernels.


@pytest.mark.tpu_test
def test_tpu_pooled_and_per_layer_kernels_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real pooled op (gather, fused kernel, scatter on a 5-D RPA pool
    with one token per row) and the real dispatched op produce the same
    outputs and states over chunked prefills and decodes."""
    monkeypatch.setattr(L, "is_conv_state_dim_first", lambda: False)
    device = torch.device("tpu")
    torch.manual_seed(0)
    heads, head_dim = 2, 128
    monkeypatch.setattr(
        sys.modules[__name__], "H", heads, raising=True
    )  # The reference helpers read the module geometry.
    monkeypatch.setattr(sys.modules[__name__], "D", head_dim)
    monkeypatch.setattr(sys.modules[__name__], "PROJ", heads * head_dim)
    monkeypatch.setattr(sys.modules[__name__], "DIM", 3 * heads * head_dim)

    def base_init(self, config, vllm_config, prefix=""):
        _base_init(False, 0)(self, config, vllm_config, prefix)
        self.head_dim = head_dim
        self.local_projection_size = heads * head_dim
        self.conv1d = _Conv(False)
        self.o_norm = _BaseNorm()
        self.dt_bias = nn.Parameter(torch.randn(heads * head_dim) * 0.3)

    monkeypatch.setattr(L.KimiGatedDeltaNetAttention, "__init__", base_init)
    token_bytes = 2 * 2 * 128 * 2
    layout = L.glm53_kda_pool_state_layout(
        conv_kernel_size=K,
        local_num_heads=heads,
        head_dim=head_dim,
        token_bytes=token_bytes,
    )
    block_size = 1 << (layout.required_tokens - 1).bit_length()
    layer = L.Glm53TpuKimiDeltaAttention(
        SimpleNamespace(), _vllm_config(block_size), PREFIX
    )
    with torch.no_grad():
        layer.conv1d.weight.mul_(0.5)
        layer.o_norm.weight.fill_(1.0)
    layer = layer.to(device)
    layer.fold_conv_weight()

    def schedule(kv_cache):
        layer.kv_cache = kv_cache
        rng = np.random.default_rng(0)
        outputs, _ = _run_schedule_on(monkeypatch, layer, rng, device)
        return outputs

    num_blocks = 12
    pooled = schedule(
        [
            torch.zeros(
                num_blocks, block_size, 2, 2, 128, dtype=torch.bfloat16, device=device
            )
        ]
    )
    dispatched = schedule(
        [
            torch.zeros(num_blocks, K - 1, 1, 3 * heads * head_dim, device=device),
            torch.zeros(num_blocks, heads, head_dim, head_dim, device=device),
        ]
    )
    for a, b in zip(pooled, dispatched, strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-2, atol=2e-2)


def _run_schedule_on(monkeypatch, layer, rng, device):
    """``_run_schedule`` without the CPU reference, on ``device``."""
    original_forward = layer._forward

    def forward_on_device(mixed_qkv, g1, g2, beta, core):
        out = torch.empty(core.shape, device=device)
        original_forward(
            mixed_qkv.to(device), g1.to(device), g2.to(device), beta.to(device), out
        )
        core.copy_(out.cpu())

    monkeypatch.setattr(layer, "_forward", forward_on_device)
    original_metadata = layer._metadata

    def metadata_on_device():
        m = original_metadata()
        if m is None:
            return None
        return SimpleNamespace(
            **{
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in vars(m).items()
            }
        )

    monkeypatch.setattr(layer, "_metadata", metadata_on_device)
    return _run_schedule(monkeypatch, layer, rng, [3, 7, 5, 9], check_reference=False)
