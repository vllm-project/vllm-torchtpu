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
"""Tests for the fence that keeps the attention-output relayout in bf16."""

import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental import pallas as pl
from jax.experimental import topologies
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear

from vllm_torchtpu import model_loader_patches
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.layers.adapter.custom_ops.optimization_barrier_op import (
    optimization_barrier_op,
)
from vllm_torchtpu.layers.adapter.linear_common import _get_quantized_matmul_op
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    ATTN_OUT_FENCE_OP_ATTR,
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
    tag_attn_out_relayout_projections,
)

_HEADS, _KV_HEADS, _HEAD_DIM = 4, 2, 128
_K, _N = _HEADS * _HEAD_DIM, 256
# vLLM rejects a second attention layer with the same name.
_LAYER_IDS = itertools.count()


@pytest.fixture(autouse=True)
def single_rank_tp_group(monkeypatch):
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(parallel_state, "_TP", group)


@pytest.fixture
def cpu_vllm_config():
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


def _linear(cls, *args, quantized: bool = True):
    quant = (
        VllmFp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic")
        if quantized
        else None
    )
    return cls(
        *args,
        bias=False,
        quant_config=quant,
        params_dtype=torch.bfloat16,
    )


def _o_proj(quantized: bool = True) -> RowParallelLinear:
    return _linear(RowParallelLinear, _K, _N, quantized=quantized)


class _AttentionBlock(torch.nn.Module):
    """An attention block whose attribute names match no model's names."""

    def __init__(self, *, qkv_heads: int | None = _HEADS, quantized: bool = True):
        super().__init__()
        self.core = Attention(
            _HEADS,
            _HEAD_DIM,
            _HEAD_DIM**-0.5,
            num_kv_heads=_KV_HEADS,
            prefix=f"layers.{next(_LAYER_IDS)}.attn",
        )
        self.out = _o_proj(quantized)
        if qkv_heads is not None:
            self.inp = _linear(QKVParallelLinear, _K, _HEAD_DIM, qkv_heads, _KV_HEADS)


class _MlpBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out = _o_proj()


class _TwoProjectionBlock(_AttentionBlock):
    def __init__(self):
        super().__init__()
        self.second_out = _o_proj()


def _set_chip(monkeypatch, chip: pltpu.ChipVersion) -> None:
    monkeypatch.setattr(fp8, "get_chip_version", lambda: chip)


def test_tag_finds_attention_o_proj_by_type(monkeypatch, cpu_vllm_config):
    _set_chip(monkeypatch, pltpu.ChipVersion.TPU_7X)
    plain = _AttentionBlock()
    no_qkv = _AttentionBlock(qkv_heads=None)
    gated = _AttentionBlock(qkv_heads=2 * _HEADS)
    unquantized = _AttentionBlock(quantized=False)
    with monkeypatch.context() as m:
        m.setenv("REQUANTIZE_WEIGHT_DTYPE", "int8")
        int8_requant = _AttentionBlock()
        assert isinstance(int8_requant.out.quant_method, VllmFp8LinearMethodTPU)
        int8_requant.out.quant_method.process_weights_after_loading(int8_requant.out)
    assert int8_requant.out.weight.dtype is torch.int8
    mlp = _MlpBlock()
    two_projections = _TwoProjectionBlock()
    untagged = (gated, unquantized, int8_requant, mlp, two_projections)
    model = torch.nn.ModuleList([plain, no_qkv, *untagged])
    assert isinstance(plain.out.quant_method, VllmFp8LinearMethodTPU)

    assert tag_attn_out_relayout_projections(model) == 2
    for block in (plain, no_qkv):
        assert getattr(block.out, ATTN_OUT_FENCE_OP_ATTR) is optimization_barrier_op()
    for block in untagged:
        assert not hasattr(block.out, ATTN_OUT_FENCE_OP_ATTR)
    assert not hasattr(two_projections.second_out, ATTN_OUT_FENCE_OP_ATTR)
    assert not hasattr(plain.inp, ATTN_OUT_FENCE_OP_ATTR)

    # A second pass clears the tag of a layer that no longer qualifies.
    plain.out.quant_method = unquantized.out.quant_method
    assert tag_attn_out_relayout_projections(model) == 1
    assert not hasattr(plain.out, ATTN_OUT_FENCE_OP_ATTR)


@pytest.mark.parametrize(
    ("chip", "expected"),
    [
        (pltpu.ChipVersion.TPU_7X, 1),
        (pltpu.ChipVersion.TPU_V6E, 1),
        (pltpu.ChipVersion.TPU_V5E, 0),
    ],
)
def test_tag_only_on_measured_chips(monkeypatch, cpu_vllm_config, chip, expected):
    block = _AttentionBlock()
    _set_chip(monkeypatch, pltpu.ChipVersion.TPU_7X)
    assert tag_attn_out_relayout_projections(block) == 1
    _set_chip(monkeypatch, chip)
    assert tag_attn_out_relayout_projections(block) == expected
    assert hasattr(block.out, ATTN_OUT_FENCE_OP_ATTR) is bool(expected)


def test_patch_tags_after_load_and_logs_the_count(monkeypatch, cpu_vllm_config):
    from vllm.model_executor.model_loader import base_loader

    _set_chip(monkeypatch, pltpu.ChipVersion.TPU_7X)
    logged: list[tuple] = []
    monkeypatch.setattr(
        model_loader_patches.logger, "info", lambda *args: logged.append(args)
    )
    monkeypatch.setattr(base_loader, "process_weights_after_loading", lambda *a: "ok")
    model_loader_patches.patch_attn_out_relayout_tags()
    wrapped = base_loader.process_weights_after_loading
    model_loader_patches.patch_attn_out_relayout_tags()
    assert base_loader.process_weights_after_loading is wrapped

    block = _AttentionBlock()
    assert wrapped(block, None, None) == "ok"
    assert hasattr(block.out, ATTN_OUT_FENCE_OP_ATTR)
    assert [args[1:] for args in logged] == [(1,)]
    assert wrapped(_MlpBlock(), None, None) == "ok"
    assert [args[1:] for args in logged] == [(1,), (0,)]


def _meta_o_proj() -> RowParallelLinear:
    layer = _o_proj()
    assert isinstance(layer.quant_method, VllmFp8LinearMethodTPU)
    layer.quant_method.process_weights_after_loading(layer)
    return layer.to("meta")


@pytest.mark.parametrize("tagged", [False, True])
def test_apply_fences_only_a_tagged_layer(monkeypatch, cpu_vllm_config, tagged):
    calls: list[str] = []
    layer = _meta_o_proj()
    if tagged:
        setattr(layer, ATTN_OUT_FENCE_OP_ATTR, lambda x: (calls.append("fence"), x)[1])
    monkeypatch.setattr(
        fp8,
        "quantized_matmul",
        lambda x, w, _s, w8a16_max_rows: (
            calls.append("matmul"),
            torch.empty((x.shape[0], w.shape[-1]), dtype=x.dtype, device="meta"),
        )[1],
    )
    assert isinstance(layer.quant_method, VllmFp8LinearMethodTPU)
    x = torch.empty((64, _K), dtype=torch.bfloat16, device="meta")
    assert layer.quant_method.apply(layer, x).shape == (64, _N)
    assert calls == (["fence", "matmul"] if tagged else ["matmul"])


def test_tagged_apply_traces_through_the_real_ops(cpu_vllm_config):
    layer = _meta_o_proj()
    setattr(layer, ATTN_OUT_FENCE_OP_ATTR, optimization_barrier_op())
    # Built before tracing, as the runner does: Dynamo cannot trace its lock.
    _get_quantized_matmul_op()
    assert isinstance(layer.quant_method, VllmFp8LinearMethodTPU)
    compiled = torch.compile(
        lambda x: layer.quant_method.apply(layer, x),
        backend="eager",
        fullgraph=True,
    )
    out = compiled(torch.empty((64, _K), dtype=torch.bfloat16, device="meta"))
    assert out.shape == (64, _N) and out.dtype == torch.bfloat16


def test_fence_removes_the_f32_relayout_copy_in_tpu_hlo():
    topo = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x=2x2x1")
    rep = NamedSharding(Mesh(topo.devices[:1], ("x",)), P())
    spec = pl.BlockSpec((1, 128, 4, 2, 512), lambda h, m: (h, m, 0, 0, 0))
    rpa_out = pl.pallas_call(
        lambda q_ref, o_ref: o_ref.__setitem__(Ellipsis, q_ref[...]),
        out_shape=jax.ShapeDtypeStruct((2, 16384, 4, 2, 512), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            0, grid=(2, 128), in_specs=[spec], out_specs=spec
        ),
    )
    args = (
        jax.ShapeDtypeStruct((2, 16384, 4, 2, 512), jnp.bfloat16, sharding=rep),
        jax.ShapeDtypeStruct((8192, 5376), jnp.float8_e4m3fn, sharding=rep),
        jax.ShapeDtypeStruct((5376,), jnp.float32, sharding=rep),
    )

    def f32_copies(fenced: bool, gated: bool = False) -> int:
        def fn(q, w, s):
            x = rpa_out(q).swapaxes(0, 1).reshape(16384, 8192)
            if gated:
                x = x * jax.nn.sigmoid(x)
            if fenced:
                # The body of the `pallas::optimization_barrier` op.
                x = jax.lax.optimization_barrier(x)
            return xla_quantized_matmul(x, w, s)

        hlo = jax.jit(fn).trace(*args).lower().compile().as_text()
        return sum(
            " copy(" in line and "f32[" in line.split(" copy(")[0]
            for line in hlo.splitlines()
        )

    assert f32_copies(fenced=False) == 1
    assert f32_copies(fenced=True) == 0
    # A gated producer has no `f32` relayout, so it is left unfenced.
    assert f32_copies(fenced=False, gated=True) == 0


@pytest.mark.tpu_test
@pytest.mark.parametrize("rows", [256, 8192])
def test_tagged_apply_matches_untagged_bytes_on_tpu(rows):
    device = torch.device("tpu:0")
    with set_current_vllm_config(VllmConfig()):
        block = _AttentionBlock()
    generator = torch.Generator().manual_seed(rows)
    layer = block.out
    layer.weight.weight_loader(
        layer.weight,
        (torch.randn((_N, _K), generator=generator) * 0.05).to(torch.float8_e4m3fn),
    )
    layer.weight_scale.data.fill_(1.0)
    layer.quant_method.process_weights_after_loading(layer)
    layer.to(device)
    _get_quantized_matmul_op()
    head_major = (
        (torch.randn((_KV_HEADS, rows, _K // _KV_HEADS), generator=generator) * 3.0)
        .to(torch.bfloat16)
        .to(device)
    )

    def run() -> np.ndarray:
        compiled = torch.compile(
            lambda x: layer.quant_method.apply(
                layer, x.transpose(0, 1).reshape(rows, _K)
            ),
            backend="tpu",
            fullgraph=True,
        )
        return compiled(head_major).cpu().view(torch.int16).numpy()

    untagged = run()
    setattr(layer, ATTN_OUT_FENCE_OP_ATTR, optimization_barrier_op())
    tagged = run()
    assert untagged.shape == (rows, _N) and np.any(untagged)
    np.testing.assert_array_equal(tagged, untagged)
