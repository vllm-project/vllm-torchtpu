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
"""CPU unit tests for `TPU_GDN_PREPACK_CONV_WEIGHT` and `conv_weight_transposed`."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn as nn
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs, weights_already_processed

import vllm_torchtpu.experimental.export as export_pkg
from vllm_torchtpu import envs
from vllm_torchtpu.experimental.export import exporter, weight_mapper
from vllm_torchtpu.kernels.gdn.head_geometry import derive_gdn_head_geometry
from vllm_torchtpu.kernels.gdn.v3 import wrapper
from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op
from vllm_torchtpu.layers.core import gdn_attention as core_gdn
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture(autouse=True)
def clear_envs_cache() -> Iterator[None]:
    cache_clear = getattr(envs.__getattr__, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    yield
    if callable(cache_clear):
        cache_clear()


@pytest.fixture
def static_tpu7x_info(monkeypatch: pytest.MonkeyPatch) -> pltpu.TPUInfo:
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return info


def test_env_default_and_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TPU_GDN_PREPACK_CONV_WEIGHT", raising=False)
    assert envs.TPU_GDN_PREPACK_CONV_WEIGHT is False

    monkeypatch.setenv("TPU_GDN_PREPACK_CONV_WEIGHT", "1")
    assert envs.TPU_GDN_PREPACK_CONV_WEIGHT is True

    monkeypatch.setenv("TPU_GDN_PREPACK_CONV_WEIGHT", "invalid")
    with pytest.raises(ValueError, match="TPU_GDN_PREPACK_CONV_WEIGHT"):
        _ = envs.TPU_GDN_PREPACK_CONV_WEIGHT


def test_wrapper_skips_transpose_and_cast_when_prepacked(
    static_tpu7x_info: pltpu.TPUInfo,
) -> None:
    del static_tpu7x_info
    n_kq, n_v, d_k, d_v, kernel_size = 2, 8, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    num_seqs = 4
    batch_size = 4
    fn = wrapper.fused_conv1d_gdn.__wrapped__

    def _trace(conv_weight: jax.Array, conv_bias: jax.Array, transposed: bool) -> str:
        return str(
            jax.make_jaxpr(
                lambda: fn(
                    qkv=jnp.zeros((batch_size, dim), jnp.bfloat16),
                    b=jnp.zeros((batch_size, n_v), jnp.bfloat16),
                    a=jnp.zeros((batch_size, n_v), jnp.bfloat16),
                    states=(
                        jnp.zeros((num_seqs + 1, kernel_size - 1, dim), jnp.float32),
                        jnp.zeros((num_seqs + 1, n_v, d_k, d_v), jnp.bfloat16),
                    ),
                    conv_weight=conv_weight,
                    conv_bias=conv_bias,
                    a_log=jnp.zeros((n_v,), jnp.float32),
                    dt_bias=jnp.zeros((n_v,), jnp.float32),
                    query_start_loc=jnp.arange(num_seqs + 1, dtype=jnp.int32),
                    state_indices=jnp.arange(1, num_seqs + 1, dtype=jnp.int32),
                    distribution=jnp.asarray(
                        [num_seqs, num_seqs, num_seqs], dtype=jnp.int32
                    ),
                    seq_lens=jnp.full((num_seqs,), 32, dtype=jnp.int32),
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=kernel_size,
                    conv_weight_transposed=transposed,
                )
            )()
        )

    default_jaxpr = _trace(
        jnp.zeros((dim, 1, kernel_size), jnp.bfloat16),
        jnp.zeros((dim,), jnp.bfloat16),
        transposed=False,
    )
    assert "transpose[permutation=(2, 1, 0)]" in default_jaxpr

    prepacked_jaxpr = _trace(
        jnp.zeros((kernel_size, 1, dim), jnp.float32),
        jnp.zeros((dim,), jnp.float32),
        transposed=True,
    )
    assert "transpose[permutation=(2, 1, 0)]" not in prepacked_jaxpr
    assert (
        default_jaxpr.count("convert_element_type")
        - prepacked_jaxpr.count("convert_element_type")
        == 2
    )


def test_wrapper_rejects_mismatched_conv_weight_shape_or_dtype(
    static_tpu7x_info: pltpu.TPUInfo,
) -> None:
    del static_tpu7x_info
    n_kq, n_v, d_k, d_v, kernel_size = 2, 8, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    fn = wrapper.fused_conv1d_gdn.__wrapped__

    def _call(conv_weight: jax.Array, conv_bias: jax.Array | None = None) -> None:
        fn(
            qkv=jnp.zeros((4, dim), jnp.bfloat16),
            b=jnp.zeros((4, n_v), jnp.bfloat16),
            a=jnp.zeros((4, n_v), jnp.bfloat16),
            states=(
                jnp.zeros((5, kernel_size - 1, dim), jnp.float32),
                jnp.zeros((5, n_v, d_k, d_v), jnp.bfloat16),
            ),
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=jnp.zeros((n_v,), jnp.float32),
            dt_bias=jnp.zeros((n_v,), jnp.float32),
            query_start_loc=jnp.arange(5, dtype=jnp.int32),
            state_indices=jnp.arange(1, 5, dtype=jnp.int32),
            distribution=jnp.asarray([4, 4, 4], dtype=jnp.int32),
            seq_lens=jnp.full((4,), 16, dtype=jnp.int32),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_weight_transposed=True,
        )

    with pytest.raises(AssertionError, match="conv_weight_transposed=True"):
        _call(jnp.zeros((dim, 1, kernel_size), jnp.float32))

    with pytest.raises(AssertionError, match="prepacked conv_weight dtype"):
        _call(jnp.zeros((kernel_size, 1, dim), jnp.bfloat16))

    with pytest.raises(AssertionError, match="prepacked conv_bias dtype"):
        _call(
            jnp.zeros((kernel_size, 1, dim), jnp.float32),
            jnp.zeros((dim,), jnp.bfloat16),
        )


def test_weight_mapper_conv1d_transposed_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ColumnParallelConv1d(nn.Module):
        def __init__(self, shape: tuple[int, int, int], tp_size: int) -> None:
            super().__init__()
            self.tp_size = tp_size
            self.weight = nn.Parameter(torch.zeros(shape), requires_grad=False)
            self.weight.output_dim = 0  # type: ignore[attr-defined]
            self.weight.input_dim = 1  # type: ignore[attr-defined]

    class _ToyModel(nn.Module):
        def __init__(self, tp_size: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].linear_attn = nn.Module()
            self.layers[0].linear_attn.conv1d = _ColumnParallelConv1d(
                (4, 1, 1536), tp_size=tp_size
            )
            setattr(
                self.layers[0].linear_attn.conv1d,
                weight_mapper.CONV_WEIGHT_TRANSPOSED_ATTR,
                True,
            )
            self.pcp_layer = nn.Module()
            self.pcp_layer.conv1d = _ColumnParallelConv1d((1536, 1, 4), tp_size=tp_size)

    model_tp4 = _ToyModel(tp_size=4)
    assert weight_mapper.infer_weight_sharding(
        "model.layers.0.linear_attn.conv1d.weight",
        (4, 1, 1536),
        model=model_tp4,
        tp_size_hint=4,
    ) == [None, None, "tp"]
    assert weight_mapper.infer_weight_sharding(
        "model.pcp_layer.conv1d.weight",
        (1536, 1, 4),
        model=model_tp4,
        tp_size_hint=4,
    ) == ["tp", None, None]

    model_tp1 = _ToyModel(tp_size=1)
    assert weight_mapper.infer_weight_sharding(
        "model.layers.0.linear_attn.conv1d.weight",
        (4, 1, 1536),
        model=model_tp1,
        tp_size_hint=1,
    ) == [None, None, None]

    monkeypatch.setenv("TPU_GDN_PREPACK_CONV_WEIGHT", "1")
    with pytest.raises(NotImplementedError, match="TPU_GDN_PREPACK_CONV_WEIGHT"):
        weight_mapper.infer_weight_sharding(
            "model.layers.0.linear_attn.conv1d.weight",
            (4, 1, 1536),
            model=None,
            tp_size_hint=4,
        )

    with pytest.raises(ValueError, match="Could not resolve submodule"):
        weight_mapper.infer_weight_sharding(
            "model.missing_layer.conv1d.weight",
            (4, 1, 1536),
            model=model_tp4,
            tp_size_hint=4,
        )

    layouts = exporter._tensor_layouts(
        model_tp4,  # type: ignore[arg-type]
        {
            "model.layers.0.linear_attn.conv1d.weight": export_pkg.ShapeDtypeStruct(
                (4, 1, 1536), "float32"
            )
        },
        frozenset(),
        {"model": 4},
    )
    assert layouts["model.layers.0.linear_attn.conv1d.weight"].sharding == [
        None,
        None,
        "model",
    ]
    assert layouts["model.layers.0.linear_attn.conv1d.weight"].shape == [4, 1, 6144]


def test_experimental_export_lazy_getattr_and_dir() -> None:
    exported_names = dir(export_pkg)
    for name in export_pkg.__all__:
        assert name in exported_names
    resolved_export = export_pkg.export
    assert export_pkg.__dict__["export"] is resolved_export
    assert export_pkg.export is resolved_export
    with pytest.raises(AttributeError, match="no_such_export_attr"):
        _ = export_pkg.no_such_export_attr


def _make_stub_gdn_attn(
    *,
    conv_dim: int = 16,
    kernel_size: int = 4,
    with_bias: bool = True,
) -> tuple[gdn_attention_op.VllmGatedDeltaNetAttention, nn.Module]:
    class _Conv1dModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = UnquantizedLinearMethod()
            weight = nn.Parameter(
                torch.arange(conv_dim * kernel_size, dtype=torch.bfloat16).reshape(
                    conv_dim, 1, kernel_size
                ),
                requires_grad=False,
            )
            set_weight_attrs(weight, {"weight_loader": default_weight_loader})
            self.register_parameter("weight", weight)
            if with_bias:
                bias = nn.Parameter(
                    torch.arange(conv_dim, dtype=torch.bfloat16),
                    requires_grad=False,
                )
                set_weight_attrs(bias, {"weight_loader": default_weight_loader})
                self.register_parameter("bias", bias)
            else:
                self.register_parameter("bias", None)

    conv1d = _Conv1dModule()
    attn = gdn_attention_op.VllmGatedDeltaNetAttention.__new__(
        gdn_attention_op.VllmGatedDeltaNetAttention
    )
    nn.Module.__init__(attn)
    attn.prefix = "model.layers.0.linear_attn"
    attn.tp_size = 1
    attn.num_k_heads = 2
    attn.num_v_heads = 4
    attn.head_k_dim = 2
    attn.head_v_dim = 2
    attn.conv_kernel_size = kernel_size
    attn.num_spec = 0
    attn._bm_kernel = False
    attn.conv1d = conv1d  # type: ignore[assignment]
    attn.get_state_dtype = lambda: (torch.float32, torch.bfloat16)  # type: ignore[method-assign]
    return attn, conv1d


def _dummy_vllm_config() -> VllmConfig:
    model_cfg = MagicMock(spec=ModelConfig)
    cache_cfg = MagicMock(spec=CacheConfig)
    cache_cfg.block_size = 16
    parallel_cfg = MagicMock(spec=ParallelConfig)
    parallel_cfg.prefill_context_parallel_size = 1
    vllm_cfg = MagicMock(spec=VllmConfig)
    vllm_cfg.model_config = model_cfg
    vllm_cfg.cache_config = cache_cfg
    vllm_cfg.parallel_config = parallel_cfg
    return vllm_cfg


def test_adapter_prepack_pcp_warning_and_op_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TPU_GDN_PREPACK_CONV_WEIGHT", "1")
    mesh = Mesh(np.array(jax.devices()[:1]), ("model",))
    vllm_cfg = _dummy_vllm_config()

    attn_pcp, _ = _make_stub_gdn_attn()
    with patch.object(gdn_attention_op.logger, "warning_once") as mock_warn:
        attn_pcp._configure_conv_weight_prepack(pcp_enabled=True)
        assert attn_pcp._prepack_conv_weight is False
        mock_warn.assert_called_once()

    attn, conv1d = _make_stub_gdn_attn(conv_dim=4, kernel_size=4, with_bias=True)
    expected_transposed = conv1d.weight.detach().transpose(0, 2).to(torch.float32)

    recorded_op_names: list[str] = []
    recorded_transposed_flags: list[bool] = []

    def _fake_jax_op(op_name: str, fn: object, **kwargs: object) -> MagicMock:
        del kwargs
        recorded_op_names.append(op_name)
        keywords = getattr(fn, "keywords", None)
        if isinstance(keywords, dict) and "conv_weight_transposed" in keywords:
            recorded_transposed_flags.append(bool(keywords["conv_weight_transposed"]))
        op = MagicMock()
        return op

    gdn_attention_op._shared_gdn_ops.clear()
    with (
        set_vllm_model_wrapper_context(mesh=mesh, vllm_config=vllm_cfg),
        patch.object(gdn_attention_op.pallas, "jax_op", side_effect=_fake_jax_op),
    ):
        attn._configure_conv_weight_prepack(pcp_enabled=False)
        assert attn._prepack_conv_weight is True
        attn.gdn_op = attn._build_gdn_op(conv_weight_transposed=False)
        attn.gdn_pooled_op = attn._build_pooled_gdn_op(
            block_major=False, conv_weight_transposed=False
        )
        assert recorded_op_names[0] == (
            "pallas::gdn_attention_model_layers_0_linear_attn"
        )
        assert recorded_transposed_flags == [False]
        assert len(gdn_attention_op._shared_gdn_ops) == 1

        # Prepack and verify square (4, 1, 4) idempotence on repeated calls.
        attn.process_weights_after_loading(torch.bfloat16)
        attn.process_weights_after_loading(torch.bfloat16)
        assert conv1d.weight.shape == (4, 1, 4)
        assert conv1d.weight.dtype == torch.float32
        assert torch.equal(conv1d.weight, expected_transposed)
        assert conv1d.bias.dtype == torch.float32
        assert getattr(conv1d, weight_mapper.CONV_WEIGHT_TRANSPOSED_ATTR) is True
        assert (
            "pallas::gdn_attention_prepack_model_layers_0_linear_attn"
            in recorded_op_names
        )
        assert recorded_transposed_flags[-1] is True
        # Pooled op cache built a distinct entry for conv_weight_transposed=True.
        assert len(gdn_attention_op._shared_gdn_ops) == 2

    # Mock conv1d without bias attribute (SimpleNamespace) also prepacks cleanly.
    mock_conv1d = SimpleNamespace(
        weight=torch.ones((8, 1, 4), dtype=torch.bfloat16),
    )
    gdn_attention_op._prepack_conv1d_weights(mock_conv1d)
    assert mock_conv1d.weight.shape == (4, 1, 8)
    assert mock_conv1d.weight.dtype == torch.float32


def test_adapter_prepack_ipc_weights_already_processed_and_layerwise_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TPU_GDN_PREPACK_CONV_WEIGHT", "1")
    mesh = Mesh(np.array(jax.devices()[:1]), ("model",))
    vllm_cfg = _dummy_vllm_config()

    # 1. IPC weight cache path: weights arrive already transposed in float32.
    attn_ipc, conv_ipc = _make_stub_gdn_attn(
        conv_dim=12, kernel_size=4, with_bias=False
    )
    prepacked_ipc_weight = torch.randn((4, 1, 12), dtype=torch.float32)
    conv_ipc.weight = nn.Parameter(prepacked_ipc_weight.clone(), requires_grad=False)
    with (
        set_vllm_model_wrapper_context(mesh=mesh, vllm_config=vllm_cfg),
        patch.object(gdn_attention_op.pallas, "jax_op", return_value=MagicMock()),
        weights_already_processed(),
    ):
        attn_ipc._configure_conv_weight_prepack(pcp_enabled=False)
        attn_ipc.process_weights_after_loading(torch.bfloat16)
    assert conv_ipc.weight.shape == (4, 1, 12)
    assert conv_ipc.weight.dtype == torch.float32
    assert torch.equal(conv_ipc.weight, prepacked_ipc_weight)
    assert getattr(conv_ipc, weight_mapper.CONV_WEIGHT_TRANSPOSED_ATTR) is True

    # 2. Layerwise reload path: construction metadata is (dim, 1, K) bf16,
    # post-load kernel tensor is (K, 1, dim) fp32, and reload re-prepacks.
    class _ParentModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn, self.conv1d = _make_stub_gdn_attn(
                conv_dim=8, kernel_size=4, with_bias=False
            )

    model = _ParentModel()
    with (
        set_vllm_model_wrapper_context(mesh=mesh, vllm_config=vllm_cfg),
        patch.object(gdn_attention_op.pallas, "jax_op", return_value=MagicMock()),
    ):
        model.attn._configure_conv_weight_prepack(pcp_enabled=False)
        record_metadata_for_reloading(model)
        model.conv1d.quant_method.process_weights_after_loading(model.conv1d)
        model.attn.process_weights_after_loading(torch.bfloat16)

    orig_ptr = model.conv1d.weight.data_ptr()
    assert model.conv1d.weight.shape == (4, 1, 8)
    assert model.conv1d.weight.dtype == torch.float32

    initialize_layerwise_reload(model)
    reloaded_untransposed = (
        torch.arange(32, dtype=torch.bfloat16).reshape(8, 1, 4) + 100
    )
    model.conv1d.weight.weight_loader(model.conv1d.weight, reloaded_untransposed)

    assert model.conv1d.weight.shape == (4, 1, 8)
    assert model.conv1d.weight.dtype == torch.float32
    assert model.conv1d.weight.data_ptr() == orig_ptr
    assert torch.equal(
        model.conv1d.weight,
        reloaded_untransposed.transpose(0, 2).to(torch.float32),
    )


def test_core_gdn_attention_in_specs_with_prepacked_weight() -> None:
    mesh = Mesh(np.array(jax.devices()[:1]), ("model",))
    captured_specs: list[tuple] = []

    def _fake_shard_map(fn: object, *, mesh: object, in_specs: tuple, **kwargs: object):
        del fn, mesh, kwargs
        captured_specs.append(in_specs)
        return lambda *args: (
            (args[3][0], args[3][1]) if isinstance(args[3], tuple) else args[3],
            jnp.zeros((4, 8), dtype=jnp.bfloat16),
        )

    with patch.object(core_gdn.jax, "shard_map", side_effect=_fake_shard_map):
        core_gdn.run_jax_gdn_attention(
            jnp.zeros((4, 24), jnp.bfloat16),
            jnp.zeros((4, 4), jnp.bfloat16),
            jnp.zeros((4, 4), jnp.bfloat16),
            jnp.zeros((5, 3, 24), jnp.float32),
            jnp.zeros((5, 4, 2, 2), jnp.bfloat16),
            jnp.zeros((4, 1, 24), jnp.float32),
            None,
            jnp.zeros((4,), jnp.float32),
            jnp.zeros((4,), jnp.float32),
            jnp.arange(1, 5, dtype=jnp.int32),
            jnp.arange(5, dtype=jnp.int32),
            jnp.asarray([4, 4, 4], dtype=jnp.int32),
            jnp.full((4,), 16, dtype=jnp.int32),
            n_kq=2,
            n_v=4,
            d_k=2,
            d_v=2,
            kernel_size=4,
            mesh=mesh,
            conv_weight_transposed=True,
        )
        core_gdn.run_jax_gdn_attention_pooled(
            jnp.zeros((4, 24), jnp.bfloat16),
            jnp.zeros((4, 4), jnp.bfloat16),
            jnp.zeros((4, 4), jnp.bfloat16),
            jnp.zeros((8, 16, 4, 128), jnp.bfloat16),
            jnp.zeros((4, 1, 24), jnp.float32),
            None,
            jnp.zeros((4,), jnp.float32),
            jnp.zeros((4,), jnp.float32),
            jnp.arange(1, 5, dtype=jnp.int32),
            jnp.arange(5, dtype=jnp.int32),
            jnp.asarray([4, 4, 4], dtype=jnp.int32),
            jnp.full((4,), 16, dtype=jnp.int32),
            n_kq=2,
            n_v=4,
            d_k=2,
            d_v=2,
            kernel_size=4,
            pool_block_tokens=16,
            mesh=mesh,
            conv_weight_transposed=True,
        )

    assert len(captured_specs) == 2
    assert captured_specs[0][4] == PartitionSpec(None, None, "model")
    assert captured_specs[1][4] == PartitionSpec(None, None, "model")
    # Unused import guard: derive_gdn_head_geometry remains valid for TP=1.
    assert derive_gdn_head_geometry(2, 4, 1).local_num_v_heads == 4
