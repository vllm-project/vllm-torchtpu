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

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parameter import BasevLLMParameter

from vllm_torchtpu.models.vllm.qwen4_exp.pd_identity import LoadedWeightIdentity


@pytest.fixture
def tp_rank_one(monkeypatch):
    for module in ("parameter", "layers.linear", "layers.vocab_parallel_embedding"):
        monkeypatch.setattr(
            f"vllm.model_executor.{module}.get_tensor_model_parallel_rank", lambda: 1
        )
        monkeypatch.setattr(
            f"vllm.model_executor.{module}.get_tensor_model_parallel_world_size",
            lambda: 2,
        )
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


def expected_digest(name, views):
    identity = LoadedWeightIdentity()
    for index, view in enumerate(views):
        identity.add(f"{name}/{index}", view)
    return identity.finish()


@pytest.mark.usefixtures("tp_rank_one")
@pytest.mark.parametrize("column", [False, True])
def test_public_parallel_loaders_hash_only_owned_copies(column):
    cls = ColumnParallelLinear if column else RowParallelLinear
    layer = cls(8, 12, bias=False, quant_config=None, params_dtype=torch.bfloat16)
    source = torch.arange(96).reshape(12, 8).to(torch.bfloat16)
    before = source.clone()
    axis = 0 if column else 1
    local = source.narrow(axis, source.shape[axis] // 2, source.shape[axis] // 2)
    identity = LoadedWeightIdentity(layer.parameters())
    with identity.observe("weight"):
        scratch = torch.empty_like(source)
        scratch.copy_(source + 10)
        layer.weight_loader(layer.get_parameter("weight"), source)
    assert identity.finish() == expected_digest("weight", [local])
    torch.testing.assert_close(layer.weight, local, rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    changed = source.clone()
    changed.narrow(axis, 0, source.shape[axis] // 2).add_(100)
    repeated = LoadedWeightIdentity(layer.parameters())
    with repeated.observe("weight"):
        layer.weight_loader(layer.get_parameter("weight"), changed)
    assert repeated.finish() == identity.finish()
    scratch.copy_(source)
    assert repeated.finish() == identity.finish()


@pytest.mark.usefixtures("tp_rank_one")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interleaved", [False, True])
def test_actual_gdn_merged_loader_observes_each_owned_projection(dtype, interleaved):
    gdn = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(gdn)
    gdn.gqa_interleaved_layout = interleaved
    gdn.num_k_heads, gdn.num_v_heads = (1 if interleaved else 2), 4
    gdn.head_k_dim, gdn.head_v_dim, gdn.tp_size = 2, 2, 2
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_default_torch_dtype(dtype):
        layer = gdn.create_qkvz_proj(
            8, gdn.num_k_heads * 2, 8, None, "gdn.in_proj_qkvz"
        )
    rows = 20 if interleaved else 16
    source = torch.arange(rows * 8).reshape(rows, 8).to(dtype)
    identity = LoadedWeightIdentity(layer.parameters())
    parameter = layer.get_parameter("weight")
    assert isinstance(parameter, BasevLLMParameter)
    with identity.observe("in_proj_qkv"):
        layer.weight_loader(parameter, source, None if interleaved else (0, 1, 2))
    if interleaved:
        expected = torch.cat([source[0:2], source[2:4], source[8:12], source[16:20]])
        assert identity.finish() == expected_digest("in_proj_qkv", [expected])
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    else:
        expected_pieces = [source[2:4], source[6:8], source[12:16]]
        assert identity.finish() == expected_digest("in_proj_qkv", expected_pieces)
        torch.testing.assert_close(
            parameter[:8], torch.cat(expected_pieces), rtol=0, atol=0
        )


@pytest.mark.usefixtures("tp_rank_one")
def test_vocab_loader_hashes_valid_owned_rows_without_padding():
    layer = VocabParallelEmbedding(10, 4, padding_size=4, params_dtype=torch.bfloat16)
    source = torch.arange(40).reshape(10, 4).to(torch.bfloat16)
    identity = LoadedWeightIdentity(layer.parameters())
    with identity.observe("embedding"):
        layer.weight_loader(layer.get_parameter("weight"), source)
    start = layer.shard_indices.org_vocab_start_index
    stop = layer.shard_indices.org_vocab_end_index
    assert (start, stop) == (6, 10)
    assert identity.finish() == expected_digest("embedding", [source[start:stop]])
    torch.testing.assert_close(
        layer.get_parameter("weight")[:4], source[6:10], rtol=0, atol=0
    )


def test_strided_hash_uses_bounded_logical_copies(monkeypatch):
    source = torch.arange(2048 * 2048, dtype=torch.float32).reshape(2048, 2048)[:, ::2]
    expected = LoadedWeightIdentity()
    expected.add("slice", source.contiguous())
    contiguous = torch.Tensor.contiguous
    sizes = []

    def bounded(view, *args, **kwargs):
        sizes.append(view.numel() * view.element_size())
        return contiguous(view, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", bounded)
    actual = LoadedWeightIdentity()
    actual.add("slice", source)
    assert actual.finish() == expected.finish()
    assert sizes and max(sizes) <= 4 * 1024**2


def test_observer_exits_after_loader_failure():
    target = torch.zeros(4)
    identity = LoadedWeightIdentity([target])
    with pytest.raises(RuntimeError, match="load failed"), identity.observe("weight"):
        target.copy_(torch.ones(4))
        raise RuntimeError("load failed")
    before = identity.finish()
    target.copy_(torch.full((4,), 9.0))
    assert identity.finish() == before
