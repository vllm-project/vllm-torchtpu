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
"""The default-off hook through which a drafter supplies the attention of
its DCP-sharded KV layers (`DraftDcpAttention`)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasAttentionBackendImpl,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.draft_dcp_attention import DraftDcpAttentionProvider
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.runner.kv_cache_manager import dcp_replicated_page_tokens
from vllm_torchtpu.spec_decode.dflash import DFlashProposer

BF16 = torch.bfloat16
NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, Q_LEN = 4, 1, 128, 3


class FakeDraftDcpAttention:
    """Records its calls; attends to a constant."""

    def __init__(self):
        self.calls = []

    def write(self, layer, key, value, kv_cache, attn_metadata):
        self.calls.append(("write", layer, key.shape, value.shape, kv_cache))

    def forward(self, layer, query, key, value, kv_cache, attn_metadata):
        self.calls.append(("forward", layer, query.shape, key.shape, kv_cache))
        return torch.full_like(query, 2.0)


@pytest.fixture
def cpu_vllm_config():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _impl() -> PallasAttentionBackendImpl:
    return PallasAttentionBackendImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=1.0,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=64,
        kv_cache_dtype="bf16",
        kv_cache_quantized_dtype=None,
    )


def _layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.register_buffer("_k_scale", torch.ones(()))
    return layer


def _metadata() -> AttentionMetadata:
    return AttentionMetadata(
        input_positions=torch.arange(Q_LEN, dtype=torch.int32),
        block_tables=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.tensor([Q_LEN], dtype=torch.int32),
        query_start_loc=torch.tensor([0, Q_LEN], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )


def test_off_by_default(cpu_vllm_config):
    impl = _impl()
    assert impl.sharded_draft_dcp is None
    assert not impl.kv_replicated_over_dcp


def test_enable_sets_rank_buffer_and_kv_heads(cpu_vllm_config):
    impl, layer, fake = _impl(), _layer(), FakeDraftDcpAttention()
    impl.enable_sharded_draft_dcp(
        layer, impl=fake, dcp=4, dcp_rank=3, kv_head_replicas=2
    )
    assert impl.sharded_draft_dcp is fake
    assert impl.kv_replicated_over_dcp
    assert impl.sharded_draft_dcp_kv_heads == 2
    assert layer.draft_dcp_rank.tolist() == [3]
    assert layer.draft_dcp_rank.dtype == torch.int32
    assert "draft_dcp_rank" not in layer.state_dict()


def test_forward_and_write_go_to_the_drafter_implementation(cpu_vllm_config):
    impl, layer, fake = _impl(), _layer(), FakeDraftDcpAttention()
    impl.enable_sharded_draft_dcp(
        layer, impl=fake, dcp=2, dcp_rank=0, kv_head_replicas=8
    )
    kv_cache = torch.zeros(
        PallasAttentionBackend.get_kv_cache_shape(2, 16, 1, HEAD_SIZE, BF16),
        dtype=BF16,
    )
    query = torch.ones(Q_LEN, NUM_HEADS * HEAD_SIZE, dtype=BF16)
    key = torch.ones(Q_LEN, NUM_KV_HEADS * HEAD_SIZE, dtype=BF16)

    def no_kernel(*args, **kwargs):
        raise AssertionError("the plain kernel must not run")

    with (
        patch.object(PallasAttentionBackendImpl, "_build_rpa_kernel", no_kernel),
        set_vllm_model_wrapper_context(mesh=SimpleNamespace()),
    ):
        out = impl.forward(
            query=query,
            key=key,
            value=key.clone(),
            kv_cache=kv_cache,
            attn_metadata=_metadata(),
            layer=layer,
        )
    assert out.shape == query.shape
    assert torch.all(out == 2.0)
    kind, seen_layer, q_shape, k_shape, seen_cache = fake.calls[-1]
    assert (kind, seen_layer, seen_cache) == ("forward", layer, kv_cache)
    assert q_shape == (Q_LEN, NUM_HEADS, HEAD_SIZE)
    assert k_shape == (Q_LEN, NUM_KV_HEADS, HEAD_SIZE)

    impl.write_sharded_draft_kv(layer, key, key.clone(), kv_cache, _metadata())
    kind, seen_layer, k_shape, v_shape, seen_cache = fake.calls[-1]
    assert (kind, seen_layer, seen_cache) == ("write", layer, kv_cache)
    assert k_shape == v_shape == (Q_LEN, NUM_KV_HEADS, HEAD_SIZE)


def _vllm_config(dcp: int):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp)
    )


def test_page_tokens_replicated_vs_sharded(cpu_vllm_config):
    impl = _impl()
    module = SimpleNamespace(impl=impl)
    assert dcp_replicated_page_tokens(module, 16, _vllm_config(2)) == 16
    impl.kv_replicated_over_dcp = True
    assert dcp_replicated_page_tokens(module, 16, _vllm_config(2)) == 32
    impl.enable_sharded_draft_dcp(
        _layer(), impl=FakeDraftDcpAttention(), dcp=2, dcp_rank=1, kv_head_replicas=8
    )
    assert dcp_replicated_page_tokens(module, 16, _vllm_config(2)) == 16


class _Provider(DraftDcpAttentionProvider):
    def __init__(self, sharded):
        self.sharded = sharded
        self.seen = None

    def make_draft_dcp_attention(self, layer, attn_impl, **kwargs):
        self.seen = (layer, attn_impl, kwargs)
        return self.sharded


def _proposer(draft_model) -> DFlashProposer:
    proposer = object.__new__(DFlashProposer)
    proposer.draft_model = draft_model
    proposer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=8)
    )
    proposer.speculative_config = SimpleNamespace(
        num_speculative_tokens=7,
        draft_model_config=SimpleNamespace(get_total_num_kv_heads=lambda: 2),
    )
    return proposer


def test_drafter_without_provider_keeps_kv_replicated(cpu_vllm_config):
    proposer = _proposer(SimpleNamespace())
    assert proposer._draft_dcp_attention(_layer(), _impl(), 2) is None


def test_provider_base_default_is_off(cpu_vllm_config):
    class Plain(DraftDcpAttentionProvider):
        pass

    draft = Plain()
    proposer = _proposer(draft)
    assert proposer._draft_dcp_attention(_layer(), _impl(), 2) is None
    assert draft.prepare_sharded_context_kv(None, 2, 4) is None


def test_provider_supplies_the_implementation(cpu_vllm_config):
    fake = FakeDraftDcpAttention()
    draft = _Provider(fake)
    layer, impl = _layer(), _impl()
    assert _proposer(draft)._draft_dcp_attention(layer, impl, 2) is fake
    seen_layer, seen_impl, kwargs = draft.seen
    assert seen_layer is layer and seen_impl is impl
    assert kwargs == dict(dcp=2, kv_head_replicas=4, max_query_len=8)
