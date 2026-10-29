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
"""Torch-side dispatch of dense MLA under DCP (the ops themselves are mocked)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.layers.adapter import attention as attention_adapter
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl

LOCAL_HEADS, DCP, RANK, LKV = 2, 2, 512, 512


def _impl():
    impl = PallasMLAttentionBackendImpl(
        num_heads=LOCAL_HEADS,
        head_size=576,
        scale=0.125,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
        logits_soft_cap=None,
        attn_type="DECODER",
        kv_sharing_target_layer_name=None,
        kv_lora_rank=LKV,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
    )
    impl.dcp_size = DCP
    return impl


@pytest.fixture
def collectives(monkeypatch):
    """Head gather = repeat; merge = this rank's head slice of the partial."""
    gathered = []

    def gather(ql, qpe, _rank):
        gathered.append(ql.shape[0])
        return ql.repeat(1, DCP, 1), qpe.repeat(1, DCP, 1)

    monkeypatch.setattr(attention_adapter, "all_gather_q_nope_pe_heads", gather)
    monkeypatch.setattr(
        attention_adapter,
        "merge_lse_partials_scatter_heads",
        lambda out, _lse, **_kw: out[:, :LOCAL_HEADS],
    )
    return gathered


def _step(num_tokens, num_seqs, num_decode):
    ql = torch.randn(num_tokens, LOCAL_HEADS, LKV)
    q_pe = torch.randn(num_tokens, LOCAL_HEADS, 64)
    kv_c, k_pe = torch.randn(num_tokens, LKV), torch.randn(num_tokens, 64)
    cache = torch.zeros(4, 8, 4, 640)
    cu = torch.arange(num_seqs + 1, dtype=torch.int32).clamp(max=num_tokens)
    md = SimpleNamespace(
        seq_lens=torch.ones(num_seqs, dtype=torch.int32),
        block_tables=torch.zeros(num_seqs, dtype=torch.int32),
        query_start_loc=cu,
        request_distribution=torch.tensor([num_decode, num_decode, num_seqs]),
    )
    decoded = torch.full((min(num_tokens, num_seqs), LOCAL_HEADS * DCP, LKV), 1.0)
    prefilled = torch.full((num_tokens, LOCAL_HEADS, LKV), 2.0)
    decode_op = MagicMock(
        return_value=(torch.ones_like(cache), decoded, torch.zeros(decoded.shape[:2]))
    )
    prefill_op = MagicMock(return_value=prefilled)
    layer = SimpleNamespace(kv_lora_rank=LKV, mla_dcp_ops=(decode_op, prefill_op))
    return layer, (ql, q_pe, kv_c, k_pe, cache, md)


def test_decode_bucket_runs_only_the_decode_op(collectives):
    layer, (ql, q_pe, kv_c, k_pe, cache, md) = _step(4, 8, 4)
    out = _impl()._forward_dense_mla_dcp(layer, ql, q_pe, kv_c, k_pe, cache, md)
    decode_op, prefill_op = layer.mla_dcp_ops
    prefill_op.assert_not_called()
    assert collectives == [4]
    args = decode_op.call_args.args
    assert args[3].shape[0] == 4  # the store sees every token
    assert torch.equal(out, torch.ones(4, LOCAL_HEADS, LKV))
    assert torch.equal(cache, torch.ones_like(cache))  # the stored cache is kept


def test_prefill_bucket_combines_by_token(collectives):
    layer, (ql, q_pe, kv_c, k_pe, cache, md) = _step(12, 8, 3)
    out = _impl()._forward_dense_mla_dcp(layer, ql, q_pe, kv_c, k_pe, cache, md)
    decode_op, prefill_op = layer.mla_dcp_ops
    assert collectives == [8]  # only the leading num_seqs tokens are gathered
    assert decode_op.call_args.args[3].shape[0] == 12
    assert prefill_op.call_args.args[1] is ql  # own heads, every token
    assert torch.equal(out[:3], torch.ones(3, LOCAL_HEADS, LKV))
    assert torch.equal(out[3:], torch.full((9, LOCAL_HEADS, LKV), 2.0))


def test_forward_requires_built_ops():
    layer, args = _step(4, 8, 4)
    layer.mla_dcp_ops = None
    with pytest.raises(AssertionError, match="never built"):
        _impl()._forward_dense_mla_dcp(layer, *args)


@pytest.mark.parametrize("num_tokens,num_decode", [(4, 4), (12, 3)])
def test_dispatch_traces_whole_graph_with_dynamic_tokens(
    collectives, num_tokens, num_decode
):
    """The backbone runs under torch.compile(fullgraph=True) with a dynamic token
    dimension: no step of the dispatch may read a device value into Python."""
    layer, (ql, q_pe, kv_c, k_pe, cache, md) = _step(num_tokens, 8, num_decode)

    def decode_op(cache, q, q_pe, kv_c, k_pe, *metadata):
        del q_pe, kv_c, k_pe, metadata
        return cache + 1, torch.ones_like(q), q.new_zeros(q.shape[:2])

    def prefill_op(cache, q, *rest):
        del cache, rest
        return torch.full_like(q, 2.0)

    layer.mla_dcp_ops = (decode_op, prefill_op)
    impl = _impl()

    def step(ql, q_pe, kv_c, k_pe, cache, qsl, dist):
        md_ = SimpleNamespace(
            seq_lens=md.seq_lens,
            block_tables=md.block_tables,
            query_start_loc=qsl,
            request_distribution=dist,
        )
        return impl._forward_dense_mla_dcp(layer, ql, q_pe, kv_c, k_pe, cache, md_)

    for t in (ql, q_pe, kv_c, k_pe):
        torch._dynamo.mark_dynamic(t, 0)
    compiled = torch.compile(step, fullgraph=True, backend="eager", dynamic=True)
    out = compiled(
        ql, q_pe, kv_c, k_pe, cache, md.query_start_loc, md.request_distribution
    )
    assert torch.equal(out[:num_decode], torch.ones(num_decode, LOCAL_HEADS, LKV))
    assert torch.equal(out[num_decode:], torch.full_like(out[num_decode:], 2.0))
