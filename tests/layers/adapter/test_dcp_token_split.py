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
"""TPU_DCP_MLA_TOKEN_SPLIT's pure helpers (CPU): the head order lands every
rank on its o_proj heads after the DCP reduce-scatter and the lane
all-to-all, the q columns follow it, lane rows slice by data, and the
TP-gathered q_b copy keeps every per-channel parameter in step."""

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import dcp_token_split


def _owned_heads_after_merge(order, dcp, lanes, hpr):
    """Global heads rank (l, d) holds after the reduce-scatter (head block d
    of the kernel order) and the lane all-to-all (block l of that)."""
    per = len(order) // dcp
    out = {}
    for lane in range(lanes):
        for d in range(dcp):
            rs_block = order[d * per : (d + 1) * per]
            out[(lane, d)] = rs_block[lane * hpr : (lane + 1) * hpr]
    return out


@pytest.mark.parametrize("dcp,lanes,hpr", [(4, 2, 8), (2, 4, 8), (8, 2, 4), (4, 4, 2)])
def test_head_order_lands_each_rank_on_its_oproj_heads(dcp, lanes, hpr):
    order = dcp_token_split.head_order(dcp, lanes, hpr)
    assert sorted(order) == list(range(dcp * lanes * hpr))
    owned = _owned_heads_after_merge(order, dcp, lanes, hpr)
    for (lane, d), heads in owned.items():
        rank = lane * dcp + d
        assert heads == list(range(rank * hpr, (rank + 1) * hpr)), (lane, d)
    # Positive control: the standard order does not.
    standard = _owned_heads_after_merge(list(range(len(order))), dcp, lanes, hpr)
    assert any(
        heads != list(range((lane * dcp + d) * hpr, (lane * dcp + d + 1) * hpr))
        for (lane, d), heads in standard.items()
    )


def test_q_column_index_follows_the_head_order():
    dcp, lanes, hpr, head_dim = 4, 2, 8, 6
    group = dcp * hpr
    # Gathered q_b columns over the TP group: rank r holds lane r // dcp's
    # `group` heads; encode (rank, lane-local head, channel) in the value.
    cols = []
    for r in range(dcp * lanes):
        for h in range(group):
            for c in range(head_dim):
                cols.append((r, (r // dcp) * group + h, c))
    index = dcp_token_split.q_column_index(dcp, lanes, hpr, head_dim)
    picked = [cols[i] for i in index.tolist()]
    order = dcp_token_split.head_order(dcp, lanes, hpr)
    for p, g in enumerate(order):
        for c in range(head_dim):
            rank, head, chan = picked[p * head_dim + c]
            assert (head, chan) == (g, c)
            assert rank // dcp == g // group  # a rank of the head's lane


def test_lane_rows_slices_by_the_lane_tensor():
    x = torch.arange(16 * 3).reshape(16, 3)
    for lanes in (2, 4):
        n = 16 // lanes
        for lane in range(lanes):
            got = dcp_token_split.lane_rows(
                x, torch.tensor([lane], dtype=torch.int32), lanes
            )
            torch.testing.assert_close(got, x[lane * n : (lane + 1) * n])


class _Group:
    def __init__(self, per_rank):
        self.per_rank = per_rank

    def all_gather(self, t, dim):
        # Every rank's tensor of this parameter, rank-major along `dim`.
        name = next(k for k, v in self.per_rank[0].items() if v.shape == t.shape)
        return torch.cat([p[name] for p in self.per_rank], dim=dim)


@pytest.mark.parametrize("flipped", [False, True])
def test_gather_linear_output_channels(flipped):
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as op
    from vllm_torchtpu.layers.adapter.linear_common import (
        FP8_PALLAS_N_ATTR,
        FP8_PALLAS_WEIGHT,
        WEIGHT_FLIPPED_ATTR,
    )

    ranks, k, n = 4, 3, 5
    per_rank = []
    for r in range(ranks):
        w = torch.randn(n, k) + 10 * r
        per_rank.append(
            dict(
                weight=w.t().contiguous() if flipped else w,
                weight_scale=torch.arange(n, dtype=torch.float32) + 100 * r,
            )
        )
    layer = torch.nn.Module()
    for name, t in per_rank[1].items():
        layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
    layer.register_parameter(
        FP8_PALLAS_WEIGHT, torch.nn.Parameter(torch.zeros(k, 8), requires_grad=False)
    )
    setattr(layer, WEIGHT_FLIPPED_ATTR, flipped)
    setattr(layer, FP8_PALLAS_N_ATTR, n)
    index = torch.tensor([7, 0, 19, 12])
    new = op._gather_linear_output_channels(layer, _Group(per_rank), index)

    want_w = torch.cat([p["weight"] for p in per_rank], dim=1 if flipped else 0)
    want_s = torch.cat([p["weight_scale"] for p in per_rank])
    got_w = new.weight
    torch.testing.assert_close(
        got_w, want_w[:, index] if flipped else want_w[index], rtol=0, atol=0
    )
    torch.testing.assert_close(new.weight_scale, want_s[index], rtol=0, atol=0)
    assert FP8_PALLAS_WEIGHT not in dict(new.named_parameters())
    assert not hasattr(new, FP8_PALLAS_N_ATTR)
    # The source layer is untouched.
    assert layer.weight is not new.weight
    assert hasattr(layer, FP8_PALLAS_N_ATTR)
    assert np.array_equal(layer.weight.numpy(), per_rank[1]["weight"].numpy())


class _FakeLinear(torch.nn.Module):
    """A processed linear: fp8-style weight + per-channel scale, applied by
    its quant method (as q_b's is)."""

    def __init__(self, k, n, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(k, n, generator=g), False)
        self.weight_scale = torch.nn.Parameter(torch.rand(n, generator=g), False)
        self.quant_method = self

    def apply(self, layer, x, bias=None):
        return (x @ layer.weight) * layer.weight_scale

    def forward(self, x):
        return self.apply(self, x), None


def _ts_wrapper(tokens):
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as op

    w = op.VllmMultiHeadLatentAttentionWrapper.__new__(
        op.VllmMultiHeadLatentAttentionWrapper
    )
    torch.nn.Module.__init__(w)
    heads, group, lanes, hd = 1, 2, 2, 5
    w.kimi_preprocess = None
    w.q_lora_rank, w.kv_lora_rank = 4, 4
    w.qk_nope_head_dim, w.qk_rope_head_dim, w.qk_head_dim = 3, 2, hd
    w.num_heads, w.v_head_dim = heads, 2
    w.dcp_q_replicate, w.dcp_group_size = True, group
    w.is_sparse, w.indexer, w.skip_topk = True, None, True
    w.topk_indices_buffer = torch.zeros((tokens, 8), dtype=torch.int32)
    w.fused_qkv_a_proj = _FakeLinear(8, 4 + 4 + 2, 1)
    w.q_a_layernorm = torch.nn.Identity()
    w.kv_a_layernorm = torch.nn.Identity()
    w.q_b_proj = _FakeLinear(4, group * heads * hd, 2)
    w.q_b_proj.qrep_active, w.q_b_proj.group_size = True, group
    w.q_b_proj._local_view = lambda t: t
    w.rotary_emb = None
    w.g_proj, w.gate_is_fused, w.qb_overlap = None, False, False

    class _Attn(torch.nn.Module):
        def forward(self, q, kv_c_normed, k_pe, **kwargs):
            return kv_c_normed[:, :2] * 1.0

    w.mla_attn = _Attn()
    w.o_proj = lambda v: (v, None)
    w.ts_lanes = lanes
    w.__dict__["q_b_ts"] = _FakeLinear(4, lanes * group * heads * hd, 3)
    w._ts_qb_names = ("weight", "weight_scale")
    w._ts_qb_ts_names = ("weight", "weight_scale")
    w.ts_rank = torch.tensor([0], dtype=torch.int32)
    w.ts_lane = torch.tensor([1], dtype=torch.int32)
    w.token_split = True
    return w


def _lifted_names(tokens, monkeypatch):
    """Graph input names Dynamo lifts tracing the wrapper at `tokens` in warmup
    (TPU_DCP_MLA_TOKEN_SPLIT_MIN_TOKENS = 8: 16 splits, 4 does not)."""
    import torch._dynamo

    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import shape_variants
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as op

    monkeypatch.setattr(shape_variants, "in_warmup", lambda: True)
    monkeypatch.setattr(op.vllm_compilation_backends, "model_tag", "backbone")
    monkeypatch.setattr(envs, "TPU_DCP_MLA_TOKEN_SPLIT_MIN_TOKENS", 8, raising=False)
    w = _ts_wrapper(tokens)
    names = set()

    def backend(gm, example_inputs):
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                names.add(node.name)
        return gm.forward

    torch._dynamo.reset()
    fn = torch.compile(w.forward, backend=backend, dynamic=False)
    out = fn(torch.arange(tokens), torch.ones(tokens, 8))
    assert out.shape[0] == tokens
    return {n for n in names if "hidden_states" not in n and "positions" not in n}


def test_both_sides_of_the_token_split_branch_lift_the_same_inputs(monkeypatch):
    """The warmup hands a bucket over only between traces that lift the same
    named buffers: the head-split and token-split traces must agree."""
    split = _lifted_names(16, monkeypatch)
    head = _lifted_names(4, monkeypatch)
    assert any("q_b_ts" in n for n in split) and any("ts_lane" in n for n in head)
    assert split == head, (sorted(split - head), sorted(head - split))
