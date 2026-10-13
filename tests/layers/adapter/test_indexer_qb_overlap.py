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
"""TPU_INDEXER_QB_OVERLAP wiring (torch, any device).

The kernel-level claim -- `streamindex_topk_dcp(overlap_fn=...)` returns the
top-k and the overlapped work byte-identical to running them apart -- is in
tests/kernels/test_dsv4_streamindex_topk_dcp.py. What this file checks is the
plumbing around it:

* the wrapper hands the indexer the q_b layer and attends with the q the
  indexer returns, exactly as it would with q from `q_b_proj` itself, and
  falls back to `q_b_proj` when the indexer returns no q;
* the indexer passes the op exactly `(q_c, q_b.weight, q_b.weight_scale)`;
* only a layer that would run `_fp8_linear_pallas_jax` on its plain weight is
  accepted.
"""

from unittest.mock import MagicMock

import pytest
import torch

# pytest prepends this directory to sys.path (no __init__.py here).
from test_mla_rope_neox_permute import _inputs, _run, _wrapper  # noqa: E402

from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import VllmTPUIndexer
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_N_ATTR,
    FP8_PALLAS_WEIGHT,
    KEEP_VLLM_LAYOUT_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

TOPK = 8


class _FakeIndexer(torch.nn.Module):
    """Stands in for the indexer: a deterministic top-k from the inputs, and
    with `q_b_layer` the q the overlap op would return -- `q_b_layer(qr)` --
    or None when told to decline."""

    def __init__(self, decline=False):
        super().__init__()
        self.decline = decline
        self.calls = []

    def forward(self, hidden_states, qr, positions, rotary_emb, q_b_layer=None):
        self.calls.append(q_b_layer)
        topk = (qr.float().sum(-1, keepdim=True).abs() * 1000).int() % 97
        topk = topk.expand(-1, TOPK).contiguous()
        if q_b_layer is None:
            return topk
        if self.decline:
            return topk, None
        return topk, q_b_layer(qr)[0]


def _sparse_wrapper(qb_overlap, decline=False, **kw):
    w = _wrapper(**kw)
    w.indexer = _FakeIndexer(decline=decline)
    w.indexer_rope_emb = None
    w.skip_topk = False
    w.qb_overlap = qb_overlap
    return w


@pytest.mark.parametrize("num_tokens", [16, 96], ids=["decode16", "prefill96"])
@pytest.mark.parametrize(
    "heads,group_size,rank", [(16, 1, 0), (4, 4, 2)], ids=["qrep_off", "qrep_on"]
)
@pytest.mark.parametrize("decline", [False, True], ids=["op_q", "fallback_q"])
def test_flag_on_matches_flag_off(num_tokens, heads, group_size, rank, decline):
    kw = dict(heads=heads, group_size=group_size, rank_in_group=rank, layout="fp8")
    off = _sparse_wrapper(False, **kw)
    on = _sparse_wrapper(True, decline=decline, **kw)
    hidden, positions = _inputs(num_tokens)
    r_off, r_on = _run(off, hidden, positions), _run(on, hidden, positions)
    # Flag off never offers the q_b layer; flag on offers this wrapper's.
    assert off.indexer.calls == [None]
    assert on.indexer.calls == [on.q_b_proj]
    for name in ("q_nope", "q_pe", "kv_c", "k_pe"):
        assert torch.equal(r_on[name], r_off[name]), name
    if r_off["rep"] is not None:
        for a, b in zip(r_off["rep"], r_on["rep"]):
            assert torch.equal(a, b)
    t_off = off.mla_attn.call_args.kwargs["topk_indices"]
    t_on = on.mla_attn.call_args.kwargs["topk_indices"]
    assert torch.equal(t_on, t_off)


def test_positive_control_wrong_q_is_caught():
    """If the op handed back anything but q_b(q_c) -- here the q of another
    layer's weights -- the comparison above must see it."""
    kw = dict(heads=16, layout="fp8")
    off = _sparse_wrapper(False, **kw)
    on = _sparse_wrapper(True, **kw)
    other = _wrapper(seed=11, **kw).q_b_proj
    real = on.indexer.forward

    def wrong(hidden_states, qr, positions, rotary_emb, q_b_layer=None):
        topk, _ = real(hidden_states, qr, positions, rotary_emb, q_b_layer)
        return topk, other(qr)[0]

    on.indexer.forward = wrong
    hidden, positions = _inputs(16)
    r_off, r_on = _run(off, hidden, positions), _run(on, hidden, positions)
    assert not torch.equal(r_on["q_nope"], r_off["q_nope"])


def _qb_layer(**overrides):
    layer = torch.nn.Module()
    layer.quant_method = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
    layer.weight = torch.nn.Parameter(
        torch.zeros(128, 256).to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(torch.ones(256), requires_grad=False)
    layer.bias = None
    setattr(layer, FP8_PALLAS_N_ATTR, 256)
    for k, v in overrides.items():
        setattr(layer, k, v)
    return layer


def test_operands_accept_the_plain_fp8_pallas_layer():
    layer = _qb_layer()
    w, s = mla_op._qb_overlap_operands(layer)
    assert w is layer.weight and s is layer.weight_scale


@pytest.mark.parametrize(
    "overrides",
    [
        {FP8_PALLAS_N_ATTR: None},
        {KEEP_VLLM_LAYOUT_ATTR: True},
        {"bias": torch.nn.Parameter(torch.zeros(256))},
        {FP8_PALLAS_WEIGHT: torch.nn.Parameter(torch.zeros(1))},
        {"quant_method": object()},
        {"gather_output": True},
    ],
    ids=["no_pallas", "vllm_layout", "bias", "padded_copy", "other_method", "gather"],
)
def test_operands_refuse_anything_else(overrides):
    assert mla_op._qb_overlap_operands(_qb_layer(**overrides)) is None


def test_indexer_hands_the_op_q_c_and_the_q_b_weights():
    """`VllmTPUIndexer.forward(..., q_b_layer=)` passes `(q_c, w_q, w_s)` and
    returns the op's pair; a layer it cannot take yields `(topk, None)`."""
    n_head, head_dim, t = 2, 128, 16
    idx = VllmTPUIndexer.__new__(VllmTPUIndexer)
    torch.nn.Module.__init__(idx)
    idx.n_head, idx.head_dim, idx.rope_dim = n_head, head_dim, 64
    idx.softmax_scale = 1.0
    idx.wq_b = lambda x: (torch.ones(x.shape[0], n_head * head_dim), None)
    idx.wk_weights_proj = lambda x: (torch.ones(x.shape[0], head_dim + n_head), None)
    idx.k_norm = torch.nn.Identity()
    op = MagicMock(return_value=("topk", "q"))
    idx.indexer_op = op
    rope = lambda pos, q, k: (q, k)  # noqa: E731
    qr = torch.randn(t, 128).bfloat16()
    hidden = torch.randn(t, 64).bfloat16()
    layer = _qb_layer()
    out = idx.forward(hidden, qr, torch.arange(t), rope, q_b_layer=layer)
    assert out == ("topk", "q")
    x, w, s = op.call_args.kwargs["qb_overlap"]
    assert torch.equal(x, qr) and w is layer.weight and s is layer.weight_scale

    op.reset_mock()
    op.return_value = "topk"
    out = idx.forward(
        hidden, qr, torch.arange(t), rope, q_b_layer=_qb_layer(bias=torch.ones(1))
    )
    assert out == ("topk", None)
    assert "qb_overlap" not in op.call_args.kwargs
