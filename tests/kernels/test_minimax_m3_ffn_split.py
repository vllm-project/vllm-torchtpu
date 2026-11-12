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
"""TPU_MINIMAX_M3_FFN_SPLIT_MIN_TOKENS: the decoder layer's post-attention
block on two token halves equals the unsplit block row for row (with an
exact all-reduce), and `pallas_all_reduce`'s chain orders the four
all-reduces attention 0, attention 1, FFN 0, FFN 1.

Runs on a CPU host: the two functions are read from their sources and run
against stand-ins for the TP collectives and the layer's modules."""

import ast
import dataclasses
import pathlib
import types
import typing

import pytest
import torch

SRC = pathlib.Path(__file__).parents[2] / "src/vllm_torchtpu"


def _extract(path: pathlib.Path, names: set[str], namespace: dict) -> types.ModuleType:
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) in names for t in targets):
                body.append(node)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    body.append(item)
    mod = types.ModuleType(path.stem)
    mod.__dict__.update(namespace)
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), mod.__dict__
    )
    return mod


class _Op:
    """The armed op: records each call (and the `prev` it was ordered after)
    and sums the ranks' partials, which the stand-in passes in stacked."""

    def __init__(self, log, ordered):
        self.log, self.ordered = log, ordered

    def __call__(self, x, rank, prev=None):
        out = x.sum(0) if x.dim() == 3 else x.clone()
        self.log.append((self.ordered, None if prev is None else id(prev), id(out)))
        return out


def _pallas(log):
    ns = dict(dataclasses=dataclasses, torch=torch, Any=typing.Any)
    mod = _extract(
        SRC / "distributed/pallas_all_reduce.py",
        {
            "_Armed",
            "_ARMED",
            "_CHAIN",
            "_DEFERRED",
            "all_reduce",
            "armed",
            "begin_chain",
            "end_chain",
        },
        ns,
    )
    mod._ARMED = mod._Armed(
        ranks=(0, 1), op=_Op(log, False), rank=None, ordered_op=_Op(log, True)
    )
    return mod


def test_chain_orders_all_reduces_and_closes():
    log = []
    pr = _pallas(log)
    x = torch.ones(4, 8, dtype=torch.bfloat16)
    pr.all_reduce((0, 1), x)  # outside a chain: plain op
    pr.begin_chain()
    outs = [pr.all_reduce((0, 1), x) for _ in range(3)]
    pr.end_chain()
    pr.all_reduce((0, 1), x)
    assert [o for o, _, _ in log] == [False, False, True, True, False]
    # Each chained call is ordered after the previous chained output.
    assert log[2][1] == id(outs[0]) and log[3][1] == id(outs[1])
    assert pr._CHAIN is None and pr.armed()
    # Not armed for other ranks or non-2-D / non-bf16 inputs.
    assert pr.all_reduce((0, 2), x) is None
    assert pr.all_reduce((0, 1), x.float()) is None


@pytest.mark.parametrize("tokens", [16, 24])
def test_ffn_halves_equal_unsplit(tokens):
    pr = _pallas([])
    ranks, hidden = 2, 8
    gen = torch.Generator().manual_seed(tokens)
    # Each rank's attention partial, token-major [T, ranks, H]; the stand-in
    # all-reduce sums the ranks exactly (float64), so split and unsplit must
    # agree bit for bit.
    partials = torch.randn(tokens, ranks, hidden, generator=gen, dtype=torch.float64)
    residual = torch.randn(tokens, hidden, generator=gen, dtype=torch.float64)
    w = torch.randn(hidden, hidden, generator=gen, dtype=torch.float64)
    order = []

    def all_reduce(x):
        order.append(("ar", x.shape[0]))
        return x.sum(1)

    def attn(positions, h, reduce=True):
        assert not reduce
        return partials

    def norm(a, r):
        s = a + r
        return s / s.norm(dim=-1, keepdim=True), s

    def ffn(h):
        order.append(("ffn", h.shape[0]))
        return h @ w

    layer = types.SimpleNamespace(self_attn=attn, post_attention_layernorm=norm)
    model = _extract(
        SRC / "models/minimax_m3/model.py",
        {"_forward_ffn_halves"},
        dict(
            torch=torch,
            pallas_all_reduce=pr,
            tensor_model_parallel_all_reduce=all_reduce,
        ),
    )
    hidden = torch.zeros(tokens, hidden, dtype=torch.float64)
    out, res = model._forward_ffn_halves(layer, None, hidden, residual, ffn)
    ref_h, ref_r = norm(partials.sum(1), residual)
    assert torch.equal(res, ref_r)
    assert torch.equal(out, ref_h @ w)
    half = tokens // 2
    assert order == [("ar", half), ("ar", half), ("ffn", half), ("ffn", half)]
    assert pr._CHAIN is None


def _pallas_ordered(log):
    ns = dict(dataclasses=dataclasses, torch=torch, Any=typing.Any)
    mod = _extract(
        SRC / "distributed/pallas_all_reduce.py",
        {
            "_Armed",
            "_ARMED",
            "_CHAIN",
            "_DEFERRED",
            "_AFTER_ROWS",
            "all_reduce",
            "all_reduce_after",
            "begin_defer",
            "end_defer",
            "armed",
        },
        ns,
    )

    class Op:
        """Two ranks holding the same partial: the sum is 2x, exact in bf16
        for small integers."""

        def __init__(self, ordered):
            self.ordered = ordered

        def __call__(self, x, rank, prev=None):
            out = x * 2
            log.append((self.ordered, None if prev is None else prev, out))
            return out

    mod._ARMED = mod._Armed(ranks=(0, 1), op=Op(False), rank=None, ordered_op=Op(True))
    return mod


def _fused_moe_stub():
    ns = dict(torch=torch)
    return _extract(
        SRC / "layers/adapter/fused_moe.py",
        {"_DISPATCH_TOKENS", "begin_dispatch_tokens", "end_dispatch_tokens"},
        ns,
    )


@pytest.mark.parametrize("moe", [True, False])
@pytest.mark.parametrize("tokens", [16, 24])
def test_ffn_halves_dispatch_order(tokens, moe):
    log = []
    pr = _pallas_ordered(log)
    fm = _fused_moe_stub()
    hidden = 8
    gen = torch.Generator().manual_seed(tokens)

    def ints(*shape):
        return torch.randint(-4, 5, shape, generator=gen).to(torch.bfloat16)

    attn_out = ints(tokens, hidden)
    residual = ints(tokens, hidden)
    dispatch = []

    def attn(positions, h, reduce=True, split_rows=None):
        assert split_rows == tokens // 2
        return attn_out[:split_rows].clone(), attn_out[split_rows:].clone()

    def norm(a, r):
        s = a + r
        return s, s

    def ffn(h):
        if moe and fm._DISPATCH_TOKENS is not None:
            token = h[:8, :4].clone()
            dispatch.append(token)
            fm._DISPATCH_TOKENS.append(token)
        out = pr.all_reduce((0, 1), h * 3)  # the FFN's own all-reduce
        return out

    layer = types.SimpleNamespace(self_attn=attn, post_attention_layernorm=norm)
    model = _extract(
        SRC / "models/minimax_m3/model.py",
        {"_forward_ffn_halves_dispatch_order"},
        dict(torch=torch, pallas_all_reduce=pr, fused_moe=fm),
    )
    x = torch.zeros(tokens, hidden, dtype=torch.bfloat16)
    out, res = model._forward_ffn_halves_dispatch_order(layer, None, x, residual, ffn)
    # Unsplit reference: all-reduce (2x), norm, FFN (3x), all-reduce (2x).
    ref_h, ref_r = norm(attn_out * 2, residual)
    assert torch.equal(res, ref_r)
    assert torch.equal(out, ref_h * 3 * 2)
    # Four all-reduces, the FFNs' deferred ones done by the layer.
    assert len(log) == 4 and [o for o, _, _ in log] == [False, True, True, True]
    a0, a1, f0, f1 = (entry[2] for entry in log)
    if moe:
        # attention 1 after FFN 0's dispatch, FFN 0 after FFN 1's dispatch.
        assert log[1][1] is dispatch[0] and log[2][1] is dispatch[1]
    else:
        assert torch.equal(log[1][1], a0[:8]) and torch.equal(log[2][1], a1[:8])
    assert torch.equal(log[3][1], f0[:8])
    assert pr._DEFERRED is None and fm._DISPATCH_TOKENS is None
