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
"""The 4-bit MSA DCP ops of a layer reading a shared scale page at a row
offset (TPU_MINIMAX_M3_KV4_SHARED_SCALES): each op's fake fits its schema
with a symbolic token count under ``FakeTensorMode`` and under
``torch.compile(fullgraph=True)`` over meta tensors, the whole DCP op runs
on a stub group for DCP 2 (one KV head) and DCP 4 (two), with the shared
``[pages, 8, 128]`` page in and out, and the ops get names of their own
(one op set per row offset). Nothing lowers or runs."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

pytest.importorskip("torch_tpu")

SCALE = 128**-0.5 * 1.4375  # unique op names in this file (torch's op registry)
HEAD_DIM = 128
TOPK = 16
I32, BF16 = torch.int32, torch.bfloat16
TOKENS = 48
# Served shapes per rank (TP 8, 4 KV heads): c16 / c48 DCP 2, c64 DCP 4.
SHAPES = {
    "dcp2": dict(dcp=2, heads=8, reps=2, g=1, seqs=16, pps=1024, pages=256, row0=6),
    "dcp4": dict(dcp=4, heads=8, reps=2, g=2, seqs=64, pps=1024, pages=256, row0=4),
}
_KEEP: dict = {}


def _ops():
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    return msa_ops


def _keep(key, build):
    if key not in _KEEP:
        _KEEP[key] = build()
    return _KEEP[key]


def _empty(shape, dtype):
    return torch.empty(shape, dtype=dtype)


def _meta(s):
    return (
        _empty((s["seqs"],), I32),
        _empty((s["seqs"] * s["pps"],), I32),
        _empty((s["seqs"] + 1,), I32),
    )


def _kv(s):
    return _empty((s["pages"], 16, 4, 2 * s["g"] * HEAD_DIM), torch.uint8)


def _index(s):
    return _empty((s["pages"], 17, 4, HEAD_DIM), torch.uint8)


def _page(s):
    return _empty((s["pages"], 8, 128), torch.float32)  # the shared page


def _symbolic():
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(TOKENS),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    return mode, seed


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
def test_attend_merge_op_fake_symbolic_and_dynamo(fused):
    s = SHAPES["dcp2"]
    op = _keep(
        ("am", fused),
        lambda: _ops()._get_msa_dcp_attend_merge_op(
            SCALE, TOPK, 2, True, "fake_kv4sh", fused=fused, scale_row0=s["row0"]
        ),
    )
    assert len(op._opoverload._schema.returns) == 1

    def args(num):
        return [
            _empty((1,), I32),
            _empty((s["heads"], num, HEAD_DIM), BF16),
            _kv(s),
            _empty((num, 2 * TOPK), torch.float32),
            *_meta(s),
            _page(s),
            _empty((HEAD_DIM, HEAD_DIM), BF16),
        ]

    mode, seed = _symbolic()
    with mode:
        num = seed.shape[0]
        out = op(*[mode.from_tensor(a) for a in args(num)])
    assert out.shape == (num, s["heads"] * HEAD_DIM) and out.dtype == BF16
    assert isinstance(out.shape[0], torch.SymInt)
    meta = [a.to("meta") for a in args(TOKENS)]
    torch._dynamo.mark_dynamic(meta[1], 1)
    torch._dynamo.mark_dynamic(meta[3], 0)
    torch._dynamo.reset()
    out = torch.compile(lambda *xs: op(*xs) + 1, backend="eager", fullgraph=True)(
        *meta
    )
    assert out.shape == (TOKENS, s["heads"] * HEAD_DIM)


def test_heads_op_fake_symbolic():
    s = SHAPES["dcp4"]
    op = _keep(
        "heads",
        lambda: _ops()._get_msa_dcp_heads_op(
            SCALE, TOPK, 4, 2, 8, "fake_kv4sh", scale_row0=s["row0"]
        ),
    )
    mode, seed = _symbolic()
    with mode:
        num = seed.shape[0]
        args = (
            _empty((1,), I32),
            _empty((num, s["heads"], HEAD_DIM), BF16),
            _kv(s),
            _page(s),
            _empty((num, s["g"] * 2 * TOPK), torch.float32),
            *_meta(s),
            _empty((3,), I32),
        )
        out = op(*[mode.from_tensor(a) for a in args])
    assert out.shape == (num, s["heads"], HEAD_DIM)
    assert isinstance(out.shape[0], torch.SymInt)


class _Group:
    def __init__(self, world_size):
        self.world_size = world_size

    def all_gather(self, x, dim):
        return torch.cat([x] * self.world_size, dim=dim)

    def reduce_scatter(self, x, dim):
        return x.chunk(self.world_size, dim=dim)[0].clone()


class _Metadata:
    def __init__(self, seq_lens, block_tables, query_start_loc, distribution):
        self.seq_lens = seq_lens
        self.block_tables = block_tables
        self.query_start_loc = query_start_loc
        self.request_distribution = distribution


def _dcp_op(name, row0):
    from vllm_torchtpu.distributed import dcp as dcp_mod

    key = ("dcp", name, row0)
    if key in _KEEP:
        return _KEEP[key]
    s = SHAPES[name]
    real = dcp_mod.get_dcp_group
    dcp_mod.get_dcp_group = lambda: _Group(s["dcp"])
    try:
        op = _ops().get_msa_dcp_op(
            SCALE * (1.5 + row0 / 16),
            TOPK,
            0,
            1,
            s["dcp"],
            s["reps"],
            fp4=True,
            decode_rows=8,
            **({"scale_row0": row0} if row0 else {}),
        )
    finally:
        dcp_mod.get_dcp_group = real
    _KEEP[key] = op
    return op


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_dcp_op_shared_page_symbolic(name):
    """A layer at row 0 (the page's owner) and one at the row offset: same
    output shapes, the shared page comes back whole, symbolic token dim."""
    s = SHAPES[name]
    ops = [_dcp_op(name, 0), _dcp_op(name, s["row0"])]
    mode, seed = _symbolic()
    with mode:
        num = seed.shape[0]
        q_shape = (num, s["heads"], HEAD_DIM) if s["g"] > 1 else (s["heads"], num, HEAD_DIM)
        page = mode.from_tensor(_page(s))
        outs = []
        for op in ops:
            outs.append(
                op(
                    mode.from_tensor(_empty(q_shape, BF16)),
                    *(mode.from_tensor(_empty((num, 1, HEAD_DIM), BF16)) for _ in range(3)),
                    mode.from_tensor(_empty((num, HEAD_DIM), BF16)),
                    mode.from_tensor(_kv(s)),
                    mode.from_tensor(_index(s)),
                    _Metadata(*(mode.from_tensor(m) for m in _meta(s)),
                              mode.from_tensor(_empty((3,), I32))),
                    dcp_rank=mode.from_tensor(_empty((1,), I32)),
                    scale_cache=page,
                    hadamard=mode.from_tensor(_empty((HEAD_DIM, HEAD_DIM), BF16)),
                )
            )  # fmt: skip
        assert page.shape == (s["pages"], 8, 128)
    for out in outs:
        assert out.shape == outs[0].shape and out.dtype == BF16
        assert any(isinstance(d, torch.SymInt) for d in out.shape)


def test_shared_ops_named_per_row(monkeypatch):
    """The row offset reaches every sub-op builder and names the op set."""
    msa_ops = _ops()
    seen = []
    real_am = msa_ops._get_msa_dcp_attend_merge_op

    def am(*a, **kw):
        seen.append(("am", a[4], kw.get("scale_row0", 0)))
        return real_am(*a, **kw)

    monkeypatch.setattr(msa_ops, "_get_msa_dcp_attend_merge_op", am)
    _KEEP.pop(("dcp", "dcp2", 2), None)
    _dcp_op("dcp2", 2)
    assert seen and all(row == 2 and suffix.endswith("_sr2") for _, suffix, row in seen)
    with pytest.raises(ValueError, match="4-bit"):
        msa_ops.get_msa_dcp_op(SCALE * 3.1, TOPK, 0, 1, 2, 2, fp4=False, scale_row0=2)
