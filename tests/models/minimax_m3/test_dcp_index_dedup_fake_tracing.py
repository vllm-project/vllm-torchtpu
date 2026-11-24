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
"""The 4-bit MSA DCP op with index lanes (TPU_MINIMAX_M3_DCP_INDEX_DEDUP)
under ``FakeTensorMode`` with a symbolic token count, on stub DCP and lane
groups: DCP 2 (4 lanes, one KV head) and DCP 4 (2 lanes, two KV heads) at
the served shapes, with lane index pages of 32 / 64 tokens. The lanes'
scoring op returns every lane's heads' candidates, the lane merge one
lane's; the op output matches the op without lanes. Also the cache spec /
shape math of the lane index pages. Nothing lowers or runs."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

pytest.importorskip("torch_tpu")

SCALE = 128**-0.5 * 1.5625  # unique op names in this file (torch's op registry)
HEAD_DIM = 128
TOPK = 16
I32, BF16 = torch.int32, torch.bfloat16
TOKENS = 48
# Served shapes per rank (TP 8, 4 KV heads): c16 / c48 DCP 2, c64 DCP 4.
SHAPES = {
    "dcp2": dict(dcp=2, lanes=4, heads=8, reps=2, g=1, seqs=16, pps=1024, pages=256),
    "dcp4": dict(dcp=4, lanes=2, heads=8, reps=2, g=2, seqs=64, pps=1024, pages=256),
}
_KEEP: dict = {}


def _ops():
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    return msa_ops


def _empty(shape, dtype):
    return torch.empty(shape, dtype=dtype)


def _meta(s):
    return (
        _empty((s["seqs"],), I32),
        _empty((s["seqs"] * s["pps"],), I32),
        _empty((s["seqs"] + 1,), I32),
    )


def _index(s, lanes):
    """Folded 4-bit index pages of 128 / lanes tokens."""
    return _empty((s["pages"], 128 // lanes // 8 + 1, 4, HEAD_DIM), torch.uint8)


class _Group:
    def __init__(self, world_size):
        self.world_size = world_size
        self.rank_in_group = 0

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


def _dcp_op(name, lanes):
    from vllm_torchtpu.distributed import dcp as dcp_mod

    key = (name, lanes)
    if key in _KEEP:
        return _KEEP[key]
    s = SHAPES[name]
    # The builder binds the group accessors it imports: stub them while it runs.
    real = dcp_mod.get_dcp_group, dcp_mod.get_dcp_lane_group
    dcp_mod.get_dcp_group = lambda: _Group(s["dcp"])
    dcp_mod.get_dcp_lane_group = lambda: _Group(s["lanes"])
    try:
        op = _ops().get_msa_dcp_op(
            SCALE,
            TOPK,
            0,
            1,
            s["dcp"],
            s["reps"],
            fp4=True,
            decode_rows=8,
            **({"index_lanes": lanes} if lanes > 1 else {}),
        )
    finally:
        dcp_mod.get_dcp_group, dcp_mod.get_dcp_lane_group = real
    _KEEP[key] = op
    return op


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_lanes_op_symbolic(name, monkeypatch):
    s = SHAPES[name]
    lanes = s["lanes"]
    base, op = _dcp_op(name, 1), _dcp_op(name, lanes)
    msa_ops = _ops()
    seen = []

    def a2a(x, group):
        seen.append((tuple(x.shape), group.world_size))
        return torch.empty_like(x)

    monkeypatch.setattr(msa_ops, "_all_to_all", a2a)
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(TOKENS),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    with mode:
        num = seed.shape[0]
        q_shape = (
            (num, s["heads"], HEAD_DIM) if s["g"] > 1 else (s["heads"], num, HEAD_DIM)
        )

        def call(fn, index, **kw):
            return fn(
                mode.from_tensor(_empty(q_shape, BF16)),
                *(mode.from_tensor(_empty((num, 1, HEAD_DIM), BF16)) for _ in range(3)),
                mode.from_tensor(_empty((num, HEAD_DIM), BF16)),
                mode.from_tensor(
                    _empty((s["pages"], 16, 4, 2 * s["g"] * HEAD_DIM), torch.uint8)
                ),
                mode.from_tensor(index),
                _Metadata(
                    *(mode.from_tensor(m) for m in _meta(s)),
                    mode.from_tensor(_empty((3,), I32)),
                ),
                dcp_rank=mode.from_tensor(_empty((1,), I32)),
                scale_cache=mode.from_tensor(
                    _empty((s["pages"], 8, 128), torch.float32)
                ),
                hadamard=mode.from_tensor(_empty((HEAD_DIM, HEAD_DIM), BF16)),
                **kw,
            )

        want = call(base, _index(s, 1))
        got = call(op, _index(s, lanes), index_lane=mode.from_tensor(_empty((1,), I32)))
        with pytest.raises(ValueError, match="index_lane"):
            call(op, _index(s, lanes))
    assert got.shape == want.shape and got.dtype == want.dtype == BF16
    assert any(isinstance(d, torch.SymInt) for d in got.shape)
    # One candidate swap: [lanes, T, G * 2 * topk] over the lane group.
    assert len(seen) == 1
    shape, world = seen[0]
    assert world == lanes and shape[0] == lanes and shape[2] == s["g"] * 2 * TOPK


def test_lane_ops_fakes_and_names():
    s = SHAPES["dcp4"]
    lanes = s["lanes"]
    msa_ops = _ops()
    score, merge = msa_ops._index_lane_ops(
        lambda *a, **kw: None, TOPK, lanes, "fake_dedup_t16_il2"
    )
    assert (
        "score_lanes" in str(score._opoverload)
        and "_il2" in str(score._opoverload)
        and "lane_merge" in str(merge._opoverload)
    )
    mode = FakeTensorMode(shape_env=ShapeEnv())
    with mode:
        num = 40
        g = s["g"]
        args = [
            _empty((num, g, HEAD_DIM), BF16),
            _empty((num, g, HEAD_DIM), BF16),
            _empty((num, lanes * g, HEAD_DIM), BF16),
            _empty((num, HEAD_DIM), BF16),
            _empty((s["pages"], 16, 4, 2 * g * HEAD_DIM), torch.uint8),
            _index(s, lanes),
            _empty((s["pages"], 8, 128), torch.float32),
            *_meta(s),
            _empty((3,), I32),
            _empty((1,), I32),
            _empty((1,), I32),
            _empty((num, s["heads"], HEAD_DIM), BF16),
        ]
        outs = score(*[mode.from_tensor(a) for a in args])
        assert outs[3].shape == (num, lanes * g * 2 * TOPK)
        assert outs[1].shape == args[5].shape and outs[4].shape == args[-1].shape
        merged = merge(
            mode.from_tensor(_empty((lanes, num, g * 2 * TOPK), torch.float32))
        )
        assert merged.shape == (num, g * 2 * TOPK)
    with pytest.raises(ValueError, match="4-bit"):
        msa_ops.get_msa_dcp_op(SCALE * 3.3, TOPK, 0, 1, 4, 2, fp4=False, index_lanes=2)


@pytest.mark.parametrize("lanes,rows,head_size", [(1, 17, 68), (2, 9, 36), (4, 5, 20)])
def test_lane_index_page_spec_matches_shape(lanes, rows, head_size, monkeypatch):
    """The budgeted page bytes of a lane's 4-bit index page equal the
    allocated page: 128 / lanes tokens of 64 B + one 512 B scale row."""
    from types import SimpleNamespace

    from vllm_torchtpu.models.minimax_m3 import attention

    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(attention, "get_current_vllm_config", lambda: config)
    cache = attention.MiniMaxM3Cache(
        HEAD_DIM, torch.uint8, f"l{lanes}", token_shards=lanes, scales_in_page=True
    )
    shape = cache.kv_cache_shape(10)
    assert shape == (10, rows, 4, HEAD_DIM)
    spec = cache.get_kv_cache_spec(config)
    assert spec.head_size == head_size
    assert spec.page_size_bytes * 10 == torch.Size(shape).numel()
    # The K/V cache (no folded scales, 2 heads): unchanged budget.
    kv = attention.MiniMaxM3Cache(4 * HEAD_DIM, torch.uint8, f"kv{lanes}")
    assert kv.get_kv_cache_spec(config).head_size == 4 * HEAD_DIM // 2
