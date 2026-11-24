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
"""The MSA DCP decode ops trace with fake tensors and a symbolic token count,
as Dynamo runs them in the runner's ``capture_model``, with MSA index reuse
(TPU_MINIMAX_M3_MSA_INDEX_REUSE_DECODE) off and on: each op's registered fake
gives as many outputs as its schema (which torch_tpu derives from the JAX
function's return annotation) and keeps the symbolic token dim.

Three ways in: a direct call of each jax_op under ``FakeTensorMode`` with a
``ShapeEnv`` and a symbolic first dim; ``torch.compile(fullgraph=True)``
over meta tensors with the token dim marked dynamic (where a one-output
annotation on a two-output op failed as "Unable to cast (...) to Tensor");
and the whole DCP op of a Full layer then a Shared layer, on a stub DCP
group, for the served c4/c16 (DCP 2, bf16), c48 (DCP 2, 4-bit) and c64
(DCP 8, 4-bit, four KV heads per group) configs. Nothing lowers or runs.
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

pytest.importorskip("torch_tpu")

SCALE = 128**-0.5
HEAD_DIM = 128
TOPK = 16
I32 = torch.int32
BF16 = torch.bfloat16
TOKENS = 72  # the hint of the symbolic token count

# Served shapes per rank (TP 8, 64 query heads, 4 KV heads): DCP size, query
# heads, KV head replicas, KV heads per DCP group (G), max sequences, pages
# per sequence, KV cache pages, 4-bit caches.
SERVED = {
    "c16": dict(dcp=2, heads=8, reps=2, g=1, seqs=16, pps=1024, pages=256, fp4=0),
    "c48": dict(dcp=2, heads=8, reps=2, g=1, seqs=48, pps=1024, pages=256, fp4=1),
    "c64": dict(dcp=8, heads=8, reps=2, g=4, seqs=64, pps=1024, pages=256, fp4=1),
}


def _schema_outputs(op) -> int:
    return len(op._opoverload._schema.returns)


def _check(op, outs, num_tokens):
    """``outs`` fit ``op``'s schema; every output has a token-sized dim."""
    flat = outs if isinstance(outs, tuple) else (outs,)
    assert len(flat) == _schema_outputs(op), (op._name, len(flat))
    for t in flat:
        assert isinstance(t, torch.Tensor)
        assert any(isinstance(d, torch.SymInt) or d == num_tokens for d in t.shape), (
            op._name,
            t.shape,
        )
    return flat


def _mode():
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(TOKENS),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    return mode, seed


def _fake_call(op, make_args):
    """Call ``op`` under FakeTensorMode with a symbolic token count ``T``
    (``make_args(T)`` builds the operands)."""
    mode, seed = _mode()
    with mode:
        num = seed.shape[0]
        assert isinstance(num, torch.SymInt)
        args = tuple(
            mode.from_tensor(a)
            if isinstance(a, torch.Tensor) and not mode.is_our_fake(a)
            else a
            for a in make_args(num)
        )
        flat = _check(op, op(*args), num)
    for t in flat:
        assert any(isinstance(d, torch.SymInt) for d in t.shape), (op._name, t.shape)
    return num, flat


def _empty(shape, dtype):
    return torch.empty(shape, dtype=dtype)


def _meta(shape):
    seqs, pps = shape["seqs"], shape["pps"]
    return _empty((seqs,), I32), _empty((seqs * pps,), I32), _empty((seqs + 1,), I32)


def _kv(shape):
    g, pages = shape["g"], shape["pages"]
    if shape["fp4"]:
        return _empty((pages, 128 // 8, 4, 2 * g * HEAD_DIM), torch.uint8)
    return _empty((pages, 64, 2, 2 * g * HEAD_DIM), BF16)


def _index(shape):
    if shape["fp4"]:
        return _empty((shape["pages"], 128 // 8 + 1, 4, HEAD_DIM), torch.uint8)
    return _empty((shape["pages"], 64, 2, HEAD_DIM), BF16)


def _scale(shape):
    return _empty((shape["pages"], 2 * shape["g"], 128), torch.float32)


def _ops():
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    return msa_ops


# Every op built here stays referenced for the whole session and is built
# once: torch_tpu registers each jax_op as a torch custom op by name, and
# redefining a name in one process corrupts the op registry (the served model
# builds each op once).
_KEEP: dict = {}


def _keep(key, build):
    if key not in _KEEP:
        _KEEP[key] = build()
    return _KEEP[key]


def _am(name, selection):
    shape = SERVED[name]
    return _keep(
        ("am", name, selection),
        lambda: _ops()._get_msa_dcp_attend_merge_op(
            SCALE, TOPK, shape["dcp"], bool(shape["fp4"]), f"fake_{name}", selection
        ),
    )


def _am_args(shape, selection):
    """The attend+merge operands: head-major ``q [Hl, T, D]``, candidates
    ``[T, 2 * topk]`` f32 (the score op's) or a selection ``[T, 1, topk]``
    int32 ("given")."""

    def make(num):
        q = _empty((shape["heads"], num, HEAD_DIM), BF16)
        if selection == "given":
            sel = _empty((num, 1, TOPK), I32)
        else:
            sel = _empty((num, 2 * TOPK), torch.float32)
        args = [_empty((1,), I32), q, _kv(shape), sel, *_meta(shape)]
        if shape["fp4"]:
            args += [_scale(shape), _empty((HEAD_DIM, HEAD_DIM), BF16)]
        return tuple(args)

    return make


# ------------------------------------------------- attend+merge (c4/c16/c48)
@pytest.mark.parametrize("selection", ["candidates", "return", "given"])
@pytest.mark.parametrize("name", ["c16", "c48"])
def test_attend_merge_fake_symbolic(name, selection):
    """Off ("candidates") one output; a Full layer's "return" also yields
    the merged selection ``[T, 1, topk]`` int32; a Shared layer's "given"
    takes it."""
    shape = SERVED[name]
    op = _am(name, selection)
    num, flat = _fake_call(op, _am_args(shape, selection))
    assert flat[0].shape == (num, shape["heads"] * HEAD_DIM) and flat[0].dtype == BF16
    if selection == "return":
        assert len(flat) == 2
        assert flat[1].shape == (num, 1, TOPK) and flat[1].dtype == I32
    else:
        assert len(flat) == 1


def test_attend_merge_schemas():
    for name in ("c16", "c48"):
        for selection, n in (("candidates", 1), ("return", 2), ("given", 1)):
            assert _schema_outputs(_am(name, selection)) == n, (name, selection)


@pytest.mark.parametrize("selection", ["return", "given"])
@pytest.mark.parametrize("name", ["c16", "c48"])
def test_attend_merge_dynamo(name, selection):
    """Dynamo traces a Full layer's "return" call (output and selection
    unpacked) and a Shared layer's "given" call on meta tensors with a
    dynamic token dim."""
    shape = SERVED[name]
    op = _am(name, selection)
    args = _am_args(shape, selection)(TOKENS)
    meta = [a.to("meta") for a in args]
    torch._dynamo.mark_dynamic(meta[1], 1)
    torch._dynamo.mark_dynamic(meta[3], 0)
    torch._dynamo.reset()
    if selection == "return":

        def fn(*xs):
            out, sel = op(*xs)
            return out + 1, sel + 1

    else:

        def fn(*xs):
            return op(*xs) + 1

    outs = torch.compile(fn, backend="eager", fullgraph=True)(*meta)
    flat = _check(op, outs, TOKENS)
    assert flat[0].shape == (TOKENS, shape["heads"] * HEAD_DIM)


# ------------------------------------------------------ Shared-layer writes
@pytest.mark.parametrize("name", sorted(SERVED))
def test_write_op_fake_symbolic(name):
    shape = SERVED[name]
    fp4, g = bool(shape["fp4"]), shape["g"]
    op = _keep(
        ("write", name),
        lambda: _ops()._get_msa_dcp_write_op(shape["dcp"], f"fake_{name}", fp4),
    )
    kv, index, scale = _kv(shape), _index(shape), _scale(shape)
    rank = _empty((1,), I32)

    def make(num):
        rows = (
            _empty((num, g, HEAD_DIM), BF16),
            _empty((num, g, HEAD_DIM), BF16),
            _empty((num, HEAD_DIM), BF16),
        )
        if fp4:
            q = _empty((num, shape["heads"], HEAD_DIM), BF16)
            return (*rows, kv, index, scale, *_meta(shape), rank, q)
        return (*rows, kv, index, *_meta(shape), rank)

    mode, seed = _mode()
    with mode:
        num = seed.shape[0]
        args = tuple(mode.from_tensor(a) for a in make(num))
        outs = op(*args)
    assert len(outs) == _schema_outputs(op) == (4 if fp4 else 2)
    assert outs[0].shape == kv.shape and outs[1].shape == index.shape
    if fp4:
        assert outs[2].shape == scale.shape
        assert outs[3].shape == (num, shape["heads"], HEAD_DIM)
        assert isinstance(outs[3].shape[0], torch.SymInt)


# --------------------------------------------- head-parallel op (c64), as is
def test_heads_op_fake_symbolic():
    """The c64 op (4 KV heads per DCP 8 group); a Shared layer calls it with
    the Full layer's candidates."""
    shape = SERVED["c64"]
    op = _keep(
        "heads",
        lambda: _ops()._get_msa_dcp_heads_op(SCALE, TOPK, 8, 2, 4, "fake_c64"),
    )
    num, flat = _fake_call(
        op,
        lambda n: (
            _empty((1,), I32),
            _empty((n, shape["heads"], HEAD_DIM), BF16),
            _kv(shape),
            _scale(shape),
            _empty((n, shape["g"] * 2 * TOPK), torch.float32),
            *_meta(shape),
            _empty((3,), I32),
        ),
    )
    assert flat[0].shape == (num, shape["heads"], HEAD_DIM)


# ---------------------------------------- the whole DCP op: Full then Shared
class _Group:
    """Shape model of a vLLM group coordinator."""

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


@pytest.fixture(scope="module")
def dcp_op():
    from vllm_torchtpu.distributed import dcp as dcp_mod

    msa_ops = _ops()
    real = dcp_mod.get_dcp_group

    def build(name, reuse):
        if ("dcp", name, reuse) in _KEEP:
            return _KEEP["dcp", name, reuse]
        shape = SERVED[name]
        # Bound by the builder: the op keeps this stub group.
        dcp_mod.get_dcp_group = lambda: _Group(shape["dcp"])
        try:
            kwargs = {"index_reuse": True} if reuse else {}
            # Off and on take different scales: the same op name registered
            # twice in one process corrupts torch's op registry.
            op = msa_ops.get_msa_dcp_op(
                SCALE if reuse else SCALE / 2,
                TOPK,
                0,
                1,
                shape["dcp"],
                shape["reps"],
                fp4=bool(shape["fp4"]),
                decode_rows=4,
                **kwargs,
            )
        finally:
            dcp_mod.get_dcp_group = real
        _KEEP["dcp", name, reuse] = op
        return op

    yield build
    # Stub-bound ops out of the factory's cache (still referenced in _KEEP).
    msa_ops.get_msa_dcp_op.cache_clear()


def _layer_call(op, shape, num, reuse_kwargs):
    """One layer's DCP op call on decode-sized fake operands."""
    heads, fp4 = shape["heads"], shape["fp4"]
    q_shape = (num, heads, HEAD_DIM) if shape["g"] > 1 else (heads, num, HEAD_DIM)
    meta = _Metadata(*_meta(shape), _empty((3,), I32))
    extra = {"dcp_rank": _empty((1,), I32)}
    if fp4:
        extra["scale_cache"] = _scale(shape)
        extra["hadamard"] = _empty((HEAD_DIM, HEAD_DIM), BF16)
    return op(
        _empty(q_shape, BF16),
        _empty((num, 1, HEAD_DIM), BF16),
        _empty((num, 1, HEAD_DIM), BF16),
        _empty((num, 1, HEAD_DIM), BF16),
        _empty((num, HEAD_DIM), BF16),
        _kv(shape),
        _index(shape),
        meta,
        **extra,
        **reuse_kwargs,
    )


@pytest.mark.parametrize("name", sorted(SERVED))
def test_dcp_op_full_then_shared_symbolic(name, dcp_op):
    """Served decode path per conc: the Full layer leaves its selection (one
    KV head) or local candidates (head-parallel op) in the step's dict, the
    Shared layer consumes it; both outputs keep the symbolic token dim and
    match the reuse-off op's."""
    shape = SERVED[name]
    off = dcp_op(name, reuse=False)
    assert off.index_reuse is False
    on = dcp_op(name, reuse=True)
    assert on.index_reuse is True
    mode, seed = _mode()
    with mode:
        num = seed.shape[0]
        want = _layer_call(off, shape, num, {})
        step = {}
        full = _layer_call(on, shape, num, {"index_reuse": step, "full": True})
        key = "candidates" if shape["g"] > 1 else "selected"
        assert set(step) == {key}
        if key == "selected":
            assert step[key].shape == (num, 1, TOPK) and step[key].dtype == I32
        else:
            assert step[key].shape == (num, shape["g"] * 2 * TOPK)
        shared = _layer_call(on, shape, num, {"index_reuse": step, "full": False})
    for got in (full, shared):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert any(isinstance(d, torch.SymInt) for d in got.shape)


def test_dcp_op_refuses_unsupported_path(dcp_op):
    """bf16 caches over several KV heads per DCP group (not served): reuse is
    refused at build time and the op ignores the step's dict."""
    SERVED["c64_bf16"] = dict(SERVED["c64"], fp4=0)
    try:
        op = dcp_op("c64_bf16", reuse=True)
        assert op.index_reuse is False
    finally:
        del SERVED["c64_bf16"]


def test_dcp_op_split_scoring_on_reuse_step(monkeypatch):
    """Served c64 layout (DCP 4, two KV heads per group) with the split index
    scoring (TPU_MINIMAX_M3_MSA_INDEX_SPLIT_MAX_TOKENS) and a shared scale
    row: on an index-reuse step the Full layer takes the split ops and leaves
    its candidates in the step's dict, the Shared layer consumes them; outputs
    match the reuse-off op's (symbolic token dim)."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.distributed import dcp as dcp_mod

    msa_ops = _ops()
    shape = dict(dcp=4, heads=8, reps=2, g=2, seqs=64, pps=1024, pages=256, fp4=1)
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_MSA_INDEX_SPLIT_MAX_TOKENS", 512, raising=False)
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_MSA_INDEX_SPLIT_MIN_TOKENS", 1, raising=False)
    monkeypatch.setattr(dcp_mod, "get_dcp_group", lambda: _Group(shape["dcp"]))
    monkeypatch.setattr(msa_ops, "get_tp_group", lambda: _Group(8))
    calls = []
    real_split = msa_ops._get_msa_dcp_split_ops

    def spy(*a, **kw):
        score, merge = real_split(*a, **kw)

        def score_spy(*args):
            calls.append("split")
            return score(*args)

        return score_spy, merge

    monkeypatch.setattr(msa_ops, "_get_msa_dcp_split_ops", spy)
    ops = {}
    for reuse in (False, True):
        ops[reuse] = msa_ops.get_msa_dcp_op(
            SCALE / (5 if reuse else 7), TOPK, 0, 1, shape["dcp"], shape["reps"],
            fp4=True, decode_rows=4, scale_row0=4,
            **({"index_reuse": True} if reuse else {}),
        )  # fmt: skip
    msa_ops.get_msa_dcp_op.cache_clear()
    assert ops[True].index_reuse is True
    mode, seed = _mode()
    with mode:
        num = seed.shape[0]
        lane = {"dcp_lane": _empty((1,), I32)}
        want = _layer_call(ops[False], shape, num, lane)
        assert calls == ["split"]
        step = {}
        full = _layer_call(ops[True], shape, num, {"index_reuse": step, "full": True, **lane})
        assert calls == ["split", "split"]
        assert set(step) == {"candidates"}
        assert step["candidates"].shape == (num, shape["g"] * 2 * TOPK)
        shared = _layer_call(ops[True], shape, num, {"index_reuse": step, "full": False, **lane})
        assert calls == ["split", "split"]  # the Shared layer scores nothing
    for got in (full, shared):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert any(isinstance(d, torch.SymInt) for d in got.shape)
