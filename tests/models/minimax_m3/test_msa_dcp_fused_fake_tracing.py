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
"""The fused DCP decode attention op (TPU_MINIMAX_M3_DCP_FUSED_DECODE_MAX_TOKENS)
as a torch_tpu ``jax_op``: its fake fits the op schema with a symbolic token
count under ``FakeTensorMode`` and under ``torch.compile(fullgraph=True)``
over meta tensors (as Dynamo runs it in the runner's warmup), and the whole
DCP op takes it for small decode steps only when the switch is on (8/16-bit
caches as c4 / c16 serve, and 4-bit caches). Nothing lowers or runs."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

pytest.importorskip("torch_tpu")

SCALE = 128**-0.5 * 1.375  # unique op names in this file (torch's op registry)
HEAD_DIM = 128
TOPK = 16
HEADS = 8  # query heads per rank (TP 8)
SEQS, PPS, PAGES = 16, 1024, 256
I32, BF16 = torch.int32, torch.bfloat16
TOKENS = 16  # hint of the symbolic token count

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


def _meta():
    return _empty((SEQS,), I32), _empty((SEQS * PPS,), I32), _empty((SEQS + 1,), I32)


def _kv(fp4):
    if fp4:
        return _empty((PAGES, 16, 4, 2 * HEAD_DIM), torch.uint8)
    return _empty((PAGES, 32, 4, 2 * HEAD_DIM), torch.float8_e4m3fn)


def _fused(fp4):
    return _keep(
        ("fused", fp4),
        lambda: _ops()._get_msa_dcp_attend_merge_op(
            SCALE, TOPK, 2, fp4, f"fake_dattn{int(fp4)}", fused=True
        ),
    )


def _args(num, fp4):
    args = [
        _empty((1,), I32),
        _empty((HEADS, num, HEAD_DIM), BF16),
        _kv(fp4),
        _empty((num, 2 * TOPK), torch.float32),
        *_meta(),
    ]
    if fp4:
        args += [
            _empty((PAGES, 8, 128), torch.float32),
            _empty((HEAD_DIM, HEAD_DIM), BF16),
        ]
    return args


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "fp4"])
def test_fused_op_fake_symbolic(fp4):
    op = _fused(fp4)
    assert len(op._opoverload._schema.returns) == 1
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(TOKENS),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    with mode:
        num = seed.shape[0]
        assert isinstance(num, torch.SymInt)
        out = op(*[mode.from_tensor(a) for a in _args(num, fp4)])
    assert out.shape == (num, HEADS * HEAD_DIM) and out.dtype == BF16
    assert isinstance(out.shape[0], torch.SymInt)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "fp4"])
def test_fused_op_dynamo(fp4):
    op = _fused(fp4)
    meta = [a.to("meta") for a in _args(TOKENS, fp4)]
    torch._dynamo.mark_dynamic(meta[1], 1)
    torch._dynamo.mark_dynamic(meta[3], 0)
    torch._dynamo.reset()
    out = torch.compile(lambda *xs: op(*xs) + 1, backend="eager", fullgraph=True)(*meta)
    assert out.shape == (TOKENS, HEADS * HEAD_DIM)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "fp4"])
def test_fused_op_fake_symbolic_max_free(fp4, monkeypatch):
    """With the max-free softmax configured (TPU_MINIMAX_M3_MSA_MAX_FREE, the
    MXU mask too) the op's fake still fits its schema symbolically."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse

    monkeypatch.setattr(msa_attention_lse, "MAX_FREE", True)
    monkeypatch.setattr(msa_attention_lse, "MXU_MASK", True)
    op = _keep(
        ("fused_mf", fp4),
        lambda: _ops()._get_msa_dcp_attend_merge_op(
            SCALE * 1.75, TOPK, 2, fp4, f"fake_dattn_mf{int(fp4)}", fused=True
        ),
    )
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(TOKENS),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    with mode:
        num = seed.shape[0]
        out = op(*[mode.from_tensor(a) for a in _args(num, fp4)])
    assert out.shape == (num, HEADS * HEAD_DIM) and out.dtype == BF16
    assert isinstance(out.shape[0], torch.SymInt)


def test_fused_rejects_selection_modes():
    with pytest.raises(ValueError, match="candidates"):
        _ops()._get_msa_dcp_attend_merge_op(
            SCALE * 1.5, TOPK, 2, False, "fake_dattn_bad", "given", fused=True
        )


# ------------------------------------------------- the whole DCP op
class _Group:
    world_size = 2

    def all_gather(self, x, dim):
        return torch.cat([x] * 2, dim=dim)


class _Metadata:
    def __init__(self):
        self.seq_lens, self.block_tables, self.query_start_loc = _meta()
        self.request_distribution = _empty((3,), I32)


class _Recorder:
    """The fused op, recording the token count of each call."""

    def __init__(self, op):
        self.op, self.calls = op, []

    def __call__(self, *args):
        self.calls.append(args[1].shape[1])
        return self.op(*args)


def _build_dcp_op(fp4, on, monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.distributed import dcp as dcp_mod

    key = ("dcp", fp4, on)
    if key in _KEEP:
        return _KEEP[key]
    msa_ops = _ops()
    built = []
    real = msa_ops._get_msa_dcp_attend_merge_op

    def spy(*a, **kw):
        op = real(*a, **kw)
        if kw.get("fused"):
            op = _Recorder(op)
            built.append(op)
        return op

    monkeypatch.setattr(msa_ops, "_get_msa_dcp_attend_merge_op", spy)
    monkeypatch.setattr(
        envs,
        "TPU_MINIMAX_M3_DCP_FUSED_DECODE_MAX_TOKENS",
        32 if on else 0,
        raising=False,
    )
    monkeypatch.setattr(dcp_mod, "get_dcp_group", lambda: _Group())
    op = msa_ops.get_msa_dcp_op(
        SCALE * (2.0 + 0.25 * fp4 + 0.5 * on), TOPK, 0, 1, 2, 2, fp4=fp4, decode_rows=8
    )
    _KEEP[key] = op, built
    return op, built


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "fp4"])
@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_dcp_op_takes_fused_call(fp4, on, monkeypatch):
    """Off: no fused op is built. On: the DCP op calls it for decode steps
    of at most 32 rows in whole 8-row tiles and the attend+merge op
    otherwise; the output keeps its shape."""
    op, built = _build_dcp_op(fp4, on, monkeypatch)
    assert len(built) == (1 if on else 0)
    from vllm_torchtpu.distributed import dcp as dcp_mod

    monkeypatch.setattr(dcp_mod, "get_dcp_group", lambda: _Group())

    def run(num):
        with FakeTensorMode():
            extra = {"dcp_rank": _empty((1,), I32)}
            if fp4:
                extra["scale_cache"] = _empty((PAGES, 8, 128), torch.float32)
                extra["hadamard"] = _empty((HEAD_DIM, HEAD_DIM), BF16)
            args = [
                _empty((HEADS, num, HEAD_DIM), BF16),
                _empty((num, 1, HEAD_DIM), BF16),
                _empty((num, 1, HEAD_DIM), BF16),
                _empty((num, 1, HEAD_DIM), BF16),
                _empty((num, HEAD_DIM), BF16),
                _kv(fp4),
                _empty((PAGES, 17, 4, HEAD_DIM), torch.uint8)
                if fp4
                else _empty((PAGES, 32, 4, HEAD_DIM), torch.float8_e4m3fn),
            ]
            return op(*args, _Metadata(), **extra)

    for num in (8, 16, 32, 40, 48):
        out = run(num)
        assert out.shape == (num, HEADS * HEAD_DIM)
    if on:
        assert built[0].calls == [8, 16, 32]
