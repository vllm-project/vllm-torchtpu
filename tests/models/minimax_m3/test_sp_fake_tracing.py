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
"""The MiniMax-M3 sequence-parallel jax_ops (TPU_MINIMAX_M3_SP_*) as torch_tpu
``jax_op``s: their fakes fit the op schemas with a symbolic token count under
``FakeTensorMode`` and under ``torch.compile(fullgraph=True)`` over meta
tensors (as Dynamo runs them in the runner's warmup). Nothing lowers or runs."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

pytest.importorskip("torch_tpu")

TP, HIDDEN, K, N = 4, 256, 128, 256
TOKENS = 64  # hint of the symbolic token count
E4M3, BF16, F32 = torch.float8_e4m3fn, torch.bfloat16, torch.float32
EPS = 1.25e-6  # unique op name in this file (torch's op registry)


def _sp_o_proj():
    from vllm_torchtpu.models.minimax_m3 import sp_o_proj

    return sp_o_proj


def _cases():
    """(name, op, args builder from the token count, expected outputs)."""
    from vllm_torchtpu.models.minimax_m3 import quant
    from vllm_torchtpu.models.minimax_m3.ops.sp_norm import get_add_norm_quant_op

    sp = _sp_o_proj()
    e = torch.empty
    return {
        "quantize_rows_pow2": (
            sp._quantize_op,
            lambda t: [e((t, K), dtype=BF16)],
            lambda t: [((t, K), E4M3), ((t, 1), F32)],
        ),
        "o_proj_e4m3": (
            sp._o_proj_e4m3_op,
            lambda t: [
                e((TP, t, K), dtype=E4M3),
                e((TP, t, 1), dtype=F32),
                e((TP, K, N), dtype=E4M3),
                e((N,), dtype=BF16),
            ],
            lambda t: [((t, N), BF16)],
        ),
        "e4m3_rows_matmul": (
            quant._e4m3_matmul,
            lambda t: [
                e((t, HIDDEN), dtype=E4M3),
                e((t, 1), dtype=F32),
                e((HIDDEN, N), dtype=E4M3),
                e((N,), dtype=BF16),
            ],
            lambda t: [((t, N), BF16)],
        ),
        "add_norm_quant": (
            get_add_norm_quant_op(EPS),
            lambda t: [e((t, HIDDEN), dtype=BF16), e((t, HIDDEN), dtype=BF16), e((HIDDEN,), dtype=F32)],
            lambda t: [((t, HIDDEN), BF16), ((t, HIDDEN), E4M3), ((t, 1), F32)],
        ),
    }


def _outs(out):
    return list(out) if isinstance(out, (tuple, list)) else [out]


@pytest.mark.parametrize(
    "name", ["quantize_rows_pow2", "o_proj_e4m3", "e4m3_rows_matmul", "add_norm_quant"]
)
def test_fake_symbolic(name):
    op, args, want = _cases()[name]
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(TOKENS),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    with mode:
        t = seed.shape[0]
        assert isinstance(t, torch.SymInt)
        outs = _outs(op(*[mode.from_tensor(a) for a in args(t)]))
    assert [(tuple(o.shape), o.dtype) for o in outs] == want(t)
    assert isinstance(outs[0].shape[0], torch.SymInt)


@pytest.mark.parametrize(
    "name", ["quantize_rows_pow2", "o_proj_e4m3", "e4m3_rows_matmul", "add_norm_quant"]
)
def test_dynamo_meta(name):
    op, args, want = _cases()[name]
    meta = [a.to("meta") for a in args(TOKENS)]
    # The token dim: axis 1 of the [TP, T, ...] o_proj operands, else axis 0.
    for a in meta:
        if a.dim() >= 2:
            torch._dynamo.mark_dynamic(a, 1 if a.dim() == 3 else 0)
    torch._dynamo.reset()
    outs = _outs(torch.compile(lambda *xs: op(*xs), backend="eager", fullgraph=True)(*meta))
    assert [(tuple(o.shape), o.dtype) for o in outs] == want(TOKENS)
