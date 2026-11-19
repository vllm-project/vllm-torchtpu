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
"""The MSA ops whose JAX bodies the indexer switches change
(TPU_MINIMAX_M3_INDEXER_HOIST_METADATA, TPU_MINIMAX_M3_INDEXER_ROW_GATHER;
`msa_indexer.block_scores` on decode-sized steps) keep their schemas and
trace with fake tensors and a symbolic token count with both switches on:
the DCP op of the served c16 (bf16) and c48 (4-bit) configs under
``FakeTensorMode`` and under ``torch.compile(fullgraph=True)`` over meta
tensors, and the JAX functions the op schemas are derived from give the same
output shapes and dtypes (``jax.eval_shape``) with the switches off and on,
on a decode-sized step that takes the cond. Nothing lowers or runs.
"""

import jax
import jax.numpy as jnp
import pytest
import torch

from tests.models.minimax_m3 import test_msa_index_reuse_fake_tracing as base

pytest.importorskip("torch_tpu")

SWITCHES = (
    "TPU_MINIMAX_M3_INDEXER_HOIST_METADATA",
    "TPU_MINIMAX_M3_INDEXER_ROW_GATHER",
)


@pytest.fixture
def switches_on(monkeypatch):
    for name in SWITCHES:
        monkeypatch.setenv(name, "1")
    jax.clear_caches()
    yield
    jax.clear_caches()


def _dcp_op(name):
    """The served DCP op of `name` (built once per process: a jax_op name
    registered twice corrupts torch's op registry; this scale is used by no
    other test)."""
    from vllm_torchtpu.distributed import dcp as dcp_mod

    key = ("hoist_dcp", name)
    if key in base._KEEP:
        return base._KEEP[key]
    shape = base.SERVED[name]
    msa_ops = base._ops()
    real = dcp_mod.get_dcp_group
    dcp_mod.get_dcp_group = lambda: base._Group(shape["dcp"])
    try:
        op = msa_ops.get_msa_dcp_op(
            base.SCALE / 4,
            base.TOPK,
            0,
            1,
            shape["dcp"],
            shape["reps"],
            fp4=bool(shape["fp4"]),
            decode_rows=4,
        )
    finally:
        dcp_mod.get_dcp_group = real
        msa_ops.get_msa_dcp_op.cache_clear()
    base._KEEP[key] = op
    return op


@pytest.mark.parametrize("name", ["c16", "c48"])
def test_dcp_op_fake_symbolic(name, switches_on):
    shape = base.SERVED[name]
    op = _dcp_op(name)
    mode, seed = base._mode()
    with mode:
        num = seed.shape[0]
        out = base._layer_call(op, shape, num, {})
    assert out.dtype == base.BF16
    assert any(isinstance(d, torch.SymInt) for d in out.shape), out.shape


@pytest.mark.parametrize("name", ["c16", "c48"])
def test_dcp_op_dynamo(name, switches_on):
    """Dynamo traces the layer call on meta tensors with a dynamic token
    dim, fullgraph."""
    shape = base.SERVED[name]
    op = _dcp_op(name)
    n = base.TOKENS
    tensors = [
        base._empty((shape["heads"], n, base.HEAD_DIM), base.BF16),
        base._empty((n, 1, base.HEAD_DIM), base.BF16),
        base._empty((n, 1, base.HEAD_DIM), base.BF16),
        base._empty((n, 1, base.HEAD_DIM), base.BF16),
        base._empty((n, base.HEAD_DIM), base.BF16),
        base._kv(shape),
        base._index(shape),
        *base._meta(shape),
        base._empty((3,), base.I32),
        base._empty((1,), base.I32),
    ]
    if shape["fp4"]:
        tensors += [
            base._scale(shape),
            base._empty((base.HEAD_DIM, base.HEAD_DIM), base.BF16),
        ]
    meta = [t.to("meta") for t in tensors]
    torch._dynamo.mark_dynamic(meta[0], 1)
    for i in range(1, 5):
        torch._dynamo.mark_dynamic(meta[i], 0)

    def fn(q, k, v, iq, ik, kv, index, seq_lens, tables, qsl, dist, rank, *fp4):
        extra = {"dcp_rank": rank}
        if fp4:
            extra["scale_cache"], extra["hadamard"] = fp4
        md = base._Metadata(seq_lens, tables, qsl, dist)
        return op(q, k, v, iq, ik, kv, index, md, **extra) + 1

    torch._dynamo.reset()
    out = torch.compile(fn, backend="eager", fullgraph=True)(*meta)
    assert out.shape[0] == n or out.shape[1] == n


def _score_shapes(num_tokens, dcp, fp4):
    """`msa_write_and_candidates_dcp`'s output shapes and dtypes on a
    decode-sized step (``jax.eval_shape``; traces `block_scores`' cond)."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_indexer

    seqs, pps, pages, d = 16, 64, 256, base.HEAD_DIM
    assert not msa_indexer.is_prefill_step(num_tokens, seqs)
    s = jax.ShapeDtypeStruct
    if fp4:
        kv = s((pages, 16, 4, 2 * d), jnp.uint8)
        index = s((pages, 17, 4, d), jnp.uint8)
        scale = (s((pages, 2, 128), jnp.float32),)
    else:
        kv = s((pages, 64, 2, 2 * d), jnp.bfloat16)
        index = s((pages, 64, 2, d), jnp.bfloat16)
        scale = ()
    args = (
        s((num_tokens, 1, d), jnp.bfloat16),
        s((num_tokens, 1, d), jnp.bfloat16),
        s((num_tokens, 1, d), jnp.bfloat16),
        s((num_tokens, d), jnp.bfloat16),
        kv,
        index,
        s((seqs,), jnp.int32),
        s((seqs * pps,), jnp.int32),
        s((seqs + 1,), jnp.int32),
        s((3,), jnp.int32),
    )

    def score(*a):
        a, sc = a[:10], a[10:]
        return msa_dcp.msa_write_and_candidates_dcp(
            *a,
            topk=base.TOPK,
            init_blocks=0,
            local_blocks=1,
            rank=0,
            dcp=dcp,
            interpret=True,
            **({"scale_cache": sc[0]} if sc else {}),
        )

    out = jax.eval_shape(score, *args, *scale)
    return [(o.shape, o.dtype) for o in out]


@pytest.mark.parametrize("fp4", [False, True], ids=["bf16", "fp4"])
def test_score_function_shapes_unchanged(monkeypatch, fp4):
    """The schema source (the score op's JAX function) gives the same
    outputs with the switches on, on a decode-sized step with a prefill
    chunk's room (the cond path)."""
    for name in SWITCHES:
        monkeypatch.setenv(name, "0")
    jax.clear_caches()
    off = _score_shapes(200, 2, fp4)
    for name in SWITCHES:
        monkeypatch.setenv(name, "1")
    jax.clear_caches()
    on = _score_shapes(200, 2, fp4)
    assert on == off
