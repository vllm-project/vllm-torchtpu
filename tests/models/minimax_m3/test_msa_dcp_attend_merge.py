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
"""`msa_dcp.attend_and_merge` (DCP 2 over one KV head, fp8 and 4-bit caches)
on 2 host devices against every rank's `msa_dcp.msa_attend_dcp` merged in
f32 (interpret mode). Needs 2 host devices:
``XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest ...``."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tests.models.minimax_m3.packing import pack
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4

PAGE, HEAD_DIM, TOPK, DCP, Q_PER_KV = 128, 128, 16, 2, 16
HL = Q_PER_KV // DCP  # query heads per rank
SM_SCALE = HEAD_DIM**-0.5


def _step(rng, q_lens, ctx_lens, pages_per_seq, num_tokens, fp4):
    num_seqs = len(q_lens)
    max_num_seqs = num_seqs + 1
    num_real = sum(q_lens)
    num_pages = max_num_seqs * pages_per_seq + 1
    block_tables = rng.permutation(np.arange(1, num_pages))[
        : max_num_seqs * pages_per_seq
    ].astype(np.int32)
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu = np.full(max_num_seqs + 1, num_real, np.int32)
    cu[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    dist = np.array([0, 0, num_seqs], np.int32)
    meta = tuple(jnp.asarray(a) for a in (seq_lens, block_tables, cu, dist))
    x = [
        jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    ]
    ik = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)
    kvs, scs, cands = [], [], []
    for rank in range(DCP):
        if fp4:
            kv = msa_fp4.pack_rows(
                jnp.asarray(
                    rng.integers(0, 16, (num_pages, PAGE, 2 * HEAD_DIM)), jnp.uint8
                )
            )
            index = msa_fp4.pack_rows(
                jnp.asarray(
                    rng.integers(0, 16, (num_pages, PAGE, HEAD_DIM)), jnp.uint8
                ),
                jnp.asarray(rng.uniform(0.5, 1.5, (num_pages, HEAD_DIM)), jnp.float32),
            )
            sc = jnp.asarray(
                rng.uniform(0.5, 1.5, (num_pages, msa_fp4.scale_rows(1), PAGE)),
                jnp.float32,
            )
            kv, _, sc, cand = msa_dcp.msa_write_and_candidates_dcp(
                *x, ik, kv, index, *meta, topk=TOPK, init_blocks=1, local_blocks=1,
                rank=rank, dcp=DCP, scale_cache=sc, interpret=True,
            )  # fmt: skip
        else:
            dt = jnp.float8_e4m3fn
            kv = pack(rng.standard_normal((num_pages, PAGE, 2 * HEAD_DIM)), dt)
            index = pack(rng.standard_normal((num_pages, PAGE, HEAD_DIM)), dt)
            sc = None
            kv, _, cand = msa_dcp.msa_write_and_candidates_dcp(
                *x, ik, kv, index, *meta, topk=TOPK, init_blocks=1, local_blocks=1,
                rank=rank, dcp=DCP, interpret=True,
            )  # fmt: skip
        kvs.append(kv), scs.append(sc), cands.append(cand)
    # Each rank's query heads, head-major (rotated for a 4-bit cache).
    q = [
        jnp.asarray(rng.standard_normal((HL, num_tokens, HEAD_DIM)), jnp.bfloat16)
        for _ in range(DCP)
    ]
    return q, kvs, scs, cands, meta, num_real


def _reference(q, kvs, scs, cands, meta, fp4):
    q_all = jnp.concatenate(q, axis=0)
    cand_all = jnp.stack(cands)
    outs, lses = [], []
    for rank in range(DCP):
        out, lse = msa_dcp.msa_attend_dcp(
            q_all, kvs[rank], cand_all, *meta[:3], sm_scale=SM_SCALE, topk=TOPK,
            rank=rank, dcp=DCP, scale_cache=scs[rank], q_head_major=True,
            interpret=True,
        )  # fmt: skip
        outs.append(out.astype(jnp.float32)), lses.append(lse)
    lses = jnp.stack(lses)  # [dcp, H, T]
    w = jnp.nan_to_num(jnp.exp(lses - jax.nn.logsumexp(lses, axis=0)))
    merged = jnp.sum(w[..., None] * jnp.stack(outs), axis=0)  # [H, T, D]
    if fp4:
        merged = msa_fp4.unrotate_output(merged)
    merged = merged.transpose(1, 0, 2)
    return [
        merged[:, r * HL : (r + 1) * HL].reshape(merged.shape[0], -1)
        for r in range(DCP)
    ]


def _spmd(q, kvs, scs, cands, meta, fp4):
    mesh = Mesh(np.asarray(jax.devices()[:DCP]), ("dcp",))
    body = functools.partial(
        msa_dcp.attend_and_merge,
        axis="dcp",
        dcp=DCP,
        sm_scale=SM_SCALE,
        topk=TOPK,
        interpret=True,
    )
    sharded, rep = P("dcp"), P()
    extra = ()
    specs = (sharded,) * 4 + (rep,) * 3
    if fp4:
        extra = (
            jnp.concatenate(scs),
            jnp.asarray(msa_fp4.hadamard(HEAD_DIM), jnp.bfloat16),
        )
        specs += (sharded, rep)
    f = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=sharded, check_vma=False
    )
    ranks = jnp.arange(DCP, dtype=jnp.int32)
    out = f(ranks, *(jnp.concatenate(a) for a in (q, kvs, cands)), *meta[:3], *extra)
    return np.split(np.asarray(out, np.float32), DCP)


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,num_tokens",
    [
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 320),  # mixed, padded tail
        ([8, 8], [3000, 5000], 16),  # decode
    ],
)
def test_attend_and_merge_matches_dcp_reference(q_lens, ctx_lens, num_tokens, fp4):
    if len(jax.devices()) < DCP:
        pytest.skip(
            "needs 2 devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)"
        )
    rng = np.random.default_rng(len(q_lens))
    *args, num_real = _step(rng, q_lens, ctx_lens, 40, num_tokens, fp4)
    want = _reference(*args, fp4)
    got = _spmd(*args, fp4)
    for r in range(DCP):
        np.testing.assert_allclose(
            got[r][:num_real], np.asarray(want[r])[:num_real], rtol=2e-2, atol=2e-2
        )
