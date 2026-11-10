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
"""`msa_dcp_heads.attend_and_merge` (DCP 8 over 4 KV heads, 4-bit caches)
on 8 host devices against every rank's `msa_dcp.msa_attend_dcp` merged in
f32 (interpret mode). Needs 8 host devices:
``XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest ...``."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_dcp,
    msa_dcp_heads,
    msa_fp4,
)

PAGE, HEAD_DIM, TOPK, DCP, HEADS, Q_PER_KV = 128, 128, 16, 8, 4, 16
REPLICAS = DCP // HEADS
HL = HEADS * Q_PER_KV // DCP  # query heads per rank
SM_SCALE = HEAD_DIM**-0.5


def _step(rng, q_lens, ctx_lens, pages_per_seq, num_tokens):
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

    kvs, scs, cands = [], [], []
    x = [
        jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    ]
    ik = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)
    for rank in range(DCP):
        kv = msa_fp4.pack_rows(
            jnp.asarray(
                rng.integers(0, 16, (num_pages, PAGE, HEADS * 2 * HEAD_DIM)), jnp.uint8
            )
        )
        index = msa_fp4.pack_rows(
            jnp.asarray(rng.integers(0, 16, (num_pages, PAGE, HEAD_DIM)), jnp.uint8),
            jnp.asarray(rng.uniform(0.5, 1.5, (num_pages, HEAD_DIM)), jnp.float32),
        )
        sc = jnp.asarray(
            rng.uniform(0.5, 1.5, (num_pages, msa_fp4.scale_rows(HEADS), PAGE)),
            jnp.float32,
        )
        kv, _, sc, cand = msa_dcp.msa_write_and_candidates_dcp(
            x[0],
            x[1],
            x[2],
            ik,
            kv,
            index,
            *meta,
            topk=TOPK,
            init_blocks=1,
            local_blocks=1,
            rank=rank,
            dcp=DCP,
            scale_cache=sc,
            interpret=True,
        )
        kvs.append(kv), scs.append(sc), cands.append(cand)
    q = [
        jnp.asarray(rng.standard_normal((num_tokens, HL, HEAD_DIM)), jnp.bfloat16)
        for _ in range(DCP)
    ]
    return q, kvs, scs, cands, meta, num_real


def _reference(q, kvs, scs, cands, meta):
    q_all = jnp.concatenate(q, axis=1)
    cand_all = jnp.stack(cands)
    outs, lses = [], []
    for rank in range(DCP):
        out, lse = msa_dcp.msa_attend_dcp(
            q_all,
            kvs[rank],
            cand_all,
            *meta[:3],
            sm_scale=SM_SCALE,
            topk=TOPK,
            rank=rank,
            dcp=DCP,
            scale_cache=scs[rank],
            interpret=True,
        )
        outs.append(out.astype(jnp.float32)), lses.append(lse)
    lses = jnp.stack(lses)  # [dcp, H, T]
    w = jnp.nan_to_num(jnp.exp(lses - jax.nn.logsumexp(lses, axis=0)))
    merged = jnp.sum(w[..., None] * jnp.stack(outs), axis=0)  # [H, T, D]
    merged = msa_fp4.unrotate_output(merged).transpose(1, 0, 2)
    return [merged[:, r * HL : (r + 1) * HL] for r in range(DCP)]


def _spmd(q, kvs, scs, cands, meta, **kw):
    mesh = Mesh(np.asarray(jax.devices()[:DCP]), ("dcp",))
    body = functools.partial(
        msa_dcp_heads.attend_and_merge,
        axis="dcp",
        dcp=DCP,
        kv_head_replicas=REPLICAS,
        sm_scale=SM_SCALE,
        topk=TOPK,
        rs_out_precision=False,
        interpret=True,
        **kw,
    )
    sharded, rep = P("dcp"), P()
    f = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(sharded,) * 5 + (rep,) * 4,
        out_specs=sharded,
        check_vma=False,
    )
    ranks = jnp.arange(DCP, dtype=jnp.int32)
    out = f(ranks, *(jnp.concatenate(a) for a in (q, kvs, scs, cands)), *meta)
    return np.split(np.asarray(out, np.float32), DCP)


@pytest.mark.parametrize(
    "q_lens,ctx_lens,num_tokens,min_tokens,max_call,width",
    [
        # Two decodes and two prefills: the prefills (9 + 4 logical blocks)
        # take two rounds of a 12-block buffer.
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 320, 64, 2048, 12),
        # Kernel calls of 48 rows: several windows per round, the last one
        # clamped back over rows already attended.
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 320, 64, 48, 12),
        # An odd block-table width: the replica-split buffer pads to 12.
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 320, 64, 2048, 11),
        # The same step below the head-parallel size: DCP attention only.
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 320, 1 << 20, 2048, 12),
    ],
)
def test_attend_and_merge_matches_dcp_reference(
    q_lens, ctx_lens, num_tokens, min_tokens, max_call, width, monkeypatch
):
    if len(jax.devices()) < DCP:
        pytest.skip(
            "needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)"
        )
    monkeypatch.setattr(msa_attention, "MAX_TOKENS_PER_CALL", max_call)
    rng = np.random.default_rng(0)
    step = _step(rng, q_lens, ctx_lens, width, num_tokens)
    *args, num_real = step
    want = _reference(*args)
    got = _spmd(*args, decode_rows=8, head_parallel_min_tokens=min_tokens)
    for r in range(DCP):
        np.testing.assert_allclose(
            got[r][:num_real], np.asarray(want[r])[:num_real], rtol=2e-2, atol=2e-2
        )
