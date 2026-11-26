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
"""SP halves (models/minimax_m3/attention.py StepHalves) on the DCP 2 MSA path
over one KV head: the first token half's write + candidates + attend-and-merge,
then the second half's on the caches the first left, in one SPMD program with
the caches donated, equal the whole step's per token (fp8 and 4-bit caches,
prefill-sized steps). Interpret mode on 2 host devices:
``XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest ...``."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tests.models.minimax_m3.packing import pack_index_fp4
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4

PAGE, HEAD_DIM, TOPK, DCP, HL = 128, 128, 16, 2, 8
SM_SCALE = HEAD_DIM**-0.5


def _split(meta, split):
    """step_halves.split_metadata on (seq_lens, block_tables, cu, dist)."""
    seq_lens, tables, qsl, dist = meta
    first_qsl = jnp.minimum(qsl, split)
    second_qsl = qsl - first_qsl
    first_q = first_qsl[1:] - first_qsl[:-1]
    tail = second_qsl[1:] - second_qsl[:-1]
    first = ((first_q > 0) * (seq_lens - tail)).astype(seq_lens.dtype)
    second = ((tail > 0) * seq_lens).astype(seq_lens.dtype)
    return (first, tables, first_qsl, dist), (second, tables, second_qsl, dist)


def _caches(rng, num_pages, fp4):
    if fp4:
        kv = msa_fp4.pack_rows(
            jnp.asarray(rng.integers(0, 16, (num_pages, PAGE, 2 * HEAD_DIM)), jnp.uint8)
        )
        index = pack_index_fp4(
            jnp.asarray(rng.integers(0, 16, (num_pages, PAGE, HEAD_DIM)), jnp.uint8),
            jnp.asarray(rng.uniform(0.5, 1.5, (num_pages, HEAD_DIM)), jnp.float32),
        )
        scales = jnp.asarray(
            rng.uniform(0.5, 1.5, (num_pages, msa_fp4.scale_rows(1), PAGE)), jnp.float32
        )
        return kv, index, scales
    dt = jnp.float8_e4m3fn
    kv = jnp.asarray(rng.standard_normal((num_pages, PAGE // 4, 4, 2 * HEAD_DIM)), dt)
    index = jnp.asarray(rng.standard_normal((num_pages, PAGE // 4, 4, HEAD_DIM)), dt)
    return kv, index, jnp.zeros((1,), jnp.float32)


def _program(fp4, split, halves):
    mesh = Mesh(np.asarray(jax.devices()[:DCP]), ("dcp",))
    hadamard = jnp.asarray(msa_fp4.hadamard(HEAD_DIM), jnp.bfloat16)

    def attend(rank, x, kv, index, scales, q, meta, prefill=None):
        k, v, iq, ik = x
        res = msa_dcp.msa_write_and_candidates_dcp(
            k, v, iq, ik, kv, index, *meta, topk=TOPK, init_blocks=1, local_blocks=1,
            rank=rank, dcp=DCP, scale_cache=scales if fp4 else None, prefill=prefill,
            interpret=True,
        )  # fmt: skip
        kv, index, *scales_out, cand = res
        scales = scales_out[0] if fp4 else scales
        extra = (scales, hadamard) if fp4 else ()
        out = msa_dcp.attend_and_merge(
            rank, q, kv, cand, *meta[:3], *extra, axis="dcp", dcp=DCP,
            sm_scale=SM_SCALE, topk=TOPK, interpret=True,
        )  # fmt: skip
        return out, kv, index, scales

    def body(rank, k, v, iq, ik, kv, index, scales, q, *meta):
        x = (k, v, iq, ik)
        if not halves:
            out, kv, index, scales = attend(rank, x, kv, index, scales, q, meta)
            return out, kv, index, scales
        first, second = _split(meta, split)
        # Each half takes the whole step's candidate path (ops/msa.py
        # `step_tokens`), here the prefill one, whatever its own rows.
        half = functools.partial(attend, prefill=True)
        out_a, kv, index, scales = half(
            rank, tuple(t[:split] for t in x), kv, index, scales, q[:, :split], first
        )
        # The second half's writes follow the first half's reads (model.py).
        edge = jnp.nan_to_num(out_a[:1, :1].astype(jnp.float32)) * 0
        x = (x[0][split:] + edge.astype(x[0].dtype),) + tuple(t[split:] for t in x[1:])
        out_b, kv, index, scales = half(
            rank, x, kv, index, scales, q[:, split:], second
        )
        return jnp.concatenate([out_a, out_b]), kv, index, scales

    sh, rep = P("dcp"), P()
    f = jax.shard_map(
        body, mesh=mesh, in_specs=(sh,) + (rep,) * 4 + (sh,) * 4 + (rep,) * 4,
        out_specs=(sh,) * 4, check_vma=False,
    )  # fmt: skip
    return jax.jit(f, donate_argnums=(5, 6, 7))


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [
        ([512], [2900]),  # one chunk straddling the split
        # decode rows in the first half only; a half's 256 rows alone would
        # take the decode candidate path (256 <= 4 x 64)
        ([8, 8, 496], [3000, 5000, 2900]),
    ],
)
def test_halves_match_whole_step(q_lens, ctx_lens, fp4):
    if len(jax.devices()) < DCP:
        pytest.skip(
            "needs 2 devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)"
        )
    num_tokens, max_num_seqs = 512, len(q_lens) + 1
    # More rows than decode tiles: the prefill (class) candidate path.
    assert num_tokens > max_num_seqs * 64
    rng = np.random.default_rng(len(q_lens) + 2 * fp4)
    pages_per_seq = -(-(max(ctx_lens) + max(q_lens)) // (DCP * PAGE)) + 1
    num_pages = max_num_seqs * pages_per_seq + 1
    tables = rng.permutation(np.arange(1, num_pages))[: max_num_seqs * pages_per_seq]
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[: len(q_lens)] = np.array(ctx_lens) + np.array(q_lens)
    cu = np.full(max_num_seqs + 1, sum(q_lens), np.int32)
    cu[: len(q_lens) + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    dist = np.array([2 * (len(q_lens) > 1)] * 2 + [len(q_lens)], np.int32)
    meta = tuple(jnp.asarray(a) for a in (seq_lens, tables.astype(np.int32), cu, dist))
    rows = [
        jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    ] + [jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)]
    q = jnp.asarray(rng.standard_normal((DCP * HL, num_tokens, HEAD_DIM)), jnp.bfloat16)
    caches = [_caches(rng, num_pages, fp4) for _ in range(DCP)]

    def run(halves):
        cat = [jnp.concatenate([c[i] for c in caches]) for i in range(3)]
        f = _program(fp4, num_tokens // 2, halves)
        out = f(jnp.arange(DCP, dtype=jnp.int32), *rows, *cat, q, *meta)[0]
        # Rank-major [DCP * T, HL * D] -> each rank's heads of the real tokens.
        return np.asarray(out, np.float32).reshape(DCP, num_tokens, -1)[:, :real]

    real = sum(q_lens)
    whole, halves = run(False), run(True)
    assert np.isfinite(whole).all()
    np.testing.assert_array_equal(halves, whole)
