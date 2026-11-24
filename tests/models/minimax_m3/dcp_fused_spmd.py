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
"""Worker of `test_msa_dcp_fused.py` on a 2-device CPU platform: the fused
DCP decode attention call against `msa_dcp.attend_and_merge`, bit for bit.

    dcp_fused_spmd.py <fp8|fp4> <ctx,ctx,...> <padding rows> <control 0|1> [maxfree|shared]

``shared`` (4-bit): the fused call reads the layer's K/V scales at rows 6 / 7
of a scale page shared with three other layers (whose rows hold other
values; TPU_MINIMAX_M3_KV4_SHARED_SCALES) and must match the unfused call
over the layer's own page.

8 verify rows per request over the given contexts, then the padding rows.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention_lse,
    msa_dcp,
    msa_dcp_fused,
    msa_fp4,
)

PAGE, D, HL, TOPK, DCP, ROWS = 128, 128, 8, 16, 2, 8
SCALE = D**-0.5


def step(ctx, pad, rng):
    """``(num_tokens, num_real, (seq_lens, block_tables, cu_q_lens), pages)``."""
    reqs = len(ctx)
    max_seqs = max(reqs, 4)
    seq = np.add(ctx, ROWS)
    logical = -(-seq // (PAGE * DCP))
    pages = int(logical.sum()) + 2
    num_real = reqs * ROWS
    sl = np.zeros(max_seqs, np.int32)
    sl[:reqs] = seq
    cu = np.full(max_seqs + 1, num_real, np.int32)
    cu[: reqs + 1] = np.arange(reqs + 1) * ROWS
    tables = np.zeros((max_seqs, int(logical.max())), np.int32)
    perm = rng.permutation(np.arange(1, pages))
    at = 0
    for j in range(reqs):
        tables[j, : logical[j]] = perm[at : at + logical[j]]
        at += logical[j]
    meta = (jnp.asarray(sl), jnp.asarray(tables.reshape(-1)), jnp.asarray(cu))
    return num_real + pad, num_real, meta, pages


def caches(fp4, pages, rng):
    """Per-rank random ``(kv, index, scale)`` caches."""
    if not fp4:

        def f8(width):
            x = rng.standard_normal((pages, PAGE // 4, 4, width))
            return jnp.asarray(x).astype(jnp.float8_e4m3fn)

        return [(f8(2 * D), f8(D), None) for _ in range(DCP)]

    def nib(width):
        return jnp.asarray(rng.integers(0, 16, (pages, PAGE, width)))

    def uniform(*shape):
        return jnp.asarray(rng.uniform(0.05, 0.2, shape), jnp.float32)

    return [
        (
            msa_fp4.pack_rows(nib(2 * D)),
            msa_fp4.pack_rows(nib(D), scales=uniform(pages, PAGE)),
            uniform(pages, msa_fp4.scale_rows(1), PAGE),
        )
        for _ in range(DCP)
    ]


def main():
    modes = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else set()
    if "maxfree" in modes:
        # Both calls run the max-free softmax (TPU_MINIMAX_M3_MSA_MAX_FREE).
        msa_attention_lse.configure_max_free(0.0, enabled=True, mxu_mask=True)
    shared = "shared" in modes
    fp4 = sys.argv[1] == "fp4"
    ctx = [int(x) for x in sys.argv[2].split(",")]
    pad, control = int(sys.argv[3]), sys.argv[4] == "1"
    rng = np.random.default_rng(len(ctx) * 7 + pad)
    num_tokens, num_real, meta, pages = step(ctx, pad, rng)

    def bf16(*shape):
        return jnp.asarray(rng.standard_normal(shape)).astype(jnp.bfloat16)

    q = bf16(num_tokens, 2 * HL, D)
    k, v, iq = (bf16(num_tokens, 1, D) for _ in range(3))
    ik = bf16(num_tokens, D)
    dist = jnp.asarray([0, 0, len(ctx)], jnp.int32)
    kv, scale, cands = [], [], []
    for r, (kv_r, index_r, scale_r) in enumerate(caches(fp4, pages, rng)):
        out = msa_dcp.msa_write_and_candidates_dcp(
            k, v, iq, ik, kv_r, index_r, *meta, dist,
            topk=TOPK, init_blocks=1, local_blocks=1, rank=r, dcp=DCP,
            scale_cache=scale_r, interpret=True,
        )  # fmt: skip
        kv.append(out[0])
        scale.append(out[2] if fp4 else None)
        cands.append(out[-1])
    qh = (msa_fp4.rotate_queries(q) if fp4 else q).transpose(1, 0, 2)  # [2 HL, T, D]
    mesh = Mesh(np.array(jax.devices()[:DCP]), ("dcp",))
    sh, rep = P("dcp"), P()
    extra, extra_specs = (), ()
    if fp4:
        hadamard = jnp.asarray(msa_fp4.hadamard(D), jnp.bfloat16)
        extra, extra_specs = (jnp.concatenate(scale), hadamard), (sh, rep)
    base = (jnp.arange(DCP, dtype=jnp.int32), qh, jnp.concatenate(kv))

    def run(fn, cand, row0=0):
        def body(*a):
            if row0:
                a = (*a[:7], msa_fp4.with_row0(a[7], row0), *a[8:])
            return fn(
                *a, axis="dcp", dcp=DCP, sm_scale=SCALE, topk=TOPK, interpret=True
            )

        out = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(sh,) * 4 + (rep,) * 3 + extra_specs,
            out_specs=sh,
            check_vma=False,
        )(*base, cand, *meta, *extra)
        return np.asarray(out.astype(jnp.float32))

    cand = jnp.concatenate(cands)
    want = run(msa_dcp.attend_and_merge, cand)
    if shared:
        own = extra[0]  # [2 * pages, 8, S]: rows 0 / 1 hold this layer's scales
        page = jnp.asarray(rng.uniform(0.5, 3.0, own.shape), jnp.float32)
        page = page.at[:, 6:8].set(own[:, :2])
        extra = (page, extra[1])
        got = run(msa_dcp_fused.fused_attend_and_merge, cand, row0=6)
        assert np.array_equal(got, run(msa_dcp.attend_and_merge, cand, row0=6))
    else:
        got = run(msa_dcp_fused.fused_attend_and_merge, cand)
    rows = np.arange(DCP * num_tokens) % num_tokens
    assert np.isfinite(want).all(), "non-finite"
    assert np.abs(want[rows < num_real]).max() > 0, "all-zero output"
    differ = int((want != got).sum())
    assert differ == 0, f"differ {differ} of {want.size}"
    assert not got[rows >= num_real].any(), "padding rows"
    if control:
        # Rank 1 drops its candidates: the merged selection (so the output)
        # changes, and the fused call still matches the unfused one.
        c = np.asarray(cand).copy()
        c[num_tokens:, :TOPK] = -np.inf
        c[num_tokens:, TOPK:] = -1
        alt = run(msa_dcp_fused.fused_attend_and_merge, jnp.asarray(c))
        assert not np.array_equal(alt, got), "control"
        assert np.array_equal(alt, run(msa_dcp.attend_and_merge, jnp.asarray(c)))
    print("OK")


if __name__ == "__main__":
    main()
