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
"""The single-KV-head DCP attend + merge SPMD program (additional_config
minimax_dcp_attend_merge, `kernels/msa_dcp_merge.py`) against per-rank
`msa_attend_dcp` and an f64 log-sum-exp merge, on CPU: a 2-device host
platform in a subprocess (the device count is fixed at JAX start-up)."""

import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_merge


def _merge_ref(parts, lses, hadamard=None):
    """``parts [R, H, T, D]``, ``lses [R, T, H]`` -> f64 ``[T, H * D]``."""
    parts = np.asarray(parts, np.float64)
    lses = np.asarray(lses, np.float64)
    m = lses.max(axis=0)
    m = np.where(np.isfinite(m), m, 0.0)
    w = np.where(np.isfinite(lses), np.exp(lses - m), 0.0)
    w = w / np.maximum(w.sum(axis=0), 1e-30)  # [R, T, H]
    merged = np.einsum("rhtd,rth->thd", parts, w)
    if hadamard is not None:
        merged = merged @ np.asarray(hadamard, np.float64) / merged.shape[-1]
    return merged.reshape(merged.shape[0], -1)


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("num_tokens", [5, 300])
def test_merge_partials(num_tokens, rotated):
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    rng = np.random.default_rng(num_tokens)
    dcp, heads, d = 2, 8, 128
    parts = jnp.asarray(rng.standard_normal((dcp, heads, num_tokens, d)), jnp.bfloat16)
    lses = rng.standard_normal((dcp, num_tokens, heads)).astype(np.float32) * 3
    lses[0, 0] = -np.inf  # rank 0 attended nothing for token 0
    parts = parts.at[0, :, 0].set(0)
    lses[:, -1] = -np.inf  # no rank attended the last token
    parts = parts.at[:, :, -1].set(0)
    hadamard = (
        jnp.asarray(msa_fp4.hadamard(d), jnp.bfloat16) if rotated else None
    )
    got = msa_dcp_merge.merge_partials(
        parts, jnp.asarray(lses), hadamard, block_tokens=128, interpret=True
    )
    want = _merge_ref(parts.astype(jnp.float32), lses, hadamard)
    got = np.asarray(got.astype(jnp.float32))
    assert np.all(np.isfinite(got))
    # One bf16 rounding of the merge (and one more of the unrotation).
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)
    assert np.all(got[-1] == 0)


_SCRIPT = r'''
import os, sys
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
sys.path[:0] = [os.getcwd()]
from tests.models.minimax_m3 import test_msa_dcp as t
from tests.models.minimax_m3.packing import pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_dcp_merge, msa_write

msa_write.write_rows = t._xla_write_rows
small = sys.argv[1] == "small"
if not small:
    msa_dcp_merge.PERMUTE_MAX_TOKENS = 0
dcp, dtype = 2, jnp.bfloat16
q_lens, ctx_lens, topk = ([1, 130, 3, 1], [1100, 200, 0, 40], 4)
rng = np.random.default_rng(0)
logical = -(-max(np.add(q_lens, ctx_lens)) // (t.PAGE * dcp))
num_tokens, seq_lens, cu_q, tables, _ = t._batch(rng, q_lens, ctx_lens, dcp, logical)
kv_rows = rounded(rng.standard_normal((dcp, t.PAGES, t.PAGE, 2 * t.HEAD_DIM)), dtype)
index_rows = rounded(rng.standard_normal((dcp, t.PAGES, t.PAGE, t.HEAD_DIM)), dtype)
bf16 = lambda *s: jnp.asarray(rng.standard_normal(s)).astype(jnp.bfloat16)
q = bf16(num_tokens, t.HEADS, t.HEAD_DIM)
k, v = bf16(num_tokens, 1, t.HEAD_DIM), bf16(num_tokens, 1, t.HEAD_DIM)
index_q, index_k = bf16(num_tokens, 1, t.HEAD_DIM), bf16(num_tokens, t.HEAD_DIM)
dist = jnp.asarray([0, 0, len(q_lens)], jnp.int32)
meta = (jnp.asarray(seq_lens), jnp.asarray(tables.reshape(-1)), jnp.asarray(cu_q))
caches, cands = [], []
for r in range(dcp):
    kv, _, cand = msa_dcp.msa_write_and_candidates_dcp(
        k, v, index_q, index_k, pack(kv_rows[r], dtype), pack(index_rows[r], dtype),
        *meta, dist, rank=r, dcp=dcp, interpret=True, topk=topk, init_blocks=0,
        local_blocks=1)
    caches.append(kv)
    cands.append(cand)
stacked = jnp.stack(cands)
hl = t.HEADS // dcp
want = []
for r in range(dcp):
    out, lse = msa_dcp.msa_attend_dcp(q, caches[r], stacked, *meta, sm_scale=t.SCALE,
        topk=topk, rank=r, dcp=dcp, interpret=True)
    want.append((np.asarray(out.astype(jnp.float32))[:, :num_tokens],
                 np.asarray(lse)[:, :num_tokens]))
parts = np.stack([w[0] for w in want])  # [R, H, T, D]
lses = np.stack([w[1] for w in want])  # [R, H, T]
ref = []
for r in range(dcp):
    heads = slice(r * hl, (r + 1) * hl)
    p = parts[:, heads]
    l = lses[:, heads].transpose(0, 2, 1)
    m = l.max(0); m = np.where(np.isfinite(m), m, 0.0)
    w = np.where(np.isfinite(l), np.exp(l - m), 0.0)
    w = w / np.maximum(w.sum(0), 1e-30)
    ref.append(np.einsum("rhtd,rth->thd", p, w).reshape(num_tokens, -1))
ref = np.concatenate(ref)

mesh = Mesh(np.array(jax.devices()[:dcp]), ("dcp",))
sh, rep = P("dcp"), P()
body = jax.shard_map(
    lambda *a: msa_dcp_merge.attend_and_merge(*a, axis="dcp", dcp=dcp,
        sm_scale=t.SCALE, topk=topk, interpret=True),
    mesh=mesh, in_specs=(sh,) * 4 + (rep,) * 3, out_specs=sh, check_vma=False)
got = body(
    jnp.arange(dcp, dtype=jnp.int32),
    jnp.concatenate([q[:, r * hl:(r + 1) * hl] for r in range(dcp)]),
    jnp.concatenate(caches),
    jnp.concatenate(cands),
    *meta)
got = np.asarray(got.astype(jnp.float32))
num_real = sum(q_lens)
real = np.concatenate([np.arange(num_real) + r * num_tokens for r in range(dcp)])
assert np.all(np.isfinite(got)), "non-finite"
err = np.abs(got[real] - ref[real]).max()
assert err < 2e-2, err
print("OK", err)
'''


@pytest.mark.parametrize("path", ["small", "large"])
def test_attend_and_merge_matches_per_rank_attention(path):
    """Permutes (small) and XLA collectives (large) paths: the merged output
    of each rank's heads within bf16 rounding of the f64 merge of per-rank
    partials (the merge rounds once)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
        PYTHONPATH=f"{root}:{root}/src",
    )
    res = subprocess.run(
        [sys.executable, "-c", _SCRIPT, path],
        cwd=root, env=env, capture_output=True, text=True, timeout=1500,
    )
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-4000:]
