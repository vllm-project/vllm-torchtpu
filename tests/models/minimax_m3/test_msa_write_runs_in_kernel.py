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
"""Token runs found in the MSA write kernel
(TPU_MINIMAX_M3_MSA_WRITE_RUNS_IN_KERNEL_MAX_TOKENS), interpret mode: the
4-bit and fp8 writes put the same cache bytes as with XLA's `token_runs`
(one / two KV heads, runs crossing a page, dropped tokens, a run cap)."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_fp4, msa_write

PAGE, D = 128, 128
ENV = "TPU_MINIMAX_M3_MSA_WRITE_RUNS_IN_KERNEL_MAX_TOKENS"


def _case(rng, heads, q_lens, starts, pages_total=24):
    """Tokens of requests with `q_lens` new rows from slot `starts`, each
    request's tokens in consecutive pages; one token dropped (page -1)."""
    pages, pos = [], []
    free = list(rng.permutation(np.arange(1, pages_total)))
    for q, s in zip(q_lens, starts):
        first = free.pop()
        second = free.pop()
        for i in range(q):
            p = s + i
            pages.append(first if p < PAGE else second)
            pos.append(p)
    pages = np.asarray(pages, np.int32)
    pages[1] = -1  # a token of another DCP rank
    kv = rng.integers(0, 256, (pages_total, PAGE // 8, 4, 2 * heads * D), np.uint8)
    idx = rng.integers(0, 256, (pages_total, PAGE // 8 + 1, 4, D), np.uint8)
    sc = rng.uniform(0.0, 1.0, (pages_total, msa_fp4.SCALE_ROWS, PAGE)).astype(
        np.float32
    )
    t = len(pos)
    k = rng.standard_normal((t, heads, D)) * rng.uniform(0.1, 4.0, (t, heads, 1))
    v = rng.standard_normal((t, heads, D))
    ik = rng.standard_normal((t, D))
    k[0, 0] = 0.0  # an all-zero vector: scale 0
    as_bf16 = lambda x: jnp.asarray(x, jnp.bfloat16)  # noqa: E731
    return (
        tuple(jnp.asarray(x) for x in (kv, idx, sc)),
        (as_bf16(k), as_bf16(v), as_bf16(ik)),
        jnp.asarray(pages),
        jnp.asarray(pos, jnp.int32),
    )


@pytest.mark.parametrize(
    "heads,q_lens,starts",
    [
        (1, [8, 8, 8], [3, 120, 64]),  # c48 decode, one run crossing a page
        (2, [8, 8], [0, 125]),  # c64 (two KV heads per rank)
        (1, [8] * 9, [8 * i for i in range(9)]),  # 72 tokens
    ],
)
def test_runs_in_kernel_bitwise(monkeypatch, heads, q_lens, starts):
    rng = np.random.default_rng(heads * 10 + len(q_lens))
    caches, (k, v, ik), pages, pos = _case(rng, heads, q_lens, starts)
    monkeypatch.setenv(ENV, "0")
    ref = msa.write_fp4(*caches, k, v, ik, pages, pos, max_num_seqs=len(q_lens) + 1,
                        interpret=True)  # fmt: skip
    monkeypatch.setenv(ENV, "512")
    got = msa.write_fp4(*caches, k, v, ik, pages, pos, max_num_seqs=len(q_lens) + 1,
                        interpret=True)  # fmt: skip
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(ref[0]), np.asarray(caches[0]))


def test_runs_in_kernel_fp8_and_cap():
    rng = np.random.default_rng(5)
    q_lens, starts = [8, 8, 8, 8], [3, 124, 40, 90]
    _, _, pages, pos = _case(rng, 1, q_lens, starts)
    t = pages.shape[0]
    kv = jnp.asarray(rng.standard_normal((24, PAGE // 4, 4, 2 * D)), jnp.float8_e4m3fn)
    rows = jnp.asarray(rng.standard_normal((t, 2 * D)), jnp.float8_e4m3fn)
    slots = pos % PAGE
    for max_runs in (16, 3):  # 3 < the 5 runs: later runs are not written
        ref = msa_write.write_runs([msa_write.Rows(kv, rows)], pages, slots,
                                   max_runs=max_runs, interpret=True)[0]  # fmt: skip
        got = msa_write.write_runs([msa_write.Rows(kv, rows)], pages, slots,
                                   max_runs=max_runs, interpret=True,
                                   runs_in_kernel=True)[0]  # fmt: skip
        np.testing.assert_array_equal(
            np.asarray(ref).view(np.uint8), np.asarray(got).view(np.uint8)
        )
