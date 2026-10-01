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
"""Byte-exact tests of the Pallas indexer K-cache insert against the XLA scatter.

Both sides start from the same random cache and the full
``u8[425, 256, 4, 256]`` cache is compared byte for byte. The reference is the
jnp scatter expression; the kernel builds its group list by an independent
formula. A kernel fed positions shifted by one slot must mismatch, so the
comparison is shown able to fail.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.k_cache_insert import (
    KERNEL_MIN_TOKENS,
    insert_indexer_k_cache,
    insert_indexer_k_cache_auto,
    insert_indexer_k_cache_ref,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

N_PAGES, BPK, KP, D = 425, 256, 4, 256
BS = BPK * KP
PPS = 264  # pages per seq (>= 262144 / 1024 + margin)


def _make_case(T, qs, ctxs, seed):
    """qs: q_len per seq (0 allowed); ctxs: context length before this step."""
    S = len(qs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_PAGES).astype(np.int32)
    stride = max(1, N_PAGES // S) if S > 1 else 0
    pi = np.stack([perm[(s * stride + np.arange(PPS)) % N_PAGES] for s in range(S)])
    cu = np.concatenate([[0], np.cumsum(qs)]).astype(np.int32)
    assert cu[-1] <= T
    sl = np.array([c + q if q > 0 else 0 for c, q in zip(ctxs, qs)], np.int32)
    touched = set()
    for s in range(S):
        if qs[s] == 0:
            continue
        pages = {int(pi[s, p // BS]) for p in range(ctxs[s], ctxs[s] + qs[s])}
        assert not (pages & touched), "seqs share a touched page"
        touched |= pages
    k = rng.integers(0, 256, (T, D), dtype=np.uint8)
    return (
        jnp.asarray(k),
        jnp.asarray(sl),
        jnp.asarray(pi.reshape(-1)),
        jnp.asarray(cu),
    )


def _fresh_cache(seed):
    return jax.random.bits(jax.random.key(seed), (N_PAGES, BPK, KP, D), dtype=jnp.uint8)


_fresh = jax.jit(_fresh_cache, static_argnums=0)
_ndiff = jax.jit(lambda a, b: jnp.sum((a != b).astype(jnp.int32)))
_ref = jax.jit(insert_indexer_k_cache_ref, donate_argnums=0)
_kernel = jax.jit(insert_indexer_k_cache, donate_argnums=0)
_auto = jax.jit(insert_indexer_k_cache_auto, donate_argnums=0)


CASES = {
    # 1 seq, 1024-token prefill chunk at aligned / unaligned / page-edge starts.
    **{
        f"prefill_T1024_C{c}": (1024, [1024], [c])
        for c in (0, 1023, 131072, 131073, 261120)
    },
    # 1 real decode token at every slot % 4, 15 padding rows.
    **{
        f"decode_T16_1tok_slotmod{r}": (16, [1], [131072 + 4 * 7 + r]) for r in range(4)
    },
    "twoseq_T1024_q300@5001_q500@131073_pad224": (
        1024,
        [300, 500],
        [5001, 131073],
    ),
    "decode_T16_16seq": (16, [1] * 16, [100000 + 37 * s for s in range(16)]),
    "T64_emptyseqs_pad": (64, [7, 0, 30, 0], [2, 0, 4093, 0]),
    "T32_all_padding": (32, [0, 0], [0, 0]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_kernel_matches_scatter_byte_exact(name):
    T, qs, ctxs = CASES[name]
    seed = sum(map(ord, name)) % 1000
    k, sl, pi, cu = _make_case(T, qs, ctxs, seed)
    base = _fresh(seed)
    ref = _ref(_fresh(seed), k, sl, pi, cu)
    out = _kernel(_fresh(seed), k, sl, pi, cu)
    n_written = int(_ndiff(ref, base))
    assert int(_ndiff(out, ref)) == 0, f"{name}: kernel differs from scatter"
    if sum(qs) == 0:
        assert n_written == 0
        return
    assert n_written > 0
    # Positive control: positions shifted by one slot must be caught.
    shifted = _kernel(_fresh(seed), k, sl + (sl > 0).astype(sl.dtype), pi, cu)
    assert int(_ndiff(shifted, ref)) > 0, f"{name}: shifted insert not caught"


@pytest.mark.parametrize("T", [16, KERNEL_MIN_TOKENS, 1024])
def test_auto_matches_scatter(T):
    k, sl, pi, cu = _make_case(T, [T // 2, T // 4], [131073, 7001], T)
    ref = _ref(_fresh(T), k, sl, pi, cu)
    out = _auto(_fresh(T), k, sl, pi, cu)
    assert int(_ndiff(out, ref)) == 0
