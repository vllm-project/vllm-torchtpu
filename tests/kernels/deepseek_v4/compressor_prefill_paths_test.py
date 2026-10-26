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
"""Correctness test for Kernel 2 (compress_norm_rope_store)."""
"""The compressor's prefill paths write exactly the bytes the per-token path does.

`proj_and_save_state`'s run-coalesced writes (``runs_min_tokens``) and the
prefill-block compress (``prefill_block``) are pure reorganizations of the same
arithmetic, so every cache array they touch -- state, NoPE / SEQ_ALONG_LANE
records, RoPE rows -- must match `compressor_forward` without them bit for bit,
on mixed steps (decode requests + prefill chunks at arbitrary, unaligned
positions, including position 0) at the served DSv4-Pro geometry.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compressor_v1, config

HIDDEN = 7168
BLOCK = 1024
# variant: (head_dim, compress_ratio, overlap, quant_block)
VARIANTS = {
    "csa": (512, 4, True, 64),
    "indexer": (128, 4, True, 128),
    "hca": (512, 128, False, 64),
}
pytestmark = pytest.mark.tpu_test

NEW = dict(proj_runs_min_tokens=8, prefill_block=128, prefill_block_min_tokens=8)


def _build(variant, decode, prefills, seed=0):
    """decode: (num_reqs, tokens each); prefills: [(length, first position)]."""
    head, ratio, overlap, qb = VARIANTS[variant]
    rng = np.random.default_rng(seed)
    segs = [(22528 + 977 * i + 3, decode[1]) for i in range(decode[0])]
    segs += [(p, n) for n, p in prefills]
    n_real = sum(n for _, n in segs)
    T = 1 << max(3, math.ceil(math.log2(n_real)))
    max_pos = max(s + n for s, n in segs) + 8
    mode = config.select_mode(head, overlap)
    sbs = config.state_block_size(
        mode, BLOCK, compress_ratio=ratio, head_dim=head, indexer_seq_along_lane=True
    )
    nreq = len(segs)
    st_cols = math.ceil(max_pos / sbs) + 1
    kv_cols = math.ceil(max_pos / BLOCK) + 1
    st = np.zeros((nreq, st_cols), np.int32)
    kv = np.zeros((nreq, kv_cols), np.int32)
    nxt = 1
    for q, (s, n) in enumerate(segs):
        lo, hi = max(0, (s - 4 * ratio) // sbs - 1), (s + n) // sbs + 1
        st[q, lo : hi + 1] = np.arange(nxt, nxt + hi + 1 - lo)
        nxt += hi + 1 - lo
        lo, hi = max(0, s // BLOCK - 1), (s + n) // BLOCK + 1
        kv[q, lo : hi + 1] = np.arange(nxt, nxt + hi + 1 - lo)
        nxt += hi + 1 - lo
    pages = nxt + 1
    remap = np.concatenate([[0], rng.permutation(np.arange(1, pages))]).astype(np.int32)
    st, kv = remap[st], remap[kv]

    def rand(shape, dtype):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        f = rng.standard_normal(nbytes // 4).astype(np.float32)
        return jnp.asarray(f.view(dtype).reshape(shape))

    rope = state = None
    if variant == "csa":
        cache = rand((pages, BLOCK // ratio, 128), np.int32)
        rope = rand((pages, BLOCK // ratio // 4, 128), np.int32)
    elif variant == "indexer":
        cache = rand((pages, config.SAL_SUBLANE_GROUPS, 4, BLOCK // ratio), np.uint8)
    else:
        cache = rand((pages, BLOCK // ratio * 2, 4, 128), np.uint8)
        state = rand((pages, BLOCK // 4, 128), np.int32)
    sw = (1 + overlap) * head
    k = jax.random.split(jax.random.key(seed + 1), 5)
    pos = np.zeros((T,), np.int32)
    o = 0
    for s, n in segs:
        pos[o : o + n] = np.arange(s, s + n)
        o += n
    args = dict(
        hidden_states=jax.random.normal(k[4], (T, HIDDEN)).astype(jnp.bfloat16),
        wkv_wgate=(jax.random.normal(k[0], (HIDDEN, 2 * sw)) * 0.02).astype(jnp.bfloat16),
        ape=jax.random.normal(k[1], (ratio, sw)) * 0.1,
        norm_weight=1.0 + 0.1 * jax.random.normal(k[2], (head,)),
        cos_sin_cache=jax.random.uniform(k[3], (max_pos + 16, 64), minval=-1, maxval=1),
        positions=jnp.asarray(pos),
        block_table=jnp.asarray(st.reshape(-1)),
        query_start_loc=jnp.asarray(np.cumsum([0] + [n for _, n in segs]), jnp.int32),
        kv_block_table=jnp.asarray(kv.reshape(-1)),
        distribution=jnp.array([decode[0], decode[0], nreq], jnp.int32),
        cache=cache,
        rope_cache=rope,
        state_cache=state,
        state_block_size=sbs,
        head_dim=head,
        compress_ratio=ratio,
        overlap=overlap,
        rms_eps=1e-6,
        quant_block=qb,
        seq_along_lane=variant == "indexer",
    )
    return args


_STATIC = ("state_block_size", "head_dim", "compress_ratio", "overlap", "rms_eps",
           "quant_block", "seq_along_lane")


def _run(args, **kw):
    static = {k: args[k] for k in _STATIC}
    dyn = {k: v for k, v in args.items() if k not in _STATIC}
    out = jax.jit(lambda a: compressor_v1.compressor_forward(**a, **static, **kw))(dyn)
    return [None if x is None else np.asarray(x) for x in out]


class CompressorPrefillPathsTest(jtu.JaxTestCase):
    def setUp(self):
        super().setUp()
        if not jtu.test_device_matches(["tpu"]):
            self.skipTest("TPU only")

    @parameterized.product(
        variant=tuple(VARIANTS),
        case=(
            ((1, 7), ((2041, 22528),)),
            ((1, 7), ((2041, 22531),)),
            ((1, 7), ((505, 0),)),
            ((2, 7), ((121, 7),)),
            ((1, 7), ((249, 5),)),
            ((2, 7), ((500, 1000), (1530, 50001))),
            ((1, 7), ((3, 40000), (2037, 7))),
            ((3, 7), ((2027, 300001),)),
            ((4, 7), ()),
        ),
    )
    def test_bit_exact(self, variant, case):
        args = _build(variant, *case)
        ref = _run(args)
        new = _run(args, **NEW)
        for r, n in zip(ref, new):
            if r is not None:
                self.assertArraysEqual(r, n)
        # The step wrote something (guards a vacuous pass).
        inputs = (args["cache"], args["rope_cache"], args["state_cache"])
        self.assertTrue(
            any(
                r is not None and not np.array_equal(r, np.asarray(i))
                for r, i in zip(ref, inputs)
            )
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
