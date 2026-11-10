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
"""VLLM_TPU_MK_WHOLE_BATCHED_INPUTS on TPU: the real ring/tag read, bitwise against the
per-layer loop, at the served whole-step shapes (11 QSA layers, 512-token FP8 blocks),
plus the per-call time of both (a microbench; printed, not asserted)."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_inputs import qsa_step_inputs

pytestmark = pytest.mark.tpu_test

L, T, MAX_GROUPS, PAGES = 11, 4, 32768, 1024


def _case(nreq, n_req, seed=0):
    rng = np.random.default_rng(seed)
    blocks = 2048
    pool = jax.random.randint(
        jax.random.PRNGKey(seed), (blocks, 512, 1, 4, 256), 0, 255, jnp.uint8
    ).view(jnp.float8_e4m3fn)
    tables = [
        (
            jnp.asarray(rng.integers(1, blocks, size=(n_req, 1)), jnp.int32),
            jnp.asarray(rng.integers(1, blocks, size=(n_req, PAGES)), jnp.int32),
            jnp.asarray(rng.integers(1, blocks, size=(n_req, 260)), jnp.int32),
        )
        for _ in range(L)
    ]
    slots = [(o % 2, o % 2) for o in range(L)]
    q_len = np.full(n_req, T, np.int32)
    if n_req > 1:
        q_len[1] = 0  # an idle request
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(q_len)]), jnp.int32)
    ctx = rng.integers(0, 100_000, size=n_req).astype(np.int32)
    ctx[-1] = 0  # a fresh request: no ring yet
    trig = jnp.asarray(rng.standard_normal((262144, 128)), jnp.bfloat16)
    args = (pool, tables, slots, trig, jnp.arange(nreq * T, dtype=jnp.int32), starts,
            jnp.asarray(ctx))
    kw = dict(M=nreq * T, T=T, pages_per_seq=PAGES, max_groups=MAX_GROUPS,
              layer_major_tables=True)
    return args, kw


def _run(batched, args, kw, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_WHOLE_BATCHED_INPUTS", batched, raising=False)
    names = None

    def fn(*a):
        out = qsa_step_inputs(*a, **kw)
        out = {k: v for k, v in out.items() if k not in ("qsa_pool", "qsa_ck")}
        nonlocal names
        names = sorted(out)
        return [out[k] for k in names]

    f = jax.jit(fn)
    out = jax.block_until_ready(f(*args))
    for _ in range(5):
        jax.block_until_ready(f(*args))
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        out = f(*args)
    jax.block_until_ready(out)
    return dict(zip(names, out)), (time.perf_counter() - t0) / n * 1e6


@pytest.mark.parametrize("nreq,n_req", [(1, 1), (4, 4), (4, 3)])
def test_batched_inputs_bitwise_on_tpu(nreq, n_req, monkeypatch):
    args, kw = _case(nreq, n_req)
    want, t_loop = _run(False, args, kw, monkeypatch)
    got, t_batched = _run(True, args, kw, monkeypatch)
    print(f"\nqsa_step_inputs nreq={nreq} n_req={n_req}: per-layer loop {t_loop:.1f} us/call, "
          f"layer-batched {t_batched:.1f} us/call")
    for k in want:
        a, b = np.asarray(want[k]), np.asarray(got[k])
        assert a.shape == b.shape and a.dtype == b.dtype, k
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8), err_msg=k)


@pytest.mark.parametrize("ns", [4, 16])
def test_batched_gdn_blocks_bitwise_on_tpu(ns, monkeypatch):
    """gdn_blocks over the whole step's 33 GDN layers: vmapped == per-layer loop."""
    from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import gdn_blocks

    rng = np.random.default_rng(4)
    n, slots = ns, 2 * ns + 1
    meta = []
    for _ in range(33):
        meta += [
            jnp.asarray(rng.integers(-1, slots, size=n), jnp.int32),
            jnp.asarray(rng.integers(0, 4, size=slots), jnp.int32),
            jnp.asarray(rng.integers(0, 4096, size=(n, 4)), jnp.int32),
            jnp.asarray(np.arange(n + 1) * 4, jnp.int32),
            jnp.asarray(rng.integers(0, 9000, size=n), jnp.int32),
        ]
    res = {}
    for batched in (False, True):
        monkeypatch.setattr(envs, "VLLM_TPU_MK_WHOLE_BATCHED_INPUTS", batched, raising=False)
        f = jax.jit(lambda *m: gdn_blocks(list(m), ns, True, 4096))
        out = jax.block_until_ready(f(*meta))
        for _ in range(5):
            jax.block_until_ready(f(*meta))
        t0 = time.perf_counter()
        for _ in range(50):
            out = f(*meta)
        jax.block_until_ready(out)
        res[batched] = (out, (time.perf_counter() - t0) / 50 * 1e6)
    print(f"\ngdn_blocks NS={ns} x33: per-layer {res[False][1]:.1f} us/call, "
          f"vmapped {res[True][1]:.1f} us/call")
    for a, b in zip(res[False][0], res[True][0], strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
