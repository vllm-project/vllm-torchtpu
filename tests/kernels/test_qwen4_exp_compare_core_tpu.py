# SPDX-License-Identifier: Apache-2.0
"""TPU checks of VLLM_TPU_QSA_PREFILL_COMPARE_CORE at the served prefill shape (run through the bench's
BENCH=script, which carries the served prefill env).

- Tables: the SparseCore build (``compare_sc_tables`` on ``union_sc_bits_flat``) equals the XLA reference.
- Numerics: ``qsa_rpa._prefill_attention`` with the flag against the same call without it (the served path under
  the pod's env: the union walk with its SparseCore list and gather) on the same inputs: an 8192-row step of two
  4096-row requests at 64k and 96k context, and s10623's shape (3200 rows at 64000 plus 682 at 70400). Gate:
  max |port - served| / mean |served| over live rows at most 0.08 (Compare's max_rel), every value finite.
- Timing (printed, not gated): each path's call and the core's parts, median wall time of a few jitted calls.

Selections: each row takes its 32 most recent groups plus 480 drawn from a request-level set of popular groups
(Zipf weights, so neighbouring rows overlap as real selections do), its own tail group by the served contract.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import qsa_compare_core as cc
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa


RATIO, TOPK = 4, 512
# The served prefill flags the numerics compare against, used where the pod's env has none.
SERVED_ENV = {
    "VLLM_TPU_QSA_PREFILL_UNION_WALK": "64", "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE": "1024",
    "VLLM_TPU_QSA_PREFILL_LEAN": "1", "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": "19200",
    "VLLM_TPU_QSA_PREFILL_UNION_CAP": "16384", "VLLM_TPU_QSA_PREFILL_Q128_FALLBACK": "1",
    "VLLM_TPU_QSA_PREFILL_UNION_LISTED": "sparsecore", "VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS": "1",
    "VLLM_TPU_QSA_PREFILL_UNION_ALPHA": "0.9", "VLLM_TPU_QSA_PREFILL_UNION_GATHERED": "sparsecore",
    "VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS": "8", "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS": "4",
    "VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE": "1", "VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET": "1",
    "VLLM_TPU_QSA_PREFILL_QUERY_TILE": "64",
}
# (name, [(rows, cached) per request], bucket rows, page tokens: a TP1 prefill stage's 3200-token kernel pages)
CASES = [
    ("ctx64k", [(4096, 64000), (4096, 64000)], 8192, 3200),
    ("ctx96k", [(4096, 96000), (4096, 96000)], 8192, 3200),
    ("s10623", [(3200, 64000), (682, 70400)], 4096, 3200),
]
MAX_REL = 0.08
REPS = int(os.environ.get("CMP_TPU_REPS", "5"))


def _report(line):
    """A result line on stdout and, with CMP_TPU_OUT set, appended to its summary.txt."""
    print(line, flush=True)
    out = os.environ.get("CMP_TPU_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "summary.txt"), "a") as f:
            f.write(line + "\n")


@pytest.fixture(autouse=True)
def served_env(monkeypatch):
    # The backend is touched here, not at import: a backend initialized during collection claims the TPU.
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")
    for k, v in SERVED_ENV.items():
        if k not in os.environ:
            print(f"[cmp-tpu] {k} unset in the pod env; using {v}", flush=True)
            monkeypatch.setenv(k, v)
    monkeypatch.delenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE", raising=False)
    yield
    jax.clear_caches()


def _step(seed, reqs, tokens, page):
    rng = np.random.default_rng(seed)
    pps = max(-(-(c + n) // page) for n, c in reqs)
    pool_pages = pps * len(reqs) + 1
    starts = np.concatenate([[0], np.cumsum([n for n, _ in reqs])]).astype(np.int32)
    owner = rng.permutation(pool_pages - 1) + 1
    lengths = np.zeros(tokens, np.int32)
    blocks = np.full((tokens, TOPK), -1, np.int32)
    table = np.zeros((tokens, pps), np.int32)
    for r, (n, ctx) in enumerate(reqs):
        pages = owner[r * pps:(r + 1) * pps]
        hot_n = 1600
        hot = rng.choice((ctx + n) // RATIO, size=hot_n, replace=False)
        weight = 1.0 / np.arange(1, hot_n + 1) ** 0.8
        for i in range(n):
            t = starts[r] + i
            length = ctx + i + 1
            lengths[t] = length
            table[t] = pages
            complete = length // RATIO
            local = np.arange(max(0, complete - 32), complete)
            ok = hot < complete
            w = weight[ok] / weight[ok].sum()
            pick = rng.choice(hot[ok], size=min(TOPK - len(local), int(ok.sum())), replace=False, p=w)
            sel = np.unique(np.concatenate([local, pick]))[:TOPK]
            blocks[t, :len(sel)] = sel
    q = jnp.asarray(rng.normal(0, 0.5, (tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, (tokens, 24, 256)), jnp.bfloat16)
    key = jax.random.key(seed)
    raw = jax.random.randint(key, (pool_pages, page, 1, 4, 256), 0, 0x40, jnp.int32).astype(jnp.uint8)
    raw = raw | (jax.random.randint(jax.random.fold_in(key, 1), raw.shape, 0, 2, jnp.int32).astype(jnp.uint8) << 7)
    pool = jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), pool, jnp.asarray(starts)


def _call(step):
    fn = jax.jit(lambda q, g, b, l, t, p, s: qsa_rpa._prefill_attention(q, g, b, l, t, p, s, kv_heads=2,
                                                                        ratio=RATIO)[0])
    return fn, fn(*step)


def _median_ms(fn, args, reps=REPS):
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def _aligned_tables(step, lister, bitset, cap):
    q, gate, blocks, lengths, table, pool, starts = step
    starts_np = np.asarray(starts)
    real = np.diff(starts_np)
    offsets = np.concatenate([[0], np.cumsum(-(-real // 16) * 16)])
    size = max(int(offsets[-1]), len(lengths))
    where = np.concatenate([offsets[r] + np.arange(n) for r, n in enumerate(real)])

    def put(x, fill):
        x = np.asarray(x)
        out = np.full((size, *x.shape[1:]), fill, x.dtype)
        out[where] = x[: starts_np[-1]]
        return jnp.asarray(out)

    page = pool.shape[1]
    ab, al, at = put(blocks, -1), put(lengths, 0), put(table, 0)
    fn = jax.jit(lambda b, l, t: cc.compare_tables(b, l, t, groups_cap=table.shape[1] * page // RATIO,
                                                   union_cap=cap, page_size=page, bitset=bitset, lister=lister))
    return fn, (ab, al, at), put(q.astype(jnp.float32), 0).astype(jnp.bfloat16)


def test_sparsecore_tables_equal_the_reference():
    step = _step(0, CASES[2][1], CASES[2][2], CASES[2][3])
    cap = int(os.environ.get("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP", "4096"))
    got_fn, args, _ = _aligned_tables(step, "sparsecore", "sparsecore", cap)
    want_fn, _, _ = _aligned_tables(step, "reference", "tensorcore", cap)
    got, want = jax.block_until_ready(got_fn(*args)), jax.block_until_ready(want_fn(*args))
    usize = np.asarray(want[2])
    _report(f"[cmp-tpu] tables s10623: usize max {usize.max()} mean live {usize[usize > 0].mean():.1f}; "
            f"tiles a live group {float(cc.tiles_per_group(want[2], cap)):.2f}")
    for name, g, w in zip(("rows", "codes", "usize", "ok"), got, want):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w), err_msg=name)


@pytest.mark.parametrize("name,reqs,tokens,page", CASES, ids=[c[0] for c in CASES])
def test_numerics_and_time_against_served(monkeypatch, name, reqs, tokens, page):
    step = _step(1, reqs, tokens, page)
    served_fn, served = _call(step)
    served = np.asarray(jax.block_until_ready(served).astype(jnp.float32))
    served_ms = _median_ms(served_fn, step)
    jax.clear_caches()
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE", "1")
    port_fn, port = _call(step)
    port = np.asarray(jax.block_until_ready(port).astype(jnp.float32))
    port_ms = _median_ms(port_fn, step)
    live = np.asarray(step[3]) > 0
    d = np.abs(port[live] - served[live])
    scale = float(np.abs(served[live]).mean())
    max_rel = float(d.max() / scale)
    finite = bool(np.isfinite(port).all())
    # The core's parts on the same step: tables, then the kernel on them.
    cap = int(os.environ.get("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP", "4096"))
    lister = os.environ.get("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_TABLES", "sparsecore")
    bitset = "sparsecore" if os.environ.get("VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET") == "1" else "tensorcore"
    tables_fn, targs, aq = _aligned_tables(step, lister, bitset, cap)
    rows, codes, usize, ok = jax.block_until_ready(tables_fn(*targs))
    tables_ms = _median_ms(tables_fn, targs)
    depth = int(os.environ.get("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_DEPTH", "8"))
    kernel_fn = jax.jit(lambda q, r, c, u, p: cc.compare_core_kernel(q, r, c, u, p, union_cap=cap, depth=depth))
    kargs = (aq, rows, codes, usize, step[5])
    kernel_ms = _median_ms(kernel_fn, kargs)
    u = np.asarray(usize)
    tiles = int(np.sum(np.where(u > 0, -(-np.minimum(u, cap) // cc.GPT), 0)))
    _report(f"[cmp-tpu] {name}: max_rel {max_rel:.4f} (max abs {d.max():.3e}, mean |served| {scale:.3e}) finite "
            f"{finite} ok {bool(ok)}; served call {served_ms:.1f} ms, port call {port_ms:.1f} ms, tables "
            f"{tables_ms:.2f} ms, kernel {kernel_ms:.2f} ms; union max {u.max()} mean live {u[u > 0].mean():.1f}, "
            f"{tiles} tiles, {tiles / max(1, int((u > 0).sum())):.2f} a live group, kernel "
            f"{kernel_ms * 1e3 / max(tiles, 1):.3f} us a tile")
    assert finite and bool(ok)
    assert max_rel <= MAX_REL, max_rel
