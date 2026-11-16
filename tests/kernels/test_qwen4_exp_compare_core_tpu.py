# SPDX-License-Identifier: Apache-2.0
"""TPU checks of VLLM_TPU_QSA_PREFILL_COMPARE_CORE at the served prefill shape (run through the bench's
BENCH=script; the block exports the served prefill QSA env).

- Tables: the SparseCore build (``compare_sc_tables`` on ``union_sc_bits_flat``) equals the XLA reference on every
  live slot (padding slots only need a valid address: the SparseCore pads with word 0's pool base, the reference
  with slot 0's row).
- Numerics: ``qsa_rpa._prefill_attention`` with the flag and without it (the served path: the union walk with its
  SparseCore list and gather), each against a float32 reference over every row's exact selection on 512 sampled
  rows. Gate (lead ruling 10.264): the port's max and RMS error at most 1.25x the served path's own, its
  max |out - ref| / mean |ref| at most 0.08 (Compare's max_rel), every value finite. The port against the served
  output is printed: both are bf16, so one output ulp at the largest values already gives ~0.09 there.
- Timing (printed, not gated): each path's call, the tables, and the kernel in full, without its KV copies
  (compute), with its copies alone (reads), with the slot rows staged one tile ahead (lead1), and with
  pool-contiguous slot pairs as one copy (pair, pair_reads).

Selections follow the recorded ones (s10623 L39: 513 groups a row in ~335 runs; two rows 16 apart share ~105):
each row keeps its 32 most recent complete groups and its predecessor's other groups but ~49, which it replaces by
new groups in runs of 1-3 drawn over the context, so 16 consecutive rows' union is ~1,250 groups and 64 rows'
~3,400 (the served walk's ~3,300 estimate at 92.8k, prefill-bound/result.md).
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
    ("t6400", [(3200, 64000), (3200, 64000)], 6400, 3200),
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


def _selections(rng, n, ctx, topk=TOPK, local=32, change=49):
    """int32 [n, topk] selections of a request's n rows after ctx cached tokens (module docstring)."""
    out = np.full((n, topk), -1, np.int32)
    keep = None
    for i in range(n):
        complete = (ctx + i + 1) // RATIO
        recent = np.arange(max(0, complete - local), complete)
        if keep is None:
            keep = rng.choice(max(complete - local, 1), size=topk - local, replace=False)
        else:
            keep = keep[rng.permutation(len(keep))[change:]]
            starts = rng.integers(0, max(complete - local - 3, 1), size=change)
            runs = np.concatenate([s0 + np.arange(rng.integers(1, 4)) for s0 in starts])
            keep = np.unique(np.concatenate([keep, runs]))
            keep = keep[keep < complete - local][: topk - local] if len(keep) > topk - local else keep
        sel = np.unique(np.concatenate([recent, keep]))[:topk]
        out[i, :len(sel)] = sel
    return out


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
        rows = slice(starts[r], starts[r] + n)
        lengths[rows] = ctx + np.arange(n) + 1
        table[rows] = owner[r * pps:(r + 1) * pps]
        blocks[rows] = _selections(rng, n, ctx)
    q = jnp.asarray(rng.normal(0, 0.5, (tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, (tokens, 24, 256)), jnp.bfloat16)
    key = jax.random.key(seed)
    raw = jax.random.randint(key, (pool_pages, page, 1, 4, 256), 0, 0x40, jnp.int32).astype(jnp.uint8)
    raw = raw | (jax.random.randint(jax.random.fold_in(key, 1), raw.shape, 0, 2, jnp.int32).astype(jnp.uint8) << 7)
    pool = jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), pool, jnp.asarray(starts)


def _reference(step, rows):
    """float32 gated attention [len(rows), 24, 256] of the sampled rows over their exact selections (the indexer's
    groups plus the partial tail group, tokens below the row's length), 32 rows a chunk."""
    q, gate, blocks, lengths, table, pool, _ = step
    pages, page = pool.shape[:2]
    flat = pool.reshape(pages * page, 4, 256)

    @jax.jit
    def chunk(r):
        length = lengths[r]
        tail = jnp.where(length % RATIO != 0, length // RATIO, -1)
        groups = jnp.concatenate([blocks[r], tail[:, None]], axis=1)                    # [c, K + 1]
        tok = groups[..., None] * RATIO + jnp.arange(RATIO)                              # [c, K + 1, 4]
        tok = tok.reshape(len(r), -1)
        live = (tok >= 0) & (tok < length[:, None]) & (groups.repeat(RATIO, axis=1) >= 0)
        safe = jnp.where(live, tok, 0)
        phys = jnp.take_along_axis(table[r], safe // page, axis=1) * page + safe % page
        kv = flat[phys].astype(jnp.float32)                                              # [c, n, 4, 256]
        qq = q[r].astype(jnp.float32).reshape(len(r), 2, 12, 256)
        k, v = kv[:, :, 0::2], kv[:, :, 1::2]                                           # [c, n, 2, 256]
        s = jnp.einsum("cgjd,cngd->cgjn", qq, k, precision="highest") / 16.0
        s = jnp.where(live[:, None, None, :], s, -jnp.inf)
        p = jax.nn.softmax(s, axis=-1)
        o = jnp.einsum("cgjn,cngd->cgjd", p, v, precision="highest").reshape(len(r), 24, 256)
        return o * jax.nn.sigmoid(gate[r].astype(jnp.float32))

    out = [np.asarray(chunk(jnp.asarray(rows[i:i + 32]))) for i in range(0, len(rows), 32)]
    return np.concatenate(out)


def _rel(got, ref):
    return float(np.abs(got - ref).max() / np.abs(ref).mean())


def _errors(got, ref):
    """(max |got - ref|, RMS of got - ref)."""
    d = np.asarray(got, np.float64) - np.asarray(ref, np.float64)
    return float(np.abs(d).max()), float(np.sqrt(np.mean(d * d)))


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
    live = np.arange(cap)[None, :] < usize[:, None]
    np.testing.assert_array_equal(np.where(live, np.asarray(got[0]), 0), np.where(live, np.asarray(want[0]), 0),
                                  err_msg="rows (live slots)")
    for name, g, w in zip(("codes", "usize", "ok"), got[1:], want[1:]):
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
    live = np.nonzero(np.asarray(step[3]) > 0)[0]
    rows = live[np.linspace(0, len(live) - 1, 512).astype(int)]
    ref = _reference(step, rows)
    port_rel, served_rel = _rel(port[rows], ref), _rel(served[rows], ref)
    (port_max, port_rms), (served_max, served_rms) = _errors(port[rows], ref), _errors(served[rows], ref)
    raw_rel = _rel(port[live], served[live])
    finite = bool(np.isfinite(port).all())
    # The core's parts on the same step: tables, then the kernel on them, in each timing mode.
    cap = int(os.environ.get("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP", "4096"))
    lister = os.environ.get("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_TABLES", "sparsecore")
    bitset = "sparsecore" if os.environ.get("VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET") == "1" else "tensorcore"
    tables_fn, targs, aq = _aligned_tables(step, lister, bitset, cap)
    rows_t, codes, usize, ok = jax.block_until_ready(tables_fn(*targs))
    tables_ms = _median_ms(tables_fn, targs)
    depth = int(os.environ.get("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_DEPTH", "8"))
    kargs = (aq, rows_t, codes, usize, step[5])
    u = np.asarray(usize)
    tiles = int(np.sum(np.where(u > 0, -(-np.minimum(u, cap) // cc.GPT), 0)))
    modes = {}
    for label, kw in (("full", {}), ("compute", dict(mode="compute")), ("reads", dict(mode="reads")),
                      ("lead1", dict(srow_lead=1)), ("pair", dict(pair=True)), ("pair_reads", dict(pair=True, mode="reads"))):
        fn = jax.jit(lambda q, r, c, us, p, kw=kw: cc.compare_core_kernel(q, r, c, us, p, union_cap=cap,
                                                                          depth=depth, **kw))
        modes[label] = _median_ms(fn, kargs)
    per_tile = {k: v * 1e3 / max(tiles, 1) for k, v in modes.items()}
    _report(f"[cmp-tpu] {name}: against float32 (512 rows) port max {port_max:.3e} rms {port_rms:.3e} max_rel "
            f"{port_rel:.4f}, served max {served_max:.3e} rms {served_rms:.3e} max_rel {served_rel:.4f}, ratios max "
            f"{port_max / max(served_max, 1e-30):.3f} rms {port_rms / max(served_rms, 1e-30):.3f}; port "
            f"against served {raw_rel:.4f}; finite {finite} ok {bool(ok)}; served call {served_ms:.1f} ms, port call "
            f"{port_ms:.1f} ms, tables {tables_ms:.2f} ms; union max {u.max()} mean live {u[u > 0].mean():.1f}, "
            f"{tiles} tiles, {tiles / max(1, int((u > 0).sum())):.2f} a live group; kernel ms "
            + ", ".join(f"{k} {v:.2f} ({per_tile[k]:.3f} us a tile)" for k, v in modes.items()))
    assert finite and bool(ok)
    assert port_rel <= MAX_REL, port_rel
    assert port_max <= 1.25 * served_max and port_rms <= 1.25 * served_rms, (port_max, served_max, port_rms,
                                                                              served_rms)


# Float64 stress cases (work-order-prefill-qsa.md, last section): one 2,048-row request at 24,000 cached tokens,
# 3200-token pages, the port and the served path each against a float64 reference on 256 sampled rows.
STRESS = ("peaked", "long_unions", "near_duplicate_keys", "fp8_range_ends", "nan_tail", "recorded_selections")
_RECORDED = ("/home/benliangw_google_com/workspace/tmp/compare-runs/prepared/objects/served/"
             "served-8f219aec5cbc3d30ba2e/served-prefill-38258702ef-stage6.json")


def _stress_step(name, seed=11, rows=2048, ctx=24000, page=3200):
    """(step, query scale) of a stress case; the pool holds one sequence's pages in a permuted order."""
    rng = np.random.default_rng(seed)
    pps = -(-(ctx + rows) // page)
    lengths = (ctx + np.arange(rows) + 1).astype(np.int32)
    table = np.broadcast_to((rng.permutation(pps) + 1).astype(np.int32), (rows, pps)).copy()
    if name == "long_unions":
        # Each 16-row group: 272 shared groups and 227 of each row's own, ~3,900 a union (cap 4,096); a row's
        # 13 most recent groups ride along so each selection stays causal.
        blocks = np.full((rows, TOPK), -1, np.int32)
        for g in range(rows // 16):
            complete = (ctx + 16 * g + 1) // RATIO
            pool = rng.permutation(complete - 16)[: 272 + 16 * 227]
            for r in range(16):
                own = pool[272 + 227 * r: 272 + 227 * (r + 1)]
                recent = np.arange(complete - 13, complete)
                sel = np.unique(np.concatenate([pool[:272], own, recent]))[:TOPK]
                blocks[16 * g + r, :len(sel)] = sel
    elif name == "recorded_selections" and os.path.exists(_RECORDED):
        # s10623 L39's recorded selections (rows 16 apart), each row taking the nearest earlier recorded row's
        # groups below its own tail (Compare's interpolated plant), shifted onto this context.
        import json

        rec = json.load(open(_RECORDED))["plant"]["selections"][0]["tables"][0]
        blocks = np.full((rows, TOPK), -1, np.int32)
        for t in range(rows):
            sel = np.asarray(rec[min(t // 16, len(rec) - 1)], np.int64)
            sel = sel[sel < lengths[t] // RATIO][:TOPK]
            blocks[t, :len(sel)] = sel
    else:
        blocks = _selections(rng, rows, ctx)
    scale = 8.0 if name == "peaked" else 1.0
    q = jnp.asarray(rng.normal(0, 0.5 * scale, (rows, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, (rows, 24, 256)), jnp.bfloat16)
    shape = (pps + 1, page, 1, 4, 256)
    if name == "near_duplicate_keys":
        # Every token's K and V one of 4 base records, a few bytes flipped: many exact and near ties.
        base = rng.integers(0, 0x40, size=(4, 1, 4, 256), dtype=np.uint8) | (rng.integers(0, 2, (4, 1, 4, 256)).astype(np.uint8) << 7)
        raw = base[rng.integers(0, 4, size=shape[:2])].copy()
        flip = rng.random(raw.shape) < 0.01
        raw[flip] ^= 1
    elif name == "fp8_range_ends":
        # Magnitudes at e4m3's ends: 448 (0x7E), the smallest subnormal (0x01) and zero, random signs.
        mag = rng.choice(np.array([0x7E, 0x01, 0x00], np.uint8), size=shape, p=[0.3, 0.4, 0.3])
        raw = mag | (rng.integers(0, 2, size=shape).astype(np.uint8) << 7)
    else:
        raw = rng.integers(0, 0x40, size=shape, dtype=np.uint8) | (rng.integers(0, 2, size=shape).astype(np.uint8) << 7)
    if name == "nan_tail":
        # Unwritten bytes past the last row in its tail group (and the rest of its page) are NaN (0x7F).
        last = ctx + rows
        page_i, off = last // page, last % page
        raw[table[0, page_i], off:] = 0x7F
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    starts = jnp.asarray([0] + [rows] * 16, jnp.int32)
    return (q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), pool, starts)


def _f64_rows(step, rows):
    """float64 gated attention of the sampled rows over their exact selections (numpy, on the host)."""
    q, gate, blocks, lengths, table, pool, _ = step
    page = pool.shape[1]
    flat = np.asarray(pool.reshape(-1, 4, 256).astype(jnp.float32)).astype(np.float64)
    qn, gn = np.asarray(q.astype(jnp.float32), np.float64), np.asarray(gate.astype(jnp.float32), np.float64)
    b, ln, tb = np.asarray(blocks), np.asarray(lengths), np.asarray(table)
    out = np.zeros((len(rows), 24, 256))
    for n, t in enumerate(rows):
        length = int(ln[t])
        groups = {int(x) for x in b[t] if x >= 0}
        if length % RATIO:
            groups.add(length // RATIO)
        toks = np.asarray(sorted(x for g in groups for x in range(g * RATIO, g * RATIO + RATIO) if x < length))
        kv = flat[tb[t][toks // page] * page + toks % page]
        for h in range(24):
            k, v = kv[:, 2 * (h // 12)], kv[:, 2 * (h // 12) + 1]
            s = k @ qn[t, h] / 16.0
            p = np.exp(s - s.max())
            out[n, h] = (p / p.sum()) @ v
    return out / (1.0 + np.exp(-gn[rows]))


@pytest.mark.parametrize("name", STRESS)
def test_stress_against_float64(monkeypatch, name):
    if name == "recorded_selections" and not os.path.exists(_RECORDED):
        pytest.skip("recorded selections not on this host")
    step = _stress_step(name)
    _, served = _call(step)
    served = np.asarray(jax.block_until_ready(served).astype(jnp.float32))
    jax.clear_caches()
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE", "1")
    _, port = _call(step)
    port = np.asarray(jax.block_until_ready(port).astype(jnp.float32))
    rows = np.linspace(0, len(port) - 1, 256).astype(int)
    ref = _f64_rows(step, rows)
    (pm, pr), (sm, sr) = _errors(port[rows], ref), _errors(served[rows], ref)
    port_rel, served_rel = _rel(port[rows], ref), _rel(served[rows], ref)
    finite = bool(np.isfinite(port).all())
    _report(f"[cmp-tpu] stress {name}: against float64 (256 rows) port max {pm:.3e} rms {pr:.3e} max_rel "
            f"{port_rel:.4f}, served max {sm:.3e} rms {sr:.3e} max_rel {served_rel:.4f}, ratios max "
            f"{pm / max(sm, 1e-30):.3f} rms {pr / max(sr, 1e-30):.3f}; finite port {finite} served "
            f"{bool(np.isfinite(served).all())}")
    assert finite
    assert port_rel <= MAX_REL, port_rel
    assert pm <= 1.25 * sm and pr <= 1.25 * sr, (pm, sm, pr, sr)
