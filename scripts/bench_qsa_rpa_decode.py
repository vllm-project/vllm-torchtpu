"""QSA sparse decode attention microbench at the Qwen3.8-Flash TP8 geometry.

Per rank: 3 Q heads, 1 KV head (d 256), FP8 main KV rows (1, 4, 256) with a
paired second layer, 512-token pages, top-512 groups of 4 tokens, MTP K=3
(4 verify rows per request). Times RPA decode across block sizes against the
fused sparse GQA decode kernel. Each sample runs ITERS calls in one program,
threading the cache, so dispatch is outside the measurement.

Run directly (python scripts/bench_qsa_rpa_decode.py) or through pytest.
"""

import functools
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa  # noqa: E402

PAGE = 512
RATIO = 4
TOPK_GROUPS = 512
Q_HEADS = 3
D = 256
ITERS = int(os.environ.get("BENCH_ITERS", "20"))


def clustered_pick(rng, visible, recent=128, mean_run=4):
    """A recent window plus geometric-length runs at random positions."""
    picked = set(range(max(visible - recent, 0), visible))
    while len(picked) < min(TOPK_GROUPS, visible):
        start = int(rng.integers(0, visible))
        for g in range(start, min(start + int(rng.geometric(1 / mean_run)), visible)):
            picked.add(g)
    out = np.fromiter(picked, np.int32)[:TOPK_GROUPS]
    return rng.permutation(out)


def pick_blocks(rng, lengths, pattern):
    blocks = np.full((len(lengths), TOPK_GROUPS), -1, np.int32)
    for i, length in enumerate(lengths):
        visible = int(length) // RATIO
        if not visible:
            continue
        if pattern == "clustered":
            pick = clustered_pick(rng, visible)
        else:
            pick = rng.choice(visible, min(TOPK_GROUPS, visible), replace=False)
        blocks[i, : len(pick)] = pick
    return blocks


def make_inputs(
    rng, *, requests, rows, overlap, pages=4096, max_ctx=65536, pattern="random"
):
    """``requests`` x 4 verify rows padded to ``rows``; ``overlap`` is the share
    of each verify row's groups taken from its request's first row. ``pattern``
    "clustered" draws a recent window plus short runs (unsorted, as the
    indexer returns them)."""
    cols = max_ctx // PAGE
    contexts = rng.integers(max_ctx // 4, max_ctx - 8, size=requests)
    table = np.zeros((rows, cols), np.int32)
    lengths = np.zeros((rows,), np.int32)
    blocks = np.full((rows, TOPK_GROUPS), -1, np.int32)
    perm = rng.permutation(pages)
    for r in range(requests):
        n_pages = -(-(contexts[r] + 4) // PAGE)
        own = perm[(r * cols) % pages : (r * cols) % pages + n_pages]
        own = np.resize(own, n_pages)
        base = None
        for t in range(4):
            i = r * 4 + t
            if i >= rows:
                break
            table[i, :n_pages] = own
            lengths[i] = contexts[r] + t + 1
            visible = lengths[i] // RATIO
            if pattern == "clustered":
                blocks[i] = clustered_pick(rng, visible)
                continue
            pick = rng.choice(visible, TOPK_GROUPS, replace=False)
            if base is None:
                base = pick
            else:
                keep = int(overlap * TOPK_GROUPS)
                rest = np.setdiff1d(pick, base[:keep])[: TOPK_GROUPS - keep]
                pick = np.concatenate((base[:keep], rest))
                if len(pick) < TOPK_GROUPS:
                    pick = np.resize(pick, TOPK_GROUPS)
            blocks[i] = np.sort(pick)
    q = rng.standard_normal((rows, Q_HEADS, D)).astype(np.float32)
    gate = rng.standard_normal((rows, Q_HEADS, D)).astype(np.float32)
    return (
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(gate, jnp.bfloat16),
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
    )


def make_cache(pages=4096):
    key = jax.random.key(0)
    cache = jax.random.normal(key, (pages, PAGE, 1, 4, D), jnp.bfloat16)
    return (cache * 0.5).astype(jnp.float8_e4m3fn)


def bench(attend, inputs, cache):
    q, gate, blocks, lengths, table = inputs

    @functools.partial(jax.jit, donate_argnums=(0,))
    def run(cache, q):
        def body(i, carry):
            cache, acc = carry
            out, cache = attend(q, gate, blocks, lengths, table, cache)
            return cache, acc + out.astype(jnp.float32).sum()

        return jax.lax.fori_loop(0, ITERS, body, (cache, jnp.float32(0)))

    cache, acc = run(cache, q)
    jax.block_until_ready(acc)
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        cache, acc = run(cache, q)
        jax.block_until_ready(acc)
        samples.append((time.perf_counter() - t0) / ITERS * 1e6)
    return float(np.median(samples)), cache


def bench_step(attend, layer_inputs, cache, trace=None):
    """12 QSA layers of one step: per-layer Q/gate/blocks, shared lengths and
    table, alternating paired-page slots; XLA may share work across layers."""
    _, _, _, lengths, table = layer_inputs[0]
    qs = jnp.stack([x[0] for x in layer_inputs])
    gates = jnp.stack([x[1] for x in layer_inputs])
    blocks = jnp.stack([x[2] for x in layer_inputs])

    @functools.partial(jax.jit, donate_argnums=(0,))
    def run(cache, qs):
        def body(i, carry):
            cache, acc = carry
            for layer in range(len(layer_inputs)):
                out, cache = attend(
                    qs[layer], gates[layer], blocks[layer], lengths, table, cache,
                    layer % 2,
                )
                acc = acc + out.astype(jnp.float32).sum()
            return cache, acc

        return jax.lax.fori_loop(0, ITERS, body, (cache, jnp.float32(0)))

    cache, acc = run(cache, qs)
    jax.block_until_ready(acc)
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        cache, acc = run(cache, qs)
        jax.block_until_ready(acc)
        samples.append((time.perf_counter() - t0) / ITERS * 1e6)
    if trace:
        with jax.profiler.trace(trace):
            cache, acc = run(cache, qs)
            jax.block_until_ready(acc)
    return float(np.median(samples)), cache


def main_step(cache, results):
    """Per-step attention time for the serving routes (decode_attention)."""
    rng = np.random.default_rng(1)
    cases = os.environ.get("BENCH_CASES", "2x64,4x64,6x24,8x32").split(",")
    patterns = os.environ.get("BENCH_PATTERNS", "random").split(",")
    def parse(arm):
        # "fslr[/bkv/lanes]": fused, shared, lean, slot read digits, then an
        # optional compact schedule with that KV block and lanes.
        flags, *compact = arm.split("/")
        return (*map(int, flags), *(map(int, compact) if compact else ()))

    arms = [
        parse(a) for a in os.environ.get("BENCH_ARMS", "000,010,100,110").split(",")
    ]
    for (reqs, rows), pattern in itertools.product(
        [tuple(map(int, c.split("x"))) for c in cases], patterns
    ):
        base = make_inputs(rng, requests=reqs, rows=rows, overlap=0.0)
        lengths = np.asarray(base[3])
        layers = []
        for _ in range(12):
            q = rng.standard_normal((rows, Q_HEADS, D)).astype(np.float32)
            g = rng.standard_normal((rows, Q_HEADS, D)).astype(np.float32)
            layers.append((
                jnp.asarray(q, jnp.bfloat16), jnp.asarray(g, jnp.bfloat16),
                jnp.asarray(pick_blocks(rng, lengths, pattern)), base[3], base[4],
            ))
        for fused, shared, kv_fetch, *rest in arms:
            fused, shared = bool(fused), bool(shared)
            slot = rest[0] if rest else 0
            bkv, lanes = rest[1:3] if len(rest) >= 3 else (0, 0)

            def attend(q, g, b, lens, t, cache, offset, fused=fused, shared=shared,
                       kv_fetch=kv_fetch, slot=slot, bkv=bkv, lanes=lanes):
                os.environ["VLLM_TPU_QSA_DECODE_LEAN_FETCH"] = str(kv_fetch)
                os.environ["VLLM_TPU_QSA_DECODE_SLOT_READ"] = str(slot)
                os.environ["VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE"] = str(int(bkv > 0))
                if bkv:
                    os.environ["VLLM_TPU_QSA_DECODE_RPA_BKV"] = str(bkv)
                    os.environ["VLLM_TPU_QSA_DECODE_RPA_LANES"] = str(lanes)
                return qsa_rpa.decode_attention(
                    q, g, b, lens, t, cache, kv_heads=1, ratio=RATIO,
                    k_scale=1.0, v_scale=1.0, fused=fused, shared_schedule=shared,
                    **({"kv_head_offset": 1} if offset else {}),
                )

            trace = None
            if os.environ.get("BENCH_XPROF"):
                trace = os.path.join(
                    os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "xprof",
                    f"r{reqs}x{rows}_{pattern}_f{int(fused)}_s{int(shared)}"
                    f"_k{kv_fetch}_r{slot}_b{bkv}x{lanes}",
                )
            try:
                us, cache = bench_step(attend, layers, cache, trace)
            except Exception as e:  # noqa: BLE001
                us = repr(e)[:200]
            row = dict(mode="step", requests=reqs, rows=rows, pattern=pattern,
                       fused=fused, shared=shared, kv_fetch=kv_fetch, slot=slot,
                       bkv=bkv, lanes=lanes,
                       us_per_step=us)
            print("STEP", json.dumps(row), flush=True)
            results.append(row)
    return cache


def main():
    rng = np.random.default_rng(0)
    cache = make_cache()
    results = []
    if os.environ.get("BENCH_MODE", "step") == "step":
        main_step(cache, results)
        out_dir = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
        Path(out_dir, "bench_qsa_step.json").write_text(json.dumps(results, indent=1))
        return
    cases = [
        # (requests, padded rows, overlap)
        (6, 24, 0.0),
        (8, 32, 0.0),
        (8, 64, 0.0),
        (2, 64, 0.0),
        (8, 32, 0.9),
    ]
    blocks_cfg = [(256, 8), (512, 8), (1024, 8), (2048, 8), (256, 16), (512, 16),
                  (256, 4), (128, 8)]
    for (reqs, rows, overlap), offset in itertools.product(cases, (0, 1)):
        inputs = make_inputs(rng, requests=reqs, rows=rows, overlap=overlap)
        common = dict(kv_heads=1, ratio=RATIO, k_scale=1.0, v_scale=1.0)
        if offset:
            common["kv_head_offset"] = 1
        arms = {}
        for bkv, batch in blocks_cfg:
            arms[f"rpa_k{bkv}_b{batch}"] = functools.partial(
                qsa_rpa._rpa_attention_batch, bkv_size=bkv, batch_size=batch, **common
            )
        arms["fused"] = functools.partial(
            qsa_rpa.decode_attention, fused=True, **common
        )
        ref = None
        for name, attend in arms.items():
            try:
                us, cache = bench(attend, inputs, cache)
                q, gate, blocks, lengths, table = inputs
                out, cache = jax.jit(attend, donate_argnums=(5,))(
                    q, gate, blocks, lengths, table, cache
                )
                out = np.asarray(out.astype(jnp.float32))
                if ref is None:
                    ref = out
                err = float(np.max(np.abs(out - ref)))
            except Exception as e:  # noqa: BLE001
                us, err = None, repr(e)[:200]
            row = dict(requests=reqs, rows=rows, overlap=overlap, offset=offset,
                       arm=name, us=us, max_err_vs_first=err)
            print("BENCH", json.dumps(row), flush=True)
            results.append(row)
    out_dir = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    Path(out_dir, "bench_qsa_rpa_decode.json").write_text(json.dumps(results, indent=1))


def test_bench():
    main()


def check_exact(cache):
    """Lean vs schedule per-page fetch: identical DMAs, so outputs must match."""
    rng = np.random.default_rng(7)
    results = []
    for pattern, (reqs, rows), offset in itertools.product(
        ("random", "clustered"), ((6, 24), (2, 64), (8, 32)), (0, 1)
    ):
        q, gate, blocks, lengths, table = make_inputs(
            rng, requests=reqs, rows=rows, overlap=0.0, pattern=pattern
        )
        outs = []
        for kv_fetch in (0, 1):
            attend = jax.jit(
                functools.partial(
                    qsa_rpa.rpa_attention, kv_heads=1, ratio=RATIO, k_scale=0.375,
                    v_scale=2.5, lean_fetch=bool(kv_fetch), decode_only=True,
                    **({"kv_head_offset": 1} if offset else {}),
                ),
                donate_argnums=(5,),
            )
            out, cache = attend(q, gate, blocks, lengths, table, cache)
            outs.append(np.asarray(out.astype(jnp.float32)))
        row = dict(mode="exact", pattern=pattern, rows=rows, offset=offset,
                   max_abs=max(float(np.max(np.abs(outs[0] - o))) for o in outs),
                   nonzero=float(np.abs(outs[0]).max()))
        print("EXACT", json.dumps(row), flush=True)
        results.append(row)
        assert row["max_abs"] == 0 and row["nonzero"] > 0
    return cache, results


def check_slot_exact(cache):
    """Slot read vs both heads with zero queries: same math, same order."""
    rng = np.random.default_rng(11)
    results = []
    for pattern, (reqs, rows), offset in itertools.product(
        ("random", "clustered"), ((6, 24), (2, 64), (8, 32)), (0, 1)
    ):
        q, gate, blocks, lengths, table = make_inputs(
            rng, requests=reqs, rows=rows, overlap=0.0, pattern=pattern
        )
        outs = []
        for slot_read in (False, True):
            attend = jax.jit(
                functools.partial(
                    qsa_rpa.rpa_attention, kv_heads=1, ratio=RATIO, k_scale=0.375,
                    v_scale=2.5, lean_fetch=True, decode_only=True,
                    slot_read=slot_read,
                    **({"kv_head_offset": 1} if offset else {}),
                ),
                donate_argnums=(5,),
            )
            out, cache = attend(q, gate, blocks, lengths, table, cache)
            outs.append(np.asarray(out.astype(jnp.float32)))
        row = dict(mode="slot_exact", pattern=pattern, rows=rows, offset=offset,
                   max_abs=float(np.max(np.abs(outs[0] - outs[1]))),
                   nonzero=float(np.abs(outs[0]).max()))
        print("EXACT", json.dumps(row), flush=True)
        results.append(row)
        assert row["max_abs"] == 0 and row["nonzero"] > 0
    return cache, results


def test_slot_read(monkeypatch):
    monkeypatch.setenv("BENCH_CASES", "6x24,8x32,2x64")
    monkeypatch.setenv("BENCH_PATTERNS", "random")
    monkeypatch.setenv("BENCH_ARMS", "0000,0100,0110,0111,0011")
    cache = make_cache()
    cache, results = check_slot_exact(cache)
    main_step(cache, results)
    out_dir = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    Path(out_dir, "bench_qsa_slot_read.json").write_text(json.dumps(results, indent=1))


def test_slot_read_xprof(monkeypatch):
    monkeypatch.setenv("BENCH_XPROF", "1")
    monkeypatch.setenv("BENCH_CASES", "6x24")
    monkeypatch.setenv("BENCH_PATTERNS", "random")
    monkeypatch.setenv("BENCH_ARMS", "0110,0111")
    main_step(make_cache(), [])


def _one_dma_copy_in(self, step, slot, total_steps):
    """Diagnostic only (wrong data): each lane's tile moves the same bytes as
    the per-page fetch in ONE contiguous DMA, isolating descriptor issue."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.experimental.batched_rpa import bref_override

    is_no_op, block_idx = bref_override._step_guard(step, total_steps)
    kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref = self.src_ref
    sem = self.sem_recvs.at[slot]
    vmem_dst = self.window_ref.at[slot, :, :, : self.cfgs.kv_hbm_stride]
    flat = kv_cache_hbm.reshape(-1, *kv_cache_hbm.shape[2:])
    page_size = self.cfgs.serve.page_size
    for b in range(self.cfgs.batch_size):
        total = 0
        for i in range(self.cfgs.bkv_p_cache):
            total += schedule_ref.dma_kv_cache[block_idx, b, i, 2]
        total = jnp.where(is_no_op, 0, total)
        p0 = schedule_ref.dma_kv_cache[block_idx, b, 0, 0]
        src = page_indices_ref[p0] * page_size
        pltpu.make_async_copy(
            flat.at[pl.ds(src, total)], vmem_dst.at[b, pl.ds(0, total)], sem
        ).start()
        entry = schedule_ref.dma_kv_new[block_idx, b, 0]
        new_total = 0
        for i in range(self.cfgs.bkv_p_new):
            new_total += schedule_ref.dma_kv_new[block_idx, b, i].fetch_val
        new_total = jnp.where(is_no_op, 0, new_total)
        pltpu.make_async_copy(
            new_kv_hbm.at[pl.ds(entry.fetch_hbm[...], new_total)],
            vmem_dst.at[b, pl.ds(entry.fetch_vmem[...], new_total)],
            sem,
        ).start()


def check_compact_exact(cache):
    """Compact vs per-page schedule. At the same KV block the kernel sees the
    same data in the same order (bit for bit); other blocks change the online
    softmax partition, so their difference is reported, not asserted."""
    rng = np.random.default_rng(13)
    results = []
    for (reqs, rows), offset in itertools.product(((6, 24), (2, 64)), (0, 1)):
        q, gate, blocks, lengths, table = make_inputs(
            rng, requests=reqs, rows=rows, overlap=0.0
        )
        outs = {}
        for name, kw in (
            ("base", {}),
            ("c256x8", dict(compact=True, bkv_size=256, batch_size=8)),
            ("c512x8", dict(compact=True, bkv_size=512, batch_size=8)),
            ("c1024x8", dict(compact=True, bkv_size=1024, batch_size=8)),
            ("c1024x4", dict(compact=True, bkv_size=1024, batch_size=4)),
        ):
            attend = jax.jit(
                functools.partial(
                    qsa_rpa._rpa_attention_batch, kv_heads=1, ratio=RATIO,
                    k_scale=0.375, v_scale=2.5, lean_fetch=True, decode_only=True,
                    slot_read=True, **kw,
                    **({"kv_head_offset": 1} if offset else {}),
                ),
                donate_argnums=(5,),
            )
            try:
                out, cache = attend(q, gate, blocks, lengths, table, cache)
                outs[name] = np.asarray(out.astype(jnp.float32))
            except Exception as e:  # noqa: BLE001
                print("COMPACT_FAIL", name, repr(e)[:300], flush=True)
        base = outs["base"]
        row = dict(mode="compact_exact", rows=rows, offset=offset,
                   nonzero=float(np.abs(base).max()),
                   **{k: float(np.max(np.abs(v - base))) for k, v in outs.items()})
        print("EXACT", json.dumps(row), flush=True)
        results.append(row)
        assert row.get("c256x8", 1) == 0
    return cache, results


def test_compact(monkeypatch):
    monkeypatch.setenv("BENCH_CASES", "6x24,8x32,2x64")
    monkeypatch.setenv("BENCH_PATTERNS", "random")
    monkeypatch.setenv(
        "BENCH_ARMS",
        "0111,0111/256/8,0111/512/8,0111/512/4,0111/1024/8,0111/1024/4,"
        "0111/1152/4,0111/2176/2",
    )
    cache = make_cache()
    cache, results = check_compact_exact(cache)
    main_step(cache, results)
    out_dir = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    Path(out_dir, "bench_qsa_compact.json").write_text(json.dumps(results, indent=1))


def test_diag_one_dma(monkeypatch):
    """Lean per-page fetch vs one DMA per lane-tile (same bytes)."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import bref_override

    monkeypatch.setenv("BENCH_CASES", "6x24,8x32")
    monkeypatch.setenv("BENCH_PATTERNS", "random")
    monkeypatch.setenv("BENCH_ARMS", "0111")
    cache = make_cache()
    results = []
    cache = main_step(cache, results)
    monkeypatch.setattr(
        bref_override.KVBufferedRefHeadAlongSublane, "copy_in", _one_dma_copy_in
    )
    print("DIAG one DMA per lane-tile:", flush=True)
    main_step(cache, results)


def test_lean_fetch(monkeypatch):
    monkeypatch.setenv("BENCH_CASES", "6x24,8x32,2x64")
    monkeypatch.setenv("BENCH_PATTERNS", "random,clustered")
    monkeypatch.setenv("BENCH_ARMS", "010,011,110")
    cache = make_cache()
    cache, results = check_exact(cache)
    main_step(cache, results)
    out_dir = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    Path(out_dir, "bench_qsa_lean_fetch.json").write_text(json.dumps(results, indent=1))


def test_lean_fetch_xprof(monkeypatch):
    monkeypatch.setenv("BENCH_XPROF", "1")
    monkeypatch.setenv("BENCH_CASES", "6x24")
    monkeypatch.setenv("BENCH_PATTERNS", "random,clustered")
    monkeypatch.setenv("BENCH_ARMS", "010,011")
    main_step(make_cache(), [])


def test_bench_xprof(monkeypatch):
    monkeypatch.setenv("BENCH_XPROF", "1")
    monkeypatch.setenv("BENCH_CASES", "2x64,6x24")
    main()


def _no_convert_strided_load_bkv(kv_in_vref, b_idx, start, *, cfgs):
    """Diagnostic only (wrong values): K and V are the packed u32 words
    reinterpreted as f32, so the kernel does no FP8 unpack/convert at all."""
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.experimental.batched_rpa import utils

    start //= cfgs.serve.packing_kv
    kv_ref = kv_in_vref.at[b_idx].bitcast(jnp.uint32).reshape(
        -1, cfgs.aligned_kv_head_dim
    )
    kv_ref = pltpu.annotate(kv_ref, no_hazard_no_deps=True)
    kv = utils.strided_load(
        kv_ref, start, cfgs.bkv_sz * cfgs.bkv_stride, cfgs.bkv_stride
    )
    word = pltpu.bitcast(kv, jnp.float32)
    pairs = 1 if cfgs.serve.kv_slot is not None else cfgs.serve.packing_kv // 2
    return [(word, word)] * pairs


def test_diag_no_convert(monkeypatch):
    """Step time with and without the in-kernel FP8 unpack (upper bound on (b))."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import kernel

    monkeypatch.setenv("BENCH_CASES", "6x24,8x32,2x64")
    monkeypatch.setenv("BENCH_PATTERNS", "random")
    monkeypatch.setenv("BENCH_ARMS", "0111")
    cache = make_cache()
    results = []
    cache = main_step(cache, results)
    monkeypatch.setattr(kernel, "strided_load_bkv", _no_convert_strided_load_bkv)
    print("DIAG no FP8 unpack:", flush=True)
    main_step(cache, results)


def check_fast_exact(cache):
    """Fast full-tile fetch vs lean fetch: identical DMAs' bytes, so outputs
    must match bit for bit (separately jitted: fast_kv_fetch is static)."""
    rng = np.random.default_rng(19)
    results = []
    for pattern, (reqs, rows), offset in itertools.product(
        ("random", "clustered"), ((2, 8), (6, 24), (8, 32), (2, 64)), (0, 1)
    ):
        q, gate, blocks, lengths, table = make_inputs(
            rng, requests=reqs, rows=rows, overlap=0.0, pattern=pattern
        )
        outs = []
        for fast in (False, True):
            attend = jax.jit(
                functools.partial(
                    qsa_rpa.rpa_attention, kv_heads=1, ratio=RATIO, k_scale=0.375,
                    v_scale=2.5, lean_fetch=True, decode_only=True, slot_read=True,
                    fast_fetch=fast, **({"kv_head_offset": 1} if offset else {}),
                ),
                donate_argnums=(5,),
            )
            out, cache = attend(q, gate, blocks, lengths, table, cache)
            outs.append(np.asarray(out.astype(jnp.float32)))
        row = dict(mode="fast_exact", pattern=pattern, rows=rows, offset=offset,
                   max_abs=float(np.max(np.abs(outs[0] - outs[1]))),
                   nonzero=float(np.abs(outs[0]).max()))
        print("EXACT", json.dumps(row), flush=True)
        results.append(row)
        assert row["max_abs"] == 0 and row["nonzero"] > 0
    return cache, results


def test_fast_fetch(monkeypatch):
    """Same pod: lean vs fast fetch (static flag, retraced), 8/24/32/64 rows."""
    monkeypatch.setenv("BENCH_CASES", "2x8,6x24,8x32,2x64")
    monkeypatch.setenv("BENCH_PATTERNS", "random")
    cache = make_cache()
    cache, results = check_fast_exact(cache)
    for fast in ("0", "1", "0", "1"):
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_FAST_FETCH", fast)
        monkeypatch.setenv("BENCH_ARMS", "0111")
        print("FAST", fast, flush=True)
        cache = main_step(cache, results)
    out_dir = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    Path(out_dir, "bench_qsa_fast_fetch.json").write_text(json.dumps(results, indent=1))


def test_diag_combined(monkeypatch):
    """Same pod: baseline, no-unpack, one-DMA-per-tile, and both together.
    If only 'both' is faster, the scalar DMA-issue chain and the vector unpack
    are each on the critical path; if none is, the step waits on memory."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import bref_override, kernel

    monkeypatch.setenv("BENCH_CASES", "6x24,8x32")
    monkeypatch.setenv("BENCH_PATTERNS", "random")
    monkeypatch.setenv("BENCH_ARMS", "0111")
    cache = make_cache()
    results = []
    copy_in = bref_override.KVBufferedRefHeadAlongSublane.copy_in
    load = kernel.strided_load_bkv
    for name, unpack, dma in (
        ("base", load, copy_in),
        ("no_unpack", _no_convert_strided_load_bkv, copy_in),
        ("one_dma", load, _one_dma_copy_in),
        ("both", _no_convert_strided_load_bkv, _one_dma_copy_in),
    ):
        monkeypatch.setattr(kernel, "strided_load_bkv", unpack)
        monkeypatch.setattr(
            bref_override.KVBufferedRefHeadAlongSublane, "copy_in", dma
        )
        # ragged_paged_attention is a module-level jax.jit with the same static
        # arguments in every arm: without this its cached executable (traced
        # with the unpatched kernel) would be reused.
        jax.clear_caches()
        print("DIAG", name, flush=True)
        cache = main_step(cache, results)


if __name__ == "__main__":
    main()
