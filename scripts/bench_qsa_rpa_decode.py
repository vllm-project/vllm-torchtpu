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
                    qs[layer],
                    gates[layer],
                    blocks[layer],
                    lengths,
                    table,
                    cache,
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
            layers.append(
                (
                    jnp.asarray(q, jnp.bfloat16),
                    jnp.asarray(g, jnp.bfloat16),
                    jnp.asarray(pick_blocks(rng, lengths, pattern)),
                    base[3],
                    base[4],
                )
            )
        for fused, shared, kv_fetch, *rest in arms:
            fused, shared = bool(fused), bool(shared)
            slot = rest[0] if rest else 0
            bkv, lanes = rest[1:3] if len(rest) >= 3 else (0, 0)

            def attend(
                q,
                g,
                b,
                lens,
                t,
                cache,
                offset,
                fused=fused,
                shared=shared,
                kv_fetch=kv_fetch,
                slot=slot,
                bkv=bkv,
                lanes=lanes,
            ):
                os.environ["VLLM_TPU_QSA_DECODE_LEAN_FETCH"] = str(kv_fetch)
                os.environ["VLLM_TPU_QSA_DECODE_SLOT_READ"] = str(slot)
                os.environ["VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE"] = str(int(bkv > 0))
                if bkv:
                    os.environ["VLLM_TPU_QSA_DECODE_RPA_BKV"] = str(bkv)
                    os.environ["VLLM_TPU_QSA_DECODE_RPA_LANES"] = str(lanes)
                return qsa_rpa.decode_attention(
                    q,
                    g,
                    b,
                    lens,
                    t,
                    cache,
                    kv_heads=1,
                    ratio=RATIO,
                    k_scale=1.0,
                    v_scale=1.0,
                    fused=fused,
                    shared_schedule=shared,
                    **({"kv_head_offset": 1} if offset else {}),
                )

            trace = None
            if os.environ.get("BENCH_XPROF"):
                trace = os.path.join(
                    os.environ.get("CDK_OUTPUT_DIR", "/tmp"),
                    "xprof",
                    f"r{reqs}x{rows}_{pattern}_f{int(fused)}_s{int(shared)}"
                    f"_k{kv_fetch}_r{slot}_b{bkv}x{lanes}",
                )
            try:
                us, cache = bench_step(attend, layers, cache, trace)
            except Exception as e:  # noqa: BLE001
                us = repr(e)[:200]
            row = dict(
                mode="step",
                requests=reqs,
                rows=rows,
                pattern=pattern,
                fused=fused,
                shared=shared,
                kv_fetch=kv_fetch,
                slot=slot,
                bkv=bkv,
                lanes=lanes,
                us_per_step=us,
            )
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
    blocks_cfg = [
        (256, 8),
        (512, 8),
        (1024, 8),
        (2048, 8),
        (256, 16),
        (512, 16),
        (256, 4),
        (128, 8),
    ]
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
            row = dict(
                requests=reqs,
                rows=rows,
                overlap=overlap,
                offset=offset,
                arm=name,
                us=us,
                max_err_vs_first=err,
            )
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
                    qsa_rpa.rpa_attention,
                    kv_heads=1,
                    ratio=RATIO,
                    k_scale=0.375,
                    v_scale=2.5,
                    lean_fetch=bool(kv_fetch),
                    decode_only=True,
                    **({"kv_head_offset": 1} if offset else {}),
                ),
                donate_argnums=(5,),
            )
            out, cache = attend(q, gate, blocks, lengths, table, cache)
            outs.append(np.asarray(out.astype(jnp.float32)))
        row = dict(
            mode="exact",
            pattern=pattern,
            rows=rows,
            offset=offset,
            max_abs=max(float(np.max(np.abs(outs[0] - o))) for o in outs),
            nonzero=float(np.abs(outs[0]).max()),
        )
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
                    qsa_rpa.rpa_attention,
                    kv_heads=1,
                    ratio=RATIO,
                    k_scale=0.375,
                    v_scale=2.5,
                    lean_fetch=True,
                    decode_only=True,
                    slot_read=slot_read,
                    **({"kv_head_offset": 1} if offset else {}),
                ),
                donate_argnums=(5,),
            )
            out, cache = attend(q, gate, blocks, lengths, table, cache)
            outs.append(np.asarray(out.astype(jnp.float32)))
        row = dict(
            mode="slot_exact",
            pattern=pattern,
            rows=rows,
            offset=offset,
            max_abs=float(np.max(np.abs(outs[0] - outs[1]))),
            nonzero=float(np.abs(outs[0]).max()),
        )
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
                    qsa_rpa._rpa_attention_batch,
                    kv_heads=1,
                    ratio=RATIO,
                    k_scale=0.375,
                    v_scale=2.5,
                    lean_fetch=True,
                    decode_only=True,
                    slot_read=True,
                    **kw,
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
        row = dict(
            mode="compact_exact",
            rows=rows,
            offset=offset,
            nonzero=float(np.abs(base).max()),
            **{k: float(np.max(np.abs(v - base))) for k, v in outs.items()},
        )
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
    kv_ref = (
        kv_in_vref.at[b_idx].bitcast(jnp.uint32).reshape(-1, cfgs.aligned_kv_head_dim)
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
                    qsa_rpa.rpa_attention,
                    kv_heads=1,
                    ratio=RATIO,
                    k_scale=0.375,
                    v_scale=2.5,
                    lean_fetch=True,
                    decode_only=True,
                    slot_read=True,
                    fast_fetch=fast,
                    **({"kv_head_offset": 1} if offset else {}),
                ),
                donate_argnums=(5,),
            )
            out, cache = attend(q, gate, blocks, lengths, table, cache)
            outs.append(np.asarray(out.astype(jnp.float32)))
        row = dict(
            mode="fast_exact",
            pattern=pattern,
            rows=rows,
            offset=offset,
            max_abs=float(np.max(np.abs(outs[0] - outs[1]))),
            nonzero=float(np.abs(outs[0]).max()),
        )
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
        monkeypatch.setattr(bref_override.KVBufferedRefHeadAlongSublane, "copy_in", dma)
        # ragged_paged_attention is a module-level jax.jit with the same static
        # arguments in every arm: without this its cached executable (traced
        # with the unpatched kernel) would be reused.
        jax.clear_caches()
        print("DIAG", name, flush=True)
        cache = main_step(cache, results)


# --- Served DCP4 decode shape: zero-size descriptor skip and per-descriptor cost ---

F9_ENV = {
    "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_RPA_BKV": "512",
    "VLLM_TPU_QSA_DECODE_RPA_LANES": "4",
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": "128",
    "VLLM_TPU_QSA_FUSED_SPARSE_DECODE": "0",
    "VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY": "0",
}
F9_DCP = 4
F9_Q_HEADS = 12  # the DCP group's gathered heads attend on every rank


def f9_inputs(rng, *, requests, rows, pattern, pages=4096, ctx=(95000, 110000)):
    """``requests`` x 4 verify rows padded to ``rows`` at long contexts, each row
    with its own 512-group selection (served verify rows), DCP4 tables."""
    cols = -(-ctx[1] // PAGE) + 4
    contexts = rng.integers(ctx[0], ctx[1], size=requests)
    table = np.zeros((rows, cols), np.int32)
    lengths = np.zeros((rows,), np.int32)
    blocks = np.full((rows, TOPK_GROUPS), -1, np.int32)
    perm = rng.permutation(pages)
    for r in range(requests):
        n_pages = -(-(contexts[r] + 4) // PAGE)
        own = np.resize(perm[(r * cols) % pages :], n_pages)
        for t in range(4):
            i = r * 4 + t
            if i >= rows:
                break
            table[i, :n_pages] = own
            lengths[i] = contexts[r] + t + 1
            visible = lengths[i] // RATIO
            if pattern == "clustered":
                blocks[i] = clustered_pick(rng, visible)
            else:
                blocks[i] = rng.choice(visible, TOPK_GROUPS, replace=False)
    return jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table)


def f9_counts(blocks, lengths, rank):
    """Lane-tiles, real page DMAs and zero-size page DMAs of one rank's call
    (qsa_rpa._dcp_page_metadata ownership, 512-token tiles of 4-token pages)."""
    blocks, lengths = np.asarray(blocks), np.asarray(lengths)
    tiles = real = 0
    for row, length in zip(blocks, lengths):
        if length <= 0:
            continue
        complete = length // RATIO
        count = min(complete, TOPK_GROUPS)
        owned = int(np.sum((row[:count] // (PAGE // RATIO)) % F9_DCP == rank))
        tail = length % RATIO if (complete // (PAGE // RATIO)) % F9_DCP == rank else 0
        tokens = owned * RATIO + tail
        tiles += -(-tokens // 512)
        real += -(-tokens // RATIO)
    tiles, real = int(tiles), int(real)
    return dict(
        tiles=tiles, real=real, zero=tiles * 128 - real, descriptors=tiles * 129
    )


def _f9_attend(q, g, b, lens, t, cache, offset, rank):
    return qsa_rpa.decode_attention(
        q, g, b, lens, t, cache, kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5,
        dcp=(rank, F9_DCP), return_lse=True,
        **({"kv_head_offset": 1} if offset else {}),
    )


def _f9_set_env(skip):
    for k, v in F9_ENV.items():
        os.environ[k] = v
    os.environ["VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY"] = str(skip)


def _f9_compact_diag_copy_in(mode):
    """Diagnostic copy_in for the compact fetch (wrong data, same bytes).

    ``one_dma``: each lane-tile's tokens in ONE DMA. ``zero_size``: the same one
    DMA plus every page slot's descriptor with its full scalar setup but size 0,
    so the difference to ``one_dma`` is the issue cost of the page descriptors
    alone and the difference to the real fetch is what moving data adds."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.experimental.batched_rpa import bref_override

    def copy_in(self, step, slot, total_steps):
        is_no_op, block_idx = bref_override._step_guard(step, total_steps)
        kv_cache_hbm, new_kv_hbm, schedule_ref, page_indices_ref = self.src_ref
        sem = self.sem_recvs.at[slot]
        vmem_dst = self.window_ref.at[slot, :, :, : self.cfgs.kv_hbm_stride]
        flat = kv_cache_hbm.reshape(-1, *kv_cache_hbm.shape[2:])
        page_size = self.cfgs.serve.page_size
        last_index = self.cfgs.serve.num_page_indices - 1
        for b in range(self.cfgs.batch_size):
            p0 = schedule_ref.dma_kv_cache[block_idx, b, 0, 0]
            tokens = schedule_ref.dma_kv_cache[block_idx, b, 0, 2]
            tokens = jnp.where(is_no_op, 0, tokens)
            if mode == "zero_size":
                zero = jnp.minimum(tokens, 0)
                for i in range(self.cfgs.bkv_p_cache):
                    sz = jnp.clip(tokens - i * page_size, 0, page_size)
                    sz = jnp.minimum(sz, zero)
                    p_idx = jnp.minimum(p0 + i, last_index) if i else p0
                    src_off = page_indices_ref[p_idx] * page_size
                    pltpu.make_async_copy(
                        flat.at[pl.ds(src_off, sz)],
                        vmem_dst.at[b, pl.ds(i * page_size, sz)],
                        sem,
                    ).start()
            src = jnp.minimum(
                page_indices_ref[p0] * page_size, flat.shape[0] - self.cfgs.bkv_sz
            )
            pltpu.make_async_copy(
                flat.at[pl.ds(src, tokens)], vmem_dst.at[b, pl.ds(0, tokens)], sem
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

    return copy_in


def f9_exact(cache, results):
    """SKIP_EMPTY vs the served fetch on the DCP4 + LSE route: out and LSE bit
    for bit, every rank, both paired slots, random and clustered selections."""
    rng = np.random.default_rng(23)
    cases = [(2, 8, (16,)), (6, 24, (16,)), (24, 128, (8, 16, 32)), (32, 128, (16,))]
    ok = True
    patterns = ("random", "clustered")
    for (reqs, rows, skips), pattern in itertools.product(cases, patterns):
        blocks, lengths, table = f9_inputs(
            rng, requests=reqs, rows=rows, pattern=pattern
        )
        q = jnp.asarray(rng.standard_normal((rows, F9_Q_HEADS, D)), jnp.bfloat16)
        g = jnp.asarray(rng.standard_normal((rows, F9_Q_HEADS, D)), jnp.bfloat16)
        for offset in (0, 1):
            outs = {}
            for skip in (0, *skips):
                _f9_set_env(skip)
                jax.clear_caches()
                fn = jax.jit(
                    functools.partial(_f9_attend, offset=offset), donate_argnums=(5,)
                )
                for rank in range(F9_DCP):
                    r = jnp.asarray([rank], jnp.int32)
                    out, lse, cache = fn(q, g, blocks, lengths, table, cache, rank=r)
                    outs[(skip, rank)] = (
                        np.asarray(out, np.float32), np.asarray(lse, np.float32)
                    )
            for skip in skips:
                for rank in range(F9_DCP):
                    o0, l0 = outs[(0, rank)]
                    o1, l1 = outs[(skip, rank)]
                    same = o0.tobytes() == o1.tobytes() and l0.tobytes() == l1.tobytes()
                    row = dict(mode="f9_exact", requests=reqs, rows=rows,
                               pattern=pattern, offset=offset, skip=skip, rank=rank,
                               bit_equal=same,
                               out_max_abs=float(np.max(np.abs(o0 - o1))),
                               out_nonzero=float(np.abs(o0).max()),
                               lse_finite=int(np.isfinite(l0).sum()))
                    ok &= same
                    print("F9", json.dumps(row), flush=True)
                    results.append(row)
    print("F9_EXACT_ALL", ok, flush=True)
    return cache, ok


def f9_step(
    cache, results, *, label, skip=0, copy_in=None, cases=((24, 128), (32, 128))
):
    """12-layer step time on the served DCP4 decode route, per rank 0..3."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import bref_override

    original = bref_override.KVBufferedRefHeadAlongSublane.copy_in
    if copy_in is not None:
        bref_override.KVBufferedRefHeadAlongSublane.copy_in = copy_in
    try:
        for (reqs, rows), pattern in itertools.product(cases, ("random", "clustered")):
            rng = np.random.default_rng(reqs * 1000 + rows + (pattern == "clustered"))
            layers = [
                f9_inputs(rng, requests=reqs, rows=rows, pattern=pattern)
                for _ in range(12)
            ]
            lengths, table = layers[0][1], layers[0][2]
            blocks = jnp.stack([x[0] for x in layers])
            shape = (12, rows, F9_Q_HEADS, D)
            qs = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
            gs = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
            _f9_set_env(skip)
            jax.clear_caches()

            @functools.partial(jax.jit, donate_argnums=(0,))
            def run(cache, qs, gs, blocks, lengths, table, rank):
                def body(i, carry):
                    cache, acc = carry
                    for layer in range(12):
                        out, lse, cache = _f9_attend(
                            qs[layer], gs[layer], blocks[layer], lengths, table, cache,
                            layer % 2, rank,
                        )
                        acc = acc + out.astype(jnp.float32).sum()
                        acc = acc + jnp.where(jnp.isfinite(lse), lse, 0).sum()
                    return cache, acc

                return jax.lax.fori_loop(0, ITERS, body, (cache, jnp.float32(0)))

            for rank in range(F9_DCP):
                r = jnp.asarray([rank], jnp.int32)
                args = (qs, gs, blocks, lengths, table, r)
                cache, acc = run(cache, *args)
                jax.block_until_ready(acc)
                samples = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    cache, acc = run(cache, *args)
                    jax.block_until_ready(acc)
                    samples.append((time.perf_counter() - t0) / ITERS * 1e6)
                counts = [f9_counts(blocks[i], lengths, rank) for i in range(12)]
                total = {k: sum(c[k] for c in counts) for k in counts[0]}
                row = dict(mode="f9_step", arm=label, skip=skip, requests=reqs,
                           rows=rows, pattern=pattern, rank=rank,
                           us_per_step=float(np.median(samples)),
                           us_min=float(np.min(samples)), us_max=float(np.max(samples)),
                           **{f"step_{k}": v for k, v in total.items()})
                print("F9", json.dumps(row), flush=True)
                results.append(row)
    finally:
        bref_override.KVBufferedRefHeadAlongSublane.copy_in = original
        jax.clear_caches()
    return cache


def f9_main():
    cache = make_cache()
    results = []
    out = Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "f9_bench.json")
    try:
        cache, _ = f9_exact(cache, results)
        out.write_text(json.dumps(results, indent=1))
        arms = [
            ("base", 0, None),
            ("skip16", 16, None),
            ("one_dma", 0, _f9_compact_diag_copy_in("one_dma")),
            ("zero_size", 0, _f9_compact_diag_copy_in("zero_size")),
            ("skip8", 8, None),
            ("skip32", 32, None),
            ("base_again", 0, None),
        ]
        for label, skip, copy_in in arms:
            print("F9_ARM", label, flush=True)
            cache = f9_step(cache, results, label=label, skip=skip, copy_in=copy_in)
            out.write_text(json.dumps(results, indent=1))
    finally:
        out.write_text(json.dumps(results, indent=1))
        print("F9_DONE", out, flush=True)


# --- Union layout on the served DCP4 shape: one sequence per request over the
# union of its verify rows' owned pages (no per-row mask: an upper bound) ---

F9U_DATA = Path(
    os.environ.get(
        "F9U_DATA", Path(__file__).resolve().parents[1] / "scratch/f9_union"
    )
)
F9U_UNIT = PAGE // RATIO  # 128 groups per 512-token main-KV page, the DCP unit
F9U_PAGES = 512  # union page slots per sequence (max owned union seen: 410)


def f9u_case(name, pages=4096):
    """Dumped AgentX selections (served blocks layout), lengths and a DCP4 table."""
    z = np.load(F9U_DATA / f"{name}.npz")
    blocks, lengths, requests = z["blocks"], z["lengths"], int(z["requests"])
    rows = len(lengths)
    cols = -(-int(lengths.max()) // (F9_DCP * PAGE)) + 1
    rng = np.random.default_rng(len(name) * 7 + requests)
    perm = rng.permutation(pages)
    table = np.zeros((rows, cols), np.int32)
    for r in range(requests):
        own = np.resize(perm[(r * cols) % pages :], cols)
        table[4 * r : 4 * r + 4] = own
    return blocks, lengths, table, requests


def f9u_union_meta(blocks, lengths, table, requests, rank):
    """Per request: the union of its 4 rows' owned groups as virtual pages."""
    seqs = requests
    pages = np.zeros((seqs, F9U_PAGES), np.int32)
    kv_lens = np.zeros((seqs,), np.int32)
    q_lens = np.zeros((seqs,), np.int32)
    for r in range(requests):
        groups = []
        for i in range(4 * r, 4 * r + 4):
            length = int(lengths[i])
            complete = length // RATIO
            count = min(complete, TOPK_GROUPS)
            groups.append(blocks[i, :count])
            if length % RATIO:
                groups.append(np.array([complete], np.int32))
        g = np.unique(np.concatenate(groups))
        g = g[(g // F9U_UNIT) % F9_DCP == rank][:F9U_PAGES]
        if not len(g):
            continue
        entry = table[4 * r, g // (F9_DCP * F9U_UNIT)]
        pages[r, : len(g)] = entry * F9U_UNIT + g % F9U_UNIT
        kv_lens[r] = len(g) * RATIO
        q_lens[r] = 4
    cu = np.concatenate(([0], np.cumsum(q_lens))).astype(np.int32)
    n = np.int32(seqs)
    return pages.reshape(-1), kv_lens, cu, np.array([n, n, n], np.int32)


def _f9u_rpa(query, pages, kv_lens, cu, dist, cache, slot, *, bq, bkv, lanes, dqs):
    """RPA on precomputed metadata, with _rpa_attention_batch's served arguments."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

    virtual_cache = cache.reshape(-1, RATIO, *cache.shape[2:])
    dummy = jnp.zeros((query.shape[0], 1, query.shape[-1]), cache.dtype)
    out, next_cache, lse = wrapper.ragged_paged_attention(
        query, dummy, dummy, virtual_cache, kv_lens, pages, cu, dist,
        sm_scale=D ** -0.5, q_scale=None, k_scale=0.375, v_scale=2.5,
        out_dtype=jnp.dtype(jnp.float32), skip_kv_update=True,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=configs.BlockSizes(
            bq_sz=bq, bq_c_sz=bq, bkv_sz=bkv, batch_size=lanes
        ),
        prefill_block_sizes=configs.BlockSizes(
            bq_sz=bq, bq_c_sz=bq, bkv_sz=bkv, batch_size=1
        ),
        decode_only=True, lean_kv_fetch=True, fast_kv_fetch=True, kv_slot=slot,
        compact_kv_schedule=True, return_lse=True, decode_query_size=dqs,
    )
    return out, lse, next_cache.reshape(cache.shape)


def f9u_time(cache, run, args):
    cache, acc = run(cache, *args)
    jax.block_until_ready(acc)
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        cache, acc = run(cache, *args)
        jax.block_until_ready(acc)
        samples.append((time.perf_counter() - t0) / ITERS * 1e6)
    med, lo, hi = np.median(samples), np.min(samples), np.max(samples)
    return cache, float(med), float(lo), float(hi)


def f9u_arm(cache, results, *, case, label, kind, bkv=512, one_dma=False):
    """12-layer step of one arm on all four DCP ranks.

    kind "served": decode_attention per row (the served route, metadata inside).
    kind "rows": the served per-row RPA on precomputed metadata (kernel only).
    kind "union": one 4-query sequence per request over its union pages."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import bref_override

    blocks, lengths, table, requests = f9u_case(case)
    rows = len(lengths)
    rng = np.random.default_rng(5)
    shape = (12, rows, F9_Q_HEADS, D)
    qs = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    gs = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    bq, dqs = (4, 4) if kind == "union" else (1, 1)

    @functools.partial(jax.jit, donate_argnums=(0,))
    def run_served(cache, qs, gs, blocks, lengths, table, rank):
        def body(i, carry):
            cache, acc = carry
            for layer in range(12):
                out, lse, cache = _f9_attend(
                    qs[layer], gs[layer], blocks[layer], lengths, table, cache,
                    layer % 2, rank,
                )
                acc = acc + out.astype(jnp.float32).sum()
                acc = acc + jnp.where(jnp.isfinite(lse), lse, 0).sum()
            return cache, acc

        return jax.lax.fori_loop(0, ITERS, body, (cache, jnp.float32(0)))

    @functools.partial(jax.jit, donate_argnums=(0,))
    def run_rpa(cache, q, pages, kv_lens, cu, dist):
        def body(i, carry):
            cache, acc = carry
            for layer in range(12):
                out, lse, cache = _f9u_rpa(
                    q[layer], pages[layer], kv_lens[layer], cu[layer], dist[layer],
                    cache, layer % 2, bq=bq, bkv=bkv, lanes=4, dqs=dqs,
                )
                acc = acc + out.astype(jnp.float32).sum()
                acc = acc + jnp.where(jnp.isfinite(lse), lse, 0).sum()
            return cache, acc

        return jax.lax.fori_loop(0, ITERS, body, (cache, jnp.float32(0)))

    original = bref_override.KVBufferedRefHeadAlongSublane.copy_in
    if one_dma:
        bref_override.KVBufferedRefHeadAlongSublane.copy_in = _f9_compact_diag_copy_in(
            "one_dma"
        )
    _f9_set_env(0)
    jax.clear_caches()
    try:
        for rank in range(F9_DCP):
            r = jnp.asarray([rank], jnp.int32)
            if kind == "union":
                metas = [
                    f9u_union_meta(blocks[i], lengths, table, requests, rank)
                    for i in range(12)
                ]
                slots = bkv // RATIO + min(1 + -(-(dqs - 1) // RATIO), bkv // RATIO)
                tiles = sum(int(np.sum(-(-m[1] // bkv))) for m in metas)
                desc = tiles * slots
                real = sum(int(np.sum(m[1] // RATIO)) for m in metas)
            else:
                counts = [f9_counts(blocks[i], lengths, rank) for i in range(12)]
                desc = sum(c["descriptors"] for c in counts)
                real = sum(c["real"] for c in counts)
            if kind == "served":
                run = run_served
                args = (qs, gs, jnp.asarray(blocks), jnp.asarray(lengths),
                        jnp.asarray(table), r)
            else:
                if kind == "rows":
                    metas = [
                        qsa_rpa.virtual_page_metadata(
                            jnp.asarray(blocks[i]), jnp.asarray(lengths),
                            jnp.asarray(table), ratio=RATIO, page_size=PAGE,
                            dcp=(r, F9_DCP),
                        )
                        for i in range(12)
                    ]
                    metas = [(m[0].reshape(-1), m[1], m[2], m[3]) for m in metas]
                    t_rows = rows
                else:
                    t_rows = 4 * requests
                stack = [
                    jnp.stack([jnp.asarray(m[j]) for m in metas]) for j in range(4)
                ]
                run = run_rpa
                args = (qs[:, :t_rows].astype(jnp.float32), *stack)
            try:
                cache, med, lo, hi = f9u_time(cache, run, args)
                err = None
            except Exception as e:  # noqa: BLE001
                med = lo = hi = None
                err = repr(e)[:400]
            row = dict(mode="f9u", case=case, requests=requests, arm=label, kind=kind,
                       bkv=bkv, one_dma=one_dma, rank=rank, us_per_step=med,
                       us_min=lo, us_max=hi, step_descriptors=int(desc),
                       step_real=int(real), error=err)
            if med is not None:
                row["us_per_request_layer"] = med / 12 / requests
                row["descriptors_per_request_layer"] = desc / 12 / requests
            print("F9U", json.dumps(row), flush=True)
            results.append(row)
    finally:
        bref_override.KVBufferedRefHeadAlongSublane.copy_in = original
        jax.clear_caches()
    return cache


def f9u_check(cache, results):
    """Union arm sanity: finite output for every query of a sequence with keys."""
    blocks, lengths, table, requests = f9u_case("s24")
    meta = f9u_union_meta(blocks[0], lengths, table, requests, 0)
    q = jnp.asarray(
        np.random.default_rng(3).standard_normal((4 * requests, F9_Q_HEADS, D)),
        jnp.float32,
    )
    _f9_set_env(0)
    for bkv in (640, 1024, 1280):
        fn = jax.jit(functools.partial(_f9u_rpa, slot=0, bq=4, bkv=bkv, lanes=4, dqs=4),
                     donate_argnums=(5,))
        try:
            out, lse, cache = fn(q, *map(jnp.asarray, meta), cache)
            out = np.asarray(out, np.float32)
            live = np.repeat(meta[1] > 0, 4)
            row = dict(mode="f9u_check", bkv=bkv,
                       finite=bool(np.isfinite(out[live]).all()),
                       nonzero=float(np.abs(out[live]).max()), error=None)
        except Exception as e:  # noqa: BLE001
            row = dict(mode="f9u_check", bkv=bkv, error=repr(e)[:400])
        print("F9U", json.dumps(row), flush=True)
        results.append(row)
    return cache


def f9u_main():
    cache = make_cache()
    results = []
    out = Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "f9u_bench.json")
    arms = [("served_rows", "served", 512, False), ("rows", "rows", 512, False),
            ("rows_one_dma", "rows", 512, True)]
    for bkv in (640, 1024, 1280):
        arms += [(f"union{bkv}", "union", bkv, False),
                 (f"union{bkv}_one_dma", "union", bkv, True)]
    try:
        cache = f9u_check(cache, results)
        for case in ("s24", "s32"):
            for label, kind, bkv, one in arms:
                print("F9U_ARM", case, label, flush=True)
                cache = f9u_arm(cache, results, case=case, label=label, kind=kind,
                                bkv=bkv, one_dma=one)
                out.write_text(json.dumps(results, indent=1))
    finally:
        out.write_text(json.dumps(results, indent=1))
        print("F9U_DONE", out, flush=True)


# --- F15: one sequence per request over its rows' union, with a per-token row
# mask (selection_bits): gate 1 against the per-row kernel, and the mask's cost ---

F15_TILE = 640
F15_P = 2080  # union page slots a sequence: >= 4 * 512 + 2 groups, whole tiles
F15_SEQS = 32


def f15_union_host(blocks, lengths, table, starts, rank, *, seqs, pages_per_seq=F15_P,
                   own_list=False):
    """One DCP rank's union inputs, built on the host.

    The design note's ``union_reference`` with the per-row path's slot rule
    (slot i < min(complete, 512), group >= 0, owned by the rank). Returns pages
    [seqs, P], sel [seqs, 4P], kv_lens [seqs], cu [seqs + 1] (= starts, padded
    sequences empty), row_live [T] (row has a set bit) and tails, the
    (virtual page, written tokens) of each request's owned final partial group.
    """
    blocks, lengths = np.asarray(blocks), np.asarray(lengths)
    table, starts = np.asarray(table), np.asarray(starts)
    requests = len(starts) - 1
    pages = np.zeros((seqs, pages_per_seq), np.int32)
    sel = np.zeros((seqs, pages_per_seq * RATIO), np.int64)
    kv_lens = np.zeros((seqs,), np.int32)
    row_live = np.zeros((len(lengths),), bool)
    tails = []
    unit = F9U_UNIT
    for r in range(requests):
        rows = range(int(starts[r]), int(starts[r + 1]))
        assert len(rows) <= 4
        gs, js, ns = [], [], []
        last = 0
        for j, i in enumerate(rows):
            length = int(lengths[i])
            last = max(last, length)
            if length <= 0:
                continue
            complete = length // RATIO
            width = blocks.shape[1] if own_list else min(complete, blocks.shape[1])
            g = blocks[i, :width].astype(np.int64)
            g = g[(g >= 0) & (g < complete) & ((g // unit) % F9_DCP == rank)]
            gs += [g, np.full(len(g), j)]
            ns.append(np.full(len(g), RATIO))
            if length % RATIO and (complete // unit) % F9_DCP == rank:
                gs += [np.array([complete]), np.array([j])]
                ns.append(np.array([length % RATIO]))
        if not ns:
            continue
        g = np.concatenate(gs[0::2]).astype(np.int64)
        j = np.concatenate(gs[1::2]).astype(np.int64)
        n = np.concatenate(ns)
        if not len(g):
            continue
        tail = last // RATIO if last % RATIO else -1
        body = np.unique(g[g != tail])
        has_tail = bool(np.any(g == tail))
        order = np.concatenate([body, [tail]]) if has_tail else body
        count = len(order)
        assert count <= pages_per_seq
        pos = np.where(g == tail, count - 1, np.searchsorted(body, g))
        words = np.zeros(count * RATIO, np.int64)
        for o in range(RATIO):
            hit = n > o
            np.bitwise_or.at(words, pos[hit] * RATIO + o, np.left_shift(1, j[hit]))
        entry = table[rows[0], order // (F9_DCP * unit)]
        pages[r, :count] = entry * unit + order % unit
        sel[r, : count * RATIO] = words
        kv_lens[r] = RATIO * count - ((RATIO - last % RATIO) if has_tail else 0)
        for jj, i in enumerate(rows):
            row_live[i] = bool(np.any((words >> jj) & 1))
        if has_tail:
            tails.append((int(pages[r, count - 1]), last % RATIO))
    cu = np.concatenate([starts, np.full(seqs - requests, starts[-1])]).astype(np.int32)
    return pages, sel.astype(np.int32), kv_lens, cu, row_live, tails


def f15_rows_check(blocks, lengths, table, starts, rank, meta, own_list=False):
    """Host check of f15_union_host: every row's attended (virtual page, token)
    set equals the per-row rule of qsa_rpa._dcp_page_metadata, nothing at or
    past kv_len is set, and padded sequences are empty."""
    pages, sel, kv_lens, cu, row_live, _ = meta
    blocks, lengths = np.asarray(blocks), np.asarray(lengths)
    table, starts = np.asarray(table), np.asarray(starts)
    unit = F9U_UNIT
    bad = 0
    for r in range(len(starts) - 1):
        tokens = sel[r].reshape(-1, RATIO)
        if np.any(sel[r, kv_lens[r] :]):
            bad += 1
        for j, i in enumerate(range(int(starts[r]), int(starts[r + 1]))):
            length = int(lengths[i])
            want = set()
            if length > 0:
                complete = length // RATIO
                width = blocks.shape[1] if own_list else min(complete, blocks.shape[1])
                for g in blocks[i, :width]:
                    if own_list and not 0 <= g < complete:
                        continue
                    if (g // unit) % F9_DCP == rank:
                        p = table[i, g // (F9_DCP * unit)] * unit + g % unit
                        want |= {(int(p), o) for o in range(RATIO)}
                if length % RATIO and (complete // unit) % F9_DCP == rank:
                    p = table[i, complete // (F9_DCP * unit)] * unit + complete % unit
                    want |= {(int(p), o) for o in range(length % RATIO)}
            hit = np.argwhere((tokens >> j) & 1)
            got = {(int(pages[r, k]), int(o)) for k, o in hit}
            bad += got != want
            bad += bool(want) != bool(row_live[i])
    for s in range(len(starts) - 1, len(kv_lens)):
        bad += bool(kv_lens[s] or cu[s + 1] != cu[s] or np.any(sel[s]))
    return bad


def f15_edge_case(rng, *, pages=4096, rows=64):
    """Requests for the note's failure modes 2-4, padded to ``rows`` rows:
    disjoint per-row group ranges (an ignored mask shows; rows lack keys in
    whole tiles), 1-3 row requests, a dead row inside a live request, contexts
    shorter than one 512-token page (ranks 1-3 own nothing), shorter than one
    4-token group, fewer than 512 complete groups, tails split over a rank
    boundary, identical selections, and a long context."""
    reqs = []

    def pick(complete, count=TOPK_GROUPS, lo=0, hi=None):
        hi = complete if hi is None else hi
        out = np.full(TOPK_GROUPS, -1, np.int32)
        g = rng.choice(np.arange(lo, hi), min(count, hi - lo), replace=False)
        out[: len(g)] = g
        return out

    base = 40000
    lens = [base + 1 + j for j in range(4)]
    reqs.append((lens, [pick(base // 4, lo=j * 2500, hi=(j + 1) * 2500) for j in range(4)]))
    for nrows, ctx in ((1, 5000), (2, 23000), (3, 61000)):
        lens = [ctx + 1 + j for j in range(nrows)]
        reqs.append((lens, [pick(n // 4) for n in lens]))
    lens = [30001, 0, 30003, 30004]
    reqs.append((lens, [pick(n // 4) if n else np.full(TOPK_GROUPS, -1, np.int32) for n in lens]))
    lens = [297, 298, 299, 300]
    reqs.append((lens, [pick(n // 4) for n in lens]))
    lens = [1, 2, 3, 4]
    reqs.append((lens, [pick(n // 4) if n >= 4 else np.full(TOPK_GROUPS, -1, np.int32) for n in lens]))
    lens = [1501, 1502, 1503, 1504]
    reqs.append((lens, [pick(n // 4) for n in lens]))
    lens = [2046, 2047, 2049, 2050]
    reqs.append((lens, [pick(n // 4) for n in lens]))
    shared = pick(77000 // 4)
    reqs.append(([77001, 77002, 77003, 77004], [shared] * 4))
    lens = [201001 + j for j in range(4)]
    reqs.append((lens, [pick(n // 4) for n in lens]))
    cols = -(-max(max(x[0]) for x in reqs) // (F9_DCP * PAGE)) + 1
    perm = rng.permutation(pages)
    blocks = np.full((rows, TOPK_GROUPS), -1, np.int32)
    lengths = np.zeros((rows,), np.int32)
    table = np.zeros((rows, cols), np.int32)
    starts, i = [0], 0
    for r, (lens, sels) in enumerate(reqs):
        own = perm[r * cols : (r + 1) * cols]
        for length, row in zip(lens, sels):
            lengths[i], blocks[i], table[i] = length, row, own
            i += 1
        starts.append(i)
    starts = np.asarray(starts, np.int32)
    assert starts[-1] <= rows - 8, starts[-1]
    return blocks, lengths, table, starts


def _f15_union_rpa(query, pages, sel, kv_lens, cu, cache, slot, *, bkv=F15_TILE, lanes=4):
    """The union kernel call: one sequence per request, q_len its rows, the
    request's union pages, and ``sel`` (None: causal over the union)."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

    virtual_cache = cache.reshape(-1, RATIO, *cache.shape[2:])
    dummy = jnp.zeros((query.shape[0], 1, query.shape[-1]), cache.dtype)
    seqs = kv_lens.shape[0]
    out, next_cache, lse = wrapper.ragged_paged_attention(
        query, dummy, dummy, virtual_cache, kv_lens, pages.reshape(-1), cu,
        jnp.full((3,), seqs, jnp.int32),
        sm_scale=D ** -0.5, q_scale=None, k_scale=0.375, v_scale=2.5,
        out_dtype=jnp.dtype(jnp.float32), skip_kv_update=True,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=configs.BlockSizes(
            bq_sz=4, bq_c_sz=4, bkv_sz=bkv, batch_size=lanes
        ),
        prefill_block_sizes=configs.BlockSizes(
            bq_sz=4, bq_c_sz=4, bkv_sz=bkv, batch_size=1
        ),
        decode_only=True, lean_kv_fetch=True, fast_kv_fetch=True, kv_slot=slot,
        compact_kv_schedule=True, return_lse=True, decode_query_size=4,
        **({"selection_bits": sel} if sel is not None else {}),
    )
    return out, lse, next_cache.reshape(cache.shape)


@functools.partial(jax.jit, static_argnames=("vpages",))
def _f15_nan_cache(idx, tail_p, tail_o, key, *, vpages):
    nan = jnp.full((vpages, RATIO, 1, 4, D), jnp.nan, jnp.float8_e4m3fn)
    vals = jax.random.normal(key, (len(idx), RATIO, 1, 4, D), jnp.bfloat16) * 0.5
    cache = nan.at[idx].set(vals.astype(jnp.float8_e4m3fn), mode="drop")
    cache = cache.at[tail_p, tail_o].set(jnp.nan, mode="drop")
    return cache.reshape(vpages // F9U_UNIT, PAGE, 1, 4, D)


def f15_nan_cache(meta, seed, pages=4096):
    """A cache whose only finite rows are the union's written tokens: NaN in
    every other 4-token page and in the unwritten tokens of each owned final
    partial group, so a read outside them shows as a non-finite output."""
    pages_arr, _, kv_lens, _, _, tails = meta
    vpages = pages * F9U_UNIT
    idx = [pages_arr[s, : -(-int(kv_lens[s]) // RATIO)] for s in range(len(kv_lens))]
    idx = np.unique(np.concatenate(idx)) if idx else np.zeros(0, np.int64)
    size = max(8192, -(-len(idx) // 8192) * 8192)
    idx = np.concatenate([idx, np.full(size - len(idx), vpages)]).astype(np.int32)
    tp = [(p, o) for p, w in tails for o in range(w, RATIO)]
    tp += [(vpages, 0)] * (128 - len(tp))
    tp = np.asarray(tp[:128], np.int32)
    return _f15_nan_cache(
        jnp.asarray(idx), jnp.asarray(tp[:, 0]), jnp.asarray(tp[:, 1]),
        jax.random.key(seed), vpages=vpages,
    )


def f15_schedule_check(cfgs, kv_lens, cu):
    """Rebuilds the union call's schedule (same inputs as the wrapper's selection
    mode) and checks that it holds exactly one task per KV tile of each sequence
    with keys, none for empty or padded sequences, and kv_lens tokens in all."""
    import dataclasses

    from vllm_torchtpu.kernels.experimental.batched_rpa import schedule

    serve = dataclasses.replace(
        cfgs.serve, kv_slot=None, kv_skip_group=0, selection_mask=False
    )
    cfgs = dataclasses.replace(cfgs, serve=serve)
    kv = jnp.asarray(kv_lens)
    seqs = len(kv_lens)
    sched = schedule.generate_rpa_metadata(
        jnp.asarray(cu), kv, kv, jnp.zeros_like(kv), jnp.full((3,), seqs, jnp.int32),
        cfgs=cfgs,
    )
    steps = int(np.asarray(sched.actual_steps)[0])
    lanes = cfgs.batch_size
    s_idx = np.asarray(sched.s_idx.data).reshape(-1, lanes)[:steps]
    k_idx = np.asarray(sched.k_idx.data).reshape(-1, lanes)[:steps]
    tokens = np.asarray(sched.dma_kv_cache.data).reshape(-1, lanes, 3)[:steps, :, 2]
    live = s_idx >= 0
    tasks = sorted(zip(s_idx[live].tolist(), k_idx[live].tolist()))
    bkv = cfgs.bkv_sz
    want = sorted(
        (s, k) for s in range(seqs) for k in range(-(-int(kv_lens[s]) // bkv))
    )
    return dict(
        tasks=len(tasks), want=len(want), same_tasks=tasks == want,
        tokens=int(tokens[live].sum()), kv_tokens=int(np.sum(kv_lens)),
        empty_seq_tasks=int(sum(kv_lens[s] == 0 for s, _ in tasks)),
    )


def f15_compare(ref, uni, row_live, query, padded):
    """Gate 1 for one call pair: rows live in the per-row path are exactly the
    union's live rows; for those, out within 1e-3 of the row's largest |out|
    and LSE within 1e-3, both finite; padded rows unchanged (nothing written)."""
    (o_r, l_r), (o_u, l_u) = ref, uni
    live_ref = np.isfinite(l_r).all(axis=1)
    n = len(live_ref)
    live = row_live[:n]
    out = dict(live_match=bool(np.array_equal(live_ref, live)), live_rows=int(live.sum()))
    if live.any():
        scale = np.abs(o_r[live]).reshape(live.sum(), -1).max(axis=1)
        d_out = np.abs(o_u[live] - o_r[live]).reshape(live.sum(), -1).max(axis=1)
        rel = d_out / np.maximum(scale, 1e-30)
        d_lse = np.abs(l_u[live] - l_r[live]).max(axis=1)
        out.update(
            out_rel_max=float(rel.max()), lse_abs_max=float(d_lse.max()),
            finite=bool(np.isfinite(o_u[live]).all() and np.isfinite(l_u[live]).all()
                        and np.isfinite(o_r[live]).all()),
            out_scale_min=float(scale.min()),
        )
        ok = out["finite"] and rel.max() <= 1e-3 and d_lse.max() <= 1e-3
    else:
        ok = True
    if padded.any():
        same = np.array_equal(o_u[padded], query[padded]) and bool(
            np.all(np.isneginf(l_u[padded]))
        )
        out["padded_untouched"] = bool(same)
        ok = ok and same
    out["ok"] = bool(ok and out["live_match"])
    return out


def f15_gate1(results, layers=range(12)):
    from vllm_torchtpu.kernels.experimental.batched_rpa import kernel as rpa_kernel

    cases = []
    for name in ("s24", "s32"):
        blocks, lengths, table, requests = f9u_case(name)
        starts = np.arange(0, 4 * requests + 1, 4, dtype=np.int32)
        for layer in layers:
            cases.append((f"{name}/L{layer}", blocks[layer], lengths, table, starts, F15_SEQS))
    rng = np.random.default_rng(15)
    cases.append(("edge", *f15_edge_case(rng), 16))
    _f9_set_env(0)
    jax.clear_caches()
    captured = {}
    original = rpa_kernel.rpa_kernel

    def spy(*args, **kwargs):
        if kwargs["cfgs"].serve.selection_mask:
            captured["cfgs"] = kwargs["cfgs"]
        return original(*args, **kwargs)

    rpa_kernel.rpa_kernel = spy
    ok_all = True
    try:
        ref_fns = {
            off: jax.jit(functools.partial(_f9_attend, offset=off), donate_argnums=(5,))
            for off in (0, 1)
        }
        uni_fns = {
            off: jax.jit(functools.partial(_f15_union_rpa, slot=off), donate_argnums=(5,))
            for off in (0, 1)
        }
        for c_idx, (name, blocks, lengths, table, starts, seqs) in enumerate(cases):
            rows = len(lengths)
            padded = np.arange(rows) >= starts[-1]
            q = jnp.asarray(
                np.random.default_rng(100 + c_idx).standard_normal((rows, F9_Q_HEADS, D)),
                jnp.bfloat16,
            )
            gate = jnp.zeros_like(q)
            q32 = q.astype(jnp.float32)
            q_np = np.asarray(q32)
            for rank in range(F9_DCP):
                try:
                    meta = f15_union_host(blocks, lengths, table, starts, rank, seqs=seqs)
                    host_bad = f15_rows_check(blocks, lengths, table, starts, rank, meta)
                    pages, sel, kv_lens, cu, row_live, tails = meta
                    cache = f15_nan_cache(meta, seed=c_idx * 8 + rank)
                    r = jnp.asarray([rank], jnp.int32)
                    for off in (0, 1):
                        o_r, l_r, cache = ref_fns[off](
                            q, gate, jnp.asarray(blocks), jnp.asarray(lengths),
                            jnp.asarray(table), cache, rank=r,
                        )
                        o_u, l_u, cache = uni_fns[off](
                            q32, jnp.asarray(pages), jnp.asarray(sel),
                            jnp.asarray(kv_lens), jnp.asarray(cu), cache,
                        )
                        row = dict(mode="f15_gate1", case=name, rank=rank, slot=off,
                                   host_bad=host_bad, union_pages=int(np.sum(-(-kv_lens // RATIO))),
                                   tails=len(tails),
                                   **f15_compare(
                                       (np.asarray(o_r, np.float32), np.asarray(l_r, np.float32)),
                                       (np.asarray(o_u, np.float32), np.asarray(l_u, np.float32)),
                                       row_live, q_np, padded,
                                   ))
                        row["ok"] = row["ok"] and host_bad == 0
                        if off == 0 and "cfgs" in captured:
                            sched = f15_schedule_check(captured["cfgs"], kv_lens, cu)
                            row["schedule"] = sched
                            row["ok"] = row["ok"] and sched["same_tasks"] and (
                                sched["tokens"] == sched["kv_tokens"]
                            ) and sched["empty_seq_tasks"] == 0
                        ok_all &= row["ok"]
                        print("F15", json.dumps(row), flush=True)
                        results.append(row)
                    del cache
                except Exception as e:  # noqa: BLE001
                    ok_all = False
                    row = dict(mode="f15_gate1", case=name, rank=rank, error=repr(e)[:1500])
                    print("F15", json.dumps(row), flush=True)
                    results.append(row)
    finally:
        rpa_kernel.rpa_kernel = original
        jax.clear_caches()
    print("F15_GATE1_ALL", ok_all, flush=True)
    return ok_all


def f15_exact(cache, meta, q, starts, slot):
    """Float64 attention of every live row over exactly its union keys (bit j
    of its request's words), from the cache's FP8 rows: (out [T, H, D] with
    zeros for rows without keys, lse [T, H] with -inf)."""
    pages, sel, kv_lens, _, row_live, _ = meta
    view = cache.reshape(-1, RATIO, *cache.shape[2:])
    used = [pages[r, : -(-int(kv_lens[r]) // RATIO)] for r in range(len(starts) - 1)]
    flat = np.unique(np.concatenate(used)) if used else np.zeros(0, np.int32)
    data = np.asarray(view[jnp.asarray(flat.astype(np.int32))]).astype(np.float64)
    where = {int(p): k for k, p in enumerate(flat)}
    q = np.asarray(q, np.float64)
    out = np.zeros(q.shape, np.float64)
    lse = np.full(q.shape[:2], -np.inf)
    for r in range(len(starts) - 1):
        n = -(-int(kv_lens[r]) // RATIO)
        idx = np.array([where[int(p)] for p in pages[r, :n]], np.int64)
        words = sel[r, : n * RATIO].reshape(n, RATIO)
        for j, i in enumerate(range(int(starts[r]), int(starts[r + 1]))):
            hit = np.argwhere((words >> j) & 1)
            if not len(hit):
                continue
            rows = data[idx[hit[:, 0]], hit[:, 1], 0]  # [keys, 4, D]
            k, v = rows[:, 2 * slot], rows[:, 2 * slot + 1]
            logits = q[i] @ k.T * (D ** -0.5) * 0.375  # [H, keys]
            top = logits.max(axis=1, keepdims=True)
            w = np.exp(logits - top)
            out[i] = (w @ v) * 2.5 / w.sum(axis=1, keepdims=True)
            lse[i] = top[:, 0] + np.log(w.sum(axis=1))
    return out, lse


def f15_err(o, l, o_x, l_x, live):
    """Largest |out - exact| over a row's largest |exact|, and |lse - exact|."""
    if not live.any():
        return None, None
    n = int(live.sum())
    scale = np.abs(o_x[live]).reshape(n, -1).max(axis=1)
    rel = np.abs(o[live] - o_x[live]).reshape(n, -1).max(axis=1) / np.maximum(scale, 1e-30)
    return float(rel.max()), float(np.abs(l[live] - l_x[live]).max())


def f15_err_parts(o, o_x, live):
    """(out - exact) / the row's largest |exact| for every element of the live
    rows, [rows, elements], and the live row ids."""
    rows = np.flatnonzero(live)
    if not len(rows):
        return np.zeros((0, 1)), rows
    diff = (o[rows] - o_x[rows]).reshape(len(rows), -1)
    scale = np.abs(o_x[rows]).reshape(len(rows), -1).max(axis=1, keepdims=True)
    return diff / np.maximum(scale, 1e-30), rows


def f15_precision(results):
    """Gate 1's out tolerance against the kernels' own precision: the union
    kernel, the per-row kernel at its served 512-token tile and the same
    per-row kernel at 1024, each against a float64 reference over the same
    keys, and the pairwise differences."""
    cases = []
    for name, layer in (("s24", 0), ("s32", 0), ("s32", 5)):
        blocks, lengths, table, requests = f9u_case(name)
        starts = np.arange(0, 4 * requests + 1, 4, dtype=np.int32)
        cases.append((f"{name}/L{layer}", blocks[layer], lengths, table, starts, F15_SEQS))
    cases.append(("edge", *f15_edge_case(np.random.default_rng(15)), 16))
    outs = {}
    arms = (("rows512", "512"), ("rows1024", "1024"), ("union640", None))
    for arm, bkv in arms:
        _f9_set_env(0)
        if bkv:
            os.environ["VLLM_TPU_QSA_DECODE_RPA_BKV"] = bkv
        jax.clear_caches()
        fns = {
            off: jax.jit(
                functools.partial(_f9_attend if bkv else _f15_union_rpa,
                                  **({"offset": off} if bkv else {"slot": off})),
                donate_argnums=(5,),
            )
            for off in (0, 1)
        }
        for c_idx, (name, blocks, lengths, table, starts, seqs) in enumerate(cases):
            rows = len(lengths)
            q = jnp.asarray(
                np.random.default_rng(300 + c_idx).standard_normal((rows, F9_Q_HEADS, D)),
                jnp.bfloat16,
            )
            for rank in range(F9_DCP):
                try:
                    meta = f15_union_host(blocks, lengths, table, starts, rank, seqs=seqs)
                    cache = f15_nan_cache(meta, seed=500 + c_idx * 8 + rank)
                    r = jnp.asarray([rank], jnp.int32)
                    for off in (0, 1):
                        if bkv:
                            o, l, cache = fns[off](
                                q, jnp.zeros_like(q), jnp.asarray(blocks), jnp.asarray(lengths),
                                jnp.asarray(table), cache, rank=r,
                            )
                        else:
                            pages, sel, kv_lens, cu = meta[:4]
                            o, l, cache = fns[off](
                                q.astype(jnp.float32), jnp.asarray(pages), jnp.asarray(sel),
                                jnp.asarray(kv_lens), jnp.asarray(cu), cache,
                            )
                        outs[(arm, name, rank, off)] = (
                            np.asarray(o, np.float64), np.asarray(l, np.float64)
                        )
                        if arm == "union640":
                            outs[("exact", name, rank, off)] = f15_exact(
                                cache, meta, q.astype(jnp.float32), starts, off
                            )
                    del cache
                except Exception as e:  # noqa: BLE001
                    row = dict(mode="f15_precision", arm=arm, case=name, rank=rank,
                               error=repr(e)[:1500])
                    print("F15", json.dumps(row), flush=True)
                    results.append(row)
    jax.clear_caches()
    _f9_set_env(0)
    # Per arm over every live row of every case, rank and slot: largest
    # error, sum of squares, sum and count (for rms and the signed mean),
    # and the worst row.
    total = {arm: dict(max=0.0, sq=0.0, sum=0.0, n=0, worst=None) for arm, _ in arms}
    for c_idx, (name, blocks, lengths, table, starts, seqs) in enumerate(cases):
        for rank in range(F9_DCP):
            for off in (0, 1):
                key = (name, rank, off)
                if any((a, *key) not in outs for a in ("rows512", "rows1024", "union640", "exact")):
                    continue
                o_x, l_x = outs[("exact", *key)]
                live = np.isfinite(l_x).all(axis=1)
                row = dict(mode="f15_precision", case=name, rank=rank, slot=off,
                           live_rows=int(live.sum()))
                for arm, _ in arms:
                    o_a, l_a = outs[(arm, *key)]
                    e_o, e_l = f15_err(o_a, l_a, o_x, l_x, live)
                    err, ids = f15_err_parts(o_a, o_x, live)
                    row[f"{arm}_vs_exact_out"], row[f"{arm}_vs_exact_lse"] = e_o, e_l
                    if err.size:
                        row[f"{arm}_vs_exact_rms"] = float(np.sqrt(np.mean(err ** 2)))
                        row[f"{arm}_vs_exact_mean"] = float(np.mean(err))
                        t = total[arm]
                        per_row = np.abs(err).max(axis=1)
                        k = int(per_row.argmax())
                        if per_row[k] > t["max"]:
                            t["max"] = float(per_row[k])
                            t["worst"] = dict(case=name, rank=rank, slot=off, row=int(ids[k]))
                        t["sq"] += float(np.sum(err ** 2))
                        t["sum"] += float(np.sum(err))
                        t["n"] += int(err.size)
                for a, b in (("union640", "rows512"), ("rows1024", "rows512")):
                    o_b, l_b = outs[(b, *key)]
                    row[f"{a}_vs_{b}_out"] = f15_err(*outs[(a, *key)], o_b, l_b, live)[0]
                print("F15", json.dumps(row), flush=True)
                results.append(row)
    summary = dict(mode="f15_precision_summary")
    for arm, t in total.items():
        if t["n"]:
            rms = (t["sq"] / t["n"]) ** 0.5
            summary[arm] = dict(max=t["max"], rms=rms, mean=t["sum"] / t["n"],
                                elements=t["n"], worst=t["worst"])
    if "union640" in summary and "rows512" in summary:
        u, r = summary["union640"], summary["rows512"]
        summary["union_over_rows512_max"] = u["max"] / r["max"]
        summary["union_over_rows512_rms"] = u["rms"] / r["rms"]
        summary["union_mean_over_rms"] = abs(u["mean"]) / u["rms"]
        summary["bar_pass"] = bool(
            summary["union_over_rows512_max"] <= 1.25
            and summary["union_over_rows512_rms"] <= 1.25
            and u["max"] <= 5e-3
            and summary["union_mean_over_rms"] <= 0.1
        )
    print("F15_PRECISION_SUMMARY", json.dumps(summary), flush=True)
    results.append(summary)


def f15_arm(cache, results, *, case, label, use_sel, no_sel_dma=False):
    """12-layer step of the union kernel at the 640 tile with P = 2080 page
    slots a sequence, with (use_sel) or without the row mask, ranks 0-3.
    ``no_sel_dma`` (diagnostic, wrong masks): the selection words are never
    fetched, so the difference to the mask arm is the words' DMA."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import bref_override

    blocks, lengths, table, requests = f9u_case(case)
    rows = len(lengths)
    starts = np.arange(0, 4 * requests + 1, 4, dtype=np.int32)
    rng = np.random.default_rng(5)
    qs = jnp.asarray(rng.standard_normal((12, rows, F9_Q_HEADS, D)), jnp.bfloat16)
    qs = qs.astype(jnp.float32)

    @functools.partial(jax.jit, donate_argnums=(0,))
    def run(cache, q, pages, sel, kv_lens, cu):
        def body(i, carry):
            cache, acc = carry
            for layer in range(12):
                out, lse, cache = _f15_union_rpa(
                    q[layer], pages[layer], sel[layer] if use_sel else None,
                    kv_lens[layer], cu[layer], cache, layer % 2,
                )
                acc = acc + out.astype(jnp.float32).sum()
                acc = acc + jnp.where(jnp.isfinite(lse), lse, 0).sum()
            return cache, acc

        return jax.lax.fori_loop(0, ITERS, body, (cache, jnp.float32(0)))

    original = (bref_override.SelectionFetch.copy_in, bref_override.SelectionFetch.wait_in)
    if no_sel_dma:
        bref_override.SelectionFetch.copy_in = lambda self, *a: None
        bref_override.SelectionFetch.wait_in = lambda self, *a: None
    jax.clear_caches()
    try:
        cache = _f15_arm_ranks(cache, results, run, qs, blocks, lengths, table, starts,
                               requests, case, label, use_sel, no_sel_dma)
    finally:
        bref_override.SelectionFetch.copy_in, bref_override.SelectionFetch.wait_in = original
        jax.clear_caches()
    return cache


def _f15_arm_ranks(cache, results, run, qs, blocks, lengths, table, starts, requests,
                   case, label, use_sel, no_sel_dma):
    for rank in range(F9_DCP):
        metas = [
            f15_union_host(blocks[i], lengths, table, starts, rank, seqs=F15_SEQS)
            for i in range(12)
        ]
        stack = [jnp.asarray(np.stack([m[j] for m in metas])) for j in range(4)]
        slots = F15_TILE // RATIO + 2
        tiles = sum(int(np.sum(-(-m[2] // F15_TILE))) for m in metas)
        real = sum(int(np.sum(-(-m[2] // RATIO))) for m in metas)
        try:
            cache, med, lo, hi = f9u_time(cache, run, (qs, *stack))
            err = None
        except Exception as e:  # noqa: BLE001
            med = lo = hi = None
            err = repr(e)[:1500]
        row = dict(mode="f15_time", case=case, requests=requests, arm=label,
                   use_sel=use_sel, no_sel_dma=no_sel_dma, rank=rank, us_per_step=med,
                   us_min=lo, us_max=hi, step_descriptors=tiles * slots, step_real=real,
                   error=err)
        if med is not None:
            row["us_per_request_layer"] = med / 12 / requests
            row["descriptors_per_request_layer"] = tiles * slots / 12 / requests
        print("F15", json.dumps(row), flush=True)
        results.append(row)
    return cache


def f15_layout_case(rng, counts, *, slots, rows, pages=4096, ctx=(30000, 200000)):
    """Requests with the given row counts (rows of a request at consecutive
    positions, own top-512 selections), padded to ``slots`` request slots and
    ``rows`` rows."""
    cols = -(-ctx[1] // (F9_DCP * PAGE)) + 2
    perm = rng.permutation(pages)
    blocks = np.full((rows, TOPK_GROUPS), -1, np.int32)
    lengths = np.zeros((rows,), np.int32)
    table = np.zeros((rows, cols), np.int32)
    starts = np.zeros((slots + 1,), np.int32)
    i = 0
    for r, n in enumerate(counts):
        base = int(rng.integers(*ctx))
        own = np.resize(perm[(r * cols) % pages :], cols)
        for j in range(n):
            lengths[i] = base + 1 + j
            blocks[i] = rng.choice(lengths[i] // RATIO, TOPK_GROUPS, replace=False)
            table[i] = own
            i += 1
        starts[r + 1] = i
    starts[len(counts) + 1 :] = i
    assert i <= rows <= slots * 4
    return blocks, lengths, table, starts


def f15_long(results):
    """Steps with a request of more than 4 rows take the per-row path inside
    the union program: outputs equal the flag-off program's. A step of 4-row
    requests in the same harness takes the union path (its outputs differ
    from the per-row path's by the kernels' rounding)."""
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa as rpa

    rng = np.random.default_rng(57)
    cases = [
        ("5 alone", [5], 8, 8),
        ("8 alone", [8], 8, 8),
        ("257 alone", [257], 128, 320),
        ("5 + 7x4", [5] + [4] * 7, 16, 64),
        ("4x7 + 8", [4] * 7 + [8], 16, 64),
        ("257 + 31x4", [4] * 16 + [257] + [4] * 15, 128, 384),
        ("dummy 127x1 + 257", [1] * 127 + [257], 128, 384),
        ("40x2 past the covered slots", [2] * 40, 128, 96),
        ("33x3 past the covered slots", [3] * 33, 128, 128),
        ("control 16x4", [4] * 16, 16, 64),
        ("control 24x4 in 128 slots", [4] * 24, 128, 96),
        ("control 48x4 in 128 slots", [4] * 48, 128, 192),
    ]

    def attend(q, gate, blocks, lengths, table, cache, starts, rank, *, offset, union):
        return rpa.qsa_attention(
            q, gate, blocks, lengths, table, cache, starts, kv_heads=1, ratio=RATIO,
            k_scale=0.375, v_scale=2.5, max_query_len=4, dcp=(rank, F9_DCP),
            **({"kv_head_offset": offset} if offset else {}),
            **({"union": True} if union else {}),
        )

    ok_all = True
    cache = make_cache()
    for name, counts, slots, rows in cases:
        blocks, lengths, table, starts = f15_layout_case(rng, counts, slots=slots, rows=rows)
        q = jnp.asarray(rng.standard_normal((rows, F9_Q_HEADS, D)), jnp.bfloat16)
        gate = jnp.zeros_like(q)
        args = tuple(map(jnp.asarray, (blocks, lengths, table)))
        for rank in (0, 3):
            try:
                r = jnp.asarray([rank], jnp.int32)
                outs = {}
                for flag in ("0", "640"):
                    _f9_set_env(0)
                    os.environ["VLLM_TPU_QSA_DECODE_RPA_UNION"] = flag
                    jax.clear_caches()
                    for off in (0, 1):
                        fn = jax.jit(
                            functools.partial(attend, offset=off, union=flag != "0"),
                            donate_argnums=(5,),
                        )
                        o, l, cache = fn(q, gate, *args, cache, jnp.asarray(starts), r)
                        outs[(flag, off)] = (np.asarray(o, np.float32), np.asarray(l, np.float32))
                for off in (0, 1):
                    (o0, l0), (o1, l1) = outs[("0", off)], outs[("640", off)]
                    live = np.isfinite(l0).all(axis=1)
                    row = dict(mode="f15_long", case=name, rank=rank, slot=off, rows=rows,
                               slots=slots, live_rows=int(live.sum()),
                               bit_equal=bool(o0.tobytes() == o1.tobytes() and l0.tobytes() == l1.tobytes()),
                               out_max_abs=float(np.max(np.abs(o0 - o1))),
                               finite=bool(np.isfinite(o1).all()))
                    if name.startswith("control"):
                        # The union path ran: same live rows, a rounding-sized gap.
                        row["ok"] = bool(row["finite"] and not row["bit_equal"]
                                         and np.array_equal(np.isfinite(l1).all(axis=1), live))
                    else:
                        row["ok"] = row["bit_equal"]
                    ok_all &= row["ok"]
                    print("F15", json.dumps(row), flush=True)
                    results.append(row)
            except Exception as e:  # noqa: BLE001
                ok_all = False
                row = dict(mode="f15_long", case=name, rank=rank, error=repr(e)[:1500])
                print("F15", json.dumps(row), flush=True)
                results.append(row)
    os.environ["VLLM_TPU_QSA_DECODE_RPA_UNION"] = "0"
    jax.clear_caches()
    print("F15_LONG_ALL", ok_all, flush=True)
    return ok_all


def f15c_main():
    """Long and short-request routing, then the decode branch timed end to end
    in the served request bucket (128 slots) at 24, 48 and 64 requests."""
    results = []
    out = Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "f15c_bench.json")
    try:
        if os.environ.get("F15C_LONG", "1") == "1":
            f15_long(results)
            out.write_text(json.dumps(results, indent=1))
        cache = make_cache()
        arms = (
            ("rows", False, False, True),
            ("union", True, False, True),
            ("union_meta_first", True, True, True),
        )
        for case in ("s24", "s48", "s64"):
            for label, union, meta_first, route in arms:
                print("F15_ARM", case, label, "bucket128", flush=True)
                cache = f15_e2e_arm(cache, results, case=case, label=label + "_b128",
                                    union=union, bucket=128, meta_first=meta_first,
                                    route=route)
                out.write_text(json.dumps(results, indent=1))
    finally:
        out.write_text(json.dumps(results, indent=1))
        print("F15_DONE", flush=True)


def f15_bounds_main():
    """Gate 1, the wiring checks and the long-request routing with the main
    RPA kernel compiled with bounds checks (a DMA past a buffer halts)."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import kernel as rpa_kernel

    rpa_kernel.BOUNDS_CHECKS = True
    results = []
    out = Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "f15_bounds.json")
    try:
        ok = {}
        ok["long"] = f15_long(results)
        out.write_text(json.dumps(results, indent=1))
        ok["gate1"] = f15_gate1(results, layers=(0, 5, 11))
        out.write_text(json.dumps(results, indent=1))
        ok["wire"] = f15_wire(results)
        out.write_text(json.dumps(results, indent=1))
        print("F15_BOUNDS_SUMMARY", json.dumps(ok), flush=True)
    finally:
        out.write_text(json.dumps(results, indent=1))
        print("F15_DONE", flush=True)


def f15_own_lists(blocks, lengths, rank, rng):
    """The rank-only lists the compressed-DCP local merge attends
    (pooled_qsa.compressed_dcp_own_blocks) from global ones: a row's selected
    groups this rank owns, -1 in every other slot, columns shuffled."""
    blocks, lengths = np.asarray(blocks), np.asarray(lengths)
    out = np.full_like(blocks, -1)
    for i, length in enumerate(lengths):
        complete = int(length) // RATIO
        row = blocks[i, : min(complete, blocks.shape[1])]
        row = row[(row >= 0) & (row < complete) & ((row // F9U_UNIT) % F9_DCP == rank)]
        out[i, : len(row)] = row
        out[i] = out[i, rng.permutation(blocks.shape[1])]
    return out


def f15_wire_own(results):
    """f15_wire on rank-only lists (VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE,
    dcp=(rank, 4, 0, True)): qsa_attention with the union path against the
    per-row path, row by row, on NaN-planted caches. Cases: s24/L0, s32/L7,
    s24+s32/L3 (56 requests, two calls), the edge batch."""
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa as rpa

    cases = []
    for name, layer in (("s24", 0), ("s32", 7)):
        blocks, lengths, table, requests = f9u_case(name)
        starts = np.arange(0, 4 * requests + 1, 4, dtype=np.int32)
        starts = np.concatenate([starts, np.full(F15_SEQS - requests, starts[-1])])
        cases.append((f"{name}/L{layer}", blocks[layer], lengths, table, starts))
    blocks, lengths, table, starts, _ = f15_compose("s56")
    cases.append(("s24+s32/L3", blocks[3], lengths, table, starts))
    blocks, lengths, table, starts = f15_edge_case(np.random.default_rng(15))
    cases.append(("edge", blocks, lengths, table,
                  np.concatenate([starts, np.full(16 - len(starts) + 1, starts[-1])])))

    def attend(q, gate, blocks, lengths, table, cache, starts, rank, *, offset, union):
        return rpa.qsa_attention(
            q, gate, blocks, lengths, table, cache, starts, kv_heads=1, ratio=RATIO,
            k_scale=0.375, v_scale=2.5, max_query_len=4, dcp=(rank, F9_DCP, 0, True),
            **({"kv_head_offset": offset} if offset else {}),
            **({"union": True} if union else {}),
        )

    ok_all = True
    rng = np.random.default_rng(77)
    for c_idx, (name, glob, lengths, table, starts) in enumerate(cases):
        rows = len(lengths)
        live_reqs = int(np.searchsorted(starts, starts[-1]))
        seqs = len(starts) - 1
        q = jnp.asarray(
            np.random.default_rng(400 + c_idx).standard_normal((rows, F9_Q_HEADS, D)),
            jnp.bfloat16,
        )
        gate = jnp.zeros_like(q)
        for rank in range(F9_DCP):
            try:
                own = f15_own_lists(glob, lengths, rank, rng)
                meta = f15_union_host(own, lengths, table, starts[: live_reqs + 1], rank,
                                      seqs=seqs, own_list=True)
                host_bad = f15_rows_check(own, lengths, table, starts[: live_reqs + 1], rank,
                                          meta, own_list=True)
                ref_meta = f15_union_host(glob, lengths, table, starts[: live_reqs + 1], rank,
                                          seqs=seqs)
                same_as_global = bool(
                    np.array_equal(meta[2], ref_meta[2]) and np.array_equal(meta[4], ref_meta[4])
                )
                cache = f15_nan_cache(meta, seed=2000 + c_idx * 8 + rank)
                r = jnp.asarray([rank], jnp.int32)
                outs = {}
                for union in (False, True):
                    _f9_set_env(0)
                    os.environ["VLLM_TPU_QSA_DECODE_RPA_UNION"] = "640" if union else "0"
                    jax.clear_caches()
                    for off in (0, 1):
                        fn = jax.jit(functools.partial(attend, offset=off, union=union),
                                     donate_argnums=(5,))
                        o, l, cache = fn(q, gate, jnp.asarray(own), jnp.asarray(lengths),
                                         jnp.asarray(table), cache, jnp.asarray(starts), r)
                        outs[(union, off)] = (np.asarray(o, np.float32), np.asarray(l, np.float32))
                for off in (0, 1):
                    (o_r, l_r), (o_u, l_u) = outs[(False, off)], outs[(True, off)]
                    live = np.isfinite(l_r).all(axis=1)
                    row = dict(mode="f15_wire_own", case=name, rank=rank, slot=off,
                               requests=live_reqs, live_rows=int(live.sum()), host_bad=host_bad,
                               same_union_as_global=same_as_global,
                               same_dead=bool(
                                   np.array_equal(np.isfinite(l_u).all(axis=1), live)
                                   and np.all(o_u[~live] == 0) and np.all(np.isneginf(l_u[~live]))
                               ))
                    if live.any():
                        scale = np.abs(o_r[live]).reshape(live.sum(), -1).max(axis=1)
                        rel = np.abs(o_u[live] - o_r[live]).reshape(live.sum(), -1).max(axis=1)
                        rel = rel / np.maximum(scale, 1e-30)
                        d_lse = np.abs(l_u[live] - l_r[live]).max(axis=1)
                        row.update(out_rel_max=float(rel.max()), lse_abs_max=float(d_lse.max()),
                                   finite=bool(np.isfinite(o_u[live]).all()
                                               and np.isfinite(o_r[live]).all()))
                        # The kernel gate's bound (lead 10.x): no row above 5e-3.
                        row["ok"] = bool(row["same_dead"] and row["finite"] and host_bad == 0
                                         and same_as_global and rel.max() <= 5e-3
                                         and d_lse.max() <= 1e-3)
                    else:
                        row["ok"] = bool(row["same_dead"] and host_bad == 0)
                    ok_all &= row["ok"]
                    print("F15", json.dumps(row), flush=True)
                    results.append(row)
                del cache
            except Exception as e:  # noqa: BLE001
                ok_all = False
                row = dict(mode="f15_wire_own", case=name, rank=rank, error=repr(e)[:1500])
                print("F15", json.dumps(row), flush=True)
                results.append(row)
    os.environ["VLLM_TPU_QSA_DECODE_RPA_UNION"] = "0"
    jax.clear_caches()
    print("F15_WIRE_OWN_ALL", ok_all, flush=True)
    return ok_all


def f15_wire(results):
    """qsa_attention(union=True) under VLLM_TPU_QSA_DECODE_RPA_UNION=640 (the
    SparseCore metadata, the glue and the 32-request calls) against the per-row
    decode branch, row by row with gate 1's tolerance, on NaN-planted caches:
    s24, s32, both together (56 requests, two calls) and the edge batch."""
    import dataclasses  # noqa: F401

    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa as rpa

    cases = []
    parts = {}
    for name in ("s24", "s32"):
        blocks, lengths, table, requests = f9u_case(name)
        rows = 4 * requests
        parts[name] = (blocks[:, :rows], lengths[:rows], table[:rows], requests)
        starts = np.arange(0, 4 * requests + 1, 4, dtype=np.int32)
        starts = np.concatenate([starts, np.full(F15_SEQS - requests, starts[-1])])
        for layer in (0, 7):
            cases.append((f"{name}/L{layer}", blocks[layer], lengths, table, starts))
    b24, l24, t24, r24 = parts["s24"]
    b32, l32, t32, r32 = parts["s32"]
    cols = max(t24.shape[1], t32.shape[1])
    pad = lambda t: np.pad(t, ((0, 0), (0, cols - t.shape[1])))  # noqa: E731
    both_rows = 4 * (r24 + r32)
    both = (
        np.concatenate([b24[3], b32[3]]),
        np.concatenate([l24, l32]),
        np.concatenate([pad(t24), pad(t32)]),
    )
    starts = np.arange(0, both_rows + 1, 4, dtype=np.int32)
    starts = np.concatenate([starts, np.full(64 - r24 - r32, starts[-1])])
    extra = 256 - both_rows
    cases.append((
        "s24+s32/L3",
        np.pad(both[0], ((0, extra), (0, 0)), constant_values=-1),
        np.pad(both[1], (0, extra)),
        np.pad(both[2], ((0, extra), (0, 0))),
        starts,
    ))
    rng = np.random.default_rng(15)
    blocks, lengths, table, starts = f15_edge_case(rng)
    cases.append(("edge", blocks, lengths, table, np.concatenate([starts, np.full(16 - len(starts) + 1, starts[-1])])))

    def attend(q, gate, blocks, lengths, table, cache, starts, rank, *, offset, union):
        return rpa.qsa_attention(
            q, gate, blocks, lengths, table, cache, starts, kv_heads=1, ratio=RATIO,
            k_scale=0.375, v_scale=2.5, max_query_len=4, dcp=(rank, F9_DCP),
            **({"kv_head_offset": offset} if offset else {}),
            **({"union": True} if union else {}),
        )

    ok_all = True
    for c_idx, (name, blocks, lengths, table, starts) in enumerate(cases):
        rows = len(lengths)
        live_reqs = int(np.searchsorted(starts, starts[-1]))
        seqs = len(starts) - 1
        q = jnp.asarray(
            np.random.default_rng(200 + c_idx).standard_normal((rows, F9_Q_HEADS, D)),
            jnp.bfloat16,
        )
        gate = jnp.zeros_like(q)
        for rank in range(F9_DCP):
            try:
                meta = f15_union_host(blocks, lengths, table, starts[: live_reqs + 1], rank, seqs=seqs)
                cache = f15_nan_cache(meta, seed=1000 + c_idx * 8 + rank)
                r = jnp.asarray([rank], jnp.int32)
                outs = {}
                for union in (False, True):
                    _f9_set_env(0)
                    os.environ["VLLM_TPU_QSA_DECODE_RPA_UNION"] = "640" if union else "0"
                    jax.clear_caches()
                    for off in (0, 1):
                        fn = jax.jit(
                            functools.partial(attend, offset=off, union=union),
                            donate_argnums=(5,),
                        )
                        o, l, cache = fn(
                            q, gate, jnp.asarray(blocks), jnp.asarray(lengths),
                            jnp.asarray(table), cache, jnp.asarray(starts), r,
                        )
                        outs[(union, off)] = (np.asarray(o, np.float32), np.asarray(l, np.float32))
                for off in (0, 1):
                    (o_r, l_r), (o_u, l_u) = outs[(False, off)], outs[(True, off)]
                    live = np.isfinite(l_r).all(axis=1)
                    row = dict(mode="f15_wire", case=name, rank=rank, slot=off,
                               requests=live_reqs, live_rows=int(live.sum()),
                               same_dead=bool(
                                   np.array_equal(np.isfinite(l_u).all(axis=1), live)
                                   and np.all(o_u[~live] == 0) and np.all(np.isneginf(l_u[~live]))
                               ))
                    if live.any():
                        scale = np.abs(o_r[live]).reshape(live.sum(), -1).max(axis=1)
                        rel = np.abs(o_u[live] - o_r[live]).reshape(live.sum(), -1).max(axis=1)
                        rel = rel / np.maximum(scale, 1e-30)
                        d_lse = np.abs(l_u[live] - l_r[live]).max(axis=1)
                        row.update(out_rel_max=float(rel.max()), lse_abs_max=float(d_lse.max()),
                                   finite=bool(np.isfinite(o_u[live]).all()))
                        row["ok"] = bool(row["same_dead"] and row["finite"]
                                         and rel.max() <= 1e-3 and d_lse.max() <= 1e-3)
                    else:
                        row["ok"] = row["same_dead"]
                    ok_all &= row["ok"]
                    print("F15", json.dumps(row), flush=True)
                    results.append(row)
                del cache
            except Exception as e:  # noqa: BLE001
                ok_all = False
                row = dict(mode="f15_wire", case=name, rank=rank, error=repr(e)[:1500])
                print("F15", json.dumps(row), flush=True)
                results.append(row)
    os.environ["VLLM_TPU_QSA_DECODE_RPA_UNION"] = "0"
    jax.clear_caches()
    print("F15_WIRE_ALL", ok_all, flush=True)
    return ok_all


def f15_compose(name, bucket=None):
    """Dumped AgentX requests at the request counts the union path serves in
    one program: s24 and s32 as dumped, s48 (s24 + 24 of s32), s56 (both) and
    s64 (s32 twice, the second copy with its layers rolled by 6). Returns
    blocks [12, T, 512], lengths [T], table [T, cols], starts [R + 1] padded
    to the request bucket R, and the live request count. Rows are 4 a
    request, the served token bucket."""
    def part(case, reqs=None, roll=0):
        blocks, lengths, table, requests = f9u_case(case)
        reqs = requests if reqs is None else reqs
        rows = 4 * reqs
        return np.roll(blocks[:, :rows], roll, axis=0), lengths[:rows], table[:rows], reqs

    plan = {
        "s24": [("s24", None, 0)],
        "s32": [("s32", None, 0)],
        "s48": [("s24", None, 0), ("s32", 24, 0)],
        "s56": [("s24", None, 0), ("s32", None, 0)],
        "s64": [("s32", None, 0), ("s32", None, 6)],
    }[name]
    parts = [part(*p) for p in plan]
    cols = max(p[2].shape[1] for p in parts)
    blocks = np.concatenate([p[0] for p in parts], axis=1)
    lengths = np.concatenate([p[1] for p in parts])
    table = np.concatenate(
        [np.pad(p[2], ((0, 0), (0, cols - p[2].shape[1]))) for p in parts]
    )
    requests = sum(p[3] for p in parts)
    bucket = bucket or max(32, 1 << (requests - 1).bit_length())
    # The served token bucket of a verify step: 4 rows a request (96 to 256
    # are all buckets).
    rows = 4 * requests
    extra = rows - len(lengths)
    blocks = np.pad(blocks, ((0, 0), (0, extra), (0, 0)), constant_values=-1)
    lengths = np.pad(lengths, (0, extra))
    table = np.pad(table, ((0, extra), (0, 0)))
    starts = np.arange(0, 4 * requests + 1, 4, dtype=np.int32)
    starts = np.concatenate([starts, np.full(bucket - requests, starts[-1])]).astype(np.int32)
    return blocks, lengths, table, starts, requests


def f15_e2e_arm(cache, results, *, case, label, union, bucket=None, meta_first=False,
                route=True):
    """12-layer step of qsa_attention's decode branch on the served DCP4 route:
    the per-row path (union=False) or the union path (SparseCore metadata,
    glue and 32-request RPA calls), ranks 0-3."""
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa as rpa

    blocks, lengths, table, starts, requests = f15_compose(case, bucket)
    rows = len(lengths)
    rng = np.random.default_rng(7)
    qs = jnp.asarray(rng.standard_normal((12, rows, F9_Q_HEADS, D)), jnp.bfloat16)
    gs = jnp.zeros_like(qs)
    _f9_set_env(0)
    os.environ["VLLM_TPU_QSA_DECODE_RPA_UNION"] = "640" if union else "0"
    rpa._UNION_META_FIRST, rpa._UNION_ROUTE = meta_first, route
    jax.clear_caches()

    @functools.partial(jax.jit, donate_argnums=(0,))
    def run(cache, qs, gs, blocks, lengths, table, starts, rank):
        def body(i, carry):
            cache, acc = carry
            for layer in range(12):
                # A read-only cache leaves every layer loop-invariant; the
                # carried term (XLA keeps a float times zero) stops XLA from
                # hoisting the attention out of the loop.
                q = qs[layer] + (acc * 0).astype(qs.dtype)
                out, lse, cache = rpa.qsa_attention(
                    q, gs[layer], blocks[layer], lengths, table, cache, starts,
                    kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5,
                    max_query_len=4, dcp=(rank, F9_DCP),
                    **({"kv_head_offset": 1} if layer % 2 else {}),
                    **({"union": True} if union else {}),
                )
                acc = acc + out.astype(jnp.float32).sum()
                acc = acc + jnp.where(jnp.isfinite(lse), lse, 0).sum()
            return cache, acc

        return jax.lax.fori_loop(0, ITERS, body, (cache, jnp.float32(0)))

    try:
        for rank in range(F9_DCP):
            r = jnp.asarray([rank], jnp.int32)
            args = tuple(map(jnp.asarray, (blocks, lengths, table, starts)))
            try:
                cache, med, lo, hi = f9u_time(cache, run, (qs, gs, *args, r))
                err = None
            except Exception as e:  # noqa: BLE001
                med = lo = hi = None
                err = repr(e)[:1500]
            row = dict(mode="f15_e2e", case=case, requests=requests, rows=rows,
                       request_bucket=len(starts) - 1, arm=label, union=union, rank=rank,
                       meta_first=meta_first, route=route,
                       us_per_step=med, us_min=lo, us_max=hi, error=err)
            if med is not None:
                row["us_per_request_layer"] = med / 12 / requests
            print("F15", json.dumps(row), flush=True)
            results.append(row)
    finally:
        os.environ["VLLM_TPU_QSA_DECODE_RPA_UNION"] = "0"
        rpa._UNION_META_FIRST, rpa._UNION_ROUTE = False, True
        jax.clear_caches()
    return cache


def f15b_main():
    """The wiring check, then the per-row and union decode branches timed end
    to end (metadata included) at 24 to 64 requests."""
    results = []
    out = Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "f15b_bench.json")
    try:
        f15_wire(results)
        out.write_text(json.dumps(results, indent=1))
        cache = make_cache()
        for case in ("s48", "s64", "s56", "s24", "s32"):
            for label, union in (("rows", False), ("union", True)):
                print("F15_ARM", case, label, flush=True)
                cache = f15_e2e_arm(cache, results, case=case, label=label, union=union)
                out.write_text(json.dumps(results, indent=1))
    finally:
        out.write_text(json.dumps(results, indent=1))
        print("F15_DONE", out, flush=True)


def f15_main():
    results = []
    out = Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "f15_bench.json")
    try:
        f15_gate1(results)
        out.write_text(json.dumps(results, indent=1))
        cache = make_cache()
        for case in ("s24", "s32"):
            for label, kind in (("served_rows", "served"), ("union640", "union")):
                print("F15_ARM", case, label, flush=True)
                cache = f9u_arm(cache, results, case=case, label=label, kind=kind,
                                bkv=640 if kind == "union" else 512)
                out.write_text(json.dumps(results, indent=1))
            for label, use_sel in (("union640_p2080", False), ("union640_sel", True)):
                print("F15_ARM", case, label, flush=True)
                cache = f15_arm(cache, results, case=case, label=label, use_sel=use_sel)
                out.write_text(json.dumps(results, indent=1))
    finally:
        out.write_text(json.dumps(results, indent=1))
        print("F15_DONE", out, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["f9"]:
        f9_main()
    elif sys.argv[1:] == ["f9u"]:
        f9u_main()
    elif sys.argv[1:] == ["f15"]:
        f15_main()
    elif sys.argv[1:] == ["f15_gate1"]:
        f15_gate1([])
    elif sys.argv[1:] == ["f15_wire"]:
        f15_wire([])
    elif sys.argv[1:] == ["f15_long"]:
        f15_long([])
    elif sys.argv[1:] == ["f15_wire_own"]:
        _rows = []
        try:
            f15_wire_own(_rows)
        finally:
            Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "f15_wire_own.json").write_text(
                json.dumps(_rows, indent=1)
            )
            print("F15_DONE", flush=True)
    elif sys.argv[1:] == ["f15_bounds"]:
        f15_bounds_main()
    elif sys.argv[1:] == ["f15c"]:
        f15c_main()
    elif sys.argv[1:] == ["f15b"]:
        f15b_main()
    elif sys.argv[1:] == ["f15_mask"]:
        _rows = []
        _out = Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "f15_mask.json")
        try:
            _cache = make_cache()
            for _case in ("s24", "s32"):
                for _label, _sel, _nodma in (
                    ("union640_p2080", False, False),
                    ("union640_sel", True, False),
                    ("union640_sel_nodma", True, True),
                    ("union640_p2080_again", False, False),
                ):
                    print("F15_ARM", _case, _label, flush=True)
                    _cache = f15_arm(_cache, _rows, case=_case, label=_label,
                                     use_sel=_sel, no_sel_dma=_nodma)
                    _out.write_text(json.dumps(_rows, indent=1))
        finally:
            _out.write_text(json.dumps(_rows, indent=1))
            print("F15_DONE", flush=True)
    elif sys.argv[1:] == ["f15_precision"]:
        _rows = []
        try:
            f15_precision(_rows)
        finally:
            Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "f15_precision.json").write_text(
                json.dumps(_rows, indent=1)
            )
            print("F15_DONE", flush=True)
    else:
        main()
