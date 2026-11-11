# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the per-shape tuning sweeps (VLLM_TPU_SHAPE_TUNED tables).

Each ``bench_<kernel>.py`` module exposes ``main(emit)``. For every shape
bucket it sweeps the kernel's tile knobs (env vars read at trace time, so each
config gets a fresh ``jax.jit``) and calls ``emit`` once per (bucket, config).

Timing: ``bench(body, carry)`` jits a ``fori_loop`` of ``ITERS`` calls of
``body(carry) -> carry`` with the carry donated, so a cache that the kernel
aliases in place is never copied (an undonated multi-GiB cache argument makes
XLA copy the pool every call and swamps the kernel). Bodies must make their
per-call work depend on the carry so XLA cannot hoist it out of the loop.

Bitwise: ``digest(*arrays)`` hashes host bytes of one call's outputs; the
summarizer compares each config's digest with the bucket's default config.
"""

import contextlib
import functools
import hashlib
import json
import os
import sys
import time
import traceback

import jax
import numpy as np

ITERS = int(os.environ.get("ST_ITERS", "20"))
REPS = int(os.environ.get("ST_REPS", "5"))
OUT_DIR = os.environ.get("CDK_OUTPUT_DIR", "/tmp")

# Served AgentX env (ixza6 recipe) for the kernels swept here. Applied with
# setdefault before any vllm_torchtpu import, so a sweep's own overrides win.
SERVED_ENV = {
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
    "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_RPA_BKV": "512",
    "VLLM_TPU_QSA_DECODE_RPA_LANES": "4",
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": "128",
    "VLLM_TPU_QSA_DECODE_KV_BLOCK": "3328",
    "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK": "1024",
    "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": "1",
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": "1",
    "VLLM_TPU_QSA_DECODE_TOKEN_SHARD": "0",
    "VLLM_TPU_QSA_INDEXER_HOIST_META": "1",
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": "1",
    "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": "1",
    "VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES": "1",
    "VLLM_TPU_QSA_INDEXER_TOKEN_SHARD": "1",
    "VLLM_TPU_QSA_MAIN_KV_DCP": "2",
    "VLLM_TPU_QSA_PREFILL_INDEXER_BKV": "1024",
    "VLLM_TPU_QSA_PREFILL_LEAN": "1",
    "VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY": "1",
    "VLLM_TPU_QSA_PREFILL_UNION_CAP": "12288",
    "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE": "1024",
    "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": "32768",
    "VLLM_TPU_QSA_PREFILL_UNION_WALK": "128",
    "VLLM_TPU_QSA_PREFILL_BITSET_OWNED": "1",
    "VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK": "0",
    "VLLM_TPU_QSA_DCP_SPLIT_LSE": "1",
    "VLLM_TPU_QSA_PREFIX_SLICE": "1",
    "TPU_RPA_DECODE_BATCHED": "1",
    "TPU_GDN_CHUNK_PAIR": "1",
    "TPU_GDN_KERNEL_ZERO_OUT": "1",
    "TPU_GDN_RAW_BA": "1",
    "TPU_GDN_RAW_QKV": "1",
    "TPU_ENABLE_GDN_DYNAMIC_TILING": "1",
    "VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR": "1",
    "VLLM_TPU_GDN_HOIST_METADATA": "1",
    "VLLM_TPU_MAMBA_TAIL_FOLD": "1",
    "VLLM_TPU_MAMBA_TAIL_FOLD_UNIT": "256",
    "VLLM_TPU_HC_PREFILL_WIDE_TILE": "256",
    "VLLM_TPU_HC_PREFILL_UP_T": "1",
    "VLLM_TPU_HC_PREFILL_COMBINE_MIX": "1",
    "VLLM_TPU_HC_PREFILL_INJECT": "1",
    "TPU_QWEN4_HC_PREFILL_NORM": "bf16ep64",
    "TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER": "1",
    "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS": "64",
    "TPU_MOE_ROUTER_TOPK": "rowmax",
    "TPU_MOE_SINGLE_SORT": "1",
    "TPU_MOE_SKIP_PADDED_TOKENS": "1",
}
for _k, _v in SERVED_ENV.items():
    os.environ.setdefault(_k, _v)


@contextlib.contextmanager
def env(**overrides):
    """Temporarily set env knobs (values str()-ed; None unsets)."""
    old = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def bench(body, carry, iters=None, reps=None):
    """Median us per ``body`` call; returns (us, carry). ``carry`` is donated."""
    iters = iters or ITERS
    reps = reps or REPS

    @functools.partial(jax.jit, donate_argnums=(0,))
    def run(carry):
        return jax.lax.fori_loop(0, iters, lambda i, c: body(c), carry)

    carry = run(carry)
    jax.block_until_ready(carry)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        carry = run(carry)
        jax.block_until_ready(carry)
        samples.append((time.perf_counter() - t0) / iters * 1e6)
    return float(np.median(samples)), carry


def digest(*arrays):
    h = hashlib.sha1()
    for a in jax.tree_util.tree_leaves(arrays):
        a = np.asarray(jax.device_get(a))
        h.update(str(a.dtype).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


class Emitter:
    def __init__(self, kernel):
        self.kernel = kernel
        self.path = os.path.join(OUT_DIR, f"shapetune_{kernel}.jsonl")
        self.f = open(self.path, "a")

    def __call__(self, bucket, config, us, digest=None, default=False, note=None):
        row = dict(kernel=self.kernel, bucket=bucket, config=config, us=us,
                   digest=digest, default=default)
        if note:
            row["note"] = note
        line = json.dumps(row)
        print("ST " + line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def error(self, bucket, config, exc):
        msg = f"{type(exc).__name__}: {str(exc)[:300]}"
        self(bucket, config, None, note="ERR " + msg)
        traceback.print_exc(limit=3, file=sys.stderr)


def try_config(emit, bucket, config, fn, default=False):
    """Run ``fn() -> (us, digest)`` and emit; errors are emitted, not raised."""
    try:
        us, dg = fn()
        emit(bucket, config, round(us, 2), dg, default=default)
        return us, dg
    except Exception as e:  # noqa: BLE001
        emit.error(bucket, config, e)
        return None, None
