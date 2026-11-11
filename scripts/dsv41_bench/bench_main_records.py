"""TPU micro-benchmark: CSA2 main-record preparation, whole-page vs block-stride.

Run on a TPU host from the repo root:
    python scripts/dsv41_bench/bench_main_records.py
Compares `prepare_main_records` with the old whole-page formula (page_stride=None)
against the new block-stride formula at the measured decode buckets, checks
both match `PagedCacheRows.row_indices`, and reports per-call time from a
1000-iteration on-device loop (min and max of 3 repeats).
"""

import pathlib
import sys
import time
import types

root = pathlib.Path.cwd()
for name in (
    "vllm_torchtpu",
    "vllm_torchtpu.kernels",
    "vllm_torchtpu.kernels.deepseek_v41",
    "vllm_torchtpu.kernels.experimental",
):
    module = types.ModuleType(name)
    module.__path__ = [str(root / "src" / name.replace(".", "/"))]
    sys.modules[name] = module

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v41.csa2 import cache as c2  # noqa: E402

PAGE = 128  # ratio-2 main page at --block-size 256
# Block stride 512 KiB (old, whole pages) vs 448 KiB (new), in 512-byte records.
OLD_STRIDE, NEW_STRIDE, OFFSET = 1024, 896, 256
LOOP = 1000
REPEATS = 3


def inputs(tokens, requests, pps, seed=0):
    rng = np.random.default_rng(seed)
    bounds = np.sort(rng.integers(0, tokens + 1, requests - 1))
    starts = np.concatenate([[0], bounds, [tokens]]).astype(np.int32)
    table = rng.integers(0, 4096, (requests, pps)).astype(np.int32)
    selected = rng.integers(-1, pps * PAGE, (tokens, 512)).astype(np.int32)
    return selected, table, starts


def timed(fn, *args):
    """Per-call device time: ``LOOP`` calls in one jit amortize host dispatch.

    Each iteration's inputs pass through a barrier with the previous output,
    so XLA can neither hoist the loop-invariant call nor drop it.
    """

    def body(_, carry):
        *xs, acc = jax.lax.optimization_barrier(carry)
        records, counts = fn(*xs)
        return (*xs, acc + records[0, 0] + counts[0])

    loop = jax.jit(
        lambda *xs: jax.lax.fori_loop(0, LOOP, body, (*xs, jnp.int32(0)))[-1]
    )
    out = jax.jit(fn)(*args)
    jax.block_until_ready(loop(*args))
    samples = []
    for _ in range(REPEATS):
        begin = time.perf_counter()
        jax.block_until_ready(loop(*args))
        samples.append((time.perf_counter() - begin) / LOOP)
    return out, min(samples) * 1e6, max(samples) * 1e6


def main():
    print(jax.devices()[0].device_kind)
    for tokens, requests in sorted(
        __import__(
            "vllm_torchtpu.kernels.experimental.decode_preparation",
            fromlist=["MAIN_RECORD_BATCHES"],
        ).MAIN_RECORD_BATCHES
    ):
        for pps in (256, 768):  # 64K and 192K compressed records per request
            selected, table, starts = inputs(tokens, requests, pps)
            args = tuple(jnp.asarray(x) for x in (selected, table, starts))
            # Old: pages() pre-scaled the table to whole pages.
            old_table = jnp.asarray(table * (OLD_STRIDE // PAGE) + OFFSET // PAGE)

            def old(s, t, b):
                return c2.prepare_main_records(s, t, b, page_size=PAGE)

            def new(s, t, b):
                return c2.prepare_main_records(
                    s, t, b, page_size=PAGE, page_stride=NEW_STRIDE, row_offset=OFFSET
                )

            (old_rec, _), old_us, old_max = timed(old, args[0], old_table, args[2])
            (new_rec, _), new_us, new_max = timed(new, *args)
            req = np.searchsorted(starts[1:], np.arange(tokens), side="right")
            logical = np.maximum(selected, 0)
            block = table[req[:, None], logical // PAGE]
            valid = selected >= 0
            for stride, got in ((OLD_STRIDE, old_rec), (NEW_STRIDE, new_rec)):
                expect = np.where(valid, OFFSET + block * stride + logical % PAGE, -1)
                np.testing.assert_array_equal(np.asarray(got), expect)
            print(
                f"T={tokens:4d} R={requests:3d} pps={pps:4d}  "
                f"old {old_us:7.2f} us (max {old_max:7.2f})  "
                f"new {new_us:7.2f} us (max {new_max:7.2f})  "
                f"ratio {new_us / old_us:5.3f}"
            )


if __name__ == "__main__":
    main()
