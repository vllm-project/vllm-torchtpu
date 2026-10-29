# SPDX-License-Identifier: Apache-2.0
"""Measure SWA publication plus CSA2 attention with native or baseline word pages."""

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import (
    encode_csa2_kv,
    prepare_swa_metadata,
)
from vllm_torchtpu.kernels.deepseek_v41.csa2.csa2 import csa2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=("native", "word"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 128, 1024])
    parser.add_argument("--mode", choices=("decode", "prefill"), default="prefill")
    parser.add_argument("--pages", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument(
        "--page-order", choices=("sequential", "shuffled"), default="sequential"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    page_size, heads = args.page_size, 64
    if args.layout == "native":
        from vllm_torchtpu.kernels.mla.paged_swa import write_swa_pages

        publish = write_swa_pages

        def view(x):
            return x
    else:
        # Run this mode against the baseline checkout, before the migration.
        from vllm_torchtpu.kernels.mla.paged_swa import (
            swa_word_pages,
            write_swa_word_pool,
        )

        publish = functools.partial(write_swa_word_pool, page_size=page_size)
        view = functools.partial(swa_word_pages, page_size=page_size)
    results = []
    for tokens in args.tokens:
        requests = tokens if args.mode == "decode" else 1
        starts = jnp.arange(requests + 1, dtype=jnp.int32) * (tokens // requests)
        positions = (
            jnp.full((tokens,), 127, jnp.int32)
            if args.mode == "decode"
            else jnp.arange(127, 127 + tokens, dtype=jnp.int32)
        )
        columns = (127 + tokens // requests + page_size - 1) // page_size
        assert requests * columns < args.pages
        physical_pages = np.arange(1, requests * columns + 1, dtype=np.int32)
        if args.page_order == "shuffled":
            np.random.default_rng(71).shuffle(physical_pages)
        table = jnp.asarray(physical_pages.reshape(requests, columns))
        metadata, _, records = prepare_swa_metadata(
            table,
            starts,
            positions,
            jnp.zeros(requests, jnp.int32),
            page_size=page_size,
            page_stride=page_size,
            row_offset=0,
        )
        rng = np.random.default_rng(47)

        def random(shape, rng=rng):
            return jnp.asarray(rng.standard_normal(shape) * 0.1, jnp.bfloat16)

        q, values = random((tokens, heads, 512)), random((tokens, 512))
        pages = jnp.zeros((args.pages, page_size, 512), jnp.bfloat16)
        history = random((requests * columns, page_size, 512))
        pages = pages.at[1 : requests * columns + 1].set(history)
        if args.layout == "word":
            bits = jax.lax.bitcast_convert_type(pages, jnp.uint16).astype(jnp.uint32)
            bits = bits.reshape(args.pages, page_size, 2, 2, 128)
            packed = bits[:, :, :, 0] | (bits[:, :, :, 1] << 16)
            pages = jax.lax.bitcast_convert_type(packed, jnp.int32).transpose(
                0, 2, 1, 3
            )
            pages = pages.reshape(-1, 128)
        words, scales = encode_csa2_kv(random((512, 512)))
        words = jnp.broadcast_to(words, (tokens, *words.shape))
        scales = jnp.broadcast_to(scales, (tokens, *scales.shape))
        counts = jnp.full((tokens,), 128, jnp.int32)
        sink = jnp.zeros(heads, jnp.float32)
        rope = jnp.stack((jnp.ones((tokens, 128)), jnp.zeros((tokens, 128))), axis=1)

        def step(
            owner,
            values,
            q,
            words,
            scales,
            starts,
            positions,
            sink,
            rope,
            metadata,
            counts,
            records,
            bq=6 if tokens <= 384 else 16,
        ):
            owner = publish(owner, records, values)
            output = csa2(
                q,
                words,
                scales,
                view(owner),
                starts,
                positions,
                sink,
                rope,
                swa_metadata=metadata,
                main_lengths=counts,
                compress_ratio=2,
                scale=512**-0.5,
                bq=bq,
            )
            return owner, output

        arrays = (
            values,
            q,
            words,
            scales,
            starts,
            positions,
            sink,
            rope,
            metadata,
            counts,
            records,
        )
        compiled = jax.jit(step, donate_argnums=(0,)).lower(pages, *arrays).compile()
        label = f"{args.mode}-{tokens}-{args.layout}"
        (args.out / f"{label}.hlo").write_text(compiled.as_text())
        for _ in range(5):
            pages, output = compiled(pages, *arrays)
            jax.block_until_ready((pages, output))
        samples = []
        for _ in range(args.steps):
            before = time.perf_counter_ns()
            pages, output = compiled(pages, *arrays)
            jax.block_until_ready((pages, output))
            samples.append((time.perf_counter_ns() - before) / 1000)
        with jax.profiler.trace(str(args.out / label / "profile")):
            for _ in range(5):
                pages, output = compiled(pages, *arrays)
                jax.block_until_ready((pages, output))
        np.save(args.out / f"{label}.npy", np.asarray(output, np.float32))
        result = dict(
            layout=args.layout,
            mode=args.mode,
            tokens=tokens,
            page_size=page_size,
            page_order=args.page_order,
            cache_bytes=pages.nbytes,
            median_us=float(np.median(samples)),
            samples_us=samples,
            memory=str(compiled.memory_analysis()),
        )
        print(json.dumps(result), flush=True)
        results.append(result)
        (args.out / "timings.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
