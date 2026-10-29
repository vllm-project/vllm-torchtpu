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
"""Single-device production PLE lookup correctness, timing and optional profile."""

import argparse
import functools
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.ple import decode_embeddings, resident_lookup_raw


def _slab_bits(start, segments, rank_tag, fp8, *, width, segment_width):
    segment = jnp.asarray(start, jnp.uint32) + jnp.arange(segments, dtype=jnp.uint32)
    alignment = math.gcd(width, segment_width)
    segments_per_group = width // alignment
    rows_per_group = segment_width // alignment
    within = (segment[:, None] % segments_per_group) * segment_width + jnp.arange(
        segment_width, dtype=jnp.uint32
    )
    row = (segment[:, None] // segments_per_group) * rows_per_group + within // width
    column = within % width
    h = row * jnp.uint32(0x9E3779B1) ^ column * jnp.uint32(0x85EBCA6B)
    h = h ^ jnp.uint32(rank_tag) * jnp.uint32(0xC2B2AE35)
    h = h ^ (h >> 16)
    h = h * jnp.uint32(0x7FEB352D)
    h = h ^ (h >> 15)
    if fp8:
        bits = ((h >> 31) << 7) | (((h >> 3) % 13 + 1) << 3) | (h & 7)
        row_value = jnp.where((row >> (column % 32)) & 1, 0x40, 0x38)
        rank_value = jnp.where((rank_tag >> ((column - 32) % 8)) & 1, 0x48, 0x30)
        bits = jnp.where(
            column < 32, row_value, jnp.where(column < 40, rank_value, bits)
        )
        return bits.astype(jnp.uint8)
    bits = ((h >> 31) << 15) | (((h >> 7) % 24 + 112) << 7) | (h & 127)
    row_value = jnp.where((row >> (column % 32)) & 1, 0x4000, 0x3F80)
    rank_value = jnp.where((rank_tag >> ((column - 32) % 8)) & 1, 0x4080, 0x3F00)
    bits = jnp.where(column < 32, row_value, jnp.where(column < 40, rank_value, bits))
    return jax.lax.bitcast_convert_type(bits.astype(jnp.uint16), jnp.int16)


def _generate(rank_tag, *, fp8, rows, width, segment_width, slab_segments):
    segments = math.ceil(rows * width / segment_width)
    slab = min(slab_segments, segments)
    dtype = jnp.uint8 if fp8 else jnp.int16
    # A dynamic fill avoids materializing a multi-GB host constant. Every slab
    # is overwritten in the loop; the last overlap regenerates identical bytes.
    initial = jnp.broadcast_to(rank_tag.astype(dtype), (segments, segment_width))

    def write(i, table):
        start = jnp.minimum(i * slab, segments - slab)
        values = _slab_bits(
            start, slab, rank_tag, fp8, width=width, segment_width=segment_width
        )
        return jax.lax.dynamic_update_slice(table, values, (start, 0))

    return jax.lax.fori_loop(0, math.ceil(segments / slab), write, initial)


def _expected_bits(rows, columns, rank_tag, fp8):
    """Independent CPU row/column oracle; no production segment addressing."""
    row, column = np.broadcast_arrays(
        np.asarray(rows, np.uint32), np.asarray(columns, np.uint32)
    )
    h = row * np.uint32(0x9E3779B1) ^ column * np.uint32(0x85EBCA6B)
    h ^= np.uint32((rank_tag * 0xC2B2AE35) & 0xFFFFFFFF)
    h ^= h >> 16
    h *= np.uint32(0x7FEB352D)
    h ^= h >> 15
    if fp8:
        bits = ((h >> 31) << 7) | (((h >> 3) % 13 + 1) << 3) | (h & 7)
        row_value = np.where((row >> (column % 32)) & 1, 0x40, 0x38)
        rank_value = np.where((rank_tag >> ((column - 32) % 8)) & 1, 0x48, 0x30)
        return np.where(
            column < 32, row_value, np.where(column < 40, rank_value, bits)
        ).astype(np.uint8)
    bits = ((h >> 31) << 15) | (((h >> 7) % 24 + 112) << 7) | (h & 127)
    row_value = np.where((row >> (column % 32)) & 1, 0x4000, 0x3F80)
    rank_value = np.where((rank_tag >> ((column - 32) % 8)) & 1, 0x4080, 0x3F00)
    return np.where(
        column < 32, row_value, np.where(column < 40, rank_value, bits)
    ).astype(np.uint16)


def benchmark(args):
    device = jax.devices("tpu")[0]
    segment_width = args.segment_width or pltpu.get_tpu_info().num_lanes
    assert min(args.rows, args.width, args.heads, segment_width, args.slab_segments) > 0
    assert all(t > 0 for t in args.tokens)
    assert args.samples > 0
    assert args.reserve_bytes >= 0 and 0 <= args.reserve_fraction < 1
    fp8 = args.encoding == "fp8"
    segments = math.ceil(args.rows * args.width / segment_width)
    table_bytes = segments * segment_width * (1 if fp8 else 2)
    stats = device.memory_stats()
    if stats is None:
        raise RuntimeError("Device memory statistics are required before allocation")
    reserve = max(
        args.reserve_bytes, math.ceil(stats["bytes_limit"] * args.reserve_fraction)
    )
    free = stats["bytes_limit"] - stats["bytes_in_use"]
    if table_bytes + reserve > free:
        raise RuntimeError("Table exceeds available memory headroom")
    args.output.mkdir(parents=True, exist_ok=True)
    rank = device.id + 1
    with jax.default_device(device):
        generator = jax.jit(
            functools.partial(
                _generate,
                fp8=fp8,
                rows=args.rows,
                width=args.width,
                segment_width=segment_width,
                slab_segments=args.slab_segments,
            )
        )
        generate = generator.lower(jnp.uint32(rank)).compile()
        memory = generate.memory_analysis()
        assert memory is not None
        assert memory.output_size_in_bytes + memory.temp_size_in_bytes + reserve < free
        table = generate(jnp.uint32(rank))
        table.block_until_ready()
        lookup = jax.jit(
            functools.partial(
                resident_lookup_raw,
                vocab_start=args.vocab_start,
                local_rows=args.rows,
                head_dim=args.width,
            )
        )
        decoder = jax.jit(decode_embeddings)
        results = []
        rng = np.random.default_rng(1966)
        for tokens in args.tokens:
            ids = rng.integers(
                args.vocab_start,
                args.vocab_start + args.rows,
                (tokens, args.heads),
                dtype=np.int32,
            )
            ids.reshape(-1)[: min(ids.size, 4)] = np.asarray(
                [
                    args.vocab_start - 1,
                    args.vocab_start,
                    args.vocab_start + args.rows - 1,
                    args.vocab_start + args.rows,
                ],
                np.int32,
            )[: min(ids.size, 4)]
            owned = (ids >= args.vocab_start) & (ids < args.vocab_start + args.rows)
            local = np.where(owned, ids - args.vocab_start, 0)
            bits = _expected_bits(local[..., None], np.arange(args.width), rank, fp8)
            bits = np.where(owned[..., None], bits, 0).astype(bits.dtype)
            expected = bits.view(np.int8 if fp8 else np.int16).reshape(tokens, -1)
            query = jnp.asarray(ids)
            started = time.perf_counter()
            run = lookup.lower(table, query).compile()
            compile_seconds = time.perf_counter() - started
            memory = run.memory_analysis()
            assert memory is not None
            planned = (
                memory.argument_size_in_bytes
                + memory.output_size_in_bytes
                - memory.alias_size_in_bytes
                + memory.temp_size_in_bytes
            )
            assert planned + reserve < free
            raw = run(table, query)
            np.testing.assert_array_equal(raw, expected)
            values = bits.view(ml_dtypes.float8_e4m3fn if fp8 else ml_dtypes.bfloat16)
            numeric = values.astype(ml_dtypes.bfloat16)
            if fp8:
                numeric = (numeric.astype(np.float32) * np.float32(0.375)).astype(
                    ml_dtypes.bfloat16
                )
            np.testing.assert_array_equal(
                decoder(raw, jnp.float32(0.375)), numeric.reshape(tokens, -1)
            )
            for _ in range(10):
                run(table, query).block_until_ready()
            samples = []
            for _ in range(args.samples):
                begin = time.perf_counter_ns()
                run(table, query).block_until_ready()
                samples.append((time.perf_counter_ns() - begin) / 1000)
            if args.save_hlo:
                (args.output / f"{tokens}.hlo").write_text(run.as_text())
            if args.profile:
                with jax.profiler.trace(str(args.output / f"trace-{tokens}")):
                    run(table, query).block_until_ready()
            np.savez_compressed(
                args.output / f"{tokens}.npz",
                ids=ids,
                actual=np.asarray(raw),
                expected=expected,
            )
            results.append(
                dict(
                    tokens=tokens,
                    compile_seconds=compile_seconds,
                    temp_bytes=memory.temp_size_in_bytes,
                    median_us=float(np.median(samples)),
                    samples_us=samples,
                )
            )
            (args.output / "results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", choices=("fp8", "bf16"), required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--segment-width", type=int)
    parser.add_argument("--vocab-start", type=int, default=0)
    parser.add_argument("--slab-segments", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--reserve-bytes",
        type=int,
        default=0,
        help="Additional memory headroom in bytes (default:0)",
    )
    parser.add_argument(
        "--reserve-fraction",
        type=float,
        default=0.2,
        help="Memory headroom as a device-capacity fraction (default:0.2)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-hlo", action="store_true")
    benchmark(parser.parse_args())
