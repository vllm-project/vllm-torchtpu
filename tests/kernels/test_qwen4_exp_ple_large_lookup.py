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
"""Opt-in, single-device real-size PLE gather comparison; never normal CI load."""

import functools
import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import SingleDeviceSharding

from vllm_torchtpu.kernels.qwen4_exp.ple import (
    compact_rows,
    decode_embeddings,
    resident_lookup_raw,
)

_ROWS = 40_000_192
_WIDTH = 160
_SEGMENTS = _ROWS * _WIDTH // 128
_START = 4096
_HEADS = 16
_ARMS = ("unchunked", "chunk8", "chunk16", "chunk32")


def _lookup(table, ids, *, chunk, local_rows=_ROWS):
    if chunk is None:
        # This arm is literally the production entry, including its masks/padding.
        return resident_lookup_raw(
            table, ids, vocab_start=_START, local_rows=local_rows, head_dim=_WIDTH
        )
    carrier = jnp.int16 if table.dtype == jnp.int16 else jnp.int8

    def gather(query):
        owned = (query >= _START) & (query < _START + local_rows)
        local = jnp.where(owned, query - _START, 0)
        bits = compact_rows(table, local, head_dim=_WIDTH)
        bits = jnp.where(owned[..., None], bits, jnp.zeros((), table.dtype))
        return jax.lax.bitcast_convert_type(bits, carrier)

    padded = jnp.pad(
        ids.astype(jnp.int32), ((0, (-len(ids)) % chunk), (0, 0)), constant_values=-1
    )
    result = jax.lax.map(gather, padded.reshape(-1, chunk, ids.shape[1]))
    return result.reshape(-1, ids.shape[1] * _WIDTH)[: len(ids)]


def _slab_bits(start, segments, rank_tag, fp8):
    segment = jnp.asarray(start, jnp.uint32) + jnp.arange(segments, dtype=jnp.uint32)
    within = (segment[:, None] % 5) * 128 + jnp.arange(128, dtype=jnp.uint32)
    row = (segment[:, None] // 5) * 4 + within // 160
    column = within % 160
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


def _generate(rank_tag, *, fp8, segments=_SEGMENTS):
    slab = min(8192, segments)
    dtype = jnp.uint8 if fp8 else jnp.int16
    # A dynamic fill avoids materializing a multi-GB host constant. Every slab
    # is overwritten in the loop; the last overlap regenerates identical bytes.
    initial = jnp.broadcast_to(rank_tag.astype(dtype), (segments, 128))

    def write(i, table):
        start = jnp.minimum(i * slab, segments - slab)
        values = _slab_bits(start, slab, rank_tag, fp8)
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


def _queries(tokens):
    rng = np.random.default_rng(1966 + tokens)
    ids = rng.integers(_START, _START + _ROWS, (tokens, _HEADS), dtype=np.int32)
    boundaries = [0, 1, 2, 3, 4, 7, 8, 127, 128, 159, 160, _ROWS - 2, _ROWS - 1]
    for power in range(1, 26):
        boundaries.extend([(1 << power) - 1, 1 << power, (1 << power) + 1])
    special = [-1, _START - 1, _START + _ROWS, _START + _ROWS + 1]
    special += [_START + x for x in boundaries]
    ids.reshape(-1)[: len(special)] = special
    return ids


def _memory(compiled):
    stats = compiled.memory_analysis()
    if stats is None:
        return {"available": False}
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {name: int(getattr(stats, name)) for name in names if hasattr(stats, name)}


def _buffer(array):
    result = dict(
        shape=list(array.shape),
        dtype=str(array.dtype),
        logical_bytes=array.nbytes,
        device_bytes=array.on_device_size_in_bytes(),
        format=str(array.format),
        devices=[str(x) for x in array.devices()],
        addressable_shards=len(array.addressable_shards),
    )
    try:
        result["opaque_pointer"] = array.unsafe_buffer_pointer()
    except Exception as error:
        result["opaque_pointer_unavailable"] = str(error)
    return result


def _write(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2) + "\n")


@pytest.mark.tpu_test
@pytest.mark.skipif(
    os.environ.get("TPU_PLE_LARGE_TABLE_TEST") != "1",
    reason="Opt in explicitly: allocates6.4/12.8GB on one TPU device",
)
@pytest.mark.parametrize("encoding", ["fp8", "bf16"])
def test_real_size_resident_lookup_variants(encoding, tmp_path):
    fp8 = encoding == "fp8"
    production_only = os.environ.get("PLE_LARGE_TABLE_PRODUCTION_ONLY") == "1"
    out = Path(os.environ.get("PLE_LARGE_TABLE_OUTPUT", tmp_path)) / encoding
    out.mkdir(parents=True, exist_ok=True)
    records = []

    def progress(phase, **extra):
        _write(
            out.parent,
            "progress.json",
            dict(dtype=encoding, phase=phase, completed=records, **extra),
        )
        print("LARGE_TABLE", encoding, phase, extra, flush=True)

    device = jax.devices("tpu")[0]
    sharding = SingleDeviceSharding(device)
    table_bytes = _ROWS * _WIDTH * (1 if fp8 else 2)
    before = device.memory_stats()
    _write(
        out,
        "device.json",
        dict(
            device=str(device),
            kind=device.device_kind,
            stats=before,
            required_table_bytes=table_bytes,
        ),
    )
    if not before or "bytes_limit" not in before:
        progress("BLOCKED_MEMORY_LIMIT_UNAVAILABLE")
        pytest.skip("No per-device memory limit available for safe large allocation")
    reserve = max(8 << 30, before["bytes_limit"] // 5)
    free = before["bytes_limit"] - before.get("bytes_in_use", 0)
    if 2 * table_bytes + (1 << 30) + reserve > free:
        progress("BLOCKED_MEMORY_HEADROOM", free=free, reserve=reserve)
        pytest.skip("Full-size table cannot retain the declared memory headroom")

    with jax.default_device(device):
        rank_tag = jax.device_put(np.uint32(device.id + 17), sharding)
        generator = jax.jit(
            functools.partial(_generate, fp8=fp8), out_shardings=sharding
        )
        start = time.perf_counter()
        gen = generator.lower(rank_tag).compile()
        generation_memory = _memory(gen)
        _write(
            out,
            "generation.json",
            dict(memory=generation_memory, compile_seconds=time.perf_counter() - start),
        )
        estimate = generation_memory.get("output_size_in_bytes", table_bytes)
        estimate += generation_memory.get("temp_size_in_bytes", table_bytes)
        assert estimate <= 2 * table_bytes + (128 << 20), generation_memory
        assert estimate + reserve < free
        (out / "generator.optimized.hlo").write_text(gen.as_text())
        progress("GENERATING_FULL_TABLE", expected_bytes=table_bytes)
        table = gen(rank_tag)
        table.block_until_ready()
        allocation = _buffer(table)
        assert table.shape == (_SEGMENTS, 128) and table.devices() == {device}
        assert allocation["addressable_shards"] == 1
        _write(
            out,
            "allocation.json",
            dict(
                before=before,
                after=device.memory_stats(),
                buffer=allocation,
                generation_memory=generation_memory,
            ),
        )

        # This is not the gather under test: read two contiguous physical
        # segments, computing the byte offset with host Python unbounded ints.
        read = jax.jit(
            lambda owner, segment: jax.lax.dynamic_slice(owner, (segment, 0), (2, 128)),
            in_shardings=(table.format, sharding),
        )
        sanity = []
        for row in [
            0,
            1,
            3,
            4,
            (1 << 16) - 1,
            1 << 16,
            (1 << 23) + 1,
            (1 << 25) - 1,
            1 << 25,
            _ROWS - 1,
        ]:
            offset = row * _WIDTH
            parts = np.asarray(read(table, jnp.asarray(offset // 128, jnp.int32)))
            got = parts.reshape(-1)[offset % 128 : offset % 128 + _WIDTH]
            expected = _expected_bits(row, np.arange(_WIDTH), device.id + 17, fp8)
            unsigned = np.uint8 if fp8 else np.uint16
            np.testing.assert_array_equal(got.view(unsigned), expected)
            sanity.append(row)
        _write(
            out,
            "generator-sanity.json",
            dict(
                rows=sanity,
                exact=True,
                method="independent contiguous dynamic_slice; CPU row/column formula",
            ),
        )
        progress("GENERATOR_VERIFIED", allocation=allocation)
        scale = jax.device_put(np.float32(0.375), sharding)
        decoder = jax.jit(decode_embeddings, out_shardings=sharding)
        failures = []
        for tokens in [9, 256, 2048]:
            ids = _queries(tokens)
            query = jax.device_put(ids, sharding)
            owned = (ids >= _START) & (ids < _START + _ROWS)
            local = np.where(owned, ids - _START, 0)
            unsigned = _expected_bits(
                local[..., None], np.arange(_WIDTH), device.id + 17, fp8
            )
            unsigned = np.where(owned[..., None], unsigned, 0).astype(unsigned.dtype)
            raw_expected = unsigned.view(np.int8 if fp8 else np.int16).reshape(
                tokens, -1
            )
            float_dtype = ml_dtypes.float8_e4m3fn if fp8 else ml_dtypes.bfloat16
            numbers = unsigned.view(float_dtype).astype(ml_dtypes.bfloat16)
            if fp8:
                numbers = (numbers.astype(np.float32) * np.float32(0.375)).astype(
                    ml_dtypes.bfloat16
                )
            decoded_expected = numbers.reshape(tokens, -1).view(np.uint16)
            compiled = {}
            variants = (
                [("unchunked", None)]
                if production_only
                else zip(_ARMS, [None, 8, 16, 32])
            )
            for arm, chunk in variants:
                item = dict(
                    dtype=encoding,
                    tokens=tokens,
                    arm=arm,
                    raw_mismatches=None,
                    decoded_mismatches=None,
                )
                begin = time.perf_counter()
                try:
                    fn = jax.jit(
                        functools.partial(_lookup, chunk=chunk),
                        in_shardings=(table.format, query.format),
                        out_shardings=sharding,
                    )
                    lowered = fn.lower(table, query)
                    run = lowered.compile()
                    item.update(
                        compile_seconds=time.perf_counter() - begin,
                        memory=_memory(run),
                        input_formats=str(run.input_formats),
                    )
                    assert run.input_formats[0][0] == table.format
                    memory = item["memory"]
                    assert (
                        table_bytes
                        + memory.get("temp_size_in_bytes", table_bytes)
                        + reserve
                        < free
                    )
                    (out / f"{tokens}-{arm}.optimized.hlo").write_text(run.as_text())
                    actual = run(table, query)
                    actual.block_until_ready()
                    raw = np.asarray(actual)
                    mismatch = raw != raw_expected
                    locations = np.argwhere(mismatch)
                    decoded = np.asarray(decoder(actual, scale)).view(np.uint16)
                    item.update(
                        raw_mismatches=int(np.count_nonzero(mismatch)),
                        decoded_mismatches=int(
                            np.count_nonzero(decoded != decoded_expected)
                        ),
                        first_examples=[
                            dict(
                                query=int(i),
                                head=int(j // _WIDTH),
                                column=int(j % _WIDTH),
                                global_id=int(ids[i, j // _WIDTH]),
                                expected=int(raw_expected[i, j]),
                                actual=int(raw[i, j]),
                            )
                            for i, j in locations[:16]
                        ],
                    )
                    item["correct"] = (
                        item["raw_mismatches"] == item["decoded_mismatches"] == 0
                    )
                    np.savez_compressed(
                        out / f"{tokens}-{arm}.npz",
                        ids=ids,
                        actual_raw=raw,
                        expected_raw=raw_expected,
                        actual_decoded=decoded,
                        expected_decoded=decoded_expected,
                    )
                    compiled[arm] = (run, item["correct"])
                    item["status"] = (
                        "CORRECT" if item["correct"] else "NUMERICAL_FAILURE"
                    )
                    if not item["correct"]:
                        failures.append((tokens, arm, item["raw_mismatches"]))
                except Exception as error:
                    item.update(
                        status="COMPILE_OR_EXECUTION_ERROR",
                        error=repr(error),
                        correct=False,
                    )
                    failures.append((tokens, arm, repr(error)))
                records.append(item)
                _write(out, "correctness.json", records)
                progress("ARM_CORRECTNESS", last=item)

            if production_only:
                continue
            good = [arm for arm in _ARMS if arm in compiled and compiled[arm][1]]
            for arm in good:
                for _ in range(10):
                    compiled[arm][0](table, query).block_until_ready()
            rng = np.random.default_rng(1966 + tokens)
            order = list(rng.permutation(good))
            samples = []
            for round_index in range(30):
                offset = round_index % len(order) if order else 0
                arms = order[offset:] + order[:offset]
                for position, arm in enumerate(arms):
                    begin = time.perf_counter_ns()
                    output = compiled[arm][0](table, query)
                    output.block_until_ready()
                    end = time.perf_counter_ns()
                    sample = dict(
                        dtype=encoding,
                        tokens=tokens,
                        arm=arm,
                        round=round_index,
                        order=arms,
                        position=position,
                        start_ns=begin,
                        end_ns=end,
                        us=(end - begin) / 1000,
                    )
                    samples.append(sample)
                    with (out / "timings.jsonl").open("a") as stream:
                        stream.write(json.dumps(sample) + "\n")
            summaries = []
            for arm in good:
                times = [s["us"] for s in samples if s["arm"] == arm]
                summaries.append(
                    dict(
                        dtype=encoding,
                        tokens=tokens,
                        arm=arm,
                        median_us=float(np.median(times)),
                        p10_us=float(np.percentile(times, 10)),
                        p90_us=float(np.percentile(times, 90)),
                        samples=len(times),
                    )
                )
            _write(out, f"{tokens}-latency.json", summaries)
            progress("TIMING_COMPLETE", tokens=tokens, summaries=summaries)
            if tokens == 2048:
                for arm, (run, _) in compiled.items():
                    try:
                        with (
                            jax.profiler.trace(str(out / f"trace-{arm}")),
                            jax.profiler.TraceAnnotation(
                                f"large-lookup/{encoding}/{arm}"
                            ),
                        ):
                            run(table, query).block_until_ready()
                    except Exception as error:
                        (out / f"profile-error-{arm}.txt").write_text(repr(error))
                    progress("PROFILE_COMPLETE", arm=arm)
        after = _buffer(table)
        assert after == allocation
        _write(
            out,
            "final-allocation.json",
            dict(buffer=after, stats=device.memory_stats()),
        )
        table.delete()
        progress("COMPLETE", failures=failures)
        assert not failures, failures
