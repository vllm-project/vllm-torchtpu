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
"""Native prefill group pooling, ordered pair writes and continued index state."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config
from vllm_torchtpu.kernels.qwen4_exp import (
    index_pool_prepare,
    pooled_qsa,
    qsa,
    qsa_state,
)
from vllm_torchtpu.kernels.qwen4_exp.index_pool_prepare_sc import (
    pool_index_groups_sc_async,
)
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import (
    use_prefill_index_preparation,
)

pytestmark = pytest.mark.tpu_test


def _bits(value):
    return np.ascontiguousarray(value).view(np.uint8)


def _equal(actual, expected):
    jax.block_until_ready((actual, expected))
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def _metadata(lengths, context, count=8192):
    starts = jnp.asarray(np.r_[0, np.cumsum(lengths)], jnp.int32)
    context = jnp.asarray(context, jnp.int32)
    slots = jnp.where(jnp.diff(starts) > 0, jnp.arange(8, dtype=jnp.int32), -1)
    req, logical, valid = qsa._token_metadata(count, starts, context, slots)
    return starts, context, slots, req, logical, valid


def _expanded_groups(raw, ring, starts, context, slots, req, ratio):
    """Original gather/select/FP32 mean, without compact-group preparation."""
    logical = context[req] + jnp.arange(len(raw)) - starts[req]
    members = logical[:, None] // ratio * ratio + jnp.arange(ratio)[None]
    local = starts[req, None] + members - context[req, None]
    current = raw[jnp.clip(local, 0, len(raw) - 1)]
    previous = ring[slots[req, None], members % ratio]
    previous = jnp.where((context[req] > 0)[:, None, None], previous, 0)
    joined = jnp.where((local >= starts[req, None])[..., None], current, previous)
    return bf16(jnp.mean(joined.astype(jnp.float32), axis=1))


def _assert_native(hlo, backend, *, writer=False):
    pool_name = (
        "qsa_index_window_mean_sc" if backend == "sc" else "qsa_index_group_mean_tc"
    )
    assert pool_name in hlo, f"{backend} pooling silently used fallback"
    if writer:
        assert "qsa_compressed_indexed_write_tc" in hlo, "writer used fallback"


@pytest.mark.parametrize("backend", ["tc", "sc"])
def test_prefill_group_pooling_preserves_dense_bf16_bits(backend, record_property):
    rng = np.random.default_rng(19103)
    raw = jnp.asarray(rng.normal(size=(8192, 128)), jnp.bfloat16)
    assert use_prefill_index_preparation(raw, 4, requests=8)
    ring = jnp.asarray(
        rng.normal(size=(8, 8, 128)) + np.arange(8)[:, None, None] * 100,
        jnp.bfloat16,
    )
    metadata = _metadata([1024] * 8, [0] * 8)
    functions = (
        _expanded_groups,
        index_pool_prepare.pool_index_groups
        if backend == "tc"
        else pool_index_groups_sc_async,
    )
    runs = []
    for fn in functions:
        lowered = jax.jit(functools.partial(fn, ratio=4)).lower(
            raw, ring, *metadata[:2], jnp.maximum(metadata[2], 0), metadata[3]
        )
        runs.append(lowered.compile())
    _assert_native(runs[1].as_text(), backend)
    record_property("native_pool", backend)
    special = np.array(
        [
            0,
            0x8000,
            1,
            0x8001,
            0x80,
            0x8080,
            0x7F7F,
            0xFF7F,
            0x7F80,
            0xFF80,
            0x7FC1,
            0xFFC1,
            0x3F80,
            0xBF80,
        ],
        np.uint16,
    )
    words = np.resize(special, raw.size).reshape(raw.shape)
    special_raw = jax.lax.bitcast_convert_type(jnp.asarray(words), jnp.bfloat16)
    # A normal BF16 quartet whose FP32 partial sum can be subnormal.
    quartets = np.array(
        [[0x00FF, 0x0180, 0x8080, 0x0180], [0x4E80, 0x3F80, 0xCE80, 0x3F80]],
        np.uint16,
    )
    cancellation = quartets[np.arange(128)[None] % 2, np.arange(8192)[:, None] % 4]
    cancellation = jax.lax.bitcast_convert_type(jnp.asarray(cancellation), jnp.bfloat16)
    cases = [([1024] * 8, [c] * 8) for c in (0, 8192, 8193, 8194, 8195)]
    cases += [
        ([127, 130, 0, 5, 29, 0, 7, 0], [0, 8193, 0, 3, 9, 0, 5, 0]),
        ([8188] + [0] * 7, [0] * 7 + [1]),
        ([0] * 8, [0] * 8),
    ]
    # One executable must handle both pooling branches and unrelated request slots.
    for values in (raw, special_raw, cancellation):
        for lengths, contexts in cases:
            starts, context, _, req, _, _ = _metadata(lengths, contexts)
            slots = jnp.asarray(rng.permutation(8), jnp.int32)
            history = jnp.full_like(ring, jnp.nan) if not any(contexts) else ring
            args = (values, history, starts, context, slots, req)
            _equal(runs[1](*args), runs[0](*args))
    negative_zero = jax.lax.bitcast_convert_type(
        jnp.full(raw.shape, 0x8000, jnp.uint16), jnp.bfloat16
    )
    for contexts in ([0] * 8, [8193] * 8):
        starts, context, slots, req, _, _ = _metadata([1024] * 8, contexts)
        args = (
            negative_zero,
            negative_zero[:64].reshape(ring.shape),
            starts,
            context,
            slots,
            req,
        )
        _equal(runs[1](*args), runs[0](*args))


def _fixture(fp8):
    rng = np.random.default_rng(443)
    table = 1 + np.arange(8 * 16, dtype=np.int32).reshape(8, 16)
    ids = np.arange(129, 137, dtype=np.int32)
    pool = rng.normal(0, 0.2, (137, 208, 1, 2, 128)).astype(jnp.bfloat16)
    if fp8:
        pool = pool.view(np.uint8).view(jnp.float8_e4m3fn).reshape(137, 208, 1, 4, 128)
    ring = rng.normal(size=(8, 8, 128)).astype(jnp.bfloat16)
    tags = np.zeros((8, 8, 2), np.int32)
    tags[:, :, 0] = 8191 - ((8191 - np.arange(8)) % 4)
    angles = np.arange(16384)[:, None] * 1e7 ** (-np.arange(32)[None] / 32)
    trig = np.pad(
        np.concatenate((np.cos(angles), np.sin(angles)), axis=1), ((0, 0), (0, 64))
    )
    norm = rng.normal(0, 0.1, 128).astype(jnp.bfloat16)
    return tuple(
        jax.device_put(x)
        for x in (pool, table, ids, ring, tags, norm, trig.astype(jnp.bfloat16))
    )


def _state_oracle(raw, positions, ring, tags, starts, context):
    raw, positions, ring, tags, starts, context = map(
        np.asarray, (raw, positions, ring, tags, starts, context)
    )
    rr, tt = ring.copy(), tags.copy()
    for request, (begin, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
        if begin == end:
            continue
        if context[request] == 0:
            rr[request], tt[request] = 0, -1
        for token in range(int(begin), int(end)):
            lane = (int(context[request]) + token - int(begin)) % 4
            rr[request, lane] = raw[token]
            tt[request, lane] = (positions[token], 0)
    return rr, tt


@pytest.mark.parametrize("backend", ["tc", "sc"])
@pytest.mark.parametrize("fp8", [False, True])
def test_prefill_cache_and_state_continue_into_decode(
    backend, fp8, monkeypatch, record_property
):
    pool, table, ids, seed_ring, seed_tags, norm, trig = _fixture(fp8)
    rng = np.random.default_rng(9304)

    def segment(owner, raw, positions, starts, context, *, reference=False):
        ring, tags = qsa_state.read_ring_and_tags(owner, ids, 1024, 16)
        ring = ring.reshape(8, 8, 128)
        slots = jnp.where(jnp.diff(starts) > 0, jnp.arange(8, dtype=jnp.int32), -1)
        pooled_values = None
        if (reference or backend == "sc") and use_prefill_index_preparation(
            raw, 4, requests=len(context)
        ):
            req, _, _ = qsa._token_metadata(len(raw), starts, context, slots)
            pool_groups = _expanded_groups if reference else pool_index_groups_sc_async
            pooled_values = pool_groups(
                raw, ring, starts, context, jnp.maximum(slots, 0), req, 4
            )
        result = qsa._indexer_updates(
            raw,
            positions,
            ring,
            tags.reshape(8, 8, 2),
            norm,
            trig,
            starts,
            context,
            slots,
            4,
            1e-6,
            64,
            pooled_values=pooled_values,
        )
        keys, complete, req, logical, ring, tags = result
        owner = pooled_qsa.write_compressed(
            owner,
            keys,
            complete,
            req,
            logical,
            starts,
            context,
            table,
            4,
            logical_block_size=1664,
        )
        owner = qsa_state.write_ring_and_tags(
            owner,
            ring.reshape(8, -1),
            tags.reshape(8, -1),
            jnp.where(jnp.diff(starts) > 0, ids, 0),
        )
        return owner, *result

    candidate = jax.jit(segment, donate_argnums=(0,))
    reference = jax.jit(functools.partial(segment, reference=True), donate_argnums=(0,))
    seed = jax.jit(qsa_state.write_ring_and_tags)(
        pool, seed_ring.reshape(8, -1), seed_tags.reshape(8, -1), ids
    ).block_until_ready()
    runs = {}
    for initial_context in (0, 8192):
        owners = [seed.copy().block_until_ready(), seed.copy().block_until_ready()]
        expected_ring, expected_tags = np.asarray(seed_ring), np.asarray(seed_tags)
        contexts = np.full(8, initial_context, np.int32)
        for step, lengths in enumerate(
            (
                [1024] * 8,
                [127, 130, 0, 5, 29, 0, 7, 0],
                [127, 130, 0, 5, 29, 0, 7, 0],
                [1] * 8,
            )
        ):
            count = 16 if step == 3 else 8192
            raw = jnp.asarray(rng.normal(size=(count, 128)), jnp.bfloat16)
            starts, context, _, _, logical, valid = _metadata(lengths, contexts, count)
            positions = jnp.where(valid, logical, 0).astype(jnp.int32)
            if count not in runs:
                args = (owners[0], raw, positions, starts, context)
                # Only the reference writer uses the original expanded pair path.
                with monkeypatch.context() as patch:
                    patch.setattr(
                        pooled_qsa,
                        "use_prefill_index_preparation",
                        lambda *a, **k: False,
                    )
                    ref_run = reference.lower(*args).compile()
                run = candidate.lower(*args).compile()
                hlo = run.as_text()
                if count == 8192:
                    assert use_prefill_index_preparation(raw, 4, requests=8)
                    _assert_native(hlo, backend, writer=True)
                    assert "qsa_compressed_indexed_write_tc" not in ref_run.as_text()
                else:
                    assert "qsa_index_window_mean_sc" not in hlo
                    assert "qsa_compressed_indexed_write_tc" not in hlo
                runs[count] = ref_run, run
            args = (raw, positions, starts, context)
            results = [
                fn(owner, *args) for fn, owner in zip(runs[count], owners, strict=True)
            ]
            jax.block_until_ready(results)
            for actual, expected in zip(results[1], results[0], strict=True):
                _equal(actual, expected)
            expected_ring, expected_tags = _state_oracle(
                raw, positions, expected_ring, expected_tags, starts, context
            )
            _equal(results[1][-2], expected_ring)
            _equal(results[1][-1], expected_tags)
            owners = [result[0] for result in results]
            contexts += np.asarray(lengths)
    record_property("native_pool", backend)
    record_property("full_owner_and_six_outputs_bitwise_equal", True)
    record_property("continued_chunks_per_initial_context", 4)


@pytest.mark.parametrize("fp8", [False, True])
def test_indexed_pair_writer_preserves_aliased_partial_pairs(fp8, record_property):
    pool, table, _, _, _, _, _ = _fixture(fp8)
    table = jnp.full_like(table, 2)
    words = np.random.default_rng(5317).integers(0, 65536, (8192, 128), np.uint16)
    words[:, ::3], words[:, 1::7] = 0x8000, 0x7FC1
    keys = jax.lax.bitcast_convert_type(jnp.asarray(words), jnp.bfloat16)
    assert use_prefill_index_preparation(keys, 4, requests=8)
    write = jax.jit(
        functools.partial(
            pooled_qsa.write_compressed, ratio=4, logical_block_size=1664
        ),
        donate_argnums=(0,),
    )
    canonical = words.copy()
    exponent = words & 0x7F80
    canonical[exponent == 0] &= 0x8000
    canonical[(exponent == 0x7F80) & ((words & 0x7F) != 0)] = 0x7FC0
    executable = None
    for lengths, contexts in (
        ([1024] * 8, [0] * 8),
        ([127, 130, 0, 5, 29, 0, 7, 0], [1, 8193, 0, 3, 9, 0, 5, 0]),
    ):
        starts, context, _, req, logical, valid = _metadata(lengths, contexts)
        complete = valid & (logical % 4 == 3)
        args = (keys, complete, req, logical, starts, context, table)
        if executable is None:
            executable = write.lower(pool, *args).compile()
            assert "qsa_compressed_indexed_write_tc" in executable.as_text()
        expected = np.asarray(pool).copy()
        # Independent per-completed-key writes, including old untouched half words.
        for row in np.flatnonzero(np.asarray(complete)):
            position = int(logical[row])
            pair, half = (position % 1664) // 8, (position // 4) % 2
            if fp8:
                target = expected.view(np.uint8)
                target[2, pair, 0, 2 * half] = (canonical[row] & 255).astype(np.uint8)
                target[2, pair, 0, 2 * half + 1] = (canonical[row] >> 8).astype(
                    np.uint8
                )
            else:
                expected.view(np.uint16)[2, pair, 0, half] = canonical[row]
        _equal(executable(pool.copy().block_until_ready(), *args), expected)
    record_property("native_indexed_writer", True)


def test_early_sc_preparation_keeps_two_callsite_owners_and_groups(record_property):
    """Execute the production early-read/barrier/join path with real top-k/attention."""
    rng = np.random.default_rng(8193)

    def rand(shape):
        return jnp.asarray(rng.normal(0, 0.2, shape), jnp.bfloat16)

    _, _, _, ring, tags, _, trig = _fixture(False)
    ids = jnp.arange(1, 9, dtype=jnp.int32)
    table = np.zeros((8, 16), np.int32)
    table[:, :3] = 1 + np.arange(24).reshape(8, 3)
    table = jnp.asarray(table)
    main = jnp.zeros((25, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    index = jnp.zeros((25, 208, 1, 2, 128), jnp.bfloat16)
    ring_owner = jnp.zeros((16, 8, 1, 2, 128), jnp.bfloat16)
    tags = tags.at[:, :, 0].set(2048 - ((2048 - jnp.arange(8)) % 4))
    ring_owner = jax.jit(qsa_state.write_ring_and_tags)(
        ring_owner, ring.reshape(8, -1), tags.reshape(8, -1), ids
    ).block_until_ready()
    second_main = main.at[1:25].set(jnp.float8_e4m3fn(0.25))
    second_index = index.at[1:25].set(jnp.bfloat16(0.125))
    second_ring = jax.jit(qsa_state.write_ring_and_tags)(
        ring_owner, (ring + 1).reshape(8, -1), tags.reshape(8, -1), ids
    ).block_until_ready()
    weights = (rand((256,)), rand((256,)), rand((128,)), rand((128,)))
    distribution = jnp.array([0, 0, 8], jnp.int32)

    def chain(
        owners, qgate, key, value, index_qk, positions, starts, context, *, early
    ):
        results = []
        for layer in range(2):
            results.extend(
                pooled_qsa.compact_pooled_qsa_step(
                    qgate,
                    key,
                    value,
                    index_qk,
                    *weights,
                    positions,
                    trig,
                    owners[3 * layer],
                    owners[3 * layer + 1],
                    owners[3 * layer + 2],
                    starts,
                    context,
                    ids,
                    table,
                    table,
                    distribution,
                    4,
                    2048,
                    1e-6,
                    64,
                    scheduling_group_namespace=layer + 1,
                    enable_sc_preparation=early,
                )
            )
        return tuple(results)

    owners = [
        [
            owner.copy().block_until_ready()
            for owner in (
                main,
                index,
                ring_owner,
                second_main,
                second_index,
                second_ring,
            )
        ]
        for _ in range(2)
    ]
    contexts = np.full(8, 2049, np.int32)
    runs = []
    for step, lengths in enumerate(
        ([1024] * 8, [127, 130, 0, 5, 29, 0, 7, 0], [127, 130, 0, 5, 29, 0, 7, 0])
    ):
        starts, context, _, _, logical, valid = _metadata(lengths, contexts)
        values = (
            rand((8192, 4 * 512)),
            rand((8192, 2 * 256)),
            rand((8192, 2 * 256)),
            rand((8192, 5 * 128)),
            jnp.where(valid, logical, 0).astype(jnp.int32),
            starts,
            context,
        )
        if step == 0:
            assert use_prefill_index_preparation(values[3][:, -128:], 4, requests=8)
            for early in (False, True):
                run = jax.jit(
                    functools.partial(chain, early=early), donate_argnums=(0,)
                )
                runs.append(run.lower(owners[len(runs)], *values).compile())
            _assert_native(runs[0].as_text(), "tc", writer=True)
            hlo = runs[1].as_text()
            _assert_native(hlo, "sc", writer=True)
            entry = hlo[hlo.index("ENTRY ") :].splitlines()
            for namespace in (1, 2):
                group = config.scoped_scheduling_group_id(namespace, 16384)
                lines = [
                    (i, line)
                    for i, line in enumerate(entry)
                    if f'_scheduling_group_id="{group}"' in line
                ]
                starts_at = [
                    i
                    for i, line in lines
                    if "qsa_index_window_mean_sc" in line
                    and ("async-start(" in line or "call-start(" in line)
                ]
                writes_at = [
                    i
                    for i, line in lines
                    if "qwen4_exp_main_kv_batched_ordered_scatter" in line
                    and " custom-call(" in line
                ]
                joins_at = [
                    i
                    for i, line in lines
                    if "qsa_index_window_mean_sc" in line
                    and ("async-done(" in line or "call-done(" in line)
                ]
                assert len(starts_at) == len(writes_at) == len(joins_at) == 1, [
                    line.split("backend_config=")[0] for _, line in lines
                ]
                assert starts_at[0] < writes_at[0] < joins_at[0]
        results = [run(state, *values) for run, state in zip(runs, owners, strict=True)]
        jax.block_until_ready(results)
        assert not np.array_equal(_bits(results[0][3]), _bits(results[0][8]))
        for actual, expected in zip(results[1], results[0], strict=True):
            _equal(actual, expected)
        owners = [[output[i] for i in (0, 1, 2, 5, 6, 7)] for output in results]
        contexts += np.asarray(lengths)
    record_property("early_sc_namespaces", [1, 2])
    record_property("full_two_callsite_outputs_and_owners_bitwise_equal", True)
