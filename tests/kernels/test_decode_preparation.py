# SPDX-License-Identifier: Apache-2.0
"""CPU guards and interpret-mode parity for decode preparation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.attention_preprocessing import compress_rope
from vllm_torchtpu.kernels.deepseek_v41.csa2 import cache
from vllm_torchtpu.kernels.experimental import decode_preparation as prep


class DispatchReached(Exception):
    pass


def stop(path):
    def reached(*args, **kwargs):
        raise DispatchReached(path)

    return reached


@pytest.fixture(autouse=True)
def fresh_policy(monkeypatch):
    monkeypatch.setattr(prep, "_mode", prep.FusionMode.AUTO)
    monkeypatch.setattr(prep, "_mode_used", False)
    monkeypatch.setattr(prep, "_reported", set())


@pytest.fixture
def tpu(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(prep, "get_tpu_version", lambda: 7)


BUCKETS = frozenset({96, 288})


def test_admit_auto_logs_once(tpu, caplog):
    assert prep.admit("r", tokens=288, buckets=BUCKETS)
    with caplog.at_level("WARNING"):
        assert not prep.admit("r", tokens=128, buckets=BUCKETS)
        assert not prep.admit("r", tokens=128, buckets=BUCKETS)
    assert len([r for r in caplog.records if "falls back" in r.message]) == 1


def test_admit_prefill_sized_is_silent_even_under_require(tpu):
    prep.configure("require")
    assert not prep.admit("r", tokens=4096, buckets=BUCKETS)


def test_admit_require_raises_with_reason(tpu):
    prep.configure("require")
    with pytest.raises(NotImplementedError, match="'r'.*page size 32"):
        prep.admit("r", "page size 32", tokens=288, buckets=BUCKETS)
    with pytest.raises(NotImplementedError, match="128 is not a measured bucket"):
        prep.admit("r", tokens=128, buckets=BUCKETS)


def test_admit_report_false_never_raises(tpu):
    prep.configure("require")
    assert not prep.admit("r", tokens=128, buckets=BUCKETS, report=False)


def test_admit_off_and_cpu_take_native(tpu, monkeypatch):
    prep.configure("off")
    assert not prep.admit("r", tokens=288, buckets=BUCKETS)
    monkeypatch.setattr(prep, "_mode", prep.FusionMode.REQUIRE)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert not prep.admit("r", tokens=128, buckets=BUCKETS)


def test_admit_tuple_buckets(tpu):
    buckets = prep.SWA_BATCHES
    assert prep.admit("swa", tokens=(288, 48), buckets=buckets)
    assert not prep.admit("swa", tokens=(288, 16), buckets=buckets)
    assert not prep.admit("swa", tokens=(1024, 48), buckets=buckets)


def test_configure_is_frozen_after_trace(tpu):
    prep.admit("r", tokens=288, buckets=BUCKETS)
    prep.configure("auto")  # same mode is fine
    with pytest.raises(RuntimeError, match="already auto"):
        prep.configure("off")


def test_tpu_version_problem(tpu, monkeypatch):
    assert prep.tpu_version_problem() is None
    monkeypatch.setattr(prep, "get_tpu_version", lambda: 6)
    assert "v6" in prep.tpu_version_problem()


@pytest.mark.parametrize(
    "tokens,requests,fused",
    [
        (96, 16, True),
        (192, 32, True),
        (288, 32, True),
        (288, 48, True),
        (96, 32, False),
        (192, 16, False),
        (288, 16, False),
        (128, 16, False),
    ],
)
def test_swa_bucket_dispatch(tpu, monkeypatch, tokens, requests, fused):
    monkeypatch.setattr(cache, "prepare_swa_metadata", stop("fallback"))
    monkeypatch.setattr(prep, "_swa_capacities", stop("fused"))
    with pytest.raises(DispatchReached, match="^fused$" if fused else "^fallback$"):
        prep.swa_prepare_publish(
            (),
            jax.ShapeDtypeStruct((requests, 8), jnp.int32),
            None,
            jax.ShapeDtypeStruct((tokens,), jnp.int32),
            None,
            None,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_capacity", 32),
        ("state_stride", 32),
        ("state_offset", 1),
        ("window", 256),
        ("main_capacity", 64),
        ("index_capacity", 128),
    ],
)
def test_compressor_layout_problem_names_field(field, value):
    default = prep.DEFAULT_COMPRESSOR_LAYOUT
    assert prep.compressor_layout_problem(default) is None
    layout = prep.CompressorLayout(**{field: value})
    problem = prep.compressor_layout_problem(layout)
    assert problem is not None and f"{field}={value}" in problem
    assert layout.tag != default.tag


@pytest.mark.cpu_test
@pytest.mark.parametrize("state_capacity", [16, 32])
def test_compressor_layout_guard_compiles_fullgraph(state_capacity):
    """Dynamo must inspect a bound layout without a graph break at warmup."""
    import dataclasses

    import torch

    def guard(x):
        layout = dataclasses.replace(
            prep.DEFAULT_COMPRESSOR_LAYOUT, state_capacity=state_capacity
        )
        problem = prep.compressor_layout_problem(layout)
        return x + (1 if problem is None else 2)

    compiled = torch.compile(guard, backend="eager", fullgraph=True, dynamic=False)
    x = torch.zeros(2)
    expected = x + (1 if state_capacity == 16 else 2)
    torch.testing.assert_close(compiled(x), expected)
    torch.testing.assert_close(compiled(x), expected)


def _compressor_inputs(tokens=96, requests=16, columns=8, seed=0):
    """Random decode batch: some empty requests, padded tail tokens."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 2 * tokens // requests, requests)
    lengths[0] = 0  # an empty request
    while lengths.sum() > tokens - 3:
        lengths[np.argmax(lengths)] -= 1
    starts = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    positions = np.zeros(tokens, np.int32)
    minimum = np.zeros(requests, np.int32)
    for r in range(requests):
        base = rng.integers(0, 500)
        positions[starts[r] : starts[r + 1]] = base + np.arange(lengths[r])
        if r % 3 == 1 and lengths[r]:
            minimum[r] = max(base - 5, 1)  # bounded replay window hits
    table = rng.integers(0, 40, (requests, columns)).astype(np.int32)
    return tuple(jnp.asarray(x) for x in (table, starts, starts, positions, minimum))


@pytest.mark.parametrize("change", ["table", "starts", "minimum", "dtype"])
def test_compressor_inputs_problem(change):
    table, starts, kv, positions, minimum = _compressor_inputs()
    assert prep.compressor_inputs_problem(table, starts, kv, positions, minimum) is None
    if change == "table":
        table = table[:-1]
    elif change == "starts":
        starts = starts[:-1]
    elif change == "minimum":
        minimum = minimum[:-1]
    else:
        positions = positions.astype(jnp.int16)
    assert prep.compressor_inputs_problem(table, starts, kv, positions, minimum)


@pytest.mark.parametrize(
    "tokens,requests,total", [(96, 16, 640), (192, 32, 640), (288, 48, 200)]
)
def test_compressor_state_packed_matches_native(tokens, requests, total):
    """Interpret-mode kernel equals native addresses + scatter compaction."""
    inputs = _compressor_inputs(tokens, requests)
    layout = prep.DEFAULT_COMPRESSOR_LAYOUT
    prev, valid, wp, wo, rp, ro = prep.compressor_state_packed(
        *inputs, total_records=total, interpret=True
    )
    n_prev, n_valid, write, read = prep._compressor_state_addresses(*inputs, layout)
    np.testing.assert_array_equal(prev[:, 0], n_prev)
    np.testing.assert_array_equal(valid[:, 0], n_valid)
    for pairs, offsets, records in ((wp, wo, write), (rp, ro, read)):
        expected = compress_rope._compact(records, total, 1, tokens)  # CPU scatter
        np.testing.assert_array_equal(pairs, expected[0])
        np.testing.assert_array_equal(offsets, expected[1])


def _reference_slots(kv_starts, positions, table, capacity, requests):
    token = np.arange(positions.size)
    request = np.minimum((token[:, None] >= kv_starts[None, 1:]).sum(1), requests - 1)
    logical = positions // 2
    column = np.minimum(logical // capacity, table.shape[1] - 1)
    block = table[request, column]
    live = token < kv_starts[-1]
    return live, block, logical % capacity


@pytest.mark.parametrize("tokens,requests", [(96, 16), (288, 48)])
@pytest.mark.parametrize("main_columns", [8, 130])  # crosses a 128-lane chunk
def test_compressor_addresses_match_reference(tokens, requests, main_columns):
    inputs = _compressor_inputs(tokens, requests)
    rng = np.random.default_rng(1)
    main = rng.integers(0, 1000, (requests, main_columns)).astype(np.int32)
    index = rng.integers(0, 1000, (requests, main_columns // 2 + 1)).astype(np.int32)
    layout = prep.CompressorLayout(main_offset=16, main_stride=144)
    out = prep.compressor_addresses_packed(
        *inputs,
        jnp.asarray(main),
        jnp.asarray(index),
        total_records=640,
        layout=layout,
        interpret=True,
    )
    state = prep.compressor_state_packed(
        *inputs, total_records=640, layout=layout, interpret=True
    )
    for got, want in zip(out[:6], state, strict=True):
        np.testing.assert_array_equal(got, want)
    kv, positions = np.asarray(inputs[2]), np.asarray(inputs[3])
    valid = np.asarray(state[1][:, 0]) != 0
    live, block, slot = _reference_slots(kv, positions, main, 128, requests)
    rows = layout.main_offset + block * layout.main_stride + slot
    np.testing.assert_array_equal(out[6], np.where(live & valid, rows, -1))
    live, block, slot = _reference_slots(kv, positions, index, 256, requests)
    np.testing.assert_array_equal(out[7], np.where(live, block * 256 + slot, -1))
