"""Decode schedules built with array ops give identical batched attention.

Run on a TPU: `PYTHONPATH=src python -m pytest -v <this file>`.

Each case runs the real batched kernel on the same inputs with the schedule
kernel and with `xla_decode_schedule=True` (the path xla_schedule),
through the paths the decode step takes: the MTP verify window, the one-query
draft passes and the one-query decode block. Outputs of the live rows and the
KV caches must be bitwise equal.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper
from vllm_torchtpu.layers.adapter import attention

_WINDOW = 4
_NUM_SEQS = 12
_PAGES_PER_SEQ = 8192
_LANE = configs.KVLayout.SEQ_ALONG_LANE
_HIST = {
    0: 131068,
    1: 3071,
    2: 3072,
    3: 70000,
    4: 255,
    5: 131000,
    6: 6143,
    7: 1,
    8: 9000,
    9: 120001,
    10: 4095,
    11: 50000,
}


@pytest.fixture(autouse=True)
def _require_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("batched RPA Pallas kernels require a TPU")


def _inputs(histories, q_lens, num_tokens, page_size=128):
    live = len(histories)
    dtype = jnp.dtype("float8_e4m3fn")
    used = {
        s: (h + q_lens[s] + page_size - 1) // page_size for s, h in histories.items()
    }
    total_pages = sum(used.values())
    cache_shape = wrapper.get_kv_cache_shape(
        total_pages, page_size, 1, 256, dtype, kv_layout=_LANE
    )
    table = np.zeros((_NUM_SEQS, _PAGES_PER_SEQ), np.int32)
    perm = np.random.default_rng(0).permutation(total_pages).astype(np.int32)
    seq_q = np.zeros(_NUM_SEQS, np.int32)
    kv_lens = np.zeros(_NUM_SEQS, np.int32)
    start = 0
    for seq in range(live):
        table[seq, : used[seq]] = perm[start : start + used[seq]]
        start += used[seq]
        seq_q[seq] = q_lens[seq]
        kv_lens[seq] = histories[seq] + q_lens[seq]
    keys = jax.random.split(jax.random.key(1), 4)
    cache = jax.random.normal(keys[0], cache_shape, jnp.float32).astype(dtype)
    q = jax.random.normal(keys[1], (num_tokens, 16, 256), jnp.float32).astype(
        jnp.bfloat16
    )
    k = jax.random.normal(keys[2], (num_tokens, 1, 256), jnp.float32).astype(dtype)
    v = jax.random.normal(keys[3], (num_tokens, 1, 256), jnp.float32).astype(dtype)
    cu_q_lens = np.concatenate(([0], np.cumsum(seq_q))).astype(np.int32)
    distribution = np.array([live] * 3, np.int32)
    return (
        q,
        k,
        v,
        cache,
        jnp.asarray(kv_lens),
        jnp.asarray(table.ravel()),
        jnp.asarray(cu_q_lens),
        jnp.asarray(distribution),
    )


def _run(fn, inputs):
    q, k, v, cache, *rest = inputs

    @jax.jit
    def step(q, k, v, cache, kv_lens, table, cu_q_lens, distribution):
        return fn(
            q, k, v, cache, kv_lens, table, cu_q_lens, distribution, kv_layout=_LANE
        )

    out, new_cache = step(q, k, v, cache.copy(), *rest)
    jax.block_until_ready((out, new_cache))
    rows = int(inputs[6][-1])
    return np.asarray(out[:rows]), np.asarray(new_cache)


def _verify_path():
    return attention._with_verify_block_override(
        functools.partial(wrapper.ragged_paged_attention, decode_query_size=_WINDOW),
        _WINDOW,
    )


def _check(fn, inputs):
    base_out, base_cache = _run(fn, inputs)
    out, new_cache = _run(functools.partial(fn, xla_decode_schedule=True), inputs)
    np.testing.assert_array_equal(out, base_out)
    np.testing.assert_array_equal(new_cache, base_cache)


def test_verify_windows(monkeypatch):
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", "window")
    q_lens = {s: _WINDOW for s in _HIST}
    _check(_verify_path(), _inputs(_HIST, q_lens, _WINDOW * _NUM_SEQS))


def test_one_query_draft_passes(monkeypatch):
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", "window")
    fn = attention._with_one_query_steps(_verify_path(), _WINDOW)
    _check(fn, _inputs(_HIST, {s: 1 for s in _HIST}, _NUM_SEQS))


def test_one_query_decode_block():
    block = configs.BlockSizes(
        bq_sz=1, bq_c_sz=1, bkv_sz=8192, batch_size=4, n_buffer=2
    )
    fn = functools.partial(wrapper.ragged_paged_attention, decode_block_sizes=block)
    _check(fn, _inputs(_HIST, {s: 1 for s in _HIST}, _NUM_SEQS))


def test_partial_live_batch(monkeypatch):
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", "window")
    hist = {s: _HIST[s] for s in range(5)}
    q_lens = {s: 1 + 3 * (s % 2) for s in hist}
    _check(_verify_path(), _inputs(hist, q_lens, _WINDOW * _NUM_SEQS))


def test_sequences_past_the_compact_table(monkeypatch):
    # A 500K-token history leaves the compact page table: the streamed
    # schedule keeps the kernel, the result must still match.
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", "window")
    hist = dict(_HIST)
    hist[3] = 500_000
    q_lens = {s: _WINDOW for s in hist}
    _check(_verify_path(), _inputs(hist, q_lens, _WINDOW * _NUM_SEQS))


def test_env_binds_the_schedule_choice(monkeypatch):
    seen = {}

    def capture(*args, rpa_func, **kwargs):
        seen["rpa_func"] = rpa_func

    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", capture)
    monkeypatch.delenv("TPU_BATCHED_RPA_DECODE_BLOCK", raising=False)
    call = functools.partial(
        attention._pallas_rpa_kernel_batched,
        *([None] * 12),
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=_LANE,
    )
    monkeypatch.setenv("TPU_DECODE_FAST_PATHS", "1")
    call()
    assert seen["rpa_func"].keywords.get("xla_decode_schedule") is True
    monkeypatch.setenv("TPU_DECODE_FAST_PATHS", "0")
    call()
    assert seen["rpa_func"] is attention.ragged_paged_attention_batched
