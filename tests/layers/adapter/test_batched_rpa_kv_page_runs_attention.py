"""KV page runs (TPU_BATCHED_RPA_KV_PAGE_RUNS) give identical batched attention.

Run on a TPU: `PYTHONPATH=src python -m pytest -v <this file>`.

Each case runs the real batched kernel on the same inputs with one DMA per KV
page and with `kv_run_pages`, through the paths a decode step takes: MTP verify
windows, one-query draft passes and the page-table bound. Page tables come from
2176-token manager blocks (17 kernel pages) in shuffled order, as vLLM builds
them. Outputs of the live rows and the KV caches must be bitwise equal.
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
_PAGE = 128
_RUN_PAGES = 17
_PAGES_PER_SEQ = 482 * _RUN_PAGES  # 1M-token table of 2176-token blocks
_LANE = configs.KVLayout.SEQ_ALONG_LANE
_HIST = {
    0: 131068,
    1: 2175,
    2: 2176,
    3: 70000,
    4: 255,
    5: 131000,
    6: 6143,
    7: 1,
    8: 4352,
    9: 120001,
    10: 3072,
    11: 50000,
}


@pytest.fixture(autouse=True)
def _require_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("batched RPA Pallas kernels require a TPU")


def _cdiv(a, b):
    return -(-a // b)


def _inputs(histories, q_lens, num_tokens):
    live = len(histories)
    dtype = jnp.dtype("float8_e4m3fn")
    blocks = {
        s: _cdiv(_cdiv(h + q_lens[s], _PAGE), _RUN_PAGES) for s, h in histories.items()
    }
    pool_blocks = sum(blocks.values()) + 9
    order = np.random.default_rng(0).permutation(np.arange(1, pool_blocks))
    table = np.zeros((_NUM_SEQS, _PAGES_PER_SEQ), np.int32)
    seq_q = np.zeros(_NUM_SEQS, np.int32)
    kv_lens = np.zeros(_NUM_SEQS, np.int32)
    start = 0
    for seq in range(live):
        ids = order[start : start + blocks[seq]]
        start += blocks[seq]
        pages = (ids[:, None] * _RUN_PAGES + np.arange(_RUN_PAGES)).ravel()
        table[seq, : len(pages)] = pages
        seq_q[seq] = q_lens[seq]
        kv_lens[seq] = histories[seq] + q_lens[seq]
    cache_shape = wrapper.get_kv_cache_shape(
        pool_blocks * _RUN_PAGES, _PAGE, 1, 256, dtype, kv_layout=_LANE
    )
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
    out, new_cache = _run(functools.partial(fn, kv_run_pages=_RUN_PAGES), inputs)
    np.testing.assert_array_equal(out.view(np.uint16), base_out.view(np.uint16))
    np.testing.assert_array_equal(new_cache.view(np.uint8), base_cache.view(np.uint8))


def test_verify_windows(monkeypatch):
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", "window")
    q_lens = {s: _WINDOW for s in _HIST}
    _check(_verify_path(), _inputs(_HIST, q_lens, _WINDOW * _NUM_SEQS))


def test_one_query_draft_passes(monkeypatch):
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", "window")
    fn = attention._with_one_query_steps(_verify_path(), _WINDOW)
    _check(fn, _inputs(_HIST, {s: 1 for s in _HIST}, _NUM_SEQS))


def test_partial_live_batch(monkeypatch):
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", "window")
    hist = {s: _HIST[s] for s in range(5)}
    q_lens = {s: 1 + 3 * (s % 2) for s in hist}
    _check(_verify_path(), _inputs(hist, q_lens, _WINDOW * _NUM_SEQS))


def test_addressable_page_table(monkeypatch):
    # A 256K max-model-len keeps 2176 of the 8194 table columns.
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", "window")
    fn = attention._with_addressable_page_table(_verify_path(), 262144, _WINDOW)
    q_lens = {s: _WINDOW for s in _HIST}
    _check(fn, _inputs(_HIST, q_lens, _WINDOW * _NUM_SEQS))
