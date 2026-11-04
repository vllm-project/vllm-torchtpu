"""One-query draft passes give identical attention on the one-query kernel.

Run on a TPU: `PYTHONPATH=src python -m pytest -v <this file>`.

MTP draft passes after the first attend one query per sequence. Each case runs
the real batched kernel on the same inputs through the verify-window path
(decode_query_size 4, TPU_BATCHED_RPA_VERIFY_BLOCK) and through
`_with_one_query_steps`, which switches on device to the one-query kernel at
the verify block's KV block. Outputs of the live rows and the KV caches must be
bitwise equal. Calls whose sequences carry more than one query must stay on the
verify-window kernel.
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
# 1M-token max_model_len at 128-token pages: the kernel's compact table path.
_PAGES_PER_SEQ = 8192
_LANE = configs.KVLayout.SEQ_ALONG_LANE

# Histories around the 3072-token KV block edges plus 128K contexts.
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
    """Live sequences are slots 0..n-1, as the draft loop steps lay them out."""
    live = len(histories)
    assert sorted(histories) == list(range(live))
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
    keys = jax.random.split(jax.random.key(0), 4)
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


def _poisoned(fn):
    """fn with every output element shifted: marks which branch ran."""

    def call(*args, **kwargs):
        out, new_cache = fn(*args, **kwargs)
        return out + 1, new_cache

    return call


def _check(monkeypatch, verify_block, histories, q_lens, num_tokens, one_query):
    """The switch must match the verify path and run the expected kernel: the
    other branch is poisoned, so taking it would change the outputs."""
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", verify_block)
    inputs = _inputs(histories, q_lens, num_tokens)
    verify = _verify_path()
    base_out, base_cache = _run(verify, inputs)
    if one_query:
        switched = attention._with_one_query_steps(_poisoned(verify), _WINDOW)
    else:
        monkeypatch.setattr(
            attention,
            "ragged_paged_attention_batched",
            _poisoned(wrapper.ragged_paged_attention),
        )
        switched = attention._with_one_query_steps(verify, _WINDOW)
    out, new_cache = _run(switched, inputs)
    np.testing.assert_array_equal(out, base_out)
    np.testing.assert_array_equal(new_cache, base_cache)
    return inputs, base_out, base_cache


def _one_query_block(verify_block, inputs):
    if verify_block != "window":
        bkv, batch, buffers = (int(x) for x in verify_block.split(","))
    else:
        q, k, _, cache, kv_lens, table, _, _ = inputs
        default = attention._kernel_decode_block(
            q, k, cache, kv_lens, table, _WINDOW, {"kv_layout": _LANE}
        )
        assert default.bq_sz == _WINDOW
        bkv, batch, buffers = default.bkv_sz, default.batch_size, default.n_buffer
    return configs.BlockSizes(
        bq_sz=1, bq_c_sz=1, bkv_sz=bkv, batch_size=batch, n_buffer=buffers
    )


@pytest.mark.parametrize("verify_block", ["window", "8192,4,2"])
def test_one_query_pass_matches_the_verify_kernel(monkeypatch, verify_block):
    inputs, base_out, base_cache = _check(
        monkeypatch, verify_block, _HIST, {s: 1 for s in _HIST}, _NUM_SEQS, True
    )
    # The one-query kernel on its own, at the verify block's KV block.
    block = _one_query_block(verify_block, inputs)
    out, new_cache = _run(
        functools.partial(wrapper.ragged_paged_attention, decode_block_sizes=block),
        inputs,
    )
    np.testing.assert_array_equal(out, base_out)
    np.testing.assert_array_equal(new_cache, base_cache)


def test_partial_live_batch(monkeypatch):
    # Five live sequences in a 12-slot batch: query_start_loc stops at 5.
    hist = {s: _HIST[s] for s in range(5)}
    _check(monkeypatch, "window", hist, {s: 1 for s in hist}, _NUM_SEQS, True)


def test_verify_windows_keep_the_verify_kernel(monkeypatch):
    # Three 4-token windows in a 12-token program: the switch is compiled in and
    # must pick the verify-window kernel.
    hist = {s: _HIST[s] for s in range(3)}
    _check(monkeypatch, "window", hist, {s: _WINDOW for s in hist}, _NUM_SEQS, False)


def test_mixed_window_lengths_keep_the_verify_kernel(monkeypatch):
    hist = {s: _HIST[s] for s in range(6)}
    q_lens = {s: 1 + (s == 2) for s in hist}
    _check(monkeypatch, "window", hist, q_lens, _NUM_SEQS, False)
