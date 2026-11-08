"""A verify window computed in one query chunk gives identical attention.

Run on a TPU: `PYTHONPATH=src python -m pytest -v <this file>`.

Each case runs the real batched kernel twice on the same verify-window inputs
(decode_query_size 4): once with the kernel's default decode block, which splits
the window into query chunks, and once through `_with_verify_block_override`
with TPU_BATCHED_RPA_VERIFY_BLOCK=window (same KV block, whole window per
chunk). Outputs and KV caches must be bitwise equal.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper
from vllm_torchtpu.layers.adapter import attention

_WINDOW = 4
# 1M-token max_model_len at 128-token pages: the kernel's compact table path.
_PAGES_PER_SEQ = 8192


@pytest.fixture(autouse=True)
def _require_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("batched RPA Pallas kernels require a TPU")


def _run_case(monkeypatch, histories, q_lens, num_seqs=12, page_size=128):
    """histories / q_lens map a sequence slot to cached and new tokens."""
    dtype = jnp.dtype("float8_e4m3fn")
    used = {
        s: (h + q_lens[s] + page_size - 1) // page_size for s, h in histories.items()
    }
    total_pages = sum(used.values())
    cache_shape = wrapper.get_kv_cache_shape(
        total_pages, page_size, 1, 256, dtype, kv_layout=configs.KVLayout.SEQ_ALONG_LANE
    )
    table = np.zeros((num_seqs, _PAGES_PER_SEQ), np.int32)
    perm = np.random.default_rng(0).permutation(total_pages).astype(np.int32)
    seq_q = np.zeros(num_seqs, np.int32)
    kv_lens = np.zeros(num_seqs, np.int32)
    start = 0
    for seq in sorted(histories):
        table[seq, : used[seq]] = perm[start : start + used[seq]]
        start += used[seq]
        seq_q[seq] = q_lens[seq]
        kv_lens[seq] = histories[seq] + q_lens[seq]
    num_tokens = num_seqs * _WINDOW
    keys = jax.random.split(jax.random.key(0), 4)
    cache = jax.random.normal(keys[0], cache_shape, jnp.float32).astype(dtype)
    q = jax.random.normal(keys[1], (num_tokens, 16, 256), jnp.float32).astype(
        jnp.bfloat16
    )
    k = jax.random.normal(keys[2], (num_tokens, 1, 256), jnp.float32).astype(dtype)
    v = jax.random.normal(keys[3], (num_tokens, 1, 256), jnp.float32).astype(dtype)
    cu_q_lens = jnp.asarray(np.concatenate(([0], np.cumsum(seq_q))), jnp.int32)
    distribution = jnp.array([num_seqs] * 3, jnp.int32)
    entry = functools.partial(wrapper.ragged_paged_attention, decode_query_size=_WINDOW)

    def run(fn):
        out, new_cache = fn(
            q,
            k,
            v,
            cache.copy(),
            jnp.asarray(kv_lens),
            jnp.asarray(table.ravel()),
            cu_q_lens,
            distribution,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )
        jax.block_until_ready((out, new_cache))
        return np.asarray(out[: int(cu_q_lens[-1])]), np.asarray(new_cache)

    base_out, base_cache = run(entry)

    defaults = []
    calculate = wrapper.calculate_block_sizes

    def spy(*args, **kwargs):
        defaults.append(calculate(*args, **kwargs)[0])
        return calculate(*args, **kwargs)

    monkeypatch.setattr(attention.rpa_batched_wrapper, "calculate_block_sizes", spy)
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", "window")
    out, new_cache = run(attention._with_verify_block_override(entry, _WINDOW))
    # The default block splits the window; the override computes it at once.
    assert defaults and defaults[0].bq_sz == _WINDOW
    assert defaults[0].bq_c_sz < _WINDOW, defaults[0]
    np.testing.assert_array_equal(out, base_out)
    np.testing.assert_array_equal(new_cache, base_cache)


# Histories around the 3072-token KV block edges plus 128K contexts.
_HIST = {
    0: 131068,
    1: 3068,
    2: 3069,
    3: 70000,
    4: 255,
    5: 131000,
    6: 6140,
    7: 1,
    8: 9000,
    9: 120001,
    10: 4093,
    11: 50000,
}


def test_full_verify_windows(monkeypatch):
    _run_case(monkeypatch, _HIST, {s: _WINDOW for s in _HIST})


def test_one_token_draft_passes(monkeypatch):
    # Draft passes after the first carry one token per sequence.
    _run_case(monkeypatch, _HIST, {s: 1 for s in _HIST})


def test_mixed_window_lengths(monkeypatch):
    _run_case(monkeypatch, _HIST, {s: 1 + s % _WINDOW for s in _HIST})


def test_partial_batch(monkeypatch):
    # Five live sequences in a 12-slot batch: the second kernel batch is short.
    hist = {s: _HIST[s] for s in (0, 3, 5, 9, 11)}
    _run_case(monkeypatch, hist, {s: _WINDOW for s in hist})
