"""The batched RPA kernel gives identical results on the addressable columns.

Run on a TPU: `PYTHONPATH=src python -m pytest -v <this file>`.

Each case runs the real kernel twice on the same inputs: once on the full block
table (as wide as the scheduler blocks round up to) and once through
`_with_addressable_page_table`, and requires bitwise-equal outputs and caches.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper
from vllm_torchtpu.layers.adapter import attention

_MAX_MODEL_LEN = 1048576
_LONG_BLOCK = configs.BlockSizes(
    bq_sz=1, bq_c_sz=1, bkv_sz=8192, batch_size=4, n_buffer=2
)


@pytest.fixture(autouse=True)
def _require_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("batched RPA Pallas kernels require a TPU")


def _run_case(page_size, pages_per_seq, num_seqs, histories, stream, block=None):
    """histories maps a sequence slot to its cached tokens; one new token each."""
    dtype = jnp.dtype("float8_e4m3fn")
    used = {s: (h + page_size) // page_size for s, h in histories.items()}
    total_pages = sum(used.values())
    cache_shape = wrapper.get_kv_cache_shape(
        total_pages, page_size, 1, 256, dtype, kv_layout=configs.KVLayout.SEQ_ALONG_LANE
    )
    table = np.zeros((num_seqs, pages_per_seq), np.int32)
    perm = np.random.default_rng(0).permutation(total_pages).astype(np.int32)
    q_lens = np.zeros(num_seqs, np.int32)
    kv_lens = np.zeros(num_seqs, np.int32)
    start = 0
    for seq in sorted(histories):
        table[seq, : used[seq]] = perm[start : start + used[seq]]
        start += used[seq]
        q_lens[seq] = 1
        kv_lens[seq] = histories[seq] + 1
    num_tokens = len(histories)
    keys = jax.random.split(jax.random.key(0), 4)
    cache = jax.random.normal(keys[0], cache_shape, jnp.float32).astype(dtype)
    q = jax.random.normal(keys[1], (num_tokens, 16, 256), jnp.float32).astype(
        jnp.bfloat16
    )
    k = jax.random.normal(keys[2], (num_tokens, 1, 256), jnp.float32).astype(dtype)
    v = jax.random.normal(keys[3], (num_tokens, 1, 256), jnp.float32).astype(dtype)
    cu_q_lens = jnp.asarray(np.concatenate(([0], np.cumsum(q_lens))), jnp.int32)
    distribution = jnp.array([num_seqs] * 3, jnp.int32)
    kwargs = {
        "kv_layout": configs.KVLayout.SEQ_ALONG_LANE,
        "stream_page_indices": stream,
    }
    if block is not None:
        kwargs["decode_block_sizes"] = block

    def run(fn):
        return jax.block_until_ready(
            fn(
                q,
                k,
                v,
                cache.copy(),
                jnp.asarray(kv_lens),
                jnp.asarray(table.ravel()),
                cu_q_lens,
                distribution,
                **kwargs,
            )
        )

    full_out, full_cache = run(wrapper.ragged_paged_attention)
    bounded = attention._with_addressable_page_table(
        wrapper.ragged_paged_attention, _MAX_MODEL_LEN, 1
    )
    out, new_cache = run(bounded)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(full_out))
    np.testing.assert_array_equal(np.asarray(new_cache), np.asarray(full_cache))


# 48 sequences at 128K on a 1M table: 8194 columns (2176-token blocks) -> 8192.
_P128 = {0: 255, 6: 1000, 7: 131071, 13: 300, 14: 5000, 47: 140000}
# 72 sequences on 256-token pages: 4104 columns (2304-token blocks) -> 4096.
_P256 = {0: 255, 14: 133000, 15: 511, 30: 70000, 71: 1000}


@pytest.mark.parametrize("stream", [None, True])
def test_page128_streamed_table(stream):
    _run_case(128, 8194, 48, _P128, stream)


def test_page128_long_decode_block():
    _run_case(128, 8194, 48, _P128, None, _LONG_BLOCK)


def test_page128_compact_table():
    # Every history fits the wrapper's compact table: the compact branch runs.
    _run_case(128, 8194, 48, {0: 255, 7: 90000, 47: 20000}, None)


@pytest.mark.parametrize("stream", [None, True])
def test_page256_streamed_table(stream):
    _run_case(256, 4104, 72, _P256, stream, _LONG_BLOCK)
