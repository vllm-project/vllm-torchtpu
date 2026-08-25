"""Sharded parity between the two batched-RPA KV cache layouts.

The KV-head axis that `sharded_ragged_paged_attention` splits sits at dim 2
under HEAD_ALONG_SUBLANE and dim 1 under SEQ_ALONG_LANE. The partition spec
only bites with more than one device, so single-chip tests cannot see this.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.kernels.experimental.batched_rpa import \
    configs as batched_rpa_configs
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper
from vllm_torchtpu.layers.common.attention_interface import (
    attention, ragged_paged_attention_batched)
from vllm_torchtpu.layers.common.attention_metadata import AttentionMetadata

pytestmark = pytest.mark.multichip

KVLayout = batched_rpa_configs.KVLayout
NUM_DEVICES = 4

HEAD_DIM = 128
NUM_KV_HEADS = 4
NUM_Q_HEADS = 8
PAGE_SIZE = 128  # SEQ_ALONG_LANE accepts no other page size
TOTAL_PAGES = 4
SEQ_LEN = 64


def _run(mesh, kv_layout, query, key, value, metadata, dtype):
    cache_shape = wrapper.get_kv_cache_shape(TOTAL_PAGES,
                                             PAGE_SIZE,
                                             NUM_KV_HEADS,
                                             HEAD_DIM,
                                             dtype,
                                             kv_layout=kv_layout)
    _, out = attention(jnp.zeros(cache_shape, dtype),
                       query,
                       key,
                       value,
                       metadata,
                       mesh,
                       rpa_func=ragged_paged_attention_batched,
                       kv_layout=kv_layout)
    return np.asarray(out.astype(jnp.float32)), cache_shape


def test_layouts_agree_when_sharded_over_kv_heads():
    if jax.device_count() < NUM_DEVICES:
        pytest.skip(f"needs {NUM_DEVICES} chips, saw {jax.device_count()}")

    dtype = jnp.bfloat16
    mesh = Mesh(np.asarray(jax.devices()[:NUM_DEVICES]), ("model", ))
    rng = np.random.default_rng(0)

    def r(*shape):
        return jnp.asarray((rng.standard_normal(shape) * 0.5), dtype)

    query = r(SEQ_LEN, NUM_Q_HEADS, HEAD_DIM)
    key = r(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
    value = r(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
    metadata = AttentionMetadata(
        input_positions=None,
        block_tables=jnp.arange(TOTAL_PAGES, dtype=jnp.int32),
        seq_lens=jnp.array([SEQ_LEN], jnp.int32),
        query_start_loc=jnp.array([0, SEQ_LEN], jnp.int32),
        request_distribution=jnp.asarray((0, 0, 1), jnp.int32),
    )

    baseline, baseline_shape = _run(mesh, KVLayout.HEAD_ALONG_SUBLANE, query,
                                    key, value, metadata, dtype)
    ported, ported_shape = _run(mesh, KVLayout.SEQ_ALONG_LANE, query, key,
                                value, metadata, dtype)

    # Guard against a vacuous pass: two all-zero outputs would also "agree".
    assert np.abs(baseline).max() > 0

    # The KV-head axis really is in different places, which is the whole point.
    assert baseline_shape[2] == NUM_KV_HEADS * 2 // 2
    assert ported_shape[1] == NUM_KV_HEADS * 2

    np.testing.assert_allclose(ported, baseline, atol=2e-2, rtol=0)
