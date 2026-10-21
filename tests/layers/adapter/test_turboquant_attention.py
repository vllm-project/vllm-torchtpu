# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""TPU tests for the TurboQuant kernel entry: real Pallas RPA + shard_map."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    ref_ragged_paged_attention,
)
from vllm_torchtpu.layers.adapter.turboquant_attention import (
    TURBOQUANT_KV_CACHE_DTYPES,
    PallasTurboQuantAttentionBackend,
    _tq_rpa_kernel_entry,
    get_tq_layout,
    get_tq_tables,
)
from vllm_torchtpu.layers.core import turboquant as tq

pytestmark = pytest.mark.tpu_test

HEAD_DIM = 128
PAGE_SIZE = 16
PAGES_PER_SEQ = 8
MAX_NUM_SEQS = 4
NUM_PAGES = 64
NUM_TOKENS = 64


def _case(
    preset: str,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int = HEAD_DIM,
    seed: int = 0,
):
    layout = get_tq_layout(preset, head_dim)
    tables = get_tq_tables(preset, head_dim)
    rng = np.random.default_rng(seed)

    kv_lens = np.array([70, 33, 20, 9], np.int32)
    q_lens = np.array([1, 1, 12, 9], np.int32)
    hist_lens = kv_lens - q_lens
    hist_cu = np.concatenate([[0], np.cumsum(hist_lens)]).astype(np.int32)
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    distribution = np.array([2, 2, 4], np.int32)
    page_indices = rng.permutation(NUM_PAGES)[: MAX_NUM_SEQS * PAGES_PER_SEQ].astype(
        np.int32
    )

    def rnd(*shape):
        return jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)

    page_words = layout.page_words_128(PAGE_SIZE)
    cache = jnp.zeros((NUM_PAGES, num_kv_heads, page_words, 128), jnp.uint8)
    n_hist = int(hist_cu[-1])
    cache, _ = tq.tq_ragged_paged_attention(
        rnd(n_hist, num_q_heads, head_dim),
        rnd(n_hist, num_kv_heads, head_dim),
        rnd(n_hist, num_kv_heads, head_dim),
        cache,
        jnp.asarray(hist_lens),
        jnp.asarray(page_indices),
        jnp.asarray(hist_cu),
        jnp.asarray([0, 3, 3], np.int32),
        layout=layout,
        tables=tables,
        rpa_fn=ref_ragged_paged_attention,
        sm_scale=head_dim**-0.5,
    )

    return dict(
        layout=layout,
        tables=tables,
        cache=cache,
        q=rnd(NUM_TOKENS, num_q_heads, head_dim),
        k=rnd(NUM_TOKENS, num_kv_heads, head_dim),
        v=rnd(NUM_TOKENS, num_kv_heads, head_dim),
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu=jnp.asarray(cu),
        distribution=jnp.asarray(distribution),
        total=int(cu[-1]),
    )


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("preset", TURBOQUANT_KV_CACHE_DTYPES)
@pytest.mark.parametrize("tp", [1, 2])
def test_kernel_entry_matches_reference(preset, tp, head_dim):
    if jax.devices()[0].platform != "tpu" or len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} TPU devices")
    num_kv_heads, num_q_heads = 2 * tp, 4 * tp
    c = _case(preset, num_kv_heads, num_q_heads, head_dim=head_dim)
    mesh = Mesh(np.asarray(jax.devices()[:tp]), ("model",))
    sm_scale = head_dim**-0.5

    entry = jax.jit(
        lambda *a: _tq_rpa_kernel_entry(
            *a,
            None,
            None,
            None,
            None,
            mesh=mesh,
            sliding_window=None,
            sm_scale=sm_scale,
            skip_kv_update=False,
            tq_preset=preset,
        )
    )
    new_cache, out = entry(
        c["cache"],
        c["q"],
        c["k"],
        c["v"],
        c["kv_lens"],
        c["page_indices"],
        c["cu"],
        c["distribution"],
    )
    ref_cache, ref_out = tq.tq_ragged_paged_attention(
        c["q"],
        c["k"],
        c["v"],
        c["cache"],
        c["kv_lens"],
        c["page_indices"],
        c["cu"],
        c["distribution"],
        layout=c["layout"],
        tables=c["tables"],
        rpa_fn=ref_ragged_paged_attention,
        sm_scale=sm_scale,
    )
    np.testing.assert_array_equal(np.asarray(new_cache), np.asarray(ref_cache))
    np.testing.assert_allclose(
        np.asarray(out[: c["total"]], np.float32),
        np.asarray(ref_out[: c["total"]], np.float32),
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("preset", TURBOQUANT_KV_CACHE_DTYPES)
def test_backend_page_geometry(preset, head_dim):
    layout = get_tq_layout(preset, head_dim)
    page_words = layout.page_words_128(PAGE_SIZE)
    shape = PallasTurboQuantAttentionBackend.get_kv_cache_shape(
        10, PAGE_SIZE, 4, head_dim, preset
    )
    assert shape == (10, 4, page_words, 128)
    page_bytes = PallasTurboQuantAttentionBackend.get_kv_cache_page_size_bytes(
        PAGE_SIZE, 4, head_dim, preset
    )
    assert page_bytes == page_words * 4 * 128
    assert page_bytes < PAGE_SIZE * 4 * 2 * head_dim * 2
