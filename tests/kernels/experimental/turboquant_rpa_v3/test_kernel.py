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
"""Deviceless and TPU tests for `kernels.experimental.turboquant_rpa_v3`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.turboquant_rpa_v3 import (
    prepare_tq_block_sizes,
    ragged_paged_attention,
    static_validate_inputs,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    RpaCase,
    ref_ragged_paged_attention,
)


def test_static_validate_inputs():
    s = jax.ShapeDtypeStruct
    base = dict(
        queries=s((16, 4, 128), jnp.bfloat16),
        keys=s((16, 2, 128), jnp.bfloat16),
        values=s((16, 2, 128), jnp.bfloat16),
        kv_cache=s((8, 2, 24, 128), jnp.uint8),
        kv_lens=s((2,), jnp.int32),
        page_indices=s((8,), jnp.int32),
        cu_q_lens=s((3,), jnp.int32),
        distribution=s((3,), jnp.int32),
        tq_page_size=16,
        tq_dequant_page_fn=lambda refs, dt, **kw: None,
        tq_new_kv=s((16, 3, 2, 128), jnp.uint8),
        tq_update_page_fn=lambda old_p, nw, mask, **kw: old_p,
    )
    static_validate_inputs(**base)  # type: ignore[arg-type]
    static_validate_inputs(
        **(base | dict(skip_kv_update=True, tq_new_kv=None, tq_update_page_fn=None))  # type: ignore[arg-type]
    )
    for overrides, match in [
        ({"tq_page_size": 0}, "tq_page_size"),
        ({"tq_dequant_page_fn": None}, "tq_dequant_page_fn"),
        ({"queries": s((16, 3, 128), jnp.bfloat16)}, "divisible"),
        ({"kv_cache": s((8, 2, 24, 128), jnp.bfloat16)}, "4D uint8 TurboQuant"),
        ({"tq_new_kv": None}, "tq_new_kv and tq_update_page_fn are required"),
        ({"tq_new_kv": s((8, 3, 2, 128), jnp.uint8)}, "4D uint8 tq_new_kv"),
        ({"d_block_sizes": (1, 24, 1, 24)}, "divisible by tq_page_size"),
        ({"sliding_window": 0}, "sliding_window"),
    ]:
        with pytest.raises(ValueError, match=match):
            static_validate_inputs(**(base | overrides))  # type: ignore[arg-type]


def test_prepare_tq_block_sizes(monkeypatch: pytest.MonkeyPatch):
    kw = dict(
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        actual_num_q_heads=8,
        actual_num_kv_heads=2,
        max_num_tokens=256,
        max_num_seqs=16,
        pages_per_seq=32,
    )
    assert prepare_tq_block_sizes(
        (2, 64, 1, 32), head_dim=128, page_size=16, case=RpaCase.MIXED, **kw
    ) == {"bq_sz": 2, "bkv_sz": 64, "bq_csz": 1, "bkv_csz": 32}

    for tpu_version in (6, 7):
        monkeypatch.setattr(
            "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel.get_tpu_version",
            lambda v=tpu_version: v,
        )
        for head_dim, page_size in ((128, 16), (128, 32), (256, 16), (256, 64)):
            for case in (RpaCase.DECODE, RpaCase.MIXED):
                s = prepare_tq_block_sizes(
                    None,
                    head_dim=head_dim,
                    page_size=page_size,
                    case=case,
                    tpu_version=tpu_version,
                    **kw,
                )
                assert s["bq_csz"] > 0 and s["bq_sz"] % s["bq_csz"] == 0
                assert s["bkv_csz"] % page_size == 0 and s["bkv_sz"] % s["bkv_csz"] == 0
                if head_dim == 256 and page_size < 64:
                    assert s["bkv_csz"] <= 64


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("preset", "head_dim", "page_size", "q_lens", "kv_lens", "dist", "kw"),
    [
        pytest.param(
            "turboquant_k8v4",
            128,
            32,
            [1, 1, 10, 8],
            [98, 40, 42, 8],
            [2, 2, 4],
            {},
            id="k8v4_p32_mixed",
        ),
        pytest.param(
            "turboquant_k3v4_nc",
            128,
            16,
            [1, 1, 10, 8],
            [66, 20, 42, 8],
            [2, 2, 4],
            {},
            id="k3v4_p16_mixed",
        ),
        pytest.param(
            "turboquant_4bit_nc",
            256,
            16,
            [1, 1, 1, 1],
            [65, 33, 17, 5],
            [4, 4, 4],
            {},
            id="4bit_hd256_decode",
        ),
        pytest.param(
            "turboquant_3bit_nc",
            256,
            64,
            [1, 1, 10, 8],
            [132, 70, 80, 8],
            [2, 2, 4],
            {},
            id="3bit_hd256_p64",
        ),
        pytest.param(
            "turboquant_4bit_nc",
            128,
            16,
            [1, 1, 9, 6],
            [70, 35, 45, 22],
            [2, 2, 4],
            dict(sliding_window=32, soft_cap=30.0),
            id="4bit_sw_softcap",
        ),
        pytest.param(
            "turboquant_k8v4",
            128,
            16,
            [1, 1, 9, 6],
            [70, 35, 45, 22],
            [2, 2, 4],
            dict(skip_kv_update=True),
            id="k8v4_skip_update",
        ),
    ],
)
def test_pallas_kernel_matches_reference(
    preset, head_dim, page_size, q_lens, kv_lens, dist, kw
):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs TPU")
    from vllm.model_executor.layers.quantization.turboquant.centroids import (
        get_centroids,
    )
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    from vllm_torchtpu.layers.core import turboquant as tq

    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim)
    layout = tq.TQLayout.from_tq_config(cfg)
    tables = tq.make_tables(
        layout, get_centroids(head_dim, cfg.centroid_bits).cpu().numpy()
    )
    sm_scale = head_dim**-0.5

    max_num_seqs = len(kv_lens)
    pages_per_seq = max(8, (max(kv_lens) + page_size - 1) // page_size + 1)
    num_pages = max_num_seqs * pages_per_seq
    num_kv_heads, num_q_heads, num_tokens = 2, 4, max(32, sum(q_lens))
    rng = np.random.default_rng(0)

    def rnd(*shape):
        return jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)

    page_indices = jnp.asarray(rng.permutation(num_pages), jnp.int32)
    kv_arr, q_arr = np.asarray(kv_lens, np.int32), np.asarray(q_lens, np.int32)
    hist_lens = kv_arr - q_arr
    cache = jnp.zeros(
        (num_pages, num_kv_heads, layout.page_words_128(page_size), 128), jnp.uint8
    )
    active = np.flatnonzero(hist_lens > 0)
    if active.size > 0:
        a_lens = hist_lens[active]
        a_cu = np.concatenate([[0], np.cumsum(a_lens)]).astype(np.int32)
        a_pages = page_indices.reshape(max_num_seqs, pages_per_seq)[active].reshape(-1)
        n_hist = int(a_cu[-1])
        cache, _ = tq.tq_ragged_paged_attention(
            rnd(n_hist, num_q_heads, head_dim),
            rnd(n_hist, num_kv_heads, head_dim),
            rnd(n_hist, num_kv_heads, head_dim),
            cache,
            jnp.asarray(a_lens),
            a_pages,
            jnp.asarray(a_cu),
            jnp.asarray([0, 0, int(active.size)], jnp.int32),
            layout=layout,
            tables=tables,
            rpa_fn=ref_ragged_paged_attention,
            sm_scale=sm_scale,
        )

    q = rnd(num_tokens, num_q_heads, head_dim)
    k = rnd(num_tokens, num_kv_heads, head_dim)
    v = rnd(num_tokens, num_kv_heads, head_dim)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_arr)]).astype(np.int32))
    common = (
        jnp.asarray(kv_arr),
        page_indices,
        cu,
        jnp.asarray(dist, jnp.int32),
    )
    got_cache, got_out = tq.tq_ragged_paged_attention(
        q.copy(),
        k.copy(),
        v.copy(),
        cache.copy(),
        *common,
        layout=layout,
        tables=tables,
        rpa_fn=ragged_paged_attention,
        sm_scale=sm_scale,
        **kw,
    )
    ref_cache, ref_out = tq.tq_ragged_paged_attention(
        q,
        k,
        v,
        cache,
        *common,
        layout=layout,
        tables=tables,
        rpa_fn=ref_ragged_paged_attention,
        sm_scale=sm_scale,
        **kw,
    )
    if kw.get("skip_kv_update"):
        np.testing.assert_array_equal(np.asarray(got_cache), np.asarray(cache))
    np.testing.assert_array_equal(np.asarray(got_cache), np.asarray(ref_cache))
    total = int(q_arr.sum())
    np.testing.assert_allclose(
        np.asarray(got_out[:total], np.float32),
        np.asarray(ref_out[:total], np.float32),
        rtol=3e-2,
        atol=3e-2,
    )
