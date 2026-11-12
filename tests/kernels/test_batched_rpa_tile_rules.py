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
"""Batched RPA tile rules on TPU7x.

By default every layer gets the tiles the block-size heuristic chose before the
tuned tile rules: the expected values for the models other than Qwen3.5 are
main's choices. On TPU7x, engine traces of the tile calls show the same values
for Qwen2.5-1.5B and Qwen3-0.6B, and the compiled kernel names show them for
Gemma-4-31B. With ``tuned_tiles`` the Qwen3.5 PD shapes get the tiles their
deployments run.
"""

import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_HEAD = configs.KVLayout.HEAD_ALONG_SUBLANE
_SEQ = configs.KVLayout.SEQ_ALONG_LANE


@pytest.fixture(autouse=True)
def tpu7x(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)


def _tiles(
    num_q_heads,
    num_kv_heads,
    head_dim,
    *,
    kv,
    layout,
    num_seqs,
    pages_per_seq,
    tokens,
    decode_query_size,
    sliding_window=None,
    serve_extra=None,
    q="bf16",
    **kwargs,
):
    model = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        mask_value=-1e30,
        sliding_window=sliding_window,
    )
    fp8 = kv == "fp8"
    fp8_q = q == "fp8"
    # A packed fp4 KV cache is stored as uint8, two fp4 values per byte.
    dtype_kv = {"bf16": jnp.bfloat16, "fp8": jnp.float8_e4m3fn, "fp4": jnp.uint8}[kv]
    serve = configs.ServingConfigs(
        num_seqs=num_seqs,
        page_size=128,
        total_q_tokens=tokens,
        num_page_indices=num_seqs * pages_per_seq,
        dtype_q=jnp.float8_e4m3fn if fp8_q else jnp.bfloat16,
        scale_q=1.0 if fp8_q else None,
        dtype_kv=dtype_kv,
        dtype_out=jnp.bfloat16,
        scale_k=1.0 if fp8 else None,
        scale_v=1.0 if fp8 else None,
        kv_layout=layout,
        **(serve_extra or {}),
    )
    decode, prefill = wrapper.calculate_block_sizes(
        model,
        serve,
        # The wrapper's default VMEM limit: the chip's capacity.
        pltpu.get_tpu_info().vmem_capacity_bytes,
        decode_query_size=decode_query_size,
        **kwargs,
    )
    return (
        (decode.bq_sz, decode.bq_c_sz, decode.bkv_sz, decode.batch_size),
        (prefill.bq_sz, prefill.bq_c_sz, prefill.bkv_sz, prefill.batch_size),
    )


def _qwen2_5_1_5b(kv, tokens):
    # 12 q heads, 2 KV heads, head_dim 128; 8 sequences of 16 pages.
    return dict(
        args=(12, 2, 128),
        kv=kv,
        layout=_HEAD,
        num_seqs=8,
        pages_per_seq=16,
        tokens=tokens,
        decode_query_size=1,
    )


def _qwen3_0_6b(kv, tokens):
    return dict(_qwen2_5_1_5b(kv, tokens), args=(16, 8, 128))


def _gemma4_31b_tp8(layer, tokens):
    # One TP8 device: 4 q heads; sliding layers have 2 KV heads of 256 and a
    # 1024-token window, global layers 1 (replicated) KV head of 512. fp8 KV,
    # head-major (HND) cache, MTP K=3, max_model_len 2048.
    sliding = layer == "sliding"
    return dict(
        args=(4, 2, 256) if sliding else (4, 1, 512),
        sliding_window=1024 if sliding else None,
        kv="fp8",
        layout=_SEQ,
        num_seqs=8,
        pages_per_seq=16,
        tokens=tokens,
        decode_query_size=4,
    )


# (layer, expected decode tiles, expected prefill tiles), as (bq, bq_c, bkv,
# batch).
_OTHER_MODELS = [
    *(
        pytest.param(
            _qwen2_5_1_5b("bf16", t),
            (1, 1, 1280, 8),
            (512, 32, 512, 2),
            id=f"qwen2.5-1.5b-bf16-{t}tok",
        )
        for t in (16, 64, 256)
    ),
    *(
        pytest.param(
            _qwen2_5_1_5b("fp8", t),
            (1, 1, 2048, 8),
            (512, 32, 512, 2),
            id=f"qwen2.5-1.5b-fp8-{t}tok",
        )
        for t in (16, 64, 256)
    ),
    *(
        pytest.param(
            _qwen3_0_6b("bf16", t),
            (1, 1, 256, 8),
            (256, 64, 256, 2),
            id=f"qwen3-0.6b-bf16-{t}tok",
        )
        for t in (16, 256)
    ),
    *(
        pytest.param(
            _qwen3_0_6b("fp8", t),
            (1, 1, 512, 8),
            (512, 32, 512, 2),
            id=f"qwen3-0.6b-fp8-{t}tok",
        )
        for t in (16, 256)
    ),
    *(
        pytest.param(
            _gemma4_31b_tp8("sliding", t),
            (4, 4, 1536, 8),
            (1024, 64, 1024, 2),
            id=f"gemma4-31b-tp8-sliding-{t}tok",
        )
        for t in (16, 512)
    ),
    *(
        pytest.param(
            _gemma4_31b_tp8("global", t),
            (4, 4, 1536, 8),
            (512, 64, 512, 2),
            id=f"gemma4-31b-tp8-global-{t}tok",
        )
        for t in (16, 512)
    ),
]


@pytest.mark.parametrize("layer, decode, prefill", _OTHER_MODELS)
def test_other_models_keep_their_tiles(layer, decode, prefill):
    layer = dict(layer)
    args = layer.pop("args")
    assert _tiles(*args, **layer) == (decode, prefill)


@pytest.mark.parametrize(
    "chip, layer, decode, prefill",
    [
        # fp8 queries: the fp8/bf16 throughput ratio of TPU7x (1.99) floors to 1.
        pytest.param(
            "TPU_7X",
            dict(_qwen3_0_6b("fp8", 256), q="fp8"),
            (1, 1, 512, 8),
            (256, 64, 256, 2),
            id="tpu7x-fp8-queries",
        ),
        # fp8 KV with bf16 queries on a chip whose ratio floors above 1 (8).
        pytest.param(
            "TPU_8I",
            _qwen2_5_1_5b("fp8", 1024),
            (1, 1, 2048, 8),
            (1280, 8, 1280, 2),
            id="tpu8i-fp8-kv",
        ),
    ],
)
def test_other_chips_keep_their_tiles(monkeypatch, chip, layer, decode, prefill):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion[chip], 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)
    layer = dict(layer)
    args = layer.pop("args")
    assert _tiles(*args, **layer) == (decode, prefill)


def _qwen3_5(shape, **serve):
    # Qwen3.5-397B-A17B: 32 q heads, 2 KV heads, head_dim 256; fp8 KV in the
    # head-major cache with 128-token kernel pages; max_model_len 262144.
    layer = dict(
        kv="fp8",
        layout=_SEQ,
        pages_per_seq=262144 // 128,
        tuned_tiles=True,
    )
    if shape == "tp1-prefill":
        # A PP=8 x TP1 prefill stage (MTP K=1).
        layer.update(args=(32, 2, 256), num_seqs=32, decode_query_size=2)
    elif shape == "tp8-decode":
        # One TP8 decode device (MTP K=3).
        layer.update(args=(4, 1, 256), num_seqs=16, tokens=64, decode_query_size=4)
    elif shape == "tp8-dcp4-cache-only":
        # DCP4: the cached-token pass attends with the queries of 4 devices.
        layer.update(
            args=(16, 1, 256),
            num_seqs=16,
            tokens=64,
            decode_query_size=4,
            serve_extra=dict(
                cp_group_size=4,
                return_lse=True,
                attention_scope=configs.AttentionScope.CACHE_ONLY,
            ),
        )
    elif shape == "tp8-dcp4-new-tokens":
        layer.update(
            args=(4, 1, 256),
            num_seqs=16,
            tokens=64,
            decode_query_size=4,
            serve_extra=dict(
                cp_group_size=4,
                return_lse=True,
                attention_scope=configs.AttentionScope.NEW_TOKENS_ONLY,
            ),
        )
    elif shape == "dp4tp2-decode":
        # One DP4 x TP2 decode device, with 4 decode lanes.
        layer.update(
            args=(16, 1, 256),
            num_seqs=32,
            tokens=128,
            decode_query_size=4,
            decode_batch_size=4,
        )
    layer.update(serve)
    return layer


_QWEN3_5 = [
    *(
        pytest.param(
            _qwen3_5("tp1-prefill", tokens=t),
            (2, 2, 1792, 8),
            # The tuned prefill table's entry.
            (256, 16, 1024, 1),
            id=f"tp1-prefill-{t}tok",
        )
        for t in (256, 2048, 8192)
    ),
    pytest.param(
        _qwen3_5("tp8-decode"), (4, 4, 4096, 8), (256, 128, 1024, 2), id="tp8-decode"
    ),
    pytest.param(
        _qwen3_5("tp8-dcp4-cache-only"),
        (4, 2, 2560, 8),
        (256, 64, 256, 2),
        id="tp8-dcp4-cache-only",
    ),
    pytest.param(
        _qwen3_5("tp8-dcp4-new-tokens"),
        (4, 4, 4096, 8),
        (256, 64, 1024, 2),
        id="tp8-dcp4-new-tokens",
    ),
    pytest.param(
        _qwen3_5("dp4tp2-decode"),
        (4, 4, 4096, 4),
        (256, 128, 256, 2),
        id="dp4tp2-decode",
    ),
]


@pytest.mark.parametrize("layer, decode, prefill", _QWEN3_5)
def test_tuned_tiles_at_the_qwen3_5_shapes(layer, decode, prefill):
    layer = dict(layer)
    args = layer.pop("args")
    assert _tiles(*args, **layer) == (decode, prefill)


def _correctness_test_shape(kv):
    # The call of test_ragged_paged_attention_tuned_tiles in
    # experimental/batched_rpa/correctness_test.py: 32 q / 2 KV heads of 256,
    # 8 sequence slots of 33 pages (4,100 tokens), 512 padded tokens, MTP K=3.
    return dict(
        args=(32, 2, 256),
        kv=kv,
        layout=_SEQ,
        num_seqs=8,
        pages_per_seq=33,
        tokens=512,
        decode_query_size=4,
        tuned_tiles=True,
    )


def test_a_packed_fp4_cache_takes_the_non_fp8_tuned_rules():
    """A packed fp4 KV cache is one byte per element, like fp8, but the tuned
    rules' fp8 branches need an fp8 cache: it takes the bf16 compute threshold
    (prefill bq_c as without tuned_tiles) and the 75% decode cap. With the fp8
    rules it would get 4,2,1792,8 and 256,128,256,1."""
    layer = _correctness_test_shape("fp4")
    args = layer.pop("args")
    assert _tiles(*args, **layer) == ((4, 2, 1536, 8), (256, 64, 256, 1))


def test_the_tpu_tuned_tiles_test_runs_different_tiles():
    """test_ragged_paged_attention_tuned_tiles runs its call with and without
    tuned_tiles on TPU7x, the chip of the TPU CI lane. Both tile sets must
    differ there, or its tuned case runs the default kernels again."""
    layer = _correctness_test_shape("fp8")
    args = layer.pop("args")
    default_decode, default_prefill = _tiles(*args, **{**layer, "tuned_tiles": False})
    tuned_decode, tuned_prefill = _tiles(*args, **layer)
    assert (default_decode, default_prefill) == ((4, 2, 1280, 8), (256, 64, 256, 1))
    assert tuned_decode != default_decode
    assert tuned_prefill != default_prefill


@pytest.mark.parametrize(
    "chip, decode, prefill",
    [
        ("TPU_V5E", (4, 2, 4096, 8), (128, 8, 1792, 2)),
        ("TPU_V5P", (4, 2, 3072, 8), (128, 16, 1024, 2)),
    ],
)
def test_a_chip_without_fp8_compute_keeps_the_bf16_threshold(
    monkeypatch, chip, decode, prefill
):
    """The tuned rules' fp8 threshold for an fp8 KV cache scales the bf16 one by
    the chip's fp8/bf16 throughput. A chip without fp8 compute
    (fp8_ops_per_second == 0) keeps the bf16 threshold: with a ratio of 0 the
    threshold would be 0 and every q tile would split into the smallest
    sub-chunks (prefill bq_c 2)."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion[chip], 1)
    assert info.fp8_ops_per_second == 0
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)
    layer = _qwen3_5("tp8-decode")
    args = layer.pop("args")
    assert _tiles(*args, **layer) == (decode, prefill)


def _tp8_decode_test_shape(**kwargs):
    # The call of test_ragged_paged_attention_tuned_tiles_at_a_tp8_decode_shape
    # in experimental/batched_rpa/correctness_test.py: 4 q / 1 KV head of 256,
    # 8 sequence slots of 40 pages (5,000 tokens), 512 padded tokens, MTP K=3.
    return dict(
        args=(4, 1, 256),
        kv="fp8",
        layout=_SEQ,
        num_seqs=8,
        pages_per_seq=40,
        tokens=512,
        decode_query_size=4,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs, decode, prefill",
    [
        (dict(), (4, 4, 3328, 8), (1024, 64, 1024, 2)),
        # The 85% decode VMEM cap reaches the 4096-token KV block, and the
        # prefill bq is clamped to the 512-token bucket.
        (dict(tuned_tiles=True), (4, 4, 4096, 8), (512, 128, 1024, 2)),
        # The DP4 x TP2 decode legs' 4 decode lanes.
        (
            dict(tuned_tiles=True, decode_batch_size=4),
            (4, 4, 4096, 4),
            (512, 128, 1024, 2),
        ),
    ],
    ids=["default", "tuned", "tuned-4-lanes"],
)
def test_the_tpu_tp8_decode_test_runs_the_tuned_rules_and_lane_counts(
    kwargs, decode, prefill
):
    """The TPU test's three cases run the tiles named here on TPU7x: the
    default tiles, the tuned ones (the decode VMEM cap and the prefill bq
    clamp both change a tile), and the tuned ones with 4 decode lanes."""
    layer = _tp8_decode_test_shape(**kwargs)
    args = layer.pop("args")
    assert _tiles(*args, **layer) == (decode, prefill)
