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

from typing import Any
from unittest.mock import patch

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import KVLayout
from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op


def _dummy_rank_rows(num_reqs: int = 4, num_rows: int = 1024) -> tp.RankRows:
    return tp.RankRows(
        seq_lens=jnp.zeros((num_reqs,), dtype=jnp.int32),
        cu_q_lens=jnp.zeros((num_reqs + 1,), dtype=jnp.int32),
        num_seqs=jnp.int32(1),
        first_req=jnp.int32(0),
        num_rows=jnp.int32(num_rows),
    )


def _invoke_select_topk(
    *,
    num_rows: int,
    natural_shape: tuple[int, int, int, int],
    kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE,
    chunk_tokens: int | None = 256,
    mixed_bkvp: int | None = 4,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_streamindex_topk(*args: Any, **kwargs: Any) -> jnp.ndarray:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return jnp.zeros((args[0].shape[0], kwargs["k"]), dtype=jnp.int32)

    select = dcp_token_parallel_op.make_select_topk(
        topk=2048,
        kv_layout=kv_layout,
        chunk_tokens=chunk_tokens,
        small_bucket=(4, None, 64),
        skip_empty_decode=True,
        skip_empty_decode_chunked=True,
        mixed_bkvp=mixed_bkvp,
        decode_pack=True,
        shared_meta=True,
    )
    q_r = jnp.zeros((num_rows, 32, 128), dtype=jnp.bfloat16)
    w_r = jnp.ones((num_rows, 32), dtype=jnp.bfloat16)
    natural = jnp.zeros(natural_shape, dtype=jnp.uint8)
    rr = _dummy_rank_rows(num_rows=num_rows)
    table = jnp.zeros((4 * natural_shape[0],), dtype=jnp.int32)

    with patch.object(
        dcp_token_parallel_op, "streamindex_topk", side_effect=fake_streamindex_topk
    ):
        out = select(q_r, w_r, natural, rr, table)

    assert out.shape == (num_rows, 2048)
    return captured


def test_should_keep_bkvp8_and_256_chunks_when_natural_kv_at_16k():
    call = _invoke_select_topk(
        num_rows=1024,
        natural_shape=(64, 33, 4, 256),
        kv_layout=KVLayout.SEQ_ALONG_LANE,
        chunk_tokens=256,
        mixed_bkvp=4,
    )
    kw = call["kwargs"]
    assert kw["num_kv_pages_per_block"] == (2, 2, 8)
    assert kw["chunk_tokens"] == 256
    assert kw["decode_as_mixed"] is True
    assert list(call["args"][6]) == [0, 0, 1]


def test_should_use_bkvp8_and_128_chunks_when_natural_kv_at_32k():
    call = _invoke_select_topk(
        num_rows=1024,
        natural_shape=(128, 33, 4, 256),
        kv_layout=KVLayout.SEQ_ALONG_LANE,
        chunk_tokens=256,
        mixed_bkvp=4,
    )
    kw = call["kwargs"]
    assert kw["num_kv_pages_per_block"] == (2, 2, 8)
    assert kw["chunk_tokens"] == 128
    assert kw["decode_as_mixed"] is True


@pytest.mark.parametrize(
    ("kv_layout", "natural_shape", "expected_bkvp"),
    [
        (KVLayout.SEQ_ALONG_LANE, (192, 33, 4, 256), 16),
        (KVLayout.SEQ_ALONG_LANE, (256, 33, 4, 256), 16),
        (KVLayout.SEQ_ALONG_LANE, (320, 33, 4, 256), 16),
        (KVLayout.HEAD_ALONG_SUBLANE, (320, 64, 4, 256), 16),
        # Non-multiple-of-16 page count must avoid KV padding by staying at bkvp=8.
        (KVLayout.SEQ_ALONG_LANE, (200, 33, 4, 256), 8),
    ],
)
def test_should_use_128_chunks_and_aligned_bkvp16_when_natural_kv_at_or_above_48k(
    kv_layout: KVLayout, natural_shape: tuple[int, int, int, int], expected_bkvp: int
):
    call = _invoke_select_topk(
        num_rows=1024,
        natural_shape=natural_shape,
        kv_layout=kv_layout,
        chunk_tokens=256,
        mixed_bkvp=4,
    )
    kw = call["kwargs"]
    assert kw["num_kv_pages_per_block"] == (2, 2, expected_bkvp)
    assert kw["chunk_tokens"] == 128
    assert kw["decode_as_mixed"] is True


def test_should_preserve_caller_knobs_when_outside_adaptive_regime():
    narrow = _invoke_select_topk(
        num_rows=1024,
        natural_shape=(4, 33, 4, 256),
        chunk_tokens=512,
        mixed_bkvp=4,
    )
    assert narrow["kwargs"]["num_kv_pages_per_block"] == (2, 2, 4)
    assert narrow["kwargs"]["chunk_tokens"] == 512

    explicit_wide = _invoke_select_topk(
        num_rows=128,
        natural_shape=(320, 33, 4, 256),
        chunk_tokens=256,
        mixed_bkvp=16,
    )
    assert explicit_wide["kwargs"]["num_kv_pages_per_block"] == (2, 2, 16)
    assert explicit_wide["kwargs"]["chunk_tokens"] == 256

    unset_bkvp = _invoke_select_topk(
        num_rows=1024,
        natural_shape=(320, 33, 4, 256),
        chunk_tokens=256,
        mixed_bkvp=None,
    )
    assert unset_bkvp["kwargs"]["num_kv_pages_per_block"] == (2, 2, 2)
    assert unset_bkvp["kwargs"]["chunk_tokens"] == 128
