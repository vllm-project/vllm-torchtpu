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
"""GLM pipeline-stage -> DCP byte-copy goldens.

The spans are replayed byte by byte through an emulation of the planner's
global-space page split and coverage rules, and every transferred token is
looked up where the decode's own DCP cache writer (`_get_dcp_page_and_slot`)
would have put it. No TPU runtime or transport is involved.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    lower_glm_dcp_row_spans,
)
from vllm_torchtpu.kernels.mla import kv_cache_utils as kvu

# Bytes per token in these goldens: the token id, then a per-tag salt.
_TOKEN_BYTES = 8
_FILLER = 0xEE


def _token_bytes(tokens, salt):
    tokens = np.asarray(tokens, dtype=np.uint32)
    out = np.empty((tokens.size, _TOKEN_BYTES), dtype=np.uint8)
    out[:, :4] = (tokens + 1).astype("<u4").view(np.uint8).reshape(-1, 4)
    out[:, 4:] = np.array([salt], dtype="<u4").view(np.uint8)
    return out


def _dcp_owner_slots(num_tokens, page_tokens, dcp, interleave, table, num_pages):
    """Per DCP rank, the (page, slot) the decode writer gives every token."""
    spec = kvu.SparseMLAKVCacheSpec.create(
        kvu.KVCacheType.NOPE, kvu.KVCacheLayout.TENSORCORE, num_pages, page_tokens, 512
    )
    seq_lens = jnp.array([num_tokens], jnp.int32)
    tables = jnp.array([table], jnp.int32)
    query_start_loc = jnp.array([0, num_tokens], jnp.int32)
    slots = []
    for rank in range(dcp):
        page, slot = kvu._get_dcp_page_and_slot(
            num_tokens,
            seq_lens,
            tables,
            query_start_loc,
            jnp.int32(rank),
            nope_spec=spec,
            dcp_size=dcp,
            interleave_size=interleave,
        )
        slots.append((np.asarray(page), np.asarray(slot)))
    return slots


def _replay(registration, *, dcp, live, dst_ids, src_pages, num_pages):
    """Apply one registration the way the planner does; return each unit's
    destination pages and check the planner's coverage rules."""
    dst = [np.zeros((num_pages, live), dtype=np.uint8) for _ in range(dcp)]
    coverage = [[[] for _ in dst_ids] for _ in range(dcp)]
    covered = 0
    assert registration.dst_space_version == 1
    for span in registration.spans:
        assert span.dst_block_index == 0
        assert span.count == 1 and not span.src_stride_bytes
        assert not span.dst_stride_bytes
        assert span.src_offset_bytes + span.size_bytes <= live
        if dcp == 1:
            assert span.dst_unit_ordinal is None
            units = [0]
        else:
            assert 0 <= span.dst_unit_ordinal < dcp
            units = [span.dst_unit_ordinal]
        src_page = src_pages[registration.block_ids[span.src_block_ordinal]]
        # SplitPlainGlobalSpan: cut at destination page boundaries.
        offset, src = span.dst_offset_bytes, span.src_offset_bytes
        remaining = span.size_bytes
        while remaining:
            index, in_page = divmod(offset, live)
            take = min(remaining, live - in_page)
            assert index < len(dst_ids), "span past the destination page set"
            for unit in units:
                coverage[unit][index].append((in_page, in_page + take))
                dst[unit][dst_ids[index], in_page : in_page + take] = src_page[
                    src : src + take
                ]
            covered += take
            offset, src, remaining = offset + take, src + take, remaining - take
    assert covered == registration.declared_bytes
    # ValidateSpanCoverage: a prefix per destination block, full except the
    # last, and the same extents on every destination unit.
    extents = None
    for unit in range(dcp):
        unit_extents = []
        for index, intervals in enumerate(coverage[unit]):
            assert intervals, f"block {index} uncovered on unit {unit}"
            until = 0
            for start, end in sorted(intervals):
                assert start == until, f"gap/overlap at {start} on unit {unit}"
                until = end
            if index != len(dst_ids) - 1:
                assert until == live
            unit_extents.append(until)
        extents = extents or unit_extents
        assert unit_extents == extents, f"unit {unit} extents differ"
    return dst


_CASES = [
    # (page_tokens, dcp, interleave)
    (1024, 16, 1024),
    (1024, 16, 256),
    (1024, 16, 64),
    (1024, 8, 64),
    (1024, 2, 512),
    (128, 4, 32),
    (128, 3, 32),
    (128, 1, 32),
    (64, 16, 4),
]


def _token_counts(page_tokens, dcp, interleave):
    cycle, block = dcp * interleave, dcp * page_tokens
    counts = {1, 2, interleave - 1, interleave, interleave + 1}
    counts |= {cycle - 1, cycle, cycle + 1, block - 1, block, block + 1}
    counts.add(2 * block + interleave + 3)
    return sorted(count for count in counts if count > 0)


@pytest.mark.parametrize(
    "page_tokens,dcp,interleave,num_tokens",
    [
        (page, dcp, interleave, tokens)
        for page, dcp, interleave in _CASES
        for tokens in _token_counts(page, dcp, interleave)
    ],
)
def test_dcp_spans_land_where_the_decode_writer_puts_each_token(
    page_tokens, dcp, interleave, num_tokens
):
    num_src_pages = math.ceil(num_tokens / page_tokens)
    num_dst_blocks = math.ceil(num_tokens / (page_tokens * dcp))
    num_pages = 2 * max(num_src_pages, num_dst_blocks) + 3
    # Deliberately non-contiguous, distinct physical ids on both sides.
    rng = np.random.default_rng(num_tokens)
    src_ids = [int(i) for i in rng.permutation(num_pages)[:num_src_pages]]
    dst_ids = [int(i) for i in rng.permutation(num_pages)[:num_dst_blocks]]
    owners = _dcp_owner_slots(
        num_tokens, page_tokens, dcp, interleave, dst_ids, num_pages
    )
    owned = sum(int((page != kvu.OOB_PAGE).sum()) for page, _ in owners)
    assert owned == num_tokens

    # Nope rows hold one token; rope and indexer rows pack four.
    for salt, tokens_per_row in enumerate((1, 4)):
        live = page_tokens * _TOKEN_BYTES
        src_pages = {}
        for ordinal, block_id in enumerate(src_ids):
            page = np.full((page_tokens, _TOKEN_BYTES), _FILLER, dtype=np.uint8)
            first = ordinal * page_tokens
            real = min(page_tokens, num_tokens - first)
            page[:real] = _token_bytes(range(first, first + real), salt)
            src_pages[block_id] = page.reshape(-1)
        registration = lower_glm_dcp_row_spans(
            tag=f"tag{salt}",
            num_tokens=num_tokens,
            page_tokens=page_tokens,
            live_bytes_per_block=live,
            row_bytes=tokens_per_row * _TOKEN_BYTES,
            block_ids=src_ids,
            dst_dcp_size=dcp,
            interleave_tokens=interleave,
        )
        assert registration.block_ids == tuple(src_ids)
        dst = _replay(
            registration,
            dcp=dcp,
            live=live,
            dst_ids=dst_ids,
            src_pages=src_pages,
            num_pages=num_pages,
        )
        for rank, (page, slot) in enumerate(owners):
            tokens = np.flatnonzero(page != kvu.OOB_PAGE)
            byte_index = slot[tokens, None] * _TOKEN_BYTES + np.arange(_TOKEN_BYTES)
            landed = dst[rank][page[tokens, None], byte_index]
            np.testing.assert_array_equal(landed, _token_bytes(tokens, salt))
