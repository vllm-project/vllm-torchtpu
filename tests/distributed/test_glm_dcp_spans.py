# SPDX-License-Identifier: Apache-2.0
"""GLM pipeline-stage -> DCP byte-copy goldens.

The spans are replayed byte by byte through an emulation of the planner's
global-space page split and coverage rules, and every transferred token is
looked up where the decode's own DCP cache writer (`_get_dcp_page_and_slot`)
would have put it. No TPU runtime or transport is involved.
"""

import math
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    lower_glm_dcp_row_spans,
)
from vllm_torchtpu.kernels.mla import kv_cache_utils as kvu

from .test_tpu_connector import _MOD, _make_raiden_worker

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


def _global_span_pieces(span, *, live, tag_skip=0):
    """SplitPlainGlobalSpan: (destination page index, in-page offset, source
    offset, size) pieces of one request-global span. The page-aligned
    dst_skip_bytes clip drops the bytes below ``tag_skip`` (their source is
    never read) and re-bases the rest, so index 0 is the first page past the
    cut."""
    assert tag_skip % live == 0, "dst_skip_bytes must be whole pages"
    clip = min(max(tag_skip - span.dst_offset_bytes, 0), span.size_bytes)
    offset = span.dst_offset_bytes + clip - tag_skip
    src = span.src_offset_bytes + clip
    remaining = span.size_bytes - clip
    while remaining:
        index, in_page = divmod(offset, live)
        take = min(remaining, live - in_page)
        yield index, in_page, src, take
        offset, src, remaining = offset + take, src + take, remaining - take


def _replay(
    registration, *, dcp, live, dst_ids, src_pages, num_pages, tag_skip=0, dst=None
):
    """Apply one registration the way the planner does; return each unit's
    destination pages and check the planner's coverage rules.

    ``tag_skip`` is the tag's dst_skip_bytes clip: ``dst_ids`` then lists only
    the destination pages past it, and ``dst`` holds the pages as they stand
    before the copy (zeros by default)."""
    if dst is None:
        dst = [np.zeros((num_pages, live), dtype=np.uint8) for _ in range(dcp)]
    coverage = [[[] for _ in dst_ids] for _ in range(dcp)]
    covered = clipped = extent = 0
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
        extent = max(extent, span.dst_offset_bytes + span.size_bytes)
        clipped += min(max(tag_skip - span.dst_offset_bytes, 0), span.size_bytes)
        src_page = src_pages[registration.block_ids[span.src_block_ordinal]]
        for index, in_page, src, take in _global_span_pieces(
            span, live=live, tag_skip=tag_skip
        ):
            assert index < len(dst_ids), "span past the destination page set"
            for unit in units:
                coverage[unit][index].append((in_page, in_page + take))
                dst[unit][dst_ids[index], in_page : in_page + take] = src_page[
                    src : src + take
                ]
            covered += take
    # The planner refuses a clip that removes the whole tag: a full local hit
    # releases the producer instead of planning.
    assert tag_skip < extent, "dst_skip_bytes removes the entire tag"
    assert covered + clipped == registration.declared_bytes
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


def _decode_tag_skips(tag_live, *, page_tokens, dcp, skip_tokens):
    """The dst_skip_bytes a GLM decode worker sends for a suffix pull past
    ``skip_tokens`` locally cached tokens, by tag."""
    worker = _make_raiden_worker(
        is_producer=False, tp_rank=0, tp_size=dcp, block_size=page_tokens
    )
    worker.vllm_config.parallel_config.decode_context_parallel_size = dcp
    # Only the live bytes (the first field) place a destination page.
    worker._stage3_row_geometry = {tag: (live, 0) for tag, live in tag_live.items()}
    with patch(f"{_MOD}._use_raiden_glm_admission", return_value=True):
        skips, logged = worker._stage3_cache_skip_bytes(list(tag_live), skip_tokens)
    assert logged == sum(skips)
    return dict(zip(tag_live, skips, strict=True))


def _suffix_cases(cases):
    """(page_tokens, dcp, interleave, num_tokens, prefix_blocks) with at
    least one cached scheduler block and one block left to pull."""
    out = []
    for page, dcp, interleave in cases:
        block = page * dcp
        for tokens in (block + 1, 2 * block, 2 * block + interleave + 3):
            num_blocks = math.ceil(tokens / block)
            for prefix in sorted({1, num_blocks - 1}):
                out.append((page, dcp, interleave, tokens, prefix))
    return out


_CACHED = 0xCA


@pytest.mark.parametrize(
    "page_tokens,dcp,interleave,num_tokens,prefix_blocks", _suffix_cases(_CASES)
)
def test_dcp_spans_suffix_pull_lands_past_a_cached_prefix(
    page_tokens, dcp, interleave, num_tokens, prefix_blocks
):
    """Decode prefix caching: the decode already holds the first
    prefix_blocks scheduler blocks, so it pulls only the rest, into its newly
    allocated pages, with the clip its worker computes."""
    block_tokens = page_tokens * dcp
    skip_tokens = prefix_blocks * block_tokens
    num_src_pages = math.ceil(num_tokens / page_tokens)
    num_dst_blocks = math.ceil(num_tokens / block_tokens)
    num_pages = 2 * max(num_src_pages, num_dst_blocks) + 3
    rng = np.random.default_rng(num_tokens + prefix_blocks)
    src_ids = [int(i) for i in rng.permutation(num_pages)[:num_src_pages]]
    # The decode's table: adopted cache pages, then the suffix it pulls into.
    dst_ids = [int(i) for i in rng.permutation(num_pages)[:num_dst_blocks]]
    cached_ids, suffix_ids = dst_ids[:prefix_blocks], dst_ids[prefix_blocks:]
    owners = _dcp_owner_slots(
        num_tokens, page_tokens, dcp, interleave, dst_ids, num_pages
    )
    live = page_tokens * _TOKEN_BYTES
    skips = _decode_tag_skips(
        {"tag0": live, "tag1": live},
        page_tokens=page_tokens,
        dcp=dcp,
        skip_tokens=skip_tokens,
    )
    assert skips == {"tag0": prefix_blocks * live, "tag1": prefix_blocks * live}

    for salt, tokens_per_row in enumerate((1, 4)):
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
        dst = [np.zeros((num_pages, live), dtype=np.uint8) for _ in range(dcp)]
        for pages in dst:
            pages[cached_ids] = _CACHED
        dst = _replay(
            registration,
            dcp=dcp,
            live=live,
            dst_ids=suffix_ids,
            src_pages=src_pages,
            num_pages=num_pages,
            tag_skip=skips[f"tag{salt}"],
            dst=dst,
        )
        for rank, (page, slot) in enumerate(owners):
            # The adopted cache pages are never transfer targets.
            assert (dst[rank][cached_ids] == _CACHED).all()
            tokens = np.flatnonzero(page != kvu.OOB_PAGE)
            tokens = tokens[tokens >= skip_tokens]
            assert np.isin(page[tokens], suffix_ids).all()
            byte_index = slot[tokens, None] * _TOKEN_BYTES + np.arange(_TOKEN_BYTES)
            landed = dst[rank][page[tokens, None], byte_index]
            np.testing.assert_array_equal(landed, _token_bytes(tokens, salt))
