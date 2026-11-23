# SPDX-License-Identifier: Apache-2.0
"""GLM pipeline stage -> DCP decode with a SEQ_ALONG_LANE indexer K cache on
both legs (TPU_RAIDEN_GLM_SAL_LANE_SPANS).

The lane spans are replayed byte by byte through an emulation of the
planner's strided global-space rebase and coverage rules, and every token's
lane-row words are looked up where the decode's own DCP cache writer
(`_get_dcp_page_and_slot`) puts the token. Also the page-agnostic
fingerprint, the indexer layout selection and the admission checks. No TPU
runtime or transport is involved.
"""

import math
from unittest.mock import patch

import numpy as np
import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import layout_fingerprint as rlf
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    GLM_LANE_WORD_BYTES,
    lower_glm_dcp_lane_spans,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    indexer_kv_layout,
)
from vllm_torchtpu.kernels.mla import kv_cache_utils as kvu

from .raiden_test_utils import glm_named_kv_caches
from .test_glm_dcp_spans import (
    _CACHED,
    _dcp_owner_slots,
    _decode_tag_skips,
    _token_counts,
)
from .test_tpu_connector import _MOD, _make_raiden_worker

# Lane rows of a GLM-5.2 indexer page: 128 fp8 dims in 32 words, then the
# scale word.
_ROWS = 33
_FILLER_WORD = 0xEEEEEEEE


def _expected_words(tokens):
    tokens = np.asarray(tokens, dtype=np.uint32)
    return (tokens[:, None] + 1) * 64 + np.arange(_ROWS, dtype=np.uint32)[None, :]


def _src_page(ordinal, page_tokens, num_tokens):
    """A source page's bytes: token t of lane row r at (r * page + t) words."""
    words = np.full((_ROWS, page_tokens), _FILLER_WORD, dtype=np.uint32)
    first = ordinal * page_tokens
    real = min(page_tokens, num_tokens - first)
    words[:, :real] = _expected_words(range(first, first + real)).T
    return words.astype("<u4").view(np.uint8).reshape(-1)


def _replay(
    registration,
    *,
    dcp,
    src_live,
    dst_live,
    dst_ids,
    num_pages,
    src_pages,
    tag_skip=0,
    dst=None,
):
    """Apply one registration the way the planner does (RebaseStridedGlobalSpan,
    then ValidateSpanCoverage); return each unit's destination pages.

    ``tag_skip`` is the tag's dst_skip_bytes clip: a strided span lies inside
    one page and the clip is whole pages, so a span below it is dropped whole
    (its source is never read) and the rest re-base past it. ``dst_ids`` then
    lists only the pages past the clip, and ``dst`` holds the pages as they
    stand before the copy (zeros by default)."""
    assert tag_skip % dst_live == 0, "dst_skip_bytes must be whole pages"
    if dst is None:
        dst = [np.zeros((num_pages, dst_live), dtype=np.uint8) for _ in range(dcp)]
    coverage = [[[] for _ in dst_ids] for _ in range(dcp)]
    covered = clipped = tag_end = 0
    assert registration.dst_space_version == 1
    for span in registration.spans:
        assert span.dst_block_index == 0
        assert span.count > 1
        assert span.src_stride_bytes > 0 and span.dst_stride_bytes > 0
        assert (
            span.src_offset_bytes
            + (span.count - 1) * span.src_stride_bytes
            + span.size_bytes
            <= src_live
        )
        index, in_page = divmod(span.dst_offset_bytes, dst_live)
        extent = (span.count - 1) * span.dst_stride_bytes + span.size_bytes
        assert in_page + extent <= dst_live, "strided span leaves its page"
        tag_end = max(tag_end, span.dst_offset_bytes + extent)
        if span.dst_offset_bytes < tag_skip:
            clipped += span.count * span.size_bytes
            continue
        index -= tag_skip // dst_live
        assert index < len(dst_ids), "span past the destination page set"
        if dcp == 1:
            assert span.dst_unit_ordinal is None
            units = [0]
        else:
            assert 0 <= span.dst_unit_ordinal < dcp
            units = [span.dst_unit_ordinal]
        src_page = src_pages[registration.block_ids[span.src_block_ordinal]]
        for repeat in range(span.count):
            src = span.src_offset_bytes + repeat * span.src_stride_bytes
            start = in_page + repeat * span.dst_stride_bytes
            end = start + span.size_bytes
            for unit in units:
                coverage[unit][index].append((start, end))
                dst[unit][dst_ids[index], start:end] = src_page[
                    src : src + span.size_bytes
                ]
            covered += span.size_bytes
    # The planner refuses a clip that removes the whole tag: a full local hit
    # releases the producer instead of planning.
    assert tag_skip < tag_end, "dst_skip_bytes removes the entire tag"
    assert covered + clipped == registration.declared_bytes
    # Every destination page is covered whole, once, on every unit.
    for unit in range(dcp):
        for index, intervals in enumerate(coverage[unit]):
            until = 0
            for start, end in sorted(intervals):
                assert start == until, f"gap/overlap at {start} on unit {unit}"
                until = end
            assert until == dst_live, f"block {index} not covered on unit {unit}"
    return dst


_CASES = [
    # (source page, dcp, interleave = destination page)
    (1024, 16, 256),
    (1024, 8, 256),
    (256, 16, 256),
    (1024, 4, 128),
    (512, 2, 512),
    (256, 1, 128),
]


@pytest.mark.parametrize(
    "page_tokens,dcp,interleave,num_tokens",
    [
        (page, dcp, interleave, tokens)
        for page, dcp, interleave in _CASES
        for tokens in _token_counts(page, dcp, interleave)
    ],
)
def test_lane_spans_land_where_the_decode_writer_puts_each_token(
    page_tokens, dcp, interleave, num_tokens
):
    num_src_pages = math.ceil(num_tokens / page_tokens)
    num_dst_blocks = math.ceil(num_tokens / (interleave * dcp))
    num_pages = 2 * max(num_src_pages, num_dst_blocks) + 3
    # Deliberately non-contiguous, distinct physical ids on both sides.
    rng = np.random.default_rng(num_tokens)
    src_ids = [int(i) for i in rng.permutation(num_pages)[:num_src_pages]]
    dst_ids = [int(i) for i in rng.permutation(num_pages)[:num_dst_blocks]]
    # The decode pages at its interleave: one chunk per page.
    owners = _dcp_owner_slots(
        num_tokens, interleave, dcp, interleave, dst_ids, num_pages
    )
    src_live = _ROWS * page_tokens * GLM_LANE_WORD_BYTES
    dst_live = _ROWS * interleave * GLM_LANE_WORD_BYTES
    src_pages = {
        block_id: _src_page(ordinal, page_tokens, num_tokens)
        for ordinal, block_id in enumerate(src_ids)
    }
    registration = lower_glm_dcp_lane_spans(
        tag="dsa.idx.l0",
        num_tokens=num_tokens,
        page_tokens=page_tokens,
        live_bytes_per_block=src_live,
        block_ids=src_ids,
        dst_dcp_size=dcp,
        interleave_tokens=interleave,
    )
    assert registration.block_ids == tuple(src_ids)
    dst = _replay(
        registration,
        dcp=dcp,
        src_live=src_live,
        dst_live=dst_live,
        dst_ids=dst_ids,
        num_pages=num_pages,
        src_pages=src_pages,
    )
    landed_tokens = 0
    for rank, (page, slot) in enumerate(owners):
        tokens = np.flatnonzero(page != kvu.OOB_PAGE)
        words = dst[rank].view("<u4").reshape(num_pages, _ROWS, interleave)
        landed = words[page[tokens], :, slot[tokens]]
        np.testing.assert_array_equal(landed, _expected_words(tokens))
        landed_tokens += tokens.size
    assert landed_tokens == num_tokens


def _suffix_cases():
    """(source page, dcp, interleave, num_tokens, prefix_blocks) with at least
    one cached scheduler block (interleave * dcp tokens) and one left to pull."""
    out = []
    for page, dcp, interleave in _CASES:
        block = interleave * dcp
        for tokens in (block + 1, 2 * block, 2 * block + interleave + 3):
            num_blocks = math.ceil(tokens / block)
            for prefix in sorted({1, num_blocks - 1}):
                out.append((page, dcp, interleave, tokens, prefix))
    return out


@pytest.mark.parametrize(
    "page_tokens,dcp,interleave,num_tokens,prefix_blocks", _suffix_cases()
)
def test_lane_spans_suffix_pull_lands_past_a_cached_prefix(
    page_tokens, dcp, interleave, num_tokens, prefix_blocks
):
    """Decode prefix caching over the lane spans: the decode already holds the
    first prefix_blocks scheduler blocks, so it pulls only the rest, into its
    newly allocated pages, with the clip its worker computes from the indexer
    page bytes."""
    block_tokens = interleave * dcp
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
        num_tokens, interleave, dcp, interleave, dst_ids, num_pages
    )
    src_live = _ROWS * page_tokens * GLM_LANE_WORD_BYTES
    dst_live = _ROWS * interleave * GLM_LANE_WORD_BYTES
    skips = _decode_tag_skips(
        {"dsa.idx.l0": dst_live},
        page_tokens=interleave,
        dcp=dcp,
        skip_tokens=skip_tokens,
    )
    assert skips == {"dsa.idx.l0": prefix_blocks * dst_live}
    src_pages = {
        block_id: _src_page(ordinal, page_tokens, num_tokens)
        for ordinal, block_id in enumerate(src_ids)
    }
    registration = lower_glm_dcp_lane_spans(
        tag="dsa.idx.l0",
        num_tokens=num_tokens,
        page_tokens=page_tokens,
        live_bytes_per_block=src_live,
        block_ids=src_ids,
        dst_dcp_size=dcp,
        interleave_tokens=interleave,
    )
    dst = [np.zeros((num_pages, dst_live), dtype=np.uint8) for _ in range(dcp)]
    for pages in dst:
        pages[cached_ids] = _CACHED
    dst = _replay(
        registration,
        dcp=dcp,
        src_live=src_live,
        dst_live=dst_live,
        dst_ids=suffix_ids,
        num_pages=num_pages,
        src_pages=src_pages,
        tag_skip=skips["dsa.idx.l0"],
        dst=dst,
    )
    for rank, (page, slot) in enumerate(owners):
        # The adopted cache pages are never transfer targets.
        assert (dst[rank][cached_ids] == _CACHED).all()
        tokens = np.flatnonzero(page != kvu.OOB_PAGE)
        tokens = tokens[tokens >= skip_tokens]
        assert np.isin(page[tokens], suffix_ids).all()
        words = dst[rank].view("<u4").reshape(num_pages, _ROWS, interleave)
        landed = words[page[tokens], :, slot[tokens]]
        np.testing.assert_array_equal(landed, _expected_words(tokens))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(tag=""), "tag must not be empty"),
        (dict(num_tokens=0), "num_tokens must be positive"),
        (dict(dst_dcp_size=0), "dst_dcp_size must be positive"),
        (dict(live_bytes_per_block=_ROWS * 1024 * 4 + 4), "whole lane rows"),
        (dict(interleave_tokens=384), "must divide the page"),
        (dict(block_ids=[7]), "complete request page set"),
    ],
)
def test_lane_spans_reject_bad_geometry(overrides, message):
    kwargs = dict(
        tag="dsa.idx.l0",
        num_tokens=1500,
        page_tokens=1024,
        live_bytes_per_block=_ROWS * 1024 * GLM_LANE_WORD_BYTES,
        block_ids=[7, 9],
        dst_dcp_size=16,
        interleave_tokens=256,
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        lower_glm_dcp_lane_spans(**kwargs)


# ---------------------------------------------------------------------------
# Page-agnostic fingerprint.
# ---------------------------------------------------------------------------


def _layout_getter(tensor):
    packing = tensor.shape[2]
    return ([3, 2, 1, 0], [[packing, 128], [packing, 1]], 0)


def _versions(package):
    return {"torch_tpu": "torch-tpu-test", "libtpu": "libtpu-test"}[package]


def _fingerprint(block, *, sal=True, layers=(0, 1), **kwargs):
    named = glm_named_kv_caches(
        block_size_tokens=block,
        idx_shape=(16, _ROWS, 4, block) if sal else None,
    )
    named = {
        name: cache
        for name, cache in named.items()
        if rpm.layer_index_from_name(name) in layers
    }
    manifest = rpm.build_glm_mla_pool_manifest(
        named_kv_caches=named,
        raw_tensors=(),
        block_size_tokens=block,
        per_layer_tags=True,
        indexer_seq_along_lane=sal,
    )
    return rlf.measured_glm_layout_fingerprint(
        manifest,
        page_tokens=block,
        layout_getter=_layout_getter,
        package_version=_versions,
        indexer_seq_along_lane=sal,
        **kwargs,
    )


@pytest.mark.parametrize("hash_indexer", [False, True])
def test_lane_span_fingerprint_matches_a_stage_page_to_the_decode_page(hash_indexer):
    stage, stage_payload = _fingerprint(
        1024, hash_indexer=hash_indexer, page_agnostic=True
    )
    decode, _ = _fingerprint(256, hash_indexer=hash_indexer, page_agnostic=True)
    assert stage == decode
    assert "page_tokens" not in stage_payload
    assert stage_payload["indexer_layout"] == "sal"
    if hash_indexer:
        assert stage_payload["layouts"]["dsa.idx"]["lane_rows"] == [_ROWS, 4]
    # A stage without an indexer layer matches when the class is unhashed.
    if not hash_indexer:
        assert (
            _fingerprint(1024, layers=(1,), page_agnostic=True, hash_indexer=False)[0]
            == decode
        )
    # Page-specific hashing keeps the two pages apart.
    assert (
        _fingerprint(1024, hash_indexer=hash_indexer)[0]
        != _fingerprint(256, hash_indexer=hash_indexer)[0]
    )


def test_lane_span_fingerprint_tells_the_indexer_layouts_apart():
    # The indexer class is out of the hash under per-layer tags; the layout
    # name alone keeps a packed-row peer from pairing with a lane one.
    sal, sal_payload = _fingerprint(256, hash_indexer=False, page_agnostic=True)
    has, has_payload = _fingerprint(
        256, sal=False, hash_indexer=False, page_agnostic=True
    )
    assert sal_payload["layouts"] == has_payload["layouts"]
    assert sal != has


# ---------------------------------------------------------------------------
# Indexer layout selection.
# ---------------------------------------------------------------------------


def _raiden_glm_env(monkeypatch, *, identity=False, lane_spans=False):
    monkeypatch.setenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER", "1")
    monkeypatch.setenv("TPU_RAIDEN_GLM_ADMISSION", "1")
    monkeypatch.setenv("TPU_RAIDEN_GLM_DCP_IDENTITY", "1" if identity else "0")
    monkeypatch.setenv("TPU_RAIDEN_GLM_SAL_LANE_SPANS", "1" if lane_spans else "0")


def test_lane_spans_admit_sal_under_raiden_glm_admission(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    _raiden_glm_env(monkeypatch)
    with pytest.raises(ValueError, match="incompatible with Raiden GLM admission"):
        indexer_kv_layout(1024)
    _raiden_glm_env(monkeypatch, lane_spans=True)
    assert indexer_kv_layout(1024) == KVLayout.SEQ_ALONG_LANE
    assert indexer_kv_layout(256) == KVLayout.SEQ_ALONG_LANE
    monkeypatch.delenv("VLLM_TPU_INDEXER_KV_CACHE_MODE")
    assert indexer_kv_layout(256) == KVLayout.SEQ_ALONG_LANE


# ---------------------------------------------------------------------------
# Admission.
# ---------------------------------------------------------------------------


def _stage_worker(dst_dcp=16, dst_interleave=256):
    worker = _make_raiden_worker(tp_rank=0, tp_size=1, pp_size=8, block_size=1024)
    worker.vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "destination_decode_context_parallel_size": dst_dcp,
        "destination_cp_kv_cache_interleave_size": dst_interleave,
    }
    return worker


def _decode_worker(*, tp_size=16, dcp=16, interleave=256, block=256):
    worker = _make_raiden_worker(
        tp_rank=3 if tp_size > 1 else 0, tp_size=tp_size, is_producer=False
    )
    worker.vllm_config.parallel_config.decode_context_parallel_size = dcp
    worker.vllm_config.parallel_config.cp_kv_cache_interleave_size = interleave
    worker.vllm_config.cache_config.block_size = block
    return worker


def _topology(worker, *, dst_shards=16):
    with (
        patch(f"{_MOD}._raiden_glm_dcp_identity", return_value=False),
        patch(f"{_MOD}._glm_indexer_seq_along_lane", return_value=True),
        patch(f"{_MOD}._raiden_dst_shards", return_value=dst_shards),
        patch("vllm_torchtpu.distributed.dcp.get_dcp_rank", return_value=3),
    ):
        return worker._raiden_glm_admission_topology()


def test_lane_span_topology_admits_a_stage_and_a_decode_paging_at_its_interleave():
    assert _topology(_stage_worker()) == "tp1_prefill"
    assert _topology(_decode_worker()) == "tp16dcp16_decode"


def test_lane_span_stage_needs_a_dcp_destination():
    with pytest.raises(ValueError, match="to a DCP destination only"):
        _topology(_stage_worker(dst_dcp=1))


@pytest.mark.parametrize(
    ("decode_kwargs", "message"),
    [
        (dict(interleave=4), "equal its block size"),
        (dict(interleave=256, block=1024), "equal its block size"),
        (dict(tp_size=1, dcp=1), "into a TP/DCP decode only"),
    ],
)
def test_lane_span_decode_fails_closed(decode_kwargs, message):
    with pytest.raises(ValueError, match=message):
        _topology(_decode_worker(**decode_kwargs))
