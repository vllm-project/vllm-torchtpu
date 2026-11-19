# SPDX-License-Identifier: Apache-2.0
"""GLM DCP producer -> DCP decode (TPU_RAIDEN_GLM_DCP_IDENTITY).

Producer rank k ships its whole pages to decode rank k. The spans are replayed
through the planner's page split and coverage rules, every token is placed on
the producer by the same DCP cache writer (`_get_dcp_page_and_slot`) the decode
uses, and must land where that writer puts it on the decode. No TPU runtime or
transport is involved.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer.raiden import layout_fingerprint as rlf
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    lower_glm_dcp_identity_spans,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    indexer_kv_layout,
)
from vllm_torchtpu.kernels.mla import kv_cache_utils as kvu

from .raiden_test_utils import glm_named_kv_caches
from .test_glm_dcp_spans import (
    _FILLER,
    _TOKEN_BYTES,
    _dcp_owner_slots,
    _decode_tag_skips,
    _global_span_pieces,
    _suffix_cases,
    _token_bytes,
)
from .test_tpu_connector import _MOD, _make_raiden_scheduler, _make_raiden_worker

# ---------------------------------------------------------------------------
# Byte goldens.
# ---------------------------------------------------------------------------

_CASES = [
    # (page_tokens, dcp, interleave)
    (256, 16, 4),
    (256, 16, 256),
    (1024, 16, 1024),
    (128, 4, 32),
    (64, 2, 4),
]


def _token_counts(page_tokens, dcp, interleave):
    cycle, block = dcp * interleave, dcp * page_tokens
    counts = {1, 2, interleave + 1, cycle - 1, cycle, cycle + 1}
    counts |= {block - 1, block, block + 1, 2 * block + interleave + 3}
    return sorted(count for count in counts if count > 0)


def _writer_pages(owners, num_pages, live):
    """Per DCP rank, its cache pages as its own writer fills them."""
    ranks = []
    for page, slot in owners:
        pages = np.full((num_pages, live), _FILLER, dtype=np.uint8)
        tokens = np.flatnonzero(page != kvu.OOB_PAGE)
        byte_index = slot[tokens, None] * _TOKEN_BYTES + np.arange(_TOKEN_BYTES)
        pages[page[tokens, None], byte_index] = _token_bytes(tokens, 0)
        ranks.append(pages)
    return ranks


@pytest.mark.parametrize(
    "page_tokens,dcp,interleave,num_tokens",
    [
        (page, dcp, interleave, tokens)
        for page, dcp, interleave in _CASES
        for tokens in _token_counts(page, dcp, interleave)
    ],
)
def test_identity_spans_land_where_the_decode_writer_puts_each_token(
    page_tokens, dcp, interleave, num_tokens
):
    num_blocks = math.ceil(num_tokens / (page_tokens * dcp))
    num_pages = 2 * num_blocks + 3
    # Deliberately non-contiguous, distinct physical ids on both sides.
    rng = np.random.default_rng(num_tokens)
    src_ids = [int(i) for i in rng.permutation(num_pages)[:num_blocks]]
    dst_ids = [int(i) for i in rng.permutation(num_pages)[:num_blocks]]
    live = page_tokens * _TOKEN_BYTES
    src = _writer_pages(
        _dcp_owner_slots(num_tokens, page_tokens, dcp, interleave, src_ids, num_pages),
        num_pages,
        live,
    )

    dst = [np.zeros((num_pages, live), dtype=np.uint8) for _ in range(dcp)]
    coverage = [[[] for _ in dst_ids] for _ in range(dcp)]
    for rank in range(dcp):
        registration = lower_glm_dcp_identity_spans(
            tag="mla.nope",
            num_tokens=num_tokens,
            page_tokens=page_tokens,
            dcp_size=dcp,
            dcp_rank=rank,
            live_bytes_per_block=live,
            block_ids=src_ids,
        )
        assert registration.dst_space_version == 1
        assert registration.block_ids == tuple(src_ids)
        assert registration.declared_bytes == num_blocks * live
        for span in registration.spans:
            assert span.dst_unit_ordinal == rank
            assert span.count == 1 and span.dst_block_index == 0
            # SplitPlainGlobalSpan: a whole page never crosses a boundary.
            index, in_page = divmod(span.dst_offset_bytes, live)
            assert in_page + span.size_bytes <= live
            assert index < len(dst_ids), "span past the destination page set"
            src_page = src[rank][registration.block_ids[span.src_block_ordinal]]
            dst[rank][dst_ids[index], in_page : in_page + span.size_bytes] = src_page[
                span.src_offset_bytes : span.src_offset_bytes + span.size_bytes
            ]
            coverage[rank][index].append((in_page, in_page + span.size_bytes))

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

    owners = _dcp_owner_slots(
        num_tokens, page_tokens, dcp, interleave, dst_ids, num_pages
    )
    for rank, (page, slot) in enumerate(owners):
        tokens = np.flatnonzero(page != kvu.OOB_PAGE)
        byte_index = slot[tokens, None] * _TOKEN_BYTES + np.arange(_TOKEN_BYTES)
        landed = dst[rank][page[tokens, None], byte_index]
        np.testing.assert_array_equal(landed, _token_bytes(tokens, 0))


@pytest.mark.parametrize(
    "page_tokens,dcp,interleave,num_tokens,prefix_blocks", _suffix_cases(_CASES)
)
def test_identity_suffix_pull_lands_past_a_cached_prefix(
    page_tokens, dcp, interleave, num_tokens, prefix_blocks
):
    """Decode prefix caching over the identity transfer: the decode already
    holds the first prefix_blocks scheduler blocks and pulls only the rest,
    into its newly allocated pages, with the per-tag clip its worker
    computes. The rope pool's pages are a quarter of the nope pool's, so one
    shared clip would misplace one of them."""
    num_blocks = math.ceil(num_tokens / (page_tokens * dcp))
    num_pages = 2 * num_blocks + 3
    rng = np.random.default_rng(num_tokens + prefix_blocks)
    src_ids = [int(i) for i in rng.permutation(num_pages)[:num_blocks]]
    # The decode's table: adopted cache pages, then the suffix it pulls into.
    dst_ids = [int(i) for i in rng.permutation(num_pages)[:num_blocks]]
    cached_ids, suffix_ids = dst_ids[:prefix_blocks], dst_ids[prefix_blocks:]
    live = {"mla.nope": page_tokens * _TOKEN_BYTES, "mla.rope": page_tokens * 2}
    skips = _decode_tag_skips(
        live,
        page_tokens=page_tokens,
        dcp=dcp,
        skip_tokens=prefix_blocks * page_tokens * dcp,
    )
    assert skips == {tag: prefix_blocks * size for tag, size in live.items()}
    # Producer pages: the nope pool as its DCP writer fills it, the rope pool
    # opaque (identity ships whole pages, whatever their layout).
    src = {
        "mla.nope": _writer_pages(
            _dcp_owner_slots(
                num_tokens, page_tokens, dcp, interleave, src_ids, num_pages
            ),
            num_pages,
            live["mla.nope"],
        ),
        "mla.rope": [
            rng.integers(0, 256, (num_pages, live["mla.rope"]), dtype=np.uint8)
            for _ in range(dcp)
        ],
    }
    for tag, size in live.items():
        for rank in range(dcp):
            # The decode's cache holds the prefix as the producer computed it.
            dst = np.zeros((num_pages, size), dtype=np.uint8)
            dst[cached_ids] = src[tag][rank][src_ids[:prefix_blocks]]
            coverage = [[] for _ in suffix_ids]
            registration = lower_glm_dcp_identity_spans(
                tag=tag,
                num_tokens=num_tokens,
                page_tokens=page_tokens,
                dcp_size=dcp,
                dcp_rank=rank,
                live_bytes_per_block=size,
                block_ids=src_ids,
            )
            for span in registration.spans:
                src_page = src[tag][rank][
                    registration.block_ids[span.src_block_ordinal]
                ]
                for index, in_page, offset, take in _global_span_pieces(
                    span, live=size, tag_skip=skips[tag]
                ):
                    assert index < len(suffix_ids), "span past the suffix pages"
                    dst[suffix_ids[index], in_page : in_page + take] = src_page[
                        offset : offset + take
                    ]
                    coverage[index].append((in_page, in_page + take))
            # Every suffix page arrives whole, exactly once.
            for index, intervals in enumerate(coverage):
                assert sorted(intervals) == [(0, size)], f"{tag} page {index}"
            np.testing.assert_array_equal(dst[dst_ids], src[tag][rank][src_ids])
            untouched = np.setdiff1d(np.arange(num_pages), dst_ids)
            assert not dst[untouched].any()

    # Every token, cached or pulled, sits where the decode's writer reads it.
    nope = live["mla.nope"]
    owners = _dcp_owner_slots(
        num_tokens, page_tokens, dcp, interleave, dst_ids, num_pages
    )
    for rank, (page, slot) in enumerate(owners):
        dst = np.zeros((num_pages, nope), dtype=np.uint8)
        dst[dst_ids] = src["mla.nope"][rank][src_ids]
        tokens = np.flatnonzero(page != kvu.OOB_PAGE)
        byte_index = slot[tokens, None] * _TOKEN_BYTES + np.arange(_TOKEN_BYTES)
        landed = dst[page[tokens, None], byte_index]
        np.testing.assert_array_equal(landed, _token_bytes(tokens, 0))


@pytest.mark.parametrize("num_tokens", [1, 2047, 2048, 2049, 5000, 9999])
def test_identity_spans_address_tp_ranks_when_tp_exceeds_dcp(num_tokens):
    # TP16 / DCP8: TP rank r holds DCP shard r % 8, and ships it whole to the
    # destination's TP rank r, which holds the same shard.
    kwargs = dict(
        tag="mla.nope",
        num_tokens=num_tokens,
        page_tokens=256,
        dcp_size=8,
        live_bytes_per_block=512 * 256,
        block_ids=list(range(math.ceil(num_tokens / (256 * 8)))),
    )
    for tp_rank in range(16):
        shard = lower_glm_dcp_identity_spans(dcp_rank=tp_rank % 8, **kwargs)
        unit = lower_glm_dcp_identity_spans(
            dcp_rank=tp_rank % 8, dst_unit=tp_rank, **kwargs
        )
        assert unit.declared_bytes == shard.declared_bytes
        assert len(unit.spans) == len(shard.spans) == len(kwargs["block_ids"])
        for got, want in zip(unit.spans, shard.spans, strict=True):
            assert got.dst_unit_ordinal == tp_rank
            assert (got.src_block_ordinal, got.dst_offset_bytes, got.size_bytes) == (
                want.src_block_ordinal,
                want.dst_offset_bytes,
                want.size_bytes,
            )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # 5000 tokens over 256-token pages at DCP16 are two scheduler blocks.
        (dict(block_ids=[7]), "expected 2"),
        (dict(dcp_rank=16), "outside dcp_size"),
        (dict(num_tokens=0), "num_tokens must be positive"),
        # A destination unit must hold the shard it receives.
        (dict(dst_unit=20), "does not hold DCP shard"),
        (dict(dst_unit=-1), "does not hold DCP shard"),
    ],
)
def test_identity_lowering_fails_closed(overrides, message):
    kwargs = dict(
        tag="dsa.idx",
        num_tokens=5000,
        page_tokens=256,
        dcp_size=16,
        dcp_rank=3,
        live_bytes_per_block=33 * 4 * 256,
        block_ids=[7, 9],
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        lower_glm_dcp_identity_spans(**kwargs)


# ---------------------------------------------------------------------------
# SEQ_ALONG_LANE indexer cache: manifest, fingerprint, layout selection.
# ---------------------------------------------------------------------------

_SAL_BLOCK = 256


def _sal_manifest(idx_shape=(16, 33, 4, _SAL_BLOCK), **kwargs):
    return rpm.build_glm_mla_pool_manifest(
        named_kv_caches=glm_named_kv_caches(
            block_size_tokens=_SAL_BLOCK, idx_shape=idx_shape
        ),
        raw_tensors=(),
        block_size_tokens=_SAL_BLOCK,
        **kwargs,
    )


def test_glm_manifest_admits_seq_along_lane_indexer_as_whole_pages():
    sal = _sal_manifest(indexer_seq_along_lane=True)
    has = rpm.build_glm_mla_pool_manifest(
        named_kv_caches=glm_named_kv_caches(block_size_tokens=_SAL_BLOCK),
        raw_tensors=(),
        block_size_tokens=_SAL_BLOCK,
    )

    (pool,) = sal.pools_of_class(rpm.TAG_DSA_IDX)
    (region,) = pool.regions
    page_bytes = 33 * 4 * _SAL_BLOCK
    assert pool.block_stride_bytes == page_bytes
    assert (region.unit_bytes, region.num_units) == (page_bytes, 1)
    assert sal.geometry_by_tag()[rpm.TAG_DSA_IDX]["live_bytes_per_block"] == (
        page_bytes
    )
    # The MLA classes keep their packed-row geometry.
    for tag in (rpm.TAG_MLA_NOPE, rpm.TAG_MLA_ROPE):
        assert [p.regions for p in sal.pools_of_class(tag)] == [
            p.regions for p in has.pools_of_class(tag)
        ]


def test_glm_manifest_rejects_seq_along_lane_indexer_without_opt_in():
    with pytest.raises(rpm.ManifestError, match="KV block size"):
        _sal_manifest()


def test_glm_manifest_rejects_seq_along_lane_width_off_the_page():
    with pytest.raises(rpm.ManifestError, match="SEQ_ALONG_LANE width 512"):
        _sal_manifest(idx_shape=(16, 33, 4, 512), indexer_seq_along_lane=True)


def _layout_getter(tensor):
    packing = tensor.shape[2]
    return ([3, 2, 1, 0], [[packing, 128], [packing, 1]], 0)


def _versions(package):
    return {"torch_tpu": "torch-tpu-test", "libtpu": "libtpu-test"}[package]


def test_glm_fingerprint_tells_the_indexer_layouts_apart():
    sal_fp, sal_payload = rlf.measured_glm_layout_fingerprint(
        _sal_manifest(indexer_seq_along_lane=True),
        page_tokens=_SAL_BLOCK,
        layout_getter=_layout_getter,
        package_version=_versions,
        indexer_seq_along_lane=True,
    )
    has_fp, has_payload = rlf.measured_glm_layout_fingerprint(
        rpm.build_glm_mla_pool_manifest(
            named_kv_caches=glm_named_kv_caches(block_size_tokens=_SAL_BLOCK),
            raw_tensors=(),
            block_size_tokens=_SAL_BLOCK,
        ),
        page_tokens=_SAL_BLOCK,
        layout_getter=_layout_getter,
        package_version=_versions,
    )
    assert sal_payload["layouts"]["dsa.idx"]["page_shape"] == [33, 4, _SAL_BLOCK]
    assert has_payload["layouts"]["dsa.idx"]["page_shape"] == [64, 4, _SAL_BLOCK]
    assert sal_payload["layouts"]["mla.nope"] == has_payload["layouts"]["mla.nope"]
    assert sal_fp != has_fp


def test_glm_fingerprint_rejects_seq_along_lane_indexer_without_opt_in():
    with pytest.raises(RuntimeError, match="page geometry"):
        rlf.measured_glm_layout_fingerprint(
            _sal_manifest(indexer_seq_along_lane=True),
            page_tokens=_SAL_BLOCK,
            layout_getter=_layout_getter,
            package_version=_versions,
        )


def _raiden_glm_env(monkeypatch, identity):
    monkeypatch.setenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER", "1")
    monkeypatch.setenv("TPU_RAIDEN_GLM_ADMISSION", "1")
    monkeypatch.setenv("TPU_RAIDEN_GLM_DCP_IDENTITY", "1" if identity else "0")


@pytest.mark.parametrize(
    ("identity", "expected"),
    [(False, KVLayout.HEAD_ALONG_SUBLANE), (True, KVLayout.SEQ_ALONG_LANE)],
)
def test_default_indexer_layout_under_raiden_glm_admission(
    monkeypatch, identity, expected
):
    _raiden_glm_env(monkeypatch, identity)
    monkeypatch.delenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", raising=False)
    assert indexer_kv_layout(_SAL_BLOCK) == expected


def test_explicit_sal_needs_dcp_identity_under_raiden_glm_admission(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    _raiden_glm_env(monkeypatch, identity=False)
    with pytest.raises(ValueError, match="incompatible with Raiden GLM admission"):
        indexer_kv_layout(_SAL_BLOCK)
    _raiden_glm_env(monkeypatch, identity=True)
    assert indexer_kv_layout(_SAL_BLOCK) == KVLayout.SEQ_ALONG_LANE


# ---------------------------------------------------------------------------
# Admission topology.
# ---------------------------------------------------------------------------


def _dcp_worker(
    *,
    is_producer,
    tp_rank=3,
    tp_size=16,
    dcp=16,
    interleave=4,
    dst_dcp=16,
    dst_interleave=4,
):
    worker = _make_raiden_worker(
        tp_rank=tp_rank, tp_size=tp_size, is_producer=is_producer
    )
    parallel = worker.vllm_config.parallel_config
    parallel.decode_context_parallel_size = dcp
    parallel.cp_kv_cache_interleave_size = interleave
    worker.vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "destination_decode_context_parallel_size": dst_dcp,
        "destination_cp_kv_cache_interleave_size": dst_interleave,
    }
    return worker


def _topology(worker, *, identity=True, dcp_rank=3, dst_shards=16):
    with (
        patch(f"{_MOD}._raiden_glm_dcp_identity", return_value=identity),
        patch(f"{_MOD}._raiden_dst_shards", return_value=dst_shards),
        patch("vllm_torchtpu.distributed.dcp.get_dcp_rank", return_value=dcp_rank),
    ):
        return worker._raiden_glm_admission_topology()


def test_identity_topology_admits_matching_dcp_peers():
    assert _topology(_dcp_worker(is_producer=True)) == "tp16dcp16_prefill"
    assert _topology(_dcp_worker(is_producer=False)) == "tp16dcp16_decode"


@pytest.mark.parametrize("is_producer", [True, False])
def test_identity_topology_admits_tp_over_dcp(is_producer):
    # TP rank 11 at DCP8 holds shard 3.
    worker = _dcp_worker(is_producer=is_producer, tp_rank=11, dcp=8, dst_dcp=8)
    role = "prefill" if is_producer else "decode"
    assert _topology(worker, dcp_rank=3) == f"tp16dcp8_{role}"
    with pytest.raises(ValueError, match="dcp_rank == tp_rank % dcp_size"):
        _topology(
            _dcp_worker(is_producer=is_producer, tp_rank=11, dcp=8, dst_dcp=8),
            dcp_rank=11 // 2,
        )


@pytest.mark.parametrize(
    ("worker_kwargs", "topology_kwargs", "message"),
    [
        (dict(dst_dcp=8), {}, "destination's DCP size"),
        (dict(dst_interleave=1024), {}, "destination's DCP size"),
        # _raiden_dst_dcp_size already refuses fewer shards than DCP ranks.
        ({}, dict(dst_shards=8), "destination DCP size must divide"),
        (dict(dcp=32, dst_dcp=16), {}, "dividing tensor_parallel_size"),
        # TP16 / DCP8 ships to as many destination units as TP ranks.
        (dict(dcp=8, dst_dcp=8), dict(dst_shards=8), "shard count"),
        ({}, dict(dcp_rank=4), "dcp_rank == tp_rank"),
        # Without the identity mode a DCP producer fails closed.
        ({}, dict(identity=False), "needs TPU_RAIDEN_GLM_DCP_IDENTITY"),
    ],
)
def test_identity_producer_topology_fails_closed(
    worker_kwargs, topology_kwargs, message
):
    with pytest.raises(ValueError, match=message):
        _topology(_dcp_worker(is_producer=True, **worker_kwargs), **topology_kwargs)


def test_identity_decode_must_be_tp_dcp():
    with pytest.raises(ValueError, match="needs a decode with"):
        _topology(_dcp_worker(is_producer=False, tp_rank=0, tp_size=1, dcp=1))
    with pytest.raises(ValueError, match="dividing tensor_parallel_size"):
        _topology(_dcp_worker(is_producer=False, dcp=32, dst_dcp=16))


# ---------------------------------------------------------------------------
# Producer send metadata and the decode's check.
# ---------------------------------------------------------------------------


def _identity_producer(dcp=16):
    producer = _make_raiden_scheduler(is_producer=True, block_size=256, tp_size=16)
    parallel = producer.vllm_config.parallel_config
    parallel.decode_context_parallel_size = dcp
    parallel.cp_kv_cache_interleave_size = 4
    producer.vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "destination_decode_context_parallel_size": dcp,
        "destination_cp_kv_cache_interleave_size": 4,
    }
    return producer


def test_tp_over_dcp_producer_publishes_its_dcp_geometry():
    producer = _identity_producer(dcp=8)
    prompt_tokens = 10_000
    req = SimpleNamespace(
        request_id="identity-dcp8",
        num_prompt_tokens=prompt_tokens,
        num_computed_tokens=prompt_tokens,
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        kv_transfer_params=None,
    )
    # 9999 transfer tokens over 256-token pages at DCP8: five 2048-token
    # scheduler blocks.
    block_ids = [100, 101, 102, 103, 104]
    with (
        patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True),
        patch(f"{_MOD}._use_raiden_glm_admission", return_value=True),
        patch(f"{_MOD}._raiden_glm_dcp_identity", return_value=True),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS", "p:27000"),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_ENGINE_ID", "producer"),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM", 16),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_MAX_TRANSFER_TOKENS", None),
        patch(f"{_MOD}._raiden_dst_shards", return_value=16),
    ):
        delay, params = producer.request_finished(req, block_ids)
    assert delay
    assert (params["dst_tp_size"], params["dst_dcp_size"]) == (16, 8)
    assert (params["src_dcp_size"], params["src_interleave_size"]) == (8, 4)
    assert producer.reqs_to_send["identity-dcp8"].local_block_ids == block_ids


@pytest.mark.parametrize("identity", [True, False])
def test_identity_producer_publishes_scheduler_blocks_and_source_geometry(identity):
    producer = _identity_producer()
    prompt_tokens = 10_000
    req = SimpleNamespace(
        request_id="identity",
        num_prompt_tokens=prompt_tokens,
        num_computed_tokens=prompt_tokens,
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        kv_transfer_params=None,
    )
    # 9999 transfer tokens over 256-token pages at DCP16: three 4096-token
    # scheduler blocks.
    block_ids = [100, 101, 102]
    with (
        patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True),
        patch(f"{_MOD}._use_raiden_glm_admission", return_value=True),
        patch(f"{_MOD}._raiden_glm_dcp_identity", return_value=identity),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS", "p:27000"),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_ENGINE_ID", "producer"),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM", 16),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_MAX_TRANSFER_TOKENS", None),
        patch(f"{_MOD}._raiden_dst_shards", return_value=16),
    ):
        delay, params = producer.request_finished(req, block_ids)
    if not identity:
        # Read as 256-token blocks, three ids cannot cover 9999 tokens.
        assert (delay, params) == (False, {})
        return
    assert delay
    assert params["num_tokens"] == prompt_tokens - 1
    assert (params["dst_dcp_size"], params["dst_interleave_size"]) == (16, 4)
    assert (params["src_dcp_size"], params["src_interleave_size"]) == (16, 4)
    assert producer.reqs_to_send["identity"].local_block_ids == block_ids


def _load_params(**overrides):
    params = {
        "req_id": "src-req",
        "uuid": 11,
        "num_tokens": 9999,
        "src_controller_address": "p:27000",
        "src_job_name": "prefill",
        "src_engine_id": "producer",
        "src_data_replica_idx": 0,
        "src_parallelism": 16,
        "dst_tp_size": 16,
        "dst_dcp_size": 16,
        "dst_interleave_size": 4,
        "src_dcp_size": 16,
        "src_interleave_size": 4,
    }
    params.update(overrides)
    return {key: value for key, value in params.items() if value is not None}


@pytest.mark.parametrize(
    ("identity", "overrides", "accepted"),
    [
        (True, {}, True),
        (True, dict(src_interleave_size=1024), False),
        (True, dict(src_dcp_size=8), False),
        # A pipeline producer publishes no source DCP geometry.
        (True, dict(src_dcp_size=None, src_interleave_size=None), False),
        (False, {}, False),
        (False, dict(src_dcp_size=None, src_interleave_size=None), True),
    ],
)
def test_decode_requires_matching_identity_producer(identity, overrides, accepted):
    decode = _make_raiden_scheduler(is_producer=False, tp_size=16)
    parallel = decode.vllm_config.parallel_config
    parallel.decode_context_parallel_size = 16
    parallel.cp_kv_cache_interleave_size = 4
    request = SimpleNamespace(request_id="dst-req", num_prompt_tokens=10_000)
    with (
        patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True),
        patch(f"{_MOD}._use_raiden_glm_admission", return_value=True),
        patch(f"{_MOD}._raiden_glm_dcp_identity", return_value=identity),
    ):
        error = decode._validate_load_params(request, _load_params(**overrides))
    if accepted:
        assert error is None
    else:
        assert "DCP identity" in error


@pytest.mark.parametrize(
    ("overrides", "accepted"),
    [
        (dict(dst_dcp_size=8, src_dcp_size=8), True),
        # A DCP16 producer's pages hold another shard of each block.
        (dict(dst_dcp_size=8, src_dcp_size=16), False),
    ],
)
def test_tp_over_dcp_decode_requires_matching_identity_producer(overrides, accepted):
    decode = _make_raiden_scheduler(is_producer=False, tp_size=16)
    parallel = decode.vllm_config.parallel_config
    parallel.decode_context_parallel_size = 8
    parallel.cp_kv_cache_interleave_size = 4
    request = SimpleNamespace(request_id="dst-req", num_prompt_tokens=10_000)
    with (
        patch(f"{_MOD}._use_raiden_stage3_transport", return_value=True),
        patch(f"{_MOD}._use_raiden_glm_admission", return_value=True),
        patch(f"{_MOD}._raiden_glm_dcp_identity", return_value=True),
    ):
        error = decode._validate_load_params(request, _load_params(**overrides))
    if accepted:
        assert error is None
    else:
        assert "DCP identity" in error


@pytest.mark.parametrize("prefix_aware", [True, False])
def test_decode_prefix_hit_pulls_only_the_uncached_scheduler_blocks(prefix_aware):
    """Decode prefix caching: a local hit of two 4096-token DCP16 blocks
    leaves only the rest to pull, into the newly allocated pages. With
    TPU_RAIDEN_PREFIX_AWARE_LOAD off (the default) the whole payload is
    pulled into every page, the hit's adopted pages included."""
    prompt_tokens = 4 * 4096
    request = MagicMock()
    request.request_id = "dst-req"
    request.num_prompt_tokens = prompt_tokens
    request.num_computed_tokens = 0
    request.prompt_token_ids = [0] * prompt_tokens
    request.kv_transfer_params = _load_params(num_tokens=prompt_tokens - 1)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([50, 51, 52, 53],)
    with (
        patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT", "raiden", create=True),
        patch(
            f"{_MOD}.tpu_envs.TPU_RAIDEN_PREFIX_AWARE_LOAD", prefix_aware, create=True
        ),
        patch(f"{_MOD}._prefix_aware_load_supported", return_value=True),
        patch(f"{_MOD}.dist_utils.get_raiden_inline_load", return_value=False),
        patch(f"{_MOD}._use_raiden_glm_admission", return_value=True),
        patch(f"{_MOD}._raiden_glm_dcp_identity", return_value=True),
    ):
        decode = _make_raiden_scheduler(
            is_producer=False, block_size=256, tp_size=16, enable_prefix_caching=True
        )
        parallel = decode.vllm_config.parallel_config
        parallel.decode_context_parallel_size = 16
        parallel.cp_kv_cache_interleave_size = 4
        matched, is_async = decode.get_num_new_matched_tokens(request, 2 * 4096)
        decode.update_state_after_alloc(request, blocks, matched)

    assert (matched, is_async) == (2 * 4096 - 1, True)
    load = decode.reqs_to_load["dst-req"]
    assert load.num_tokens == prompt_tokens - 1
    assert not load.release_only
    if prefix_aware:
        assert load.skip_tokens == 2 * 4096
        assert load.local_block_ids == [52, 53]
    else:
        assert load.skip_tokens == 0
        assert load.local_block_ids == [50, 51, 52, 53]


# ---------------------------------------------------------------------------
# The pinned vLLM's P/D interleave override.
# ---------------------------------------------------------------------------


def _interleave_config(connector, *, interleave=4, dcp=16, dcp_interleave=1):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector=connector),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=interleave,
            dcp_kv_cache_interleave_size=dcp_interleave,
        ),
    )


def _kv_cache_config(block_size=256):
    spec = SimpleNamespace(block_size=block_size)
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])


def _unpatched_interleave_adjust(monkeypatch):
    """Start from vLLM's method; monkeypatch restores both after the test."""
    from vllm.config import VllmConfig

    monkeypatch.setattr(
        VllmConfig,
        "adjust_dcp_kv_cache_interleave_size",
        VllmConfig.adjust_dcp_kv_cache_interleave_size,
    )
    monkeypatch.setattr(
        VllmConfig, "_tpu_raiden_keep_interleave_patch", False, raising=False
    )
    return VllmConfig


@pytest.mark.parametrize(
    ("identity", "connector", "expected"),
    [
        (True, "TPURaidenConnector", 4),
        # Other connectors keep vLLM's block-size alignment.
        (True, "NixlConnector", 256),
        # Without the identity transfer nothing is patched.
        (False, "TPURaidenConnector", 256),
    ],
)
def test_identity_keeps_configured_interleave(
    monkeypatch, identity, connector, expected
):
    import vllm_torchtpu

    config_cls = _unpatched_interleave_adjust(monkeypatch)
    _raiden_glm_env(monkeypatch, identity)
    vllm_torchtpu._patch_raiden_identity_keeps_cp_interleave()
    config = _interleave_config(connector)
    config_cls.adjust_dcp_kv_cache_interleave_size(config, _kv_cache_config())
    assert config.parallel_config.cp_kv_cache_interleave_size == expected


def test_identity_interleave_patch_keeps_deprecated_dcp_override(monkeypatch):
    import vllm_torchtpu

    config_cls = _unpatched_interleave_adjust(monkeypatch)
    _raiden_glm_env(monkeypatch, identity=True)
    vllm_torchtpu._patch_raiden_identity_keeps_cp_interleave()
    vllm_torchtpu._patch_raiden_identity_keeps_cp_interleave()  # idempotent
    config = _interleave_config("TPURaidenConnector", dcp_interleave=64)
    config_cls.adjust_dcp_kv_cache_interleave_size(config, _kv_cache_config())
    assert config.parallel_config.cp_kv_cache_interleave_size == 64
