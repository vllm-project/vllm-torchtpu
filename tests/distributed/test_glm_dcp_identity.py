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
from unittest.mock import patch

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
from .test_glm_dcp_spans import _FILLER, _TOKEN_BYTES, _dcp_owner_slots, _token_bytes
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
    ("overrides", "message"),
    [
        # 5000 tokens over 256-token pages at DCP16 are two scheduler blocks.
        (dict(block_ids=[7]), "expected 2"),
        (dict(dcp_rank=16), "outside dcp_size"),
        (dict(num_tokens=0), "num_tokens must be positive"),
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


@pytest.mark.parametrize(
    ("worker_kwargs", "topology_kwargs", "message"),
    [
        (dict(dst_dcp=8), {}, "destination's DCP size"),
        (dict(dst_interleave=1024), {}, "destination's DCP size"),
        # _raiden_dst_dcp_size already refuses fewer shards than DCP ranks.
        ({}, dict(dst_shards=8), "destination DCP size must divide"),
        (dict(dcp=8, dst_dcp=8), {}, r"tensor_parallel_size == decode_context"),
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
    with pytest.raises(ValueError, match="decode with tensor_parallel_size"):
        _topology(_dcp_worker(is_producer=False, tp_rank=0, tp_size=1, dcp=1))


# ---------------------------------------------------------------------------
# Producer send metadata and the decode's check.
# ---------------------------------------------------------------------------


def _identity_producer():
    producer = _make_raiden_scheduler(is_producer=True, block_size=256, tp_size=16)
    parallel = producer.vllm_config.parallel_config
    parallel.decode_context_parallel_size = 16
    parallel.cp_kv_cache_interleave_size = 4
    producer.vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "destination_decode_context_parallel_size": 16,
        "destination_cp_kv_cache_interleave_size": 4,
    }
    return producer


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
