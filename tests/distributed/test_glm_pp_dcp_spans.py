# SPDX-License-Identifier: Apache-2.0
"""GLM PP x DCP producer -> larger DCP decode (lower_glm_dcp_page_scatter_spans).

A pipeline stage's TP ranks are its DCP ranks. Both legs page at the DCP
interleave, so each page holds one interleave chunk and ships whole to the
decode rank that holds those tokens. Every token is placed on both legs by the
DCP cache writer (`_get_dcp_page_and_slot`) and must land where the decode's
writer puts it. No TPU runtime or transport is involved.
"""

import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    TPUConnectorMetadata,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    lower_glm_dcp_identity_spans,
    lower_glm_dcp_page_scatter_spans,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnectorScheduler,
)
from vllm_torchtpu.kernels.mla import kv_cache_utils as kvu

from . import test_tpu_connector as _tc
from .test_glm_dcp_identity import _writer_pages
from .test_glm_dcp_spans import (
    _FILLER,
    _TOKEN_BYTES,
    _dcp_owner_slots,
    _decode_tag_skips,
    _global_span_pieces,
    _token_bytes,
)
from .test_tpu_connector import (
    _MOD,
    _FakeRaidenControllerFacade,
    _FakeRaidenEngine,
    _make_raiden_worker,
    _make_vllm_config,
)

# (page_tokens == interleave, src_dcp, dst_dcp)
_CASES = [
    (256, 2, 16),
    (256, 4, 16),
    (256, 2, 8),
    (256, 1, 16),
    (64, 2, 4),
    (256, 16, 16),
]


def _token_counts(page_tokens, src_dcp, dst_dcp):
    src_block, dst_block = page_tokens * src_dcp, page_tokens * dst_dcp
    counts = {1, 2, page_tokens, page_tokens + 1, src_block - 1, src_block}
    counts |= {src_block + 1, dst_block - 1, dst_block, dst_block + 1}
    counts |= {3 * dst_block + page_tokens + 5}
    return sorted(c for c in counts if c > 0)


def _scatter(num_tokens, page_tokens, src_dcp, dst_dcp, src_pages, src_ids, live, skip):
    """Replay every source rank's spans; return per-unit coverage and pages."""
    num_dst = math.ceil(num_tokens / (page_tokens * dst_dcp))
    landed = [dict() for _ in range(dst_dcp)]
    for rank in range(src_dcp):
        reg = lower_glm_dcp_page_scatter_spans(
            tag="mla.nope",
            num_tokens=num_tokens,
            page_tokens=page_tokens,
            src_dcp_size=src_dcp,
            src_dcp_rank=rank,
            dst_dcp_size=dst_dcp,
            live_bytes_per_block=live,
            block_ids=src_ids,
        )
        assert reg.dst_space_version == 1
        assert reg.block_ids == tuple(src_ids)
        assert reg.declared_bytes == sum(span.size_bytes for span in reg.spans)
        # This rank feeds dst_dcp / src_dcp units, one page per decode block each.
        assert len(reg.spans) == (dst_dcp // src_dcp) * num_dst
        for span in reg.spans:
            unit = span.dst_unit_ordinal if dst_dcp > 1 else 0
            assert unit % src_dcp == rank, (
                "a unit is fed by the rank holding its tokens"
            )
            assert span.count == 1 and span.dst_block_index == 0
            assert span.src_offset_bytes == 0 and span.size_bytes == live
            assert 0 <= span.src_block_ordinal < len(src_ids)
            page = src_pages[rank][reg.block_ids[span.src_block_ordinal]]
            for index, in_page, offset, take in _global_span_pieces(
                span, live=live, tag_skip=skip
            ):
                assert in_page == 0 and take == live, "pages move whole"
                assert index not in landed[unit], f"unit {unit} page {index} twice"
                landed[unit][index] = page[offset : offset + take]
    return landed


@pytest.mark.parametrize(
    "page_tokens,src_dcp,dst_dcp,num_tokens",
    [
        (page, src, dst, tokens)
        for page, src, dst in _CASES
        for tokens in _token_counts(page, src, dst)
    ],
)
def test_page_scatter_lands_where_the_decode_writer_puts_each_token(
    page_tokens, src_dcp, dst_dcp, num_tokens
):
    num_src = math.ceil(num_tokens / (page_tokens * src_dcp))
    num_dst = math.ceil(num_tokens / (page_tokens * dst_dcp))
    num_pages = 2 * max(num_src, num_dst) + 3
    rng = np.random.default_rng(num_tokens + 7 * src_dcp)
    src_ids = [int(i) for i in rng.permutation(num_pages)[:num_src]]
    dst_ids = [int(i) for i in rng.permutation(num_pages)[:num_dst]]
    live = page_tokens * _TOKEN_BYTES
    src_pages = _writer_pages(
        _dcp_owner_slots(
            num_tokens, page_tokens, src_dcp, page_tokens, src_ids, num_pages
        ),
        num_pages,
        live,
    )
    landed = _scatter(
        num_tokens, page_tokens, src_dcp, dst_dcp, src_pages, src_ids, live, skip=0
    )
    # ValidateSpanCoverage: every destination page of every unit, whole, so
    # every unit has the same extent.
    for unit in range(dst_dcp):
        assert sorted(landed[unit]) == list(range(num_dst)), f"unit {unit}"

    owners = _dcp_owner_slots(
        num_tokens, page_tokens, dst_dcp, page_tokens, dst_ids, num_pages
    )
    for unit, (page, slot) in enumerate(owners):
        dst = np.full((num_pages, live), _FILLER, dtype=np.uint8)
        for index, data in landed[unit].items():
            dst[dst_ids[index]] = data
        tokens = np.flatnonzero(page != kvu.OOB_PAGE)
        byte_index = slot[tokens, None] * _TOKEN_BYTES + np.arange(_TOKEN_BYTES)
        np.testing.assert_array_equal(
            dst[page[tokens, None], byte_index], _token_bytes(tokens, 0)
        )


@pytest.mark.parametrize(
    "page_tokens,src_dcp,dst_dcp,num_tokens,prefix_blocks",
    [
        (page, src, dst, tokens, prefix)
        for page, src, dst in _CASES
        for tokens in (page * dst + 1, 2 * page * dst, 3 * page * dst + page + 5)
        for prefix in sorted({1, math.ceil(tokens / (page * dst)) - 1})
    ],
)
def test_page_scatter_suffix_pull_lands_past_a_cached_prefix(
    page_tokens, src_dcp, dst_dcp, num_tokens, prefix_blocks
):
    """Decode prefix caching: the decode holds the first prefix_blocks of its
    scheduler blocks and pulls the rest with the per-tag clip its worker
    computes; every suffix page arrives once, whole."""
    num_src = math.ceil(num_tokens / (page_tokens * src_dcp))
    num_dst = math.ceil(num_tokens / (page_tokens * dst_dcp))
    num_pages = 2 * max(num_src, num_dst) + 3
    rng = np.random.default_rng(num_tokens + prefix_blocks)
    src_ids = [int(i) for i in rng.permutation(num_pages)[:num_src]]
    live = page_tokens * _TOKEN_BYTES
    (skip,) = _decode_tag_skips(
        {"mla.nope": live},
        page_tokens=page_tokens,
        dcp=dst_dcp,
        skip_tokens=prefix_blocks * page_tokens * dst_dcp,
    ).values()
    assert skip == prefix_blocks * live
    src_pages = _writer_pages(
        _dcp_owner_slots(
            num_tokens, page_tokens, src_dcp, page_tokens, src_ids, num_pages
        ),
        num_pages,
        live,
    )
    full = _scatter(
        num_tokens, page_tokens, src_dcp, dst_dcp, src_pages, src_ids, live, skip=0
    )
    suffix = _scatter(
        num_tokens, page_tokens, src_dcp, dst_dcp, src_pages, src_ids, live, skip=skip
    )
    for unit in range(dst_dcp):
        assert sorted(suffix[unit]) == list(range(num_dst - prefix_blocks))
        for index, data in suffix[unit].items():
            np.testing.assert_array_equal(data, full[unit][index + prefix_blocks])


@pytest.mark.parametrize("num_tokens", [1, 255, 4096, 4097, 9999])
def test_equal_dcp_sizes_match_the_identity_lowering(num_tokens):
    kwargs = dict(
        tag="mla.nope",
        num_tokens=num_tokens,
        page_tokens=256,
        live_bytes_per_block=256 * 576,
        block_ids=list(range(math.ceil(num_tokens / (256 * 16)))),
    )
    for rank in range(16):
        got = lower_glm_dcp_page_scatter_spans(
            src_dcp_size=16, src_dcp_rank=rank, dst_dcp_size=16, **kwargs
        )
        want = lower_glm_dcp_identity_spans(dcp_size=16, dcp_rank=rank, **kwargs)
        assert got == want


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # 5000 tokens over 256-token pages at DCP2 are ten scheduler blocks.
        (dict(block_ids=[7]), "expected 10"),
        (dict(src_dcp_rank=2), "outside src_dcp_size"),
        (dict(dst_dcp_size=12, src_dcp_size=8, src_dcp_rank=0), "multiple"),
        (dict(num_tokens=0), "num_tokens must be positive"),
    ],
)
def test_page_scatter_lowering_fails_closed(overrides, message):
    kwargs = dict(
        tag="dsa.idx",
        num_tokens=5000,
        page_tokens=256,
        src_dcp_size=2,
        src_dcp_rank=1,
        dst_dcp_size=16,
        live_bytes_per_block=33 * 4 * 256,
        block_ids=list(range(10)),
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        lower_glm_dcp_page_scatter_spans(**kwargs)


# ---------------------------------------------------------------------------
# Connector: a PP8 x TP2/DCP2 producer feeding a TP16/DCP16 decode.
# ---------------------------------------------------------------------------


def _pp_dcp_worker(tp_rank=1, block_size=256, interleave=256, dst_interleave=256):
    worker = _make_raiden_worker(
        tp_rank=tp_rank, tp_size=2, pp_size=8, block_size=block_size
    )
    parallel = worker.vllm_config.parallel_config
    parallel.decode_context_parallel_size = 2
    parallel.cp_kv_cache_interleave_size = interleave
    worker.vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "destination_decode_context_parallel_size": 16,
        "destination_cp_kv_cache_interleave_size": dst_interleave,
    }
    return worker


def _env(parallelism=16):
    return (
        patch(f"{_MOD}.tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER", True),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_GLM_ADMISSION", True),
        patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT", "raiden"),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM", parallelism),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER", True, create=True),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_GLM_DCP_IDENTITY", False, create=True),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_DST_SHARDS", 16, create=True),
    )


class _Env:
    def __init__(self, patches):
        self.patches = patches

    def __enter__(self):
        for p in self.patches:
            p.__enter__()

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.__exit__(*exc)


@pytest.mark.parametrize(
    ("stage", "tp_rank", "unit"), [(0, 0, 0), (0, 1, 1), (5, 1, 11)]
)
def test_transfer_rank_is_stage_major_over_the_stage_dcp_ranks(stage, tp_rank, unit):
    worker = _pp_dcp_worker(tp_rank=tp_rank)
    group = SimpleNamespace(rank_in_group=stage)
    with patch("vllm.distributed.parallel_state.get_pp_group", return_value=group):
        assert worker._local_raiden_transfer_rank() == unit


def test_parallelism_and_completion_count_cover_every_stage_rank():
    worker = _pp_dcp_worker()
    cfg = _make_vllm_config(is_producer=True, pp_size=8, tp_size=2)
    cfg.parallel_config.decode_context_parallel_size = 2
    with _Env(_env()):
        worker._validate_stage3_transfer_parallelism(16)
        with pytest.raises(ValueError, match="DCP sizes"):
            worker._validate_stage3_transfer_parallelism(8)
        assert TPURaidenConnectorScheduler(cfg).get_finished_count() == 16


def test_topology_admits_a_pp_dcp_producer():
    worker = _pp_dcp_worker(tp_rank=1)
    with (
        _Env(_env()),
        patch("vllm_torchtpu.distributed.dcp.get_dcp_rank", return_value=1),
    ):
        assert worker._raiden_glm_admission_topology() == "pp8tp2dcp2_prefill"


@pytest.mark.parametrize(
    ("kwargs", "dcp_rank", "message"),
    [
        (dict(interleave=4), 1, "whole pages"),
        (dict(dst_interleave=4), 1, "whole pages"),
        (dict(), 0, "dcp_rank == tp_rank"),
    ],
)
def test_pp_dcp_producer_topology_fails_closed(kwargs, dcp_rank, message):
    worker = _pp_dcp_worker(tp_rank=1, **kwargs)
    with (
        _Env(_env()),
        patch("vllm_torchtpu.distributed.dcp.get_dcp_rank", return_value=dcp_rank),
        pytest.raises(ValueError, match=message),
    ):
        worker._raiden_glm_admission_topology()


def test_stage_rank_registers_its_pages_for_its_decode_ranks():
    worker = _pp_dcp_worker(tp_rank=1)
    stage_manifest = _tc.TestGlmPipelineParallelProducer._stage_manifest
    with _Env(_env()):
        worker._stage3_row_geometry = worker._measure_glm_tag_geometry(
            stage_manifest({1})
        )
    worker._raiden_transfer_engine = _FakeRaidenEngine()
    worker._raiden_controller_facade = facade = _FakeRaidenControllerFacade()
    worker._raiden_controller_address = "prefill.test:27000"
    worker._raiden_work_unit = SimpleNamespace(rank=11)
    metadata = TPUConnectorMetadata()
    # 5000 tokens: ten 512-token scheduler blocks, two 4096-token decode blocks.
    metadata.reqs_to_send["req"] = SimpleNamespace(
        uuid=7,
        local_block_ids=list(range(20, 30)),
        num_tokens=5000,
        expiration_time=1e20,
    )
    group = SimpleNamespace(rank_in_group=5)
    with (
        _Env(_env()),
        patch("vllm.distributed.parallel_state.get_pp_group", return_value=group),
    ):
        worker.process_send_load(metadata)

    (call,) = facade.register_request_blocks_calls
    assert call["block_ids"] == list(range(20, 30))
    for reg in call["pool_spans"]:
        assert reg.tag.endswith(".l1")
        units = sorted({span.dst_unit_ordinal for span in reg.spans})
        assert units == list(range(1, 16, 2)), "DCP rank 1 feeds the odd units"
        assert len(reg.spans) == 8 * 2
