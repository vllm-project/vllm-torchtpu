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
"""Kimi-K3.1's index-key cache in the PP4 -> TP32 Raiden transfer."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_torchtpu.distributed.kv_transfer import tpu_connector as tc
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    lower_dcp_page_spans,
)

from .raiden_test_utils import FakeTensor, kimi_pool_manifest
from .test_tpu_connector import (
    _MOD,
    _FakeRaidenControllerFacade,
    _make_raiden_worker,
    _make_worker,
)

K31 = "KimiK31ForConditionalGeneration"
# The MLA layers, and those of the last PP4 stage.
MLA_LAYERS = [*range(3, 92, 4), 92]
STAGE3_LAYERS = [75, 79, 83, 87, 91, 92]
# A decode page: 512 tokens, then 32 scale rows.
PAGE_TOKENS = 512
PAGE_ROWS = PAGE_TOKENS + 32


def _index(
    layers, rows, *, slots=None, blocks=4, dtype="torch.uint8", itemsize=1, width=128
):
    tensor = FakeTensor(
        (blocks, len(layers) if slots is None else slots, rows, width),
        itemsize,
        dtype=dtype,
    )
    return {f"model.layers.{i}.self_attn.indexer.k_cache": tensor for i in layers}


def _decoder(index=None, kv_dtype="float8_e4m3fn", rows=128):
    # 128 fp8 rows of four tokens: a 512-token decode block.
    return kimi_pool_manifest(
        3, kv_dtype=kv_dtype, rows=rows, per_layer_tags=True, index=index
    )


def _stage(index=None):
    return kimi_pool_manifest(
        3,
        kv_dtype="float8_e4m3fn",
        rows=128,
        sub_blocks=4,
        per_layer_tags=True,
        index=index,
    )


def test_decoder_manifest_places_each_layers_index_page():
    manifest = _decoder(_index(MLA_LAYERS, PAGE_ROWS))

    pools = manifest.pools_of_class("dsa.idx")
    assert [pool.tag for pool in pools] == [f"dsa.idx.l{i}" for i in MLA_LAYERS]
    assert [pool.base_offset_bytes for pool in pools] == [
        m * PAGE_ROWS * 128 for m in range(24)
    ]
    assert {pool.block_stride_bytes for pool in pools} == {24 * PAGE_ROWS * 128}
    assert {pool.dtype_tag for pool in pools} == {"uint8"}
    assert manifest.geometry_by_tag()["dsa.idx"]["live_bytes_per_block"] == (
        PAGE_ROWS * 128
    )
    # The K3 pools are those of the manifest without an index.
    assert [p for p in manifest.pools if p not in pools] == _decoder().pools


def test_stage_manifest_stacks_four_decode_pages():
    manifest = _stage(_index(STAGE3_LAYERS, 4 * PAGE_ROWS))

    pools = manifest.pools_of_class("dsa.idx")
    assert [pool.tag for pool in pools] == [f"dsa.idx.l{i}" for i in STAGE3_LAYERS]
    assert manifest.geometry_by_tag()["dsa.idx"]["live_bytes_per_block"] == (
        4 * PAGE_ROWS * 128
    )
    for j, pool in enumerate(pools):
        (region,) = pool.regions
        assert (region.num_units, region.unit_bytes) == (4, PAGE_ROWS * 128)
        assert pool.block_stride_bytes == 6 * 4 * PAGE_ROWS * 128
        assert pool.block_stride_bytes == 24 * PAGE_ROWS * 128
        assert pool.base_offset_bytes == j * 4 * PAGE_ROWS * 128


@pytest.mark.parametrize(
    "index",
    [
        {
            **_index(MLA_LAYERS[:12], PAGE_ROWS, slots=24),
            **_index(MLA_LAYERS[12:], PAGE_ROWS, slots=24),
        },
        _index(MLA_LAYERS, PAGE_ROWS + 1),
        _index(MLA_LAYERS, PAGE_ROWS, blocks=3),
        _index(MLA_LAYERS, PAGE_ROWS, dtype="torch.bfloat16", itemsize=2),
        _index(MLA_LAYERS[:-1], PAGE_ROWS, slots=24),
    ],
    ids=["two-tensors", "rows", "num-blocks", "bf16", "missing-layer"],
)
def test_index_pool_is_refused_unless_one_uint8_tensor_of_the_pool_blocks(index):
    with pytest.raises(ValueError, match="index-key cache"):
        _decoder(index)


def _layout(tile_rows):
    def getter(tensor):
        if str(tensor.dtype).endswith("uint8"):
            return ([3, 2, 1, 0], [[tile_rows, 128], [4, 1]], 0)
        packing = tensor.shape[2]
        return ([3, 2, 1, 0], [[packing, 128], [packing, 1]], 0)

    return getter


def _fingerprints(manifests, tile_rows=32):
    """Fingerprints of ``(role, block_size, manifest)``: a PP4 x PCP8 stage
    rank, a TP32 slice prefill rank or a TP32 decode rank."""
    workers = {
        "stage": _make_raiden_worker(
            tp_rank=0, tp_size=1, pcp_size=8, pp_size=4, architecture=K31
        ),
        "slice": _make_raiden_worker(tp_rank=0, tp_size=32, architecture=K31),
        "decoder": _make_raiden_worker(
            tp_rank=0, tp_size=32, is_producer=False, architecture=K31
        ),
    }
    with (
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM", 32),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_DST_SHARDS", 32),
        patch(
            "vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint."
            "_default_layout_getter",
            side_effect=_layout(tile_rows),
        ),
        patch("importlib.metadata.version", return_value="test"),
    ):
        fingerprints = []
        for role, block_size, manifest in manifests:
            worker = workers[role]
            worker.vllm_config.cache_config.block_size = block_size
            fingerprints.append(worker._measure_stage3_layout(manifest)[0])
        return fingerprints


def test_stage_and_decoder_index_fingerprints_agree():
    stage, tp_slice, decoder, k3 = _fingerprints(
        [
            ("stage", 4 * PAGE_TOKENS, _stage(_index(STAGE3_LAYERS, 4 * PAGE_ROWS))),
            ("slice", PAGE_TOKENS, _decoder(_index(MLA_LAYERS, PAGE_ROWS))),
            ("decoder", PAGE_TOKENS, _decoder(_index(MLA_LAYERS, PAGE_ROWS))),
            ("decoder", PAGE_TOKENS, _decoder()),
        ]
    )
    assert stage == tp_slice == decoder
    assert decoder != k3
    # bf16 rows of two tokens: K3 fingerprints alike across decode page
    # sizes, but a 256-token index page is not a 512-token one.
    pages = []
    for page_tokens in (256, PAGE_TOKENS):
        rows = page_tokens // 2
        k3, k31 = _fingerprints(
            [
                ("decoder", page_tokens, _decoder(kv_dtype="bfloat16", rows=rows)),
                (
                    "decoder",
                    page_tokens,
                    _decoder(
                        _index(MLA_LAYERS, page_tokens + 32),
                        kv_dtype="bfloat16",
                        rows=rows,
                    ),
                ),
            ]
        )
        pages.append((k3, k31))
    assert pages[0][0] == pages[1][0]
    assert pages[0][1] != pages[1][1]
    # A tile that splits a page leaves it no single byte range.
    with pytest.raises(RuntimeError, match="index"):
        _fingerprints(
            [("decoder", PAGE_TOKENS, _decoder(_index(MLA_LAYERS, PAGE_ROWS)))], 64
        )


def test_stage_row_spans_carry_every_index_tag_to_its_decode_ranks():
    manifest = _stage(_index(STAGE3_LAYERS, 4 * PAGE_ROWS))
    geometry = tc._kimi_row_geometry(manifest)
    index_tags = [f"dsa.idx.l{i}" for i in STAGE3_LAYERS]
    assert list(geometry) == ["fa.l4", *index_tags]
    page = PAGE_ROWS * 128
    assert all(geometry[tag] == (4 * page, page) for tag in index_tags)

    worker = _make_raiden_worker(
        tp_rank=0,
        tp_size=1,
        pcp_size=8,
        pp_size=4,
        block_size=4 * PAGE_TOKENS,
        interleave_size=4 * PAGE_TOKENS,
        architecture=K31,
    )
    worker.vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "destination_decode_context_parallel_size": 32
    }
    worker._raiden_manifest = manifest
    worker._stage3_row_geometry = geometry
    worker._stage3_state_group_count = 1
    worker._raiden_controller_facade = facade = _FakeRaidenControllerFacade()
    worker._raiden_controller_address = "prefill.test:27000"
    worker._raiden_work_unit = SimpleNamespace(rank=21)
    metadata = tc.TPUConnectorMetadata()
    metadata.reqs_to_send["request"] = SimpleNamespace(
        uuid=123,
        local_block_ids=[40, 31],
        mamba_state_block_ids=[9],
        num_tokens=6000,
        expiration_time=1e20,
    )
    with (
        patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT", "raiden"),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM", 32),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_DST_SHARDS", 32),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER", True),
        patch(
            "vllm.distributed.parallel_state.get_pp_group",
            return_value=SimpleNamespace(rank_in_group=2),
        ),
        patch("vllm_torchtpu.distributed.pcp.get_pcp_rank", return_value=5),
    ):
        worker._register_stage3_request_row_spans(metadata)

    (call,) = facade.register_request_blocks_calls
    spans = {registration.tag: registration for registration in call["pool_spans"]}
    assert [r.tag for r in call["pool_spans"]][: 1 + len(index_tags)] == [
        "fa.l4",
        *index_tags,
    ]
    for tag in index_tags:
        assert spans[tag].block_ids == (40, 31)
        assert {s.dst_unit_ordinal for s in spans[tag].spans} == {20, 21, 22, 23}
        assert {s.size_bytes for s in spans[tag].spans} == {page}


def test_index_pools_stage_one_request_of_pages_per_storage():
    worker = _make_raiden_worker(tp_rank=0, tp_size=1, architecture=K31)
    groups = (
        SimpleNamespace(layer_names=("model.layers.4.attn",)),
        SimpleNamespace(layer_names=("model.layers.1.kda",)),
    )
    k3 = worker._raiden_staging_blocks_per_pool(_decoder(), groups)
    k31 = worker._raiden_staging_blocks_per_pool(
        _decoder(_index(MLA_LAYERS, PAGE_ROWS)), groups
    )
    assert k31 == k3 + [worker._max_request_blocks()] * 24


@pytest.mark.parametrize("layer", [75, 92])
@pytest.mark.parametrize("core", [0, 5])
def test_pp4_spans_carry_index_sub_pages_to_their_decode_ranks(core, layer):
    rows = 4 * PAGE_ROWS
    manifest = _stage(_index(STAGE3_LAYERS, rows))
    (pool,) = [p for p in manifest.pools if p.tag == f"dsa.idx.l{layer}"]
    j = STAGE3_LAYERS.index(layer)
    # Every 32-bit word of row r of slot j in block b reads (b, j, r).
    words = np.zeros((4, 6, rows, 32), np.uint32)
    for b in range(4):
        for slot in range(6):
            words[b, slot] = ((b << 24) | (slot << 16) | np.arange(rows))[:, None]
    stage = words.view(np.uint8)
    assert stage.shape == (4, 6, rows, 128)
    assert pool.block_stride_bytes == stage[0].nbytes
    block_ids = [2, 3]

    spans = lower_dcp_page_spans(
        tag=pool.tag,
        transfer_rank=core,
        live_bytes_per_block=4 * PAGE_ROWS * 128,
        block_ids=block_ids,
        sub_blocks=4,
    )
    units = {}
    for span in spans.spans:
        block = stage[spans.block_ids[span.src_block_ordinal]].reshape(-1)
        start = pool.base_offset_bytes + span.src_offset_bytes
        unit = units.setdefault(
            span.dst_unit_ordinal, np.zeros(len(block_ids) * PAGE_ROWS * 128, np.uint8)
        )
        unit[span.dst_offset_bytes : span.dst_offset_bytes + span.size_bytes] = block[
            start : start + span.size_bytes
        ]

    assert sorted(units) == [4 * core + i for i in range(4)]
    page = PAGE_ROWS * 128
    for i in range(4):
        for c, block_id in enumerate(block_ids):
            expected = stage[block_id, j, i * PAGE_ROWS : (i + 1) * PAGE_ROWS]
            got = units[4 * core + i][c * page : (c + 1) * page]
            assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize(
    "skip_tokens", [PAGE_TOKENS * 32 // 2, PAGE_TOKENS * 32, 3 * PAGE_TOKENS * 32]
)
def test_consumer_registers_index_tags_after_fa_with_whole_page_skips(
    monkeypatch, skip_tokens, queued
):
    dcp = 32
    worker = _make_raiden_worker(
        is_producer=False,
        tp_rank=0,
        tp_size=32,
        block_size=PAGE_TOKENS,
        architecture=K31,
    )
    worker.vllm_config.parallel_config.decode_context_parallel_size = dcp
    worker._raiden_manifest = _decoder(_index(MLA_LAYERS, PAGE_ROWS))
    worker._stage3_state_group_count = 1
    worker._raiden_work_unit = SimpleNamespace(job_name="decode")
    facade = MagicMock()
    facade.start_transfer.return_value = True
    worker._stage3_source_facades["localhost:27000"] = facade
    monkeypatch.setattr(
        worker,
        "_stage3_source_work_units",
        lambda meta: [SimpleNamespace(job_name="prefill")],
    )
    monkeypatch.setattr(worker, "_raiden_hbm_memory_type", lambda: 3)
    monkeypatch.setattr(tc.tpu_envs, "TPU_RAIDEN_POOL_TAGS_PER_LAYER", True)
    req = SimpleNamespace(
        src_controller_address="localhost:27000",
        skip_tokens=skip_tokens,
        mamba_state_block_ids=[11],
    )
    args = dict(
        req_meta=req,
        source_req_id="p",
        destination_req_id="d",
        uuid=123,
        num_tokens=4 * PAGE_TOKENS * dcp,
        local_blocks=[7],
        dst_controller_address="localhost:27001",
        dst_units=[worker._raiden_work_unit],
    )
    if queued:
        pending = tc._Stage3PendingSubmit(
            **args, defer_deadline=None, first_attempt_s=0
        )
        task = worker._stage_stage3_submit_task(pending, synchronous=False)
        assert task is not None
        assert worker._execute_stage3_submit(task).error is None
    else:
        outcome = worker._submit_stage3_load_as_leader(**args)
        assert outcome[0] == "accepted", outcome

    kwargs = facade.start_transfer.call_args.kwargs
    index_tags = [f"dsa.idx.l{i}" for i in MLA_LAYERS]
    assert kwargs["transfer_pool_tags"] == [
        "fa.l4",
        *index_tags,
        "gdn.conv.g0.l1",
        "gdn.ssm.g0.l1",
    ]
    assert kwargs["dst_device_block_ids"] == [7] * 25 + [11, 11]
    assert kwargs["dst_block_counts"] == [1] * 27
    # 640 FA bytes per token: four tokens in each 2560-byte row.
    fa_skip = skip_tokens // dcp * 640
    index_skip = (skip_tokens // dcp) // PAGE_TOKENS * PAGE_ROWS * 128
    assert kwargs["dst_skip_bytes"] == [fa_skip] + [index_skip] * 24 + [0, 0]


class _PastTheGuard(Exception):
    pass


def _raise_past(*args, **kwargs):
    raise _PastTheGuard


@pytest.mark.parametrize("architecture", [K31, "GlmMoeDsaForCausalLM"])
def test_zmq_refuses_kimi_k31(architecture):
    worker = _make_worker()
    worker.vllm_config.model_config.architecture = architecture
    worker._extract_kv_layout = _raise_past
    runner = SimpleNamespace(device="cpu", kv_caches=[torch.zeros(4, 8)])
    if architecture == K31:
        with pytest.raises(NotImplementedError, match="index-key cache"):
            worker.register_runner(runner)
    else:
        with pytest.raises(_PastTheGuard):
            worker.register_runner(runner)


@pytest.mark.parametrize(
    ("architecture", "admission", "stage3", "refused"),
    [
        (K31, False, False, True),
        (K31, True, False, True),
        (K31, True, True, False),
        ("GlmMoeDsaForCausalLM", False, False, False),
    ],
    ids=["k31-legacy", "k31-admission-without-stage3", "k31-stage3", "glm-legacy"],
)
def test_legacy_raiden_refuses_kimi_k31(
    monkeypatch, architecture, admission, stage3, refused
):
    worker = _make_raiden_worker(architecture=architecture)
    for name, value in {
        "TPU_USE_RAIDEN_KV_CACHE_MANAGER": admission,
        "TPU_RAIDEN_KIMIK3_ADMISSION": admission,
        "TPU_RAIDEN_QWEN35_ADMISSION": False,
        "TPU_RAIDEN_GLM_ADMISSION": False,
        "TPU_RAIDEN_DSV4_ADMISSION": False,
    }.items():
        monkeypatch.setattr(tc.tpu_envs, name, value)
    worker._raiden_stage3_enabled = lambda: stage3
    worker._admit_raiden_hybrid_kv_cache = _raise_past
    worker._ensure_raiden_transfer_engine = _raise_past
    if refused:
        with pytest.raises(NotImplementedError, match="Stage-3 Raiden"):
            worker.register_runner(SimpleNamespace())
    else:
        with pytest.raises(_PastTheGuard):
            worker.register_runner(SimpleNamespace())


def _offload_connector(architecture):
    from vllm_torchtpu.offload.raiden_store import KimiIndexGeometry

    worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=4,
            kv_cache_groups=[object()],
            hisparse_host_num_blocks=None,
            prefix_cacheable_group_ids=(0,),
        ),
        _init_worker=lambda canonical: None,
        spec=SimpleNamespace(
            pool_sources=[],
            kimi_index=KimiIndexGeometry(rows=2, layers=1),
            pp_size=1,
        ),
    )
    return SimpleNamespace(
        connector_worker=worker,
        _vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(architecture=architecture)
        ),
    )


@pytest.mark.parametrize("architecture", [K31, "GlmMoeDsaForCausalLM"])
def test_offload_carries_kimi_k31_index_pages(architecture):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    pool = torch.zeros(4, 128, dtype=torch.uint8)
    index = torch.zeros(4, 1, 2, 128, dtype=torch.uint8)
    kv_caches = {
        "model.layers.3.self_attn": pool,
        "model.layers.3.self_attn.indexer.k_cache": index,
    }
    connector = _offload_connector(architecture)
    TPURaidenOffloadingConnector.register_kv_caches(connector, kv_caches)
    # Kimi-K3.1's index-key cache is a block array of its own beside the
    # unified pools; the GLM indexer shares the name and is a plain layer.
    expected = [1] if architecture == K31 else []
    assert connector.connector_worker.spec.index_arrays == expected
