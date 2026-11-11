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
"""Model connector planning and multi-group generation integrity on CPU."""

import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast
from unittest.mock import patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.request import RequestStatus

from tests.runner.test_qwen4_exp_pd import cache_groups
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata
from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector

from .test_tpu_connector import (
    _MOD,
    _dsv4_worker_plan,
    _FakeRaidenControllerFacade,
    _FakeRaidenEngine,
    _make_raiden_worker,
    _make_vllm_config,
)

if TYPE_CHECKING:
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager
    from vllm.config import SpeculativeConfig


@pytest.fixture(autouse=True)
def stage3(monkeypatch):
    for name, value in {
        "TPU_KV_RESHARD_TRANSPORT": "raiden",
        "TPU_RAIDEN_CONTROLLER_ADDRESS": "source:27000",
        "TPU_RAIDEN_ENGINE_ID": "source-engine",
        "TPU_RAIDEN_TRANSFER_PARALLELISM": "1",
        "TPU_RAIDEN_QWEN35_ADMISSION": "0",
        "TPU_RAIDEN_KIMIK3_ADMISSION": "0",
        "TPU_RAIDEN_GLM_ADMISSION": "0",
        "TPU_RAIDEN_DSV4_ADMISSION": "0",
    }.items():
        monkeypatch.setenv(name, value)


def connector(config, *, state_first=False):
    groups = cache_groups()
    if state_first:
        groups = groups[2:] + groups[:2]
    caches = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups)
    with (
        patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100),
    ):
        return Qwen4ExpRaidenConnector(config, KVConnectorRole.SCHEDULER, caches)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("state_first", [False, True])
def test_constructor_routes_every_group_without_checkpoint_tag_shortcut(
    tp, state_first, monkeypatch
):
    monkeypatch.setenv("TPU_RAIDEN_TRANSFER_PARALLELISM", str(tp))
    obj = connector(_make_vllm_config(tp_size=tp), state_first=state_first)
    assert obj._stage3_mamba_group_indices == []
    assert not obj.supports_divergent_local_hybrid_hits
    scheduler = obj.connector_scheduler
    assert scheduler._stage3_transfer_groups is obj._stage3_transfer_groups
    assert scheduler._stage3_truncates_prompt()
    tables = []
    page = 1
    for group in obj._stage3_transfer_groups:
        count = group.page_count(129)
        tables.append(list(range(page, page + count)))
        page += count
    assert sorted(map(len, tables)) == [1, 1, 1, 2, 2]
    req = SimpleNamespace(
        request_id="qwen-state",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        num_prompt_tokens=129,
        num_computed_tokens=129,
        kv_transfer_params={"_p_side_truncated": True},
    )
    delay, params = obj.request_finished_all_groups(req, tuple(tables))
    assert delay and params["num_tokens"] == 129
    send = scheduler.build_connector_meta().reqs_to_send[req.request_id]
    assert send.group_block_ids == tables
    assert send.mamba_state_block_ids is None


@pytest.mark.parametrize("option", ["pcp", "dcp", "pp", "apc", "mtp"])
def test_execution_switches_do_not_replace_required_transfer_contract(option):
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenWorker

    cfg = _make_vllm_config()
    if option == "apc":
        cfg.cache_config.enable_prefix_caching = True
        with pytest.raises(ValueError, match="prefix-aware|token-page-aligned"):
            connector(cfg)
        return
    elif option == "mtp":
        cfg.speculative_config = object()
    else:
        name = {
            "pcp": "prefill_context_parallel_size",
            "dcp": "decode_context_parallel_size",
            "pp": "pipeline_parallel_size",
        }[option]
        setattr(cfg.parallel_config, name, 2)
    obj = connector(cfg)
    assert (
        obj._stage3_transfer_groups
        == connector(_make_vllm_config())._stage3_transfer_groups
    )
    worker = object.__new__(Qwen4ExpRaidenWorker)
    worker.vllm_config = cfg
    worker.tp_rank, worker.tp_size = 0, 1
    worker._validate_stage3_transfer_parallelism(1)


@pytest.mark.parametrize("admission", ["QWEN35", "KIMIK3", "GLM", "DSV4"])
def test_constructor_cannot_borrow_another_model_admission(admission, monkeypatch):
    monkeypatch.setenv(f"TPU_RAIDEN_{admission}_ADMISSION", "1")
    with pytest.raises(ValueError, match="another model"):
        connector(_make_vllm_config())


@pytest.mark.parametrize("terminal", [False, True])
def test_grouped_worker_replay_checks_secondary_pages(terminal, monkeypatch):
    monkeypatch.setenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER", "1")
    monkeypatch.setenv("TPU_RAIDEN_DSV4_ADMISSION", "1")
    worker = _make_raiden_worker(
        tp_rank=0, tp_size=1, is_producer=True, dp_size=8, block_size=1024
    )
    worker._stage3_dsv4_plan, _ = _dsv4_worker_plan(worker)
    facade = _FakeRaidenControllerFacade()
    engine = _FakeRaidenEngine()
    engine.poll_results = [(["request"], [], [])]
    worker._raiden_transfer_engine = engine
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = "source:27000"
    worker._raiden_work_unit = SimpleNamespace(data_name="grouped")
    tables = [
        [10],
        [20],
        [30, 31, 32, 33],
        [0] * 7 + [47],
        [50],
        [60],
        [70, 71, 72, 73],
    ]
    meta = TPUConnectorMetadata()
    meta.reqs_to_send["request"] = SimpleNamespace(
        uuid=99,
        num_tokens=1024,
        expiration_time=time.perf_counter() + 300,
        local_block_ids=[10],
        group_block_ids=tables,
    )
    worker._register_stage3_request_blocks_grouped(meta)
    if terminal:
        worker._poll_finished(engine)
        assert "request" not in worker._stage3_registered_sends
        assert worker._stage3_terminal_sends["request"].group_block_ids[1] == (20,)
    worker._register_stage3_request_blocks_grouped(meta)
    assert len(facade.register_request_blocks_calls) == 1
    tables[1] = [21]
    with pytest.raises(ValueError, match="Conflicting"):
        worker._register_stage3_request_blocks_grouped(meta)
    assert len(facade.register_request_blocks_calls) == 1


def test_scheduler_rejects_changed_state_group_under_same_request():
    obj = connector(_make_vllm_config())
    req = SimpleNamespace(
        request_id="live",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        num_prompt_tokens=127,
        num_computed_tokens=127,
        kv_transfer_params={"_p_side_truncated": True},
    )
    tables = ([10], [20], [30], [40], [50])
    delay, params = obj.request_finished_all_groups(req, tables)
    assert delay
    obj.connector_scheduler.build_connector_meta()
    with pytest.raises(ValueError, match="Conflicting group pages"):
        obj.request_finished_all_groups(req, ([10], [20], [31], [40], [50]))
    assert not obj.connector_scheduler.build_connector_meta().reqs_to_send
    _, replay = obj.request_finished_all_groups(req, tables)
    assert replay["uuid"] == params["uuid"]


def qwen_worker(monkeypatch, *, tp=1, rank=0, producer=True, as_connector=False):
    from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
        BINDING_ALIASED_RAW,
        PoolEntry,
        PoolManifest,
        RegionSpec,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import build_transfer_groups
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import (
        Qwen4ExpRaidenWorker,
        QwenPagePlan,
    )

    monkeypatch.setenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER", "1")
    monkeypatch.setenv("TPU_RAIDEN_TRANSFER_PARALLELISM", str(tp))
    cfg = _make_vllm_config(tp_size=tp, is_producer=producer)
    caches = KVCacheConfig(
        num_blocks=256, kv_cache_tensors=[], kv_cache_groups=cache_groups()
    )
    with (
        patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=rank),
        patch(f"{_MOD}.get_tensor_model_parallel_world_size", return_value=tp),
        patch(f"{_MOD}.dist_utils.get_node_id", return_value=0),
        patch(f"{_MOD}.dist_utils.get_host_ip", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_transfer_port", return_value=9100),
    ):
        obj = Qwen4ExpRaidenConnector(cfg, KVConnectorRole.WORKER, caches)
    worker = obj.connector_worker
    assert isinstance(worker, Qwen4ExpRaidenWorker)
    groups = build_transfer_groups(caches.kv_cache_groups)
    # Admission boundary double: transport geometry, not TPU layout evidence.
    pools = [
        PoolEntry(
            tag=f"g{g.group_index}.{name}",
            layer_name=name,
            storage_index=0,
            base_offset_bytes=0,
            block_stride_bytes=4096,
            num_blocks=256,
            regions=(RegionSpec("page", 0, 4096, 4096, 1),),
            dtype_tag="test-page",
        )
        for g in groups
        for name in g.layer_names
    ]
    worker._qwen_page_plan = QwenPagePlan(
        groups, PoolManifest(BINDING_ALIASED_RAW, [], pools)
    )
    worker._raiden_transfer_engine = _FakeRaidenEngine()
    worker._raiden_controller_facade = _FakeRaidenControllerFacade()
    worker._raiden_controller_address = "source:27000"
    worker._raiden_work_unit = SimpleNamespace(data_name="qwen")
    return obj if as_connector else worker


def send_pages(num_tokens=129):
    count = (num_tokens + 127) // 128
    return SimpleNamespace(
        uuid=7,
        num_tokens=num_tokens,
        expiration_time=time.perf_counter() + 300,
        local_block_ids=list(range(1, count + 1)),
        group_block_ids=[
            list(range(1, count + 1)),
            list(range(70, 70 + count)),
            [140],
            [141],
            [142],
        ],
    )


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 1), (4, 2), (8, 7)])
def test_qwen_producer_dispatch_registers_full_pages_to_matching_tp(
    monkeypatch, tp, rank
):
    worker = qwen_worker(monkeypatch, tp=tp, rank=rank)
    metadata = TPUConnectorMetadata()
    metadata.reqs_to_send["qwen"] = send_pages()
    worker.process_send_load(metadata)
    worker.process_send_load(metadata)
    calls = worker._raiden_controller_facade.register_request_blocks_calls
    assert len(calls) == 1
    registrations = calls[0]["pool_spans"]
    assert [len(r.block_ids) for r in registrations] == [2, 2, 1, 1, 1]
    assert sum(r.declared_bytes for r in registrations) == 7 * 4096
    from tpu_sync.api.torch.reshard_client import _span_entry_dict

    for ordinal, reg in enumerate(registrations):
        token_pages = ordinal < 2
        assert reg.dst_space_version == int(token_pages)
        for index, span in enumerate(reg.spans):
            assert span.dst_unit_ordinal == rank
            assert span.src_block_ordinal == index
            assert span.dst_block_index == 0
            assert span.src_offset_bytes == 0
            assert span.dst_offset_bytes == (index * 4096 if token_pages else 0)
            assert span.size_bytes == 4096
        wire = _span_entry_dict(reg)
        assert wire == dict(
            tag=reg.tag,
            block_ids=list(reg.block_ids),
            declared_bytes=len(reg.block_ids) * 4096,
            dst_space_version=int(token_pages),
            spans=[
                (index, 0, 0, index * 4096 if token_pages else 0, 4096, 0, 0, 1, rank)
                for index in range(len(reg.block_ids))
            ],
        )


@pytest.mark.parametrize("suppressed", [False, True])
@pytest.mark.parametrize("cleanup", [False, True])
def test_multigroup_failed_receive_is_request_scoped(monkeypatch, suppressed, cleanup):
    from vllm import SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

    obj = qwen_worker(monkeypatch, producer=False, as_connector=True)
    worker = obj.connector_worker
    worker._raiden_transfer_engine.poll_results = [([], [], [])]
    worker._record_stage3_load_failure("failed", [5])
    if suppressed:
        worker._suppress_done_recving.add("failed")
    result = obj.get_transfer_results({"failed"} if cleanup else set())
    assert result.failed_recving == {"failed"}
    assert result.finished_recving == {"failed"}
    assert obj.get_block_ids_with_load_errors() == set()
    assert obj.get_transfer_results(set()).failed_recving == set()
    scheduler = object.__new__(Scheduler)
    request = Request(
        request_id="failed",
        prompt_token_ids=[1] * 17,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = 17
    scheduler.requests = {"failed": request}
    scheduler.recompute_kv_load_failures = False
    assert scheduler._handle_failed_recving(result.failed_recving) == {"failed"}
    assert request.num_computed_tokens == 17


@pytest.mark.parametrize("with_state", [False, True])
def test_shared_failure_scoping_uses_all_cache_groups(with_state):
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector

    groups = cache_groups()
    caches = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[groups[0], groups[3]] if with_state else [groups[0]],
    )
    with (
        patch(f"{_MOD}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{_MOD}.get_tensor_model_parallel_world_size", return_value=1),
        patch(f"{_MOD}.dist_utils.get_node_id", return_value=0),
        patch(f"{_MOD}.dist_utils.get_host_ip", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_transfer_port", return_value=9100),
    ):
        obj = TPURaidenConnector(
            _make_vllm_config(is_producer=False), KVConnectorRole.WORKER, caches
        )
    assert len(obj._stage3_transfer_groups) == 1
    worker = obj.connector_worker
    worker._raiden_transfer_engine = _FakeRaidenEngine()
    worker._raiden_transfer_engine.poll_results = [([], [], [])]
    worker._record_stage3_load_failure("failed", [5])
    assert obj.get_transfer_results(set()).failed_recving == {"failed"}
    assert obj.get_block_ids_with_load_errors() == (set() if with_state else {5})


@pytest.mark.parametrize(
    "fault", ["missing", "capacity", "overlap", "state_extra", "primary", "unadmitted"]
)
def test_qwen_worker_rejects_invalid_pages_before_native_registration(
    monkeypatch, fault
):
    worker = qwen_worker(monkeypatch)
    send = send_pages()
    if fault == "missing":
        send.group_block_ids = None
    elif fault == "capacity":
        send.group_block_ids[2] = [256]
    elif fault == "overlap":
        send.group_block_ids[2] = [70]
    elif fault == "state_extra":
        send.group_block_ids[2] = [140, 143]
    elif fault == "primary":
        send.local_block_ids = [3, 4]
    else:
        worker._qwen_page_plan = None
    meta = TPUConnectorMetadata()
    meta.reqs_to_send["qwen"] = send
    with pytest.raises((ValueError, RuntimeError)):
        worker.process_send_load(meta)
    assert not worker._raiden_controller_facade.register_request_blocks_calls


@pytest.mark.parametrize("fault", [None, "tp", "prefix", "primary"])
def test_qwen_consumer_stages_every_tag_with_its_own_pages(monkeypatch, fault):
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import _Stage3PendingSubmit

    worker = qwen_worker(monkeypatch, producer=False)
    request = send_pages()
    request.src_controller_address = "source:27000"
    request.src_parallelism = 2 if fault == "tp" else 1
    request.skip_tokens = 1 if fault == "prefix" else 0
    request.mamba_state_block_ids = None
    if fault == "primary":
        request.local_block_ids = [3, 4]
    worker._stage3_source_facades["source:27000"] = worker._raiden_controller_facade
    monkeypatch.setattr(worker, "_stage3_source_work_units", lambda _: ["source"])
    monkeypatch.setattr(worker, "_raiden_hbm_memory_type", lambda: "HBM")
    pending = _Stage3PendingSubmit(
        req_meta=request,
        source_req_id="src",
        destination_req_id="dst",
        uuid=7,
        num_tokens=129,
        local_blocks=request.local_block_ids,
        dst_controller_address="dest:27000",
        dst_units=["dest"],
        defer_deadline=None,
        first_attempt_s=time.perf_counter(),
    )
    task = worker._stage_stage3_submit_task(pending, synchronous=True)
    if fault:
        assert task is None
    else:
        assert task.call_kwargs["dst_block_counts"] == [2, 2, 1, 1, 1]
        assert task.call_kwargs["dst_device_block_ids"] == [1, 2, 70, 71, 140, 141, 142]
        assert len(task.call_kwargs["transfer_pool_tags"]) == 5
        assert "dst_skip_bytes" not in task.call_kwargs


def test_qwen_terminal_cleanup_retains_all_group_identity(monkeypatch):
    worker = qwen_worker(monkeypatch)
    meta = TPUConnectorMetadata()
    meta.reqs_to_send["qwen"] = send_pages()
    worker.process_send_load(meta)
    engine = worker._raiden_transfer_engine
    engine.poll_results = [(["qwen"], [], [])]
    worker._poll_finished(engine)
    worker.process_send_load(meta)
    meta.reqs_to_send["qwen"].group_block_ids[2] = [143]
    with pytest.raises(ValueError, match="Conflicting replay"):
        worker.process_send_load(meta)
    assert len(worker._raiden_controller_facade.register_request_blocks_calls) == 1


@pytest.mark.parametrize("tokens", [1, 2, 3, 129, 8192])
def test_qwen_handoff_stops_before_decode_token(tokens, monkeypatch):
    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import ngram_context_from_store

    monkeypatch.setenv("TPU_RAIDEN_STAGE3_REGISTER_AHEAD", "1")
    obj = connector(_make_vllm_config())
    scheduler = obj.connector_scheduler
    assert not scheduler._stage3_register_ahead_enabled()
    prompt = list(range(10, 10 + tokens))
    request = SimpleNamespace(
        request_id="handoff",
        prompt_token_ids=prompt.copy(),
        _all_token_ids=prompt.copy(),
        num_prompt_tokens=tokens,
        kv_transfer_params=None,
    )
    scheduler.on_new_request(request)
    scheduler.on_new_request(request)
    committed = tokens - 1
    assert request.prompt_token_ids == (prompt[:-1] if committed else prompt)
    assert (
        scheduler._stage3_transfer_tokens(request, request.num_prompt_tokens)
        == committed
    )
    # D retains the full authoritative prompt. Its reordered row holds the
    # committed history; the uncomputed last token is not part of PLE history.
    store = torch.tensor([prompt[::-1], prompt], dtype=torch.int32)
    history = ngram_context_from_store(
        store, torch.tensor([1]), torch.tensor([committed]), 248044, 3
    )
    expected = ([248044, 248044] + prompt[:committed])[-2:]
    assert history.tolist() == [expected]
    assert store[1, committed].item() == prompt[-1]


@pytest.mark.parametrize("native_first", [False, True])
def test_qwen_loaded_requires_controller_and_native_completion(
    monkeypatch, native_first
):
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import _Stage3LoadMeta

    from .test_tpu_connector import _flush_stage3_submits

    worker = qwen_worker(monkeypatch, producer=False)
    engine = worker._raiden_transfer_engine
    engine.poll_results = []
    facade = worker._raiden_controller_facade
    entered, release = threading.Event(), threading.Event()

    def transfer(**kwargs):
        facade.start_transfer_calls.append(kwargs)
        entered.set()
        return release.wait(5)

    monkeypatch.setattr(facade, "start_transfer", transfer)
    worker._stage3_source_facades["source:27000"] = facade
    monkeypatch.setattr(worker, "_stage3_source_work_units", lambda _: ["source"])
    monkeypatch.setattr(worker, "_raiden_hbm_memory_type", lambda: "HBM")
    monkeypatch.setattr(
        worker, "_require_stage3_controller", lambda: (facade, "dest:27000")
    )
    pages = send_pages()
    meta = TPUConnectorMetadata()
    meta.reqs_to_load["dst"] = _Stage3LoadMeta(
        uuid=7,
        source_req_id="src",
        local_block_ids=pages.local_block_ids,
        num_tokens=129,
        src_controller_address="source:27000",
        src_job_name="source",
        src_engine_id="source-engine",
        src_data_replica_idx=0,
        src_parallelism=1,
        group_block_ids=pages.group_block_ids,
    )
    try:
        worker._submit_stage3_loads(meta, engine)
        assert entered.wait(2)
        worker._submit_stage3_loads(meta, engine)
        # A reused scheduler object must not change an already armed target.
        meta.reqs_to_load["dst"].group_block_ids[2][0] = 143
        with pytest.raises(ValueError, match="Conflicting grouped"):
            worker._submit_stage3_loads(meta, engine)
        assert worker._stage3_submitted_load_metas["dst"].group_block_ids[2] == [140]
        meta.reqs_to_load["dst"].group_block_ids[2][0] = 140
        if native_first:
            engine.poll_results = [([], ["src"], [])]
            assert worker.get_finished() == (set(), set())
            assert worker._stage3_deferred_native_failures == {"dst": False}
        release.set()
        _flush_stage3_submits(worker)
        if not native_first:
            assert worker.get_finished() == (set(), set())
            engine.poll_results = [([], ["src"], [])]
        assert worker.get_finished() == (set(), {"dst"})
        worker._submit_stage3_loads(meta, engine)
        meta.reqs_to_load["dst"].group_block_ids[2][0] = 143
        with pytest.raises(ValueError, match="Conflicting grouped"):
            worker._submit_stage3_loads(meta, engine)
        assert worker.get_finished() == (set(), set())
        assert worker.get_block_ids_with_load_errors() == set()
        assert len(facade.start_transfer_calls) == 1
        assert facade.start_transfer_calls[0]["dst_block_counts"] == [2, 2, 1, 1, 1]
    finally:
        release.set()
        _flush_stage3_submits(worker)


def _registered_qwen_pool(
    monkeypatch,
    *,
    fault=None,
    ssm_dtype: Literal["bfloat16", "float32"] = "bfloat16",
    tp=1,
    producer=True,
    scales=(0.5, 0.25),
    blocks=3,
    max_model_len=None,
    state_shape_change=False,
    feature=None,
    gather_fault=None,
    text_overrides=None,
):
    import torch

    from tests.runner.test_qwen4_exp_pd_layout import native_layout
    from tests.runner.test_qwen4_exp_pd_manifest import live_pool
    from vllm_torchtpu.models.vllm.qwen4_exp import pd_connector as module
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import measure_transfer_layout
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache

    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")
    worker = qwen_worker(monkeypatch, tp=tp, producer=producer)
    worker._qwen_page_plan = None
    worker._raiden_transfer_engine = None
    worker.vllm_config.parallel_config.enable_expert_parallel = False
    if max_model_len is not None:
        worker.vllm_config.model_config.max_model_len = max_model_len
    worker.vllm_config.cache_config.mamba_ssm_cache_dtype = ssm_dtype
    if feature == "apc":
        worker.vllm_config.cache_config.enable_prefix_caching = True
    elif feature == "mtp":
        worker.vllm_config.speculative_config = cast("SpeculativeConfig", object())
    elif feature == "enable_expert_parallel":
        worker.vllm_config.parallel_config.enable_expert_parallel = True
    elif feature == "dp_ep":
        worker.vllm_config.parallel_config.data_parallel_size = 2
        worker.vllm_config.parallel_config.enable_expert_parallel = True
    elif feature is not None:
        setattr(worker.vllm_config.parallel_config, feature, 2)
    text_overrides = text_overrides or {}
    config, materialized = live_pool(
        tp=tp, ssm=ssm_dtype, num_blocks=blocks, **text_overrides
    )
    from tests.runner.test_qwen4_exp_pool import pool_config
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    cache_config = pool_config(tp=tp, ssm=ssm_dtype)
    for key, value in text_overrides.items():
        setattr(cache_config.model_config.hf_text_config, key, value)
    cache = Qwen4ExpPooledCache.__new__(Qwen4ExpPooledCache)
    cache.native_config, cache.materialized = config, materialized
    cache.draft_model = None
    cache.layout, cache.specs = pooled_cache_specs(cache_config)
    if state_shape_change:
        for name, spec in cache.specs.items():
            if name.endswith(".linear_attn"):
                assert isinstance(spec, MambaSpec)
                conv, ssm = spec.shapes
                cache.specs[name] = replace(spec, shapes=(conv, ssm[::-1]))
    attention = SimpleNamespace(
        _k_scale_float=scales[0],
        _v_scale_float=scales[1],
        _k_scale_cpu=torch.tensor(scales[0]),
        _v_scale_cpu=torch.tensor(scales[1]),
    )
    cache.model = SimpleNamespace(
        checkpoint_path="/producer/checkpoint" if producer else "/consumer/checkpoint",
        config=SimpleNamespace(
            to_dict=lambda: pytest.fail("transfer admission read the full model config")
        ),
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_type="indexed_attention",
                    self_attn=attention,
                    ple=SimpleNamespace(
                        weight=torch.arange(8 if producer else 16),
                        owner=0 if producer else 1,
                    ),
                )
            ]
        ),
    )
    worker.vllm_config.additional_config = dict(
        qwen4_exp=dict(
            ple_host_backend="torch" if producer else "native_dma",
            engram_config=dict(
                cpu_offload=True,
                dp_shared_memory=producer,
                embedding_across_dp=not producer,
            ),
        )
    )
    if fault == "scale":
        attention._k_scale_cpu.fill_(0.75)
    adapter = Qwen4ExpAdapter.__new__(Qwen4ExpAdapter)
    adapter.cache = cache
    runner = SimpleNamespace(
        model_adapter=adapter,
        kv_cache_config=config,
        kv_cache_raw_tensors=materialized.raw_tensors,
    )
    worker.named_kv_caches = dict(materialized.kv_caches)
    if fault == "legacy_env":
        monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "0")
    if fault == "layout_dtype":
        original = materialized.raw_tensors[0]
        changed_dtype = (
            torch.bfloat16
            if original.dtype == torch.float8_e4m3fn
            else torch.float8_e4m3fn
        )
        changed = original.view(changed_dtype)
        assert changed.dtype != original.dtype
        assert changed.data_ptr() == original.data_ptr()
        caches = {
            name: [changed if tensor is original else tensor for tensor in value]
            if isinstance(value, list)
            else changed
            if value is original
            else value
            for name, value in materialized.kv_caches.items()
        }
        materialized = replace(
            materialized,
            raw_tensors=[
                changed if tensor is original else tensor
                for tensor in materialized.raw_tensors
            ],
            kv_caches=caches,
            layer_to_raw={
                name: changed if tensor is original else tensor
                for name, tensor in materialized.layer_to_raw.items()
            },
        )
        cache.materialized = materialized
        runner.kv_cache_raw_tensors = materialized.raw_tensors
        worker.named_kv_caches = dict(materialized.kv_caches)
    if fault == "binding":
        worker.named_kv_caches.pop(next(iter(worker.named_kv_caches)))
    monkeypatch.setattr(
        module,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=tp, cpu_group="cpu"),
    )

    def gather(output, value, *, group):
        assert group == "cpu"
        output[:] = [dict(value, rank=i, tp_rank=i) for i in range(tp)]
        if gather_fault == "missing":
            output[1] = None
        elif gather_fault == "rank":
            output[1]["rank"] = 0
        elif gather_fault == "tp":
            output[1]["tp_rank"] = tp
        elif gather_fault == "tp_duplicate":
            output[1]["tp_rank"] = 0
        elif gather_fault == "state":
            output[1]["cache_specs"] = {}

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(module, "materialize_storages", lambda _: None)
    monkeypatch.setattr(
        module,
        "measure_transfer_layout",
        lambda manifest: measure_transfer_layout(
            manifest,
            layout_getter=native_layout,
            package_version=lambda _: "test",
        ),
    )
    monkeypatch.setattr(worker, "_maybe_host_reshard_store", lambda: None)
    monkeypatch.setattr(
        worker, "_new_raiden_id", lambda fields: SimpleNamespace(**fields)
    )
    facade = _FakeRaidenControllerFacade()
    monkeypatch.setattr(worker, "_new_raiden_controller_facade", lambda _: facade)
    pools = []
    pool_registration = {}

    def register_pools(values, **kwargs):
        pools.extend(values)
        pool_registration.update(kwargs)
        return {}

    engine = SimpleNamespace(
        transfer_address="data:1",
        listener_address="control:2",
        register_pools=register_pools,
    )
    builds = []

    def construct(storages, *, num_slots):
        assert num_slots == 1
        assert all(a is b for a, b in zip(storages, materialized.raw_tensors))
        builds.append(storages)
        return engine

    monkeypatch.setattr(worker, "_construct_raiden_transfer_engine", construct)
    try:
        worker.register_runner(runner)
    except (ValueError, RuntimeError):
        assert not builds and not facade.register_work_unit_calls
        raise
    worker.register_runner(runner)
    assert len(builds) == len(facade.register_work_unit_calls) == 1
    assert len(pools) == len(materialized.kv_caches)
    assert worker._qwen_page_plan is not None
    assert worker._qwen_page_plan.manifest.storages[0] is materialized.raw_tensors[0]
    return SimpleNamespace(
        worker=worker,
        runner=runner,
        facade=facade,
        pool_registration=pool_registration,
    )


@pytest.mark.parametrize("producer", [False, True])
@pytest.mark.parametrize("extra_pages", [0, 2])
@pytest.mark.parametrize("leases", [1, 3])
def test_qwen_registration_bounds_staging_by_transferred_group_ids(
    monkeypatch, producer, extra_pages, leases
):
    from tests.runner.test_qwen4_exp_pd_manifest import live_pool
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import (
        TransferExtent,
        build_transfer_groups,
    )

    config, _ = live_pool(num_blocks=1)
    groups = build_transfer_groups(config.kv_cache_groups)
    page_tokens = min(
        group.block_size
        for group in groups
        if group.extent is TransferExtent.TOKEN_PAGES
    )
    max_tokens = extra_pages * page_tokens + 1
    blocks = 1 + sum(group.page_count(max_tokens) for group in groups)
    monkeypatch.setenv("TPU_RAIDEN_POOL_STAGING_LEASES", str(leases))
    registered = _registered_qwen_pool(
        monkeypatch, producer=producer, blocks=blocks, max_model_len=max_tokens
    )
    assert registered.pool_registration.get("staging_leases") == leases
    plan = registered.worker._page_plan()
    tables = []
    next_id = 1
    for group in plan.groups:
        count = group.page_count(max_tokens)
        tables.append(list(range(next_id, next_id + count)))
        next_id += count
    pages = plan.request_pages(tables, max_tokens)
    touched = {}
    for pool, group_index in plan.pool_groups():
        touched.setdefault(pool.storage_index, set()).update(pages[group_index])
    hints = registered.pool_registration["staging_blocks_per_pool"]
    assert hints == [len(touched[pool.storage_index]) for pool in plan.manifest.pools]
    assert all(0 < hint < blocks for hint in hints)


def test_qwen_registration_preserves_explicit_full_mirror_override(monkeypatch):
    monkeypatch.setenv("TPU_RAIDEN_POOL_STAGING_LEASES", "0")
    registered = _registered_qwen_pool(monkeypatch)
    assert registered.pool_registration == {}


def test_equal_carriers_with_different_ring_encoding_do_not_admit_as_peers(monkeypatch):
    peers = []
    for ratio in (4, 8):
        peer = _registered_qwen_pool(
            monkeypatch,
            text_overrides=dict(
                layer_types=["indexed_attention"],
                num_hidden_layers=1,
                ple_layer_ids=[],
                indexer_compress_ratio=ratio,
            ),
        )
        peers.append(peer.worker._raiden_layout_fingerprint_payload)
        ring = peer.runner.model_adapter.cache.layout
        assert ring.compressed_kernel_rows == 128 // ratio
    assert peers[0]["layout"] == peers[1]["layout"]
    from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
        canonical_layout_fingerprint,
    )

    assert canonical_layout_fingerprint(peers[0]) != canonical_layout_fingerprint(
        peers[1]
    )
    assert [
        payload["semantics"]["ranks"][0]["cache_specs"][
            "model.layers.0.self_attn.raw_ring"
        ]["block_size"]
        for payload in peers
    ] == [4, 8]


@pytest.mark.parametrize(
    "fault", [None, "scale", "legacy_env", "layout_dtype", "binding"]
)
@pytest.mark.parametrize("ssm_dtype", ["bfloat16", "float32"])
def test_qwen_register_runner_admits_actual_materialized_owners(
    monkeypatch, fault, ssm_dtype
):
    if fault not in (None, "legacy_env"):
        with pytest.raises((ValueError, RuntimeError)):
            _registered_qwen_pool(monkeypatch, fault=fault, ssm_dtype=ssm_dtype)
    else:
        fixture = _registered_qwen_pool(monkeypatch, fault=fault, ssm_dtype=ssm_dtype)
        payload = fixture.worker._raiden_layout_fingerprint_payload
        rank = payload["semantics"]["ranks"][0]
        assert set(rank) == {
            "rank",
            "tp_rank",
            "tp_size",
            "cache_specs",
            "kv_scales",
            "state_flags",
        }
        assert rank["state_flags"] == {
            "gdn_conv_dtype": "torch.float32",
            "gdn_granule_bytes": 512,
        }
        assert rank["kv_scales"] == [(0, 0.5, 0.25)]
        recurrent = [
            value
            for name, value in rank["cache_specs"].items()
            if name.endswith(".linear_attn")
        ]
        assert recurrent and all(
            value["dtypes"] == ["torch.float32", "torch." + ssm_dtype]
            for value in recurrent
        )


@pytest.mark.parametrize("tp", [1, 2])
def test_registered_peers_compare_cache_bytes_not_local_ple_or_checkpoint(
    monkeypatch, tp
):
    producer = _registered_qwen_pool(monkeypatch, tp=tp, producer=True, blocks=3)
    consumer = _registered_qwen_pool(monkeypatch, tp=tp, producer=False, blocks=5)
    assert (
        producer.worker.vllm_config.additional_config
        != consumer.worker.vllm_config.additional_config
    )
    pmodel = producer.runner.model_adapter.cache.model
    dmodel = consumer.runner.model_adapter.cache.model
    assert pmodel.checkpoint_path != dmodel.checkpoint_path
    assert (
        pmodel.model.layers[0].ple.weight.shape
        != dmodel.model.layers[0].ple.weight.shape
    )
    assert (
        producer.worker._raiden_layout_fingerprint
        == consumer.worker._raiden_layout_fingerprint
    )
    assert (
        producer.worker._raiden_layout_fingerprint_payload
        == consumer.worker._raiden_layout_fingerprint_payload
    )
    assert (
        producer.facade.register_work_unit_calls[0]["layout_fingerprint"]
        == consumer.facade.register_work_unit_calls[0]["layout_fingerprint"]
    )


@pytest.mark.parametrize("change", ["scale", "ssm_dtype", "state_shape", "tp"])
def test_registered_peers_keep_wire_encoding_and_scale_differences(monkeypatch, change):
    producer = _registered_qwen_pool(monkeypatch)
    if change == "scale":
        consumer = _registered_qwen_pool(
            monkeypatch, producer=False, scales=(0.75, 0.25)
        )
    elif change == "ssm_dtype":
        consumer = _registered_qwen_pool(
            monkeypatch, producer=False, ssm_dtype="float32"
        )
    elif change == "state_shape":
        consumer = _registered_qwen_pool(
            monkeypatch, producer=False, state_shape_change=True
        )
    else:
        assert change == "tp"
        consumer = _registered_qwen_pool(monkeypatch, producer=False, tp=2)
    assert (
        producer.worker._raiden_layout_fingerprint
        != consumer.worker._raiden_layout_fingerprint
    )


@pytest.mark.parametrize(
    "feature",
    [
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "pipeline_parallel_size",
        "enable_expert_parallel",
        "dp_ep",
        "apc",
        "mtp",
    ],
)
def test_feature_flags_do_not_fingerprint_an_unchanged_cache(monkeypatch, feature):
    producer = _registered_qwen_pool(monkeypatch)
    consumer = _registered_qwen_pool(monkeypatch, producer=False, feature=feature)
    assert (
        producer.worker._raiden_layout_fingerprint
        == consumer.worker._raiden_layout_fingerprint
    )


@pytest.mark.parametrize(
    "gather_fault", ["missing", "rank", "tp", "tp_duplicate", "state"]
)
def test_actual_collector_rejects_incomplete_or_inconsistent_tp_records(
    monkeypatch, gather_fault
):
    with pytest.raises(ValueError, match="every rank|cache encoding or TP map"):
        _registered_qwen_pool(monkeypatch, tp=2, gather_fault=gather_fault)


@pytest.mark.parametrize("size,rank,transfer", [(2, 0, 1), (2, 2, 2), (0, 0, 0)])
def test_whole_shard_span_mapping_rejects_an_invalid_transfer_rank_set(
    monkeypatch, size, rank, transfer
):
    worker = qwen_worker(monkeypatch)
    worker.tp_size, worker.tp_rank = size, rank
    with pytest.raises(ValueError, match="whole-shard spans"):
        worker._validate_stage3_transfer_parallelism(transfer)


@pytest.mark.parametrize("source_units,destination_units", [(2, 2), (2, 1), (1, 2)])
def test_whole_page_ordinal_mapping_matches_actual_destination_coverage(
    monkeypatch, source_units, destination_units
):
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
        SendMeta,
        _Stage3LoadMeta,
    )

    ordinals = set()
    for rank in range(source_units):
        producer = qwen_worker(monkeypatch, tp=source_units, rank=rank)
        producer.vllm_config.parallel_config.data_parallel_size = 8
        _, registrations = producer._stage3_grouped_spans(
            cast(SendMeta, send_pages()), 129
        )
        assert sum(reg.declared_bytes for reg in registrations) == 7 * 4096
        for ordinal, reg in enumerate(registrations):
            token_pages = ordinal < 2
            assert reg.dst_space_version == int(token_pages)
            for index, span in enumerate(reg.spans):
                assert span.src_offset_bytes == 0
                assert span.dst_block_index == 0
                assert span.dst_offset_bytes == (index * 4096 if token_pages else 0)
                assert span.size_bytes == 4096
                assert span.dst_unit_ordinal == rank
                ordinals.add(span.dst_unit_ordinal)
    consumer = qwen_worker(monkeypatch, tp=destination_units, producer=False)
    consumer.vllm_config.parallel_config.data_parallel_size = 4
    request = cast(_Stage3LoadMeta, send_pages())
    request.src_parallelism, request.skip_tokens = source_units, 0
    required_ordinals = set(range(destination_units))
    if ordinals != required_ordinals:
        with pytest.raises(ValueError, match="ordinal routing has no shard conversion"):
            consumer._stage3_grouped_destination_pages(request, 129)
    else:
        tags, pages, counts = consumer._stage3_grouped_destination_pages(request, 129)
        assert len(tags) == 5 and counts == [2, 2, 1, 1, 1]
        assert pages == [1, 2, 70, 71, 140, 141, 142]


@pytest.mark.parametrize("qwen", [False, True], ids=["dsv4", "qwen"])
@pytest.mark.parametrize("changed", [False, True])
def test_shared_consumer_replay_preserves_each_contract(monkeypatch, qwen, changed):
    from dataclasses import replace
    from unittest.mock import Mock

    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import _Stage3LoadMeta

    monkeypatch.setenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER", "1")
    monkeypatch.setenv("TPU_RAIDEN_DSV4_ADMISSION", "1" if not qwen else "0")
    if qwen:
        worker = qwen_worker(monkeypatch, producer=False)
    else:
        worker = _make_raiden_worker(
            tp_rank=0, tp_size=1, is_producer=False, dp_size=8, block_size=1024
        )
        worker._stage3_dsv4_plan, _ = _dsv4_worker_plan(worker)
        worker._raiden_controller_facade = _FakeRaidenControllerFacade()
        worker._raiden_controller_address = "destination:27000"
        worker._raiden_work_unit = SimpleNamespace(data_name="grouped")
    assert worker._uses_stage3_grouped_pages()
    worker._dispatch_stage3_load_submit = Mock()
    tables = [[value] for value in range(10, 60 if qwen else 80, 10)]
    first = _Stage3LoadMeta(
        uuid=99,
        num_tokens=128,
        local_block_ids=[10],
        source_req_id="source",
        src_controller_address="source:27000",
        src_job_name="prefill",
        src_engine_id="source-engine",
        src_data_replica_idx=0,
        src_parallelism=1,
        group_block_ids=tables,
    )
    meta = TPUConnectorMetadata()
    cast(dict[str, _Stage3LoadMeta], meta.reqs_to_load)["destination"] = first
    worker._submit_stage3_loads(meta, cast("KVCacheManager", _FakeRaidenEngine()))
    assert worker._dispatch_stage3_load_submit.call_count == 1
    submitted = worker._stage3_submitted_load_metas["destination"]
    replay_tables = [list(pages) for pages in tables]
    if changed:
        replay_tables[1] = [21]
    repeated = replace(first, group_block_ids=replay_tables)
    cast(dict[str, _Stage3LoadMeta], meta.reqs_to_load)["destination"] = repeated
    if qwen and changed:
        with pytest.raises(ValueError, match="Conflicting grouped"):
            worker._submit_stage3_loads(
                meta, cast("KVCacheManager", _FakeRaidenEngine())
            )
    else:
        worker._submit_stage3_loads(meta, cast("KVCacheManager", _FakeRaidenEngine()))
    assert worker._dispatch_stage3_load_submit.call_count == 1
    assert worker._stage3_submitted_load_metas["destination"] is submitted
    assert submitted.group_block_ids == tables


def test_non_grouped_replay_does_not_read_released_metadata(monkeypatch):
    from unittest.mock import Mock

    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import _Stage3LoadMeta

    worker = _make_raiden_worker(tp_rank=0, tp_size=1, is_producer=False)
    worker._raiden_controller_facade = _FakeRaidenControllerFacade()
    worker._raiden_controller_address = "destination:27000"
    worker._raiden_work_unit = SimpleNamespace(data_name="ordinary")
    worker._dispatch_stage3_load_submit = Mock()
    worker._stage3_submitted_loads["destination"] = 99
    worker._stage3_submitted_load_tokens["destination"] = 128
    assert not worker._uses_stage3_grouped_pages()
    assert not worker._stage3_submitted_load_metas
    meta = TPUConnectorMetadata()
    cast(dict[str, _Stage3LoadMeta], meta.reqs_to_load)["destination"] = (
        _Stage3LoadMeta(
            uuid=99,
            source_req_id="source",
            local_block_ids=[10],
            num_tokens=128,
            src_controller_address="source:27000",
            src_job_name="prefill",
            src_engine_id="source-engine",
            src_data_replica_idx=0,
            src_parallelism=1,
        )
    )
    worker._submit_stage3_loads(meta, cast("KVCacheManager", _FakeRaidenEngine()))
    worker._dispatch_stage3_load_submit.assert_not_called()
