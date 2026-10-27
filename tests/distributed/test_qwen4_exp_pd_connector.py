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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import KVCacheConfig
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
def test_constructor_rejects_unimplemented_state_lifecycles(option):
    cfg = _make_vllm_config()
    if option == "apc":
        cfg.cache_config.enable_prefix_caching = True
    elif option == "mtp":
        cfg.speculative_config = object()
    else:
        name = {
            "pcp": "prefill_context_parallel_size",
            "dcp": "decode_context_parallel_size",
            "pp": "pipeline_parallel_size",
        }[option]
        setattr(cfg.parallel_config, name, 2)
    with pytest.raises(NotImplementedError, match="CP1/PP1"):
        connector(cfg)


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
    worker._register_stage3_request_blocks_dsv4(meta)
    if terminal:
        worker._poll_finished(engine)
        assert "request" not in worker._stage3_registered_sends
        assert worker._stage3_terminal_sends["request"].group_block_ids[1] == (20,)
    worker._register_stage3_request_blocks_dsv4(meta)
    assert len(facade.register_request_blocks_calls) == 1
    tables[1] = [21]
    with pytest.raises(ValueError, match="Conflicting"):
        worker._register_stage3_request_blocks_dsv4(meta)
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


def qwen_worker(monkeypatch, *, tp=1, rank=0, producer=True):
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
    return worker


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
    for reg in registrations:
        assert reg.dst_space_version == 0
        for index, span in enumerate(reg.spans):
            assert span.dst_unit_ordinal == rank
            assert span.src_block_ordinal == span.dst_block_index == index
            assert span.src_offset_bytes == span.dst_offset_bytes == 0
            assert span.size_bytes == 4096


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


@pytest.mark.parametrize(
    "fault", [None, "checkpoint", "scale", "layout_flag", "binding"]
)
@pytest.mark.parametrize("ssm_dtype", ["bfloat16", "float32"])
def test_qwen_register_runner_admits_actual_materialized_owners(
    monkeypatch, fault, ssm_dtype
):
    import torch

    from tests.runner.test_qwen4_exp_pd_layout import native_layout
    from tests.runner.test_qwen4_exp_pd_manifest import live_pool
    from vllm_torchtpu.models.vllm.qwen4_exp import pd_connector as module
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import measure_transfer_layout
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache

    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")
    worker = qwen_worker(monkeypatch)
    worker._qwen_page_plan = None
    worker._raiden_transfer_engine = None
    worker.vllm_config.parallel_config.enable_expert_parallel = False
    worker.vllm_config.cache_config.mamba_ssm_cache_dtype = ssm_dtype
    config, materialized = live_pool(tp=8, ssm=ssm_dtype)
    cache = Qwen4ExpPooledCache.__new__(Qwen4ExpPooledCache)
    cache.native_config, cache.materialized = config, materialized
    attention = SimpleNamespace(
        _k_scale_float=0.5,
        _v_scale_float=0.25,
        _k_scale_cpu=torch.tensor(0.5),
        _v_scale_cpu=torch.tensor(0.25),
    )
    cache.model = SimpleNamespace(
        pd_loaded_weight_identity=None if fault == "checkpoint" else "a" * 64,
        pd_loaded_ple_identity="b" * 64,
        pd_gdn_qk_pair_layout=fault != "layout_flag",
        config=SimpleNamespace(to_dict=lambda: {"hidden_size": 16}),
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_type="full_attention",
                    self_attn=attention,
                )
            ]
        ),
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
    if fault == "binding":
        worker.named_kv_caches.pop(next(iter(worker.named_kv_caches)))
    monkeypatch.setattr(
        module,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1, cpu_group="cpu"),
    )

    def gather(output, value, *, group):
        assert group == "cpu"
        output[:] = [value]

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
    engine = SimpleNamespace(
        transfer_address="data:1",
        listener_address="control:2",
        register_pools=lambda values: pools.extend(values) or {},
    )
    builds = []

    def construct(storages, *, num_slots):
        assert num_slots == 1
        assert all(a is b for a, b in zip(storages, materialized.raw_tensors))
        builds.append(storages)
        return engine

    monkeypatch.setattr(worker, "_construct_raiden_transfer_engine", construct)
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            worker.register_runner(runner)
        assert not builds and not facade.register_work_unit_calls
    else:
        worker.register_runner(runner)
        worker.register_runner(runner)
        assert len(builds) == len(facade.register_work_unit_calls) == 1
        from vllm_torchtpu.models.vllm.qwen4_exp.pd import TransferExtent

        token_sizes = {
            g.block_size
            for g in cache.get_transfer_groups()
            if g.extent is TransferExtent.TOKEN_PAGES
        }
        assert token_sizes == {facade.register_work_unit_calls[0]["page_tokens"]}
        assert len(pools) == len(materialized.kv_caches)
        assert (
            worker._qwen_page_plan.manifest.storages[0] is materialized.raw_tensors[0]
        )
        payload = worker._raiden_layout_fingerprint_payload
        assert payload["semantics"]["ranks"][0]["state_flags"]["ssm_dtype"] == ssm_dtype
        assert payload["semantics"]["ranks"][0]["checkpoint"] == "a" * 64
        assert payload["semantics"]["ranks"][0]["kv_scales"] == [(0, 0.5, 0.25)]
        assert (
            "moe_combine_dtype" not in payload["semantics"]["ranks"][0]["state_flags"]
        )
