# SPDX-License-Identifier: Apache-2.0
"""Real connector methods with only network/device construction replaced."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer import tpu_connector as tc
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol

from .raiden_test_utils import FakeTensor
from .test_tpu_connector import (
    _FakeRaidenControllerFacade,
    _make_raiden_scheduler,
    _make_raiden_worker,
)


def _environment(monkeypatch, pcp=4, tp=2, dtp=8):
    monkeypatch.setattr(tc, "_raiden_seq_on_lane_layout", lambda: True)
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setenv("TPU_RAIDEN_TRANSFER_PARALLELISM", str(pcp * tp))
    monkeypatch.setenv("TPU_RAIDEN_DST_SHARDS", str(dtp))
    monkeypatch.setenv("TPU_RAIDEN_CONTROLLER_ADDRESS", "localhost:27000")
    monkeypatch.setenv("TPU_RAIDEN_ENGINE_ID", "producer")


def _manifest(heads):
    geometry = sol.SolFaGeometry(2 * heads, 64, 128)
    stride = 384 * geometry.bytes_per_token
    conv = (
        rpm.RegionSpec("gdn_conv_qk", 0, 1536, 512, 3),
        rpm.RegionSpec("gdn_conv_v", 512, 1536, 256, 3, 4),
    )
    ssm = (rpm.RegionSpec("gdn_ssm", 0, 32768, 32768, 4),)
    return rpm.PoolManifest(
        binding=rpm.BINDING_ALIASED_RAW,
        storages=[FakeTensor((48, 2 * heads, 64, 4, 128), 1)],
        pools=[
            rpm.PoolEntry(
                "fa",
                "fa.0",
                0,
                0,
                stride,
                16,
                sol.sol_fa_regions(block_size_tokens=384, geometry=geometry),
                "float8_e4m3fn",
            ),
            rpm.PoolEntry(
                "gdn.conv.g0", "gdn.0", 0, 131072, stride, 16, conv, "bfloat16"
            ),
            rpm.PoolEntry("gdn.ssm.g0", "gdn.0", 0, 0, stride, 16, ssm, "bfloat16"),
        ],
    )


@pytest.mark.parametrize("pcp,tp,dtp,dcp,heads", [(4, 2, 8, 8, 1), (16, 2, 32, 8, 4)])
@pytest.mark.parametrize("tokens", [127, 8193])
def test_producer_registers_fa_and_state_with_separate_coordinates(
    monkeypatch, pcp, tp, dtp, dcp, heads, tokens
):
    _environment(monkeypatch, pcp, tp, dtp)
    for p in range(pcp):
        for t in range(tp):
            worker = _make_raiden_worker(
                tp_rank=t, tp_size=tp, pcp_size=pcp, block_size=384, interleave_size=128
            )
            worker.vllm_config.kv_transfer_config.kv_connector_extra_config = {
                "destination_decode_context_parallel_size": dcp
            }
            worker.vllm_config.model_config.hf_config.num_key_value_heads = heads
            worker._raiden_manifest = _manifest(max(1, heads // tp))
            worker._raiden_layout_fingerprint_payload = {
                "minor_to_major": [4, 3, 2, 1, 0],
                "tiles": [[4, 128], [4, 1]],
                "element_size_in_bits": 8,
                "fa_kv_layout": sol.SOL_LAYOUT_TOKEN,
                "fa_kernel_page_tokens": 128,
            }
            worker._stage3_state_group_count = 1
            facade = _FakeRaidenControllerFacade()
            worker._raiden_controller_facade = facade
            worker._raiden_controller_address = "localhost:27000"
            worker._raiden_work_unit = SimpleNamespace(
                job_name="prefill", rank=p * tp + t
            )
            monkeypatch.setattr(
                worker, "_local_raiden_transfer_rank", lambda p=p, t=t: p * tp + t
            )
            meta = tc.TPUConnectorMetadata()
            ids = [3 + i * 2 for i in range(math.ceil(tokens / (384 * pcp)))]
            meta.reqs_to_send["request"] = SimpleNamespace(
                uuid=123,
                local_block_ids=ids,
                num_tokens=tokens,
                expiration_time=1e20,
                mamba_state_block_ids=[11],
            )
            worker._validate_stage3_transfer_parallelism(pcp * tp)
            worker._register_stage3_request_blocks(meta)
            reg = facade.register_request_blocks_calls[0]
            state = [s for s in reg["pool_spans"] if s.tag.startswith("gdn.")]
            assert len(state) == 2
            assert {s.dst_unit_ordinal for r in state for s in r.spans} == {t * pcp + p}
            fa = [s for s in reg["pool_spans"] if s.tag == "fa"]
            if heads == 1 and t == 1:
                assert not fa  # Replicated FA has one canonical sender.
            for r in fa:
                for s in r.spans:
                    local_page = s.dst_offset_bytes // (2 * 256 * 128)
                    global_page = local_page * dcp + s.dst_unit_ordinal % dcp
                    assert global_page % pcp == p
            # Empty FA ownership does not make a GDN contributor terminal.
            assert "request" not in worker._done_sending
            assert "request" in worker._stage3_registered_sends


@pytest.mark.parametrize("tokens", [8191, 32767])
def test_producer_metadata_uses_worker_count_but_pcp_block_extent(monkeypatch, tokens):
    _environment(monkeypatch)
    scheduler = _make_raiden_scheduler(
        is_producer=True, tp_size=2, pcp_size=4, block_size=384
    )
    scheduler.vllm_config.parallel_config.cp_kv_cache_interleave_size = 128
    scheduler.vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "destination_decode_context_parallel_size": 8
    }
    request = MagicMock(
        request_id="request",
        num_computed_tokens=tokens + 1,
        num_prompt_tokens=tokens + 1,
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        kv_transfer_params={},
    )
    ids = list(range(math.ceil(tokens / (384 * 4))))
    delay, params = scheduler.request_finished(request, ids)
    assert delay
    assert params["src_parallelism"] == 8
    assert params["dst_tp_size"] == params["dst_dcp_size"] == 8
    assert params["num_tokens"] == tokens
    assert scheduler.reqs_to_send["request"].local_block_ids == ids


@pytest.mark.parametrize("dcp", [2, 8])
def test_consumer_allocations_use_dcp_logical_block_size(monkeypatch, dcp):
    _environment(monkeypatch)
    scheduler = _make_raiden_scheduler(is_producer=False, tp_size=8, block_size=384)
    scheduler.vllm_config.parallel_config.decode_context_parallel_size = dcp
    request = MagicMock(
        request_id="request",
        num_prompt_tokens=8193,
        kv_transfer_params={
            "req_id": "producer-request",
            "uuid": 123,
            "num_tokens": 8193,
            "src_controller_address": "localhost:27000",
            "src_job_name": "prefill",
            "src_engine_id": "producer",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
            "dst_tp_size": 8,
            "dst_dcp_size": dcp,
        },
    )
    blocks = MagicMock()
    ids = list(range(math.ceil(8193 / (384 * dcp))))
    blocks.get_block_ids.return_value = (ids,)
    scheduler.update_state_after_alloc(request, blocks, 8193)
    assert scheduler.reqs_to_load["request"].local_block_ids == ids
    assert scheduler.reqs_to_load["request"].num_tokens == 8193


@pytest.mark.parametrize(
    "field,value",
    [
        ("dst_dcp_size", 2),
        ("dst_tp_size", 4),
        ("dst_tp_size", None),
        ("dst_dcp_size", None),
        ("dst_tp_size", "invalid"),
        ("dst_dcp_size", True),
        ("dst_dcp_size", 8.5),
    ],
)
def test_consumer_rejects_invalid_geometry_before_transfer(monkeypatch, field, value):
    _environment(monkeypatch)
    scheduler = _make_raiden_scheduler(is_producer=False, tp_size=8, block_size=384)
    scheduler.vllm_config.parallel_config.decode_context_parallel_size = 8
    request = MagicMock(
        request_id="request",
        num_prompt_tokens=1024,
        kv_transfer_params={
            "req_id": "p",
            "uuid": 1,
            "num_tokens": 1024,
            "src_controller_address": "localhost:27000",
            "src_job_name": "prefill",
            "src_engine_id": "producer",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
            "dst_tp_size": 8,
            "dst_dcp_size": 8,
        },
    )
    if value is None:
        del request.kv_transfer_params[field]
    else:
        request.kv_transfer_params[field] = value
    blocks = MagicMock()
    scheduler.update_state_after_alloc(request, blocks, 1024)
    blocks.get_block_ids.assert_not_called()
    assert "destination" in request.kv_transfer_params[tc._KV_PARAMS_REJECTED]
    assert "geometry" in request.kv_transfer_params[tc._KV_PARAMS_REJECTED]
    assert scheduler.reqs_to_load["request"].release_only
    assert not scheduler._stage3_active_source_req_ids


@pytest.mark.parametrize("tp,dcp", [(8, 8), (8, 2), (2, 1)])
def test_consumer_prefix_clip_uses_local_heads_and_dcp_tokens(monkeypatch, tp, dcp):
    _environment(monkeypatch, dtp=tp)
    worker = _make_raiden_worker(
        is_producer=False, tp_rank=0, tp_size=tp, block_size=384
    )
    worker.vllm_config.parallel_config.decode_context_parallel_size = dcp
    worker._raiden_manifest = _manifest(1)
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
    req = SimpleNamespace(
        src_controller_address="localhost:27000", skip_tokens=384 * dcp
    )
    outcome = worker._submit_stage3_load_as_leader(
        req_meta=req,
        source_req_id="p",
        destination_req_id="d",
        uuid=123,
        num_tokens=384 * dcp + 1,
        local_blocks=[7],
        dst_controller_address="localhost:27001",
        dst_units=[worker._raiden_work_unit],
    )
    assert outcome[0] == "accepted", outcome
    assert facade.start_transfer.call_args.kwargs["dst_skip_bytes"] == [384 * 512]


@pytest.mark.parametrize(
    "producer,pcp,tp,dcp,name",
    [
        (True, 4, 2, 1, "pcp4tp2_prefill"),
        (True, 16, 2, 1, "pcp16tp2_prefill"),
        (False, 1, 8, 8, "tp8dcp8_decode"),
        (False, 1, 32, 8, "tp32dcp8_decode"),
    ],
)
def test_hnd_pd_topology_admission(monkeypatch, producer, pcp, tp, dcp, name):
    _environment(monkeypatch)
    worker = _make_raiden_worker(
        is_producer=producer, pcp_size=pcp, tp_size=tp, tp_rank=0, dp_size=1
    )
    worker.vllm_config.parallel_config.decode_context_parallel_size = dcp
    assert worker._raiden_qwen35_admission_topology() == name
    # Ports remain distinct even when PCP peers share the same TP rank.
    ranks = set()
    for rank in range(pcp * tp):
        worker.tp_rank = rank % tp
        monkeypatch.setattr(
            worker, "_local_raiden_transfer_rank", lambda rank=rank: rank
        )
        endpoint, _ = worker._raiden_endpoint_identity()
        ranks.add(worker._rank_control_port(9100, rank=endpoint))
    assert len(ranks) == pcp * tp


@pytest.mark.parametrize("layout,dcp", [("NHD", 8), ("HND", 3)])
def test_pd_topology_rejects_invalid_layout_or_dcp(monkeypatch, layout, dcp):
    _environment(monkeypatch)
    monkeypatch.setattr(tc, "_raiden_seq_on_lane_layout", lambda: layout == "HND")
    worker = _make_raiden_worker(is_producer=False, tp_size=8, tp_rank=0, dp_size=1)
    worker.vllm_config.parallel_config.decode_context_parallel_size = dcp
    with pytest.raises(ValueError):
        worker._raiden_qwen35_admission_topology()


@pytest.mark.parametrize(
    "pcp,tp,dtp,k,v,reason",
    [
        (4, 2, 4, 4, 32, "Q/K replication"),
        (1, 1, 8, 4, 32, "Q/K replication"),
        (2, 2, 8, 16, 32, "producer parallelism"),
    ],
)
def test_gdn_unsupported_resharding_fails_before_store_or_pool_registration(
    monkeypatch, pcp, tp, dtp, k, v, reason
):
    _environment(monkeypatch, pcp=pcp, tp=tp, dtp=dtp)
    worker = _make_raiden_worker(
        pcp_size=pcp, tp_size=tp, tp_rank=0, dp_size=4 if pcp * tp == 1 else 1
    )
    config = worker.vllm_config.model_config.hf_config
    config.linear_num_key_heads = k
    config.linear_num_value_heads = v
    host_store = MagicMock()
    construct_engine = MagicMock()
    monkeypatch.setattr(worker, "_maybe_host_reshard_store", host_store)
    monkeypatch.setattr(worker, "_construct_raiden_transfer_engine", construct_engine)
    with pytest.raises(ValueError, match=reason):
        worker._admit_raiden_hybrid_kv_cache(MagicMock())
    host_store.assert_not_called()
    construct_engine.assert_not_called()


@pytest.mark.parametrize(
    "pcp,tp,dtp,k,v",
    [
        (4, 2, 8, 4, 32),  # Scaled model with Q/K replication on both sides.
        (16, 2, 32, 16, 128),  # Full target geometry.
        (4, 2, 4, 8, 32),  # Fan-in without Q/K replication remains valid.
        (8, 1, 2, 16, 32),  # Existing PCP -> TP2.
        (1, 1, 8, 16, 32),  # Existing full-state producer fan-out.
        (1, 1, 1, 16, 32),
    ],
)
def test_gdn_supported_transfer_geometry(monkeypatch, pcp, tp, dtp, k, v):
    _environment(monkeypatch, pcp=pcp, tp=tp, dtp=dtp)
    worker = _make_raiden_worker(pcp_size=pcp, tp_size=tp, tp_rank=0)
    config = worker.vllm_config.model_config.hf_config
    config.linear_num_key_heads = k
    config.linear_num_value_heads = v
    worker._validate_gdn_transfer_geometry(worker._gdn_head_geometry())
