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
"""CPU contracts for the optional runner extension; no model or device loading."""

from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.runner import kv_cache_manager, model_adapters, tpu_runner
from vllm_torchtpu.runner.model_adapters import ModelInputSnapshot


@pytest.mark.parametrize("architecture", ["Qwen3ForCausalLM", "UnknownModel"])
def test_unregistered_model_uses_default_runner(architecture):
    assert (
        model_adapters.select_model_adapter(
            NS(model_config=NS(architecture=architecture))
        )
        is None
    )


@pytest.mark.parametrize("payload", ["tensor", "mapping", "none"])
def test_snapshot_field_contract_is_frozen(payload):
    context = {"tensor": torch.zeros(2, 2), "mapping": {"rows": (3, 4)}, "none": None}[
        payload
    ]
    ids = torch.arange(4)
    metadata = {"layer": object()}
    snapshot = ModelInputSnapshot(context, 3, 2, 4, metadata, ids)
    assert snapshot.step_context is context
    assert snapshot.attention_metadata is metadata and snapshot.input_ids is ids
    with pytest.raises(FrozenInstanceError):
        snapshot.row_start = 0
    with pytest.raises(FrozenInstanceError):
        snapshot.step_context = object()


@pytest.mark.parametrize("kwargs", [None, {}, {"model_input": 7}])
@pytest.mark.parametrize("with_intermediate", [False, True])
def test_forward_preserves_kwargs_and_existing_outputs(kwargs, with_intermediate):
    output, aux = torch.tensor([3]), torch.tensor([4])
    intermediate = object() if with_intermediate else None
    model = Mock(return_value=(output, aux))
    ids, positions = torch.tensor([1]), torch.tensor([0])
    before = None if kwargs is None else dict(kwargs)
    actual = tpu_runner.TPUModelRunner.forward_model(
        NS(model=model),
        ids,
        positions,
        intermediate_tensors=intermediate,
        model_kwargs=kwargs,
    )
    assert actual[0] is output and actual[1] is aux
    expected = dict(
        input_ids=ids, positions=positions, inputs_embeds=None, **(kwargs or {})
    )
    if with_intermediate:
        expected["intermediate_tensors"] = intermediate
    model.assert_called_once_with(**expected)
    assert kwargs == before


def test_cache_hooks_bypass_generic_spec_and_allocation(monkeypatch):
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    specs, config = {"owner": object()}, object()
    adapter = NS(
        get_kv_cache_spec=Mock(return_value=specs),
        initialize_kv_cache=Mock(),
    )
    manager.model_adapter = adapter
    monkeypatch.setattr(
        kv_cache_manager,
        "get_layers_from_vllm_config",
        Mock(side_effect=AssertionError("generic spec path")),
    )
    assert manager.get_kv_cache_spec() is specs
    # This object has no num_blocks: the adapter must run before generic validation.
    manager.initialize_kv_cache(config)
    adapter.get_kv_cache_spec.assert_called_once_with()
    adapter.initialize_kv_cache.assert_called_once_with(config)


def test_no_adapter_uses_generic_spec_and_unified_initialization(monkeypatch):
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager.model_adapter = None
    manager.runner = NS(
        vllm_config=NS(cache_config=NS(block_size=16, cache_dtype="auto")),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        _unified_kv_layout=True,
        _is_async_drafter=False,
        kv_cache_dtype=torch.bfloat16,
        device="cpu",
    )
    get_layers = Mock(return_value={})
    monkeypatch.setattr(kv_cache_manager, "get_layers_from_vllm_config", get_layers)
    monkeypatch.setattr(
        kv_cache_manager.TpuPlatform, "_find_non_ssm_backend", lambda cfg: None
    )
    assert manager.get_kv_cache_spec() == {}
    get_layers.assert_called_once()
    manager._maybe_add_kv_sharing_layers_to_kv_cache_groups = Mock()
    manager._initialize_unified_kv_cache = Mock()
    config = NS(num_blocks=1, kv_cache_groups=[])
    manager.initialize_kv_cache(config)
    allocated = manager.runner.kv_cache_config
    assert allocated is not config and allocated.num_blocks == 1
    assert manager.runner.empty_slot_mappings == {}
    manager._initialize_unified_kv_cache.assert_called_once_with(allocated)


def test_dummy_run_delegates_before_generic_device_work():
    events = []
    adapter = NS(
        dummy_run=Mock(side_effect=lambda *a, **kw: events.append("adapter") or 1.25)
    )
    assert (
        tpu_runner.TPUModelRunner._dummy_run(
            NS(
                model_adapter=adapter,
                _pp_wave=None,
                _wait_input_staging_fence=lambda: events.append("fence"),
            ),
            16,
            2,
            num_blocks=3,
            dp_lockstep=True,
            profile_prefix=5,
        )
        == 1.25
    )
    adapter.dummy_run.assert_called_once_with(16, 2, 5, dp_lockstep=True)
    assert events == ["fence", "adapter"]


@pytest.mark.parametrize("enabled", [False, True])
def test_load_selects_adapter_after_device_setup_before_weights(monkeypatch, enabled):
    events = []
    device = torch.device("cpu")
    model = torch.nn.Identity()
    runner = NS(
        kv_cache_manager=NS(),
        device_config=NS(),
        vllm_config=NS(),
        model_config=NS(quantization=None),
        load_config=object(),
        mesh=None,
        is_pooling_model=False,
        parallel_config=NS(pipeline_parallel_size=1),
        _pp_is_last=False,
        _is_async_drafter=False,
        _initialize_pallas_kernels=Mock(),
    )

    def bind(loaded):
        assert loaded is model and runner.model is model
        events.append("bind")

    adapter = NS(bind_model=Mock(side_effect=bind)) if enabled else None

    def select(actual):
        assert actual is runner
        assert runner.device is runner.device_config.device is device
        assert runner.vllm_config.device_config is runner.device_config
        events.append("select")
        return adapter

    def load(**kwargs):
        assert runner.model_adapter is runner.kv_cache_manager.model_adapter is adapter
        assert kwargs == dict(
            vllm_config=runner.vllm_config, model_config=runner.model_config
        )
        events.append("load")
        return model

    # Exercise the real orchestration with CPU device/model collaborators.
    monkeypatch.setattr(
        tpu_runner, "torch", NS(device=Mock(return_value=device), nn=torch.nn)
    )
    monkeypatch.setattr(
        model_adapters, "select_model_adapter", Mock(side_effect=select)
    )
    loader = NS(load_model=Mock(side_effect=load))
    monkeypatch.setattr(tpu_runner, "get_model_loader", Mock(return_value=loader))
    monkeypatch.setattr(tpu_runner, "get_tpu_quantization_config", lambda cfg: None)
    monkeypatch.setattr(tpu_runner, "validate_online_fp8", Mock())
    for name in ("set_vllm_model_wrapper_context", "set_current_vllm_config"):
        monkeypatch.setattr(tpu_runner, name, lambda *a, **kw: nullcontext())
    for name in (
        "TPU_ROPE_CACHE_TRUNCATE",
        "TPU_ROPE_CACHE_ROW_MAJOR",
        "TPU_MOE_HASH_TABLE_ROW_MAJOR",
        "TPU_GREEDY_TOP_TOKENS",
        "FUSED_MOE_DECODE_KERNEL",
        "TPU_MOE_SKIP_PADDED_TOKENS",
        "USE_MOE_TP_DECODE_KERNEL",
    ):
        monkeypatch.setattr(tpu_runner.envs, name, False)
    monkeypatch.setattr(
        "vllm_torchtpu.model_loader_patches.evict_loaded_weights_page_cache", Mock()
    )
    tpu_runner.TPUModelRunner.load_model(runner)
    model_adapters.select_model_adapter.assert_called_once_with(runner)
    tpu_runner.get_model_loader.assert_called_once_with(runner.load_config)
    loader.load_model.assert_called_once()
    runner._initialize_pallas_kernels.assert_called_once()
    assert runner.model_adapter is runner.kv_cache_manager.model_adapter is adapter
    assert events == (["select", "load", "bind"] if enabled else ["select", "load"])


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("payload", ["tensor", "mapping", "none"])
def test_execute_step_snapshots_after_reorder_and_forwards_substituted_ids(
    monkeypatch, enabled, payload
):
    class ForwardReached(Exception):
        pass

    events, snapshots, forwards = [], [], []
    context = {"tensor": torch.zeros(2, 2), "mapping": {"rows": (3, 4)}, "none": None}[
        payload
    ]
    input_ids = torch.arange(4, dtype=torch.int32)
    substituted = input_ids + 10
    metadata = {"layer": object()}
    kwargs = {"model_input": object()}

    def begin():
        events.append("begin")
        return context

    def prepare(snapshot):
        snapshots.append(snapshot)
        return kwargs

    def forward(**kw):
        forwards.append(kw)
        raise ForwardReached

    trace = Mock(return_value=nullcontext())
    trace.is_enabled.return_value = False
    monkeypatch.setattr(tpu_runner, "TraceAnnotation", trace)
    monkeypatch.setattr(
        tpu_runner, "set_forward_context", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(tpu_runner.dist_utils, "get_raiden_inline_load", lambda: False)
    runner = NS(
        execute_model_state=None,
        _pp_wave=None,
        _mtp_prefill_only=False,
        scheduler_config=NS(async_scheduling=False),
        _update_states=lambda s: events.append("update"),
        _reorder_batch_for_rpa=lambda s: (events.append("reorder") or (0, 0)),
        _count_input_chunks=lambda s: (1, 2),
        _dp_coordinated_step=lambda *a, **kw: (4, 1),
        supports_mm_inputs=False,
        phase_based_profiler=None,
        _is_async_drafter=False,
        _greedy_top_tokens_this_step=lambda: False,
        vllm_config=NS(),
        input_batch=NS(num_reqs=2),
        maybe_setup_kv_connector=lambda *a, **kw: None,
        _prepare_inputs=lambda *a: (metadata, None, 2, 2, 2, [], [], None, None, None),
        _flush_mamba_state_seed_copies=lambda: None,
        _reset_read_offsets_for_new_requests=lambda *a: None,
        _apply_async_token_substitution=lambda *a: substituted,
        input_ids=input_ids,
        _record_input_staging_fence=lambda: None,
        _pp_is_first=True,
        _get_model_inputs=lambda ids, mm: (ids, None),
        _token_padding_update=lambda n: None,
        _dp_num_tokens_across_dp=lambda n: None,
        mesh=None,
        _kv_cache_bundle=None,
        _stage_timing=NS(stage=lambda n: nullcontext()),
        position_ids=torch.arange(4),
        forward_model=forward,
        model_adapter=NS(begin_step=begin, prepare_inputs=prepare) if enabled else None,
    )
    scheduler = NS(total_num_scheduled_tokens=4, scheduled_encoder_inputs={})
    with pytest.raises(ForwardReached):
        tpu_runner.TPUModelRunner.execute_model(runner, scheduler)
    assert events == (
        ["update", "reorder", "begin"] if enabled else ["update", "reorder"]
    )
    assert len(forwards) == 1
    assert forwards[0]["input_ids"] is substituted
    assert forwards[0]["model_kwargs"] is (kwargs if enabled else None)
    if enabled:
        (snapshot,) = snapshots
        assert snapshot.step_context is context and snapshot.input_ids is substituted
        assert snapshot.attention_metadata is metadata
        assert (snapshot.row_start, snapshot.num_requests, snapshot.num_tokens) == (
            0,
            2,
            4,
        )
    else:
        assert not snapshots
