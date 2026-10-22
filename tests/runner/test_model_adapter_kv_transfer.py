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
"""Model-owned caches must register the same live buffers after binding."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner import kv_cache_manager


@pytest.mark.parametrize("transfer_enabled", [False, True])
@pytest.mark.parametrize("runner_hooks", [False, True])
def test_adapter_cache_registration_order_and_identity(
    monkeypatch, transfer_enabled, runner_hooks
):
    events = []
    raw = torch.zeros(2, 4)
    caches = {"attention": raw, "ring": raw, "state": [raw]}
    config = object()
    runner = SimpleNamespace()

    class Adapter:
        def initialize_kv_cache(self, actual_config):
            assert actual_config is config
            runner.kv_caches = [raw]
            events.append("bound")
            return caches

    class Connector:
        def register_kv_caches(self, actual_caches):
            assert actual_caches is caches
            assert runner.kv_caches[0] is actual_caches["ring"]
            events.append("registered")

    class ConnectorWithHooks(Connector):
        def set_host_xfer_buffer_ops(self, copy_op):
            assert copy_op is kv_cache_manager.copy_kv_blocks
            events.append("copy")

        def register_runner(self, actual_runner):
            assert actual_runner is runner
            events.append("runner")

    connector = ConnectorWithHooks() if runner_hooks else Connector()

    def get_connector():
        assert transfer_enabled
        return connector

    monkeypatch.setattr(
        kv_cache_manager, "has_kv_transfer_group", lambda: transfer_enabled
    )
    monkeypatch.setattr(kv_cache_manager, "get_kv_transfer_group", get_connector)
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager.runner = runner
    manager.model_adapter = Adapter()
    manager.initialize_kv_cache(config)
    expected = ["bound"]
    if transfer_enabled:
        expected += ["registered"]
        if runner_hooks:
            expected += ["copy", "runner"]
    assert events == expected


@pytest.mark.parametrize("transfer_enabled", [False, True])
def test_missing_live_cache_mapping_is_rejected_only_for_transfer(
    monkeypatch, transfer_enabled
):
    monkeypatch.setattr(
        kv_cache_manager, "has_kv_transfer_group", lambda: transfer_enabled
    )
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager.runner = SimpleNamespace()
    manager.model_adapter = SimpleNamespace(initialize_kv_cache=lambda config: None)
    if transfer_enabled:
        with pytest.raises(NotImplementedError, match="live named buffers"):
            manager.initialize_kv_cache(object())
    else:
        manager.initialize_kv_cache(object())


@pytest.mark.parametrize("ssm_dtype", ["bfloat16", "float32"])
def test_qwen_pool_adapter_registers_materialized_carriers(monkeypatch, ssm_dtype):
    from tests.runner.test_qwen4_exp_compact_pool import compact_config
    from vllm_torchtpu.models.vllm.qwen4_exp import pooled_cache
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import build_transfer_manifest
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    cfg = compact_config(kv="fp8_e4m3", prefix=False, layers=1, length=128)
    cfg.cache_config.mamba_ssm_cache_dtype = ssm_dtype
    runner = SimpleNamespace(
        vllm_config=cfg,
        model_config=cfg.model_config,
        cache_config=cfg.cache_config,
        device=torch.device("cpu"),
        enforce_eager=True,
        may_reinitialize_input_batch=lambda *args: None,
        _precompile_mamba_state_seed_copies=lambda: None,
    )
    runner._build_mamba_copy_plan = TPUModelRunner._build_mamba_copy_plan.__get__(
        runner
    )
    bound = {}
    layers = [
        SimpleNamespace(
            layer_type="linear_attention",
            linear_attn=SimpleNamespace(kv_cache=None),
            ple=SimpleNamespace(bind_pool=lambda raw: bound.update(ple=raw)),
        ),
        SimpleNamespace(
            layer_type="full_attention",
            self_attn=SimpleNamespace(
                bind_pool=lambda main, index: bound.update(main=main, index=index)
            ),
            ple=None,
        ),
    ]
    # CPU fixture replaces only HBM discovery, metadata construction and device
    # precompilation; planning, allocation, binding and registration are real.
    monkeypatch.setattr(
        pooled_cache,
        "compute_hbm_budget",
        lambda *a, **kw: SimpleNamespace(available=1 << 40),
    )
    monkeypatch.setattr(pooled_cache, "AttentionMetadataBuilder", lambda *a, **kw: None)
    adapter = Qwen4ExpAdapter(runner)
    adapter.bind_model(SimpleNamespace(model=SimpleNamespace(layers=layers)))
    cache = adapter.cache
    cfg.cache_config.num_gpu_blocks_override = cache.pages
    cfg.cache_config.mamba_block_size = cfg.model_config.max_model_len
    cache.allocate()
    materialized = cache.materialized
    bound["ple"].fill_(7)
    history = bound["ple"].clone()
    events = []

    class Connector:
        def register_kv_caches(self, caches):
            assert caches is materialized.kv_caches
            assert caches["model.layers.1.self_attn"] is bound["main"]
            assert caches["model.layers.1.self_attn.compressed_k"] is bound["index"]
            assert caches["model.layers.1.self_attn.raw_ring"] is bound["index"]
            assert caches["model.layers.0.ple"][0] is bound["ple"]
            assert all(
                a is b
                for a, b in zip(runner.kv_cache_raw_tensors, materialized.raw_tensors)
            )
            manifest = build_transfer_manifest(cache.native_config, materialized)
            assert {p.layer_name for p in manifest.pools} == set(caches)
            events.append("caches")

        def register_runner(self, actual_runner):
            assert actual_runner is runner
            assert events == ["caches"]
            events.append("runner")

    monkeypatch.setattr(kv_cache_manager, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kv_cache_manager, "get_kv_transfer_group", Connector)
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager.runner, manager.model_adapter = runner, adapter
    manager.initialize_kv_cache(cache.native_config)
    assert events == ["caches", "runner"]
    assert torch.equal(bound["ple"], history)
    assert set(materialized.layer_to_raw) == {
        name for group in cache.get_transfer_groups() for name in group.layer_names
    }
