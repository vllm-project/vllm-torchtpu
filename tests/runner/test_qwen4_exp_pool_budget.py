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
"""Physical capacity selection and real native cache ownership, without TPU."""

import copy
from types import SimpleNamespace as NS

import pytest

from tests.runner.test_qwen4_exp_layout_lifecycle import (
    configuration,
    worker_and_adapter,
)
from vllm_torchtpu.executors.tpu_multiproc_executor import TpuMultiprocExecutor
from vllm_torchtpu.models.vllm.qwen4_exp import pooled_cache
from vllm_torchtpu.models.vllm.qwen4_exp.config import native_cache_capacity
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("override", [None, "fits", "too_large", "too_small"])
def test_real_worker_selects_budgeted_capacity_before_profile_and_freezes_owners(
    monkeypatch, prefix, override
):
    cfg = configuration(prefix, "fp8_e4m3", 1, 128, 128)
    layout, _ = pooled_cache_specs(cfg)
    cfg.model_config.max_model_len = 8 * layout.block_size + 1
    layout, specs = pooled_cache_specs(cfg)
    ceiling = native_cache_capacity(cfg, specs)
    from vllm.v1.core.kv_cache_planning import (
        get_kv_cache_groups,
    )

    groups = get_kv_cache_groups(cfg, specs)
    bytes_per_block = layout.page_bytes * max(len(g.layer_names) for g in groups)
    budget_pages = ceiling - 1
    cfg.cache_config.num_gpu_blocks_override = {
        None: None,
        "fits": budget_pages,
        "too_large": ceiling,
        "too_small": 1,
    }[override]
    monkeypatch.setattr(
        pooled_cache,
        "compute_hbm_budget",
        lambda *_: NS(
            available=budget_pages * bytes_per_block + pooled_cache.TEMP_HEADROOM
        ),
    )
    worker, adapter, _ = worker_and_adapter(cfg, [])
    if override in ("too_large", "too_small"):
        with pytest.raises(ValueError, match="budget|memory"):
            worker.get_kv_cache_spec()
        assert adapter.cache.materialized is None
        return
    worker.get_kv_cache_spec()
    cache = adapter.cache
    assert cache.pages == worker.get_num_gpu_blocks_override() == budget_pages
    driver = copy.deepcopy(cfg)
    driver.cache_config.num_gpu_blocks_override = (
        None if override is None else budget_pages
    )
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor.vllm_config = driver
    monkeypatch.setattr(
        executor,
        "collective_rpc",
        lambda method, *_args, **_kwargs: [
            worker.get_kv_cache_spec()
            if method == "get_kv_cache_spec"
            else worker.get_num_gpu_blocks_override()
        ],
    )
    assert executor.get_kv_cache_specs() == [cache.specs]
    assert driver.cache_config.num_gpu_blocks_override == budget_pages
    cache.allocate()
    assert cache.native_config.num_blocks == budget_pages
    assert cache.native_config.kv_cache_groups == cache.groups
    storages = {
        raw.untyped_storage()._cdata: raw.untyped_storage().nbytes()
        for raw in cache.materialized.raw_tensors
    }
    assert sum(storages.values()) == budget_pages * bytes_per_block
    assert {t.size for t in cache.native_config.kv_cache_tensors} == {
        sum(storages.values())
    }
    owners = [id(raw) for raw in cache.materialized.raw_tensors]
    # The existing allocation consumes HBM; another query must reuse its plan.
    monkeypatch.setattr(
        pooled_cache,
        "compute_hbm_budget",
        lambda *_: (_ for _ in ()).throw(AssertionError("Rebudgeted allocated cache")),
    )
    assert worker.get_kv_cache_spec() == cache.specs
    assert cache.pages == budget_pages
    assert owners == [id(raw) for raw in cache.materialized.raw_tensors]
    assert cfg.model_config.max_model_len == 8 * layout.block_size + 1
    assert cfg.scheduler_config.max_num_seqs == 2


def test_budgeted_cache_uses_existing_allocator_admission_and_reuse(monkeypatch):
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.request import Request

    cfg = configuration(True, "fp8_e4m3", 1, 128, 128)
    cfg.model_config.hf_text_config.layer_types = ["indexed_attention"]
    cfg.model_config.hf_text_config.num_hidden_layers = 1
    cfg.model_config.hf_text_config.ple_layer_ids = []
    layout, _ = pooled_cache_specs(cfg)
    cfg.model_config.max_model_len = 8 * layout.block_size
    cache = pooled_cache.Qwen4ExpPooledCache(
        NS(vllm_config=cfg, cache_config=cfg.cache_config), None
    )
    cache.get_specs()
    per_request = sum(
        g.kv_cache_spec.max_num_blocks_per_req(cfg, cfg.model_config.max_model_len)
        for g in cache.groups
    )
    capacity = 1 + per_request
    bytes_per_block = layout.page_bytes * max(len(g.layer_names) for g in cache.groups)
    cache.get_specs(available_memory=capacity * bytes_per_block)
    assert cache.pages == capacity
    from vllm.v1.core.kv_cache_planning import (
        get_kv_cache_config_from_groups,
    )

    plan = get_kv_cache_config_from_groups(
        cfg, cache.groups, capacity * bytes_per_block
    )
    block, hashed = resolve_kv_cache_block_sizes(plan, cfg)
    manager = KVCacheManager(
        plan,
        max_model_len=cfg.model_config.max_model_len,
        scheduler_block_size=block,
        hash_block_size=hashed,
        enable_caching=False,
    )
    requests = [
        Request(
            str(i),
            [7] * cfg.model_config.max_model_len,
            SamplingParams(max_tokens=1),
            None,
        )
        for i in range(2)
    ]
    assert (
        manager.allocate_slots(
            requests[0], cfg.model_config.max_model_len, full_sequence_must_fit=True
        )
        is not None
    )
    assert manager.allocate_slots(requests[1], 1, full_sequence_must_fit=True) is None
    manager.free(requests[0])
    assert (
        manager.allocate_slots(
            requests[1], cfg.model_config.max_model_len, full_sequence_must_fit=True
        )
        is not None
    )


def test_worker_budget_coordinates_actual_tp_then_ep_dp_minimum(monkeypatch):
    cfg = configuration(True, "fp8_e4m3", 2, 128, 128)
    layout, specs = pooled_cache_specs(cfg)
    cfg.model_config.max_model_len = 8 * layout.block_size + 1
    layout, specs = pooled_cache_specs(cfg)
    ceiling = native_cache_capacity(cfg, specs)
    from vllm.v1.core.kv_cache_planning import (
        get_kv_cache_groups,
    )

    groups = get_kv_cache_groups(cfg, specs)
    stride = layout.page_bytes * max(len(g.layer_names) for g in groups)
    monkeypatch.setattr(
        pooled_cache,
        "compute_hbm_budget",
        lambda *_: NS(available=ceiling * stride + pooled_cache.TEMP_HEADROOM),
    )
    group = object()
    monkeypatch.setattr(pooled_cache, "get_tp_group", lambda: NS(cpu_group=group))
    order = []

    def reduce(value, op, group):
        assert op is pooled_cache.torch.distributed.ReduceOp.MIN
        assert group is expected_group
        assert value.item() == ceiling * stride
        value.fill_((ceiling - 1) * stride)
        order.append("TP")

    expected_group = group
    monkeypatch.setattr(pooled_cache.torch.distributed, "all_reduce", reduce)

    def dp_min(parallel, value):
        assert parallel is cfg.parallel_config and value == (ceiling - 1) * stride
        order.append("EP/DP")
        return (ceiling - 2) * stride

    monkeypatch.setattr(pooled_cache, "min_across_ep_dp", dp_min)
    worker, adapter, _ = worker_and_adapter(cfg, [])
    worker.get_kv_cache_spec()
    assert order == ["TP", "EP/DP"]
    assert adapter.cache.pages == ceiling - 2
