# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
from types import SimpleNamespace as NS
from typing import cast

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CompilationConfig, VllmConfig
from vllm.model_executor.models import ModelRegistry
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    resolve_cache_hit_alignment_tokens,
    resolve_kv_cache_block_sizes,
)

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.executors.tpu_multiproc_executor import TpuMultiprocExecutor
from vllm_torchtpu.models.vllm.qwen4_exp import pooled_cache
from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpForCausalLM
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.worker.tpu_worker import TPUWorker


class ModelOptions(NS):
    def get_num_kv_heads(self, parallel):
        return max(
            self.hf_text_config.num_key_value_heads // parallel.tensor_parallel_size, 1
        )

    def get_head_size(self):
        return self.hf_text_config.head_dim


def configuration(prefix, dtype, tp, requested, dim):
    cfg = pool_config(tp=tp, kv=dtype, ssm="float32")
    cfg.cache_config.enable_prefix_caching = prefix
    cfg.cache_config.mamba_cache_mode = "align" if prefix else "none"
    cfg.cache_config.block_size = requested
    cfg.scheduler_config.max_num_seqs = 2
    cfg.compilation_config = CompilationConfig()
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.parallel_config.world_size = tp
    cfg.model_config = ModelOptions(
        hf_config=PretrainedConfig.from_dict(
            {"architectures": ["Qwen4ExpForCausalLM"]}
        ),
        architecture="Qwen4ExpForCausalLM",
        is_hybrid=True,
        dtype=torch.bfloat16,
        max_model_len=512,
        hf_text_config=q4exp_text_config(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=dim,
            indexer_head_dim=64,
            indexer_n_heads=1,
            indexer_budget=64,
            num_hidden_layers=2,
            layer_types=["linear_attention", "indexed_attention"],
            ple_layer_ids=[1],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        ),
    )
    return cast(VllmConfig, cfg)


def worker_and_adapter(cfg, events):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.vllm_config, runner.cache_config, runner.model_config = (
        cfg,
        cfg.cache_config,
        cfg.model_config,
    )
    runner.device = torch.device("cpu")
    runner.block_size = cfg.cache_config.block_size
    runner.max_model_len, runner.max_num_reqs, runner.max_num_tokens = 512, 2, 16
    runner.max_num_blocks_per_req = math.ceil(512 / runner.block_size)
    runner.most_model_len, runner.kv_cache_config = None, None
    runner._attention_capacity = {}
    runner._decode_device_cache_key = None
    adapter = Qwen4ExpAdapter(runner)
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.runner, manager.model_adapter = runner, adapter
    runner.kv_cache_manager = manager
    attention = Qwen4ExpQSA.__new__(Qwen4ExpQSA)
    nn.Module.__init__(attention)
    cfg.compilation_config.static_forward_context["model.layers.1.self_attn"] = (
        attention
    )
    owners = []
    model = NS(
        model=NS(
            layers=[
                NS(
                    layer_type="linear_attention",
                    linear_attn=NS(kv_cache=None),
                    ple=NS(bind_pool=lambda owner: owners.append(owner)),
                ),
                NS(
                    layer_type="indexed_attention",
                    self_attn=NS(bind_pool=lambda *pools: owners.extend(pools)),
                    ple=None,
                ),
            ]
        )
    )
    events.append("load")
    adapter.bind_model(model)
    worker = TPUWorker.__new__(TPUWorker)
    worker.vllm_config, worker.model_runner = cfg, runner
    return worker, adapter, owners


@pytest.mark.parametrize(
    "prefix,dtype,tp,requested,dim",
    [
        (False, "auto", 1, 128, 128),
        (True, "fp8", 2, 2048, 64),
        (True, "bfloat16", 2, 2048, 256),
        (False, "fp8_e4m3", 1, 128, 256),
    ],
)
def test_load_negotiate_specs_profile_scheduler_with_independent_configs(
    monkeypatch, prefix, dtype, tp, requested, dim
):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda arch, **kwargs: (Qwen4ExpForCausalLM, arch),
    )
    monkeypatch.setattr(
        pooled_cache, "compute_hbm_budget", lambda *args: NS(available=1 << 40)
    )
    driver = configuration(prefix, dtype, tp, requested, dim)
    events, workers, adapters, owner_lists = [], [], [], []
    actual_specs = pooled_cache.pooled_cache_specs

    def observe_specs(cfg):
        events.append("specs")
        return actual_specs(cfg)

    monkeypatch.setattr(pooled_cache, "pooled_cache_specs", observe_specs)
    for _ in range(tp):
        cfg = copy.deepcopy(driver)
        worker, adapter, owners = worker_and_adapter(cfg, events)
        workers.append(worker)
        adapters.append(adapter)
        owner_lists.append(owners)
        assert events[-1] == "load"
        assert cfg.cache_config.num_gpu_blocks_override is None
        events.append("platform")
        TpuPlatform.update_block_size_for_backend(cfg)
    assert "specs" not in events
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor.vllm_config = driver

    def rpc(method, *args, **kwargs):
        return [
            worker.get_kv_cache_spec()
            if method == "get_kv_cache_spec"
            else worker.get_num_gpu_blocks_override()
            for worker in workers
        ]

    monkeypatch.setattr(executor, "collective_rpc", rpc)
    specs = executor.get_kv_cache_specs()
    assert driver.cache_config is not workers[0].vllm_config.cache_config
    for worker, adapter, owners, per_worker in zip(
        workers, adapters, owner_lists, specs
    ):
        cfg, runner, cache = worker.vllm_config, worker.model_runner, adapter.cache
        assert cache is not None
        layout = cache.layout
        assert layout.block_size >= requested
        assert (
            runner.block_size
            == adapter.block_size
            == cfg.cache_config.block_size
            == layout.block_size
        )
        assert runner.max_num_blocks_per_req == math.ceil(512 / layout.block_size)
        assert (
            cfg.cache_config.mamba_page_size_padded
            == layout.main_page_bytes + layout.index_page_bytes
        )
        assert all(
            spec.page_size_bytes == layout.page_bytes for spec in per_worker.values()
        )
        assert layout.main_page_bytes >= max(
            layout.gdn_state_bytes, layout.ple_state_bytes
        )
        expected_pages = 1 + 2 * sum(
            group.kv_cache_spec.max_num_blocks_per_req(cfg, 512)
            for group in cache.groups
        )
        assert (
            worker.get_num_gpu_blocks_override()
            == driver.cache_config.num_gpu_blocks_override
            == expected_pages
        )
        events.append("profile")
        cache.allocate()
        assert len(owners) == 3 and owners[-1] is not owners[-2]
        assert cache.materialized is not None
        raw_bytes = []
        for index, value in enumerate(cache.materialized.raw_tensors):
            if value.dtype == torch.bfloat16:
                assert torch.count_nonzero(value.view(torch.uint8)) == 0
            value.view(torch.uint8).fill_(0x81 + index)
            raw_bytes.append(value.view(torch.uint8).clone())
        events.append("scheduler")
        groups = get_kv_cache_groups(driver, per_worker.copy())
        plan = get_kv_cache_config_from_groups(
            driver,
            groups,
            expected_pages
            * layout.page_bytes
            * max(len(g.layer_names) for g in groups),
        )
        assert plan.num_blocks == cache.native_config.num_blocks == expected_pages
        participating = [
            g.kv_cache_spec.block_size
            for g in plan.kv_cache_groups
            if g.kv_cache_spec.prefix_cacheable
        ]
        driver.cache_config.block_size = min(participating)
        scheduler_block, hash_block = resolve_kv_cache_block_sizes(plan, driver)
        expected_quantum = math.lcm(
            *(g.kv_cache_spec.block_size for g in plan.kv_cache_groups)
        )
        assert scheduler_block == expected_quantum
        if prefix:
            assert scheduler_block == layout.block_size
        else:
            assert hash_block == scheduler_block
        plan.kv_cache_layout = driver.cache_config.kv_cache_layout
        plan.hash_block_size = hash_block
        plan.cache_hit_alignment_tokens = resolve_cache_hit_alignment_tokens(
            plan, driver, scheduler_block, hash_block
        )
        assert plan == cache.native_config
        for descriptor in plan.kv_cache_tensors:
            for index, name in enumerate(descriptor.layers):
                assert (
                    descriptor.offset
                    + index * descriptor.layer_stride
                    + (expected_pages - 1) * descriptor.block_stride
                    + per_worker[name].page_size_bytes
                    <= descriptor.size
                )
        before = (
            layout,
            cache.specs,
            cache.pages,
            tuple(map(id, cache.materialized.raw_tensors)),
        )
        TpuPlatform.update_block_size_for_backend(cfg)
        assert worker.get_kv_cache_spec() == per_worker
        assert cfg.cache_config.mamba_page_size_padded == layout.page_bytes
        cache.allocate()
        assert all(
            torch.equal(value.view(torch.uint8), expected)
            for value, expected in zip(cache.materialized.raw_tensors, raw_bytes)
        )
        assert before == (
            cache.layout,
            cache.specs,
            cache.pages,
            tuple(map(id, cache.materialized.raw_tensors)),
        )
        cfg.scheduler_config.max_num_seqs += 1
        with pytest.raises(ValueError, match="geometry changed after allocation"):
            worker.get_kv_cache_spec()
        assert cache.pages == expected_pages
        cfg.scheduler_config.max_num_seqs -= 1
        cfg.cache_config.block_size *= 2
        with pytest.raises(ValueError, match="geometry changed after allocation"):
            worker.get_kv_cache_spec()
        assert all(
            torch.equal(value.view(torch.uint8), expected)
            for value, expected in zip(cache.materialized.raw_tensors, raw_bytes)
        )
    assert (
        events.index("platform")
        < events.index("specs")
        < events.index("profile")
        < events.index("scheduler")
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_non_pooled_backend_negotiation_is_idempotent(monkeypatch, sparse):
    from vllm_torchtpu.layers.adapter.attention import (
        PallasAttentionBackend,
        PallasMLAttentionBackend,
    )
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
        deepseek_v4_attention_op,
    )

    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    cfg = configuration(False, "auto", 1, 128, 128)
    cast(ModelOptions, cfg.model_config).is_hybrid = False
    cfg.cache_config.user_specified_block_size = False
    cfg.model_config.hf_config.architectures = [
        "DeepseekV4ForCausalLM" if sparse else "LlamaForCausalLM"
    ]
    backend = (
        deepseek_v4_attention_op.VllmDeepseekSparseSWABackend
        if sparse
        else PallasAttentionBackend
    )
    monkeypatch.setattr(
        TpuPlatform,
        "_find_non_ssm_backend",
        classmethod(lambda cls, cfg: backend),
    )
    expected = (
        PallasMLAttentionBackend.get_page_size(cfg)
        if sparse
        else PallasAttentionBackend.get_preferred_block_size(
            PallasAttentionBackend.get_page_size(cfg)
        )
    )
    for _ in range(2):
        TpuPlatform.update_block_size_for_backend(cfg)
        assert cfg.cache_config.block_size == expected
        assert cfg.cache_config.mamba_page_size_padded is None
