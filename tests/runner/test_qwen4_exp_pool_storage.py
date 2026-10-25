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
"""Actual pool allocation regression, runnable against the preceding source."""

import math
from types import SimpleNamespace as NS

import pytest
import torch

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache


def test_actual_pool_backing_does_not_allocate_main_sized_compressed_pages(monkeypatch):
    cfg = pool_config(kv="bfloat16")
    cfg.model_config.hf_text_config = q4exp_text_config(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
        indexer_n_heads=1,
        indexer_compress_ratio=4,
        num_hidden_layers=2,
        layer_types=["linear_attention", "indexed_attention"],
        ple_layer_ids=[],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    cfg.model_config.max_model_len = 512
    cfg.scheduler_config.max_num_seqs = 2
    runner = NS(
        vllm_config=cfg,
        cache_config=cfg.cache_config,
        kv_cache_dtype="bfloat16",
        device=torch.device("cpu"),
    )
    bound = []
    # No checkpoint/model computation: exercise the production planner,
    # allocator and binding with two logical layers, not a mocked allocator.
    layer0 = NS(layer_type="linear_attention", linear_attn=NS(kv_cache=None), ple=None)
    layer1 = NS(
        layer_type="indexed_attention",
        self_attn=NS(bind_pool=lambda *owners: bound.append(owners)),
        ple=None,
    )
    model = NS(model=NS(layers=[layer0, layer1]))
    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache.compute_hbm_budget",
        lambda *a, **kw: NS(available=1 << 40),
    )
    cache = Qwen4ExpPooledCache(runner, model)
    cfg.cache_config.block_size = cache.layout.block_size
    cfg.cache_config.mamba_block_size = cache.layout.block_size
    cfg.cache_config.num_gpu_blocks_override = cache.pages
    cache.allocate()
    assert bound
    # Two requests: four full-attention blocks, four recurrent checkpoints,
    # one non-cacheable ring each, plus the native null block. The compressed
    # index belongs to those same attention blocks, not four more blocks.
    expected_blocks = 1 + 2 * (math.ceil(512 / 128) * 2 + 1)
    main_per_block = 128 * 2 * 2 * 256 * 2
    index_per_block = max(8192, (128 // 4) * 128 * 2)
    expected_bytes = expected_blocks * (main_per_block + index_per_block)
    unique = {
        t.untyped_storage()._cdata: t.untyped_storage().nbytes()
        for t in cache.materialized.raw_tensors
    }
    assert sum(unique.values()) == expected_bytes
    assert cache.native_config.num_blocks == expected_blocks
    assert len(bound[0]) == 2


def _startup_cache(
    monkeypatch, *, prefix=False, explicit_prefix_unit=False, kv="bfloat16"
):
    from tests.runner.test_qwen4_exp_compact_pool import compact_config
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    cfg = compact_config(prefix=prefix, layers=1, kv=kv)
    cfg.num_prefill_lookahead_tokens = 0
    cfg.model_config.original_max_model_len = cfg.model_config.max_model_len
    runner = NS(
        vllm_config=cfg,
        cache_config=cfg.cache_config,
        device=torch.device("cpu"),
        enforce_eager=True,
        max_num_reqs=cfg.scheduler_config.max_num_seqs,
        _pool_block_split=1,
        may_reinitialize_input_batch=lambda *args: None,
        _precompile_mamba_state_seed_copies=lambda: None,
    )
    runner._build_mamba_copy_plan = TPUModelRunner._build_mamba_copy_plan.__get__(
        runner
    )
    # Layer/runner lifecycle only are fixtures; planner, storage, binding,
    # metadata builders and state-copy planning are production implementations.
    layers = [
        NS(
            layer_type="linear_attention",
            linear_attn=NS(kv_cache=None),
            ple=NS(bind_pool=lambda owner: None),
        ),
        NS(
            layer_type="indexed_attention",
            self_attn=NS(bind_pool=lambda *owners: None),
            ple=None,
        ),
    ]
    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache.compute_hbm_budget",
        lambda *a, **kw: NS(available=1 << 40),
    )
    cache = Qwen4ExpPooledCache(runner, NS(model=NS(layers=layers)))
    cfg.cache_config.block_size = cache.layout.block_size
    cfg.cache_config.mamba_block_size = cache.layout.block_size if prefix else None
    cfg.cache_config.num_gpu_blocks_override = cache.pages
    if explicit_prefix_unit:
        cfg.cache_config.prefix_match_unit = cache.layout.block_size
    runner.input_batch = NS(
        block_table=[
            NS(
                max_num_blocks_per_req=group.kv_cache_spec.max_num_blocks_per_req(
                    cfg, cfg.model_config.max_model_len
                )
            )
            for group in cache.groups
        ]
    )
    cache.allocate()
    return cache


def _engine_plan(cache):
    import copy

    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_configs,
        resolve_cache_hit_alignment_tokens,
        resolve_kv_cache_block_sizes,
    )

    cfg = cache.runner.vllm_config
    memory = (
        cache.pages
        * cache.layout.page_bytes
        * max(len(group.layer_names) for group in cache.groups)
    )
    plans = get_kv_cache_configs(cfg, [copy.deepcopy(cache.specs)], [memory])
    final = plans[0]
    final.kv_cache_layout = cfg.cache_config.kv_cache_layout
    scheduler = generate_scheduler_kv_cache_config(plans)
    block, hashed = resolve_kv_cache_block_sizes(scheduler, cfg)
    final.hash_block_size = hashed
    final.cache_hit_alignment_tokens = resolve_cache_hit_alignment_tokens(
        scheduler, cfg, block, hashed
    )
    return final


@pytest.mark.parametrize(
    "prefix,explicit_unit", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("kv", ["bfloat16", "fp8_e4m3"])
def test_initialize_retains_engine_scheduler_metadata(
    monkeypatch, prefix, explicit_unit, kv
):
    cache = _startup_cache(
        monkeypatch, prefix=prefix, explicit_prefix_unit=explicit_unit, kv=kv
    )
    final = _engine_plan(cache)
    owners = tuple(cache.materialized.raw_tensors)
    cache.initialize(final)
    assert cache.runner.kv_cache_config is final
    assert cache.native_config == final
    assert final.hash_block_size is not None
    assert final.cache_hit_alignment_tokens is not None
    assert all(a is b for a, b in zip(owners, cache.runner.kv_caches))
    assert all(group[0].metadata_builders for group in cache.runner.attn_groups)


@pytest.mark.parametrize(
    "mutation",
    [
        "blocks",
        "bytes",
        "offset",
        "stride",
        "layout",
        "layers",
        "groups",
        "dtype",
        "shape",
        "block_size",
        "hash",
    ],
)
def test_initialize_rejects_changed_pool_contract(monkeypatch, mutation):
    import copy
    from dataclasses import replace

    from vllm.v1.kv_cache_interface import MambaSpec

    cache = _startup_cache(monkeypatch)
    final = _engine_plan(cache)
    cache.initialize(final)
    changed = copy.deepcopy(final)
    if mutation == "blocks":
        changed.num_blocks += 1
    elif mutation in {"bytes", "offset", "stride"}:
        descriptor = changed.kv_cache_tensors[0]
        field = {"bytes": "size", "offset": "offset", "stride": "block_stride"}[
            mutation
        ]
        changed.kv_cache_tensors[0] = replace(
            descriptor, **{field: getattr(descriptor, field) + 1}
        )
    elif mutation == "layout":
        changed.kv_cache_layout = "LBHNC"
    elif mutation == "layers":
        group = changed.kv_cache_groups[0]
        changed.kv_cache_groups[0] = replace(
            group, layer_names=["wrong.owner", *group.layer_names[1:]]
        )
    elif mutation == "groups":
        changed.kv_cache_groups.reverse()
    elif mutation == "hash":
        changed.hash_block_size += 1
    else:
        gid = next(
            i
            for i, group in enumerate(changed.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        )
        group = changed.kv_cache_groups[gid]
        spec = group.kv_cache_spec
        if mutation == "dtype":
            spec = replace(spec, dtypes=(torch.float32, *spec.dtypes[1:]))
        elif mutation == "shape":
            spec = replace(spec, shapes=((1,), *spec.shapes[1:]))
        else:
            spec = replace(spec, block_size=spec.block_size + 1)
        changed.kv_cache_groups[gid] = replace(group, kv_cache_spec=spec)
    assert changed != final
    with pytest.raises(AssertionError, match="pool plan changed"):
        cache.initialize(changed)
    assert cache.runner.kv_cache_config is final
