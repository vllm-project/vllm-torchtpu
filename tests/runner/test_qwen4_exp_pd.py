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
"""PD page ownership at partial compression and scheduler-page boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config.cache import CacheConfig
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from vllm_torchtpu.models.vllm.qwen4_exp.pd import (
    TransferExtent,
    build_transfer_groups,
    handoff_block_ids,
)
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs
from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache


def cache_groups():
    specs = [
        FullAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=128, dtype=torch.float8_e4m3fn
        ),
        MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            tokens_per_state=4,
        ),
        CircularBufferSpec(
            block_size=4, num_kv_heads=1, head_size=132, dtype=torch.bfloat16
        ),
        MambaSpec(
            block_size=9216,
            shapes=((3, 1, 256), (2, 128, 128)),
            dtypes=(torch.float32, torch.bfloat16),
            mamba_cache_mode="none",
        ),
        MambaSpec(
            block_size=9216,
            shapes=((9, 256),),
            dtypes=(torch.bfloat16,),
            mamba_cache_mode="none",
        ),
    ]
    names = ["qsa", "qsa.compressed_k", "qsa.raw_ring", "gdn", "ple"]
    return [
        KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
        for name, spec in zip(names, specs)
    ]


@pytest.mark.parametrize("tokens", [0, 1, 3, 4, 5, 127, 128, 129, 8191, 8192])
def test_committed_prefix_uses_paged_kv_and_single_request_states(tokens):
    plan = build_transfer_groups(cache_groups())
    expected = [(tokens + 127) // 128] * 2 + [int(tokens > 0)] * 3
    tables = []
    next_page = 1
    for count in expected:
        tables.append(list(range(next_page, next_page + count)))
        next_page += count
    assert [group.page_count(tokens) for group in plan] == expected
    assert [g.extent for g in plan[2:]] == [TransferExtent.REQUEST_STATE] * 3
    assert handoff_block_ids(
        plan, tables, committed_tokens=tokens, computed_tokens=tokens
    ) == tuple(tuple(table) for table in tables)


@pytest.mark.parametrize("computed", [126, 128])
def test_reject_state_from_a_different_prefix(computed):
    with pytest.raises(ValueError, match="published committed prefix"):
        handoff_block_ids(
            build_transfer_groups(cache_groups()),
            [[1], [2], [3], [4], [5]],
            committed_tokens=127,
            computed_tokens=computed,
        )


@pytest.mark.parametrize(
    "tables,reason",
    [
        ([[1], [2], [3], [4]], "one block table"),
        ([[1], [2], [0], [4], [5]], "null or invalid"),
        ([[1], [2], [3, 0], [4], [5]], "needs 1 pages"),
        ([[1], [], [3], [4], [5]], "needs 1 pages"),
        ([[True], [2], [3], [4], [5]], "null or invalid"),
    ],
)
def test_reject_missing_null_or_padded_scheduler_tables(tables, reason):
    with pytest.raises(ValueError, match=reason):
        handoff_block_ids(
            build_transfer_groups(cache_groups()),
            tables,
            committed_tokens=127,
            computed_tokens=127,
        )


def test_reject_duplicate_pages_and_reordered_groups():
    plan = build_transfer_groups(cache_groups())
    with pytest.raises(ValueError, match="repeats a physical page"):
        handoff_block_ids(
            plan,
            [[1, 1], [2, 3], [4], [5], [6]],
            committed_tokens=129,
            computed_tokens=129,
        )
    with pytest.raises(ValueError, match="scheduler group order"):
        handoff_block_ids(
            tuple(reversed(plan)),
            [[1], [2], [3], [4], [5]],
            committed_tokens=1,
            computed_tokens=1,
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"mamba_cache_mode": "align"},
        {"mamba_cache_mode": "all"},
        {"num_speculative_blocks": 1},
    ],
)
def test_state_capacity_flags_accept_only_an_explicit_committed_snapshot(fields):
    groups = cache_groups()
    groups[3] = replace(
        groups[3],
        kv_cache_spec=replace(groups[3].kv_cache_spec, **fields),
    )
    plan = build_transfer_groups(groups)
    count = 1 + fields.get("num_speculative_blocks", 0)
    assert plan[3].request_state_pages == count
    tables = [[1], [2], [3], [4], [10]]
    assert handoff_block_ids(
        plan, tables, committed_tokens=127, computed_tokens=127
    ) == tuple(tuple(t) for t in tables)
    for state_ids in ([], list(range(4, 5 + count))):
        with pytest.raises(ValueError, match="needs 1 pages"):
            handoff_block_ids(
                plan,
                [*tables[:3], state_ids, tables[4]],
                committed_tokens=127,
                computed_tokens=127,
            )
    with pytest.raises(ValueError, match="published committed prefix"):
        handoff_block_ids(plan, tables, committed_tokens=127, computed_tokens=128)


def test_duplicate_cache_owners_are_rejected():
    groups = cache_groups()
    groups.append(groups[0])
    with pytest.raises(ValueError, match="every cache owner once"):
        build_transfer_groups(groups)


def test_pool_exposes_only_the_bound_transfer_plan():
    cache = object.__new__(Qwen4ExpPooledCache)
    cache.materialized = None
    with pytest.raises(ValueError, match="Initialize Qwen caches"):
        cache.get_transfer_groups()
    cache.materialized = object()
    cache.native_config = SimpleNamespace(kv_cache_groups=cache_groups())
    assert cache.get_transfer_groups() == build_transfer_groups(cache_groups())


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_plan_covers_the_native_qwen_scheduler_groups(tp):
    cfg = SimpleNamespace(
        additional_config={},
        cache_config=CacheConfig(
            block_size=128,
            cache_dtype="fp8_e4m3",
            enable_prefix_caching=False,
            mamba_cache_mode="none",
            mamba_ssm_cache_dtype="bfloat16",
        ),
        model_config=SimpleNamespace(
            hf_text_config=q4exp_text_config(),
            max_model_len=9216,
            dtype=torch.bfloat16,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_transfer_config=None,
    )
    cfg.cache_config.kv_cache_layout = "BLHNC"
    layout, specs = pooled_cache_specs(cfg)
    cfg.cache_config.block_size = layout.block_size
    cfg.cache_config.mamba_block_size = cfg.model_config.max_model_len
    groups = get_kv_cache_groups(cfg, specs)
    plan = build_transfer_groups(groups)
    assert {name for group in plan for name in group.layer_names} == set(specs)
    for group in plan:
        spec = groups[group.group_index].kv_cache_spec
        if isinstance(spec, (CircularBufferSpec, MambaSpec)):
            assert group.page_count(8191) == 1
        else:
            assert (
                group.page_count(8191)
                == (8191 + spec.block_size - 1) // spec.block_size
            )


@pytest.mark.parametrize("k", [0, 1, 3])
def test_mtp_wire_uses_one_state_page_without_changing_allocation(k):
    groups = cache_groups()
    for i in (3, 4):
        groups[i] = replace(
            groups[i],
            kv_cache_spec=replace(groups[i].kv_cache_spec, num_speculative_blocks=k),
        )
    plan = build_transfer_groups(groups)
    assert [g.request_state_pages for g in plan] == [1, 1, 1, k + 1, k + 1]
    assert [g.page_count(129) for g in plan] == [2, 2, 1, 1, 1]
    tables = [[1, 2], [3, 4], [5], [6], [10]]
    handoff_block_ids(plan, tables, committed_tokens=129, computed_tokens=129)
    if k:
        tables[3] = list(range(6, 7 + k))
        with pytest.raises(ValueError, match="needs 1 pages"):
            handoff_block_ids(plan, tables, committed_tokens=129, computed_tokens=129)
