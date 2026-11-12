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
"""Live carrier binding; CPU bytes do not prove the TPU physical layout."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)

from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import ManifestError
from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import build_transfer_manifest
from vllm_torchtpu.models.vllm.qwen4_exp.pool_allocation import materialize_pool
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs


def live_pool(
    tp=1,
    device="cpu",
    num_blocks=64,
    ssm="bfloat16",
    k=0,
    apc=False,
    **text_overrides,
):
    cfg = pool_config(tp=tp, ssm=ssm)
    cfg.speculative_config = (
        SimpleNamespace(
            num_speculative_tokens=k,
            method="mtp",
            use_eagle=lambda: True,
            use_eagle_block_drop=lambda: False,
        )
        if k
        else None
    )
    # These are normalized text-only pool/transfer fixtures.
    cfg.model_config.hf_text_config.rope_parameters.pop("mrope_section", None)
    cfg.model_config.hf_text_config.rope_parameters.pop("mrope_interleaved", None)
    for key, value in text_overrides.items():
        setattr(cfg.model_config.hf_text_config, key, value)
    cfg.cache_config.enable_prefix_caching = apc
    cfg.cache_config.mamba_cache_mode = "align" if apc else "none"
    layout, specs = pooled_cache_specs(cfg)
    expected_ssm = torch.float32 if ssm == "float32" else torch.bfloat16
    assert all(
        spec.dtypes[1] == expected_ssm
        for name, spec in specs.items()
        if name.endswith(".linear_attn")
    )
    cfg.cache_config.block_size = layout.block_size
    cfg.cache_config.mamba_block_size = (
        layout.block_size if apc else cfg.model_config.max_model_len
    )
    groups = get_kv_cache_groups(cfg, specs)
    regions = max(len(g.layer_names) for g in groups)
    config = get_kv_cache_config_from_groups(
        cfg, groups, num_blocks * layout.page_bytes * regions
    )
    config.kv_cache_layout = "BLHNC"
    materialized = materialize_pool(cfg, config, layout, torch.device(device))
    return config, materialized


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("ssm", ["bfloat16", "float32"])
def test_full_pages_keep_all_owners_and_group_specific_aliases(tp, ssm):
    config, materialized = live_pool(tp, ssm=ssm)
    manifest = build_transfer_manifest(config, materialized)
    ring_keys = {k for k in materialized.kv_caches if k.endswith(".raw_ring")}
    assert len(manifest.pools) == len(materialized.kv_caches) - len(ring_keys)
    assert len({p.tag for p in manifest.pools}) == len(manifest.pools)
    assert all(a is b for a, b in zip(manifest.storages, materialized.raw_tensors))
    by_name = {p.layer_name: p for p in manifest.pools}
    for p in manifest.pools:
        assert p.live_bytes_per_block == p.block_stride_bytes
        assert p.regions[0].extent_end_bytes == p.block_stride_bytes
        if p.layer_name.endswith(".self_attn"):
            compressed = by_name[p.layer_name + ".compressed_k"]
            assert p.storage_index != compressed.storage_index
            assert compressed.num_blocks == p.num_blocks == config.num_blocks
            assert p.block_stride_bytes + compressed.block_stride_bytes == (
                config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
            )
            assert len({p.tag, compressed.tag}) == 2
        if p.layer_name.endswith(".ple"):
            linear = by_name[p.layer_name.removesuffix(".ple") + ".linear_attn"]
            assert p.storage_index != linear.storage_index
            assert p.num_blocks == config.num_blocks


def test_reject_equal_shape_clone_instead_of_live_carrier():
    config, materialized = live_pool()
    name = next(iter(materialized.kv_caches))
    caches = dict(materialized.kv_caches)
    raw = materialized.layer_to_raw[name]
    clone = torch.empty_like(raw)
    caches[name] = [clone] if isinstance(caches[name], list) else clone
    with pytest.raises(ManifestError, match="not its live carrier"):
        build_transfer_manifest(config, replace(materialized, kv_caches=caches))


def test_reject_missing_owner_and_duplicate_storage():
    config, materialized = live_pool()
    caches = dict(materialized.kv_caches)
    caches.pop(next(iter(caches)))
    with pytest.raises(ManifestError, match="every live cache owner"):
        build_transfer_manifest(config, replace(materialized, kv_caches=caches))
    with pytest.raises(ManifestError, match="repeats a carrier"):
        build_transfer_manifest(
            config, replace(materialized, raw_tensors=materialized.raw_tensors * 2)
        )


def test_reject_replaced_raw_storage():
    config, materialized = live_pool()
    replacement = list(materialized.raw_tensors)
    replacement[0] = torch.empty_like(replacement[0])
    with pytest.raises(ManifestError, match="not its live carrier"):
        build_transfer_manifest(config, replace(materialized, raw_tensors=replacement))


def test_reject_wrong_page_capacity():
    config, materialized = live_pool()
    config.num_blocks += 1
    with pytest.raises((ManifestError, AssertionError)):
        build_transfer_manifest(config, materialized)


@pytest.mark.parametrize(
    "tp,heads,dim,index_dim,ratio,blocks",
    [
        (1, 4, 128, 96, 2, 7),
        (2, 4, 256, 192, 8, 11),
        (4, 2, 256, 128, 16, 13),
        (8, 2, 512, 256, 4, 17),
    ],
)
def test_derived_regions_and_capacity_independent_peer_encoding(
    tp, heads, dim, index_dim, ratio, blocks
):
    from tests.runner.test_qwen4_exp_pd_layout import native_layout
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import measure_transfer_layout

    fingerprints = []
    for capacity in (blocks, blocks + 1):
        config, live = live_pool(
            tp,
            num_blocks=capacity,
            num_key_value_heads=heads,
            head_dim=dim,
            indexer_head_dim=index_dim,
            indexer_compress_ratio=ratio,
        )
        manifest = build_transfer_manifest(config, live)
        assert config.num_blocks == capacity
        for pool in manifest.pools:
            raw = live.kv_caches[pool.layer_name]
            raw = raw[0] if isinstance(raw, list) else raw
            assert manifest.storages[pool.storage_index] is raw
            assert pool.block_stride_bytes == raw[0].nbytes
        fingerprint, payload = measure_transfer_layout(
            manifest, layout_getter=native_layout, package_version=lambda _: "test"
        )
        assert payload["schema"] == "qwen4-dual-carrier-page-layout-v2"
        fingerprints.append(fingerprint)
    assert fingerprints[0] == fingerprints[1]


@pytest.mark.parametrize("extra_pages", [0, 1, 2])
def test_actual_dual_carriers_follow_owner_tables_and_preserve_other_bytes(extra_pages):
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import build_transfer_groups
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import QwenPagePlan

    config, live = live_pool(
        tp=8,
        num_blocks=32,
        num_hidden_layers=3,
        layer_types=["full_attention", "linear_attention", "linear_attention"],
    )
    manifest = build_transfer_manifest(config, live)
    groups = build_transfer_groups(config.kv_cache_groups)
    plan = QwenPagePlan(groups, manifest)
    tokens = (
        next(
            g.block_size
            for g in groups
            if any(n.endswith(".self_attn") for n in g.layer_names)
        )
        * extra_pages
        + 1
    )
    tables, cursor = [], 1
    for group in groups:
        count = group.page_count(tokens)
        tables.append(list(range(cursor, cursor + count)))
        cursor += count
    assert cursor < config.num_blocks // 2
    source = [
        torch.arange(raw.nbytes, dtype=torch.int32).to(torch.uint8)
        for raw in manifest.storages
    ]
    actual = [torch.full_like(x, 211) for x in source]
    expected = [x.clone() for x in actual]
    # Independent oracle: enumerate actual model bindings from scheduler owners,
    # adding compressed keys to the main table and co-owned PLE to its GDN table.
    for group, pages in zip(groups, tables):
        for owner in group.layer_names:
            names = [owner]
            if owner.endswith(".self_attn"):
                names.append(owner + ".compressed_k")
            elif (
                owner.endswith(".linear_attn")
                and (owner.removesuffix(".linear_attn") + ".ple") in live.kv_caches
            ):
                names.append(owner.removesuffix(".linear_attn") + ".ple")
            for name in names:
                raw = live.kv_caches[name]
                raw = raw[0] if isinstance(raw, list) else raw
                index = next(i for i, x in enumerate(manifest.storages) if x is raw)
                stride = raw[0].nbytes
                for page in pages:
                    dest = config.num_blocks - page
                    expected[index][dest * stride : (dest + 1) * stride] = source[
                        index
                    ][page * stride : (page + 1) * stride]
    pages = plan.request_pages(tables, tokens)
    for pool, gid in plan.pool_groups():
        stride = pool.block_stride_bytes
        for page in pages[gid]:
            dest = config.num_blocks - page
            actual[pool.storage_index][dest * stride : (dest + 1) * stride] = source[
                pool.storage_index
            ][page * stride : (page + 1) * stride]
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    # Same physical main page cannot be assigned to two groups sharing a carrier.
    main_gid = next(
        g.group_index
        for g in groups
        if any(n.endswith(".self_attn") for n in g.layer_names)
    )
    gdn_gid = next(
        g.group_index
        for g in groups
        if any(n.endswith(".linear_attn") for n in g.layer_names)
    )
    tables[gdn_gid] = tables[main_gid][:1]
    with pytest.raises(ValueError, match="overlap"):
        plan.request_pages(tables, tokens)


@pytest.mark.parametrize("k", [0, 1, 3])
def test_mtp_manifest_enumerates_draft_owners_and_checkpoint_geometry(k):
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import build_transfer_groups
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import QwenPagePlan

    config, live = live_pool(k=k, num_blocks=256)
    manifest = build_transfer_manifest(config, live)
    owners = [p.layer_name for p in manifest.pools]
    draft = "mtp.layers.48.self_attn"
    assert (draft in owners) is bool(k)
    if k:
        assert {draft, draft + ".compressed_k"} <= set(owners)
        assert draft + ".raw_ring" not in owners
    groups = build_transfer_groups(config.kv_cache_groups)
    cursor, tables = 1, []
    for group in groups:
        count = group.page_count(129)
        tables.append(list(range(cursor, cursor + count)))
        cursor += count
    QwenPagePlan(groups, manifest).request_pages(tables, num_tokens=129)
