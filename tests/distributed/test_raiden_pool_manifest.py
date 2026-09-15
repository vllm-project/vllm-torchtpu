# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Raiden pool-manifest builder.

Primary oracles are mocked *materializations* (fake typed tensors shaped like
the live acceptance runs); known live geometry appears only as a secondary
cross-check. Decode FA geometry must be derived as 1024-token blocks from the
cache spec.
"""

import types

import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm

from .raiden_test_utils import FakeTensor, glm_named_kv_caches

pytestmark = pytest.mark.cpu_test


def _fa_group(layer_names, *, block_size, num_kv_heads, head_size):
    return types.SimpleNamespace(
        layer_names=tuple(layer_names),
        kv_cache_spec=types.SimpleNamespace(block_size=block_size,
                                            num_kv_heads=num_kv_heads,
                                            head_size=head_size),
    )


def _gdn_group(layer_names):
    return types.SimpleNamespace(layer_names=tuple(layer_names),
                                 kv_cache_spec=types.SimpleNamespace())


def _pooled_gdn_group(layer_names, *, shapes, dtypes, page_size_bytes):
    return types.SimpleNamespace(
        layer_names=tuple(layer_names),
        kv_cache_spec=types.SimpleNamespace(shapes=tuple(shapes),
                                            dtypes=tuple(dtypes),
                                            page_size_bytes=page_size_bytes),
    )


QWEN35_FA_LAYERS = tuple(range(3, 60, 4))
QWEN35_GDN_LAYERS = tuple(i for i in range(60) if i not in QWEN35_FA_LAYERS)


def _qwen35_materialization(*, fa_shape, fa_esz, fa_dtype, conv_shape,
                            conv_esz, ssm_shape, ssm_esz):
    """Builds named_kv_caches shaped like the live runner materialization."""
    named = {}
    for idx in QWEN35_FA_LAYERS:
        named[f"model.layers.{idx}.self_attn.attn"] = FakeTensor(
            fa_shape, fa_esz, dtype=fa_dtype)
    for idx in QWEN35_GDN_LAYERS:
        conv = FakeTensor(conv_shape, conv_esz, dtype="torch.bfloat16")
        ssm = FakeTensor(ssm_shape, ssm_esz, dtype="torch.float32")
        named[f"model.layers.{idx}.linear_attn"] = (conv, ssm)
    return named


def _qwen35_groups(named, *, block_size, num_kv_heads, head_size):
    fa_names = [n for n in named if "self_attn" in n]
    gdn_names = [n for n in named if "linear_attn" in n]
    return (
        _fa_group(fa_names,
                  block_size=block_size,
                  num_kv_heads=num_kv_heads,
                  head_size=head_size),
        _gdn_group(gdn_names[:15]),
        _gdn_group(gdn_names[15:30]),
        _gdn_group(gdn_names[30:]),
    )


PCP8_GEOMETRY = rpm.GdnHeadGeometry(local_key_heads=16,
                                    local_value_heads=64,
                                    key_head_dim=128,
                                    value_head_dim=128)
TP2_GEOMETRY = rpm.GdnHeadGeometry(local_key_heads=8,
                                   local_value_heads=32,
                                   key_head_dim=128,
                                   value_head_dim=128)


def _build_pcp8():
    named = _qwen35_materialization(
        fa_shape=(256, 256, 1, 4, 256),
        fa_esz=1,
        fa_dtype="torch.float8_e4m3fn",
        conv_shape=(16, 3, 1, 12288),
        conv_esz=2,
        ssm_shape=(16, 64, 128, 128),
        ssm_esz=4,
    )
    groups = _qwen35_groups(named,
                            block_size=4096,
                            num_kv_heads=2,
                            head_size=256)
    manifest = rpm.build_qwen35_pool_manifest(named_kv_caches=named,
                                              kv_cache_groups=groups,
                                              raw_tensors=(),
                                              gdn_geometry=PCP8_GEOMETRY)
    return named, manifest


def _build_tp2dp4():
    named = _qwen35_materialization(
        fa_shape=(64, 256, 1, 4, 256),
        fa_esz=1,
        fa_dtype="torch.float8_e4m3fn",
        conv_shape=(16, 3, 1, 6144),
        conv_esz=2,
        ssm_shape=(16, 32, 128, 128),
        ssm_esz=4,
    )
    groups = _qwen35_groups(named,
                            block_size=1024,
                            num_kv_heads=1,
                            head_size=256)
    manifest = rpm.build_qwen35_pool_manifest(named_kv_caches=named,
                                              kv_cache_groups=groups,
                                              raw_tensors=(),
                                              gdn_geometry=TP2_GEOMETRY)
    return named, manifest


# --------------------------------------------------------------------------
# PCP8 manifest derivation from the mocked materialization.
# --------------------------------------------------------------------------


def test_pcp8_manifest_matches_live_geometry():
    named, manifest = _build_pcp8()

    assert manifest.binding == rpm.BINDING_PRIVATE_TYPED
    assert len(manifest.pools) == 105
    assert len(manifest.storages) == 105
    assert manifest.tag_counts() == {
        rpm.TAG_FA: 15,
        rpm.TAG_GDN_CONV: 45,
        rpm.TAG_GDN_SSM: 45,
    }
    geometry = manifest.geometry_by_tag()
    # Known live values as secondary cross-checks.
    assert geometry[rpm.TAG_FA] == {
        "num_blocks": 16,
        "block_stride_bytes": 4_194_304,
        "live_bytes_per_block": 4_194_304,
    }
    assert geometry[rpm.TAG_GDN_CONV] == {
        "num_blocks": 16,
        "block_stride_bytes": 73_728,
        "live_bytes_per_block": 73_728,
    }
    assert geometry[rpm.TAG_GDN_SSM] == {
        "num_blocks": 16,
        "block_stride_bytes": 4_194_304,
        "live_bytes_per_block": 4_194_304,
    }

    fa_pool = next(p for p in manifest.pools if p.tag == rpm.TAG_FA)
    (region, ) = fa_pool.regions
    assert (region.offset_bytes, region.stride_bytes, region.unit_bytes,
            region.num_units, region.units_per_stride) == (0, 1024, 512, 4096,
                                                           2)
    assert fa_pool.dtype_tag == "float8_e4m3fn"

    conv_pool = next(p for p in manifest.pools if p.tag == rpm.TAG_GDN_CONV)
    q, k, v = conv_pool.regions
    assert (q.offset_bytes, q.stride_bytes, q.unit_bytes, q.num_units,
            q.units_per_stride) == (0, 24_576, 256, 3, 16)
    assert k.offset_bytes == 4_096
    assert (v.offset_bytes, v.units_per_stride) == (8_192, 64)

    ssm_pool = next(p for p in manifest.pools if p.tag == rpm.TAG_GDN_SSM)
    (ssm_region, ) = ssm_pool.regions
    assert (ssm_region.offset_bytes, ssm_region.stride_bytes,
            ssm_region.unit_bytes, ssm_region.num_units) == (0, 65_536, 65_536,
                                                             64)


def test_per_layer_tags_name_each_layer_and_fold_into_class_geometry():
    # A pipeline stage materializes only its own layers (here 0..4, one FA
    # layer at 3); under per-layer tags every pool names its layer and the
    # class geometry view still sees one FA and one GDN geometry.
    named_all = _qwen35_materialization(
        fa_shape=(256, 256, 1, 4, 256),
        fa_esz=1,
        fa_dtype="torch.float8_e4m3fn",
        conv_shape=(16, 3, 1, 12288),
        conv_esz=2,
        ssm_shape=(16, 64, 128, 128),
        ssm_esz=4,
    )
    stage_layers = {f"model.layers.{i}." for i in range(5)}
    named = {
        name: tensor
        for name, tensor in named_all.items() if any(
            name.startswith(prefix) for prefix in stage_layers)
    }
    groups = _qwen35_groups(named,
                            block_size=4096,
                            num_kv_heads=2,
                            head_size=256)
    manifest = rpm.build_qwen35_pool_manifest(
        named_kv_caches=named,
        kv_cache_groups=groups,
        raw_tensors=(),
        gdn_geometry=PCP8_GEOMETRY,
        mamba_group_ordinal_by_layer={
            name: 0
            for name in named if "linear_attn" in name
        },
        per_layer_tags=True,
    )

    assert [pool.tag for pool in manifest.pools] == [
        "gdn.conv.g0.l0", "gdn.ssm.g0.l0", "gdn.conv.g0.l1", "gdn.ssm.g0.l1",
        "gdn.conv.g0.l2", "gdn.ssm.g0.l2", "fa.l3", "gdn.conv.g0.l4",
        "gdn.ssm.g0.l4"
    ]
    assert set(
        manifest.geometry_by_tag()) == {"fa", "gdn.conv.g0", "gdn.ssm.g0"}
    assert [pool.tag for pool in manifest.pools_of_class("fa")] == ["fa.l3"]
    # The class view is what the default tagging produces for the same
    # layers.
    plain = rpm.build_qwen35_pool_manifest(
        named_kv_caches=named,
        kv_cache_groups=groups,
        raw_tensors=(),
        gdn_geometry=PCP8_GEOMETRY,
        mamba_group_ordinal_by_layer={
            name: 0
            for name in named if "linear_attn" in name
        },
    )
    assert manifest.geometry_by_tag() == plain.geometry_by_tag()


def test_layer_tag_helpers_round_trip():
    from vllm_torchtpu.distributed.kv_transfer.raiden import tags

    assert tags.layer_tag("fa", 3) == "fa.l3"
    assert tags.split_layer_tag("fa.l3") == ("fa", 3)
    assert tags.split_layer_tag("gdn.conv.g0.l12") == ("gdn.conv.g0", 12)
    assert tags.split_layer_tag("gdn.conv.g0") == ("gdn.conv.g0", None)
    assert tags.class_tag("fa.l3") == "fa"
    assert tags.class_tag("fa") == "fa"
    with pytest.raises(ValueError):
        tags.layer_tag("fa", -1)


def test_canonical_pool_order_is_layer_major_conv_before_ssm():
    _, manifest = _build_pcp8()
    # Layer 0..2 are GDN (conv, ssm), layer 3 is FA.
    assert [p.tag for p in manifest.pools[:7]] == [
        rpm.TAG_GDN_CONV,
        rpm.TAG_GDN_SSM,
        rpm.TAG_GDN_CONV,
        rpm.TAG_GDN_SSM,
        rpm.TAG_GDN_CONV,
        rpm.TAG_GDN_SSM,
        rpm.TAG_FA,
    ]
    assert [
        rpm.layer_index_from_name(p.layer_name) for p in manifest.pools[:7]
    ] == [0, 0, 1, 1, 2, 2, 3]


# --------------------------------------------------------------------------
# TP2DP4 decode manifest derives its FA geometry from the cache spec.
# --------------------------------------------------------------------------


def test_tp2dp4_decode_fa_geometry_is_derived():
    _, manifest = _build_tp2dp4()

    assert manifest.binding == rpm.BINDING_PRIVATE_TYPED
    assert len(manifest.pools) == 105
    geometry = manifest.geometry_by_tag()
    assert geometry[rpm.TAG_FA] == {
        "num_blocks": 16,
        "block_stride_bytes": 1_048_576,
        "live_bytes_per_block": 524_288,
    }
    assert geometry[rpm.TAG_GDN_CONV] == {
        "num_blocks": 16,
        "block_stride_bytes": 36_864,
        "live_bytes_per_block": 36_864,
    }
    assert geometry[rpm.TAG_GDN_SSM] == {
        "num_blocks": 16,
        "block_stride_bytes": 2_097_152,
        "live_bytes_per_block": 2_097_152,
    }

    fa_pool = next(p for p in manifest.pools if p.tag == rpm.TAG_FA)
    (region, ) = fa_pool.regions
    assert region.num_units == 1024
    assert (region.offset_bytes, region.stride_bytes, region.unit_bytes,
            region.units_per_stride) == (0, 1024, 512, 1)


# --------------------------------------------------------------------------
# Variant geometry: the builder follows the tensors, not constants.
# --------------------------------------------------------------------------


def test_variant_geometry_derives_from_tensors():
    geometry = rpm.GdnHeadGeometry(local_key_heads=4,
                                   local_value_heads=8,
                                   key_head_dim=64,
                                   value_head_dim=32)
    conv_dim = 2 * 4 * 64 + 8 * 32  # 768
    named = {
        "model.layers.0.linear_attn": (
            FakeTensor((8, 2, conv_dim), 2),
            FakeTensor((8, 8, 64, 32), 4),
        ),
        "model.layers.1.self_attn.attn":
        FakeTensor((16, 32, 1, 4, 64), 2),
    }
    groups = (
        _fa_group(["model.layers.1.self_attn.attn"],
                  block_size=64,
                  num_kv_heads=2,
                  head_size=64),
        _gdn_group(["model.layers.0.linear_attn"]),
    )
    manifest = rpm.build_qwen35_pool_manifest(named_kv_caches=named,
                                              kv_cache_groups=groups,
                                              raw_tensors=(),
                                              gdn_geometry=geometry)

    conv_pool, ssm_pool, fa_pool = manifest.pools
    q, k, v = conv_pool.regions
    assert q.offset_bytes == 0
    assert k.offset_bytes == 4 * 64 * 2  # key heads × key dim × esz
    assert v.offset_bytes == 2 * 4 * 64 * 2
    assert v.unit_bytes == 32 * 2
    assert conv_pool.block_stride_bytes == 2 * conv_dim * 2
    (ssm_region, ) = ssm_pool.regions
    assert ssm_region.unit_bytes == 64 * 32 * 4
    assert ssm_region.num_units == 8
    # fa: 16×32 = 512 tokens / 64-token blocks = 8 blocks.
    assert fa_pool.num_blocks == 8
    (fa_region, ) = fa_pool.regions
    assert fa_region.num_units == 64
    assert fa_region.units_per_stride == 2

    # Conv tensor that disagrees with the head geometry must be rejected.
    named_bad = dict(named)
    named_bad["model.layers.0.linear_attn"] = (
        FakeTensor((8, 2, conv_dim + 64), 2),
        named["model.layers.0.linear_attn"][1],
    )
    with pytest.raises(rpm.ManifestError, match="head geometry"):
        rpm.build_qwen35_pool_manifest(named_kv_caches=named_bad,
                                       kv_cache_groups=groups,
                                       raw_tensors=(),
                                       gdn_geometry=geometry)


# --------------------------------------------------------------------------
# Binding resolution with storage identity oracles.
# --------------------------------------------------------------------------


def _aliased_gdn_layer(raw, *, conv_shape, conv_esz, ssm_shape, ssm_esz):
    conv = FakeTensor(conv_shape,
                      conv_esz,
                      storage=raw.untyped_storage(),
                      storage_offset_elems=0)
    conv_bytes = 1
    for d in conv_shape[1:]:
        conv_bytes *= d
    conv_bytes *= conv_esz
    ssm = FakeTensor(ssm_shape,
                     ssm_esz,
                     storage=raw.untyped_storage(),
                     storage_offset_elems=conv_bytes // ssm_esz)
    return conv, ssm


def test_private_binding_uses_typed_storages():
    named, manifest = _build_pcp8()
    # Storage identity: every pool's storage IS one of the typed tensors.
    typed = set()
    for cache in named.values():
        tensors = cache if isinstance(cache, tuple) else (cache, )
        typed.update(id(t) for t in tensors)
    assert {id(s) for s in manifest.storages} == typed
    assert all(p.base_offset_bytes == 0 for p in manifest.pools)
    rpm.verify_storage_binding(manifest, named, raw_tensors=())


def test_aliased_binding_resolves_raw_storage_offsets():
    # One raw unified page per layer, PCP8 geometry: page 4,268,032 B,
    # conv at base 0, ssm at base 73,728.
    page = 4_268_032
    raw = FakeTensor((16 * page, ), 1, dtype="torch.int8")
    conv, ssm = _aliased_gdn_layer(raw,
                                   conv_shape=(16, 3, 12288),
                                   conv_esz=2,
                                   ssm_shape=(16, 64, 128, 128),
                                   ssm_esz=4)
    named = {"model.layers.0.linear_attn": (conv, ssm)}
    groups = (_gdn_group(["model.layers.0.linear_attn"]), )
    manifest = rpm.build_qwen35_pool_manifest(named_kv_caches=named,
                                              kv_cache_groups=groups,
                                              raw_tensors=(raw, ),
                                              gdn_geometry=PCP8_GEOMETRY)

    assert manifest.binding == rpm.BINDING_ALIASED_RAW
    assert manifest.storages == [raw]
    conv_pool, ssm_pool = manifest.pools
    assert conv_pool.block_stride_bytes == page
    assert conv_pool.base_offset_bytes == 0
    assert ssm_pool.block_stride_bytes == page
    assert ssm_pool.base_offset_bytes == 73_728
    rpm.verify_storage_binding(manifest, named, raw_tensors=(raw, ))


@pytest.mark.parametrize(("ssm_dtype", "ssm_itemsize"), [
    ("torch.float32", 4),
    ("torch.bfloat16", 2),
])
def test_unified_pool_derives_logical_gdn_views_without_torch_ops(
        ssm_dtype, ssm_itemsize):
    geometry = rpm.GdnHeadGeometry(local_key_heads=4,
                                   local_value_heads=8,
                                   key_head_dim=64,
                                   value_head_dim=32,
                                   conv_kernel_size=3)
    conv_shape = (2, 768)
    ssm_shape = (8, 64, 32)
    manager_page_bytes = 131_072
    # Four attention-shaped manager pages.  The same object is bound to FA,
    # exposed as [pool] to GDN, and listed as the runner's raw tensor.
    pool = FakeTensor((4, 128, 1, 4, 256), 1, dtype="torch.float8_e4m3fn")
    named = {
        "model.layers.0.linear_attn": [pool],
        "model.layers.1.self_attn.attn": pool,
    }
    groups = (
        _pooled_gdn_group(["model.layers.0.linear_attn"],
                          shapes=(conv_shape, ssm_shape),
                          dtypes=("torch.bfloat16", ssm_dtype),
                          page_size_bytes=manager_page_bytes),
        _fa_group(["model.layers.1.self_attn.attn"],
                  block_size=128,
                  num_kv_heads=2,
                  head_size=256),
    )
    manifest = rpm.build_qwen35_pool_manifest(
        named_kv_caches=named,
        kv_cache_groups=groups,
        raw_tensors=(pool, ),
        gdn_geometry=geometry,
        mamba_group_ordinal_by_layer={"model.layers.0.linear_attn": 0},
    )

    assert manifest.binding == rpm.BINDING_ALIASED_RAW
    assert manifest.storages == [pool]
    assert len(manifest.pools) == 3
    conv, ssm, fa = manifest.pools
    assert [entry.tag for entry in manifest.pools] == [
        f"{rpm.TAG_GDN_CONV}.g0",
        f"{rpm.TAG_GDN_SSM}.g0",
        rpm.TAG_FA,
    ]
    assert all(entry.storage_index == 0 for entry in manifest.pools)
    assert all(entry.num_blocks == 4 for entry in manifest.pools)
    assert all(entry.block_stride_bytes == manager_page_bytes
               for entry in manifest.pools)
    ssm_bytes = 8 * 64 * 32 * ssm_itemsize
    assert (conv.base_offset_bytes, conv.dtype_tag,
            conv.live_bytes_per_block) == (ssm_bytes, "bfloat16", 2 * 768 * 2)
    assert (ssm.base_offset_bytes, ssm.dtype_tag,
            ssm.live_bytes_per_block) == (0, ssm_dtype.removeprefix("torch."),
                                          ssm_bytes)
    assert (fa.base_offset_bytes, fa.dtype_tag,
            fa.live_bytes_per_block) == (0, "float8_e4m3fn",
                                         manager_page_bytes)
    assert max(region.extent_end_bytes
               for region in conv.regions) + conv.base_offset_bytes < \
        manager_page_bytes
    rpm.verify_storage_binding(manifest, named, raw_tensors=(pool, ))


def test_unified_pool_views_are_kernel_tied_not_spec_tied():
    """A drifted Conv MambaSpec declaration must not leak into the manifest.

    The layer-level ``get_state_shape``/``get_state_dtype`` overrides can
    diverge from the pooled kernel's fixed Conv byte model (PR #437 declared
    the Conv state FP32 with an extra singleton dim). The manifest describes
    the Conv bytes the kernel actually moves, so manifests built from drifted
    and kernel-matching Conv declarations must be identical. SSM storage dtype
    remains spec-defined.
    """
    geometry = rpm.GdnHeadGeometry(local_key_heads=4,
                                   local_value_heads=8,
                                   key_head_dim=64,
                                   value_head_dim=32,
                                   conv_kernel_size=3)
    layer = "model.layers.0.linear_attn"
    page = 131_072
    pool = FakeTensor((4, 128, 1, 4, 256), 1, dtype="torch.float8_e4m3fn")

    def build(shapes, dtypes):
        groups = (_pooled_gdn_group([layer],
                                    shapes=shapes,
                                    dtypes=dtypes,
                                    page_size_bytes=page), )
        return rpm.build_qwen35_pool_manifest(named_kv_caches={layer: [pool]},
                                              kv_cache_groups=groups,
                                              raw_tensors=(pool, ),
                                              gdn_geometry=geometry)

    drifted = build(shapes=((2, 1, 768), (8, 64, 32)),
                    dtypes=("torch.float32", "torch.float32"))
    matching = build(shapes=((2, 768), (8, 64, 32)),
                     dtypes=("torch.bfloat16", "torch.float32"))
    assert drifted.pool_dicts() == matching.pool_dicts()

    conv, ssm = drifted.pools
    ssm_bytes = 8 * 64 * 32 * 4
    assert (conv.dtype_tag, conv.base_offset_bytes,
            conv.live_bytes_per_block) == ("bfloat16", ssm_bytes, 2 * 768 * 2)
    assert (ssm.dtype_tag, ssm.base_offset_bytes,
            ssm.live_bytes_per_block) == ("float32", 0, ssm_bytes)


def test_unified_pool_conv_region_matches_kernel_state_plan():
    """Pin the manifest's state model to the pooled kernel's state plan.

    Uses the live Qwen3.5 full-width geometry and the same
    ``gdn_pool_layout`` helpers ``_build_v3_pool_state_plan`` consumes, so a
    kernel-side layout change breaks this test instead of the nightly
    reshard pair.
    """
    from vllm_torchtpu.gdn_pool_layout import (derive_pooled_gdn_state_layout,
                                               pooled_gdn_conv_state_bytes,
                                               pooled_gdn_ssm_state_bytes)

    geometry = PCP8_GEOMETRY  # 16 K heads / 64 V heads / 128-dim, kernel 4
    fa_page = 8_388_608
    token_bytes = 1024
    conv_bytes = pooled_gdn_conv_state_bytes(
        kernel_size=geometry.conv_kernel_size, conv_dim=geometry.conv_dim)
    ssm_bytes = pooled_gdn_ssm_state_bytes(
        num_v_heads=geometry.local_value_heads,
        head_k_dim=geometry.key_head_dim,
        head_v_dim=geometry.value_head_dim)

    layer = "model.layers.0.linear_attn"
    pool = FakeTensor((2 * fa_page, ), 1, dtype="torch.float8_e4m3fn")
    groups = (
        _pooled_gdn_group(
            [layer],
            # Post-#437 declaration: ignored by the kernel-tied manifest.
            shapes=((3, 1, 12288), (64, 128, 128)),
            dtypes=("torch.float32", "torch.float32"),
            page_size_bytes=fa_page,
        ), )
    manifest = rpm.build_qwen35_pool_manifest(named_kv_caches={layer: [pool]},
                                              kv_cache_groups=groups,
                                              raw_tensors=(pool, ),
                                              gdn_geometry=geometry)

    conv, ssm = manifest.pools
    assert conv.live_bytes_per_block == conv_bytes == 73_728
    assert ssm.live_bytes_per_block == ssm_bytes == 4_194_304
    # The conv view starts exactly where the kernel's token plan places the
    # conv region (right after the SSM tokens).
    kernel_plan = derive_pooled_gdn_state_layout(ssm_bytes=ssm_bytes,
                                                 conv_bytes=conv_bytes,
                                                 token_bytes=token_bytes)
    assert conv.base_offset_bytes == kernel_plan.ssm_tokens * token_bytes
    assert kernel_plan.required_tokens * token_bytes <= fa_page


@pytest.mark.parametrize(
    ("shapes", "dtypes", "page_size", "raw_size", "match"),
    [
        (((2, 768), ), ("torch.bfloat16", ), 131_072, 524_288,
         "requires conv and SSM specs"),
        (((2, 768), (8, 64, 32)), ("torch.bfloat16", "torch.float32"), 65_536,
         524_288, "exceed one manager page"),
        (((2, 768), (8, 64, 32)), ("torch.bfloat16", "torch.float32"), 131_072,
         524_289, "must be divisible"),
    ],
)
def test_unified_pool_rejects_invalid_gdn_spec(shapes, dtypes, page_size,
                                               raw_size, match):
    pool = FakeTensor((raw_size, ), 1, dtype="torch.float8_e4m3fn")
    layer = "model.layers.0.linear_attn"
    named = {layer: [pool]}
    groups = (_pooled_gdn_group([layer],
                                shapes=shapes,
                                dtypes=dtypes,
                                page_size_bytes=page_size), )
    geometry = rpm.GdnHeadGeometry(local_key_heads=4,
                                   local_value_heads=8,
                                   key_head_dim=64,
                                   value_head_dim=32,
                                   conv_kernel_size=3)
    with pytest.raises(rpm.ManifestError, match=match):
        rpm.build_qwen35_pool_manifest(named_kv_caches=named,
                                       kv_cache_groups=groups,
                                       raw_tensors=(pool, ),
                                       gdn_geometry=geometry)


def test_unified_pool_requires_a_complete_listed_raw_storage():
    layer = "model.layers.0.linear_attn"
    pool = FakeTensor((524_288, ), 1, dtype="torch.float8_e4m3fn")
    groups = (_pooled_gdn_group(
        [layer],
        shapes=((2, 768), (8, 64, 32)),
        dtypes=("torch.bfloat16", "torch.float32"),
        page_size_bytes=131_072,
    ), )
    geometry = rpm.GdnHeadGeometry(local_key_heads=4,
                                   local_value_heads=8,
                                   key_head_dim=64,
                                   value_head_dim=32)
    with pytest.raises(rpm.ManifestError, match="must share raw pool storage"):
        rpm.build_qwen35_pool_manifest(named_kv_caches={layer: [pool]},
                                       kv_cache_groups=groups,
                                       raw_tensors=(),
                                       gdn_geometry=geometry)

    partial = FakeTensor((262_144, ),
                         1,
                         dtype="torch.float8_e4m3fn",
                         storage=pool.untyped_storage())
    with pytest.raises(rpm.ManifestError, match="complete raw pool"):
        rpm.build_qwen35_pool_manifest(named_kv_caches={layer: [partial]},
                                       kv_cache_groups=groups,
                                       raw_tensors=(pool, ),
                                       gdn_geometry=geometry)


def test_mixed_binding_is_rejected():
    page = 4_268_032
    raw = FakeTensor((16 * page, ), 1, dtype="torch.int8")
    conv, ssm = _aliased_gdn_layer(raw,
                                   conv_shape=(16, 3, 12288),
                                   conv_esz=2,
                                   ssm_shape=(16, 64, 128, 128),
                                   ssm_esz=4)
    private_fa = FakeTensor((256, 256, 1, 4, 256), 1)
    named = {
        "model.layers.0.linear_attn": (conv, ssm),
        "model.layers.3.self_attn.attn": private_fa,
    }
    groups = (
        _gdn_group(["model.layers.0.linear_attn"]),
        _fa_group(["model.layers.3.self_attn.attn"],
                  block_size=4096,
                  num_kv_heads=2,
                  head_size=256),
    )
    with pytest.raises(rpm.ManifestError, match="mixed KV cache binding"):
        rpm.build_qwen35_pool_manifest(named_kv_caches=named,
                                       kv_cache_groups=groups,
                                       raw_tensors=(raw, ),
                                       gdn_geometry=PCP8_GEOMETRY)


# --------------------------------------------------------------------------
# Registering dead raw storage must hard-fail.
# --------------------------------------------------------------------------


def test_dead_raw_storage_hard_fails():
    named, manifest = _build_pcp8()
    # The runner still allocates raw unified pages under the alias fallback;
    # they share storage with nothing the kernels touch.
    raw_tensors = tuple(
        FakeTensor((16 * 4_268_032, ), 1, dtype="torch.int8")
        for _ in range(15))

    # The correctly-built manifest passes.
    rpm.verify_storage_binding(manifest, named, raw_tensors)

    # Registering the raw unified pages while typed caches are private must be
    # impossible to express silently.
    dead_raw_manifest = rpm.PoolManifest(binding=rpm.BINDING_PRIVATE_TYPED,
                                         storages=list(raw_tensors),
                                         pools=manifest.pools)
    with pytest.raises(rpm.DeadStorageError, match="raw unified pages"):
        rpm.verify_storage_binding(dead_raw_manifest, named, raw_tensors)

    # Missing typed storages (partial coverage) also fail.
    partial = rpm.PoolManifest(binding=rpm.BINDING_PRIVATE_TYPED,
                               storages=manifest.storages[:-1],
                               pools=manifest.pools)
    with pytest.raises(rpm.DeadStorageError, match="do not match"):
        rpm.verify_storage_binding(partial, named, raw_tensors)


# --------------------------------------------------------------------------
# Manifest serialization round-trip (future stage-3 handshake payload).
# --------------------------------------------------------------------------


def test_pool_dicts_round_trip():
    import json

    _, manifest = _build_tp2dp4()
    payload = json.dumps({
        "binding": manifest.binding,
        "pools": manifest.pool_dicts(),
    })
    decoded = json.loads(payload)
    assert decoded["binding"] == rpm.BINDING_PRIVATE_TYPED
    assert len(decoded["pools"]) == 105
    fa_dicts = [p for p in decoded["pools"] if p["tag"] == rpm.TAG_FA]
    assert fa_dicts[0]["block_stride_bytes"] == 1_048_576
    assert fa_dicts[0]["regions"][0]["num_units"] == 1024
    assert fa_dicts[0]["dtype_tag"] == "float8_e4m3fn"


def test_pool_dicts_coerce_into_raiden_pool_specs():
    pool_layout = pytest.importorskip("tpu_sync.api.torch.pool_layout")

    _, manifest = _build_tp2dp4()
    for pool_dict, entry in zip(manifest.pool_dicts(), manifest.pools):
        spec = pool_layout.coerce_pool_spec(pool_dict)
        spec.validate()
        assert spec.tag == entry.tag
        assert spec.block_stride_bytes == entry.block_stride_bytes
        assert spec.live_bytes_per_block == entry.live_bytes_per_block


# --------------------------------------------------------------------------
# GLM-5.2 MLA manifest (row-granular regions over replicated caches).
# --------------------------------------------------------------------------

_GLM_BLOCK_SIZE = 1024


def test_glm_manifest_requires_private_typed_binding():
    raw = FakeTensor((16, 3 * 256 * 2560), 1, dtype="torch.uint8")
    named = {
        "model.layers.0.self_attn.mla_attn": (
            FakeTensor((16, 1024, 4, 128),
                       1,
                       dtype="torch.uint8",
                       storage=raw.untyped_storage()),
            FakeTensor((16, 256, 4, 128),
                       1,
                       dtype="torch.uint8",
                       storage=raw.untyped_storage()),
        ),
        "model.layers.0.self_attn.indexer":
        FakeTensor((16, 256, 4, 256),
                   1,
                   dtype="torch.uint8",
                   storage=raw.untyped_storage()),
    }
    with pytest.raises(rpm.ManifestError, match="private typed"):
        rpm.build_glm_mla_pool_manifest(named_kv_caches=named,
                                        raw_tensors=(raw, ),
                                        block_size_tokens=_GLM_BLOCK_SIZE)


def test_glm_manifest_sparse_mla_pairs():
    named = glm_named_kv_caches()
    manifest = rpm.build_glm_mla_pool_manifest(
        named_kv_caches=named,
        raw_tensors=(),
        block_size_tokens=_GLM_BLOCK_SIZE)

    assert manifest.binding == rpm.BINDING_PRIVATE_TYPED
    assert manifest.tag_counts() == {
        rpm.TAG_MLA_NOPE: 2,
        rpm.TAG_MLA_ROPE: 2,
        rpm.TAG_DSA_IDX: 1,
    }
    assert len(manifest.storages) == 5
    geometry = manifest.geometry_by_tag()
    assert geometry[rpm.TAG_MLA_NOPE] == {
        "num_blocks": 16,
        "block_stride_bytes": 1024 * 512,
        "live_bytes_per_block": 1024 * 512,
    }
    assert geometry[rpm.TAG_MLA_ROPE]["live_bytes_per_block"] == 256 * 512
    assert geometry[rpm.TAG_DSA_IDX]["live_bytes_per_block"] == 256 * 1024
    for pool in manifest.pools:
        (region, ) = pool.regions
        assert region.offset_bytes == 0
        assert region.stride_bytes == region.unit_bytes
        assert region.units_per_stride == 1
        assert pool.base_offset_bytes == 0
        if pool.tag == rpm.TAG_MLA_NOPE:
            assert region.name == "mla_nope_rows"
            assert region.unit_bytes == 512
            assert region.num_units == 1024
        elif pool.tag == rpm.TAG_MLA_ROPE:
            assert region.name == "mla_rope_rows"
            assert region.unit_bytes == 512
            assert region.num_units == 256
        else:
            assert region.name == "dsa_rows"
            assert region.unit_bytes == 1024
            assert region.num_units == 256
    rpm.verify_storage_binding(manifest, named, raw_tensors=())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(nope_shape=(16, 512, 4, 128)), "nope rows 512 do not match"),
        (dict(rope_shape=(16, 512, 4, 128)), "KV block size"),
        (dict(nope_shape=(16, 1024, 2, 128)), "32-bit word"),
        (dict(nope_shape=(16, 1024, 512)),
         r"\[blocks, rows, packing, width\]"),
        (dict(rope_shape=(16, 256, 4, 120)), "lane-aligned"),
        (dict(idx_shape=(8, 256, 4, 256)), "disagree on num_blocks"),
    ],
)
def test_glm_manifest_rejects_bad_geometry(overrides, message):
    named = glm_named_kv_caches(**overrides)
    with pytest.raises(rpm.ManifestError, match=message):
        rpm.build_glm_mla_pool_manifest(named_kv_caches=named,
                                        raw_tensors=(),
                                        block_size_tokens=_GLM_BLOCK_SIZE)
