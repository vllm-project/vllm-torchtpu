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
"""Deviceless goldens for the Stage-3 physical-layout identity."""

import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import layout_fingerprint as rlf
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm

from .raiden_test_utils import FakeTensor, dsv4_materialization, glm_named_kv_caches


def _manifest(*, page_tokens=4096):
    storage = object()
    return rpm.PoolManifest(
        binding=rpm.BINDING_PRIVATE_TYPED,
        storages=[storage],
        pools=[
            rpm.PoolEntry(
                tag=rpm.TAG_FA,
                layer_name="model.layers.3.self_attn.attn",
                storage_index=0,
                base_offset_bytes=0,
                block_stride_bytes=page_tokens * 1024,
                num_blocks=16,
                regions=(
                    rpm.RegionSpec(
                        name="fa_payload",
                        offset_bytes=0,
                        stride_bytes=1024,
                        unit_bytes=512,
                        num_units=page_tokens,
                        units_per_stride=2,
                    ),
                ),
                dtype_tag="float8_e4m3fn",
            )
        ],
    )


def test_measured_fa_layout_fingerprint_golden(monkeypatch):
    monkeypatch.delenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", raising=False)
    manifest = _manifest()
    fingerprint, payload = rlf.measured_fa_layout_fingerprint(
        manifest,
        layout_getter=lambda tensor: ([4, 3, 2, 1, 0], [[4, 128], [4, 1]], 0),
        package_version=lambda package: {
            "torch_tpu": "0.1.1.dev20260630090312",
            "libtpu": "0.0.42.1",
        }[package],
    )

    assert payload == {
        "schema": "qwen35-fa-raw-layout-fingerprint-v1",
        "torch_tpu": "0.1.1.dev20260630090312",
        "libtpu": "0.0.42.1",
        "minor_to_major": [4, 3, 2, 1, 0],
        "tiles": [[4, 128], [4, 1]],
        "element_size_in_bits": 8,
        "kv_layout": "layer-major",
        "gdn_conv_layout": "legacy-split-qk",
    }
    assert fingerprint == (
        "491e9b375a38f65d175e6e38919d29af59bf5149a2c617d0bea69872e3431b77"
    )
    assert (
        rlf.canonical_layout_fingerprint(dict(reversed(list(payload.items()))))
        == fingerprint
    )
    assert rlf.fa_page_tokens(manifest) == 4096


def test_block_major_fingerprint_names_the_layout(monkeypatch):
    monkeypatch.delenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", raising=False)
    manifest = _manifest()
    versions = {"torch_tpu": "0.1.1.dev20260630090312", "libtpu": "0.0.42.1"}
    row_layout = ([5, 4, 3, 2, 1, 0], [[4, 128], [4, 1]], 8)
    page_layout = ([4, 3, 2, 1, 0], [[4, 128], [4, 1]], 8)
    bm_fingerprint, bm_payload = rlf.measured_fa_layout_fingerprint(
        manifest,
        block_major=True,
        layout_getter=lambda tensor: row_layout,
        package_version=versions.__getitem__,
    )
    lm_fingerprint, lm_payload = rlf.measured_fa_layout_fingerprint(
        manifest,
        layout_getter=lambda tensor: page_layout,
        package_version=versions.__getitem__,
    )
    assert bm_payload["kv_layout"] == "block-major"
    assert bm_payload["minor_to_major"] == [5, 4, 3, 2, 1, 0]
    assert lm_payload["kv_layout"] == "layer-major"
    assert bm_fingerprint != lm_fingerprint
    # Each layout accepts only its own rank.
    with pytest.raises(RuntimeError, match="minor-to-major"):
        rlf.measured_fa_layout_fingerprint(
            manifest,
            block_major=True,
            layout_getter=lambda tensor: page_layout,
            package_version=versions.__getitem__,
        )
    with pytest.raises(RuntimeError, match="minor-to-major"):
        rlf.measured_fa_layout_fingerprint(
            manifest,
            layout_getter=lambda tensor: row_layout,
            package_version=versions.__getitem__,
        )


def test_fingerprint_diverges_across_gdn_conv_layouts(monkeypatch):
    """A mixed-layout disagg pair must not share a fingerprint: the conv
    geometries are size-identical, so the layout version in the payload is
    the only thing that fails such a pair closed."""

    def measure():
        return rlf.measured_fa_layout_fingerprint(
            _manifest(),
            layout_getter=lambda tensor: ([4, 3, 2, 1, 0], [[4, 128], [4, 1]], 0),
            package_version=lambda package: "unused",
        )

    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")
    pair_fingerprint, pair_payload = measure()
    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "0")
    legacy_fingerprint, legacy_payload = measure()

    assert pair_payload["gdn_conv_layout"] == "qk-pair-v1"
    assert legacy_payload["gdn_conv_layout"] == "legacy-split-qk"
    assert pair_fingerprint != legacy_fingerprint


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        (([3, 4, 2, 1, 0], [[4, 128], [4, 1]], 8), "minor-to-major"),
        (([4, 3, 2, 1, 0], [[8, 128], [4, 1]], 8), "tile gate"),
        (([4, 3, 2, 1, 0], [[4, 128], [4, 1]], 16), "element-size"),
    ],
)
def test_measured_fa_layout_fingerprint_fails_closed(layout, message):
    with pytest.raises(RuntimeError, match=message):
        rlf.measured_fa_layout_fingerprint(
            _manifest(),
            layout_getter=lambda tensor: layout,
            package_version=lambda package: "unused",
        )


def test_fa_page_tokens_rejects_divergent_pool_geometry():
    manifest = _manifest()
    second = _manifest(page_tokens=1024).pools[0]
    manifest.pools.append(second)
    with pytest.raises(ValueError, match="differs across pools"):
        rlf.fa_page_tokens(manifest)


# ---------------------------------------------------------------------------
# GLM-5.2 MLA fingerprint (row-granular transfers).
# ---------------------------------------------------------------------------


def _glm_layout_getter(tensor):
    if len(tensor.shape) == 3:
        # DSv4's int32 word rows, as measured on TPU7x: one (8, 128) tile,
        # shrunk to the row count on pages under 8 rows.
        return ([2, 1, 0], [[min(8, tensor.shape[1]), 128]], 0)
    packing = tensor.shape[2]
    return ([3, 2, 1, 0], [[packing, 128], [packing, 1]], 0)


def _glm_versions(package):
    return {
        "torch_tpu": "torch-tpu-test-version",
        "libtpu": "libtpu-test-version",
    }[package]


@pytest.mark.parametrize(
    ("layout", "page_tokens", "message"),
    [
        (([2, 3, 1, 0], [[4, 128], [4, 1]], 0), 1024, "physical order"),
        (([3, 2, 1, 0], [[8, 128], [4, 1]], 0), 1024, "tile shape"),
        (None, 1024, "no materialized"),
        (([3, 2, 1, 0], [[4, 128], [4, 1]], 0), 512, "page geometry"),
    ],
)
def test_measured_glm_layout_fingerprint_fails_closed(layout, page_tokens, message):
    with pytest.raises(RuntimeError, match=message):
        rlf.measured_glm_layout_fingerprint(
            _glm_manifest(),
            page_tokens=page_tokens,
            layout_getter=lambda tensor: layout,
            package_version=lambda package: "unused",
        )


def _glm_manifest(**kwargs):
    return rpm.build_glm_mla_pool_manifest(
        named_kv_caches=glm_named_kv_caches(**kwargs),
        raw_tensors=(),
        block_size_tokens=1024,
    )


def test_measured_glm_layout_fingerprint_golden():
    fingerprint, payload = rlf.measured_glm_layout_fingerprint(
        _glm_manifest(),
        page_tokens=1024,
        layout_getter=_glm_layout_getter,
        package_version=_glm_versions,
    )

    # Per-page geometry only: prefill and decode admit different pool
    # capacities, so num_blocks must not enter the identity.
    assert payload == {
        "schema": "glm-mla-raw-layout-fingerprint-v1",
        "torch_tpu": "torch-tpu-test-version",
        "libtpu": "libtpu-test-version",
        "page_tokens": 1024,
        "layouts": {
            "mla.nope": {
                "page_shape": [1024, 4, 128],
                "row_bytes": 512,
                "minor_to_major": [3, 2, 1, 0],
                "tiles": [[4, 128], [4, 1]],
                "element_size_in_bits": 8,
            },
            "mla.rope": {
                "page_shape": [256, 4, 128],
                "row_bytes": 512,
                "minor_to_major": [3, 2, 1, 0],
                "tiles": [[4, 128], [4, 1]],
                "element_size_in_bits": 8,
            },
            "dsa.idx": {
                "page_shape": [256, 4, 256],
                "row_bytes": 1024,
                "minor_to_major": [3, 2, 1, 0],
                "tiles": [[4, 128], [4, 1]],
                "element_size_in_bits": 8,
            },
        },
    }
    assert fingerprint == (
        "bfea2b664510bfb11f87b5092431492fddaf535f0f321f01d59a79e139c3bc2b"
    )


def test_glm_fingerprint_rejects_shape_divergence_within_tag():
    manifest = _glm_manifest()
    divergent = _glm_manifest(rope_shape=(16, 256, 4, 256))
    rope_pool = next(p for p in divergent.pools if p.tag == rpm.TAG_MLA_ROPE)
    manifest.pools.append(rope_pool)
    manifest.storages.append(FakeTensor((16, 256, 4, 256), 1, dtype="torch.uint8"))
    manifest.pools[-1].storage_index = len(manifest.storages) - 1
    with pytest.raises(RuntimeError, match="disagree on shape"):
        rlf.measured_glm_layout_fingerprint(
            manifest,
            page_tokens=1024,
            layout_getter=_glm_layout_getter,
            package_version=lambda package: "unused",
        )


def _glm_sparsecore_manifest():
    # TPU_SPARSE_MLA_{NOPE,ROPE}_LAYOUT=sparsecore: uint32 word rows.
    named = glm_named_kv_caches()
    for idx in range(2):
        named[f"model.layers.{idx}.self_attn.mla_attn"] = (
            FakeTensor((16, 1024, 128), 4, dtype="torch.uint32"),
            FakeTensor((16, 256, 128), 4, dtype="torch.uint32"),
        )
    return rpm.build_glm_mla_pool_manifest(
        named_kv_caches=named, raw_tensors=(), block_size_tokens=1024
    )


def test_measured_glm_layout_fingerprint_accepts_sparsecore_rows():
    fingerprint, payload = rlf.measured_glm_layout_fingerprint(
        _glm_sparsecore_manifest(),
        page_tokens=1024,
        layout_getter=_glm_layout_getter,
        package_version=_glm_versions,
    )

    word_rows = {
        "row_bytes": 512,
        "minor_to_major": [2, 1, 0],
        "tiles": [[8, 128]],
        "element_size_in_bits": 32,
    }
    assert payload["layouts"]["mla.nope"] == word_rows | {"page_shape": [1024, 1, 128]}
    assert payload["layouts"]["mla.rope"] == word_rows | {"page_shape": [256, 1, 128]}
    # Same bytes per row, but a tensorcore peer must not pair with it.
    tensorcore, _ = rlf.measured_glm_layout_fingerprint(
        _glm_manifest(),
        page_tokens=1024,
        layout_getter=_glm_layout_getter,
        package_version=_glm_versions,
    )
    assert fingerprint != tensorcore


@pytest.mark.parametrize(
    ("layout", "page_tokens", "message"),
    [
        (([1, 2, 0], [[8, 128]], 0), 1024, "physical order"),
        (([2, 1, 0], [[8, 256]], 0), 1024, "tile shape"),
        (([2, 1, 0], [[3, 128]], 0), 1024, "tile shape"),
        (([2, 1, 0], [[8, 128]], 0), 512, "page geometry"),
    ],
)
def test_measured_glm_sparsecore_fingerprint_fails_closed(layout, page_tokens, message):
    def getter(tensor):
        return layout if len(tensor.shape) == 3 else _glm_layout_getter(tensor)

    with pytest.raises(RuntimeError, match=message):
        rlf.measured_glm_layout_fingerprint(
            _glm_sparsecore_manifest(),
            page_tokens=page_tokens,
            layout_getter=getter,
            package_version=lambda package: "unused",
        )


# ---------------------------------------------------------------------------
# DeepSeek-V4 fingerprint (per-role page geometry).
# ---------------------------------------------------------------------------


def _dsv4_manifest(**kwargs):
    named_kv_caches, kv_cache_groups = dsv4_materialization(**kwargs)
    return rpm.build_dsv4_pool_manifest(
        named_kv_caches=named_kv_caches, kv_cache_groups=kv_cache_groups, raw_tensors=()
    )


def test_measured_dsv4_layout_fingerprint_records_every_role():
    _, payload = rlf.measured_dsv4_layout_fingerprint(
        _dsv4_manifest(),
        layout_getter=_glm_layout_getter,
        package_version=_glm_versions,
    )

    assert payload["schema"] == "dsv4-raw-layout-fingerprint-v1"
    # No page_tokens key: the roles disagree on how many logical tokens sit on
    # a page, so the identity is per-tag page geometry instead.
    assert "page_tokens" not in payload
    assert set(payload["layouts"]) == {
        "dsv4.csa.nope.g0",
        "dsv4.csa.rope.g0",
        "dsv4.idx.g1",
        "dsv4.hca.g2",
        "dsv4.swa.g3",
        "dsv4.state.csa.g4",
        "dsv4.state.idx.g5",
        "dsv4.state.hca.g6",
    }
    # CSA is int32 word rows: packing-1 rows of the same 512 bytes.
    assert payload["layouts"]["dsv4.csa.nope.g0"] == {
        "page_shape": [256, 1, 128],
        "row_bytes": 512,
        "minor_to_major": [2, 1, 0],
        "tiles": [[8, 128]],
        "element_size_in_bits": 32,
    }
    assert payload["layouts"]["dsv4.hca.g2"]["page_shape"] == [512, 4, 128]
    assert payload["layouts"]["dsv4.idx.g1"]["row_bytes"] == 1024


def test_dsv4_fingerprint_matches_its_host_array_for_aliased_pools():
    _, payload = rlf.measured_dsv4_layout_fingerprint(
        _dsv4_manifest(),
        layout_getter=_glm_layout_getter,
        package_version=_glm_versions,
    )
    host = payload["layouts"]["dsv4.csa.nope.g0"]
    for aliased in ("dsv4.swa.g3", "dsv4.state.csa.g4", "dsv4.state.hca.g6"):
        assert payload["layouts"][aliased] == host


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        (([2, 3, 1, 0], [[4, 128], [4, 1]], 0), "physical order"),
        (([2, 1, 0], [[8, 128], [4, 1]], 0), "tile shape"),
        # Taller than the 64-row RoPE page, so it would pad the page.
        (([2, 1, 0], [[128, 128]], 0), "tile shape"),
        (None, "no materialized"),
    ],
)
def test_measured_dsv4_layout_fingerprint_fails_closed(layout, message):
    with pytest.raises(RuntimeError, match=message):
        rlf.measured_dsv4_layout_fingerprint(
            _dsv4_manifest(),
            layout_getter=lambda tensor: layout,
            package_version=lambda package: "unused",
        )


def test_dsv4_fingerprint_admits_the_shrunk_tile_of_a_small_page():
    """At 16 tokens per page RoPE has 4 rows, which XLA tiles (4, 128)."""
    _, payload = rlf.measured_dsv4_layout_fingerprint(
        _dsv4_manifest(tokens_per_page=16),
        layout_getter=_glm_layout_getter,
        package_version=_glm_versions,
    )
    assert payload["layouts"]["dsv4.csa.rope.g0"]["tiles"] == [[4, 128]]


def test_dsv4_fingerprint_is_stable_across_pool_capacity():
    """Prefill and decode admit different block counts on the same layout."""
    small, _ = rlf.measured_dsv4_layout_fingerprint(
        _dsv4_manifest(num_blocks=16),
        layout_getter=_glm_layout_getter,
        package_version=_glm_versions,
    )
    large, _ = rlf.measured_dsv4_layout_fingerprint(
        _dsv4_manifest(num_blocks=64),
        layout_getter=_glm_layout_getter,
        package_version=_glm_versions,
    )
    assert small == large


def test_glm_stage_without_indexer_fingerprints_like_every_layer():
    # A pipeline stage holding only layer 1 has no indexer cache; with the
    # indexer class left out of the hash it matches a peer holding all.
    full = glm_named_kv_caches()
    layer = "model.layers.1.self_attn.mla_attn"
    stage = {layer: full[layer]}

    def fingerprint(named, **kwargs):
        manifest = rpm.build_glm_mla_pool_manifest(
            named_kv_caches=named,
            raw_tensors=(),
            block_size_tokens=1024,
            per_layer_tags=True,
        )
        return rlf.measured_glm_layout_fingerprint(
            manifest,
            page_tokens=1024,
            layout_getter=_glm_layout_getter,
            package_version=_glm_versions,
            **kwargs,
        )

    stage_fp, stage_payload = fingerprint(stage, hash_indexer=False)
    full_fp, full_payload = fingerprint(full, hash_indexer=False)
    assert stage_fp == full_fp
    assert set(full_payload["layouts"]) == {"mla.nope", "mla.rope"}
    assert fingerprint(full)[0] != full_fp
