# SPDX-License-Identifier: Apache-2.0
"""Deviceless goldens for the Stage-3 physical-layout identity."""

import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import \
    layout_fingerprint as rlf
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm


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
                regions=(rpm.RegionSpec(
                    name="fa_payload",
                    offset_bytes=0,
                    stride_bytes=1024,
                    unit_bytes=512,
                    num_units=page_tokens,
                    units_per_stride=2,
                ), ),
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
        "gdn_conv_layout": "legacy-split-qk",
    }
    assert fingerprint == (
        "0ddf18228f657fb11c8b6a5ddf0826b1f3020cf88fde96833832eedbd3cf9811")
    assert rlf.canonical_layout_fingerprint(
        dict(reversed(list(payload.items())))) == fingerprint
    assert rlf.fa_page_tokens(manifest) == 4096


def test_fingerprint_diverges_across_gdn_conv_layouts(monkeypatch):
    """A mixed-layout disagg pair must not share a fingerprint: the conv
    geometries are size-identical, so the layout version in the payload is
    the only thing that fails such a pair closed."""

    def measure():
        return rlf.measured_fa_layout_fingerprint(
            _manifest(),
            layout_getter=lambda tensor:
            ([4, 3, 2, 1, 0], [[4, 128], [4, 1]], 0),
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
