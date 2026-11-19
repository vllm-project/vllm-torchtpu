# SPDX-License-Identifier: Apache-2.0
"""Measured page-layout admission and an isolated raw-byte TPU calibration."""

from dataclasses import replace

import pytest

from tests.runner.test_qwen4_exp_pd_manifest import live_pool
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import ManifestError
from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import (
    build_transfer_manifest,
    measure_transfer_layout,
)

_LAYOUT = ((4, 3, 2, 1, 0), ((4, 128), (4, 1)), 8)


def test_fingerprint_reads_each_carrier_and_includes_page_geometry():
    config, materialized = live_pool()
    manifest = build_transfer_manifest(config, materialized)
    visited = []

    def read(raw):
        visited.append(raw)
        return _LAYOUT

    fingerprint, payload = measure_transfer_layout(
        manifest, layout_getter=read, package_version=lambda name: name + "-test"
    )
    assert len(fingerprint) == 64
    assert len(visited) == len(manifest.storages)
    assert all(a is b for a, b in zip(visited, manifest.storages))
    assert len(payload["pools"]) == len(manifest.pools)
    other = measure_transfer_layout(
        manifest, layout_getter=read, package_version=lambda name: name + "-other"
    )[0]
    assert other != fingerprint


@pytest.mark.parametrize(
    "layout",
    [
        None,
        ((0, 4, 3, 2, 1), _LAYOUT[1], 8),
        (_LAYOUT[0], ((8, 128),), 8),
        (_LAYOUT[0], _LAYOUT[1], 16),
    ],
)
def test_refuse_missing_or_different_native_layout(layout):
    config, materialized = live_pool()
    with pytest.raises(ManifestError):
        measure_transfer_layout(
            build_transfer_manifest(config, materialized),
            layout_getter=lambda raw: layout,
        )


def test_refuse_page_extents_that_do_not_partition_storage():
    config, materialized = live_pool()
    manifest = build_transfer_manifest(config, materialized)
    manifest.pools[0] = replace(manifest.pools[0], block_stride_bytes=1)
    with pytest.raises(ManifestError, match="partition"):
        measure_transfer_layout(manifest, layout_getter=lambda raw: _LAYOUT)


@pytest.mark.tpu_test
def test_actual_qwen_carrier_raw_bytes_and_whole_pages_tpu():
    import hashlib
    import json

    import numpy as np
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import sync
    from torch_tpu._internal.batch_transfer import batch_transfer_d2h_sync
    from torch_tpu._internal.compile import tpu_torch_compile

    from tests.distributed.test_raiden_fa_layout_calibration_tpu import (
        _apply_xla_tiled_layout,
        _logical_probe,
    )

    records = []
    for tp in (1, 2, 4, 8):
        config, materialized = live_pool(tp, device="tpu")
        manifest = build_transfer_manifest(config, materialized)
        measurements = []
        for index, raw in enumerate(manifest.storages):
            logical = _logical_probe(tuple(raw.shape))
            raw.copy_(torch.from_numpy(logical).view(torch.float8_e4m3fn).to("tpu"))
            sync.synchronize([raw], wait=True)
            order, tiles, bits = tpu_torch_compile.get_device_layout_if_materialized(
                raw
            )
            host = torch.empty(raw.nbytes, dtype=torch.uint8)
            batch_transfer_d2h_sync([raw], [host])
            expected = _apply_xla_tiled_layout(logical, order, tiles)
            assert expected.nbytes == raw.nbytes
            np.testing.assert_array_equal(host.numpy(), expected)
            stride = next(
                p.block_stride_bytes for p in manifest.pools if p.storage_index == index
            )
            labels = np.repeat(
                np.arange(config.num_blocks, dtype=np.int32), stride
            ).reshape(raw.shape)
            np.testing.assert_array_equal(
                _apply_xla_tiled_layout(labels, order, tiles), labels.reshape(-1)
            )
            measurements.append(
                dict(
                    shape=list(raw.shape),
                    page_bytes=stride,
                    raw_bytes=host.nbytes,
                    raw_sha256=hashlib.sha256(host.numpy().tobytes()).hexdigest(),
                )
            )
        fingerprint, payload = measure_transfer_layout(manifest)
        records.append(
            dict(
                tp_geometry=tp,
                fingerprint=fingerprint,
                payload=payload,
                measurements=measurements,
            )
        )
    print("QWEN_PD_NATIVE_LAYOUT=" + json.dumps(records, sort_keys=True))
