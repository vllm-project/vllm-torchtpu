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
"""Measured page-layout admission and an isolated raw-byte TPU calibration."""

from dataclasses import replace

import pytest
import torch

from tests.runner.test_qwen4_exp_pd_manifest import live_pool
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import ManifestError
from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import (
    build_transfer_manifest,
    measure_transfer_layout,
)

_LAYOUT = ((4, 3, 2, 1, 0), ((4, 128), (4, 1)), 8)


def native_layout(raw):
    packing = 4 // raw.element_size()
    return ((4, 3, 2, 1, 0), ((packing, 128), (packing, 1)), raw.element_size() * 8)


def test_fingerprint_reads_each_carrier_and_includes_page_geometry():
    config, materialized = live_pool()
    manifest = build_transfer_manifest(config, materialized)
    visited = []

    def read(raw):
        visited.append(raw)
        return native_layout(raw)

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
        measure_transfer_layout(manifest, layout_getter=native_layout)


@pytest.mark.tpu_test
def test_actual_qwen_carrier_raw_bytes_and_whole_pages_tpu():
    import hashlib
    import json

    import numpy as np
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
            logical = (
                _logical_probe((raw.nbytes,))
                .view(np.uint8 if raw.element_size() == 1 else np.uint16)
                .reshape(raw.shape)
            )
            raw.copy_(
                torch.from_numpy(logical.view(np.uint8).reshape(-1))
                .view(raw.dtype)
                .reshape(raw.shape)
                .to("tpu")
            )
            sync.synchronize([raw], wait=True)
            order, tiles, bits = tpu_torch_compile.get_device_layout_if_materialized(
                raw
            )
            print(
                "QWEN_PD_CARRIER_LAYOUT="
                + json.dumps(
                    dict(
                        tp=tp,
                        index=index,
                        shape=list(raw.shape),
                        dtype=str(raw.dtype),
                        order=list(order),
                        tiles=[list(t) for t in tiles],
                        bits=bits,
                    )
                ),
                flush=True,
            )
            host = torch.empty(raw.nbytes, dtype=torch.uint8)
            batch_transfer_d2h_sync([raw], [host])
            expected = _apply_xla_tiled_layout(logical, order, tiles)
            assert expected.nbytes == raw.nbytes
            np.testing.assert_array_equal(host.numpy(), expected.view(np.uint8))
            stride = next(
                p.block_stride_bytes for p in manifest.pools if p.storage_index == index
            )
            labels = np.repeat(
                np.arange(config.num_blocks, dtype=np.int32),
                stride // raw.element_size(),
            ).reshape(raw.shape)
            np.testing.assert_array_equal(
                _apply_xla_tiled_layout(labels, order, tiles), labels.reshape(-1)
            )
            measurements.append(
                dict(
                    shape=list(raw.shape),
                    layout=[list(order), [list(t) for t in tiles], bits],
                    dtype=str(raw.dtype),
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
