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
"""Host-only unit tests for the block-major layout contract and offload plane.

Validates:
  - Pure-config contract resolution and deterministic layout fingerprinting.
  - Namespace isolation ensuring layer-major and block-major cache keys never collide.
  - Worker-side single-bundle tensor reconstruction and shape validation.
  - Fail-closed validation against non-uniform or unaligned cache geometries.

Executes entirely on host without TPU hardware or active controller connections.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import KVCacheTensor

# Synthetic test geometry: 1 kernel row = (16, 2, 2, 4) bf16 = 512 bytes.
# Fixed to factor == 1 (device_block_size == kernel_block_size) as required by the contract.
_KERNEL_BLOCK_SIZE = 16
_PER_BLOCK_SHAPE = (16, 2, 2, 4)
_ROW_BYTES = 512
_DEVICE_BLOCK_SIZE = 16
_FACTOR = 1

_GEOMETRY = (_KERNEL_BLOCK_SIZE, _PER_BLOCK_SHAPE, torch.bfloat16, _DEVICE_BLOCK_SIZE)


def _kv_cache_config(num_tensors=3, num_blocks=4, page_bytes=None):
    page_bytes = page_bytes or _FACTOR * _ROW_BYTES
    tensors = [
        KVCacheTensor(
            size=num_blocks * page_bytes * num_tensors,
            layers=[f"layers.{i}"],
            layer_stride=page_bytes,
            block_stride=page_bytes * num_tensors,
            offset=i * page_bytes,
        )
        for i in range(num_tensors)
    ]
    return SimpleNamespace(
        kv_cache_tensors=tensors, num_blocks=num_blocks, kv_cache_groups=[MagicMock()]
    )


def _resolve(kv_cache_config, flag=True):
    from vllm_torchtpu.offload import block_major_layout as bml

    with (
        patch.object(bml.tpu_envs, "VLLM_TPU_BLOCK_MAJOR_KV", flag),
        patch(
            "vllm_torchtpu.offload.raiden_store.resolve_kernel_geometry",
            return_value=_GEOMETRY,
        ),
    ):
        return bml.resolve_block_major_contract(MagicMock(), kv_cache_config)


class TestContractResolution(unittest.TestCase):
    def test_flag_off_returns_none(self):
        """Verifies that contract resolution returns None when VLLM_TPU_BLOCK_MAJOR_KV is disabled."""
        self.assertIsNone(_resolve(_kv_cache_config(), flag=False))

    def test_uniform_dense_contract(self):
        """Verifies contract metrics and row sizes for uniform dense model topologies."""
        contract = _resolve(_kv_cache_config(num_tensors=3))
        self.assertEqual(contract.fragment_count, 3)
        self.assertEqual(contract.fragment_row_bytes, _ROW_BYTES)
        self.assertEqual(contract.bundle_row_bytes, 3 * _ROW_BYTES)

    def test_contract_is_deterministic(self):
        """Verifies that identical configurations yield identical layout fingerprints."""
        a = _resolve(_kv_cache_config())
        b = _resolve(_kv_cache_config())
        self.assertEqual(a, b)
        self.assertEqual(a.logical_fingerprint, b.logical_fingerprint)

    def test_fingerprint_tracks_fragment_count(self):
        """Verifies that altering fragment counts changes the layout fingerprint."""
        a = _resolve(_kv_cache_config(num_tensors=3))
        b = _resolve(_kv_cache_config(num_tensors=4))
        self.assertNotEqual(a.logical_fingerprint, b.logical_fingerprint)

    def test_non_uniform_sizes_fail_closed(self):
        """Verifies that non-uniform fragment tensor sizes fail closed with ValueError."""
        config = _kv_cache_config(num_tensors=3)
        config.kv_cache_tensors[1].size *= 2
        with self.assertRaisesRegex(ValueError, "not uniform"):
            _resolve(config)

    def test_untiled_geometry_fails_closed(self):
        """Verifies that page sizes not evenly tiling kernel rows fail closed."""
        config = _kv_cache_config(page_bytes=_FACTOR * _ROW_BYTES + 512)
        with self.assertRaisesRegex(ValueError, "does not tile"):
            _resolve(config)

    def test_factor_gt_one_fails_closed(self):
        """Verifies that device_block_size != kernel_block_size (factor > 1) fails closed."""
        from vllm_torchtpu.offload import block_major_layout as bml

        geometry = (
            _KERNEL_BLOCK_SIZE,
            _PER_BLOCK_SHAPE,
            torch.bfloat16,
            2 * _KERNEL_BLOCK_SIZE,
        )
        config = _kv_cache_config(page_bytes=2 * _ROW_BYTES)
        with (
            patch.object(bml.tpu_envs, "VLLM_TPU_BLOCK_MAJOR_KV", True),
            patch(
                "vllm_torchtpu.offload.raiden_store.resolve_kernel_geometry",
                return_value=geometry,
            ),
            self.assertRaisesRegex(ValueError, "factor"),
        ):
            bml.resolve_block_major_contract(MagicMock(), config)


class TestNamespaceIsolation(unittest.TestCase):
    def _namespace(self, contract):
        from vllm_torchtpu.offload.raiden_store import derive_offload_namespace

        vllm_config = MagicMock()
        vllm_config.model_config.model = "m"
        vllm_config.model_config.revision = None
        vllm_config.model_config.quantization = None
        vllm_config.cache_config.prefix_caching_hash_algo = "sha256"
        return derive_offload_namespace(
            vllm_config,
            kernel_block_size=_KERNEL_BLOCK_SIZE,
            per_block_shape=_PER_BLOCK_SHAPE,
            kv_dtype=torch.bfloat16,
            device_block_size=_DEVICE_BLOCK_SIZE,
            world_size=4,
            num_kv_cache_groups=1,
            num_kv_cache_tensors=3,
            block_major_contract=contract,
        )

    def test_block_major_never_shares_layer_major_namespace(self):
        """Verifies namespace isolation between block-major and layer-major deployments."""
        contract = _resolve(_kv_cache_config())
        self.assertNotEqual(self._namespace(None), self._namespace(contract))

    def test_namespace_tracks_layout_fingerprint(self):
        """Verifies that differing layout fingerprints produce distinct offload namespaces."""
        a = _resolve(_kv_cache_config(num_tensors=3))
        b = _resolve(_kv_cache_config(num_tensors=4))
        self.assertNotEqual(self._namespace(a), self._namespace(b))


class TestWorkerBundledView(unittest.TestCase):
    def _make_worker(self, kv_caches, contract):
        from vllm_torchtpu.offload.raiden_store import RaidenStoreOffloadingWorker

        with patch(
            "vllm_torchtpu.offload.raiden_store.resolve_kernel_geometry",
            return_value=_GEOMETRY,
        ):
            return RaidenStoreOffloadingWorker(
                kv_caches,
                vllm_config=MagicMock(),
                kv_cache_config=MagicMock(),
                host_blocks_to_allocate=8,
                # Intentionally unreachable loopback port to avoid dialing external controller endpoints during unit tests.
                controller_address="127.0.0.1:1",
                rank=0,
                block_major_contract=contract,
            )

    def _canonical(self, num_tensors, num_blocks, page_bytes):
        from vllm.v1.kv_offload.base import (
            CanonicalKVCacheRef,
            CanonicalKVCaches,
            CanonicalKVCacheTensor,
        )

        tensors = [
            CanonicalKVCacheTensor(
                tensor=torch.zeros(num_blocks, page_bytes, dtype=torch.int8),
                page_size_bytes=page_bytes,
            )
            for _ in range(num_tensors)
        ]
        refs = [
            [
                CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=page_bytes)
                for i in range(num_tensors)
            ]
        ]
        return CanonicalKVCaches(tensors=tensors, group_data_refs=refs)

    def test_bundled_view_shape(self):
        """Verifies that worker device tensors are correctly reshaped to [blocks, F, *R]."""
        contract = _resolve(_kv_cache_config(num_tensors=3))
        num_blocks = 4
        bundle_page = _FACTOR * contract.bundle_row_bytes
        kv_caches = self._canonical(1, num_blocks, bundle_page)
        worker = self._make_worker(kv_caches, contract)
        try:
            (view,) = worker._device_tensors[0]
            self.assertEqual(
                tuple(view.shape),
                (num_blocks * _FACTOR, contract.fragment_count) + _PER_BLOCK_SHAPE,
            )
            self.assertEqual(view.dtype, torch.bfloat16)
        finally:
            worker.shutdown()

    def test_bundled_requires_single_storage(self):
        """Verifies that worker registration asserts single canonical storage under block-major."""
        contract = _resolve(_kv_cache_config(num_tensors=3))
        kv_caches = self._canonical(3, 4, _FACTOR * _ROW_BYTES)
        with self.assertRaisesRegex(AssertionError, "one canonical storage"):
            self._make_worker(kv_caches, contract)

    def test_layer_major_views_unchanged(self):
        """Verifies that layer-major registration leaves device tensor shapes unchanged."""
        kv_caches = self._canonical(3, 4, _FACTOR * _ROW_BYTES)
        worker = self._make_worker(kv_caches, None)
        try:
            self.assertEqual(len(worker._device_tensors), 3)
            (view,) = worker._device_tensors[0]
            self.assertEqual(tuple(view.shape), (4 * _FACTOR,) + _PER_BLOCK_SHAPE)
        finally:
            worker.shutdown()


if __name__ == "__main__":
    unittest.main()
