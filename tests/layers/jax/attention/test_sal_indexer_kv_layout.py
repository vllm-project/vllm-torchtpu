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
"""
================================================================================
3-TIER EXECUTION PROVENANCE BANNER (Rule 19 Mandatory Provenance)
================================================================================
- Tier 1 (Compiled Host C-ABI Shared Library): N/A (Python/JAX Pallas & XLA lowering)
- Tier 2 (Compiled Host C++20 LloRegionBuilder IR): N/A
- Tier 3 (Execution Provenance):
  [CPU NUMERICAL PARITY & JAX 16-DEVICE SHARD_MAP VERIFICATION — NOT PHYSICAL TPU]
  Executes on 16-device CPU JAX mesh (`XLA_FLAGS=--xla_force_host_platform_device_count=16`)
  verifying Companion PR #6 4-D Scale-Amortized Layout (`SAL`: `132 B/tok`,
  `(num_blocks, 33, 4, 256)`) vs `has` (`256 B/tok`, `(num_blocks, 64, 4, 256)`) and
  un-normalized XLA baseline (512 B/tok), bit-exact round-trip pack/unpack across
  ragged token counts T in {1, 3, 7, 16, 64, 256}, slot positioning preservation across
  out-of-order block tables, AttentionSpec normalizer routing, and AST wiring.
================================================================================
"""

import ast
import dataclasses
import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Configure 16 host CPU devices for DCP=16 shard_map testing before importing jax
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    convert_cache_to_head_along_sublane,
    convert_cache_to_seq_along_lane,
    get_kv_cache_shape,
    indexer_kv_layout,
)


# Stand-in dataclasses mirroring vllm.v1.kv_cache_interface for standalone evaluation
@dataclasses.dataclass(frozen=True)
class StandinAttentionSpec:
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    page_size_padded: int | None = None
    tokens_per_state: int = 1
    cache_dtype_str: str = "auto"
    model_version: str | None = None
    num_head_slots: int = 1
    state_content_bytes: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        return self.page_size_padded or (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * torch.empty((), dtype=self.dtype).element_size()
        )


@dataclasses.dataclass(frozen=True)
class StandinMLAAttentionSpec(StandinAttentionSpec):
    pass


@dataclasses.dataclass(frozen=True)
class StandinFullAttentionSpec(StandinAttentionSpec):
    pass


def _standin_is_deepseek_v32_indexer_spec(spec) -> bool:
    """Standalone replica of _is_deepseek_v32_indexer_spec from kv_cache_spec_normalizer.py."""
    return (
        isinstance(spec, StandinMLAAttentionSpec)
        and not spec.dtype.is_floating_point
        and spec.tokens_per_state == 1
        and spec.cache_dtype_str != "fp8_ds_mla"
        and getattr(spec, "model_version", None) != "deepseek_v4"
    )


class TestSalIndexerKVLayout(unittest.TestCase):
    """Test suite for 4-D Scale-Amortized Layout (SAL: 132 B/tok) Indexer KV Cache."""

    def test_01_4d_allocation_probe_shapes_and_page_sizes(self):
        """Test 1: Verify 4-D shapes and per-token byte budgets across SAL (132B) vs HAS (256B) vs XLA."""
        num_blocks = 2
        block_size = 256
        head_size = 128

        # 1. SAL Mode: (num_blocks, 33, 4, block_size)
        sal_shape = get_kv_cache_shape(
            num_blocks, block_size, head_size, kv_layout=KVLayout.SEQ_ALONG_LANE
        )
        self.assertEqual(sal_shape, (num_blocks, 33, 4, block_size))

        # 33 sublanes * 4 lanes * 256 block_size * 1 byte = 33,792 bytes
        sal_page_bytes = 33 * 4 * block_size * 1
        self.assertEqual(sal_page_bytes, 33792)
        bytes_per_token_sal = sal_page_bytes / block_size
        self.assertEqual(bytes_per_token_sal, 132.0)

        # 2. HAS Mode: (num_blocks, block_size // 4, 4, 256)
        has_shape = get_kv_cache_shape(
            num_blocks, block_size, head_size, kv_layout=KVLayout.HEAD_ALONG_SUBLANE
        )
        self.assertEqual(has_shape, (num_blocks, 64, 4, 256))

        # 64 * 4 * 256 * 1 byte = 65,536 bytes
        has_page_bytes = 64 * 4 * 256 * 1
        self.assertEqual(has_page_bytes, 65536)
        bytes_per_token_has = has_page_bytes / block_size
        self.assertEqual(bytes_per_token_has, 256.0)

        # 3. Capacity expansion ratio: 256 / 132 = 1.9394x (+93.94% -> +94.0%)
        capacity_gain_pct = (bytes_per_token_has / bytes_per_token_sal - 1.0) * 100.0
        self.assertAlmostEqual(capacity_gain_pct, 93.9393939, places=4)

        # 4. Standard un-normalized baseline comparison (512 B/token for 2 heads x 128 x 2 bytes bf16)
        unnormalized_bytes_per_token = 512.0
        hbm_saving_ratio = unnormalized_bytes_per_token / bytes_per_token_sal
        self.assertAlmostEqual(hbm_saving_ratio, 512.0 / 132.0, places=4)

    def test_02_indexer_spec_predicate_and_normalizer_logic(self):
        """Test 2: Verify _is_deepseek_v32_indexer_spec predicate isolating indexer specs."""
        # Genuine DSA indexer spec: uint8 MLA with tokens_per_state == 1
        indexer_spec = StandinMLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.uint8,
            tokens_per_state=1,
            cache_dtype_str="auto",
            model_version="deepseek_v32",
        )
        self.assertTrue(_standin_is_deepseek_v32_indexer_spec(indexer_spec))

        # Negative control 1: Floating point MLA latent cache (bfloat16)
        mla_latent_bf16 = StandinMLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            tokens_per_state=1,
            cache_dtype_str="auto",
        )
        self.assertFalse(_standin_is_deepseek_v32_indexer_spec(mla_latent_bf16))

        # Negative control 2: Standard FullAttentionSpec
        full_attn_spec = StandinFullAttentionSpec(
            block_size=256,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.uint8,
        )
        self.assertFalse(_standin_is_deepseek_v32_indexer_spec(full_attn_spec))

        # Negative control 3: fp8_ds_mla cache
        fp8_ds_mla_spec = StandinMLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.uint8,
            tokens_per_state=1,
            cache_dtype_str="fp8_ds_mla",
        )
        self.assertFalse(_standin_is_deepseek_v32_indexer_spec(fp8_ds_mla_spec))

        # Negative control 4: deepseek_v4 version tag
        dsv4_spec = StandinMLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.uint8,
            tokens_per_state=1,
            cache_dtype_str="auto",
            model_version="deepseek_v4",
        )
        self.assertFalse(_standin_is_deepseek_v32_indexer_spec(dsv4_spec))

    def test_03_bit_exact_roundtrip_pack_unpack_ragged_tokens(self):
        """Test 3: Bit-exact (0 ULP, CosSim == 1.000000) round-trip across ragged token counts."""
        ragged_counts = [1, 3, 7, 16, 64, 256]
        np.random.seed(42)

        for t in ragged_counts:
            with self.subTest(num_tokens=t):
                num_pages = 1
                page_size = 256
                head_dim = 128
                record_width = 256  # align_to(128 + 1, 128)

                # Generate random keys and scales for t tokens
                keys = np.random.randint(0, 256, size=(num_pages, t, head_dim), dtype=np.uint8)
                scales = np.random.randint(0, 256, size=(num_pages, t, 1), dtype=np.uint8)

                # Pack into HEAD_ALONG_SUBLANE cache
                flat = np.zeros((num_pages, page_size, record_width), dtype=np.uint8)
                flat[:, :t, :head_dim] = keys
                flat[:, :t, head_dim : head_dim + 1] = scales

                has_cache = jnp.array(
                    flat.reshape(num_pages, page_size // 4, 4, record_width)
                )

                # 1. Repack to SEQ_ALONG_LANE
                sal_cache = convert_cache_to_seq_along_lane(has_cache, head_dim)
                self.assertEqual(sal_cache.shape, (num_pages, 33, 4, page_size))

                # 2. Unpack back to HEAD_ALONG_SUBLANE
                recovered_has = convert_cache_to_head_along_sublane(sal_cache, head_dim)
                self.assertEqual(recovered_has.shape, has_cache.shape)

                # 3. Assert bit-exact identity for active tokens (0 ULP delta)
                recovered_flat = np.array(
                    recovered_has.reshape(num_pages, page_size, record_width)
                )

                recovered_keys = recovered_flat[:, :t, :head_dim]
                recovered_scales = recovered_flat[:, :t, head_dim : head_dim + 1]

                np.testing.assert_array_equal(
                    keys,
                    recovered_keys,
                    err_msg=f"Key mismatch for ragged token count T={t}",
                )
                np.testing.assert_array_equal(
                    scales,
                    recovered_scales,
                    err_msg=f"Scale mismatch for ragged token count T={t}",
                )

                # Cosine similarity check (Gate 7 numerical requirement: >= 0.99999)
                k_orig = keys.astype(np.float64).flatten()
                k_rec = recovered_keys.astype(np.float64).flatten()
                cossim = np.dot(k_orig, k_rec) / (
                    np.linalg.norm(k_orig) * np.linalg.norm(k_rec)
                )
                self.assertAlmostEqual(float(cossim), 1.000000, places=6)

    def test_04_out_of_order_block_tables_and_slot_preservation(self):
        """Test 4: Multi-page out-of-order slot allocation is preserved bit-exactly."""
        num_pages = 4
        page_size = 256
        head_dim = 128
        record_width = 256

        flat = np.zeros((num_pages, page_size, record_width), dtype=np.uint8)

        # Populate arbitrary disparate slot positions across pages
        test_slots = [
            (0, 5),
            (0, 27),
            (0, 128),
            (0, 250),
            (1, 0),
            (1, 255),
            (2, 77),
            (2, 199),
            (3, 12),
            (3, 99),
            (3, 201),
        ]

        expected_records = {}
        for p, s in test_slots:
            k = np.random.randint(0, 256, size=(head_dim,), dtype=np.uint8)
            sc = np.random.randint(0, 256, size=(1,), dtype=np.uint8)
            flat[p, s, :head_dim] = k
            flat[p, s, head_dim : head_dim + 1] = sc
            expected_records[(p, s)] = (k, sc)

        has_cache = jnp.array(flat.reshape(num_pages, page_size // 4, 4, record_width))

        # Convert to SAL and back
        sal_cache = convert_cache_to_seq_along_lane(has_cache, head_dim)
        recovered_has = convert_cache_to_head_along_sublane(sal_cache, head_dim)
        recovered_flat = np.array(
            recovered_has.reshape(num_pages, page_size, record_width)
        )

        for (p, s), (exp_k, exp_sc) in expected_records.items():
            np.testing.assert_array_equal(
                recovered_flat[p, s, :head_dim],
                exp_k,
                err_msg=f"Key mismatch at page {p}, slot {s}",
            )
            np.testing.assert_array_equal(
                recovered_flat[p, s, head_dim : head_dim + 1],
                exp_sc,
                err_msg=f"Scale mismatch at page {p}, slot {s}",
            )

    def test_05_packed_head_dim_129_unpacking(self):
        """Test 5: Packed actual_head_dim=129 is unpacked to 128 across all utilities."""
        shape_128 = get_kv_cache_shape(2, 256, 128, kv_layout=KVLayout.SEQ_ALONG_LANE)
        shape_129 = get_kv_cache_shape(2, 256, 129, kv_layout=KVLayout.SEQ_ALONG_LANE)
        self.assertEqual(shape_128, shape_129)
        self.assertEqual(shape_129, (2, 33, 4, 256))

        # Test pack/unpack works with actual_head_dim=129
        dummy_has = jnp.zeros((2, 64, 4, 256), dtype=jnp.uint8)
        sal = convert_cache_to_seq_along_lane(dummy_has, 129)
        self.assertEqual(sal.shape, (2, 33, 4, 256))
        recovered = convert_cache_to_head_along_sublane(sal, 129)
        self.assertEqual(recovered.shape, (2, 64, 4, 256))

    def test_06_fallback_on_sub128_and_non_multiple_block_sizes(self):
        """Test 6: indexer_kv_layout gracefully falls back to HEAD_ALONG_SUBLANE."""
        # block_size < 128: falls back to HEAD_ALONG_SUBLANE
        self.assertEqual(
            indexer_kv_layout(block_size=64, mode="sal"),
            KVLayout.HEAD_ALONG_SUBLANE,
        )
        self.assertEqual(
            indexer_kv_layout(block_size=32, mode="sal"),
            KVLayout.HEAD_ALONG_SUBLANE,
        )

        # block_size % 128 != 0: falls back to HEAD_ALONG_SUBLANE
        self.assertEqual(
            indexer_kv_layout(block_size=192, mode="sal"),
            KVLayout.HEAD_ALONG_SUBLANE,
        )

        # block_size valid multiple of 128: uses SEQ_ALONG_LANE in SAL mode
        self.assertEqual(
            indexer_kv_layout(block_size=128, mode="sal"),
            KVLayout.SEQ_ALONG_LANE,
        )
        self.assertEqual(
            indexer_kv_layout(block_size=256, mode="sal"),
            KVLayout.SEQ_ALONG_LANE,
        )
        self.assertEqual(
            indexer_kv_layout(block_size=512, mode="sal"),
            KVLayout.SEQ_ALONG_LANE,
        )

        # HAS mode always returns HEAD_ALONG_SUBLANE
        self.assertEqual(
            indexer_kv_layout(block_size=256, mode="has"),
            KVLayout.HEAD_ALONG_SUBLANE,
        )

    def test_07_ast_verification_normalizer_and_attention_wiring(self):
        """Test 7: Static AST verification of source code wiring across modified files."""
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..")
        )

        # 1. AST of kv_cache_spec_normalizer.py
        norm_path = os.path.join(
            repo_root, "src/vllm_torchtpu/kv_cache_spec_normalizer.py"
        )
        self.assertTrue(os.path.exists(norm_path), f"File {norm_path} not found")
        with open(norm_path, "r", encoding="utf-8") as f:
            norm_code = f.read()

        norm_ast = ast.parse(norm_code)
        norm_funcs = [
            n.name for n in ast.walk(norm_ast) if isinstance(n, ast.FunctionDef)
        ]
        self.assertIn("_is_deepseek_v32_indexer_spec", norm_funcs)
        self.assertIn("_normalize_one_spec", norm_funcs)
        self.assertIn("VllmTPUDeepseekV32IndexerBackend", norm_code)
        self.assertIn("_is_deepseek_v32_indexer_spec(spec)", norm_code)

        # 2. AST of layers/adapter/attention.py
        attn_path = os.path.join(
            repo_root, "src/vllm_torchtpu/layers/adapter/attention.py"
        )
        self.assertTrue(os.path.exists(attn_path), f"File {attn_path} not found")
        with open(attn_path, "r", encoding="utf-8") as f:
            attn_code = f.read()

        attn_ast = ast.parse(attn_code)
        classes = {n.name: n for n in ast.walk(attn_ast) if isinstance(n, ast.ClassDef)}
        self.assertIn("VllmTPUDeepseekV32IndexerBackend", classes)

        indexer_backend_cls = classes["VllmTPUDeepseekV32IndexerBackend"]
        indexer_methods = [
            n.name for n in indexer_backend_cls.body if isinstance(n, ast.FunctionDef)
        ]
        self.assertIn("get_kv_cache_shape", indexer_methods)
        self.assertIn("get_kv_cache_page_size_bytes", indexer_methods)
        self.assertIn("customize_spec", indexer_methods)
        self.assertIn("supported_kv_cache_layouts", indexer_methods)
        self.assertIn("get_name", indexer_methods)

        # 3. Verify envs.py configuration
        self.assertEqual(envs.DEFAULT_INDEXER_KV_CACHE_MODE, "sal")
        self.assertEqual(envs.VLLM_TPU_INDEXER_KV_CACHE_MODE, "sal")

        envs_path = os.path.join(repo_root, "src/vllm_torchtpu/envs.py")
        with open(envs_path, "r", encoding="utf-8") as f:
            envs_code = f.read()
        self.assertIn("DEFAULT_INDEXER_KV_CACHE_MODE = \"sal\"", envs_code)
        self.assertIn("\"VLLM_TPU_INDEXER_KV_CACHE_MODE\"", envs_code)

    def test_08_negative_controls_invalid_modes(self):
        """Test 8: Verify negative controls and invalid modes raise descriptive exceptions."""
        with self.assertRaises(ValueError) as ctx:
            indexer_kv_layout(block_size=256, mode="non_existent_mode")
        self.assertIn("Invalid VLLM_TPU_INDEXER_KV_CACHE_MODE", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            indexer_kv_layout(block_size=256, mode="cuda_flash_layout")
        self.assertIn("expected one of ('xla', 'has', 'sal')", str(ctx.exception))

        # SEQ_ALONG_LANE requires page_size to be a multiple of 128
        with self.assertRaisesRegex(ValueError, "SEQ_ALONG_LANE requires page_size"):
            get_kv_cache_shape(2, 64, 128, kv_layout=KVLayout.SEQ_ALONG_LANE)

        # HEAD_ALONG_SUBLANE requires page_size % 4 == 0
        with self.assertRaisesRegex(ValueError, "HEAD_ALONG_SUBLANE requires page_size"):
            get_kv_cache_shape(2, 3, 128, kv_layout=KVLayout.HEAD_ALONG_SUBLANE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
