# SPDX-License-Identifier: Apache-2.0
"""100% Coverage Suite for DeepSeek-V4.1 v2 Async Fence, Ops, Cache.

Covers 100% of lines and branches in:
- vllm_torchtpu.models.vllm.deepseek_v41.v2.async_fence (lines 1-37)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.ops (lines 1-133)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.cache (lines 1-856)
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

# Ensure CPU backend and skip MDS query before importing torch/jax
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import ROW_WORDS
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import async_fence as v2_fence
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as v2_cache
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ops as v2_ops

# ==============================================================================
# 1. Tests for v2/async_fence.py
# ==============================================================================


class TestAsyncFence:
    def test_install_lookback_fence_and_execution(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from vllm_torchtpu.runner import tpu_runner

        # Create dummy original method
        called_original = []

        def dummy_original(runner_self, num_reqs: int):
            called_original.append(num_reqs)
            return torch.tensor([1, 2, 3])

        # Patch TPUModelRunner._prepare_lookback_token_ids
        monkeypatch.setattr(
            tpu_runner.TPUModelRunner,
            "_prepare_lookback_token_ids",
            dummy_original,
            raising=False,
        )

        # 1. First installation
        v2_fence.install_lookback_fence()
        patched_fn = tpu_runner.TPUModelRunner._prepare_lookback_token_ids
        assert getattr(patched_fn, "_deepseek_v41_fenced", False) is True

        # 2. Second installation (idempotency branch)
        v2_fence.install_lookback_fence()
        assert tpu_runner.TPUModelRunner._prepare_lookback_token_ids is patched_fn

        # 3. Execution when fence is None
        runner = SimpleNamespace()
        mock_event = MagicMock()
        monkeypatch.setattr(
            torch, "tpu", SimpleNamespace(Event=lambda: mock_event), raising=False
        )

        res1 = patched_fn(runner, 4)
        assert torch.equal(res1, torch.tensor([1, 2, 3]))
        assert called_original == [4]
        mock_event.record.assert_called_once()
        assert getattr(runner, "_deepseek_v41_lookback_fence", None) is mock_event

        # 4. Execution when fence is present (synchronize branch)
        mock_event.reset_mock()
        mock_event2 = MagicMock()
        monkeypatch.setattr(
            torch, "tpu", SimpleNamespace(Event=lambda: mock_event2), raising=False
        )

        res2 = patched_fn(runner, 2)
        assert torch.equal(res2, torch.tensor([1, 2, 3]))
        assert called_original == [4, 2]
        mock_event.synchronize.assert_called_once()
        mock_event2.record.assert_called_once()
        assert runner._deepseek_v41_lookback_fence is mock_event2


# ==============================================================================
# 2. Tests for v2/ops.py
# ==============================================================================


class TestOps:
    def test_device_hooks(self):
        mod = SimpleNamespace(alpha=10, beta="hello")

        # Normal execution
        with v2_ops.device_hooks(mod, alpha=99, beta="world"):
            assert mod.alpha == 99
            assert mod.beta == "world"

        assert mod.alpha == 10
        assert mod.beta == "hello"

        # Exception safety
        with (
            pytest.raises(RuntimeError, match="test_err"),
            v2_ops.device_hooks(mod, alpha=100),
        ):
            assert mod.alpha == 100
            raise RuntimeError("test_err")

        assert mod.alpha == 10

    def test_bind_device_ops(self):
        def sample_fn(x, factor=2):
            """Docstring."""
            return custom_add(x * factor, offset)  # noqa: F821

        bound = v2_ops.bind_device_ops(
            sample_fn, custom_add=lambda a, b: a + b, offset=5
        )

        assert bound(10) == 25  # 10 * 2 + 5
        assert bound(10, factor=3) == 35  # 10 * 3 + 5
        assert bound.__name__ == sample_fn.__name__
        assert bound.__qualname__ == sample_fn.__qualname__

    def test_compact_rope_caches(self):
        # max_model_len <= 0 error
        with pytest.raises(ValueError, match="max_model_len must be positive"):
            v2_ops.compact_rope_caches([], 0)

        # Cache sharing and slicing
        shared_cache = torch.randn(100, 16)
        att1 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=shared_cache))
        att2 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=shared_cache))
        small_cache = torch.randn(30, 16)
        att3 = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=small_cache))

        v2_ops.compact_rope_caches([att1, att2, att3], max_model_len=50)

        # att1 and att2 share the sliced copy
        assert att1._tpu_rope_cache.shape == (50, 16)
        assert att1._tpu_rope_cache is att2._tpu_rope_cache
        assert torch.equal(att1._tpu_rope_cache, shared_cache[:50])

        # att3 is <= max_model_len, kept identical
        assert att3._tpu_rope_cache is small_cache

    def test_rotary_dimension_validation(self):
        x = torch.randn(1, 2, 32)
        pos = torch.tensor([0, 1])

        # dim < 2
        with pytest.raises(ValueError, match="invalid rotary dimension"):
            v2_ops.rotary(
                x,
                pos,
                {"qk_rope_head_dim": 1, "rope_theta": 10000.0},
                compressed=False,
            )

        # dim % 2 != 0
        with pytest.raises(ValueError, match="invalid rotary dimension"):
            v2_ops.rotary(
                x,
                pos,
                {"qk_rope_head_dim": 7, "rope_theta": 10000.0},
                compressed=False,
            )

        # dim > x.shape[-1]
        with pytest.raises(ValueError, match="invalid rotary dimension"):
            v2_ops.rotary(
                x,
                pos,
                {"qk_rope_head_dim": 64, "rope_theta": 10000.0},
                compressed=False,
            )

    def test_rotary_uncompressed_and_inverse(self):
        # 3D tensor: shape (batch=1, tokens=2, dim=16)
        x_3d = torch.randn(1, 2, 16)
        pos = torch.tensor([0, 1])
        cfg = {"qk_rope_head_dim": 8, "rope_theta": 10000.0}

        out = v2_ops.rotary(x_3d, pos, cfg, compressed=False, inverse=False)
        assert out.shape == x_3d.shape
        # Untouched prefix
        assert torch.equal(out[..., :-8], x_3d[..., :-8])

        # Inverse rotation should invert the transform
        inv = v2_ops.rotary(out, pos, cfg, compressed=False, inverse=True)
        assert torch.allclose(inv, x_3d, atol=1e-5)

        # 4D tensor: shape (batch=1, tokens=2, heads=4, dim=16)
        x_4d = torch.randn(1, 2, 4, 16)
        out_4d = v2_ops.rotary(x_4d, pos, cfg, compressed=False, inverse=False)
        assert out_4d.shape == x_4d.shape

    def test_rotary_compressed_with_yarn(self):
        x = torch.randn(1, 2, 16)
        pos = torch.tensor([10, 20])

        # 1. YaRN with original_max_position_embeddings > 0
        cfg_yarn = {
            "qk_rope_head_dim": 8,
            "compress_rope_theta": 50000.0,
            "rope_scaling": {
                "original_max_position_embeddings": 4096,
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 2.0,
            },
        }
        out_yarn = v2_ops.rotary(x, pos, cfg_yarn, compressed=True)
        assert out_yarn.shape == x.shape

        # 2. YaRN with rope_parameters fallback
        cfg_yarn2 = {
            "qk_rope_head_dim": 8,
            "compress_rope_theta": 50000.0,
            "rope_parameters": {
                "original_max_position_embeddings": 4096,
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 2.0,
            },
        }
        out_yarn2 = v2_ops.rotary(x, pos, cfg_yarn2, compressed=True)
        assert torch.allclose(out_yarn, out_yarn2)

        # 3. YaRN with original_max_position_embeddings <= 0 (skipped correction)
        cfg_no_correction = {
            "qk_rope_head_dim": 8,
            "compress_rope_theta": 50000.0,
            "rope_scaling": {
                "original_max_position_embeddings": 0,
            },
        }
        out_no_corr = v2_ops.rotary(x, pos, cfg_no_correction, compressed=True)
        assert out_no_corr.shape == x.shape

        # 4. Compressed without any YaRN dict
        cfg_plain_comp = {
            "qk_rope_head_dim": 8,
            "compress_rope_theta": 50000.0,
        }
        out_plain = v2_ops.rotary(x, pos, cfg_plain_comp, compressed=True)
        assert out_plain.shape == x.shape

    def test_round_fp8_activation(self):
        # Invalid block_size
        x = torch.randn(2, 40)
        with pytest.raises(
            ValueError, match="FP8 block size must divide contraction dimension"
        ):
            v2_ops.round_fp8_activation(x, block_size=32)

        # Valid block_size
        x_valid = torch.randn(2, 64, dtype=torch.bfloat16)
        out = v2_ops.round_fp8_activation(x_valid, block_size=32)
        assert out.shape == x_valid.shape
        assert out.dtype == torch.bfloat16

    def test_rms_norm(self):
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        w = torch.ones(16, dtype=torch.bfloat16)
        eps = 1e-6
        out = v2_ops.rms_norm(x, w, eps)
        assert out.shape == x.shape
        assert out.dtype == torch.bfloat16
        # Check variance is ~ 1.0
        var = out.float().square().mean(-1)
        assert torch.allclose(var, torch.tensor(1.0), atol=1e-2)


# ==============================================================================
# 3. Tests for v2/cache.py: Sizing, Materialization, and Views
# ==============================================================================


class TestCacheSizingAndConfig:
    def test_uses_fp8_rope_companion(self):
        # True case
        vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
            ),
            cache_config=SimpleNamespace(cache_dtype="fp8"),
        )
        assert v2_cache.uses_fp8_rope_companion(vllm_cfg) is True

        # False cases: architecture mismatch
        vllm_cfg.model_config.hf_config.architectures = ["LlamaForCausalLM"]
        assert v2_cache.uses_fp8_rope_companion(vllm_cfg) is False

        # False cases: architectures is None
        vllm_cfg.model_config.hf_config.architectures = None
        assert v2_cache.uses_fp8_rope_companion(vllm_cfg) is False

        # False cases: cache_dtype is bfloat16
        vllm_cfg.model_config.hf_config.architectures = ["DeepseekV41ForCausalLM"]
        vllm_cfg.cache_config.cache_dtype = "bfloat16"
        assert v2_cache.uses_fp8_rope_companion(vllm_cfg) is False

    def test_reserve_rope_companion(self):
        # Non-FP8 case
        vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
            ),
            cache_config=SimpleNamespace(cache_dtype="bfloat16"),
        )
        assert v2_cache.reserve_rope_companion(vllm_cfg, 1000) == 1000

        # FP8 case below cap
        vllm_cfg.cache_config.cache_dtype = "fp8_e4m3"
        assert v2_cache.reserve_rope_companion(vllm_cfg, 1000) == 800

        # FP8 case above cap
        huge_bytes = 100 << 30
        assert (
            v2_cache.reserve_rope_companion(vllm_cfg, huge_bytes)
            == v2_cache.SPARSECORE_MAX_POOL_BYTES
        )

    def test_native_cache_view(self):
        raw = torch.arange(1024, dtype=torch.int8)
        shape = (2, 4, 8)  # 64 elements, float32 -> 256 bytes per page
        dtype = torch.float32
        block_stride_bytes = 512
        offset_bytes = 64

        view = v2_cache.native_cache_view(
            raw, shape, dtype, block_stride_bytes, offset_bytes
        )
        assert view.shape == shape
        assert view.dtype == dtype

    def test_materialize_upstream_kv_cache(self, monkeypatch: pytest.MonkeyPatch):
        # Case 1: empty views
        monkeypatch.setattr(v2_cache, "allocate_upstream_kv_cache", lambda *a: {})
        res_empty = v2_cache.materialize_upstream_kv_cache(
            None, torch.device("cpu"), None
        )
        assert res_empty.kv_caches == {}
        assert res_empty.raw_tensors == []

        # Case 2: non-empty views, layer_compact = False, word_rows = False
        dummy_view = torch.zeros(16, 1, 4, 8, dtype=torch.bfloat16)
        dummy_views = {"layer0": dummy_view}
        monkeypatch.setattr(
            v2_cache, "allocate_upstream_kv_cache", lambda *a: dummy_views
        )

        cfg = SimpleNamespace(
            kv_cache_tensors=[SimpleNamespace(size=16 * 512)],
            num_blocks=16,
        )
        layout = SimpleNamespace(is_layer_compact=False)

        res = v2_cache.materialize_upstream_kv_cache(
            cfg, torch.device("cpu"), layout, word_rows=False
        )
        assert "layer0" in res.kv_caches
        assert len(res.raw_tensors) == 1
        assert res.raw_tensors[0].shape == (16 * 512,)

        # Case 3: layer_compact = True, word_rows = True, with FP8 views
        fp8_view = torch.zeros(16, 1, 4, 128, dtype=torch.uint8)
        fp8_views = {"fp8_layer": fp8_view}
        monkeypatch.setattr(
            v2_cache, "allocate_upstream_kv_cache", lambda *a: fp8_views
        )

        from vllm_torchtpu import kv_cache_materializer

        monkeypatch.setattr(
            kv_cache_materializer, "layer_to_pool_index", lambda c: {"fp8_layer": 0}
        )

        cfg_fp8 = SimpleNamespace(
            kv_cache_tensors=[SimpleNamespace(size=16 * 512)],
            num_blocks=16,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=512))
            ],
        )
        layout_fp8 = SimpleNamespace(is_layer_compact=True)

        res_fp8 = v2_cache.materialize_upstream_kv_cache(
            cfg_fp8, torch.device("cpu"), layout_fp8, word_rows=True
        )
        assert "fp8_layer" in res_fp8.rope_pools
        # Companion raw_tensor appended
        assert len(res_fp8.raw_tensors) == 2


# ==============================================================================
# 4. Tests for v2/cache.py: PagedCacheRows
# ==============================================================================


class TestPagedCacheRows:
    def test_init_validations(self):
        valid_raw = torch.zeros((16, ROW_WORDS), dtype=torch.int32)

        # 1. Invalid raw dtype
        with pytest.raises(
            ValueError, match="Native cache requires DSV4 int32\\[..., 128\\] word rows"
        ):
            v2_cache.PagedCacheRows(
                valid_raw.float(), ((16, 4, 64), torch.bfloat16, 512, 0)
            )

        # 2. Invalid raw ndim or shape
        with pytest.raises(
            ValueError, match="Native cache requires DSV4 int32\\[..., 128\\] word rows"
        ):
            v2_cache.PagedCacheRows(
                torch.zeros((16, 64), dtype=torch.int32),
                ((16, 4, 64), torch.bfloat16, 512, 0),
            )

        # 3. Invalid record dtype
        with pytest.raises(
            ValueError, match="Native cache records must be BF16, FP32 or FP8 bytes"
        ):
            v2_cache.PagedCacheRows(valid_raw, ((16, 4, 64), torch.int64, 512, 0))

        # 4. FP8 uint8 without companion or wrong shape
        with pytest.raises(
            ValueError, match="FP8 records are one pool row plus a RoPE companion"
        ):
            v2_cache.PagedCacheRows(
                valid_raw, ((16, 1, 512), torch.uint8, 512, 0), rope=None
            )

        with pytest.raises(
            ValueError, match="FP8 records are one pool row plus a RoPE companion"
        ):
            v2_cache.PagedCacheRows(
                valid_raw,
                ((16, 1, 512), torch.uint8, 512, 0),
                rope=torch.zeros((2, ROW_WORDS), dtype=torch.int32),
            )

        # 5. Record bytes must divide or be multiple of 512
        with pytest.raises(
            ValueError, match="Cache records must divide or be a multiple of 512 bytes"
        ):
            v2_cache.PagedCacheRows(valid_raw, ((16, 1, 100), torch.bfloat16, 200, 0))

        # 6. Record alignment for pages
        with pytest.raises(ValueError, match="Cache pages must be record-aligned"):
            v2_cache.PagedCacheRows(valid_raw, ((16, 1, 256), torch.bfloat16, 513, 0))

        with pytest.raises(ValueError, match="Cache pages must be record-aligned"):
            v2_cache.PagedCacheRows(valid_raw, ((16, 1, 256), torch.bfloat16, 512, 100))

        # 7. Native cache row counts must fit int32
        huge_raw = SimpleNamespace(
            shape=(torch.iinfo(torch.int32).max + 1, ROW_WORDS),
            ndim=2,
            dtype=torch.int32,
            nbytes=1024,
            device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="Native cache row counts must fit int32"):
            v2_cache.PagedCacheRows(huge_raw, ((16, 1, 256), torch.bfloat16, 512, 0))

    def test_numel_row_indices_pages(self):
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # 16 pages, 4 records per page, 128 words per record
        layout = ((16, 4, 128), torch.bfloat16, 1024, 512)
        paged = v2_cache.PagedCacheRows(raw, layout)

        assert paged.numel() == 16 * 4 * 128
        slots = torch.tensor([0, 1, 4, 7])
        # row_offset = 512 // 256 = 2, page_stride = 1024 // 256 = 4
        # slots // 4 * 4 + slots % 4 + 2 -> slot + 2
        assert torch.equal(paged.row_indices(slots), slots + 2)

        # Aligned pages
        # rows = shape[1] = 4, rows % 4 == 0, row_offset % 4 == 2 != 0 -> returns None
        assert paged.pages(torch.tensor([0, 1])) is None

        # Fully aligned layout
        layout_aligned = ((16, 4, 128), torch.bfloat16, 1024, 0)
        paged_aligned = v2_cache.PagedCacheRows(raw, layout_aligned)
        assert torch.equal(
            paged_aligned.pages(torch.tensor([1, 2])), torch.tensor([1, 2])
        )

    def test_getitem_and_write_cpu(self):
        raw = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
        layout = ((8, 4, 128), torch.bfloat16, 1024, 0)
        paged = v2_cache.PagedCacheRows(raw, layout)

        # Write BF16 values on CPU
        slots = torch.tensor([0, 2])
        values = torch.ones(2, 128, dtype=torch.bfloat16) * 3.5
        paged.write(slots, values)

        # Read back via __getitem__ on CPU
        blocks = torch.tensor([0, 0])
        rows = torch.tensor([0, 2])
        read_back = paged[blocks, rows]
        assert read_back.shape == (2, 128)
        assert torch.allclose(read_back, values)

    def test_getitem_and_write_custom_op_paths(self, monkeypatch: pytest.MonkeyPatch):
        raw = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
        layout = ((8, 4, 128), torch.bfloat16, 1024, 0)
        paged = v2_cache.PagedCacheRows(raw, layout)

        # Simulate non-CPU device on paged.raw
        paged.raw = SimpleNamespace(
            device=SimpleNamespace(type="tpu"),
            copy_=MagicMock(),
        )

        # Test __getitem__ through _read_values_op
        mock_read = MagicMock(return_value=torch.zeros(2, 128, dtype=torch.bfloat16))
        monkeypatch.setattr(v2_cache, "_read_values_op", mock_read)
        out = paged[torch.tensor([0, 0]), torch.tensor([0, 1])]
        assert out.shape == (2, 128)
        mock_read.assert_called_once()

        # Test write without after through _write_values_op
        mock_write = MagicMock(return_value=torch.zeros(1))
        monkeypatch.setattr(v2_cache, "_write_values_op", mock_write)
        paged.write(torch.tensor([0]), torch.zeros(1, 128, dtype=torch.bfloat16))
        mock_write.assert_called_once()
        paged.raw.copy_.assert_called_once()

        # Test write with after through _write_values_after_op
        paged.raw.copy_.reset_mock()
        mock_write_after = MagicMock(return_value=torch.zeros(1))
        monkeypatch.setattr(v2_cache, "_write_values_after_op", mock_write_after)
        paged.write(
            torch.tensor([0]),
            torch.zeros(1, 128, dtype=torch.bfloat16),
            after=torch.tensor(1),
        )
        mock_write_after.assert_called_once()
        paged.raw.copy_.assert_called_once()

    def test_write_fp8_cpu_and_custom_op(self, monkeypatch: pytest.MonkeyPatch):
        raw = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
        rope = torch.zeros((4, ROW_WORDS), dtype=torch.int32)
        layout = ((16, 1, 512), torch.uint8, 512, 0)
        paged = v2_cache.PagedCacheRows(raw, layout, rope=rope)

        # 1. CPU FP8 write
        latents = torch.randn(2, 512, dtype=torch.bfloat16)
        slots = torch.tensor([0, 1])
        paged.write(slots, latents)
        # raw and rope should now be mutated
        assert not torch.equal(raw, torch.zeros_like(raw))
        assert not torch.equal(rope, torch.zeros_like(rope))

        # 2. Non-CPU FP8 write
        paged.raw = SimpleNamespace(
            device=SimpleNamespace(type="tpu"),
            copy_=MagicMock(),
        )
        paged.rope = SimpleNamespace(copy_=MagicMock())

        mock_encode = MagicMock(
            return_value=(
                torch.zeros(2, ROW_WORDS, dtype=torch.int32),
                torch.zeros(2, v2_cache.ROPE_WORDS, dtype=torch.int32),
            )
        )
        mock_write_val = MagicMock(side_effect=[torch.zeros(1), torch.zeros(1)])
        monkeypatch.setattr(v2_cache, "_encode_fp8_op", mock_encode)
        monkeypatch.setattr(v2_cache, "_write_values_op", mock_write_val)

        paged.write(slots, latents)
        mock_encode.assert_called_once()
        assert mock_write_val.call_count == 2
        paged.raw.copy_.assert_called_once()
        paged.rope.copy_.assert_called_once()


# ==============================================================================
# 5. Tests for v2/cache.py: NativeCacheView
# ==============================================================================


class TestNativeCacheView:
    def test_native_cache_view_property_and_binding(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        class BaseModule:
            def bind_kv_cache(self, cache):
                self._bound_cache = cache

        class DummyAdapter(v2_cache.NativeCacheView, BaseModule):
            pass

        adapter = DummyAdapter()
        bound_tensor = torch.zeros(16, 4, 128, dtype=torch.bfloat16)

        # 1. Non-compiling or raw is None -> returns bound_cache
        adapter.bind_kv_cache(bound_tensor)
        assert adapter.kv_cache is bound_tensor

        # 2. Compiling with raw_cache present -> returns PagedCacheRows
        raw_cpu = torch.zeros(64, ROW_WORDS, dtype=torch.int32)
        adapter.bind_kv_cache(bound_tensor, raw_cache=raw_cpu)

        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        view_result = adapter.kv_cache
        assert isinstance(view_result, v2_cache.PagedCacheRows)

        # 3. Setter
        new_tensor = torch.zeros(8, 4, 128)
        adapter.kv_cache = new_tensor
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        assert adapter.kv_cache is new_tensor

        # 4. Error when raw_cache is on TPU without explicit layout
        tpu_raw = SimpleNamespace(device=SimpleNamespace(type="tpu"))
        with pytest.raises(
            ValueError, match="TPU native caches require explicit allocator layout"
        ):
            adapter.bind_kv_cache(bound_tensor, raw_cache=tpu_raw, layout=None)

        # 5. Explicit layout provided (lines 316-317)
        explicit_layout = ((8, 2, 128), torch.bfloat16, 512, 0)
        adapter.bind_kv_cache(bound_tensor, layout=explicit_layout)
        assert adapter._tpu_cache_layout == explicit_layout


# ==============================================================================
# 6. Tests for v2/cache.py: JAX Helper Functions
# ==============================================================================


class TestCacheJaxHelpers:
    def test_scatter_rows(self):
        cache = jnp.zeros((8, 16), dtype=jnp.float32)
        slots = jnp.array([1, -1, 3], dtype=jnp.int32)
        values = jnp.ones((3, 16), dtype=jnp.float32)

        out = v2_cache._scatter_rows(cache, slots, values)
        assert jnp.all(out[1] == 1.0)
        assert jnp.all(out[3] == 1.0)
        # Slot -1 should be dropped
        assert jnp.all(out[0] == 0.0)

    def test_gather_words(self):
        cache = jnp.arange(16 * ROW_WORDS, dtype=jnp.int32).reshape(16, ROW_WORDS)

        # 1. record_words >= ROW_WORDS (chunked)
        records = jnp.array([0, 2], dtype=jnp.int32)
        out_chunk = v2_cache._gather_words(cache, records, 2 * ROW_WORDS)
        assert out_chunk.shape == (2, 2 * ROW_WORDS)

        # 2. record_words < ROW_WORDS (sub-row)
        out_sub = v2_cache._gather_words(cache, records, 32)
        assert out_sub.shape == (2, 32)

    def test_permute_and_halves(self):
        x = jnp.arange(256, dtype=jnp.float32).reshape(1, 256)
        split = v2_cache.split_halves(x)
        assert split.shape == x.shape
        merged = v2_cache.merge_halves(split)
        assert jnp.allclose(merged, x)

    def test_split_and_merge_pieces(self):
        x = jnp.arange(2 * 4 * 128, dtype=jnp.float32).reshape(2, 4, 128)
        # Split into 2 pieces: shape [2, 2, 4, 64]
        pieces = v2_cache.split_pieces(x, pieces=2)
        assert pieces.shape == (2, 2, 4, 64)
        merged = v2_cache.merge_pieces(pieces)
        assert jnp.allclose(merged, x)

    def test_gather_cache_pieces_and_values(self):
        cache = jnp.arange(32 * ROW_WORDS, dtype=jnp.int32).reshape(32, ROW_WORDS)
        records = jnp.array([0, 1, -1], dtype=jnp.int32)

        # gather_cache_pieces small (<= ROW_WORDS)
        pieces_small = v2_cache.gather_cache_pieces(cache, records, elements=64)
        assert pieces_small.ndim == 3

        # gather_cache_pieces chunked (> ROW_WORDS)
        pieces_chunked = v2_cache.gather_cache_pieces(cache, records, elements=512)
        assert pieces_chunked.ndim == 3

        # gather_cache_values split=True small and chunked
        val_split_small = v2_cache.gather_cache_values(
            cache, records, elements=64, split=True
        )
        assert val_split_small.shape == (3, 64)

        val_split_chunk = v2_cache.gather_cache_values(
            cache, records, elements=512, split=True
        )
        assert val_split_chunk.shape == (3, 512)

        # gather_cache_values split=False fp32
        val_fp32 = v2_cache.gather_cache_values(
            cache, records, elements=32, fp32=True, split=False
        )
        assert val_fp32.shape == (3, 32)
        assert val_fp32.dtype == jnp.float32

        # gather_cache_values split=False bf16
        val_bf16 = v2_cache.gather_cache_values(
            cache, records, elements=64, fp32=False, split=False
        )
        assert val_bf16.shape == (3, 64)
        assert val_bf16.dtype == jnp.bfloat16

    def test_gather_cache_page_values(self):
        cache = jnp.zeros((64, ROW_WORDS), dtype=jnp.int32)

        # Error on unaligned record_words
        with pytest.raises(ValueError, match="Pages must cover whole pool word rows"):
            v2_cache.gather_cache_page_values(
                cache, jnp.array([0]), page_records=3, elements=100
            )

        # Valid call
        first_records = jnp.array([0, 4, -1], dtype=jnp.int32)
        out = v2_cache.gather_cache_page_values(
            cache, first_records, page_records=4, elements=64
        )
        assert out.shape == (12, 64)
        assert out.dtype == jnp.bfloat16

    def test_encode_fp8_kv(self):
        key = jax.random.PRNGKey(42)
        latent = jax.random.normal(key, (4, 512)).astype(jnp.bfloat16)
        words, rope = v2_cache.encode_fp8_kv(latent)
        assert words.shape == (4, ROW_WORDS)
        assert words.dtype == jnp.int32
        assert rope.shape == (4, v2_cache.ROPE_WORDS)
        assert rope.dtype == jnp.int32


# ==============================================================================
# 7. Tests for v2/cache.py: Ops Initialization & Cache Mutation
# ==============================================================================


class TestCacheMutationOps:
    def test_initialize_cache_ops_and_fakes(self, monkeypatch: pytest.MonkeyPatch):
        from vllm_torchtpu.models.vllm import vllm_model_wrapper_context

        mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tp",))
        ctx = SimpleNamespace(mesh=mesh)

        monkeypatch.setattr(
            vllm_model_wrapper_context,
            "get_vllm_model_wrapper_context",
            lambda: ctx,
        )

        # Intercept jax_op to capture write_values_after (lines 642-643)
        from torch_tpu._internal import pallas

        captured_ops = {}
        captured_kwargs = {}
        orig_jax_op = pallas.jax_op

        def intercept_jax_op(name, fn, *args, **kwargs):
            captured_ops[name] = fn
            captured_kwargs[name] = kwargs
            return orig_jax_op(name, fn, *args, **kwargs)

        monkeypatch.setattr(pallas, "jax_op", intercept_jax_op)

        # Reset global ops to force re-initialization
        monkeypatch.setattr(v2_cache, "_write_op", None)
        v2_cache.initialize_cache_ops()
        assert v2_cache._write_op is not None
        assert captured_kwargs["pallas::dsv41v2_cache_write"].get("donate_argnums") == (
            0,
        )

        # Idempotency check
        v2_cache.initialize_cache_ops()

        # Exercise real write_values_after (lines 642-643)
        wva = captured_ops.get("pallas::dsv41v2_cache_write_values_after")
        assert wva is not None
        c_jax = jnp.zeros((8, ROW_WORDS), dtype=jnp.int32)
        r_jax = jnp.zeros(2, dtype=jnp.int32)
        v_jax = jnp.zeros((2, ROW_WORDS), dtype=jnp.int32)
        after_jax = jnp.zeros(2, dtype=jnp.int32)
        res_wva = wva(c_jax, r_jax, v_jax, after_jax)
        assert res_wva.shape == c_jax.shape

        # Exercise registered fake handlers
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            c_fake = torch.empty(16, ROW_WORDS, dtype=torch.int32)
            r_fake = torch.empty(4, dtype=torch.int32)
            val_fake = torch.empty(4, 64, dtype=torch.bfloat16)

            # _read_values_op fake
            out_read = v2_cache._read_values_op(c_fake, r_fake, 64, False)
            assert out_read.shape == (4, 64)
            assert out_read.dtype == torch.bfloat16

            # _write_op fake
            out_write = v2_cache._write_op(c_fake, r_fake, val_fake)
            assert out_write.shape == c_fake.shape

            # _write_values_op fake
            out_wv = v2_cache._write_values_op(c_fake, r_fake, val_fake)
            assert out_wv.shape == c_fake.shape

            # _write_values_after_op fake
            out_wva = v2_cache._write_values_after_op(
                c_fake, r_fake, val_fake, out_read
            )
            assert out_wva.shape == c_fake.shape

            # _encode_fp8_op fake
            lat_fake = torch.empty(4, 512, dtype=torch.bfloat16)
            w_fake, rp_fake = v2_cache._encode_fp8_op(lat_fake)
            assert w_fake.shape == (4, ROW_WORDS)
            assert rp_fake.shape == (4, v2_cache.ROPE_WORDS)

    def test_write_cache_dispatch(self, monkeypatch: pytest.MonkeyPatch):
        # 1. PagedCacheRows branch
        mock_paged = MagicMock(spec=v2_cache.PagedCacheRows)
        v2_cache.write_cache(
            mock_paged,
            torch.tensor([0, 1]),
            torch.zeros(2, 64),
            torch.tensor([True, True]),
        )
        mock_paged.write.assert_called_once()

        # 2. CPU tensor: cache.ndim == values.ndim
        cache_2d = torch.zeros(8, 64)
        slots = torch.tensor([1, 2])
        values = torch.ones(2, 64) * 4.0
        v2_cache.write_cache(cache_2d, slots, values, torch.tensor([True, True]))
        assert torch.all(cache_2d[1] == 4.0)
        assert torch.all(cache_2d[2] == 4.0)

        # 3. CPU tensor: cache.ndim != values.ndim
        cache_3d = torch.zeros(4, 2, 64)
        v2_cache.write_cache(cache_3d, slots, values, torch.tensor([True, True]))
        # Slot 1 -> (0, 1), Slot 2 -> (1, 0)
        assert torch.all(cache_3d[0, 1] == 4.0)
        assert torch.all(cache_3d[1, 0] == 4.0)

        # 4. Non-CPU tensor: calls _write_op
        mock_non_cpu = SimpleNamespace(
            device=SimpleNamespace(type="tpu"),
            copy_=MagicMock(),
        )
        mock_write_op = MagicMock(return_value=torch.zeros(1))
        monkeypatch.setattr(v2_cache, "_write_op", mock_write_op)

        v2_cache.write_cache(mock_non_cpu, slots, values, torch.tensor([True, False]))
        mock_write_op.assert_called_once()
        mock_non_cpu.copy_.assert_called_once()

    def test_merge_sub_rows_and_pack_bf16_words(self):
        cache = jnp.zeros((8, ROW_WORDS), dtype=jnp.int32)
        records = jnp.array([0, 1], dtype=jnp.int32)
        words = jnp.ones((2, 64), dtype=jnp.int32)

        rows, merged = v2_cache.merge_sub_rows(cache, records, words)
        assert rows.shape == (2,)
        assert merged.shape == (2, ROW_WORDS)

        # _pack_bf16_words
        bf16_vals = jnp.ones((2, 2 * ROW_WORDS), dtype=jnp.bfloat16)
        packed = v2_cache._pack_bf16_words(bf16_vals)
        assert packed.shape == (2, ROW_WORDS)
        assert packed.dtype == jnp.int32

    def test_scatter_cache_values_branches(self, monkeypatch: pytest.MonkeyPatch):
        cache = jnp.zeros((8, ROW_WORDS), dtype=jnp.int32)
        records = jnp.array([0, 1], dtype=jnp.int32)

        # 1. Empty values -> returns cache immediately
        empty_vals = jnp.zeros((0, ROW_WORDS), dtype=jnp.int32)
        assert v2_cache.scatter_cache_values(cache, records[:0], empty_vals) is cache

        # 2. int32 values on CPU
        int32_vals = jnp.ones((2, 64), dtype=jnp.int32)
        out_i32 = v2_cache.scatter_cache_values(cache, records, int32_vals)
        assert out_i32.shape == cache.shape

        # 3. width >= ROW_WORDS on CPU
        full_vals = jnp.ones((2, ROW_WORDS), dtype=jnp.int32)
        out_full = v2_cache.scatter_cache_values(cache, records, full_vals)
        assert out_full.shape == cache.shape

        # Mock pl.pallas_call before entering TPU backend branches
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        pallas_called = False

        def mock_pallas(kernel, *args, **kwargs):
            nonlocal pallas_called
            pallas_called = True

            def invoke(rec, w_flat, c):
                rec_ref = [0, 1, -1] + [0] * 64
                val_ref = MagicMock()
                out_ref = MagicMock()
                sem = MagicMock()
                orig_pid = getattr(pl, "program_id", None)
                orig_ds = getattr(pl, "ds", None)
                orig_copy = getattr(pltpu, "make_async_copy", None)
                orig_when = getattr(pl, "when", None)
                orig_fori = getattr(jax.lax, "fori_loop", None)
                try:
                    pl.program_id = lambda _: 0
                    pl.ds = lambda s, sz: slice(s, s + sz)
                    copy_mock = SimpleNamespace(start=MagicMock(), wait=MagicMock())
                    pltpu.make_async_copy = lambda s, d, sm: copy_mock
                    pl.when = lambda c: lambda fn: fn()
                    jax.lax.fori_loop = lambda lower, upper, body, init: [
                        body(i, init) for i in range(lower, min(upper, 2))
                    ]
                    kernel(rec_ref, val_ref, None, out_ref, sem)
                finally:
                    if orig_pid:
                        pl.program_id = orig_pid
                    if orig_ds:
                        pl.ds = orig_ds
                    if orig_copy:
                        pltpu.make_async_copy = orig_copy
                    if orig_when:
                        pl.when = orig_when
                    if orig_fori:
                        jax.lax.fori_loop = orig_fori
                return c

            return invoke

        monkeypatch.setattr(pl, "pallas_call", mock_pallas)

        # 4. TPU backend with width < ROW_WORDS (merge_sub_rows branch)
        monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
        out_tpu_sub = v2_cache.scatter_cache_values(cache, records, int32_vals)
        assert out_tpu_sub.shape == cache.shape
        assert pallas_called is True

        # 5. TPU backend with width >= ROW_WORDS (Pallas DMA branch)
        pallas_called = False
        out_tpu_pallas = v2_cache.scatter_cache_values(cache, records, full_vals)
        assert pallas_called is True
        assert out_tpu_pallas.shape == cache.shape
