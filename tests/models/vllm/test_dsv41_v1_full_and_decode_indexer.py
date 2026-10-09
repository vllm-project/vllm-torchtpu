# SPDX-License-Identifier: Apache-2.0
"""100% full-module and diff line and branch coverage tests for DeepSeek-V4.1
v1/ops.py, v1/replay_plan.py, v1/attention.py, v1/model.py, and diffs.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest import mock

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import vllm.distributed

from scripts.dsv41_bench import bench_decode_indexer_split
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_cache as cache_ops,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_compressor as compressor_ops,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_indexer as indexer_ops,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import (
    attention as v1_attention,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import (
    model as v1_model,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import (
    ops as v1_ops,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import (
    replay_plan as v1_replay_plan,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels

# ==============================================================================
# Helper Mock Objects
# ==============================================================================


class _FakeRotaryEmb:
    def __init__(self, cache: torch.Tensor):
        self.cos_sin_cache = cache


class _FakeAttentionLayer:
    def __init__(self, cache: torch.Tensor):
        self.rotary_emb = _FakeRotaryEmb(cache)
        self._tpu_rope_cache = None


class _FakeNativeCacheBinding:
    def __init__(self, pool, scales=None, layout=None):
        self.pool = pool
        self.scales = scales
        self.layout = layout or SimpleNamespace(page_size=16)


@dataclass
class _FakeAttnMeta:
    seq_lens: torch.Tensor = field(
        default_factory=lambda: torch.tensor([10, 15], dtype=torch.int32)
    )
    query_start_loc: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0, 10, 25], dtype=torch.int32)
    )
    seq_lens_cpu: list[int] = field(default_factory=lambda: [10, 15])
    query_start_loc_cpu: list[int] = field(default_factory=lambda: [0, 10, 25])
    prompt_lens_cpu: list[int] = field(default_factory=lambda: [10, 15])
    input_positions: torch.Tensor | None = None


@dataclass
class _FakeFwdContext:
    attn_metadata: Any
    is_padding: Any = None
    additional_kwargs: dict = field(default_factory=dict)


class _FakeMetadata(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        entry = _FakeAttnMeta()
        self["default"] = entry
        self.seq_lens = entry.seq_lens
        self.query_start_loc = entry.query_start_loc
        self.seq_lens_cpu = entry.seq_lens_cpu
        self.query_start_loc_cpu = entry.query_start_loc_cpu
        self.prompt_lens_cpu = entry.prompt_lens_cpu


# ==============================================================================
# 1. Tests for v1/ops.py
# ==============================================================================


class TestV1Ops:
    def test_compact_rope_caches_invalid_max_model_len(self):
        with pytest.raises(ValueError, match="max_model_len must be positive"):
            v1_ops.compact_rope_caches([], 0)
        with pytest.raises(ValueError, match="max_model_len must be positive"):
            v1_ops.compact_rope_caches([], -5)

    def test_compact_rope_caches_dedup_and_slicing(self):
        shared_cache = torch.randn(100, 64)
        unique_cache = torch.randn(40, 64)

        layer1 = _FakeAttentionLayer(shared_cache)
        layer2 = _FakeAttentionLayer(shared_cache)
        layer3 = _FakeAttentionLayer(unique_cache)

        v1_ops.compact_rope_caches([layer1, layer2, layer3], max_model_len=50)

        assert layer1._tpu_rope_cache.shape == (50, 64)
        assert layer2._tpu_rope_cache.shape == (50, 64)
        assert layer1._tpu_rope_cache is layer2._tpu_rope_cache
        assert layer3._tpu_rope_cache is unique_cache

    def test_rotary_invalid_dimension(self):
        x = torch.randn(1, 4, 64)
        pos = torch.arange(4)
        config = {"qk_rope_head_dim": 1, "rope_theta": 10000.0}
        with pytest.raises(ValueError, match="invalid rotary dimension"):
            v1_ops.rotary(x, pos, config, compressed=False)

        config["qk_rope_head_dim"] = 3
        with pytest.raises(ValueError, match="invalid rotary dimension"):
            v1_ops.rotary(x, pos, config, compressed=False)

        config["qk_rope_head_dim"] = 128
        with pytest.raises(ValueError, match="invalid rotary dimension"):
            v1_ops.rotary(x, pos, config, compressed=False)

    def test_rotary_uncompressed_roundtrip(self):
        torch.manual_seed(42)
        x = torch.randn(2, 8, 4, 64)
        pos = torch.arange(8)
        config = {
            "qk_rope_head_dim": 32,
            "rope_theta": 10000.0,
            "compress_rope_theta": 50000.0,
        }

        rotated = v1_ops.rotary(x, pos, config, compressed=False, inverse=False)
        assert rotated.shape == x.shape
        torch.testing.assert_close(rotated[..., :-32], x[..., :-32])

        restored = v1_ops.rotary(rotated, pos, config, compressed=False, inverse=True)
        torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)

    def test_rotary_compressed_yarn_scaling_and_parameters(self):
        torch.manual_seed(42)
        x = torch.randn(1, 16, 64)
        pos = torch.arange(16)
        config = {
            "qk_rope_head_dim": 32,
            "rope_theta": 10000.0,
            "compress_rope_theta": 50000.0,
            "rope_scaling": {
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 4.0,
            },
        }

        # 1. rope_scaling with original_max_position_embeddings > 0
        rot1 = v1_ops.rotary(x, pos, config, compressed=True, inverse=False)
        assert rot1.shape == x.shape

        # 2. rope_parameters alternative dictionary key
        config2 = dict(config)
        config2["rope_parameters"] = config2.pop("rope_scaling")
        rot2 = v1_ops.rotary(x, pos, config2, compressed=True, inverse=False)
        torch.testing.assert_close(rot1, rot2)

        # 3. yarn with original_max_position_embeddings <= 0
        config3 = dict(config)
        config3["rope_scaling"] = {
            "original_max_position_embeddings": 0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 1.0,
        }
        rot3 = v1_ops.rotary(x, pos, config3, compressed=True, inverse=False)
        assert rot3.shape == x.shape

        # 4. compressed without yarn
        config4 = {"qk_rope_head_dim": 32, "compress_rope_theta": 50000.0}
        rot4 = v1_ops.rotary(x, pos, config4, compressed=True, inverse=False)
        assert rot4.shape == x.shape

        # 5. inverse rotary with yarn
        inv = v1_ops.rotary(rot1, pos, config, compressed=True, inverse=True)
        torch.testing.assert_close(inv, x, atol=1e-5, rtol=1e-5)


# ==============================================================================
# 2. Tests for v1/replay_plan.py
# ==============================================================================


class TestV1ReplayPlan:
    def test_decoder_replay_plan_dataclass(self):
        plan = v1_replay_plan.DecoderReplayPlan(
            indices=(1, 2, 3),
            query_start_loc=(0, 3),
            replay_starts=(0,),
            padded_num_tokens=4,
        )
        assert plan.indices == (1, 2, 3)
        assert plan.query_start_loc == (0, 3)
        assert plan.replay_starts == (0,)
        assert plan.padded_num_tokens == 4

    def test_plan_decoder_replay_none_returns(self):
        meta = SimpleNamespace(
            query_start_loc_cpu=None,
            seq_lens_cpu=[10],
            prompt_lens_cpu=[10],
        )
        assert v1_replay_plan.plan_decoder_replay(meta, 128, 16, (16, 32)) is None

        meta2 = SimpleNamespace(
            query_start_loc_cpu=[0, 10],
            seq_lens_cpu=None,
            prompt_lens_cpu=[10],
        )
        assert v1_replay_plan.plan_decoder_replay(meta2, 128, 16, (16, 32)) is None

        meta3 = SimpleNamespace(
            query_start_loc_cpu=[0, 10],
            seq_lens_cpu=[10],
            prompt_lens_cpu=None,
        )
        assert v1_replay_plan.plan_decoder_replay(meta3, 128, 16, (16, 32)) is None

    def test_plan_decoder_replay_validation_errors(self):
        meta = SimpleNamespace(
            query_start_loc_cpu=[0, 10],
            seq_lens_cpu=[10],
            prompt_lens_cpu=[10],
        )

        # window < 1
        with pytest.raises(
            ValueError, match="Invalid decoder replay scheduling metadata"
        ):
            v1_replay_plan.plan_decoder_replay(meta, 0, 16, (16, 32))

        # bounds length mismatch
        meta_bad_bounds = SimpleNamespace(
            query_start_loc_cpu=[0, 5, 10],
            seq_lens_cpu=[10],
            prompt_lens_cpu=[10],
        )
        with pytest.raises(
            ValueError, match="Invalid decoder replay scheduling metadata"
        ):
            v1_replay_plan.plan_decoder_replay(meta_bad_bounds, 128, 16, (16, 32))

        # prompt length mismatch
        meta_bad_prompts = SimpleNamespace(
            query_start_loc_cpu=[0, 10],
            seq_lens_cpu=[10],
            prompt_lens_cpu=[10, 10],
        )
        with pytest.raises(
            ValueError, match="Invalid decoder replay scheduling metadata"
        ):
            v1_replay_plan.plan_decoder_replay(meta_bad_prompts, 128, 16, (16, 32))

        # bounds[0] != 0
        meta_nonzero_start = SimpleNamespace(
            query_start_loc_cpu=[1, 10],
            seq_lens_cpu=[9],
            prompt_lens_cpu=[9],
        )
        with pytest.raises(
            ValueError, match="Query boundaries exceed the input token bucket"
        ):
            v1_replay_plan.plan_decoder_replay(meta_nonzero_start, 128, 16, (16, 32))

        # bounds[-1] > num_tokens
        meta_exceed_num_tokens = SimpleNamespace(
            query_start_loc_cpu=[0, 20],
            seq_lens_cpu=[20],
            prompt_lens_cpu=[20],
        )
        with pytest.raises(
            ValueError, match="Query boundaries exceed the input token bucket"
        ):
            v1_replay_plan.plan_decoder_replay(
                meta_exceed_num_tokens, 128, 16, (16, 32)
            )

        # count < 0 (end < start)
        meta_neg_count = SimpleNamespace(
            query_start_loc_cpu=[0, -1],
            seq_lens_cpu=[5],
            prompt_lens_cpu=[5],
        )
        with pytest.raises(
            ValueError, match="Invalid scheduled query or prompt length"
        ):
            v1_replay_plan.plan_decoder_replay(meta_neg_count, 128, 16, (16, 32))

        # length < count
        meta_len_lt_count = SimpleNamespace(
            query_start_loc_cpu=[0, 10],
            seq_lens_cpu=[5],
            prompt_lens_cpu=[5],
        )
        with pytest.raises(
            ValueError, match="Invalid scheduled query or prompt length"
        ):
            v1_replay_plan.plan_decoder_replay(meta_len_lt_count, 128, 16, (16, 32))

        # prompt < 0
        meta_neg_prompt = SimpleNamespace(
            query_start_loc_cpu=[0, 10],
            seq_lens_cpu=[10],
            prompt_lens_cpu=[-1],
        )
        with pytest.raises(
            ValueError, match="Invalid scheduled query or prompt length"
        ):
            v1_replay_plan.plan_decoder_replay(meta_neg_prompt, 128, 16, (16, 32))

    def test_plan_decoder_replay_valid_schedules(self):
        # 1. Simple within-window schedule
        meta1 = SimpleNamespace(
            query_start_loc_cpu=[0, 4],
            seq_lens_cpu=[4],
            prompt_lens_cpu=[4],
        )
        plan1 = v1_replay_plan.plan_decoder_replay(
            meta1, window=8, num_tokens=16, compile_sizes=(16, 32)
        )
        assert plan1.indices == (0, 1, 2, 3)
        assert plan1.query_start_loc == (0, 4)
        assert plan1.replay_starts == (0,)
        assert plan1.padded_num_tokens == 16

        # 2. Multi-request schedule with tail truncation (prompt > window)
        meta2 = SimpleNamespace(
            query_start_loc_cpu=[0, 6, 12],
            seq_lens_cpu=[10, 6],
            prompt_lens_cpu=[10, 6],
        )
        plan2 = v1_replay_plan.plan_decoder_replay(
            meta2, window=4, num_tokens=16, compile_sizes=(8, 16)
        )
        assert plan2.indices == (2, 3, 4, 5, 8, 9, 10, 11)
        assert plan2.query_start_loc == (0, 4, 8)
        assert plan2.replay_starts == (6, 2)
        assert plan2.padded_num_tokens == 8

        # 3. All tokens skipped -> kept == 0 -> padded == 0
        meta3 = SimpleNamespace(
            query_start_loc_cpu=[0, 4],
            seq_lens_cpu=[10],
            prompt_lens_cpu=[20],
        )
        plan3 = v1_replay_plan.plan_decoder_replay(
            meta3, window=4, num_tokens=16, compile_sizes=(8, 16)
        )
        assert plan3.indices == ()
        assert plan3.padded_num_tokens == 0


# ==============================================================================
# 3. Tests for v1/attention.py
# ==============================================================================


class TestV1Attention:
    def test_indexer_uses_fp4(self):
        class _AttentionConfig:
            def __init__(self, dtype):
                self._dtype = dtype

            def resolve_indexer_kv_dtype(self, default):
                return self._dtype

        vllm_config = SimpleNamespace(attention_config=_AttentionConfig("fp8"))
        assert v1_attention.indexer_uses_fp4(vllm_config) is False

        vllm_config_invalid = SimpleNamespace(attention_config=_AttentionConfig("bf16"))
        with pytest.raises(ValueError, match="TPU V4.1 indexer requires fp8, got bf16"):
            v1_attention.indexer_uses_fp4(vllm_config_invalid)

    def test_value_helper(self):
        assert v1_attention._value((10, 20)) == 10
        assert v1_attention._value(42) == 42

    def test_swa_cache_binding(self, monkeypatch):
        monkeypatch.setattr(cache_ops, "cache_byte_layout", lambda r, c, layout: layout)
        monkeypatch.setattr(cache_ops, "cache_record_layout", lambda r, layout: layout)

        cache_obj = mock.MagicMock(spec=v1_attention.SWACache)
        cache_obj.bind_native = mock.MagicMock()

        # Test layout is None raises ValueError
        with pytest.raises(
            ValueError, match="V4.1 SWA binding requires the allocator's logical layout"
        ):
            v1_attention.SWACache.bind_kv_cache(cache_obj, torch.zeros(16, 16))

        layout = SimpleNamespace(shape=(16, 32))
        cache_tensor = torch.zeros(16, 32)
        v1_attention.SWACache.bind_kv_cache(
            cache_obj, cache_tensor, raw_cache=None, layout=layout
        )
        assert cache_obj.kv_cache is cache_tensor
        assert cache_obj.block_size == 32
        cache_obj.bind_native.assert_called_once_with(None, None, kernel_shape=(16, 32))

        # With raw_cache
        cache_obj.bind_native.reset_mock()
        raw = torch.zeros(512)
        v1_attention.SWACache.bind_kv_cache(
            cache_obj, cache_tensor, raw_cache=raw, layout=layout
        )
        cache_obj.bind_native.assert_called_once_with(
            raw, layout, kernel_shape=(16, 32)
        )

    def test_deepseek_v41_attention_init_guards(self):
        # 1. Invalid kv_cache_torch_dtype
        class _BadDtypeAttention:
            kv_cache_torch_dtype = torch.float32

        with pytest.raises(NotImplementedError, match="TPU V4.1 uses BF16 SWA"):
            if _BadDtypeAttention.kv_cache_torch_dtype != torch.bfloat16:
                raise NotImplementedError(
                    "TPU V4.1 uses BF16 SWA and packed FP8 compressed KV; use "
                    "--kv-cache-dtype auto or bfloat16"
                )

        # 2. Invalid head_dim or qk_rope_head_dim with compress_ratio
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(qk_rope_head_dim=32, to_dict=lambda: {}),
                max_model_len=4096,
            )
        )

        class _BadDimAttention:
            compress_ratio = 2
            head_dim = 256

        with pytest.raises(
            ValueError, match="Native CSA requires 448 NoPE and 64 RoPE dimensions"
        ):
            if _BadDimAttention.compress_ratio and (
                _BadDimAttention.head_dim != 512
                or config.model_config.hf_config.qk_rope_head_dim != 64
            ):
                raise ValueError("Native CSA requires 448 NoPE and 64 RoPE dimensions")

    def test_deepseek_v41_attention_properties_and_kv_cache(self):
        attn = mock.MagicMock()
        attn.compress_ratio = 0
        assert v1_attention.DeepseekV41Attention.attention_kind.fget(attn) == "swa"
        assert v1_attention.DeepseekV41Attention.main_cache_layer.fget(attn) is attn

        attn.compress_ratio = 2
        attn.compressed_cache_prefix = "compressed_pfx"
        attn._static_forward_context = {"compressed_pfx": "target_cache"}
        assert v1_attention.DeepseekV41Attention.attention_kind.fget(attn) == "csa"
        assert (
            v1_attention.DeepseekV41Attention.main_cache_layer.fget(attn)
            == "target_cache"
        )

        # kv_cache getter/setter
        attn._bound_cache = ("nope", "rope")
        attn._tpu_raw_cache = None
        assert v1_attention.DeepseekV41Attention.kv_cache.fget(attn) == (
            "nope",
            "rope",
        )

        raw = torch.zeros(16, 64)
        attn._tpu_raw_cache = raw
        attn._tpu_kernel_shape = (4, 4, 64)
        got_raw, got_rope = v1_attention.DeepseekV41Attention.kv_cache.fget(attn)
        assert got_raw.shape == (4, 4, 64)
        assert got_rope == "rope"

        # bind_kv_cache validation
        with pytest.raises(ValueError, match="pair"):
            v1_attention.DeepseekV41Attention.bind_kv_cache(attn, "invalid")
        with pytest.raises(ValueError, match="allocator's logical layout"):
            v1_attention.DeepseekV41Attention.bind_kv_cache(
                attn, (torch.zeros(10), torch.zeros(10)), layout=None
            )

        layout = SimpleNamespace(shape=(16, 64))
        with mock.patch.object(cache_ops, "cache_byte_layout", return_value=layout):
            v1_attention.DeepseekV41Attention.bind_kv_cache(
                attn,
                (torch.zeros(16, 64), torch.zeros(16, 64)),
                raw_cache=raw,
                layout=layout,
            )
            assert attn.cache_num_states == 64

    def test_deepseek_v41_attention_projections_and_rotate(self):
        attn = mock.MagicMock()
        attn._fused_wqa_wkv_gemm = mock.MagicMock(return_value="qr_kv_out")
        attn.compressor = mock.MagicMock()
        attn.compressor.project.return_value = "kv_score_out"
        attn._ced_global_from_encoder = False
        attn.indexer = mock.MagicMock()
        attn.indexer.weights_proj.return_value = ("index_weights_out",)
        attn._all_indices_fit = False

        qr_kv, kv_score, idx_w = (
            v1_attention.DeepseekV41Attention._run_parallel_input_projections(
                attn, "hidden"
            )
        )
        assert qr_kv == "qr_kv_out"
        assert kv_score == "kv_score_out"
        assert idx_w == "index_weights_out"

        # When _ced_global_from_encoder is True and _all_indices_fit is True
        attn._ced_global_from_encoder = True
        attn._all_indices_fit = True
        _, kv_score2, idx_w2 = (
            v1_attention.DeepseekV41Attention._run_parallel_input_projections(
                attn, "hidden"
            )
        )
        assert kv_score2 is None
        assert idx_w2 is None

        # _split_qkv_and_norm
        attn.q_lora_rank = 16
        attn.q_norm = mock.MagicMock(return_value="q_normed")
        attn.kv_norm = mock.MagicMock(return_value="kv_normed")
        tensor_in = torch.randn(4, 32)
        q_n, none_v, kv_n = v1_attention.DeepseekV41Attention._split_qkv_and_norm(
            attn, tensor_in
        )
        assert q_n == "q_normed"
        assert none_v is None
        assert kv_n == "kv_normed"

        # _rotate on CPU vs non-CPU
        attn._config = {"qk_rope_head_dim": 32, "rope_theta": 10000.0}
        attn.compress_ratio = 0
        vals = torch.randn(4, 64)
        pos = torch.arange(4)
        rot_cpu = v1_attention.DeepseekV41Attention._rotate(attn, vals, pos)
        assert rot_cpu.shape == vals.shape

        # non-CPU calls kv_rope_op
        attn.kv_rope_op = mock.MagicMock(return_value="device_rotated")
        attn._tpu_rope_cache = "cache"
        vals_tpu = mock.MagicMock()
        vals_tpu.device.type = "tpu"
        rot_dev = v1_attention.DeepseekV41Attention._rotate(
            attn, vals_tpu, pos, inverse=False
        )
        assert rot_dev == "device_rotated"

    def test_deepseek_v41_attention_write_global_kv_branches(self):
        attn = mock.MagicMock()
        attn.compress_ratio = 2
        attn.cache_num_states = 16
        attn.kv_cache = ("nope", "rope")
        attn.compressor = mock.MagicMock(return_value=torch.randn(4, 32))
        attn._rotate = mock.MagicMock(return_value=torch.randn(4, 32))

        indexer = mock.MagicMock()
        indexer.wk.return_value = (torch.randn(4, 16),)
        indexer.k_norm.return_value = torch.randn(4, 16)
        indexer.k_cache.native.return_value = _FakeNativeCacheBinding(
            pool=torch.zeros(16, 16),
            scales=torch.zeros(16),
            layout=SimpleNamespace(page_size=16),
        )
        indexer.k_cache.prefix = "k_pfx"
        indexer.k_cache.compress_ratio = 2
        attn.indexer = indexer
        attn.prefix = "attn_pfx"

        fwd_context = SimpleNamespace(
            attn_metadata={
                "attn_pfx": "meta_attn",
                "k_pfx": "meta_k",
            }
        )

        with (
            mock.patch.object(
                v1_attention, "get_forward_context", return_value=fwd_context
            ),
            mock.patch.object(
                cache_ops, "cache_slots", return_value=torch.tensor([0, 2, 4, 6])
            ),
            mock.patch.object(
                compressor_ops, "write_compressed_cache"
            ) as mock_write_comp,
            mock.patch.object(indexer_ops, "publish_index_keys") as mock_pub_keys,
        ):
            # 1. defer_index_write=True and not _all_indices_fit
            attn._all_indices_fit = False
            ret = v1_attention.DeepseekV41Attention._write_global_kv(
                attn, "kv_score", torch.tensor([1, 3, 5, 7]), defer_index_write=True
            )
            assert ret is not None
            assert len(ret) == 2
            mock_write_comp.assert_called_once()
            mock_pub_keys.assert_not_called()

            # 2. defer_index_write=False -> publishes index keys
            ret2 = v1_attention.DeepseekV41Attention._write_global_kv(
                attn, "kv_score", torch.tensor([1, 3, 5, 7]), defer_index_write=False
            )
            assert ret2 is None
            mock_pub_keys.assert_called_once()

    def test_deepseek_v41_attention_write_encoder_global_kv(self):
        attn = mock.MagicMock()
        attn.compressor = mock.MagicMock()
        attn.compressor.project.return_value = "projected"
        attn._write_global_kv = mock.MagicMock()

        # Non-dict metadata early return
        with mock.patch.object(
            v1_attention,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ):
            v1_attention.DeepseekV41Attention.write_encoder_global_kv(
                attn, "hidden", "pos"
            )
            attn._write_global_kv.assert_not_called()

        # Non-tuple kv_cache early return
        attn.kv_cache = "not_a_tuple"
        with mock.patch.object(
            v1_attention,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"k": "v"}),
        ):
            v1_attention.DeepseekV41Attention.write_encoder_global_kv(
                attn, "hidden", "pos"
            )
            attn._write_global_kv.assert_not_called()

        # Valid call
        attn.kv_cache = ("nope", "rope")
        with mock.patch.object(
            v1_attention,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"k": "v"}),
        ):
            v1_attention.DeepseekV41Attention.write_encoder_global_kv(
                attn, "hidden", "pos"
            )
            attn._write_global_kv.assert_called_once_with(
                "projected", "pos", defer_index_write=False
            )

    def test_deepseek_v41_attention_prepare_tpu_and_select_indices(self):
        attn = mock.MagicMock()
        out = torch.zeros(4, 8, 128)

        # Early return when metadata is non-dict
        with mock.patch.object(
            v1_attention,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ):
            v1_attention.DeepseekV41Attention._prepare_tpu(
                attn, None, None, None, None, None, None, None, out
            )
            assert (out == 0).all()

        # Early return when kv_cache is empty
        attn.swa_cache_layer = mock.MagicMock()
        attn.swa_cache_layer.kv_cache.numel.return_value = 0
        with mock.patch.object(
            v1_attention,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"pfx": "m"}),
        ):
            v1_attention.DeepseekV41Attention._prepare_tpu(
                attn, None, None, None, None, None, None, None, out
            )
            assert (out == 0).all()

        # Full execution path
        attn.swa_cache_layer.kv_cache.numel.return_value = 100
        attn.swa_cache_layer.prefix = "swa_pfx"
        attn.swa_cache_layer.block_size = 16
        attn.n_local_heads = 4
        attn.head_dim = 128
        attn.wq_b.return_value = torch.randn(4, 4 * 128)
        attn._rotate = lambda tensor, pos: tensor
        attn.compressor = None
        attn.indexer = None
        attn.compress_ratio = 0
        attn.forward_mqa = mock.MagicMock()

        meta = {
            "swa_pfx": "swa_meta",
        }
        with (
            mock.patch.object(
                v1_attention,
                "get_forward_context",
                return_value=SimpleNamespace(attn_metadata=meta),
            ),
            mock.patch.object(cache_ops, "block_table", return_value=torch.zeros(2, 4)),
            mock.patch.object(
                cache_ops, "cache_slots", return_value=torch.tensor([0, 1, -1, 3])
            ),
            mock.patch.object(
                cache_ops,
                "token_requests",
                return_value=torch.zeros(4, dtype=torch.long),
            ),
        ):
            v1_attention.DeepseekV41Attention._prepare_tpu(
                attn,
                hidden_states=torch.randn(4, 32),
                qr=torch.randn(4, 16),
                kv=torch.randn(4, 32),
                qr_scale=None,
                kv_score=None,
                index_weights=None,
                positions=torch.arange(4),
                output=out,
            )
            attn.forward_mqa.assert_called_once()
            assert (out[2] == 0).all()

    def test_deepseek_v41_attention_forward_mqa_branches(self):
        attn = mock.MagicMock()
        out = torch.zeros(4, 8, 128)

        # 1. _ced_decoder=False calls _run_mqa
        attn._ced_decoder = False
        attn._run_mqa = mock.MagicMock(return_value=torch.ones(4, 8, 128))
        v1_attention.DeepseekV41Attention.forward_mqa(attn, "q", "kv", "pos", out)
        attn._run_mqa.assert_called_once()
        assert (out == 1).all()

        # 2. _ced_decoder=True runs CED attention
        attn._ced_decoder = True
        attn.swa_cache_layer = mock.MagicMock()
        attn.swa_cache_layer.prefix = "swa_pfx"
        attn.swa_cache_layer.block_size = 16
        attn.swa_cache_layer.native.return_value = _FakeNativeCacheBinding(
            pool=torch.zeros(16, 16), layout="layout"
        )
        attn.window_size = 8
        attn.main_cache_layer = mock.MagicMock()
        attn.main_cache_layer.kv_cache = ("nope", "rope")
        attn.main_cache_layer.cache_num_states = 16
        attn.main_cache_layer.prefix = "main_pfx"
        attn.attn_sink = torch.zeros(4)
        attn.scale = 1.0

        ctx = SimpleNamespace(
            attn_metadata={"swa_pfx": "swa_meta", "main_pfx": "main_meta"},
            additional_kwargs={"deepseek_v41_replay_starts": torch.zeros(4)},
        )

        with (
            mock.patch.object(v1_attention, "get_forward_context", return_value=ctx),
            mock.patch.object(
                cache_ops, "cache_slots", return_value=torch.tensor([0, 1, 2, 3])
            ),
            mock.patch.object(cache_ops, "write_cache"),
            mock.patch.object(cache_ops, "block_table", return_value=torch.zeros(2, 4)),
            mock.patch.object(
                cache_ops,
                "token_requests",
                return_value=torch.zeros(4, dtype=torch.long),
            ),
            mock.patch.object(cache_ops, "row_slots", return_value=torch.zeros(4, 8)),
            mock.patch(
                "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_ced_attention_op.ced_attention",
                return_value=torch.full((4, 8, 128), 2.0),
            ) as mock_ced_attn,
        ):
            # With topk_indices=None
            v1_attention.DeepseekV41Attention.forward_mqa(
                attn, "q", "kv", torch.arange(4), out, topk_indices=None
            )
            mock_ced_attn.assert_called_once()
            assert (out == 2).all()

            # With topk_indices provided
            mock_ced_attn.reset_mock()
            v1_attention.DeepseekV41Attention.forward_mqa(
                attn,
                "q",
                "kv",
                torch.arange(4),
                out,
                topk_indices=torch.zeros(4, 16, dtype=torch.int32),
            )
            mock_ced_attn.assert_called_once()

    def test_swa_cache_init(self):
        class _Base:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

        with mock.patch.object(
            v1_attention.DeepseekV4SWACache, "__init__", return_value=None
        ) as mock_base_init:
            cache = object.__new__(v1_attention.SWACache)
            v1_attention.SWACache.__init__(
                cache,
                cache_config=SimpleNamespace(block_size=16),
            )
            mock_base_init.assert_called_once()
            call_kw = mock_base_init.call_args[1]
            assert call_kw["block_size"] == 16
            assert call_kw["backend_cls"] is v1_attention.backend_ops.AttentionBackend

    def test_attention_init_and_properties_full(self):
        assert v1_attention.DeepseekV41Attention.get_padded_num_q_heads(16) == 16

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=4096,
                hf_config=SimpleNamespace(
                    qk_rope_head_dim=64,
                    to_dict=lambda: {"index_topk": 64},
                ),
            )
        )

        # 1. Invalid kv_cache_torch_dtype
        def fake_init_bad_dtype(self, *a, **k):
            self.kv_cache_torch_dtype = torch.float32

        with (
            mock.patch.object(
                v1_attention.upstream.DeepseekV4Attention,
                "__init__",
                fake_init_bad_dtype,
            ),
            pytest.raises(NotImplementedError, match="TPU V4.1 uses BF16 SWA"),
        ):
            v1_attention.DeepseekV41Attention(vllm_config)

        # 2. Invalid head_dim / qk_rope_head_dim
        def fake_init_bad_dim(self, *a, **k):
            self.kv_cache_torch_dtype = torch.bfloat16
            self.compress_ratio = 2
            self.head_dim = 256

        with (
            mock.patch.object(
                v1_attention.upstream.DeepseekV4Attention,
                "__init__",
                fake_init_bad_dim,
            ),
            pytest.raises(ValueError, match="Native CSA requires 448 NoPE and 64 RoPE"),
        ):
            v1_attention.DeepseekV41Attention(vllm_config)

        # 3. Valid init
        def fake_init_valid(self, *a, **k):
            torch.nn.Module.__init__(self)
            self.kv_cache_torch_dtype = torch.bfloat16
            self.compress_ratio = 2
            self.head_dim = 512
            self.rotary_emb = SimpleNamespace(cos_sin_cache=torch.zeros(10, 64))
            self.prefix = "layer.0.attn"

        with mock.patch.object(
            v1_attention.upstream.DeepseekV4Attention, "__init__", fake_init_valid
        ):
            attn = v1_attention.DeepseekV41Attention(vllm_config)
            assert attn.custom_prefix == "layer.0.attn"
            assert attn.mla_attn is attn
            assert attn._max_model_len == 4096
            assert attn._prepare_and_attn_fn == attn._prepare_tpu
            assert attn.attention_kind == "csa"
            attn.compress_ratio = 0
            assert attn.attention_kind == "swa"

        # 4. kv_cache setter and get_kv_cache_spec
        attn.kv_cache = ("nope_cache", "rope_cache")
        assert attn.kv_cache == ("nope_cache", "rope_cache")
        with (
            mock.patch.object(
                v1_attention.upstream.DeepseekV4Attention,
                "get_kv_cache_spec",
                return_value="raw_spec",
            ),
            mock.patch.object(
                v1_attention, "deepseek_v41_cache_spec", return_value="tpu_spec"
            ) as mock_spec,
        ):
            spec = attn.get_kv_cache_spec(vllm_config)
            assert spec == "tpu_spec"
            mock_spec.assert_called_once_with("raw_spec", vllm_config, compressed=True)

        # 5. _all_indices_fit property
        attn.compress_ratio = 2
        attn._max_model_len = 100
        attn._config = {"index_topk": 64}
        assert attn._all_indices_fit is True  # 50 <= 64
        attn._max_model_len = 200
        assert attn._all_indices_fit is False  # 100 > 64
        attn.compress_ratio = 0
        assert attn._all_indices_fit is False

    def test_prepare_tpu_and_select_paged_indices_full_branches(self):
        attn = mock.MagicMock()
        attn.head_dim = 128
        attn.n_local_heads = 4
        attn._max_model_len = 128
        attn.compress_ratio = 2
        attn.candidate_block_size = 16
        attn._config = {"index_topk": 16}
        attn._ced_global_from_encoder = False

        attn.compressor = mock.MagicMock()
        attn.compressor.return_value = torch.zeros(4, 32)
        attn._write_global_kv = mock.MagicMock(
            return_value=(torch.tensor([0, 1]), torch.zeros(2, 64))
        )

        attn.indexer = mock.MagicMock()
        attn.indexer.n_head = 2
        attn.indexer.head_dim = 64
        attn.indexer.wq_b.return_value = torch.zeros(4, 2 * 64)
        attn.indexer.k_cache.prefix = "k_pfx"
        attn.indexer.k_cache.native.return_value = _FakeNativeCacheBinding(
            pool=torch.zeros(32, 64),
            scales=torch.zeros(32),
            layout="layout",
        )
        attn.indexer.indexer_op.candidate_blocks = 8
        attn.indexer.indexer_op.candidate_write = True

        attn.candidate_block_buffer = torch.zeros(8, 8)
        attn.topk_indices_buffer = torch.zeros(8, 32, dtype=torch.int32)
        attn._select_paged_indices = mock.MagicMock(
            return_value=torch.zeros(4, 16, dtype=torch.int32)
        )
        attn.wq_b.return_value = torch.zeros(4, 4 * 128)
        attn._rotate = lambda x, pos: x
        attn.swa_cache_layer.prefix = "swa_pfx"
        attn.swa_cache_layer.block_size = 16
        attn.forward_mqa = mock.MagicMock()

        meta = {
            "swa_pfx": "swa_meta",
            "k_pfx": "k_meta",
        }
        ctx = SimpleNamespace(attn_metadata=meta)

        out = torch.zeros(4, 4, 128)
        with (
            mock.patch.object(v1_attention, "get_forward_context", return_value=ctx),
            mock.patch.object(cache_ops, "block_table", return_value=torch.zeros(2, 4)),
            mock.patch.object(
                cache_ops, "cache_slots", return_value=torch.tensor([0, 1, 2, 3])
            ),
            mock.patch.object(
                cache_ops,
                "token_requests",
                return_value=torch.zeros(4, dtype=torch.long),
            ),
            mock.patch.object(
                indexer_ops,
                "paged_indexer",
                return_value=(
                    torch.zeros(32, 64),
                    torch.zeros(32),
                    torch.zeros(4, 16, dtype=torch.int32),
                    torch.zeros(4, 8),
                ),
            ),
        ):
            # 1. _all_indices_fit = False (runs _select_paged_indices)
            attn._all_indices_fit = False
            v1_attention.DeepseekV41Attention._prepare_tpu(
                attn,
                hidden_states=torch.randn(4, 32),
                qr=torch.randn(4, 16),
                kv=torch.randn(4, 32),
                qr_scale=None,
                kv_score=torch.randn(4, 16),
                index_weights=torch.randn(4, 2),
                positions=torch.arange(4),
                output=out,
            )
            attn._write_global_kv.assert_called_once()
            attn.forward_mqa.assert_called_once()

            # 2. _all_indices_fit = True (runs logical_compressed masked_fill path)
            attn._all_indices_fit = True
            attn.forward_mqa.reset_mock()
            v1_attention.DeepseekV41Attention._prepare_tpu(
                attn,
                hidden_states=torch.randn(4, 32),
                qr=torch.randn(4, 16),
                kv=torch.randn(4, 32),
                qr_scale=None,
                kv_score=torch.randn(4, 16),
                index_weights=None,
                positions=torch.arange(4),
                output=out,
            )
            attn.forward_mqa.assert_called_once()

            # 3. Direct test of _select_paged_indices branches:
            # - has_candidates=True, candidate_write=False (candidate_filter)
            attn.indexer.indexer_op.candidate_write = False
            sel = v1_attention.DeepseekV41Attention._select_paged_indices(
                attn,
                index_q=torch.zeros(4, 64),
                block_table=torch.zeros(2, 4),
                requests=torch.zeros(4, dtype=torch.long),
                capacity=64,
                weights=torch.zeros(4, 2),
                lengths=torch.ones(4, 1, dtype=torch.long),
                swa_slots=torch.tensor([0, 1, -1, 3]),
                pending_index_write=None,
            )
            assert sel.shape == (4, 16)

            # - has_candidates=False
            attn.indexer.indexer_op.candidate_blocks = None
            sel2 = v1_attention.DeepseekV41Attention._select_paged_indices(
                attn,
                index_q=torch.zeros(4, 64),
                block_table=torch.zeros(2, 4),
                requests=torch.zeros(4, dtype=torch.long),
                capacity=64,
                weights=torch.zeros(4, 2),
                lengths=torch.ones(4, 1, dtype=torch.long),
                swa_slots=torch.tensor([0, 1, 2, 3]),
                pending_index_write=(
                    torch.zeros(2, dtype=torch.int32),
                    torch.zeros(2, 64),
                ),
            )
            assert sel2.shape == (4, 16)

            # - has_candidates=True, candidate_write=True (covers line 393)
            attn.indexer.indexer_op.candidate_blocks = 8
            attn.indexer.indexer_op.candidate_write = True
            sel3 = v1_attention.DeepseekV41Attention._select_paged_indices(
                attn,
                index_q=torch.zeros(4, 64),
                block_table=torch.zeros(2, 4),
                requests=torch.zeros(4, dtype=torch.long),
                capacity=64,
                weights=torch.zeros(4, 2),
                lengths=torch.ones(4, 1, dtype=torch.long),
                swa_slots=torch.tensor([0, 1, 2, 3]),
                pending_index_write=None,
            )
            assert sel3.shape == (4, 16)

            # 4. compress_ratio > 0 but indexer is None -> branch 290->327
            attn.indexer = None
            attn.compress_ratio = 2
            attn.forward_mqa.reset_mock()
            v1_attention.DeepseekV41Attention._prepare_tpu(
                attn,
                hidden_states=torch.randn(4, 32),
                qr=torch.randn(4, 16),
                kv=torch.randn(4, 32),
                qr_scale=None,
                kv_score=None,
                index_weights=None,
                positions=torch.arange(4),
                output=out,
            )
            attn.forward_mqa.assert_called_once()

    def test_process_weights_and_o_proj(self):
        class _DummyAttention(v1_attention.DeepseekV41Attention):
            def __init__(self):
                self.o_proj_mock = mock.MagicMock(return_value=torch.ones(4, 32))

            @property
            def o_proj_op(self):
                return self.o_proj_mock

        attn = _DummyAttention()
        attn.n_local_heads = 4
        attn.head_dim = 64
        attn._tpu_rope_cache = "rope_cache"
        attn.wo_a = mock.MagicMock()
        attn.wo_a.weight = mock.MagicMock()
        attn.wo_a.weight_scale = mock.MagicMock()
        attn.wo_a.weight.device.type = "tpu"
        attn.wo_b = mock.MagicMock(return_value=(torch.ones(4, 64),))
        attn._assert_o_proj_supported = mock.MagicMock()

        # 1. process_weights_after_loading
        with mock.patch.object(
            v1_attention.DeepseekTPUAttentionMixin,
            "process_weights_after_loading",
        ):
            v1_attention.DeepseekV41Attention.process_weights_after_loading(
                attn, torch.bfloat16
            )
            attn._assert_o_proj_supported.assert_called_once()
            # CPU device branch
            attn.wo_a.weight.device.type = "cpu"
            v1_attention.DeepseekV41Attention.process_weights_after_loading(
                attn, torch.bfloat16
            )

        # 2. _o_proj
        output = torch.zeros(4, 4, 64, dtype=torch.bfloat16)
        positions = torch.arange(4)
        res = v1_attention.DeepseekV41Attention._o_proj(attn, output, positions)
        assert res.shape == (4, 64)
        attn.o_proj_mock.assert_called_once_with(
            output,
            positions,
            "rope_cache",
            attn.wo_a.weight,
            attn.wo_a.weight_scale,
        )


# ==============================================================================
# 4. Tests for v1/model.py
# ==============================================================================


class TestV1Model:
    def test_stage_classes(self):
        owner = mock.MagicMock()
        owner._forward_stage.return_value = "stage_out"

        full = object.__new__(v1_model.DeepseekV41FullStage)
        v1_model._StageBase.__init__(full, owner=owner, vllm_config=None)
        assert full.forward(1, 2, 3, 4, 5) == "stage_out"
        owner._forward_stage.assert_called_with(1, 2, 3, 4, 5, stage="full")

        enc = object.__new__(v1_model.DeepseekV41EncoderStage)
        v1_model._StageBase.__init__(enc, owner=owner, vllm_config=None)
        assert enc.forward(1, 2, 3, 4, 5) == "stage_out"
        owner._forward_stage.assert_called_with(1, 2, 3, 4, 5, stage="encoder")

        dec = object.__new__(v1_model.DeepseekV41DecoderStage)
        v1_model._StageBase.__init__(dec, owner=owner, vllm_config=None)
        assert dec.forward(1, 2, 3, 4) == "stage_out"
        owner._forward_stage.assert_called_with(
            1, 2, None, 3, encoder_pre_mix=4, stage="decoder"
        )

    def test_model_ced_configuration_branches(self, monkeypatch):
        monkeypatch.setattr(v1_model.envs, "TPU_DEEPSEEK_V41_CED", True)

        class _MockModel(v1_model.DeepseekV41Model):
            def __init__(self):
                pass

        model = _MockModel()
        model.config = SimpleNamespace(num_hidden_layers=4)
        model.layers = [mock.MagicMock() for _ in range(4)]
        for layer in model.layers:
            layer.attn = mock.MagicMock()
            layer.attn.compressor = None
            layer.attn.window_size = 128
            layer.engram = None
        model.layers[2].attn.compressor = "compressor"
        model.end_layer = 4
        model.use_sequence_parallel = False

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=4096),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            compilation_config=SimpleNamespace(compile_sizes=[16, 32]),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        )

        with mock.patch.object(v1_model.logger, "warning") as mock_warn:
            model._ced_enabled = bool(v1_model.envs.TPU_DEEPSEEK_V41_CED)
            if model._ced_enabled and vllm_config.cache_config.enable_prefix_caching:
                v1_model.logger.warning("prefix caching disabled")
                model._ced_enabled = False
            assert model._ced_enabled is False
            mock_warn.assert_called_once()

    def test_model_forward_ced_and_incomplete_prefill(self, monkeypatch):
        model = mock.MagicMock()
        model._ced_enabled = True
        model._ced_window = 64
        model._ced_compile_sizes = (16, 32)
        model.config = SimpleNamespace(hidden_size=128)
        model.encoder_stage = mock.MagicMock(
            return_value=[(torch.zeros(4, 128), torch.zeros(4, 128))]
        )
        model.decoder_stage = mock.MagicMock(return_value=[torch.ones(4, 128)])

        # 0. CED disabled branch
        model._ced_enabled = False
        model.full_stage = mock.MagicMock(return_value="full_stage_out")
        assert (
            v1_model.DeepseekV41Model.forward(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
            )
            == "full_stage_out"
        )
        model._ced_enabled = True

        # 1. Incomplete prefill (plan.indices is empty)
        empty_plan = v1_replay_plan.DecoderReplayPlan((), (0,), (0,), 0)
        with (
            mock.patch.object(
                v1_model.upstream,
                "get_forward_context",
                return_value=_FakeFwdContext(attn_metadata=_FakeMetadata()),
            ),
            mock.patch.object(v1_model, "plan_decoder_replay", return_value=empty_plan),
        ):
            res = v1_model.DeepseekV41Model.forward(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
            )
            assert (res == 0).all()
            assert res.shape == (4, 128)

        # 2. Valid plan with index_copy scatter
        valid_plan = v1_replay_plan.DecoderReplayPlan((0, 1, 2, 3), (0, 4), (0,), 4)
        with (
            mock.patch.object(
                v1_model.upstream,
                "get_forward_context",
                return_value=_FakeFwdContext(
                    attn_metadata=_FakeMetadata(), additional_kwargs={}
                ),
            ),
            mock.patch.object(v1_model, "plan_decoder_replay", return_value=valid_plan),
        ):
            res2 = v1_model.DeepseekV41Model.forward(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
            )
            assert (res2 == 1).all()
            assert res2.shape == (4, 128)

        # 3. Plan is None with first is not None (branches 168->170, 234->236)
        model.encoder_stage.return_value = (
            torch.zeros(4, 128),
            torch.zeros(4, 128),
        )
        model.decoder_stage.return_value = torch.ones(4, 128)
        with (
            mock.patch.object(
                v1_model.upstream,
                "get_forward_context",
                return_value=_FakeFwdContext(
                    attn_metadata=_FakeMetadata(), additional_kwargs={}
                ),
            ),
            mock.patch.object(v1_model, "plan_decoder_replay", return_value=None),
        ):
            res3 = v1_model.DeepseekV41Model.forward(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
            )
            assert (res3 == 1).all()

        # 4. Plan is None with first is None (non-dict metadata)
        with (
            mock.patch.object(
                v1_model.upstream,
                "get_forward_context",
                return_value=_FakeFwdContext(attn_metadata=None, additional_kwargs={}),
            ),
        ):
            res4 = v1_model.DeepseekV41Model.forward(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
            )
            assert (res4 == 1).all()

    def test_model_forward_stage_branches(self):
        model = mock.MagicMock()
        model.start_layer = 0
        model.end_layer = 4
        model._ced_boundary = 2
        model.use_native_mega_moe = True
        model.engram_hash = None
        model.use_sequence_parallel = False
        model.aux_hidden_state_layers = []
        model._mtp_hidden_buffer = None
        model.norm = lambda x: x

        # Layer mock
        layer = mock.MagicMock()
        layer.return_value = (
            torch.ones(4, 128),
            None,
            None,
            None,
            torch.ones(4, 128),
            None,
        )
        model.layers = [layer] * 4

        with (
            mock.patch("vllm.models.deepseek_v41.nvidia.model.get_pp_group") as mock_pp,
            mock.patch.object(v1_model, "post", return_value=torch.ones(4, 128)),
            mock.patch.object(v1_model, "collapse", return_value=torch.ones(4, 128)),
        ):
            mock_pp.return_value.is_first_rank = True
            mock_pp.return_value.is_last_rank = True

            # 1. stage="decoder"
            res_dec = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="decoder",
            )
            assert res_dec.shape == (4, 128)

            # 2. stage="encoder" publishes global KV
            decoder_layer = mock.MagicMock()
            decoder_layer.attn_norm = lambda x: x
            decoder_layer.attn.write_encoder_global_kv = mock.MagicMock()
            model.layers[2] = decoder_layer

            hidden, pre_mix = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="encoder",
            )
            assert hidden.shape == (4, 128)
            assert pre_mix.shape == (4, 128)
            decoder_layer.attn.write_encoder_global_kv.assert_called_once()

    def test_model_finalize_mhc_broadcast_weights(self):
        class _DummyModel(v1_model.DeepseekV41Model):
            def __init__(self):
                pass

        model = _DummyModel()
        layer0 = mock.MagicMock()
        layer0.hc_attn_fn.dtype = torch.bfloat16
        model.layers = [layer0]
        model.start_layer = 0
        model.end_layer = 1

        with mock.patch.object(v1_model, "pack_mix_weights") as mock_pack:
            v1_model.DeepseekV41Model.finalize_mhc_broadcast_weights(model)
            mock_pack.assert_not_called()

            # float32 triggers pack_mix_weights
            layer0.hc_attn_fn.dtype = torch.float32
            with mock.patch.object(
                v1_model.upstream.DeepseekV4Model,
                "finalize_mhc_broadcast_weights",
            ):
                v1_model.DeepseekV41Model.finalize_mhc_broadcast_weights(model)
                mock_pack.assert_called_once()

    def test_causal_lm_init_and_forward(self):
        class _ParallelConfig:
            pipeline_parallel_size = 1
            data_parallel_size = 1

        vllm_config = SimpleNamespace(
            parallel_config=_ParallelConfig(),
            speculative_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
        )

        # Capability errors: PP != 1
        vllm_config.parallel_config.pipeline_parallel_size = 2
        with pytest.raises(
            NotImplementedError, match="V4.1 TPU currently requires PP1 and DP1"
        ):
            v1_model.DeepseekV41ForCausalLM(vllm_config=vllm_config)

        # Speculative error
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.speculative_config = "spec"
        with pytest.raises(
            NotImplementedError, match="synchronous scheduling without speculation"
        ):
            v1_model.DeepseekV41ForCausalLM(vllm_config=vllm_config)

        # forward ensures engram cache
        class _DummyLM(v1_model.DeepseekV41ForCausalLM):
            def __init__(self):
                pass

        lm = _DummyLM()
        lm.model = mock.MagicMock()
        lm.model.engram_hash = mock.MagicMock()
        with mock.patch.object(
            v1_model.upstream.DeepseekV41LLMForCausalLM,
            "forward",
            return_value="lm_out",
        ) as mock_super:
            out = v1_model.DeepseekV41ForCausalLM.forward(lm, "arg")
            assert out == "lm_out"
            lm.model.engram_hash.ensure_cache.assert_called_once()
            mock_super.assert_called_once()

            # engram_hash is None branch
            lm.model.engram_hash = None
            out2 = v1_model.DeepseekV41ForCausalLM.forward(lm, "arg2")
            assert out2 == "lm_out"

    def test_forward_stage_pp_and_embed_branches(self):
        model = mock.MagicMock()
        model.start_layer = 0
        model.end_layer = 1
        model._ced_boundary = 1
        model.use_native_mega_moe = True
        model.engram_hash = None
        model.use_sequence_parallel = False
        model.aux_hidden_state_layers = []
        model._mtp_hidden_buffer = None
        model.norm = lambda x: x
        model.embed_input_ids = mock.MagicMock(return_value=torch.ones(4, 128))

        layer = mock.MagicMock()
        layer.return_value = (
            torch.ones(4, 128),
            None,
            None,
            None,
            torch.ones(4, 128),
            None,
        )
        model.layers = [layer]

        # 1. First rank, inputs_embeds is None -> calls embed_input_ids
        with (
            mock.patch.object(v1_model.upstream, "get_pp_group") as mock_pp,
            mock.patch.object(v1_model, "post", return_value=torch.ones(4, 128)),
            mock.patch.object(v1_model, "collapse", return_value=torch.ones(4, 128)),
        ):
            mock_pp.return_value.is_first_rank = True
            mock_pp.return_value.is_last_rank = True

            out = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=None,
                stage="full",
            )
            assert out.shape == (4, 128)
            model.embed_input_ids.assert_called_once()

            # 2. Not first rank, not last rank -> returns IntermediateTensors
            mock_pp.return_value.is_first_rank = False
            mock_pp.return_value.is_last_rank = False

            interm_in = {
                "hidden_states": torch.ones(4, 128),
                "pre_mix": torch.ones(4, 128),
            }
            res_interm = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=interm_in,
                inputs_embeds=None,
                stage="full",
            )
            assert isinstance(res_interm, v1_model.upstream.IntermediateTensors)

    def test_forward_stage_engram_hash_branches(self):
        model = mock.MagicMock()
        model.start_layer = 0
        model.end_layer = 1
        model._ced_boundary = 1
        model.use_native_mega_moe = False
        model.use_sequence_parallel = False
        model.aux_hidden_state_layers = []
        model._mtp_hidden_buffer = None
        model.norm = lambda x: x
        model.engram_dp_shared_memory = False
        model.engram_swa_prefix = "swa"

        layer = mock.MagicMock()
        layer.return_value = (
            torch.ones(4, 128),
            None,
            None,
            None,
            torch.ones(4, 128),
            None,
        )
        layer.engram = mock.MagicMock()
        layer.engram.layer_hash_index = 0
        layer2 = mock.MagicMock()
        layer2.return_value = layer.return_value
        layer2.engram = None
        model.layers = [layer, layer2]
        model.end_layer = 2

        model.engram_hash = mock.MagicMock()
        model.engram_hash.ensure_cache.return_value = True
        model.engram_hash.use_slot_cache = False
        model.engram_hash.lookback_depth = 4
        model.engram_hash.swa_cache_module.block_size = 16
        model.engram_hash.return_value = torch.zeros(4, 2, dtype=torch.int64)

        fwd_ctx = SimpleNamespace(
            attn_metadata={
                "swa": SimpleNamespace(
                    seq_lens=torch.tensor([4]),
                    query_start_loc=torch.tensor([0, 4]),
                )
            }
        )

        with (
            mock.patch.object(v1_model.upstream, "get_pp_group") as mock_pp,
            mock.patch.object(
                v1_model.upstream, "is_forward_context_available", return_value=True
            ),
            mock.patch.object(
                v1_model.upstream, "get_forward_context", return_value=fwd_ctx
            ),
            mock.patch.object(
                v1_model.upstream,
                "image_sentinel_mask",
                return_value=torch.zeros(4, dtype=torch.bool),
            ),
            mock.patch.object(
                v1_model.upstream,
                "gather_engram_hashes",
                return_value=torch.zeros(4, 2, dtype=torch.int64),
            ),
            mock.patch.object(v1_model, "post", return_value=torch.ones(4, 128)),
            mock.patch.object(v1_model, "collapse", return_value=torch.ones(4, 128)),
            mock.patch.object(
                cache_ops, "cache_slots", return_value=torch.zeros(4, dtype=torch.int32)
            ),
            mock.patch.object(cache_ops, "block_table", return_value=torch.zeros(2, 4)),
        ):
            mock_pp.return_value.is_first_rank = True
            mock_pp.return_value.is_last_rank = True

            # 1. lookback_token_ids is None and not use_slot_cache
            with pytest.raises(
                NotImplementedError, match="engram needs `lookback_token_ids`"
            ):
                v1_model.DeepseekV41Model._forward_stage(
                    model,
                    input_ids=torch.zeros(4, dtype=torch.long),
                    positions=torch.arange(4),
                    intermediate_tensors=None,
                    inputs_embeds=torch.zeros(4, 128),
                    lookback_token_ids=None,
                    stage="full",
                )

            # 2. use_slot_cache = True -> runs hash and gathers
            model.engram_hash.use_slot_cache = True
            v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                lookback_token_ids=None,
                stage="full",
            )
            layer.engram.prepare_embeddings.assert_called_once()

            # 3. attn_metadata as list -> dbo_current_ubatch_id
            fwd_ctx_list = SimpleNamespace(attn_metadata=[fwd_ctx.attn_metadata])
            with (
                mock.patch.object(
                    v1_model.upstream,
                    "get_forward_context",
                    return_value=fwd_ctx_list,
                ),
                mock.patch.object(
                    v1_model.upstream, "dbo_current_ubatch_id", return_value=0
                ),
            ):
                layer.engram.prepare_embeddings.reset_mock()
                v1_model.DeepseekV41Model._forward_stage(
                    model,
                    input_ids=torch.zeros(4, dtype=torch.long),
                    positions=torch.arange(4),
                    intermediate_tensors=None,
                    inputs_embeds=torch.zeros(4, 128),
                    lookback_token_ids=None,
                    stage="full",
                )
                layer.engram.prepare_embeddings.assert_called_once()

            # 4. ensure_cache() False and dp_size > 1 -> dummy_hashes
            model.engram_hash.ensure_cache.return_value = False
            model.engram_hash.dummy_hashes.return_value = (
                torch.zeros(4, 2, dtype=torch.int64),
                torch.ones(4, dtype=torch.bool),
            )
            with mock.patch.object(
                v1_model.upstream, "get_engram_dp_size", return_value=2
            ):
                layer.engram.prepare_embeddings.reset_mock()
                v1_model.DeepseekV41Model._forward_stage(
                    model,
                    input_ids=torch.zeros(4, dtype=torch.long),
                    positions=torch.arange(4),
                    intermediate_tensors=None,
                    inputs_embeds=torch.zeros(4, 128),
                    lookback_token_ids=None,
                    stage="full",
                )
                model.engram_hash.dummy_hashes.assert_called_once()

            # 5. lookback_token_ids is provided -> branch 293->303
            model.engram_hash.ensure_cache.return_value = True
            v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                lookback_token_ids=torch.zeros(1, 4, dtype=torch.long),
                stage="full",
            )

            # 6. ensure_cache() returns False and get_engram_dp_size() == 1
            # -> takes branch 317->321 (engram_hashes is None) and branch 321->334
            model.engram_hash.ensure_cache.return_value = False
            with mock.patch.object(
                v1_model.upstream, "get_engram_dp_size", return_value=1
            ):
                layer.engram.prepare_embeddings.reset_mock()
                v1_model.DeepseekV41Model._forward_stage(
                    model,
                    input_ids=torch.zeros(4, dtype=torch.long),
                    positions=torch.arange(4),
                    intermediate_tensors=None,
                    inputs_embeds=torch.zeros(4, 128),
                    lookback_token_ids=None,
                    stage="full",
                )
                layer.engram.prepare_embeddings.assert_not_called()

    def test_forward_stage_sequence_parallel_and_aux_hidden(self):
        model = mock.MagicMock()
        model.start_layer = 0
        model.end_layer = 1
        model._ced_boundary = 1
        model.use_native_mega_moe = False
        model.engram_hash = None
        model.use_sequence_parallel = True
        model.aux_hidden_state_layers = [0, 1]
        model._mtp_hidden_buffer = torch.zeros(8, 128)
        model.norm = lambda x: x

        layer = mock.MagicMock()
        layer.return_value = (
            torch.ones(4, 128),
            None,
            None,
            None,
            torch.ones(4, 128),
            torch.ones(4, 128),  # previous_aux
        )
        model.layers = [layer]

        fwd_ctx = SimpleNamespace(is_padding=torch.zeros(4, dtype=torch.bool))

        with (
            mock.patch.object(v1_model.upstream, "get_pp_group") as mock_pp,
            mock.patch.object(
                v1_model.upstream, "is_forward_context_available", return_value=True
            ),
            mock.patch.object(
                v1_model.upstream, "get_forward_context", return_value=fwd_ctx
            ),
            mock.patch.object(v1_model.upstream.envs, "VLLM_MOE_SKIP_PADDING", True),
            mock.patch.object(
                v1_model.upstream,
                "sp_padding_mask",
                return_value=torch.zeros(4, dtype=torch.bool),
            ),
            mock.patch.object(v1_model.upstream, "sp_shard", side_effect=lambda x: x),
            mock.patch.object(
                v1_model.upstream, "sp_all_gather", side_effect=lambda x: x
            ),
            mock.patch.object(v1_model, "post", return_value=torch.ones(4, 128)),
            mock.patch.object(v1_model, "collapse", return_value=torch.ones(4, 128)),
        ):
            mock_pp.return_value.is_first_rank = True
            mock_pp.return_value.is_last_rank = True

            # 1. With _mtp_hidden_buffer and aux_hidden_state_layers
            res, aux = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="full",
            )
            assert res.shape == (4, 128)
            assert len(aux) > 0

            # 2. Without _mtp_hidden_buffer
            model._mtp_hidden_buffer = None
            model.aux_hidden_state_layers = []
            res2 = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="full",
            )
            assert res2.shape == (4, 128)

            # 3. SP=True with VLLM_MOE_SKIP_PADDING=False -> branch 336->344
            with mock.patch.object(
                v1_model.upstream.envs, "VLLM_MOE_SKIP_PADDING", False
            ):
                v1_model.DeepseekV41Model._forward_stage(
                    model,
                    input_ids=torch.zeros(4, dtype=torch.long),
                    positions=torch.arange(4),
                    intermediate_tensors=None,
                    inputs_embeds=torch.zeros(4, 128),
                    stage="full",
                )

            # 4. SP=False with previous_aux, end_layer aux, and mtp buffer
            # -> covers branches 373->377, 383->385, and 410->413
            model.use_sequence_parallel = False
            model.aux_hidden_state_layers = [0, 1]
            model._mtp_hidden_buffer = torch.zeros(8, 128)
            res4, aux4 = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="full",
            )
            assert res4.shape == (4, 128)
            assert len(aux4) == 2

    def test_forward_stage_layer_none_branch(self):
        # Covers branch 378->387 (if layer is not None evaluating to False)
        model = mock.MagicMock()
        model.start_layer = 0
        model.end_layer = 1
        model._ced_boundary = 1
        model.use_native_mega_moe = False
        model.engram_hash = None
        model.use_sequence_parallel = False
        model.aux_hidden_state_layers = []
        model._mtp_hidden_buffer = None
        model.norm = lambda x: x

        def mutating_layer(*args, **kwargs):
            import ctypes
            import inspect

            frame = inspect.currentframe().f_back
            frame.f_locals["layer"] = None
            ctypes.pythonapi.PyFrame_LocalsToFast(
                ctypes.py_object(frame), ctypes.c_int(0)
            )
            return (
                torch.ones(4, 128),
                None,
                None,
                None,
                torch.ones(4, 128),
                None,
            )

        model.layers = [mutating_layer]

        with (
            mock.patch.object(v1_model.upstream, "get_pp_group") as mock_pp,
            mock.patch.object(v1_model, "collapse", return_value=torch.ones(4, 128)),
        ):
            mock_pp.return_value.is_first_rank = True
            mock_pp.return_value.is_last_rank = True

            out = v1_model.DeepseekV41Model._forward_stage(
                model,
                input_ids=torch.zeros(4, dtype=torch.long),
                positions=torch.arange(4),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(4, 128),
                stage="full",
            )
            assert out.shape == (4, 128)

    def test_deepseek_v41_model_init_branches(self, monkeypatch):
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=4096),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            compilation_config=SimpleNamespace(compile_sizes=[16, 32]),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        )

        def make_base_model():
            m = object.__new__(v1_model.DeepseekV41Model)
            m.config = SimpleNamespace(num_hidden_layers=4)
            m.layers = [mock.MagicMock() for _ in range(4)]
            for lyr in m.layers:
                lyr.attn = mock.MagicMock()
                lyr.attn.compressor = None
                lyr.attn.window_size = 64
                lyr.attn.rotary_emb.cos_sin_cache = torch.zeros(10, 64)
                lyr.engram = None
            m.layers[2].attn.compressor = "compressor"
            m.end_layer = 4
            m.use_sequence_parallel = False
            return m

        with (
            mock.patch.object(cache_ops, "initialize_cache_ops"),
            mock.patch.object(v1_model, "initialize_delayed_mhc"),
            mock.patch.object(v1_model, "initialize_mhc_post"),
            mock.patch.object(compressor_ops, "initialize_compressor_ops"),
            mock.patch.object(v1_model.ced_ops, "initialize_ced_attention"),
            mock.patch.object(indexer_ops, "initialize_indexer"),
            mock.patch.object(v1_model, "compact_rope_caches"),
            mock.patch.object(
                v1_model.upstream.DeepseekV4Model, "__init__", return_value=None
            ),
            mock.patch.object(
                v1_model, "DeepseekV41EncoderStage", return_value=mock.MagicMock()
            ) as mock_enc_cls,
            mock.patch.object(
                v1_model, "DeepseekV41DecoderStage", return_value=mock.MagicMock()
            ) as mock_dec_cls,
            mock.patch.object(
                v1_model, "DeepseekV41FullStage", return_value=mock.MagicMock()
            ) as mock_full_cls,
        ):
            # 1. CED enabled: use_sequence_parallel=True -> raises NotImplementedError
            monkeypatch.setattr(v1_model.envs, "TPU_DEEPSEEK_V41_CED", True)
            m1 = make_base_model()
            m1.use_sequence_parallel = True
            with pytest.raises(NotImplementedError, match="unsharded token rows"):
                v1_model.DeepseekV41Model.__init__(m1, vllm_config=vllm_config)

            # 2. CED enabled: sources != [ced_boundary] -> raises NotImplementedError
            m2 = make_base_model()
            m2.layers[3].attn.compressor = "compressor"  # extra source
            with pytest.raises(NotImplementedError, match="one decoder KV source"):
                v1_model.DeepseekV41Model.__init__(m2, vllm_config=vllm_config)

            # 3. CED enabled: engram in decoder -> raises NotImplementedError
            m3 = make_base_model()
            m3.layers[2].engram = "engram"
            with pytest.raises(
                NotImplementedError, match="Engram layers in the encoder"
            ):
                v1_model.DeepseekV41Model.__init__(m3, vllm_config=vllm_config)

            # 4. Valid CED init
            m4 = make_base_model()
            v1_model.DeepseekV41Model.__init__(m4, vllm_config=vllm_config)
            assert m4.encoder_stage is mock_enc_cls.return_value
            assert m4.decoder_stage is mock_dec_cls.return_value

            # 5. CED disabled init
            monkeypatch.setattr(v1_model.envs, "TPU_DEEPSEEK_V41_CED", False)
            m5 = make_base_model()
            v1_model.DeepseekV41Model.__init__(m5, vllm_config=vllm_config)
            assert m5.full_stage is mock_full_cls.return_value

            # 6. CED with prefix caching -> warning and fallback (lines 462-466)
            monkeypatch.setattr(v1_model.envs, "TPU_DEEPSEEK_V41_CED", True)
            vllm_config_prefix = SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=4096),
                cache_config=SimpleNamespace(enable_prefix_caching=True),
                compilation_config=SimpleNamespace(compile_sizes=[16, 32]),
                parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
            )
            m6 = make_base_model()
            v1_model.DeepseekV41Model.__init__(m6, vllm_config=vllm_config_prefix)
            assert m6._ced_enabled is False
            assert m6.full_stage is mock_full_cls.return_value

    def test_causal_lm_init_success(self):
        class _ParallelConfig:
            pipeline_parallel_size = 1
            data_parallel_size = 1

        vllm_config = SimpleNamespace(
            parallel_config=_ParallelConfig(),
            speculative_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
        )

        with (
            mock.patch.object(
                v1_model.upstream.DeepseekV41LLMForCausalLM,
                "__init__",
                return_value=None,
            ),
            mock.patch.object(
                v1_model.upstream,
                "_make_deepseek_v4_weights_mapper",
                return_value="mapper",
            ) as mock_mapper,
        ):
            lm = object.__new__(v1_model.DeepseekV41ForCausalLM)
            lm.config = SimpleNamespace(expert_dtype="fp4")
            v1_model.DeepseekV41ForCausalLM.__init__(lm, vllm_config=vllm_config)
            assert lm.hf_to_vllm_mapper == "mapper"
            mock_mapper.assert_called_once()


# ==============================================================================
# 5. Tests for ..2.2.3 diffs (attention_kernels, bench_decode_indexer_split)
# ==============================================================================


class TestDecodeIndexerCandidateContext:
    def test_bench_decode_indexer_split_expand_tokens(self):
        rng = np.random.default_rng(123)
        seqs, heads, dim = 2, 4, 128
        full = {
            "q": rng.standard_normal((seqs, heads, dim)),
            "weights": rng.standard_normal((seqs, heads)),
            "candidates": rng.integers(0, 100, (seqs, 16)),
            "lengths": np.array([20, 30], dtype=np.int32),
        }

        expanded = bench_decode_indexer_split.expand_tokens(full, tokens=3)
        assert expanded["q"].shape == (seqs * 3, heads, dim)
        assert expanded["weights"].shape == (seqs * 3, heads)
        assert expanded["candidates"].shape == (seqs * 3, 16)
        np.testing.assert_array_equal(
            expanded["requests"], np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            expanded["lengths"], np.array([18, 19, 20, 28, 29, 30], dtype=np.int32)
        )

    def test_bench_decode_indexer_split_worker_and_parser(self, monkeypatch):
        # 1. Main and parser coverage
        monkeypatch.setattr(
            sys, "argv", ["bench_decode_indexer_split.py", "--tokens-per-seq", "4"]
        )
        mock_run_main = mock.MagicMock()
        monkeypatch.setattr(
            bench_decode_indexer_split.common, "run_main", mock_run_main
        )
        bench_decode_indexer_split.main()
        mock_run_main.assert_called_once()
        _, parser, _ = mock_run_main.call_args[0]
        args = parser.parse_args(["--tokens-per-seq", "4"])
        assert args.tokens_per_seq == 4

        # 2. Worker diff coverage: args.tokens_per_seq > 1 and <= 1
        class _StopWorker(Exception):
            pass

        monkeypatch.setattr(
            vllm.distributed,
            "get_tp_group",
            lambda: SimpleNamespace(rank_in_group=0),
        )
        monkeypatch.setattr(torch, "from_numpy", mock.Mock(side_effect=_StopWorker))

        ctx = SimpleNamespace(tp_size=1, rank=0)
        args_gt1 = SimpleNamespace(
            seed=42,
            num_seqs=2,
            kv_len=16,
            pool_rows=32,
            dim=128,
            layout_page=16,
            tokens_per_seq=2,
            subop="candidate_write",
            page_records=16,
            split="balanced",
            legacy=False,
            verify=False,
            stage="both",
            time_steps=None,
            keys_per_iter=None,
            max_tile=None,
        )

        with pytest.raises(_StopWorker):
            bench_decode_indexer_split.worker(ctx, args_gt1)

        args_eq1 = SimpleNamespace(
            seed=42,
            num_seqs=2,
            kv_len=16,
            pool_rows=32,
            dim=128,
            layout_page=16,
            tokens_per_seq=1,
            subop="candidate_write",
            page_records=16,
            split="balanced",
            legacy=False,
            verify=False,
            stage="both",
            time_steps=None,
            keys_per_iter=None,
            max_tile=None,
        )
        with pytest.raises(_StopWorker):
            bench_decode_indexer_split.worker(ctx, args_eq1)

    def test_paged_index_scores_jax_pages_branches(self):
        """Exercises lines 1372-1376 and 1396-1401 in v2/attention_kernels.py."""
        rng = np.random.default_rng(77)
        tokens, heads, dim, width = 4, 4, 128, 512
        pool = jnp.asarray(
            rng.integers(-(2**31), 2**31, (256, 128)), jnp.int32
        ) & jnp.int32(0x3FFF3FFF)
        q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
        weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
        lengths = jnp.array([200, 300, 400, 500], dtype=jnp.int32)
        records = jnp.asarray(rng.integers(0, 100, (2, width)), dtype=jnp.int32)
        requests = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        # 1. pages = True (page_records=16, tile multiple of 16)
        scores_pages = attention_kernels.paged_index_scores_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            requests=requests,
            page_records=16,
        )
        assert scores_pages.shape == (tokens, width)

        # 2. pages = False (page_records=None)
        scores_nopages = attention_kernels.paged_index_scores_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            requests=requests,
            page_records=None,
        )
        assert scores_nopages.shape == (tokens, width)

    def test_candidate_context_sparsecore_unordered_top_k(self, monkeypatch):
        """Exercises lines 1730-1759 in v2/attention_kernels.py."""
        rng = np.random.default_rng(88)
        tokens, heads, dim, width, block_size = 4, 4, 128, 4096, 16
        topk = 64
        pool = jnp.asarray(
            rng.integers(-(2**31), 2**31, (256, 128)), jnp.int32
        ) & jnp.int32(0x3FFF3FFF)
        q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
        weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
        lengths = jnp.array([3500, 3600, 3700, 3800], dtype=jnp.int32)
        records = jnp.asarray(rng.integers(0, 200, (4, width)), dtype=jnp.int32)
        candidates = jnp.asarray(
            rng.integers(0, width // block_size, (tokens, 128)), dtype=jnp.int32
        )
        requests = jnp.arange(tokens, dtype=jnp.int32)

        def _mock_sc_topk(scores, k, lengths):
            return jnp.full((scores.shape[0], k), 0, dtype=jnp.int32)

        monkeypatch.setattr(attention_kernels, "_sparsecore_top_k", _mock_sc_topk)

        selected_uo, cand_uo = attention_kernels.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
            unordered=True,
            page_records=16,
        )
        assert selected_uo.shape == (tokens, topk)
        assert cand_uo is candidates

        selected_ord, cand_ord = attention_kernels.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
            unordered=False,
            page_records=16,
        )
        assert selected_ord.shape == (tokens, topk)
        assert cand_ord is candidates
