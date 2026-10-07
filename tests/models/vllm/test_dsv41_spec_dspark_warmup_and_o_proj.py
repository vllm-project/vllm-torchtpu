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

"""100% full-module line and branch coverage tests for

src/vllm_torchtpu/spec_decode/dspark.py,
src/vllm_torchtpu/kernels/deepseek_v4/o_projection.py,
and src/vllm_torchtpu/runner/tpu_runner.py spec-warmup diffs (eab4fc815) on CPU.
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec

from vllm_torchtpu.kernels.deepseek_v4 import o_projection as op
from vllm_torchtpu.kernels.deepseek_v4.rope import LANE
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.spec_decode.dspark import DSparkProposer


def _norm_idx(idx):
    if isinstance(idx, tuple):
        return tuple(_norm_idx(i) for i in idx)
    if hasattr(idx, "start") and hasattr(idx, "size"):
        stride = getattr(idx, "stride", 1)
        stride = 1 if stride is None else int(stride)
        return slice(int(idx.start), int(idx.start + idx.size), stride)
    return idx


class MockArrayRef:
    """Mock HBM / VMEM buffer reference for simulating Pallas kernel steps on CPU."""

    def __init__(self, arr: np.ndarray | jax.Array):
        self.arr = np.asarray(arr).copy()
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def __getitem__(self, idx):
        try:
            return self.arr[_norm_idx(idx)]
        except Exception:
            return self.arr

    def __setitem__(self, idx, val):
        with contextlib.suppress(Exception):
            self.arr[_norm_idx(idx)] = np.asarray(val)


# ==============================================================================
# 1. DSparkProposer Full-Module Coverage Tests
# ==============================================================================


class TestDSparkProposerFull:
    """Tests for 100% full-module line and branch coverage of spec_decode/dspark.py."""

    def _make_valid_vllm_config(
        self,
        *,
        has_to_dict: bool = True,
        sample_from_anchor: bool | None = None,
        dspark_block_size: int | None = 5,
        block_size: int | None = None,
        num_speculative_tokens: int = 5,
        draft_sample_method: str = "greedy",
        enable_adaptive_verification: bool = False,
        dspark_draft_topk: int | None = None,
        target_layer_ids: list[int] | None = None,
    ):
        if target_layer_ids is None:
            target_layer_ids = [10, 20]

        hf_dict = {}
        if sample_from_anchor is not None:
            hf_dict["sample_from_anchor"] = sample_from_anchor
        if dspark_block_size is not None:
            hf_dict["dspark_block_size"] = dspark_block_size
        if block_size is not None:
            hf_dict["block_size"] = block_size

        if has_to_dict:
            hf_config = SimpleNamespace(to_dict=lambda: dict(hf_dict))
        else:
            hf_config = SimpleNamespace()
            for k, v in hf_dict.items():
                setattr(hf_config, k, v)

        draft_model_config = SimpleNamespace(
            hf_config=hf_config,
            model="test-draft-model",
        )
        speculative_config = SimpleNamespace(
            draft_model_config=draft_model_config,
            num_speculative_tokens=num_speculative_tokens,
            draft_sample_method=draft_sample_method,
            enable_adaptive_verification=enable_adaptive_verification,
            dspark_draft_topk=dspark_draft_topk,
            method="dspark",
        )

        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.speculative_config = speculative_config
        vllm_config.model_config = SimpleNamespace(max_model_len=4096)

        return vllm_config, target_layer_ids

    def test_init_success_variants(self):
        # 1. Default sample_from_anchor, dspark_block_size=5, has_to_dict=True
        cfg1, target_layers = self._make_valid_vllm_config(
            has_to_dict=True,
            sample_from_anchor=None,
            dspark_block_size=5,
        )
        with patch.object(
            DSparkProposer.__bases__[0],
            "__init__",
            lambda self, runner, v_cfg: (
                setattr(
                    self,
                    "dflash_config",
                    {"target_layer_ids": target_layers},
                )
                or setattr(self, "speculative_config", v_cfg.speculative_config)
            ),
        ):
            p1 = DSparkProposer(MagicMock(), cfg1)
            assert p1.sample_from_anchor is True
            assert p1._query_block_size(5) == 5
            assert p1._force_share_target_embeddings is False

        # 2. Explicit sample_from_anchor=False, block_size=6, has_to_dict=True
        cfg2, target_layers2 = self._make_valid_vllm_config(
            has_to_dict=True,
            sample_from_anchor=False,
            dspark_block_size=None,
            block_size=6,
            num_speculative_tokens=6,
        )
        with patch.object(
            DSparkProposer.__bases__[0],
            "__init__",
            lambda self, runner, v_cfg: (
                setattr(
                    self,
                    "dflash_config",
                    {"target_layer_ids": target_layers2},
                )
                or setattr(self, "speculative_config", v_cfg.speculative_config)
            ),
        ):
            p2 = DSparkProposer(MagicMock(), cfg2)
            assert p2.sample_from_anchor is False
            assert p2._query_block_size(6) == 7

        # 3. Neither dspark_block_size nor block_size specified (block is None)
        cfg3, target_layers3 = self._make_valid_vllm_config(
            has_to_dict=True,
            sample_from_anchor=True,
            dspark_block_size=None,
            block_size=None,
            num_speculative_tokens=5,
        )
        with patch.object(
            DSparkProposer.__bases__[0],
            "__init__",
            lambda self, runner, v_cfg: (
                setattr(
                    self,
                    "dflash_config",
                    {"target_layer_ids": target_layers3},
                )
                or setattr(self, "speculative_config", v_cfg.speculative_config)
            ),
        ):
            p3 = DSparkProposer(MagicMock(), cfg3)
            assert p3.sample_from_anchor is True
            assert p3._query_block_size(5) == 5

        # 4. has_to_dict=False (covers hasattr(hf_config, "to_dict") False branch)
        cfg4, target_layers4 = self._make_valid_vllm_config(
            has_to_dict=False,
            sample_from_anchor=None,
            dspark_block_size=5,
            num_speculative_tokens=5,
        )
        with patch.object(
            DSparkProposer.__bases__[0],
            "__init__",
            lambda self, runner, v_cfg: (
                setattr(
                    self,
                    "dflash_config",
                    {"target_layer_ids": target_layers4},
                )
                or setattr(self, "speculative_config", v_cfg.speculative_config)
            ),
        ):
            p4 = DSparkProposer(MagicMock(), cfg4)
            assert p4.sample_from_anchor is True
            assert p4._query_block_size(5) == 5

    def test_init_missing_target_layers_error(self):
        cfg, _ = self._make_valid_vllm_config()
        with (
            patch.object(
                DSparkProposer.__bases__[0],
                "__init__",
                lambda self, runner, v_cfg: (
                    setattr(self, "dflash_config", {})
                    or setattr(self, "speculative_config", v_cfg.speculative_config)
                ),
            ),
            pytest.raises(ValueError, match="DSpark requires target layer ids"),
        ):
            DSparkProposer(MagicMock(), cfg)

    def test_init_block_size_mismatch_error(self):
        cfg, target_layers = self._make_valid_vllm_config(
            dspark_block_size=4,
            num_speculative_tokens=5,
        )
        with (
            patch.object(
                DSparkProposer.__bases__[0],
                "__init__",
                lambda self, runner, v_cfg: (
                    setattr(
                        self,
                        "dflash_config",
                        {"target_layer_ids": target_layers},
                    )
                    or setattr(self, "speculative_config", v_cfg.speculative_config)
                ),
            ),
            pytest.raises(
                ValueError,
                match="does not match the DSpark checkpoint's block size 4",
            ),
        ):
            DSparkProposer(MagicMock(), cfg)

    def test_init_unsupported_options_errors(self):
        # 1. draft_sample_method != "greedy"
        cfg1, target_layers = self._make_valid_vllm_config(draft_sample_method="sample")
        with (
            patch.object(
                DSparkProposer.__bases__[0],
                "__init__",
                lambda self, runner, v_cfg: (
                    setattr(
                        self,
                        "dflash_config",
                        {"target_layer_ids": target_layers},
                    )
                    or setattr(self, "speculative_config", v_cfg.speculative_config)
                ),
            ),
            pytest.raises(
                ValueError,
                match="DSpark on TPU does not support draft_sample_method='sample'",
            ),
        ):
            DSparkProposer(MagicMock(), cfg1)

        # 2. enable_adaptive_verification = True
        cfg2, target_layers = self._make_valid_vllm_config(
            enable_adaptive_verification=True
        )
        with (
            patch.object(
                DSparkProposer.__bases__[0],
                "__init__",
                lambda self, runner, v_cfg: (
                    setattr(
                        self,
                        "dflash_config",
                        {"target_layer_ids": target_layers},
                    )
                    or setattr(self, "speculative_config", v_cfg.speculative_config)
                ),
            ),
            pytest.raises(ValueError, match="enable_adaptive_verification=True"),
        ):
            DSparkProposer(MagicMock(), cfg2)

        # 3. dspark_draft_topk is not None
        cfg3, target_layers = self._make_valid_vllm_config(dspark_draft_topk=8)
        with (
            patch.object(
                DSparkProposer.__bases__[0],
                "__init__",
                lambda self, runner, v_cfg: (
                    setattr(
                        self,
                        "dflash_config",
                        {"target_layer_ids": target_layers},
                    )
                    or setattr(self, "speculative_config", v_cfg.speculative_config)
                ),
            ),
            pytest.raises(ValueError, match="dspark_draft_topk=8"),
        ):
            DSparkProposer(MagicMock(), cfg3)

    def test_load_model_checks(self):
        cfg, target_layers = self._make_valid_vllm_config()
        with patch.object(
            DSparkProposer.__bases__[0],
            "__init__",
            lambda self, runner, v_cfg: (
                setattr(
                    self,
                    "dflash_config",
                    {"target_layer_ids": target_layers},
                )
                or setattr(self, "speculative_config", v_cfg.speculative_config)
            ),
        ):
            proposer = DSparkProposer(MagicMock(), cfg)

        target_model = MagicMock()

        # Missing one attribute
        attrs = [
            "compute_draft_logits",
            "markov_embed",
            "markov_bias",
            "map_draft_to_target",
            "has_own_lm_head",
            "has_own_embed_tokens",
        ]
        for missing_attr in attrs:
            mock_draft = MagicMock()
            for a in attrs:
                if a == missing_attr:
                    delattr(mock_draft, a)
                else:
                    setattr(mock_draft, a, MagicMock())

            proposer.draft_model = mock_draft
            with (
                patch.object(
                    DSparkProposer.__bases__[0], "load_model", lambda self, tm: None
                ),
                pytest.raises(
                    RuntimeError, match=f"DSpark draft model lacks {missing_attr}\\(\\)"
                ),
            ):
                proposer.load_model(target_model)

        # All attributes present
        mock_draft = MagicMock()
        for a in attrs:
            setattr(mock_draft, a, MagicMock())
        proposer.draft_model = mock_draft
        with patch.object(
            DSparkProposer.__bases__[0], "load_model", lambda self, tm: None
        ):
            proposer.load_model(target_model)

    def test_dflash_forward_and_sample_wrapped_execution(self):
        # Test unwrapped method body across both sample_from_anchor=True
        # (sample_start=0) and sample_from_anchor=False (sample_start=1).
        fn = DSparkProposer._dflash_forward_and_sample.__wrapped__

        proposer = MagicMock()
        proposer.sample_from_anchor = True

        # Set up mock draft model
        hidden_dim = 64
        draft_vocab = 32
        target_vocab = 100
        block_size = 4
        padded_num_reqs = 2
        total_tokens = padded_num_reqs * block_size

        proposer.draft_model.model.return_value = torch.ones(total_tokens, hidden_dim)
        proposer.draft_model.compute_draft_logits.return_value = torch.zeros(
            total_tokens, draft_vocab
        )
        proposer.draft_model.markov_embed.side_effect = lambda prev: torch.zeros(
            padded_num_reqs, 16
        )
        proposer.draft_model.markov_bias.side_effect = lambda emb: torch.zeros(
            padded_num_reqs, draft_vocab
        )
        proposer.draft_model.map_draft_to_target.side_effect = lambda idx: (
            (idx + 10) % target_vocab
        )

        input_ids = torch.arange(total_tokens, dtype=torch.int64)
        positions = torch.arange(total_tokens, dtype=torch.int64)

        # 1. sample_from_anchor = True -> sample_start = 0 -> iterates 4 slots
        draft_chunk, hidden = fn(proposer, input_ids, positions, block_size)
        assert draft_chunk.shape == (padded_num_reqs, block_size)
        assert hidden.shape == (total_tokens, hidden_dim)

        # 2. sample_from_anchor = False -> sample_start = 1 -> iterates 3 slots
        proposer.sample_from_anchor = False
        draft_chunk_bonus, hidden_bonus = fn(proposer, input_ids, positions, block_size)
        assert draft_chunk_bonus.shape == (padded_num_reqs, block_size - 1)
        assert hidden_bonus.shape == (total_tokens, hidden_dim)


# ==============================================================================
# 2. TPUModelRunner Spec-Warmup Diff (eab4fc815) Tests
# ==============================================================================


class TestTpuRunnerSpecWarmupDiff:
    """Tests for 100% diff coverage of eab4fc815 in tpu_runner.py."""

    def test_spec_warmup_group_num_blocks_logic(self):
        # Case 1: CircularBufferSpec where g_nblk > max_num_blocks_per_req
        ring_spec = CircularBufferSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
        )
        ring_group = SimpleNamespace(kv_cache_spec=ring_spec)
        runner = SimpleNamespace(
            max_model_len=8192,
            vllm_config=SimpleNamespace(),
        )
        # max_pos = 2060 -> cdiv(2060, 8) = 258, max_blocks = 1 -> clamped to 1
        assert tpu_runner._spec_warmup_group_num_blocks(runner, ring_group, 2060) == 1

        # Case 2: CircularBufferSpec where max_pos is limited by runner.max_model_len
        runner_short = SimpleNamespace(
            max_model_len=4,
            vllm_config=SimpleNamespace(),
        )
        # max_pos = 2060, max_model_len = 4 -> min(2060, 4) = 4 -> cdiv(4, 8) = 1
        assert (
            tpu_runner._spec_warmup_group_num_blocks(runner_short, ring_group, 2060)
            == 1
        )

        # Case 3: CircularBufferSpec where max_num_blocks_per_req > g_nblk
        custom_ring_spec = MagicMock(spec=CircularBufferSpec)
        custom_ring_spec.block_size = 128
        custom_ring_spec.max_num_blocks_per_req.return_value = 10
        custom_group = SimpleNamespace(kv_cache_spec=custom_ring_spec)
        # max_pos = 64, max_model_len = 8192 -> cdiv(64, 128) = 1, min(1, 10) = 1
        assert tpu_runner._spec_warmup_group_num_blocks(runner, custom_group, 64) == 1

        # Case 4: Non-CircularBufferSpec (e.g. FullAttentionSpec)
        attn_spec = FullAttentionSpec(
            block_size=256,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        )
        attn_group = SimpleNamespace(kv_cache_spec=attn_spec)
        # max_pos = 2060 -> cdiv(2060, 256) = 9
        assert tpu_runner._spec_warmup_group_num_blocks(runner, attn_group, 2060) == 9
        # max_pos = 10000 -> min(10000, 8192) = 8192 -> cdiv(8192, 256) = 32
        assert tpu_runner._spec_warmup_group_num_blocks(runner, attn_group, 10000) == 32

    def test_warmup_one_pcp_mtp_prefill_wiring(self):
        # Covers line 5920 in tpu_runner.py
        ring_spec = CircularBufferSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
        )
        attn_spec = FullAttentionSpec(
            block_size=256,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        )
        executed = []
        runner = SimpleNamespace(
            max_model_len=8192,
            max_num_tokens=2048,
            kv_cache_config=SimpleNamespace(
                num_blocks=200,
                kv_cache_groups=[
                    SimpleNamespace(kv_cache_spec=attn_spec),
                    SimpleNamespace(kv_cache_spec=ring_spec),
                ],
            ),
            execute_model=MagicMock(
                side_effect=lambda so: executed.append(so) and None
            ),
            sample_tokens=MagicMock(),
            take_draft_token_ids=MagicMock(return_value=None),
            _warmup_spec_decode_cleanup=MagicMock(return_value=True),
        )

        res = TPUModelRunner._warmup_one_pcp_mtp_prefill(runner, P=2048)
        assert res is True
        assert len(executed) == 1
        new_req = executed[0].scheduled_new_reqs[0]
        # Attention group has 9 blocks, CircularBufferSpec has 1 block
        assert len(new_req.block_ids[0]) == 9
        assert len(new_req.block_ids[1]) == 1

    def test_warmup_spec_batch_and_one_spec_request_wiring(self):
        # Covers lines 6223 and 6339 in tpu_runner.py
        ring_spec = CircularBufferSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
        )
        attn_spec = SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(
                block_size=256,
                num_kv_heads=8,
                head_size=128,
                dtype=torch.bfloat16,
            )
        )
        ring_group = SimpleNamespace(kv_cache_spec=ring_spec)

        executed = []
        runner = SimpleNamespace(
            max_model_len=8192,
            max_num_tokens=2048,
            speculative_config=SimpleNamespace(num_speculative_tokens=5),
            kv_cache_config=SimpleNamespace(
                num_blocks=200,
                kv_cache_groups=[attn_spec, ring_group],
            ),
            execute_model=MagicMock(
                side_effect=lambda so: executed.append(so) and None
            ),
            sample_tokens=MagicMock(),
            take_draft_token_ids=MagicMock(return_value=None),
            _warmup_spec_decode_cleanup=MagicMock(return_value=True),
        )

        # 1. _warmup_one_spec_request (line 6339)
        assert TPUModelRunner._warmup_one_spec_request(runner, 2048, idx=0) is True
        req_one = executed[0].scheduled_new_reqs[0]
        assert req_one.block_ids == (list(range(1, 10)), [1])

        # 2. _warmup_spec_batch (line 6223)
        executed.clear()
        TPUModelRunner._warmup_spec_batch(runner, R=2, P=2048, greedy=True)
        assert len(executed) == 9  # 2 prefill chunks + 7 decode steps
        req0 = executed[0].scheduled_new_reqs[0]
        req1 = executed[1].scheduled_new_reqs[0]
        assert req0.block_ids == (list(range(1, 10)), [1])
        assert req1.block_ids == (list(range(10, 19)), [10])


# ==============================================================================
# 3. O-Projection Full-Module Coverage Tests
# ==============================================================================


class TestOProjectionFull:
    """Tests for 100% line and branch coverage of o_projection.py."""

    def test_largest_divisor(self):
        # Hits direct divisor
        assert op._largest_divisor(128, cap=128) == 128
        assert op._largest_divisor(100, cap=16) == 10
        assert op._largest_divisor(10, cap=20) == 10
        # Loops down to 1
        assert op._largest_divisor(7, cap=6) == 1
        assert op._largest_divisor(1, cap=1) == 1
        # Loop exhaustion (empty range when min(x, cap) <= 0, covers line 40)
        assert op._largest_divisor(0, cap=10) == 1
        assert op._largest_divisor(10, cap=0) == 1
        assert op._largest_divisor(-5, cap=10) == 1

    def test_cos_sin_body(self):
        tile_n = 4
        rotary_dim = 64
        cos_sin_in = np.ones((tile_n, rotary_dim), dtype=np.float32) * 0.5
        ref_in = MockArrayRef(cos_sin_in)
        ref_out = MockArrayRef(np.zeros((tile_n, 2 * LANE), dtype=np.float32))

        # inverse = False
        op._cos_sin_body(
            ref_in,
            ref_out,
            rotary_dim=rotary_dim,
            inverse=False,
            out_dtype=jnp.float32,
        )
        assert ref_out.arr.shape == (tile_n, 2 * LANE)
        assert np.isfinite(ref_out.arr).all()

        # inverse = True
        ref_out_inv = MockArrayRef(np.zeros((tile_n, 2 * LANE), dtype=np.float32))
        op._cos_sin_body(
            ref_in,
            ref_out_inv,
            rotary_dim=rotary_dim,
            inverse=True,
            out_dtype=jnp.bfloat16,
        )
        assert ref_out_inv.arr.shape == (tile_n, 2 * LANE)

    def test_cos_sin_kernel_wiring(self):
        captured_specs = []

        def fake_emit(pipeline_fn, grid=None, in_specs=None, out_specs=None):
            captured_specs.extend(in_specs or [])
            captured_specs.extend(out_specs or [])

            def runner(*args, allocations=None):
                pass

            return runner

        mock_alloc = SimpleNamespace()

        with (
            patch.object(op.pltpu, "emit_pipeline", side_effect=fake_emit),
            patch.object(op.CosSinRef, "create", return_value=mock_alloc),
            patch.object(
                op.pl,
                "with_scoped",
                side_effect=lambda allocations: lambda fn: lambda: fn(allocations),
            ),
            patch.object(op.pltpu.BufferedRef, "output", return_value=mock_alloc),
        ):
            pos_ref = MockArrayRef(np.zeros(8, dtype=np.int32))
            cs_cache = MockArrayRef(np.zeros((64, 64), dtype=np.float32))
            out_hbm = MockArrayRef(np.zeros((8, 2 * LANE), dtype=np.float32))

            op._cos_sin_kernel(
                pos_ref,
                cs_cache,
                out_hbm,
                num_tiles=2,
                tile_n=4,
                rotary_dim=64,
                inverse=False,
                cos_sin_dtype=jnp.float32,
                out_dtype=jnp.float32,
            )

            # Test index_maps for both specs
            assert len(captured_specs) == 2
            for s in captured_specs:
                idx = s.index_map(3)
                assert idx == (3, 0)

    def test_gather_cos_sin_assertions(self):
        # 1. positions.ndim != 1
        with pytest.raises(AssertionError):
            op.gather_cos_sin(
                jnp.zeros((2, 2), dtype=jnp.int32),
                jnp.zeros((16, 64), dtype=jnp.float32),
            )

        # 2. cos_sin_cache.ndim != 2
        with pytest.raises(AssertionError):
            op.gather_cos_sin(
                jnp.zeros((4,), dtype=jnp.int32),
                jnp.zeros((16, 64, 2), dtype=jnp.float32),
            )

        # 3. rotary_dim % 2 != 0
        with pytest.raises(AssertionError):
            op.gather_cos_sin(
                jnp.zeros((4,), dtype=jnp.int32), jnp.zeros((16, 63), dtype=jnp.float32)
            )

        # 4. rotary_dim > LANE (LANE = 128)
        with pytest.raises(AssertionError):
            op.gather_cos_sin(
                jnp.zeros((4,), dtype=jnp.int32),
                jnp.zeros((16, 130), dtype=jnp.float32),
            )

    def test_gather_cos_sin_execution(self):
        positions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        cache = jnp.zeros((16, 64), dtype=jnp.float32)

        fake_out = jnp.ones((4, 2 * LANE), dtype=jnp.float32)

        with patch.object(
            op.pl, "pallas_call", return_value=lambda pos, c: fake_out
        ) as mock_pallas:
            # Test gather_cos_sin.__wrapped__
            res = op.gather_cos_sin.__wrapped__(
                positions,
                cache,
                inverse=True,
                out_dtype=jnp.float32,
                name="test_gather",
            )
            assert res.shape == (4, 2 * LANE)
            assert mock_pallas.called

    def test_rope_heads(self):
        # 1. head_dim = 128 (lo = 0, entire head roped)
        x_128 = jnp.ones((2, 4, 128), dtype=jnp.bfloat16)
        cs = jnp.ones((2, 2 * LANE), dtype=jnp.float32)
        roped_128 = op._rope_heads(x_128, cs, head_dim=128)
        assert roped_128.shape == (2, 4, 128)
        assert roped_128.dtype == jnp.bfloat16

        # 2. head_dim = 256 (lo = 128, first 128 kept, last 128 roped)
        x_256 = jnp.ones((2, 4, 256), dtype=jnp.bfloat16)
        roped_256 = op._rope_heads(x_256, cs, head_dim=256)
        assert roped_256.shape == (2, 4, 256)
        assert roped_256.dtype == jnp.bfloat16
        # lo part should match exactly
        np.testing.assert_array_equal(
            np.asarray(roped_256[:, :, :128]), np.asarray(x_256[:, :, :128])
        )

    def test_kernel_quantized_and_unquantized(self):
        tile_t = 4
        num_sub_t = 2
        H = 8
        head_dim = 128
        D = H * head_dim
        tile_r = 16

        x_arr = np.ones((tile_t, H, head_dim), dtype=np.float32)
        w_arr = np.ones((D, tile_r), dtype=np.int8)  # fp8 mock
        scale_arr = np.ones((1, tile_r), dtype=np.float32) * 0.1
        cs_arr = np.ones((tile_t, 2 * LANE), dtype=np.float32)

        x_ref = MockArrayRef(x_arr)
        w_ref = MockArrayRef(w_arr)
        scale_ref = MockArrayRef(scale_arr)
        cs_ref = MockArrayRef(cs_arr)
        out_ref = MockArrayRef(np.zeros((tile_t, tile_r), dtype=np.float32))

        # 1. quantize_activations = True (covers lines 199-204, 214-216)
        op._kernel(
            x_ref,
            w_ref,
            scale_ref,
            cs_ref,
            out_ref,
            tile_t=tile_t,
            num_sub_t=num_sub_t,
            quantize_activations=True,
            head_dim=head_dim,
        )
        assert out_ref.arr.shape == (tile_t, tile_r)
        assert np.isfinite(out_ref.arr).all()

        # 2. quantize_activations = False (covers line 206)
        out_ref_unquant = MockArrayRef(np.zeros((tile_t, tile_r), dtype=np.float32))
        op._kernel(
            x_ref,
            w_ref,
            scale_ref,
            cs_ref,
            out_ref_unquant,
            tile_t=tile_t,
            num_sub_t=num_sub_t,
            quantize_activations=False,
            head_dim=head_dim,
        )
        assert out_ref_unquant.arr.shape == (tile_t, tile_r)
        assert np.isfinite(out_ref_unquant.arr).all()

        # 3. quantize_activations = True with zero activations (covers amax == 0)
        x_zero_ref = MockArrayRef(np.zeros((tile_t, H, head_dim), dtype=np.float32))
        out_ref_zero = MockArrayRef(np.zeros((tile_t, tile_r), dtype=np.float32))
        op._kernel(
            x_zero_ref,
            w_ref,
            scale_ref,
            cs_ref,
            out_ref_zero,
            tile_t=tile_t,
            num_sub_t=num_sub_t,
            quantize_activations=True,
            head_dim=head_dim,
        )
        assert np.isfinite(out_ref_zero.arr).all()

    def test_wo_a_projection_assertions(self):
        T = 16
        G = 2
        H = 8  # SUBLANE = 8
        head_dim = 128
        D = H * head_dim  # 1024
        R = 32
        out_features = G * R  # 64

        valid_x = jnp.zeros((T, G * H, head_dim), dtype=jnp.bfloat16)
        valid_w = jnp.zeros((D, out_features), dtype=jnp.int8)
        valid_scale = jnp.zeros((out_features,), dtype=jnp.float32)
        valid_cs = jnp.zeros((T, 2 * LANE), dtype=jnp.float32)

        # 1. x.ndim != 3
        with pytest.raises(AssertionError):
            op.wo_a_projection(
                jnp.zeros((T, G * H), dtype=jnp.bfloat16),
                valid_w,
                valid_scale,
                valid_cs,
            )

        # 2. x.dtype != jnp.bfloat16
        with pytest.raises(AssertionError):
            op.wo_a_projection(
                jnp.zeros((T, G * H, head_dim), dtype=jnp.float32),
                valid_w,
                valid_scale,
                valid_cs,
            )

        # 3. wo_a.ndim != 2 or wo_a_scale.ndim != 1
        with pytest.raises(AssertionError):
            op.wo_a_projection(
                valid_x,
                jnp.zeros((D, out_features, 1), dtype=jnp.int8),
                valid_scale,
                valid_cs,
            )
        with pytest.raises(AssertionError):
            op.wo_a_projection(
                valid_x,
                valid_w,
                jnp.zeros((out_features, 1), dtype=jnp.float32),
                valid_cs,
            )

        # 4. out_features != wo_a_scale.shape[0]
        with pytest.raises(AssertionError):
            op.wo_a_projection(
                valid_x,
                valid_w,
                jnp.zeros((out_features + 1,), dtype=jnp.float32),
                valid_cs,
            )

        # 5. reduction % head_dim != 0
        with pytest.raises(AssertionError):
            bad_w = jnp.zeros((D + 1, out_features), dtype=jnp.int8)
            op.wo_a_projection(valid_x, bad_w, valid_scale, valid_cs)

        # 6. heads_per_group != SUBLANE (heads_per_group = 4)
        with pytest.raises(AssertionError):
            bad_w = jnp.zeros((4 * head_dim, out_features), dtype=jnp.int8)
            op.wo_a_projection(valid_x, bad_w, valid_scale, valid_cs)

        # 7. num_heads % heads_per_group != 0
        with pytest.raises(AssertionError):
            bad_x = jnp.zeros((T, 15, head_dim), dtype=jnp.bfloat16)
            op.wo_a_projection(bad_x, valid_w, valid_scale, valid_cs)

        # 8. out_features % num_groups != 0 (num_groups = 2, out_features = 65)
        with pytest.raises(AssertionError):
            bad_w = jnp.zeros((D, 65), dtype=jnp.int8)
            bad_scale = jnp.zeros((65,), dtype=jnp.float32)
            op.wo_a_projection(valid_x, bad_w, bad_scale, valid_cs)

        # 9. cos_sin.shape != (num_tokens, 2 * LANE)
        with pytest.raises(AssertionError):
            bad_cs = jnp.zeros((T, LANE), dtype=jnp.float32)
            op.wo_a_projection(valid_x, valid_w, valid_scale, bad_cs)

        # 10. head_dim % LANE != 0
        with pytest.raises(AssertionError):
            bad_x = jnp.zeros((T, G * H, 100), dtype=jnp.bfloat16)
            bad_w = jnp.zeros((H * 100, out_features), dtype=jnp.int8)
            op.wo_a_projection(bad_x, bad_w, valid_scale, valid_cs)

        # 11. lora_rank % tile_r != 0 (lora_rank = 32, tile_r = 7)
        with pytest.raises(AssertionError):
            op.wo_a_projection(valid_x, valid_w, valid_scale, valid_cs, tile_r=7)

        # 12. num_tokens % tile_t != 0 (T = 16, tile_t = 5)
        with pytest.raises(AssertionError):
            op.wo_a_projection(valid_x, valid_w, valid_scale, valid_cs, tile_t=5)

        # 13. tile_t % sub_t != 0 (tile_t = 16, sub_t = 6)
        with pytest.raises(AssertionError):
            op.wo_a_projection(
                valid_x, valid_w, valid_scale, valid_cs, tile_t=16, sub_t=6
            )

    def test_wo_a_projection_execution_and_specs(self):
        T = 16
        G = 2
        H = 8
        head_dim = 128
        D = H * head_dim
        R = 32
        out_features = G * R

        x = jnp.zeros((T, G * H, head_dim), dtype=jnp.bfloat16)
        w = jnp.zeros((D, out_features), dtype=jnp.int8)
        scale = jnp.zeros((out_features,), dtype=jnp.float32)
        cs = jnp.zeros((T, 2 * LANE), dtype=jnp.float32)

        captured_in_specs = None
        captured_out_specs = None
        captured_grid = None

        def fake_pallas_call(
            kernel, out_shape, grid, in_specs, out_specs, compiler_params, name
        ):
            nonlocal captured_in_specs, captured_out_specs, captured_grid
            captured_in_specs = in_specs
            captured_out_specs = out_specs
            captured_grid = grid

            def runner(*args):
                return jnp.ones(out_shape.shape, dtype=out_shape.dtype)

            return runner

        with patch.object(op.pl, "pallas_call", side_effect=fake_pallas_call):
            # Test default tile_r=None, tile_t=None, sub_t=None
            res = op.wo_a_projection.__wrapped__(x, w, scale, cs)
            assert res.shape == (T, out_features)

            # Check grid and specs
            assert captured_grid == (
                G,
                1,
                1,
            )  # num_groups=2, num_t_tiles=1, num_r_tiles=1
            assert len(captured_in_specs) == 4

            # Test index_maps for each BlockSpec
            # 1. x: (t, g, 0)
            assert captured_in_specs[0].index_map(1, 2, 3) == (2, 1, 0)
            # 2. wo_a: (0, g * num_r_tiles + r)
            assert captured_in_specs[1].index_map(1, 2, 0) == (0, 1)
            # 3. wo_a_scale: (0, g * num_r_tiles + r)
            assert captured_in_specs[2].index_map(1, 2, 0) == (0, 1)
            # 4. cos_sin: (t, 0)
            assert captured_in_specs[3].index_map(1, 2, 3) == (2, 0)
            # out_spec: (t, g * num_r_tiles + r)
            assert captured_out_specs.index_map(1, 2, 0) == (2, 1)

            # Test custom tile_r, tile_t, sub_t, quantize_activations=False
            res_custom = op.wo_a_projection.__wrapped__(
                x,
                w,
                scale,
                cs,
                tile_t=8,
                tile_r=16,
                sub_t=4,
                quantize_activations=False,
                name="custom_wo_a",
            )
            assert res_custom.shape == (T, out_features)
            # T=16, tile_t=8 -> num_t_tiles=2; R=32, tile_r=16 -> num_r_tiles=2
            assert captured_grid == (G, 2, 2)

    def test_fused_reverse_rope_wo_a_projection(self):
        T = 16
        G = 2
        H = 8
        head_dim = 128
        D = H * head_dim
        R = 32
        out_features = G * R

        x = jnp.zeros((T, G * H, head_dim), dtype=jnp.bfloat16)
        positions = jnp.arange(T, dtype=jnp.int32)
        cache = jnp.zeros((64, 64), dtype=jnp.float32)
        w = jnp.zeros((D, out_features), dtype=jnp.int8)
        scale = jnp.zeros((out_features,), dtype=jnp.float32)

        # 1. Assertion: x.ndim != 3
        with pytest.raises(AssertionError):
            op.fused_reverse_rope_wo_a_projection(
                jnp.zeros((T, G * H), dtype=jnp.bfloat16),
                positions,
                cache,
                w,
                scale,
                head_dim=head_dim,
            )

        # 2. Assertion: x.shape[2] != head_dim
        with pytest.raises(AssertionError):
            op.fused_reverse_rope_wo_a_projection(
                x,
                positions,
                cache,
                w,
                scale,
                head_dim=head_dim + 16,
            )

        # 3. Successful execution with mocked gather_cos_sin and wo_a_projection
        fake_cs = jnp.zeros((T, 2 * LANE), dtype=jnp.float32)
        fake_out = jnp.zeros((T, out_features), dtype=jnp.bfloat16)

        with (
            patch.object(op, "gather_cos_sin", return_value=fake_cs) as mock_gather,
            patch.object(op, "wo_a_projection", return_value=fake_out) as mock_proj,
        ):
            res = op.fused_reverse_rope_wo_a_projection.__wrapped__(
                x,
                positions,
                cache,
                w,
                scale,
                head_dim=head_dim,
                inverse=True,
                name="test_fused",
            )
            assert res.shape == (T, out_features)
            mock_gather.assert_called_once_with(
                positions,
                cache,
                inverse=True,
                name="test_fused_gather_cos_sin",
            )
            mock_proj.assert_called_once()
