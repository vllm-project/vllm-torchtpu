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
"""Unit tests for MoE chunk pipelining, mathematical derivation, and contract checks."""

import inspect
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.fused_moe import fused_moe_gmm
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    calculate_moe_chunks, enable_pipelined_collective_and_compute,
    pipelined_fused_moe_gmm)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16 import \
    VllmCompressedTensorsW4A16MoEMethod
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import \
    VllmCompressedTensorsW4ANMxfp4MoEMethod
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import VllmMxfp4MoEMethod
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod
from vllm_torchtpu.layers.adapter.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod


def _make_mock_moe_config():
    return SimpleNamespace(
        experts_per_token=8,
        moe_parallel_config=SimpleNamespace(use_ep=False),
        group_size=32,
    )


def test_env_var_default_and_override():
    """Verify TPU_MOE_COLLECTION_CHUNK_SIZE default is 0 and parses positive ints."""
    with patch.dict(os.environ, {}, clear=True):
        assert envs.TPU_MOE_COLLECTION_CHUNK_SIZE == 0

    with patch.dict(os.environ, {"TPU_MOE_COLLECTION_CHUNK_SIZE": "16384"}):
        assert envs.TPU_MOE_COLLECTION_CHUNK_SIZE == 16384

    with patch.dict(os.environ, {"TPU_MOE_COLLECTION_CHUNK_SIZE": "0"}):
        assert envs.TPU_MOE_COLLECTION_CHUNK_SIZE == 0


def test_enable_pipelined_collective_and_compute():
    """Verify enable_pipelined_collective_and_compute reflects TPU_MOE_COLLECTION_CHUNK_SIZE > 0."""
    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 0):
        assert enable_pipelined_collective_and_compute() is False

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384):
        assert enable_pipelined_collective_and_compute() is True


def test_enable_pipelined_collective_allows_pcp():
    """Verify PCP does not disable MoE collective chunking."""
    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384):
        assert enable_pipelined_collective_and_compute() is True


@pytest.mark.parametrize(
    "method_cls",
    [
        VllmFp8MoEMethodTPU,
        VllmUnquantizedFusedMoEMethod,
        VllmMxfp4MoEMethod,
        VllmNvfp4MoEMethod,
        VllmCompressedTensorsW4A16MoEMethod,
        VllmCompressedTensorsW4ANMxfp4MoEMethod,
    ],
)
def test_supports_internal_mk_property(method_cls):
    """Verify supports_internal_mk is True when chunk_size > 0, False when 0 across all 6 quantization classes."""
    prop = getattr(method_cls, "supports_internal_mk")
    assert isinstance(prop, property)

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 0):
        assert prop.fget(None) is False

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384):
        assert prop.fget(None) is True


def test_calculate_moe_chunks_math():
    """Verify chunk calculation: N_chunk = ceil((DP * S) / C), S_chunk = S // N_chunk."""
    # S=8192, DP=4, C=16384 -> T_global=32768 -> N_chunk=2, S_chunk=4096
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=8192,
                                                        parallel_size=4,
                                                        chunk_size=16384)
    assert num_chunks == 2
    assert chunk_size_local == 4096

    # S=4096, DP=4, C=16384 -> T_global=16384 -> N_chunk=1, S_chunk=4096
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=4096,
                                                        parallel_size=4,
                                                        chunk_size=16384)
    assert num_chunks == 1
    assert chunk_size_local == 4096

    # S=4096, DP=8, C=16384 -> T_global=32768 -> N_chunk=2, S_chunk=2048
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=4096,
                                                        parallel_size=8,
                                                        chunk_size=16384)
    assert num_chunks == 2
    assert chunk_size_local == 2048

    # Sub-threshold: S=1024, DP=4, C=16384 -> T_global=4096 -> N_chunk=1, S_chunk=1024
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=1024,
                                                        parallel_size=4,
                                                        chunk_size=16384)
    assert num_chunks == 1
    assert chunk_size_local == 1024

    # Non-multiple global: S=3072, DP=8, C=16384 -> T_global=24576 -> N_chunk=2, S_chunk=1536
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=3072,
                                                        parallel_size=8,
                                                        chunk_size=16384)
    assert num_chunks == 2
    assert chunk_size_local == 1536

    # Non-multiple global: S=8000, DP=4, C=16384 -> T_global=32000 -> N_chunk=2, S_chunk=4000
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=8000,
                                                        parallel_size=4,
                                                        chunk_size=16384)
    assert num_chunks == 2
    assert chunk_size_local == 4000

    # Zero/Empty sequence length guard
    num_chunks, chunk_size_local = calculate_moe_chunks(seq_len=0,
                                                        parallel_size=4,
                                                        chunk_size=16384)
    assert num_chunks == 1
    assert chunk_size_local == 0


def test_calculate_moe_chunks_validation_errors():
    """Verify invalid token/chunk geometries raise clear ValueErrors."""
    # Local tokens (5001) with DP=4, C=16384 -> T=20004 -> N_chunk=2, but 5001 % 2 != 0
    with pytest.raises(ValueError, match="not divisible by chunk count"):
        calculate_moe_chunks(seq_len=5001, parallel_size=4, chunk_size=16384)

    # Local tokens (3) with DP=4, C=6 -> T=12 -> N_chunk=2, but 3 % 2 != 0
    with pytest.raises(ValueError, match="not divisible by chunk count"):
        calculate_moe_chunks(seq_len=3, parallel_size=4, chunk_size=6)


class _MockCollectiveGroup:
    """Mock collective group tracking calls and verifying order."""

    def __init__(self, world_size=4, rank=0):
        self.world_size = world_size
        self.rank_in_group = rank
        self.call_log = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.call_log.append(("all_gather", tensor.shape))
        # Replicate along dim=0 across world_size
        return torch.cat([tensor] * self.world_size, dim=dim)

    def reduce_scatter(self,
                       tensor: torch.Tensor,
                       dim: int = 0) -> torch.Tensor:
        self.call_log.append(("reduce_scatter", tensor.shape))
        chunks = tensor.chunk(self.world_size, dim=dim)
        return chunks[self.rank_in_group]


def test_pipelined_moe_execution_flow_and_numerical_parity():
    """Verify pipelined execution interleaving collectives and compute with numerical parity."""
    dp_group = _MockCollectiveGroup(world_size=4, rank=0)
    seq_len = 8192
    hidden_dim = 128
    topk = 2
    num_experts = 4

    # Dummy inputs
    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.ones(seq_len, topk, dtype=torch.bfloat16) / topk
    topk_ids = torch.zeros(seq_len, topk, dtype=torch.int32)

    # Dummy weights
    w1 = torch.randn(num_experts,
                     hidden_dim,
                     hidden_dim * 2,
                     dtype=torch.bfloat16)
    w2 = torch.randn(num_experts,
                     hidden_dim * 2,
                     hidden_dim,
                     dtype=torch.bfloat16)

    def mock_kernel_fn(hidden_states, *args, **kwargs):
        # Linear compute for parity check: hs * 2.0
        return hidden_states * 2.0

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.get_pcp_group", return_value=None), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.get_dp_group", return_value=dp_group), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )

    # Output shape should match original local hidden_states (S, H)
    assert out.shape == (seq_len, hidden_dim)

    # Numerical parity check: each token should be exactly hs * 2.0
    torch.testing.assert_close(out, hidden_states * 2.0)

    # Call log should record 2 all-gathers (for hidden_states, topk_weights, topk_ids) and 2 reduce-scatters
    ag_calls = [c for c in dp_group.call_log if c[0] == "all_gather"]
    rs_calls = [c for c in dp_group.call_log if c[0] == "reduce_scatter"]
    # 3 tensors gathered per chunk * 2 chunks = 6 all_gather calls
    assert len(ag_calls) == 6
    # 1 tensor reduce-scattered per chunk * 2 chunks = 2 reduce_scatter calls
    assert len(rs_calls) == 2


def test_pcp_uses_chunk_pipeline_and_not_dp_collectives():
    """Verify PCP selects its own group and executes multiple MoE chunks."""
    pcp_group = _MockCollectiveGroup(world_size=4, rank=0)
    dp_group = _MockCollectiveGroup(world_size=1, rank=0)
    seq_len = 8192
    hidden_dim = 16
    topk = 2
    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.ones(seq_len, topk, dtype=torch.bfloat16) / topk
    topk_ids = torch.zeros(seq_len, topk, dtype=torch.int32)

    def mock_kernel_fn(hidden_states, *args, **kwargs):
        return hidden_states * 2.0

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.get_pcp_group", return_value=pcp_group), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.get_dp_group", return_value=dp_group), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=torch.empty(0),
            w2=torch.empty(0),
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )

    torch.testing.assert_close(out, hidden_states * 2.0)
    assert len([c for c in pcp_group.call_log if c[0] == "all_gather"]) == 6
    assert len([c for c in pcp_group.call_log
                if c[0] == "reduce_scatter"]) == 2
    assert dp_group.call_log == []


class TestMoEForwardPrecisionBranches:
    """Tests protecting apply_monolithic call paths and signature parity across all MoE precision methods."""

    def test_fused_moe_and_pipelined_signature_parity(self):
        """Ensure fused_moe_gmm and pipelined_fused_moe_gmm maintain identical signatures.

        Prevents changes to fused_moe_gmm parameter list that break pipelined_fused_moe_gmm.
        """
        sig_fused = inspect.signature(fused_moe_gmm)
        sig_pipelined = inspect.signature(pipelined_fused_moe_gmm)

        assert list(sig_fused.parameters.keys()) == list(
            sig_pipelined.parameters.keys()
        ), (f"Signature parameter mismatch between fused_moe_gmm ({list(sig_fused.parameters.keys())}) "
            f"and pipelined_fused_moe_gmm ({list(sig_pipelined.parameters.keys())})."
            )

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_fp8_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify FP8 apply_monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight = torch.randn(4, 64, 128, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn)
        layer.w2_weight = torch.randn(4, 64, 64, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn)
        layer.w13_weight_scale_inv = torch.ones(4,
                                                1,
                                                1,
                                                128,
                                                dtype=torch.float32)
        layer.w2_weight_scale_inv = torch.ones(4,
                                               1,
                                               1,
                                               64,
                                               dtype=torch.float32)
        layer.w13_bias = torch.zeros(4, 1, 128, dtype=torch.float32)
        layer.w2_bias = torch.zeros(4, 1, 64, dtype=torch.float32)
        layer.moe_config.experts_per_token = 2
        layer.moe_config.moe_parallel_config.use_ep = False

        method = MagicMock(spec=VllmFp8MoEMethodTPU)
        method._tpu_activation_str = "silu"
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.adapter.quantization.fp8.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.fp8.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.fp8.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.fp8.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmFp8MoEMethodTPU.apply_monolithic(method, layer, x,
                                                       router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_unquantized_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify unquantized apply_monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        layer.w2_weight = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        layer.w13_bias = torch.zeros(4, 1, 128, dtype=torch.float32)
        layer.w2_bias = torch.zeros(4, 1, 64, dtype=torch.float32)
        layer.moe_config.experts_per_token = 2

        method = MagicMock()
        method._tpu_activation_str = "silu"
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.adapter.quantization.unquantized.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.unquantized.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.unquantized.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.unquantized.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmUnquantizedFusedMoEMethod._forward_monolithic_tpu(
                method, layer, x, router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_mxfp4_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify MXFP4 forward monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        layer.w2_weight = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        layer.w13_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.w13_bias = None
        layer.w2_bias = None
        layer.moe_config.experts_per_token = 2

        method = MagicMock()
        method._tpu_activation_str = "silu"
        method.rhs_quant_dtype = None
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.adapter.quantization.mxfp4.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.mxfp4.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.mxfp4.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.mxfp4.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmMxfp4MoEMethod._forward_monolithic_tpu(
                method, layer, x, router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_nvfp4_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify NVFP4 apply_monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        layer.w2_weight = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        layer.w13_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.moe_config.experts_per_token = 2

        method = MagicMock(spec=VllmNvfp4MoEMethod)
        method._tpu_activation_str = "silu"
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.adapter.quantization.nvfp4.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.nvfp4.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.nvfp4.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.nvfp4.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmNvfp4MoEMethod.apply_monolithic(method, layer, x,
                                                      router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_w4a16_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify compressed tensors W4A16 apply_monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer.activation = "silu"
        layer.moe_config = _make_mock_moe_config()
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight_packed = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        layer.w2_weight_packed = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        layer.w13_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)

        method = MagicMock(spec=VllmCompressedTensorsW4A16MoEMethod)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 8, dtype=torch.bfloat16) * 0.125
        topk_ids = torch.zeros(4, 8, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmCompressedTensorsW4A16MoEMethod.apply_monolithic(
                method, layer, x, router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight_packed

    @pytest.mark.parametrize("pipelined", [False, True])
    def test_w4an_mxfp4_apply_monolithic_dispatches_with_exact_kwargs(
            self, pipelined):
        """Verify compressed tensors W4AN_MXFP4 apply_monolithic passes identical valid kwargs to both branches."""
        layer = MagicMock()
        layer._experts_start = torch.zeros((), dtype=torch.int32)
        layer.w13_weight = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        layer.w2_weight = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        layer.w13_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(4, 1, 1, dtype=torch.float32)
        layer.moe_config.experts_per_token = 2

        # Exercise the real apply_with_routing helper, not a MagicMock child.
        method = VllmCompressedTensorsW4ANMxfp4MoEMethod.__new__(
            VllmCompressedTensorsW4ANMxfp4MoEMethod)
        method._tpu_activation_str = "silu"
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16) * 0.5
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)

        mock_fused = MagicMock(return_value=x)
        mock_pipelined = MagicMock(return_value=x)

        with patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.moe_routing.route",
                return_value=(topk_weights, topk_ids),
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.enable_pipelined_collective_and_compute",
                return_value=pipelined,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.fused_moe_gmm",
                mock_fused,
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.pipelined_fused_moe_gmm",
                mock_pipelined,
        ):
            out = VllmCompressedTensorsW4ANMxfp4MoEMethod.apply_monolithic(
                method, layer, x, router_logits)

        assert out is x
        called_mock = mock_pipelined if pipelined else mock_fused
        called_mock.assert_called_once()
        call_kwargs = called_mock.call_args.kwargs
        target_fn = pipelined_fused_moe_gmm if pipelined else fused_moe_gmm
        bound = inspect.signature(target_fn).bind(**call_kwargs)
        assert bound.arguments["hidden_states"] is x
        assert bound.arguments["w1"] is layer.w13_weight


def test_pipelined_moe_single_chunk_dp_greater_than_one():
    """Verify pipelined MoE execution when num_chunks == 1 with DP > 1."""
    dp_group = _MockCollectiveGroup(world_size=4, rank=0)
    seq_len = 1024
    hidden_dim = 64
    topk = 2
    num_experts = 4

    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.ones(seq_len, topk, dtype=torch.bfloat16) * 0.5
    topk_ids = torch.zeros(seq_len, topk, dtype=torch.int32)
    w1 = torch.randn(num_experts,
                     hidden_dim,
                     hidden_dim * 2,
                     dtype=torch.bfloat16)
    w2 = torch.randn(num_experts,
                     hidden_dim * 2,
                     hidden_dim,
                     dtype=torch.bfloat16)

    def mock_kernel_fn(hidden_states, *args, **kwargs):
        return hidden_states * 3.0

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.get_dp_group", return_value=dp_group), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )

    assert out.shape == (seq_len, hidden_dim)
    torch.testing.assert_close(out, hidden_states * 3.0)

    # In single-chunk mode (num_chunks=1): 3 all_gathers (hs, weights, ids) and 1 reduce_scatter
    ag_calls = [c for c in dp_group.call_log if c[0] == "all_gather"]
    rs_calls = [c for c in dp_group.call_log if c[0] == "reduce_scatter"]
    assert len(ag_calls) == 3
    assert len(rs_calls) == 1


def test_pipelined_moe_dp_size_one_no_collectives():
    """Verify pipelined MoE bypasses collectives when DP == 1 or dp_group is None."""
    seq_len = 1024
    hidden_dim = 64
    topk = 2
    num_experts = 4

    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.ones(seq_len, topk, dtype=torch.bfloat16) * 0.5
    topk_ids = torch.zeros(seq_len, topk, dtype=torch.int32)
    w1 = torch.randn(num_experts,
                     hidden_dim,
                     hidden_dim * 2,
                     dtype=torch.bfloat16)
    w2 = torch.randn(num_experts,
                     hidden_dim * 2,
                     hidden_dim,
                     dtype=torch.bfloat16)

    def mock_kernel_fn(hidden_states, *args, **kwargs):
        return hidden_states * 1.5

    # Case 1: dp_group is None
    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.get_dp_group", return_value=None), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )
    assert out.shape == (seq_len, hidden_dim)
    torch.testing.assert_close(out, hidden_states * 1.5)

    # Case 2: dp_group world_size == 1
    dp_group_1 = _MockCollectiveGroup(world_size=1, rank=0)
    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.get_dp_group", return_value=dp_group_1), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )
    assert out.shape == (seq_len, hidden_dim)
    torch.testing.assert_close(out, hidden_states * 1.5)
    assert len(dp_group_1.call_log) == 0


def test_pipelined_moe_with_none_routing_tensors():
    """Verify pipelining works cleanly when topk_weights or topk_ids are None."""
    dp_group = _MockCollectiveGroup(world_size=4, rank=0)
    seq_len = 8192
    hidden_dim = 64
    num_experts = 4

    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(num_experts,
                     hidden_dim,
                     hidden_dim * 2,
                     dtype=torch.bfloat16)
    w2 = torch.randn(num_experts,
                     hidden_dim * 2,
                     hidden_dim,
                     dtype=torch.bfloat16)

    def mock_kernel_fn(hidden_states,
                       topk_weights=None,
                       topk_ids=None,
                       *args,
                       **kwargs):
        assert topk_weights is None
        assert topk_ids is None
        return hidden_states * 2.0

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.get_dp_group", return_value=dp_group), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=None,
            topk_ids=None,
            experts_start=None,
            topk=2,
            activation="silu",
        )

    assert out.shape == (seq_len, hidden_dim)
    torch.testing.assert_close(out, hidden_states * 2.0)
    # 1 tensor gathered per chunk * 2 chunks = 2 all_gather calls
    ag_calls = [c for c in dp_group.call_log if c[0] == "all_gather"]
    rs_calls = [c for c in dp_group.call_log if c[0] == "reduce_scatter"]
    assert len(ag_calls) == 2
    assert len(rs_calls) == 2


def test_pipelined_moe_four_stage_pipeline():
    """Verify pipelining with 4 chunks exercises multi-iteration pipeline steady state."""
    dp_group = _MockCollectiveGroup(world_size=4, rank=0)
    # S=16384, DP=4, C=16384 -> T_global=65536 -> N_chunk=4, S_chunk=4096
    seq_len = 16384
    hidden_dim = 64
    topk = 2
    num_experts = 4

    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    topk_weights = torch.ones(seq_len, topk, dtype=torch.bfloat16) * 0.5
    topk_ids = torch.zeros(seq_len, topk, dtype=torch.int32)
    w1 = torch.randn(num_experts,
                     hidden_dim,
                     hidden_dim * 2,
                     dtype=torch.bfloat16)
    w2 = torch.randn(num_experts,
                     hidden_dim * 2,
                     hidden_dim,
                     dtype=torch.bfloat16)

    def mock_kernel_fn(hidden_states, *args, **kwargs):
        return hidden_states * 4.0

    with patch.object(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 16384), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.get_dp_group", return_value=dp_group), \
         patch("vllm_torchtpu.layers.adapter.pipelined_fused_moe.fused_moe_gmm", side_effect=mock_kernel_fn):
        out = pipelined_fused_moe_gmm(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=None,
            topk=topk,
            activation="silu",
        )

    assert out.shape == (seq_len, hidden_dim)
    torch.testing.assert_close(out, hidden_states * 4.0)
    ag_calls = [c for c in dp_group.call_log if c[0] == "all_gather"]
    rs_calls = [c for c in dp_group.call_log if c[0] == "reduce_scatter"]
    # 3 tensors * 4 chunks = 12 all_gather calls
    assert len(ag_calls) == 12
    # 4 reduce_scatter calls
    assert len(rs_calls) == 4
