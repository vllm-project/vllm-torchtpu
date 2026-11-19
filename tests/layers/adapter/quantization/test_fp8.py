# Copyright 2025 Google LLC
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
Tests for FP8 quantization weight processing on TPU.

Verifies that process_weights_after_loading correctly transforms FP8
weights and scales for the GMM kernel, and that FP8 linear layers keep
runtime FP8 weights with dynamically quantized activations.

Run on CPU (default):
    pytest tests/layers/adapter/quantization/test_fp8.py -v

Run on TPU (when available):
    pytest tests/layers/adapter/quantization/test_fp8.py -v --use-tpu
"""

import jax.numpy as jnp
import pytest
import torch

from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
)


def test_fp8_config_accepts_default_store_dtype():
    config = VllmFp8Config.from_config({"activation_scheme": "dynamic"})
    assert config.store_dtype is None


def test_fp8_config_rejects_store_dtype():
    with pytest.raises(NotImplementedError, match="store_dtype"):
        VllmFp8Config.from_config(
            {
                "activation_scheme": "dynamic",
                "store_dtype": "mxfp4",
            }
        )


@pytest.mark.parametrize(
    "block_size,shard_size,deepseek_v4",
    [
        ([128, 128], 256, False),
        ([128, 128], 192, False),
        ([128, 128], 64, False),
        (None, 256, False),
        ([128, 128], 256, True),
    ],
)
def test_fp8_routed_experts_checkpoint_allocation(
    monkeypatch, block_size, shard_size, deepseek_v4
):
    """Exercise upstream construction and loading with TPU checkpoint geometry."""
    from math import gcd
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        MoEActivation,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    parallel = FusedMoEParallelConfig(
        tp_size=2,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=1,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=256,
        intermediate_size=2 * shard_size,
        num_local_experts=2,
        num_logical_experts=2,
        activation=MoEActivation.SILU,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    expert_map = ExpertMapManager(
        max_num_batched_tokens=16,
        top_k=1,
        global_num_experts=2,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy="linear",
        enable_eplb=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    quant_cls = VllmFp8Config
    if deepseek_v4:
        from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
            VllmDeepseekV4Fp8Config,
        )

        quant_cls = VllmDeepseekV4Fp8Config
    quant = quant_cls(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=block_size,
    )
    quant.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp8")),
    )
    with set_current_vllm_config(quant.vllm_config):
        layer = RoutedExperts(
            layer_name="model.layers.0.mlp.experts",
            params_dtype=torch.bfloat16,
            moe_config=config,
            quant_config=quant,
            expert_map_manager=expert_map,
        )

    assert isinstance(layer.quant_method, VllmFp8MoEMethodTPU)
    eff_block = gcd(128, shard_size) if block_size else None
    assert layer.w13_weight.shape == (2, 2 * shard_size, 256)
    assert layer.w2_weight.shape == (2, 256, shard_size)
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.float8_e4m3fn
    suffix = "weight_scale_inv" if block_size else "weight_scale"
    w13_scale = getattr(layer, f"w13_{suffix}")
    w2_scale = getattr(layer, f"w2_{suffix}")
    assert w13_scale.shape == (
        (2, 2 * shard_size // eff_block, 256 // eff_block) if block_size else (2, 2)
    )
    assert w2_scale.shape == (
        (2, 256 // eff_block, shard_size // eff_block) if block_size else (2,)
    )
    assert w13_scale.dtype == w2_scale.dtype == torch.float32
    assert layer.w13_weight.weight_loader == layer.weight_loader
    assert w13_scale.weight_loader == layer.weight_loader
    assert layer.w13_input_scale is None and layer.w2_input_scale is None
    if block_size and shard_size == 256:
        scales = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        layer.weight_loader(w2_scale, scales, "w2_weight_scale_inv", "w2", 0)
        torch.testing.assert_close(w2_scale[0], scales[:, 2:])


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tp_size,full_inter",
    [(1, 64), (1, 192), (4, 768), (8, 768), (16, 1024)],
)
def test_fp8_routed_experts_tp_shard_loading(monkeypatch, tp_size, full_inter):
    """Sub-block TP shards load without overflow and dequantize to the checkpoint."""
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        MoEActivation,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

    from vllm_torchtpu.layers.adapter.quantization.fp8 import _process_fp8_moe_weights

    hidden = 128
    shard = full_inter // tp_size
    padded = (shard + 127) // 128 * 128
    n_blocks = (full_inter + 127) // 128
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    gen = torch.Generator().manual_seed(0)
    w1 = torch.ones(full_inter, hidden, dtype=torch.float8_e4m3fn)
    w3 = torch.full((full_inter, hidden), 2.0, dtype=torch.float8_e4m3fn)
    w2 = torch.ones(hidden, full_inter, dtype=torch.float8_e4m3fn)
    s1 = torch.rand(n_blocks, hidden // 128, generator=gen) + 0.5
    s3 = torch.rand(n_blocks, hidden // 128, generator=gen) + 0.5
    s2 = torch.rand(hidden // 128, n_blocks, generator=gen) + 0.5
    ref_w1 = (
        w1.float()
        * s1.repeat_interleave(128, dim=0)[:full_inter].repeat_interleave(128, dim=1)[
            :, :hidden
        ]
    )
    ref_w3 = (
        w3.float()
        * s3.repeat_interleave(128, dim=0)[:full_inter].repeat_interleave(128, dim=1)[
            :, :hidden
        ]
    )
    ref_w2 = (
        w2.float()
        * s2.repeat_interleave(128, dim=0)[:hidden].repeat_interleave(128, dim=1)[
            :, :full_inter
        ]
    )

    for tp_rank in range(tp_size):
        parallel = FusedMoEParallelConfig(
            tp_size=tp_size,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=tp_rank,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        )
        config = FusedMoEConfig(
            num_experts=1,
            experts_per_token=1,
            hidden_dim=hidden,
            intermediate_size=full_inter,
            num_local_experts=1,
            num_logical_experts=1,
            activation=MoEActivation.SILU,
            device=torch.device("cpu"),
            routing_method=RoutingMethodType.Renormalize,
            moe_parallel_config=parallel,
            in_dtype=torch.bfloat16,
        )
        expert_map = ExpertMapManager(
            max_num_batched_tokens=16,
            top_k=1,
            global_num_experts=1,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=parallel,
            placement_strategy="linear",
            enable_eplb=False,
        )
        quant = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        quant.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp8")),
        )
        with set_current_vllm_config(quant.vllm_config):
            layer = RoutedExperts(
                layer_name="model.layers.0.mlp.experts",
                params_dtype=torch.bfloat16,
                moe_config=config,
                quant_config=quant,
                expert_map_manager=expert_map,
            )

        # Dirty the uninitialized weight buffers so tp_size=1 padding tests
        # verify that _process_fp8_moe_weights zeroes the padded tail.
        layer.w13_weight.data.fill_(7.0)
        layer.w2_weight.data.fill_(7.0)
        for sid, w, s in (("w1", w1, s1), ("w3", w3, s3), ("w2", w2, s2)):
            attr = "w2" if sid == "w2" else "w13"
            layer.weight_loader(
                getattr(layer, f"{attr}_weight"), w, f"{sid}.weight", sid, 0
            )
            layer.weight_loader(
                getattr(layer, f"{attr}_weight_scale_inv"),
                s,
                f"{sid}.weight_scale_inv",
                sid,
                0,
            )

        w13_q, w13_s, w2_q, w2_s, _, _ = _process_fp8_moe_weights(
            layer,
            weight_block_size=tuple(layer.quant_method.moe_block_shape),
            activation="silu",
        )
        sl = slice(tp_rank * shard, (tp_rank + 1) * shard)
        got_w13 = (w13_q[0].float() * w13_s[0, 0, 0]).T
        got_w2 = (w2_q[0].float() * w2_s[0, 0, 0]).T
        torch.testing.assert_close(got_w13[:shard], ref_w1[sl], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            got_w13[padded : padded + shard], ref_w3[sl], rtol=1e-2, atol=1e-2
        )
        assert not got_w13[shard:padded].any()
        assert not got_w13[padded + shard :].any()
        torch.testing.assert_close(
            got_w2[:, :shard], ref_w2[:, sl], rtol=1e-2, atol=1e-2
        )
        assert not got_w2[:, shard:].any()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tp_size,full_inter,expected_block",
    [
        (1, 192, [128, 128]),  # tp_size=1 non-aligned: padded to 256, not refined.
        (16, 1024, [64, 64]),  # Qwen3.5-397B shared geometry: shard 64.
        (8, 768, [32, 32]),  # Shard 96: the smallest block still allowed.
        (2, 512, [128, 128]),  # Shard 256: whole blocks, nothing to refine.
        (16, 640, None),  # Shard 40: gcd 8 < 32, refused.
    ],
)
def test_fp8_routed_experts_block_refine(
    monkeypatch, tp_size, full_inter, expected_block
):
    """Sub-block shards use upstream's refine: blocks under 32 are refused."""
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        MoEActivation,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    parallel = FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=1,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=full_inter,
        num_local_experts=1,
        num_logical_experts=1,
        activation=MoEActivation.SILU,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    expert_map = ExpertMapManager(
        max_num_batched_tokens=16,
        top_k=1,
        global_num_experts=1,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy="linear",
        enable_eplb=False,
    )
    quant = VllmFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    quant.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp8")),
    )

    def build():
        with set_current_vllm_config(quant.vllm_config):
            return RoutedExperts(
                layer_name="model.layers.0.mlp.experts",
                params_dtype=torch.bfloat16,
                moe_config=config,
                quant_config=quant,
                expert_map_manager=expert_map,
            )

    if expected_block is None:
        with pytest.raises(ValueError, match="no common block >= 32"):
            build()
        return

    layer = build()
    method = layer.quant_method
    assert list(method.moe_block_shape) == expected_block
    if tp_size == 1:
        assert (
            layer.moe_config.intermediate_size_per_partition
            == (full_inter + 127) // 128 * 128
        )
    if expected_block == [128, 128]:
        assert method.weight_scale_refine is None
    else:
        assert method.weight_scale_refine == (
            128 // expected_block[0],
            128 // expected_block[1],
        )


class TestFp8LinearSubBlockTP:
    """Tests for sub-block TP partitioning on FP8 block-quantized linear layers
    (e.g. Qwen3.5-397B-A17B-FP8 shared_expert with intermediate_size=1024 at TP=16).
    """

    @staticmethod
    def _mock_tp(monkeypatch, tp_size: int, tp_rank: int):
        for target in (
            "vllm.distributed",
            "vllm.distributed.parallel_state",
            "vllm.model_executor.layers.linear",
            "vllm.model_executor.parameter",
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
        ):
            monkeypatch.setattr(
                f"{target}.get_tensor_model_parallel_world_size",
                lambda: tp_size,
                raising=False,
            )
            monkeypatch.setattr(
                f"{target}.get_tensor_model_parallel_rank",
                lambda: tp_rank,
                raising=False,
            )

    @pytest.mark.cpu_test
    @pytest.mark.parametrize("fused_checkpoint", [False, True])
    def test_merged_column_parallel_tp16_shared_expert_gate_up(
        self, monkeypatch, fused_checkpoint
    ):
        """MergedColumnParallelLinear with output_sizes=[1024, 1024] at TP=16
        (64 rows per gate/up shard, smaller than block_size=128) must allocate
        separate scale blocks for gate and up and load the exact block scale for
        each of the 16 TP ranks without out-of-bounds slicing or overwriting.
        """
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            BlockQuantScaleParameterTPU,
        )
        from vllm_torchtpu.layers.core.quantization import dequantize_tensor

        tp_size = 16
        hidden = 256
        inter = 1024
        shard = inter // tp_size  # 64 < 128

        gen = torch.Generator().manual_seed(42)
        gate_w = torch.ones(inter, hidden, dtype=torch.float8_e4m3fn)
        up_w = torch.full((inter, hidden), 2.0, dtype=torch.float8_e4m3fn)
        gate_s = torch.arange(
            1, 1 + (inter // 128) * (hidden // 128), dtype=torch.float32
        ).reshape(inter // 128, hidden // 128)
        up_s = (
            torch.arange(
                100, 100 + (inter // 128) * (hidden // 128), dtype=torch.float32
            ).reshape(inter // 128, hidden // 128)
            + torch.rand(inter // 128, hidden // 128, generator=gen) * 0.1
        )

        ref_gate = dequantize_tensor(
            gate_w, gate_s, axis=(0, 1), out_dtype=torch.float32
        )
        ref_up = dequantize_tensor(up_w, up_s, axis=(0, 1), out_dtype=torch.float32)

        quant = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

        for tp_rank in range(tp_size):
            self._mock_tp(monkeypatch, tp_size, tp_rank)
            layer = MergedColumnParallelLinear(
                input_size=hidden,
                output_sizes=[inter, inter],
                bias=False,
                quant_config=quant,
                params_dtype=torch.bfloat16,
            )
            assert layer.weight.shape == (2 * shard, hidden)
            assert layer.weight_block_size == [64, 128]
            assert isinstance(layer.weight_scale_inv, BlockQuantScaleParameterTPU)
            assert layer.weight_scale_inv.shape == (2, hidden // 128)

            if fused_checkpoint:
                fused_w = torch.cat([gate_w, up_w], dim=0)
                fused_s = torch.cat([gate_s, up_s], dim=0)
                layer.weight.weight_loader(layer.weight, fused_w)
                layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, fused_s)
            else:
                layer.weight.weight_loader(layer.weight, gate_w, 0)
                layer.weight.weight_loader(layer.weight, up_w, 1)
                layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, gate_s, 0)
                layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, up_s, 1)

            expected_block_idx = tp_rank // 2
            torch.testing.assert_close(
                layer.weight_scale_inv.data[0], gate_s[expected_block_idx]
            )
            torch.testing.assert_close(
                layer.weight_scale_inv.data[1], up_s[expected_block_idx]
            )

            layer.quant_method.process_weights_after_loading(layer)
            assert layer.weight.shape == (hidden, 2 * shard)
            assert layer.weight_scale.shape == (2 * shard,)

            sl = slice(tp_rank * shard, (tp_rank + 1) * shard)
            got = (layer.weight.float() * layer.weight_scale[None, :]).T
            torch.testing.assert_close(got[:shard], ref_gate[sl], rtol=5e-2, atol=2e-1)
            torch.testing.assert_close(got[shard:], ref_up[sl], rtol=5e-2, atol=2e-1)

    @pytest.mark.cpu_test
    @pytest.mark.parametrize("enable_blockwise_kernel", [False, True])
    def test_row_parallel_tp16_shared_expert_down(
        self, monkeypatch, enable_blockwise_kernel
    ):
        """RowParallelLinear with input_size=1024 at TP=16 (input_size_per_partition=64,
        smaller than block_k=128) must pass block validation, refine the scale grid
        along input_dim, and fall back cleanly to per-channel quantization when
        input_size_per_partition < REQUANTIZE_BLOCK_SIZE.
        """
        from vllm.model_executor.layers.linear import RowParallelLinear

        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            BlockQuantScaleParameterTPU,
        )
        from vllm_torchtpu.layers.core.quantization import dequantize_tensor

        if enable_blockwise_kernel:
            monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
            monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "128")

        tp_size = 16
        inter = 1024
        hidden = 256
        shard = inter // tp_size  # 64 < 128

        down_w = torch.ones(hidden, inter, dtype=torch.float8_e4m3fn)
        down_s = torch.arange(
            1, 1 + (hidden // 128) * (inter // 128), dtype=torch.float32
        ).reshape(hidden // 128, inter // 128)
        ref_down = dequantize_tensor(
            down_w, down_s, axis=(0, 1), out_dtype=torch.float32
        )

        quant = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

        for tp_rank in range(tp_size):
            self._mock_tp(monkeypatch, tp_size, tp_rank)
            layer = RowParallelLinear(
                input_size=inter,
                output_size=hidden,
                bias=False,
                quant_config=quant,
                params_dtype=torch.bfloat16,
                reduce_results=False,
            )
            assert layer.weight.shape == (hidden, shard)
            assert layer.weight_block_size == [128, 64]
            assert isinstance(layer.weight_scale_inv, BlockQuantScaleParameterTPU)
            assert layer.weight_scale_inv.shape == (hidden // 128, 1)

            layer.weight.weight_loader(layer.weight, down_w)
            layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, down_s)

            expected_block_idx = tp_rank // 2
            torch.testing.assert_close(
                layer.weight_scale_inv.data[:, 0], down_s[:, expected_block_idx]
            )

            layer.quant_method.process_weights_after_loading(layer)
            assert layer.weight.shape == (shard, hidden)
            assert layer.weight_scale.shape == (hidden,)

            sl = slice(tp_rank * shard, (tp_rank + 1) * shard)
            got = (layer.weight.float() * layer.weight_scale[None, :]).T
            torch.testing.assert_close(got, ref_down[:, sl], rtol=5e-2, atol=2e-1)

    @pytest.mark.cpu_test
    def test_column_parallel_tp16_sub_block(self, monkeypatch):
        """ColumnParallelLinear with output_size=1024 at TP=16 (64 rows per shard)."""
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        from vllm_torchtpu.layers.core.quantization import dequantize_tensor

        tp_size = 16
        out_size = 1024
        in_size = 256
        shard = out_size // tp_size  # 64

        w = torch.ones(out_size, in_size, dtype=torch.float8_e4m3fn)
        s = torch.arange(
            1, 1 + (out_size // 128) * (in_size // 128), dtype=torch.float32
        ).reshape(out_size // 128, in_size // 128)
        ref = dequantize_tensor(w, s, axis=(0, 1), out_dtype=torch.float32)

        quant = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

        for tp_rank in range(tp_size):
            self._mock_tp(monkeypatch, tp_size, tp_rank)
            layer = ColumnParallelLinear(
                input_size=in_size,
                output_size=out_size,
                bias=False,
                quant_config=quant,
                params_dtype=torch.bfloat16,
                gather_output=False,
            )
            assert layer.weight.shape == (shard, in_size)
            assert layer.weight_scale_inv.shape == (1, in_size // 128)

            layer.weight.weight_loader(layer.weight, w)
            layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, s)
            torch.testing.assert_close(layer.weight_scale_inv.data[0], s[tp_rank // 2])

            layer.quant_method.process_weights_after_loading(layer)
            sl = slice(tp_rank * shard, (tp_rank + 1) * shard)
            got = (layer.weight.float() * layer.weight_scale[None, :]).T
            torch.testing.assert_close(got, ref[sl], rtol=5e-2, atol=2e-1)

    @pytest.mark.cpu_test
    def test_qkv_parallel_tp16_sub_block_with_kv_replication(self, monkeypatch):
        """QKVParallelLinear with KV head replication and sub-block shards at TP=16."""
        from vllm.model_executor.layers.linear import QKVParallelLinear

        tp_size = 16
        head_size = 64
        total_num_heads = 16  # 1024 total Q rows -> 64 per TP rank
        total_num_kv_heads = 4  # 256 total K/V rows -> 64 per TP rank (4x replicated)
        hidden = 256

        q_w = torch.ones(total_num_heads * head_size, hidden, dtype=torch.float8_e4m3fn)
        k_w = torch.full(
            (total_num_kv_heads * head_size, hidden), 2.0, dtype=torch.float8_e4m3fn
        )
        v_w = torch.full(
            (total_num_kv_heads * head_size, hidden), 3.0, dtype=torch.float8_e4m3fn
        )
        q_s = torch.arange(1, 1 + 8 * 2, dtype=torch.float32).reshape(8, 2)
        k_s = torch.arange(50, 50 + 2 * 2, dtype=torch.float32).reshape(2, 2)
        v_s = torch.arange(90, 90 + 2 * 2, dtype=torch.float32).reshape(2, 2)

        quant = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

        for tp_rank in range(tp_size):
            self._mock_tp(monkeypatch, tp_size, tp_rank)
            layer = QKVParallelLinear(
                hidden_size=hidden,
                head_size=head_size,
                total_num_heads=total_num_heads,
                total_num_kv_heads=total_num_kv_heads,
                bias=False,
                quant_config=quant,
                params_dtype=torch.bfloat16,
            )
            assert layer.weight.shape == (192, hidden)
            assert layer.weight_scale_inv.shape == (3, hidden // 128)

            for sid, w, s in (("q", q_w, q_s), ("k", k_w, k_s), ("v", v_w, v_s)):
                layer.weight.weight_loader(layer.weight, w, sid)
                layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, s, sid)

            kv_group = tp_rank // (tp_size // total_num_kv_heads)
            torch.testing.assert_close(
                layer.weight_scale_inv.data[0], q_s[tp_rank // 2]
            )
            torch.testing.assert_close(
                layer.weight_scale_inv.data[1], k_s[kv_group // 2]
            )
            torch.testing.assert_close(
                layer.weight_scale_inv.data[2], v_s[kv_group // 2]
            )


class FakeQuant:
    """Minimal quant config for testing."""

    def __init__(
        self,
        weight_block_size=None,
        activation_scheme="dynamic",
        is_checkpoint_fp8_serialized=True,
    ):
        self.weight_block_size = weight_block_size
        self.activation_scheme = activation_scheme
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized


class FakeMoEConfig:
    """Minimal MoE config for testing."""

    def __init__(self, has_bias=False, experts_per_token=2):
        self.has_bias = has_bias
        self.experts_per_token = experts_per_token


class FakeActivation:
    """Mimics MoEActivation enum."""

    def __init__(self, value):
        self.value = value


class FakeFusedMoELayer(torch.nn.Module):
    """Minimal FusedMoE-like layer for testing process_weights_after_loading."""

    def __init__(
        self,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        activation="silu",
        has_bias=False,
        experts_per_token=2,
    ):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        self.w13_weight_scale_inv = torch.nn.Parameter(w13_scale, requires_grad=False)
        self.w2_weight_scale_inv = torch.nn.Parameter(w2_scale, requires_grad=False)
        self.activation = FakeActivation(activation)
        self.moe_config = FakeMoEConfig(
            has_bias=has_bias, experts_per_token=experts_per_token
        )

    # Make isinstance(layer, FusedMoE) work via duck typing in the assert
    def __class_getitem__(cls, item):
        return cls


@pytest.mark.tpu_test
class TestFp8MoEScaleReshape:
    """Tests for block-quantized FP8 MoE scale reshaping."""

    @pytest.mark.parametrize(
        "num_experts,intermediate,hidden,block_size",
        [
            (4, 256, 512, 128),
            (8, 384, 768, 128),
            (2, 128, 256, 128),
        ],
    )
    def test_scale_output_shape(
        self, device, num_experts, intermediate, hidden, block_size
    ):
        """Verify scales are reshaped to [E, in/B, 1, out_dim] for GMM."""
        E, inter, H, B = num_experts, intermediate, hidden, block_size

        # Input scale shape: [E, 2*inter/B, H/B] (w13) and [E, H/B, inter/B] (w2)
        w13_scale = torch.randn(E, 2 * inter // B, H // B, device=device)
        w2_scale = torch.randn(E, H // B, inter // B, device=device)

        # Simulate the reshape logic from process_weights_after_loading
        # Step 1: transpose [E, out/B, in/B] -> [E, in/B, out/B]
        w13_out = w13_scale.transpose(1, 2)
        # Step 2: expand by block_h then reshape
        w13_out = (
            w13_out.unsqueeze(-1)
            .expand(*w13_out.shape, B)
            .reshape(w13_out.shape[0], w13_out.shape[1], -1)
        )
        # Step 3: unsqueeze for GMM
        w13_out = w13_out.unsqueeze(2).to(torch.float32)

        # Expected: [E, H/B, 1, 2*inter]
        assert w13_out.shape == (E, H // B, 1, 2 * inter)

        # Same for w2
        w2_out = w2_scale.transpose(1, 2)
        w2_out = (
            w2_out.unsqueeze(-1)
            .expand(*w2_out.shape, B)
            .reshape(w2_out.shape[0], w2_out.shape[1], -1)
        )
        w2_out = w2_out.unsqueeze(2).to(torch.float32)

        assert w2_out.shape == (E, inter // B, 1, H)

    def test_scale_values_are_repeated(self, device):
        """Each block scale should be repeated block_h times in output dim."""
        E, B = 2, 128
        out_blocks, in_blocks = 4, 2  # out=512, in=256

        scale = torch.arange(
            E * out_blocks * in_blocks, dtype=torch.float32, device=device
        ).reshape(E, out_blocks, in_blocks)

        # Apply reshape
        out = scale.transpose(1, 2)
        out = (
            out.unsqueeze(-1)
            .expand(*out.shape, B)
            .reshape(out.shape[0], out.shape[1], -1)
        )
        out = out.unsqueeze(2)

        # Shape: [E, in_blocks, 1, out_blocks * B]
        assert out.shape == (E, in_blocks, 1, out_blocks * B)

        # Check each block_h-sized chunk has the same value
        for e in range(E):
            for ib in range(in_blocks):
                for ob in range(out_blocks):
                    chunk = out[e, ib, 0, ob * B : (ob + 1) * B]
                    assert torch.all(chunk == chunk[0]), (
                        f"Scale not constant within block at e={e}, ib={ib}, ob={ob}"
                    )


@pytest.mark.tpu_test
class TestFp8LinearRuntimeQuant:
    """Tests for ullm-style runtime FP8 dense linear weights."""

    def test_block_dequant_shape(self, device):
        """Runtime FP8 weight should preserve original shape."""
        out_dim, in_dim = 256, 512
        block_h, block_w = 128, 128

        weight_fp8 = torch.randn(
            out_dim, in_dim, device=device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        scale_inv = torch.ones(
            out_dim // block_h, in_dim // block_w, device=device, dtype=torch.float32
        )

        layer = torch.nn.Module()
        layer.logical_widths = None
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scale_inv, requires_grad=False)

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[block_h, block_w]))
        method.process_weights_after_loading(layer)

        # Canonical (k, n): the runtime matmul is (m, k) @ (k, n), so the
        # checkpoint's [n_out, n_in] weight is stored transposed.
        assert layer.weight.shape == (in_dim, out_dim)
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.shape == (out_dim,)
        assert layer.weight_scale.dtype == torch.float32
        assert not hasattr(layer, "weight_scale_inv")

    def test_block_dequant_applies_scales(self, device):
        """Block scales should actually scale the weights."""
        out_dim, in_dim = 256, 256
        block_h, block_w = 128, 128

        # Use ones as FP8 weight — after dequant, result should equal scale
        weight_fp8 = torch.ones(
            out_dim, in_dim, device=device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        # Set different scales per block
        scale_inv = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], device=device, dtype=torch.float32
        )

        layer = torch.nn.Module()
        layer.logical_widths = None
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scale_inv, requires_grad=False)

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[block_h, block_w]))
        method.process_weights_after_loading(layer)

        # Weight is stored (k, n), so it indexes [in, out] and the
        # per-output-channel scale broadcasts along the trailing axis.
        w = layer.weight.float() * layer.weight_scale[None, :]
        # fp8 ones = 1.0, so each block should be scaled by its scale_inv.
        # w[i, o] == w_checkpoint[o, i], so the off-diagonal blocks swap
        # relative to the checkpoint's [out, in] grid:
        # scale_inv[0,0] -> w[:128, :128]  ~1.0
        # scale_inv[1,0] -> w[:128, 128:]  ~3.0
        # scale_inv[0,1] -> w[128:, :128]  ~2.0
        # scale_inv[1,1] -> w[128:, 128:]  ~4.0
        assert torch.allclose(
            w[:128, :128].mean(), torch.tensor(1.0, device=device), rtol=0.06, atol=0.05
        )
        assert torch.allclose(
            w[:128, 128:].mean(), torch.tensor(3.0, device=device), rtol=0.06, atol=0.05
        )
        assert torch.allclose(
            w[128:, :128].mean(), torch.tensor(2.0, device=device), rtol=0.06, atol=0.05
        )
        assert torch.allclose(
            w[128:, 128:].mean(), torch.tensor(4.0, device=device), rtol=0.06, atol=0.05
        )

    def test_per_tensor_dequant(self, device):
        """Per-tensor (non-block) dequant should scale entire weight."""
        out_dim, in_dim = 64, 128

        weight_fp8 = torch.ones(
            out_dim, in_dim, device=device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        scale = torch.tensor([2.5], device=device, dtype=torch.float32)

        layer = torch.nn.Module()
        layer.logical_widths = None
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=None))
        method.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.shape == (out_dim,)
        # fp8 ones * 2.5 should be ~2.5
        # (k, n) storage: scale is per output channel, so it broadcasts
        # along the trailing axis.
        runtime_deq = layer.weight.float() * layer.weight_scale[None, :]
        assert torch.allclose(
            runtime_deq.mean(), torch.tensor(2.5, device=device), atol=0.1
        )

    def test_per_tensor_dequant_with_logical_widths(self, device):
        """Per-tensor dequant with multiple scales should use
        logical_widths repeat_interleave.
        """
        # Simulate QKVParallelLinear with logical_widths [64, 32, 32]
        # (total out_dim = 128)
        logical_widths = [64, 32, 32]
        out_dim = sum(logical_widths)
        in_dim = 128

        weight_fp8 = torch.ones(
            out_dim, in_dim, device=device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        scale = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        layer.logical_widths = logical_widths

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=None))
        method.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.shape == (out_dim,)

        w = layer.weight.float() * layer.weight_scale[:, None]
        assert torch.allclose(w[:64].mean(), torch.tensor(1.0, device=device), atol=0.1)
        assert torch.allclose(
            w[64:96].mean(), torch.tensor(2.0, device=device), atol=0.1
        )
        assert torch.allclose(w[96:].mean(), torch.tensor(3.0, device=device), atol=0.1)

    def test_apply_is_linear(self, device):
        """apply() should use the runtime FP8 quantized matmul."""
        if device.type != "tpu":
            pytest.skip("Pallas quantized matmul bridge requires TPU.")

        out_dim, in_dim = 64, 128
        batch = 4

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[128, 128]))

        layer = torch.nn.Module()
        # apply() consumes the canonical (k, n) layout that
        # process_weights_after_loading produces, i.e. [n_in, n_out].
        layer.weight = torch.nn.Parameter(
            torch.ones(in_dim, out_dim, device=device, dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(out_dim, device=device, dtype=torch.float32), requires_grad=False
        )

        x = torch.ones(batch, in_dim, device=device, dtype=torch.bfloat16)
        result = method.apply(layer, x)

        assert result.shape == (batch, out_dim)
        expected = torch.full(
            (batch, out_dim), float(in_dim), device=device, dtype=torch.bfloat16
        )
        assert torch.allclose(result, expected, rtol=0.01, atol=0.01)

    def test_blockwise_runtime_scale_shape(self, device, monkeypatch):
        """ullm blockwise env gate changes dense-linear runtime scale layout."""
        monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "128")

        out_dim, in_dim = 256, 256
        block_h, block_w = 128, 128

        layer = torch.nn.Module()
        layer.logical_widths = None
        layer.weight = torch.nn.Parameter(
            torch.ones(out_dim, in_dim, device=device, dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        layer.weight_scale_inv = torch.nn.Parameter(
            torch.ones(
                out_dim // block_h,
                in_dim // block_w,
                device=device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[block_h, block_w]))
        method.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        # gmm_v2 rhs_scale layout: [size_group, num_blocks, 1, out_size].
        assert layer.weight_scale.shape == (1, in_dim // 128, 1, out_dim)

    def test_blockwise_kernel_requires_block_size(self, monkeypatch):
        """Match ullm: enabling the blockwise kernel requires block size."""
        monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
        monkeypatch.delenv("REQUANTIZE_BLOCK_SIZE", raising=False)

        with pytest.raises(ValueError, match="REQUANTIZE_BLOCK_SIZE"):
            VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[128, 128]))

    def test_block_size_requires_blockwise_kernel(self, monkeypatch):
        """Match ullm: REQUANTIZE_BLOCK_SIZE only applies to the kernel path."""
        monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "0")
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "128")

        with pytest.raises(ValueError, match="Blockwise quantization"):
            VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[128, 128]))

    def test_blockwise_scale_output_dim_validated(self):
        """Blockwise matmul should reject scales with the wrong output dim."""
        x = jnp.ones((2, 4), dtype=jnp.bfloat16)
        w_q = jnp.ones((4, 3), dtype=jnp.float8_e4m3fn)  # (k, n)
        w_scale = jnp.ones((1, 1, 1, 2), dtype=jnp.float32)  # n=2 != 3

        with pytest.raises(ValueError, match="output dim"):
            _quantized_matmul_jax(x, w_q, w_scale)

    def test_blockwise_scale_block_count_validated(self):
        """Blockwise matmul should reject ambiguous scale block counts."""
        x = jnp.ones((2, 5), dtype=jnp.bfloat16)
        w_q = jnp.ones((5, 3), dtype=jnp.float8_e4m3fn)  # (k, n), k=5
        w_scale = jnp.ones((1, 2, 1, 3), dtype=jnp.float32)  # 5 % 2 != 0

        with pytest.raises(ValueError, match="divisible by block scale count"):
            _quantized_matmul_jax(x, w_q, w_scale)


@pytest.mark.tpu_test
class TestFp8WeightTranspose:
    """Tests for weight transpose in MoE processing."""

    def test_weights_transposed(self, device):
        """Weights should be transposed from [E, out, in] to [E, in, out]."""
        E, inter, H = 4, 256, 512

        w13 = torch.randn(E, 2 * inter, H, device=device, dtype=torch.bfloat16)
        w2 = torch.randn(E, H, inter, device=device, dtype=torch.bfloat16)

        w13_t = w13.transpose(1, 2).contiguous()
        w2_t = w2.transpose(1, 2).contiguous()

        assert w13_t.shape == (E, H, 2 * inter)
        assert w2_t.shape == (E, inter, H)
        # Verify the transpose is correct
        assert torch.equal(w13_t[0, :, 0], w13[0, 0, :])

    def test_swigluoai_deinterleave(self, device):
        """swigluoai activation should de-interleave w13 weights."""
        E, inter, H = 2, 128, 256

        # Create w13 with known interleaved pattern
        w13 = torch.zeros(E, 2 * inter, H, device=device, dtype=torch.bfloat16)
        # Even rows (gate/w1) = 1.0, odd rows (up/w3) = 2.0
        w13[:, ::2, :] = 1.0
        w13[:, 1::2, :] = 2.0

        # De-interleave
        w1 = w13[:, ::2, :]
        w3 = w13[:, 1::2, :]
        w13_deinterleaved = torch.cat([w1, w3], dim=1)

        # First half should be all 1.0 (w1), second half all 2.0 (w3)
        assert torch.all(w13_deinterleaved[:, :inter, :] == 1.0)
        assert torch.all(w13_deinterleaved[:, inter:, :] == 2.0)

    def test_swigluoai_deinterleave_scale(self, device):
        """swigluoai should de-interleave w13_scale to match w13."""
        E, inter = 2, 128

        # Create scale with known interleaved pattern: one scale per row
        w13_scale = torch.zeros(E, 2 * inter, 1, device=device)
        w13_scale[:, ::2, :] = 1.0  # w1 scales
        w13_scale[:, 1::2, :] = 2.0  # w3 scales

        s1 = w13_scale[:, ::2, :]
        s3 = w13_scale[:, 1::2, :]
        w13_scale_deinterleaved = torch.cat([s1, s3], dim=1)

        # First half should be w1 scales (1.0), second half w3 scales (2.0)
        assert torch.all(w13_scale_deinterleaved[:, :inter, :] == 1.0)
        assert torch.all(w13_scale_deinterleaved[:, inter:, :] == 2.0)


@pytest.mark.tpu_test
class TestOnlineFp8Quantization:
    """Tests for _quantize_bf16_moe_weights."""

    def test_roundtrip_error(self, device):
        """FP8 quantize -> dequantize should have small error."""
        E, inter, H = 4, 64, 128
        w13 = torch.randn(E, 2 * inter, H, device=device, dtype=torch.float32)

        from vllm_torchtpu.layers.core.quantization import quantize_tensor

        w13_q, w13_s = quantize_tensor(w13, quant_dtype=torch.float8_e4m3fn, axis=-1)
        w13_deq = w13_q.to(torch.float32) * w13_s

        rel_err = (w13 - w13_deq).abs().mean() / (w13.abs().mean() + 1e-8)
        assert rel_err < 0.05, f"Roundtrip relative error too high: {rel_err:.4f}"

    def test_output_shapes(self, device):
        """_quantize_bf16_moe_weights should produce correct shapes."""
        from unittest.mock import MagicMock

        from vllm.model_executor.layers.fused_moe import RoutedExperts

        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            _quantize_bf16_moe_weights,
        )

        E, inter, H = 4, 64, 128
        layer = MagicMock(spec=RoutedExperts)
        layer.w13_weight = torch.nn.Parameter(
            torch.randn(E, 2 * inter, H, dtype=torch.bfloat16)
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.randn(E, H, inter, dtype=torch.bfloat16)
        )

        w13, w13_s, w2, w2_s, dtype_name, block_size = _quantize_bf16_moe_weights(
            layer, activation="silu"
        )

        # _quantize_and_format_moe_weights rounds the intermediate size up to
        # a multiple of 128 before quantizing, padding w13/w2 accordingly.
        aligned_inter = (inter + 127) // 128 * 128

        # Weights should be transposed: [E, in, out]
        assert w13.shape == (E, H, 2 * aligned_inter)
        assert w2.shape == (E, aligned_inter, H)
        assert w13.dtype == torch.float8_e4m3fn
        assert w2.dtype == torch.float8_e4m3fn

        # Scales should be 4D: [E, num_blocks, 1, N]
        assert w13_s.shape == (E, 1, 1, 2 * aligned_inter)
        assert w2_s.shape == (E, 1, 1, H)
        assert w13_s.dtype == torch.float32

    def test_custom_routing_called(self):
        """FP8 apply_monolithic should use custom_routing_function when present."""
        from unittest.mock import MagicMock, patch

        from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

        # Create mock layer with custom routing
        layer = MagicMock()
        layer.moe_config.experts_per_token = 2
        layer.moe_config.moe_parallel_config.use_ep = False
        layer.renormalize = True

        mock_routing = MagicMock(
            return_value=(torch.ones(4, 2), torch.zeros(4, 2, dtype=torch.int32))
        )
        layer.custom_routing_function = mock_routing

        method = MagicMock(spec=VllmFp8MoEMethodTPU)
        # MagicMock does not inherit the class attribute's False value.
        method._use_adaptive_fused_moe = False
        method._tpu_activation_str = "silu"
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)

        # Call the real apply_monolithic
        with patch(
            "vllm_torchtpu.layers.adapter.quantization.fp8.fused_moe_gmm",
            return_value=x,
        ):
            VllmFp8MoEMethodTPU.apply_monolithic(method, layer, x, router_logits)

        mock_routing.assert_called_once()


class FakeInt8MoEConfig:
    """The `moe_config` attributes `_process_fp8_moe_weights` actually reads."""

    def __init__(self, intermediate):
        self.intermediate_size_per_partition = intermediate
        self.intermediate_size_per_partition_unpadded = intermediate
        self.tp_size = 1
        self.tp_rank = 0
        self.has_bias = False


class TestInt8MoERequantization:
    """`MOE_REQUANTIZE_WEIGHT_DTYPE=int8` end to end through the loader.

    int8 is a supported value of that env var but no recipe, CI config or other
    test exercises it, which is how the truncating cast in `quantize_tensor`
    survived. This runs the real `_process_fp8_moe_weights`. CPU-only, so it
    takes no `device` fixture.
    """

    E, INTER, H, BLOCK = 2, 128, 384, 128

    def _layer(self):
        gen = torch.Generator().manual_seed(0)
        E, inter, H, B = self.E, self.INTER, self.H, self.BLOCK
        layer = FakeFusedMoELayer(
            (torch.randn(E, 2 * inter, H, generator=gen) * 40).to(torch.float8_e4m3fn),
            (torch.randn(E, H, inter, generator=gen) * 40).to(torch.float8_e4m3fn),
            torch.rand(E, 2 * inter // B, H // B, generator=gen) + 0.5,
            torch.rand(E, H // B, inter // B, generator=gen) + 0.5,
        )
        layer.moe_config = FakeInt8MoEConfig(inter)
        return layer

    def test_int8_requant_produces_int8_weights_rounded_to_nearest(self, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            _process_fp8_moe_weights,
        )
        from vllm_torchtpu.layers.core.quantization import dequantize_tensor

        monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "int8")
        layer = self._layer()

        w13, w13_scale, w2, _, dtype_name, block_size = _process_fp8_moe_weights(
            layer, weight_block_size=(self.BLOCK, self.BLOCK), activation="silu"
        )

        assert dtype_name == "int8" and block_size is None
        assert w13.dtype == torch.int8 and w2.dtype == torch.int8
        # The GMM kernel wants [E, in, out]; the loader transposes on the way.
        assert w13.shape == (self.E, self.H, 2 * self.INTER)
        assert w2.shape == (self.E, self.INTER, self.H)

        # Per-channel scale over the contracting axis, broadcast back to [E, out, in].
        scale = w13_scale.reshape(self.E, 1, -1).transpose(1, 2)
        reference = dequantize_tensor(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            axis=(1, 2),
            out_dtype=torch.float32,
        )
        err_lsb = (w13.float().transpose(1, 2) * scale - reference).abs() / scale
        # Round-to-nearest bounds this by half an LSB; truncation gives a full one.
        assert err_lsb.max().item() <= 0.5 + 1e-4


def _native_fp4_values(w):
    """[E, K, N] float32 values of a float4_e2m1fn_x2 [E, K, N/2] weight (the
    pair of each byte along N)."""
    import jax
    import numpy as np

    u8 = jnp.asarray(w.view(torch.uint8).numpy())
    fp4 = jax.lax.bitcast_convert_type(u8, jnp.float4_e2m1fn)
    return np.asarray(fp4.astype(jnp.float32)).reshape(*w.shape[:2], -1)


def _kblock_fp4_reference(param, scale, block):
    """The dequantized FP8 checkpoint weight [E, N, K] quantized to FP4 per
    `block` elements of K in that orientation (absmax / 6 scales, as XLA
    rounds them), laid out K-major: values [E, K, N] and scales
    [E, K / block, 1, N], float32."""
    import jax
    import numpy as np

    from vllm_torchtpu.layers.core.quantization import dequantize_tensor

    @jax.jit
    def quantize(w):
        e, n, k = w.shape
        blocks = w.reshape(e, n, k // block, block)
        s = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True) / 6.0
        inv = jnp.where(s == 0, 0.0, 1.0 / s)
        q = jnp.clip(blocks * inv, -6.0, 6.0).astype(jnp.float4_e2m1fn)
        return q.astype(jnp.float32).reshape(e, n, k), s[..., 0]

    w = dequantize_tensor(param.data, scale.data, axis=(1, 2), out_dtype=torch.float32)
    vals, s = quantize(jnp.asarray(w.numpy()))
    return (
        np.asarray(vals).transpose(0, 2, 1),
        np.asarray(s).transpose(0, 2, 1)[:, :, None, :],
    )


def _cpu_jax_op(_name, fn, **_kwargs):
    """`pallas.jax_op` on CPU: the JAX function on numpy views of the tensors."""
    import jax
    import numpy as np

    def to_jax(t):
        if t.dtype == torch.bfloat16:
            u16 = t.view(torch.uint16).numpy()
            return jax.lax.bitcast_convert_type(jnp.asarray(u16), jnp.bfloat16)
        return jnp.asarray(t.numpy())

    def to_torch(a):
        if a.dtype == jnp.float4_e2m1fn:
            pairs = a.reshape(*a.shape[:-1], a.shape[-1] // 2, 2)
            u8 = np.array(jax.lax.bitcast_convert_type(pairs, jnp.uint8))
            return torch.from_numpy(u8).view(torch.float4_e2m1fn_x2)
        return torch.from_numpy(np.array(a))

    class _Op:
        def __call__(self, *args):
            out = fn(*(to_jax(a) for a in args))
            if isinstance(out, tuple):
                return tuple(to_torch(o) for o in out)
            return to_torch(out)

    return _Op()


class TestFp4MoERequantization:
    """`MOE_REQUANTIZE_WEIGHT_DTYPE=fp4` through the real loader functions.

    A block-FP8 checkpoint's routed experts become K-major FP4 with one scale
    per 512 rows of K: packed two to a byte for the GMM path, unpacked
    `float4_e2m1fn_x2` for the fused EP W4A8 kernel. MTP layers stay FP8
    unless VLLM_TPU_MOE_REQUANT_MTP is set. CPU-only: the torch_tpu op
    boundary is replaced by a JAX call on CPU.
    """

    E, INTER, H, BLOCK = 2, 512, 512, 128

    @pytest.fixture(autouse=True)
    def _cpu_ops(self, monkeypatch):
        if not hasattr(torch, "float4_e2m1fn_x2"):
            pytest.skip("torch.float4_e2m1fn_x2 unavailable")
        from vllm_torchtpu.layers.adapter import fused_moe
        from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

        monkeypatch.setattr(fused_moe.pallas, "jax_op", _cpu_jax_op)
        monkeypatch.setattr(fused_moe, "_quantize_native_fp4_kmajor_ops", {})
        monkeypatch.setattr(fp8_mod, "fresh_buffer", lambda t: t)
        monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")
        monkeypatch.delenv("MOE_REQUANTIZE_BLOCK_SIZE", raising=False)

    def _layer(self, inter=INTER, H=H):
        gen = torch.Generator().manual_seed(0)
        E, B = self.E, self.BLOCK
        layer = FakeFusedMoELayer(
            (torch.randn(E, 2 * inter, H, generator=gen) * 40).to(torch.float8_e4m3fn),
            (torch.randn(E, H, inter, generator=gen) * 40).to(torch.float8_e4m3fn),
            torch.rand(E, 2 * inter // B, H // B, generator=gen) + 0.5,
            torch.rand(E, H // B, inter // B, generator=gen) + 0.5,
        )
        layer.moe_config = FakeInt8MoEConfig(inter)
        return layer

    @staticmethod
    def _unpack_kmajor(w, scale, packed):
        """[E, K, N] float32 from the loader's FP4 weight and [E, K/b, 1, N] scale."""
        import jax
        import numpy as np

        if packed:  # uint8 [E, K/2, N], the pair of each byte along K
            fp4 = jax.lax.bitcast_convert_type(
                jnp.asarray(w.numpy()), jnp.float4_e2m1fn
            )
            fp4 = jnp.swapaxes(fp4, -1, -2)  # [E, K/2, 2, N]
            vals = np.asarray(fp4.astype(jnp.float32)).reshape(
                w.shape[0], -1, w.shape[2]
            )
        else:
            vals = _native_fp4_values(w)
        block = vals.shape[1] // scale.shape[1]
        return vals * np.repeat(scale[:, :, 0, :].float().numpy(), block, axis=1)

    @pytest.mark.parametrize("fused_ep_w4a8", [False, True])
    def test_fp8_checkpoint_to_kmajor_fp4(self, monkeypatch, fused_ep_w4a8):
        import numpy as np

        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            _process_fp8_moe_weights,
        )
        from vllm_torchtpu.layers.core.quantization import dequantize_tensor

        if fused_ep_w4a8:
            monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "1")
            monkeypatch.setenv("MOE_FUSED_EP_ENABLE_W4A8", "1")
        layer = self._layer()

        w13, w13_scale, w2, w2_scale, dtype_name, block_size = _process_fp8_moe_weights(
            layer, weight_block_size=(self.BLOCK, self.BLOCK), activation="silu"
        )

        E, inter, H = self.E, self.INTER, self.H
        assert dtype_name == "fp4" and block_size == 512
        if fused_ep_w4a8:
            assert w13.dtype == torch.float4_e2m1fn_x2
            assert w13.shape == (E, H, inter) and w2.shape == (E, inter, H // 2)
        else:
            assert w13.dtype == torch.uint8
            assert w13.shape == (E, H // 2, 2 * inter) and w2.shape == (
                E,
                inter // 2,
                H,
            )
        assert w13_scale.shape == (E, H // 512, 1, 2 * inter)
        assert w2_scale.shape == (E, inter // 512, 1, H)

        for got_w, got_s, param, scale in (
            (w13, w13_scale, layer.w13_weight, layer.w13_weight_scale_inv),
            (w2, w2_scale, layer.w2_weight, layer.w2_weight_scale_inv),
        ):
            want = dequantize_tensor(
                param.data, scale.data, axis=(1, 2), out_dtype=torch.float32
            )
            want = want.transpose(1, 2).numpy()  # K-major [E, K, N]
            got = self._unpack_kmajor(got_w, got_s, packed=not fused_ep_w4a8)
            rel_err = np.linalg.norm(got - want) / np.linalg.norm(want)
            assert rel_err < 0.22

    @pytest.mark.parametrize(
        "inter, w2_block",
        [
            pytest.param(256, 256, id="ep2xtp8-shard"),
            pytest.param(512, 512, id="ep2xtp4-shard"),
        ],
    )
    def test_moe_tp_decode_fp4_layout(self, monkeypatch, inter, w2_block):
        """With USE_MOE_TP_DECODE_NVFP4_KERNEL an admitted layer gets the layout
        moe_tp_decode_nvfp4 reads: native FP4, w13 at block 512 and w2 at
        min(512, I). Block 512 alone cannot tile the EP2xTP8 shard's I = 256.
        The values are those of a K-blocked quantization of the checkpoint in
        its own [E, N, K] orientation."""
        import numpy as np

        from vllm_torchtpu.layers.adapter import moe_tp_decode_nvfp4
        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            _process_fp8_moe_weights,
        )

        prechecks = []

        def admit(layer, *, requant_block):
            prechecks.append(requant_block)

        monkeypatch.setattr(moe_tp_decode_nvfp4, "precheck_admission", admit)
        monkeypatch.setenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", "1")
        layer = self._layer(inter=inter, H=1024)

        w13, w13_scale, w2, w2_scale, _, block_size = _process_fp8_moe_weights(
            layer, weight_block_size=(self.BLOCK, self.BLOCK), activation="silu"
        )

        assert prechecks == [512] and block_size == 512
        for got_w, got_s, param, scale, block in (
            (w13, w13_scale, layer.w13_weight, layer.w13_weight_scale_inv, 512),
            (w2, w2_scale, layer.w2_weight, layer.w2_weight_scale_inv, w2_block),
        ):
            assert got_w.dtype == torch.float4_e2m1fn_x2
            want_vals, want_scale = _kblock_fp4_reference(param, scale, block)
            assert np.array_equal(_native_fp4_values(got_w), want_vals)
            assert np.array_equal(got_s.numpy(), want_scale)

    @pytest.mark.parametrize("case", ["refused", "fused_ep_w4a8"])
    def test_moe_tp_decode_fp4_layout_only_where_it_arms(self, monkeypatch, case):
        """A layer the precheck refuses, or one the fused EP W4A8 kernel is asked
        to serve, keeps exactly the layout it gets without the flag."""
        from vllm_torchtpu.layers.adapter import moe_tp_decode_nvfp4
        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            _process_fp8_moe_weights,
        )

        if case == "fused_ep_w4a8":
            monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "1")
            monkeypatch.setenv("MOE_FUSED_EP_ENABLE_W4A8", "1")
        layer = self._layer()
        kwargs = dict(weight_block_size=(self.BLOCK, self.BLOCK), activation="silu")
        want = _process_fp8_moe_weights(layer, **kwargs)
        prechecks = []

        def refuse(layer, *, requant_block):
            prechecks.append(requant_block)
            return "refused by the test"

        monkeypatch.setattr(moe_tp_decode_nvfp4, "precheck_admission", refuse)
        monkeypatch.setenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", "1")
        got = _process_fp8_moe_weights(layer, **kwargs)

        assert prechecks == ([512] if case == "refused" else [])
        assert got[4:] == want[4:]
        for a, b in zip(want[:4], got[:4]):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    def test_mtp_layers_stay_fp8(self, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            _get_moe_quant_config,
            _mtp_moe_keeps_fp8,
            _process_fp8_moe_weights,
        )

        mtp = "mtp.layers.48.mlp.experts"
        assert _mtp_moe_keeps_fp8(mtp)
        assert _mtp_moe_keeps_fp8("model.mtp.layers.0.mlp.experts")
        assert not _mtp_moe_keeps_fp8("model.layers.47.mlp.experts")
        monkeypatch.setenv("VLLM_TPU_MOE_REQUANT_MTP", "1")
        assert not _mtp_moe_keeps_fp8(mtp)
        monkeypatch.delenv("VLLM_TPU_MOE_REQUANT_MTP")
        # Without FP4 requantization an MTP layer is loaded like the others.
        monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn")
        assert not _mtp_moe_keeps_fp8(mtp)
        monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")

        assert _get_moe_quant_config("test", force_fp8=False)[1] == (
            torch.float4_e2m1fn_x2
        )
        assert _get_moe_quant_config("test", force_fp8=True) == (
            "float8_e4m3fn",
            torch.float8_e4m3fn,
            None,
        )

        w13, w13_scale, w2, _, dtype_name, block_size = _process_fp8_moe_weights(
            self._layer(),
            weight_block_size=(self.BLOCK, self.BLOCK),
            activation="silu",
            force_fp8=True,
        )
        assert dtype_name == "float8_e4m3fn" and block_size is None
        assert w13.dtype == torch.float8_e4m3fn and w2.dtype == torch.float8_e4m3fn
        assert w13_scale.shape == (self.E, 1, 1, 2 * self.INTER)

    def test_fused_scale_buffers_read_scale_inv(self):
        from vllm_torchtpu.layers.adapter.fused_moe_ep import (
            register_fused_scale_buffers,
        )

        layer = torch.nn.Module()
        layer.w13_weight_scale_inv = torch.nn.Parameter(
            torch.ones((4, 20, 1, 1280)), requires_grad=False
        )
        layer.w2_weight_scale_inv = torch.nn.Parameter(
            torch.full((4, 5, 1, 2560), 2.0), requires_grad=False
        )
        register_fused_scale_buffers(layer)
        assert layer._tpu_fused_w13_scale.shape == (4, 20, 1280)
        assert layer._tpu_fused_w2_scale.shape == (4, 5, 2560)
        assert torch.equal(
            layer._tpu_fused_w2_scale, layer.w2_weight_scale_inv.squeeze(2)
        )


class TestMtpMoELayerDetection:
    """Which MoE layers belong to an MTP draft model.

    vLLM's DeepSeek-style MTP drafters (GLM-5 included) name their layers
    `model.layers.<i>` with `i` counted from the target's num_hidden_layers
    up, so GLM-5's drafter MoE is `model.layers.78.mlp.experts` next to the
    target's last MoE layer `model.layers.77.mlp.experts`.
    """

    NUM_HIDDEN_LAYERS = 78
    MTP = "model.layers.78.mlp.experts"
    MAIN = "model.layers.77.mlp.experts"

    @staticmethod
    def _current_config(monkeypatch, num_hidden_layers):
        from types import SimpleNamespace

        from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=num_hidden_layers)
            )
        )
        monkeypatch.setattr(fp8_mod, "get_current_vllm_config_or_none", lambda: config)

    def test_layer_index_at_or_past_num_hidden_layers_is_mtp(self):
        from vllm_torchtpu.layers.adapter.quantization.fp8 import _is_mtp_moe_layer

        n = self.NUM_HIDDEN_LAYERS
        assert _is_mtp_moe_layer(self.MTP, n)
        assert _is_mtp_moe_layer("model.layers.79.mlp.experts", n)
        assert not _is_mtp_moe_layer(self.MAIN, n)
        assert not _is_mtp_moe_layer("model.layers.3.mlp.experts", n)
        assert not _is_mtp_moe_layer("model.layers.780x.mlp.experts", n)

    def test_mtp_prefix_forms_match_without_num_hidden_layers(self):
        from vllm_torchtpu.layers.adapter.quantization.fp8 import _is_mtp_moe_layer

        for prefix in ("mtp.layers.0.mlp.experts", "model.mtp.layers.0.mlp.experts"):
            assert _is_mtp_moe_layer(prefix)
            assert _is_mtp_moe_layer(prefix, self.NUM_HIDDEN_LAYERS)
        # No vLLM config is current here: an index alone cannot tell.
        assert not _is_mtp_moe_layer(self.MTP)

    def test_num_hidden_layers_defaults_to_the_current_config(self, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization.fp8 import _is_mtp_moe_layer

        self._current_config(monkeypatch, self.NUM_HIDDEN_LAYERS)
        assert _is_mtp_moe_layer(self.MTP)
        assert not _is_mtp_moe_layer(self.MAIN)

    def test_drafter_moe_keeps_fp8_under_fp4_requant(self, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization.fp8 import _mtp_moe_keeps_fp8

        if not hasattr(torch, "float4_e2m1fn_x2"):
            pytest.skip("torch.float4_e2m1fn_x2 unavailable")
        n = self.NUM_HIDDEN_LAYERS
        monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")
        monkeypatch.delenv("VLLM_TPU_MOE_REQUANT_MTP", raising=False)
        assert _mtp_moe_keeps_fp8(self.MTP, n)
        assert _mtp_moe_keeps_fp8("mtp.layers.0.mlp.experts", n)
        assert not _mtp_moe_keeps_fp8(self.MAIN, n)
        monkeypatch.setenv("VLLM_TPU_MOE_REQUANT_MTP", "1")
        assert not _mtp_moe_keeps_fp8(self.MTP, n)
        monkeypatch.delenv("VLLM_TPU_MOE_REQUANT_MTP")
        monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn")
        assert not _mtp_moe_keeps_fp8(self.MTP, n)

    def test_method_reads_num_hidden_layers_when_built(self, monkeypatch):
        from types import SimpleNamespace

        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            VllmFp8MoEMethodTPU,
        )

        self._current_config(monkeypatch, self.NUM_HIDDEN_LAYERS)
        method = VllmFp8MoEMethodTPU(
            SimpleNamespace(weight_block_size=[128, 128]),
            SimpleNamespace(),
            prefix=self.MTP,
        )
        assert method._num_hidden_layers == self.NUM_HIDDEN_LAYERS


def test_concat_fp4_experts_op_has_annotated_signature():
    """`pallas.jax_op` needs a `jax.Array` annotation on every argument of
    the JAX function: the op that joins unpacked FP4 expert chunks cannot be
    a `*xs` lambda."""
    import inspect
    import uuid

    import jax
    from torch_tpu._internal import pallas

    from vllm_torchtpu.layers.adapter.quantization.fp8 import _build_concat_fp4_fn

    fn = _build_concat_fp4_fn(3)
    sig = inspect.signature(fn)
    assert list(sig.parameters) == ["x0", "x1", "x2"]
    for param in sig.parameters.values():
        assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert param.annotation is jax.Array
    assert sig.return_annotation is jax.Array
    out = fn(
        jnp.ones((2, 4), jnp.int8),
        jnp.zeros((3, 4), jnp.int8),
        jnp.ones((1, 4), jnp.int8),
    )
    assert out.shape == (6, 4)

    suffix = uuid.uuid4().hex[:8]
    with pytest.raises(ValueError, match="Missing argument type annotation"):
        pallas.jax_op(
            f"pallas::test_fp4_concat_lambda_{suffix}",
            lambda *xs: jnp.concatenate(xs, axis=0),
        )
    pallas.jax_op(f"pallas::test_fp4_concat_typed_{suffix}", fn)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.tpu_test
class TestMoERequantizationChunks:
    """Requantizing a few experts at a time must give the whole-tensor
    result."""

    @staticmethod
    def _layer(device, dtype, num_experts=6, inter=256, hidden=512, block=128):
        from types import SimpleNamespace

        def param(t):
            return torch.nn.Parameter(t, requires_grad=False)

        w13 = torch.randn(
            num_experts, 2 * inter, hidden, device=device, dtype=torch.bfloat16
        ).to(dtype)
        w2 = torch.randn(
            num_experts, hidden, inter, device=device, dtype=torch.bfloat16
        ).to(dtype)
        layer = SimpleNamespace(
            w13_weight=param(w13),
            w2_weight=param(w2),
            w13_weight_scale_inv=param(
                torch.rand(
                    num_experts, 2 * inter // block, hidden // block, device=device
                )
                + 0.5
            ),
            w2_weight_scale_inv=param(
                torch.rand(num_experts, hidden // block, inter // block, device=device)
                + 0.5
            ),
            moe_config=SimpleNamespace(
                intermediate_size_per_partition=inter,
                intermediate_size_per_partition_unpadded=inter,
                tp_size=1,
                tp_rank=0,
            ),
        )
        # w13 chunks of two experts: 2 * (2*inter*hidden) float32 bytes
        return layer, 2 * 2 * inter * hidden * 4

    @staticmethod
    def _same(whole, chunked):
        for a, b in zip(whole[:4], chunked[:4]):
            assert a.shape == b.shape
            assert torch.equal(a.float(), b.float())

    def test_fp8_checkpoint(self, device, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

        torch.manual_seed(0)
        layer, chunk_bytes = self._layer(device, torch.float8_e4m3fn)
        kwargs = dict(weight_block_size=(128, 128), activation="silu")
        whole = fp8_mod._process_fp8_moe_weights(layer, **kwargs)
        monkeypatch.setattr(fp8_mod, "_REQUANT_CHUNK_BYTES", chunk_bytes)
        chunked = fp8_mod._process_fp8_moe_weights(layer, **kwargs)
        self._same(whole, chunked)

    def test_bf16_checkpoint(self, device, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

        torch.manual_seed(0)
        layer, chunk_bytes = self._layer(device, torch.bfloat16)
        whole = fp8_mod._quantize_bf16_moe_weights(layer, activation="silu")
        monkeypatch.setattr(fp8_mod, "_REQUANT_CHUNK_BYTES", chunk_bytes)
        chunked = fp8_mod._quantize_bf16_moe_weights(layer, activation="silu")
        self._same(whole, chunked)
