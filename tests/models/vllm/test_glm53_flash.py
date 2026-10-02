"""Unit and serving tests for GLM 5.3 Flash (`zai-org/GLM-5.3-Flash`) on TorchTPU."""

from __future__ import annotations

import types
import unittest

import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops import (
    glm53_kda_conv_tpu,
    glm53_kda_vllm_layer_tpu,
    glm53_mhc_tpu,
)
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.glm5_next import model as glm5_next_model

pytestmark = pytest.mark.cpu_test


class TestGlm53FlashTorchTpu(unittest.TestCase):
    def test_model_registry_and_oot_kda_layer(self):
        register_models()
        from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
            KimiGatedDeltaNetAttention,
        )
        from vllm.model_executor.models.registry import ModelRegistry

        supported = ModelRegistry.get_supported_archs()
        self.assertIn("Glm5NextForConditionalGeneration", supported)
        self.assertIn("Glm5NextForCausalLM", supported)
        # The GLM entries share `register_models` with the other TPU models;
        # registering them must not displace any of those.
        for arch in (
            "KimiLinearForCausalLM",
            "KimiK3ForConditionalGeneration",
            "DeepseekV4ForCausalLM",
            "K3DSparkModel",
            "DSparkDraftModel",
        ):
            self.assertIn(arch, supported)
        self.assertIs(
            ModelRegistry.models["Glm5NextForConditionalGeneration"].load_model_cls(),
            glm5_next_model.Glm5NextForConditionalGeneration,
        )
        self.assertIs(
            ModelRegistry.models["Glm5NextForCausalLM"].load_model_cls(),
            glm5_next_model.Glm5NextForCausalLM,
        )

        obj = KimiGatedDeltaNetAttention.__new__(KimiGatedDeltaNetAttention)
        self.assertIsInstance(
            obj, glm53_kda_vllm_layer_tpu.TpuKimiGatedDeltaNetAttention
        )

    def test_glm5_text_config_schedule(self):
        raw_cfg = types.SimpleNamespace(
            hidden_size=4096,
            num_hidden_layers=45,
            vocab_size=154880,
            num_attention_heads=64,
            rms_norm_eps=1e-5,
        )
        cfg = glm5_next_model._ensure_glm5_text_config_attrs(raw_cfg)
        self.assertEqual(cfg.linear_attn_config["num_heads"], 64)
        self.assertEqual(cfg.linear_attn_config["head_dim"], 128)
        self.assertEqual(cfg.linear_attn_config["short_conv_kernel_size"], 4)
        self.assertEqual(sum(1 for t in cfg.layer_types if t == "linear_attention"), 34)
        self.assertEqual(
            sum(1 for t in cfg.layer_types if t == "deepseek_sparse_attention"),
            11,
        )
        self.assertEqual(sum(1 for t in cfg.mlp_layer_types if t == "dense"), 3)
        self.assertEqual(sum(1 for t in cfg.mlp_layer_types if t == "sparse"), 42)

    def test_mhc_vpu_fp32_stream_combine(self):
        mod = glm53_mhc_tpu.FusedGlm53HyperConnectionTpu(
            hidden_size=64, hc_mult=4, hc_sinkhorn_iters=20
        )
        with torch.no_grad():
            mod.fn.normal_(0, 0.02)
            mod.base.normal_(0, 0.1)
            mod.scale.fill_(1.0)
        streams = torch.randn(1, 4, 4, 64, dtype=torch.bfloat16)
        post, comb, normed = mod.collapse_and_norm(streams)
        self.assertEqual(post.shape, (1, 4, 4))
        self.assertEqual(comb.shape, (1, 4, 4, 4))
        self.assertEqual(normed.shape, (1, 4, 64))
        # Verify Sinkhorn-Knopp doubly-stochastic invariant (row & col sums == 1)
        torch.testing.assert_close(
            comb.sum(dim=-1), torch.ones_like(comb.sum(dim=-1)), rtol=0, atol=1e-4
        )
        torch.testing.assert_close(
            comb.sum(dim=-2), torch.ones_like(comb.sum(dim=-2)), rtol=0, atol=1e-4
        )
        out_streams = mod.combine(streams, normed, post, comb)
        self.assertEqual(out_streams.shape, (1, 4, 4, 64))

    def test_paged_causal_conv1d_update(self):
        x = torch.randn(4, 128, dtype=torch.bfloat16)
        conv_state = torch.randn(8, 128, 3, dtype=torch.bfloat16)
        weight = torch.randn(128, 4, dtype=torch.bfloat16)
        slots = torch.tensor([2, 0, 5, 1], dtype=torch.int32)
        out = glm53_kda_conv_tpu.causal_conv1d_update_paged(
            x, conv_state, weight, conv_state_indices=slots
        )
        self.assertEqual(out.shape, (4, 128))


if __name__ == "__main__":
    unittest.main()
