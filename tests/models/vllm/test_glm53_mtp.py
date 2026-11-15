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
"""Unit tests for the GLM-5.3-Flash Multi-Token Prediction (MTP) draft model."""

from __future__ import annotations

import inspect
import types
import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np
import torch
from vllm.config import CompilationMode, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.registry import ModelRegistry

from vllm_torchtpu import gdn_pool_layout
from vllm_torchtpu.layers.adapter.custom_ops import (
    glm53_collectives_tpu,
    glm53_mhc_tpu,
)
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.glm5_next import config as glm5_next_config
from vllm_torchtpu.models.vllm.glm5_next import model as glm5_next_model
from vllm_torchtpu.models.vllm.glm5_next import mtp as glm5_next_mtp
from vllm_torchtpu.spec_decode.utils import (
    iter_mtp_shared_heads,
    maybe_share_embeddings,
    maybe_share_lm_head,
)


def _ensure_dist_initialized() -> None:
    import socket

    import torch.distributed as dist
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    if not dist.is_initialized():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)


class TestGlm53MtpDraftModel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        from vllm.platforms import current_platform

        for attr in ("device_name", "device_type"):
            p = mock.patch.object(current_platform, attr, "cpu")
            p.start()
            self.addCleanup(p.stop)

    def test_mtp_model_and_config_registration(self) -> None:
        register_models()
        supported = ModelRegistry.get_supported_archs()
        self.assertIn("Glm5NextMTPModel", supported)
        self.assertIs(
            ModelRegistry.models["Glm5NextMTPModel"].load_model_cls(),
            glm5_next_mtp.Glm5NextMTP,
        )
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP

        for arch in ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"):
            self.assertIs(MODELS_CONFIG_MAP[arch], glm5_next_config.TPUGlm5NextConfig)

        none_cfg = types.SimpleNamespace(num_nextn_predict_layers=None)
        self.assertEqual(glm5_next_mtp.glm53_num_mtp_layers(none_cfg), 1)
        self.assertEqual(
            glm5_next_mtp.glm53_num_mtp_layers(
                types.SimpleNamespace(num_nextn_predict_layers=None, n_predict=3)
            ),
            3,
        )

    def test_spec_decode_config_hook(self) -> None:
        from vllm.model_executor.models.config import Glm5NextModelConfig

        hook = glm5_next_config.TPUGlm5NextConfig
        cache = types.SimpleNamespace(
            cache_dtype="auto", enable_prefix_caching=True, mamba_cache_mode="align"
        )
        vcfg = types.SimpleNamespace(
            speculative_config=object(),
            model_config=types.SimpleNamespace(
                architecture="Glm5NextForConditionalGeneration",
                hf_text_config=types.SimpleNamespace(
                    qk_rope_head_dim=0,
                    mhc=True,
                    num_attention_heads=64,
                    num_key_value_heads=64,
                    linear_num_heads=64,
                    index_topk=2048,
                    index_kpool=64,
                    index_kpool_always_select_tail=True,
                ),
                max_model_len=4096,
            ),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=8),
            cache_config=cache,
        )
        with (
            mock.patch.object(
                Glm5NextModelConfig,
                "verify_and_update_config",
                staticmethod(lambda vc: None),
            ),
            mock.patch.object(
                gdn_pool_layout,
                "unified_kv_layout_enabled_for_architecture",
                return_value=True,
            ),
        ):
            hook.verify_and_update_config(vcfg)
        self.assertTrue(cache.enable_prefix_caching)
        self.assertEqual(cache.mamba_cache_mode, "align")

        with (
            mock.patch.object(
                Glm5NextModelConfig,
                "verify_and_update_config",
                staticmethod(lambda vc: None),
            ),
            mock.patch.object(
                gdn_pool_layout,
                "unified_kv_layout_enabled_for_architecture",
                return_value=False,
            ),
        ):
            hook.verify_and_update_config(vcfg)
        self.assertFalse(cache.enable_prefix_caching)
        self.assertEqual(cache.mamba_cache_mode, "none")

        # Without spec decode nothing changes.
        cache.enable_prefix_caching = True
        vcfg.speculative_config = None
        with mock.patch.object(
            Glm5NextModelConfig,
            "verify_and_update_config",
            staticmethod(lambda vc: None),
        ):
            hook.verify_and_update_config(vcfg)
        self.assertTrue(cache.enable_prefix_caching)

    def test_normalize_mtp_checkpoint_name_mapping(self) -> None:
        norm = glm5_next_mtp.Glm5NextMTP._normalize_mtp_checkpoint_name
        cases = [
            (
                "model.language_model.layers.45.enorm.weight",
                "model.layers.45.enorm.weight",
            ),
            (
                "model.language_model.layers.45.hnorm.weight",
                "model.layers.45.hnorm.weight",
            ),
            (
                "model.language_model.layers.45.eh_proj.weight",
                "model.layers.45.eh_proj.weight",
            ),
            (
                "model.language_model.layers.45.shared_head.norm.weight",
                "model.layers.45.shared_head.norm.weight",
            ),
            (
                "model.language_model.layers.45.shared_head.head.weight",
                "model.layers.45.shared_head.head.weight",
            ),
            (
                "model.language_model.layers.45.embed_tokens.weight",
                "model.embed_tokens.weight",
            ),
            (
                "model.language_model.layers.45.input_layernorm.weight",
                "model.layers.45.mtp_block.input_layernorm.weight",
            ),
            (
                "model.language_model.layers.45.post_attention_layernorm.weight",
                "model.layers.45.mtp_block.post_attention_layernorm.weight",
            ),
            (
                "model.language_model.layers.45.self_attn.kv_b_proj.weight",
                "model.layers.45.mtp_block.self_attn.kv_b_proj.weight",
            ),
            (
                "model.language_model.layers.45.self_attn.q_a_proj.weight_scale_inv",
                "model.layers.45.mtp_block.self_attn.q_a_proj.weight_scale_inv",
            ),
            (
                "model.language_model.layers.45.mlp.gate.weight",
                "model.layers.45.mtp_block.mlp.gate.weight",
            ),
            (
                "model.language_model.layers.45.mlp.gate.e_score_correction_bias",
                "model.layers.45.mtp_block.mlp.gate.e_score_correction_bias",
            ),
            (
                "model.language_model.layers.45.mlp.experts.0.gate_proj.weight",
                "model.layers.45.mtp_block.mlp.experts.0.gate_proj.weight",
            ),
            (
                "model.language_model.layers.45.mlp.shared_experts.gate_proj.weight",
                "model.layers.45.mtp_block.mlp.shared_experts.gate_proj.weight",
            ),
            # Skipped: DSA indexer weights, target layers 0..44, visual tower,
            # and non-layer weights.
            ("model.language_model.layers.45.self_attn.indexer.wk.weight", None),
            (
                "model.language_model.layers.45.self_attn.indexer.weights_proj.weight",
                None,
            ),
            ("model.language_model.layers.44.self_attn.q_a_proj.weight", None),
            ("model.language_model.layers.0.mlp.down_proj.weight", None),
            ("model.language_model.embed_tokens.weight", None),
            ("model.visual.blocks.0.attn.qkv.weight", None),
        ]
        for ckpt_name, expected in cases:
            with self.subTest(ckpt_name=ckpt_name):
                self.assertEqual(norm(ckpt_name, 45, 1), expected)

    def test_fused_eh_norm_and_add_rms_norm_match_jax_and_torch(self) -> None:
        for fn in (
            glm5_next_mtp._fused_eh_norm_jax,
            glm5_next_mtp._fused_add_rms_norm_jax,
        ):
            for name, param in inspect.signature(fn).parameters.items():
                self.assertIsNot(
                    param.annotation,
                    inspect.Parameter.empty,
                    f"{fn.__name__}.{name} is missing type annotation",
                )

        torch.manual_seed(42)
        tokens, hidden, eps = 6, 128, 1e-5
        positions_1d = torch.tensor([0, 1, 2, 0, 5, 6], dtype=torch.int32)
        positions_2d = torch.stack(
            [positions_1d + 10, positions_1d + 20, positions_1d], dim=0
        )
        embeds = (torch.randn(tokens, hidden) * 2.0).to(torch.bfloat16)
        prev_h = (torch.randn(tokens, hidden) * 2.0).to(torch.bfloat16)
        enorm_w = (torch.rand(hidden) + 0.5).to(torch.bfloat16)
        hnorm_w = (torch.rand(hidden) + 0.5).to(torch.bfloat16)

        out_1d = glm5_next_mtp.fused_eh_norm_tpu(
            positions_1d, embeds, prev_h, enorm_w, hnorm_w, eps
        )
        out_2d = glm5_next_mtp.fused_eh_norm_tpu(
            positions_2d, embeds, prev_h, enorm_w, hnorm_w, eps
        )
        torch.testing.assert_close(out_1d, out_2d, rtol=0, atol=0)
        self.assertEqual(out_1d.shape, (tokens, hidden * 2))

        self.assertTrue(torch.all(out_1d[0, :hidden] == 0))
        self.assertTrue(torch.all(out_1d[3, :hidden] == 0))
        expected_h = glm53_mhc_tpu._rms_norm_weighted_fn(prev_h, hnorm_w, eps)
        torch.testing.assert_close(out_1d[:, hidden:], expected_h, rtol=0, atol=0)

        def to_jax(t: torch.Tensor) -> jnp.ndarray:
            if t.dtype == torch.bfloat16:
                return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
            return jnp.asarray(t.numpy())

        j_eh = glm5_next_mtp._fused_eh_norm_jax(
            to_jax(positions_1d),
            to_jax(embeds),
            to_jax(prev_h),
            to_jax(enorm_w),
            to_jax(hnorm_w),
            eps=eps,
        )
        torch.testing.assert_close(
            torch.from_numpy(np.asarray(j_eh.astype(jnp.float32)).copy()),
            out_1d.float(),
            rtol=1e-2,
            atol=1e-2,
        )

        x = (torch.randn(tokens, hidden) * 1.5).to(torch.bfloat16)
        residual = (torch.randn(tokens, hidden) * 1.5).to(torch.bfloat16)
        post_w = (torch.rand(hidden) + 0.5).to(torch.bfloat16)
        normed_t, res_t = glm5_next_mtp.fused_add_rms_norm_tpu(x, residual, post_w, eps)
        normed_j, res_j = glm5_next_mtp._fused_add_rms_norm_jax(
            to_jax(x), to_jax(residual), to_jax(post_w), eps=eps
        )
        torch.testing.assert_close(
            torch.from_numpy(np.asarray(res_j.astype(jnp.float32)).copy()),
            res_t.float(),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            torch.from_numpy(np.asarray(normed_j.astype(jnp.float32)).copy()),
            normed_t.float(),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_glm5_mtp_construction_weight_loading_and_forward(self) -> None:
        from vllm.platforms import current_platform

        _ensure_dist_initialized()
        hidden = 32
        inter = 48
        vocab = 64
        heads = 2
        q_lora = 16
        kv_lora = 12
        qk_nope = 8
        v_hd = 8
        n_target_layers = 45

        hf_config = types.SimpleNamespace(
            hidden_size=hidden,
            intermediate_size=inter,
            moe_intermediate_size=16,
            vocab_size=vocab,
            num_hidden_layers=n_target_layers,
            num_nextn_predict_layers=1,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=0,
            v_head_dim=v_hd,
            rms_norm_eps=1e-5,
            swiglu_limit=10.0,
            n_routed_experts=0,
            mlp_layer_types=["dense"] * n_target_layers,
        )
        vllm_config = VllmConfig()
        vllm_config.compilation_config.mode = CompilationMode.NONE
        vllm_config.speculative_config = types.SimpleNamespace(
            num_speculative_tokens=1,
            draft_model_config=types.SimpleNamespace(
                hf_config=hf_config,
                compute_hash=lambda: "test_mtp_draft_model",
            ),
            compute_hash=lambda: "test_mtp_spec",
        )

        class _FakeRpaAttention(torch.nn.Module):
            def __init__(self, num_heads: int, head_size: int, **kwargs) -> None:
                super().__init__()
                del kwargs
                self.num_heads = num_heads
                self.head_size = head_size

            def forward(
                self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                return (q + k + v)[:, : self.num_heads * self.head_size]

        with (
            mock.patch.object(current_platform, "device_name", "cpu"),
            mock.patch.object(current_platform, "device_type", "cpu"),
            mock.patch.object(glm5_next_model, "Attention", _FakeRpaAttention),
            set_current_vllm_config(vllm_config),
        ):
            mtp_model = glm5_next_mtp.Glm5NextMTP(vllm_config=vllm_config, prefix="")

        mtp_layer = mtp_model.model.layers[str(n_target_layers)]
        heads_found = list(iter_mtp_shared_heads(mtp_model))
        self.assertEqual(len(heads_found), 1)
        self.assertIs(heads_found[0], mtp_layer.shared_head)

        self.assertEqual(
            mtp_layer.mtp_block.self_attn.prefix,
            f"model.layers.{n_target_layers}.self_attn",
        )

        torch.manual_seed(7)
        ckpt_prefix = f"model.language_model.layers.{n_target_layers}"
        ckpt_weights = {
            f"{ckpt_prefix}.enorm.weight": torch.rand(hidden) + 0.5,
            f"{ckpt_prefix}.hnorm.weight": torch.rand(hidden) + 0.5,
            f"{ckpt_prefix}.eh_proj.weight": torch.randn(hidden, hidden * 2) * 0.05,
            f"{ckpt_prefix}.shared_head.norm.weight": torch.rand(hidden) + 0.5,
            f"{ckpt_prefix}.input_layernorm.weight": torch.rand(hidden) + 0.5,
            f"{ckpt_prefix}.post_attention_layernorm.weight": torch.rand(hidden) + 0.5,
            f"{ckpt_prefix}.self_attn.q_a_proj.weight": torch.randn(q_lora, hidden)
            * 0.05,
            f"{ckpt_prefix}.self_attn.q_a_layernorm.weight": torch.rand(q_lora) + 0.5,
            f"{ckpt_prefix}.self_attn.q_b_proj.weight": torch.randn(
                heads * qk_nope, q_lora
            )
            * 0.05,
            f"{ckpt_prefix}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(
                kv_lora, hidden
            )
            * 0.05,
            f"{ckpt_prefix}.self_attn.kv_a_layernorm.weight": torch.rand(kv_lora) + 0.5,
            f"{ckpt_prefix}.self_attn.kv_b_proj.weight": torch.randn(
                heads * (qk_nope + v_hd), kv_lora
            )
            * 0.05,
            f"{ckpt_prefix}.self_attn.o_proj.weight": torch.randn(hidden, heads * v_hd)
            * 0.05,
            f"{ckpt_prefix}.mlp.gate_proj.weight": torch.randn(inter, hidden) * 0.05,
            f"{ckpt_prefix}.mlp.up_proj.weight": torch.randn(inter, hidden) * 0.05,
            f"{ckpt_prefix}.mlp.down_proj.weight": torch.randn(hidden, inter) * 0.05,
            f"{ckpt_prefix}.self_attn.indexer.wk.weight": torch.randn(8, hidden),
            "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(
                hidden, heads * v_hd
            ),
        }
        loaded = mtp_model.load_weights(ckpt_weights.items())
        self.assertIn(f"model.layers.{n_target_layers}.enorm.weight", loaded)
        self.assertIn(f"model.layers.{n_target_layers}.eh_proj.weight", loaded)
        self.assertIn(
            f"model.layers.{n_target_layers}.mtp_block.self_attn.kv_b_proj.weight",
            loaded,
        )
        self.assertIn(
            f"model.layers.{n_target_layers}.mtp_block.mlp.gate_up_proj.weight",
            loaded,
        )
        mtp_model.process_weights_after_loading()

        target_embed = torch.nn.Embedding(vocab, hidden)
        target_lm_head = ParallelLMHead(vocab, hidden)
        fake_target = types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=target_embed),
            lm_head=target_lm_head,
        )
        maybe_share_embeddings(mtp_model, fake_target, tp_layout_matches=True)
        maybe_share_lm_head(
            mtp_model, fake_target, draft_replicated=False, tp_layout_matches=True
        )
        self.assertIs(
            mtp_model.model.embed_tokens.weight,
            target_embed.weight,
        )
        self.assertIs(
            mtp_layer.shared_head.head.weight,
            target_lm_head.weight,
        )

        input_ids = torch.tensor([3, 7, 11, 15], dtype=torch.long)
        positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        prev_hidden = torch.randn(4, hidden)
        hidden_out, state_out = mtp_model(
            input_ids=input_ids,
            positions=positions,
            hidden_states=prev_hidden,
        )
        self.assertEqual(hidden_out.shape, (4, hidden))
        torch.testing.assert_close(hidden_out, state_out, rtol=0, atol=0)
        logits = mtp_model.compute_logits(hidden_out)
        self.assertIsNotNone(logits)
        assert logits is not None
        self.assertEqual(logits.shape, (4, vocab))

    def test_glm5_mtp_sparse_moe_weight_loading(self) -> None:
        from vllm.platforms import current_platform

        _ensure_dist_initialized()
        hidden = 16
        inter = 24
        moe_inter = 8
        vocab = 32
        heads = 2
        q_lora = 8
        kv_lora = 8
        qk_nope = 4
        v_hd = 4
        n_target_layers = 45
        n_experts = 4

        hf_config = types.SimpleNamespace(
            hidden_size=hidden,
            intermediate_size=inter,
            moe_intermediate_size=moe_inter,
            vocab_size=vocab,
            num_hidden_layers=n_target_layers,
            num_nextn_predict_layers=1,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=0,
            v_head_dim=v_hd,
            rms_norm_eps=1e-5,
            swiglu_limit=10.0,
            n_routed_experts=n_experts,
            num_experts_per_tok=2,
            n_shared_experts=1,
            routed_scaling_factor=2.5,
            norm_topk_prob=True,
            scoring_func="sigmoid",
            mlp_layer_types=["sparse"] * n_target_layers,
        )
        vllm_config = VllmConfig()
        vllm_config.compilation_config.mode = CompilationMode.NONE
        vllm_config.speculative_config = types.SimpleNamespace(
            num_speculative_tokens=1,
            draft_model_config=types.SimpleNamespace(
                hf_config=hf_config,
                compute_hash=lambda: "test_mtp_draft_model",
            ),
            compute_hash=lambda: "test_mtp_spec",
        )

        class _FakeRpaAttention(torch.nn.Module):
            def __init__(self, num_heads: int, head_size: int, **kwargs) -> None:
                super().__init__()
                del num_heads, head_size, kwargs

        with (
            mock.patch.object(current_platform, "device_name", "cpu"),
            mock.patch.object(current_platform, "device_type", "cpu"),
            mock.patch.object(glm5_next_model, "Attention", _FakeRpaAttention),
            set_current_vllm_config(vllm_config),
        ):
            mtp_model = glm5_next_mtp.Glm5NextMTP(vllm_config=vllm_config, prefix="")

        mtp_layer = mtp_model.model.layers[str(n_target_layers)]
        self.assertIsInstance(mtp_layer.mtp_block.mlp, glm5_next_model.Glm5NextMoE)
        self.assertEqual(mtp_model.num_moe_layers, 1)
        self.assertIs(mtp_model.moe_mlp_layers[0], mtp_layer.mtp_block.mlp)
        self.assertIs(mtp_model.moe_layers[0], mtp_layer.mtp_block.mlp.experts)
        self.assertEqual(mtp_model.num_routed_experts, mtp_model.num_logical_experts)
        self.assertGreater(mtp_model.num_logical_experts, 0)
        mtp_model.update_physical_experts_metadata(mtp_model.num_logical_experts + 2, 3)
        self.assertEqual(mtp_model.num_local_physical_experts, 3)

        ckpt_prefix = f"model.language_model.layers.{n_target_layers}"
        ckpt_weights: dict[str, torch.Tensor] = {
            f"{ckpt_prefix}.enorm.weight": torch.ones(hidden),
            f"{ckpt_prefix}.hnorm.weight": torch.ones(hidden),
            f"{ckpt_prefix}.eh_proj.weight": torch.zeros(hidden, hidden * 2),
            f"{ckpt_prefix}.shared_head.norm.weight": torch.ones(hidden),
            f"{ckpt_prefix}.input_layernorm.weight": torch.ones(hidden),
            f"{ckpt_prefix}.post_attention_layernorm.weight": torch.ones(hidden),
            f"{ckpt_prefix}.self_attn.q_a_proj.weight": torch.zeros(q_lora, hidden),
            f"{ckpt_prefix}.self_attn.q_a_layernorm.weight": torch.ones(q_lora),
            f"{ckpt_prefix}.self_attn.q_b_proj.weight": torch.zeros(
                heads * qk_nope, q_lora
            ),
            f"{ckpt_prefix}.self_attn.kv_a_proj_with_mqa.weight": torch.zeros(
                kv_lora, hidden
            ),
            f"{ckpt_prefix}.self_attn.kv_a_layernorm.weight": torch.ones(kv_lora),
            f"{ckpt_prefix}.self_attn.kv_b_proj.weight": torch.zeros(
                heads * (qk_nope + v_hd), kv_lora
            ),
            f"{ckpt_prefix}.self_attn.o_proj.weight": torch.zeros(hidden, heads * v_hd),
            f"{ckpt_prefix}.mlp.gate.weight": torch.zeros(n_experts, hidden),
            f"{ckpt_prefix}.mlp.gate.e_score_correction_bias": torch.zeros(n_experts),
            f"{ckpt_prefix}.mlp.shared_experts.gate_proj.weight": torch.zeros(
                moe_inter, hidden
            ),
            f"{ckpt_prefix}.mlp.shared_experts.up_proj.weight": torch.zeros(
                moe_inter, hidden
            ),
            f"{ckpt_prefix}.mlp.shared_experts.down_proj.weight": torch.zeros(
                hidden, moe_inter
            ),
        }
        for exp_idx in range(n_experts):
            ckpt_weights[f"{ckpt_prefix}.mlp.experts.{exp_idx}.gate_proj.weight"] = (
                torch.zeros(moe_inter, hidden)
            )
            ckpt_weights[f"{ckpt_prefix}.mlp.experts.{exp_idx}.up_proj.weight"] = (
                torch.zeros(moe_inter, hidden)
            )
            ckpt_weights[f"{ckpt_prefix}.mlp.experts.{exp_idx}.down_proj.weight"] = (
                torch.zeros(hidden, moe_inter)
            )

        loaded = mtp_model.load_weights(ckpt_weights.items())
        self.assertIn(
            f"model.layers.{n_target_layers}.mtp_block.mlp.experts.routed_experts.w13_weight",
            loaded,
        )
        self.assertIn(
            f"model.layers.{n_target_layers}.mtp_block.mlp.experts.routed_experts.w2_weight",
            loaded,
        )
        self.assertIn(
            f"model.layers.{n_target_layers}.mtp_block.mlp.shared_experts.gate_up_proj.weight",
            loaded,
        )

    def test_mtp_compiled_logits_all_gather(self) -> None:
        from vllm.platforms import current_platform

        _ensure_dist_initialized()
        hidden = 16
        inter = 24
        vocab = 32
        heads = 2
        n_target_layers = 45

        hf_config = types.SimpleNamespace(
            hidden_size=hidden,
            intermediate_size=inter,
            moe_intermediate_size=8,
            vocab_size=vocab,
            num_hidden_layers=n_target_layers,
            num_nextn_predict_layers=1,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=0,
            v_head_dim=4,
            rms_norm_eps=1e-5,
            swiglu_limit=10.0,
            n_routed_experts=0,
            mlp_layer_types=["dense"] * n_target_layers,
        )
        vllm_config = VllmConfig()
        vllm_config.compilation_config.mode = CompilationMode.NONE
        vllm_config.speculative_config = types.SimpleNamespace(
            num_speculative_tokens=2,
            draft_model_config=types.SimpleNamespace(
                hf_config=hf_config,
                compute_hash=lambda: "test_mtp_draft_model_k2",
            ),
            compute_hash=lambda: "test_mtp_spec_k2",
        )

        class _FakeRpaAttention(torch.nn.Module):
            def __init__(self, num_heads: int, head_size: int, **kwargs) -> None:
                super().__init__()
                del kwargs
                self.num_heads = num_heads
                self.head_size = head_size

            def forward(
                self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                return (q + k + v)[:, : self.num_heads * self.head_size]

        with (
            mock.patch.object(current_platform, "device_name", "cpu"),
            mock.patch.object(current_platform, "device_type", "cpu"),
            mock.patch.object(glm5_next_model, "Attention", _FakeRpaAttention),
            set_current_vllm_config(vllm_config),
        ):
            mtp_model = glm5_next_mtp.Glm5NextMTP(vllm_config=vllm_config, prefix="")

        mtp_layer = mtp_model.model.layers[str(n_target_layers)]
        target_lm_head = ParallelLMHead(vocab // 2, hidden)
        target_lm_head.tp_size = 2
        mtp_layer.shared_head.head = target_lm_head
        mtp_model.model.logits_processor.org_vocab_size = vocab - 2
        hidden_in = torch.randn(3, hidden)
        with (
            mock.patch.object(torch.compiler, "is_compiling", return_value=True),
            mock.patch.object(
                glm53_collectives_tpu,
                "vocab_all_gather",
                side_effect=lambda x: torch.cat([x, x], dim=-1),
            ) as mock_vag,
        ):
            compiled_logits = mtp_model.compute_logits(hidden_in)
        mock_vag.assert_called_once()
        self.assertIsNotNone(compiled_logits)
        assert compiled_logits is not None
        self.assertEqual(compiled_logits.shape, (3, vocab - 2))


if __name__ == "__main__":
    unittest.main()
