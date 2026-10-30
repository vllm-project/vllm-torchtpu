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
"""Unit tests for Mamba3ForCausalLM, Mamba3Model, and Mamba3Config."""

from __future__ import annotations

import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from transformers import AutoConfig
from vllm.config import (
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.interfaces import (
    SupportsMambaPrefixCaching,
    supports_mamba_prefix_caching,
)
from vllm.platforms import current_platform
from vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op import (  # pyrefly: ignore
    VllmMambaMixer3,
)

from vllm_torchtpu.layers.adapter.custom_ops import (  # pyrefly: ignore
    mamba_mixer3_op,  # pyrefly: ignore
)
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.mamba3 import (
    Mamba3Config,
    Mamba3ForCausalLM,
    Mamba3GatedMLP,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture(autouse=True)
def _empty_shared_mamba3_ops(monkeypatch):
    """Isolate shared Mamba-3 jax_op cache across unit tests."""
    monkeypatch.setattr(mamba_mixer3_op, "_shared_mamba3_ops", {})


@pytest.fixture(scope="module", autouse=True)
def _dist_and_cpu_platform():
    import torch.distributed as dist
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    orig_device_name = getattr(current_platform, "device_name", "cpu")
    orig_device_type = getattr(current_platform, "device_type", "cpu")
    current_platform.device_name = "cpu"
    current_platform.device_type = "cpu"

    if not dist.is_initialized():
        import portpicker

        port = portpicker.pick_unused_port()
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)
    try:
        yield
    finally:
        current_platform.device_name = orig_device_name
        current_platform.device_type = orig_device_type


def _jax_cpu_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1]).reshape((1,))
    return Mesh(devices, axis_names=("model",))


def _make_vllm_config(
    hf_config: Mamba3Config,
    *,
    tp_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> VllmConfig:
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        dtype=dtype,
        head_dtype=None,
    )
    vllm_config.cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
    )
    vllm_config.parallel_config = SimpleNamespace(
        tensor_parallel_size=tp_size,
        data_parallel_size=1,
        pipeline_parallel_size=1,
    )
    vllm_config.compilation_config.mode = CompilationMode.NONE
    return vllm_config


def _build_mamba3_model(
    hf_config: Mamba3Config,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    use_real_jax_op: bool = False,
) -> tuple[Mamba3ForCausalLM, VllmConfig]:
    vllm_config = _make_vllm_config(hf_config, tp_size=tp_size)
    mesh = (
        _jax_cpu_mesh()
        if use_real_jax_op
        else SimpleNamespace(
            shape={"attn_dp": 1, "expert": 1, "model": 1},
            axis_names=("attn_dp", "expert", "model"),
        )
    )

    def _passthrough_jax_op(op_name, fn, donate_argnums=()):
        def _op(*torch_args):
            jax_args = [
                jax.numpy.asarray(t.detach().cpu().numpy())
                if isinstance(t, torch.Tensor)
                else None
                for t in torch_args
            ]
            new_aux, new_rec, out = fn(*jax_args)
            return (
                torch.from_numpy(np.array(new_aux, copy=True)),
                torch.from_numpy(np.array(new_rec, copy=True)),
                torch.from_numpy(np.array(out, copy=True)),
            )

        _op.register_fake = lambda _: None
        return _op

    with ExitStack() as stack:
        stack.enter_context(set_current_vllm_config(vllm_config))
        stack.enter_context(
            set_vllm_model_wrapper_context(mesh=mesh, vllm_config=vllm_config)
        )
        if tp_size > 1:
            for target in (
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_tensor_model_parallel_world_size",
                "vllm_torchtpu.models.vllm.mamba3.mamba3.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_world_size",
            ):
                stack.enter_context(patch(target, return_value=tp_size))
            for target in (
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_tensor_model_parallel_rank",
                "vllm_torchtpu.models.vllm.mamba3.mamba3.get_tensor_model_parallel_rank",
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
                "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
            ):
                stack.enter_context(patch(target, return_value=tp_rank))
        if use_real_jax_op:
            stack.enter_context(
                patch(
                    "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.pallas.jax_op",
                    side_effect=_passthrough_jax_op,
                )
            )
        else:
            fake_jax_op = MagicMock()
            stack.enter_context(
                patch(
                    "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.pallas.jax_op",
                    return_value=fake_jax_op,
                )
            )

        model = Mamba3ForCausalLM(vllm_config=vllm_config)

    return model, vllm_config


class TestMamba3RegistrationAndConfig:
    def test_mamba3_model_registration(self):
        register_models()
        entry = ModelRegistry.models.get("Mamba3ForCausalLM")
        assert entry is not None, "Mamba3ForCausalLM is not registered"
        model_cls = entry.load_model_cls()
        assert model_cls is Mamba3ForCausalLM

    def test_autoconfig_detects_siso_and_mimo_checkpoint_configs(self, tmp_path):
        register_models()

        siso_dir = tmp_path / "mamba3-siso-1.5b"
        siso_dir.mkdir()
        siso_json = {
            "d_model": 2048,
            "d_intermediate": 4096,
            "n_layer": 24,
            "vocab_size": 128256,
            "ssm_cfg": {
                "layer": "Mamba3",
                "d_state": 128,
                "expand": 2,
                "headdim": 64,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "dt_min": 0.001,
                "dt_max": 0.1,
                "dt_init_floor": 0.0001,
                "A_floor": 0.0001,
                "chunk_size": 64,
                "is_mimo": False,
                "mimo_rank": 4,
                "is_outproj_norm": False,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "tie_embeddings": True,
            "pad_vocab_size_multiple": 16,
        }
        (siso_dir / "config.json").write_text(json.dumps(siso_json))

        siso_cfg = AutoConfig.from_pretrained(str(siso_dir))
        assert isinstance(siso_cfg, Mamba3Config)
        assert siso_cfg.model_type == "mamba3"
        assert siso_cfg.architectures == ["Mamba3ForCausalLM"]
        assert siso_cfg.hidden_size == 2048
        assert siso_cfg.num_hidden_layers == 24
        assert siso_cfg.d_intermediate == 4096
        assert siso_cfg.num_heads == 64
        assert siso_cfg.head_dim == 64
        assert siso_cfg.state_size == 128
        assert siso_cfg.is_mimo is False
        assert siso_cfg.mimo_rank == 1
        assert siso_cfg.rope_fraction == 0.5
        assert siso_cfg.chunk_size == 64

        mimo_dir = tmp_path / "mamba3-mimo-1.5b"
        mimo_dir.mkdir()
        mimo_json = dict(siso_json, d_intermediate=3824)
        mimo_json["ssm_cfg"] = dict(
            siso_json["ssm_cfg"], is_mimo=True, mimo_rank=4, chunk_size=16
        )
        (mimo_dir / "config.json").write_text(json.dumps(mimo_json))

        mimo_cfg = AutoConfig.from_pretrained(str(mimo_dir))
        assert isinstance(mimo_cfg, Mamba3Config)
        assert mimo_cfg.is_mimo is True
        assert mimo_cfg.mimo_rank == 4
        assert mimo_cfg.chunk_size == 16
        assert mimo_cfg.d_intermediate == 3824

    def test_state_shape_and_dtype_match_mixer_layer(self):
        siso_cfg = Mamba3Config(
            d_model=64,
            n_layer=2,
            vocab_size=256,
            d_intermediate=128,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 16,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": False,
                "mimo_rank": 4,
            },
        )
        mimo_cfg = Mamba3Config(
            d_model=64,
            n_layer=2,
            vocab_size=256,
            d_intermediate=112,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 16,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": True,
                "mimo_rank": 4,
            },
        )

        for cfg in (siso_cfg, mimo_cfg):
            model, vllm_cfg = _build_mamba3_model(cfg)
            mixer = model.backbone.layers[0].mixer
            assert isinstance(mixer, VllmMambaMixer3)
            assert (
                Mamba3ForCausalLM.get_mamba_state_shape_from_config(vllm_cfg)
                == mixer.get_state_shape()
            )
            assert (
                Mamba3ForCausalLM.get_mamba_state_dtype_from_config(vllm_cfg)
                == mixer.get_state_dtype()
            )

        assert SupportsMambaPrefixCaching not in Mamba3ForCausalLM.__mro__
        assert not supports_mamba_prefix_caching(Mamba3ForCausalLM)
        assert not hasattr(Mamba3ForCausalLM, "get_mamba_state_copy_func")


class TestMamba3ArchitectureAndWeights:
    def test_decoder_layer_variants_with_and_without_mlp(self):
        cfg_no_mlp = Mamba3Config(
            d_model=64,
            n_layer=2,
            vocab_size=128,
            d_intermediate=0,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 16,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
            },
        )
        model_no_mlp, _ = _build_mamba3_model(cfg_no_mlp)
        assert model_no_mlp.backbone.layers[0].mlp is None
        assert model_no_mlp.backbone.layers[0].norm2 is None

        cfg_mimo_mlp = Mamba3Config(
            d_model=64,
            n_layer=2,
            vocab_size=128,
            d_intermediate=112,  # Non-multiple of 32 -> padded to 128
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 16,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "is_mimo": True,
                "mimo_rank": 4,
            },
        )
        model_mimo, _ = _build_mamba3_model(cfg_mimo_mlp)
        layer0 = model_mimo.backbone.layers[0]
        assert isinstance(layer0.mlp, Mamba3GatedMLP)
        assert layer0.mlp.orig_intermediate_size == 112
        assert layer0.mlp.intermediate_size == 128
        assert layer0.mlp.fc1.weight.shape == (256, 64)
        assert layer0.mlp.fc2.weight.shape == (64, 128)
        assert layer0.mixer.mimo_x is layer0.mixer.Psi
        assert layer0.mixer.mimo_o is layer0.mixer.Phi
        assert layer0.mixer.mimo_z is layer0.mixer.Zeta

    def test_load_weights_siso_checkpoint_with_3d_bc_bias(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=1,
            vocab_size=64,
            d_intermediate=64,
            tie_embeddings=False,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": False,
                "mimo_rank": 4,
                "is_outproj_norm": False,
            },
        )
        model, _ = _build_mamba3_model(cfg)
        mixer = model.backbone.layers[0].mixer
        total_proj = mixer.total_proj_size

        emb_w = torch.randn(64, 32)
        lm_head_w = torch.randn(64, 32)
        norm_w = torch.randn(32)
        norm2_w = torch.randn(32)
        norm_f_w = torch.randn(32)
        in_proj_w = torch.randn(total_proj, 32)
        out_proj_w = torch.randn(32, 64)
        b_norm_w = torch.randn(8)
        c_norm_w = torch.randn(8)
        # SISO HF checkpoint stores 3D (H, 1, N) for B_bias and C_bias
        b_bias_3d = torch.randn(4, 1, 8)
        c_bias_3d = torch.randn(4, 1, 8)
        d_w = torch.randn(4)
        dt_bias_w = torch.randn(4)
        fc1_w = torch.randn(128, 32)
        fc2_w = torch.randn(32, 64)

        ckpt = [
            ("backbone.embedding.weight", emb_w),
            ("backbone.layers.0.norm.weight", norm_w),
            ("backbone.layers.0.mixer.in_proj.weight", in_proj_w),
            ("backbone.layers.0.mixer.out_proj.weight", out_proj_w),
            ("backbone.layers.0.mixer.B_norm.weight", b_norm_w),
            ("backbone.layers.0.mixer.C_norm.weight", c_norm_w),
            ("backbone.layers.0.mixer.B_bias", b_bias_3d),
            ("backbone.layers.0.mixer.C_bias", c_bias_3d),
            ("backbone.layers.0.mixer.D", d_w),
            ("backbone.layers.0.mixer.dt_bias", dt_bias_w),
            ("backbone.layers.0.norm2.weight", norm2_w),
            ("backbone.layers.0.mlp.fc1.weight", fc1_w),
            ("backbone.layers.0.mlp.fc2.weight", fc2_w),
            ("backbone.norm_f.weight", norm_f_w),
            ("lm_head.weight", lm_head_w),
        ]
        loaded = model.load_weights(ckpt)

        assert "backbone.embedding.weight" in loaded
        assert "lm_head.weight" in loaded
        assert "backbone.layers.0.mixer.B_norm_weight" in loaded
        assert "backbone.layers.0.mixer.C_norm_weight" in loaded
        torch.testing.assert_close(mixer.B_bias, b_bias_3d.squeeze(1))
        torch.testing.assert_close(mixer.C_bias, c_bias_3d.squeeze(1))
        torch.testing.assert_close(mixer.B_norm_weight, b_norm_w)
        torch.testing.assert_close(mixer.C_norm_weight, c_norm_w)
        torch.testing.assert_close(mixer.in_proj.weight, in_proj_w)
        torch.testing.assert_close(model.backbone.layers[0].mlp.fc1.weight, fc1_w)
        torch.testing.assert_close(model.backbone.layers[0].mlp.fc2.weight, fc2_w)
        torch.testing.assert_close(model.lm_head.weight, lm_head_w)

    def test_load_weights_mimo_with_unpadded_and_prepadded_mlp(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=1,
            vocab_size=64,
            d_intermediate=48,  # Padded to 64 (multiple of 32)
            tie_embeddings=True,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": True,
                "mimo_rank": 4,
            },
        )
        model, _ = _build_mamba3_model(cfg)
        mixer = model.backbone.layers[0].mixer
        mlp = model.backbone.layers[0].mlp
        assert mlp is not None and mlp.intermediate_size == 64

        psi_w = torch.randn(4, 4, 16)
        phi_w = torch.randn(4, 4, 16)
        zeta_w = torch.randn(4, 4, 16)
        # Unpadded fc1 (2 * 48 = 96, 32) and fc2 (32, 48)
        fc1_unpadded = torch.randn(96, 32)
        fc2_unpadded = torch.randn(32, 48)

        model.load_weights(
            [
                ("model.embed_tokens.weight", torch.ones(64, 32)),
                ("model.layers.0.mixer.mimo_x", psi_w),
                ("model.layers.0.mixer.mimo_o", phi_w),
                ("model.layers.0.mixer.mimo_z", zeta_w),
                ("model.layers.0.mlp_norm.weight", torch.ones(32) * 2.0),
                ("model.layers.0.mlp.fc1.weight", fc1_unpadded),
                ("model.layers.0.mlp.fc2.weight", fc2_unpadded),
            ]
        )
        torch.testing.assert_close(mixer.Psi, psi_w)
        torch.testing.assert_close(mixer.Phi, phi_w)
        torch.testing.assert_close(mixer.Zeta, zeta_w)
        # Gate is rows [0:48], zero-padded on [48:64]; Up is [64:112], padded [112:128]
        torch.testing.assert_close(mlp.fc1.weight[:48], fc1_unpadded[:48])
        assert torch.all(mlp.fc1.weight[48:64] == 0)
        torch.testing.assert_close(mlp.fc1.weight[64:112], fc1_unpadded[48:96])
        assert torch.all(mlp.fc1.weight[112:128] == 0)
        torch.testing.assert_close(mlp.fc2.weight[:, :48], fc2_unpadded)
        assert torch.all(mlp.fc2.weight[:, 48:64] == 0)

        # Load pre-padded fc1 (128, 32) and fc2 (32, 64) as in HF mamba3-mimo-1.5b
        fc1_padded = torch.randn(128, 32)
        fc2_padded = torch.randn(32, 64)
        model.load_weights(
            [
                ("backbone.layers.0.mlp.fc1.weight", fc1_padded),
                ("backbone.layers.0.mlp.fc2.weight", fc2_padded),
            ]
        )
        torch.testing.assert_close(mlp.fc1.weight, fc1_padded)
        torch.testing.assert_close(mlp.fc2.weight, fc2_padded)

    def test_multiprocess_tp_sharding_with_single_group(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=1,
            vocab_size=64,
            d_intermediate=48,  # Padded to 64 -> 32 per rank on TP=2
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": False,
            },
        )
        model_r1, _ = _build_mamba3_model(cfg, tp_size=2, tp_rank=1)
        mixer_r1 = model_r1.backbone.layers[0].mixer
        mlp_r1 = model_r1.backbone.layers[0].mlp
        assert mlp_r1 is not None

        # Global in_proj: z(64), x(64), B(8), C(8), dt/A/trap(4 each), angles(2) = 158
        # Rank 1 in_proj: z(32), x(32), B(8), C(8), dt/A/trap(2 each), angles(2) = 88
        assert mixer_r1.in_proj.weight.shape == (88, 32)

        in_proj_global = torch.arange(158 * 32, dtype=torch.float32).reshape(158, 32)
        b_bias_3d = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 1, 8)
        fc1_padded = torch.arange(128 * 32, dtype=torch.float32).reshape(128, 32)
        fc2_padded = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)

        model_r1.load_weights(
            [
                ("backbone.layers.0.mixer.in_proj.weight", in_proj_global),
                ("backbone.layers.0.mixer.B_bias", b_bias_3d),
                ("backbone.layers.0.mlp.fc1.weight", fc1_padded),
                ("backbone.layers.0.mlp.fc2.weight", fc2_padded),
            ]
        )

        expected_in_proj_r1 = torch.cat(
            [
                in_proj_global[32:64],  # z shard 1
                in_proj_global[96:128],  # x shard 1
                in_proj_global[128:136],  # B replicated
                in_proj_global[136:144],  # C replicated
                in_proj_global[146:148],  # dd_dt shard 1
                in_proj_global[150:152],  # dd_A shard 1
                in_proj_global[154:156],  # trap shard 1
                in_proj_global[156:158],  # angles replicated
            ],
            dim=0,
        )
        torch.testing.assert_close(mixer_r1.in_proj.weight, expected_in_proj_r1)
        torch.testing.assert_close(mixer_r1.B_bias, b_bias_3d.squeeze(1)[2:4])
        expected_fc1_r1 = torch.cat([fc1_padded[32:64], fc1_padded[96:128]], dim=0)
        torch.testing.assert_close(mlp_r1.fc1.weight, expected_fc1_r1)
        torch.testing.assert_close(mlp_r1.fc2.weight, fc2_padded[:, 32:64])


class TestMamba3EndToEndForward:
    def test_siso_and_mimo_full_model_prefill_then_decode_matches_full_prefill(
        self,
    ):
        """Verifies full 2-layer Mamba3ForCausalLM E2E forward + logits continuity."""
        for is_mimo in (False, True):
            torch.manual_seed(1234)
            cfg = Mamba3Config(
                d_model=32,
                n_layer=2,
                vocab_size=64,
                d_intermediate=64,
                ssm_cfg={
                    "layer": "Mamba3",
                    "d_state": 8,
                    "expand": 2,
                    "headdim": 16,
                    "ngroups": 1,
                    "rope_fraction": 0.5,
                    "chunk_size": 4,
                    "is_mimo": is_mimo,
                    "mimo_rank": 4,
                    "is_outproj_norm": False,
                },
            )
            model, vllm_cfg = _build_mamba3_model(cfg, use_real_jax_op=True)
            for p in model.parameters():
                torch.nn.init.normal_(p, mean=0.0, std=0.05)

            aux_shape, ssm_shape = Mamba3ForCausalLM.get_mamba_state_shape_from_config(
                vllm_cfg
            )
            max_seqs = 2
            total_len = 6
            prefill_len = 4

            input_ids = torch.tensor([3, 11, 27, 42, 19, 55], dtype=torch.long)
            positions = torch.arange(total_len, dtype=torch.long)

            # 1. Contiguous full prefill over all 6 tokens
            caches_contig = []
            for layer in model.backbone.layers:
                aux_c = torch.zeros((max_seqs, *aux_shape), dtype=torch.float32)
                rec_c = torch.zeros((max_seqs, *ssm_shape), dtype=torch.float32)
                layer.mixer.kv_cache = (aux_c, rec_c)
                caches_contig.append((aux_c, rec_c))

            meta_contig = SimpleNamespace(
                mamba_state_indices=torch.tensor([0], dtype=torch.int32),
                query_start_loc=torch.tensor([0, total_len], dtype=torch.int32),
                request_distribution=torch.tensor([0, 1, 1], dtype=torch.int32),
                seq_lens=torch.tensor([total_len], dtype=torch.int32),
                block_tables=None,
            )
            fc_contig = SimpleNamespace(
                attn_metadata={
                    f"backbone.layers.{i}.mixer": meta_contig for i in range(2)
                }
            )
            pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

            with (
                patch(
                    "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_forward_context",
                    return_value=fc_contig,
                ),
                patch(
                    "vllm_torchtpu.models.vllm.mamba3.mamba3.get_pp_group",
                    return_value=pp_group,
                ),
            ):
                hidden_contig = model(input_ids, positions)
                logits_contig = model.compute_logits(hidden_contig)

            # 2. Split execution: prefill(0..4) + 2 decode steps (4..5, 5..6)
            caches_split = []
            for layer in model.backbone.layers:
                aux_s = torch.zeros((max_seqs, *aux_shape), dtype=torch.float32)
                rec_s = torch.zeros((max_seqs, *ssm_shape), dtype=torch.float32)
                layer.mixer.kv_cache = (aux_s, rec_s)
                caches_split.append((aux_s, rec_s))

            meta_prefill = SimpleNamespace(
                mamba_state_indices=torch.tensor([0], dtype=torch.int32),
                query_start_loc=torch.tensor([0, prefill_len], dtype=torch.int32),
                request_distribution=torch.tensor([0, 1, 1], dtype=torch.int32),
                seq_lens=torch.tensor([prefill_len], dtype=torch.int32),
                block_tables=None,
            )
            fc_prefill = SimpleNamespace(
                attn_metadata={
                    f"backbone.layers.{i}.mixer": meta_prefill for i in range(2)
                }
            )
            with (
                patch(
                    "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_forward_context",
                    return_value=fc_prefill,
                ),
                patch(
                    "vllm_torchtpu.models.vllm.mamba3.mamba3.get_pp_group",
                    return_value=pp_group,
                ),
            ):
                hidden_pre = model(input_ids[:prefill_len], positions[:prefill_len])

            hidden_dec_list = []
            for step_idx in range(prefill_len, total_len):
                meta_dec = SimpleNamespace(
                    mamba_state_indices=torch.tensor([0], dtype=torch.int32),
                    query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                    request_distribution=torch.tensor([1, 1, 1], dtype=torch.int32),
                    seq_lens=torch.tensor([step_idx + 1], dtype=torch.int32),
                    block_tables=None,
                )
                fc_dec = SimpleNamespace(
                    attn_metadata={
                        f"backbone.layers.{i}.mixer": meta_dec for i in range(2)
                    }
                )
                with (
                    patch(
                        "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_forward_context",
                        return_value=fc_dec,
                    ),
                    patch(
                        "vllm_torchtpu.models.vllm.mamba3.mamba3.get_pp_group",
                        return_value=pp_group,
                    ),
                ):
                    h_dec = model(
                        input_ids[step_idx : step_idx + 1],
                        positions[step_idx : step_idx + 1],
                    )
                    hidden_dec_list.append(h_dec)

            hidden_split = torch.cat([hidden_pre, *hidden_dec_list], dim=0)
            logits_split = model.compute_logits(hidden_split)

            torch.testing.assert_close(
                hidden_split, hidden_contig, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                logits_split, logits_contig, rtol=1e-4, atol=1e-4
            )
            for i in range(2):
                torch.testing.assert_close(
                    caches_split[i][0][0],
                    caches_contig[i][0][0],
                    rtol=1e-4,
                    atol=1e-4,
                )
                torch.testing.assert_close(
                    caches_split[i][1][0],
                    caches_contig[i][1][0],
                    rtol=1e-4,
                    atol=1e-4,
                )
