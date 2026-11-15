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
"""Tests for Qwen3-VL ViT torch.compile patch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.qwen3_vl_patch import (
    _COMPILED_VIT_FNS,
    _patch_qwen3_vl_vllm_vit_compile,
    maybe_patch_qwen3_vl,
)


class TestQwen3VLVitCompilePatch:
    """Verify Qwen3_VisionTransformer compile patch behavior and guards."""

    @staticmethod
    def _stub_hf_patches(monkeypatch: pytest.MonkeyPatch) -> None:
        for fn_name in (
            "_patch_qwen3_vl_get_rope_index",
            "_patch_qwen3_vl_vision_attention",
            "_patch_qwen3_vl_vision_transformer",
        ):
            monkeypatch.setattr(
                f"vllm_torchtpu.models.vllm.qwen3_vl_patch.{fn_name}",
                lambda _m: None,
            )

    def test_disabled_by_default_without_env_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

        self._stub_hf_patches(monkeypatch)
        monkeypatch.delenv("VLLM_TPU_QWEN3_VL_COMPILE_VIT", raising=False)
        assert envs.VLLM_TPU_QWEN3_VL_COMPILE_VIT is False
        assert "VLLM_TPU_QWEN3_VL_COMPILE_VIT" in _TPU_COMPILE_ENV_IGNORED

        called = False

        def _fake_patch() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.qwen3_vl_patch._patch_qwen3_vl_vllm_vit_compile",
            _fake_patch,
        )
        cfg = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl", architectures=[]),
            model="Qwen/Qwen3-VL-4B-Instruct",
            enforce_eager=False,
        )
        # pyrefly: ignore[bad-argument-type]
        maybe_patch_qwen3_vl(cfg)
        assert not called

    def test_enabled_calls_vit_compile_patch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_hf_patches(monkeypatch)
        called = False

        def _fake_patch() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.qwen3_vl_patch._patch_qwen3_vl_vllm_vit_compile",
            _fake_patch,
        )
        monkeypatch.setattr(envs, "VLLM_TPU_QWEN3_VL_COMPILE_VIT", True)
        cfg = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl", architectures=[]),
            model="Qwen/Qwen3-VL-4B-Instruct",
            enforce_eager=False,
        )
        # pyrefly: ignore[bad-argument-type]
        maybe_patch_qwen3_vl(cfg)
        assert called

    def test_enforce_eager_skips_vit_compile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_hf_patches(monkeypatch)
        called = False

        def _fake_patch() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.qwen3_vl_patch._patch_qwen3_vl_vllm_vit_compile",
            _fake_patch,
        )
        monkeypatch.setattr(envs, "VLLM_TPU_QWEN3_VL_COMPILE_VIT", True)
        cfg = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl", architectures=[]),
            model="Qwen/Qwen3-VL-4B-Instruct",
            enforce_eager=True,
        )
        # pyrefly: ignore[bad-argument-type]
        maybe_patch_qwen3_vl(cfg)
        assert not called

    @pytest.mark.tpu_test
    def test_compiled_vit_matches_eager_on_tpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state
        from vllm.model_executor.models.qwen3_vl import (
            Qwen3_VisionBlock,
            Qwen3_VisionTransformer,
        )

        class _DummyTPGroup:
            world_size = 1
            rank_in_group = 0
            rank = 0

            def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
                return x

        monkeypatch.setattr(parallel_state, "_TP", _DummyTPGroup())
        monkeypatch.setattr(envs, "VLLM_TPU_QWEN3_VL_COMPILE_VIT", True)
        orig_vit_forward = Qwen3_VisionTransformer.forward
        monkeypatch.setattr(Qwen3_VisionTransformer, "forward", orig_vit_forward)

        class _DummyMerger(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.squeeze(1)

        class _DummyViT(torch.nn.Module):
            def __init__(self, device: torch.device) -> None:
                super().__init__()
                self.device = device
                self.dtype = torch.bfloat16
                self.input_norm = lambda x, dtype: x.to(dtype)
                self.patch_embed = torch.nn.Identity()
                self.deepstack_visual_indexes = [0]
                self.blocks = torch.nn.ModuleList(
                    [
                        Qwen3_VisionBlock(dim=256, num_heads=4, mlp_hidden_dim=512).to(
                            device=device, dtype=torch.bfloat16
                        )
                    ]
                )
                self.deepstack_merger_list = torch.nn.ModuleList([_DummyMerger()])
                self.merger = _DummyMerger()

            def prepare_encoder_metadata(
                self, grid_thw_list: list[list[int]]
            ) -> dict[str, torch.Tensor]:
                cu = [0]
                for t, h, w in grid_thw_list:
                    cu.append(cu[-1] + t * h * w)
                seq_len = cu[-1]
                return {
                    "pos_embeds": torch.zeros(
                        (seq_len, 256), dtype=torch.bfloat16, device=self.device
                    ),
                    "cu_seqlens": torch.tensor(cu, dtype=torch.int32),
                    "rotary_pos_emb_cos": torch.ones(
                        (seq_len, 32), dtype=torch.bfloat16, device=self.device
                    ),
                    "rotary_pos_emb_sin": torch.zeros(
                        (seq_len, 32), dtype=torch.bfloat16, device=self.device
                    ),
                    "max_seqlen": torch.tensor(seq_len, dtype=torch.int32),
                }

        _patch_qwen3_vl_vllm_vit_compile()
        _patch_qwen3_vl_vllm_vit_compile()
        assert getattr(Qwen3_VisionTransformer.forward, "_tpu_vit_compile_patch", False)

        tpu = torch.device("tpu")
        torch.manual_seed(0)
        with set_current_vllm_config(VllmConfig()):
            vit = _DummyViT(tpu)
            for param in vit.parameters():
                torch.nn.init.normal_(param, std=0.02)
            grid = [[1, 16, 16], [1, 16, 16]]
            x = torch.randn((512, 256), dtype=torch.bfloat16, device=tpu)
            with torch.inference_mode():
                # 1) Default metadata path with multi-image cu_seqlens.
                out_compiled = Qwen3_VisionTransformer.forward(vit, x, grid)
                assert vit in _COMPILED_VIT_FNS
                # 2) Pre-computed encoder_metadata (with CPU sequence_lengths)
                # passed positionally.
                meta = vit.prepare_encoder_metadata(grid)
                meta["sequence_lengths"] = torch.tensor([256, 256], dtype=torch.int32)
                out_precomp = Qwen3_VisionTransformer.forward(
                    vit, x, torch.tensor(grid), meta
                )
                # 3) Outer torch.compile(fullgraph=True) skips inner compile.
                fresh_vit = _DummyViT(tpu)
                fresh_vit.load_state_dict(vit.state_dict())
                outer_compiled = torch.compile(
                    lambda xt, gt, m: Qwen3_VisionTransformer.forward(
                        fresh_vit, xt, gt, encoder_metadata=m
                    ),
                    backend="eager",
                    fullgraph=True,
                )
                out_outer = outer_compiled(x, torch.tensor(grid), meta)
                assert fresh_vit not in _COMPILED_VIT_FNS

                hs = (x + meta["pos_embeds"]).unsqueeze(1)
                h0 = vit.blocks[0](
                    hs,
                    cu_seqlens=meta["cu_seqlens"].to(tpu),
                    rotary_pos_emb_cos=meta["rotary_pos_emb_cos"],
                    rotary_pos_emb_sin=meta["rotary_pos_emb_sin"],
                    max_seqlen=meta["max_seqlen"].to(tpu),
                    sequence_lengths=meta["sequence_lengths"].to(tpu),
                )
                out_ref = torch.cat(
                    [vit.merger(h0), vit.deepstack_merger_list[0](h0)], dim=1
                )

        torch.testing.assert_close(
            out_compiled.cpu().float(),
            out_ref.cpu().float(),
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            out_precomp.cpu().float(),
            out_ref.cpu().float(),
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            out_outer.cpu().float(),
            out_ref.cpu().float(),
            atol=2e-2,
            rtol=2e-2,
        )
