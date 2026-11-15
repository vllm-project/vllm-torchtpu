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
"""TPU tests for the Qwen-Image text_encoder CPU offload patch."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QwenImagePipeline,
)

from vllm_torchtpu import envs
from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED
from vllm_torchtpu.omni.patches import (
    _QWEN_IMAGE_TEXT_ENCODER_OFFLOAD_PATCHED,
    _module_device,
    _patch_qwen_image_text_encoder_offload,
)

pytestmark = [pytest.mark.omni]


class _RecordingTextEncoder(torch.nn.Module):
    """Linear encoder that records the device of every ``.to()`` move."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.moves: list[str] = []

    def to(self, *args: Any, **kwargs: Any) -> _RecordingTextEncoder:
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self.moves.append(device.type)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _tpu_encoder() -> _RecordingTextEncoder:
    encoder = _RecordingTextEncoder().to(
        device=torch.device("tpu"), dtype=torch.bfloat16
    )
    encoder.moves.clear()
    return encoder


class _DummyQwenPipeline:
    """Stands in for QwenImagePipeline; methods are installed on the real class."""

    def __init__(
        self, text_encoder: torch.nn.Module | None, *, raise_on_encode: bool = False
    ) -> None:
        self.text_encoder = text_encoder
        self.raise_on_encode = raise_on_encode
        self.seen_devices: list[str] = []

    def encode_prompt(
        self, x: torch.Tensor, prompt_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.raise_on_encode:
            raise RuntimeError("boom")
        if prompt_embeds is not None:
            return prompt_embeds
        if self.text_encoder is None:
            return x
        device = _module_device(self.text_encoder)
        assert device is not None
        self.seen_devices.append(device.type)
        return self.text_encoder(x.to(device))

    def _prepare_generation_context(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors upstream true CFG: positive and negative prompt encodes.
        positive = QwenImagePipeline.encode_prompt(self, x)
        negative = QwenImagePipeline.encode_prompt(self, x)
        return positive, negative


def _x() -> torch.Tensor:
    return torch.ones((1, 4), dtype=torch.bfloat16)


def _install_dummy_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        QwenImagePipeline, "encode_prompt", _DummyQwenPipeline.encode_prompt
    )
    monkeypatch.setattr(
        QwenImagePipeline,
        "_prepare_generation_context",
        _DummyQwenPipeline._prepare_generation_context,
    )


@pytest.fixture
def offload_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_methods(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_TPU_OMNI_TEXT_ENCODER_OFFLOAD", True)
    _patch_qwen_image_text_encoder_offload()


class TestQwenImageTextEncoderOffload:
    """Verify the QwenImagePipeline text_encoder CPU offload lifecycle on TPU."""

    def test_disabled_by_default_without_env_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VLLM_TPU_OMNI_TEXT_ENCODER_OFFLOAD", raising=False)
        assert envs.VLLM_TPU_OMNI_TEXT_ENCODER_OFFLOAD is False
        assert "VLLM_TPU_OMNI_TEXT_ENCODER_OFFLOAD" in _TPU_COMPILE_ENV_IGNORED
        _install_dummy_methods(monkeypatch)
        _patch_qwen_image_text_encoder_offload()
        assert QwenImagePipeline.encode_prompt is _DummyQwenPipeline.encode_prompt
        assert (
            QwenImagePipeline._prepare_generation_context
            is _DummyQwenPipeline._prepare_generation_context
        )

    def test_upstream_qwen_image_pipeline_signatures(self) -> None:
        """Canary against upstream vllm-omni renames of patched hooks/params."""
        assert hasattr(QwenImagePipeline, "_prepare_generation_context")
        encode_params = inspect.signature(QwenImagePipeline.encode_prompt).parameters
        assert "prompt_embeds" in encode_params

    @pytest.mark.usefixtures("offload_patch")
    def test_patch_is_idempotent(self) -> None:
        encode = QwenImagePipeline.encode_prompt
        prepare = QwenImagePipeline._prepare_generation_context
        assert getattr(encode, _QWEN_IMAGE_TEXT_ENCODER_OFFLOAD_PATCHED, False)
        _patch_qwen_image_text_encoder_offload()
        assert QwenImagePipeline.encode_prompt is encode
        assert QwenImagePipeline._prepare_generation_context is prepare

    @pytest.mark.usefixtures("offload_patch")
    def test_true_cfg_request_round_trips_encoder_once(self) -> None:
        encoder = _tpu_encoder()
        pipe = _DummyQwenPipeline(encoder)

        # Request 1: loaded on TPU, both encodes run there, then one offload.
        QwenImagePipeline._prepare_generation_context(pipe, _x())
        assert encoder.moves == ["cpu"]
        # Request 2: one upload for both encodes, one offload.
        QwenImagePipeline._prepare_generation_context(pipe, _x())
        assert encoder.moves == ["cpu", "tpu", "cpu"]
        assert pipe.seen_devices == ["tpu"] * 4

    @pytest.mark.usefixtures("offload_patch")
    def test_direct_encode_prompt_offloads_and_restores(self) -> None:
        encoder = _tpu_encoder()
        pipe = _DummyQwenPipeline(encoder)

        out = QwenImagePipeline.encode_prompt(pipe, _x())
        assert out.device.type == "tpu"
        assert encoder.moves == ["cpu"]
        QwenImagePipeline.encode_prompt(pipe, _x())
        assert encoder.moves == ["cpu", "tpu", "cpu"]
        assert pipe.seen_devices == ["tpu", "tpu"]

    @pytest.mark.usefixtures("offload_patch")
    def test_precomputed_prompt_embeds_do_not_move_encoder(self) -> None:
        encoder = _tpu_encoder()
        pipe = _DummyQwenPipeline(encoder)
        QwenImagePipeline.encode_prompt(pipe, _x())  # first offload to CPU
        embeds = torch.zeros((1, 4), dtype=torch.bfloat16)

        out = QwenImagePipeline.encode_prompt(pipe, _x(), prompt_embeds=embeds)
        assert out is embeds
        assert encoder.moves == ["cpu"]
        assert pipe.seen_devices == ["tpu"]

    @pytest.mark.usefixtures("offload_patch")
    def test_offloads_to_cpu_on_exception(self) -> None:
        encoder = _tpu_encoder()
        pipe = _DummyQwenPipeline(encoder, raise_on_encode=True)
        with pytest.raises(RuntimeError, match="boom"):
            QwenImagePipeline.encode_prompt(pipe, _x())
        device = _module_device(encoder)
        assert device is not None and device.type == "cpu"

    @pytest.mark.usefixtures("offload_patch")
    def test_passes_through_cpu_and_none_text_encoder(self) -> None:
        cpu_encoder = _RecordingTextEncoder().to(dtype=torch.bfloat16)
        cpu_encoder.moves.clear()
        cpu_pipe = _DummyQwenPipeline(cpu_encoder)
        out = QwenImagePipeline.encode_prompt(cpu_pipe, _x())
        assert out.device.type == "cpu"
        assert cpu_encoder.moves == []

        none_pipe = _DummyQwenPipeline(None)
        assert QwenImagePipeline.encode_prompt(none_pipe, _x()).device.type == "cpu"
