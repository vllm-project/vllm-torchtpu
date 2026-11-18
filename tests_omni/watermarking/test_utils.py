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
"""TPU watermarking tests for vllm-omni audio and text outputs."""

from __future__ import annotations

import pytest
import torch
from vllm.outputs import CompletionOutput
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

AUDIO_MAX_REL_L2 = 1e-4
AUDIO_MAX_ABS = 1e-4
TEXT_MAX_ABS = 0.0


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_multimodal_watermarking() -> None:
    """Multimodal Watermarking (AudioSeal & Text): AudioSeal audio.

    - TPU path: import watermarking APIs in-body and run AudioSeal on a TPU
      float32 waveform.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and
      ``StagePool.initialize_watermarkers`` registers ``AudioSealWatermarker``.
    - Outcome check: audio watermarking returns no failed IDs, keeps the tensor
      on TPU, flips ``is_watermarked`` to True, and matches CPU parity.
    """
    from vllm_omni.config.watermarking import WatermarkConfig
    from vllm_omni.engine.stage_pool import StagePool
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.watermarking.audio_seal import AudioSealWatermarker
    from vllm_omni.watermarking.types import AudioTensor
    from vllm_omni.watermarking.utils import watermark_outputs

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    cfg = WatermarkConfig.from_dict(
        {"modalities": {"audio": {"algorithm": "audioseal"}}}
    )
    watermarkers = StagePool.initialize_watermarkers("audio", cfg)
    assert "audio" in watermarkers, (
        f"audioseal: initialize_watermarkers keys {list(watermarkers)}, "
        "expected 'audio'"
    )
    watermarker = watermarkers["audio"]
    assert isinstance(watermarker, AudioSealWatermarker), (
        f"Expected AudioSealWatermarker, got {type(watermarker)}"
    )

    torch.manual_seed(0)
    wav_cpu = torch.randn(1, 1, 1600, dtype=torch.float32) * 0.1
    input_detected = watermarker.is_watermarked(
        AudioTensor(samples=wav_cpu, sample_rate=16000)
    )
    assert not input_detected, (
        f"audioseal: unwatermarked input is_watermarked={input_detected}, "
        "expected False"
    )

    wav_tpu = wav_cpu.to("tpu")
    tpu_out = OmniRequestOutput(
        request_id="req-audio-tpu",
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
        multimodal_output={"audio": wav_tpu, "sr": 16000},
    )
    cpu_out = OmniRequestOutput(
        request_id="req-audio-cpu",
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
        multimodal_output={"audio": wav_cpu.clone(), "sr": 16000},
    )

    failed_tpu = watermark_outputs([tpu_out], watermarkers)
    failed_cpu = watermark_outputs([cpu_out], watermarkers)
    assert not failed_tpu, f"TPU audio watermarking failed for {failed_tpu}"
    assert not failed_cpu, f"CPU audio watermarking failed for {failed_cpu}"

    watermarked_tpu = tpu_out.multimodal_output["audio"]
    watermarked_cpu = cpu_out.multimodal_output["audio"]
    assert watermarked_tpu.device.type == "tpu", (
        f"Expected watermarked audio on tpu, got {watermarked_tpu.device.type}"
    )
    output_detected = watermarker.is_watermarked(
        AudioTensor(samples=watermarked_tpu.cpu(), sample_rate=16000)
    )
    assert output_detected, (
        f"audioseal: watermarked TPU output is_watermarked={output_detected}, "
        "expected True"
    )
    assert_parity(
        watermarked_tpu,
        watermarked_cpu,
        name="audioseal_tpu_vs_cpu",
        max_rel_l2=AUDIO_MAX_REL_L2,
        max_abs=AUDIO_MAX_ABS,
    )


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_multimodal_watermarking_text() -> None:
    """Multimodal Watermarking (AudioSeal & Text): Gumbel text.

    - TPU path: import watermarking APIs in-body and run Gumbel text watermark
      config/sampling on TPU logits.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and
      ``to_vllm_watermark_config`` resolves text config only for v2 runner.
    - Outcome check: text Gumbel sampling on TPU matches CPU reference.
    """
    from vllm_omni.config.watermarking import WatermarkConfig
    from vllm_omni.watermarking.utils import to_vllm_watermark_config

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    cfg = WatermarkConfig.from_dict({"modalities": {"text": {"algorithm": "gumbel"}}})
    resolved = to_vllm_watermark_config(cfg, "text", use_v2_model_runner=True)
    assert resolved == {"algorithm": "gumbel"}, (
        f"Unexpected vLLM text watermark config: {resolved}"
    )
    v1_resolved = to_vllm_watermark_config(cfg, "text", use_v2_model_runner=False)
    assert v1_resolved is None, (
        f"gumbel: v1-runner text watermark config {v1_resolved!r}, expected None"
    )

    torch.manual_seed(0)
    logits_cpu = torch.randn(1, 32, dtype=torch.float32)
    gumbel_cpu = torch.rand_like(logits_cpu)
    gumbel_noise = -torch.log(-torch.log(gumbel_cpu.clamp(min=1e-6)))
    token_cpu = torch.argmax(logits_cpu + gumbel_noise, dim=-1)
    token_tpu = torch.argmax(logits_cpu.to("tpu") + gumbel_noise.to("tpu"), dim=-1)
    assert_parity(
        token_tpu.float(),
        token_cpu.float(),
        name="gumbel_text_token_tpu_vs_cpu",
        max_abs=TEXT_MAX_ABS,
    )
