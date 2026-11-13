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
"""Nightly OmniRunner test for Qwen3-Omni Thinker on TPU v7x via TPUARWorker."""

import numpy as np
import pytest

pytest.importorskip("vllm_omni", reason="vllm_omni is required for omni model tests")

from vllm import SamplingParams

from tests_omni.conftest import OmniRunner

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.spawns_tpu_workers,
]

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEPLOY_CONFIG = "qwen3_omni_moe_thinking.yaml"
SAMPLE_RATE = 16000
# 8 seconds at 16 kHz -> 800 mel frames (8 full 100-frame windows), exercising
# the TPUARWorker / TPUARModelRunner wiring, make_omni_output, and the
# padded-window audio encoder patch together in a single Thinker-only stage.
AUDIO_DURATION_SECONDS = 8

TEXT_PROMPT = (
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _build_8s_audio_prompt() -> dict[str, object]:
    t = np.linspace(
        0.0,
        AUDIO_DURATION_SECONDS,
        SAMPLE_RATE * AUDIO_DURATION_SECONDS,
        endpoint=False,
        dtype=np.float32,
    )
    waveform = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    return {
        "prompt": (
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
            "Describe the sound briefly.<|im_end|>\n<|im_start|>assistant\n"
        ),
        "multi_modal_data": {
            "audio": (waveform, SAMPLE_RATE),
        },
    }


@pytest.mark.parametrize(
    "omni_runner,prompt",
    [
        pytest.param((MODEL, DEPLOY_CONFIG), TEXT_PROMPT, id="text"),
        pytest.param((MODEL, DEPLOY_CONFIG), _build_8s_audio_prompt(), id="audio_8s"),
    ],
    indirect=["omni_runner"],
)
def test_qwen3_omni_thinker_only_tpu(
    omni_runner: OmniRunner, prompt: str | dict[str, object]
) -> None:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
    outputs = list(omni_runner.generate(prompt, [sampling_params]))
    assert len(outputs) == 1
    assert outputs[0].final_output_type == "text"
    assert len(outputs[0].outputs) == 1
    assert len(outputs[0].outputs[0].text.strip()) > 0
