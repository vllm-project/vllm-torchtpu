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
"""PR CI running tests for k2-fsa/OmniVoice on TPU v7x.

End-to-end text-to-speech through the offline Omni entrypoint. OmniVoice loads
its generator and decoder weights with ``safetensors.torch.load_file(...,
device="tpu:N")``, so this test fails at model load without the patch in
src/vllm_torchtpu/omni/patches/safetensors_load.py. Each generated clip is
transcribed with Whisper on CPU and compared with the input text, so silent or
garbled audio fails too.
"""

import re

import numpy as np
import pytest
import torch
from transformers import pipeline
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner

pytestmark = [
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]

MODEL = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000
ASR_MODEL = "openai/whisper-small"

# vLLM-Omni's default omnivoice.yaml selects the "mp" executor, which loads the
# model in a separate worker process. The other single-chip engine tests in this
# lane use the "uni" default and run the engine in the pytest process, which then
# keeps the TPU open for the rest of the session. On the lane's one-chip host a
# separate worker started after them fails with "Failed to acquire a TPU device
# node", so run OmniVoice in-process too.
OMNI_RUNNER_PARAM = (
    MODEL,
    None,
    {"trust_remote_code": True, "distributed_executor_backend": "uni"},
)

# Word error rate of the Whisper transcript against the input text, after
# lowercasing and dropping punctuation. Measured on TPU v7x (torch_tpu 0.2.0,
# vllm-omni 0.32.0rc1): 0.00 for both cases in two runs; corpus WER over 200
# Seed-TTS-en utterances was 0.5-1.9% (Whisper-large-v3). Another sentence
# scores >= 1.0 against these transcripts, so 0.25 leaves margin for sampling
# and Whisper spelling variants ("seatbelt" -> "seat belt") while still failing
# on wrong or garbled speech.
MAX_WER = 0.25

CASES = [
    pytest.param(
        "The weather is nice today, perfect for a walk in the park.",
        "English",
        "english",
        id="english",
    ),
    pytest.param(
        "안녕하세요. 오늘 날씨가 정말 좋네요.",
        "Korean",
        "korean",
        id="korean",
    ),
]


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _wer(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance divided by the reference length."""
    ref, hyp = _words(reference), _words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / max(len(ref), 1)


def _extract_audio(output) -> tuple[np.ndarray, int]:
    """Audio samples and sample rate from an Omni generate output."""
    mm = getattr(output, "multimodal_output", None)
    if not mm and getattr(output, "outputs", None):
        mm = getattr(output.outputs[0], "multimodal_output", None)
    assert mm and "audio" in mm, f"No audio in output: {mm!r}"
    audio = mm["audio"]
    if isinstance(audio, list):
        audio = torch.cat([chunk for chunk in audio if chunk.numel() > 0], dim=-1)
    if isinstance(audio, torch.Tensor):
        audio = audio.float().cpu().numpy()
    sr = mm.get("sr", SAMPLE_RATE)
    if isinstance(sr, list):
        sr = sr[-1]
    return np.asarray(audio, dtype=np.float32).squeeze(), int(sr)


@pytest.fixture(scope="module")
def asr():
    return pipeline("automatic-speech-recognition", model=ASR_MODEL, device="cpu")


@pytest.mark.parametrize(
    "omni_runner",
    [OMNI_RUNNER_PARAM],
    indirect=True,
)
class TestOmniVoicePrCiOnTpu:
    """Fast PR CI end-to-end test for OmniVoice on TPU v7x."""

    @pytest.mark.parametrize("text,lang,asr_lang", CASES)
    def test_omnivoice_tts_matches_input_text(
        self, omni_runner: OmniRunner, asr, text: str, lang: str, asr_lang: str
    ):
        prompt = {"prompt": text, "lang": lang}
        params = OmniDiffusionSamplingParams(extra_args={"seed": 42})
        outputs = list(omni_runner.generate(prompt, [params]))
        assert outputs, "Omni returned no output"

        audio, sr = _extract_audio(outputs[-1])
        assert sr == SAMPLE_RATE
        assert np.isfinite(audio).all(), "Audio has NaN or Inf samples"
        duration = audio.size / sr
        assert 1.0 < duration < 15.0, f"Unexpected duration {duration:.2f}s"
        rms = float(np.sqrt(np.mean(audio**2)))
        assert rms > 0.01, f"Audio RMS {rms:.4f} is too low, likely silence"

        transcript = asr(
            {"raw": audio, "sampling_rate": sr},
            generate_kwargs={"language": asr_lang, "task": "transcribe"},
        )["text"]
        wer = _wer(text, transcript)
        print(f"{duration:.2f}s rms={rms:.3f} wer={wer:.2f} asr={transcript!r}")
        assert wer <= MAX_WER, f"WER {wer:.2f} > {MAX_WER}: {transcript!r}"
        # Must-fail control: the same transcript against another sentence.
        assert _wer("A red flower in an open sunny green meadow", transcript) > MAX_WER
