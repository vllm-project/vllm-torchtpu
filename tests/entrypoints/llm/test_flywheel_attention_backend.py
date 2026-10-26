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
"""--attention-backend flywheel against the default RPA backend: greedy
Qwen3-0.6B generations on one TPU core.

Each engine runs in its own process (libtpu is process-exclusive), in process
with the engine core (VLLM_ENABLE_V1_MULTIPROCESSING=0), which starts no child
process for --attention-backend flywheel's runtime-registered name to reach.
The prompts outgrow max_num_batched_tokens, so the steps cover fresh and
chunked prefill, mixed prefill + decode, and decode alone.
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.tpu_test

_GENERATE = """
import json, sys
from vllm import LLM, SamplingParams

backend = sys.argv[1] or None
base = "The quick brown fox jumps over the lazy dog. "
prompts = [
    "The capital of France is",
    "Write a haiku about the ocean.",
    "List the planets of the solar system in order:",
    "Explain why the sky is blue in one paragraph.",
    "def fibonacci(n):",
    base * 20 + "Summarize the text above in one sentence.",
    base * 45 + "How many times does the word fox appear above?",
    "1, 2, 3, 4, 5,",
    "Translate to German: The weather is nice today.",
    base * 70 + "What animal jumps in the text above?",
]
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=2048,
    max_num_batched_tokens=256,
    max_num_seqs=8,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    attention_backend=backend,
)
outputs = llm.generate(
    prompts, SamplingParams(temperature=0.0, max_tokens=32, logprobs=2)
)
print("OUTPUTS=" + json.dumps([
    [
        (token, sorted((-lp.logprob for lp in step.values()))[:2])
        for token, step in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)
    ]
    for o in outputs
]))
"""

# bf16 logits resolve log-probabilities this coarsely, so two tokens this close
# are a tie that a different reduction order may break the other way.
_TIE = 0.25


def _generate(attention_backend: str | None) -> list[list[tuple[int, list[float]]]]:
    """Per prompt, each greedy token with its two best negative log-probs."""
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0")
    result = subprocess.run(
        [sys.executable, "-c", _GENERATE, attention_backend or ""],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("OUTPUTS=")]
    assert result.returncode == 0 and lines, (
        f"{attention_backend or 'default'} backend failed "
        f"(rc={result.returncode}):\n{result.stderr[-6000:]}"
    )
    return json.loads(lines[-1].removeprefix("OUTPUTS="))


def test_flywheel_generates_rpa_tokens():
    rpa = _generate(None)
    flywheel = _generate("flywheel")

    assert len(flywheel) == len(rpa)
    exact = 0
    for i, (got, want) in enumerate(zip(flywheel, rpa)):
        got_tokens, want_tokens = [t for t, _ in got], [t for t, _ in want]
        exact += got_tokens == want_tokens
        part = next(
            (k for k, (a, b) in enumerate(zip(got_tokens, want_tokens)) if a != b),
            None,
        )
        if part is not None:
            # A different kernel may break RPA's own ties the other way, and
            # only those.
            best, second = want[part][1]
            assert second - best <= _TIE, (
                f"prompt {i} parts at token {part}, where RPA is not tied: "
                f"{got_tokens} vs {want_tokens}"
            )
    assert exact >= len(rpa) - 2, f"{exact}/{len(rpa)} identical"
