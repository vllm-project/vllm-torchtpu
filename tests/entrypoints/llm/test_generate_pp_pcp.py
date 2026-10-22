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
"""End-to-end test of pipeline stages that span a prefill context-parallel
group on TPU: two stages, each over half of the host's devices, must produce
the same greedy first tokens as two stages of one device each.

Each layout runs in a process of its own: the TPU runtime takes the
context-parallel slice flags once per process, so a second engine with other
flags cannot start in the same one.
"""

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.multichip

MODEL_NAME = "Qwen/Qwen3-0.6B"
# Long enough for every context-parallel rank to hold a chunk of each prompt.
PROMPTS = [
    "The capital of France is " + "Paris, a city on the Seine. " * 24,
    "A pipeline stage can span a context-parallel group. " * 20 + "In short,",
    "The quick brown fox jumps over the lazy dog. " * 30 + "Therefore the",
]

# Runs one engine and prints the greedy first token of each prompt, one
# prompt per step so every layout sees the same batch shape. Every token
# bucket must divide into the context-parallel ranks' Q blocks of 512
# tokens, so the default buckets from 16 up cannot be used.
_CHILD = """
import json, sys
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

model, parallel, prompts = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
llm = LLM(
    model=model,
    max_model_len=2048,
    max_num_batched_tokens=2048,
    compilation_config={"compile_sizes": [512, 1024, 2048]},
    async_scheduling=False,
    gpu_memory_utilization=0.5,
    enforce_eager=False,
    disable_log_stats=True,
    **parallel,
)
try:
    params = SamplingParams(temperature=0.0, max_tokens=1)
    tokens = [
        list(llm.generate([prompt], sampling_params=params)[0].outputs[0].token_ids)
        for prompt in prompts
    ]
finally:
    del llm
    cleanup_dist_env_and_memory()
print("FIRST_TOKENS " + json.dumps(tokens), flush=True)
"""


def _get_local_tpu_device_count() -> int:
    try:
        from tpu_info import device as tpu_device

        _, num_chips = tpu_device.get_local_chips()
        return num_chips
    except Exception as e:
        pytest.skip(f"Unable to detect local TPU chip count: {e}")


def _first_tokens(**parallel) -> list[list[int]]:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            MODEL_NAME,
            json.dumps(parallel),
            json.dumps(PROMPTS),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, f"{parallel}: {proc.stderr[-4000:]}"
    lines = [
        line for line in proc.stdout.splitlines() if line.startswith("FIRST_TOKENS ")
    ]
    assert lines, f"{parallel}: no tokens in {proc.stdout[-2000:]}"
    return json.loads(lines[-1][len("FIRST_TOKENS ") :])


def test_pipeline_stages_over_context_parallel_groups_match_plain_stages():
    devices = _get_local_tpu_device_count()
    if devices < 8:
        pytest.skip(
            f"PP x PCP test expects an 8-device agent, found {devices}; "
            "running here would silently reduce the stage width"
        )
    reference = _first_tokens(pipeline_parallel_size=2)
    staged = _first_tokens(
        pipeline_parallel_size=2, prefill_context_parallel_size=devices // 2
    )
    assert len(staged) == len(PROMPTS)
    assert all(len(tokens) == 1 for tokens in staged)
    assert staged == reference
