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
"""Pin the upstream forwards that sequence-parallel prefill mirrors."""

import hashlib
import inspect

import pytest
import vllm
from vllm.model_executor.models import qwen3, qwen3_vl

from vllm_torchtpu.models.vllm import sp_prefill

# SequenceParallelPrefill copies these from vLLM v0.33.0 (pyproject.toml pin).
PINNED_VLLM_VERSION = "0.33.0"


@pytest.mark.skipif(
    vllm.__version__ != PINNED_VLLM_VERSION,
    reason=f"the mirrored forwards are copies of vLLM {PINNED_VLLM_VERSION}",
)
@pytest.mark.parametrize(
    "forward, pinned_hash",
    [
        (
            qwen3.Qwen3Attention.forward,
            "06bb52cad82441114178e752abe4cc7b8cafc4f71eba8e821abb8ddce0f66a26",
        ),
        (
            qwen3.Qwen3DecoderLayer.forward,
            "2dbae90ca479be517b3db00d35fc22cdc4d3cd97520bca9bec82a4c56277d74a",
        ),
        (
            sp_prefill._UPSTREAM_FORWARD or qwen3_vl.Qwen3LLMModel.forward,
            "d27f45da621148d8777ce884fddd558fa8b44de8b191623129190ee22e2418ac",
        ),
    ],
)
def test_mirrored_upstream_forward_unchanged(forward, pinned_hash):
    actual = hashlib.sha256(inspect.getsource(forward).encode("utf-8")).hexdigest()
    assert actual == pinned_hash, (
        f"Upstream {forward.__qualname__} has drifted. Re-sync "
        "SequenceParallelPrefill in src/vllm_torchtpu/models/vllm/sp_prefill.py."
    )
