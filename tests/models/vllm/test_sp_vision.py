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
"""Pin the upstream forwards that the sequence-parallel vision tower mirrors."""

import hashlib
import inspect

import pytest
import vllm
from vllm.model_executor.models import qwen2_5_vl, qwen3_vl

from vllm_torchtpu.models.vllm import sp_vision

# SequenceParallelVision copies these from vLLM v0.33.0 (pyproject.toml pin).
PINNED_VLLM_VERSION = "0.33.0"


@pytest.mark.skipif(
    vllm.__version__ != PINNED_VLLM_VERSION,
    reason=f"the mirrored forwards are copies of vLLM {PINNED_VLLM_VERSION}",
)
@pytest.mark.parametrize(
    "forward, pinned_hash",
    [
        (
            sp_vision._UPSTREAM_FORWARD or qwen3_vl.Qwen3_VisionTransformer.forward,
            "6ef9bf87825621c11ab3f4871fb5cdadd7cc1e5b8e05737df63284ffc8c8fcaf",
        ),
        (
            qwen3_vl.Qwen3_VisionBlock.forward,
            "e92ce5aad55b0340042055ea57edacf749579529f2145e439cd18ba031ea1e64",
        ),
        (
            qwen2_5_vl.Qwen2_5_VisionAttention.forward,
            "d65da66fac09eae5db93eedae4d304c9b546723a8b7285f7c3fba7573bb49c6b",
        ),
        (
            qwen3_vl.Qwen3_VisionPatchMerger.forward,
            "100752bade898c81d6efab01e0ab40e5f638307b7619ffd6a8b4f88c32df813a",
        ),
    ],
)
def test_mirrored_upstream_forward_unchanged(forward, pinned_hash):
    actual = hashlib.sha256(inspect.getsource(forward).encode("utf-8")).hexdigest()
    assert actual == pinned_hash, (
        f"Upstream {forward.__qualname__} has drifted. Re-sync "
        "SequenceParallelVision in src/vllm_torchtpu/models/vllm/sp_vision.py."
    )
