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
"""Shared builders for the spec-decode proposer tests."""

from types import SimpleNamespace
from unittest import mock

from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer


def make_proposer(
    draft_tp: int | None = 1, target_tp: int = 1, method: str = "eagle3"
) -> Eagle3Proposer:
    """An `Eagle3Proposer` on a mock runner, without a loaded draft."""
    speculative_config = SimpleNamespace(
        draft_tensor_parallel_size=draft_tp, method=method, draft_model_config=None
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=target_tp, pipeline_parallel_size=1
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config, parallel_config=parallel_config
    )
    return Eagle3Proposer(
        runner=mock.MagicMock(
            _pcp_enabled=False, _mtp_enabled=method == "mtp", _mtp_prefill_only=False
        ),
        vllm_config=vllm_config,
    )
