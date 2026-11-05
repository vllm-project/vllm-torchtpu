# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The slice prefill's ops expose the streams the MLA DSpark drafts read."""

import pytest

from vllm_torchtpu.megakernel.kimi_k3.draft_mla_prefill import is_snapshot
from vllm_torchtpu.megakernel.kimi_k3.prefill import PrefillProgram

pytestmark = pytest.mark.cpu_test


def tapped(aux_layers):
    return [layer for layer in aux_layers if not is_snapshot(layer)]


def test_kimi_k3_draft_taps():
    assert PrefillProgram.stream_sources(tapped((3, 24, 48, 72, 90))) == {
        3: ("first", 0),
        90: ("tap", 10),
    }


def test_kimi_k31_draft_taps():
    """76 enters group 19 (pair 9's first), 80 group 20 (pair 9's second),
    88 group 22 (pair 10's second); 92 the final op; 84 is a snapshot."""
    assert PrefillProgram.stream_sources(tapped((76, 80, 84, 88, 92))) == {
        76: ("input", 9),
        80: ("middle", 9),
        88: ("middle", 10),
        92: ("final", 0),
    }


@pytest.mark.parametrize("layer", [0, 5, 77, 91, 93])
def test_streams_no_op_exposes(layer):
    with pytest.raises(NotImplementedError, match=f"layer {layer}"):
        PrefillProgram.stream_sources([layer])
