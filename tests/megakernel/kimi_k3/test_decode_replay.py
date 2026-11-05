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
"""A recompute on the K3 decoder writes its KDA state into vLLM's state block
at every block boundary the request holds one for (CPU, stub ops)."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3.decode import KimiDecoder
from vllm_torchtpu.megakernel.kimi_k3.pools import KDA_LAYERS

pytestmark = pytest.mark.cpu_test


def test_replay_writes_the_state_into_each_boundary_block():
    page, slot = 4, 5
    written = []
    decoder = SimpleNamespace(
        groups=1,
        control=None,
        layout=None,
        sub_base=9,
        _kda_groups=[(1 + k % 3, page) for k in range(KDA_LAYERS)],
        _sync_plan=lambda entries, tables: None,
        ops=SimpleNamespace(
            batch_for=lambda n: n,
            plain_step=lambda control, *args: (None, control),
            slots=("conv", "recurrent"),
        ),
        _importer=SimpleNamespace(
            export_state=lambda conv, recurrent, blocks, at: written.append(
                (blocks.tolist(), at)
            )
        ),
    )
    # A hit at 8 tokens (column 1), then this request's blocks; align mode left
    # column 4 null (a boundary inside a longer chunk).
    kda = [
        np.array([0, 10 * g, 10 * g + 1, 10 * g + 2, 0, 10 * g + 3]) for g in (1, 2, 3)
    ]
    block_tables = (np.arange(1, 4, dtype=np.int32), *kda)
    KimiDecoder._replay(
        decoder, slot, np.arange(8, 22), 8, block_tables[0], block_tables
    )
    assert written == [
        ([10 * (1 + k % 3) + c for k in range(KDA_LAYERS)], slot) for c in (1, 2)
    ]
