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

"""The MTP proposer records each draft forward's form before running it
(`Eagle3Proposer._draft_rows_sliced`, then `sequence_parallel.set_active`), so
an engine with an MTP drafter keeps the per-layout sequence-parallel row
thresholds the Qwen3.5 model patch measured (64 rows at DP4 x TP2, 512 at
TP8). `sequence_parallel.DRAFT_RECORDS_STEP_FORM` declares that record;
without it every sequence-parallel step would stay sliced and these tests
fail."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import sequence_parallel
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("layout,threshold", [((4, 2), 64), ((1, 8), 512)])
def test_an_mtp_drafter_keeps_the_layouts_row_threshold(monkeypatch, layout, threshold):
    """With a drafter and a model that follows each step's form, the row
    threshold is the layout's, and the draft path slices a draft forward's
    rows exactly from that threshold."""
    monkeypatch.setattr(sequence_parallel, "_FOLLOWS_STEP_FORM", True)
    monkeypatch.setattr(sequence_parallel, "_HAS_DRAFTER", True)
    monkeypatch.setattr(
        sequence_parallel, "parallel_layout", lambda parallel_config=None: layout
    )
    assert sequence_parallel.min_rows() == threshold
    proposer = SimpleNamespace(_draft_sequence_parallel=True)
    assert not Eagle3Proposer._draft_rows_sliced(proposer, threshold - 1)
    assert Eagle3Proposer._draft_rows_sliced(proposer, threshold)
    assert sequence_parallel.gates_rows()
