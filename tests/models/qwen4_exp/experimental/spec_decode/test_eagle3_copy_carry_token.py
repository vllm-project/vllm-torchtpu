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
"""Draft step 0's fused position carry and first token on the model-owned
EAGLE/MTP proposer."""

import pytest
import torch

from tests.models.qwen4_exp.experimental.spec_decode.eagle3_copy_utils import (
    make_proposer,
)


@pytest.mark.parametrize("use_d2t", [False, True])
@pytest.mark.parametrize("mrope", [False, True])
def test_draft_rows_carry_and_token_matches_separate_helpers(
    use_d2t: bool, mrope: bool
):
    proposer = make_proposer(draft_tp=1, method="eagle3")
    hidden_size, draft_vocab, target_vocab = 8, 12, 24
    lm_head = torch.nn.Linear(hidden_size, draft_vocab, bias=False)
    d2t = torch.arange(draft_vocab, dtype=torch.int32) * 2 if use_d2t else None

    class _Draft:
        draft_id_to_target_id = d2t

        def __init__(self):
            self.lm_head = lm_head

        def logits_processor(self, head, hidden):
            return head(hidden)

        def compute_logits(self, hidden):
            return torch.nn.functional.pad(
                lm_head(hidden), (0, target_vocab - draft_vocab), value=-1e9
            )

    proposer.draft_model = _Draft()
    positions = (
        torch.arange(48, dtype=torch.int32).reshape(3, 16)
        if mrope
        else torch.arange(100, 116, dtype=torch.int32)
    )
    # Include -1 (PCP unowned row) as a negative control: must clamp to index 0.
    indices = torch.tensor([2, -1, 7, 0], dtype=torch.int64)
    last_hidden = torch.randn((4, hidden_size), dtype=torch.float32)

    pos_carry, tok0 = proposer._draft_rows_carry_and_token(
        positions, indices, last_hidden
    )
    clamped = indices.clamp_min(0)
    expected_pos = torch.index_select(positions, positions.ndim - 1, clamped)
    if use_d2t:
        draft_id = lm_head(last_hidden).argmax(dim=-1)
        expected_tok = (draft_id + d2t[draft_id]).to(torch.int32)
    else:
        expected_tok = (
            proposer.draft_model.compute_logits(last_hidden)
            .argmax(dim=-1)
            .to(torch.int32)
        )
    assert torch.equal(pos_carry, expected_pos)
    assert torch.equal(tok0, expected_tok)
