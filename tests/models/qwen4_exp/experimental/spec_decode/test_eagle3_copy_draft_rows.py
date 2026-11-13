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
"""Draft rows (MTP draft step 0 at each request's last token only) on the
model-owned EAGLE/MTP proposer."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.models.qwen4_exp.experimental.spec_decode.eagle3_copy_utils import (
    make_chunk,
    make_proposer,
)


def _draft_rows_proposer(monkeypatch, **runner):
    proposer = make_proposer(draft_tp=1, method="mtp")
    proposer.draft_model = SimpleNamespace(supports_draft_rows=True)
    proposer.runner = SimpleNamespace(
        _pcp_enabled=runner.get("pcp", False),
        _dp_lockstep_enabled=lambda: runner.get("lockstep", False),
    )
    proposer._share_mtp_indices = runner.get("share", False)
    return proposer


@pytest.mark.parametrize(
    "supported,runner,expected",
    [
        (True, {}, True),
        (False, {}, False),
        (True, {"share": True}, False),
        (True, {"pcp": True}, False),
        (True, {"lockstep": True}, False),
    ],
)
def test_resolve_draft_rows(monkeypatch, supported, runner, expected):
    proposer = _draft_rows_proposer(monkeypatch, **runner)
    proposer.draft_model.supports_draft_rows = supported
    assert proposer._resolve_draft_rows() is expected


def test_propose_with_draft_rows_uses_row_outputs_as_carries():
    """Step 0 receives last_token_indices as draft_rows and its per-row
    outputs become the carries; positions are still gathered per token."""
    import torch._dynamo

    device = torch.device("cpu")
    torch._dynamo.reset()
    hidden_size, vocab = 8, 32
    proposer = make_proposer(draft_tp=1, method="mtp")
    proposer.speculative_config.num_speculative_tokens = 2
    proposer._draft_rows = True
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=2),
        max_num_reqs=16,
        num_reqs_max_model_len=16,
        num_reqs_most_model_len=16,
        num_tokens_paddings=[16],
        device=device,
        uses_mrope=False,
        _dp_lockstep_enabled=lambda: False,
    )
    chunk = make_chunk(
        input_ids=torch.zeros(16, dtype=torch.int32, device=device),
        position_ids=torch.zeros(16, dtype=torch.int32, device=device),
        query_start_loc_np=np.array([0, 3, 7], dtype=np.int32),
        start_index=0,
        num_reqs=2,
        hidden=hidden_size,
        device=device,
        attn_ctx=SimpleNamespace(use_max_model_len=True),
        hidden_states=torch.zeros(16, hidden_size, device=device),
    )
    proposer.draft_chunks = [chunk]

    class _Draft:
        def compute_logits(self, hidden):
            # Token = the row's first hidden value.
            return torch.nn.functional.one_hot(
                hidden[:, 0].long() % vocab, vocab
            ).float()

    proposer.draft_model = _Draft()
    # The compiled helpers need the TPU backend; run their bodies eagerly.
    proposer._draft_gather_positions = lambda pos, idx: pos.index_select(0, idx)
    proposer._draft_propose_token = lambda hidden: (
        proposer.draft_model.compute_logits(hidden).argmax(-1).to(torch.int32)
    )
    last = torch.tensor([2, 6, 6, 6], device=device)
    positions = torch.arange(100, 116, dtype=torch.int32, device=device)

    def _prepare(chunk, *_args, **_kwargs):
        return (
            torch.zeros(16, dtype=torch.int32, device=device),
            positions,
            last,
            np.zeros(2, dtype=np.int32),
        )

    proposer._prepare_draft_inputs = _prepare
    calls = []

    def _forward_draft(*, input_ids, positions, step_idx, draft_rows=None, **_):
        calls.append((step_idx, draft_rows, positions.clone()))
        n = input_ids.shape[0] if draft_rows is None else len(draft_rows)
        if step_idx == 0:
            # Padded with -1 to the chunk's request capacity so the traced
            # program's shape does not depend on the padded request count.
            assert n == 16
            value = torch.zeros(16, device=device)
            value[:2] = torch.tensor([11.0, 22.0], device=device)
        else:
            value = torch.full((n,), 5.0 + step_idx, device=device)
        hidden = value[:, None].expand(n, hidden_size).contiguous()
        return hidden, hidden

    proposer._forward_draft = _forward_draft
    drafts = proposer.propose(
        sampled_token_ids=[[0], [0]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
    )
    assert drafts == [[11, 6], [22, 6]]
    assert calls[0][0] == 0 and torch.equal(calls[0][1][:4], last)
    assert (calls[0][1][4:] == -1).all()
    # The loop step starts from the gathered step-0 positions, plus one.
    assert calls[1][1] is None
    assert calls[1][2][:2].tolist() == [103, 107]


@pytest.mark.parametrize(
    "use_max_model_len, capacity, p", [(True, 16, 4), (False, 8, 16), (True, 4, 4)]
)
def test_draft_rows_fixed_length_per_chunk(use_max_model_len, capacity, p):
    """draft_rows always has the chunk's request capacity as its length
    (whatever the padded request count p), so every token bucket's compiled
    draft program sees one static draft_rows shape; outputs go back to p."""
    proposer = make_proposer(draft_tp=1, method="mtp")
    proposer.runner = SimpleNamespace(
        max_num_reqs=32,
        num_reqs_max_model_len=capacity if use_max_model_len else 32,
        num_reqs_most_model_len=32 if use_max_model_len else capacity,
    )
    chunk = SimpleNamespace(
        attn_ctx=SimpleNamespace(use_max_model_len=use_max_model_len)
    )
    num_rows = proposer._draft_rows_len(chunk)
    assert num_rows == capacity
    last = torch.arange(p, dtype=torch.int64) * 3
    rows = proposer._padded_draft_rows(last, num_rows)
    assert rows.shape == (capacity,)
    n = min(p, capacity)
    assert torch.equal(rows[:n], last[:n])
    assert (rows[n:] == -1).all()
    out = torch.arange(capacity * 2, dtype=torch.float32).view(capacity, 2)
    fitted = proposer._fit_draft_rows_output(out, p)
    assert fitted.shape == (p, 2)
    assert torch.equal(fitted[:n], out[:n])
    assert (fitted[n:] == 0).all()
