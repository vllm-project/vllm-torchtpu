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

"""Two-chunk asynchronous spec steps hand every chunk's drafts to the runner.

On an asynchronous spec step the MTP proposer returns its drafts on device and
the runner builds the next step's substitution source from them
(``_assemble_async_spec_substitution``). These tests run the real ``propose``
and the real assembly on CPU and check that every request finds its own drafts
in the substitution source."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.spec_decode.test_eagle3 import _make_chunk, _make_proposer
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

pytestmark = pytest.mark.cpu_test

K = 3
HIDDEN = 4
LOOP_BUCKET = 16
REQ_BUCKET = 8
INVALID = -1


def _draft_code(chunk: int, row: int) -> int:
    return 1000 * (chunk + 1) + 10 * row


def _bonus(chunk: int, row: int) -> int:
    return 500 + 10 * chunk + row


def _proposer(monkeypatch, num_reqs: list[int]) -> Eagle3Proposer:
    """An MTP proposer with a TP-sharded draft (draft TP 2) whose draft model
    is replaced by stand-ins: chunk ``c``'s request ``r`` drafts
    ``_draft_code(c, r) + s`` at draft step ``s``."""
    proposer = _make_proposer(draft_tp=2, target_tp=2, method="mtp")
    proposer.speculative_config.num_speculative_tokens = K
    proposer.draft_model = SimpleNamespace()
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=sum(num_reqs)),
        max_num_reqs=2 * REQ_BUCKET,
        num_reqs_paddings=[REQ_BUCKET, 2 * REQ_BUCKET],
        num_tokens_paddings=[LOOP_BUCKET, 2 * LOOP_BUCKET],
        num_reqs_max_model_len=LOOP_BUCKET,
        num_reqs_most_model_len=None,
        device=torch.device("cpu"),
        uses_mrope=False,
        _dp_lockstep_enabled=lambda: False,
    )

    def chunk_of(chunk) -> int:
        return 0 if chunk.start_index == 0 else 1

    def prepare(self, chunk, *_args, **_kwargs):
        n = chunk.num_reqs
        last = torch.cat([torch.arange(n), torch.full((LOOP_BUCKET - n,), n - 1)]).to(
            torch.int64
        )
        return (
            torch.zeros(LOOP_BUCKET, dtype=torch.int32),
            torch.zeros(LOOP_BUCKET, dtype=torch.int32),
            last,
            torch.zeros(REQ_BUCKET, dtype=torch.int32),
        )

    def forward(self, *, chunk, step_idx, num_tokens_padded, **kwargs):
        if step_idx == 0:
            hidden = torch.zeros((num_tokens_padded, HIDDEN))
            hidden[:, 0] = torch.tensor(
                [_draft_code(chunk_of(chunk), t) for t in range(num_tokens_padded)],
                dtype=torch.float32,
            )
        else:
            hidden = kwargs["target_hidden_states"] + 1
        return hidden, hidden

    def gather(self, hidden, positions, last_hidden, idx, *_rest):
        return hidden[idx], positions[idx], last_hidden[idx]

    def propose_token(self, hidden):
        return hidden[:, 0].to(torch.int32)

    monkeypatch.setattr(Eagle3Proposer, "_prepare_draft_inputs", prepare)
    monkeypatch.setattr(Eagle3Proposer, "_forward_draft", forward)
    monkeypatch.setattr(Eagle3Proposer, "_draft_gather_carries", gather)
    monkeypatch.setattr(Eagle3Proposer, "_draft_propose_token", propose_token)
    proposer.draft_chunks = [
        _make_chunk(
            input_ids=torch.zeros(LOOP_BUCKET, dtype=torch.int32),
            position_ids=torch.zeros(LOOP_BUCKET, dtype=torch.int32),
            query_start_loc_np=np.arange(n + 1, dtype=np.int32),
            start_index=sum(num_reqs[:ci]),
            num_reqs=n,
            device=torch.device("cpu"),
            hidden_states=torch.zeros((LOOP_BUCKET, HIDDEN)),
            attn_ctx=SimpleNamespace(use_max_model_len=True, seq_lens=None),
        )
        for ci, n in enumerate(num_reqs)
    ]
    return proposer


def _verify_outputs(num_reqs: list[int]):
    """Each chunk's rejection-sampler output ``[REQ_BUCKET, 1 + K]`` (every
    draft rejected, so each row holds its bonus token) and its metadata."""
    next_tokens, metadata = [], []
    for c, n in enumerate(num_reqs):
        nt = torch.full((REQ_BUCKET, 1 + K), INVALID, dtype=torch.int32)
        nt[:, 0] = 0
        nt[:n, 0] = torch.tensor([_bonus(c, r) for r in range(n)], dtype=torch.int32)
        next_tokens.append(nt)
        lengths = torch.zeros(REQ_BUCKET, dtype=torch.int32)
        lengths[:n] = K
        metadata.append(
            SimpleNamespace(
                draft_lengths=lengths,
                draft_lengths_cpu=np.full(n, K, dtype=np.int32),
            )
        )
    return next_tokens, SimpleNamespace(
        num_reqs_list=list(num_reqs), spec_decode_metadata_list=metadata
    )


def _runner(drafter: Eagle3Proposer, draft_tp: int = 2) -> TPUModelRunner:
    runner = object.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.max_num_reqs = 2 * REQ_BUCKET
    runner.speculative_config = SimpleNamespace(
        num_speculative_tokens=K, draft_tensor_parallel_size=draft_tp
    )
    runner.drafter = drafter
    runner._adaptive_verification = None
    return runner


def _assert_each_request_has_its_drafts(num_reqs, assembled):
    chunks, next_token_indices, _, _ = assembled
    source = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
    stride = 1 + K
    g = 0
    for c, n in enumerate(num_reqs):
        for r in range(n):
            at = next_token_indices[g] * stride
            want = [_bonus(c, r)] + [_draft_code(c, r) + s for s in range(K)]
            assert source[at : at + stride].tolist() == want, (c, r)
            g += 1


@pytest.mark.parametrize("num_reqs", [[3, 2], [8, 8], [1, 5]])
def test_two_chunk_async_step_hands_each_chunk_its_drafts(monkeypatch, num_reqs):
    """propose returns both chunks' drafts in request order, and the assembly
    puts each request's drafts in its own rows of the source."""
    drafter = _proposer(monkeypatch, num_reqs)
    runner = _runner(drafter)
    next_tokens, state = _verify_outputs(num_reqs)
    drafts = drafter.propose(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
        return_device=True,
        next_tokens_per_chunk=next_tokens,
    )
    assert drafts.tolist() == [
        [_draft_code(c, r) + s for s in range(K)]
        for c, n in enumerate(num_reqs)
        for r in range(n)
    ]
    _assert_each_request_has_its_drafts(
        num_reqs, runner._assemble_async_spec_substitution(drafts, next_tokens, state)
    )
