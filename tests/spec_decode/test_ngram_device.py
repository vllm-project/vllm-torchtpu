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
"""DeviceNgramProposer drafts what vLLM's host n-gram matcher drafts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.ngram_proposer import (
    _find_longest_matched_ngram_and_propose_tokens,
)

from vllm_torchtpu.spec_decode.ngram_device import DeviceNgramProposer

MAX_MODEL_LEN = 64
MAX_NUM_REQS = 4
X = -1  # INVALID_TOKEN_ID: no token in this slot.


def _make_proposer(device, histories, stale, min_n, max_n, k):
    token_ids_cpu = np.full((MAX_NUM_REQS, MAX_MODEL_LEN), stale, dtype=np.int32)
    for row, history in enumerate(histories):
        token_ids_cpu[row, : len(history)] = history
    num_tokens_no_spec = np.zeros(MAX_NUM_REQS, dtype=np.int32)
    num_tokens_no_spec[: len(histories)] = [len(h) for h in histories]
    runner = SimpleNamespace(
        device=device,
        max_num_reqs=MAX_NUM_REQS,
        input_batch=SimpleNamespace(
            num_reqs=len(histories),
            req_ids=[f"req-{row}" for row in range(len(histories))],
            token_ids_cpu=token_ids_cpu,
            num_tokens_no_spec=num_tokens_no_spec,
        ),
        _registered_h2d_sources=set(),
        _free_h2d_sources={},
        _unfenced_h2d_sources=[],
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            prompt_lookup_min=min_n,
            prompt_lookup_max=max_n,
            num_speculative_tokens=k,
        ),
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
    )
    proposer = DeviceNgramProposer(runner, vllm_config)
    proposer.load_model(target_model=None)
    return proposer


def _expected(sequence, min_n, max_n, k):
    """vLLM's host matcher, its short proposal filled to K with token 0."""
    drafts = _find_longest_matched_ngram_and_propose_tokens(
        np.array(sequence, dtype=np.int32), min_n, max_n, MAX_MODEL_LEN, k
    ).tolist()
    return drafts + [0] * (k - len(drafts))


@pytest.mark.parametrize(
    ("histories", "stale", "chunks", "steps", "min_n", "max_n", "k"),
    [
        pytest.param(
            [[2, 3, 7, 1, 2, 3, 9, 1, 2]],
            0,
            [(0, 1, 2)],
            [[[[3], [5]]]],
            2,
            3,
            4,
            id="longest_ngram_wins_over_earlier_shorter",
        ),
        pytest.param(
            [[5, 6, 1, 5, 6, 2, 5]],
            0,
            [(0, 1, 2)],
            [[[[6], [5]]]],
            2,
            2,
            3,
            id="earliest_occurrence_wins",
        ),
        pytest.param(
            [[4, 4, 7, 8, 4]],
            0,
            [(0, 1, 2)],
            [[[[4], [5]]]],
            2,
            2,
            7,
            id="tail_shorter_than_k_is_filled",
        ),
        pytest.param(
            [[1, 2, 3, 4]],
            77,
            [(0, 1, 2)],
            [[[[1], [5]]]],
            1,
            2,
            5,
            id="stale_tokens_past_seq_len_ignored",
        ),
        pytest.param(
            [[3, 1, 4, 1, 5]],
            0,
            [(0, 1, 2)],
            [
                [[[9, 2, X, X], [5, 5, 5, 5]]],
                [[[5, 9, X, X], [5, 5, 5, 5]]],
            ],
            2,
            2,
            3,
            id="invalid_padded_append_lands_at_base_len",
        ),
        pytest.param(
            [[1, 2, 3, 4], [6, 7, 8, 9, 6]],
            0,
            [(0, 1, 2), (1, 1, 2)],
            # Chunk 0's padded row must not reach chunk 1's request at row 1.
            [[[[1], [3]], [[7], [8]]]],
            2,
            2,
            2,
            id="padded_rows_never_written",
        ),
        pytest.param(
            [[1, 2, 3]],
            0,
            [(0, 1, 2)],
            [[[[4], [5]]], [[[1], [5]]]],
            1,
            1,
            4,
            id="no_match_drafts_filler_and_seq_len_advances",
        ),
    ],
)
def test_propose_matches_host_ngram(
    device, histories, stale, chunks, steps, min_n, max_n, k
):
    """Each step drafts what the host matcher drafts from the committed tokens
    plus the step's valid tokens. Between steps only the committed lengths
    advance: the drafter must hold the appended tokens itself. One-token steps
    arrive as a prefill's `[rows]` seed, longer ones as a verify output."""
    proposer = _make_proposer(device, histories, stale, min_n, max_n, k)
    input_batch = proposer.runner.input_batch
    proposer.draft_chunks = [
        SimpleNamespace(start_index=start, num_reqs=num_reqs)
        for start, num_reqs, _ in chunks
    ]
    sequences = [list(h) for h in histories]
    for step, step_tokens in enumerate(steps):
        scheduler_output = SchedulerOutput.make_empty()
        if step == 0:
            scheduler_output.scheduled_new_reqs = [
                SimpleNamespace(req_id=req_id) for req_id in input_batch.req_ids
            ]
        new_tokens = [
            torch.tensor(rows_tokens, dtype=torch.int32).to(device)
            for rows_tokens in step_tokens
        ]
        if new_tokens[0].shape[1] == 1:
            seeds = [t.view(-1).long() for t in new_tokens]
            drafts = proposer.propose(
                [], [], None, scheduler_output, True, device_seed=seeds
            )
        else:
            drafts = proposer.propose(
                None, [], None, scheduler_output, True, next_tokens_per_chunk=new_tokens
            )

        for (start, num_reqs, rows), chunk_tokens, chunk_drafts in zip(
            chunks, step_tokens, drafts, strict=True
        ):
            assert chunk_drafts.shape == (rows, k)
            for row in range(num_reqs):
                appended = [t for t in chunk_tokens[row] if t != X]
                sequences[start + row] += appended
                input_batch.num_tokens_no_spec[start + row] += len(appended)
                assert chunk_drafts[row].tolist() == _expected(
                    sequences[start + row], min_n, max_n, k
                )
            assert (chunk_drafts.cpu() >= 0).all()
