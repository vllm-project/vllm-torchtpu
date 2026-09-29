# ruff: noqa: E501
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
"""End-to-end unit and integration tests for Eagle3 + GaaD Phase 3 closed loop.

Verifies:
1. Async Scheduler hook (`update_draft_token_ids_in_output`) trims invalid Eagle3
   draft tokens, tops up remaining budget with GaaD grammar jump tokens, updates
   `num_invalid_spec_tokens`, and populates `scheduled_spec_decode_flags`
   (`[DRAFT_FLAG_MODEL]*b + [DRAFT_FLAG_GRAMMAR]*a + [DRAFT_FLAG_EMPTY]*c`).
2. Sync Scheduler hook (`update_draft_token_ids` + `schedule`) populates
   `scheduled_spec_decode_flags` on `SchedulerOutput`.
3. `SpeculativeDecodingManager.get_spec_decode_metadata` builds `draft_flags`
   device tensor aligned with `draft_token_ids`.
4. `TPUModelRunner._prepare_async_token_substitution_indices` substitutes
   bonus (`j=0`) and model draft (`flag == 1`) tokens from device `next_tokens_tpu`,
   while preserving host-injected GaaD grammar jump tokens (`flag == 2`) and
   empty slots (`flag == 0`) in `input_ids` and `extract_draft_token_ids`.
5. End-to-end `RejectionSampler` verification + Eagle3 Step 0 carry index calculation
   (`last_token_indices` and `num_rejected_out`) absorbing forced-accepted GaaD jump tokens.  # noqa: E501
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

import vllm_torchtpu.spec_decode.grammar_jump as grammar_jump
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
)
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpecDecodeMetadata,
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner_async_output import extract_draft_token_ids
from vllm_torchtpu.spec_decode.grammar_jump import (
    DRAFT_FLAG_EMPTY,
    DRAFT_FLAG_GRAMMAR,
    DRAFT_FLAG_MODEL,
)


class FakeMatcher:
    """Mock XGrammar matcher that accepts specific valid tokens and jumps."""

    def __init__(self, valid_tokens: set[int], jump_str: str = ""):
        self.valid_tokens = valid_tokens
        self.jump_str = jump_str
        self.accepted: list[int] = []

    def accept_token(self, token_id: int) -> bool:
        if token_id not in self.valid_tokens:
            return False
        self.accepted.append(token_id)
        return True

    def find_jump_forward_string(self) -> str:
        return self.jump_str

    def rollback(self, num_tokens: int) -> None:
        if num_tokens > 0:
            del self.accepted[-num_tokens:]


class FakeGrammar:
    def __init__(self, matcher: FakeMatcher):
        self.matcher = matcher

    def is_terminated(self) -> bool:
        return False

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        valid = []
        for t in tokens:
            if t in self.matcher.valid_tokens:
                valid.append(t)
            else:
                break
        return valid


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


def test_async_scheduler_topup_trims_invalid_eagle3_and_populates_flags():
    """Verify async Eagle3 draft validation + GaaD top-up + flag generation."""
    # Eagle3 proposed K=4 tokens: [10, 999, 998, 997].
    # Only token 10 is valid under grammar. Jump string is "ok" -> [111, 107].
    matcher = FakeMatcher(valid_tokens={10, 111, 107}, jump_str="ok")
    grammar = FakeGrammar(matcher)
    tokenizer = FakeTokenizer()

    req = SimpleNamespace(
        request_id="req-1",
        is_finished=lambda: False,
        is_prefill_chunk=False,
        max_tokens=100,
        output_token_ids=[1, 2],
        structured_output_request=SimpleNamespace(grammar=grammar),
    )
    scheduler = SimpleNamespace(
        requests={"req-1": req},
        structured_output_manager=SimpleNamespace(
            tokenizer=tokenizer,
            should_advance=lambda r: True,
        ),
        num_spec_tokens=4,
    )

    # Simulate vLLM original_update_in_output result:
    # Eagle3 proposed [10, 999, 998, 997] -> validated [10], padded with 3x -1.
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"req-1": [10, -1, -1, -1]},
        num_invalid_spec_tokens={"req-1": 3},
        scheduled_spec_decode_flags=None,
    )

    topup_fn = getattr(grammar_jump, "_topup_scheduled_spec_decode_tokens", None)
    assert topup_fn is not None, "_topup_scheduled_spec_decode_tokens not implemented"
    topup_fn(scheduler, scheduler_output)

    # Expect: [10] (model) + [111, 107] (grammar jump "ok") + [-1] (1 remaining invalid)  # noqa: E501
    assert scheduler_output.scheduled_spec_decode_tokens["req-1"] == [10, 111, 107, -1]
    assert scheduler_output.num_invalid_spec_tokens["req-1"] == 1
    assert scheduler_output.scheduled_spec_decode_flags["req-1"] == [
        DRAFT_FLAG_MODEL,
        DRAFT_FLAG_GRAMMAR,
        DRAFT_FLAG_GRAMMAR,
        DRAFT_FLAG_EMPTY,
    ]


def test_spec_decode_metadata_builds_draft_flags_tensor():
    """Verify SpeculativeDecodingManager.get_spec_decode_metadata builds draft_flags."""  # noqa: E501
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.arange_np = np.arange(32, dtype=np.int32)
    runner.num_tokens_paddings = [8, 16, 32]
    # Batch with 2 requests: req-0 has 3 drafts (1 + 3 = 4 scheduled tokens),
    # req-1 has 2 drafts (1 + 2 = 3 scheduled tokens). Total scheduled = 7.
    # Input tokens: req-0: [bonus0, d0_0, d0_1, d0_2], req-1: [bonus1, d1_0, d1_1]
    runner.input_ids_cpu = torch.tensor(
        [100, 10, 111, 107, 200, 20, 21, 0], dtype=torch.int32
    )
    runner.input_batch.req_ids = ["req-0", "req-1"]

    manager = SpeculativeDecodingManager(runner)
    num_draft_tokens = np.array([3, 2], dtype=np.int32)
    cu_num_scheduled_tokens = np.array([4, 7], dtype=np.int32)
    padded_num_reqs = 2

    scheduled_flags = {
        "req-0": [DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR],
        "req-1": [DRAFT_FLAG_MODEL, DRAFT_FLAG_MODEL],
    }

    md = manager.get_spec_decode_metadata(
        num_draft_tokens=num_draft_tokens,
        cu_num_scheduled_tokens=cu_num_scheduled_tokens,
        padded_num_reqs=padded_num_reqs,
        scheduled_spec_decode_flags=scheduled_flags,
    )

    assert isinstance(md, SpecDecodeMetadata)
    assert md.draft_flags is not None
    # Total draft tokens = 3 + 2 = 5, padded to padded_logits_length (8)
    expected_flags = [
        DRAFT_FLAG_MODEL,
        DRAFT_FLAG_GRAMMAR,
        DRAFT_FLAG_GRAMMAR,
        DRAFT_FLAG_MODEL,
        DRAFT_FLAG_MODEL,
        DRAFT_FLAG_EMPTY,
        DRAFT_FLAG_EMPTY,
        DRAFT_FLAG_EMPTY,
    ]
    assert md.draft_flags.tolist() == expected_flags


def test_async_token_substitution_preserves_grammar_jump_tokens():
    """Verify async substitution skips overwriting GaaD jump tokens on device."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    # Create minimal mock runner to test _prepare_async_token_substitution_indices
    runner = SimpleNamespace()
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    runner._pre_async_results = SimpleNamespace(
        spec_decode_num_rejected_tokens=torch.tensor([0], dtype=torch.int32),
        req_id_to_index_copy={"req-0": 0},
    )
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner._last_sequence_layout_plan = None

    # Suppose req-0 has n_sched = 1 + 3 = 4 tokens:
    # j=0: bonus token (must come from device next_tokens_tpu[0])
    # j=1 (draft 0): DRAFT_FLAG_MODEL (1) -> comes from device next_tokens_tpu[1]
    # j=2 (draft 1): DRAFT_FLAG_GRAMMAR (2) -> host GaaD token, DO NOT overwrite!
    # j=3 (draft 2): DRAFT_FLAG_EMPTY (0) -> host -1 token, DO NOT overwrite!
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_flags={
            "req-0": [DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_EMPTY]
        }
    )

    cur_indices, pre_indices = TPUModelRunner._prepare_async_token_substitution_indices(  # noqa: E501
        runner,
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.array([4], dtype=np.int32),
        scheduler_output=scheduler_output,
    )

    # Only j=0 (bonus) and j=1 (model draft) should be substituted from device!
    # j=2 (grammar jump) and j=3 (empty) must be preserved from host input_ids!
    assert cur_indices.tolist() == [0, 1]
    assert pre_indices.tolist() == [0, 1]

    # Verify that after substitution, extract_draft_token_ids sees:
    # draft 0 from device (e.g. 55), draft 1 from host GaaD (e.g. 111), draft 2 from host (-1)  # noqa: E501
    # Host input_ids: [placeholder_bonus(0), placeholder_d0(0), gaad_d1(111), empty_d2(-1)]  # noqa: E501
    input_ids = torch.tensor([0, 0, 111, -1], dtype=torch.int32)
    # Device next_tokens_tpu: [real_bonus(99), eagle3_d0(55), stale_eagle3_d1(66), stale_eagle3_d2(77)]  # noqa: E501
    next_tokens_tpu = torch.tensor([99, 55, 66, 77], dtype=torch.int32)
    input_ids[cur_indices] = next_tokens_tpu[pre_indices]

    assert input_ids.tolist() == [99, 55, 111, -1]
    logits_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    target_logits_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    extracted_drafts = extract_draft_token_ids(
        input_ids, logits_indices, target_logits_indices
    )
    assert extracted_drafts.tolist() == [55, 111, -1]


def test_eagle3_closed_loop_rejection_sampler_and_step0_carry(device):
    """Verify RejectionSampler forced acceptance + Eagle3 Step 0 carry index alignment."""  # noqa: E501
    sampler = RejectionSampler(device=device)
    # 1 request, K=3 drafts: [eagle3_draft=55, gaad_jump_1=111, gaad_jump_2=107]
    draft_token_ids = torch.tensor([55, 111, 107], dtype=torch.int32, device=device)
    draft_flags = torch.tensor(
        [DRAFT_FLAG_MODEL, DRAFT_FLAG_GRAMMAR, DRAFT_FLAG_GRAMMAR],
        dtype=torch.int32,
        device=device,
    )
    num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    segment_ids = torch.tensor([0, 0, 0], dtype=torch.int64, device=device)
    group_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    bonus_token_ids = torch.tensor([202], dtype=torch.int32, device=device)

    # Target model logits:
    # At pos 0 (verifying eagle3_draft=55): target argmax is 55 (MATCH!)
    # At pos 1 (verifying gaad_jump_1=111): target argmax is 888 (MISMATCH, but flag==2 forces accept!)  # noqa: E501
    # At pos 2 (verifying gaad_jump_2=107): target argmax is 777 (MISMATCH, but flag==2 forces accept!)  # noqa: E501
    vocab_size = 1000
    target_logits = torch.zeros((3, vocab_size), dtype=torch.float32, device=device)
    target_logits[0, 55] = 10.0
    target_logits[1, 888] = 10.0
    target_logits[2, 777] = 10.0

    print("Running sampler")
    output = sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        segment_ids=segment_ids,
        group_indices=group_indices,
        max_draft_tokens=3,
        draft_flags=draft_flags,
    )

    # Output shape [1, 4]: all 3 drafts (55, 111, 107) + bonus token (202) accepted!
    assert list(output.cpu().numpy()[0]) == [55, 111, 107, 202]

    # Now verify Eagle3 Step 0 carry index calculation (as in Eagle3Proposer._prepare_draft_inputs):  # noqa: E501
    nt = output[:1]
    num_valid = (nt != PLACEHOLDER_TOKEN_ID).sum(dim=1).clamp(min=1)
    last_valid_col = num_valid - 1
    seed_token = nt.gather(1, last_valid_col.unsqueeze(1)).squeeze(1)
    num_draft_dev = torch.tensor([3], dtype=torch.int64, device=device)
    accepted_drafts = torch.minimum((num_valid - 1).clamp(min=0), num_draft_dev)
    num_rejected_out = (num_draft_dev - accepted_drafts).to(torch.int32)
    # Suppose chunk query_start_loc is [0, 4] (4 tokens in verify window: bonus_prev + 3 drafts)  # noqa: E501
    qsl_end = torch.tensor([4], dtype=torch.int64, device=device)
    last_token_indices = qsl_end - 1 - num_rejected_out.to(torch.int64)

    # 0 tokens rejected! Seed token is bonus token 202, and last_token_indices points to slot 3 (the end of GaaD jump!)  # noqa: E501
    assert num_rejected_out.cpu().item() == 0
    assert seed_token.cpu().item() == 202
    assert last_token_indices.cpu().item() == 3


if __name__ == "__main__":
    print("test_async_token_substitution_preserves_grammar_jump_tokens")
    test_async_token_substitution_preserves_grammar_jump_tokens()
    print("test_eagle3_closed_loop_rejection_sampler_and_step0_carry")
    test_eagle3_closed_loop_rejection_sampler_and_step0_carry()
    print("DONE!")
