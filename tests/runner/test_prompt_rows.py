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
"""PromptTokenIds: the same list, and the same persistent-batch rows."""

import copy
import pickle
import random

import numpy as np
import torch
from vllm import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_torchtpu.runner.prompt_rows import PromptTokenIds, cache_prompt_rows


def test_is_the_same_list():
    ids = [5, 7, 11, 13]
    prompt = PromptTokenIds(ids)
    assert prompt == ids and isinstance(prompt, list)
    assert prompt[1:3] == [7, 11] and type(prompt[1:3]) is list
    assert list(prompt) == ids and len(prompt) == 4 and prompt[-1] == 13
    assert prompt + [17] == ids + [17]


def test_numpy_reads_the_array_until_the_list_changes():
    prompt = PromptTokenIds([5, 7, 11], np.array([5, 7, 11], dtype=np.int32))
    assert np.asarray(prompt) is prompt._array
    row = np.zeros(8, dtype=np.int32)
    row[:3] = prompt
    assert row[:3].tolist() == [5, 7, 11]
    for change in (
        lambda p: p.append(13),
        lambda p: p.extend([13]),
        lambda p: p.__setitem__(0, 3),
        lambda p: p.__delitem__(0),
        lambda p: p.insert(0, 3),
        lambda p: p.pop(),
        lambda p: p.remove(7),
        lambda p: p.sort(reverse=True),
        lambda p: p.reverse(),
        lambda p: p.__iadd__([13]),
    ):
        prompt = PromptTokenIds([5, 7, 11], np.array([5, 7, 11], dtype=np.int32))
        change(prompt)
        assert np.asarray(prompt).tolist() == list(prompt)


def test_copies_and_pickles_as_the_same_ids():
    prompt = PromptTokenIds([5, 7, 11], np.array([5, 7, 11], dtype=np.int32))
    for clone in (pickle.loads(pickle.dumps(prompt)), copy.deepcopy(prompt)):
        assert clone == prompt and isinstance(clone, PromptTokenIds)
        assert np.asarray(clone).tolist() == [5, 7, 11]


def _request(req_id: str, prompt: list[int], output: list[int]) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        mm_features=[],
        block_ids=([1, 2],),
        generator=None,
        num_computed_tokens=len(prompt) + len(output),
        output_token_ids=list(output),
    )


def _batch() -> InputBatch:
    return InputBatch(
        max_num_reqs=4,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        device=torch.device("cpu"),
        vocab_size=50000,
        block_sizes=[16],
        max_num_blocks_per_req=[128],
    )


def _rows(batch: InputBatch) -> dict[str, list[int]]:
    return {
        req_id: batch.token_ids_cpu[i, : batch.num_tokens_no_spec[i]].tolist()
        for req_id, i in batch.req_id_to_index.items()
    }


def test_re_added_requests_get_the_rows_the_list_gives():
    rng = random.Random(0)
    specs = [
        (f"r{i}", [rng.randrange(50000) for _ in range(n)], [rng.randrange(50000)])
        for i, n in enumerate((1500, 700, 1999))
    ]
    lists = {rid: _request(rid, p, o) for rid, p, o in specs}
    cached = {rid: _request(rid, p, o) for rid, p, o in specs}
    plain_batch, cached_batch = _batch(), _batch()
    for rid in lists:
        plain_batch.add_request(lists[rid])
        cached_batch.add_request(cached[rid])
    cache_prompt_rows(cached, cached_batch, list(cached))
    for rid, prompt, _ in specs:
        assert isinstance(cached[rid].prompt_token_ids, PromptTokenIds)
        assert cached[rid].prompt_token_ids == prompt
    # r1 and r0 sit a step out, then come back in the other order.
    for batch in (plain_batch, cached_batch):
        batch.remove_request("r1")
        batch.remove_request("r0")
        batch.condense()
    for rid in ("r1", "r0"):
        plain_batch.add_request(lists[rid])
        cached_batch.add_request(cached[rid])
    assert _rows(cached_batch) == _rows(plain_batch)
    assert cached_batch.req_id_to_index == plain_batch.req_id_to_index
    for rid, prompt, output in specs:
        assert _rows(cached_batch)[rid] == prompt + output
