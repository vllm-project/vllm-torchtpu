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
"""Runner host-path shortcuts with and without the PCP manager.

The host path skips work when every chunk is a plain slice of the request
list: a max-model-len chunk takes the next `num_reqs_max_model_len`
requests, the first request over `most_model_len` settles a chunk, and the
async token substitution of an identity layout is cached. The PCP manager
breaks all three: it sizes a chunk from every remaining request's tokens, and
it repacks tokens rank-major. These tests pin both sides with stand-in
runners.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

MAX_MODEL_LEN_REQS = 8
MOST_MODEL_LEN_REQS = 16


class _Manager:
    """PCP manager stand-in that records the requests it sizes chunks from."""

    def __init__(self, per_chunk, plan=None):
        self.per_chunk = per_chunk
        self.seen = []
        self.sharding_plan = plan

    def num_reqs_in_chunk(self, q_lens, seq_lens, capacity):
        assert len(q_lens) == len(seq_lens)
        self.seen.append(len(q_lens))
        return min(self.per_chunk, len(q_lens))


class _Reserved(Exception):
    """Stops _prepare_inputs at the host reservation, carrying its size."""


def _runner(tokens, most_model_len, manager):
    n = len(tokens)
    req_ids = [f"r{i}" for i in range(n)]

    def reserve(_runner, num_tokens):
        raise _Reserved(num_tokens)

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=n,
            req_ids=req_ids,
            num_computed_tokens_cpu=np.zeros(n, dtype=np.int32),
        ),
        most_model_len=most_model_len,
        pcp_manager=manager,
        num_reqs_max_model_len=MAX_MODEL_LEN_REQS,
        num_reqs_most_model_len=MOST_MODEL_LEN_REQS,
        reorder_batch_threshold=1,
        _wait_input_staging_fence=lambda: None,
        get_pcp_planner=lambda: SimpleNamespace(reserve_host_token_capacity=reserve),
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=sum(tokens),
        num_scheduled_tokens=dict(zip(req_ids, tokens)),
    )
    return runner, scheduler_output


def _reserved_tokens(runner, scheduler_output):
    with pytest.raises(_Reserved) as stop:
        TPUModelRunner._prepare_inputs(runner, scheduler_output, 0, 0)
    return stop.value.args[0]


def test_count_chunks_with_the_manager_and_no_most_model_len():
    manager = _Manager(per_chunk=2)
    runner, so = _runner([5, 5, 5, 5, 5], None, manager)
    assert TPUModelRunner._count_input_chunks(runner, so) == (3, 2)
    # Every chunk is sized from all the requests that remain.
    assert manager.seen == [5, 3, 1]


def test_count_chunks_with_the_manager_reads_past_a_long_request():
    manager = _Manager(per_chunk=2)
    runner, so = _runner([5, 500, 5, 5, 5], 100, manager)
    assert TPUModelRunner._count_input_chunks(runner, so) == (3, 2)
    assert manager.seen == [5, 3, 1]


def test_count_chunks_without_the_manager_keeps_the_shortcuts():
    runner, so = _runner([1] * 20, None, None)
    assert TPUModelRunner._count_input_chunks(runner, so) == (3, 8)
    runner, so = _runner([1] * 20, 100, None)
    assert TPUModelRunner._count_input_chunks(runner, so) == (2, 16)
    # A long request makes the first chunk a max-model-len chunk of 8; the
    # 12 short requests left fit one most-model-len chunk.
    runner, so = _runner([1, 500] + [1] * 18, 100, None)
    assert TPUModelRunner._count_input_chunks(runner, so) == (2, 12)


def test_prepare_inputs_sizes_the_chunk_with_the_manager():
    manager = _Manager(per_chunk=2)
    runner, so = _runner([3, 5, 7, 11, 13], None, manager)
    # The host reservation covers the two requests the manager chose.
    assert _reserved_tokens(runner, so) == 3 + 5
    assert manager.seen == [5]


def test_prepare_inputs_without_the_manager_takes_the_max_model_len_slice():
    runner, so = _runner([3, 5, 7, 11, 13] * 2, None, None)
    # The first num_reqs_max_model_len requests, without asking anyone.
    assert _reserved_tokens(runner, so) == sum([3, 5, 7, 11, 13] + [3, 5, 7])


class _Plan:
    """Rank-major plan stand-in: token i is local at 100 + i when i is even."""

    def local_index_for_request_major_token(self, i):
        return 100 + i if i % 2 == 0 else None


def _substitution_runner(manager):
    return SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["a", "b", "c"]),
        _pre_async_results=SimpleNamespace(
            req_id_to_index_copy={"a": 0, "b": 1, "c": 2},
            spec_decode_num_rejected_tokens=None,
        ),
        speculative_config=None,
        _last_sequence_layout_plan=None,
        pcp_manager=manager,
    )


def test_async_substitution_goes_through_the_manager_plan():
    runner = _substitution_runner(_Manager(per_chunk=1, plan=_Plan()))
    cur, src = TPUModelRunner._prepare_async_token_substitution_indices(
        runner, 0, 3, np.asarray([1, 1, 1], dtype=np.int32)
    )
    assert cur.tolist() == [100, 102]
    assert src.tolist() == [0, 2]
    # A rank-major mapping is never cached as an identity mapping.
    assert not getattr(runner, "_async_subst_prep_cache", None)


def test_async_substitution_without_the_manager_is_the_cached_identity():
    runner = _substitution_runner(None)
    cur, src = TPUModelRunner._prepare_async_token_substitution_indices(
        runner, 0, 3, np.asarray([1, 1, 1], dtype=np.int32)
    )
    assert cur.tolist() == [0, 1, 2]
    assert src.tolist() == [0, 1, 2]
    assert len(runner._async_subst_prep_cache) == 1
