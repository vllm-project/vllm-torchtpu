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
"""Per-step index uploads from temporary host tensors go through
`stage_h2d_upload`, which keeps each host tensor alive until the runner's next
input-staging fence has been waited on: a non-blocking upload reads its host
tensor after the call returns. On CPU an upload returns its host tensor, so a
staged upload hands its consumer exactly a tensor held in
`_unfenced_h2d_sources`.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from runner_input_test_utils import _make_runner, _scheduler_output

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _ids(tensors):
    return {id(t) for t in tensors}


def _staged(runner):
    return [host for host, _ in runner._unfenced_h2d_sources]


def _substitution_runner(fast):
    return SimpleNamespace(
        _fast_token_substitution=fast,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.tensor([101, 102, 103, 104], dtype=torch.int64)
        ),
        _unfenced_h2d_sources=[],
        _registered_h2d_sources=set(),
    )


@pytest.mark.parametrize("fast", [False, True], ids=["plain", "cached"])
def test_token_substitution_stages_its_index_uploads(fast):
    """Both upload sites: the plain path, and the fast path's cached indices
    (indices that are not 1:1 skip the direct slice). With the fast path a
    second step with the same indices reuses the cached device indices and
    uploads nothing."""
    runner = _substitution_runner(fast)
    input_ids = torch.tensor([-1, 7, -1, 8], dtype=torch.int32)
    cur = np.array([0, 2], dtype=np.int32)
    pre = np.array([1, 0], dtype=np.int32)
    received = []

    def kernel(ids, cur_t, pre_t, _next_tokens):
        received.append((cur_t, pre_t))
        return ids

    with patch("vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token", kernel):
        TPUModelRunner._apply_async_token_substitution(runner, input_ids, cur, pre)
        staged = _staged(runner)
        assert len(staged) == 2
        assert _ids(received[0]) == _ids(staged)
        assert [t.tolist() for t in received[0]] == [[0, 2, 1, 3], [1, 0, -1, -1]]
        if fast:
            TPUModelRunner._apply_async_token_substitution(runner, input_ids, cur, pre)
            assert len(_staged(runner)) == 2  # nothing uploaded again
            assert _ids(received[1]) == _ids(received[0])


def test_logits_indices_upload_is_staged():
    """The default layout's logits indices are `query_start_loc - 1`, a
    temporary host tensor; its upload is staged, so the step's fence keeps it
    alive."""
    scheduled = [3, 2]
    runner = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=scheduled,
        scheduled_tokens=scheduled,
        token_paddings=[16],
        pcp_size=1,
    )
    prepared = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, num_decode_reqs=0
    )
    logits_indices = prepared[1]
    assert id(logits_indices) in _ids(_staged(runner))
    assert logits_indices.tolist()[:2] == [2, 4]
