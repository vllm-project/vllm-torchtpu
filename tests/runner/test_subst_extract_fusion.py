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
"""TPU_SPEC_SUBST_EXTRACT_FUSION: the fused program gives the eager fast
token substitution + extract_draft_token_ids values bit for bit, and the
fused path is taken exactly when the eager first fast path would be."""

import types

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    _substitute_and_extract_drafts,
    extract_draft_token_ids,
)


@pytest.mark.parametrize("num_reqs,k,src_reqs", [(1, 7, 4), (3, 7, 4), (4, 7, 4), (6, 7, 16)])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_fused_matches_eager(num_reqs, k, src_reqs, ids_dtype):
    gen = torch.Generator().manual_seed(num_reqs)
    n = num_reqs * (k + 1)
    src = torch.randint(0, 200000, (src_reqs * (k + 1),), generator=gen, dtype=torch.int32)
    input_ids = torch.zeros(n, dtype=ids_dtype)
    logits_idx = torch.arange(n, dtype=torch.int32)
    seg = torch.arange(num_reqs, dtype=torch.int32).repeat_interleave(k)
    target = torch.arange(num_reqs * k, dtype=torch.int32) + seg
    # eager: fast path 1 then the gather
    ref_ids = src.flatten().to(ids_dtype)[:n]
    ref_drafts = extract_draft_token_ids(ref_ids, logits_idx, target)
    ids, drafts = _substitute_and_extract_drafts(src, input_ids, logits_idx, target)
    assert ids.dtype == ref_ids.dtype and torch.equal(ids, ref_ids)
    assert drafts.dtype == ref_drafts.dtype and torch.equal(drafts, ref_drafts)


def _runner(src):
    return types.SimpleNamespace(
        _fast_token_substitution=True,
        _pre_async_results=types.SimpleNamespace(next_tokens_tpu=src),
    )


def test_identity_source_only_on_first_fast_path():
    fn = TPUModelRunner._identity_substitution_source
    src = torch.zeros(32, dtype=torch.int32)
    ids = torch.zeros(16, dtype=torch.int32)
    a = np.arange(16, dtype=np.int32)
    assert fn(_runner(src), ids, a, a) is src
    assert fn(_runner(src), ids, a[:8], a[:8]) is None  # not every row substituted
    b = a.copy(); b[[0, 1]] = b[[1, 0]]
    assert fn(_runner(src), ids, b, a) is None  # reordered rows
    assert fn(_runner(src), ids, a, b) is None
    assert fn(_runner(torch.zeros(8, dtype=torch.int32)), ids, a, a) is None  # short source
    r = _runner(src); r._fast_token_substitution = False
    assert fn(r, ids, a, a) is None
    r = _runner(None)
    assert fn(r, ids, a, a) is None
