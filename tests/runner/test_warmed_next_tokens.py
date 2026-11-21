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
"""The substitution source is padded and cast to a source the precompile warmed."""

from types import SimpleNamespace

import torch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(*, async_drafter: bool, K: int = 3, max_seqs: int = 32):
    runner = SimpleNamespace(
        num_reqs_paddings=[8, 16, 32],
        _is_async_drafter=async_drafter,
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        scheduler_config=SimpleNamespace(max_num_seqs=max_seqs),
        max_num_reqs=max_seqs,
        num_reqs_max_model_len=max_seqs,
        num_reqs_most_model_len=None,
        most_model_len=None,
        _substitute_lens_cache=None,
    )
    for name in (
        "_chunk_request_capacities",
        "_async_spec_source_rows",
        "_substitute_next_lens",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


def _warm(runner, t):
    return TPUModelRunner._warmed_next_tokens(runner, t)


def test_async_drafter_source_is_cast_to_the_warmed_int64_span():
    """An async drafter parks a fixed (rows * (1 + K)) source of int32 drafts;
    the precompile warmed that length as int64."""
    runner = _runner(async_drafter=True)
    bound = runner._async_spec_source_rows() * 4
    assert runner._substitute_next_lens() == [8, 16, 32, bound]
    src = torch.arange(bound, dtype=torch.int32)
    out = _warm(runner, src)
    assert out.dtype == torch.int64 and out.shape == (bound,)
    assert torch.equal(out, src.long())
    warmed = torch.arange(bound, dtype=torch.int64)
    assert _warm(runner, warmed) is warmed
    big = torch.arange(bound + 1, dtype=torch.int64)
    assert _warm(runner, big) is big  # no warmed length fits: unchanged


def test_stride_one_source_pads_to_the_next_request_bucket():
    """Without an async drafter, several chunks' sampled tokens concatenated
    give an odd length; it is padded to the next num_reqs bucket."""
    runner = _runner(async_drafter=False)
    assert runner._substitute_next_lens() == [8, 16, 32]
    t = torch.arange(12, dtype=torch.int64)
    out = _warm(runner, t)
    assert out.shape == (16,) and torch.equal(out[:12], t) and not out[12:].any()
    two_d = torch.arange(20, dtype=torch.int32).view(20, 1)
    out = _warm(runner, two_d)
    assert out.dtype == torch.int64 and out.shape == (32,)
    assert torch.equal(out[:20], two_d.reshape(-1).long())
    same = torch.arange(16, dtype=torch.int64)
    assert _warm(runner, same) is same
    assert runner._substitute_lens_cache == [8, 16, 32]
