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
"""Every substitution source the runtime parks is one the precompile warmed."""

import contextlib
from types import SimpleNamespace

import torch

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(*, async_drafter: bool, K: int = 3, max_seqs: int = 32, chunk: int = 8):
    # Chunks of `chunk` requests: a step of max_seqs requests parks up to
    # max_seqs / chunk chunks, more rows than one chunk's largest bucket.
    runner = SimpleNamespace(
        num_reqs_paddings=[8],
        _is_async_drafter=async_drafter,
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        scheduler_config=SimpleNamespace(max_num_seqs=max_seqs, async_scheduling=True),
        max_num_reqs=chunk,
        num_reqs_max_model_len=chunk,
        num_reqs_most_model_len=None,
        most_model_len=None,
        num_tokens_paddings=[16],
        device=torch.device("cpu"),
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    for name in (
        "_chunk_request_capacities",
        "_async_spec_source_rows",
        "_substitute_source_rows",
        "_pad_substitute_source",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


def _warmed_sources(runner, monkeypatch):
    warmed = []

    def substitute(input_ids, cur, pre, next_tokens):
        warmed.append((tuple(next_tokens.shape), next_tokens.dtype))
        return input_ids

    monkeypatch.setattr(tpu_runner, "_substitute_placeholder_token", substitute)
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: None)
    TPUModelRunner._precompile_substitute_placeholder_token(runner)
    return set(warmed)


def test_async_drafter_span_is_warmed_with_its_int32_drafts(monkeypatch):
    """The async drafter parks [bonus, drafts] in the drafts' dtype (int32
    for eagle3); the precompile warms that span as int32, not only int64."""
    runner = _runner(async_drafter=True)
    span = runner._async_spec_source_rows() * 4
    warmed = _warmed_sources(runner, monkeypatch)
    assert ((span,), torch.int32) in warmed
    assert ((span,), torch.int64) in warmed


def test_multi_chunk_stride_one_source_pads_to_a_warmed_row_count(monkeypatch):
    """Without an async drafter, several chunks' [num_reqs, 1] sampled tokens
    concatenated give a row count no single chunk parks; it is padded to a
    warmed one."""
    runner = _runner(async_drafter=False)
    warmed = _warmed_sources(runner, monkeypatch)
    rows = runner._async_spec_source_rows()
    assert rows > 8 and runner._substitute_source_rows() == [8, rows]
    assert {((8, 1), torch.int64), ((rows, 1), torch.int64)} == warmed
    src = torch.arange(12, dtype=torch.int64).view(12, 1)
    out = runner._pad_substitute_source(src)
    assert out.shape == (rows, 1)
    assert torch.equal(out[:12], src) and not out[12:].any()
    assert (tuple(out.shape), out.dtype) in warmed
    single = torch.arange(8, dtype=torch.int64).view(8, 1)
    assert runner._pad_substitute_source(single) is single
    flat = torch.arange(12, dtype=torch.int64)
    assert runner._pad_substitute_source(flat) is flat
