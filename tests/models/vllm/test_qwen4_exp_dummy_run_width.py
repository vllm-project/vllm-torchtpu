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
"""Dummy FA tables use pooled page geometry independently of other groups."""

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from vllm_torchtpu.models.vllm.qwen4_exp import inputs
from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter


def _make_adapter(monkeypatch, captured, *, page_size, context_limit, tokens, requests):
    def context(per_layer, *args, **kwargs):
        captured["per_layer"] = per_layer
        return nullcontext()

    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.setattr(inputs, "set_forward_context", context)
    monkeypatch.setattr(
        inputs, "set_vllm_model_wrapper_context", lambda **kwargs: nullcontext()
    )
    monkeypatch.setattr(inputs, "synchronize_tensors", lambda tensor: None)

    def forward(**kwargs):
        captured.update(kwargs)
        return torch.zeros((tokens, 1)), None

    runner = SimpleNamespace(
        device="cpu",
        _pp_is_first=True,
        supports_mm_inputs=False,
        _record_input_staging_fence=lambda: None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=requests),
        input_batch=SimpleNamespace(num_reqs=0),
        max_model_len=context_limit,
        most_model_len=None,
        vllm_config=SimpleNamespace(),
        mesh=None,
        _dp_num_tokens_across_dp=lambda n: torch.full((1,), n, dtype=torch.int32),
        forward_model=forward,
        compute_selected_logits=lambda output, indices: None,
        model_config=SimpleNamespace(hf_text_config=q4exp_text_config()),
        cache_config=SimpleNamespace(block_size=page_size),
    )
    adapter = Qwen4ExpAdapter(runner)
    adapter.cache = SimpleNamespace(
        allocate=lambda: None,
        pages=requests * math.ceil(context_limit / page_size) + 1,
        dummy_metadata=lambda metadata, active, **kwargs: {"qsa": metadata},
    )
    return adapter


@pytest.mark.parametrize(
    "page_size,context_limit,tokens,requests",
    [(1664, 32768, 8192, 8), (128, 4096, 1024, 4), (384, 6144, 2048, 3)],
)
@pytest.mark.parametrize("use_max_model_len", [False, True])
@pytest.mark.parametrize("other_group_columns", [None, 1, 64])
def test_dummy_fa_table_uses_context_and_page_geometry(
    monkeypatch,
    page_size,
    context_limit,
    tokens,
    requests,
    use_max_model_len,
    other_group_columns,
):
    captured = {}
    adapter = _make_adapter(
        monkeypatch,
        captured,
        page_size=page_size,
        context_limit=context_limit,
        tokens=tokens,
        requests=requests,
    )
    if not use_max_model_len:
        adapter.runner.most_model_len = context_limit // 2
    expected_context = context_limit if use_max_model_len else context_limit // 2
    expected_columns = math.ceil(expected_context / page_size)
    adapter.dummy_run(
        tokens,
        requests,
        num_blocks=other_group_columns,
        use_max_model_len=use_max_model_len,
    )
    metadata = captured["per_layer"]["qsa"]
    rows = metadata.block_tables.reshape(requests, expected_columns)
    query_lengths = metadata.query_start_loc[1:] - metadata.query_start_loc[:-1]
    assert int(query_lengths.sum()) == tokens
    next_page = 1
    for row, length in zip(rows, query_lengths.tolist(), strict=True):
        pages = math.ceil(length / page_size)
        torch.testing.assert_close(
            row[:pages], torch.arange(next_page, next_page + pages, dtype=row.dtype)
        )
        assert torch.count_nonzero(row[pages:]) == 0
        next_page += pages
    assert next_page <= adapter.cache.pages
