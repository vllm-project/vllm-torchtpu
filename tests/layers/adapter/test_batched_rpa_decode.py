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
"""Decode lanes of the batched RPA kernel: the shape table and how the batched
RPA entry hands them to the kernel."""

import functools
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter import attention, batched_rpa_decode

pytestmark = pytest.mark.cpu_test

FP8 = torch.float8_e4m3fn


@pytest.mark.parametrize(
    "num_kv_heads, head_dim, dtype, max_model_len, lanes",
    [
        # The measured shape: one KV head per device, head_dim 256, fp8 KV,
        # long context (the DP4xTP2 decode leg serves max_model_len 262144).
        (1, 256, FP8, 262144, 4),
        (1, 256, torch.float8_e5m2, 98304, 4),
        # Anything else keeps the wrapper default.
        (1, 256, FP8, 98303, None),
        (1, 256, FP8, 8192, None),
        (2, 256, FP8, 262144, None),
        (1, 128, FP8, 262144, None),
        (1, 256, None, 262144, None),
        (1, 256, torch.uint8, 262144, None),
    ],
)
def test_decode_lanes_table(num_kv_heads, head_dim, dtype, max_model_len, lanes):
    assert (
        batched_rpa_decode.decode_lanes(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_cache_dtype=dtype,
            max_model_len=max_model_len,
        )
        == lanes
    )


def _entry_rpa_func(monkeypatch, *, rows, decode_query_size, decode_lanes):
    seen = {}

    def impl(*args, rpa_func, **kwargs):
        seen["rpa_func"] = rpa_func

    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", impl)
    attention._pallas_rpa_kernel_batched(
        None,
        SimpleNamespace(shape=(rows, 16, 256)),
        *([None] * 10),
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=None,
        decode_query_size=decode_query_size,
        decode_lanes=decode_lanes,
    )
    return seen["rpa_func"]


@pytest.mark.parametrize("rows", [64, 112, 2048])
def test_decode_lanes_reach_the_kernel(monkeypatch, rows):
    rpa_func = _entry_rpa_func(
        monkeypatch, rows=rows, decode_query_size=4, decode_lanes=4
    )
    assert isinstance(rpa_func, functools.partial)
    assert rpa_func.func is attention.ragged_paged_attention_batched
    assert rpa_func.keywords == {"decode_batch_size": 4, "decode_query_size": 4}


def test_no_lanes_keeps_the_kernel_default(monkeypatch):
    rpa_func = _entry_rpa_func(
        monkeypatch, rows=64, decode_query_size=1, decode_lanes=None
    )
    assert rpa_func is attention.ragged_paged_attention_batched
    rpa_func = _entry_rpa_func(
        monkeypatch, rows=64, decode_query_size=4, decode_lanes=None
    )
    assert rpa_func.keywords == {"decode_query_size": 4}
