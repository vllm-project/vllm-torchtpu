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
"""TPU_RPA_DECODE_BATCHED: the decode-program settings and how the batched
RPA entry hands them to the kernel."""

import functools
from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import attention, batched_rpa_decode

pytestmark = pytest.mark.cpu_test

MIB = 1 << 20


def _vllm_config(max_num_seqs, num_speculative_tokens=0, max_model_len=9216):
    spec = (
        SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
        if num_speculative_tokens
        else None
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=spec,
    )


@pytest.fixture
def v7x(monkeypatch):
    monkeypatch.setattr(
        batched_rpa_decode.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(vmem_capacity_bytes=64 * MIB),
    )


def test_off_by_default(monkeypatch, v7x):
    monkeypatch.delenv("TPU_RPA_DECODE_BATCHED", raising=False)
    assert batched_rpa_decode.decode_tuning(_vllm_config(4)) is None
    assert batched_rpa_decode.kernel_kwargs(None, 8) == {}


# The decode batch follows the engine's request slots, capped at the
# wrapper's default of 8: 4 at 4 users, 8 from 8 users up.
@pytest.mark.parametrize("max_num_seqs, batch", [(1, 1), (4, 4), (8, 8), (32, 8)])
def test_tuning_halves_vmem_and_caps_the_batch(monkeypatch, v7x, max_num_seqs, batch):
    monkeypatch.setenv("TPU_RPA_DECODE_BATCHED", "1")
    tuning = batched_rpa_decode.decode_tuning(_vllm_config(max_num_seqs))
    assert tuning == batched_rpa_decode.DecodeTuning(
        vmem_limit_bytes=32 * MIB, decode_batch_size=batch
    )


# An 8k/1k deployment takes the decode settings; a long-context one (AgentX
# serves max_model_len 262144) keeps the default, larger KV blocks.
@pytest.mark.parametrize(
    "max_model_len, takes",
    [(9216, True), (12288, True), (12289, False), (262144, False)],
)
def test_tuning_refused_past_the_model_length_bound(
    monkeypatch, v7x, max_model_len, takes
):
    monkeypatch.setenv("TPU_RPA_DECODE_BATCHED", "1")
    tuning = batched_rpa_decode.decode_tuning(
        _vllm_config(8, 3, max_model_len=max_model_len)
    )
    assert (tuning is not None) == takes
    kwargs = batched_rpa_decode.kernel_kwargs(tuning, 32)
    assert ("vmem_limit_bytes" in kwargs) == takes


@pytest.mark.parametrize(
    "max_num_seqs, k, armed", [(16, 3, "decode"), (32, 3, "default"), (64, 0, "decode")]
)
def test_tuning_logs_which_settings_decode_takes(
    monkeypatch, v7x, max_num_seqs, k, armed
):
    monkeypatch.setenv("TPU_RPA_DECODE_BATCHED", "1")
    messages = []
    monkeypatch.setattr(
        batched_rpa_decode.logger,
        "info_once",
        lambda msg, *args: messages.append(msg % args),
    )
    batched_rpa_decode.decode_tuning(_vllm_config(max_num_seqs, k))
    (message,) = messages
    assert f"runs the {armed} settings" in message
    assert f"<= {64 // (1 + k)} requests" in message


TUNING = batched_rpa_decode.DecodeTuning(vmem_limit_bytes=32 * MIB, decode_batch_size=4)


@pytest.mark.parametrize(
    "rows, decode_query_size, tuning, expected",
    [
        (64, 1, TUNING, {"vmem_limit_bytes": 32 * MIB, "decode_batch_size": 4}),
        (128, 1, TUNING, {}),
        (
            64,
            4,
            TUNING,
            {
                "vmem_limit_bytes": 32 * MIB,
                "decode_batch_size": 4,
                "decode_query_size": 4,
            },
        ),
        (128, 4, TUNING, {"decode_query_size": 4}),
        (8, 1, None, {}),
    ],
)
def test_batched_entry_passes_the_settings_to_the_kernel(
    monkeypatch, rows, decode_query_size, tuning, expected
):
    seen = {}

    def impl(*args, rpa_func, **kwargs):
        seen["rpa_func"] = rpa_func

    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", impl)
    query = SimpleNamespace(shape=(rows, 4, 128))
    attention._pallas_rpa_kernel_batched(
        None,
        query,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=None,
        decode_query_size=decode_query_size,
        decode_tuning=tuning,
    )
    rpa_func = seen["rpa_func"]
    if not expected:
        assert rpa_func is attention.ragged_paged_attention_batched
    else:
        assert isinstance(rpa_func, functools.partial)
        assert rpa_func.func is attention.ragged_paged_attention_batched
        assert rpa_func.keywords == expected
