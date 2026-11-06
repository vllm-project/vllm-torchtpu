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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.output_processor import OutputProcessor, RequestState

from vllm_torchtpu.tracing.utils import (
    SPAN_ATTR_CACHED_TOKENS,
    SPAN_ATTR_TIME_BEFORE_QUEUE,
    _patch_vllm_request_tracing_phase_gaps,
    extract_kv_lens_for_tracing,
    extract_request_ids_for_tracing,
    trim_request_id_suffix,
)


def test_trim_request_id_suffix():
    # Case A: Standard long cmpl string correctly truncates to the first 6 parts.
    assert trim_request_id_suffix("cmpl-1-2-3-4-5-6-7-8") == "cmpl-1-2-3-4-5"

    # Case B: A short string is gracefully returned untouched.
    assert trim_request_id_suffix("cmpl-1234") == "cmpl-1234"

    # Case C: Non-"cmpl" prefixed IDs are bypassed.
    assert (
        trim_request_id_suffix("custom-req-id-1-2-3-4-5-6-7")
        == "custom-req-id-1-2-3-4-5-6-7"
    )


def test_extract_request_ids_for_tracing():
    req_ids = ["cmpl-1", "cmpl-2", None, "cmpl-3", "cmpl-4"]

    # Case A: Test pagination by passing start_index and num_reqs
    res = extract_request_ids_for_tracing(req_ids, start_index=1, num_reqs=2)
    # the active_ids will be `["cmpl-2", None]`
    # and the trimmed won't contain None
    assert res == {"request_id1": "cmpl-2"}

    # Check another slice
    res_2 = extract_request_ids_for_tracing(req_ids, start_index=3, num_reqs=2)
    assert res_2 == {"request_id1": "cmpl-3", "request_id2": "cmpl-4"}

    # Case C: Mock the internal components to fail and verify it swallows error
    class BadItem:
        def __str__(self):
            raise ValueError("Intentional crash")

    with patch("vllm_torchtpu.tracing.utils.logger.warning") as mock_warning:
        res_3 = extract_request_ids_for_tracing(["cmpl-1", BadItem()], 0, 2)
        assert res_3 == {}  # the list comprehension exception catches in outer try
        mock_warning.assert_called_once()


def test_extract_kv_lens_for_tracing():
    num_computed_tokens = [100, 200, 300, 400]

    # Case A: Test pagination by passing start_index and num_reqs
    res = extract_kv_lens_for_tracing(num_computed_tokens, start_index=1, num_reqs=2)
    assert res == {
        "kv_len1": 200,
        "kv_len2": 300,
        "min_kv_len": 200,
        "max_kv_len": 300,
        "avg_kv_len": 250,
    }

    # Case B: Default slicing (start_index=0, num_reqs=-1) uses the whole list
    res_2 = extract_kv_lens_for_tracing(num_computed_tokens)
    assert res_2 == {
        "kv_len1": 100,
        "kv_len2": 200,
        "kv_len3": 300,
        "kv_len4": 400,
        "min_kv_len": 100,
        "max_kv_len": 400,
        "avg_kv_len": 250,
    }

    # Case C: Empty slice yields no kwargs and no summary stats
    res_3 = extract_kv_lens_for_tracing(num_computed_tokens, start_index=4, num_reqs=2)
    assert res_3 == {}


def test_patch_vllm_request_tracing_phase_gaps_emits_attributes(monkeypatch):
    import vllm.v1.engine.output_processor as op_mod

    _patch_vllm_request_tracing_phase_gaps()
    # Calling a second time must be idempotent.
    _patch_vllm_request_tracing_phase_gaps()
    assert getattr(OutputProcessor, "_tpu_phase_gaps_patched", False) is True

    req_state = RequestState(
        request_id="req-internal-1",
        external_req_id="cmpl-bench-2p2d-0",
        parent_req=None,
        request_index=0,
        lora_request=None,
        output_kind=RequestOutputKind.FINAL_ONLY,
        prompt="hello",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        prompt_embeds=None,
        logprobs_processor=None,
        detokenizer=None,
        max_tokens_param=64,
        arrival_time=1000.0,
        queue=None,
        log_stats=True,
        stream_interval=1,
    )
    assert req_state.stats is not None

    req_state.num_cached_tokens = 112
    req_state.stats.queued_ts = 508.601
    req_state.stats.scheduled_ts = 508.6655
    req_state.stats.first_token_ts = 508.7073
    req_state.stats.last_token_ts = 509.1444
    req_state.stats.first_token_latency = 8.7073
    req_state.stats.num_generation_tokens = 64

    captured: dict[str, object] = {}

    def fake_instrument_manual(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(op_mod, "instrument_manual", fake_instrument_manual)

    processor = OutputProcessor(tokenizer=None, log_stats=True, tracing_enabled=True)
    engine_core_output = EngineCoreOutput(
        request_id="req-internal-1",
        new_token_ids=[42],
        trace_headers={"traceparent": "00-90d86cb618c6615e4dd701e734a854a7-0123456789abcdef-01"},
    )
    iteration_stats = SimpleNamespace(iteration_timestamp=1009.1525)

    processor.do_tracing(engine_core_output, req_state, iteration_stats)  # type: ignore[arg-type]

    assert op_mod.instrument_manual is fake_instrument_manual
    attrs = captured["attributes"]
    assert isinstance(attrs, dict)
    assert attrs[SPAN_ATTR_TIME_BEFORE_QUEUE] == pytest.approx(8.601, abs=1e-6)
    assert attrs[SPAN_ATTR_CACHED_TOKENS] == 112
