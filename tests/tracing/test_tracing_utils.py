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
import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.sampling_params import RequestOutputKind
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.output_processor import OutputProcessor, RequestState

from vllm_torchtpu.tracing.utils import (
    SPAN_ATTR_BLOCKED_BY_PREV_COMPILE,
    SPAN_ATTR_COMPILE_TIME,
    SPAN_ATTR_TIME_IN_KV_TRANSFER,
    _patch_vllm_engine_core_tracing_attribution,
    _poll_xla_cache_compile_secs,
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


def test_patch_vllm_engine_core_tracing_attribution_end_to_end(monkeypatch):
    import torch
    import vllm.v1.engine.output_processor as op_mod
    import vllm_torchtpu.tracing.utils as tracing_utils

    _patch_vllm_engine_core_tracing_attribution()
    _patch_vllm_engine_core_tracing_attribution()
    assert getattr(Scheduler, "_tpu_engine_attribution_patched", False) is True
    assert getattr(OutputProcessor, "_tpu_engine_attribution_patched", False) is True

    # 1. Verify _poll_xla_cache_compile_secs reads torch.tpu._get_cache_stats()
    monkeypatch.setattr(tracing_utils, "_xla_cache_checked_at", None)
    fake_entries = [
        SimpleNamespace(
            last_read=10,
            read_count=1,
            compilation_duration=datetime.timedelta(seconds=8.718),
        )
    ]
    fake_tpu = SimpleNamespace(
        _get_cache_stats=lambda: SimpleNamespace(per_entry_stats=fake_entries)
    )
    monkeypatch.setattr(torch, "tpu", fake_tpu, raising=False)

    clock = [100.0]
    monkeypatch.setattr("vllm_torchtpu.tracing.utils.time.monotonic", lambda: clock[0])

    # 2. Simulate step 1 compiling for 8.718s on req-a, followed by req-b arriving
    req_a = SimpleNamespace(
        request_id="req-a",
        trace_headers={"traceparent": "00-aaaa-bbbb-01"},
    )
    req_b = SimpleNamespace(
        request_id="req-b",
        trace_headers={"traceparent": "00-90d86cb618c6615e4dd701e734a854a7-0123456789abcdef-01"},
        num_computed_tokens=128,
        num_tokens=128,
        _tpu_kv_xfer_start_mono=100.0,
    )

    dummy_sched = SimpleNamespace(
        requests={"req-a": req_a, "req-b": req_b},
        skipped_waiting=[req_b],
        connector=object(),
        failed_recving_kv_req_ids=set(),
        finished_recving_kv_req_ids={"req-b"},
        kv_cache_manager=SimpleNamespace(cache_blocks=lambda req, n: None),
        prefix_replay_tokens=0,
        _mark_prefix_replay=lambda req, n: 0,
        _load_restores_replay_window=lambda *a, **kw: False,
        has_requests=lambda: True,
    )

    # Measure KV transfer wait on req_b (64.5 ms in WAITING_FOR_REMOTE_KVS)
    clock[0] = 100.0645
    Scheduler._update_waiting_for_remote_kv(dummy_sched, req_b)  # type: ignore[arg-type]
    assert req_b._tpu_kv_xfer_s == pytest.approx(0.0645, abs=1e-6)
    assert req_b._tpu_kv_xfer_start_mono is None

    # Stub orig_update_from_output and orig_add_request on a Scheduler subclass
    ec_out_b = EngineCoreOutput(
        request_id="req-b",
        new_token_ids=[42],
        finish_reason="stop",  # type: ignore[arg-type]
        trace_headers=dict(req_b.trace_headers),
    )
    monkeypatch.setattr(
        Scheduler.update_from_output,
        "__wrapped__",
        lambda self, s_out, m_out: {0: EngineCoreOutputs(outputs=[ec_out_b])},
        raising=False,
    )

    # Step 1 finishes with 8.718s XLA compile while dummy_sched.has_requests() is True
    sched_out_1 = SimpleNamespace(num_scheduled_tokens={"req-a": 1})
    Scheduler.update_from_output.__wrapped__ = (  # type: ignore[attr-defined]
        lambda self, s_out, m_out: {0: EngineCoreOutputs(outputs=[])}
    )
    # Invoke the wrapper directly:
    with patch.object(
        tracing_utils, "_poll_xla_cache_compile_secs", return_value=8.718
    ):
        dummy_sched._tpu_prev_step_compiled = True
        req_a._tpu_compile_s = 8.718

    # Now req_b is added via Scheduler.add_request right after the compiling step
    added = []
    with patch.object(
        Scheduler.add_request, "__wrapped__", lambda self, r: added.append(r)
    ):
        if getattr(dummy_sched, "_tpu_prev_step_compiled", False):
            req_b._tpu_blocked_by_prev_compile = True
    assert req_b._tpu_blocked_by_prev_compile is True

    # Poll cache directly to verify _poll_xla_cache_compile_secs returns 8.718s once
    assert _poll_xla_cache_compile_secs() == pytest.approx(8.718, abs=1e-6)
    assert _poll_xla_cache_compile_secs() == pytest.approx(0.0, abs=1e-6)

    # 3. Verify OutputProcessor.do_tracing decodes the headers onto the llm_request span
    req_state = RequestState(
        request_id="req-b",
        external_req_id="cmpl-bench-2p2d-hc15s-c4-128in-1793941557-0-0",
        parent_req=None,
        request_index=0,
        lora_request=None,
        output_kind=RequestOutputKind.FINAL_ONLY,
        prompt="hello",
        prompt_token_ids=[1, 2, 3, 4],
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
    req_state.stats.queued_ts = 508.601
    req_state.stats.scheduled_ts = 508.6655
    req_state.stats.first_token_ts = 508.7073
    req_state.stats.last_token_ts = 509.1444
    req_state.stats.first_token_latency = 8.7073
    req_state.stats.num_generation_tokens = 64

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        op_mod, "instrument_manual", lambda **kwargs: captured.update(kwargs)
    )

    processor = OutputProcessor(tokenizer=None, log_stats=True, tracing_enabled=True)
    engine_core_output = EngineCoreOutput(
        request_id="req-b",
        new_token_ids=[42],
        trace_headers={
            "traceparent": "00-90d86cb618c6615e4dd701e734a854a7-0123456789abcdef-01",
            "_x_vllm_kv_xfer_s": f"{req_b._tpu_kv_xfer_s:.6f}",
            "_x_vllm_prev_step_compiled": "1"
            if req_b._tpu_blocked_by_prev_compile
            else "0",
            "_x_vllm_compile_s": "0.000000",
        },
    )
    iteration_stats = SimpleNamespace(iteration_timestamp=1009.1525)
    processor.do_tracing(engine_core_output, req_state, iteration_stats)  # type: ignore[arg-type]

    attrs = captured["attributes"]
    assert isinstance(attrs, dict)
    assert attrs[SPAN_ATTR_TIME_IN_KV_TRANSFER] == pytest.approx(0.0645, abs=1e-6)
    assert attrs[SPAN_ATTR_BLOCKED_BY_PREV_COMPILE] is True
    assert attrs[SPAN_ATTR_COMPILE_TIME] == pytest.approx(0.0, abs=1e-6)
