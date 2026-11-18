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
"""TPU tests for distributed OpenTelemetry tracing in vLLM-Omni."""

import pytest
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

_W3C_TRACE_ID = "12345678901234567890123456789012"
_INBOUND_HEADERS = {
    "traceparent": f"00-{_W3C_TRACE_ID}-1234567890123456-01",
    "tracestate": "vendor=state",
    "baggage": "tenant=gold",
}


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="vllm_omni.tracing is not in vllm-omni 0.32.0rc1",
)
def test_distributed_opentelemetry_tracing() -> None:
    """Distributed OpenTelemetry Tracing

    - TPU path: Imports ``vllm_omni.tracing`` with ``OmniTpuPlatform`` active and
      records ``RequestTrace`` and ``SessionTrace`` spans.
    - Setup check: Verifies ``OmniTpuPlatform`` is active,
      ``create_tracer_provider(None)`` returns ``None``, and an endpoint
      returns a ``TracerProvider``.
    - Outcome check: Propagates W3C headers via ``capture_trace_headers`` and verifies
      exported ``omni.pipeline``/``omni.stage`` and ``omni.session``/``omni.turn``
      span hierarchy and ``trace_id``.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from vllm_omni.tracing import (
        RequestTrace,
        SessionTrace,
        capture_trace_headers,
        create_tracer_provider,
    )

    assert create_tracer_provider(None) is None, (
        "create_tracer_provider(None) must return None"
    )
    otlp_provider = create_tracer_provider("http://127.0.0.1:4317")
    assert isinstance(otlp_provider, TracerProvider), (
        f"expected TracerProvider, got {type(otlp_provider).__name__}"
    )
    otlp_provider.shutdown()

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    try:
        carrier = capture_trace_headers(_INBOUND_HEADERS)
        assert carrier["tracestate"] == _INBOUND_HEADERS["tracestate"], (
            f"tracestate {carrier.get('tracestate')!r} != "
            f"{_INBOUND_HEADERS['tracestate']!r}"
        )
        assert carrier["baggage"] == _INBOUND_HEADERS["baggage"], (
            f"baggage {carrier.get('baggage')!r} != {_INBOUND_HEADERS['baggage']!r}"
        )

        req_trace = RequestTrace(tracer, "req-0", carrier)
        req_trace.start_stage(0, 0, "diffusion")
        req_trace.output(0)
        req_trace.end()

        sess_trace = SessionTrace(tracer, "sess-0", carrier)
        sess_trace.start_turn("resp-0")
        sess_trace.finish_turn("resp-0", "completed")
        sess_trace.end("completed")
    finally:
        provider.shutdown()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    expected_trace_id = int(_W3C_TRACE_ID, 16)

    pipeline_span = spans["omni.pipeline"]
    stage_span = spans["omni.stage"]
    assert pipeline_span.context.trace_id == expected_trace_id, (
        f"pipeline trace_id {pipeline_span.context.trace_id:#x} != "
        f"{expected_trace_id:#x}"
    )
    assert stage_span.parent.span_id == pipeline_span.context.span_id, (
        "omni.stage parent span_id does not match omni.pipeline"
    )
    stage_events = [event.name for event in stage_span.events]
    assert "omni.stage.first_output" in stage_events, (
        f"missing omni.stage.first_output event in {stage_events}"
    )

    session_span = spans["omni.session"]
    turn_span = spans["omni.turn"]
    assert session_span.context.trace_id == expected_trace_id, (
        f"session trace_id {session_span.context.trace_id:#x} != {expected_trace_id:#x}"
    )
    assert turn_span.parent.span_id == session_span.context.span_id, (
        "omni.turn parent span_id does not match omni.session"
    )
