# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Unit tests for Stage 3: flag-gated vLLM lifecycle hooks."""

from types import SimpleNamespace
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.telemetry.vllm_hooks import (
    AGENT_TRACE_HEADERS,
    AgenticTelemetryService,
    extract_agent_headers,
    install_agentic_telemetry_patches,
    patch_block_pool,
    patch_output_processor,
    patch_trace_headers,
)


def test_flag_off_is_strict_noop(monkeypatch) -> None:
    """When `VLLM_TPU_AGENTIC_TELEMETRY=False`, `install_agentic_telemetry_patches` does nothing."""
    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", False)
    called = []
    monkeypatch.setattr(
        "vllm_torchtpu.telemetry.vllm_hooks.patch_trace_headers",
        lambda *a, **kw: called.append("headers"),
    )
    install_agentic_telemetry_patches()
    assert called == []


def test_extract_agent_headers_and_trace_header_patch() -> None:
    """Agent headers are extracted case-insensitively and added idempotently."""
    fake_tracing_mod = SimpleNamespace(TRACE_HEADERS=["traceparent", "tracestate"])
    patch_trace_headers(fake_tracing_mod)
    patch_trace_headers(fake_tracing_mod)
    for hdr in AGENT_TRACE_HEADERS:
        assert fake_tracing_mod.TRACE_HEADERS.count(hdr) == 1

    sid, parent_id, turn_id = extract_agent_headers(
        {
            "X-Agent-Session-Id": "sess-42",
            "X-Agent-Parent-Session-Id": "sess-root",
            "X-Agent-Turn-Id": "3",
        }
    )
    assert (sid, parent_id, turn_id) == ("sess-42", "sess-root", 3)

    assert extract_agent_headers({"x-agent-turn-id": "not-an-int"}) == (
        None,
        None,
        None,
    )
    assert extract_agent_headers(None) == (None, None, None)


def test_block_pool_eviction_hook_records_timestamps() -> None:
    """Wrapped `BlockPool._maybe_evict_cached_block` records evicted block hashes."""

    class FakeBlockPool:
        def _maybe_evict_cached_block(self, block: Any) -> bool:
            if block.block_hash is None:
                return False
            block.block_hash = None
            return True

    svc = AgenticTelemetryService(block_size=4, replica_id="tpu-node-0")
    patch_block_pool(FakeBlockPool, service=svc)
    patch_block_pool(FakeBlockPool, service=svc)  # idempotent

    pool = FakeBlockPool()
    blk_cached = SimpleNamespace(block_hash=123456)
    blk_empty = SimpleNamespace(block_hash=None)

    assert pool._maybe_evict_cached_block(blk_empty) is False
    assert len(svc.eviction_tracker) == 0

    assert pool._maybe_evict_cached_block(blk_cached) is True
    assert len(svc.eviction_tracker) == 1
    age_ms = svc.eviction_tracker.time_since_eviction_ms([123456])
    assert age_ms is not None and age_ms >= 0.0


def test_output_processor_hooks_enrich_spans_and_counters() -> None:
    """Wrapped `OutputProcessor` classifies prefill tokens and enriches `llm_request` spans."""
    captured_spans: list[dict[str, Any]] = []

    class FakeOutputProcessor:
        def __init__(self, tracing_enabled: bool = True) -> None:
            self.request_states: dict[str, Any] = {}
            self.tracing_enabled = tracing_enabled

        def process_outputs(
            self,
            engine_core_outputs: list[Any],
            engine_core_timestamp: float | None = None,
            iteration_stats: Any = None,
        ) -> list[str]:
            for out in engine_core_outputs:
                st = self.request_states.get(out.request_id)
                if st is None:
                    continue
                st.num_cached_tokens = out.num_cached_tokens
                st.is_prefilling = False
                if out.finish_reason is not None and self.tracing_enabled:
                    self.do_tracing(out, st, iteration_stats)
            return [o.request_id for o in engine_core_outputs]

        def do_tracing(
            self, engine_core_output: Any, req_state: Any, iteration_stats: Any
        ) -> None:
            fake_mod.instrument_manual(
                span_name="llm_request",
                start_time=1000,
                attributes={"gen_ai.request.id": req_state.request_id},
                context=None,
                kind="SERVER",
            )

    def fake_instrument_manual(
        span_name: str,
        start_time: int,
        attributes: dict[str, Any] | None = None,
        context: Any = None,
        kind: Any = None,
    ) -> None:
        captured_spans.append(
            {"span_name": span_name, "attributes": dict(attributes or {})}
        )

    fake_mod = SimpleNamespace(
        OutputProcessor=FakeOutputProcessor,
        instrument_manual=fake_instrument_manual,
    )

    svc = AgenticTelemetryService(block_size=4, replica_id="tpu-v6e-0")
    patch_output_processor(FakeOutputProcessor, fake_mod, service=svc)
    patch_output_processor(FakeOutputProcessor, fake_mod, service=svc)  # idempotent

    proc = FakeOutputProcessor(tracing_enabled=True)
    req_state = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=[10, 11, 12, 13, 14, 15, 16, 17],
        num_cached_tokens=0,
        is_prefilling=True,
        detokenizer=SimpleNamespace(output_token_ids=[90, 91]),
    )
    proc.request_states["req-1"] = req_state
    out = SimpleNamespace(
        request_id="req-1",
        num_cached_tokens=4,
        new_token_ids=[90, 91],
        finish_reason="stop",
        trace_headers={"x-agent-session-id": "sess-1", "x-agent-turn-id": "1"},
        kv_transfer_params={"dram_tokens": 4, "tpu_shape_padding_tokens": 8},
        num_external_computed_tokens=4,
    )

    proc.process_outputs([out])

    assert len(captured_spans) == 1
    attrs = captured_spans[0]["attributes"]
    assert attrs["gen_ai.request.id"] == "req-1"
    assert attrs["gen_ai.session.id"] == "sess-1"
    assert attrs["gen_ai.session.turn_id"] == 1
    assert attrs["gen_ai.server.replica_id"] == "tpu-v6e-0"
    assert attrs["gen_ai.usage.input_tokens.cache_hit"] == 4
    assert attrs["gen_ai.usage.input_tokens.new_input"] == 4
    assert attrs["vllm.kv_cache.dram_tokens"] == 4
    assert attrs["vllm.tpu.shape_padding_tokens"] == 8
    assert svc.category_totals["cache_hit"] == 4
    assert svc.category_totals["new_input"] == 4


def test_vllm_v1_prefill_stats_weight_version_and_contextvar_headers(tmp_path) -> None:
    """Handles vLLM V1 `prefill_stats`, `weight_version`, state pop, and ContextVar headers."""
    import json

    fake_tracing_mod = SimpleNamespace(
        TRACE_HEADERS=["traceparent", "tracestate"],
        contains_trace_headers=lambda h: False,
        extract_trace_headers=lambda h: {},
    )
    patch_trace_headers(fake_tracing_mod)

    raw_http_headers = {"X-Agent-Session-Id": "sess-v1", "X-Agent-Turn-Id": "1"}
    assert fake_tracing_mod.contains_trace_headers(raw_http_headers) is True
    extracted = fake_tracing_mod.extract_trace_headers(raw_http_headers)
    assert extracted["x-agent-session-id"] == "sess-v1"

    class FakeV1OutputProcessor:
        def __init__(self) -> None:
            self.request_states: dict[str, Any] = {}
            self.tracing_enabled = False

        def add_request(self, request: Any, prompt: str | None = None) -> None:
            self.request_states[request.request_id] = SimpleNamespace(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                num_cached_tokens=0,
                is_prefilling=True,
                detokenizer=SimpleNamespace(output_token_ids=[99]),
            )

        def process_outputs(
            self,
            engine_core_outputs: list[Any],
            engine_core_timestamp: float | None = None,
            iteration_stats: Any = None,
            weight_version: str | None = None,
        ) -> list[str]:
            for out in engine_core_outputs:
                st = self.request_states.get(out.request_id)
                if st is None:
                    continue
                if st.is_prefilling and out.prefill_stats is not None:
                    st.num_cached_tokens = out.prefill_stats.num_cached_tokens
                    st.is_prefilling = False
                if out.finish_reason is not None:
                    self.request_states.pop(out.request_id, None)
            return [o.request_id for o in engine_core_outputs]

        def do_tracing(
            self, engine_core_output: Any, req_state: Any, iteration_stats: Any
        ) -> None:
            pass

    span_log = tmp_path / "spans.jsonl"
    svc = AgenticTelemetryService(
        block_size=4,
        replica_id="tpu-v6e-live",
        span_log_path=str(span_log),
    )
    fake_mod = SimpleNamespace(
        OutputProcessor=FakeV1OutputProcessor,
        instrument_manual=lambda **kw: None,
    )
    patch_output_processor(FakeV1OutputProcessor, fake_mod, service=svc)

    proc = FakeV1OutputProcessor()
    req = SimpleNamespace(
        request_id="req-v1",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        trace_headers=None,  # Simulates AsyncLLM clearing trace_headers when OTel disabled
    )
    proc.add_request(req)
    assert req.trace_headers == extracted

    out = SimpleNamespace(
        request_id="req-v1",
        prefill_stats=SimpleNamespace(
            num_cached_tokens=4,
            num_external_cached_tokens=4,
        ),
        new_token_ids=[99],
        finish_reason="stop",
        trace_headers=None,
    )
    proc.process_outputs([out], weight_version="v1")

    lines = span_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["gen_ai.session.id"] == "sess-v1"
    assert logged["gen_ai.session.turn_id"] == 1
    assert logged["gen_ai.usage.input_tokens.cache_hit"] == 4
    assert logged["gen_ai.usage.input_tokens.new_input"] == 4
    assert logged["vllm.kv_cache.dram_tokens"] == 4
    assert logged["vllm.tpu.shape_padding_tokens"] == 12


def test_golden_fixtures_match_schema_contract() -> None:
    """All golden spans satisfy 7-category token conservation and schema keys."""
    import json
    from pathlib import Path

    from vllm_torchtpu.telemetry import CATEGORIES, SPAN_ATTRIBUTE_KEYS

    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "fixtures"
        / "cuj2_golden_spans.jsonl"
    )
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 5
    for row in rows:
        total = sum(row[f"gen_ai.usage.input_tokens.{cat}"] for cat in CATEGORIES)
        assert total == row["gen_ai.usage.prompt_tokens"]
        for k in row:
            assert k in SPAN_ATTRIBUTE_KEYS or k.startswith("gen_ai.")
    assert rows[1]["gen_ai.usage.input_tokens.unreused_reply_loss"] == 37
