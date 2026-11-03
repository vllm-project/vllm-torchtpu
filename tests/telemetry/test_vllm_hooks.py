# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Unit tests for CUJ-2 telemetry schema contract and golden trace replay."""

from pathlib import Path
from types import SimpleNamespace

from vllm_torchtpu import envs
from vllm_torchtpu.telemetry import (
    AGENT_TRACE_HEADERS,
    CATEGORIES,
    SPAN_ATTRIBUTE_KEYS,
    extract_agent_headers,
    install_agentic_telemetry_patches,
    patch_trace_headers,
)


def test_flag_off_is_strict_noop(monkeypatch) -> None:
    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", False)
    called = []
    monkeypatch.setattr(
        "vllm_torchtpu.telemetry.vllm_hooks.patch_trace_headers",
        lambda *a, **kw: called.append("headers"),
    )
    install_agentic_telemetry_patches()
    assert called == []


def test_extract_agent_headers_and_trace_header_patch() -> None:
    fake_tracing_mod = SimpleNamespace(
        TRACE_HEADERS=["traceparent", "tracestate"],
        contains_trace_headers=lambda h: False,
        extract_trace_headers=lambda h: {},
    )
    patch_trace_headers(fake_tracing_mod)
    patch_trace_headers(fake_tracing_mod)
    for hdr in AGENT_TRACE_HEADERS:
        assert fake_tracing_mod.TRACE_HEADERS.count(hdr) == 1

    raw = {
        "X-Agent-Session-Id": "sess-42",
        "X-Agent-Parent-Session-Id": "sess-root",
        "X-Agent-Turn-Id": "3",
    }
    assert fake_tracing_mod.contains_trace_headers(raw) is True
    extracted = fake_tracing_mod.extract_trace_headers(raw)
    sid, parent_id, turn_id = extract_agent_headers(extracted)
    assert (sid, parent_id, turn_id) == ("sess-42", "sess-root", 3)


def test_golden_fixtures_match_schema_contract() -> None:
    import json

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
