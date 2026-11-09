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
    assert "vllm.tpu.shape_padding_tokens" not in logged


def test_batch_eviction_sync_and_server_padding_hook(monkeypatch) -> None:
    """Verify once-per-step batch eviction sync and TPU round padding counter hook."""
    from vllm_torchtpu.telemetry.vllm_hooks import patch_tpu_padding_metrics

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    svc = AgenticTelemetryService(block_size=4, replica_id="test-rep-batch")

    class _SlottedBlock:
        __slots__ = ("block_id", "block_hash", "is_null")

        def __init__(self, block_id: int, block_hash: int) -> None:
            self.block_id = block_id
            self.block_hash: int | None = block_hash
            self.is_null = False

    class FakeBatchBlockPool:
        def __init__(self) -> None:
            self.blocks = [
                _SlottedBlock(0, 111),
                _SlottedBlock(1, 222),
            ]

        def cache_full_blocks(
            self,
            request: Any,
            blocks: list[Any],
            num_cached_blocks: int,
            num_full_blocks: int,
            block_size: int,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            pass

        def _maybe_evict_cached_block(self, block: Any) -> bool:
            if block.block_hash is None:
                return False
            block.block_hash = None
            return True

        def get_new_blocks(self, num_blocks: int) -> list[Any]:
            for b in self.blocks[:num_blocks]:
                self._maybe_evict_cached_block(b)
            return self.blocks[:num_blocks]

    patch_block_pool(FakeBatchBlockPool, service=svc)
    pool = FakeBatchBlockPool()
    req_a = SimpleNamespace(
        trace_headers={"x-agent-session-id": "sess-batch-a"},
        all_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    pool.cache_full_blocks(req_a, pool.blocks, 0, 2, 4)
    assert pool._cuj2_block_session[0] == "sess-batch-a"
    assert pool._cuj2_block_session[1] == "sess-batch-a"

    flush_calls = []
    orig_flush = svc.flush_eviction_sync
    monkeypatch.setattr(
        svc,
        "flush_eviction_sync",
        lambda: (flush_calls.append(1), orig_flush()),
    )
    pool.get_new_blocks(2)
    assert len(flush_calls) == 1
    assert svc.eviction_tracker.session_evicted_at_ms("sess-batch-a") is not None

    class FakeRunner:
        def __init__(self) -> None:
            self._last_sequence_layout_plan = SimpleNamespace(
                global_num_tokens=430,
                global_padded_num_tokens=512,
            )

        def _prepare_inputs(
            self, scheduler_output: Any, start_index: int
        ) -> tuple[Any, ...]:
            return (None, None, 1, 1)

    patch_tpu_padding_metrics(FakeRunner, service=svc)
    runner = FakeRunner()
    runner._prepare_inputs(SimpleNamespace(total_num_scheduled_tokens=430), 0)
    runner._last_sequence_layout_plan = SimpleNamespace(
        global_num_tokens=1,
        global_padded_num_tokens=16,
    )
    runner._prepare_inputs(SimpleNamespace(total_num_scheduled_tokens=1), 0)

    counts = svc.sync_padding_state()
    assert counts["prompt_rounds"] == 1
    assert counts["prompt_real_tokens"] == 430
    assert counts["prompt_filler_tokens"] == 82
    assert counts["reply_rounds"] == 1
    assert counts["reply_real_tokens"] == 1
    assert counts["reply_filler_tokens"] == 15


def test_installer_fails_loudly_on_missing_anchor(tmp_path) -> None:
    """`install_into_tree` raises `RuntimeError` if target files or anchors are missing."""
    import pytest
    from vllm_torchtpu.telemetry.install import install_into_tree

    pkg_dir = tmp_path / "vllm_torchtpu"
    pkg_dir.mkdir()
    (pkg_dir / "envs.py").write_text("environment_variables = {}\n", encoding="utf-8")
    (pkg_dir / "patch_registry.py").write_text("def foo(): pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to locate"):
        install_into_tree(pkg_dir)


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
        assert "vllm.kv_cache.lru_horizon_ms" not in row
        assert "vllm.tpu.shape_padding_tokens" not in row
        for k in row:
            assert k in SPAN_ATTRIBUTE_KEYS or k.startswith("gen_ai.")
    assert rows[1]["gen_ai.usage.input_tokens.unreused_reply_loss"] == 37


def test_delta_eviction_sync_compaction_and_lru_horizon_omitted(
    monkeypatch, tmp_path
) -> None:
    """Delta sync writes only changed sessions per step, reads incrementally, compacts, and omits lru_horizon_ms."""
    import json

    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    sync_base = str(tmp_path / ".evict_delta")
    monkeypatch.setattr(hooks_mod, "_EVICTION_SYNC_PATH", sync_base)

    engine_svc = AgenticTelemetryService(block_size=4, replica_id="rep-delta-0")
    api_svc = AgenticTelemetryService(block_size=4, replica_id="rep-delta-0")
    sync_file = tmp_path / ".evict_delta_rep-delta-0"

    # Step 1: Evict sess-A only -> line 1 contains ONLY sess-A.
    engine_svc.on_block_evicted("blk-a1", now=10.0, session_id="sess-A")
    lines_1 = sync_file.read_text(encoding="utf-8").splitlines()
    assert len(lines_1) == 1
    assert list(json.loads(lines_1[0])["sessions"].keys()) == ["sess-A"]

    api_svc._maybe_sync_eviction_state()
    offset_after_1 = api_svc._eviction_sync_offset
    assert offset_after_1 > 0
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-A") is not None
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-B") is None

    # Step 2: Evict sess-B only -> line 2 contains ONLY sess-B (not sess-A again).
    engine_svc.on_block_evicted("blk-b1", now=20.0, session_id="sess-B")
    lines_2 = sync_file.read_text(encoding="utf-8").splitlines()
    assert len(lines_2) == 2
    assert list(json.loads(lines_2[1])["sessions"].keys()) == ["sess-B"]

    api_svc._maybe_sync_eviction_state()
    assert api_svc._eviction_sync_offset > offset_after_1
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-A") is not None
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-B") is not None

    # No-op sync when file unchanged does not move offset.
    steady_offset = api_svc._eviction_sync_offset
    api_svc._maybe_sync_eviction_state()
    assert api_svc._eviction_sync_offset == steady_offset

    # Step 3: Trigger compaction when sync file exceeds threshold.
    monkeypatch.setattr(hooks_mod, "_MAX_EVICTION_SYNC_BYTES", 64)
    engine_svc.on_block_evicted("blk-c1", now=30.0, session_id="sess-C")
    lines_3 = sync_file.read_text(encoding="utf-8").splitlines()
    assert len(lines_3) == 1
    compacted_keys = set(json.loads(lines_3[0])["sessions"].keys())
    assert compacted_keys == {"sess-A", "sess-B", "sess-C"}

    # Reader detects new inode / smaller size and resets offset cleanly.
    api_svc._maybe_sync_eviction_state()
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-C") is not None

    # Verify per-turn span attributes omit lru_horizon_ms while keeping time_since_eviction_ms & session_evicted_at_ms.
    req_t1 = SimpleNamespace(
        request_id="req-a-1",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        num_cached_tokens=0,
        is_prefilling=True,
        detokenizer=SimpleNamespace(output_token_ids=[9, 10, 11, 12]),
    )
    out_t1 = SimpleNamespace(
        request_id="req-a-1",
        num_cached_tokens=0,
        new_token_ids=[9, 10, 11, 12],
        finish_reason="stop",
        trace_headers={"x-agent-session-id": "sess-A", "x-agent-turn-id": "1"},
    )
    api_svc.on_prefill_output(req_t1, out_t1)
    api_svc.build_span_attributes(req_t1, out_t1, now=8.0)

    req_t2 = SimpleNamespace(
        request_id="req-a-2",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        num_cached_tokens=4,
        is_prefilling=True,
        detokenizer=SimpleNamespace(output_token_ids=[17, 18]),
    )
    out_t2 = SimpleNamespace(
        request_id="req-a-2",
        num_cached_tokens=4,
        new_token_ids=[17, 18],
        finish_reason="stop",
        trace_headers={"x-agent-session-id": "sess-A", "x-agent-turn-id": "2"},
    )
    api_svc.on_prefill_output(req_t2, out_t2)
    attrs_t2 = api_svc.build_span_attributes(req_t2, out_t2, now=12.5)
    assert attrs_t2["gen_ai.usage.input_tokens.affinity_honored_eviction"] == 8
    assert attrs_t2["vllm.kv_cache.time_since_eviction_ms"] == 2500.0
    assert "vllm.kv_cache.session_evicted_at_ms" in attrs_t2
    assert "vllm.kv_cache.lru_horizon_ms" not in attrs_t2
