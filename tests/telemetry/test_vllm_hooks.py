# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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
"""Unit tests for flag-gated vLLM span and header lifecycle hooks."""

import importlib.util
import json
import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.telemetry import CATEGORIES, SPAN_ATTRIBUTE_KEYS
from vllm_torchtpu.telemetry.vllm_hooks import (
    AGENT_TRACE_HEADERS,
    AgenticTelemetryService,
    extract_agent_headers,
    install_agentic_telemetry_patches,
    patch_output_processor,
    patch_trace_headers,
    reset_telemetry_service,
    sync_block_size_from_config,
)


@pytest.fixture(autouse=True)
def _ensure_vllm_logger_for_standalone_cpu(monkeypatch) -> None:
    """Provide a `vllm.logger` stand-in only when `vllm` is absent on CPU hosts."""
    if importlib.util.find_spec("vllm") is not None:
        return

    class _StandInLogger(logging.Logger):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self._seen_once: set[str] = set()

        def warning_once(self, msg: str, *args: Any, **kwargs: Any) -> None:
            if msg not in self._seen_once:
                self._seen_once.add(msg)
                self.warning(msg, *args, **kwargs)

    vllm_pkg = ModuleType("vllm")
    vllm_logger_mod: Any = ModuleType("vllm.logger")
    vllm_logger_mod.init_logger = lambda name: _StandInLogger(name)
    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger_mod)


def test_env_flag_defaults_off(monkeypatch) -> None:
    """Telemetry environment variables default to off/empty."""
    monkeypatch.delenv("VLLM_TPU_AGENTIC_TELEMETRY", raising=False)
    assert envs.environment_variables["VLLM_TPU_AGENTIC_TELEMETRY"]() is False

    monkeypatch.setenv("VLLM_TPU_AGENTIC_TELEMETRY", "1")
    assert envs.environment_variables["VLLM_TPU_AGENTIC_TELEMETRY"]() is True


def test_flag_off_is_strict_noop(monkeypatch) -> None:
    """When telemetry flag is False, `install_agentic_telemetry_patches` is a no-op."""
    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", False)
    called: list[str] = []
    monkeypatch.setattr(
        "vllm_torchtpu.telemetry.vllm_hooks.patch_trace_headers",
        lambda *a, **kw: called.append("headers"),
    )
    install_agentic_telemetry_patches()
    assert called == []


def test_extract_agent_headers_and_trace_header_patch() -> None:
    """Agent headers delegate to upstream helpers and extract case-insensitively."""
    calls: list[str] = []
    trace_headers = ["traceparent", "tracestate"]
    fake_tracing_mod = SimpleNamespace(
        TRACE_HEADERS=trace_headers,
        contains_trace_headers=lambda h: (
            calls.append("contains"),
            any(k in h for k in trace_headers),
        )[1],
        extract_trace_headers=lambda h: (
            calls.append("extract"),
            {k: h[k] for k in trace_headers if k in h},
        )[1],
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
    assert calls == ["contains", "extract"]
    assert extract_agent_headers(extracted) == ("sess-42", "sess-root", 3)

    assert extract_agent_headers({"x-agent-turn-id": "not-an-int"}) == (
        None,
        None,
        None,
    )
    assert extract_agent_headers(None) == (None, None, None)


def test_output_processor_hooks_enrich_spans_and_counters() -> None:
    """Wrapped `OutputProcessor` classifies prefill and enriches `llm_request`."""
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
                if out.prefill_stats is not None:
                    st.num_cached_tokens = out.prefill_stats.num_cached_tokens
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
    patch_output_processor(FakeOutputProcessor, fake_mod, service=svc)

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
        prefill_stats=SimpleNamespace(
            num_cached_tokens=4,
            num_external_cached_tokens=4,
        ),
        new_token_ids=[90, 91],
        finish_reason="stop",
        trace_headers={"x-agent-session-id": "sess-1", "x-agent-turn-id": "1"},
        kv_transfer_params={"dram_tokens": 4, "tpu_shape_padding_tokens": 8},
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
    for key in attrs:
        assert key in SPAN_ATTRIBUTE_KEYS or key.startswith("gen_ai.")
    assert sum(attrs[f"gen_ai.usage.input_tokens.{c}"] for c in CATEGORIES) == 8


def test_vllm_v1_prefill_stats_weight_version_and_contextvar_headers(tmp_path) -> None:
    """Handles vLLM V1 `prefill_stats`, `n > 1` children, and ContextVar isolation."""
    trace_headers = ["traceparent", "tracestate"]
    fake_tracing_mod = SimpleNamespace(
        TRACE_HEADERS=trace_headers,
        contains_trace_headers=lambda h: any(k in h for k in trace_headers),
        extract_trace_headers=lambda h: {k: h[k] for k in trace_headers if k in h},
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

        def add_request(
            self,
            request: Any,
            prompt: str | None = None,
            parent_req: Any = None,
            request_index: int = 0,
        ) -> None:
            self.request_states[request.request_id] = SimpleNamespace(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                num_cached_tokens=0,
                is_prefilling=True,
                detokenizer=SimpleNamespace(output_token_ids=[]),
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
        trace_headers=None,
    )
    proc.add_request(req)
    assert req.trace_headers == extracted

    # Empty detok.output_token_ids falls back cleanly to out.new_token_ids.
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
    assert logged["gen_ai.usage.completion_tokens"] == 1
    assert logged["gen_ai.usage.input_tokens.cache_hit"] == 4
    assert logged["gen_ai.usage.input_tokens.new_input"] == 4
    assert logged["vllm.kv_cache.dram_tokens"] == 4
    assert "vllm.tpu.shape_padding_tokens" not in logged

    # Parallel sampling (n=2) propagates ContextVar headers to both child requests,
    # then clears before a subsequent stateless request.
    fake_tracing_mod.extract_trace_headers({"x-agent-session-id": "sess-n2"})
    parent = SimpleNamespace(n=2)
    c0 = SimpleNamespace(request_id="c0", prompt_token_ids=[1, 2], trace_headers=None)
    c1 = SimpleNamespace(request_id="c1", prompt_token_ids=[1, 2], trace_headers=None)
    proc.add_request(c0, None, parent, 0)
    proc.add_request(c1, None, parent, 1)
    assert c0.trace_headers == {"x-agent-session-id": "sess-n2"}
    assert c1.trace_headers == {"x-agent-session-id": "sess-n2"}

    req_stateless = SimpleNamespace(
        request_id="req-v1-stateless",
        prompt_token_ids=[10, 11, 12, 13],
        trace_headers=None,
    )
    proc.add_request(req_stateless)
    assert req_stateless.trace_headers is None


def test_sync_block_size_from_config_direct_hook(monkeypatch) -> None:
    """`TpuPlatform` calls `sync_block_size_from_config` directly without patching."""
    import ast
    from pathlib import Path

    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_BLOCK_SIZE", 0)
    monkeypatch.setattr(hooks_mod, "_DETECTED_VLLM_BLOCK_SIZE", None)

    stub_cfg = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))

    # 1. Strict no-op when `VLLM_TPU_AGENTIC_TELEMETRY` is disabled.
    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", False)
    svc_before = reset_telemetry_service(AgenticTelemetryService(replica_id="rep-bs"))
    assert svc_before.session_tracker.block_size == 16
    sync_block_size_from_config(stub_cfg)
    assert svc_before.session_tracker.block_size == 16
    assert hooks_mod._DETECTED_VLLM_BLOCK_SIZE is None

    # 2. Updates both active and subsequently created services when enabled.
    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    sync_block_size_from_config(stub_cfg)
    assert svc_before.session_tracker.block_size == 64

    svc_after = AgenticTelemetryService(replica_id="rep-bs")
    assert svc_after.session_tracker.block_size == 64

    # 3. Verify `TpuPlatform.update_block_size_for_backend` invokes it directly.
    platform_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "vllm_torchtpu"
        / "platforms"
        / "tpu_platform.py"
    )
    tree = ast.parse(platform_path.read_text(encoding="utf-8"))
    direct_calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "sync_block_size_from_config" in direct_calls
    reset_telemetry_service(None)


def test_compile_env_ignored_and_fail_open_safety(monkeypatch) -> None:
    """Telemetry env vars stay out of compile keys and hooks fail open cleanly."""
    import ast
    from pathlib import Path

    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    try:
        from vllm_torchtpu.compilation.tpu_compiler import (
            _TPU_COMPILE_ENV_IGNORED as ignored_vars,
        )
    except ModuleNotFoundError:
        compiler_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "vllm_torchtpu"
            / "compilation"
            / "tpu_compiler.py"
        )
        tree = ast.parse(compiler_path.read_text(encoding="utf-8"))
        ignored_vars = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "_TPU_COMPILE_ENV_IGNORED"
                        and isinstance(node.value, ast.Set)
                    ):
                        ignored_vars = {
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)
                        }
    assert {
        "VLLM_TPU_AGENTIC_TELEMETRY",
        "VLLM_TPU_AGENTIC_SPAN_LOG",
        "VLLM_TPU_REPLICA_ID",
        "VLLM_TPU_AGENTIC_BLOCK_SIZE",
    }.issubset(ignored_vars)

    # Refresh updates isolated consumer module bindings without touching real modules.
    fake_target_name = "_fake_vllm_serving_consumer"
    monkeypatch.setattr(hooks_mod, "_TRACING_CONSUMER_MODULES", (fake_target_name,))
    trace_headers = ["traceparent"]
    fake_tracing_mod = SimpleNamespace(
        TRACE_HEADERS=trace_headers,
        contains_trace_headers=lambda h: any(k in h for k in trace_headers),
        extract_trace_headers=lambda h: {k: h[k] for k in trace_headers if k in h},
    )
    patch_trace_headers(fake_tracing_mod, refresh_consumers=True)
    late_consumer = SimpleNamespace(
        contains_trace_headers=lambda h: False,
        extract_trace_headers=lambda h: {},
    )
    monkeypatch.setitem(sys.modules, fake_target_name, late_consumer)
    patch_trace_headers(fake_tracing_mod, refresh_consumers=True)
    assert (
        late_consumer.contains_trace_headers({"X-Agent-Session-Id": "late-sess"})
        is True
    )

    class FakeFailOpenProcessor:
        def __init__(self) -> None:
            self.request_states: dict[str, Any] = {
                "req-ok": SimpleNamespace(
                    request_id="req-ok",
                    prompt_token_ids=[1, 2, 3, 4],
                    is_prefilling=True,
                ),
                "req-abort": SimpleNamespace(
                    request_id="req-abort",
                    prompt_token_ids=[5, 6, 7, 8],
                    is_prefilling=True,
                ),
            }
            self.tracing_enabled = False

        def abort_requests(
            self, request_ids: list[str], internal: bool = False
        ) -> list[str]:
            for rid in request_ids:
                self.request_states.pop(rid, None)
            return list(request_ids)

        def process_outputs(self, engine_core_outputs: list[Any]) -> list[str]:
            return [o.request_id for o in engine_core_outputs]

        def do_tracing(self, *args: Any) -> None:
            pass

    svc = AgenticTelemetryService(block_size=4, replica_id="tpu-fail-open")
    fake_mod = SimpleNamespace(
        OutputProcessor=FakeFailOpenProcessor,
        instrument_manual=lambda **kw: None,
    )
    patch_output_processor(FakeFailOpenProcessor, fake_mod, service=svc)
    proc = FakeFailOpenProcessor()

    svc.on_prefill_output(
        proc.request_states["req-abort"],
        SimpleNamespace(
            request_id="req-abort",
            prefill_stats=SimpleNamespace(
                num_cached_tokens=0, num_external_cached_tokens=0
            ),
        ),
    )
    assert svc._inflight.peek("req-abort") is not None
    assert proc.abort_requests(["req-abort"]) == ["req-abort"]
    assert svc._inflight.peek("req-abort") is None

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated telemetry failure")

    monkeypatch.setattr(svc, "on_prefill_output", _boom)
    out = SimpleNamespace(request_id="req-ok", finish_reason="stop")
    assert proc.process_outputs([out]) == ["req-ok"]


def test_block_pool_eviction_hook_records_timestamps(monkeypatch, tmp_path) -> None:
    """Wrapped `BlockPool._maybe_evict_cached_block` records evicted block hashes."""
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod
    from vllm_torchtpu.telemetry.vllm_hooks import patch_block_pool

    monkeypatch.setattr(hooks_mod, "_EVICTION_SYNC_PATH", str(tmp_path / ".ev_single"))

    class FakeBlockPool:
        def _maybe_evict_cached_block(self, block: Any) -> bool:
            if block.block_hash is None:
                return False
            block.block_hash = None
            return True

    svc = AgenticTelemetryService(block_size=4, replica_id="tpu-node-0")
    patch_block_pool(FakeBlockPool, service=svc)
    patch_block_pool(FakeBlockPool, service=svc)

    pool = FakeBlockPool()
    blk_cached = SimpleNamespace(block_hash=123456)
    blk_empty = SimpleNamespace(block_hash=None)

    assert pool._maybe_evict_cached_block(blk_empty) is False
    assert len(svc.eviction_tracker) == 0

    assert pool._maybe_evict_cached_block(blk_cached) is True
    assert len(svc.eviction_tracker) == 1
    age_ms = svc.eviction_tracker.time_since_eviction_ms([123456])
    assert age_ms is not None and age_ms >= 0.0


def test_batch_eviction_allocate_slots_and_evict_blocks(monkeypatch, tmp_path) -> None:
    """Verify `O(new_blocks)` caching, `allocate_slots`, and `evict_blocks`."""
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod
    from vllm_torchtpu.telemetry import patch_block_pool

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    monkeypatch.setattr(hooks_mod, "_EVICTION_SYNC_PATH", str(tmp_path / ".evict_b"))
    svc = AgenticTelemetryService(block_size=4, replica_id="test-rep-batch", dp_rank=0)

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
                _SlottedBlock(2, 333),
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

        def evict_blocks(self, block_ids: set[int]) -> None:
            for bid in block_ids:
                self._maybe_evict_cached_block(self.blocks[bid])

    class FakeKVCacheManager:
        def __init__(self, block_pool: Any) -> None:
            self.block_pool = block_pool

        def allocate_slots(
            self,
            request: Any,
            num_new_tokens: int,
            num_new_computed_tokens: int = 0,
            new_computed_blocks: Any = None,
        ) -> Any:
            return self.block_pool.get_new_blocks(1)

    patch_block_pool(
        FakeBatchBlockPool, service=svc, kv_cache_manager_cls=FakeKVCacheManager
    )
    pool = FakeBatchBlockPool()
    mgr = FakeKVCacheManager(pool)

    # Tag block 1 for sess-own and block 2 for sess-other (block 0 masked out).
    req_own = SimpleNamespace(
        trace_headers={"x-agent-session-id": "sess-own"},
        all_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    pool.cache_full_blocks(req_own, pool.blocks, 0, 2, 4, 0, block_mask=[False, True])
    assert 0 not in pool._agentic_block_session
    assert pool._agentic_block_session[1] == "sess-own"

    # Self-eviction during sess-own's own allocate_slots records session_id=None,
    # while tagging matched prefix-hit block 2 onto sess-own.
    pool._agentic_block_session[0] = "sess-own"
    hit_blocks = SimpleNamespace(blocks=([pool.blocks[2]],))
    mgr.allocate_slots(req_own, 4, 4, hit_blocks)
    assert svc.eviction_tracker.session_evicted_at_ms("sess-own") is None
    assert pool._agentic_block_session[2] == "sess-own"

    # Explicit evict_blocks flushes once and attributes block 2 eviction to sess-own.
    flush_calls: list[int] = []
    orig_flush = svc.flush_eviction_sync
    monkeypatch.setattr(
        svc,
        "flush_eviction_sync",
        lambda: (flush_calls.append(1), orig_flush()),
    )
    pool.evict_blocks({1, 2})
    assert len(flush_calls) == 1
    assert svc.eviction_tracker.session_evicted_at_ms("sess-own") is not None


def test_delta_eviction_sync_compaction_and_multi_dp(monkeypatch, tmp_path) -> None:
    """Multi-DP delta sync merges per-DP files, compacts, and omits lru_horizon_ms."""
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    sync_base = str(tmp_path / ".evict_delta")
    monkeypatch.setattr(hooks_mod, "_EVICTION_SYNC_PATH", sync_base)

    engine_dp0 = AgenticTelemetryService(
        block_size=4, replica_id="rep-delta", dp_rank=0
    )
    engine_dp1 = AgenticTelemetryService(
        block_size=4, replica_id="rep-delta", dp_rank=1
    )
    api_svc = AgenticTelemetryService(block_size=4, replica_id="rep-delta")
    sync_file_dp0 = tmp_path / ".evict_delta_rep-delta_dp0"
    sync_file_dp1 = tmp_path / ".evict_delta_rep-delta_dp1"

    engine_dp0.on_block_evicted("blk-a1", now=10.0, session_id="sess-A")
    engine_dp1.on_block_evicted("blk-d1", now=15.0, session_id="sess-DP1")
    assert sync_file_dp0.exists() and sync_file_dp1.exists()

    api_svc._maybe_sync_eviction_state()
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-A") is not None
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-DP1") is not None
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-B") is None

    engine_dp0.on_block_evicted("blk-b1", now=20.0, session_id="sess-B")
    lines_2 = sync_file_dp0.read_text(encoding="utf-8").splitlines()
    assert len(lines_2) == 2
    assert list(json.loads(lines_2[1])["sessions"].keys()) == ["sess-B"]

    api_svc._maybe_sync_eviction_state()
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-B") is not None

    steady_cursor = api_svc._eviction_sync_cursors[str(sync_file_dp0)]
    api_svc._maybe_sync_eviction_state()
    assert api_svc._eviction_sync_cursors[str(sync_file_dp0)] == steady_cursor

    monkeypatch.setattr(hooks_mod, "_MAX_EVICTION_SYNC_BYTES", 64)
    engine_dp0.on_block_evicted("blk-c1", now=30.0, session_id="sess-C")
    lines_3 = sync_file_dp0.read_text(encoding="utf-8").splitlines()
    assert len(lines_3) == 1
    compacted_keys = set(json.loads(lines_3[0])["sessions"].keys())
    assert compacted_keys == {"sess-A", "sess-B", "sess-C"}

    api_svc._maybe_sync_eviction_state()
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-C") is not None

    req_t1 = SimpleNamespace(
        request_id="req-a-1",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        num_cached_tokens=0,
        is_prefilling=True,
        detokenizer=SimpleNamespace(output_token_ids=[9, 10, 11, 12]),
    )
    out_t1 = SimpleNamespace(
        request_id="req-a-1",
        prefill_stats=SimpleNamespace(
            num_cached_tokens=0, num_external_cached_tokens=0
        ),
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
        prefill_stats=SimpleNamespace(
            num_cached_tokens=4, num_external_cached_tokens=0
        ),
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


def test_session_evicted_at_ms_clears_after_warm_turn_and_dram_recall(
    monkeypatch, tmp_path
) -> None:
    """`session_evicted_at_ms` clears once re-warmed and fires on DRAM recall."""
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    sync_base = str(tmp_path / ".evict_clear")
    monkeypatch.setattr(hooks_mod, "_EVICTION_SYNC_PATH", sync_base)

    engine_svc = AgenticTelemetryService(
        block_size=4, replica_id="rep-clr-0", dp_rank=0
    )
    api_svc = AgenticTelemetryService(block_size=4, replica_id="rep-clr-0")

    def run_turn(
        req_id: str,
        prompt: list[int],
        out_toks: list[int],
        cached: int,
        turn_id: int,
        now: float,
        ext_cached: int = 0,
    ) -> dict[str, Any]:
        req = SimpleNamespace(
            request_id=req_id,
            prompt_token_ids=prompt,
            num_cached_tokens=cached,
            is_prefilling=True,
            detokenizer=SimpleNamespace(output_token_ids=out_toks),
        )
        out = SimpleNamespace(
            request_id=req_id,
            prefill_stats=SimpleNamespace(
                num_cached_tokens=cached,
                num_external_cached_tokens=ext_cached,
            ),
            kv_transfer_params={"dram_tokens": ext_cached} if ext_cached else None,
            new_token_ids=out_toks,
            finish_reason="stop",
            trace_headers={
                "x-agent-session-id": "sess-W",
                "x-agent-turn-id": str(turn_id),
            },
        )
        api_svc.on_prefill_output(req, out)
        return api_svc.build_span_attributes(req, out, now=now)

    p1 = [1, 2, 3, 4, 5, 6, 7, 8]
    o1 = [9, 10, 11, 12]
    attrs_1 = run_turn("r1", p1, o1, cached=0, turn_id=1, now=5.0)
    assert "vllm.kv_cache.session_evicted_at_ms" not in attrs_1

    engine_svc.eviction_tracker.record_eviction(
        "blk-w1", now=10.0, wall_time_ms=1700000010000, session_id="sess-W"
    )
    engine_svc._eviction_sync_dirty = True
    engine_svc.flush_eviction_sync()

    p2 = p1 + o1 + [13, 14, 15, 16]
    o2 = [17, 18, 19, 20]
    attrs_2 = run_turn("r2", p2, o2, cached=4, turn_id=2, now=12.0)
    assert attrs_2["gen_ai.usage.input_tokens.affinity_honored_eviction"] == 8
    assert attrs_2["vllm.kv_cache.session_evicted_at_ms"] == 1700000010000
    assert attrs_2["vllm.kv_cache.time_since_eviction_ms"] == 2000.0
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-W") is None

    monkeypatch.setattr(hooks_mod, "_MAX_EVICTION_SYNC_BYTES", 16)
    engine_svc.on_block_evicted("blk-other", now=15.0, session_id="sess-other")

    p3 = p2 + o2 + [21, 22, 23, 24]
    o3 = [25, 26, 27, 28]
    attrs_3 = run_turn("r3", p3, o3, cached=20, turn_id=3, now=18.0)
    assert attrs_3["gen_ai.usage.input_tokens.affinity_honored_eviction"] == 0
    assert "vllm.kv_cache.session_evicted_at_ms" not in attrs_3
    assert "vllm.kv_cache.time_since_eviction_ms" not in attrs_3

    engine_svc.eviction_tracker.record_eviction(
        "blk-w2", now=25.0, wall_time_ms=1700000025000, session_id="sess-W"
    )
    engine_svc._eviction_sync_dirty = True
    engine_svc.flush_eviction_sync()

    p4 = p3 + o3 + [29, 30, 31, 32]
    o4 = [33, 34, 35, 36]
    attrs_4 = run_turn("r4", p4, o4, cached=28, turn_id=4, now=26.5, ext_cached=20)
    assert attrs_4["gen_ai.usage.input_tokens.affinity_honored_eviction"] == 0
    assert attrs_4["vllm.kv_cache.dram_tokens"] == 20
    assert attrs_4["vllm.kv_cache.session_evicted_at_ms"] == 1700000025000
    assert attrs_4["vllm.kv_cache.time_since_eviction_ms"] == 1500.0
    assert api_svc.eviction_tracker.session_evicted_at_ms("sess-W") is None
