# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Flag-gated vLLM lifecycle hooks for CUJ-2 agentic serving telemetry."""

import contextlib
import contextvars
import json
import logging
import os
import socket
import sys
import time
from collections.abc import Mapping
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.telemetry.eviction_tracker import (
    BlockEvictionTracker,
    TpuPaddingCounterTracker,
    compute_tpu_bucket_padding_tokens,
    extract_hardware_span_attributes,
)
from vllm_torchtpu.telemetry.token_breakdown import (
    CATEGORIES,
    SessionTurnTracker,
    TokenBreakdown,
    compute_rolling_block_hashes,
)

logger = logging.getLogger("vllm." + __name__)

AGENT_TRACE_HEADERS: tuple[str, ...] = (
    "x-agent-session-id",
    "x-agent-parent-session-id",
    "x-agent-turn-id",
    "x-agent-turn-index",
)

SPAN_ATTRIBUTE_KEYS: tuple[str, ...] = (
    "gen_ai.session.id",
    "gen_ai.session.turn_id",
    "gen_ai.session.parent_id",
    "gen_ai.server.replica_id",
    "gen_ai.usage.input_tokens.cache_hit",
    "gen_ai.usage.input_tokens.cross_session_opportunity",
    "gen_ai.usage.input_tokens.affinity_honored_eviction",
    "gen_ai.usage.input_tokens.routing_loss",
    "gen_ai.usage.input_tokens.prefix_mutation_loss",
    "gen_ai.usage.input_tokens.unreused_reply_loss",
    "gen_ai.usage.input_tokens.new_input",
    "gen_ai.usage.mutation_offset_token_idx",
    "vllm.kv_cache.time_since_eviction_ms",
    "vllm.kv_cache.lru_horizon_ms",
    "vllm.kv_cache.session_evicted_at_ms",
    "vllm.kv_cache.dram_tokens",
    "vllm.kv_cache.dram_bytes",
    "vllm.kv_cache.dram_ms",
    "vllm.kv_cache.ssd_tokens",
    "vllm.kv_cache.ssd_bytes",
    "vllm.kv_cache.ssd_ms",
    "vllm.kv_cache.external_computed_tokens",
    "vllm.kv_cache.remote_tokens",
    "vllm.kv_cache.transfer_ms",
    "vllm.tpu.shape_padding_tokens",
)

_ACTIVE_TRACE_HEADERS: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar("_ACTIVE_TRACE_HEADERS", default=None)
)


def _ipc_base_dir() -> str:
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return "/tmp"


_BLOCK_SIZE_SYNC_PATH = f"{_ipc_base_dir()}/.vllm_tpu_agentic_block_size"
_EVICTION_SYNC_PATH = f"{_ipc_base_dir()}/.vllm_tpu_agentic_eviction_state"
_PADDING_SYNC_PATH = f"{_ipc_base_dir()}/.vllm_tpu_agentic_padding_state"
_MAX_EVICTION_SYNC_BYTES = 524_288
_COMPACT_RECENT_SESSIONS = 1024


def _resolve_trace_headers(
    req_state: Any, engine_core_output: Any
) -> Mapping[str, str] | None:
    """Return trace headers from `engine_core_output` or `req_state` fallback."""
    return getattr(engine_core_output, "trace_headers", None) or getattr(
        req_state, "trace_headers", None
    )


def _resolve_num_cached_tokens(req_state: Any, engine_core_output: Any) -> int:
    """Extract `num_cached_tokens` across direct fields, `prefill_stats`, or `req_state`."""
    num_cached = getattr(engine_core_output, "num_cached_tokens", None)
    if num_cached is None:
        prefill_stats = getattr(engine_core_output, "prefill_stats", None)
        if prefill_stats is not None:
            num_cached = getattr(prefill_stats, "num_cached_tokens", None)
    if num_cached is None:
        num_cached = getattr(req_state, "num_cached_tokens", 0) or 0
    return int(num_cached)


def _resolve_num_external_cached_tokens(
    req_state: Any, engine_core_output: Any
) -> int:
    """Extract external/DRAM cached tokens from `prefill_stats`, `engine_core_output`, or `req_state`."""
    ext = getattr(engine_core_output, "num_external_computed_tokens", None)
    if ext is None:
        ext = getattr(engine_core_output, "num_external_cached_tokens", None)
    if ext is None:
        prefill_stats = getattr(engine_core_output, "prefill_stats", None)
        if prefill_stats is not None:
            ext = getattr(prefill_stats, "num_external_cached_tokens", None)
            if ext is None:
                ext = getattr(prefill_stats, "num_external_computed_tokens", None)
    if ext is None:
        ext = getattr(req_state, "_cuj2_external_cached_tokens", None)
    if ext is None:
        ext = getattr(req_state, "num_external_cached_tokens", 0) or 0
    return int(ext)


def extract_agent_headers(
    trace_headers: Mapping[str, str] | None,
) -> tuple[str | None, str | None, int | None]:
    """Extract `(session_id, parent_session_id, turn_id)` from trace headers."""
    if not trace_headers:
        return None, None, None
    lowered = {k.lower(): v for k, v in trace_headers.items() if v is not None}
    session_id = lowered.get("x-agent-session-id") or None
    parent_session_id = lowered.get("x-agent-parent-session-id") or None
    raw_turn = lowered.get("x-agent-turn-id")
    if raw_turn is None:
        raw_turn = lowered.get("x-agent-turn-index")
    turn_id: int | None = None
    if raw_turn is not None:
        try:
            turn_id = int(raw_turn)
        except (TypeError, ValueError):
            turn_id = None
    return session_id, parent_session_id, turn_id


class AgenticTelemetryService:
    """Coordinates session turn tracking, block eviction age, padding metrics, and span emission."""

    def __init__(
        self,
        *,
        block_size: int | None = None,
        replica_id: str | None = None,
        span_log_path: str | None = None,
        session_tracker: SessionTurnTracker | None = None,
        eviction_tracker: BlockEvictionTracker | None = None,
        padding_tracker: TpuPaddingCounterTracker | None = None,
    ) -> None:
        self.replica_id = (
            replica_id
            or os.environ.get("VLLM_REPLICA_ID")
            or os.environ.get("HOSTNAME")
            or socket.gethostname()
        )
        self.span_log_path = (
            span_log_path
            if span_log_path is not None
            else getattr(envs, "VLLM_TPU_AGENTIC_SPAN_LOG", "")
        )
        env_bs = os.environ.get("VLLM_TPU_AGENTIC_BLOCK_SIZE")
        self._explicit_block_size = (
            block_size is not None
            or session_tracker is not None
            or bool(env_bs)
        )
        self._block_size_checked = self._explicit_block_size
        resolved_block_size = (
            block_size
            if block_size is not None
            else int(env_bs or 16)
        )
        self.session_tracker = session_tracker or SessionTurnTracker(
            block_size=resolved_block_size
        )
        self.eviction_tracker = eviction_tracker or BlockEvictionTracker()
        self.padding_tracker = padding_tracker or TpuPaddingCounterTracker()
        self.category_totals: dict[str, int] = {cat: 0 for cat in CATEGORIES}
        self._req_breakdowns: dict[str, TokenBreakdown] = {}
        self._req_external_tokens: dict[str, int] = {}
        self._req_prefill_ts: dict[str, float] = {}
        self._req_eviction_metrics: dict[
            str, tuple[float | None, int | None]
        ] = {}
        self._req_eviction_ts: dict[str, float | None] = {}
        self._eviction_sync_dirty: bool = False
        self._eviction_writer_initialized: bool = False
        self._eviction_sync_inode: int | None = None
        self._eviction_sync_offset: int = 0

    def _safe_replica_id(self) -> str:
        return "".join(
            c if c.isalnum() or c in ("-", "_", ".") else "_"
            for c in str(self.replica_id)
        )

    def _eviction_sync_path(self) -> str:
        return f"{_EVICTION_SYNC_PATH}_{self._safe_replica_id()}"

    def _padding_sync_path(self) -> str:
        return f"{_PADDING_SYNC_PATH}_{self._safe_replica_id()}"

    def on_tpu_round_padded(
        self,
        scheduled_tokens: int,
        padded_tokens: int,
        *,
        is_prompt_round: bool,
        sync_ipc: bool = True,
    ) -> None:
        """Record one TPU round's real vs filler tokens and sync to shared memory."""
        self.padding_tracker.record_round(
            scheduled_tokens,
            padded_tokens,
            is_prompt_round=is_prompt_round,
        )
        if sync_ipc and envs.VLLM_TPU_AGENTIC_TELEMETRY:
            sync_path = self._padding_sync_path()
            tmp_path = f"{sync_path}.tmp.{os.getpid()}"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(self.padding_tracker.as_dict(), f)
                os.replace(tmp_path, sync_path)
            except OSError:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

    def sync_padding_state(self) -> dict[str, int]:
        """Sync TPU round padding counters from EngineCore shared memory."""
        sync_path = self._padding_sync_path()
        if envs.VLLM_TPU_AGENTIC_TELEMETRY and os.path.exists(sync_path):
            try:
                with open(sync_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, Mapping):
                    self.padding_tracker.update_from_dict(data)
            except (OSError, ValueError):
                pass
        return self.padding_tracker.as_dict()

    def flush_eviction_sync(self) -> None:
        """Flush changed per-session eviction timestamps once per step as a delta line."""
        if not self._eviction_sync_dirty or not envs.VLLM_TPU_AGENTIC_TELEMETRY:
            return
        self._eviction_sync_dirty = False
        changed_sessions = self.eviction_tracker.pop_dirty_session_state()
        last_pair = [
            self.eviction_tracker._last_evicted_ts,
            self.eviction_tracker._last_evicted_wall_ms,
        ]
        sync_path = self._eviction_sync_path()
        try:
            needs_replace = not self._eviction_writer_initialized
            if not needs_replace and os.path.exists(sync_path):
                needs_replace = (
                    os.path.getsize(sync_path) >= _MAX_EVICTION_SYNC_BYTES
                )
            if needs_replace:
                sessions_payload = (
                    changed_sessions
                    if not self._eviction_writer_initialized
                    else self.eviction_tracker.export_recent_session_state(
                        _COMPACT_RECENT_SESSIONS
                    )
                )
                payload = {
                    "sessions": sessions_payload,
                    "last": last_pair,
                }
                tmp_path = f"{sync_path}.tmp.{os.getpid()}"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(payload, separators=(",", ":")) + "\n")
                os.replace(tmp_path, sync_path)
                self._eviction_writer_initialized = True
            else:
                payload = {
                    "sessions": changed_sessions,
                    "last": last_pair,
                }
                line = json.dumps(payload, separators=(",", ":")) + "\n"
                with open(sync_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except OSError:
            pass

    def _apply_eviction_sync_record(self, raw_line: str) -> bool:
        """Parse and merge one eviction sync JSONL line (or legacy record)."""
        if not raw_line:
            return False
        try:
            if raw_line.startswith("{"):
                data = json.loads(raw_line)
                sessions = data.get("sessions")
                if isinstance(sessions, Mapping) and sessions:
                    self.eviction_tracker.import_session_state(sessions)
                last = data.get("last")
                if (
                    isinstance(last, list)
                    and len(last) >= 2
                    and last[0] is not None
                    and last[1] is not None
                ):
                    ts = float(last[0])
                    wall_ms = int(last[1])
                    cur_ts = self.eviction_tracker._last_evicted_ts
                    if cur_ts is None or ts > cur_ts:
                        self.eviction_tracker._last_evicted_ts = ts
                        self.eviction_tracker._last_evicted_wall_ms = wall_ms
                return True
            parts = raw_line.split()
            if len(parts) >= 2:
                ts = float(parts[0])
                wall_ms = int(parts[1])
                cur_ts = self.eviction_tracker._last_evicted_ts
                if cur_ts is None or ts > cur_ts:
                    self.eviction_tracker._last_evicted_ts = ts
                    self.eviction_tracker._last_evicted_wall_ms = wall_ms
                return True
        except ValueError:
            return False
        return False

    def _maybe_sync_eviction_state(self) -> None:
        """Incrementally read newly appended per-session eviction deltas from shared memory."""
        sync_path = self._eviction_sync_path()
        if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
            return
        try:
            st = os.stat(sync_path)
        except OSError:
            return
        if (
            self._eviction_sync_inode == st.st_ino
            and self._eviction_sync_offset == st.st_size
        ):
            return
        if (
            self._eviction_sync_inode != st.st_ino
            or st.st_size < self._eviction_sync_offset
        ):
            self._eviction_sync_inode = st.st_ino
            self._eviction_sync_offset = 0
        try:
            with open(sync_path, "rb") as f:
                f.seek(self._eviction_sync_offset)
                chunk = f.read()
            if not chunk:
                return
            last_nl = chunk.rfind(b"\n")
            if last_nl == -1:
                if self._eviction_sync_offset == 0:
                    raw_text = chunk.decode("utf-8", errors="replace").strip()
                    if self._apply_eviction_sync_record(raw_text):
                        self._eviction_sync_offset = len(chunk)
                return
            complete = chunk[: last_nl + 1]
            self._eviction_sync_offset += len(complete)
            for raw_line in complete.decode("utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if line:
                    self._apply_eviction_sync_record(line)
        except OSError:
            pass

    def _maybe_sync_vllm_block_size(self) -> None:
        """Auto-detect `cache_config.block_size` (e.g. 64/128 on TPU) once if not explicitly set."""
        if self._explicit_block_size or self._block_size_checked:
            return
        self._block_size_checked = True
        env_bs = os.environ.get("VLLM_TPU_AGENTIC_BLOCK_SIZE")
        if env_bs:
            try:
                bs = int(env_bs)
                if bs > 0:
                    self.session_tracker.block_size = bs
                    self._explicit_block_size = True
                    return
            except ValueError:
                pass
        if envs.VLLM_TPU_AGENTIC_TELEMETRY and os.path.exists(_BLOCK_SIZE_SYNC_PATH):
            try:
                with open(_BLOCK_SIZE_SYNC_PATH, encoding="utf-8") as f:
                    bs = int(f.read().strip())
                if bs > 0:
                    self.session_tracker.block_size = bs
                    self._explicit_block_size = True
                    return
            except (OSError, ValueError):
                pass
        try:
            from vllm.config.vllm import get_current_vllm_config

            cfg = get_current_vllm_config()
            bs = getattr(getattr(cfg, "cache_config", None), "block_size", None)
            if isinstance(bs, int) and bs > 0:
                self.session_tracker.block_size = bs
                self._explicit_block_size = True
        except Exception:
            pass

    def on_block_evicted(
        self,
        block_hash: Any,
        now: float | None = None,
        *,
        session_id: str | None = None,
        defer_sync: bool = False,
    ) -> None:
        """Record an HBM block eviction event (with optional deferred batch IPC flush)."""
        self.eviction_tracker.record_eviction(
            block_hash, now=now, session_id=session_id
        )
        self._eviction_sync_dirty = True
        if not defer_sync:
            self.flush_eviction_sync()

    def on_prefill_output(
        self, req_state: Any, engine_core_output: Any
    ) -> TokenBreakdown:
        """Classify prefill tokens on the first output step of a request."""
        req_id = getattr(engine_core_output, "request_id", None) or getattr(
            req_state, "request_id", ""
        )
        ext_cached = _resolve_num_external_cached_tokens(
            req_state, engine_core_output
        )
        if ext_cached > 0:
            if req_id:
                self._req_external_tokens[req_id] = ext_cached
            try:
                req_state._cuj2_external_cached_tokens = ext_cached
            except AttributeError:
                pass
        if req_id and req_id in self._req_breakdowns:
            return self._req_breakdowns[req_id]

        self._maybe_sync_vllm_block_size()
        self._maybe_sync_eviction_state()
        prefill_now = time.monotonic()
        if req_id:
            self._req_prefill_ts[req_id] = prefill_now

        session_id, parent_session_id, turn_id = extract_agent_headers(
            _resolve_trace_headers(req_state, engine_core_output)
        )
        prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
        num_cached = _resolve_num_cached_tokens(req_state, engine_core_output)

        breakdown = self.session_tracker.classify_prefill(
            prompt_token_ids,
            num_cached,
            session_id=session_id,
            turn_id=turn_id,
            parent_session_id=parent_session_id,
        )
        for cat, count in breakdown.as_category_dict().items():
            self.category_totals[cat] += count
        if req_id:
            self._req_breakdowns[req_id] = breakdown
            self._req_eviction_ts[req_id] = (
                self.eviction_tracker.session_evicted_ts(session_id)
            )
            evict_age, evict_wall = self._compute_eviction_snapshot(
                req_state,
                breakdown,
                session_id=session_id,
                eval_now=prefill_now,
                ext_cached=ext_cached,
            )
            self._req_eviction_metrics[req_id] = (evict_age, evict_wall)
        return breakdown

    def _compute_eviction_snapshot(
        self,
        req_state: Any,
        breakdown: TokenBreakdown,
        *,
        session_id: str | None,
        eval_now: float | None,
        ext_cached: int = 0,
    ) -> tuple[float | None, int | None]:
        eviction_age_ms: float | None = None
        session_evicted_wall: int | None = None
        if breakdown.affinity_honored_eviction > 0 or ext_cached > 0:
            prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
            block_size = self.session_tracker.block_size
            num_prompt_blocks = len(prompt_token_ids) // block_size
            t1_state = (
                self.session_tracker.tier1.peek(session_id)
                if session_id
                else None
            )
            if t1_state is not None and len(t1_state.block_hashes) >= num_prompt_blocks:
                all_hashes = t1_state.block_hashes[:num_prompt_blocks]
            else:
                all_hashes = compute_rolling_block_hashes(
                    prompt_token_ids, block_size
                )
            hbm_hit = max(0, breakdown.cache_hit - max(0, int(ext_cached)))
            start_blk = hbm_hit // block_size
            missed_hashes = all_hashes[start_blk:]
            eviction_age_ms = self.eviction_tracker.time_since_eviction_ms(
                missed_hashes, now=eval_now, session_id=session_id
            )
            session_evicted_wall = self.eviction_tracker.session_evicted_at_ms(
                session_id
            )

        return eviction_age_ms, session_evicted_wall

    def on_request_finished(
        self, req_state: Any, engine_core_output: Any
    ) -> TokenBreakdown:
        """Finalize request state and update two-tier session history."""
        req_id = getattr(engine_core_output, "request_id", None) or getattr(
            req_state, "request_id", ""
        )
        breakdown = self._req_breakdowns.pop(req_id, None) if req_id else None
        if breakdown is None:
            breakdown = self.on_prefill_output(req_state, engine_core_output)
            if req_id:
                self._req_breakdowns.pop(req_id, None)

        session_id, _, turn_id = extract_agent_headers(
            _resolve_trace_headers(req_state, engine_core_output)
        )
        prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
        detok = getattr(req_state, "detokenizer", None)
        output_token_ids = (
            getattr(detok, "output_token_ids", None)
            if detok is not None
            else getattr(engine_core_output, "new_token_ids", None)
        ) or ()
        full_hashes, reused_blocks = self.session_tracker.record_turn_completion(
            session_id,
            prompt_token_ids,
            output_token_ids,
            turn_id=turn_id if turn_id is not None else 1,
        )
        if session_id and full_hashes:
            self.eviction_tracker.register_session_blocks(
                session_id,
                full_hashes[reused_blocks:],
            )
        return breakdown

    def build_span_attributes(
        self,
        req_state: Any,
        engine_core_output: Any,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Build all CUJ-2 OTel span attributes for a completed request."""
        req_id = getattr(engine_core_output, "request_id", None) or getattr(
            req_state, "request_id", ""
        )
        ext_computed = (
            self._req_external_tokens.pop(req_id, 0) if req_id else 0
        )
        if not ext_computed:
            ext_computed = _resolve_num_external_cached_tokens(
                req_state, engine_core_output
            )

        snap_evict = (
            self._req_eviction_metrics.pop(req_id, None) if req_id else None
        )
        breakdown = self.on_request_finished(req_state, engine_core_output)
        if snap_evict is None and req_id:
            snap_evict = self._req_eviction_metrics.pop(req_id, None)
        prefill_evict_ts = (
            self._req_eviction_ts.pop(req_id, None) if req_id else None
        )
        prefill_ts = self._req_prefill_ts.pop(req_id, None) if req_id else None
        eval_now = now if now is not None else prefill_ts

        attrs = breakdown.to_span_attributes()
        attrs["gen_ai.server.replica_id"] = self.replica_id

        session_id, parent_session_id, turn_id = extract_agent_headers(
            _resolve_trace_headers(req_state, engine_core_output)
        )
        if session_id:
            attrs["gen_ai.session.id"] = session_id
        if turn_id is not None:
            attrs["gen_ai.session.turn_id"] = turn_id
        if parent_session_id and "gen_ai.session.parent_id" not in attrs:
            attrs["gen_ai.session.parent_id"] = parent_session_id

        if snap_evict is not None and now is None:
            eviction_age_ms, session_evicted_wall = snap_evict
        else:
            self._maybe_sync_eviction_state()
            if prefill_evict_ts is None:
                prefill_evict_ts = self.eviction_tracker.session_evicted_ts(
                    session_id
                )
            eviction_age_ms, session_evicted_wall = (
                self._compute_eviction_snapshot(
                    req_state,
                    breakdown,
                    session_id=session_id,
                    eval_now=eval_now,
                    ext_cached=ext_computed,
                )
            )

        had_eviction = (
            breakdown.affinity_honored_eviction > 0 or ext_computed > 0
        )
        if (
            session_id
            and prefill_evict_ts is not None
            and (
                had_eviction
                or breakdown.cache_hit > 0
                or (turn_id is not None and turn_id > 1)
            )
        ):
            self.eviction_tracker.clear_session_eviction(
                session_id, up_to_ts=prefill_evict_ts
            )

        hw_attrs = extract_hardware_span_attributes(
            kv_transfer_params=getattr(engine_core_output, "kv_transfer_params", None),
            num_external_computed_tokens=ext_computed,
            time_since_eviction_ms=eviction_age_ms,
            lru_horizon_ms=None,
            session_evicted_at_ms=session_evicted_wall,
            tpu_shape_padding_tokens=None,
        )
        attrs.update(hw_attrs)
        if self.span_log_path:
            log_record = dict(attrs)
            trace_hdrs = _resolve_trace_headers(req_state, engine_core_output) or {}
            lowered_hdrs = {
                k.lower(): v for k, v in trace_hdrs.items() if v is not None
            }
            if "traceparent" in lowered_hdrs:
                log_record.setdefault("traceparent", lowered_hdrs["traceparent"])
            prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
            detok = getattr(req_state, "detokenizer", None)
            output_token_ids = (
                getattr(detok, "output_token_ids", None)
                if detok is not None
                else getattr(engine_core_output, "new_token_ids", None)
            ) or ()
            log_record.setdefault(
                "gen_ai.usage.prompt_tokens",
                len(prompt_token_ids) or breakdown.total_tokens,
            )
            log_record.setdefault(
                "gen_ai.usage.completion_tokens", len(output_token_ids)
            )
            ext_req_id = getattr(req_state, "external_req_id", None) or getattr(
                req_state, "request_id", None
            )
            if ext_req_id:
                log_record.setdefault("gen_ai.request.id", ext_req_id)
            stats = getattr(req_state, "stats", None)
            if stats is not None:
                ttft = getattr(stats, "first_token_latency", None)
                if isinstance(ttft, (int, float)):
                    log_record.setdefault(
                        "gen_ai.latency.time_to_first_token", round(float(ttft), 6)
                    )
                scheduled_ts = getattr(stats, "scheduled_ts", 0.0) or 0.0
                queued_ts = getattr(stats, "queued_ts", 0.0) or 0.0
                first_token_ts = getattr(stats, "first_token_ts", 0.0) or 0.0
                last_token_ts = getattr(stats, "last_token_ts", 0.0) or 0.0
                if scheduled_ts > 0 and queued_ts > 0:
                    log_record.setdefault(
                        "gen_ai.latency.time_in_queue",
                        round(max(0.0, float(scheduled_ts - queued_ts)), 6),
                    )
                if first_token_ts > 0 and scheduled_ts > 0:
                    log_record.setdefault(
                        "gen_ai.latency.time_in_model_prefill",
                        round(max(0.0, float(first_token_ts - scheduled_ts)), 6),
                    )
                if last_token_ts > 0 and first_token_ts > 0:
                    decode_sec = max(0.0, float(last_token_ts - first_token_ts))
                    log_record.setdefault(
                        "gen_ai.latency.time_in_model_decode",
                        round(decode_sec, 6),
                    )
                    if isinstance(ttft, (int, float)) and ttft > 0:
                        log_record.setdefault(
                            "gen_ai.latency.e2e",
                            round(float(ttft) + decode_sec, 6),
                        )
            try:
                with open(self.span_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_record, sort_keys=True) + "\n")
            except OSError:
                logger.debug("Failed to write CUJ-2 span log to %s", self.span_log_path)
        return attrs


_SERVICE: AgenticTelemetryService | None = None


def get_telemetry_service() -> AgenticTelemetryService:
    """Return the process-local `AgenticTelemetryService` singleton."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AgenticTelemetryService()
    return _SERVICE


def reset_telemetry_service(
    service: AgenticTelemetryService | None = None,
) -> AgenticTelemetryService:
    """Reset or replace the process-local telemetry service (for testing)."""
    global _SERVICE
    _SERVICE = service if service is not None else AgenticTelemetryService()
    return _SERVICE


def patch_trace_headers(tracing_utils_mod: Any = None) -> None:
    """Extend `vllm.tracing.utils.TRACE_HEADERS` with agent session headers."""
    if tracing_utils_mod is None:
        from vllm.tracing import utils as tracing_utils_mod

    headers: list[str] = tracing_utils_mod.TRACE_HEADERS
    for hdr in AGENT_TRACE_HEADERS:
        if hdr not in headers:
            headers.append(hdr)

    if getattr(tracing_utils_mod, "_tpu_agentic_trace_patch", False):
        return

    def _wrapped_contains_trace_headers(raw_headers: Mapping[str, str]) -> bool:
        lowered = {k.lower() for k in raw_headers}
        return any(h.lower() in lowered for h in tracing_utils_mod.TRACE_HEADERS)

    def _wrapped_extract_trace_headers(
        raw_headers: Mapping[str, str],
    ) -> Mapping[str, str]:
        lowered = {k.lower(): v for k, v in raw_headers.items()}
        extracted = {
            h: lowered[h.lower()]
            for h in tracing_utils_mod.TRACE_HEADERS
            if h.lower() in lowered
        }
        _ACTIVE_TRACE_HEADERS.set(dict(extracted) if extracted else None)
        return extracted

    if hasattr(tracing_utils_mod, "contains_trace_headers"):
        tracing_utils_mod.contains_trace_headers = _wrapped_contains_trace_headers
    if hasattr(tracing_utils_mod, "extract_trace_headers"):
        tracing_utils_mod.extract_trace_headers = _wrapped_extract_trace_headers
    tracing_utils_mod._tpu_agentic_trace_patch = True

    for mod_name in ("vllm.tracing", "vllm.entrypoints.serve.engine.serving"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            if hasattr(mod, "contains_trace_headers"):
                mod.contains_trace_headers = _wrapped_contains_trace_headers
            if hasattr(mod, "extract_trace_headers"):
                mod.extract_trace_headers = _wrapped_extract_trace_headers


def patch_block_pool(
    block_pool_cls: Any = None,
    service: AgenticTelemetryService | None = None,
) -> None:
    """Wrap `BlockPool` to track per-session block ownership and batch eviction sync."""
    if block_pool_cls is None:
        from vllm.v1.core.block_pool import BlockPool as block_pool_cls

    if block_pool_cls.__dict__.get("_tpu_agentic_eviction_patch", False):
        return

    orig_evict = block_pool_cls._maybe_evict_cached_block
    orig_cache_full = getattr(block_pool_cls, "cache_full_blocks", None)
    orig_get_new_blocks = getattr(block_pool_cls, "get_new_blocks", None)
    orig_evict_blocks = getattr(block_pool_cls, "evict_blocks", None)

    if orig_cache_full is not None:

        def _wrapped_cache_full_blocks(
            self: Any,
            request: Any,
            blocks: list[Any],
            num_cached_blocks: int,
            num_full_blocks: int,
            block_size: int,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            res = orig_cache_full(
                self,
                request,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                block_size,
                *args,
                **kwargs,
            )
            sid, _, _ = extract_agent_headers(getattr(request, "trace_headers", None))
            if not sid:
                sid = getattr(request, "session_id", None)
            all_tokens = (
                getattr(request, "all_token_ids", None)
                or getattr(request, "prompt_token_ids", None)
                or ()
            )
            blk_sessions = getattr(self, "_cuj2_block_session", None)
            if blk_sessions is None:
                blk_sessions = {}
                self._cuj2_block_session = blk_sessions
            blk_hashes = getattr(self, "_cuj2_block_rolling_hash", None)
            if blk_hashes is None:
                blk_hashes = {}
                self._cuj2_block_rolling_hash = blk_hashes

            prev_hash = 0
            if num_cached_blocks > 0 and num_cached_blocks - 1 < len(blocks):
                prev_blk = blocks[num_cached_blocks - 1]
                prev_id = getattr(prev_blk, "block_id", None)
                prev_hash = (
                    blk_hashes.get(prev_id)
                    if prev_id is not None and prev_id in blk_hashes
                    else getattr(prev_blk, "_cuj2_rolling_hash", 0)
                ) or 0
            if sid:
                for i in range(min(num_cached_blocks, len(blocks))):
                    blk = blocks[i]
                    if not getattr(blk, "is_null", False):
                        blk_id = getattr(blk, "block_id", None)
                        if blk_id is not None:
                            blk_sessions[blk_id] = sid
                        try:
                            blk._cuj2_session_id = sid
                        except AttributeError:
                            pass
            for i in range(num_cached_blocks, min(num_full_blocks, len(blocks))):
                blk = blocks[i]
                if getattr(blk, "is_null", False):
                    continue
                blk_id = getattr(blk, "block_id", None)
                if all_tokens and (i + 1) * block_size <= len(all_tokens):
                    chunk = tuple(all_tokens[i * block_size : (i + 1) * block_size])
                    prev_hash = hash((prev_hash, chunk))
                    if blk_id is not None:
                        blk_hashes[blk_id] = prev_hash
                    try:
                        blk._cuj2_rolling_hash = prev_hash
                    except AttributeError:
                        pass
                if sid:
                    if blk_id is not None:
                        blk_sessions[blk_id] = sid
                    try:
                        blk._cuj2_session_id = sid
                    except AttributeError:
                        pass
            return res

        block_pool_cls.cache_full_blocks = _wrapped_cache_full_blocks

    def _wrapped_maybe_evict_cached_block(self: Any, block: Any) -> bool:
        raw_hash = getattr(block, "block_hash", None)
        blk_id = getattr(block, "block_id", None)
        blk_sessions = getattr(self, "_cuj2_block_session", None)
        blk_hashes = getattr(self, "_cuj2_block_rolling_hash", None)
        rolling_hash = (
            blk_hashes.get(blk_id)
            if blk_hashes is not None and blk_id is not None and blk_id in blk_hashes
            else getattr(block, "_cuj2_rolling_hash", None)
        )
        evicted_sid = (
            blk_sessions.get(blk_id)
            if blk_sessions is not None
            and blk_id is not None
            and blk_id in blk_sessions
            else getattr(block, "_cuj2_session_id", None)
        )
        evicted = orig_evict(self, block)
        if evicted and raw_hash is not None:
            if blk_sessions is not None and blk_id is not None:
                blk_sessions.pop(blk_id, None)
            if blk_hashes is not None and blk_id is not None:
                blk_hashes.pop(blk_id, None)
            for attr in ("_cuj2_rolling_hash", "_cuj2_session_id"):
                if hasattr(block, attr):
                    try:
                        setattr(block, attr, None)
                    except AttributeError:
                        pass
            alloc_sid = getattr(self, "_cuj2_allocating_session_id", None)
            record_sid = (
                None
                if (evicted_sid is not None and evicted_sid == alloc_sid)
                else evicted_sid
            )
            in_batch = bool(getattr(self, "_cuj2_in_batch_evict", False))
            svc = service or get_telemetry_service()
            svc.on_block_evicted(
                rolling_hash if rolling_hash is not None else raw_hash,
                session_id=record_sid,
                defer_sync=in_batch,
            )
        return evicted

    block_pool_cls._maybe_evict_cached_block = _wrapped_maybe_evict_cached_block

    if orig_get_new_blocks is not None:

        def _wrapped_get_new_blocks(
            self: Any, num_blocks: int, *args: Any, **kwargs: Any
        ) -> Any:
            self._cuj2_in_batch_evict = True
            try:
                return orig_get_new_blocks(self, num_blocks, *args, **kwargs)
            finally:
                self._cuj2_in_batch_evict = False
                svc = service or get_telemetry_service()
                svc.flush_eviction_sync()

        block_pool_cls.get_new_blocks = _wrapped_get_new_blocks

    if orig_evict_blocks is not None:

        def _wrapped_evict_blocks(
            self: Any, block_ids: Any, *args: Any, **kwargs: Any
        ) -> Any:
            self._cuj2_in_batch_evict = True
            try:
                return orig_evict_blocks(self, block_ids, *args, **kwargs)
            finally:
                self._cuj2_in_batch_evict = False
                svc = service or get_telemetry_service()
                svc.flush_eviction_sync()

        block_pool_cls.evict_blocks = _wrapped_evict_blocks

    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        if not KVCacheManager.__dict__.get("_tpu_agentic_alloc_patch", False):
            orig_allocate_slots = KVCacheManager.allocate_slots

            def _wrapped_allocate_slots(
                self: Any, request: Any, *args: Any, **kwargs: Any
            ) -> Any:
                sid, _, _ = extract_agent_headers(
                    getattr(request, "trace_headers", None)
                )
                if not sid:
                    sid = getattr(request, "session_id", None)
                bp = getattr(self, "block_pool", None)
                if bp is not None:
                    bp._cuj2_allocating_session_id = sid
                try:
                    return orig_allocate_slots(self, request, *args, **kwargs)
                finally:
                    if bp is not None:
                        bp._cuj2_allocating_session_id = None

            KVCacheManager.allocate_slots = _wrapped_allocate_slots
            KVCacheManager._tpu_agentic_alloc_patch = True
    except Exception:
        pass

    block_pool_cls._tpu_agentic_eviction_patch = True


class _Cuj2TpuPaddingCollector:
    """Prometheus collector exposing server-level TPU round padding counters."""

    _name = "cuj2_tpu_padding"

    def __init__(self, service: AgenticTelemetryService | None = None) -> None:
        self._service = service

    def collect(self) -> Any:
        from prometheus_client.core import CounterMetricFamily

        svc = self._service or get_telemetry_service()
        counts = svc.sync_padding_state()
        rep = str(svc.replica_id)

        rounds_metric = CounterMetricFamily(
            "vllm:tpu_padding_rounds_total",
            "Total TPU model execution rounds by phase (prompt vs reply)",
            labels=["phase", "replica_id"],
        )
        rounds_metric.add_metric(["prompt", rep], float(counts["prompt_rounds"]))
        rounds_metric.add_metric(["reply", rep], float(counts["reply_rounds"]))
        yield rounds_metric

        tokens_metric = CounterMetricFamily(
            "vllm:tpu_padding_tokens_total",
            "Total TPU scheduled real vs static bucket filler tokens by phase",
            labels=["phase", "kind", "replica_id"],
        )
        tokens_metric.add_metric(
            ["prompt", "real", rep], float(counts["prompt_real_tokens"])
        )
        tokens_metric.add_metric(
            ["prompt", "filler", rep], float(counts["prompt_filler_tokens"])
        )
        tokens_metric.add_metric(
            ["reply", "real", rep], float(counts["reply_real_tokens"])
        )
        tokens_metric.add_metric(
            ["reply", "filler", rep], float(counts["reply_filler_tokens"])
        )
        yield tokens_metric


def _register_padding_collector_on_registry(
    registry: Any, service: AgenticTelemetryService | None = None
) -> None:
    if getattr(registry, "_cuj2_padding_collector_registered", False):
        return
    try:
        registry.register(_Cuj2TpuPaddingCollector(service=service))
        registry._cuj2_padding_collector_registered = True
    except ValueError:
        pass


def _is_padding_sync_writer(runner: Any) -> bool:
    """Return True only for per-replica worker rank 0 to avoid `/dev/shm` races."""
    if getattr(runner, "is_driver_worker", None) is False:
        return False
    pc = getattr(runner, "parallel_config", None)
    if pc is not None:
        world = int(getattr(pc, "world_size", 1) or 1)
        rank = int(getattr(pc, "rank", 0) or 0)
        return (rank % max(1, world)) == 0
    rank_attr = getattr(runner, "rank", None)
    if rank_attr is not None:
        return int(rank_attr) == 0
    return True


def patch_tpu_padding_metrics(
    runner_cls: Any = None,
    service: AgenticTelemetryService | None = None,
) -> None:
    """Hook `TPUModelRunner._prepare_inputs` and Prometheus `/metrics` for server-level padding."""
    if runner_cls is None:
        try:
            from vllm_torchtpu.runner.tpu_runner import TPUModelRunner as runner_cls
        except Exception:
            runner_cls = None

    if runner_cls is not None and not runner_cls.__dict__.get(
        "_tpu_agentic_padding_patch", False
    ):
        orig_prepare_inputs = getattr(runner_cls, "_prepare_inputs", None)
        if orig_prepare_inputs is not None:

            def _wrapped_prepare_inputs(
                self: Any,
                scheduler_output: Any,
                start_index: int,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                res = orig_prepare_inputs(
                    self, scheduler_output, start_index, *args, **kwargs
                )
                try:
                    layout_plan = getattr(self, "_last_sequence_layout_plan", None)
                    if layout_plan is not None:
                        real_tokens = int(
                            getattr(layout_plan, "global_num_tokens", 0) or 0
                        )
                        padded_tokens = int(
                            getattr(layout_plan, "global_padded_num_tokens", 0) or 0
                        )
                    else:
                        real_tokens = int(
                            getattr(scheduler_output, "total_num_scheduled_tokens", 0)
                            or 0
                        )
                        padded_tokens = real_tokens + compute_tpu_bucket_padding_tokens(
                            real_tokens
                        )
                    if (
                        isinstance(res, tuple)
                        and len(res) >= 4
                        and isinstance(res[3], int)
                    ):
                        chunk_num_reqs = res[3]
                    else:
                        chunk_num_reqs = int(
                            getattr(getattr(self, "input_batch", None), "num_reqs", 1)
                            or 1
                        )
                    svc = service or get_telemetry_service()
                    svc.on_tpu_round_padded(
                        real_tokens,
                        padded_tokens,
                        is_prompt_round=(real_tokens > chunk_num_reqs),
                        sync_ipc=_is_padding_sync_writer(self),
                    )
                except Exception:
                    pass
                return res

            runner_cls._prepare_inputs = _wrapped_prepare_inputs
            runner_cls._tpu_agentic_padding_patch = True

    try:
        from prometheus_client import REGISTRY, multiprocess

        _register_padding_collector_on_registry(REGISTRY, service=service)
        mp_cls = multiprocess.MultiProcessCollector
        if not getattr(mp_cls, "_tpu_agentic_mp_patch", False):
            orig_mp_collect = mp_cls.collect

            def _wrapped_mp_collect(self: Any) -> Any:
                yield from orig_mp_collect(self)
                yield from _Cuj2TpuPaddingCollector(service=service).collect()

            mp_cls.collect = _wrapped_mp_collect
            mp_cls._tpu_agentic_mp_patch = True
    except Exception:
        pass

    try:
        import vllm.v1.metrics.prometheus as prom_mod

        if not getattr(prom_mod, "_tpu_agentic_prom_patch", False):
            orig_get_reg = prom_mod.get_prometheus_registry

            def _wrapped_get_prometheus_registry(*args: Any, **kwargs: Any) -> Any:
                reg = orig_get_reg(*args, **kwargs)
                if os.getenv("PROMETHEUS_MULTIPROC_DIR") is None:
                    _register_padding_collector_on_registry(reg, service=service)
                return reg

            prom_mod.get_prometheus_registry = _wrapped_get_prometheus_registry
            prom_mod._tpu_agentic_prom_patch = True
            inst_mod = sys.modules.get(
                "vllm.entrypoints.serve.instrumentator.metrics"
            )
            if inst_mod is not None and hasattr(inst_mod, "get_prometheus_registry"):
                inst_mod.get_prometheus_registry = _wrapped_get_prometheus_registry
    except Exception:
        pass


def patch_output_processor(
    output_processor_cls: Any = None,
    output_processor_mod: Any = None,
    service: AgenticTelemetryService | None = None,
) -> None:
    """Wrap `OutputProcessor` to classify prefill turns and enrich OTel spans."""
    if output_processor_mod is None or output_processor_cls is None:
        import vllm.v1.engine.output_processor as output_processor_mod

        output_processor_cls = output_processor_mod.OutputProcessor

    if output_processor_cls.__dict__.get("_tpu_agentic_output_patch", False):
        return

    orig_add_request = getattr(output_processor_cls, "add_request", None)
    orig_process_outputs = output_processor_cls.process_outputs
    orig_do_tracing = output_processor_cls.do_tracing

    if orig_add_request is not None:

        def _wrapped_add_request(
            self: Any,
            request: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            active_hdrs = _ACTIVE_TRACE_HEADERS.get()
            if getattr(request, "trace_headers", None) is None and active_hdrs:
                try:
                    request.trace_headers = dict(active_hdrs)
                except AttributeError:
                    pass
            if not getattr(request, "session_id", None):
                hdrs = getattr(request, "trace_headers", None) or active_hdrs
                sid, _, _ = extract_agent_headers(hdrs)
                if sid:
                    try:
                        request.session_id = sid
                    except AttributeError:
                        pass
            res = orig_add_request(self, request, *args, **kwargs)
            req_states = getattr(self, "request_states", {})
            req_state = req_states.get(getattr(request, "request_id", None))
            if req_state is not None and getattr(request, "trace_headers", None):
                try:
                    req_state.trace_headers = dict(request.trace_headers)
                except AttributeError:
                    pass
            return res

        output_processor_cls.add_request = _wrapped_add_request

    def _wrapped_process_outputs(
        self: Any,
        engine_core_outputs: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        svc = service or get_telemetry_service()
        prefilling_outputs: list[tuple[Any, Any]] = []
        finished_without_tracing: list[tuple[Any, Any]] = []
        req_states = getattr(self, "request_states", {})
        tracing_enabled = getattr(self, "tracing_enabled", False)

        for out in engine_core_outputs:
            req_state = req_states.get(getattr(out, "request_id", None))
            if req_state is None:
                continue
            if getattr(req_state, "is_prefilling", False):
                prefilling_outputs.append((req_state, out))
            if getattr(out, "finish_reason", None) is not None and not tracing_enabled:
                finished_without_tracing.append((req_state, out))

        for req_state, out in prefilling_outputs:
            svc.on_prefill_output(req_state, out)

        result = orig_process_outputs(
            self,
            engine_core_outputs,
            *args,
            **kwargs,
        )

        for req_state, out in finished_without_tracing:
            if svc.span_log_path:
                svc.build_span_attributes(req_state, out)
            else:
                svc.on_request_finished(req_state, out)

        return result

    def _wrapped_do_tracing(
        self: Any,
        engine_core_output: Any,
        req_state: Any,
        iteration_stats: Any,
    ) -> None:
        svc = service or get_telemetry_service()
        extra_attrs = svc.build_span_attributes(req_state, engine_core_output)
        orig_instrument = output_processor_mod.instrument_manual

        def _enriched_instrument_manual(
            span_name: str,
            start_time: int,
            attributes: dict[str, Any] | None = None,
            context: Any = None,
            kind: Any = None,
        ) -> None:
            merged = dict(attributes or {})
            merged.update(extra_attrs)
            return orig_instrument(
                span_name=span_name,
                start_time=start_time,
                attributes=merged,
                context=context,
                kind=kind,
            )

        output_processor_mod.instrument_manual = _enriched_instrument_manual
        try:
            orig_do_tracing(self, engine_core_output, req_state, iteration_stats)
        finally:
            output_processor_mod.instrument_manual = orig_instrument

    output_processor_cls.process_outputs = _wrapped_process_outputs
    output_processor_cls.do_tracing = _wrapped_do_tracing
    output_processor_cls._tpu_agentic_output_patch = True


def _sync_block_size_from_config(vllm_config: Any) -> None:
    """Sync `cache_config.block_size` into env, IPC file, and active service."""
    bs = getattr(getattr(vllm_config, "cache_config", None), "block_size", None)
    if isinstance(bs, int) and bs > 0:
        os.environ.setdefault("VLLM_TPU_AGENTIC_BLOCK_SIZE", str(bs))
        try:
            with open(_BLOCK_SIZE_SYNC_PATH, "w", encoding="utf-8") as f:
                f.write(str(bs))
        except OSError:
            pass
        if _SERVICE is not None and not _SERVICE._explicit_block_size:
            _SERVICE.session_tracker.block_size = bs
            _SERVICE._explicit_block_size = True
            _SERVICE._block_size_checked = True


def patch_tpu_block_size() -> None:
    """Hook `TpuPlatform.update_block_size_for_backend` to sync TPU KV block size."""
    try:
        from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

        if getattr(TpuPlatform, "_tpu_agentic_block_size_patch", False):
            return
        orig_update = TpuPlatform.update_block_size_for_backend

        @classmethod  # type: ignore[misc]
        def _wrapped_update_block_size(cls: Any, vllm_config: Any) -> Any:
            res = orig_update(vllm_config)
            _sync_block_size_from_config(vllm_config)
            return res

        TpuPlatform.update_block_size_for_backend = _wrapped_update_block_size
        TpuPlatform._tpu_agentic_block_size_patch = True
    except Exception:
        logger.debug("Skipping TpuPlatform block_size sync patch")


def install_agentic_telemetry_patches() -> None:
    """Install CUJ-2 agentic telemetry hooks when `VLLM_TPU_AGENTIC_TELEMETRY=1`."""
    if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
        return

    patch_trace_headers()
    patch_tpu_block_size()
    patch_tpu_padding_metrics()
    try:
        patch_block_pool()
    except ImportError:
        logger.debug("Skipping BlockPool telemetry patch; vllm BlockPool unavailable")
    try:
        patch_output_processor()
    except ImportError:
        logger.debug(
            "Skipping OutputProcessor telemetry patch; OutputProcessor unavailable"
        )
    logger.info("Installed vllm-torchtpu CUJ-2 agentic telemetry patches")
