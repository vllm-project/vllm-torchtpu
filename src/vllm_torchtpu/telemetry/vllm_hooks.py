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
"""Flag-gated vLLM lifecycle hooks for multi-turn agentic serving telemetry."""

import contextlib
import contextvars
import functools
import glob
import json
import os
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.telemetry.eviction_tracker import (
    BlockEvictionTracker,
    TpuPaddingCounterTracker,
    extract_hardware_span_attributes,
)
from vllm_torchtpu.telemetry.padding_metrics import register_tpu_padding_collector
from vllm_torchtpu.telemetry.token_breakdown import (
    CATEGORIES,
    BoundedLRU,
    SessionTurnTracker,
    TokenBreakdown,
    compute_rolling_block_hashes,
)

logger = init_logger(__name__)

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
    *(f"gen_ai.usage.input_tokens.{cat}" for cat in CATEGORIES),
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

_TRACING_CONSUMER_MODULES: tuple[str, ...] = (
    "vllm.tracing",
    "vllm.entrypoints.serve.engine.serving",
    "vllm.entrypoints.openai.engine.serving",
    "vllm.entrypoints.pooling.base.serving",
    "vllm.entrypoints.generate.base.serving",
    "vllm.entrypoints.generate.generative_scoring.serving",
)

_ACTIVE_TRACE_HEADERS: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar("_ACTIVE_TRACE_HEADERS", default=None)
)


def _ipc_base_dir() -> str:
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return "/dev/shm"
    return tempfile.gettempdir()


def _sanitize_replica_id(replica_id: str) -> str:
    return "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(replica_id)
    )


def _atomic_write_text(path: str, content: str) -> None:
    """Atomically write `content` to `path`, cleaning up temp file on error."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)


def _resolve_default_dp_rank() -> int:
    try:
        from vllm.config.vllm import get_current_vllm_config

        cfg = get_current_vllm_config()
        pc = getattr(cfg, "parallel_config", None)
        return max(0, int(getattr(pc, "data_parallel_rank", 0) or 0))
    except Exception:
        return 0


def _proc_stat_fields(pid: int) -> tuple[str, str, int, int] | None:
    """Return `(comm, state, ppid, starttime_ticks)` from `/proc/<pid>/stat`."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            raw = f.read()
        rparen = raw.rfind(")")
        if rparen < 0:
            return None
        comm = raw[raw.find("(") + 1 : rparen]
        rest = raw[rparen + 2 :].split()
        return comm, rest[0], int(rest[1]), int(rest[19])
    except (OSError, ValueError, IndexError):
        return None


_LINEAGE_CACHE: tuple[int, int, list[int]] | None = None


def _current_process_lineage() -> tuple[int, int, list[int]]:
    """Return `(pid, proc_start_ticks, ancestor_pids)` cached per PID."""
    global _LINEAGE_CACHE
    pid = os.getpid()
    if _LINEAGE_CACHE is not None and _LINEAGE_CACHE[0] == pid:
        return _LINEAGE_CACHE
    stat = _proc_stat_fields(pid)
    proc_start = stat[3] if stat is not None else 0
    ancestors: list[int] = [pid]
    cur = stat[2] if stat is not None else os.getppid()
    for _ in range(16):
        if cur <= 1 or cur in ancestors:
            break
        ancestors.append(cur)
        cur_stat = _proc_stat_fields(cur)
        if cur_stat is None:
            break
        cur = cur_stat[2]
    _LINEAGE_CACHE = (pid, proc_start, ancestors)
    return _LINEAGE_CACHE


def _is_writer_alive(pid: int, expected_start: int = 0) -> bool:
    if pid <= 0:
        return False
    stat = _proc_stat_fields(pid)
    if stat is not None:
        _, state, _, start_ticks = stat
        if state == "Z":
            return False
        return not (expected_start > 0 and start_ticks != expected_start)
    if not os.path.isdir("/proc"):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    return False


def _belongs_to_current_server(ancestors: Sequence[int]) -> bool:
    if not ancestors:
        return True
    my_pid = os.getpid()
    if my_pid in ancestors:
        return True
    my_ppid = os.getppid()
    if my_ppid > 1 and my_ppid in ancestors:
        parent_stat = _proc_stat_fields(my_ppid)
        if parent_stat is not None and parent_stat[0].startswith(("python", "vllm")):
            return True
    return False


_PADDING_COUNTER_KEYS: tuple[str, ...] = (
    "prompt_rounds",
    "prompt_real_tokens",
    "prompt_filler_tokens",
    "reply_rounds",
    "reply_real_tokens",
    "reply_filler_tokens",
)
_IPC_UID_PREFIX = f"{_ipc_base_dir()}/.vllm_tpu_agentic_{os.getuid()}"
_EVICTION_SYNC_PATH = f"{_IPC_UID_PREFIX}_eviction_state"
_PADDING_SYNC_PATH = f"{_IPC_UID_PREFIX}_padding_state"
_PADDING_SYNC_MIN_INTERVAL_S = 0.25
_MAX_EVICTION_SYNC_BYTES = 524_288
_COMPACT_RECENT_SESSIONS = 1024
_INFLIGHT_REQUESTS_MAXSIZE = 16384
_DETECTED_VLLM_BLOCK_SIZE: int | None = None
_PATCHES_LOGGED: bool = False


@dataclass(slots=True)
class _InFlightRequestState:
    session_id: str | None = None
    breakdown: TokenBreakdown | None = None
    external_tokens: int = 0
    prefill_ts: float | None = None
    eviction_metrics: tuple[float | None, int | None] | None = None
    eviction_ts: float | None = None


def _extract_request_id(req_state: Any, engine_core_output: Any) -> str:
    return str(
        getattr(engine_core_output, "request_id", None)
        or getattr(req_state, "request_id", "")
        or ""
    )


def _extract_output_token_ids(req_state: Any, engine_core_output: Any) -> Sequence[int]:
    detok = getattr(req_state, "detokenizer", None)
    detok_ids = getattr(detok, "output_token_ids", None) if detok is not None else None
    if detok_ids:
        return detok_ids
    return getattr(engine_core_output, "new_token_ids", None) or ()


def _resolve_trace_headers(
    req_state: Any, engine_core_output: Any
) -> Mapping[str, str] | None:
    """Return trace headers from `engine_core_output` or `req_state` fallback."""
    return getattr(engine_core_output, "trace_headers", None) or getattr(
        req_state, "trace_headers", None
    )


def _resolve_num_cached_tokens(req_state: Any, engine_core_output: Any) -> int:
    """Extract `num_cached_tokens` from vLLM V1 `prefill_stats` or `RequestState`."""
    prefill_stats = getattr(engine_core_output, "prefill_stats", None)
    if prefill_stats is not None:
        return int(prefill_stats.num_cached_tokens)
    remote_cached = getattr(req_state, "remote_prefill_cached_tokens", None)
    if remote_cached is not None:
        return int(remote_cached)
    return int(getattr(req_state, "num_cached_tokens", 0) or 0)


def _resolve_num_external_cached_tokens(req_state: Any, engine_core_output: Any) -> int:
    """Extract external/DRAM cached tokens from `PrefillStats` or stashed state."""
    prefill_stats = getattr(engine_core_output, "prefill_stats", None)
    if prefill_stats is not None:
        return int(prefill_stats.num_external_cached_tokens)
    stashed = getattr(req_state, "_agentic_external_cached_tokens", None)
    return int(stashed or 0)


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


def _extract_request_session_id(
    request: Any, fallback_headers: Mapping[str, str] | None = None
) -> str | None:
    hdrs = getattr(request, "trace_headers", None) or fallback_headers
    sid, _, _ = extract_agent_headers(hdrs)
    return sid or getattr(request, "session_id", None) or None


class AgenticTelemetryService:
    """Coordinates session tracking, eviction age, and OTel span attributes."""

    def __init__(
        self,
        *,
        block_size: int | None = None,
        replica_id: str | None = None,
        dp_rank: int | None = None,
        span_log_path: str | None = None,
        session_tracker: SessionTurnTracker | None = None,
        eviction_tracker: BlockEvictionTracker | None = None,
        padding_tracker: TpuPaddingCounterTracker | None = None,
        inflight_maxsize: int = _INFLIGHT_REQUESTS_MAXSIZE,
    ) -> None:
        self.replica_id = replica_id or envs.VLLM_TPU_REPLICA_ID or socket.gethostname()
        self.dp_rank = dp_rank
        self.span_log_path = (
            span_log_path
            if span_log_path is not None
            else envs.VLLM_TPU_AGENTIC_SPAN_LOG
        )
        env_bs = int(envs.VLLM_TPU_AGENTIC_BLOCK_SIZE or 0)
        detected_bs = _DETECTED_VLLM_BLOCK_SIZE or 0
        auto_bs = env_bs if env_bs > 0 else detected_bs
        self._explicit_block_size = (
            block_size is not None or session_tracker is not None or auto_bs > 0
        )
        self._block_size_checked = self._explicit_block_size
        resolved_block_size = (
            block_size if block_size is not None else (auto_bs if auto_bs > 0 else 16)
        )
        self.session_tracker = session_tracker or SessionTurnTracker(
            block_size=resolved_block_size
        )
        self.eviction_tracker = eviction_tracker or BlockEvictionTracker()
        self.padding_tracker = padding_tracker or TpuPaddingCounterTracker()
        self.category_totals: dict[str, int] = {cat: 0 for cat in CATEGORIES}
        self._inflight: BoundedLRU[str, _InFlightRequestState] = BoundedLRU(
            inflight_maxsize
        )
        self._eviction_sync_dirty: bool = False
        self._eviction_writer_initialized: bool = False
        self._eviction_sync_cursors: dict[str, tuple[int, int]] = {}
        self._padding_lock = threading.RLock()
        self._padding_writer_started_at: float = time.monotonic()
        self._padding_dp_size: int | None = None
        self._padding_sync_dirty: bool = False
        self._padding_dirty_dp_rank: int | None = None
        self._padding_sync_timer: threading.Timer | None = None
        self._last_padding_sync_ts: float = 0.0

    def _safe_replica_id(self) -> str:
        return _sanitize_replica_id(self.replica_id)

    def _resolved_dp_rank(self) -> int:
        if self.dp_rank is None:
            self.dp_rank = _resolve_default_dp_rank()
        return self.dp_rank

    def _eviction_sync_prefix(self) -> str:
        return f"{_EVICTION_SYNC_PATH}_{self._safe_replica_id()}"

    def _eviction_sync_path(self, dp_rank: int | None = None) -> str:
        rank = self._resolved_dp_rank() if dp_rank is None else int(dp_rank)
        return f"{self._eviction_sync_prefix()}_dp{rank}"

    def _padding_sync_prefix(self) -> str:
        return f"{_PADDING_SYNC_PATH}_{self._safe_replica_id()}"

    def _padding_sync_path(self, dp_rank: int | None = None) -> str:
        rank = self._resolved_dp_rank() if dp_rank is None else int(dp_rank)
        return f"{self._padding_sync_prefix()}_dp{rank}_{os.getpid()}"

    def _write_padding_snapshot_locked(self, rank: int, now: float) -> None:
        pid, proc_start, ancestors = _current_process_lineage()
        payload: dict[str, Any] = {
            **self.padding_tracker.as_dict(),
            "dp_rank": rank,
            "pid": pid,
            "proc_start": proc_start,
            "ancestors": ancestors,
            "started_at": self._padding_writer_started_at,
        }
        if self._padding_dp_size is not None:
            payload["dp_size"] = self._padding_dp_size
        self._last_padding_sync_ts = now
        self._padding_sync_dirty = False
        _atomic_write_text(
            self._padding_sync_path(rank),
            json.dumps(payload, separators=(",", ":")),
        )

    def _prune_stale_padding_files_locked(self) -> None:
        prefix = f"{self._padding_sync_prefix()}_dp"
        for sync_path in glob.glob(f"{prefix}*"):
            if ".tmp." in sync_path:
                continue
            try:
                with open(sync_path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, Mapping):
                    continue
                pid = data.get("pid")
                proc_start = int(data.get("proc_start", 0) or 0)
                if isinstance(pid, int) and not _is_writer_alive(pid, proc_start):
                    with contextlib.suppress(OSError):
                        os.unlink(sync_path)
                    continue
                ancestors = data.get("ancestors")
                if isinstance(ancestors, list) and not _belongs_to_current_server(
                    ancestors
                ):
                    continue
                file_rank = int(data.get("dp_rank", -1))
                file_started = float(data.get("started_at", 0.0) or 0.0)
                if (
                    self._padding_dp_size is not None
                    and file_rank >= self._padding_dp_size
                    and file_started < self._padding_writer_started_at
                ):
                    with contextlib.suppress(OSError):
                        os.unlink(sync_path)
            except (OSError, ValueError, TypeError):
                pass

    def init_padding_sync_writer(
        self, dp_rank: int | None = None, dp_size: int | None = None
    ) -> None:
        """Initialize or replace this DP engine's `/dev/shm` padding snapshot."""
        if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
            return
        with self._padding_lock:
            if dp_size is not None:
                self._padding_dp_size = max(1, int(dp_size))
            rank = self._resolved_dp_rank() if dp_rank is None else int(dp_rank)
            if rank == 0:
                self._prune_stale_padding_files_locked()
            self._write_padding_snapshot_locked(rank, time.monotonic())
            # Allow the first live round after warmup to flush immediately.
            self._last_padding_sync_ts = 0.0

    def _flush_deferred_padding_sync(self) -> None:
        with self._padding_lock:
            self._padding_sync_timer = None
            if not self._padding_sync_dirty or not envs.VLLM_TPU_AGENTIC_TELEMETRY:
                return
            rank = (
                self._resolved_dp_rank()
                if self._padding_dirty_dp_rank is None
                else self._padding_dirty_dp_rank
            )
            self._write_padding_snapshot_locked(rank, time.monotonic())

    def flush_padding_sync(self, dp_rank: int | None = None) -> None:
        """Immediately write any rate-limited pending padding counters to IPC."""
        with self._padding_lock:
            if self._padding_sync_timer is not None:
                self._padding_sync_timer.cancel()
                self._padding_sync_timer = None
            if not self._padding_sync_dirty or not envs.VLLM_TPU_AGENTIC_TELEMETRY:
                return
            rank = (
                int(dp_rank)
                if dp_rank is not None
                else (
                    self._resolved_dp_rank()
                    if self._padding_dirty_dp_rank is None
                    else self._padding_dirty_dp_rank
                )
            )
            self._write_padding_snapshot_locked(rank, time.monotonic())

    def on_tpu_round_padded(
        self,
        scheduled_tokens: int,
        padded_tokens: int,
        *,
        is_prompt_round: bool,
        sync_ipc: bool = True,
        dp_rank: int | None = None,
        dp_size: int | None = None,
    ) -> None:
        """Record one TPU round's real vs filler tokens and rate-limit IPC sync."""
        with self._padding_lock:
            self.padding_tracker.record_round(
                scheduled_tokens,
                padded_tokens,
                is_prompt_round=is_prompt_round,
            )
            if dp_size is not None:
                self._padding_dp_size = max(1, int(dp_size))
            if not sync_ipc or not envs.VLLM_TPU_AGENTIC_TELEMETRY:
                return
            rank = self._resolved_dp_rank() if dp_rank is None else int(dp_rank)
            now = time.monotonic()
            elapsed = now - self._last_padding_sync_ts
            if self._last_padding_sync_ts == 0.0 or (
                elapsed >= _PADDING_SYNC_MIN_INTERVAL_S
            ):
                if self._padding_sync_timer is not None:
                    self._padding_sync_timer.cancel()
                    self._padding_sync_timer = None
                self._write_padding_snapshot_locked(rank, now)
                return
            self._padding_sync_dirty = True
            self._padding_dirty_dp_rank = rank
            if self._padding_sync_timer is None:
                delay = max(0.001, _PADDING_SYNC_MIN_INTERVAL_S - elapsed)
                timer = threading.Timer(delay, self._flush_deferred_padding_sync)
                timer.daemon = True
                self._padding_sync_timer = timer
                timer.start()

    def sync_padding_state(self) -> dict[str, int]:
        """Sum TPU round padding counters across live DP engine shared-memory files."""
        if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
            return self.padding_tracker.as_dict()
        prefix = f"{self._padding_sync_prefix()}_dp"
        paths = [p for p in sorted(glob.glob(f"{prefix}*")) if ".tmp." not in p]
        if not paths:
            return self.padding_tracker.as_dict()
        best_by_rank: dict[int, tuple[float, int | None, dict[str, int]]] = {}
        for sync_path in paths:
            try:
                with open(sync_path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, Mapping):
                    continue
                pid = data.get("pid")
                proc_start = int(data.get("proc_start", 0) or 0)
                if isinstance(pid, int) and not _is_writer_alive(pid, proc_start):
                    with contextlib.suppress(OSError):
                        os.unlink(sync_path)
                    continue
                ancestors = data.get("ancestors")
                if isinstance(ancestors, list) and not _belongs_to_current_server(
                    ancestors
                ):
                    continue
                raw_rank = data.get("dp_rank")
                if raw_rank is None:
                    suffix = sync_path[len(prefix) :].split("_", 1)[0]
                    rank = int(suffix)
                else:
                    rank = int(raw_rank)
                started_at = float(data.get("started_at", 0.0) or 0.0)
                raw_dp_size = data.get("dp_size")
                dp_size = int(raw_dp_size) if raw_dp_size is not None else None
                counts = {
                    k: max(0, int(data.get(k, 0) or 0)) for k in _PADDING_COUNTER_KEYS
                }
                prev = best_by_rank.get(rank)
                if prev is None or started_at >= prev[0]:
                    best_by_rank[rank] = (started_at, dp_size, counts)
            except (OSError, ValueError, TypeError):
                pass
        if not best_by_rank:
            return self.padding_tracker.as_dict()
        newest_started, newest_dp_size, _ = max(
            best_by_rank.values(), key=lambda item: item[0]
        )
        totals: dict[str, int] = {k: 0 for k in _PADDING_COUNTER_KEYS}
        for rank, (started_at, _, counts) in best_by_rank.items():
            if (
                newest_dp_size is not None
                and rank >= newest_dp_size
                and started_at < newest_started
            ):
                continue
            for key, val in counts.items():
                totals[key] += val
        return totals

    def flush_eviction_sync(self) -> None:
        """Flush changed per-session eviction deltas after each eviction batch."""
        if not self._eviction_sync_dirty or not envs.VLLM_TPU_AGENTIC_TELEMETRY:
            return
        self._eviction_sync_dirty = False
        changed_sessions = self.eviction_tracker.pop_dirty_session_state()
        last_pair = self.eviction_tracker.last_evicted_pair()
        sync_path = self._eviction_sync_path()
        try:
            needs_replace = not self._eviction_writer_initialized
            if not needs_replace and os.path.exists(sync_path):
                needs_replace = os.path.getsize(sync_path) >= _MAX_EVICTION_SYNC_BYTES
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
                _atomic_write_text(
                    sync_path,
                    json.dumps(payload, separators=(",", ":")) + "\n",
                )
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
        """Parse and merge one eviction sync JSONL line."""
        if not raw_line or not raw_line.startswith("{"):
            return False
        try:
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
                self.eviction_tracker.merge_last_evicted(float(last[0]), int(last[1]))
            return True
        except ValueError:
            return False

    def _sync_single_eviction_file(self, sync_path: str) -> None:
        try:
            st = os.stat(sync_path)
        except OSError:
            return
        prev_ino, offset = self._eviction_sync_cursors.get(sync_path, (-1, 0))
        if prev_ino == st.st_ino and offset == st.st_size:
            return
        if prev_ino != st.st_ino or st.st_size < offset:
            offset = 0
        try:
            with open(sync_path, "rb") as f:
                f.seek(offset)
                chunk = f.read()
            if not chunk:
                self._eviction_sync_cursors[sync_path] = (st.st_ino, offset)
                return
            last_nl = chunk.rfind(b"\n")
            if last_nl == -1:
                if offset == 0:
                    raw_text = chunk.decode("utf-8", errors="replace").strip()
                    if self._apply_eviction_sync_record(raw_text):
                        offset = len(chunk)
                self._eviction_sync_cursors[sync_path] = (st.st_ino, offset)
                return
            complete = chunk[: last_nl + 1]
            offset += len(complete)
            self._eviction_sync_cursors[sync_path] = (st.st_ino, offset)
            for raw_line in complete.decode("utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if line:
                    self._apply_eviction_sync_record(line)
        except OSError:
            pass

    def _maybe_sync_eviction_state(self) -> None:
        """Incrementally merge session eviction deltas across all DP engines."""
        if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
            return
        prefix = f"{self._eviction_sync_prefix()}_dp"
        for sync_path in sorted(glob.glob(f"{prefix}*")):
            if ".tmp." in sync_path:
                continue
            self._sync_single_eviction_file(sync_path)

    def discard_request(self, request_id: str | None) -> None:
        """Drop in-flight bookkeeping for an aborted or rejected request."""
        if request_id:
            inflight = self._inflight.pop(request_id)
            if inflight is not None and inflight.session_id:
                self.session_tracker.discard_pending(inflight.session_id)

    def _maybe_sync_vllm_block_size(self) -> None:
        """Auto-detect `cache_config.block_size` once if not explicitly set."""
        if self._explicit_block_size or self._block_size_checked:
            return
        self._block_size_checked = True
        env_bs = int(envs.VLLM_TPU_AGENTIC_BLOCK_SIZE or 0)
        resolved = env_bs if env_bs > 0 else (_DETECTED_VLLM_BLOCK_SIZE or 0)
        if resolved > 0:
            self.session_tracker.block_size = resolved
            self._explicit_block_size = True
            return
        try:
            from vllm.config.vllm import get_current_vllm_config

            cfg = get_current_vllm_config()
            bs = getattr(getattr(cfg, "cache_config", None), "block_size", None)
            if isinstance(bs, int) and bs > 0:
                self.session_tracker.block_size = bs
                self._explicit_block_size = True
        except Exception as exc:
            logger.debug("Skipping vllm_config block_size lookup: %s", exc)

    def on_block_evicted(
        self,
        block_hash: Any,
        now: float | None = None,
        *,
        session_id: str | None = None,
        defer_sync: bool = False,
    ) -> None:
        """Record an HBM block eviction event (with optional deferred IPC flush)."""
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
        req_id = _extract_request_id(req_state, engine_core_output)
        ext_cached = _resolve_num_external_cached_tokens(req_state, engine_core_output)
        inflight = self._inflight.get(req_id) if req_id else None
        if ext_cached > 0:
            if req_id:
                if inflight is None:
                    inflight = _InFlightRequestState()
                    self._inflight.put(req_id, inflight)
                inflight.external_tokens = ext_cached
            with contextlib.suppress(AttributeError):
                req_state._agentic_external_cached_tokens = ext_cached
        if inflight is not None and inflight.breakdown is not None:
            return inflight.breakdown

        self._maybe_sync_vllm_block_size()
        self._maybe_sync_eviction_state()
        prefill_now = time.monotonic()

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
            if inflight is None:
                inflight = _InFlightRequestState()
            inflight.session_id = session_id
            inflight.breakdown = breakdown
            inflight.prefill_ts = prefill_now
            inflight.eviction_ts = self.eviction_tracker.session_evicted_ts(session_id)
            inflight.eviction_metrics = self._compute_eviction_snapshot(
                req_state,
                breakdown,
                session_id=session_id,
                eval_now=prefill_now,
                ext_cached=ext_cached,
            )
            self._inflight.put(req_id, inflight)
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
            t1_state = (
                self.session_tracker.tier1.peek(session_id) if session_id else None
            )
            verified_len = (
                self.session_tracker._verified_match_len.peek(session_id)
                if session_id
                else None
            )
            reusable_blocks = (
                (verified_len // block_size)
                if (t1_state is not None and verified_len is not None)
                else 0
            )
            all_hashes = compute_rolling_block_hashes(
                prompt_token_ids,
                block_size,
                prefix_hashes=t1_state.block_hashes if t1_state is not None else (),
                start_block=reusable_blocks,
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
        req_id = _extract_request_id(req_state, engine_core_output)
        inflight = self._inflight.pop(req_id) if req_id else None
        breakdown = inflight.breakdown if inflight is not None else None
        if breakdown is None:
            breakdown = self.on_prefill_output(req_state, engine_core_output)
            if req_id:
                self._inflight.pop(req_id)

        session_id, _, turn_id = extract_agent_headers(
            _resolve_trace_headers(req_state, engine_core_output)
        )
        prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
        output_token_ids = _extract_output_token_ids(req_state, engine_core_output)
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
        """Build agentic OTel span attributes for a completed request."""
        req_id = _extract_request_id(req_state, engine_core_output)
        prior_inflight = self._inflight.peek(req_id) if req_id else None
        ext_computed = prior_inflight.external_tokens if prior_inflight else 0
        if not ext_computed:
            ext_computed = _resolve_num_external_cached_tokens(
                req_state, engine_core_output
            )

        snap_evict = prior_inflight.eviction_metrics if prior_inflight else None
        prefill_evict_ts = prior_inflight.eviction_ts if prior_inflight else None
        prefill_ts = prior_inflight.prefill_ts if prior_inflight else None

        if prior_inflight is None and req_id:
            self.on_prefill_output(req_state, engine_core_output)
            populated = self._inflight.peek(req_id)
            if populated is not None:
                snap_evict = populated.eviction_metrics
                prefill_evict_ts = populated.eviction_ts
                prefill_ts = populated.prefill_ts

        breakdown = self.on_request_finished(req_state, engine_core_output)
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
                prefill_evict_ts = self.eviction_tracker.session_evicted_ts(session_id)
            eviction_age_ms, session_evicted_wall = self._compute_eviction_snapshot(
                req_state,
                breakdown,
                session_id=session_id,
                eval_now=eval_now,
                ext_cached=ext_computed,
            )

        had_eviction = breakdown.affinity_honored_eviction > 0 or ext_computed > 0
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
            self._append_span_log(attrs, req_state, engine_core_output, breakdown)
        return attrs

    def _append_span_log(
        self,
        attrs: Mapping[str, Any],
        req_state: Any,
        engine_core_output: Any,
        breakdown: TokenBreakdown,
    ) -> None:
        log_record = dict(attrs)
        trace_hdrs = _resolve_trace_headers(req_state, engine_core_output) or {}
        lowered_hdrs = {k.lower(): v for k, v in trace_hdrs.items() if v is not None}
        if "traceparent" in lowered_hdrs:
            log_record.setdefault("traceparent", lowered_hdrs["traceparent"])
        prompt_token_ids = getattr(req_state, "prompt_token_ids", None) or ()
        output_token_ids = _extract_output_token_ids(req_state, engine_core_output)
        log_record.setdefault(
            "gen_ai.usage.prompt_tokens",
            len(prompt_token_ids) or breakdown.total_tokens,
        )
        log_record.setdefault("gen_ai.usage.completion_tokens", len(output_token_ids))
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
        except OSError as exc:
            logger.warning_once(
                "Failed to write agentic span log to %s: %s", self.span_log_path, exc
            )


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
    if _SERVICE is not None and _SERVICE._padding_sync_timer is not None:
        _SERVICE._padding_sync_timer.cancel()
        _SERVICE._padding_sync_timer = None
    _SERVICE = service if service is not None else AgenticTelemetryService()
    return _SERVICE


def patch_trace_headers(
    tracing_utils_mod: Any = None,
    *,
    refresh_consumers: bool | None = None,
) -> None:
    """Extend `vllm.tracing.utils.TRACE_HEADERS` and refresh consumer bindings."""
    should_refresh = (
        (tracing_utils_mod is None) if refresh_consumers is None else refresh_consumers
    )
    if tracing_utils_mod is None:
        from vllm.tracing import utils as tracing_utils_mod

    headers: list[str] = tracing_utils_mod.TRACE_HEADERS
    for hdr in AGENT_TRACE_HEADERS:
        if hdr not in headers:
            headers.append(hdr)

    if not getattr(tracing_utils_mod, "_tpu_agentic_trace_patch", False):
        orig_contains = getattr(tracing_utils_mod, "contains_trace_headers", None)
        orig_extract = getattr(tracing_utils_mod, "extract_trace_headers", None)

        def _wrapped_contains_trace_headers(raw_headers: Mapping[str, str]) -> bool:
            lowered = {k.lower(): v for k, v in raw_headers.items()}
            matched = (
                bool(orig_contains(lowered))
                if callable(orig_contains)
                else any(h.lower() in lowered for h in tracing_utils_mod.TRACE_HEADERS)
            )
            if not matched:
                _ACTIVE_TRACE_HEADERS.set(None)
            return matched

        def _wrapped_extract_trace_headers(
            raw_headers: Mapping[str, str],
        ) -> Mapping[str, str]:
            lowered = {k.lower(): v for k, v in raw_headers.items()}
            extracted = (
                dict(orig_extract(lowered))
                if callable(orig_extract)
                else {
                    h: lowered[h.lower()]
                    for h in tracing_utils_mod.TRACE_HEADERS
                    if h.lower() in lowered
                }
            )
            _ACTIVE_TRACE_HEADERS.set(dict(extracted) if extracted else None)
            return extracted

        if callable(orig_contains):
            _wrapped_contains_trace_headers = functools.wraps(orig_contains)(
                _wrapped_contains_trace_headers
            )
            tracing_utils_mod.contains_trace_headers = _wrapped_contains_trace_headers
        if callable(orig_extract):
            _wrapped_extract_trace_headers = functools.wraps(orig_extract)(
                _wrapped_extract_trace_headers
            )
            tracing_utils_mod.extract_trace_headers = _wrapped_extract_trace_headers
        tracing_utils_mod._agentic_contains_fn = _wrapped_contains_trace_headers
        tracing_utils_mod._agentic_extract_fn = _wrapped_extract_trace_headers
        tracing_utils_mod._tpu_agentic_trace_patch = True

    if not should_refresh:
        return

    contains_fn = getattr(tracing_utils_mod, "_agentic_contains_fn", None)
    extract_fn = getattr(tracing_utils_mod, "_agentic_extract_fn", None)
    for mod_name in _TRACING_CONSUMER_MODULES:
        mod: Any = sys.modules.get(mod_name)
        if mod is not None:
            if contains_fn is not None and hasattr(mod, "contains_trace_headers"):
                mod.contains_trace_headers = contains_fn
            if extract_fn is not None and hasattr(mod, "extract_trace_headers"):
                mod.extract_trace_headers = extract_fn


def _tag_computed_blocks_for_session(
    block_pool: Any, new_computed_blocks: Any, session_id: str | None
) -> None:
    """Tag matched prefix-hit blocks once at `allocate_slots` (`O(hit_blocks)`)."""
    if block_pool is None or new_computed_blocks is None or not session_id:
        return
    groups = getattr(new_computed_blocks, "blocks", None)
    if not groups:
        return
    blk_sessions = getattr(block_pool, "_agentic_block_session", None)
    if blk_sessions is None:
        blk_sessions = {}
        block_pool._agentic_block_session = blk_sessions
    for group_blocks in groups:
        for blk in group_blocks or ():
            if getattr(blk, "is_null", False):
                continue
            blk_id = getattr(blk, "block_id", None)
            if blk_id is not None:
                blk_sessions[blk_id] = session_id
            with contextlib.suppress(AttributeError):
                blk._agentic_session_id = session_id


def patch_block_pool(
    block_pool_cls: Any = None,
    service: AgenticTelemetryService | None = None,
    *,
    kv_cache_manager_cls: Any = None,
) -> None:
    """Wrap `BlockPool` to track session block ownership and batch eviction sync."""
    is_default_cls = block_pool_cls is None
    if block_pool_cls is None:
        from vllm.v1.core.block_pool import BlockPool as block_pool_cls

    if not block_pool_cls.__dict__.get("_tpu_agentic_eviction_patch", False):
        orig_evict = block_pool_cls._maybe_evict_cached_block
        orig_cache_full = getattr(block_pool_cls, "cache_full_blocks", None)
        orig_get_new_blocks = getattr(block_pool_cls, "get_new_blocks", None)
        orig_evict_blocks = getattr(block_pool_cls, "evict_blocks", None)

        if orig_cache_full is not None:

            @functools.wraps(orig_cache_full)
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
                try:
                    if num_cached_blocks >= num_full_blocks:
                        return res
                    sid = _extract_request_session_id(request)
                    all_tokens = (
                        getattr(request, "all_token_ids", None)
                        or getattr(request, "prompt_token_ids", None)
                        or ()
                    )
                    block_mask = kwargs.get("block_mask")
                    if block_mask is None and len(args) > 1:
                        block_mask = args[1]
                    blk_sessions = getattr(self, "_agentic_block_session", None)
                    if blk_sessions is None:
                        blk_sessions = {}
                        self._agentic_block_session = blk_sessions
                    blk_hashes = getattr(self, "_agentic_block_rolling_hash", None)
                    if blk_hashes is None:
                        blk_hashes = {}
                        self._agentic_block_rolling_hash = blk_hashes

                    prev_hash = 0
                    if num_cached_blocks > 0 and num_cached_blocks - 1 < len(blocks):
                        prev_blk = blocks[num_cached_blocks - 1]
                        prev_id = getattr(prev_blk, "block_id", None)
                        prev_hash = (
                            blk_hashes.get(prev_id)
                            if prev_id is not None and prev_id in blk_hashes
                            else getattr(prev_blk, "_agentic_rolling_hash", 0)
                        ) or 0
                    for i in range(
                        num_cached_blocks, min(num_full_blocks, len(blocks))
                    ):
                        blk = blocks[i]
                        rel_idx = i - num_cached_blocks
                        masked_out = (
                            block_mask is not None
                            and rel_idx < len(block_mask)
                            and not block_mask[rel_idx]
                        )
                        if getattr(blk, "is_null", False) or masked_out:
                            if all_tokens and (i + 1) * block_size <= len(all_tokens):
                                chunk = tuple(
                                    all_tokens[i * block_size : (i + 1) * block_size]
                                )
                                prev_hash = hash((prev_hash, chunk))
                            continue
                        blk_id = getattr(blk, "block_id", None)
                        if all_tokens and (i + 1) * block_size <= len(all_tokens):
                            chunk = tuple(
                                all_tokens[i * block_size : (i + 1) * block_size]
                            )
                            prev_hash = hash((prev_hash, chunk))
                            if blk_id is not None:
                                blk_hashes[blk_id] = prev_hash
                            with contextlib.suppress(AttributeError):
                                blk._agentic_rolling_hash = prev_hash
                        if sid:
                            if blk_id is not None:
                                blk_sessions[blk_id] = sid
                            with contextlib.suppress(AttributeError):
                                blk._agentic_session_id = sid
                except Exception as exc:
                    logger.warning_once(
                        "Agentic telemetry cache_full_blocks hook failed: %s", exc
                    )
                return res

            block_pool_cls.cache_full_blocks = _wrapped_cache_full_blocks

        @functools.wraps(orig_evict)
        def _wrapped_maybe_evict_cached_block(self: Any, block: Any) -> bool:
            raw_hash = getattr(block, "block_hash", None)
            blk_id = getattr(block, "block_id", None)
            blk_sessions = getattr(self, "_agentic_block_session", None)
            blk_hashes = getattr(self, "_agentic_block_rolling_hash", None)
            rolling_hash = (
                blk_hashes.get(blk_id)
                if blk_hashes is not None
                and blk_id is not None
                and blk_id in blk_hashes
                else getattr(block, "_agentic_rolling_hash", None)
            )
            evicted_sid = (
                blk_sessions.get(blk_id)
                if blk_sessions is not None
                and blk_id is not None
                and blk_id in blk_sessions
                else getattr(block, "_agentic_session_id", None)
            )
            evicted = orig_evict(self, block)
            if evicted and raw_hash is not None:
                try:
                    if blk_sessions is not None and blk_id is not None:
                        blk_sessions.pop(blk_id, None)
                    if blk_hashes is not None and blk_id is not None:
                        blk_hashes.pop(blk_id, None)
                    for attr in ("_agentic_rolling_hash", "_agentic_session_id"):
                        if hasattr(block, attr):
                            with contextlib.suppress(AttributeError):
                                setattr(block, attr, None)
                    alloc_sid = getattr(self, "_agentic_allocating_session_id", None)
                    record_sid = (
                        None
                        if (evicted_sid is not None and evicted_sid == alloc_sid)
                        else evicted_sid
                    )
                    in_batch = bool(getattr(self, "_agentic_in_batch_evict", False))
                    svc = service or get_telemetry_service()
                    svc.on_block_evicted(
                        rolling_hash if rolling_hash is not None else raw_hash,
                        session_id=record_sid,
                        defer_sync=in_batch,
                    )
                except Exception as exc:
                    logger.warning_once(
                        "Agentic telemetry block eviction hook failed: %s", exc
                    )
            return evicted

        block_pool_cls._maybe_evict_cached_block = _wrapped_maybe_evict_cached_block

        def _wrap_batch_evict_fn(orig_fn: Any) -> Any:
            @functools.wraps(orig_fn)
            def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                self._agentic_in_batch_evict = True
                try:
                    return orig_fn(self, *args, **kwargs)
                finally:
                    self._agentic_in_batch_evict = False
                    try:
                        svc = service or get_telemetry_service()
                        svc.flush_eviction_sync()
                    except Exception as exc:
                        logger.warning_once(
                            "Agentic telemetry batch eviction flush failed: %s", exc
                        )

            return _wrapped

        if orig_get_new_blocks is not None:
            block_pool_cls.get_new_blocks = _wrap_batch_evict_fn(orig_get_new_blocks)

        if orig_evict_blocks is not None:
            block_pool_cls.evict_blocks = _wrap_batch_evict_fn(orig_evict_blocks)

        block_pool_cls._tpu_agentic_eviction_patch = True

    if kv_cache_manager_cls is None and is_default_cls:
        try:
            from vllm.v1.core.kv_cache_manager import (
                KVCacheManager as kv_cache_manager_cls,
            )
        except ImportError as exc:
            logger.warning_once(
                "Skipping KVCacheManager allocate_slots telemetry patch: %s", exc
            )

    if kv_cache_manager_cls is not None and not kv_cache_manager_cls.__dict__.get(
        "_tpu_agentic_alloc_patch", False
    ):
        orig_allocate_slots = kv_cache_manager_cls.allocate_slots

        @functools.wraps(orig_allocate_slots)
        def _wrapped_allocate_slots(
            self: Any, request: Any, *args: Any, **kwargs: Any
        ) -> Any:
            bp = getattr(self, "block_pool", None)
            sid = _extract_request_session_id(request)
            if bp is not None:
                bp._agentic_allocating_session_id = sid
            try:
                res = orig_allocate_slots(self, request, *args, **kwargs)
            finally:
                if bp is not None:
                    bp._agentic_allocating_session_id = None
            if res is not None:
                try:
                    computed = (
                        args[2] if len(args) > 2 else kwargs.get("new_computed_blocks")
                    )
                    _tag_computed_blocks_for_session(bp, computed, sid)
                except Exception as exc:
                    logger.warning_once(
                        "Agentic telemetry allocate_slots tag failed: %s", exc
                    )
            return res

        kv_cache_manager_cls.allocate_slots = _wrapped_allocate_slots
        kv_cache_manager_cls._tpu_agentic_alloc_patch = True


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
    orig_abort_requests = getattr(output_processor_cls, "abort_requests", None)
    orig_process_outputs = output_processor_cls.process_outputs
    orig_do_tracing = output_processor_cls.do_tracing

    if orig_add_request is not None:

        @functools.wraps(orig_add_request)
        def _wrapped_add_request(
            self: Any,
            request: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            try:
                active_hdrs = _ACTIVE_TRACE_HEADERS.get()
                parent_req = args[1] if len(args) > 1 else kwargs.get("parent_req")
                req_idx = (
                    args[2] if len(args) > 2 else int(kwargs.get("request_index", 0))
                )
                if active_hdrs is not None and (
                    parent_req is None
                    or req_idx + 1 >= int(getattr(parent_req, "n", 1) or 1)
                ):
                    _ACTIVE_TRACE_HEADERS.set(None)
                if getattr(request, "trace_headers", None) is None and active_hdrs:
                    with contextlib.suppress(AttributeError):
                        request.trace_headers = dict(active_hdrs)
                if not getattr(request, "session_id", None):
                    sid = _extract_request_session_id(request, active_hdrs)
                    if sid:
                        with contextlib.suppress(AttributeError):
                            request.session_id = sid
            except Exception as exc:
                logger.warning_once(
                    "Agentic telemetry pre-add_request hook failed: %s", exc
                )

            res = orig_add_request(self, request, *args, **kwargs)

            try:
                req_states = getattr(self, "request_states", {})
                req_state = req_states.get(getattr(request, "request_id", None))
                if req_state is not None and getattr(request, "trace_headers", None):
                    with contextlib.suppress(AttributeError):
                        req_state.trace_headers = dict(request.trace_headers)
            except Exception as exc:
                logger.warning_once(
                    "Agentic telemetry post-add_request hook failed: %s", exc
                )
            return res

        output_processor_cls.add_request = _wrapped_add_request

    if orig_abort_requests is not None:

        @functools.wraps(orig_abort_requests)
        def _wrapped_abort_requests(
            self: Any,
            request_ids: Iterable[str],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            aborted = orig_abort_requests(self, request_ids, *args, **kwargs)
            try:
                svc = service or get_telemetry_service()
                for req_id in aborted or ():
                    svc.discard_request(req_id)
            except Exception as exc:
                logger.warning_once(
                    "Agentic telemetry abort_requests cleanup failed: %s", exc
                )
            return aborted

        output_processor_cls.abort_requests = _wrapped_abort_requests

    @functools.wraps(orig_process_outputs)
    def _wrapped_process_outputs(
        self: Any,
        engine_core_outputs: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        svc: AgenticTelemetryService | None = None
        candidates_without_tracing: list[tuple[str | None, Any, Any]] = []
        errored_req_ids: list[str] = []
        req_states: Mapping[str, Any] = {}
        try:
            svc = service or get_telemetry_service()
            svc.session_tracker.configure_from_tokenizer(
                getattr(self, "tokenizer", None)
            )
            req_states = getattr(self, "request_states", {})
            tracing_enabled = getattr(self, "tracing_enabled", False)

            for out in engine_core_outputs:
                req_id = getattr(out, "request_id", None)
                if getattr(out, "error", None) is not None:
                    if req_id:
                        errored_req_ids.append(req_id)
                    continue
                req_state = req_states.get(req_id)
                if req_state is None:
                    continue
                if getattr(req_state, "is_prefilling", False):
                    svc.on_prefill_output(req_state, out)
                if not tracing_enabled:
                    candidates_without_tracing.append((req_id, req_state, out))
        except Exception as exc:
            logger.warning_once(
                "Agentic telemetry pre-process_outputs hook failed: %s", exc
            )

        result = orig_process_outputs(
            self,
            engine_core_outputs,
            *args,
            **kwargs,
        )

        if svc is not None:
            try:
                for req_id in errored_req_ids:
                    svc.discard_request(req_id)
                post_states = getattr(self, "request_states", {})
                for req_id, req_state, out in candidates_without_tracing:
                    finished = getattr(out, "finish_reason", None) is not None or (
                        req_id is not None and req_id not in post_states
                    )
                    if not finished:
                        continue
                    if svc.span_log_path:
                        svc.build_span_attributes(req_state, out)
                    else:
                        svc.on_request_finished(req_state, out)
            except Exception as exc:
                logger.warning_once(
                    "Agentic telemetry post-process_outputs hook failed: %s", exc
                )

        return result

    @functools.wraps(orig_do_tracing)
    def _wrapped_do_tracing(
        self: Any,
        engine_core_output: Any,
        req_state: Any,
        iteration_stats: Any,
    ) -> None:
        extra_attrs: dict[str, Any] | None = None
        orig_instrument: Any = None
        try:
            svc = service or get_telemetry_service()
            extra_attrs = svc.build_span_attributes(req_state, engine_core_output)
            orig_instrument = getattr(output_processor_mod, "instrument_manual", None)
        except Exception as exc:
            logger.warning_once("Agentic telemetry do_tracing hook failed: %s", exc)

        if not extra_attrs or not callable(orig_instrument):
            orig_do_tracing(self, engine_core_output, req_state, iteration_stats)
            return

        @functools.wraps(orig_instrument)
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


def sync_block_size_from_config(vllm_config: Any) -> None:
    """Sync `cache_config.block_size` into process state and active service."""
    if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
        return
    global _DETECTED_VLLM_BLOCK_SIZE
    try:
        bs = getattr(getattr(vllm_config, "cache_config", None), "block_size", None)
        if isinstance(bs, int) and bs > 0:
            _DETECTED_VLLM_BLOCK_SIZE = bs
            if _SERVICE is not None and int(envs.VLLM_TPU_AGENTIC_BLOCK_SIZE or 0) <= 0:
                _SERVICE.session_tracker.block_size = bs
                _SERVICE._explicit_block_size = True
                _SERVICE._block_size_checked = True
    except Exception as exc:
        logger.warning_once("Agentic telemetry block_size sync hook failed: %s", exc)


def install_agentic_telemetry_patches() -> None:
    """Install or refresh telemetry hooks when `VLLM_TPU_AGENTIC_TELEMETRY=1`."""
    global _PATCHES_LOGGED
    if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
        return

    patch_trace_headers()
    register_tpu_padding_collector()
    try:
        patch_block_pool()
    except ImportError as exc:
        logger.warning_once(
            "Skipping BlockPool telemetry patch; BlockPool unavailable: %s", exc
        )
    try:
        patch_output_processor()
    except ImportError as exc:
        logger.warning_once(
            "Skipping OutputProcessor telemetry patch; OutputProcessor unavailable: %s",
            exc,
        )
    if not _PATCHES_LOGGED:
        _PATCHES_LOGGED = True
        logger.info("Installed vllm-torchtpu agentic telemetry patches")
