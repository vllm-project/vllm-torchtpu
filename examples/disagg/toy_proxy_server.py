# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import codecs
import hashlib
import itertools
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _PrefixAffinityRouter:
    """Prefix-hash sticky router across multiple prefill servers (e.g., 2P1D).

    Computes cumulative 8-byte blake2b digests over fixed-size prompt byte
    chunks and routes multi-turn conversation extensions to the prefill server
    that already holds their cached KV + GDN blocks, falling back to least
    in-flight load balancing on cold conversations or when a prefill server's
    in-flight queue exceeds ``min(inflight) + slack``.

    With ``max_sessions`` > 0 the router also models each server's KV cache as
    an LRU of that many conversations (a 397B prefill leg holds about 54
    contexts of 92k tokens without a host tier, about 240 with one). A cold
    conversation goes to the server with the fewest resident conversations,
    and a conversation whose modelled residency was evicted is routed as cold
    again, so the chunk owners never point at KV that the server has dropped.
    With ``saturated_inflight`` > 0 the slack check is skipped while every
    server has at least that many requests in flight: a cold placement there
    costs a full re-prefill, which is worse than queueing behind the owner.

    ``load_metric`` picks the per-server load that the slack check and the
    non-sticky placement compare: ``inflight`` counts requests from route()
    to done(), ``running`` counts only those between start_running() and
    done(). A decode leg uses ``running``: its batch is the requests it is
    decoding, while a request still in the prefill hop holds no decode slot.
    Ties fall back to the in-flight count, then to the number placed.
    """

    def __init__(
        self,
        n: int,
        chunk: int = 1024,
        min_frac: float = 0.5,
        slack: int = 8,
        max_sessions: int = 0,
        saturated_inflight: int = 0,
        load_metric: str = "inflight",
        quad_attn_coeff: float = 0.0,
        token_slack: float = 0.0,
    ) -> None:
        if load_metric not in ("inflight", "running"):
            raise ValueError(
                f"load_metric must be inflight or running, got {load_metric!r}"
            )
        self.n = n
        self.load_metric = load_metric
        self.chunk = max(64, chunk)
        self.min_frac = min_frac
        self.slack = slack
        self.max_sessions = max(0, max_sessions)
        self.saturated_inflight = max(0, saturated_inflight)
        self.quad_attn_coeff = max(0.0, float(quad_attn_coeff))
        self.token_slack = max(0.0, float(token_slack))
        self._lock = threading.Lock()
        self.owners: dict[bytes, int] = {}
        self.inflight = [0] * n
        self.inflight_eff_tokens: list[float] = [0.0] * n
        self._req_eff_tokens: dict[str, float] = {}
        self._anon_eff_tokens: list[list[float]] = [[] for _ in range(n)]
        self.running = [0] * n
        self.placed = [0] * n
        self.sticky = 0
        self.sticky_saturated = 0
        self.sticky_quad_saved = 0
        self.cold = 0
        # Residency model: digest -> conversation id, conversation id -> rank
        # holding its KV (None once evicted), and per rank an LRU of resident
        # conversation ids.
        self.session_of: dict[bytes, int] = {}
        self.session_rank: dict[int, int | None] = {}
        self.resident: list[OrderedDict[int, float]] = [OrderedDict() for _ in range(n)]
        self.evicted = 0
        self.rerouted = 0
        self._next_session = 0

    def effective_tokens(self, uncached_tokens: int, cached_tokens: int = 0) -> float:
        """Quadratic FLOPs-equivalent prefill token cost of ``uncached_tokens``
        behind ``cached_tokens`` of resident prefix."""
        u = max(0, int(uncached_tokens))
        if u <= 0:
            return 0.0
        p = max(0, int(cached_tokens))
        return float(u) + self.quad_attn_coeff * float(u) * (float(p) + float(u) / 2.0)

    @staticmethod
    def extract_text(body: dict[str, Any]) -> str:
        if msgs := body.get("messages"):
            parts: list[str] = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "")
                c = m.get("content", "")
                if isinstance(c, str):
                    parts.append(f"{role}:{c}")
                elif isinstance(c, list):
                    sub = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                    parts.append(f"{role}:{sub}")
            return "\n".join(parts)
        p = body.get("prompt", "")
        if isinstance(p, str):
            return p
        if isinstance(p, list):
            if p and isinstance(p[0], int):
                return ",".join(map(str, p))
            return "\n".join(str(x) for x in p)
        return ""

    def hashes(self, text: str) -> list[bytes]:
        raw = text.encode("utf-8", errors="ignore")
        out: list[bytes] = []
        h = hashlib.blake2b(digest_size=8)
        for i in range(0, len(raw) - self.chunk + 1, self.chunk):
            h.update(raw[i : i + self.chunk])
            out.append(h.digest())
        return out

    def route(
        self,
        hs: list[bytes],
        prompt_tokens: int = 0,
        req_id: str | None = None,
    ) -> tuple[int, bool]:
        with self._lock:
            best_rank: int | None = None
            matched = 0
            sid: int | None = None
            for i in range(len(hs) - 1, -1, -1):
                owner = self.owners.get(hs[i])
                if owner is not None:
                    best_rank = owner
                    matched = i + 1
                    sid = self.session_of.get(hs[i])
                    break
            known = best_rank is not None and hs and matched >= len(hs) * self.min_frac
            # The owner only counts while the conversation is still resident
            # in the modelled cache of that server.
            if (
                self.max_sessions > 0
                and known
                and (sid is None or self.session_rank.get(sid) != best_rank)
            ):
                known = False
                self.rerouted += 1
            load = self.running if self.load_metric == "running" else self.inflight
            lo = min(load)
            saturated = (
                self.saturated_inflight > 0
                and min(self.inflight) >= self.saturated_inflight
            )
            est_tokens = (
                prompt_tokens
                if prompt_tokens > 0
                else (max(1, int(len(hs) * self.chunk / 3.4)) if hs else 0)
            )
            use_quad = self.quad_attn_coeff > 0.0 and est_tokens > 0
            cold_eff = self.effective_tokens(est_tokens, 0) if use_quad else 0.0
            sticky_eff = cold_eff
            quad_prefers_sticky = False
            if known and use_quad and best_rank is not None:
                frac = min(1.0, matched / max(len(hs), 1))
                cached_toks = int(est_tokens * frac)
                uncached_toks = max(1, est_tokens - cached_toks)
                sticky_eff = self.effective_tokens(uncached_toks, cached_toks)
                min_eff = min(self.inflight_eff_tokens)
                quad_prefers_sticky = (
                    self.inflight_eff_tokens[best_rank] + sticky_eff
                    <= min_eff + cold_eff + self.token_slack
                )
            if known and (
                saturated
                or load[best_rank] <= lo + self.slack
                or quad_prefers_sticky
            ):
                rank = best_rank
                is_sticky = True
                req_eff = sticky_eff
                self.sticky += 1
                if saturated and load[best_rank] > lo + self.slack:
                    self.sticky_saturated += 1
                elif quad_prefers_sticky and load[best_rank] > lo + self.slack:
                    self.sticky_quad_saved += 1
            else:
                if self.max_sessions > 0:
                    rank = min(
                        range(self.n),
                        key=lambda r: (
                            len(self.resident[r]),
                            self.inflight_eff_tokens[r] if use_quad else 0.0,
                            load[r],
                            self.inflight[r],
                            self.placed[r],
                            r,
                        ),
                    )
                else:
                    rank = min(
                        range(self.n),
                        key=lambda r: (
                            self.inflight_eff_tokens[r] if use_quad else 0.0,
                            load[r],
                            self.inflight[r],
                            self.placed[r],
                            r,
                        ),
                    )
                is_sticky = False
                req_eff = cold_eff
                self.cold += 1
            if sid is None:
                sid = self._next_session
                self._next_session += 1
            for d in hs:
                self.owners[d] = rank
                self.session_of[d] = sid
            self._touch_session(sid, rank)
            self.inflight[rank] += 1
            if use_quad:
                self.inflight_eff_tokens[rank] += req_eff
                if req_id is not None:
                    self._req_eff_tokens[req_id] = req_eff
                else:
                    self._anon_eff_tokens[rank].append(req_eff)
            self.placed[rank] += 1
            total = self.sticky + self.cold
            if total % 64 == 0:
                print(
                    f"PERF PROXY prefix_router | total={total} "
                    f"sticky={self.sticky} ({100.0 * self.sticky / total:.1f}%) "
                    f"sticky_saturated={self.sticky_saturated} "
                    f"sticky_quad_saved={self.sticky_quad_saved} "
                    f"cold={self.cold} rerouted={self.rerouted} placed={self.placed} "
                    f"inflight={self.inflight} "
                    f"inflight_eff_tokens={[int(x) for x in self.inflight_eff_tokens]} "
                    f"unique_chunks={len(self.owners)} "
                    f"resident={[len(r) for r in self.resident]} "
                    f"evicted={self.evicted} sessions={self._next_session} "
                    f"running={self.running} load_metric={self.load_metric}",
                    flush=True,
                )
            return rank, is_sticky

    def _touch_session(self, sid: int, rank: int) -> None:
        """Moves the conversation to rank's LRU head, evicting past max_sessions."""
        prev = self.session_rank.get(sid)
        if prev is not None and prev != rank:
            self.resident[prev].pop(sid, None)
        self.session_rank[sid] = rank
        lru = self.resident[rank]
        lru[sid] = time.monotonic()
        lru.move_to_end(sid)
        if self.max_sessions > 0:
            while len(lru) > self.max_sessions:
                old_sid, _ = lru.popitem(last=False)
                self.session_rank[old_sid] = None
                self.evicted += 1

    def start_running(self, rank: int) -> None:
        """Counts a request routed to ``rank`` as running there."""
        with self._lock:
            self.running[rank] += 1

    def done(
        self, rank: int, was_running: bool = False, req_id: str | None = None
    ) -> None:
        with self._lock:
            self.inflight[rank] = max(0, self.inflight[rank] - 1)
            if req_id is not None and req_id in self._req_eff_tokens:
                eff = self._req_eff_tokens.pop(req_id)
                self.inflight_eff_tokens[rank] = max(
                    0.0, self.inflight_eff_tokens[rank] - eff
                )
            elif self._anon_eff_tokens[rank]:
                eff = self._anon_eff_tokens[rank].pop(0)
                self.inflight_eff_tokens[rank] = max(
                    0.0, self.inflight_eff_tokens[rank] - eff
                )
            if self.inflight[rank] == 0:
                self.inflight_eff_tokens[rank] = 0.0
                self._anon_eff_tokens[rank].clear()
            if was_running:
                self.running[rank] = max(0, self.running[rank] - 1)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "placed": list(self.placed),
                "inflight": list(self.inflight),
                "inflight_eff_tokens": list(self.inflight_eff_tokens),
                "running": list(self.running),
                "load_metric": self.load_metric,
                "sticky": self.sticky,
                "sticky_saturated": self.sticky_saturated,
                "sticky_quad_saved": self.sticky_quad_saved,
                "cold": self.cold,
                "rerouted": self.rerouted,
                "unique_chunks": len(self.owners),
                "sessions": self._next_session,
                "resident": [len(r) for r in self.resident],
                "evicted": self.evicted,
            }


class _DPRankRouter:
    """Prefix-affinity DP-rank routing inside data-parallel decode hosts.

    Opt-in with PROXY_DECODE_ROUTING=dp_affinity; any other value leaves this
    class unused and the proxy on its old behaviour.

    A DP>1 decode host (e.g. DP4xTP2) runs one vLLM engine per DP rank behind
    one API server. vLLM's internal balancer picks the rank by request count
    and knows nothing about prefixes, so a conversation's next turn lands on
    the rank that holds its previous turn's KV only 1 time in DP size, and the
    prefill re-sends KV the decode already holds. This router pins each decode
    request to a (host, rank) slot with vLLM's ``X-data-parallel-rank`` header:

    * Preferred slot: the slot that last served the longest known prefix of the
      prompt (cumulative blake2b digests over fixed-size prompt byte chunks, as
      _PrefixAffinityRouter), if at least ``min_frac`` of the chunks match.
    * Host: the preferred slot's host while that host's unfinished requests are
      at most ``host_slack`` above the host mean; otherwise the host with the
      fewest unfinished requests. Chosen before the prefill hop (the decode
      admission budget is per host).
    * Rank, chosen at the decode send with the real prompt length: the
      preferred rank while its unfinished prompt tokens are at most
      ``(1 + kv_slack_frac)`` x the host's per-rank mean and its unfinished
      request count at most ``count_slack`` above the per-rank mean. Otherwise
      the rank with the fewest unfinished prompt tokens among the ranks within
      ``fallback_count_slack`` of the fewest unfinished requests.

    The guard exists because the DP ranks step in lockstep: the rank holding
    the most resident KV sets every step's attention time, so a pure-affinity
    policy that piles long contexts onto one rank costs TPOT (DP-balance
    analysis, reports/dp_balance.md). "Unfinished" counts a request from its
    decode send (rank) or host choice (host) until its stream ends.
    """

    def __init__(
        self,
        n_hosts: int,
        ranks: int,
        chunk: int = 4096,
        min_frac: float = 0.5,
        kv_slack_frac: float = 0.10,
        count_slack: float = 1.0,
        fallback_count_slack: int = 1,
        host_slack: int = 8,
        max_owners: int = 2_000_000,
    ) -> None:
        if n_hosts < 1 or ranks < 1:
            raise ValueError(f"n_hosts and ranks must be >= 1, got {n_hosts}, {ranks}")
        self.n_hosts = n_hosts
        self.ranks = ranks
        self.chunk = max(64, int(chunk))
        self.min_frac = float(min_frac)
        self.kv_slack_frac = max(0.0, float(kv_slack_frac))
        self.count_slack = max(0.0, float(count_slack))
        self.fallback_count_slack = max(0, int(fallback_count_slack))
        self.host_slack = max(0, int(host_slack))
        self.max_owners = max(1, int(max_owners))
        self._lock = threading.Lock()
        # digest -> (host, rank) of the slot that last served that prefix
        self.owners: OrderedDict[bytes, tuple[int, int]] = OrderedDict()
        self.host_unfinished = [0] * n_hosts
        self.count = [[0] * ranks for _ in range(n_hosts)]
        self.tokens = [[0.0] * ranks for _ in range(n_hosts)]
        # req_id -> (host, rank or None, tokens)
        self._reqs: dict[str, tuple[int, int | None, float]] = {}
        self.routed = 0
        self.sticky = 0
        self.guard_kv = 0
        self.guard_count = 0
        self.other_host = 0
        self.cold = 0

    def hashes(self, text: str) -> list[bytes]:
        raw = text.encode("utf-8", errors="ignore")
        out: list[bytes] = []
        h = hashlib.blake2b(digest_size=8)
        for i in range(0, len(raw) - self.chunk + 1, self.chunk):
            h.update(raw[i : i + self.chunk])
            out.append(h.digest())
        return out

    def _owner(self, hs: list[bytes]) -> tuple[int, int] | None:
        for i in range(len(hs) - 1, -1, -1):
            slot = self.owners.get(hs[i])
            if slot is not None:
                if (i + 1) >= len(hs) * self.min_frac:
                    return slot
                return None
        return None

    def choose_host(self, hs: list[bytes], req_id: str) -> int:
        """Picks the decode host before the prefill hop and counts the request there."""
        with self._lock:
            owner = self._owner(hs)
            mean = sum(self.host_unfinished) / self.n_hosts
            if owner is not None and self.host_unfinished[owner[0]] <= mean + self.host_slack:
                host = owner[0]
            else:
                host = min(range(self.n_hosts), key=lambda h: (self.host_unfinished[h], h))
            self.host_unfinished[host] += 1
            self._reqs[req_id] = (host, None, 0.0)
            return host

    def choose_rank(
        self, host: int, hs: list[bytes], prompt_tokens: int, req_id: str
    ) -> tuple[int, str]:
        """Picks the DP rank on ``host`` at the decode send. Returns (rank, reason)."""
        with self._lock:
            prev = self._reqs.get(req_id)
            if prev is not None and prev[1] is not None:
                raise ValueError(f"choose_rank called twice for {req_id}")
            owner = self._owner(hs)
            cnt, tok = self.count[host], self.tokens[host]
            reason = "cold"
            rank: int | None = None
            if owner is not None and owner[0] != host:
                reason = "other_host"
            elif owner is not None:
                r = owner[1]
                mean_tok = sum(tok) / self.ranks
                mean_cnt = sum(cnt) / self.ranks
                if tok[r] > (1.0 + self.kv_slack_frac) * mean_tok + 1.0:
                    reason = "guard_kv"
                elif cnt[r] > mean_cnt + self.count_slack:
                    reason = "guard_count"
                else:
                    rank, reason = r, "sticky"
            if rank is None:
                lo = min(cnt)
                rank = min(
                    (r for r in range(self.ranks) if cnt[r] <= lo + self.fallback_count_slack),
                    key=lambda r: (tok[r], cnt[r], r),
                )
            ntok = float(max(0, int(prompt_tokens)))
            cnt[rank] += 1
            tok[rank] += ntok
            if prev is None:
                # choose_host was not called (single-host use): count the host here.
                self.host_unfinished[host] += 1
            self._reqs[req_id] = (host, rank, ntok)
            for d in hs:
                self.owners[d] = (host, rank)
                self.owners.move_to_end(d)
            while len(self.owners) > self.max_owners:
                self.owners.popitem(last=False)
            self.routed += 1
            if reason == "sticky":
                self.sticky += 1
            elif reason == "guard_kv":
                self.guard_kv += 1
            elif reason == "guard_count":
                self.guard_count += 1
            elif reason == "other_host":
                self.other_host += 1
            else:
                self.cold += 1
            return rank, reason

    def done(self, req_id: str) -> None:
        with self._lock:
            ent = self._reqs.pop(req_id, None)
            if ent is None:
                return
            host, rank, ntok = ent
            self.host_unfinished[host] = max(0, self.host_unfinished[host] - 1)
            if rank is not None:
                self.count[host][rank] = max(0, self.count[host][rank] - 1)
                self.tokens[host][rank] = max(0.0, self.tokens[host][rank] - ntok)

    def describe(self) -> str:
        return (
            f"dp_affinity ranks={self.ranks} hosts={self.n_hosts} chunk={self.chunk} "
            f"min_frac={self.min_frac} kv_slack_frac={self.kv_slack_frac} "
            f"count_slack={self.count_slack} fallback_count_slack={self.fallback_count_slack} "
            f"host_slack={self.host_slack} max_owners={self.max_owners}"
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "routed": self.routed,
                "sticky": self.sticky,
                "guard_kv": self.guard_kv,
                "guard_count": self.guard_count,
                "other_host": self.other_host,
                "cold": self.cold,
                "host_unfinished": list(self.host_unfinished),
                "count": [list(c) for c in self.count],
                "tokens": [[int(t) for t in ts] for ts in self.tokens],
                "owners": len(self.owners),
            }


def _make_dp_router(n_hosts: int) -> "_DPRankRouter | None":
    """The DP-rank router when PROXY_DECODE_ROUTING=dp_affinity, else None."""
    mode = os.getenv("PROXY_DECODE_ROUTING", "round_robin").strip().lower()
    if mode != "dp_affinity":
        return None
    return _DPRankRouter(
        n_hosts=n_hosts,
        ranks=_env_int("PROXY_DP_RANKS", 4),
        chunk=_env_int("PROXY_DP_PREFIX_CHUNK", 4096),
        min_frac=_env_float("PROXY_DP_MIN_FRAC", 0.5),
        kv_slack_frac=_env_float("PROXY_DP_KV_SLACK_FRAC", 0.10),
        count_slack=_env_float("PROXY_DP_COUNT_SLACK", 1.0),
        fallback_count_slack=_env_int("PROXY_DP_FALLBACK_COUNT_SLACK", 1),
        host_slack=_env_int("PROXY_DP_HOST_SLACK", 8),
        max_owners=_env_int("PROXY_DP_MAX_OWNERS", 2_000_000),
    )


def _decode_headers_with_rank(forwarded: dict, rank: int | None) -> dict:
    """Headers for the decode hop only: the client's own X-data-parallel-rank
    wins; otherwise the router's rank, if any. The prefill hop never gets a
    router rank (the prefill leg is not data parallel)."""
    if rank is None or "X-data-parallel-rank" in forwarded:
        return forwarded
    return {**forwarded, "X-data-parallel-rank": str(rank)}


class _AdmissionBudget:
    """Bounds the requests one backend leg holds, by count and by tokens.

    A request takes ``n_tokens`` of the budget when admitted and gives them
    back when released; ``adjust`` corrects an estimate once the real prompt
    length is known. Either limit at 0 is unlimited. A request larger than the
    whole token budget is admitted only when the leg is empty, so a single
    long context never deadlocks the leg.
    """

    def __init__(self, name: str, max_count: int, max_tokens: int) -> None:
        self.name = name
        self.max_count = max(0, max_count)
        self.max_tokens = max(0, max_tokens)
        self.count = 0
        self.tokens = 0
        self.waiting = 0
        self.admitted = 0
        self._cond = asyncio.Condition()

    @property
    def enabled(self) -> bool:
        return self.max_count > 0 or self.max_tokens > 0

    def _fits(self, n_tokens: int) -> bool:
        if self.max_count > 0 and self.count >= self.max_count:
            return False
        if self.max_tokens > 0 and self.count > 0:
            return self.tokens + n_tokens <= self.max_tokens
        return True

    async def acquire(self, n_tokens: int) -> float:
        """Waits for room, takes it, and returns the wait in milliseconds."""
        t0 = time.perf_counter()
        async with self._cond:
            self.waiting += 1
            try:
                await self._cond.wait_for(lambda: self._fits(n_tokens))
            finally:
                self.waiting -= 1
            self.count += 1
            self.tokens += n_tokens
            self.admitted += 1
        return (time.perf_counter() - t0) * 1000.0

    async def release(self, n_tokens: int) -> None:
        async with self._cond:
            self.count = max(0, self.count - 1)
            self.tokens = max(0, self.tokens - n_tokens)
            self._cond.notify_all()

    async def adjust(self, delta_tokens: int) -> None:
        if delta_tokens == 0:
            return
        async with self._cond:
            self.tokens = max(0, self.tokens + delta_tokens)
            if delta_tokens < 0:
                self._cond.notify_all()

    def describe(self) -> str:
        return (
            f"{self.name} inflight={self.count} tokens={self.tokens} "
            f"waiting={self.waiting} admitted={self.admitted}"
        )

    def stats(self) -> dict[str, Any]:
        return {
            "max_count": self.max_count,
            "max_tokens": self.max_tokens,
            "inflight": self.count,
            "tokens": self.tokens,
            "waiting": self.waiting,
            "admitted": self.admitted,
        }


def _estimate_prompt_tokens(text: str, bytes_per_token: float) -> int:
    """Pre-prefill prompt length estimate from the request bytes."""
    raw = len(text.encode("utf-8", errors="ignore"))
    return max(1, int(raw / max(0.5, bytes_per_token)))


def _expected_output_tokens(req_data: dict, default: int) -> int:
    for key in ("max_completion_tokens", "max_tokens"):
        v = req_data.get(key)
        if isinstance(v, int) and v > 0:
            return min(v, default) if default > 0 else v
    return max(1, default)


def _prompt_tokens_from_prefill(
    response_json: dict, kv_transfer_params: Any
) -> int | None:
    """The real prompt length once prefill has run: usage first, then the
    connector's num_tokens (the prompt minus the last token)."""
    usage = response_json.get("usage") if isinstance(response_json, dict) else None
    if isinstance(usage, dict):
        v = usage.get("prompt_tokens")
        if isinstance(v, int) and v > 0:
            return v
    if isinstance(kv_transfer_params, dict):
        v = kv_transfer_params.get("num_tokens")
        if isinstance(v, int) and v > 0:
            return v + 1
    return None


class _ProxyLatencyTracker:
    """Small running latency summary for the proxy path."""

    def __init__(self, log_interval_s: float):
        self._log_interval_s = log_interval_s
        self._lock = threading.Lock()
        self._stats: dict[str, list[float]] = {}
        self._next_log = time.monotonic() + log_interval_s

    def record(self, phase: str, ms: float) -> None:
        if self._log_interval_s <= 0:
            return
        msg: str | None = None
        with self._lock:
            stat = self._stats.setdefault(phase, [0.0, 0.0, ms, ms])
            stat[0] += 1.0
            stat[1] += ms
            stat[2] = min(stat[2], ms)
            stat[3] = max(stat[3], ms)

            now = time.monotonic()
            if now >= self._next_log:
                self._next_log = now + self._log_interval_s
                parts = []
                for key in sorted(self._stats):
                    n, total, min_ms, max_ms = self._stats[key]
                    parts.append(
                        f"{key}: n={int(n)} avg={total / n:.2f}ms "
                        f"min={min_ms:.2f}ms max={max_ms:.2f}ms"
                    )
                msg = f"PERF PROXY latency summary | {' | '.join(parts)}"
        if msg is not None:
            print(msg, flush=True)
            logger.info(msg)


def _proxy_latency_interval() -> float:
    val = os.getenv("PROXY_LATENCY_LOG_INTERVAL", "30")
    try:
        return float(val)
    except ValueError:
        return 30.0


_PROXY_LATENCY = _ProxyLatencyTracker(_proxy_latency_interval())


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# PROXY_PREFILL_ROUTING=prefix_hash_sticky is prefix_hash with the saturation
# override on by default: PROXY_PREFIX_HASH_SATURATED_INFLIGHT defaults to 1
# instead of 0, and an explicit value still wins. The alias exists because the
# DP4xTP2 recipe lineage always passes PROXY_PREFILL_ROUTING but drops proxy
# variables that are not on its allowlist, PROXY_PREFIX_HASH_SATURATED_INFLIGHT
# among them. Opt-in: prefix_hash and every other value keep the default 0.
_PREFIX_HASH_MODES = ("prefix_hash", "prefix_hash_sticky")


def _prefill_routing_mode() -> str:
    return os.getenv("PROXY_PREFILL_ROUTING", "prefix_hash").strip().lower()


def _prefix_hash_saturated_inflight() -> int:
    default = 1 if _prefill_routing_mode() == "prefix_hash_sticky" else 0
    return _env_int("PROXY_PREFIX_HASH_SATURATED_INFLIGHT", default)


# Retries here are deliberately limited to failures that provably occur before
# any request byte is written to the socket, so the backend holds no state for
# the request and a replay is indistinguishable from a first attempt:
#   ConnectError   - the TCP/TLS connection was never established.
#   ConnectTimeout - timed out establishing that connection.
#   PoolTimeout    - timed out waiting for a free slot in the local pool, so we
#                    never even got a socket to write to.
# ConnectTimeout and PoolTimeout subclass TimeoutException, *not* ConnectError,
# so they have to be named explicitly.
#
# Anything that fails *after* the request hits the wire (ReadError, WriteError,
# ReadTimeout, 5xx) must NOT be retried from this proxy: vLLM derives its
# internal request id straight from the X-Request-Id header we send, so a
# replay is seen by the engine as the same request arriving twice. That trips
# `assert existing.streaming_queue is not None, "duplicate request id"` in
# Scheduler.add_request, which raises inside the engine loop and takes down the
# whole EngineCore rather than failing the single request. On the decode hop it
# additionally collides with the Raiden Stage-3 KV registration keyed by the
# same id. Recovering those cases requires re-running prefill under a fresh
# request id to mint a new KV uuid, which has to happen at the request-flow
# level (see _handle_completions), not inside a per-hop wrapper.
_PRE_SEND_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_MAX_ATTEMPTS = max(1, _env_int("PROXY_MAX_ATTEMPTS", 5))
_RETRY_BASE_DELAY_S = _env_float("PROXY_RETRY_BASE_DELAY_S", 0.5)


def _log_unsafe_transport_failure(
    phase: str, request_id: str, exc: Exception, detail: str = ""
) -> None:
    """Flag a failure that cannot be retried safely at this layer."""
    msg = (
        f"[UNSAFE-{phase}] req_id={request_id} failed after the request "
        f"was already sent: {exc!r}. Not retrying: the server may have "
        f"admitted this request, and replaying it with the same "
        f"X-Request-Id would collide with the in-flight vLLM request id "
        f"(and the Raiden KV registration on the decode hop). Recovery "
        f"needs a fresh request id + new KV uuid and must be handled at "
        f"the request-flow level, not in this per-hop retry wrapper."
    )
    if detail:
        msg = f"{msg} {detail}"
    print(msg, flush=True)
    logger.error(msg)


async def _backoff_before_retry(
    attempt: int, phase: str, request_id: str, exc: Exception
) -> None:
    delay = _RETRY_BASE_DELAY_S * (2**attempt)
    msg = (
        f"[RETRY-{phase}] attempt {attempt + 1}/{_MAX_ATTEMPTS} "
        f"failed for req_id={request_id}: {exc!r}, "
        f"retrying in {delay:.1f}s"
    )
    print(msg, flush=True)
    logger.warning(msg)
    await asyncio.sleep(delay)


async def _post_with_retries(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    headers: dict,
    request_id: str,
    phase: str,
):
    """POST to a backend, retrying only connection-establishment failures."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = await client_info["client"].post(
                endpoint, json=req_data, headers=headers
            )
            response.raise_for_status()
            return response
        except _PRE_SEND_ERRORS as e:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            await _backoff_before_retry(attempt, phase, request_id, e)
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                _log_unsafe_transport_failure(phase, request_id, e)
            raise
        except httpx.TransportError as e:
            _log_unsafe_transport_failure(phase, request_id, e)
            raise


def _render_endpoint_for_api(api: str) -> str | None:
    # Upstream vLLM does not expose a /render endpoint; tokenization occurs
    # directly within the prefill engine.
    return None


def _token_ids_from_completion_render(rendered: Any) -> list[int] | list[list[int]]:
    if not isinstance(rendered, list) or not rendered:
        raise ValueError("Completion render response must be a non-empty list")

    prompts: list[list[int]] = []
    for item in rendered:
        if not isinstance(item, dict):
            raise ValueError("Completion render item must be an object")
        token_ids = item.get("token_ids")
        if not isinstance(token_ids, list) or not all(
            type(token_id) is int and token_id >= 0 for token_id in token_ids
        ):
            raise ValueError("Completion render item must contain token_ids")
        prompts.append(token_ids)

    return prompts[0] if len(prompts) == 1 else prompts


def _replace_prompt_with_rendered_token_ids(req_data: dict, rendered: Any) -> dict:
    req_data = req_data.copy()
    req_data["prompt"] = _token_ids_from_completion_render(rendered)
    req_data.pop("prompt_embeds", None)
    return req_data


def _json_escape_non_ascii(text: str) -> bytes:
    """Encode text as ASCII bytes while preserving JSON string semantics."""
    if text.isascii():
        return text.encode("ascii")

    parts = []
    for char in text:
        codepoint = ord(char)
        if codepoint < 128:
            parts.append(char)
        elif codepoint <= 0xFFFF:
            parts.append(f"\\u{codepoint:04x}")
        else:
            codepoint -= 0x10000
            high_surrogate = 0xD800 + (codepoint >> 10)
            low_surrogate = 0xDC00 + (codepoint & 0x3FF)
            parts.append(f"\\u{high_surrogate:04x}\\u{low_surrogate:04x}")
    return "".join(parts).encode("ascii")


class _AsciiSafeStreamEncoder:
    """Converts UTF-8 stream chunks to ASCII-only JSON-compatible chunks."""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def encode(self, chunk: bytes = b"", *, final: bool = False) -> bytes:
        text = self._decoder.decode(chunk, final=final)
        if not text:
            return b""
        return _json_escape_non_ascii(text)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # keepalive_expiry must stay *below* the backend's keep-alive timeout,
    # otherwise the pool hands out sockets the server has already reaped and
    # the next request dies with ReadError/RemoteProtocolError -- which we
    # cannot retry safely (see _PRE_SEND_ERRORS above). vLLM runs uvicorn with
    # timeout_keep_alive=VLLM_HTTP_TIMEOUT_KEEP_ALIVE, default 5s, and httpx's
    # own default expiry is also 5.0, so out of the box the two race exactly.
    # 4.0 buys a second of margin; raise VLLM_HTTP_TIMEOUT_KEEP_ALIVE on the
    # backends if you want longer-lived connections.
    keepalive_expiry = _env_float("PROXY_KEEPALIVE_EXPIRY_S", 4.0)
    limits = httpx.Limits(
        max_connections=None,
        max_keepalive_connections=None,
        keepalive_expiry=keepalive_expiry,
    )

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None, base_url=prefiller_base_url, limits=limits
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None, base_url=decoder_base_url, limits=limits
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))
    routing_mode = _prefill_routing_mode()
    # Residency-aware prefill routing: PROXY_PREFILL_MAX_SESSIONS conversations
    # per prefill leg (0 = off; ~54 for a 397B leg without a host tier, ~240
    # with one) and PROXY_PREFIX_HASH_SATURATED_INFLIGHT, the per-leg in-flight
    # count above which the sticky slack no longer applies (0 = off; 1 by
    # default under PROXY_PREFILL_ROUTING=prefix_hash_sticky).
    if len(app.state.prefill_clients) > 1 and routing_mode in _PREFIX_HASH_MODES:
        app.state.prefill_router = _PrefixAffinityRouter(
            n=len(app.state.prefill_clients),
            chunk=_env_int("PROXY_PREFIX_HASH_CHUNK", 1024),
            min_frac=_env_float("PROXY_PREFIX_HASH_MIN_FRAC", 0.5),
            slack=_env_int("PROXY_PREFIX_HASH_SLACK", 8),
            max_sessions=_env_int("PROXY_PREFILL_MAX_SESSIONS", 0),
            saturated_inflight=_prefix_hash_saturated_inflight(),
            quad_attn_coeff=_env_float("PROXY_PREFILL_QUAD_ATTN_COEFF", 1.23e-5),
            token_slack=_env_float("PROXY_PREFIX_HASH_TOKEN_SLACK", 32768.0),
        )
    else:
        app.state.prefill_router = None
    decode_routing = os.getenv("PROXY_DECODE_ROUTING", "round_robin").strip().lower()
    # Decode prefix routing keeps a conversation on its leg while that leg runs
    # at most PROXY_DECODE_HASH_SLACK more requests than the least-loaded leg,
    # and otherwise picks the leg with the fewest running requests.
    # PROXY_DECODE_LOAD_METRIC=inflight counts requests from routing to stream
    # end instead (prefill hop included), with a default slack of 8.
    decode_load_metric = (
        os.getenv("PROXY_DECODE_LOAD_METRIC", "running").strip().lower()
    )
    if decode_load_metric not in ("running", "inflight"):
        raise ValueError(
            "PROXY_DECODE_LOAD_METRIC must be running or inflight, "
            f"got {decode_load_metric!r}"
        )
    decode_slack_default = 2 if decode_load_metric == "running" else 8
    if len(app.state.decode_clients) > 1 and decode_routing == "prefix_hash":
        app.state.decode_router = _PrefixAffinityRouter(
            n=len(app.state.decode_clients),
            chunk=_env_int("PROXY_PREFIX_HASH_CHUNK", 1024),
            min_frac=_env_float("PROXY_PREFIX_HASH_MIN_FRAC", 0.5),
            slack=_env_int("PROXY_DECODE_HASH_SLACK", decode_slack_default),
            load_metric=decode_load_metric,
        )
    else:
        app.state.decode_router = None
    # PROXY_DECODE_ROUTING=dp_affinity: pin each decode request to a DP rank of
    # its decode host (X-data-parallel-rank), prefix-affine with a KV-aware load
    # guard; it also replaces the host round robin (host affinity with a slack).
    app.state.dp_router = _make_dp_router(len(app.state.decode_clients))
    if app.state.dp_router is not None:
        print(f"Decode DP-rank routing: {app.state.dp_router.describe()}", flush=True)
    # Admission per decode leg, so the contexts a decode leg holds (running
    # plus arriving KV) stay within its KV cache; above the cap, requests wait
    # here holding no KV anywhere. Without it a burst fills the decode KV,
    # vLLM preempts and recomputes whole ~90k-token contexts on the decode
    # leg, and E2E explodes. PROXY_DECODE_MAX_INFLIGHT counts requests,
    # PROXY_DECODE_MAX_INFLIGHT_TOKENS counts prompt plus expected output
    # tokens (contexts differ 3x in length, so the token budget tracks the KV
    # the leg actually holds); both 0 = unlimited, both set = both apply.
    # PROXY_DECODE_ADMIT_POINT=before_prefill takes the budget before the
    # prefill hop with a byte-based prompt estimate (corrected after prefill);
    # after_prefill takes it between the prefill hop and the decode hop with
    # the real prompt length, so only decode-side work counts against it.
    decode_cap = _env_int("PROXY_DECODE_MAX_INFLIGHT", 0)
    decode_cap_tokens = _env_int("PROXY_DECODE_MAX_INFLIGHT_TOKENS", 0)
    admit_point = (
        os.getenv("PROXY_DECODE_ADMIT_POINT", "before_prefill").strip().lower()
    )
    if admit_point not in ("before_prefill", "after_prefill"):
        raise ValueError(
            "PROXY_DECODE_ADMIT_POINT must be before_prefill or after_prefill, "
            f"got {admit_point!r}"
        )
    app.state.decode_admit_point = admit_point
    app.state.decode_budgets = [
        _AdmissionBudget(f"decode[{i}]", decode_cap, decode_cap_tokens)
        for i in range(len(app.state.decode_clients))
    ]
    # Bound on the requests a prefill leg has in prefill or has finished
    # prefilling that no decode leg has pulled yet (the producer keeps their KV
    # pinned through the Stage-3 request registration until the pull or the
    # TPU_P2P_WAIT_PULL_TIMEOUT expiry). Taken before the prefill hop with
    # the estimate, corrected after prefill, released at the decode's first
    # chunk; a request that waits for decode admission therefore holds a
    # pending slot, and once the pending set is full the queue forms here,
    # before prefill, instead of in the prefill leg's KV pool.
    pending_cap = _env_int("PROXY_PREFILL_PENDING_MAX", 0)
    pending_cap_tokens = _env_int("PROXY_PREFILL_PENDING_MAX_TOKENS", 0)
    app.state.prefill_pending = [
        _AdmissionBudget(f"prefill[{i}]", pending_cap, pending_cap_tokens)
        for i in range(len(app.state.prefill_clients))
    ]
    # Prompt bytes per token of the AgentX trace: bytes / 4.0 came out at 0.84x
    # the real prompt_tokens (p10 to p90), bytes / 3.4 matches.
    app.state.bytes_per_token = _env_float("PROXY_TOKEN_BYTES_EST", 3.4)
    app.state.output_tokens_est = _env_int("PROXY_DECODE_OUTPUT_TOKENS_EST", 1024)

    saturated_inflight = _prefix_hash_saturated_inflight()
    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"(routing={routing_mode if app.state.prefill_router else 'round_robin'}"
        f" max_sessions={_env_int('PROXY_PREFILL_MAX_SESSIONS', 0) or 'unlimited'}"
        f" saturated_inflight={saturated_inflight or 'off'}"
        f" pending_max={pending_cap or 'unlimited'}"
        f" pending_max_tokens={pending_cap_tokens or 'unlimited'}) "
        f"and {len(app.state.decode_clients)} decode clients.",
        flush=True,
    )
    decode_router = app.state.decode_router
    decode_route_desc = (
        f"prefix_hash load_metric={decode_router.load_metric} "
        f"slack={decode_router.slack}"
        if decode_router
        else "round_robin"
    )
    print(
        f"Decode routing={decode_route_desc} "
        f"max_inflight_per_decode={decode_cap or 'unlimited'} "
        f"max_inflight_tokens_per_decode={decode_cap_tokens or 'unlimited'} "
        f"admit_point={admit_point}",
        flush=True,
    )

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def _llmd_proxy_path() -> str | None:
    """The final lineage's llm-d proxy, toy_proxy_server_llmd.py: beside this file,
    or in the tree locations the agentx recipes use (they run a copy of this file
    from /tmp, so the directory of __file__ alone is not enough)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (
        here,
        "/workspace/vllm-torchtpu/examples/disagg",
        "/root/torchtpu-vllm/examples/disagg",
    ):
        p = os.path.join(d, "toy_proxy_server_llmd.py")
        if os.path.isfile(p):
            return p
    return None


def _wants_llmd_proxy() -> bool:
    """PROXY_PREFILL_ROUTING=llmd or PROXY_DECODE_ROUTING=llmd selects the llm-d
    proxy (2-tier cache affinity + token-load TTFT router on both legs). This
    file has no llmd mode of its own: before the merge it fell back to round
    robin for that value, which is identical to prefix_hash at one pod per leg."""
    return "llmd" in (
        os.getenv("PROXY_PREFILL_ROUTING", "").strip().lower(),
        os.getenv("PROXY_DECODE_ROUTING", "").strip().lower(),
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # For prefiller instances
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8400]
    )

    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[9400]
    )

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_prefill(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    forwarded: dict | None = None,
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    # Must overwrite these for prefill workers.
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **(forwarded or {}),
    }

    return await _post_with_retries(
        client_info, endpoint, req_data, headers, request_id, phase="PREFILL"
    )


async def render_completion_prompt(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    forwarded: dict | None = None,
):
    """
    Render/tokenize a completion request once so downstream P/D requests can
    use prompt token IDs instead of retokenizing raw text independently.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **(forwarded or {}),
    }

    return await _post_with_retries(
        client_info, endpoint, req_data, headers, request_id, phase="RENDER"
    )


async def stream_from_decode(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    forwarded: dict | None = None,
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **(forwarded or {}),
    }

    for attempt in range(_MAX_ATTEMPTS):
        yielded_any = False
        try:
            async with client_info["client"].stream(
                "POST", endpoint, json=req_data, headers=headers
            ) as response:
                if response.status_code >= 400:
                    # Streaming responses are not read on entry, so pull the
                    # body first or the raised error carries no vLLM detail.
                    await response.aread()
                response.raise_for_status()
                encoder = _AsciiSafeStreamEncoder()
                async for chunk in response.aiter_bytes():
                    chunk = encoder.encode(chunk)
                    if chunk:
                        yielded_any = True
                        yield chunk
                chunk = encoder.encode(final=True)
                if chunk:
                    yielded_any = True
                    yield chunk
            return
        except _PRE_SEND_ERRORS as e:
            # Never reached the decode server, so the KV handle minted by
            # prefill is still parked and unclaimed: safe to re-send as-is.
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            await _backoff_before_retry(attempt, "DECODE", request_id, e)
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if yielded_any:
                detail = (
                    "Stream had already emitted bytes to the client, "
                    "so the response is truncated."
                )
            else:
                detail = (
                    "Failed before the first byte; the client will "
                    "still see HTTP 200 followed by an empty/short "
                    "body because StreamingResponse has already "
                    "flushed headers."
                )
            _log_unsafe_transport_failure("DECODE", request_id, e, detail)
            raise


async def _handle_completions(api: str, request: Request):
    try:
        t_request_recv_perf = time.perf_counter()
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        # vLLM sends a request carrying X-data-parallel-rank to that DP rank of
        # the engine. Passed to both hops, so that a client can hold a request
        # to one prefill replica and one decode replica.
        forwarded = {}
        if (dp_rank := request.headers.get("X-data-parallel-rank")) is not None:
            forwarded["X-data-parallel-rank"] = dp_rank
        print(
            f"PERF PROXY req_received req_id={request_id} ts={time.time():.6f}",
            flush=True,
        )

        router: _PrefixAffinityRouter | None = getattr(
            request.app.state, "prefill_router", None
        )
        decode_router: _PrefixAffinityRouter | None = getattr(
            request.app.state, "decode_router", None
        )
        prefill_idx: int | None = None
        prefill_sticky = False
        decode_idx: int | None = None
        hs: list[bytes] = []
        if router is not None or decode_router is not None:
            hasher = router if router is not None else decode_router
            hs = hasher.hashes(hasher.extract_text(req_data))
        dp_router: _DPRankRouter | None = getattr(request.app.state, "dp_router", None)
        dp_hs: list[bytes] = []
        if dp_router is not None:
            dp_hs = dp_router.hashes(_PrefixAffinityRouter.extract_text(req_data))
        # The decode leg is chosen before the prefill hop so that the
        # admission slot is taken while the request holds no KV. A
        # conversation keeps its decode leg, so a later turn hits that leg's
        # own prefix cache and pulls only the new tokens' KV from the prefill.
        if dp_router is not None:
            decode_idx = dp_router.choose_host(dp_hs, request_id)
        elif decode_router is not None:
            decode_idx, _ = decode_router.route(hs)
        else:
            decode_idx = next(request.app.state.decode_iterator)
        decode_client_info = request.app.state.decode_clients[decode_idx]
        state = request.app.state
        decode_budget: _AdmissionBudget = state.decode_budgets[decode_idx]
        admit_point: str = state.decode_admit_point
        prompt_est = _estimate_prompt_tokens(
            _PrefixAffinityRouter.extract_text(req_data), state.bytes_per_token
        )
        output_est = _expected_output_tokens(req_data, state.output_tokens_est)
        # The budget a request holds, in tokens, on each leg; corrected once the
        # prefill response reports the real prompt length.
        decode_tokens = prompt_est + output_est
        pending_tokens = prompt_est
        decode_held = False
        pending_held = False
        pending_budget: _AdmissionBudget | None = None
        decode_released = False
        decode_running = False
        pending_released = False
        t_prefill_recv_perf: float | None = None

        async def release_decode() -> None:
            nonlocal decode_released
            if decode_released:
                return
            decode_released = True
            if decode_held:
                await decode_budget.release(decode_tokens)
            if decode_router is not None:
                decode_router.done(decode_idx, was_running=decode_running)
            if dp_router is not None:
                dp_router.done(request_id)

        async def release_pending(reason: str) -> None:
            nonlocal pending_released
            if pending_released:
                return
            pending_released = True
            if pending_held and pending_budget is not None:
                await pending_budget.release(pending_tokens)
                if t_prefill_recv_perf is not None:
                    pending_ms = (time.perf_counter() - t_prefill_recv_perf) * 1000.0
                    _PROXY_LATENCY.record("prefill_pending", pending_ms)
                    print(
                        f"PERF PROXY prefill_pending_done req_id={request_id} "
                        f"prefill_id={prefill_idx} reason={reason} "
                        f"pending_ms={pending_ms:.2f} ts={time.time():.6f}",
                        flush=True,
                    )

        async def admit_decode(point: str) -> None:
            nonlocal decode_held
            if not decode_budget.enabled:
                return
            wait_ms = await decode_budget.acquire(decode_tokens)
            decode_held = True
            _PROXY_LATENCY.record("decode_admit", wait_ms)
            print(
                f"PERF PROXY decode_admit req_id={request_id} "
                f"decode_id={decode_client_info.get('id', decode_idx)} "
                f"point={point} tokens={decode_tokens} "
                f"wait_ms={wait_ms:.2f} ts={time.time():.6f}",
                flush=True,
            )
            if decode_budget.admitted % 64 == 0:
                print(
                    "PERF PROXY admission | "
                    + " | ".join(b.describe() for b in state.decode_budgets)
                    + " | "
                    + " | ".join(b.describe() for b in state.prefill_pending),
                    flush=True,
                )

        try:
            # Inside the try: a client that disconnects while this waits for
            # admission still returns its decode router slot.
            if admit_point == "before_prefill":
                await admit_decode(admit_point)

            if router is not None:
                prefill_idx, prefill_sticky = router.route(
                    hs, prompt_tokens=prompt_est, req_id=request_id
                )
                prefill_client_info = request.app.state.prefill_clients[prefill_idx]
            else:
                prefill_client_info = get_next_client(request.app, "prefill")
                prefill_idx = int(prefill_client_info.get("id", 0))
            pending_budget = state.prefill_pending[prefill_idx]
            if pending_budget.enabled:
                wait_ms = await pending_budget.acquire(pending_tokens)
                pending_held = True
                _PROXY_LATENCY.record("prefill_pending_admit", wait_ms)
                print(
                    f"PERF PROXY prefill_pending_admit req_id={request_id} "
                    f"prefill_id={prefill_idx} tokens={pending_tokens} "
                    f"wait_ms={wait_ms:.2f} ts={time.time():.6f}",
                    flush=True,
                )

            try:
                render_endpoint = _render_endpoint_for_api(api)
                if render_endpoint is not None:
                    render_response = await render_completion_prompt(
                        prefill_client_info,
                        render_endpoint,
                        req_data,
                        request_id,
                        forwarded,
                    )
                    req_data = _replace_prompt_with_rendered_token_ids(
                        req_data, render_response.json()
                    )

                # Send request to prefill service
                t_prefill_send = time.time()
                t_prefill_send_perf = time.perf_counter()
                print(
                    f"PERF PROXY prefill_send req_id={request_id} "
                    f"prefill_id={prefill_client_info.get('id', 0)} "
                    f"sticky={int(prefill_sticky)} "
                    f"ts={t_prefill_send:.6f}",
                    flush=True,
                )
                response = await send_request_to_prefill(
                    prefill_client_info, api, req_data, request_id, forwarded
                )
            finally:
                if router is not None and prefill_idx is not None:
                    router.done(prefill_idx, req_id=request_id)

            t_prefill_recv = time.time()
            t_prefill_recv_perf = time.perf_counter()
            prefill_ms = (t_prefill_recv_perf - t_prefill_send_perf) * 1000.0
            _PROXY_LATENCY.record("prefill", prefill_ms)
            _PROXY_LATENCY.record(
                "prefill_from_req", (t_prefill_recv_perf - t_request_recv_perf) * 1000.0
            )

            # Extract the needed fields
            response_json = response.json()
            kv_transfer_params = response_json.get("kv_transfer_params", {})
            kv_uuid = kv_transfer_params.get("uuid") if kv_transfer_params else None
            kv_uuid_log = f" kv_uuid={kv_uuid}" if kv_uuid is not None else ""
            prompt_tokens = _prompt_tokens_from_prefill(
                response_json, kv_transfer_params
            )
            print(
                f"PERF PROXY prefill_recv req_id={request_id} ts={t_prefill_recv:.6f} "
                f"dur_ms={prefill_ms:.2f} prompt_tokens={prompt_tokens} "
                f"prompt_est={prompt_est}{kv_uuid_log}",
                flush=True,
            )
            if prompt_tokens is not None:
                if pending_held and pending_budget is not None:
                    await pending_budget.adjust(prompt_tokens - pending_tokens)
                pending_tokens = prompt_tokens
                if decode_held:
                    await decode_budget.adjust(prompt_tokens - prompt_est)
                decode_tokens = prompt_tokens + output_est

            if kv_transfer_params:
                req_data["kv_transfer_params"] = kv_transfer_params

            if admit_point == "after_prefill":
                await admit_decode(admit_point)

            logger.debug("Using %s %s", prefill_client_info, decode_client_info)

            # Stream response from decode service
            async def generate_stream():
                nonlocal decode_running
                try:
                    if decode_router is not None:
                        decode_router.start_running(decode_idx)
                        decode_running = True
                    decode_forwarded = forwarded
                    if dp_router is not None and "X-data-parallel-rank" not in forwarded:
                        dp_rank, dp_reason = dp_router.choose_rank(
                            decode_idx,
                            dp_hs,
                            prompt_tokens if prompt_tokens is not None else prompt_est,
                            request_id,
                        )
                        decode_forwarded = _decode_headers_with_rank(forwarded, dp_rank)
                        print(
                            f"PERF PROXY dp_route req_id={request_id} "
                            f"decode_id={decode_idx} rank={dp_rank} reason={dp_reason} "
                            f"tokens={prompt_tokens if prompt_tokens is not None else prompt_est}",
                            flush=True,
                        )
                        if dp_router.routed % 256 == 0:
                            print(f"PERF PROXY dp_router | {dp_router.stats()}", flush=True)
                    t_decode_send = time.time()
                    t_decode_send_perf = time.perf_counter()
                    print(
                        f"PERF PROXY decode_send req_id={request_id} "
                        f"ts={t_decode_send:.6f}{kv_uuid_log}",
                        flush=True,
                    )
                    first = True
                    async for chunk in stream_from_decode(
                        decode_client_info, api, req_data, request_id, decode_forwarded
                    ):
                        if first:
                            t_first = time.time()
                            t_first_perf = time.perf_counter()
                            decode_first_ms = (
                                t_first_perf - t_decode_send_perf
                            ) * 1000.0
                            ttft_ms = (t_first_perf - t_request_recv_perf) * 1000.0
                            _PROXY_LATENCY.record("decode_first", decode_first_ms)
                            _PROXY_LATENCY.record("ttft", ttft_ms)
                            print(
                                f"PERF PROXY first_chunk req_id={request_id} "
                                f"ts={t_first:.6f} "
                                f"dur_from_decode_send_ms={decode_first_ms:.2f} "
                                f"ttft_ms={ttft_ms:.2f}{kv_uuid_log}",
                                flush=True,
                            )
                            first = False
                            await release_pending("first_chunk")
                        yield chunk
                finally:
                    await release_pending("stream_end")
                    await release_decode()

            # text/event-stream when the caller asked for a stream, which is what
            # the decode instance is actually sending back. Declared as
            # application/json, a client parses each SSE frame - "data: {...}" prefix
            # and all - as one JSON document, and every request fails; aiperf's
            # agentic replay does exactly that, while the non-streaming callers this
            # proxy had until now (check_completions.py, lm_eval) never noticed.
            media_type = (
                "text/event-stream" if req_data.get("stream") else "application/json"
            )
            return StreamingResponse(generate_stream(), media_type=media_type)
        except BaseException:
            await release_pending("error")
            await release_decode()
            raise

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


@app.get("/v1/models")
async def models():
    """The decode instance's model list, which is the pair's.

    Part of the OpenAI surface this proxy presents: a client resolves the served
    model name here, and the ModelCard's max_model_len is how a benchmark client
    learns the window it is replaying against. Both instances serve the same
    model; decode is the one that finishes a request.
    """
    client = app.state.decode_clients[0]["client"]
    response = await client.get("/v1/models")
    return JSONResponse(status_code=response.status_code, content=response.json())


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    router: _PrefixAffinityRouter | None = getattr(app.state, "prefill_router", None)
    res: dict[str, Any] = {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }
    if router is not None:
        res["prefill_router"] = router.stats()
    decode_router: _PrefixAffinityRouter | None = getattr(
        app.state, "decode_router", None
    )
    if decode_router is not None:
        res["decode_router"] = decode_router.stats()
    dp_router = getattr(app.state, "dp_router", None)
    if dp_router is not None:
        res["dp_router"] = dp_router.stats()
    budgets = getattr(app.state, "decode_budgets", None)
    if budgets:
        res["decode_admission"] = [b.stats() for b in budgets]
    pending = getattr(app.state, "prefill_pending", None)
    if pending:
        res["prefill_pending"] = [b.stats() for b in pending]
    return res


if __name__ == "__main__":
    if _wants_llmd_proxy():
        _llmd = _llmd_proxy_path()
        if _llmd is None:
            raise SystemExit(
                "PROXY_PREFILL_ROUTING/PROXY_DECODE_ROUTING=llmd needs "
                "examples/disagg/toy_proxy_server_llmd.py next to this file or under "
                "/workspace/vllm-torchtpu/examples/disagg"
            )
        print(f"toy_proxy_server: routing=llmd, running the llm-d proxy {_llmd}", flush=True)
        import runpy

        runpy.run_path(_llmd, run_name="__main__")
        raise SystemExit(0)
    global global_args
    global_args = parse_args()

    import uvicorn

    # The benchmark client (aiohttp) reuses its connections to this proxy; with
    # uvicorn's default timeout_keep_alive of 5 s the server closes an idle
    # connection while the client is writing the next request body on it, which
    # surfaces as "ClientOSError: Can not write request body" or
    # ServerDisconnectedError on the client (2 to 4 requests per hour in the
    # AgentX replay). PROXY_KEEPALIVE_S=5 restores the default.
    limit_concurrency = _env_int("PROXY_UVICORN_LIMIT_CONCURRENCY", 0)
    uvicorn.run(
        app,
        host=global_args.host,
        port=global_args.port,
        timeout_keep_alive=_env_int("PROXY_KEEPALIVE_S", 300),
        limit_concurrency=limit_concurrency if limit_concurrency > 0 else None,
    )
