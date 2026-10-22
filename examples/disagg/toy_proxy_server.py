# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import codecs
import hashlib
import itertools
import logging
import math
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


class _DecodeStepCostModel:
    """Hardware-calibrated Decode step latency & DualPath marginal cost model.

    Models each Decode pod d's step latency across both hardware dimensions of
    TPU v7x Megakernel execution (Qwen3.5-397B-A17B, TP8xEP8xDCP4, MTP=3):
      1. Active batch size b_d = min(active_reqs_d, mk_ceiling), quantized to
         compiled Megakernel verify buckets.
      2. Active running KV token sum Sigma_L_d = running_tokens_d scanned from
         HBM across all full-attention layers every decode step.

    Per-step latency (ms):
        T_step(b_d, Sigma_L_d) = base_step_ms
                                 + mk_seq_ms * bucket(b_d)
                                 + kv_token_ms * Sigma_L_d

    DualPath marginal step cost (ms) of adding a request with L_req tokens:
        dC_step(d, L_req) = interactivity_steps * (
            T_step(b_d', Sigma_L_d + L_req)
            + b_d * (mk_seq_ms * (b_d' - b_d) + kv_token_ms * L_req)
        )
    where the cross-term (kv_token_ms * Sigma_L_d + b_d * kv_token_ms * L_req)
    routes long-context requests to low-concurrency, low-KV-bandwidth pods and
    packs short-context requests onto complementary pods.
    """

    def __init__(
        self,
        mk_ceiling: float = 112.0,
        base_step_ms: float = 5.2,
        mk_seq_ms: float = 0.18,
        kv_token_ms: float = 1.8e-6,
        bucket_step: float = 2.0,
        interactivity_steps: float = 45.0,
        queue_drain_steps: float = 120.0,
        raiden_pull_tok_per_ms: float = 280.0,
        sticky_cost_slack_ms: float = 60.0,
    ) -> None:
        self.mk_ceiling = max(1.0, float(mk_ceiling))
        self.base_step_ms = max(0.1, float(base_step_ms))
        self.mk_seq_ms = max(0.0, float(mk_seq_ms))
        self.kv_token_ms = max(0.0, float(kv_token_ms))
        self.bucket_step = max(1.0, float(bucket_step))
        self.interactivity_steps = max(1.0, float(interactivity_steps))
        self.queue_drain_steps = max(1.0, float(queue_drain_steps))
        self.raiden_pull_tok_per_ms = max(1.0, float(raiden_pull_tok_per_ms))
        self.sticky_cost_slack_ms = max(0.0, float(sticky_cost_slack_ms))

    @classmethod
    def from_env(cls, mk_ceiling: float | None = None) -> "_DecodeStepCostModel":
        default_ceiling = (
            float(mk_ceiling)
            if mk_ceiling is not None and mk_ceiling > 0
            else _env_float(
                "PROXY_DECODE_MK_CEILING",
                _env_float("MAX_NUM_SEQS_DECODE", 112.0),
            )
        )
        return cls(
            mk_ceiling=default_ceiling,
            base_step_ms=_env_float("PROXY_DECODE_STEP_BASE_MS", 5.2),
            mk_seq_ms=_env_float("PROXY_DECODE_STEP_MK_SEQ_MS", 0.18),
            kv_token_ms=_env_float("PROXY_DECODE_STEP_KV_TOK_MS", 1.8e-6),
            bucket_step=_env_float("PROXY_DECODE_STEP_BUCKET_STEP", 2.0),
            interactivity_steps=_env_float(
                "PROXY_DECODE_COST_INTERACTIVITY_STEPS", 45.0
            ),
            queue_drain_steps=_env_float("PROXY_DECODE_COST_QUEUE_DRAIN_STEPS", 120.0),
            raiden_pull_tok_per_ms=_env_float(
                "PROXY_DECODE_RAIDEN_PULL_TOK_PER_MS", 280.0
            ),
            sticky_cost_slack_ms=_env_float(
                "PROXY_DECODE_STICKY_COST_SLACK_MS", 60.0
            ),
        )

    def mk_bucket(self, b: float) -> float:
        b_clamped = max(0.0, min(float(b), self.mk_ceiling))
        if b_clamped <= 4.0 or self.bucket_step <= 1.0:
            return b_clamped
        q = math.ceil(b_clamped / self.bucket_step) * self.bucket_step
        return min(self.mk_ceiling, float(q))

    def step_latency_ms(self, active_reqs: float, running_tokens: float) -> float:
        b = max(0.0, min(float(active_reqs), self.mk_ceiling))
        toks = max(0.0, float(running_tokens))
        return (
            self.base_step_ms
            + self.mk_seq_ms * self.mk_bucket(b)
            + self.kv_token_ms * toks
        )

    def marginal_step_cost_ms(
        self, active_reqs: float, running_tokens: float, req_tokens: int
    ) -> float:
        b = max(0.0, min(float(active_reqs), self.mk_ceiling))
        b_next = max(1.0, min(float(active_reqs) + 1.0, self.mk_ceiling))
        toks = max(0.0, float(running_tokens))
        l_req = max(0.0, float(req_tokens))
        t_new = self.step_latency_ms(b_next, toks + l_req)
        externality = b * (
            self.mk_seq_ms * (b_next - b) + self.kv_token_ms * l_req
        )
        return self.interactivity_steps * (t_new + externality)

    def queue_wait_ms(self, active_reqs: float, running_tokens: float) -> float:
        if float(active_reqs) < self.mk_ceiling:
            return 0.0
        overflow = float(active_reqs) - self.mk_ceiling + 1.0
        t_ceil = self.step_latency_ms(self.mk_ceiling, running_tokens)
        return overflow * (self.queue_drain_steps / self.mk_ceiling) * t_ceil

    def pull_cost_ms(self, uncached_tokens: int) -> float:
        return max(0.0, float(uncached_tokens)) / self.raiden_pull_tok_per_ms

    def total_cost_ms(
        self,
        active_reqs: float,
        running_tokens: float,
        req_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        uncached = max(0, int(req_tokens) - max(0, int(cached_tokens)))
        return (
            self.marginal_step_cost_ms(active_reqs, running_tokens, req_tokens)
            + self.queue_wait_ms(active_reqs, running_tokens)
            + self.pull_cost_ms(uncached)
        )

    def allows_sticky(
        self,
        owner_reqs: float,
        min_reqs: float,
        owner_cost_ms: float,
        min_cost_ms: float,
        configured_slack: int,
    ) -> bool:
        cap_slack = max(1, int(round(0.20 * self.mk_ceiling)))
        if float(owner_reqs) < self.mk_ceiling:
            eff_slack = max(max(0, configured_slack), cap_slack)
        else:
            eff_slack = min(max(1, configured_slack), cap_slack)
        if float(owner_reqs) > float(min_reqs) + eff_slack:
            return False
        return float(owner_cost_ms) <= float(min_cost_ms) + self.sticky_cost_slack_ms


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
        decode_cost_model: _DecodeStepCostModel | None = None,
        hbm_max_chunks: int = 58000,
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
        self.decode_cost_model = decode_cost_model
        self.hbm_max_chunks = max(0, int(hbm_max_chunks))
        self._lock = threading.Lock()
        self.owners: dict[bytes, int] = {}
        self.inflight = [0] * n
        self.inflight_eff_tokens: list[float] = [0.0] * n
        self._req_eff_tokens: dict[str, float] = {}
        self._anon_eff_tokens: list[list[float]] = [[] for _ in range(n)]
        self.running = [0] * n
        self.running_tokens: list[int] = [0] * n
        self.inflight_decode_tokens: list[int] = [0] * n
        self._req_decode_tokens: dict[str, list[Any]] = {}
        self._pending_run_tokens: list[list[int]] = [[] for _ in range(n)]
        self._anon_run_tokens: list[list[int]] = [[] for _ in range(n)]
        self.pod_hbm_lru: list[OrderedDict[bytes, None]] = [
            OrderedDict() for _ in range(n)
        ]
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

    def _match_pod_hbm(self, rank: int, hs: list[bytes]) -> int:
        if not hs or self.hbm_max_chunks <= 0:
            return 0
        hbm = self.pod_hbm_lru[rank]
        hits = 0
        for d in hs:
            if d in hbm:
                hits += 1
            else:
                break
        return hits

    def _record_pod_hbm(self, rank: int, hs: list[bytes]) -> None:
        if not hs or self.hbm_max_chunks <= 0:
            return
        hbm = self.pod_hbm_lru[rank]
        for d in hs:
            if d in hbm:
                hbm.move_to_end(d)
            else:
                hbm[d] = None
                if len(hbm) > self.hbm_max_chunks:
                    hbm.popitem(last=False)

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
            use_dscm = self.decode_cost_model is not None and (est_tokens > 0 or bool(hs))
            if use_dscm and self.decode_cost_model is not None:
                req_toks = max(64, est_tokens if est_tokens > 0 else 4096)
                active_reqs = [float(load[r]) for r in range(self.n)]
                active_toks = [
                    float(
                        self.running_tokens[r]
                        if self.load_metric == "running" and self.running_tokens[r] > 0
                        else self.inflight_decode_tokens[r]
                    )
                    for r in range(self.n)
                ]
                cached_toks_per_pod = [0] * self.n
                for r in range(self.n):
                    hits_r = self._match_pod_hbm(r, hs)
                    if r == best_rank and matched > hits_r:
                        hits_r = matched
                    frac_r = min(1.0, hits_r / max(len(hs), 1)) if hs else 0.0
                    cached_toks_per_pod[r] = int(req_toks * frac_r)
                cost_ms = [
                    self.decode_cost_model.total_cost_ms(
                        active_reqs=active_reqs[r],
                        running_tokens=active_toks[r],
                        req_tokens=req_toks,
                        cached_tokens=cached_toks_per_pod[r],
                    )
                    for r in range(self.n)
                ]
                min_cost_ms = min(cost_ms)
                dscm_prefers_sticky = (
                    known
                    and best_rank is not None
                    and self.decode_cost_model.allows_sticky(
                        owner_reqs=active_reqs[best_rank],
                        min_reqs=float(lo),
                        owner_cost_ms=cost_ms[best_rank],
                        min_cost_ms=min_cost_ms,
                        configured_slack=self.slack,
                    )
                )
                if known and best_rank is not None and (saturated or dscm_prefers_sticky):
                    rank = best_rank
                    is_sticky = True
                    req_eff = 0.0
                    self.sticky += 1
                    if saturated and load[best_rank] > lo + self.slack:
                        self.sticky_saturated += 1
                    elif dscm_prefers_sticky and load[best_rank] > lo + self.slack:
                        self.sticky_quad_saved += 1
                else:
                    rank = min(
                        range(self.n),
                        key=lambda r: (
                            len(self.resident[r]) if self.max_sessions > 0 else 0,
                            round(cost_ms[r], 2),
                            load[r],
                            active_toks[r],
                            self.inflight[r],
                            self.placed[r],
                            r,
                        ),
                    )
                    is_sticky = False
                    req_eff = 0.0
                    self.cold += 1
                use_quad = False
            else:
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
            self._record_pod_hbm(rank, hs)
            self.inflight[rank] += 1
            if est_tokens > 0:
                self.inflight_decode_tokens[rank] += est_tokens
                if req_id is not None:
                    self._req_decode_tokens[req_id] = [est_tokens, False]
                else:
                    self._pending_run_tokens[rank].append(est_tokens)
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
                    f"running_tokens={self.running_tokens} "
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

    def start_running(
        self, rank: int, req_id: str | None = None, tokens: int = 0
    ) -> None:
        """Counts a request routed to ``rank`` as running there."""
        with self._lock:
            self.running[rank] += 1
            if req_id is not None and req_id in self._req_decode_tokens:
                entry = self._req_decode_tokens[req_id]
                if tokens > 0 and tokens != entry[0]:
                    self.inflight_decode_tokens[rank] = max(
                        0, self.inflight_decode_tokens[rank] + (tokens - entry[0])
                    )
                    entry[0] = tokens
                if not entry[1]:
                    entry[1] = True
                    self.running_tokens[rank] += entry[0]
            elif tokens > 0:
                self.running_tokens[rank] += tokens
                self._anon_run_tokens[rank].append(tokens)
            elif self._pending_run_tokens[rank]:
                tok = self._pending_run_tokens[rank].pop(0)
                self.running_tokens[rank] += tok
                self._anon_run_tokens[rank].append(tok)

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
            if req_id is not None and req_id in self._req_decode_tokens:
                tok, is_run = self._req_decode_tokens.pop(req_id)
                self.inflight_decode_tokens[rank] = max(
                    0, self.inflight_decode_tokens[rank] - tok
                )
                if is_run or was_running:
                    self.running_tokens[rank] = max(0, self.running_tokens[rank] - tok)
            elif was_running and self._anon_run_tokens[rank]:
                tok = self._anon_run_tokens[rank].pop(0)
                self.inflight_decode_tokens[rank] = max(
                    0, self.inflight_decode_tokens[rank] - tok
                )
                self.running_tokens[rank] = max(0, self.running_tokens[rank] - tok)
            elif self._pending_run_tokens[rank]:
                tok = self._pending_run_tokens[rank].pop(0)
                self.inflight_decode_tokens[rank] = max(
                    0, self.inflight_decode_tokens[rank] - tok
                )
            if was_running:
                self.running[rank] = max(0, self.running[rank] - 1)
            if self.running[rank] == 0:
                self.running_tokens[rank] = 0
                self._anon_run_tokens[rank].clear()
            if self.inflight[rank] == 0:
                self.inflight_decode_tokens[rank] = 0
                self._pending_run_tokens[rank].clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "placed": list(self.placed),
                "inflight": list(self.inflight),
                "inflight_eff_tokens": list(self.inflight_eff_tokens),
                "inflight_decode_tokens": list(self.inflight_decode_tokens),
                "running": list(self.running),
                "running_tokens": list(self.running_tokens),
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
    routing_mode = os.getenv("PROXY_PREFILL_ROUTING", "prefix_hash").strip().lower()
    # Residency-aware prefill routing: PROXY_PREFILL_MAX_SESSIONS conversations
    # per prefill leg (0 = off; ~54 for a 397B leg without a host tier, ~240
    # with one) and PROXY_PREFIX_HASH_SATURATED_INFLIGHT, the per-leg in-flight
    # count above which the sticky slack no longer applies (0 = off).
    if len(app.state.prefill_clients) > 1 and routing_mode in (
        "prefix_hash",
        "llmd",
    ):
        app.state.prefill_router = _PrefixAffinityRouter(
            n=len(app.state.prefill_clients),
            chunk=_env_int("PROXY_PREFIX_HASH_CHUNK", 1024),
            min_frac=_env_float("PROXY_PREFIX_HASH_MIN_FRAC", 0.5),
            slack=_env_int("PROXY_PREFIX_HASH_SLACK", 8),
            max_sessions=_env_int("PROXY_PREFILL_MAX_SESSIONS", 0),
            saturated_inflight=_env_int("PROXY_PREFIX_HASH_SATURATED_INFLIGHT", 0),
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
    if len(app.state.decode_clients) > 1 and decode_routing in (
        "prefix_hash",
        "llmd",
    ):
        decode_mk_ceiling = _env_float(
            "PROXY_DECODE_MK_CEILING",
            _env_float("MAX_NUM_SEQS_DECODE", 112.0),
        )
        decode_cost_model = (
            _DecodeStepCostModel.from_env(mk_ceiling=decode_mk_ceiling)
            if _env_int("PROXY_DECODE_STEP_COST_MODEL", 1) != 0
            else None
        )
        app.state.decode_router = _PrefixAffinityRouter(
            n=len(app.state.decode_clients),
            chunk=_env_int("PROXY_PREFIX_HASH_CHUNK", 1024),
            min_frac=_env_float("PROXY_PREFIX_HASH_MIN_FRAC", 0.5),
            slack=_env_int("PROXY_DECODE_HASH_SLACK", decode_slack_default),
            load_metric=decode_load_metric,
            decode_cost_model=decode_cost_model,
        )
    else:
        app.state.decode_router = None
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
    default_admit_point = (
        "after_prefill" if app.state.decode_router is not None else "before_prefill"
    )
    admit_point = (
        os.getenv("PROXY_DECODE_ADMIT_POINT", default_admit_point).strip().lower()
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

    saturated_inflight = _env_int("PROXY_PREFIX_HASH_SATURATED_INFLIGHT", 0)
    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"(routing={'prefix_hash' if app.state.prefill_router else 'round_robin'}"
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
        state = request.app.state
        admit_point: str = state.decode_admit_point
        prompt_est = _estimate_prompt_tokens(
            _PrefixAffinityRouter.extract_text(req_data), state.bytes_per_token
        )
        output_est = _expected_output_tokens(req_data, state.output_tokens_est)
        actual_prompt_tokens = prompt_est
        # The budget a request holds, in tokens, on each leg; corrected once the
        # prefill response reports the real prompt length.
        decode_tokens = prompt_est + output_est
        pending_tokens = prompt_est
        decode_client_info: dict | None = None
        decode_budget: _AdmissionBudget | None = None
        if admit_point == "before_prefill":
            if decode_router is not None:
                decode_idx, _ = decode_router.route(
                    hs, prompt_tokens=prompt_est, req_id=request_id
                )
            else:
                decode_idx = next(request.app.state.decode_iterator)
            decode_client_info = request.app.state.decode_clients[decode_idx]
            decode_budget = state.decode_budgets[decode_idx]
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
            if decode_held and decode_budget is not None:
                await decode_budget.release(decode_tokens)
            if decode_router is not None and decode_idx is not None:
                decode_router.done(
                    decode_idx, was_running=decode_running, req_id=request_id
                )

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
            if decode_budget is None or not decode_budget.enabled:
                return
            wait_ms = await decode_budget.acquire(decode_tokens)
            decode_held = True
            _PROXY_LATENCY.record("decode_admit", wait_ms)
            decode_id = (
                decode_client_info.get("id", decode_idx)
                if decode_client_info is not None
                else decode_idx
            )
            print(
                f"PERF PROXY decode_admit req_id={request_id} "
                f"decode_id={decode_id} "
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
                actual_prompt_tokens = prompt_tokens
                if pending_held and pending_budget is not None:
                    await pending_budget.adjust(prompt_tokens - pending_tokens)
                pending_tokens = prompt_tokens
                if decode_held and decode_budget is not None:
                    await decode_budget.adjust(prompt_tokens - prompt_est)
                decode_tokens = prompt_tokens + output_est

            if kv_transfer_params:
                req_data["kv_transfer_params"] = kv_transfer_params

            if admit_point == "after_prefill":
                if decode_router is not None:
                    decode_idx, _ = decode_router.route(
                        hs, prompt_tokens=actual_prompt_tokens, req_id=request_id
                    )
                else:
                    decode_idx = next(request.app.state.decode_iterator)
                decode_client_info = request.app.state.decode_clients[decode_idx]
                decode_budget = state.decode_budgets[decode_idx]
                await admit_decode(admit_point)
                if decode_router is not None:
                    decode_router.start_running(
                        decode_idx,
                        req_id=request_id,
                        tokens=actual_prompt_tokens,
                    )
                    decode_running = True

            logger.debug("Using %s %s", prefill_client_info, decode_client_info)

            # Stream response from decode service
            async def generate_stream():
                nonlocal decode_running
                try:
                    if decode_router is not None and not decode_running:
                        decode_router.start_running(
                            decode_idx,
                            req_id=request_id,
                            tokens=actual_prompt_tokens,
                        )
                        decode_running = True
                    t_decode_send = time.time()
                    t_decode_send_perf = time.perf_counter()
                    print(
                        f"PERF PROXY decode_send req_id={request_id} "
                        f"ts={t_decode_send:.6f}{kv_uuid_log}",
                        flush=True,
                    )
                    first = True
                    async for chunk in stream_from_decode(
                        decode_client_info, api, req_data, request_id, forwarded
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
    budgets = getattr(app.state, "decode_budgets", None)
    if budgets:
        res["decode_admission"] = [b.stats() for b in budgets]
    pending = getattr(app.state, "prefill_pending", None)
    if pending:
        res["prefill_pending"] = [b.stats() for b in pending]
    return res


if __name__ == "__main__":
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
