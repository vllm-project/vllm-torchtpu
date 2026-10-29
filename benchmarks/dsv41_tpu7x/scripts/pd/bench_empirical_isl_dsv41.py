#!/usr/bin/env python3
"""5-or-N Trunks + Exact Temperatures Continuous Closed-Loop Empirical ISL Benchmark for DeepSeek-V4.1-Flash (1P1D).

Implements the exact empirical 50:20:12:9:9 ISL distribution benchmark methodology & harness
from the GLM-5.3 PD Disaggregation report, adapted for vLLM-TorchTPU DeepSeek-V4.1-Flash 1P1D:
  1. Trunk Cache Strategy (`eff_trunks = max(5, sessions_per_dp)` per DP rank, or scaled across DP=8):
     - Partitioned across the 5 ISL buckets (`1-8K`, `8-16K`, `16-32K`, `32-64K`, `64K+`)
       in exact `50:20:12:9:9` proportion.
     - Each trunk is deterministically pinned to its assigned `dp_rank` on `1P` (`ROLE=prefill`),
       populating `1P`'s prefix cache (with `--enable-prefix-caching --prefix-cache-retention-interval 1024`),
       while `1D` (`ROLE=decode`) pulls KV over Raiden Stage 3 with least-in-flight DP load balancing.
  2. Exact Sampling Parameters from `requests_mocked.jsonl(.xz)`:
     - Preserves all 30 rows' exact `(temperature, top_p, top_k)` specs and cycles them across dispatched requests.
  3. Zero Runtime Compilation Guarantee:
     - Optional Pass-1 micro-warmup followed by `/reset_prefix_cache` before the measured sweep.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import lzma
from pathlib import Path
import statistics
import time
from typing import Any
import aiohttp


CHARS_PER_TOKEN = 3.85

ISL_BUCKETS = [
    ("1-8K", 0.50, 5376, 8064),
    ("8-16K", 0.20, 8320, 15872),
    ("16-32K", 0.12, 16384, 31744),
    ("32-64K", 0.09, 32768, 63488),
    ("64K+", 0.09, 65536, 92160),
]


def _extract_text(content: Any) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for p in content:
      if isinstance(p, dict) and isinstance(p.get("text"), str):
        parts.append(p["text"])
      else:
        parts.append(json.dumps(p, ensure_ascii=False))
    return "\n".join(parts)
  return str(content or "")


def _msg_char_len(m: dict[str, Any]) -> int:
  c = len(_extract_text(m.get("content"))) + len(str(m.get("reasoning_content") or ""))
  if m.get("tool_calls"):
    c += len(json.dumps(m["tool_calls"], ensure_ascii=False))
  return max(16, c + 16)


def load_raw_requests(path: str) -> list[dict[str, Any]]:
  """Loads requests_mocked.jsonl(.xz) preserving structured messages, tools, and max_tokens."""
  p = Path(path)
  if p.suffix == ".xz":
    with lzma.open(p, "rt", encoding="utf-8") as f:
      raw_lines = [line.strip() for line in f if line.strip()]
  else:
    with open(p, "r", encoding="utf-8") as f:
      raw_lines = [line.strip() for line in f if line.strip()]

  raw_reqs: list[dict[str, Any]] = []
  for line in raw_lines:
    obj = json.loads(line)
    r = obj.get("body", obj)
    if isinstance(r, str):
      r = json.loads(r)
    raw_reqs.append(r)
  tool_bounded = [
      r for r in raw_reqs if r.get("tools") and int(r.get("max_tokens") or 0) <= 32768
  ]
  if tool_bounded:
    return tool_bounded
  tool_any = [r for r in raw_reqs if r.get("tools")]
  return tool_any if tool_any else (raw_reqs[1:] if len(raw_reqs) > 1 else raw_reqs)


def load_all_sampling_specs(path: str) -> list[dict[str, Any]]:
  """Loads exact per-row sampling parameters (temperature, top_p, top_k) from all rows of requests_mocked.jsonl(.xz)."""
  p = Path(path)
  if p.suffix == ".xz":
    with lzma.open(p, "rt", encoding="utf-8") as f:
      raw_lines = [line.strip() for line in f if line.strip()]
  else:
    with open(p, "r", encoding="utf-8") as f:
      raw_lines = [line.strip() for line in f if line.strip()]
  specs: list[dict[str, Any]] = []
  for line in raw_lines:
    obj = json.loads(line)
    r = obj.get("body", obj)
    if isinstance(r, str):
      r = json.loads(r)
    sp: dict[str, Any] = {}
    if "temperature" in r and r["temperature"] is not None:
      sp["temperature"] = r["temperature"]
    if "top_p" in r and r["top_p"] is not None:
      sp["top_p"] = r["top_p"]
    if "top_k" in r and r["top_k"] is not None:
      sp["top_k"] = r["top_k"]
    specs.append(sp)
  return specs


def _bucket_target_tokens(b_idx: int) -> int:
  _, _, lo_tok, hi_tok = ISL_BUCKETS[b_idx]
  return (((lo_tok + hi_tok) // 2) // 128) * 128


def _allocate_bucket_counts(total_trunks: int) -> list[int]:
  """Allocates `total_trunks >= len(ISL_BUCKETS)` across ISL_BUCKETS in 50:20:12:9:9 ratio with >=1 per bucket."""
  n_b = len(ISL_BUCKETS)
  assert total_trunks >= n_b
  if total_trunks == n_b:
    return [1] * n_b
  raw = [total_trunks * w for _, w, _, _ in ISL_BUCKETS]
  floors = [max(1, int(x)) for x in raw]
  while sum(floors) > total_trunks:
    idx = max(
        (i for i in range(n_b) if floors[i] > 1),
        key=lambda i: (floors[i] - raw[i], -ISL_BUCKETS[i][1]),
    )
    floors[idx] -= 1
  leftover = total_trunks - sum(floors)
  order = sorted(
      range(n_b),
      key=lambda i: (raw[i] - floors[i], ISL_BUCKETS[i][1]),
      reverse=True,
  )
  for i in order[:leftover]:
    floors[i] += 1
  return floors


def build_5_or_n_trunk_specs(
    num_sessions: int, dp_size: int = 1, trunks_above_n16: int = 0
) -> tuple[list[dict[str, Any]], dict[int, dict[int, list[dict[str, Any]]]]]:
  """Builds `max(5, sessions_per_dp)` distinct trunk specs per DP rank."""
  per_dp_counts: dict[int, int] = {r: 0 for r in range(dp_size)}
  for k in range(num_sessions):
    per_dp_counts[k % dp_size] += 1

  all_trunks: list[dict[str, Any]] = []
  per_dp_bucket_trunks: dict[int, dict[int, list[dict[str, Any]]]] = {
      r: {b: [] for b in range(len(ISL_BUCKETS))} for r in range(dp_size)
  }

  global_trunk_id = 0
  for r in range(dp_size):
    if trunks_above_n16 > 0 and num_sessions > 16:
      cnt_r = max(len(ISL_BUCKETS), trunks_above_n16 // max(1, dp_size))
    else:
      cnt_r = max(len(ISL_BUCKETS), per_dp_counts[r])
    target_counts = _allocate_bucket_counts(cnt_r)
    assigned = [0] * len(ISL_BUCKETS)
    for step in range(cnt_r):
      candidates = [b for b in range(len(ISL_BUCKETS)) if assigned[b] < target_counts[b]]
      b_chosen = max(
          candidates,
          key=lambda b: ((step + 1) * ISL_BUCKETS[b][1] - assigned[b], ISL_BUCKETS[b][1]),
      )
      assigned[b_chosen] += 1
      target_tok = _bucket_target_tokens(b_chosen)
      sp = {
          "trunk_id": global_trunk_id,
          "dp_rank": r,
          "bucket_idx": b_chosen,
          "bucket": ISL_BUCKETS[b_chosen][0],
          "warmup_target_isl_tokens": target_tok,
      }
      all_trunks.append(sp)
      per_dp_bucket_trunks[r][b_chosen].append(sp)
      global_trunk_id += 1

  return all_trunks, per_dp_bucket_trunks


def build_trunk_chat_payload(
    orig_req: dict[str, Any],
    *,
    model_name: str,
    trunk_namespace: str,
    session_id: int,
    trunk_id: int,
    bucket_idx: int,
    dp_rank: int,
    target_isl_tokens: int,
    cache_hit_ratio: float,
    uid: str,
    max_tokens: int,
    bucket_orig_max_tokens: int,
    ignore_eos: bool,
    is_warmup: bool = False,
    sampling_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Builds a structured multi-turn + tools payload for `trunk_id` with shared prefix."""
  total_chars = max(1024, int(target_isl_tokens * CHARS_PER_TOKEN))
  # Center 1024-token block quantization (`floor(prefix_tok / 1024) * 1024`) around `cache_hit_ratio`
  eff_ratio = min(0.92, (target_isl_tokens * cache_hit_ratio + 512.0) / max(1, target_isl_tokens))
  prefix_budget = int(total_chars * eff_ratio)
  suffix_budget = max(256, total_chars - prefix_budget)

  trunk_tag = f"[trunk_key={trunk_namespace}_k{trunk_id:04d}_b{bucket_idx}_dp{dp_rank:02d}] "
  raw_tools = copy.deepcopy(orig_req.get("tools") or [])

  max_tools_chars = max(0, prefix_budget - 1024)
  kept_tools: list[dict[str, Any]] = []
  tools_chars = 0
  for t in raw_tools:
    tc_len = len(json.dumps(t, ensure_ascii=False)) + 2
    if tools_chars + tc_len <= max_tools_chars or (not kept_tools and tc_len <= max_tools_chars):
      kept_tools.append(t)
      tools_chars += tc_len
    else:
      break

  orig_msgs = copy.deepcopy(orig_req.get("messages") or [])
  if kept_tools:
    if "function" in kept_tools[0] and isinstance(kept_tools[0]["function"], dict):
      desc = str(kept_tools[0]["function"].get("description") or "")
      kept_tools[0]["function"]["description"] = trunk_tag + desc
  if orig_msgs and isinstance(orig_msgs[0].get("content"), str):
    orig_msgs[0]["content"] = trunk_tag + orig_msgs[0]["content"]
  elif not kept_tools:
    orig_msgs.insert(0, {"role": "system", "content": trunk_tag})

  rem_prefix = max(256, prefix_budget - tools_chars)
  prefix_messages: list[dict[str, Any]] = []
  cur_p = 0
  rep = 0
  while cur_p < rem_prefix:
    for m in orig_msgs:
      m_copy = copy.deepcopy(m)
      if isinstance(m_copy.get("content"), list):
        m_copy["content"] = _extract_text(m_copy["content"])
      mlen = _msg_char_len(m_copy)
      if cur_p + mlen <= rem_prefix:
        prefix_messages.append(m_copy)
        cur_p += mlen
      else:
        need = rem_prefix - cur_p
        if need >= 32:
          txt = _extract_text(m_copy.get("content"))
          if len(txt) < need:
            pad_unit = txt + "\n" + json.dumps(m_copy, ensure_ascii=False)
            txt = pad_unit * ((need // max(1, len(pad_unit))) + 2)
          m_slice: dict[str, Any] = {
              "role": m_copy.get("role", "user"),
              "content": txt[:need],
          }
          if m_slice["role"] == "tool" and m_copy.get("tool_call_id"):
            m_slice["tool_call_id"] = m_copy["tool_call_id"]
          prefix_messages.append(m_slice)
          cur_p += need
        cur_p = rem_prefix
        break
    rep += 1
    if rep > 50:
      break

  turn_tag = f"[turn_uid=s{session_id:04d}_k{trunk_id:04d}_{uid}]\n"
  tail_texts: list[str] = []
  cur_s = len(turn_tag)
  while cur_s < suffix_budget:
    for m in reversed(orig_msgs):
      txt = _extract_text(m.get("content"))
      if txt.strip():
        tail_texts.append(txt)
        cur_s += len(txt) + 2
        if cur_s >= suffix_budget:
          break
    if not tail_texts:
      tail_texts.append("Continue the analysis and call the appropriate tool if needed.")
      break
  suffix_str = (turn_tag + "\n".join(tail_texts))[:suffix_budget]
  suffix_messages = [{"role": "user", "content": suffix_str}]

  if max_tokens > 0:
    req_max_tokens = max_tokens
  elif target_isl_tokens < 8192:
    req_max_tokens = 1024
  else:
    req_max_tokens = bucket_orig_max_tokens
  req_ignore_eos = ignore_eos
  if is_warmup:
    req_max_tokens = 1
    req_ignore_eos = False

  payload: dict[str, Any] = {
      "model": model_name,
      "messages": prefix_messages + suffix_messages,
      "chat_template_kwargs": {
          "enable_thinking": True,
          "reasoning_effort": "high",
          "clear_thinking": False,
      },
      "max_tokens": req_max_tokens,
      "ignore_eos": req_ignore_eos,
      "stream": True,
      "stream_options": {"include_usage": True},
      "dp_rank": int(dp_rank),
      "rid": f"s{session_id}_k{trunk_id}_dp{dp_rank}_{uid}",
  }
  if is_warmup:
    payload["temperature"] = 0.0
  elif sampling_spec is not None:
    if "temperature" in sampling_spec:
      payload["temperature"] = sampling_spec["temperature"]
    if "top_p" in sampling_spec:
      payload["top_p"] = sampling_spec["top_p"]
    if "top_k" in sampling_spec:
      payload["top_k"] = sampling_spec["top_k"]
  else:
    if "temperature" in orig_req and orig_req["temperature"] is not None:
      payload["temperature"] = orig_req["temperature"]
    if "top_p" in orig_req and orig_req["top_p"] is not None:
      payload["top_p"] = orig_req["top_p"]
  if kept_tools:
    payload["tools"] = kept_tools
    if orig_req.get("tool_choice") is not None:
      payload["tool_choice"] = orig_req["tool_choice"]
  return payload


async def send_chat_request(
    http_session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
  t_start = time.perf_counter()
  ttft: float | None = None
  chunk_times: list[float] = []
  prompt_tokens = 0
  completion_tokens = 0
  cached_tokens = 0
  status_code = 0
  error_msg: str | None = None
  num_content_chunks = 0
  finish_reason: str | None = None
  has_tool_calls = False

  headers = {
      "X-SMG-Routing-Key": f"trunk-{meta.get('trunk_id', meta.get('session_id', 0))}",
      "X-Session-ID": f"session-{meta.get('session_id', 0)}",
  }

  try:
    async with http_session.post(url, json=payload, headers=headers) as resp:
      status_code = resp.status
      if status_code != 200:
        err_body = await resp.text()
        error_msg = f"HTTP {status_code}: {err_body[:300]}"
      else:
        async for raw_line in resp.content:
          line = raw_line.decode("utf-8", errors="replace").strip()
          if not line or not line.startswith("data:"):
            continue
          data_str = line[5:].strip()
          if data_str == "[DONE]":
            break
          try:
            chunk = json.loads(data_str)
          except Exception:
            continue

          now = time.perf_counter()
          choices = chunk.get("choices")
          if choices and len(choices) > 0:
            c0 = choices[0]
            if c0.get("finish_reason"):
              finish_reason = c0.get("finish_reason")
            delta = c0.get("delta", {})
            text_val = c0.get("text", "")
            if delta.get("tool_calls"):
              has_tool_calls = True
            has_token = bool(
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
                or delta.get("tool_calls")
                or text_val
            )
            if has_token:
              if ttft is None:
                ttft = now - t_start
              chunk_times.append(now)
              num_content_chunks += 1

          usage = chunk.get("usage")
          if isinstance(usage, dict) and usage:
            prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
            completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict) and details.get("cached_tokens") is not None:
              cached_tokens = int(details["cached_tokens"])
  except Exception as e:
    error_msg = f"{type(e).__name__}: {e}"

  t_end = time.perf_counter()
  e2e = t_end - t_start
  if ttft is None and error_msg is None:
    ttft = e2e

  eff_compl = completion_tokens or num_content_chunks
  decode_dur = max(0.0, e2e - (ttft or 0.0))
  per_token_itl: float | None = None
  decode_tps: float | None = None
  if eff_compl > 1 and decode_dur > 0:
    per_token_itl = decode_dur / (eff_compl - 1)
    decode_tps = (eff_compl - 1) / decode_dur

  chunk_itls: list[float] = []
  if len(chunk_times) >= 2:
    chunk_itls = [chunk_times[i] - chunk_times[i - 1] for i in range(1, len(chunk_times))]
  elif per_token_itl is not None:
    chunk_itls = [per_token_itl]

  return {
      **meta,
      "t_start": t_start,
      "t_end": t_end,
      "ttft": ttft,
      "e2e": e2e,
      "decode_dur": decode_dur,
      "per_token_itl": per_token_itl,
      "decode_tps": decode_tps,
      "itls": chunk_itls,
      "chunk_times": chunk_times,
      "prompt_tokens": prompt_tokens,
      "completion_tokens": eff_compl,
      "cached_tokens": cached_tokens,
      "finish_reason": finish_reason,
      "has_tool_calls": has_tool_calls,
      "status_code": status_code,
      "error": error_msg,
  }


def percentile(vals: list[float], p: float) -> float:
  if not vals:
    return 0.0
  s = sorted(vals)
  k = (len(s) - 1) * (p / 100.0)
  f = int(k)
  c = min(f + 1, len(s) - 1)
  if f == c:
    return s[f]
  return s[f] + (s[c] - s[f]) * (k - f)


async def flush_all_caches(
    http_session: aiohttp.ClientSession,
    *,
    base_url: str,
    prefill_url: str,
    decode_url: str,
    max_retries: int = 15,
) -> None:
  """Drains in-flight batches and resets prefix caches on prefill, decode, and proxy."""
  await asyncio.sleep(2.0)
  for label, u in [("prefill", prefill_url), ("decode", decode_url), ("lb", base_url)]:
    if not u:
      continue
    flushed = False
    last_status = 0
    for attempt in range(max_retries):
      for ep in ("/reset_prefix_cache", "/flush_cache"):
        try:
          async with http_session.post(
              f"{u.rstrip('/')}{ep}", timeout=aiohttp.ClientTimeout(total=15)
          ) as resp:
            last_status = resp.status
            if resp.status == 200:
              try:
                body = await resp.json()
                if isinstance(body, dict) and body.get("success") is False:
                  continue
              except Exception:
                pass
              flushed = True
              break
        except Exception:
          pass
      if flushed:
        break
      await asyncio.sleep(1.0)
    if label in ("prefill", "decode"):
      print(f"  [reset_prefix_cache] {label}: {'OK (HTTP 200)' if flushed else f'WARN (last HTTP {last_status})'}")
  await asyncio.sleep(1.0)


async def scrape_spec_metrics(http_session: aiohttp.ClientSession, decode_url: str) -> dict[str, Any] | None:
  """Scrape DSpark speculative decoding counters from 1D /metrics."""
  if not decode_url:
    return None
  try:
    async with http_session.get(
        f"{decode_url.rstrip('/')}/metrics",
        timeout=aiohttp.ClientTimeout(total=10),
    ) as resp:
      if resp.status != 200:
        return None
      text = await resp.text()
  except Exception:
    return None
  import re
  num_drafts = sum(
      float(m.group(1))
      for m in re.finditer(r"^vllm:spec_decode_num_drafts_total\{[^}]*\}\s+([0-9.eE+-]+)", text, re.M)
  )
  num_draft_tokens = sum(
      float(m.group(1))
      for m in re.finditer(r"^vllm:spec_decode_num_draft_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)", text, re.M)
  )
  num_accepted = sum(
      float(m.group(1))
      for m in re.finditer(r"^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)", text, re.M)
  )
  per_pos: dict[int, float] = {}
  for m in re.finditer(
      r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([0-9.eE+-]+)',
      text,
      re.M,
  ):
    pos = int(m.group(1))
    per_pos[pos] = per_pos.get(pos, 0.0) + float(m.group(2))
  return {
      "num_drafts": num_drafts,
      "num_draft_tokens": num_draft_tokens,
      "num_accepted": num_accepted,
      "per_pos": per_pos,
  }


async def run_micro_xla_warmup_pass(
    http_session: aiohttp.ClientSession,
    *,
    model_name: str,
    base_url: str,
    prefill_url: str,
    decode_url: str,
    raw_reqs: list[dict[str, Any]],
    all_sampling_specs: list[dict[str, Any]],
    session_list: list[int],
    warmup_pass_s: float,
    cache_hit_ratio: float,
    dp_size: int,
) -> None:
  """Fast Pass-1 micro-burst across all `n` values + sampling specs to guarantee 0 runtime XLA compiles."""
  target_url = f"{base_url.rstrip('/')}/v1/chat/completions"
  print(
      f"\n================================================================================\n"
      f"[PASS 1 — FAST XLA / KERNEL WARMUP] {warmup_pass_s:.0f}s/n across {session_list} (max_tokens=48)\n"
      f"================================================================================"
  )
  await flush_all_caches(
      http_session, base_url=base_url, prefill_url=prefill_url, decode_url=decode_url
  )
  warm_trunks, per_dp_bucket_trunks = build_5_or_n_trunk_specs(num_sessions=5, dp_size=dp_size)
  p1_sem = asyncio.Semaphore(max(1, min(8, len(warm_trunks))))
  p1_dp_sems = {r: asyncio.Semaphore(1) for r in range(dp_size)}

  async def _p1_warm_one(sp: dict[str, Any]) -> None:
    b_idx = sp["bucket_idx"]
    dp_rank = sp["dp_rank"]
    orig_req = raw_reqs[(b_idx + dp_rank * 5) % len(raw_reqs)]
    b_max_tok = int(orig_req.get("max_tokens") or 8192)
    w_payload = build_trunk_chat_payload(
        orig_req,
        model_name=model_name,
        trunk_namespace="pass1_warm",
        session_id=sp["trunk_id"],
        trunk_id=sp["trunk_id"],
        bucket_idx=b_idx,
        dp_rank=dp_rank,
        target_isl_tokens=sp["warmup_target_isl_tokens"],
        cache_hit_ratio=cache_hit_ratio,
        uid=f"p1_warm_k{sp['trunk_id']}",
        max_tokens=1,
        bucket_orig_max_tokens=b_max_tok,
        ignore_eos=False,
        is_warmup=True,
    )
    async with p1_dp_sems[dp_rank], p1_sem:
      await send_chat_request(
          http_session,
          target_url,
          w_payload,
          {"session_id": sp["trunk_id"], "trunk_id": sp["trunk_id"]},
      )

  warm_trunks_ordered = sorted(warm_trunks, key=lambda sp: (-sp["bucket_idx"], sp["dp_rank"], sp["trunk_id"]))
  await asyncio.gather(*(_p1_warm_one(sp) for sp in warm_trunks_ordered))

  for n in session_list:
    t0 = time.perf_counter()
    t_end = t0 + warmup_pass_s
    disp = [0]

    async def _p1_worker(s_id: int) -> int:
      dp_rank = s_id % dp_size
      cnt = 0
      while time.perf_counter() < t_end:
        idx = disp[0]
        disp[0] += 1
        b_idx = idx % len(ISL_BUCKETS)
        if idx >= len(ISL_BUCKETS):
          b_idx = 0 if (idx % 3 != 0) else 1
        trunk_sp = per_dp_bucket_trunks[dp_rank][b_idx][0]
        orig_req = raw_reqs[(b_idx + dp_rank * 5) % len(raw_reqs)]
        samp_spec = all_sampling_specs[idx % len(all_sampling_specs)]
        payload = build_trunk_chat_payload(
            orig_req,
            model_name=model_name,
            trunk_namespace="pass1_warm",
            session_id=s_id,
            trunk_id=trunk_sp["trunk_id"],
            bucket_idx=b_idx,
            dp_rank=dp_rank,
            target_isl_tokens=trunk_sp["warmup_target_isl_tokens"],
            cache_hit_ratio=cache_hit_ratio,
            uid=f"p1_n{n}_s{s_id}_c{cnt}_{int(time.perf_counter()*1000)%100000}",
            max_tokens=48,
            bucket_orig_max_tokens=48,
            ignore_eos=True,
            is_warmup=False,
            sampling_spec=samp_spec,
        )
        await send_chat_request(
            http_session,
            target_url,
            payload,
            {"session_id": s_id, "trunk_id": trunk_sp["trunk_id"]},
        )
        cnt += 1
      return cnt

    counts = await asyncio.gather(*(_p1_worker(s_id) for s_id in range(n)))
    print(f"  [Pass-1 Warmup n={n:2d}] Completed {sum(counts)} micro-requests in {time.perf_counter() - t0:.1f}s")

  print("[PASS 1 COMPLETE] Flushing all prefix/KV caches before Measured Pass 2...")
  await flush_all_caches(
      http_session, base_url=base_url, prefill_url=prefill_url, decode_url=decode_url
  )


async def run_single_n(
    http_session: aiohttp.ClientSession,
    *,
    model_name: str,
    base_url: str,
    prefill_url: str,
    decode_url: str,
    target_url: str,
    raw_reqs: list[dict[str, Any]],
    all_sampling_specs: list[dict[str, Any]],
    num_sessions: int,
    turns_per_session: int,
    duration_per_n: float,
    cache_hit_ratio: float,
    max_tokens: int,
    ignore_eos: bool,
    dp_size: int,
    max_pinned_slots: int,
    flush_before_warmup: bool,
    warmup_concurrency: int = 8,
    trunks_above_n16: int = 0,
    num_chips: int = 8,
    chip_hourly_usd: float = 5.40,
) -> dict[str, Any]:
  all_trunks, per_dp_bucket_trunks = build_5_or_n_trunk_specs(
      num_sessions=num_sessions, dp_size=dp_size, trunks_above_n16=trunks_above_n16
  )
  trunk_namespace = f"n{num_sessions:03d}_tr{len(all_trunks):03d}"

  if flush_before_warmup:
    await flush_all_caches(
        http_session, base_url=base_url, prefill_url=prefill_url, decode_url=decode_url
    )

  stage1_specs: list[dict[str, Any]] = []
  stage2_specs: list[dict[str, Any]] = []
  per_dp_seen: dict[int, int] = {r: 0 for r in range(dp_size)}
  for sp in all_trunks:
    r = sp["dp_rank"]
    if per_dp_seen[r] < max_pinned_slots:
      stage1_specs.append(sp)
    else:
      stage2_specs.append(sp)
    per_dp_seen[r] += 1

  bucket_trunk_counts = {
      ISL_BUCKETS[b][0]: sum(1 for sp in all_trunks if sp["bucket_idx"] == b)
      for b in range(len(ISL_BUCKETS))
  }
  total_warm_isl = sum(sp["warmup_target_isl_tokens"] for sp in all_trunks)
  pinned_warm_isl = sum(sp["warmup_target_isl_tokens"] for sp in stage1_specs)
  print(
      f"\n[n={num_sessions}] Phase 0 (Warmup — {len(all_trunks)} Distinct Trunks {bucket_trunk_counts}): "
      f"Stage1={len(stage1_specs)} trunks ({int(pinned_warm_isl * cache_hit_ratio):,} prefix tok), "
      f"Stage2={len(stage2_specs)} trunks "
      f"(total target ISL={total_warm_isl:,} tok, {int(cache_hit_ratio*100)}% prefix={int(total_warm_isl * cache_hit_ratio):,} tok)..."
  )
  t_w0 = time.perf_counter()
  warmup_sem = asyncio.Semaphore(max(1, min(warmup_concurrency, len(all_trunks))))
  dp_sems = {r: asyncio.Semaphore(1) for r in range(dp_size)}

  async def _warm_one_trunk(sp: dict[str, Any]) -> dict[str, Any]:
    trunk_id = sp["trunk_id"]
    b_idx = sp["bucket_idx"]
    dp_rank = sp["dp_rank"]
    target_isl = sp["warmup_target_isl_tokens"]
    orig_req = raw_reqs[(b_idx + dp_rank * 5) % len(raw_reqs)]
    b_max_tok = int(orig_req.get("max_tokens") or 8192)
    async with dp_sems[dp_rank], warmup_sem:
      for attempt in range(2):
        w_payload = build_trunk_chat_payload(
            orig_req,
            model_name=model_name,
            trunk_namespace=trunk_namespace,
            session_id=trunk_id,
            trunk_id=trunk_id,
            bucket_idx=b_idx,
            dp_rank=dp_rank,
            target_isl_tokens=target_isl,
            cache_hit_ratio=cache_hit_ratio,
            uid=f"warmup_n{num_sessions}_dp{dp_rank}_k{trunk_id}_a{attempt}",
            max_tokens=1,
            bucket_orig_max_tokens=b_max_tok,
            ignore_eos=False,
            is_warmup=True,
        )
        res = await send_chat_request(
            http_session,
            target_url,
            w_payload,
            {
                "session_id": trunk_id,
                "trunk_id": trunk_id,
                "dp_rank": dp_rank,
                "bucket": sp["bucket"],
                "target_isl": target_isl,
                "turn": -1,
            },
        )
        if res["error"] is None:
          return res
        await asyncio.sleep(1.0)
      return res

  # Warm up each DP rank's trunks in descending bucket order (64K+ -> 32-64K -> 16-32K -> 8-16K -> 1-8K)
  # so `remove_skipped_blocks` shrinks each long-context trunk before smaller trunks are cached.
  stage1_ordered = sorted(stage1_specs, key=lambda sp: (-sp["bucket_idx"], sp["dp_rank"], sp["trunk_id"]))
  stage2_ordered = sorted(stage2_specs, key=lambda sp: (-sp["bucket_idx"], sp["dp_rank"], sp["trunk_id"]))
  w_stage1 = await asyncio.gather(*(_warm_one_trunk(sp) for sp in stage1_ordered))
  w_stage2: list[dict[str, Any]] = []
  if stage2_ordered:
    await asyncio.sleep(0.5)
    w_stage2 = await asyncio.gather(*(_warm_one_trunk(sp) for sp in stage2_ordered))
  w_all = w_stage1 + w_stage2
  w_ok = sum(1 for x in w_all if x["error"] is None)
  w_prompt_sum = sum(x["prompt_tokens"] for x in w_all if x["error"] is None)
  print(
      f"[n={num_sessions}] Phase 0 Warmup complete: {w_ok}/{len(all_trunks)} distinct trunks populated "
      f"({w_prompt_sum:,} total prompt tokens) in {time.perf_counter() - t_w0:.2f}s. Starting measured benchmark..."
  )
  await asyncio.sleep(1.0)

  spec_m0 = await scrape_spec_metrics(http_session, decode_url)
  t_bench_start = time.perf_counter()
  t_window_end = (t_bench_start + duration_per_n) if duration_per_n > 0 else 0.0
  active_in_flight = [0] * len(ISL_BUCKETS)
  dispatched_counts = [0] * len(ISL_BUCKETS)
  bucket_rr_cursors: dict[tuple[int, int], int] = {
      (r, b): 0 for r in range(dp_size) for b in range(len(ISL_BUCKETS))
  }

  def sample_next_bucket_and_trunk(dp_rank: int) -> tuple[int, dict[str, Any], int]:
    d_tot = sum(dispatched_counts)
    b_chosen = max(
        range(len(ISL_BUCKETS)),
        key=lambda b_idx: (
            (d_tot + 1) * ISL_BUCKETS[b_idx][1] - dispatched_counts[b_idx],
            ISL_BUCKETS[b_idx][1],
        ),
    )
    active_in_flight[b_chosen] += 1
    dispatched_counts[b_chosen] += 1
    b_trunks = per_dp_bucket_trunks[dp_rank][b_chosen]
    cur = bucket_rr_cursors[(dp_rank, b_chosen)]
    trunk_sp = b_trunks[cur % len(b_trunks)]
    bucket_rr_cursors[(dp_rank, b_chosen)] = cur + 1
    return b_chosen, trunk_sp, d_tot

  async def session_worker(s_id: int) -> list[dict[str, Any]]:
    dp_rank = s_id % dp_size
    out = []
    t = 0
    while True:
      if duration_per_n > 0:
        if time.perf_counter() >= t_window_end:
          break
      else:
        if t >= turns_per_session:
          break

      b_idx, trunk_sp, req_dispatch_idx = sample_next_bucket_and_trunk(dp_rank)
      samp_spec = all_sampling_specs[req_dispatch_idx % len(all_sampling_specs)]
      bucket = ISL_BUCKETS[b_idx][0]
      target_isl = _bucket_target_tokens(b_idx)
      trunk_id = trunk_sp["trunk_id"]

      bucket_ref_req = raw_reqs[(b_idx + dp_rank * 5) % len(raw_reqs)]
      b_orig_max_tokens = int(bucket_ref_req.get("max_tokens") or 8192)

      uid = f"n{num_sessions}_s{s_id}_t{t}_{int(time.perf_counter() * 1000) % 1000000}"
      payload = build_trunk_chat_payload(
          bucket_ref_req,
          model_name=model_name,
          trunk_namespace=trunk_namespace,
          session_id=s_id,
          trunk_id=trunk_id,
          bucket_idx=b_idx,
          dp_rank=dp_rank,
          target_isl_tokens=target_isl,
          cache_hit_ratio=cache_hit_ratio,
          uid=uid,
          max_tokens=max_tokens,
          bucket_orig_max_tokens=b_orig_max_tokens,
          ignore_eos=ignore_eos,
          is_warmup=False,
          sampling_spec=samp_spec,
      )
      try:
        res = await send_chat_request(
            http_session,
            target_url,
            payload,
            {
                "session_id": s_id,
                "trunk_id": trunk_id,
                "dp_rank": dp_rank,
                "bucket": bucket,
                "target_isl": target_isl,
                "req_max_tokens": payload["max_tokens"],
                "req_temperature": str(samp_spec.get("temperature", "<omitted>")),
                "req_top_p": str(samp_spec.get("top_p", "<omitted>")),
                "num_tools": len(payload.get("tools") or []),
                "turn": t,
            },
        )
      finally:
        active_in_flight[b_idx] = max(0, active_in_flight[b_idx] - 1)
      out.append(res)
      t += 1
    return out

  worker_lists = await asyncio.gather(*(session_worker(s_id) for s_id in range(num_sessions)))
  t_bench_end = time.perf_counter()
  spec_m1 = await scrape_spec_metrics(http_session, decode_url)
  wall_clock = t_bench_end - t_bench_start
  eff_window_s = duration_per_n if duration_per_n > 0 else wall_clock

  spec_decode_info: dict[str, Any] = {}
  if spec_m0 and spec_m1:
    d_drafts = max(0.0, spec_m1["num_drafts"] - spec_m0["num_drafts"])
    d_draft_tokens = max(0.0, spec_m1["num_draft_tokens"] - spec_m0["num_draft_tokens"])
    d_accepted = max(0.0, spec_m1["num_accepted"] - spec_m0["num_accepted"])
    tau = (1.0 + d_accepted / d_drafts) if d_drafts > 0 else 1.0
    d_pos = {
        k: max(0.0, spec_m1["per_pos"].get(k, 0.0) - spec_m0["per_pos"].get(k, 0.0))
        for k in spec_m1["per_pos"]
    }
    per_pos_rates = [round(d_pos.get(k, 0.0) / d_drafts, 4) for k in sorted(d_pos.keys())] if d_drafts > 0 else []
    spec_decode_info = {
        "num_drafts": d_drafts,
        "num_draft_tokens": d_draft_tokens,
        "num_accepted": d_accepted,
        "mean_accept_len_tau": round(tau, 4),
        "per_pos_accept_rates": per_pos_rates,
    }

  results = [r for lst in worker_lists for r in lst]
  ok_results = [r for r in results if r["error"] is None]
  err_results = [r for r in results if r["error"] is not None]
  distinct_trunks_used = len({r["trunk_id"] for r in results})

  window_output_tokens = 0.0
  window_input_tokens = 0.0
  window_uncached_tokens = 0.0
  window_completed_reqs = 0
  for r in ok_results:
    c_times = r.get("chunk_times") or []
    compl = float(r["completion_tokens"] or 0)
    prompt = float(r["prompt_tokens"] or 0)
    cached = float(r["cached_tokens"] or 0)
    uncached = max(0.0, prompt - cached)
    if duration_per_n <= 0 or r["t_end"] <= t_window_end:
      window_output_tokens += compl
      window_input_tokens += prompt
      window_uncached_tokens += uncached
      window_completed_reqs += 1
    elif c_times:
      in_win_chunks = sum(1 for ct in c_times if ct <= t_window_end)
      if in_win_chunks > 0:
        frac = in_win_chunks / len(c_times)
        window_output_tokens += compl * frac
        window_input_tokens += prompt
        window_uncached_tokens += uncached

  win_ok_results = [r for r in ok_results if duration_per_n <= 0 or r["t_end"] <= t_window_end]
  eval_results = win_ok_results if len(win_ok_results) >= max(4, num_sessions // 2) else ok_results

  ttfts = [r["ttft"] for r in eval_results if r["ttft"] is not None]
  e2es = [r["e2e"] for r in eval_results if r["e2e"] is not None]
  per_tok_itls = [r["per_token_itl"] for r in eval_results if r.get("per_token_itl") is not None]
  decode_tps_list = [r["decode_tps"] for r in eval_results if r.get("decode_tps") is not None]
  all_chunk_itls = [itl for r in eval_results for itl in r["itls"]]

  total_prompt = sum(r["prompt_tokens"] for r in ok_results)
  total_cached = sum(r["cached_tokens"] for r in ok_results)
  total_uncached = max(0, total_prompt - total_cached)
  total_compl = sum(r["completion_tokens"] for r in ok_results)
  cache_hit_pct = (100.0 * total_cached / total_prompt) if total_prompt > 0 else 0.0
  tool_call_reqs = sum(
      1 for r in ok_results if r.get("has_tool_calls") or r.get("finish_reason") == "tool_calls"
  )

  bucket_breakdown = {}
  for b_name, _, _, _ in ISL_BUCKETS:
    b_res = [r for r in ok_results if r["bucket"] == b_name]
    b_ttfts = [r["ttft"] for r in b_res if r["ttft"] is not None]
    b_e2es = [r["e2e"] for r in b_res if r["e2e"] is not None]
    b_tps = [r["decode_tps"] for r in b_res if r.get("decode_tps") is not None]
    b_p = sum(r["prompt_tokens"] for r in b_res)
    b_c = sum(r["cached_tokens"] for r in b_res)
    b_out = sum(r["completion_tokens"] for r in b_res)
    bucket_breakdown[b_name] = {
        "count": len(b_res),
        "trunks_configured": bucket_trunk_counts[b_name],
        "mean_isl": (b_p / len(b_res)) if b_res else 0.0,
        "mean_osl": (b_out / len(b_res)) if b_res else 0.0,
        "cache_hit_pct": (100.0 * b_c / b_p) if b_p > 0 else 0.0,
        "median_ttft_s": statistics.median(b_ttfts) if b_ttfts else 0.0,
        "p90_ttft_s": percentile(b_ttfts, 90),
        "median_e2e_s": statistics.median(b_e2es) if b_e2es else 0.0,
        "median_decode_tps": statistics.median(b_tps) if b_tps else 0.0,
        "p95_low_decode_tps": percentile(b_tps, 5),
        "p90_low_decode_tps": percentile(b_tps, 10),
        "p90_hi_decode_tps": percentile(b_tps, 90),
    }

  for r in results:
    r.pop("chunk_times", None)

  win_out_tps = window_output_tokens / eff_window_s if eff_window_s > 0 else 0.0
  win_in_tps = window_input_tokens / eff_window_s if eff_window_s > 0 else 0.0
  win_uncached_tps = window_uncached_tokens / eff_window_s if eff_window_s > 0 else 0.0
  drain_out_tps = total_compl / wall_clock if wall_clock > 0 else 0.0
  drain_in_tps = total_prompt / wall_clock if wall_clock > 0 else 0.0
  drain_uncached_tps = total_uncached / wall_clock if wall_clock > 0 else 0.0

  hourly_usd = num_chips * chip_hourly_usd
  sec_usd = hourly_usd / 3600.0

  def _cost_per_1m(tps: float) -> float:
    return (sec_usd * 1e6 / tps) if tps > 0 else 0.0

  summary = {
      "num_sessions": num_sessions,
      "num_trunks_configured": len(all_trunks),
      "bucket_trunk_counts": bucket_trunk_counts,
      "num_trunks_pinned": len(stage1_specs),
      "distinct_trunks_used": distinct_trunks_used,
      "warmup_trunks_ok": w_ok,
      "turns_per_session": turns_per_session,
      "duration_per_n_s": duration_per_n,
      "total_requests": len(results),
      "successful_requests": len(ok_results),
      "tool_call_requests": tool_call_reqs,
      "window_completed_requests": window_completed_reqs,
      "failed_requests": len(err_results),
      "errors_sample": [r["error"] for r in err_results[:5]],
      "wall_clock_s": wall_clock,
      "steady_state_window_s": eff_window_s,
      "achieved_rps": len(ok_results) / wall_clock if wall_clock > 0 else 0.0,
      "tokens": {
          "mean_prompt_tokens": total_prompt / len(ok_results) if ok_results else 0.0,
          "mean_cached_tokens": total_cached / len(ok_results) if ok_results else 0.0,
          "mean_uncached_tokens": total_uncached / len(ok_results) if ok_results else 0.0,
          "mean_completion_tokens": total_compl / len(ok_results) if ok_results else 0.0,
          "observed_cache_hit_pct": cache_hit_pct,
      },
      "throughput": {
          "window_output_tok_per_s": win_out_tps,
          "window_input_tok_per_s": win_in_tps,
          "window_uncached_prefill_tok_per_s": win_uncached_tps,
          "output_tok_per_s": drain_out_tps,
          "input_tok_per_s": drain_in_tps,
          "uncached_prefill_tok_per_s": drain_uncached_tps,
          "total_tok_per_s": (total_prompt + total_compl) / wall_clock if wall_clock > 0 else 0.0,
      },
      "cost_3yr_cud": {
          "num_chips": num_chips,
          "chip_hourly_usd": chip_hourly_usd,
          "hourly_cluster_usd": hourly_usd,
          "usd_per_1m_output_tokens_window": _cost_per_1m(win_out_tps),
          "usd_per_1m_output_tokens_drain": _cost_per_1m(drain_out_tps),
          "usd_per_1m_input_tokens_window": _cost_per_1m(win_in_tps),
          "usd_per_1m_uncached_input_tokens_window": _cost_per_1m(win_uncached_tps),
          "blended_usd_per_1m_tokens_window": _cost_per_1m(win_out_tps + win_in_tps),
      },
      "client_decode_tps_per_stream": {
          "mean": statistics.mean(decode_tps_list) if decode_tps_list else 0.0,
          "median": statistics.median(decode_tps_list) if decode_tps_list else 0.0,
          "p5_low": percentile(decode_tps_list, 5),
          "p10_low": percentile(decode_tps_list, 10),
          "p90": percentile(decode_tps_list, 90),
      },
      "per_token_itl_ms": {
          "mean": statistics.mean(per_tok_itls) * 1000.0 if per_tok_itls else 0.0,
          "median": statistics.median(per_tok_itls) * 1000.0 if per_tok_itls else 0.0,
          "p90": percentile(per_tok_itls, 90) * 1000.0,
      },
      "ttft_s": {
          "mean": statistics.mean(ttfts) if ttfts else 0.0,
          "median": statistics.median(ttfts) if ttfts else 0.0,
          "p90": percentile(ttfts, 90),
          "p99": percentile(ttfts, 99),
      },
      "chunk_itl_ms": {
          "mean": statistics.mean(all_chunk_itls) * 1000.0 if all_chunk_itls else 0.0,
          "median": statistics.median(all_chunk_itls) * 1000.0 if all_chunk_itls else 0.0,
          "p90": percentile(all_chunk_itls, 90) * 1000.0,
      },
      "e2e_s": {
          "mean": statistics.mean(e2es) if e2es else 0.0,
          "median": statistics.median(e2es) if e2es else 0.0,
          "p90": percentile(e2es, 90),
      },
      "temperature_distribution": dict(
          sorted(
              {
                  k: sum(1 for r in ok_results if r.get("req_temperature") == k)
                  for k in {r.get("req_temperature", "<omitted>") for r in ok_results}
              }.items()
          )
      ),
      "top_p_distribution": dict(
          sorted(
              {
                  k: sum(1 for r in ok_results if r.get("req_top_p") == k)
                  for k in {r.get("req_top_p", "<omitted>") for r in ok_results}
              }.items()
          )
      ),
      "spec_decode": spec_decode_info,
      "buckets": bucket_breakdown,
  }

  mode_label = (
      f"Trunks={len(all_trunks)}, Continuous {duration_per_n:.0f}s Window"
      if duration_per_n > 0
      else f"Trunks={len(all_trunks)}, {turns_per_session} turn(s)/session"
  )
  print("=" * 108)
  print(
      f"RESULTS FOR n={num_sessions} SESSIONS ({mode_label}) | "
      f"TrunksUsed={distinct_trunks_used}/{len(all_trunks)} | "
      f"OK={len(ok_results)}/{len(results)} (ToolCalls={tool_call_reqs}, In-Win={window_completed_reqs}) | "
      f"Win={eff_window_s:.1f}s | DrainWall={wall_clock:.2f}s"
  )
  if err_results:
    print(f"  [ERRORS] {len(err_results)} failed requests; sample: {err_results[0]['error']}")
  print("-" * 108)
  print(
      f"  Mean ISL: {summary['tokens']['mean_prompt_tokens']:,.1f} tok | "
      f"Cached: {summary['tokens']['mean_cached_tokens']:,.1f} tok "
      f"({cache_hit_pct:.2f}%) | "
      f"Mean Output: {summary['tokens']['mean_completion_tokens']:,.1f} tok"
  )
  print(
      f"  Steady-State Window Tput : Output={win_out_tps:,.1f} tok/s | "
      f"Input={win_in_tps:,.1f} tok/s | "
      f"Uncached={win_uncached_tps:,.1f} tok/s"
  )
  print(
      f"  Full-Drain Wall Tput     : Output={drain_out_tps:,.1f} tok/s | "
      f"Input={drain_in_tps:,.1f} tok/s | "
      f"RPS={summary['achieved_rps']:.2f}"
  )
  print(
      f"  3-Yr CUD Cost (${hourly_usd:.2f}/hr) : "
      f"Out(Win)=${summary['cost_3yr_cud']['usd_per_1m_output_tokens_window']:.2f}/1M | "
      f"Out(Drain)=${summary['cost_3yr_cud']['usd_per_1m_output_tokens_drain']:.2f}/1M | "
      f"In(Win)=${summary['cost_3yr_cud']['usd_per_1m_input_tokens_window']:.3f}/1M | "
      f"Blended=${summary['cost_3yr_cud']['blended_usd_per_1m_tokens_window']:.3f}/1M"
  )
  print(
      f"  Client Decode Speed/Strm : Median={summary['client_decode_tps_per_stream']['median']:.2f} tok/s "
      f"(P95_low={summary['client_decode_tps_per_stream']['p5_low']:.2f}, "
      f"P90_low={summary['client_decode_tps_per_stream']['p10_low']:.2f}, "
      f"P90_hi={summary['client_decode_tps_per_stream']['p90']:.2f}) | "
      f"Per-Token ITL={summary['per_token_itl_ms']['median']:.2f}ms "
      f"(StepChunk={summary['chunk_itl_ms']['median']:.2f}ms, Mean={summary['chunk_itl_ms']['mean']:.2f}ms)"
  )
  print(
      f"  Latency                  : Median TTFT={summary['ttft_s']['median']:.2f}s (P90={summary['ttft_s']['p90']:.2f}s) | "
      f"Median E2E={summary['e2e_s']['median']:.2f}s"
  )
  print("  Per-Bucket Breakdown:")
  for b_name, b_stat in bucket_breakdown.items():
    print(
        f"    {b_name:8s} (trunks={b_stat['trunks_configured']:2d}): reqs={b_stat['count']:3d} | "
        f"Mean ISL={b_stat['mean_isl']:7,.0f} | Mean OSL={b_stat['mean_osl']:6,.1f} | "
        f"Hit={b_stat['cache_hit_pct']:5.1f}% | Med TTFT={b_stat['median_ttft_s']:6.2f}s | "
        f"P90 TTFT={b_stat['p90_ttft_s']:6.2f}s | Med TPS/strm={b_stat['median_decode_tps']:5.1f}"
    )
  print("=" * 108)
  return summary


async def wait_for_health(
    http_session: aiohttp.ClientSession,
    prefill_url: str,
    decode_url: str,
) -> None:
  for label, url in [("prefill", prefill_url), ("decode", decode_url)]:
    if not url:
      continue
    health_url = f"{url.rstrip('/')}/health"
    print(f"Waiting for {label} health at {health_url} ...")
    while True:
      try:
        async with http_session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
          if resp.status == 200:
            print(f"  [{label}] Healthy (HTTP 200)!")
            break
      except Exception:
        pass
      await asyncio.sleep(5)


async def main_async(args: argparse.Namespace) -> None:
  session_list = [int(x.strip()) for x in args.sessions.split(",") if x.strip()]
  raw_reqs = load_raw_requests(args.dataset)
  all_sampling_specs = load_all_sampling_specs(args.dataset)
  print(
      f"Loaded {len(raw_reqs)} tool-calling templates and {len(all_sampling_specs)} "
      f"exact sampling specs from {args.dataset} (model={args.model})"
  )

  cur_base_url = args.base_url
  cur_prefill_url = args.prefill_url
  cur_decode_url = args.decode_url

  timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
  connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, force_close=False)

  all_summaries = []
  async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http_session:
    await wait_for_health(
        http_session,
        prefill_url=cur_prefill_url,
        decode_url=cur_decode_url,
    )
    if args.warmup_pass_s > 0:
      await run_micro_xla_warmup_pass(
          http_session,
          model_name=args.model,
          base_url=cur_base_url,
          prefill_url=cur_prefill_url,
          decode_url=cur_decode_url,
          raw_reqs=raw_reqs,
          all_sampling_specs=all_sampling_specs,
          session_list=session_list,
          warmup_pass_s=args.warmup_pass_s,
          cache_hit_ratio=args.cache_hit_rate,
          dp_size=args.dp_size,
      )

    for idx, n in enumerate(session_list):
      eff_trunks = len(
          build_5_or_n_trunk_specs(
              num_sessions=n, dp_size=args.dp_size, trunks_above_n16=args.trunks_above_n16
          )[0]
      )
      target_url = f"{cur_base_url.rstrip('/')}/v1/chat/completions"
      print(
          f"\n>>> [n={n} SESSIONS, TRUNKS={eff_trunks}] ({args.duration_per_n:.0f}s) "
          f"Measured Zero-Compile Benchmark Run (Exact Temps, 50:20:12:9:9) <<<"
      )
      s = await run_single_n(
          http_session,
          model_name=args.model,
          base_url=cur_base_url,
          prefill_url=cur_prefill_url,
          decode_url=cur_decode_url,
          target_url=target_url,
          raw_reqs=raw_reqs,
          all_sampling_specs=all_sampling_specs,
          num_sessions=n,
          turns_per_session=args.turns_per_session,
          duration_per_n=args.duration_per_n,
          cache_hit_ratio=args.cache_hit_rate,
          max_tokens=args.max_tokens,
          ignore_eos=args.ignore_eos,
          dp_size=args.dp_size,
          max_pinned_slots=args.max_pinned_prefix_slots,
          flush_before_warmup=((idx == 0 and not args.no_flush_on_start) or args.flush_each_n),
          warmup_concurrency=args.warmup_concurrency,
          trunks_above_n16=args.trunks_above_n16,
          num_chips=args.num_chips,
          chip_hourly_usd=args.chip_hourly_usd,
      )
      s["pass"] = f"measured_{int(args.duration_per_n)}s_5_or_n_trunks_exact_temps"
      all_summaries.append(s)
      if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
          json.dump(all_summaries, f, indent=2)

  if args.output_json:
    print(f"\nSaved full 5-or-N trunks exact-temps sweep summary to {args.output_json}")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
      description="5-or-N Trunks + Exact Temperatures Empirical ISL Benchmark for DeepSeek-V4.1-Flash"
  )
  p.add_argument("--base-url", type=str, required=True)
  p.add_argument("--prefill-url", type=str, default="")
  p.add_argument("--decode-url", type=str, default="")
  p.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V4.1-Flash")
  p.add_argument("--trunks-above-n16", type=int, default=0)
  p.add_argument("--dataset", type=str, default="/data/requests_mocked.jsonl.xz")
  p.add_argument("--sessions", type=str, default="8,16,32,64,128")
  p.add_argument("--turns-per-session", type=int, default=1)
  p.add_argument("--duration-per-n", type=float, default=210.0)
  p.add_argument("--warmup-pass-s", type=float, default=10.0)
  p.add_argument("--cache-hit-rate", type=float, default=0.80)
  p.add_argument("--max-tokens", type=int, default=0)
  p.add_argument("--dp-size", type=int, default=8)
  p.add_argument("--max-pinned-prefix-slots", type=int, default=16)
  p.add_argument("--num-chips", type=int, default=8)
  p.add_argument("--chip-hourly-usd", type=float, default=5.40)
  p.add_argument("--ignore-eos", action="store_true", default=False)
  p.add_argument("--no-ignore-eos", action="store_false", dest="ignore_eos")
  p.add_argument("--flush-each-n", action="store_true", default=True)
  p.add_argument("--no-flush-each-n", action="store_false", dest="flush_each_n")
  p.add_argument("--no-flush-on-start", action="store_true", default=False)
  p.add_argument("--warmup-concurrency", type=int, default=8)
  p.add_argument("--timeout", type=float, default=3600.0)
  p.add_argument("--output-json", type=str, default="/tmp/dsv41_pd_empirical_isl_summary.json")
  return p.parse_args()


if __name__ == "__main__":
  asyncio.run(main_async(parse_args()))
