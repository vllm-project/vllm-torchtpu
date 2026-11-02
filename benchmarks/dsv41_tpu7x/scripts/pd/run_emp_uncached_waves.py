#!/usr/bin/env python3
"""Runs the exact Empirical ISL (50:20:12:9:9) Uncached Prefill (0% Cache Hit, c=8, OSL=1)
per-bucket waves (8 reqs/bucket) and mixed 5-bucket sweeps (40 reqs and 128 reqs) on 1P1D,
capturing both pure 1P node latency/throughput (from PERF PROXY prefill_send -> prefill_recv)
and E2E 1P1D TTFT/throughput (including Stage-3 DCN KV transfer + 1D 128-token tail replay).
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from pathlib import Path
import aiohttp

import bench_empirical_isl_dsv41 as B


def parse_proxy_log_window(log_path: str, start_byte: int) -> list[dict]:
    p = Path(log_path)
    if not p.exists():
        return []
    with open(p, "rb") as f:
        f.seek(start_byte)
        text = f.read().decode("utf-8", errors="replace")
    send_ts = {}
    recv_ts = {}
    pf_dur_ms = {}
    first_chunk_ms = {}
    ttft_ms = {}
    for line in text.splitlines():
        if "PERF PROXY prefill_send" in line:
            m = re.search(r"req_id=(\S+)\s+ts=([0-9.]+)", line)
            if m:
                send_ts[m.group(1)] = float(m.group(2))
        elif "PERF PROXY prefill_recv" in line:
            m = re.search(r"req_id=(\S+)\s+ts=([0-9.]+)\s+dur_ms=([0-9.]+)", line)
            if m:
                recv_ts[m.group(1)] = float(m.group(2))
                pf_dur_ms[m.group(1)] = float(m.group(3))
        elif "PERF PROXY first_chunk" in line:
            m = re.search(
                r"req_id=(\S+)\s+ts=([0-9.]+)\s+dur_from_decode_send_ms=([0-9.]+)\s+ttft_ms=([0-9.]+)",
                line,
            )
            if m:
                first_chunk_ms[m.group(1)] = float(m.group(3))
                ttft_ms[m.group(1)] = float(m.group(4))
    out = []
    for rid, s_ts in send_ts.items():
        if rid in recv_ts:
            out.append({
                "req_id": rid,
                "send_ts": s_ts,
                "recv_ts": recv_ts[rid],
                "pf_s": pf_dur_ms[rid] / 1000.0,
                "decode_first_s": first_chunk_ms.get(rid, 0.0) / 1000.0,
                "ttft_s": ttft_ms.get(rid, 0.0) / 1000.0,
            })
    return out


async def main() -> None:
    base_url = "http://127.0.0.1:8000"
    target_url = f"{base_url}/v1/chat/completions"
    proxy_log = "/dev/shm/pd_proxy.log"
    dataset = "/data/requests_mocked.jsonl.xz"
    model_name = "deepseek-ai/DeepSeek-V4.1-Flash"
    raw_reqs = B.load_raw_requests(dataset)

    timeout = aiohttp.ClientTimeout(total=1800, sock_read=1800)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, force_close=False)

    summary = {"per_bucket_c8": {}, "mixed_sweeps_c8": {}}

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # 1. Per-Bucket Uncached Prefill (c = 8, 8 Reqs per Bucket)
        all_trunks_40, _ = B.build_5_or_n_trunk_specs(num_sessions=8, dp_size=8)
        for b_idx in reversed(range(len(B.ISL_BUCKETS))):
            b_name = B.ISL_BUCKETS[b_idx][0]
            b_specs = [sp for sp in all_trunks_40 if sp["bucket_idx"] == b_idx]
            await B.flush_all_caches(
                session, base_url=base_url, prefill_url="", decode_url=""
            )
            start_byte = Path(proxy_log).stat().st_size if Path(proxy_log).exists() else 0

            async def _run_one(sp: dict, tag: str) -> dict:
                trunk_id = sp["trunk_id"]
                dp_rank = sp["dp_rank"]
                target_isl = sp["warmup_target_isl_tokens"]
                orig_req = raw_reqs[(b_idx + dp_rank * 5) % len(raw_reqs)]
                b_max_tok = int(orig_req.get("max_tokens") or 8192)
                payload = B.build_trunk_chat_payload(
                    orig_req,
                    model_name=model_name,
                    trunk_namespace=tag,
                    session_id=trunk_id,
                    trunk_id=trunk_id,
                    bucket_idx=b_idx,
                    dp_rank=dp_rank,
                    target_isl_tokens=target_isl,
                    cache_hit_ratio=0.95,
                    uid=f"uncached_{tag}_dp{dp_rank}_k{trunk_id}",
                    max_tokens=1,
                    bucket_orig_max_tokens=b_max_tok,
                    ignore_eos=False,
                    is_warmup=True,
                )
                return await B.send_chat_request(
                    session,
                    target_url,
                    payload,
                    {
                        "session_id": trunk_id,
                        "trunk_id": trunk_id,
                        "dp_rank": dp_rank,
                        "bucket": b_name,
                        "target_isl": target_isl,
                        "turn": -1,
                    },
                )

            t0 = time.perf_counter()
            res_list = await asyncio.gather(*(_run_one(sp, f"wave_{b_name}") for sp in b_specs))
            e2e_wall_s = time.perf_counter() - t0
            await asyncio.sleep(0.2)
            proxy_recs = parse_proxy_log_window(proxy_log, start_byte)

            ok = [r for r in res_list if r["error"] is None]
            wave_tok = sum(r["prompt_tokens"] for r in ok)
            mean_isl = wave_tok / len(ok) if ok else 0.0
            e2e_ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
            pf_s_list = [r["pf_s"] for r in proxy_recs]
            dc_s_list = [r["decode_first_s"] for r in proxy_recs]
            if proxy_recs:
                pf_wall_s = max(r["recv_ts"] for r in proxy_recs) - min(r["send_ts"] for r in proxy_recs)
            else:
                pf_wall_s = e2e_wall_s

            pf_tput = wave_tok / pf_wall_s if pf_wall_s > 0 else 0.0
            e2e_tput = wave_tok / e2e_wall_s if e2e_wall_s > 0 else 0.0

            rec = {
                "bucket": b_name,
                "n_reqs": len(ok),
                "mean_isl": round(mean_isl, 1),
                "wave_tokens": wave_tok,
                "pf_wall_s": round(pf_wall_s, 3),
                "e2e_wall_s": round(e2e_wall_s, 3),
                "pf_lat_med_s": round(statistics.median(pf_s_list), 3) if pf_s_list else 0.0,
                "pf_lat_mean_s": round(statistics.mean(pf_s_list), 3) if pf_s_list else 0.0,
                "kv_xfer_1d_med_s": round(statistics.median(dc_s_list), 3) if dc_s_list else 0.0,
                "e2e_ttft_med_s": round(statistics.median(e2e_ttfts), 3) if e2e_ttfts else 0.0,
                "e2e_ttft_p90_s": round(B.percentile(e2e_ttfts, 90), 3) if e2e_ttfts else 0.0,
                "pf_node_tok_s": round(pf_tput, 1),
                "pf_chip_tok_s": round(pf_tput / 4.0, 1),
                "e2e_node_tok_s": round(e2e_tput, 1),
                "e2e_chip_tok_s": round(e2e_tput / 4.0, 1),
            }
            summary["per_bucket_c8"][b_name] = rec
            print(f"[BUCKET {b_name:8s}] {json.dumps(rec)}", flush=True)

        # 2. Mixed 5-Bucket Uncached Sweeps (40 Reqs and 128 Reqs at c = 8)
        for num_sess, label in [(8, "mixed_40_reqs"), (128, "mixed_128_reqs")]:
            all_trunks, _ = B.build_5_or_n_trunk_specs(num_sessions=num_sess, dp_size=8)
            await B.flush_all_caches(
                session, base_url=base_url, prefill_url="", decode_url=""
            )
            start_byte = Path(proxy_log).stat().st_size if Path(proxy_log).exists() else 0
            sem = asyncio.Semaphore(8)
            dp_sems = {r: asyncio.Semaphore(1) for r in range(8)}
            ordered = sorted(
                all_trunks, key=lambda sp: (-sp["bucket_idx"], sp["dp_rank"], sp["trunk_id"])
            )

            async def _warm_sp(sp: dict, tag: str) -> dict:
                trunk_id = sp["trunk_id"]
                b_idx = sp["bucket_idx"]
                dp_rank = sp["dp_rank"]
                target_isl = sp["warmup_target_isl_tokens"]
                orig_req = raw_reqs[(b_idx + dp_rank * 5) % len(raw_reqs)]
                b_max_tok = int(orig_req.get("max_tokens") or 8192)
                async with dp_sems[dp_rank], sem:
                    payload = B.build_trunk_chat_payload(
                        orig_req,
                        model_name=model_name,
                        trunk_namespace=tag,
                        session_id=trunk_id,
                        trunk_id=trunk_id,
                        bucket_idx=b_idx,
                        dp_rank=dp_rank,
                        target_isl_tokens=target_isl,
                        cache_hit_ratio=0.95,
                        uid=f"mixed_{tag}_dp{dp_rank}_k{trunk_id}",
                        max_tokens=1,
                        bucket_orig_max_tokens=b_max_tok,
                        ignore_eos=False,
                        is_warmup=True,
                    )
                    return await B.send_chat_request(
                        session,
                        target_url,
                        payload,
                        {
                            "session_id": trunk_id,
                            "trunk_id": trunk_id,
                            "dp_rank": dp_rank,
                            "bucket": sp["bucket"],
                            "target_isl": target_isl,
                            "turn": -1,
                        },
                    )

            t0 = time.perf_counter()
            res_list = await asyncio.gather(*(_warm_sp(sp, label) for sp in ordered))
            e2e_wall_s = time.perf_counter() - t0
            await asyncio.sleep(0.2)
            proxy_recs = parse_proxy_log_window(proxy_log, start_byte)

            ok = [r for r in res_list if r["error"] is None]
            total_tok = sum(r["prompt_tokens"] for r in ok)
            mean_isl = total_tok / len(ok) if ok else 0.0
            if proxy_recs:
                pf_wall_s = max(r["recv_ts"] for r in proxy_recs) - min(r["send_ts"] for r in proxy_recs)
            else:
                pf_wall_s = e2e_wall_s

            pf_tput = total_tok / pf_wall_s if pf_wall_s > 0 else 0.0
            e2e_tput = total_tok / e2e_wall_s if e2e_wall_s > 0 else 0.0
            rec = {
                "label": label,
                "n_reqs": len(ok),
                "mean_isl": round(mean_isl, 1),
                "total_tokens": total_tok,
                "pf_wall_s": round(pf_wall_s, 3),
                "e2e_wall_s": round(e2e_wall_s, 3),
                "pf_node_tok_s": round(pf_tput, 1),
                "pf_chip_tok_s": round(pf_tput / 4.0, 1),
                "e2e_node_tok_s": round(e2e_tput, 1),
                "e2e_chip_tok_s": round(e2e_tput / 4.0, 1),
            }
            summary["mixed_sweeps_c8"][label] = rec
            print(f"[MIXED {label:16s}] {json.dumps(rec)}", flush=True)

    Path("/dev/shm/dsv41_emp_uncached_waves_p2.json").write_text(json.dumps(summary, indent=2))
    print("DONE_EMP_UNCACHED_WAVES", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
