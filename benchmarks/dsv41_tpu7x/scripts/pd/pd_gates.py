#!/usr/bin/env python3
"""DSv4.1 1P1D correctness gates + a small TTFT/throughput probe (runs in the prefill pod).

Gates (PD_DESIGN.md "Correctness tests"):
  * 2+2 x16 through the proxy (the decode driver's payload), all must contain "4"
  * USA 50-states prompt through the proxy (recorded; known to fail until decodefix lands)
  * greedy first-64-token equality: each prompt through the proxy (P/D) vs. the same
    prompt sent straight to the decode server with no kv_transfer_params (decode-local
    prefill). Prompts span 1 page to many pages of the 1024-token MLA group and cross the
    128-token SWA window so the window trim is exercised.
  * a TTFT/throughput probe through the proxy (streaming), and the same against decode.

Writes one JSON with everything; exits 0 iff the hard gates pass (2+2 and token equality).
Stdlib only.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import statistics
import sys
import time
import urllib.request

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming",
]
PARAS = [
    "In modern distributed computer systems and machine learning compilers, high-bandwidth "
    "memory hierarchy design governs end-to-end throughput across large transformer models. ",
    "Mixture-of-Experts architectures route each token to a sparse subset of specialized "
    "feed-forward networks using learned gating scores and load-balancing auxiliary biases. ",
    "Speculative decoding accelerates autoregressive generation by proposing multiple draft "
    "tokens per step with a lightweight draft module and verifying all candidates in parallel. ",
    "Block-wise low-precision quantization formats such as MXFP4 and FP8 E4M3 reduce static "
    "weight footprints and key-value cache bandwidth requirements by up to fourfold. ",
]


def post(url: str, payload: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def chat(base: str, content: str, max_tokens: int) -> str:
    out = post(f"{base}/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": max_tokens, "temperature": 0.0,
    })
    return out["choices"][0]["message"]["content"].strip()


def complete(base: str, prompt, max_tokens: int) -> dict:
    out = post(f"{base}/v1/completions", {
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "min_tokens": max_tokens,
        "ignore_eos": True, "temperature": 0.0, "skip_special_tokens": False,
        "return_token_ids": True,
    })
    ch = out["choices"][0]
    return {"text": ch.get("text", ""), "token_ids": ch.get("token_ids"),
            "prompt_tokens": out.get("usage", {}).get("prompt_tokens")}


def scrape_spec_metrics(decode_base: str) -> dict:
    num_drafts, num_acc, per_pos = 0.0, 0.0, {}
    try:
        with urllib.request.urlopen(f"{decode_base}/metrics", timeout=15) as r:
            txt = r.read().decode("utf-8", errors="replace")
        for line in txt.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith("vllm:spec_decode_num_drafts_total") or line.startswith("vllm:spec_decode_num_drafts{"):
                num_drafts += float(line.split()[-1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
                m = re.search(r'position="(\d+)"', line)
                if m:
                    p = int(m.group(1))
                    per_pos[p] = per_pos.get(p, 0.0) + float(line.split()[-1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total{") or line.startswith("vllm:spec_decode_num_accepted_tokens{"):
                num_acc += float(line.split()[-1])
    except Exception:
        pass
    return {"num_drafts": num_drafts, "num_accepted": num_acc, "per_pos": per_pos}


def stream_one(base: str, prompt_ids: list[int], max_tokens: int, t_bench0: float = 0.0) -> dict:
    payload = {"model": MODEL, "prompt": prompt_ids, "max_tokens": max_tokens,
               "min_tokens": max_tokens, "ignore_eos": True, "temperature": 0.0,
               "stream": True}
    req = urllib.request.Request(f"{base}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, n_chunks, t_last, t_prev = None, 0, t0, t0
    itls: list[float] = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            obj = json.loads(line[6:])
            if obj.get("choices") and obj["choices"][0].get("text"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - t_prev)
                t_prev = now
                n_chunks += 1
                t_last = now
    e2e = t_last - t0
    return {
        "ttft_s": ttft, "e2e_s": e2e, "chunks": n_chunks,
        "tpot_ms": (e2e - ttft) * 1000 / max(1, max_tokens - 1) if ttft else None,
        "start_offset_s": t0 - t_bench0,
        "first_token_offset_s": (t0 - t_bench0 + ttft) if ttft is not None else None,
        "comp_offset_s": t_last - t_bench0,
        "itls": itls,
    }


def build_prompt_ids(tok, n_tokens: int, salt: str, treatise: bool = False) -> list[int]:
    hdr = tok.encode(
        "<｜begin▁of▁sentence｜><｜User｜>The secret passcode is 739204.\nBackground notes:\n",
        add_special_tokens=False,
    ).ids
    body = tok.encode("".join(PARAS), add_special_tokens=False).ids
    if treatise:
        tail = tok.encode(
            f"\nQuestion ({salt}): Provide a comprehensive, rigorous step-by-step architectural "
            "treatise covering memory hierarchy, attention compression, expert routing, and "
            "hardware execution.<｜Assistant｜></think>",
            add_special_tokens=False,
        ).ids
    else:
        tail = tok.encode(
            "\nReply with ONLY the 6-digit secret passcode followed by a "
            "comma-separated list of the 50 US states in alphabetical order, nothing else."
            "<｜Assistant｜></think>",
            add_special_tokens=False,
        ).ids
    mid_len = max(1, n_tokens - len(hdr) - len(tail))
    mid = (body * (mid_len // max(1, len(body)) + 2))[:mid_len]
    return hdr + mid + tail


def first_divergence(a, b):
    if a is None or b is None:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def bench(base: str, prompts: list[list[int]], out_len: int, conc: int, decode_base: str | None = None) -> dict:
    m0 = scrape_spec_metrics(decode_base) if decode_base else None
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda p: stream_one(base, p, out_len, t0), prompts))
    dur = time.perf_counter() - t0
    m1 = scrape_spec_metrics(decode_base) if decode_base else None

    nat_accept_len = 1.0
    per_pos_rates = []
    if m0 and m1:
        d_drafts = m1["num_drafts"] - m0["num_drafts"]
        d_acc = m1["num_accepted"] - m0["num_accepted"]
        if d_drafts > 0:
            nat_accept_len = round(1.0 + d_acc / d_drafts, 4)
            d_pos = {k: m1["per_pos"].get(k, 0.0) - m0["per_pos"].get(k, 0.0) for k in m1["per_pos"]}
            per_pos_rates = [round(d_pos.get(k, 0.0) / d_drafts, 4) for k in sorted(d_pos.keys())]

    ttfts = [r["ttft_s"] for r in res if r["ttft_s"] is not None]
    tpots = [r["tpot_ms"] for r in res if r["tpot_ms"] is not None]
    first_abs = [r["first_token_offset_s"] for r in res if r.get("first_token_offset_s") is not None]
    comp_abs = [r["comp_offset_s"] for r in res if r.get("comp_offset_s") is not None]

    prefill_window_s = max(first_abs) if first_abs else dur
    prefill_active_tok_s = round(len(prompts) * len(prompts[0]) / max(1e-6, prefill_window_s), 2)

    interior_tok_s = None
    int_steps_s = None
    step_ms = None
    if len(first_abs) >= conc and len(comp_abs) >= conc:
        t_win_start = max(first_abs[:conc])
        t_win_end = min(comp_abs[-conc:])
        if t_win_end - t_win_start > 0.5:
            win_tokens = 0.0
            for r in res:
                f_t = r.get("first_token_offset_s")
                itls = r.get("itls") or []
                if f_t is None or not itls:
                    continue
                tok_per_ch = max(1, out_len - 1) / len(itls)
                t_cur = f_t
                for dt in itls:
                    t_cur += dt
                    if t_win_start <= t_cur <= t_win_end:
                        win_tokens += tok_per_ch
            interior_tok_s = round(win_tokens / (t_win_end - t_win_start), 2)
            int_steps_s = round(interior_tok_s / max(1.0, nat_accept_len), 2)
            if int_steps_s > 0:
                step_ms = round(conc * 1000.0 / int_steps_s, 2)

    # Strip verbose per-chunk itls before saving JSON
    clean_res = [{k: v for k, v in r.items() if k != "itls"} for r in res]
    wall_tok_s = round(len(prompts) * out_len / dur, 2)
    wall_steps_s = round(wall_tok_s / max(1.0, nat_accept_len), 2)
    return {
        "n": len(prompts), "concurrency": conc, "prompt_tokens": len(prompts[0]),
        "out_len": out_len, "duration_s": round(dur, 3),
        "ttft_mean_s": round(statistics.mean(ttfts), 4) if ttfts else None,
        "ttft_p50_s": round(statistics.median(ttfts), 4) if ttfts else None,
        "ttft_max_s": round(max(ttfts), 4) if ttfts else None,
        "tpot_mean_ms": round(statistics.mean(tpots), 3) if tpots else None,
        "output_tok_s": wall_tok_s,
        "input_tok_s": round(len(prompts) * len(prompts[0]) / dur, 2),
        "prefill_active_window_s": round(prefill_window_s, 4),
        "prefill_active_input_tok_s": prefill_active_tok_s,
        "natural_mean_accept_len": nat_accept_len,
        "per_pos_accept_rates": per_pos_rates,
        "steady_state_interior_window_tok_s": interior_tok_s,
        "steady_state_interior_step_throughput_steps_s": int_steps_s,
        "steady_state_step_ms": step_ms,
        "steady_state_interior_throughput_at_accept_3p5_tok_s": round(int_steps_s * 3.5, 2) if int_steps_s else None,
        "steady_state_interior_throughput_at_accept_4p25_tok_s": round(int_steps_s * 4.25, 2) if int_steps_s else None,
        "steady_state_interior_throughput_at_accept_5p0_tok_s": round(int_steps_s * 5.0, 2) if int_steps_s else None,
        "wall_clock_step_throughput_steps_s": wall_steps_s,
        "per_request": clean_res,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="http://127.0.0.1:8000")
    ap.add_argument("--decode", required=True, help="http://<decode-ip>:9400")
    ap.add_argument("--tokenizer", default="/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json")
    ap.add_argument("--out", default="/dev/shm/pd_gates.json")
    ap.add_argument("--eq-lengths", default="64,700,1500,5000,20000")
    ap.add_argument("--eq-tokens", type=int, default=64)
    ap.add_argument("--bench-n", type=int, default=16)
    ap.add_argument("--bench-conc", type=int, default=8)
    ap.add_argument("--bench-len", type=int, default=8192)
    ap.add_argument("--bench-out", type=int, default=128)
    ap.add_argument("--skip-bench", action="store_true")
    a = ap.parse_args()
    res: dict = {"proxy": a.proxy, "decode": a.decode, "started": time.time()}

    def save():
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=2)
        os.replace(tmp, a.out)

    # --- 2+2 x16 through the proxy (decode driver payload), plus decode-local reference.
    q = [f"Question {i + 1}: what is 2+2? Answer with the single number only." for i in range(16)]
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        pd_ans = list(ex.map(lambda s: chat(a.proxy, s, 16), q))
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        local_ans = list(ex.map(lambda s: chat(a.decode, s, 16), q))
    res["gate_2p2"] = {"pd_answers": pd_ans, "local_answers": local_ans,
                       "pd_passed": all("4" in x for x in pd_ans),
                       "local_passed": all("4" in x for x in local_ans)}
    print(f"[GATE-2+2] pd_passed={res['gate_2p2']['pd_passed']} pd={pd_ans!r}", flush=True)
    print(f"[GATE-2+2] local_passed={res['gate_2p2']['local_passed']} local={local_ans!r}", flush=True)
    save()

    # --- USA prompt (recorded, not asserted: decode multi-step text is wrong on 1d02e4a).
    usa_q = "what is usa? how many states it has? list all the states by name"
    m_usa0 = scrape_spec_metrics(a.decode)
    usa_pd = chat(a.proxy, usa_q, 512)
    m_usa1 = scrape_spec_metrics(a.decode)
    usa_local = chat(a.decode, usa_q, 512)
    d_usa_drafts = m_usa1["num_drafts"] - m_usa0["num_drafts"]
    d_usa_acc = m_usa1["num_accepted"] - m_usa0["num_accepted"]
    usa_tau = round(1.0 + (d_usa_acc / d_usa_drafts if d_usa_drafts > 0 else 0.0), 4)
    res["gate_usa"] = {
        "pd_output": usa_pd, "local_output": usa_local,
        "pd_states": sum(s.lower() in usa_pd.lower() for s in US_STATES),
        "local_states": sum(s.lower() in usa_local.lower() for s in US_STATES),
        "pd_equals_local": usa_pd == usa_local,
        "pd_mean_accept_len": usa_tau,
    }
    print(f"[GATE-USA] pd_states={res['gate_usa']['pd_states']}/50 "
          f"local_states={res['gate_usa']['local_states']}/50 "
          f"pd==local={res['gate_usa']['pd_equals_local']} tau={usa_tau} "
          f"pd_tail={usa_pd[-160:]!r}", flush=True)
    save()

    # --- greedy first-N-token equality, P/D vs decode-local (sequential, batch of 1).
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    eq = []
    for n in [int(x) for x in a.eq_lengths.split(",") if x]:
        ids = build_prompt_ids(tok, n, f"eq{n}")
        pd = complete(a.proxy, ids, a.eq_tokens)
        local = complete(a.decode, ids, a.eq_tokens)
        div = first_divergence(pd["token_ids"], local["token_ids"])
        same = (pd["token_ids"] == local["token_ids"]) if pd["token_ids"] is not None \
            else (pd["text"] == local["text"])
        eq.append({"prompt_tokens": len(ids), "equal": same, "first_divergence": div,
                   "pd_text": pd["text"], "local_text": local["text"],
                   "pd_token_ids": pd["token_ids"], "local_token_ids": local["token_ids"]})
        print(f"[GATE-EQ] n={len(ids)} equal={same} first_div={div} "
              f"pd={pd['text'][:80]!r} local={local['text'][:80]!r}", flush=True)
        save()
    res["gate_token_equality"] = {"n_tokens": a.eq_tokens, "cases": eq,
                                  "all_equal": all(c["equal"] for c in eq)}
    save()

    # --- TTFT / throughput probe (P/D through the proxy vs decode-local).
    if not a.skip_bench:
        prompts = [build_prompt_ids(tok, a.bench_len, f"bench{i}") for i in range(a.bench_n)]
        res["bench_pd"] = bench(a.proxy, prompts, a.bench_out, a.bench_conc, decode_base=a.decode)
        b = res["bench_pd"]
        print(f"[BENCH-PD] n={b['n']} c={b['concurrency']} in={b['prompt_tokens']} out={b['out_len']} "
              f"ttft_mean={b['ttft_mean_s']}s p50={b['ttft_p50_s']}s tpot={b['tpot_mean_ms']}ms "
              f"tau={b['natural_mean_accept_len']} out_tok/s={b['output_tok_s']} "
              f"int_tok/s={b['steady_state_interior_window_tok_s']} step_ms={b['steady_state_step_ms']} "
              f"in_tok/s={b['input_tok_s']} active_in_tok/s={b['prefill_active_input_tok_s']}", flush=True)
        save()
        prompts = [build_prompt_ids(tok, a.bench_len, f"benchL{i}") for i in range(a.bench_n)]
        res["bench_local"] = bench(a.decode, prompts, a.bench_out, a.bench_conc, decode_base=a.decode)
        b = res["bench_local"]
        print(f"[BENCH-LOCAL] n={b['n']} c={b['concurrency']} in={b['prompt_tokens']} out={b['out_len']} "
              f"ttft_mean={b['ttft_mean_s']}s p50={b['ttft_p50_s']}s tpot={b['tpot_mean_ms']}ms "
              f"tau={b['natural_mean_accept_len']} out_tok/s={b['output_tok_s']}", flush=True)
        save()
        # High-concurrency disaggregated decode steady-state scale probe (c=64, 1 request per DP rank * 8 = 8 per rank).
        prompts_c64 = [build_prompt_ids(tok, 1024, f"c64_{i}", treatise=True) for i in range(64)]
        res["bench_pd_c64"] = bench(a.proxy, prompts_c64, 512, 64, decode_base=a.decode)
        b64 = res["bench_pd_c64"]
        print(f"[BENCH-PD-C64] n={b64['n']} c={b64['concurrency']} in={b64['prompt_tokens']} out={b64['out_len']} "
              f"ttft_mean={b64['ttft_mean_s']}s p50={b64['ttft_p50_s']}s tpot={b64['tpot_mean_ms']}ms "
              f"tau={b64['natural_mean_accept_len']} out_tok/s={b64['output_tok_s']} "
              f"int_tok/s={b64['steady_state_interior_window_tok_s']} "
              f"@3.5={b64['steady_state_interior_throughput_at_accept_3p5_tok_s']} "
              f"step_ms={b64['steady_state_step_ms']}", flush=True)
        save()

    res["hard_gates_passed"] = bool(res["gate_2p2"]["pd_passed"]
                                    and res["gate_token_equality"]["all_equal"])
    res["finished"] = time.time()
    save()
    print(f"[PD-GATES] hard_gates_passed={res['hard_gates_passed']}", flush=True)
    return 0 if res["hard_gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
