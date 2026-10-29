#!/usr/bin/env python3
"""按 token 数汇总 Stage-3 KV 传输耗时（P → D）。

    python3 transfer_stats.py <decode_log> <prefill_log> [--since "2026-10-28 06:00:00"]

耗时：decode 日志 raiden_stage3_receiver_complete 的 reshard_e2e_latency_ms
     （D 开始拉取 → D 的 worker 发现接收完成，含 P 发送、网络、D 端重组）。
字节：prefill 日志里同一 uuid 的 declared_bytes 之和（各 KV pool 的实际传输字节）。
用来看传输是否随长度线性增长：耗时 / tokens² 若在长段近似常数，说明有 O(n²) 的处理。
"""
import argparse
import collections
import json
import re
import statistics
from datetime import datetime

TS = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d)")


def after(line, since):
    if since is None:
        return True
    m = TS.search(line)
    return m is None or datetime.strptime(f"{since.year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S") >= since


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decode_log")
    ap.add_argument("prefill_log")
    ap.add_argument("--since", default=None, help="只统计这个时间之后的记录（日志里的时间，UTC）")
    args = ap.parse_args()
    since = datetime.fromisoformat(args.since) if args.since else None

    lat = {}
    for line in open(args.decode_log, errors="replace"):
        if "raiden_stage3_receiver_complete" in line and after(line, since):
            d = json.loads(line[line.index("{"):])
            if d.get("reshard_e2e_latency_ms") is not None:
                lat[d["uuid"]] = (d["num_tokens"], d["reshard_e2e_latency_ms"])
    nbytes = {}
    for line in open(args.prefill_log, errors="replace"):
        if '"declared_bytes"' in line:
            try:
                d = json.loads(line[line.index("{"):])
            except (ValueError, json.JSONDecodeError):
                continue
            if d.get("uuid") is not None:
                nbytes[d["uuid"]] = sum(d["declared_bytes"].values())

    rows = collections.defaultdict(list)
    for uuid, (tokens, ms) in lat.items():
        rows[tokens].append((ms, nbytes.get(uuid)))
    print(f"{'tokens':>9} {'n':>4} {'latency_ms':>11} {'MB':>7} {'MB/s':>6} {'B/token':>7} {'ms/Mtok^2':>10}")
    for tokens in sorted(rows):
        ms = statistics.median(x[0] for x in rows[tokens])
        bs = [x[1] for x in rows[tokens] if x[1]]
        b = statistics.median(bs) if bs else None
        mb = f"{b / 1e6:.0f}" if b else "-"
        rate = f"{b / 1e6 / (ms / 1000):.0f}" if b else "-"
        per_tok = f"{b / tokens:.0f}" if b else "-"
        quad = ms / (tokens / 1e6) ** 2
        print(f"{tokens:>9} {len(rows[tokens]):>4} {ms:>11.0f} {mb:>7} {rate:>6} {per_tok:>7} {quad:>10.0f}")


if __name__ == "__main__":
    main()
