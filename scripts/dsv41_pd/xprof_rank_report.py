#!/usr/bin/env python3
"""按 TPU 设备（= DP rank 的 TensorCore）汇总 xprof 抓到的 device op 耗时。

用法：python xprof_rank_report.py <xprof 目录或 .xplane.pb> [--top 25] [--group 'name=regex' ...]
  - 每个设备：XLA Ops 行的总忙碌时间、各 op（按 hlo 名去掉数字后缀）耗时 top N；
  - --group 把 op 名按正则归类（如 indexer=fp8_paged_mqa|topk），输出"类 × 设备"表，看各 rank 的差异；
  - XLA Modules 行：每个 module 的执行次数和平均耗时（每步一次 forward 时可看各 rank 每步时间）。
需要 PYTHONPATH 指向 vllm_torchtpu/src（用其中的 tools/xplane_pb2）。
"""
import argparse
import collections
import glob
import gzip
import os
import re

from vllm_torchtpu.tools import xplane_pb2


def load_planes(path):
    files = [path] if os.path.isfile(path) else sorted(
        glob.glob(os.path.join(path, "**", "*.xplane.pb*"), recursive=True))
    for f in files:
        data = open(f, "rb").read()
        if f.endswith(".gz"):
            data = gzip.decompress(data)
        xs = xplane_pb2.XSpace()
        xs.ParseFromString(data)
        for p in xs.planes:
            yield os.path.basename(f), p


def norm(name):
    # XLA Ops 的名字是整段 HLO（"%fusion.12 = bf16[...] fusion(...)"），只取 op 名并去掉编号。
    name = name.split(" = ", 1)[0].lstrip("%")
    name = re.sub(r"\.cloned(\.\d+)*", "", name)
    name = re.sub(r"\.refs_\d+_", "", name)
    return re.sub(r"(\.\d+)+(?=(\.|$))", "", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--list-lines", action="store_true")
    a = ap.parse_args()
    groups = [(g.split("=", 1)[0], re.compile(g.split("=", 1)[1])) for g in a.group]
    per_dev = {}
    modules = {}
    for fname, p in load_planes(a.path):
        if not p.name.startswith("/device:TPU"):
            continue
        dev = f"{fname.split('.')[0][:24]}:{p.name}"
        md = p.event_metadata
        if a.list_lines:
            print(dev, [(l.name, len(l.events)) for l in p.lines])
        for line in p.lines:
            if line.name == "XLA Ops":
                acc = per_dev.setdefault(dev, collections.Counter())
                for e in line.events:
                    acc[norm(md[e.metadata_id].name)] += e.duration_ps
            elif line.name == "XLA Modules":
                m = modules.setdefault(dev, collections.defaultdict(list))
                for e in line.events:
                    m[norm(md[e.metadata_id].name)].append(e.duration_ps / 1e9)
    for dev in sorted(per_dev):
        acc = per_dev[dev]
        tot = sum(acc.values()) / 1e9
        print(f"\n== {dev}  busy {tot:.1f} ms")
        for name, ps in acc.most_common(a.top):
            print(f"  {ps/1e9:9.2f} ms {ps/1e9/tot:6.1%}  {name[:110]}")
        for mod, ds in sorted(modules.get(dev, {}).items(), key=lambda kv: -sum(kv[1]))[:6]:
            print(f"  [module] n={len(ds):4d} mean={sum(ds)/len(ds):8.2f} ms max={max(ds):8.2f}  {mod[:80]}")
    if groups:
        devs = sorted(per_dev)
        print("\n" + "group".ljust(16) + "".join(d.split("TPU:")[-1].rjust(9) for d in devs))
        for gname, rx in groups + [("(total)", re.compile(""))]:
            row = [sum(ps for n, ps in per_dev[d].items() if rx.search(n)) / 1e9 for d in devs]
            print(gname.ljust(16) + "".join(f"{v:9.1f}" for v in row))


if __name__ == "__main__":
    main()
