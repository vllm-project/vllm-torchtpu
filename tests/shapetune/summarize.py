# SPDX-License-Identifier: Apache-2.0
"""Summarize shapetune logs: python summarize.py <log or jsonl>...

Per (kernel, bucket): default us, best us, gain, best config, and whether the
best config's output digest equals the default's (bitwise).
"""

import collections
import json
import sys


def rows(paths):
    for p in paths:
        for line in open(p, errors="replace"):
            line = line.strip()
            if line.startswith("ST "):
                line = line[3:]
            if line.startswith("{") and '"kernel"' in line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def main(paths):
    groups = collections.OrderedDict()
    for r in rows(paths):
        key = (r["kernel"], json.dumps(r["bucket"], sort_keys=True))
        groups.setdefault(key, []).append(r)
    for (kernel, bucket), rs in groups.items():
        ok = [r for r in rs if r["us"] is not None]
        errs = len(rs) - len(ok)
        d = next((r for r in ok if r["default"]), None)
        if not ok:
            print(f"{kernel}\t{bucket}\tall {errs} configs failed")
            continue
        best = min(ok, key=lambda r: r["us"])
        same = [r for r in ok if d and r["digest"] == d["digest"]]
        best_bw = min(same, key=lambda r: r["us"]) if same else None
        dus = d["us"] if d else float("nan")
        print(f"{kernel}\t{bucket}\tdefault {dus:.1f}\tbest {best['us']:.1f} "
              f"({100 * (1 - best['us'] / dus):+.1f}%)\t{json.dumps(best['config'])}\t"
              f"bitwise={'yes' if d and best['digest'] == d['digest'] else 'NO'}"
              + (f"\tbest_bitwise {best_bw['us']:.1f} {json.dumps(best_bw['config'])}"
                 if best_bw and best_bw is not best else "")
              + (f"\terrs={errs}" if errs else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
