#!/usr/bin/env python3
"""Compare TPU_RAIDEN_DSV4_DEBUG_CHECKSUMS page digests between P and D logs.

The producer logs `dsv4_debug_digests` (role=producer) when it registers a send
and the consumer (role=consumer) when the receive completes, both keyed by the
producer request id. For every request present on both sides this prints, per
pool tag, how many pages match on their live bytes and which pages differ.

    python3 dsv4_digest_diff.py <prefill.log> <decode.log> [req_id_substring]
"""
import json
import re
import sys


def events(path, role):
    out = {}
    for line in open(path, errors="replace"):
        if '"dsv4_debug_digests"' not in line or f'"role": "{role}"' not in line:
            continue
        m = re.search(r"(\{.*\})\s*$", line)
        if not m:
            continue
        try:
            ev = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        out[ev["req_id"]] = ev
    return out


def main():
    p_log, d_log = sys.argv[1], sys.argv[2]
    want = sys.argv[3] if len(sys.argv) > 3 else ""
    prod, cons = events(p_log, "producer"), events(d_log, "consumer")
    common = [r for r in cons if r in prod and want in r]
    print(f"producer events {len(prod)}, consumer events {len(cons)}, matched {len(common)}")
    for req in common:
        p, d = prod[req]["digests"], cons[req]["digests"]
        print(f"\n== {req}  P tokens={prod[req]['num_tokens']}  D tokens={cons[req]['num_tokens']}")
        for tag in sorted(set(p) | set(d)):
            pp = {e["page"]: e for e in p.get(tag, [])}
            dd = {e["page"]: e for e in d.get(tag, [])}
            pages = sorted(set(pp) | set(dd))
            bad = [
                pg for pg in pages
                if pg not in pp or pg not in dd or pp[pg]["live"] != dd[pg]["live"]
            ]
            status = "OK " if not bad else "BAD"
            detail = ""
            if bad:
                detail = " differing pages " + ",".join(
                    f"{pg}{'' if pg in pp else '(D only)'}{'' if pg in dd else '(P only)'}"
                    for pg in bad[:12]
                )
                last = pages[-1]
                detail += f" | last page {last} live_bytes P={pp.get(last, {}).get('live_bytes')} D={dd.get(last, {}).get('live_bytes')}"
            print(f"  {status} {tag:34s} pages={len(pages):4d}{detail}")


if __name__ == "__main__":
    main()
