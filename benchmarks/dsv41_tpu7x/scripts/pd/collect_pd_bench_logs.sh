#!/usr/bin/env bash
# Post-benchmark log collector for 1P1D Empirical ISL sweeps (run on the prefill pod,
# with the decode pod reachable via kubectl from the driver for its log).
#
# Usage (in prefill pod):  bash /tmp/collect_pd_bench_logs.sh /tmp/<bench_tag>
# Produces <outdir>/{proxy_transport_failures.txt, proxy_tail.log, bench_requests_summary.txt}
# and prints the failure correlation needed to root-cause
# "ClientPayloadError: TransferEncodingError" seen by the harness (E0.1).
set -euo pipefail
OUT="${1:-/tmp/pd_bench_collect}"
mkdir -p "$OUT"

PROXY_LOG=/dev/shm/pd_proxy.log
if [[ -f "$PROXY_LOG" ]]; then
  # Mid-stream decode transport failures logged by toy_proxy_server._log_unsafe_transport_failure
  grep -n -i "UNSAFE\|TransportError\|RemoteProtocolError\|ReadError\|HTTPStatusError\|Traceback\|Error occurred in disagg" \
    "$PROXY_LOG" > "$OUT/proxy_transport_failures.txt" || true
  tail -n 2000 "$PROXY_LOG" > "$OUT/proxy_tail.log"
  echo "[collect] proxy failure lines: $(wc -l < "$OUT/proxy_transport_failures.txt")"
fi

# Harness per-request records written next to --output-json (E0: *_requests.jsonl)
shopt -s nullglob
for f in /tmp/*_requests.jsonl; do
  python3 - "$f" >> "$OUT/bench_requests_summary.txt" <<'EOF'
import json, sys, collections
p = sys.argv[1]
recs = [json.loads(l) for l in open(p)]
fail = [r for r in recs if r.get("error")]
ok = [r for r in recs if not r.get("error")]
print(f"== {p}: total={len(recs)} ok={len(ok)} failed={len(fail)}")
if fail:
    by_b = collections.Counter(r.get("bucket") for r in fail)
    by_r = collections.Counter(r.get("dp_rank") for r in fail)
    tok = [r.get("completion_tokens") or 0 for r in fail]
    e2e = [r.get("e2e") or 0 for r in fail]
    had_tc = sum(1 for r in fail if r.get("has_tool_calls"))
    print(f"   failed by bucket: {dict(by_b)}")
    print(f"   failed by dp_rank: {dict(sorted(by_r.items()))}")
    print(f"   failed completion_tokens: min={min(tok)} med={sorted(tok)[len(tok)//2]} max={max(tok)}")
    print(f"   failed e2e_s: min={min(e2e):.1f} med={sorted(e2e)[len(e2e)//2]:.1f} max={max(e2e):.1f}")
    print(f"   failed with tool_calls delta seen before failure: {had_tc}/{len(fail)}")
    print(f"   finish_reason seen: {collections.Counter(r.get('finish_reason') for r in fail)}")
    print(f"   error kinds: {collections.Counter((r.get('error') or '')[:60] for r in fail)}")
EOF
done
[[ -f "$OUT/bench_requests_summary.txt" ]] && cat "$OUT/bench_requests_summary.txt"
echo "[collect] done -> $OUT"
echo "[collect] NEXT (from driver): kubectl exec <decode-pod> -- bash -c \"grep -n -B3 -A25 'Traceback\\|ERROR\\|tool_parser\\|deepseek_v41' /dev/shm/pd_decode.log | tail -n 400\" > $OUT/decode_errors.txt"
