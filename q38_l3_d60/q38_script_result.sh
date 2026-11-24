# q38_script_result.sh: source from a BENCH=script run body so the CDK job status is honest.
# L3 15:2xZ 11-13, bash port of q38_xfer4/q38_script_result.py (XFER4 lane; credit).
#
# Why (C, infx-qwen38-flash-bench render, script branch): the recipe runs BENCH_CMD, takes its exit code as rc, then
# scores the job from $CDK_OUTPUT_DIR/q38f-bench/script/run<NN>_<arm>/result.json for each arm in ${ARMS:-A}
# (run01_A by default). A run counts only if that file has "ok": true and a non-empty "cases" list. With no file,
# runs_ok = 0 and rc is forced to 1, so a body that finished with exit 0 still shows Failed (RA1, ARB1, K3, KB-D24,
# T84g RFOLD, T84h PRIME). The recipe publishes only known metric keys from cases[0]; extra keys are harmless.
#
# Use:
#   source q38_script_result.sh
#   q38_result_clear                      # first thing in the body: replaces any stale result with ok=false
#   ... run the bench and the reader ...
#   q38_result_from_verdict "$rr" "$O/x_verdict.txt" "l3prime"   # last thing in the body
#   exit "$rr"
# ok = (rc == 0) and a line starting "VERDICT" exists and it is not "VERDICT INVALID" or "VERDICT VOID".
# KEEP / NOT KEEP / FAIL-EXACT are valid measurements, so they score ok (Succeeded). Lanes whose readers print
# something else can call the primitive: q38_result_write <1|0> <label> [detail].

q38_result_paths() {
  local root="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38f-bench/script" arms="${ARMS:-A}" i=0 a
  local IFS=','
  for a in $arms; do
    i=$((i + 1))
    printf '%s/run%02d_%s/result.json\n' "$root" "$i" "$a"
  done
}

q38_result_clear() {  # overwrite with a not-ok placeholder (no deletes): a body that dies later still scores Failed
  q38_result_write 0 "cleared" "q38_result_clear placeholder; the body did not reach its result line" > /dev/null
}

q38_result_write() {  # $1 = 1 (ok) or 0, $2 = label, $3 = detail (optional, cut to 300 chars)
  local p ok=false
  [ "$1" = 1 ] && ok=true
  while read -r p; do
    mkdir -p "$(dirname "$p")"
    python3 -c 'import json,sys; json.dump({"ok": sys.argv[2] == "true", "cases": [{"label": sys.argv[3], "detail": sys.argv[4][:300]}]}, open(sys.argv[1], "w"))' \
      "$p" "$ok" "$2" "${3:-}"
    echo "Q38_SCRIPT_RESULT {\"path\": \"$p\", \"ok\": $ok}"
  done < <(q38_result_paths)
}

q38_result_from_verdict() {  # $1 = rc of the deciding step, $2 = verdict file, $3 = label
  local v="" ok=0
  [ -f "$2" ] && v=$(grep -m1 '^VERDICT' "$2")
  if [ "$1" -eq 0 ] && [ -n "$v" ] && ! printf '%s' "$v" | grep -q -E '^VERDICT (INVALID|VOID)'; then ok=1; fi
  q38_result_write "$ok" "$3" "rc=$1; ${v:-no VERDICT line in $2}"
}
