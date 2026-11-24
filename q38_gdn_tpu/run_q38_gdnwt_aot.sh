#!/bin/bash
# DEC-FB-2: GDNWT re-run pre-lock CPU job of record (cwd /tmp/src = the fixed tree). Fail-closed; exit = verdict.
# STEP0 versions must equal the image's (jax 0.11.2, libtpu 0.0.50, torch-tpu 0.2.0);
# STEP1 the rank-1 ref reshape guard (tp8, tp2) passes on the fixed tree;
# STEP2 the guard FAILS on the mutant tree (the old `weights_ref.conv.bias.reshape(-1, lanes)[...]` restored);
# STEP3 deviceless TPU7x AOT of window_tile at all 15 on-device gate cases, every step: all compile, window_tile traced at
#       the expected tile;
# STEP4 the same AOT on the mutant tree for the 7 TP8 (conv bias) cases: each FAILS with "Reshape 1D ref is not supported".
cd /tmp/src || exit 3
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu PYTHONPATH=src
echo "== STEP0 versions $(date -u +%H:%M:%SZ)"
python3 - <<'PY'; S0=$?
from importlib import metadata as m
v = {d: m.version(d) for d in ("jax", "jaxlib", "libtpu", "torch-tpu", "torch")}
print("VERSIONS", v)
want = {"jax": "0.11.2", "libtpu": "0.0.50", "torch-tpu": "0.2.0"}
bad = {k: (v[k], w) for k, w in want.items() if v[k] != w}
print("VERSIONS match the image" if not bad else f"VERSIONS MISMATCH {bad}")
raise SystemExit(1 if bad else 0)
PY
echo "STEP0_RC=$S0"
echo "== STEP1 guard on the fixed tree $(date -u +%H:%M:%SZ)"
python3 -m pytest -v -p no:cacheprovider tests/kernels/test_gdn_v3_window_tile.py -k rank1_ref > /tmp/s1.log 2>&1; S1=$?
grep -aE "PASSED|FAILED|ERROR| passed| failed| error" /tmp/s1.log | cut -c1-240 | tail -6; echo "STEP1_RC=$S1"
[ $S1 -ne 0 ] && tail -40 /tmp/s1.log | cut -c1-300
M=/tmp/q38_gdnwt_mut; rm -rf $M; mkdir -p $M; cp -R src tests q38_gdn_tpu $M/
python3 - $M/src/vllm_torchtpu/kernels/gdn/v3/window_tile.py <<'PY'; SM=$?
import sys
p = sys.argv[1]; s = open(p).read()
a = "        bias = weights_ref.conv.bias[...].reshape(-1, lanes)\n"
b = "        bias = weights_ref.conv.bias.reshape(-1, lanes)[...]\n"
assert s.count(a) == 1, s.count(a)
open(p, "w").write(s.replace(a, b)); print("mutant: rank-1 ref reshape of the conv bias restored")
PY
[ $SM -eq 0 ] || { echo "VERDICT FAIL (mutant edit did not apply)"; exit 1; }
echo "== STEP2 guard on the mutant tree must FAIL $(date -u +%H:%M:%SZ)"
(cd $M && PYTHONPATH=src python3 -m pytest -v -p no:cacheprovider tests/kernels/test_gdn_v3_window_tile.py -k rank1_ref) > /tmp/s2.log 2>&1; R2=$?
grep -aE "PASSED|FAILED|rank-1 ref reshape| passed| failed" /tmp/s2.log | cut -c1-240 | tail -6
F2=$(grep -ac "^FAILED.*rank1_ref" /tmp/s2.log)
[ $R2 -ne 0 ] && [ "$F2" = 2 ] && S2=0 || S2=1; echo "STEP2_RC=$S2 (pytest rc $R2, $F2 of 2 failed)"
echo "== STEP3 deviceless AOT, fixed tree, all 15 gate cases $(date -u +%H:%M:%SZ)"
python3 q38_gdn_tpu/q38_gdn_wt_aot.py /tmp/src > /tmp/s3.log 2>&1; S3=$?
grep -aE "^Q38_AOT_VERSIONS|^Q38_GDN_WT_AOT |^Q38_GDN_WT_AOT_CASE|^Q38_GDN_WT_AOT_SUMMARY|CHILD_ERROR" /tmp/s3.log | cut -c1-420
echo "STEP3_RC=$S3"
echo "== STEP4 deviceless AOT, mutant tree, the 7 TP8 cases must FAIL with the Mosaic rejection $(date -u +%H:%M:%SZ)"
TP8C="tp8-verify_full-t2 tp8-verify_partial-t3 tp8-verify_prefill-t2 tp8-verify_odd-t2 tp8-verify_one-t1 tp8-verify_partial-t4 tp8-verify_wide-tdyn"
python3 q38_gdn_tpu/q38_gdn_wt_aot.py $M $TP8C > /tmp/s4.log 2>&1; R4=$?
grep -aE "^Q38_GDN_WT_AOT |^Q38_GDN_WT_AOT_CASE|^Q38_GDN_WT_AOT_SUMMARY|CHILD_ERROR" /tmp/s4.log | cut -c1-420
N4=$(python3 -c "
import json
n = 0
for l in open('/tmp/s4.log'):
    if l.startswith('Q38_GDN_WT_AOT_CASE '):
        pass
seen = {}
for l in open('/tmp/s4.log'):
    if l.startswith('Q38_GDN_WT_AOT '):
        r = json.loads(l.split(' ', 1)[1])
        hit = (not r['compiled']) and 'Reshape 1D ref is not supported' in r.get('error', '')
        seen[r['case']] = seen.get(r['case'], False) or hit
print(sum(seen.values()))
")
[ $R4 -ne 0 ] && [ "$N4" = 7 ] && S4=0 || S4=1; echo "STEP4_RC=$S4 (aot rc $R4, $N4 of 7 TP8 cases fail with the Mosaic rejection)"
echo "== END $(date -u +%H:%M:%SZ) S0=$S0 S1=$S1 S2=$S2 S3=$S3 S4=$S4"
if [ $S0 = 0 ] && [ $S1 = 0 ] && [ $S2 = 0 ] && [ $S3 = 0 ] && [ $S4 = 0 ]; then echo "VERDICT PASS"; exit 0; fi
echo "VERDICT FAIL"; exit 1
