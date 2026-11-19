#!/bin/bash
# q38 L3-HCX CPU proof (CDK CPU-job body; 0 TPU chips): HCX (VLLM_TPU_HC_XLA_MAX_ROWS), port of q38mm 20e6fcf23 onto
# FINAL 5d368f166. Run from the root of the extracted job tree (git archive of q38-l3-hcx + q38_plugins/ + COMMIT +
# ctl/ = git archive of 5d368f166 src tests pyproject.toml). Pattern of q38_ragged/run_ra_proof.sh.
# Legs (each prints one Q38_L3HCX_LEG line; fail-closed: exit 0 only if every leg passes):
#   P0  versions, unpacked src hash == WANT_SRC, ctl src hash == WANT_CTL, plugin md5s, SRC-AT-START
#   L1  tests/q38_l3_hcx/test_hcx_rows.py on the real packages (image mode): >= 38 PASSED, none skipped
#   L2  test_hcx_ladder.py's compile-hash test (the real compute_tpu_compilation_hash): PASSED, none skipped
#   L2b test_hcx_ladder.py's compile ladder (vLLM compile pipeline + shape-variant hand-over), INFORMATIONAL: PASS, or BLOCKED when
#       the same harness fails on ctl (5d368f166) with the same image error (v1 run j-73461306: '_C' has no 'rotary_embedding' in
#       this vllm 0.31.0+empty image); a ladder failure while ctl's harness passes FAILS L2
#   L3  regression (skipped with L3HCX_SKIP_L3=1: proven PASS on the same src c578cdbf6d by j-73461306): the existing HC / compile-key / shape-variant suites on this tree vs ctl (5d368f166); no test that
#       passes on ctl may fail here
#   L4  mutants (q38_l3_hcx_mutrun.py --mode image --tests test_hcx_rows.py): every one KILLED, controls pass, copy restored
#   L5  SRC-AT-END == SRC-AT-START
set -u
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
unset JAX_PLATFORMS
ROOT=$(pwd)
WANT_SRC=c578cdbf6d
WANT_CTL=4cca600214
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
PT="python3 -m pytest -q -rA -p no:cacheprovider -o addopts= -p q38_cpu_device -p q38_tpu_device_gate_v2"
CPU="env JAX_PLATFORMS=cpu VLLM_TARGET_DEVICE=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8"
cnt() { grep -aE '[0-9]+ (passed|failed|error)' "$1" | tail -1; }
leg() { echo "Q38_L3HCX_LEG $1 $([ "$2" = 0 ] && echo PASS || echo FAIL) $3"; }
TOUCHED="src/vllm_torchtpu/kernels/qwen4_exp/hc_decode.py src/vllm_torchtpu/models/vllm/qwen4_exp/hyperconnection.py src/vllm_torchtpu/compilation/tpu_compiler.py src/vllm_torchtpu/envs.py tests/q38_l3_hcx/test_hcx_rows.py tests/q38_l3_hcx/test_hcx_ladder.py tests/q38_l3_hcx/hcx_load.py tests/q38_l3_hcx/q38_l3_hcx_mutrun.py tests/q38_l3_hcx/data/hyperconnection_5d368f166.py.txt"
echo "Q38_L3HCX start $(date -u +%FT%TZ) commit $(cat COMMIT 2>/dev/null) nproc $(nproc 2>/dev/null)"

# P0
python3 - <<'PY'; R0=$?
import importlib.metadata as md
def v(n):
    try:
        return md.version(n)
    except md.PackageNotFoundError:
        return None
print("Q38_L3HCX versions python-torch %s jax %s jaxlib %s torch-tpu %s vllm %s pytest %s" % (
    v("torch"), v("jax"), v("jaxlib"), v("torch-tpu") or v("torch_tpu"), v("vllm"), v("pytest")))
raise SystemExit(0 if v("vllm") and v("torch") else 1)
PY
ST=$(srch "$ROOT"); SC=$(srch "$ROOT/ctl")
echo "Q38_L3HCX src $ST (want $WANT_SRC) ctl $SC (want $WANT_CTL)"
[ "$ST" = "$WANT_SRC" ] && [ "$SC" = "$WANT_CTL" ] || R0=1
PM1=$(md5sum < q38_plugins/q38_cpu_device.py | cut -c1-8); PM2=$(md5sum < q38_plugins/q38_tpu_device_gate_v2.py | cut -c1-8)
echo "Q38_L3HCX plugins q38_cpu_device $PM1 (want b452635b) q38_tpu_device_gate_v2 $PM2 (want 31bfd921)"
[ "$PM1" = b452635b ] && [ "$PM2" = 31bfd921 ] || R0=1
md5sum $TOUCHED > /tmp/l3hcx_src_start.txt || R0=1; cat /tmp/l3hcx_src_start.txt
leg P0 $R0 "versions+hash+plugins"

run_pt() {  # tag, min PASSED, pytest args...
  local tag=$1 min=$2; shift 2
  (cd "$ROOT" && timeout 2400 $CPU Q38_L3_HCX_MODE=image PYTHONPATH=src:q38_plugins $PT "$@" > /tmp/l3hcx_$tag.log 2>&1); local r=$?
  local np=$(grep -ac '^PASSED' /tmp/l3hcx_$tag.log)
  local ns=$(grep -aE '[0-9]+ skipped' /tmp/l3hcx_$tag.log | tail -1)
  echo "Q38_L3HCX $tag rc=$r $(cnt /tmp/l3hcx_$tag.log); PASSED lines $np (min $min); skipped: ${ns:-none}; $(grep -ao 'Q38_CPU_DEVICE.*' /tmp/l3hcx_$tag.log | head -1); $(grep -ao 'Q38_TPU_DEVICE_GATE.*' /tmp/l3hcx_$tag.log | head -1)"
  [ $r = 0 ] && [ "$np" -ge "$min" ] && [ -z "$ns" ] || { tail -80 /tmp/l3hcx_$tag.log; return 1; }
  return 0
}
run_pt L1 38 tests/q38_l3_hcx/test_hcx_rows.py; R1=$?; leg L1 $R1 "HCX routing / identity / bit-equality / compile key on the real packages"
run_pt L2 1 -k compile_hash tests/q38_l3_hcx/test_hcx_ladder.py; R2=$?
# L2b: the ladder needs the vLLM compile harness of tests/compilation/test_shape_variants.py; check that harness on ctl too
(cd "$ROOT" && timeout 1200 $CPU Q38_L3_HCX_MODE=image PYTHONPATH=src:q38_plugins $PT -s -k ladder tests/q38_l3_hcx/test_hcx_ladder.py > /tmp/l3hcx_L2b.log 2>&1); rlad=$?
cp -a q38_plugins ctl/ 2>/dev/null
(cd "$ROOT/ctl" && timeout 1200 $CPU PYTHONPATH=src:q38_plugins $PT -k one_trace_serves_every_bucket tests/compilation/test_shape_variants.py > /tmp/l3hcx_L2b_ctl.log 2>&1); rctl=$?
grep -a 'HCX-LADDER' /tmp/l3hcx_L2b.log
LADX=$(grep -aoc "has no attribute 'rotary_embedding'" /tmp/l3hcx_L2b.log); CTLX=$(grep -aoc "has no attribute 'rotary_embedding'" /tmp/l3hcx_L2b_ctl.log)
echo "Q38_L3HCX L2b ladder rc=$rlad $(cnt /tmp/l3hcx_L2b.log) image-error lines $LADX | ctl harness rc=$rctl $(cnt /tmp/l3hcx_L2b_ctl.log) image-error lines $CTLX"
if [ $rlad = 0 ] && [ "$(grep -ac 'HCX-LADDER' /tmp/l3hcx_L2b.log)" -ge 3 ]; then LAD=PASS
elif [ $rctl != 0 ] && [ "$LADX" -ge 1 ] && [ "$CTLX" -ge 1 ]; then LAD=BLOCKED
else LAD=FAIL; R2=1; tail -60 /tmp/l3hcx_L2b.log; fi
leg L2 $R2 "compile hash keyed only when set; ladder $LAD (BLOCKED = the same vLLM compile harness fails on 5d368f166 in this image)"

# L3 regression: this tree vs ctl (5d368f166)
if [ "${L3HCX_SKIP_L3:-0}" = 1 ]; then R3=0; leg L3 0 "skipped: proven PASS on the same src c578cdbf6d by j-73461306 (no new failure vs 5d368f166)"; else
REG=""
for f in tests/models/vllm/test_qwen4_exp_hc_engagement.py tests/models/vllm/test_qwen4_exp_math.py \
         tests/models/vllm/test_qwen4_exp_hc_defer.py tests/models/vllm/test_qwen4_exp_sequence_parallel.py \
         tests/kernels/test_qwen4_exp_hc_decode.py tests/kernels/test_qwen4_exp_hc_prefill_fused.py \
         tests/compilation/test_shape_variants.py tests/compilation/test_bucket_filter.py \
         tests/compilation/test_tpu_compiler.py tests/compilation/test_tpu_compile_key_registration.py; do
  [ -f "$f" ] && [ -f "ctl/$f" ] && REG="$REG $f"
done
echo "Q38_L3HCX L3 files:$REG"
fails() { grep -aE '^(FAILED|ERROR) ' "$1" | sed -E 's/^(FAILED|ERROR) //; s/ - .*//' | sort -u; }
cp -a q38_plugins ctl/ 2>/dev/null
(cd "$ROOT" && timeout 3600 $CPU PYTHONPATH=src:q38_plugins $PT $REG > /tmp/l3hcx_L3_tip.log 2>&1); r3t=$?
(cd "$ROOT/ctl" && timeout 3600 $CPU PYTHONPATH=src:q38_plugins $PT $REG > /tmp/l3hcx_L3_ctl.log 2>&1); r3c=$?
fails /tmp/l3hcx_L3_tip.log > /tmp/l3hcx_L3_tip.f; fails /tmp/l3hcx_L3_ctl.log > /tmp/l3hcx_L3_ctl.f
NEW=$(comm -23 /tmp/l3hcx_L3_tip.f /tmp/l3hcx_L3_ctl.f)
NPT=$(grep -ac '^PASSED' /tmp/l3hcx_L3_tip.log); NPC=$(grep -ac '^PASSED' /tmp/l3hcx_L3_ctl.log)
echo "Q38_L3HCX L3 tip rc=$r3t $(cnt /tmp/l3hcx_L3_tip.log) PASSED $NPT | ctl rc=$r3c $(cnt /tmp/l3hcx_L3_ctl.log) PASSED $NPC"
echo "Q38_L3HCX L3 failing on ctl: $(tr '\n' ' ' < /tmp/l3hcx_L3_ctl.f)"; echo "Q38_L3HCX L3 NEW on tip: ${NEW:-none}"
R3=0; [ -z "$NEW" ] && [ "$NPT" -ge "$NPC" ] && [ "$NPT" -ge 1 ] || R3=1
[ $R3 = 0 ] || tail -60 /tmp/l3hcx_L3_tip.log
leg L3 $R3 "regression vs 5d368f166"
fi

# L4 mutants
(cd "$ROOT" && timeout 3600 $CPU PYTHONPATH=src:q38_plugins Q38_L3_HCX_PYTEST_PLUGINS=1 \
  python3 tests/q38_l3_hcx/q38_l3_hcx_mutrun.py --mode image --tests test_hcx_rows.py > /tmp/l3hcx_L4.log 2>&1); R4=$?
cat /tmp/l3hcx_L4.log | tail -40
grep -aq '^Q38_L3_HCX_MUTANTS .* PASS$' /tmp/l3hcx_L4.log || R4=1
leg L4 $R4 "$(grep -a '^Q38_L3_HCX_MUTANTS' /tmp/l3hcx_L4.log | tail -1)"

# L5
md5sum $TOUCHED > /tmp/l3hcx_src_end.txt; cmp -s /tmp/l3hcx_src_start.txt /tmp/l3hcx_src_end.txt; R5=$?
[ "$(srch "$ROOT")" = "$WANT_SRC" ] || R5=1
leg L5 $R5 "SRC-AT-END"

V=PASS; for r in $R0 $R1 $R2 $R3 $R4 $R5; do [ "$r" = 0 ] || V=FAIL; done
echo "Q38_L3HCX_SUMMARY P0=$R0 L1=$R1 L2=$R2 L3=$R3 L4=$R4 L5=$R5 $V $(date -u +%FT%TZ)"
[ $V = PASS ]
