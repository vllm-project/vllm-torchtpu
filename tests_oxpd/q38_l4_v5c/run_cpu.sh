#!/bin/bash
# q38 L4 v5 composition (q38-l4-v5c) CPU proof (CDK CPU pod only; never the Mac or the VM). V5COMP-BUILD (helper of L4-E).
# Usage (from the unpacked bundle dir holding head/ base/ q38_l3_regress_perfile.sh):
#   bash head/tests/q38_l4_v5c/run_cpu.sh smoke         P0 P0b P1 P2 P3 P4 P6
#   bash head/tests/q38_l4_v5c/run_cpu.sh mut LO HI     P0 P0b P5 (composition mutants LO..HI) P6
#   bash head/tests/q38_l4_v5c/run_cpu.sh rp A|B        P0 P0b P7 (per-file regression, half A or B of RP_FILES) P6
#   bash head/tests/q38_l4_v5c/run_cpu.sh h3k           P0 P0b P8 (H3 kernel / wiring / pack suites + H3 regression legs) P6
#   bash head/tests/q38_l4_v5c/run_cpu.sh h3c A|B       P0 P0b P9 (H3 compose: ATTNWIRE's T2 subset + v6 composition) P6
# v6 (q38-l4-v6c2, V6COMP-BUILD): head = q38-l4-v6c cd3d204d7 + T1-K1FUSE (q38-l4-k1fuse f6d018c7e) + L1-H3 (q38-L1-h3
#   f8c7da033's commits after ecf4cc34a) + ATTNWIRE (q38-l4-attnwire 6a1e7901b .. 42ae764d9) + the v6 composition commits.
#   FOLDFIX (2fde9f5e8, the manager hands a folded DraftAsmResult / K1Result to the runner): P0 diff set 24 entries
#   (+ runner/speculative_decoding_manager.py), P0b 31 modules; P4f FOLDFIX's real-path suite 16 and flag-off 3, and
#   test_v6c_foldfix_k1.py 4 (K1's take through the real manager); P7 with L3's regress_perfile v2 (4fb6f596, pinned by
#   the bundle's run script).
#   (before FOLDFIX) P0 diff set 23 entries; P0b 29 modules (+ k1fuse; H3's modules are origin-checked by P8 / P9's staged guard);
#   P2 flags 5 (F5: the two v6 configurations); P3k K1FUSE's suites; P4 test_v6c_k1.py 24 (K1 x DRAFT-CAP / VRNG /
#   SEEDFIX / KEYLITE / l4_agree); the H3 legs run in ATTNWIRE's environment (VLLM_TARGET_DEVICE=cpu, L1's pinned
#   plugins q38_cpu_device / q38_tpu_device_gate_v2 / q38_staged_guard from the bundle's plugins/).
# head = q38-l4-v5c (src EXPECT_SRC) = q38-L1-v4 ecf4cc34a + VPSHOST stage 1 / SAMPFB-TOKEN / T-1(i) + DRAFT-ASM + VFY-ONE
#        + DRAFT1-B DH-B / DH-D + DRAFT-2 (c) / EMB-REP-TGT + AGREE / KEYLITE + DRAFT-CAP + VRNG-VPS / S2 + LMH-STREAM
#        stage 0 + SEEDFIX + the composition commits; base = ecf4cc34a (EXPECT_BASE_SRC).
# PASS rules, fixed before the run:
# P0  head src == EXPECT_SRC, base src == EXPECT_BASE_SRC; the src diff set is exactly WANT_DIFF (20 entries).
# P0b import origin (cd <tree>; PYTHONPATH=src): vllm_torchtpu and the 17 touched modules resolve inside head/src (18/18),
#     the 10 that exist on the base inside base/src (10/10); with PYTHONPATH = an empty dir all 28 read FALLBACK.
# P1  tests/q38_l4_v5c/test_v5c_flagoff.py (Q38_V5C_BASE_SRC=base/src): exactly 13 passed (chained flag-off AST identity
#     of the 10 changed files vs ecf4cc34a with the allowances counted; 786f7bac2 resolves away; bites).
# P2  compile-key registration (whole file): 0 failed, > 7 passed; tests/q38_l4_v5c/test_v5c_flags.py exactly 4 passed
#     (flag table, LEVERS union, render-grep exemption on the intended v5 decode env, on-values).
# P3  each piece's own suites on the composed tree, exact counts, 0 failed / skipped / errors:
#     a  VPSHOST test_l4_vpshost.py 58 (+ its census line counts); its compile-identity census: base COMPILED lines all on
#        head, head's extra lines exactly {_vfy_one_impl, _vfy_one_rng_impl, _vps_verify_rng_impl} (VFY-ONE's and
#        VRNG-VPS's new programs), no LEVERREAD;
#     b  VFY-ONE test_vfy_one.py 78, test_vfy_agree.py 15, test_vfy_dha_compose.py 17; D83 test_vps_ro_fold.py 86;
#     c  DRAFT-ASM units 56, propose 51, multirank 8, flagoff 1 (its 3 AST tests vs 3d2f53ec9 deselected: P1 replaces);
#     d  DRAFT1-B units 27, stagea 42, multirank 6, dhd 42, flagoff 1 (its 4 tests vs 9b8d39a74 deselected: P1 replaces);
#     e  DRAFT-2 (c) / EMB-REP-TGT test_draft_embed_repl.py + test_target_embed_repl.py 27; q38_draftemb_fit.py PASS;
#     f  AGREE / KEYLITE test_l4agree_units 17, test_l4agree_multirank 17, test_l4agree_flagoff 2 (its 4 tests vs
#        2e166829a / 8344dce32 deselected: P1 replaces), test_vfy_keylite 23, test_vfy_one with KEYLITE=1 78;
#     j  SEEDFIX test_seedfix.py + test_seedfix_vrng.py -k "not distribution" 52 (the composed branch's suites, ported to the
#        AGREE clock; the primary's ecf4cc34a-only AST test deselected);
#     i  LMH-STREAM stage 0 test_lmh_hidden_major.py 25 (its 4 AST tests vs 2e166829a deselected: P1 replaces); the v5
#        HBM fit q38_v5c_fit.py PASS (DRAFT-2 (c) table + LMH copy + L47 before pool sizing: still 98,304 blocks);
#     h  DRAFT-CAP test_draftcap.py 46 (+ its marker line counts), test_draftcap_flagoff.py 3 (its draft_one.py flag-off
#        identity vs DRAFT1-B 29ea1e98d, the bite and the flag row; its 2 byte-identity tests vs 29ea1e98d deselected).
#     g  VRNG-VPS / S2 tests/q38_l4_samp2v/ -k "not v1_regime" 100 (+ its census line counts), on l4_agree.
#     (VFY-ONE's test_vfy_flagoff.py vs f24edcc94 and test_vfy_v5_flagoff.py vs 8344dce32 do not apply: P1 replaces.)
# P4  composition: test_v5c_runner.py 15, test_v5c_drafter.py 11, test_v5c_union.py 4, test_v5c_emb_agree.py 6 (exact,
#     0 failed / skipped; it proves the AGREE x DRAFT-2 (c) gather fix on 8 simulated ranks), test_v5c_vrng_levers.py
#     5 (VRNG-VPS x VFY-ONE with sampsnap + statsasync on, against the stock VRNG chain), test_v5c_keylite_vrng.py 3
#     (composition fix 3: VRNG windows on / off over 300 steps after the real warm-up, KEYLITE off / on, never cold).
# P5  (mut) q38_v5c_mutrun.py: control ok, origin INSIDE, 'Q38_V5C_MUT PASS killed N/N of N range LO-HI of TOTAL'.
# P7  (rp) base screen of the RP_FILES half (4 at a time): a KEY file counting 0 on base FAILS, a RUN file counting 0 is
#     EXCLUDED and named; then L3's regress_perfile v2 (4fb6f596) HEAD vs BASE over the rest. P7's pytest runs with
#     --tb=short on both trees (screen and regress_perfile read the same $PT): under the CPU device fixture (conftest
#     f7d4e6e36) pytest's long traceback renders frame arguments (repr_args -> saferepr) and an argument's repr in
#     tests/spec_decode/test_eagle3.py aborts the process (SIGABRT; w15 rp B j-6711f631 P7-key, w17 j-e6bd2e3c D1 alone).
# P6  head src unchanged by the run.  Last line: Q38_V5C_SUMMARY PASS|FAIL mode=<mode> [failed steps].
set -u
ulimit -c 0   # no core files: aborting pytest processes dumped core and the pod was evicted at 20Gi ephemeral (w17)
MODE=${1:-smoke}
W=$(pwd); H=$W/head; B=$W/base; Q=$H/tests/q38_l4_v5c
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
for v in $(env | grep -oE '^VLLM_TPU_(L4|L6|DRAFT_ASM|VPS_VFY|DRAFT_EMBED|TARGET_EMBED)[A-Z0-9_]*'); do unset "$v"; done
unset VLLM_TPU_VPS_FUSED_READ_OFFSETS VLLM_TPU_SERVED_AB VLLM_TPU_HOST_AB_PERIOD_MS Q38MM_SRC_ROOT Q38_VPSHOST_EXPECT_ROOT
export Q38_V5C_BASE_SRC=$B/src Q38_V5C_BASE_ENVS=$B/src/vllm_torchtpu/envs.py
unset VLLM_TPU_K1FUSE VLLM_TPU_K1FUSE_AB VLLM_TPU_K1FUSE_CHECK VLLM_TPU_MK_DRAFT_ATTN VLLM_TARGET_DEVICE XLA_FLAGS
PL=$W/plugins; export Q38_V6C_PLUGINS=$PL
PTN="python3 -m pytest -q -p no:cacheprovider --noconftest -rfEs"   # logging kept (caplog tests)
PTC="python3 -m pytest -q -p no:cacheprovider -p no:logging -rfEs"
PT=$PTC; CPUENV="JAX_PLATFORMS=cpu PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1"
FAIL=""; L=/tmp/v5c_logs; mkdir -p $L
srch() { (cd "$1" && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum | sha1sum | cut -c1-10); }
summ() { grep -aE '^=*[ ]*([0-9]+ (passed|failed|error|errors|skipped|deselected)(, )?)+.* in [0-9.]+s' "$1" | tail -1; }
count() { local n; n=$(summ "$1" | grep -oE "[0-9]+ $2" | grep -oE '^[0-9]+' | head -1); echo "${n:-0}"; }
exact() { [ "$(count $1 passed)" = "$2" ] && [ "$(count $1 failed)" = 0 ] && [ "$(count $1 skipped)" = 0 ] \
  && [ "$(count $1 error)" = 0 ] && [ "$(count $1 errors)" = 0 ]; }

HS=$(srch "$H"); BS=$(srch "$B")
echo "P0 $(date -u +%FT%TZ) mode=$MODE $(python3 -c 'import sys, torch, importlib.metadata as m; print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", m.version("vllm"))' 2>&1 | tail -1)"
echo "P0 host $(hostname) nproc $(nproc)"
echo "P0 head src $HS (expect $(cat $Q/EXPECT_SRC)); base src $BS (expect $(cat $Q/EXPECT_BASE_SRC))"
[ "$HS" = "$(cat $Q/EXPECT_SRC)" ] && [ "$BS" = "$(cat $Q/EXPECT_BASE_SRC)" ] || FAIL="$FAIL P0-src"
DIFF=$(diff -rq "$B/src" "$H/src" 2>&1 | sed "s#$W/##g" | LC_ALL=C sort)
echo "$DIFF" | sed 's/^/P0 diff /'
WANT=$(LC_ALL=C sort "$Q/WANT_DIFF")
[ "$DIFF" = "$WANT" ] && echo "P0 diff set OK ($(echo "$WANT" | wc -l | tr -d ' ') entries)" || { echo "P0 diff set MISMATCH"; FAIL="$FAIL P0-diff"; }

echo "== P0b import origin (cd <tree>; PYTHONPATH=src) and its empty-PYTHONPATH mutant"
EMPTY=$(mktemp -d /tmp/v5c_empty.XXXXXX)
origin() {  # origin <tree> <module> <pythonpath> -> "<INSIDE|FALLBACK> <path>"
  local want got
  want=$(cd "$1" && realpath src)
  got=$(cd "$1" && env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$3 python3 -c "import importlib, os; m = importlib.import_module('$2'); print(os.path.realpath(m.__file__))" 2>/dev/null | tail -1)
  case "$got" in "$want"/*) echo "INSIDE $got";; *) echo "FALLBACK ${got:-none}";; esac
}
nin=0; nkill=0; nmod=0
for t in head base; do
  for mod in vllm_torchtpu vllm_torchtpu.envs vllm_torchtpu.compilation.tpu_compiler vllm_torchtpu.core.l6_ab \
             vllm_torchtpu.runner.tpu_runner vllm_torchtpu.spec_decode.eagle3 vllm_torchtpu.models.vllm.qwen4_exp.mtp \
             vllm_torchtpu.runner.l4_vpshost vllm_torchtpu.runner.vfy_one vllm_torchtpu.spec_decode.draft_asm \
             vllm_torchtpu.spec_decode.draft_one vllm_torchtpu.runner.l4_agree vllm_torchtpu.spec_decode.draft_cap \
             vllm_torchtpu.runner.l4_samp2 vllm_torchtpu.layers.adapter.sample.vp_sampler vllm_torchtpu.runner.utils \
             vllm_torchtpu.runner.seedfix vllm_torchtpu.platforms.tpu_platform vllm_torchtpu.spec_decode.k1fuse \
             vllm_torchtpu.runner.speculative_decoding_manager; do
    if [ $t = base ]; then case $mod in *l4_vpshost|*vfy_one|*draft_asm|*draft_one|*l4_agree|*draft_cap|*l4_samp2|*seedfix|*k1fuse) continue;; esac; fi
    nmod=$((nmod + 1))
    r=$(origin "$W/$t" $mod src); echo "ORIGIN $t $mod ${r#* } ${r%% *}"; [ "${r%% *}" = INSIDE ] && nin=$((nin + 1))
    m=$(origin "$W/$t" $mod "$EMPTY"); echo "ORIGIN $t:MUTANT $mod ${m#* } ${m%% *}"; [ "${m%% *}" = FALLBACK ] && nkill=$((nkill + 1))
  done
done
echo "P0b inside $nin/$nmod; empty-PYTHONPATH mutants reading FALLBACK $nkill/$nmod"
[ $nmod = 31 ] && [ $nin = 31 ] && [ $nkill = 31 ] || FAIL="$FAIL P0b"

finish() {
  HS2=$(srch "$H"); echo "P6 head src after $HS2 (before $HS)"; [ "$HS2" = "$HS" ] || FAIL="$FAIL P6"
  [ -z "$FAIL" ] && { echo "Q38_V5C_SUMMARY PASS mode=$1"; exit 0; }
  echo "Q38_V5C_SUMMARY FAIL mode=$1:$FAIL"; exit 1
}
run() {  # run <tag> <want passed> <cmd...> (in head/)
  local tag=$1 want=$2; shift 2
  local t0=$(date +%s)
  (cd "$H" && env $CPUENV "$@") > $L/$tag.log 2>&1; local rc=$?
  echo "$tag rc=$rc $(( $(date +%s) - t0 ))s want $want :: $(summ $L/$tag.log)"
  if [ $rc = 0 ] && exact $L/$tag.log $want; then :; else
    FAIL="$FAIL $tag"; grep -aE '^(FAILED|ERROR) |^E  ' $L/$tag.log | head -40 | cut -c1-600; tail -15 $L/$tag.log | cut -c1-400; fi
}

if [ "$MODE" = mut ]; then
  LO=${2:?LO}; HI=${3:?HI}; N=$((HI - LO + 1))
  echo "== P5 composition mutants $LO-$HI"
  t0=$(date +%s)
  (cd "$H" && python3 -u tests/q38_l4_v5c/q38_v5c_mutrun.py . $LO $HI) > $L/p5.log 2>&1; rc=$?
  grep -aE '^MUTRUN|^CONTROL|^MUT |^Q38_V5C_MUT' $L/p5.log | cut -c1-700
  echo "P5 rc=$rc $(( $(date +%s) - t0 ))s"
  TOTAL=$(python3 -c "import ast; t=ast.parse(open('$Q/q38_v5c_mutrun.py').read()); print(next(len(n.value.elts) for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MUTANTS'))")
  [ $rc = 0 ] && grep -aq "^Q38_V5C_MUT PASS killed $N/$N of $N range $LO-$HI of $TOTAL " $L/p5.log \
    && grep -aq '^MUTRUN_ORIGIN .* INSIDE$' $L/p5.log || { FAIL="$FAIL P5"; tail -40 $L/p5.log; }
  finish "mut range $LO-$HI"
fi

# ---------------------------------------------------------------- v6: H3 legs in ATTNWIRE's environment
H3MODS=vllm_torchtpu.megakernel_runtime.qwen38_draft_attn,vllm_torchtpu.megakernel.wholestep.qsa_draft,vllm_torchtpu.models.vllm.qwen4_exp.mtp
H3PT="python3 -B -m pytest -q -p no:cacheprovider -W ignore -rfEs -s -p q38_cpu_device -p q38_tpu_device_gate_v2 -p q38_staged_guard"
h3plugins() {
  local ok=1 x
  for x in "q38_cpu_device.py b452635b" "q38_tpu_device_gate_v2.py 31bfd921" "q38_staged_guard.py 1d1dd665"; do
    set -- $x; [ "$(md5sum < $PL/$1 | cut -c1-8)" = $2 ] || { echo "P8 plugin pin mismatch $1"; ok=0; }
  done
  [ $ok = 1 ] && echo "P8 plugins pinned (L1's q38_cpu_device b452635b, q38_tpu_device_gate_v2 31bfd921, q38_staged_guard 1d1dd665)"
  [ $ok = 1 ]
}
h3node() {  # h3node <tag> <timeout s> <want passed> <pytest node...>: one H3 leg in head/, verdict into $L/<tag>.v
  local tag=$1 to=$2 want=$3; shift 3
  local t0=$(date +%s)
  (cd "$H" && env Q38_STAGED_ROOT=$H/src Q38_STAGED_MODULES=$H3MODS JAX_PLATFORMS=cpu VLLM_TARGET_DEVICE=cpu \
     TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PL:$H/src:$H \
     timeout -s KILL $to $H3PT "$@") > $L/$tag.log 2>&1
  local rc=$? g; g=$(grep -ac '^Q38_STAGED_GUARD OK' $L/$tag.log)
  local v=PASS
  { [ $rc = 0 ] && exact $L/$tag.log $want && [ "$g" -ge 4 ]; } || v=FAIL
  echo "$tag $v rc=$rc $(( $(date +%s) - t0 ))s want $want guard $g :: $(summ $L/$tag.log)" > $L/$tag.v
}
export -f h3node summ count exact; export H L PL H3MODS H3PT
h3run() {  # h3run <nodes file>: lines "<tag> <timeout> <want> <node...>", 4 at a time
  : > $L/h3_tags
  while read -r tag to want rest; do [ -n "$tag" ] && echo "$tag" >> $L/h3_tags; done < "$1"
  xargs -P ${H3P:-4} -L 1 bash -c 'h3node "$@"' _ < "$1"
  for t in $(cat $L/h3_tags); do
    cat $L/$t.v 2>/dev/null || echo "$t FAIL (no verdict)"
    grep -q " PASS " $L/$t.v 2>/dev/null || { FAIL="$FAIL $t"; grep -aE '^(FAILED|ERROR) |^E  ' $L/$t.log | head -30 | cut -c1-500; tail -12 $L/$t.log | cut -c1-300; }
    grep -aE '^(V6C_H3|ATTNWIRE|Q38_CPU_DEVICE|Q38_STAGED_GUARD)' $L/$t.log | cut -c1-700 | sed "s/^/$t /"
  done
}

if [ "$MODE" = h3k ]; then
  echo "== P8 H3 kernel / wiring / pack suites (ATTNWIRE's environment) and the H3 regression legs"
  h3plugins || FAIL="$FAIL P8-plugins"
  BK=tests/kernels/qwen4_exp/dcp2/test_attnwire_bucket.py
  BKN=test_kernel_at_rows_b_with_operands_packed_at_max
  cat > $L/h3k_nodes <<EOF2
WIRING 1800 9 tests/q38_attnwire/test_attnwire_wiring.py
L1WIRE 1800 6 tests/megakernel_runtime/test_l1h3_draft_attn_wiring.py
PACK 3600 5 $BK::test_pack_is_rows_invariant_tiny $BK::test_pack_is_rows_invariant_full_tp8
KERN 5400 10 tests/megakernel/test_l1h3_draft_attn.py
CALL 3600 4 tests/kernels/qwen4_exp/dcp2/test_l1h3_draft_attn_call.py
BK_21 9000 5 $BK::$BKN[2-1-8] $BK::$BKN[2-1-16] $BK::$BKN[2-1-32] $BK::$BKN[2-1-48] $BK::$BKN[2-1-64]
BK_22 9000 5 $BK::$BKN[2-2-8] $BK::$BKN[2-2-16] $BK::$BKN[2-2-32] $BK::$BKN[2-2-48] $BK::$BKN[2-2-64]
BK_44 9000 5 $BK::$BKN[4-4-8] $BK::$BKN[4-4-16] $BK::$BKN[4-4-32] $BK::$BKN[4-4-48] $BK::$BKN[4-4-64]
EOF2
  h3run $L/h3k_nodes
  echo "== P8 H3 regression legs HEAD vs BASE (ecf4cc34a), ATTNWIRE's environment, L3's regress_perfile"
  . "$W/q38_l3_regress_perfile.sh"
  CPUENV="JAX_PLATFORMS=cpu VLLM_TARGET_DEVICE=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PL:src"
  PT="python3 -B -m pytest -q -p no:cacheprovider -W ignore -rfEs -p q38_cpu_device"
  regress_perfile v6h3 "$H" "$B" 3600 tests/compilation/test_tpu_compile_key_registration.py \
    tests/q38_deco44_c32mk/test_imported_megakernel_identity.py tests/megakernel_runtime/test_qwen38_mtp_tail.py \
    tests/kernels/qwen4_exp/dcp2/test_dcp2_draft.py tests/kernels/qwen4_exp/dcp2/test_dcp2_draft_no_index_state.py \
    tests/models/vllm/test_qwen4_exp_draft_no_index_state.py tests/kernels/test_qwen4_exp_shared_selection.py \
    || FAIL="$FAIL P8-regress"
  finish h3k
fi

if [ "$MODE" = h3c ]; then
  SET=${2:?A, B or L}
  echo "== P9 H3 compose set $SET (ATTNWIRE's T2 subset on the composed tree + test_v6c_h3_compose.py)"
  h3plugins || FAIL="$FAIL P9-plugins"
  CP=tests/kernels/qwen4_exp/dcp2/test_attnwire_compose.py
  V6=tests/kernels/qwen4_exp/dcp2/test_v6c_h3_compose.py
  if [ "$SET" = A ]; then
    cat > $L/h3c_nodes <<EOF2
CPA 10800 2 $CP::test_dhd_with_draft_attention_kind_over_8_steps[2-2-16] $CP::test_dhd_with_draft_attention_kind_over_8_steps[2-2-48]
CPB 10800 1 $CP::test_dhd_with_draft_attention_kind_over_8_steps[4-4-48]
CPC 10800 2 $CP::test_meta_i32_on_off_bitwise[4-4-16] $CP::test_one_build_per_rows_across_window_switched_paths
HL 10800 2 $V6::test_hl_kind_beside_lmh_hidden_major_draft_head $V6::test_zz_counts
EOF2
  elif [ "$SET" = L ]; then
    cat > $L/h3c_nodes <<EOF2
HL 10800 2 $V6::test_hl_kind_beside_lmh_hidden_major_draft_head $V6::test_zz_counts
EOF2
  else
    cat > $L/h3c_nodes <<EOF2
HC1 10800 2 $V6::test_hc_kind_inside_draft_cap_programs_at_r_d[2-2-16] $V6::test_hc_kind_inside_draft_cap_programs_at_r_d[2-2-48]
HC2 10800 1 $V6::test_hc_kind_inside_draft_cap_programs_at_r_d[4-4-16]
HK1 10800 2 $V6::test_hk_kind_inside_k1fuse_program_is_bitwise_the_chain[16-False] $V6::test_hk_kind_inside_k1fuse_program_is_bitwise_the_chain[16-True]
HK2 10800 2 $V6::test_hk_kind_inside_k1fuse_program_is_bitwise_the_chain[48-False] $V6::test_hk_kind_inside_k1fuse_program_is_bitwise_the_chain[48-True]
EOF2
  fi
  h3run $L/h3c_nodes
  finish "h3c $SET"
fi

if [ "$MODE" = h3bar ]; then
  SET=${2:?a..f}
  echo "== P10 the FRESH H3 composed run under the locked v6 bar (prereg q38_l4_h3_v6_bar_DRAFT.md e7c7dad0), set $SET"
  h3plugins || FAIL="$FAIL P10-plugins"
  T=tests/kernels/qwen4_exp/dcp2/test_v6c_h3bar.py
  case $SET in
    a) NODES="CP218:$T::test_cp_bar[2-1-8] CP2116:$T::test_cp_bar[2-1-16] CP2132:$T::test_cp_bar[2-1-32] CP2148:$T::test_cp_bar[2-1-48]" ;;
    b) NODES="CP2164:$T::test_cp_bar[2-1-64] CP228:$T::test_cp_bar[2-2-8] CP2216:$T::test_cp_bar[2-2-16] CP2232:$T::test_cp_bar[2-2-32]" ;;
    c) NODES="CP2248:$T::test_cp_bar[2-2-48] CP2264:$T::test_cp_bar[2-2-64] PATHS:$T::test_paths_bar HC2216:$T::test_hc_bar[2-2-16]" ;;
    d) NODES="CP448:$T::test_cp_bar[4-4-8] CP4416:$T::test_cp_bar[4-4-16] CP4432:$T::test_cp_bar[4-4-32] HC4416:$T::test_hc_bar[4-4-16]" ;;
    e) NODES="CP4448:$T::test_cp_bar[4-4-48] CP4464:$T::test_cp_bar[4-4-64] HC2248:$T::test_hc_bar[2-2-48]" ;;
    f) NODES="HK16F:$T::test_hk_bar[16-False] HK16T:$T::test_hk_bar[16-True] HK48F:$T::test_hk_bar[48-False] HK48T:$T::test_hk_bar[48-True]" ;;
    # (4, 4) worlds two at a time (w6 set d j-13233d7b: four 4-rank nodes in parallel were OOMKilled at 64Gi)
    d1) H3P=2; NODES="CP448:$T::test_cp_bar[4-4-8] CP4416:$T::test_cp_bar[4-4-16]" ;;
    d2) H3P=2; NODES="CP4432:$T::test_cp_bar[4-4-32] HC4416:$T::test_hc_bar[4-4-16]" ;;
    e1) H3P=1; NODES="CP4448:$T::test_cp_bar[4-4-48]" ;;
    e2) H3P=1; NODES="CP4464:$T::test_cp_bar[4-4-64]" ;;
    e3) H3P=1; NODES="HC2248:$T::test_hc_bar[2-2-48]" ;;
    *) echo "unknown set"; FAIL="$FAIL P10-set" ;;
  esac
  : > $L/h3bar_nodes
  for x in $NODES; do echo "${x%%:*} 13800 1 ${x#*:}" >> $L/h3bar_nodes; done
  export H3P
  h3run $L/h3bar_nodes
  for t in $(cat $L/h3_tags); do grep -aE '^(H3BAR_VERDICT|H3BAR_RAW|H3BAR_DRAFTS|H3BAR )' $L/$t.log | grep -v 'per-pass shares' | cut -c1-900 | sed "s/^/$t /"; done
  finish "h3bar $SET"
fi

if [ "$MODE" = h3barelt ]; then
  SET=${2:?S1 or S2}
  echo "== P12 M-elt as written (ratio <= 1.5, |want| >= 1e-3; L1's ruling) swept over the H3 bar tests, set $SET"
  h3plugins || FAIL="$FAIL P12-plugins"
  T=tests/kernels/qwen4_exp/dcp2/test_v6c_h3bar.py
  if [ "$SET" = S1 ]; then
    P=4; NODES="$T::test_cp_bar[2-1-8] $T::test_cp_bar[2-1-16] $T::test_cp_bar[2-1-32] $T::test_cp_bar[2-1-48] \
$T::test_cp_bar[2-1-64] $T::test_cp_bar[2-2-8] $T::test_cp_bar[2-2-16] $T::test_cp_bar[2-2-32] $T::test_cp_bar[2-2-48] \
$T::test_cp_bar[2-2-64] $T::test_paths_bar $T::test_hc_bar[2-2-16] $T::test_hc_bar[2-2-48] $T::test_hk_bar[16-False] \
$T::test_hk_bar[16-True] $T::test_hk_bar[48-False] $T::test_hk_bar[48-True]"
  elif [ "$SET" = S1D ]; then  # the D = 2 tests, after the D > 1 oracle fix
    P=4; NODES="$T::test_cp_bar[2-2-8] $T::test_cp_bar[2-2-16] $T::test_cp_bar[2-2-32] $T::test_cp_bar[2-2-48] \
$T::test_cp_bar[2-2-64] $T::test_hc_bar[2-2-16] $T::test_hc_bar[2-2-48] $T::test_hk_bar[16-False] $T::test_hk_bar[16-True] \
$T::test_hk_bar[48-False] $T::test_hk_bar[48-True]"
  else
    P=2; NODES="$T::test_cp_bar[4-4-8] $T::test_cp_bar[4-4-16] $T::test_cp_bar[4-4-32] $T::test_hc_bar[4-4-16] \
$T::test_cp_bar[4-4-48] $T::test_cp_bar[4-4-64]"
  fi
  t0=$(date +%s)
  (cd "$H" && python3 -u tests/kernels/qwen4_exp/dcp2/q38_h3bar_eltsweep.py . $PL $P $NODES) > $L/p12.log 2>&1; rc=$?
  grep -aE '^(ELT|Q38_H3BAR_ELT)' $L/p12.log | cut -c1-600
  echo "P12 rc=$rc $(( $(date +%s) - t0 ))s"
  grep -aq '^Q38_H3BAR_ELT ' $L/p12.log || { FAIL="$FAIL P12"; tail -30 $L/p12.log; }
  grep -aq '^Q38_H3BAR_ELT .* bad=0 ' $L/p12.log || FAIL="$FAIL P12-bad"
  finish "h3barelt $SET"
fi

if [ "$MODE" = h3barmut ]; then
  echo "== P11 the v6 bar's mutants (M-elt, M-agg, M-elt-hi, ATTNWIRE A1 / A2 / A4) on test_cp_bar[2-1-8]"
  h3plugins || FAIL="$FAIL P11-plugins"
  t0=$(date +%s)
  ONLY=$(echo "${2:-}" | tr '+' ','); NWANT=10; [ -n "$ONLY" ] && NWANT=$(echo "$ONLY" | tr ',' '\n' | grep -c .)
  (cd "$H" && python3 -u tests/kernels/qwen4_exp/dcp2/q38_h3bar_mutrun.py . $PL "$ONLY") > $L/p11.log 2>&1; rc=$?
  grep -aE '^CONTROL|^MUT |^Q38_H3BAR_MUT' $L/p11.log | cut -c1-1500
  echo "P11 rc=$rc $(( $(date +%s) - t0 ))s"
  [ $rc = 0 ] && grep -aq "^Q38_H3BAR_MUT PASS killed $NWANT/$NWANT control_ok=1" $L/p11.log || { FAIL="$FAIL P11"; tail -30 $L/p11.log; }
  finish h3barmut
fi

if [ "$MODE" = rp ]; then
  PT="$PT --tb=short"   # see the P7 header: the long traceback's frame-argument repr aborts test_eagle3.py
  HALF=${2:?A or B}
  KEY=$(grep -a '^KEY ' $Q/RP_FILES | awk '{print $2}')
  ALL=$(grep -aE '^(KEY|RUN) ' $Q/RP_FILES | awk '{print $2}')
  NALL=$(echo "$ALL" | wc -l | tr -d ' '); NA=$(( (NALL + 1) / 2 ))
  if [ "$HALF" = A ]; then CANDS=$(echo "$ALL" | head -n $NA); else CANDS=$(echo "$ALL" | tail -n +$((NA + 1))); fi
  echo "== P7 base screen of the RP_FILES half $HALF ($(echo "$CANDS" | wc -l | tr -d ' ') of $NALL files, 4 at a time)"
  screen1() { local f=$1 log=/tmp/v5c_scr_$(echo $1 | tr '/.' '__').log t0=$(date +%s)
    (cd "$B" && timeout -s KILL 1500 env $CPUENV $PT "$f" > "$log" 2>&1)
    echo "SCREEN $f $(( $(date +%s) - t0 ))s :: $(summ "$log")"; }
  export -f screen1 summ; export B CPUENV PT
  echo "$CANDS" | xargs -P 4 -I{} bash -c 'screen1 {}' | LC_ALL=C sort > /tmp/v5c_screen.txt
  sed 's/^/P7 /' /tmp/v5c_screen.txt
  RUN=""
  for f in $CANDS; do
    log=/tmp/v5c_scr_$(echo $f | tr '/.' '__').log
    n=$(summ "$log" | grep -oE '[0-9]+ (passed|failed|error|errors)' | awk '{n += $1} END {print n + 0}')
    if [ "${n:-0}" -ge 1 ]; then RUN="$RUN $f"
    else case " $(echo $KEY) " in *" $f "*) echo "P7 KEY file counts 0 on base: $f"; FAIL="$FAIL P7-key";; *) echo "P7 EXCLUDED (0 counted on base) $f";; esac; fi
  done
  echo "== P7 per-file regression HEAD vs BASE (ecf4cc34a) over $(echo $RUN | wc -w) files (half $HALF)"
  . "$W/q38_l3_regress_perfile.sh"
  regress_perfile v5c "$H" "$B" 1500 $RUN > $L/p7rp.log 2>&1 || FAIL="$FAIL P7"
  cat $L/p7rp.log
  # every file with a NEW failure on the head: one traceback per new id from the head's log (L4-E's ask)
  i=0
  for f in $RUN; do
    i=$((i + 1))
    grep -aq "^v5c RP $f: .*NEW failures on tip" $L/p7rp.log || continue
    for id in $(grep -a "^v5c RP $f: " $L/p7rp.log | sed -E 's/.*NEW failures on tip: //; s/-> FAIL//'); do
      nm=${id##*::}
      echo "P7 NEW $id -- head traceback (first 40 lines):"
      awk -v n="$nm" 'index($0, " " n " ") && /^_+ /{f=1} f && /^_+ / && !index($0, " " n " "){exit} f' /tmp/v5c_rp_${i}_tip.log \
        | head -40 | sed 's/^/P7 TB /' | cut -c1-300
    done
  done
  finish "rp $HALF"
fi

echo "== P1 chained flag-off AST identity vs ecf4cc34a"
run P1 13 $PTN -s tests/q38_l4_v5c/test_v5c_flagoff.py
grep -a '^V5C_FLAGOFF' $L/P1.log | cut -c1-900

echo "== P2 compile-key rows, flag table, LEVERS union, render-grep"
(cd "$H" && env $CPUENV $PTC tests/compilation/test_tpu_compile_key_registration.py) > $L/P2reg.log 2>&1; rc=$?
echo "P2reg rc=$rc :: $(summ $L/P2reg.log)"
[ $rc = 0 ] && [ "$(count $L/P2reg.log failed)" = 0 ] && [ "$(count $L/P2reg.log passed)" -gt 7 ] || { FAIL="$FAIL P2reg"; tail -30 $L/P2reg.log; }
run P2flags 5 $PTC -s tests/q38_l4_v5c/test_v5c_flags.py
grep -a '^V5C_FLAGS' $L/P2flags.log | cut -c1-1200

echo "== P3a VPSHOST"
run P3a-vpshost 58 $PTC -s tests/q38_l4_vpshost/test_l4_vpshost.py
while read name want; do
  [ -n "$name" ] || continue
  got=$(grep -ac "^$name " $L/P3a-vpshost.log); echo "P3a census $name $got (want $want)"
  [ "$got" = "$want" ] || FAIL="$FAIL P3a-census-$name"
done < $H/tests/q38_l4_vpshost/EXPECT_CENSUS
(cd "$H" && python3 tests/q38_l4_vpshost/q38_vpshost_compiled_identity.py "$H") > $L/P3a-cid-head.log 2>&1; rh=$?
(cd "$H" && python3 tests/q38_l4_vpshost/q38_vpshost_compiled_identity.py "$B") > $L/P3a-cid-base.log 2>&1; rb=$?
EXTRA=$(comm -13 <(grep -a '^COMPILED ' $L/P3a-cid-base.log | grep -v COMPILED_ALL | sort) <(grep -a '^COMPILED ' $L/P3a-cid-head.log | grep -v COMPILED_ALL | sort))
GONE=$(comm -23 <(grep -a '^COMPILED ' $L/P3a-cid-base.log | grep -v COMPILED_ALL | sort) <(grep -a '^COMPILED ' $L/P3a-cid-head.log | grep -v COMPILED_ALL | sort))
NLR=$(cat $L/P3a-cid-head.log $L/P3a-cid-base.log | grep -ac '^LEVERREAD')
echo "P3a compile identity head rc=$rh base rc=$rb; head-only: [$(echo $EXTRA)]; base-only: [$(echo $GONE)]; LEVERREAD $NLR"
# head-only compiled programs: VFY-ONE's, and VRNG-VPS's two in-graph-draw programs (all new on the branch)
WANT_EXTRA=$(printf '%s\n' tpu_runner:_vfy_one_impl tpu_runner:_vfy_one_rng_impl tpu_runner:_vps_verify_rng_impl | sort)
[ -z "$GONE" ] && [ "$(echo "$EXTRA" | awk '{print $2}' | sort)" = "$WANT_EXTRA" ] && [ "$NLR" = 0 ] \
  || FAIL="$FAIL P3a-compile-identity"

echo "== P3b VFY-ONE + D83"
run P3b-vfy-one 78 env PYTHONPATH=src:q38_l4vfy $PTN tests/l4vfy/test_vfy_one.py
run P3b-vfy-agree 15 env PYTHONPATH=src:q38_l4vfy $PTN tests/l4vfy/test_vfy_agree.py
run P3b-vfy-dha 17 $PTC -s tests/l4vfy/test_vfy_dha_compose.py
run P3b-d83 86 $PTN tests/l4tail/test_vps_ro_fold.py

echo "== P3c DRAFT-ASM"
for fw in units:56 propose:51 multirank:8; do
  f=${fw%%:*}; n=${fw#*:}
  run P3c-$f $n env PYTHONPATH=src:tests/q38_draftasm $PTC tests/q38_draftasm/test_draftasm_$f.py
done
run P3c-flagoff 1 env PYTHONPATH=src:tests/q38_draftasm $PTC tests/q38_draftasm/test_draftasm_flagoff.py \
  --deselect tests/q38_draftasm/test_draftasm_flagoff.py::test_eagle3_flag_off_is_3d2f53ec9 \
  --deselect tests/q38_draftasm/test_draftasm_flagoff.py::test_tpu_runner_flag_off_is_3d2f53ec9 \
  --deselect tests/q38_draftasm/test_draftasm_flagoff.py::test_flag_off_check_bites

echo "== P3d DRAFT1-B"
for fw in units:27 stagea:42 multirank:6 dhd:42; do
  f=${fw%%:*}; n=${fw#*:}
  run P3d-$f $n env PYTHONPATH=src:tests/q38_draft1b $PTC -s tests/q38_draft1b/test_draft1b_$f.py
done
D=tests/q38_draft1b/test_draft1b_flagoff.py
run P3d-flagoff 1 env PYTHONPATH=src:tests/q38_draft1b $PTC $D \
  --deselect $D::test_draft_asm_runner_and_fused_loop_are_byte_identical_to_9b8d39a74 \
  --deselect $D::test_eagle3_stage_flags_off_is_9b8d39a74 --deselect $D::test_mtp_meta_i32_off_is_9b8d39a74 \
  --deselect $D::test_flag_off_check_bites

echo "== P3e DRAFT-2 (c) / EMB-REP-TGT"
run P3e-emb 27 $PTC -s tests/q38_draftemb/test_draft_embed_repl.py tests/q38_draftemb/test_target_embed_repl.py
(cd "$H" && env $CPUENV python3 tests/q38_draftemb/q38_draftemb_fit.py) > $L/P3e-fit.log 2>&1; rc=$?
echo "P3e-fit rc=$rc :: $(tail -1 $L/P3e-fit.log)"
[ $rc = 0 ] && tail -1 $L/P3e-fit.log | grep -q '^DRAFTEMB_FIT PASS' || { FAIL="$FAIL P3e-fit"; tail -20 $L/P3e-fit.log; }

echo "== P3f AGREE / KEYLITE"
run P3f-units 17 $PTC -s tests/q38_l4agree/test_l4agree_units.py
run P3f-multirank 17 $PTC -s tests/q38_l4agree/test_l4agree_multirank.py
A=tests/q38_l4agree/test_l4agree_flagoff.py
run P3f-flagoff 2 env Q38_L4AGREE_EXPECT_ROOT=$H/src $PTC -s $A --deselect $A::test_tpu_runner_agree_off_is_2e166829a \
  --deselect $A::test_eagle3_is_2e166829a_byte_for_byte --deselect $A::test_all_flags_off_is_the_composed_base \
  --deselect $A::test_the_check_bites_on_an_unguarded_tick_or_registration
run P3f-keylite 23 $PTC -s tests/q38_l4agree/test_vfy_keylite.py
run P3f-vfyone-keylite 78 env VLLM_TPU_VPS_VFY_ONE_KEYLITE=1 PYTHONPATH=src:q38_l4vfy $PTN tests/l4vfy/test_vfy_one.py

echo "== P3h DRAFT-CAP"
run P3h-cap 46 env PYTHONPATH=src:tests/q38_draftcap $PTC -s tests/q38_draftcap/test_draftcap.py
for f in $(cat $H/tests/q38_draftcap/EXPECT_LINES); do
  pre=${f%%:*}; want=${f##*:}
  [ "$pre" = FLAGOFF ] && continue  # the flag-off lines come from test_draftcap_flagoff.py (partly deselected here)
  got=$(grep -ac "^DRAFTCAP_$pre " $L/P3h-cap.log); echo "P3h lines DRAFTCAP_$pre $got (want $want)"
  [ "$got" = "$want" ] || FAIL="$FAIL P3h-lines-$pre"
done
C=tests/q38_draftcap/test_draftcap_flagoff.py
run P3h-flagoff 3 env PYTHONPATH=src:tests/q38_draftcap $PTC -s $C \
  --deselect $C::test_untouched_modules_are_byte_identical_to_29ea1e98d --deselect $C::test_envs_is_29ea1e98d_plus_the_variable

echo "== P3g VRNG-VPS / S2 (ported to the agreement clock)"
run P3g-vrng 100 timeout -s KILL 5400 $PTC -s tests/q38_l4_samp2v/ -k "not v1_regime"
while read -r tag want; do
  [ -n "$tag" ] || continue
  got=$(grep -acE "^$tag " $L/P3g-vrng.log); echo "P3g census $tag $got (want $want)"; [ "$got" = "$want" ] || FAIL="$FAIL P3g-census-$tag"
done < $H/tests/q38_l4_samp2v/EXPECT_CENSUS

echo "== P3i LMH-STREAM stage 0 + the v5 HBM fit"
T=tests/q38_lmhs0/test_lmh_hidden_major.py
run P3i-lmh 25 env PYTHONPATH=src:tests/q38_lmhs0 $PTC -s $T --deselect $T::test_flag_off_check_bites \
  --deselect "$T::test_flags_off_is_the_base_ast_for_ast[tpu_runner-runner/tpu_runner.py]" \
  --deselect "$T::test_flags_off_is_the_base_ast_for_ast[utils-runner/utils.py]" \
  --deselect "$T::test_flags_off_is_the_base_ast_for_ast[envs-envs.py]"
grep -aE '^LMHS0_(CENSUS|ENGAGE|WIRING|SHARDED)' $L/P3i-lmh.log | cut -c1-400
(cd "$H" && env $CPUENV python3 tests/q38_l4_v5c/q38_v5c_fit.py) > $L/P3i-fit.log 2>&1; rc=$?
grep -a '^V5C_FIT' $L/P3i-fit.log | cut -c1-400
[ $rc = 0 ] && tail -1 $L/P3i-fit.log | grep -q '^V5C_FIT PASS' || { FAIL="$FAIL P3i-fit"; tail -20 $L/P3i-fit.log; }

echo "== P3j SEEDFIX (its composed-tree suites; the primary's ecf4cc34a-only AST test deselected: P1 replaces)"
S=tests/q38_l4_seedfix
run P3j-seedfix 52 timeout -s KILL 5400 $PTC -s $S/test_seedfix.py $S/test_seedfix_vrng.py -k "not distribution" \
  --deselect $S/test_seedfix.py::test_flag_off_runner_ast_is_base
grep -aE '^SEEDFIX_[A-Z_]+ ' $L/P3j-seedfix.log | cut -c1-300 | head -40

echo "== P3k T1-K1FUSE (its suites on the composed tree, ported to the AGREE clock; its 3 tests vs 7867a18ef deselected: P1 replaces)"
for fw in k1fuse:31 multirank:4 compile:3; do
  f=${fw%%:*}; n=${fw#*:}; file=tests/q38_k1fuse/test_k1fuse.py; [ "$f" = k1fuse ] || file=tests/q38_k1fuse/test_k1fuse_$f.py
  run P3k-$f $n $PTC -s $file
done
KF=tests/q38_k1fuse/test_k1fuse_flagoff.py
run P3k-flagoff 5 $PTC -s $KF --deselect $KF::test_eagle3_flag_off_is_the_composed_base \
  --deselect $KF::test_tpu_runner_flag_off_is_the_composed_base --deselect $KF::test_flag_off_check_bites_on_each_hook
for f in $(cat $H/tests/q38_k1fuse/EXPECT_LINES); do
  pre=${f%%:*}; want=${f##*:}
  [ "$pre" = FLAGOFF ] && continue  # its FLAGOFF lines come from the 3 deselected ref-based tests
  got=$(cat $L/P3k-*.log | grep -ac "^K1FUSE_$pre "); echo "P3k lines K1FUSE_$pre $got (want $want)"
  [ "$got" = "$want" ] || FAIL="$FAIL P3k-lines-$pre"
done
bad=$(cat $L/P3k-*.log | grep -a '^K1FUSE_CENSUS ' | grep -avc 'calls 2 -> 1 ')
echo "P3k census lines not reading 'calls 2 -> 1': $bad"; [ "$bad" = 0 ] || FAIL="$FAIL P3k-census"
cat $L/P3k-*.log | grep -aE '^(K1FUSE_[A-Z]+|Q38_K1FUSE_COUNTS) ' | cut -c1-500 | head -60

echo "== P4f FOLDFIX: the real propose -> manager -> runner path (its suite) and K1's take through it; the manager's flag-off identity"
run P4f-realpath 16 $PTC -rfEs -s tests/q38_l4_foldfix/test_foldfix_realpath.py
run P4f-flagoff 3 $PTC -s tests/q38_l4_foldfix/test_foldfix_flagoff.py
run P4f-k1 4 $PTC -s tests/q38_l4_v5c/test_v6c_foldfix_k1.py
cat $L/P4f-*.log | grep -aE '^(FOLDFIX_|V6C_FOLDFIX_K1)' | cut -c1-900

echo "== P4 composition"
run P4-runner 15 $PTN -s tests/q38_l4_v5c/test_v5c_runner.py
run P4-drafter 11 env PYTHONPATH=src:tests/q38_draft1b $PTC -s tests/q38_l4_v5c/test_v5c_drafter.py
run P4-union 4 $PTN -s tests/q38_l4_v5c/test_v5c_union.py
run P4-embagree 6 $PTC -s tests/q38_l4_v5c/test_v5c_emb_agree.py
run P4-cap 9 env PYTHONPATH=src:tests/q38_draftcap $PTC -s tests/q38_l4_v5c/test_v5c_cap.py
run P4-vrnglev 5 $PTC -s tests/q38_l4_v5c/test_v5c_vrng_levers.py
run P4-keylite-vrng 3 $PTC -s tests/q38_l4_v5c/test_v5c_keylite_vrng.py
run P4-v6k1 25 $PTC -s tests/q38_l4_v5c/test_v6c_k1.py
cat $L/P4-*.log | grep -aE '^V5C_(RUNNER|DRAFTER|UNION|EMBAGREE|CAP|VRNGLEV|KEYLITE_VRNG)|^V6C_K1' | cut -c1-900
finish smoke
