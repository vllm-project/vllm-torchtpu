#!/bin/bash
# q38mm DM P15 CPU job body (runs from the tree root, /tmp/src in the parasim-cpu pod, PYTHONPATH=/tmp/src/src).
# Sections in INFRA's marker format; exits non-zero if any section fails (import_guard failure exits 3 at once).
bad=0
echo "Q38_SECTIONS_EXPECTED import_guard,pta_tests,pta_mut,pta_bench,flag_reach"
echo "Q38_SECTION_BEGIN import_guard"
python3 -c "import sys, vllm_torchtpu; f = vllm_torchtpu.__file__; print('vllm_torchtpu from', f); sys.exit(0 if f.startswith('/tmp/src/') else 3)"
rc=$?; echo "Q38_SECTION_END import_guard rc=${rc}"
[ ${rc} -eq 0 ] || { echo "Q38_SECTIONS_DONE bad=1"; exit 3; }
echo "Q38_SECTION_BEGIN pta_tests"
timeout 1800 python3 -m pytest -v -rfEs -p no:cacheprovider tests/q38mm_dm/test_prompt_ids_ndarray.py
rc=$?; echo "Q38_SECTION_END pta_tests rc=${rc}"; [ ${rc} -eq 0 ] || bad=1
echo "Q38_SECTION_BEGIN pta_mut"
timeout 2400 python3 tests/q38mm_dm/mutants_pta.py
rc=$?; echo "Q38_SECTION_END pta_mut rc=${rc}"; [ ${rc} -eq 0 ] || bad=1
echo "Q38_SECTION_BEGIN pta_bench"
timeout 1200 python3 tests/q38mm_dm/q38mm_pta_bench.py --reps 25
rc=$?; echo "Q38_SECTION_END pta_bench rc=${rc}"; [ ${rc} -eq 0 ] || bad=1
echo "Q38_SECTION_BEGIN flag_reach"
hits=$(grep -rl -e VLLM_TPU_PROMPT_IDS_NDARRAY -e prompt_ids_ndarray src | sort | paste -sd, -)
echo "flag reach: ${hits}"
[ "${hits}" = "src/vllm_torchtpu/core/prefix_aware_scheduler.py,src/vllm_torchtpu/core/prompt_ids_ndarray.py" ] && ! grep -rq -e VLLM_TPU_PROMPT_IDS_NDARRAY -e prompt_ids_ndarray src/vllm_torchtpu/runner
rc=$?; echo "Q38_SECTION_END flag_reach rc=${rc}"; [ ${rc} -eq 0 ] || bad=1
echo "Q38_SECTIONS_DONE bad=${bad}"
exit ${bad}
