#!/bin/bash
# L44 + tail kernel (VLLM_TPU_MK_WHOLE_L47) vs one L45 kernel (VLLM_TPU_MK_WHOLE_L45): call time and device module span.
D=$(dirname "$0")
for rep in 1 2; do
  for cfg in "--L 44 --tail" "--L 45"; do
    echo "CFG $cfg rep $rep"
    python3 "$D/tpu_bench_wholestep_call.py" $cfg --buckets 4,16 --ctx 3000 --iters 20 2>&1 | grep -E '^B=.*median|Error|Traceback' | tail -4
  done
done
for cfg in "--L 44 --tail" "--L 45"; do
  t=/tmp/p$(echo $cfg | tr -d ' -')
  echo "PROF $cfg"
  python3 "$D/tpu_bench_wholestep_call.py" $cfg --buckets 4 --ctx 3000 --iters 5 --profile $t --profile-iters 2 2>&1 | grep -E '^B=.*median|Error' | tail -2
  python3 "$D/parse_perfetto.py" $t 20 2>&1 | grep -vE "^thread" | tail -40
done
