#!/bin/bash
# Whole-step MK phase ablation on TPU: per-call time with each phase skipped.
# usage: tpu_ablate_wholestep.sh [rows...]   (stage C, full geometry, L=4 and 8 -> per-group slope)
cd "$(dirname "$0")/../.."
ROWS=${*:-4 16}
for skip in "" comm experts dense attn gdn "comm,experts"; do
  echo "=== SKIP=$skip"
  MKB_SKIP=$skip timeout 2400 python3 tests/megakernel/tpu_bench_wholestep.py --stage-c --rows $ROWS \
    --layers 4 8 --chains 4 16 --identical 0 --no-ref 2>&1 | grep -E "^RESULT|Error|error" | sed "s/^/SKIP=$skip /"
done
echo ABLATE_DONE
