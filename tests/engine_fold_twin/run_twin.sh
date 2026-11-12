#!/bin/bash
# Fold twin on one exos pod: hooks via .pth, real weights copied once to /dev/shm, arms
# off / on / nockpt one after the other, then the comparisons. Logs live in $W (the output
# dir is write-only) and are copied to $O as each step ends.
set -u
D=/root/fab
O=${CDK_OUTPUT_DIR:-/tmp/foldtwin_out}
W=/tmp/foldtwin
mkdir -p $O $W
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp $D/ab/foldcp_force.py $D/ab/foldtwin_hook.py $SP/
printf "import foldcp_force\nimport foldtwin_hook\n" > $SP/foldtwin.pth
echo "hooks installed in $SP"
export WMB_CORE=$D/wmb/core
export FOLDTWIN_BURN=${FOLDTWIN_BURN:-16000}
SRC=/exos/gcs/cloud-devkit/ckpt/hf/Qwen/Qwen3.8-Flash-Next-FP8
MD=/dev/shm/q38f_model
df -h /dev/shm /tmp | tee $W/df.txt
if [ -d $SRC ]; then
  FREE=$(df -B1G --output=avail /dev/shm | tail -1 | tr -d ' ')
  if [ "$FREE" -gt 230 ]; then
    t0=$(date +%s); mkdir -p $MD
    ls $SRC | grep -vE '^\.' | xargs -P 24 -I{} cp -r $SRC/{} $MD/
    echo "model copy $(( $(date +%s) - t0 )) s, $(du -sh $MD | cut -f1)"
    export FOLDTWIN_MODEL_DIR=$MD
  else
    echo "shm free ${FREE}G < 230G: load from the fuse path"; export FOLDTWIN_MODEL_DIR=$SRC
  fi
else
  echo "NO $SRC: dummy weights"; 
fi
export FOLDTWIN_GSM=$D/steps/gsm72.json
for ARM in ${FOLDTWIN_ARMS:-off on on_sp0 on_moe on_qsa}; do
  if [ $ARM = off ]; then export FOLDCP_FORCE=0; else export FOLDCP_FORCE=1; fi
  if [ $ARM = nockpt ]; then export FOLDTWIN_NOCKPT=1; else export FOLDTWIN_NOCKPT=0; fi
  export FOLDTWIN_LOG=$W/sched_$ARM.jsonl
  rm -f $FOLDTWIN_LOG
  timeout 3300 python3 $D/ab/fold_twin.py $ARM $D/steps/${FOLDTWIN_PLAN:-c32-twin.steps.json} $W > $W/arm_$ARM.log 2>&1
  echo "ARM=$ARM EXIT=$?"
  grep -aE "GSM |ARM |FINGERPRINT|engine init|FOLDCP_FORCE|FOLDTWIN|worker exports|exported [0-9]+ checkpoints|folded [0-9]+ tails|TAIL_FOLD: (on|disabled)|Traceback|Error: |DUMP |PHASE |sched step|  [ABRC][0-9]+ P=" $W/arm_$ARM.log | head -150
  tail -5 $W/arm_$ARM.log
  cp $W/arm_$ARM.log $W/arm_$ARM.json $W/sched_$ARM.jsonl $W/groups_$ARM.json $W/cols_${ARM}_*.json $O/ 2>/dev/null
done
for PAIR in "on off" "on_sp0 off" "on_moe off" "on_qsa off"; do
  python3 $D/ab/fold_twin_compare.py $W $PAIR > $W/compare_${PAIR// /_}.txt 2>&1
  cat $W/compare_${PAIR// /_}.txt | head -400
  cp $W/compare_${PAIR// /_}.txt $O/
done
ls -la $W | head -80
