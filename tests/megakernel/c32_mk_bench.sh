#!/bin/bash
# c32 MK microbench: device module time per whole-step call (L=45, served MK flags, TP8, FP8,
# ctx ~5000, nreq = rows/4, max_groups 65536) per row bucket and phase-skip mode.
# Runs inside an exos-simple-run-command pod; submitted by ~/q38f/d1/c32b/submit_mk.sh.
#   BUCKETS   row buckets            (default "32 48 64 96")
#   SKIPS     VLLM_TPU_MK_DEBUG_SKIP (default "none experts"; none = full kernel)
#   TMO       seconds per run        (default 2400)
#   MK_ENV    extra env, space separated (e.g. "VLLM_TPU_MK_ROW_BUCKET=1")
#   MK_ENVS   several arms back to back, ';' separated, each a MK_ENV value
#             (e.g. "VLLM_TPU_MK_ROW_BUCKET=0;VLLM_TPU_MK_ROW_BUCKET=1"); overrides MK_ENV
#   BENCH_EXTRA extra tpu_bench_wholestep_call args (e.g. "--hash": bitwise check across arms)
# Overlay: a --map-dir of a src tree at /root/ovsrc is imported ahead of the image's.
D=$(cd "$(dirname "$0")" && pwd)
if [ -d /root/ovsrc/vllm_torchtpu ]; then export PYTHONPATH=/root/ovsrc${PYTHONPATH:+:$PYTHONPATH}; fi
cd /tmp
python3 -c "import vllm_torchtpu, os; print('IMPORT_PATH', vllm_torchtpu.__file__)" 2>&1 | grep IMPORT_PATH
python3 - <<'PYEOF'
from vllm_torchtpu.megakernel import stream
body = open(stream.__file__).read().split("def fetch_rel")[1].split("def flush_rel")[0]
print("REL_FETCH_PARTIAL_CYCLE_FIX", "g < sp.T_TOTAL" in body)
PYEOF
export VLLM_TPU_MK_HC_FP8=1 VLLM_TPU_MK_QSEL_ASYNC=1 VLLM_TPU_MK_STACKED_QSA_WRITES=1 VLLM_TPU_MK_WHOLE_BATCHED_INPUTS=1 \
  VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE=1 VLLM_TPU_MK_BUCKET_TUNE=1 VLLM_TPU_MK_WHOLE_L45=1 VLLM_TPU_MK_SCATTER_DEEP=1
COMMON="--L 45 --shard --loop --union --budget 512 --max-groups 65536 --ctx 5000 --iters 10"
IFS=';' read -ra ARMS <<< "${MK_ENVS:-${MK_ENV:-}}"
[ ${#ARMS[@]} -eq 0 ] && ARMS=("")
for ai in "${!ARMS[@]}"; do (
for kv in ${ARMS[$ai]}; do export "$kv"; done
echo "ARM a$ai [${ARMS[$ai]}] env: $(env | grep -E '^VLLM_TPU_MK_' | sort | tr '\n' ' ')"
for b in ${BUCKETS:-32 48 64 96}; do for sk in ${SKIPS:-none experts}; do
  s=$sk; [ "$s" = none ] && s=
  tag=a${ai}_b${b}_${sk}
  echo "RUN a$ai b$b skip=$sk start $(date -u +%H:%M:%S)"
  VLLM_TPU_MK_DEBUG_SKIP="$s" timeout ${TMO:-2400} python3 $D/tpu_bench_wholestep_call.py --buckets $b $COMMON ${BENCH_EXTRA:-} \
    --watchdog 900 --profile /tmp/p_$tag --profile-iters 3 > /tmp/log_$tag 2>&1
  rc=$?
  echo "rc=$rc end $(date -u +%H:%M:%S)"
  grep -E "STAGE|HASH|call median|RESOURCE_EXHAUSTED|Error" /tmp/log_$tag | grep -v TPU_WORKER | cut -c1-250 | head -8
  if [ $rc -ne 0 ]; then
    # Watchdog stacks: the innermost frames of the main thread show where it waited.
    grep -E 'File "|Thread|Current thread' /tmp/log_$tag | tail -12 | cut -c1-200
  fi
  # Perfetto (go/cdk-perfetto) is built from the PJRT VLOG lines in the job log.
  if [ -n "${TPU_VMODULE:-}" ]; then grep -E 'tpu_pjrt_client|pjrt_stream_executor|tpu_pjrt_compiler' /tmp/log_$tag; fi
  if [ -d /tmp/p_$tag ]; then
    python3 $D/analyze_scopes.py /tmp/p_$tag 3 2>&1 | grep -E "XLA Modules|qwen38_wholestep_decode_megakernel" \
      | head -3 | cut -c1-160 | sed "s/^/[$tag] /"
  fi
done; done
); done
echo "ALL DONE $(date -u +%H:%M:%S)"
