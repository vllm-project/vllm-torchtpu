# Per-bucket MK knob sweep (VLLM_TPU_MK_TILES); served-like flags.
cd /workspace/vllm-torchtpu
export VLLM_TPU_MK_HC_FP8=1 VLLM_TPU_MK_QSEL_ASYNC=1 VLLM_TPU_MK_STACKED_QSA_WRITES=1 VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE=1
B=/root/cloud-devkit/vllm-torchtpu/tests/megakernel/tpu_bench_wholestep_call.py
for cfg in "" "nbd=14" "nbd=18" "nbd=22" "nbe=16" "nbe=20" "rel_fetch=true" "lazy=false" "dmm_unroll=4" "dense_acc_pair=true" "nbd=18,rel_fetch=true" ""; do
  echo "CFG [$cfg]"
  VLLM_TPU_MK_TILES="$cfg" timeout 1500 python3 $B --buckets ${BUCKETS:-4,8,16} --iters 20 --shard --loop --union --budget 512 --hash 2>&1 \
    | grep -E "HASH|call median|Error|Oom|OOM" | grep -v TPU_WORKER | cut -c1-200
done
