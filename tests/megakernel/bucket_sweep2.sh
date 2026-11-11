# Sweep 2: rel_fetch on every bucket + combos / ring trades.
cd /workspace/vllm-torchtpu
export VLLM_TPU_MK_HC_FP8=1 VLLM_TPU_MK_QSEL_ASYNC=1 VLLM_TPU_MK_STACKED_QSA_WRITES=1 VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE=1
B=/root/cloud-devkit/vllm-torchtpu/tests/megakernel/tpu_bench_wholestep_call.py
run() { echo "CFG [$2] buckets $1"; VLLM_TPU_MK_TILES="$2" timeout 1800 python3 $B --buckets $1 --iters 20 --shard --loop --union --budget 512 --hash 2>&1 \
    | grep -E "HASH|call median|Error|Oom|OOM|exceeds" | grep -v TPU_WORKER | cut -c1-200; }
for cfg in "" "rel_fetch=true" "rel_fetch=true,lazy=false" "rel_fetch=true,nbe=14,nbd=8" "rel_fetch=true,nbe=10,nbd=12" "rel_fetch=true"; do run 4,8,16 "$cfg"; done
for cfg in "" "rel_fetch=true" "rel_fetch=true,lazy=false" "rel_fetch=true,nbe=8,nbd=8" ""; do run 24,32 "$cfg"; done
