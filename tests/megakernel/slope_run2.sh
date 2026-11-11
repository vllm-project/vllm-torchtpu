# Follow-up: device module time (xplane "XLA Modules") at b32/b48/b64, base and rel_fetch; long timeouts, raw tail on failure.
cd /workspace/vllm-torchtpu
export VLLM_TPU_MK_HC_FP8=1 VLLM_TPU_MK_QSEL_ASYNC=1 VLLM_TPU_MK_STACKED_QSA_WRITES=1 VLLM_TPU_MK_WHOLE_BATCHED_INPUTS=1 \
  VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE=1 VLLM_TPU_MK_BUCKET_TUNE=1 VLLM_TPU_MK_WHOLE_L45=1 VLLM_TPU_MK_SCATTER_DEEP=1
D=/root/cloud-devkit/vllm-torchtpu/tests/megakernel
COMMON="--L 45 --shard --loop --union --budget 512 --max-groups 65536 --ctx 5000 --iters 5"
for cfg in "" "rel_fetch=true"; do for b in 32 48 64; do
  echo "RUN b$b [$cfg]"
  VLLM_TPU_MK_TILES="$cfg" timeout 3000 python3 $D/tpu_bench_wholestep_call.py --buckets $b $COMMON \
    --profile /tmp/p_${b}_${#cfg} --profile-iters 3 > /tmp/log_$b 2>&1; echo "rc=$?"
  grep -E "call median" /tmp/log_$b || tail -5 /tmp/log_$b | cut -c1-250
  python3 $D/analyze_scopes.py /tmp/p_${b}_${#cfg} 3 2>&1 | grep -E "XLA Modules|qwen38_wholestep_decode_megakernel" | head -3 | cut -c1-120
done; done
