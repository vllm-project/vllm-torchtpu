# Expert-phase ablation: device module time per call, skip none vs experts, served flags (GMM diagnosis).
cd /workspace/vllm-torchtpu
export VLLM_TPU_MK_HC_FP8=1 VLLM_TPU_MK_QSEL_ASYNC=1 VLLM_TPU_MK_STACKED_QSA_WRITES=1 VLLM_TPU_MK_WHOLE_BATCHED_INPUTS=1 \
  VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE=1 VLLM_TPU_MK_BUCKET_TUNE=1 VLLM_TPU_MK_WHOLE_L45=1 VLLM_TPU_MK_SCATTER_DEEP=1
D=/root/cloud-devkit/vllm-torchtpu/tests/megakernel
COMMON="--L 45 --shard --loop --union --budget 512 --max-groups 65536 --ctx 5000 --iters 5"
for b in 4 16 32 64; do for sk in "" experts; do
  echo "RUN b$b skip=[$sk]"
  VLLM_TPU_MK_DEBUG_SKIP="$sk" timeout 3000 python3 $D/tpu_bench_wholestep_call.py --buckets $b $COMMON \
    --profile /tmp/p_${b}_${sk:-none} --profile-iters 3 > /tmp/log_${b}_${sk:-none} 2>&1; echo "rc=$?"
  grep -E "call median" /tmp/log_${b}_${sk:-none} || tail -5 /tmp/log_${b}_${sk:-none} | cut -c1-250
  python3 $D/analyze_scopes.py /tmp/p_${b}_${sk:-none} 3 2>&1 | grep -E "XLA Modules|qwen38_wholestep_decode_megakernel" | head -3 | cut -c1-160
done; done
