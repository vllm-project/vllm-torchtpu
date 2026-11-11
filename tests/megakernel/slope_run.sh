# c32 slope: served-mode whole-step call at 32/48/64/96 rows (L=45), then scopes at b64/b96.
cd /workspace/vllm-torchtpu
export VLLM_TPU_MK_HC_FP8=1 VLLM_TPU_MK_QSEL_ASYNC=1 VLLM_TPU_MK_STACKED_QSA_WRITES=1 \
  VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE=1 VLLM_TPU_MK_BUCKET_TUNE=1 VLLM_TPU_MK_WHOLE_L45=1 VLLM_TPU_MK_SCATTER_DEEP=1
D=/root/cloud-devkit/vllm-torchtpu/tests/megakernel
COMMON="--L 45 --shard --loop --union --budget 512 --max-groups ${MAXG:-2048} --ctx ${CTX:-5000} --iters 20"
f() { grep -E "call median|HASH|Error|Oom|OOM|exceeds|RESOURCE" | grep -v TPU_WORKER | cut -c1-220; }
for b in 32 48 64 96; do echo "TIME b$b"; timeout 1500 python3 $D/tpu_bench_wholestep_call.py --buckets $b $COMMON 2>&1 | f; done
echo "TIME rel_fetch b48/b64/b96 (not in BUCKET_TUNE)"
for b in 48 64 96; do VLLM_TPU_MK_TILES=rel_fetch=true timeout 1500 python3 $D/tpu_bench_wholestep_call.py --buckets $b $COMMON 2>&1 | f; done
for b in 64 96 32; do
  echo "SCOPES b$b"
  VLLM_TPU_MK_SCOPES=2 timeout 1500 python3 $D/tpu_bench_wholestep_call.py --buckets $b $COMMON --iters 5 \
    --profile /tmp/prof_b$b --profile-iters 3 2>&1 | f
  python3 $D/analyze_scopes.py /tmp/prof_b$b 3 2>&1 | head -120
done
