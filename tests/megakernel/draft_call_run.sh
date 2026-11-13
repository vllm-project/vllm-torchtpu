cd /workspace/vllm-torchtpu
B=/root/cloud-devkit/vllm-torchtpu/tests/megakernel/tpu_bench_draft_call.py
python3 $B --reqs 1,4,16 --T 1 --served-flags --iters 5 --profile /tmp/dprof 2>&1 | grep -E "T=|Error|File \"/(root|workspace)"
python3 $B --reqs 1,4 --T 4 --served-flags --iters 5 --profile /tmp/dprof4 2>&1 | grep -E "T=|Error|File \"/(root|workspace)"
