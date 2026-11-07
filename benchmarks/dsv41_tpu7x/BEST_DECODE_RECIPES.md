# DeepSeek-V4.1-Flash on TPU7x — Best-Case Decode: Numbers, Manifest, Repro

Units: one TPU7x host = `tpu7x-8` = `2x2x1` = **4 chips**. Throughput is always labelled engine-total / per node (4 chips) / per chip / per user.

## 1. Headline

| # | Recipe | Chips | Decode shape | Point | Engine-total tok/s | Per node | Per chip | Per user | Step ms |
|---|---|---|---|---|---|---|---|---|---|
| A | Two-host ICI decode unit (decode-only) | 8 (2x2x2) | DP16/TP1/EP16, DSpark K=5, async | 512 conc (32/rank), ctx 20K | **26,727** | 13,363 | **3,341** | 52.2 | 55.8 |
| B | Single-host decode-only | 4 (2x2x1) | DP8/TP1/EP8, K=5, async | 256 conc (32/rank), ctx 20K | **12,937** | 12,937 | **3,234** | 50.5 | 57.5 |
| C | 1P1D production shape (71% hit, 600 s, empirical ISL) | 4P + 4D | 1P 4×(TP2,EP2) 20L-CED; 1D DP8/EP8 K=5, MNBT 384, 64-tok tail | n=384 sessions | **11,992** | 11,992 (D node) | **2,998 / D-chip** (1,499 / total chip) | 27.6 med | — |

Reference (InferenceX, GB300): Dynamo disagg 13.7–18.8K tok/s per decode GPU @ 79–203 tok/s/user; GB300 aggregated ~3.1K @ 62.

## 2. Benchmarks

**A — two-host ICI DP16/EP16** ([k5 20K](results/best_decode/x1/x1_mh16_k5_default_warm.log) · [k5 100K](results/best_decode/x1/x1_mh16_k5_ctx100k_b.log) · [k0](results/best_decode/x1/x1_mh16_k0_warm.log) · [ICI check](results/best_decode/mh/ici_check.txt))

| K | ctx | conc | seqs/rank | step ms | engine tok/s | per chip | per user |
|---|---|---|---|---|---|---|---|
| 5 | 20K | 128 | 8 | 32.2 | 11,667 | 1,458 | 91.1 |
| 5 | 20K | 256 | 16 | 41.1 | 18,023 | 2,253 | 70.4 |
| 5 | 20K | 512 | 32 | 55.8 | **26,727** | **3,341** | 52.2 |
| 5 | 100K | 512 | 32 | 58.7 | 24,472 | 3,059 | 47.8 |
| 0 | 20K | 1024 | 64 | 39.1 | 26,205 | 3,276 | 25.6 |
| 0 | 2K | 6144 | 384 | ~78 | ~79,000 | ~9,800 | 13 |

Fit: `step_ms = 24.8 + 0.163·tokens/rank` (K=5, 20K); `25.4 + 0.175·t` (100K); `26.8 + 0.188·t` (K=0). ICI verified: both pods `TPU_ACCELERATOR_TYPE=tpu7x-16, TPU_TOPOLOGY=2x2x2, TPU_HOST_BOUNDS=1,1,2`; NIC ≤ 0.3 MB/s under c=512 load (EP a2a over DCN would be ≥ 1 GB/s).

**B — single-host DP8/EP8** ([log](results/best_decode/x1/x1_k5_ctx20k_repeat.log))

| conc | seqs/rank | step ms | engine tok/s | per chip | per user |
|---|---|---|---|---|---|
| 64 | 8 | 38.8 | 4,737 | 1,184 | 74.0 |
| 128 | 16 | 45.2 | 8,012 | 2,003 | 62.6 |
| 256 | 32 | 57.5 | **12,937** | **3,234** | 50.5 |

Fit `32.6 + 0.130·t`; an earlier run without a co-scheduled prefill pod fit `27.3 + 0.126·t` → 14,298 tok/s (3,575/chip, 56/user).

**C — 1P1D, 71% hit, 600 s window, 0 failed requests** ([n384](results/best_decode/e1/e6c_tail64_hit71_n384.json) · [n256](results/best_decode/e1/e4c_hit71_n256.json))

| Point | n | out tok/s (D node) | per D-chip | per total chip | TTFT med / P90 | stream tok/s med / P10 | ITL med ms | τ | uncached prefill tok/s per P-chip |
|---|---|---|---|---|---|---|---|---|---|
| tail-64 (E6-c) | 384 | **11,992** | **2,998** | 1,499 | 1.04 / 2.29 s | 27.6 / 22.9 | 36.0 | 3.18 | 13,287 |
| tail-128 (E4-c) | 256 | 10,972 | 2,743 | 1,371 | 0.94 / 1.80 s | 38.0 / 32.2 | 26.0 | 3.27 | 11,708 |

## 3. Summary

Best decode on TPU7x is ~3.3K tok/s/chip at ~50 tok/s/user, reached identically on one host (DP8) and on a two-host ICI slice (DP16); the two-host unit scales linearly with no floor penalty, so ICI/EP communication is not the limiter. The step model is `≈25 ms fixed + 0.13–0.19 ms per token per rank`, independent of context (2K–100K) and of K; the fixed floor plus the `max-num-seqs 64 × MNBT 384` admission budget are what cap per-chip output. Dead ends: K=3, K=7 (never compiled), async off (+8 ms), `DSPARK_SHARDED_CTX_KV=0`, MNBT 768 (compile-range `ShapeSpecializationError`). In the production 1P1D shape the 64-token handoff tail moves the admission knee from n=256 to n=384 (+9% output) at a TTFT/interactivity cost. Next levers: device-level xprof of one decode step to attribute the floor, then unblocking MNBT ≥ 768 / seqs ≥ 96 per rank.

## 4. Manifest

| Item | Value |
|---|---|
| vllm-torchtpu | `repos/vtt-combined`, branch **`cpk/dsv41-mh-decode`** (`04daa1857`, `4f6f05de0`) on `cpk/dsv41-combined` @ `61b27298c` |
| vllm | `repos/vllm-decode`, branch `cpk/dsv41-vllm-decode` @ `51e61e3bf0` (unchanged) |
| Wheels | `benchmarks/dsv41_tpu7x/scripts/pd/whls/` incl. `tpu_sync_torch-0.0.1.dev20260923040928` |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash`, staged to `/dev/shm` by `in_pod_setup.sh` |
| Dataset (C) | `requests_mocked.jsonl.xz` (empirical ISL / tool-calling mock) |
| Cluster | `gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps`; multi-host pool `cpk-spot-2x2x2` (tpu7x-standard-4t, 2x2x2, workload policy `cpk-v7x-2x2x2-wp`, us-central1-c, spot) |
| Decode serve args | `--tensor-parallel-size 1 --enable-expert-parallel --kv-cache-dtype fp8 --max-model-len 131072 --max-num-seqs 64 --max-num-batched-tokens 384 --gpu-memory-utilization 0.85 --async-scheduling --compilation-config '{"compile_sizes":[16,40,48,64,96,192,384]}' --speculative-config '{"method":"dspark","num_speculative_tokens":5}'` (+ `--enable-prefix-caching --prefix-cache-retention-interval 1024` for A/B; off on the 1D role in C) |
| Decode env | `USE_MOE_FUSED_EP_KERNEL=1 MOE_FUSED_EP_KERNEL_IMPL=v2 MOE_FUSED_EP_ENABLE_W4A8=1 MOE_FUSED_EP_V2_SHARDED_PLAN=1 VLLM_TPU_MOE_FP8_DISPATCH=1 VLLM_DSV41_FAST_GROUPED_ATTN=1 VLLM_DSV41_REPL_SHARED_EXPERTS=1 VLLM_DSV41_TP_INDEXER=1 VLLM_DSV41_FUSED_SEAM_QUANT=1 VLLM_DSV41_FUSED_Q_ROPE=1 VLLM_DSV41_DSPARK_SHARDED_CTX_KV=1` (full list in `scripts/mh/launch_dsv41_mh_decode.sh`) |
| Prefill (C) | `PREFILL_MODE=4x_tp2 PREFILL_4X_TP2_GPU_MEM=0.92 --prefill-ced 1`, 4 servers ports 8400-8403, toy proxy `:8000`, Raiden Stage-3 KV over DCN, `VLLM_DSV41_PD_HANDOFF_TAIL_TOKENS=64` |
| Pass gates | A: ICI check (§2). C: `[prefill] HEALTHY`, `[decode] HEALTHY`, `STAGE3_CHECK_OK 1`, `hard_gates_passed=True`, `[GATE-EQ] n=5 equal=5`; harness `pass=measured_600s_5_or_n_trunks_exact_temps`, `failed_requests=0` |

## 5. Reproduction

```bash
cd /usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41
export KUBE_CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps
```

**A — two-host ICI DP16/EP16**
```bash
# 1. slice (retry loop handles GCE_STOCKOUT; creates pool with workload policy if missing)
bash scripts/mh/acquire_and_launch.sh | tee runs/logs/acquire_and_launch.log
#    equivalent manual launch once 2 nodes of cpk-spot-2x2x2 are Ready:
python3 -u scripts/mh/run_mh_decode.py --ctx $KUBE_CTX --topology 2x2x2 --hosts 2 --suffix mh1 \
  --dspark-tokens 5 --max-num-seqs 64 --mnbt 384        # ~25 min boot; add --reuse --no-setup to relaunch
# 2. gate
bash scripts/mh/mh_ici_check.sh
# 3. sweeps  (tag K concs ctx out_len) → runs/pd/x1_results/x1_<tag>.{json,log}
bash scripts/mh/mh_sweep.sh mh16_k5 5 128,256,512 20000 256
bash scripts/mh/mh_sweep.sh mh16_k5_100k 5 128,256,512 100000 256
python3 -u scripts/mh/run_mh_decode.py --ctx $KUBE_CTX --suffix mh1 --reuse --no-setup --dspark-tokens 0
bash scripts/mh/mh_sweep.sh mh16_k0 0 128,256,512,1024 20000 256
# 4. tear down
kubectl --context $KUBE_CTX delete job chaitanyapk-dsv41-mh-mh1
gcloud container node-pools delete cpk-spot-2x2x2 --cluster bodaborg-tpu7x-spot-sps --region us-central1 --project cloud-tpu-multipod-dev --quiet
```

**B — single-host DP8/EP8** (uses the 1D pod from C, relaunched with prefix caching on)
```bash
cd scripts/pd; P=chaitanyapk-dsv41-pd-dc-e1
kubectl --context $KUBE_CTX cp x1_relaunch.sh $P:/tmp/; kubectl --context $KUBE_CTX cp x1_step_anatomy.py $P:/tmp/
kubectl --context $KUBE_CTX exec $P -- bash /tmp/x1_relaunch.sh ENABLE_PREFIX_CACHING=1 PREFIX_CACHE_RETENTION_INTERVAL=1024 DSPARK_TOKENS=5
# wait for http://127.0.0.1:9400/health in-pod, then:
kubectl --context $KUBE_CTX exec $P -- bash -c "nohup setsid python3 -u /tmp/x1_step_anatomy.py --decode http://127.0.0.1:9400 \
  --ctx 20000 --out-len 512 --concs 64,128,256 --dp 8 --k 5 --out /tmp/x1_k5.json > /tmp/x1_k5.log 2>&1 < /dev/null & disown"
kubectl --context $KUBE_CTX cp $P:/tmp/x1_k5.log ../../runs/pd/x1_results/
```

**C — 1P1D production shape**
```bash
cd scripts/pd
python3 -u run_pd.py --ctx $KUBE_CTX --suffix e1 --tag dsv41_e1 --prefill-ced 1 --decode-mnbt 384 --gpu-mem 0.85 \
  --env PREFILL_MODE=4x_tp2 --env PREFILL_4X_TP2_GPU_MEM=0.92 --env VLLM_DSV41_PD_HANDOFF_TAIL_TOKENS=64 \
  --keep-pods --allow-foreign --resume --pending-timeout 3600     # omit TAIL env for the 128-tail point
# (provision_e1_loop.sh / e1_orchestrate.sh wrap this in a capacity-wait + retry loop)
# check gates in the run_pd output, then:
HIT=0.71 DUR=600 bash run_e1a_levels.sh ../../runs/pd/e1_results 256,384 40 tail64_hit71
# → runs/pd/e1_results/tail64_hit71_n{256,384}.{json,log} + per-request JSONL
```

**Push**
```bash
cd repos/vtt-combined && git log --oneline cpk/dsv41-combined..cpk/dsv41-mh-decode && git push -u origin cpk/dsv41-mh-decode
```
