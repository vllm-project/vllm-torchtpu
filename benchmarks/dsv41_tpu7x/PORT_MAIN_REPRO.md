# DeepSeek-V4.1-Flash on TPU7x — port onto upstream main: repro recipe

One 1P1D deployment (1 prefill host + 1 decode host, 4 chips each) built from the fork ported onto
upstream `vllm-torchtpu` main, validated 2026-11-16.

## 1. What is being reproduced

| Item | Value |
|---|---|
| vllm-torchtpu | branch `cpk/dsv41-mh-decode-main` = upstream main `aba4aad88` + 4 port commits (validated source tree `7bc0c2b1f`) + 1 commit that only adds `benchmarks/` files |
| vLLM | tag `v0.33.0` + 4 patches in `benchmarks/dsv41_tpu7x/patches/vllm-v0.33.0/` |
| Runtime (upstream main's pins) | torch 2.13.0, torch_tpu 0.2.0, libtpu 0.0.50, jax 0.11.2, tpu-sync-torch 0.0.1.dev20261029081917 |
| Shape | 1P = 4 × (TP2, EP2) 20-layer CED prefill; 1D = DP8 / TP1 / EP8, DSpark K=5, `max-num-seqs 128`, MNBT 768 |
| Required settings on this runtime | `VLLM_DSV41_STREAMIDX_SCHED_GROUPS=0` (both roles), `PREFILL_4X_TP2_MNBT=8192` (prefill), `VLLM_DSV41_CSA_FUSED_MAX_TOKENS=768` (decode) |

## 2. Checkouts

```bash
git clone https://github.com/vllm-project/vllm-torchtpu.git vtt && git -C vtt checkout cpk/dsv41-mh-decode-main
git clone https://github.com/vllm-project/vllm.git vllm && git -C vllm checkout v0.33.0
git -C vllm am ../vtt/benchmarks/dsv41_tpu7x/patches/vllm-v0.33.0/*.patch     # 4 commits
```

The run below was driven from the local project layout, which also holds the staged wheels
(torch 2.13.0+cpu, torchvision 0.28.0, frozendict, immutabledict) and the checkpoint/GCS settings:

```bash
cd /usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/scripts/pd
export KUBE_CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps
gcloud auth login --update-adc        # kubectl and the checkpoint fetch both need a fresh token
```

## 3. One command: provision, gates, decode sweeps

```bash
VTT_DIR=/path/to/vtt VLLM_DIR=/path/to/vllm \
EXTRA_ENV="VLLM_DSV41_STREAMIDX_SCHED_GROUPS=0 PREFILL_4X_TP2_MNBT=8192" \
bash validate_port_main.sh ../../runs/pd/port_main_$(date +%Y%m%d) 3
```

It waits for two free `2x2x1` hosts, provisions pods `chaitanyapk-dsv41-pd-{pf,dc}-pm`, boots both
roles, runs the PD gates, then three decode sweeps. Spot preemptions are retried; a real server
failure stops it with the tracebacks saved in the output directory. Expect about 40 minutes when the
hosts hold (prefill up in ~10, decode in ~20–35).

Equivalent manual launch:

```bash
python3 -u run_pd.py --ctx $KUBE_CTX --suffix pm --tag dsv41_port_main \
  --vtt /path/to/vtt --vllm /path/to/vllm --runtime main \
  --prefill-ced 1 --decode-mnbt 768 --enable-dspark 1 --gpu-mem 0.85 \
  --env PREFILL_MODE=4x_tp2 --env PREFILL_4X_TP2_GPU_MEM=0.92 \
  --env VLLM_DSV41_STREAMIDX_SCHED_GROUPS=0 --env PREFILL_4X_TP2_MNBT=8192 \
  --decode-env MAX_NUM_SEQS=128 --decode-env "COMPILE_SIZES=[16, 40, 48, 64, 96, 192, 384, 512, 768]" \
  --decode-env VLLM_DSV41_CSA_FUSED_MAX_TOKENS=768 \
  --decode-env ENABLE_PREFIX_CACHING=1 --decode-env PREFIX_CACHE_RETENTION_INTERVAL=1024 \
  --keep-pods --allow-foreign --resume --pending-timeout 3600
# decode sweep on the live decode pod (discard the first, cold-cache repeat):
#   x1_step_anatomy.py --decode http://127.0.0.1:9400 --ctx 4000 --out-len 1024 --concs 256,512 --dp 8 --k 5
```

`--runtime main` installs upstream's pinned runtime in the pods instead of the fork's older wheels.
Decode prefix caching is on only so the decode-only sweep can share its warm prefix.

## 4. Pass criteria

| Check | Expected |
|---|---|
| `[GATE-2+2]` | `pd_passed=True`, `local_passed=True` |
| `[GATE-USA]` | `pd_states=50/50 local_states=50/50 pd==local=True` |
| `[GATE-EQ]` n = 64, 700, 1500, 5000, 20000 | `equal=True` on all five |
| `[PD-GATES]` | `hard_gates_passed=True` |
| KV transfer | `STAGE3_COMPLETED_SENDS 102`, `STAGE3_SKIPPED_SENDS 0`, `STAGE3_CHECK_OK 1` |
| Server logs, both roles | no `annotation groups with gaps`, no `RuntimeProgramAllocationFailure`, no `sized globally` |

## 5. Expected decode throughput

Decode-only sweep, K=5, greedy, 4K context, 1,024 output tokens, warm repeats 2 and 3, per 4-chip host:

| Sequences per rank | Port on `aba4aad88` | Fork on its old runtime |
|---|---|---|
| 64 (512 concurrent) | 21,951 / 21,906 tok/s (about 5,480 per chip, 43 tok/s per user) | 21,162 / 21,233 |
| 32 (256 concurrent) | 16,534 / 16,559 tok/s (about 4,140 per chip, 65 tok/s per user) | 15,616 / 15,502 |

Acceptance length 3.10–3.11 (fork: 3.10). Step fit `23.5 + 0.128 × tokens/rank` ms. A repeat should
land within about 3 % of these.

## 6. Optional: compare tokens with the fork's recorded outputs

```bash
kubectl --context $KUBE_CTX cp port_vs_fork_tokens.py chaitanyapk-dsv41-pd-dc-pm:/tmp/
kubectl --context $KUBE_CTX cp <fork run>/prefill__pd_gates.json chaitanyapk-dsv41-pd-dc-pm:/tmp/fork_ref_gates.json
kubectl --context $KUBE_CTX exec chaitanyapk-dsv41-pd-dc-pm -- \
  python3 -u /tmp/port_vs_fork_tokens.py /tmp/fork_ref_gates.json http://127.0.0.1:9400
```

Expected: `ALL IDENTICAL: True` (five 64-token greedy continuations and the 50-states answer). The
fork reference used here is `runs/pd/20261109-064334_dsv41_validate_best_k5_mnbt768_e1/prefill__pd_gates.json`.

## 7. Known limits of this recipe

- **Prefill chunk is 8,192, not the fork's 16,384.** At 16,384 the prefill program needs about 5.4 GB
  at load and only 3.4 GB is reservable on this runtime. Prefill throughput at 8,192 has not been
  measured, and the cause of the extra memory is not known (it is not the libtpu version).
- **Indexer scheduling groups are off.** With them on, the stream-index kernel fails to compile on
  this runtime (`annotation groups with gaps`), on libtpu 0.0.50 and 0.0.48 alike.
- **Only the decode-only sweep is measured.** The 600 s empirical-ISL production benchmark has not been
  run on the ported stack.
- **Draft acceptance varies between deployments.** One earlier deployment of the same code measured
  acceptance 3.35–3.39 and 24.6–25.0K tok/s at 64 sequences per rank; two later ones measured 3.10–3.11.
  Outputs were token-identical to the fork in every case. Unexplained.
- The driver was run from the local `dsv41/scripts/pd` layout. The copy under
  `benchmarks/dsv41_tpu7x/scripts/pd` is identical, but running from there with `--runtime main` is
  untested (the torch and frozendict wheels are not in the repo).

## 8. Evidence

| Run | Location |
|---|---|
| Rebased stack (`aba4aad88`): gates, sweeps, token comparison | `benchmarks/dsv41_tpu7x/results/port_main/rebased_aba4aad88/` (local: `runs/pd/port_main_rebased_aba4aad88/`, `runs/pd/20261116-122630_dsv41_port_main_validate_pm/`) |
| Pre-rebase stack (`06f761829`) | `benchmarks/dsv41_tpu7x/results/port_main/pre_rebase_06f761829/` |
| Pre-rebase validated tree, kept as a ref | branch `cpk/dsv41-mh-decode-main-validated-06f761829` |
