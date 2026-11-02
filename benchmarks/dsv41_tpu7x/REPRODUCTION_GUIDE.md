# DeepSeek-V4.1-Flash on TPU7x: End-to-End Reproduction Guide

Self-contained runbook to reproduce the **Prefill-Only**, **Decode-Only**, and **1P1D Disaggregated** benchmarks and accuracy gates for **DeepSeek-V4.1-Flash** on `tpu7x-8` (`2x2x1` = 4 physical TPU7x chips / 8 TensorCores per node) from a single unified branch.

* **Unified `vllm-torchtpu` Branch**: `cpk/dsv41-combined`
  * **GitHub Branch**: https://github.com/vllm-project/vllm-torchtpu/tree/cpk/dsv41-combined
  * **Bundled Harness Path in Repo**: [`benchmarks/dsv41_tpu7x/`](https://github.com/vllm-project/vllm-torchtpu/tree/cpk/dsv41-combined/benchmarks/dsv41_tpu7x)
* **GCS Archives (Bundles, Tarballs, Wheels, Patches & Verified Run Logs)**:
  * **Browser URL**: https://console.cloud.google.com/storage/browser/chaitanyapk-data/dsv4.1_opt/dsv4.1
  * **GCS URI**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/` & `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/phase2_4xtp2_1p1d_20261030/`
* **Model Checkpoint (`48` shards, `~476 GiB`)**: `gs://rsinghc-storage/DeepSeek-V4.1-Flash/`

---

## 1. Prerequisites & One-Time Workspace Setup (~5 mins, No TPU Required)

### 1.1 Authenticate `gcloud` & Select a GKE `tpu7x` (`2x2x1`) Cluster

```bash
gcloud auth login --update-adc
kubectl config get-contexts   # pick a cluster with free tpu7x 2x2x1 nodes (1 node for P or D; 2 nodes for 1P1D)
export CTX="gke_cloud-tpu-multipod-dev_us-central1_bodaborg-tpu7x-spot-sps"
```

### 1.2 Option A — Fast Setup from GCS Archive (Recommended, 100% Offline-Reproducible)

Restores the exact workspace (`scripts/`, `support/`, `patches/`, `whls/`, and the `repos/vtt-combined`, `repos/vllm-prefill`, and `repos/vllm-decode` git trees) directly from `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/`:

```bash
mkdir -p ~/dsv41_repro && cd ~/dsv41_repro

# 1. Pull the harness, patches, and support files
gcloud storage cp gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/repos/dsv41-harness.tar.gz .
tar -xzf dsv41-harness.tar.gz && rm -f dsv41-harness.tar.gz

# 2. Pull the 14 pinned TPU wheels (torch_tpu, tpu_sync_torch, etc.) and verify checksums
mkdir -p whls
gcloud storage cp "gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/whls/*.whl" whls/
(cd whls && sha256sum -c SHA256SUMS.txt)

# 3. Restore the unified vllm-torchtpu and vllm worktrees from GCS git bundles
mkdir -p repos
gcloud storage cp gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/repos/vllm-torchtpu-combined-latest.bundle repos/
gcloud storage cp gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/repos/vllm-prefill-a8cb9ec41.bundle repos/
gcloud storage cp gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/repos/vllm-decode-51e61e3bf0.bundle repos/

git clone -b cpk/dsv41-combined repos/vllm-torchtpu-combined-latest.bundle repos/vtt-combined
git clone -b dsv41-vllm-prefill repos/vllm-prefill-a8cb9ec41.bundle repos/vllm-prefill
git clone -b cpk/dsv41-vllm-decode repos/vllm-decode-51e61e3bf0.bundle repos/vllm-decode
```

### 1.3 Option B — Clone Directly from GitHub (`cpk/dsv41-combined`)

If cloning from GitHub, the reproduction harness is already checked into the branch at `benchmarks/dsv41_tpu7x/`:

```bash
mkdir -p ~/dsv41_repro/repos && cd ~/dsv41_repro
git clone -b cpk/dsv41-combined https://github.com/vllm-project/vllm-torchtpu.git repos/vtt-combined
cp -r repos/vtt-combined/benchmarks/dsv41_tpu7x/* .

# Build the patched vllm trees from upstream f9ad9dd6b495 + patches/0001..0004
git clone --filter=blob:none https://github.com/vllm-project/vllm.git repos/vllm-base
git -C repos/vllm-base worktree add -b dsv41-vllm-prefill ../vllm-prefill f9ad9dd6b495f7492fa975e0ac48164b090d99ad
git -C repos/vllm-prefill apply ../../patches/0001-vllm-common-f9ad9dd6b4.patch && git -C repos/vllm-prefill commit -qam common
git -C repos/vllm-base worktree add -b cpk/dsv41-vllm-decode ../vllm-decode dsv41-vllm-prefill
git -C repos/vllm-decode apply ../../patches/0002-vllm-decode-runtime-v2.patch && git -C repos/vllm-decode commit -qam decode-v2
git -C repos/vllm-decode apply ../../patches/0003-vllm-tpu-spec-chunk-align.patch && git -C repos/vllm-decode commit -qam spec-chunk-align
git -C repos/vllm-decode apply ../../patches/0004-vllm-disjoint-kv-partitions.patch && git -C repos/vllm-decode commit -qam disjoint-kv-partitions

# Fetch the 14 TPU wheels from GCS
gcloud storage cp "gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/whls/*.whl" whls/
(cd whls && sha256sum -c SHA256SUMS.txt)
```

> [!TIP]
> **If using the existing workspace on `chaitanyapk-dev1`**: Everything is already staged at `/usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41` — simply `cd /usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41` and run the commands in Sections 2–4 below.

---

## 2. Mode 1: Reproduce Standalone Prefill-Only (`1 Node = 4 Chips`, ~20 mins)

### What It Runs
* Provisions 1 `tpu7x-8` (`2x2x1`) pod (`<user>-dsv41-combined-pf-<suffix>`), downloads only the 24 Context-Encoder checkpoint shards (`shards 2..23, 47, 48`), and launches **4 independent `TP=2, EP=2` servers** (`1` per chip, `VLLM_DSV41_CED=1`, `VLLM_DSV41_PURE_20L_PREFILL=1`, `max_num_batched_tokens=16384`, `libtpu==0.0.44`) behind a round-robin proxy on `:9000`.
* Executes the 6-workload `vllm bench serve` sweep (`output_len = 1`): `128K` (`c=4, 8, 16`), `100K` (`c=8, 16`), and `32K` (`c=8`).

### Command

```bash
python3 -u scripts/prefill/run_prefill.py \
  --ctx "$CTX" \
  --thread combined \
  --suffix pf1 \
  --vtt repos/vtt-combined \
  --vllm repos/vllm-prefill
```

### Expected Results & Verification

Results are saved to `runs/combined/<timestamp>_prefill_pf1/summary.json` (and printed via `python3 scripts/compare_results.py`):

| Workload (`output_len = 1`) | Expected Node Input Throughput (`4 Chips`) | Per-Chip Throughput | Mean TTFT | Reference Archive Run |
| :--- | ---: | ---: | ---: | :--- |
| **`128K` Input (`c=8`, `n=32`)** | **`~198,541 tok/s/node`** (`82.7%` GB200) | **`~49,635 tok/s/chip`** | `~4,947 ms` | `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/runs/combined/20261021-200912_prefill_opt2/` |
| **`128K` Input (`c=16`, `n=32`)** | **`~197,891 tok/s/node`** | **`~49,473 tok/s/chip`** | `~8,610 ms` | `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/runs/combined/20261021-200912_prefill_opt2/` |
| **`100K` Input (`c=8`, `n=32`)** | **`~196,644 tok/s/node`** (`81.9%` GB200) | **`~49,161 tok/s/chip`** | `~3,810 ms` | `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/runs/combined/20261021-200912_prefill_opt2/` |
| **`100K` Input (`c=16`, `n=32`)** | **`~195,840 tok/s/node`** | **`~48,960 tok/s/chip`** | `~6,649 ms` | `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/runs/combined/20261021-200912_prefill_opt2/` |
| **`128K` Input (`c=4`, `n=16`)** | **`~186,703 tok/s/node`** | **`~46,676 tok/s/chip`** | `~2,795 ms` | `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/runs/combined/20261021-200912_prefill_opt2/` |
| **`32K` Input (`c=8`, `n=64`)** | **`~189,563 tok/s/node`** | **`~47,391 tok/s/chip`** | `~1,335 ms` | `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/runs/combined/20261021-200912_prefill_opt2/` |

---

## 3. Mode 2: Reproduce Standalone Decode-Only (`1 Node = 4 Chips`, ~35 mins)

### What It Runs
* Provisions 1 `tpu7x-8` (`2x2x1`) pod (`<user>-dsv41-combined-dc-dp8all-<suffix>`), downloads all 48 checkpoint shards into `/dev/shm/DeepSeek-V4.1-Flash`, and boots 1 `vllm serve` instance (`DP=8, TP=1, EP=8`, `VLLM_DSV41_CED=0`, **DSpark $K=5$**, `--async-scheduling`, `libtpu==0.0.48`, `compile_sizes=[48, 96, 192, 384]`, `dp_rr=1` `X-data-parallel-rank` round-robin pinning).
* Runs the multi-step coherence gates (`2+2` $\times 8$ and `USA` 50-states prompt) followed by the **100K-context (`99,936` cached + `64` uncached input, `1,024` output)** steady-state decode sweep at **`c=64` (`1x_c64`)**, **`c=256` (`4x_c256`)**, and **`c=512` (`8x_c512`)**.

### Command

```bash
# Run all three concurrency points (c=64, c=256, c=512) + accuracy gates in one pass:
python3 -u scripts/decode/run_decode.py \
  --ctx "$CTX" \
  --mode dp8_all \
  --thread combined \
  --suffix dc1 \
  --vtt repos/vtt-combined \
  --vllm repos/vllm-decode \
  --libtpu 0.0.48 \
  --dp-rr 1 \
  --buckets '[48, 96, 192, 384]'
```
*(You can also run a single concurrency point by passing `--mode dp8_c64`, `--mode dp8_c256`, or `--mode dp8_c512`.)*

### Expected Results & Verification

Results are saved to `runs/combined/<timestamp>_decode_dp8_all_dc1/`:

| Gate / Benchmark (`100K` Context, `1,024` Out) | Expected Step Time (`TPOT`) | Expected Node Throughput @ $\tau=3.5$ | Expected Natural Throughput ($\tau$) | Reference Archive Run |
| :--- | ---: | ---: | ---: | :--- |
| **Accuracy Gates (`gate_meta.json`)** | — | `2+2`: **`8/8` (`all_passed=true`)** | `USA`: **`50/50` states**, **$\tau \approx 4.49$** | `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/runs/combined/20261021-200912_decode_dp8_all_opt2/` |
| **`1x_c64` (`c=64`, `8` reqs/rank, `T=48`)** | **`~28.23 ms/step`** (`6.82 ms` TPOT) | **`~8,037.6 tok/s/node`** (`2,009/chip`) | **`~9,502.5 tok/s/node`** ($\tau \approx 4.14$) | `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/runs/combined/20261021-155644_decode_dp8_c64_xprof3/` |
| **`4x_c256` (`c=256`, `32` reqs/rank, `T=192`)** | **`~52.81 ms/step`** (`14.85 ms` TPOT) | **`~16,966.0 tok/s/node`** (`4,242/chip`) | **`~23,316.9 tok/s/node`** ($\tau \approx 4.81$) | `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/runs/combined/20261021-200912_decode_dp8_all_opt2/` |
| **`8x_c512` (`c=512`, `64` reqs/rank, 2-wave)** | `20.77 ms` Mean TPOT | `~8,542–9,023 tok/s/node` E2E wall | **`~24,648.5 tok/s/node` implied** (`6,162/chip`) | `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/runs/combined/20261021-200912_decode_dp8_all_opt2/` |

---

## 4. Mode 3: Reproduce 1P1D Disaggregated Serving (`2 Nodes = 8 Chips`, ~25 mins)

### What It Runs
* Provisions **2 `tpu7x-8` (`2x2x1`) pods** on the same cluster (`<user>-dsv41-pd-pf-<suffix>` and `<user>-dsv41-pd-dc-<suffix>`):
  * **Prefill Node (`1P`, ports `:8400..8403`, `--prefill-mode 4x_tp2`)**: **4 independent `TP=2, EP=2` servers** (`1` chip per server), `VLLM_DSV41_CED=1`, `VLLM_DSV41_PURE_20L_PREFILL=1` (20-layer Context Encoder + 128-token SWA tail), `VLLM_DISJOINT_KV_PARTITIONS=1`, `PREFILL_4X_TP2_GPU_MEM=0.92` (`1,659` blocks/partition = `212,352` tokens/chip = `849,408` tokens/node per partition across disjoint `main`, `state`, `swa_0`, `swa_1`, `swa_2` `BlockPool` queues), `MAX_NUM_BATCHED_TOKENS=16384`, `COMPILE_SIZES=[512, 2048, 8192, 16384]`, `TPURaidenConnector` (`kv_producer`) with `TP=2 → TP=1` Stage-3 KV resharding.
  * **Decode Node (`1D`, `:9400`)**: `DP=8, TP=1, EP=8`, `VLLM_DSV41_CED=0`, **DSpark $K=5$** (`ENABLE_DSPARK=1`), `--async-scheduling`, `VLLM_DISJOINT_KV_PARTITIONS=1` (`720` blocks/partition = `92,160` tokens/rank = `737,280` tokens/node per partition), `MAX_NUM_BATCHED_TOKENS=96`, `COMPILE_SIZES=[16, 40, 48, 64, 96, 128, 192, 384]`, `TPURaidenConnector` (`kv_consumer`).
  * **Disagg Proxy (`:8000` on `1P`)**: [`toy_proxy_server.py`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/repos/vtt-combined/examples/disagg/toy_proxy_server.py) with `--prefiller-ports 8400,8401,8402,8403 --prefiller-dp-size 1 --decoder-ports 9400 --decoder-dp-size 8` deterministic DP rank routing + least-inflight load balancing and multi-producer `/reset_prefix_cache` broadcast.
* Executes [`pd_gates.py`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/scripts/pd/pd_gates.py) and the Stage-3 transfer audit (`check_dsv4_stage3_log.py`):
  1. **`[GATE-2+2]`**: 16 concurrent math prompts (`pd` vs. `local`).
  2. **`[GATE-USA]`**: 512-token greedy generation on the 50-states prompt (`pd_states=50/50`, `pd==local`).
  3. **`[GATE-EQ]`**: Bitwise greedy 64-token equality (`pd == local`) + NIAH passkey (`739204`) across `n = 64, 700, 1500, 5000, 20000` input tokens.
  4. **`[BENCH-PD]` & `[BENCH-LOCAL]`**: `n=16, c=8, input=8192, output=128` end-to-end disaggregated vs. co-located benchmark.
  5. **`[BENCH-PD-C64]`**: `n=64, c=64, input=1024, output=512` high-concurrency steady-state disaggregated decode benchmark.

### 4.1 Command — Launch `1P1D` (`4 × TP=2, EP=2` 20L CED `1P`) & Run Gates

```bash
PREFILL_4X_TP2_GPU_MEM=0.92 python3 -u scripts/pd/run_pd.py \
  --ctx "$CTX" \
  --suffix p2 \
  --tag dsv41_pd_4xtp2_20l_disjoint \
  --vtt repos/vtt-combined \
  --vllm repos/vllm-decode \
  --prefill-mode 4x_tp2 \
  --prefill-ced 1 \
  --prefill-mnbt 16384 \
  --decode-mnbt 96 \
  --enable-dspark 1 \
  --allow-foreign \
  --keep-pods
```
*(Note: Pass `--prefill-mode dp8` if you want to run the single-server `DP=8, TP=1, EP=8` `1P` mode instead.)*

### 4.2 Command — Reproduce 1P1D Empirical ISL (`50:20:12:9:9`, 95% Prefix Cache Hit) Sweep

With the pods running from Step 4.1, execute [`bench_empirical_isl_dsv41.py`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/scripts/pd/bench_empirical_isl_dsv41.py) on the `1P` pod:

```bash
COMPACT_WARMUP_SUFFIX=1 python3 -u /tmp/bench_empirical_isl_dsv41.py \
  --base-url http://127.0.0.1:8000 \
  --prefill-url http://127.0.0.1:8400,http://127.0.0.1:8401,http://127.0.0.1:8402,http://127.0.0.1:8403 \
  --decode-url http://<DECODE_POD_IP>:9400 \
  --dataset /data/requests_mocked.jsonl.xz \
  --sessions 8,16,32,64,128 \
  --duration-per-n 180 \
  --cache-hit-rate 0.95 \
  --max-tokens 1024 \
  --ignore-eos \
  --dp-size 8 \
  --trunks-above-n16 40 \
  --warmup-pass-s 30 \
  --output-json /tmp/dsv41_pd_empirical_isl_95_summary.json
```

### 4.3 Expected Results & Verification

Results are saved to `runs/pd/<timestamp>_dsv41_pd_4xtp2_20l_disjoint_p2/` (see [`EMPIRICAL_ISL_1P1D_REPORT.md`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/repos/vtt-combined/benchmarks/dsv41_tpu7x/EMPIRICAL_ISL_1P1D_REPORT.md) and [`UNCACHED_PREFILL_BENCHMARK_SUMMARY.md`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/repos/vtt-combined/benchmarks/dsv41_tpu7x/UNCACHED_PREFILL_BENCHMARK_SUMMARY.md)):

| Gate / Benchmark | Expected Result (`--prefill-mode 4x_tp2`) | Reference Archive Run |
| :--- | :--- | :--- |
| **`[GATE-2+2]` & `[GATE-USA]`** | `pd=16/16`, `local=16/16`; `USA`: **`pd=50/50`, `local=50/50`** ($\tau \approx 3.95$) | `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/phase2_4xtp2_1p1d_20261030/20261030-204736_dsv41_pd_4xtp2_20l_disjoint_p2/` |
| **`[GATE-EQ]` (`64, 700, 1.5K, 5K, 20K`)** | **5/5 `equal=True` (`first_div=None`)**, **5/5 NIAH needle `739204` recovered** | `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/phase2_4xtp2_1p1d_20261030/20261030-204736_dsv41_pd_4xtp2_20l_disjoint_p2/` |
| **Raiden Stage-3 Audit (`stage3_check.txt`)** | `STAGE3_REGISTERED_SENDS 110`, `STAGE3_COMPLETED_SENDS 110`, `STAGE3_SKIPPED_SENDS 0`, **`STAGE3_CHECK_OK 1`** | `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/phase2_4xtp2_1p1d_20261030/20261030-204736_dsv41_pd_4xtp2_20l_disjoint_p2/` |
| **Uncached Prefill (`0%` Cache Hit, `output_len = 1`)** | **`32K` (`c=8`)**: **`193,402.4 tok/s` (`1P`)** / **`184,322.3 tok/s` (`1P1D` E2E)**<br>**`128K` (`c=8`)**: **`191,583.0 tok/s` (`1P`)** / **`180,350.6 tok/s` (`1P1D` E2E)**<br>**`100K` (`c=8`)**: **`190,141.7 tok/s` (`1P`)** / **`179,012.2 tok/s` (`1P1D` E2E)** | `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/phase2_4xtp2_1p1d_20261030/20261030-074500_dsv41_pd_4xtp2_20l_phase2/` |
| **`[BENCH-PD-C64]` (`1K` in / `512` out, `c=64`)** | **E2E Wall Output Throughput**: **`1,710.88 out tok/s`** (`4,623.34 int tok/s`, **`23.13 ms` TPOT**) | `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/phase2_4xtp2_1p1d_20261030/20261030-204736_dsv41_pd_4xtp2_20l_disjoint_p2/` |
| **Empirical ISL (`50:20:12:9:9`, `95%` Cache Hit)** | **Cache Hit Rate**: **`93.08%–94.22%`** across `n = 8..128`<br>**`n = 8`**: `908.7 out tok/s`, `91.23 tok/s/stream` (`10.96 ms` ITL), **`0.60 s` med TTFT**<br>**`n = 32`**: `2,852.1 out tok/s`, `78.71 tok/s/stream` (`12.71 ms` ITL), **`0.53 s` med TTFT**<br>**`n = 64`**: `5,001.6 out tok/s`, `69.72 tok/s/stream` (`14.34 ms` ITL), **`0.59 s` med TTFT**<br>**`n = 128`**: **`6,197.7 out tok/s`**, **`116,289.2 in tok/s`** (`122,486.9 total tok/s`), `46.44 tok/s/stream` (`21.53 ms` ITL), **`0.76 s` med TTFT** (`0.89 s` med on `64K+`) | `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/phase2_4xtp2_1p1d_20261030/20261030-204736_dsv41_pd_4xtp2_20l_disjoint_p2/` |

---

## 5. Quick Troubleshooting Checklist

1. **`WARNING: other users' running pods on <ctx>`**: By default, the runners refuse to start if another user has running pods on the cluster to avoid node contention. Pass `--allow-foreign` if the cluster has enough free `2x2x1` nodes.
2. **`No gcloud access token` / CAA error**: Run `gcloud auth login --update-adc` on your workstation before launching.
3. **`LOCKED: locks/... exists`**: A previous run with the same `--suffix` is either still active or was killed with `SIGKILL`. Check the PID inside the `.lock` file; if dead, remove the `.lock` file or pass a new `--suffix`.
4. **Re-running on kept pods (`--reuse-pods`)**: `in_pod_serve.sh` automatically `pkill -9`s old `vllm`/`python3` processes and cleans `/dev/shm/pd_cache_*` before relaunching so `/dev/shm` never fills up across iterations.
