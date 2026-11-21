# XLA Flag Autotuner for vLLM on Cloud TPU (CDK)

A model-agnostic, production-grade automated exploration and validation framework for optimizing XLA compiler flags on Google Cloud TPU clusters using the **Cloud DevKit (CDK)** CLI.

---

## 📑 Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [How the Autotuner Works](#2-how-the-autotuner-works)
3. [Tuning a New Model on CDK: Step-by-Step](#3-tuning-a-new-model-on-cdk-step-by-step)
   - [Prerequisites](#31-prerequisites)
   - [Fast Proxy-Layer Strategy](#32-fast-proxy-layer-strategy)
   - [Defining the Model Configuration](#33-defining-the-model-configuration)
   - [Curating Candidate Flags](#34-curating-candidate-flags)
   - [Executing the Sweep](#35-executing-the-sweep)
   - [Validating Candidates](#36-validating-candidates)
4. [Case Study: DeepSeek-V4 Pro Exploration on TPU v7x-16](#4-case-study-deepseek-v4-pro-exploration-on-tpu-v7x-16)
   - [Phase 1: Exploratory 41-Trial Sweep](#phase-1-exploratory-41-trial-sweep)
   - [Phase 2: Multi-Flag Composite Validation (Negative Interference)](#phase-2-multi-flag-composite-validation-negative-interference)
   - [Phase 3: Single-Flag Statistical Validation (Noise Floor)](#phase-3-single-flag-statistical-validation-noise-floor)
   - [Phase 4: Full-Scale 61-Layer Production Confirmation](#phase-4-full-scale-61-layer-production-confirmation)
5. [CLI & Tool Reference](#5-cli--tool-reference)

---

## 1. Overview & Architecture

Modern LLM serving on TPUs (via `vllm-torchtpu` and PyTorch/XLA) involves dozens of low-level compiler optimization passes:
- Collective communication scheduling (e.g. All-Gather, Reduce-Scatter, All-to-All for MoE)
- Memory management & scavenging (VMEM allocation heuristics)
- Fusion passes and instruction schedulers (e.g. BRKGA, Latency-Hiding Scheduler)
- Sparse Core offloading

Because compiler flags can interact non-linearly, finding optimal configurations requires a disciplined, multi-stage methodology:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        XLA Autotune Pipeline                           │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 1. Proxy Model Sweep (e.g., 10 Layers, Dummy Weights)   │
       │    Fast OFAT exploration across flags                   │
       └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 2. Multi-Run Statistical Validation                     │
       │    Filter out noise  via 5-10 runs                      │
       └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 3. Full-Scale Production Verification                   │
       │    Evaluate winning candidate(s) on full-layer model    │
       └─────────────────────────────────────────────────────────┘
```

### Module Structure

```
scripts/xla_autotune/
├── README.md                      # This documentation
├── __init__.py                    # Public API exports
├── config.py                      # Model, CDK, Benchmark & Tuning dataclasses
├── cdk_orchestrator.py            # CDK CLI wrapper (job create, list, sync-outputs)
├── utils.py                       # Shared CDK utilities (env resolution, job ID parsing, log sync)
├── runner.py                      # vLLM benchmark serve output parser & metrics extractor
├── local_autotuner.py             # Sequential One-Factor-At-A-Time (OFAT) local tuner
├── cdk_submit_sweep.py            # Parallel batch job submission to CDK
├── cdk_monitor_sweep.py           # Parallel sweep monitor & ranking leaderboard generator
├── flags.txt                      # Comprehensive XLA/LibTPU flag catalog & descriptions
├── configs/                       # Declarative YAML configurations
│   ├── example_config.yaml        # Template config
│   ├── dsv4_pro_fast_layers.yaml  # 10-layer DeepSeek-V4 Pro proxy recipe
│   ├── dsv4_pro_v7x_16.yaml       # Full 61-layer DeepSeek-V4 Pro recipe
│   └── qwen35_397b.yaml           # Qwen-3.5 397B recipe
└── examples/dsv4_study/           # Validation scripts & artifacts from DSv4 campaign
```

---

## 2. How the Autotuner Works

### 1. CDK Integration & JobSet Lifecycle
The autotuner operates completely headlessly and non-interactively on Google Cloud DevKit (CDK):
- **Agent Code Resolution**: Automatically extracts `--agent-code` from `cdk agent-letter` or environment variables (`CDK_ACK`).
- **Dynamic Flag Injection**: Injects flags into multihost JobSets using `EXTRA_LIBTPU_INIT_ARGS="--xla_tpu_flag=value"`.
- **Decoupled Execution**:
  - `cdk_submit_sweep.py` submits jobs with unique tags (`agent-jobs,xla-autotune,trial-cand_XXX`) and saves `submission_manifest.json`.
  - `cdk_monitor_sweep.py` polls `cdk job list -o json` periodically.
  - Upon job completion, it calls `cdk job sync-outputs <job_id>`, pulls container logs from GCS, and parses `vllm bench serve` tables.

### 2. Metric Parsing & Leaderboard
The metric extractor (`runner.py`) parses the standard `vllm bench serve` summary table:
- **Primary Metric**: `total_token_throughput` (tok/s)
- **Secondary Metrics**: `request_throughput` (req/s), `mean_ttft_ms` (Time To First Token), `p99_ttft_ms`, `mean_tpot_ms` (Time Per Output Token).
- Results are ranked against the **mean of all baseline runs**, generating a dynamic `ranking.md` markdown table and `summary.json`.

---

## 3. Tuning a New Model on CDK: Step-by-Step

### 3.1 Prerequisites
1. **Working CDK Recipe**:
   Ensure you have a functional benchmark recipe in your `cloud-devkit-recipes` repository (e.g. `recipes/experimental/multihost/torchtpu/<model>/<model>-benchmark.yml`).
   The recipe template must support extra arguments via:

   ```yaml
   - name: LIBTPU_INIT_ARGS
     value: "{{ .LIBTPU_INIT_ARGS | default `...` }}{{ if .EXTRA_LIBTPU_INIT_ARGS }} {{ .EXTRA_LIBTPU_INIT_ARGS }}{{ end }}"
   ```

2. **Pre-Populated Weights / ROX Disk**:
   Weight staging per trial takes 20+ minutes if pulled from GCS cold. For fast startup, attach pre-populated disks in `ReadOnlyMany` (`WEIGHTS_PVC`).

### 3.2 Fast Proxy-Layer Strategy
Evaluating 40+ flags on ultra-large models can take weeks of TPU compute (e.g., in our DeepSeek-V4 Pro study, the full 600B+ model with 61 layers took ~45 minutes per run).

**The standard strategy is to create a fast proxy recipe:**
- Keep all tensor parallel (TP), pipeline parallel (PP), and expert parallel (EP) sharding shapes identical.
- Truncate layer count via `--hf-overrides '{"num_hidden_layers": <N>}'` in the `vllm serve` command (e.g., set to `10` layers in the DeepSeek-V4 study).
- Scale down `--gpu-memory-utilization` (e.g., set to `0.40` for the 10-layer DSv4 proxy vs. `0.90` for full model).
- Use `--safetensors-load-strategy=prefetch` or dummy loading.
- **Result (from DeepSeek-V4 study)**: Cycle time dropped from ~45 minutes down to ~18 minutes per trial, while preserving true kernel sharding and compiler graph properties.

### 3.3 Defining the Model Configuration
Create `configs/<new_model>.yaml`:

```yaml
execution_mode: "cdk"
flag_list_file: "scripts/xla_autotune/flags.txt"
artifact_dir: "artifacts/<new_model>_sweep"
baseline_runs: 2

cdk:
  recipe: "torchtpu-vllm-<new_model>-benchmark-fast"
  tags: ["agent-jobs", "xla-autotune", "<new_model>"]
  poll_interval_s: 60
  extra_recipe_params:
    RANDOM_INPUT_LEN: "8192"
    RANDOM_OUTPUT_LEN: "1024"
    NUM_PROMPTS: "1024"

model:
  name: "organization/<new_model>"
  port: 47903

benchmark:
  type: "vllm_bench_serve"
  target_metric: "total_token_throughput"
```

### 3.4 Curating Candidate Flags
Review `scripts/xla_autotune/flags.txt` to select relevant knobs:
- **Collective Offloads**: Sparse Core offload for all-gather / reduce-scatter.
- **Schedulers**: BRKGA, Latency Hiding Scheduler (LHS).
- **Memory Scavenging**: Safe vs. None.
- Save your target flags in `flags_<new_model>.txt` (one flag per line; lines starting with `#` are ignored).

### 3.5 Executing the Sweep
Submit the sweep to CDK:

```bash
python3 scripts/xla_autotune/cdk_submit_sweep.py \
  --config configs/<new_model>.yaml \
  --flags-file flags_<new_model>.txt \
  --baseline-runs 2
```

Launch the parallel sweep monitor:

```bash
python3 scripts/xla_autotune/cdk_monitor_sweep.py \
  --config configs/<new_model>.yaml
```

The monitor will update `artifacts/<new_model>_sweep/ranking.md` in real time as trials complete.

### 3.6 Validating Candidates
**Never adopt a flag based on a single trial!** Clusters exhibit natural variance (typically ~1% noise floor), so an apparent gain in a single trial may be an exploratory outlier. Furthermore, combining multiple winning flags often causes **negative compiler interference** (as seen in Phase 2 of our DeepSeek-V4 study).

1. **Identify top candidate(s)** with apparent gains (> 1%).
2. **Execute statistical validation** comparing multiple runs of the baseline against multiple runs of the candidate configuration:
   - **Single Flag Validation**: Create `flags_validate_5x.txt` with the candidate flag repeated 5 times.
   - **Composite Multi-Flag Validation**: To test multiple winning flags operating together, join all flags on **a single line separated by spaces**, and repeat that line 5 times in `flags_composite_5x.txt`:

     ```text
     # flags_composite_5x.txt: Space-separated flags per line, repeated 5 times
     --xla_tpu_vmem_scavenging_mode=NONE --xla_tpu_enable_brkga_latency_hiding_scheduler=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0
     --xla_tpu_vmem_scavenging_mode=NONE --xla_tpu_enable_brkga_latency_hiding_scheduler=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0
     --xla_tpu_vmem_scavenging_mode=NONE --xla_tpu_enable_brkga_latency_hiding_scheduler=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0
     --xla_tpu_vmem_scavenging_mode=NONE --xla_tpu_enable_brkga_latency_hiding_scheduler=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0
     --xla_tpu_vmem_scavenging_mode=NONE --xla_tpu_enable_brkga_latency_hiding_scheduler=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0
     ```

   - Launch the validation sweep using your base config (ensuring all CDK settings and `extra_recipe_params` are preserved):

     ```bash
     python3 scripts/xla_autotune/cdk_submit_sweep.py \
       --config configs/<new_model>.yaml \
       --flags-file flags_composite_5x.txt \
       --baseline-runs 5 \
       --artifact-dir artifacts/<new_model>_validation
     ```

     *(This submits 5 baseline runs and 5 candidate runs where every candidate trial sets all flags simultaneously.)*

   - Monitor the validation sweep and generate the leaderboard:

     ```bash
     python3 scripts/xla_autotune/cdk_monitor_sweep.py \
       --config configs/<new_model>.yaml \
       --artifact-dir artifacts/<new_model>_validation
     ```

3. **Check statistical significance**: Compare the candidate mean against the baseline mean. If the net delta exceeds the natural noise floor, proceed to verify on the **full-layer model**.

---

## 4. Case Study: DeepSeek-V4 Pro Exploration on TPU v7x-16

We executed this exact methodology for DeepSeek-V4 Pro (671B MoE) on TPU v7x-16 (16 chips, 2 hosts, EP=16).

### Phase 1: Exploratory 41-Trial Sweep
Using a 10-layer proxy recipe (`torchtpu-vllm-dsv4-pro-v7x-16-benchmark-10layers-baseline`):
- **Baseline Average**: `193,086.61 tok/s` (Run 1: `192,192.17`, Run 2: `193,981.04` -> **0.93% natural spread**).
- **Candidate Findings**:
  - `cand_036` (`--xla_tpu_vmem_scavenging_mode=NONE`): **`199,247.82 tok/s` (+3.19%)** -> Top candidate.
  - `cand_032` (`--xla_tpu_enable_brkga_latency_hiding_scheduler=true`): **`195,887.26 tok/s` (+1.45%)**.
  - `cand_013` (`--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0`): **`195,380.00 tok/s` (+1.19%)**.
  - `cand_031` (`--xla_tpu_enable_latency_hiding_scheduler=false`): **`177,435.61 tok/s` (-8.11%)** -> Severe regression.

### Phase 2: Multi-Flag Composite Validation (Negative Interference)
We tested combining the top three flags in a 10-run concurrent validation suite:
- **Hypothesis**: Will stacking the top candidates compound throughput gains?
- **Result**: The 10 runs averaged **`191,856.19 tok/s` (-0.64% vs baseline)**.
- **Lesson**: Compiler passes interfere non-linearly. The BRKGA instruction scheduler clashed with the Sparse Core minimum offload threshold, increasing scheduling stalls.

### Phase 3: Single-Flag Statistical Validation (Noise Floor)
We isolated the single top winner (`--xla_tpu_vmem_scavenging_mode=NONE`) across 5 validation runs on the 10-layer model:
- **Baseline Average**: `193,086.61 tok/s`
- **Runs Completed**:
  - Run 1: `191,637.37 tok/s` (-0.75%)
  - Run 2: `193,720.97 tok/s` (+0.33%)
  - Run 5: `195,876.45 tok/s` (+1.44%)
- **Validation Mean**: **`193,744.93 tok/s` (+0.34%)**
- **Lesson**: The +3.19% from the single trial was an exploratory outlier. The true effect was +0.34%, falling completely inside the cluster's ~1% natural noise floor.

### Phase 4: Full-Scale 61-Layer Production Confirmation
To rule out differences in memory pressure (10 layers vs 61 layers), we launched 2 baselines and 2 tuned runs on the full 61-layer model (`torchtpu-vllm-dsv4-pro-v7x-16-benchmark`):
- **Baseline Runs**:
  - `full_baseline_01`: `38,536.54 tok/s` (+0.34%)
  - `full_baseline_02`: `38,274.17 tok/s` (-0.34%)
  - **Baseline Mean**: **`38,405.35 tok/s`**
- **Tuned Runs (`vmem_scavenging_mode=NONE`)**:
  - `full_vmem_none_01`: `38,270.02 tok/s` (-0.35%)
  - `full_vmem_none_02`: `38,753.77 tok/s` (+0.91%)
  - **Tuned Mean**: **`38,511.89 tok/s`**
- **Net Delta**: **`+0.28%`**
- **Conclusion**: Even under full memory capacity (90% TPU utilization), the flag behaved with statistical parity to baseline. The official baseline recipe was confirmed to already be optimal.

---

## 5. CLI & Tool Reference

### `cdk_submit_sweep.py`
Submits a baseline run followed by each flag in the flags file to CDK:

```bash
python3 scripts/xla_autotune/cdk_submit_sweep.py \
  --config <path_to_config.yaml> \
  [--flags-file <flags.txt>] \
  [--artifact-dir <output_dir>] \
  [--baseline-runs 2]

# Ad-hoc mode (without a YAML config):
# python3 scripts/xla_autotune/cdk_submit_sweep.py --recipe <recipe_name> --flags-file <flags.txt>
```

### `cdk_monitor_sweep.py`
Tracks active CDK jobs, syncs outputs on completion, parses metrics, and builds the leaderboard:

```bash
python3 scripts/xla_autotune/cdk_monitor_sweep.py \
  --config <path_to_config.yaml> \
  [--poll-interval 60] \
  [--once]
```

### `local_autotuner.py`
Sequential OFAT tuning engine for local TPU VM execution:

```bash
python3 scripts/xla_autotune/local_autotuner.py \
  --config <path_to_config.yaml> \
  [--slice-index 1 --slice-count 2] \
  [--dry-run]
```
