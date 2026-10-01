# TPU Inference & Structured Sparsity Benchmarking Suite

An automated, modular performance and accuracy evaluation framework for Large Language Models running on Google Cloud TPU v6e / v5e using vLLM and JAX.

---

## 1. System Architecture

The architecture adheres strictly to **SOLID design principles**, separating parameter grid generation, dynamic model resolution/splicing, polymorphic runner execution, and metric persistence.

```
                                  ┌────────────────────────────────┐
                                  │   run_sweep_orchestrator.py    │
                                  │   (Cartesian Grid Sequencer)   │
                                  └───────────────┬────────────────┘
                                                  │
                  ┌───────────────────────────────┴───────────────────────────────┐
                  ▼                                                               ▼
  ┌────────────────────────────────┐                             ┌────────────────────────────────┐
  │      ResolverRegistry          │                             │         RunnerRegistry         │
  │────────────────────────────────│                             │────────────────────────────────│
  │ • SparseGptResolver            │                             │ • TpuEvalRunner (lm_eval)      │
  │ • NaiveMagnitudeResolver       │                             │ • TpuProfileRunner (throughput)│
  │ • WeightScalerResolver         │                             └────────────────┬───────────────┘
  └───────────────┬────────────────┘                                              │
                  │ (resolves checkpoint path)                                    │ (executes benchmark)
                  ▼                                                               ▼
  ┌────────────────────────────────┐                             ┌────────────────────────────────┐
  │   Model Splicer & Pruners      │                             │     vLLM on Cloud TPU v6e      │
  │────────────────────────────────│                             │────────────────────────────────│
  │ • Cross-Shard ModelSplicer     │                             │ • 4-way Tensor Parallelism     │
  │ • SparseGPT Hessian Pruner     │                             │ • Native Pallas FlashAttention │
  │ • Naive N:M Magnitude Pruner   │                             │ • bfloat16 / FP8 KV Cache      │
  │ • Checkpoint Integrity Shards  │                             └────────────────┬───────────────┘
  └────────────────────────────────┘                                              │
                                                                                  ▼
                                                                 ┌────────────────────────────────┐
                                                                 │      JsonResultRecorder        │
                                                                 │  • results/<id>/sweep.json     │
                                                                 │  • results/<id>/results.csv    │
                                                                 │  • GCS sync (incoming/...)     │
                                                                 └────────────────────────────────┘
```

---

## 2. Directory Structure

```text
experiments/
├── configs/                                                     # YAML configurations for sweeps and benchmark runs
│   ├── 0_baseline_mmlu_pro_eval.yaml                            # Baseline MMLU-Pro accuracy evaluation
│   ├── 0_baseline_mmlu_pro_fast.yaml                            # Fast smoke-test baseline configuration
│   ├── 0_baseline_mmlu_pro_full.yaml                            # Full-dataset baseline configuration
│   ├── 1_prefill_sweep.yaml                                     # Input sequence length scaling sweeps
│   ├── 1_prefill_sweep_mxutracing.yaml                          # Prefill sweep with MXU profile tracing
│   ├── 2_decode_batch_sweep.yaml                                # Batch size scaling decode sweeps
│   ├── 3_decode_context_sweep.yaml                              # Context length decode sweeps
│   ├── 4_sparsity_layer_sweep.yaml                              # Layer sparsity progression sweeps
│   ├── 5_sparsity_mmlu_pro_eval.yaml                            # Full MMLU-Pro accuracy vs sparsity sweeps
│   ├── 5_sparsity_mmlu_pro_eval_fast.yaml                       # Fast MMLU-Pro accuracy vs sparsity sweeps
│   ├── 6_sparsity_front_to_back_2of4_mmlu_pro_eval_fast.yaml    # 2:4 SparseGPT front-to-back layer sweep
│   ├── 7_sparsity_front_to_back_1of4_mmlu_pro_eval_fast.yaml    # 1:4 SparseGPT front-to-back layer sweep
│   ├── 8_sparsity_front_to_back_2of4_256c4_mmlu_pro_eval_fast.yaml # 2:4 SparseGPT with 256 C4 calibration samples
│   ├── 9_sparsity_front_to_back_1of4_256c4_mmlu_pro_eval_fast.yaml # 1:4 SparseGPT with 256 C4 calibration samples
│   ├── 10_sparsity_front_to_back_2of4_magnitude_mmlu_pro_eval_fast.yaml # 2:4 data-free magnitude pruning sweep
│   ├── 13_gemma4_2of4_canonical_weight_scaling_eval_fast.yaml   # 2:4 SparseGPT constant weight multiplier sweeps
│   ├── 14_gemma4_1of4_canonical_weight_scaling_eval_fast.yaml   # 1:4 SparseGPT constant weight multiplier sweeps
│   ├── 15_gemma4_2of4_256c4_canonical_weight_scaling_eval_fast.yaml # 2:4 (256 c4) weight scaling sweeps
│   ├── 16_gemma4_1of4_256c4_canonical_weight_scaling_eval_fast.yaml # 1:4 (256 c4) weight scaling sweeps
│   └── test_advanced_trace.yaml                                 # JAX/XLA profiler trace test configuration
├── core/                                                        # Core interfaces, models, builders, and recorders
│   ├── __init__.py                                              # Package initialization
│   ├── env.py                                                   # TPU environment setup, GCS bucket constants
│   ├── json_recorder.py                                         # Canonical JSON recorder (sweep.json) with GCS sync
│   ├── csv_recorder.py                                          # Dynamic schema-evolving CSV logger (legacy fallback)
│   ├── engine_builder.py                                        # vLLM kwargs builder and TPU environment setup
│   ├── interfaces.py                                            # IRunner, IModelResolver, IResultRecorder, ISparsityEngine
│   └── models.py                                                # ExecutionContext, BenchmarkResult, ModelArchitectureSpec
├── resolvers/                                                   # Dynamic checkpoint resolvers and layer transformers
│   ├── __init__.py                                              # Package initialization
│   ├── base_resolver.py                                         # Abstract base class for all checkpoint resolvers
│   ├── base_sparsity_resolver.py                                # Unified base class with owner-marked cache reaping
│   ├── dataset_providers.py                                     # Calibration dataset providers (C4, WikiText, custom)
│   ├── model_splicer.py                                         # Cross-shard layer splicer with disk shard integrity check
│   ├── naive_magnitude_resolver.py                              # Data-free magnitude pruning resolver
│   ├── registry.py                                              # Central resolver factory
│   ├── sparsegpt_resolver.py                                    # SparseGPT Hessian-based error-compensated resolver
│   └── weight_scaler_resolver.py                                # Constant scalar multiplier (c * W) resolver
├── runners/                                                     # Benchmark and evaluation execution engines
│   ├── __init__.py                                              # Package initialization
│   ├── base_runner.py                                           # Abstract base runner interface
│   ├── eval_runner.py                                           # Quality & accuracy evaluation runner (MMLU-Pro via lm_eval)
│   ├── profile_runner.py                                        # Throughput & latency profiling runner
│   └── registry.py                                              # Central runner factory
├── sparsity/                                                    # Structured sparsity algorithms
│   ├── LICENSE                                                  # Sparsity library license
│   ├── README.md                                                # Sparsity package documentation
│   ├── __init__.py                                              # Package initialization
│   ├── datautils.py                                             # Calibration sequence tokenization utilities
│   ├── gemma4.py                                                # Gemma 4 layer-0 capture and the CPU calibrator
│   ├── llama.py                                                 # LLaMA architecture layer hooks
│   ├── lrc.py                                                   # Low-Rank Compensated (LRC) alternating minimization
│   ├── modelutils.py                                            # PyTorch module traversal & model structure helpers
│   ├── naive_magnitude_pruner.py                                # Vectorized N:M magnitude pruner (dim=-1)
│   ├── quant.py                                                 # Quantization helper routines
│   ├── sparsegpt.py                                             # PyTorch reference SparseGPT implementation
│   ├── tpu_gemma4.py                                            # TPU calibrator: Gemma 4 decoder layer in JAX
│   └── tpu_sparsegpt.py                                         # SparseGPT OBS solver in JAX
├── tests/                                                       # Comprehensive unit, integration & TPU test suite
│   ├── __init__.py                                              # Test package initialization
│   ├── fixtures.py                                              # Synthetic multi-shard fixtures & dynamic path discovery
│   ├── run_coverage.py                                          # Coverage report generator with sysmon engine
│   ├── test_core.py                                             # Domain models, JSON/CSV recorders, engine builder
│   ├── test_gemma4_sparsity.py                                  # Gemma 4 calibration hooks & loader
│   ├── test_integration.py                                      # End-to-end simulated sweep pipeline
│   ├── test_jax_options.py                                      # JAX TPU execution options parser tests
│   ├── test_orchestrator.py                                     # Cartesian grid sequencer & resume logic
│   ├── test_real_weights_splicing.py                            # Real Gemma 4 31B-IT weight extraction & splicing
│   ├── test_resolvers.py                                        # Layer range strategies, ModelSplicer, integrity checks
│   ├── test_runners.py                                          # TpuProfileRunner, TpuEvalRunner lifecycle
│   ├── test_sparsity_algorithms.py                              # 2:4 / 1:4 magnitude math, SparseGPT Hessian, quant
│   ├── test_tpu_hardware.py                                     # Physical TPU v6e chip detection & on-chip pruning
│   └── test_tpu_sparsegpt_solver.py                             # TPU vs CPU SparseGPT solve on one matrix (needs a TPU)
├── .coveragerc                                                  # Coverage scoping configuration
├── pytest.ini                                                   # Pytest discovery configuration
├── rescore.py                                                   # Re-score stored lm_eval samples with new filters
├── run_sweep_orchestrator.py                                    # Sweep orchestrator CLI (entry point)
└── README.md                                                    # This documentation file
```

---

## 3. How to Sparsify Models Directly (Without `run_sweep_orchestrator`)

You can create 2:4 or 1:4 structured sparse checkpoints standalone without running a parameter sweep.

### A. Standalone SparseGPT Pruning (Hessian-Compensated)
Runs second-order error-compensated pruning using calibration data (e.g. C4 dataset):

```python
from sparsity import prune_gemma4

prune_gemma4(
    model_path="/mnt/pd/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475",
    dataset="c4",
    nsamples=256,              # Number of calibration sequences
    prunen=2,                  # N in N:M (2 for 2:4 50% sparse, 1 for 1:4 75% sparse)
    prunem=4,                  # M in N:M
    minlayer=0,                # Start layer index
    maxlayer=60,               # End layer index
    save_path="/mnt/pd/sparse_checkpoints/my_gemma4_2of4_256c4_full"
)
```

Pass `calibrator="tpu"` (or `resolver_config: {use_tpu_sparsegpt: true}` in a
sweep config) to run the same pipeline with the JAX calibrator in
`sparsity/tpu_gemma4.py`: ~2.6 s per Gemma-4 31B layer instead of ~370 s, same
checkpoint format. Because SparseGPT's top-k mask choice is discontinuous the
two calibrators are separate lineages (~97% mask agreement, indistinguishable
MMLU-Pro); cached checkpoints carry a `_tpu` tag so prefixes are only ever
spliced within a lineage, and `SparseGptResolver` runs the TPU calibrator in a
child process (`python -m sparsity.tpu_gemma4 --json ...`) because libtpu
admits one process per TPU and vLLM needs the chips right afterwards.

### B. Standalone Naive Magnitude Pruning (Data-Free)
Applies top-K magnitude pruning along the contracting dimension (dim=-1) across all linear projection weights without calibration:

```bash
python3 sparsity/naive_magnitude_pruner.py \
  --model /mnt/pd/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --save /mnt/pd/sparse_checkpoints/my_gemma4_2of4_magnitude_full \
  --prunen 2 \
  --prunem 4 \
  --minlayer 0 \
  --maxlayer 60
```

### C. Standalone Cross-Shard Layer Splicing
Splices specific layers from a 100% sparse reference model into a dense base model in seconds using `safe_open` zero-recompute streaming:

```python
from resolvers.model_splicer import ModelSplicer

splicer = ModelSplicer()
splicer.splice_checkpoint(
    dense_model_dir="/mnt/pd/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475",
    fully_sparse_model_dir="/mnt/pd/sparse_checkpoints/gemma4_31b_2of4_256c4_full",
    sparse_layer_indices={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},  # Splice first 10 prefix layers
    output_dir="/mnt/pd/sparse_checkpoints/gemma4_spliced_10_layers"
)
```

### D. Low-Rank Compensated (LRC) 2:4 Pruning
LRC (`sparsity/lrc.py`) decomposes each pruned linear layer as `W ≈ S + U @ V`, where `S` is 2:4 structured-sparse and `U @ V` is a rank-`k` residual adapter solved via `T` iterations of alternating minimization in whitened activation space (`||(W - S - U @ V) @ L||_F^2`, where `H = L @ L^T`).

To enable LRC in a sweep YAML config, use `resolvers.lrc_resolver` (`LrcResolver`) and set `lrc_rank` and `lrc_iters` inside `model_resolver.config`:

```yaml
model_resolver:
  module: resolvers.lrc_resolver
  function: resolve_model
  config:
    base_model: google/gemma-4-12B-it
    reference_sparse_model: null
    strategy: first_to_last
    prunen: 2
    prunem: 4
    nsamples: 64
    lrc_rank: 128
    lrc_iters: 3

sweep_matrix:
  layers:
    - [0, 1, 2, 3, 4, 5, 6, 7]
```

---

## 4. How to Run Automated Sweeps

### A. Local Execution on Cloud TPU v6e

Launch under GNU `screen` to survive SSH disconnects:

```bash
cd /mnt/pd/vllm-torchtpu/experiments
source /mnt/pd/torchtpu_env/bin/activate

screen -d -m -S my_sweep bash -c 'exec python -u run_sweep_orchestrator.py \
  --config configs/gemma4/eval_front_to_back_12b_mmlu_pro.yaml \
  -m "My experiment description" >> sweep.log 2>&1'
```

### B. Monitoring, Results & GCS Sync

Results stream canonically into `sweep.json`:

```bash
cat results/<sweep_id>/sweep.json
```

On completion, `JsonResultRecorder` automatically synchronizes results to Google Cloud Storage under
`gs://cloud-tpu-inference-eval-staging/incoming/<user>/<sweep_id>/`, where `<user>` is resolved from
the recorder owner (defaults to `$USER`, see `core/json_recorder.py`). Pass `--no-upload-gcs` to disable the upload.

### C. Resuming an Interrupted Sweep

If a sweep was stopped, preempted, or ran out of disk space, use `--resume` with the existing experiment ID. The orchestrator verifies that `--config` matches `metadata.yaml`, reads already-completed coordinates from `sweep.json`, and skips them:

```bash
python -u run_sweep_orchestrator.py \
  --config configs/gemma4/eval_front_to_back_12b_mmlu_pro.yaml \
  --experiment-id 20260924_214452 \
  --resume \
  -m "Resume after preemption"
```

---

## 5. Running Unit, Integration & TPU Hardware Tests

### A. Run All Unit & Integration Tests via Pytest

```bash
cd /mnt/pd/vllm-torchtpu/experiments
source /mnt/pd/torchtpu_env/bin/activate
VLLM_MOCK_TPU=1 pytest tests/ -q
```

Hardware tests (`hardware_smoke` marker) are deselected by default in `pytest.ini`; run them explicitly on a TPU host with `pytest tests/ -m hardware_smoke`.

### B. Run Code Coverage Analysis

```bash
COVERAGE_CORE=sysmon python3 tests/run_coverage.py
```

*An interactive HTML report is compiled into `experiments/htmlcov/index.html`.*
