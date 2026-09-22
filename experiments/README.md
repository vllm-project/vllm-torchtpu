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
  │ • TPU SparseGPT Hessian Kernel │                             │ • Native Pallas FlashAttention │
  │ • Naive N:M Magnitude Pruner   │                             │ • bfloat16 / FP8 KV Cache      │
  └────────────────────────────────┘                             └────────────────┬───────────────┘
                                                                                  │
                                                                                  ▼
                                                                 ┌────────────────────────────────┐
                                                                 │       CsvResultRecorder        │
                                                                 │   (results/<timestamp>/results.csv)
                                                                 └────────────────────────────────┘
```

---

## 2. Directory Structure

*(Excludes git-ignored paths such as `results/`, `plots/`, `plot_scripts/`, `scripts/`, `scratch/`, `htmlcov/`, `.pytest_cache/`, `__pycache__/`, `*.csv`, `*.log`, and compiled artifacts)*

```
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
│   ├── csv_recorder.py                                          # Dynamic schema-evolving CSV logger
│   ├── engine_builder.py                                        # vLLM kwargs builder and TPU environment setup
│   ├── interfaces.py                                            # IRunner, IModelResolver, IResultRecorder, ISparsityEngine
│   └── models.py                                                # ExecutionContext, BenchmarkResult, ModelArchitectureSpec
├── resolvers/                                                   # Dynamic checkpoint resolvers and layer transformers
│   ├── __init__.py                                              # Package initialization
│   ├── base_resolver.py                                         # Abstract base class for all checkpoint resolvers
│   ├── base_sparsity_resolver.py                                # Unified base class for sparsity resolvers
│   ├── dataset_providers.py                                     # Calibration dataset providers (C4, WikiText, custom)
│   ├── model_splicer.py                                         # Cross-shard zero-recompute safetensors layer splicer
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
├── sparsity/                                                    # Structured sparsity algorithms and TPU kernels
│   ├── LICENSE                                                  # Sparsity library license
│   ├── README.md                                                # Sparsity package documentation
│   ├── __init__.py                                              # Package initialization
│   ├── datautils.py                                             # Calibration sequence tokenization utilities
│   ├── gemma4.py                                                # Gemma 4 layer capture and Hessian orchestrator
│   ├── llama.py                                                 # LLaMA architecture layer hooks
│   ├── modelutils.py                                            # PyTorch module traversal & model structure helpers
│   ├── naive_magnitude_pruner.py                                # Vectorized N:M magnitude pruner (dim=-1)
│   ├── quant.py                                                 # Quantization helper routines
│   └── sparsegpt.py                                             # PyTorch reference SparseGPT implementation
├── tests/                                                       # Comprehensive unit, integration & TPU test suite
│   ├── __init__.py                                              # Test package initialization
│   ├── fixtures.py                                              # Synthetic multi-shard fixtures & dynamic path discovery
│   ├── run_coverage.py                                          # Coverage report generator with sysmon engine
│   ├── test_core.py                                             # Domain models, dynamic CSV recorder, engine builder
│   ├── test_gemma4_sparsity.py                                  # Gemma 4 calibration hooks & loader
│   ├── test_integration.py                                      # End-to-end simulated sweep pipeline
│   ├── test_jax_options.py                                      # JAX TPU execution options parser tests
│   ├── test_orchestrator.py                                     # Cartesian grid sequencer & resume logic
│   ├── test_real_weights_splicing.py                            # Real Gemma 4 31B-IT weight extraction & splicing
│   ├── test_resolvers.py                                        # Layer range strategies, ModelSplicer, registries
│   ├── test_runners.py                                          # TpuProfileRunner, TpuEvalRunner lifecycle
│   ├── test_sparsity_algorithms.py                              # 2:4 / 1:4 magnitude math, SparseGPT Hessian, quant
│   └── test_tpu_hardware.py                                     # Physical TPU v6e chip detection & on-chip pruning
├── .coveragerc                                                  # Coverage scoping configuration (>85% coverage)
├── pytest.ini                                                   # Pytest discovery configuration
├── run_sweep_orchestrator.py                                    # Master sweep orchestrator CLI
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
    prunen=2,                  # N in N:M (2 for 2:4, 1 for 1:4)
    prunem=4,                  # M in N:M
    minlayer=0,                # Start layer index
    maxlayer=60,               # End layer index
    save_path="/mnt/pd/sparse_checkpoints/my_gemma4_2of4_256c4_full"
)
```

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
    sparse_layer_indices={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},  # Splice first 10 layers
    output_dir="/mnt/pd/sparse_checkpoints/gemma4_spliced_10_layers"
)
```

### D. Standalone Constant Weight Scaling ($W \leftarrow c \cdot W$)
Scales only the 2D linear projection weights of a sparse model by a constant multiplier $c$ (e.g. $c = 1.10$):

```python
from resolvers.weight_scaler_resolver import WeightScalerResolver

resolver = WeightScalerResolver()
resolver._scale_checkpoint(
    base_checkpoint="/mnt/pd/sparse_checkpoints/gemma4_31b_2of4_256c4_full",
    output_dir="/mnt/pd/sparse_checkpoints/gemma4_2of4_scaled_c1p100",
    scale_factor=1.10
)
```

---

## 4. How to Run Automated Sweeps

To execute a multi-coordinate parameter sweep (e.g. evaluating 0, 5, 10, 20, 30, 40, 50, 60 sparse layers):

```bash
cd /mnt/pd/vllm-torchtpu/experiments
source /mnt/pd/torchtpu_env/bin/activate
export PYTHONPATH="/mnt/pd/vllm:/mnt/pd/vllm-torchtpu/src:${PYTHONPATH:-}"

python run_sweep_orchestrator.py --config configs/0_baseline_mmlu_pro_fast.yaml
```

**View Live Results:**

```bash
cat results/*/results.csv
```

**Resume an Interrupted Sweep:**
If a sweep was stopped or preempted, pass the previous experiment ID to resume without re-running finished coordinates:

```bash
python run_sweep_orchestrator.py --config configs/0_baseline_mmlu_pro_fast.yaml --experiment-id 20260826_190146
```

---

## 5. Running Unit, Integration & TPU Hardware Tests

The repository includes a comprehensive, modular test suite adhering to **SOLID design principles** and the **Arrange-Act-Assert (AAA)** pattern with **>85% line coverage**.

### A. Run All Unit & Hardware Tests via Pytest
Executes all 50 unit, integration, and TPU hardware tests:

```bash
cd /mnt/pd/vllm-torchtpu/experiments
source /mnt/pd/torchtpu_env/bin/activate
pytest tests/ -v
```

### B. Run Code Coverage Analysis
Executes tests under Python 3.12 PEP 669 event monitoring (`COVERAGE_CORE=sysmon`) and prints a full terminal breakdown:

```bash
cd /mnt/pd/vllm-torchtpu/experiments
source /mnt/pd/torchtpu_env/bin/activate
COVERAGE_CORE=sysmon python3 tests/run_coverage.py
```

*An interactive HTML report is automatically compiled into `experiments/htmlcov/index.html`.*

### C. Test Suite Architecture & Dual Verification
1. **Hermetic & Stub-Based (CPU / Any Machine)**:
   - High-speed, isolated unit tests using synthetic multi-shard safetensors models ([`tests/fixtures.py`](tests/fixtures.py)).
   - Runs anywhere in sub-second time without requiring GPU/TPU hardware or downloading large models.
2. **Real Weights Verification**:
   - [`tests/test_real_weights_splicing.py`](tests/test_real_weights_splicing.py) extracts Layer 0 and Layer 1 projection matrices directly from the real Gemma 4 31B-IT checkpoint shards, testing real-weight 2:4 structured pruning, layer splicing, and scalar weight scaling.
3. **Physical Cloud TPU v6e Hardware Execution**:
   - [`tests/test_tpu_hardware.py`](tests/test_tpu_hardware.py) verifies PJRT runtime initialization, physical tensor allocation on Cloud TPU v6e chips, and on-chip vectorized 2:4 structured pruning kernels.

---

## 6. Production vLLM Benchmarking Suite

For standardized production benchmarking and baseline regression tracking:

```bash
cd /mnt/pd/vllm-torchtpu
source /mnt/pd/torchtpu_env/bin/activate
export SKIP_PERF_BENCH=1

LM_EVAL_TASKS='mmlu_pro' LM_EVAL_LIMIT=3 bash ./scripts/vllm/benchmarking/run_eval_configs.sh gemma-4-31b-it-tp4
```
