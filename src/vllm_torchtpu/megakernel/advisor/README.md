# TPU Megakernel Advisor

Offline CPU specification package for TPU megakernel analysis. Resolves sourced
hardware parameters (`resolve_chip_spec`), HuggingFace model configurations and
locked model presets (`resolve_model_profile`, `model_profile_from_hf_config`),
6-axis deployment layouts (`TP x DP x EP x PP x PCP x DCP`), and measurement
records (`load_measurements`) without requiring TPU hardware or serving imports.

## Supported Models

Three locked basis models are shipped in `data/presets/models.json`, each verified
against a pinned HuggingFace `config.json` snapshot and Hub parameter manifest:

- `qwen3.5-397b` (`Qwen/Qwen3.5-397B-A17B-FP8`)
- `kimi-k3` (`moonshotai/Kimi-K3`)
- `glm-5.2` (`zai-org/GLM-5.2-FP8`)

Any HuggingFace config whose architecture does not match a validated preset raises
`UnsupportedArchitectureError`.

## CLI Usage

Run single-point estimates, multi-regime sweeps, or layer-group searches from CPU:

```bash
python -m vllm_torchtpu.megakernel.advisor estimate \
  --model qwen3.5-397b --chip tpu7x --phase decode --batch 16 --tp 8 --ep 8 \
  --explain --unresolved

python -m vllm_torchtpu.megakernel.advisor --model qwen3.5-397b --regime-sweep
python -m vllm_torchtpu.megakernel.advisor --model qwen3.5-397b --tp 8 --ep 8 --search-layer-groups
python -m vllm_torchtpu.megakernel.advisor --list-models
python -m vllm_torchtpu.megakernel.advisor calibrate \
  --trace path/to/trace.xplane.pb --out measurements.json \
  --chip tpu7x --model qwen3.5-397b --phase decode
```

## Key CLI Flags

| Flag | Description |
| --- | --- |
| `--model` / `--config` / `--hf-repo` | Shipped model preset name, local `config.json` file or model directory, or HF repo ID. |
| `--list-models` | Print validated presets, chip core-budget status, and 6-axis reference layouts. |
| `--chip` | Target TPU (`tpu7x`, `v6e`, `auto`). |
| `--act-dtype` / `--kv-dtype` / `--state-dtype` | Optional activation, KV cache, or recurrent state precision overrides (e.g. `bf16`, `fp8`, `fp32`). |
| `--phase` / `--batch` / `--decoding-seqs` | Serving phase and active sequences per step (default: `1` prefill, `16` decode). |
| `--isl` / `--osl` / `--context` | Input length (`4096` default), output length (`0` prefill, `1024` otherwise), and context length. |
| `--spec-tokens` / `--layer-groups` | Speculative draft token count and comma-separated `layer_group` sizes. |
| `--output-tok-s` / `--mean-tpot-ms` | Derive active decoding sequences `B = output_tok_s * (mean_tpot_ms / 1000)`. |
| `--tp` / `--ep` / `--dp` / `--pp` / `--dcp` / `--pcp` | Explicit 6-axis parallelism mesh degrees (`--tp` auto-selected if omitted). |
| `--hbm-utilization-cap` | Serving HBM pool headroom fraction (`HBM_UTILIZATION_CAP = 0.92` default). Physical OOM (`hbm_fits=False`) triggers only above `1.00x` HBM; exceeding the pool cap sets `hbm_pool_overflow=True`. |
| `--measurements` | Path to an `advisor.measurements.v1` JSON file, or `bundled`/`auto` for `tpu7x` priors. |
| `--host-head-us` | Optional per-step host dispatch head `per_op=<us>,fused=<us>` (default: `per_op=0,fused=0`). |
| `--per-op-step-us` / `--step-coverage` | Measured baseline `per_op` E2E step latency (`us`) and fused-eligible step fraction `(0, 1]`. |
| `--max-rows-per-call` | Most decode or verify rows served per fused call (`0` defaults to `CORE_LIMITS.max_batch_rows`). |
| `--objective` | Optimization objective (`step`, `throughput`, or `both`). |
| `--regime-sweep` / `--no-deployment-preset` | Evaluate the canonical phase, context, and batch regime table (optionally ignoring presets). |
| `--search-layer-groups` | Compare candidate `layer_group` partitions against `per_op`. |
| `--explain` / `--unresolved` | Print provenance tags and unresolved `ASSUMED` inputs ranked by sensitivity. |
| `--json [out.json]` | Emit machine-readable JSON output to stdout (when omitted) or `out.json`. |

## Physics & Drift Gates

- `HBM_UTILIZATION_CAP = 0.92` models serving non-KV runtime headroom without masking physical HBM OOM (`1.00x` HBM capacity).
- `CoreLimits` in `constants.py` reads live execution limits directly from `core/exec_spec.py`, `core/dma/vmem.py`, `core/profiles/default.py`, and `core/runtime/constants.py` (`max_batch_rows`, `chip_kinds`, `dense_ring_slots`, `expert_ring_slots`, `dense_slot_bytes`, `expert_slot_bytes`, `vmem_mib`, `smem_mib`, `wide_rows`, `default_fit_max_rows`, `fit_vmem_headroom_mib`, `default_decode_bucket_top`, `default_verify_bucket_top`), alongside `verified_ranks` and `max_ar_slots_per_layer`.
- `verify_core_limits_sync()` imports `megakernel/core` and runs 9 cross-source and structural drift checks verifying `verified_ranks`, `chip_budgets`, `min_ring_depth`, `core_module_families`, `core_supported_dtypes`, `kv_page_dtype`, `recurrent_state_dtype`, `default_bucket_tops_within_max_bucket_rows`, and `max_ar_slots_per_layer` stay synchronized with the core kernel package.
- Built-in chip geometry and sublane matmul tiles come directly from JAX TPU info queries in `chips.py`.

## Adding a Model

1. Add a pinned HuggingFace `config.json` snapshot under
   `tests/megakernel/data/advisor/hf_configs/`.
2. Add the model geometry and dtype entry to `data/presets/models.json` and
   reference topologies to `data/presets/deployments.json`.
