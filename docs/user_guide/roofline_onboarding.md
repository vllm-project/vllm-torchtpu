# Roofline Model and Hardware Onboarding

`vllm_torchtpu.roofline` separates Hugging Face architecture parsing (`ModelGeometry` and `Dtypes`) from capability-driven Tensor Processing Unit (TPU) hardware profiles (`ChipSpec` and `SliceTopology`). Models and hardware onboard through structural config inspection and declarative JSON profiles without chip-name branching or empirical efficiency multipliers.

> **L1 Shortlist Scope (`bound_*` Pruning vs. `est_*` Heuristic Ranking):**
> By default, `solve` and `emit` run as an uncalibrated Layer-1 (L1) model (`provenance.calibrated == false`). Physical lower bounds (`bound_ttft_ms`, `bound_tpot_ms`, `bound_interactivity_tok_s`, `floor_throughput_per_chip`) safely prune provably infeasible layouts (`slo_verdict == "ruled_out"`), whereas uncalibrated serial-phase estimates (`est_ttft_ms`, `est_tpot_ms`, `est_interactivity_tok_s`, `token_throughput_per_chip`) rank `recommended`, `frontier`, and `top_k`, and `bound_families_by_concurrency` shortlists structural layout families by active Time-Per-Output-Token (TPOT) lower bound. Verify exact P/D instance ratios, chip counts, and operating concurrencies with L2/L3 profiling, or supply measured calibration inputs such as `--collective-latency-us`.

---

## 1. Three-Tier Model Onboarding

| Tier | Trigger (`onboarding_tier`) | Action |
| :--- | :--- | :--- |
| **Tier 0: Zero-code Day-0** | `tier0_ready` (`listed` or `unlisted_structural_ok`) | Run `solve`, `emit`, `ceiling`, `plan`, or `explain` directly (pass `--allow-unlisted-model-type` for unseen `model_type` strings whose structural keys validate). |
| **Tier 1: Config-only / Tech-report synthesis** | `tier1_config_needed` (`unlisted_blocked` or missing structural keys) | Add `--allow-unlisted-model-type`, synthesize a Hugging Face `config.json` from published architecture dimensions, or register `model_type` in `SUPPORTED_MODEL_TYPES` (`src/vllm_torchtpu/roofline/constants.py`). |
| **Tier 2: 4-file primitive seam** | `tier2_primitive_needed` (novel attention, recurrence, or Feed-Forward Network (FFN) primitive, or a pure State-Space Model (SSM) without attention, Gated DeltaNet (GDN), or Kimi Delta Attention (KDA) heads) | Implement the primitive across `constants.py`, `geometry.py`, `memory.py`, and `modules.py`. |

### Supported Structural Controls and Key Audit (`inspect --json`)

Run `inspect` before planning to audit structural key coverage and parameter conservation (`inspect` writes `vllm_torchtpu.roofline.inspect/v1` JSON to stdout even when blocked, exiting `0` for `tier0_ready` and `2` otherwise):

```bash
python -m vllm_torchtpu.roofline inspect \
  --config /path/to/config.json \
  --allow-unlisted-model-type \
  --json
```

Pass `--model <hf_repo_or_dir>` to load a Hugging Face repository ID or local model directory, and optionally `--chip-spec <chip.json>` to bind a custom hardware profile.

During inspection, `audit_hf_config` first checks structural primitives via `supports(raw_cfg, allow_unlisted_model_type=True)` and then enforces the `model_type` gate via `supports(raw_cfg, allow_unlisted_model_type=allow_unlisted_model_type)`. Next, `_run_tracked_extraction` wraps `raw_cfg` in `KeyTracker` across `from_hf_config` and `dtypes_from_hf_config` to record every dotted key path read (also running as a dry-run on `unlisted_blocked` configs to surface `unmapped_structural_keys`):

- **`unmapped_structural_keys`**: text-config keys whose values were not read and that are absent from `BENIGN_HF_METADATA_KEYS`. Non-empty output flags structural fields that the roofline loader did not model.
- **`ignored_non_text_keys`**: top-level multimodal keys from `MULTIMODAL_KEYS` (such as `vision_config`) skipped during text-only extraction.
- **`param_counts` conservation oracle**: computes `total_params` and `active_params_per_token` (along with `attention`, `linear_attention`, `dense_ffn`, `moe_routed_ffn`, `moe_shared_ffn`, `router`, `embedding`, `lm_head`, `mtp`, and `hc`) using the same single-source weight functions as `hbm_breakdown_per_core_bytes`. Compare `total_params` and `active_params_per_token` against published model specifications to catch dropped or double-counted weights.

When synthesizing a tech-report `config.json`, include exact structural fields where applicable: `ffn_num_matrices` (`3` for Gated Linear Unit (GLU) layers, `2` for non-gated or `is_gated_mlp: false`), `tie_word_embeddings`, `dsa_layers` or `indexer_layers` (for partial-layer DeepSeek Sparse Attention (DSA) over Multi-head Latent Attention (MLA)), and `linear_attn_config` (for GDN or KDA recurrence).

### Tier 2 Four-File Primitive Seam

When a model introduces an unmodeled attention, recurrence, or FFN building block, extend these four files under `src/vllm_torchtpu/roofline/`:

1. `constants.py`: register layer tokens in `VALID_LAYER_TOKENS` (plus `LINEAR_ATTENTION_LAYER_TOKENS`, `LAYER_TYPE_MAP`, and `NON_FULL_ATTN_TOKENS` if recurrent or non-full attention) and add `model_type` to `SUPPORTED_MODEL_TYPES`.
2. `geometry.py`: extract the new fields onto `ModelGeometry` in `from_hf_config`, and update `supports`, `LayerCounts`, and `count_layer_types`.
3. `memory.py`: add static weight element accounting (shared with `param_counts`) and Key-Value (KV) or recurrent state sizing in `_compute_kv_and_state`.
4. `modules.py`: add per-step FLOPs, High Bandwidth Memory (HBM) traffic, and Inter-Chip Interconnect (ICI) collectives in `pass_costs` (and `kv_bytes_per_token_per_layer` if KV packing changes).

---

## 2. Zero-Code Hardware Onboarding (`--chip-spec` and `from_jax`)

All compute, memory, Scalar Memory (SMEM) page-table, alignment, and torus collective formulas read capability traits directly from `ChipSpec` (`cores_per_chip`, `chips_per_host`, `ici_links_per_chip`, `fp4_flops`, `int8_flops`, `int4_flops`, `vector_flops`, `smem_page_table_bytes`, `physical_alignment_elems`, `subchannel_block_k_floor`, `min_torus_dim`), while `has_sparse_core` and `dma_granule_bytes` are stored on `ChipSpec` and surfaced as metadata in `chip_catalog(chip)`.

### Live JAX Hydration (`from_jax`)

`from_jax(name, version, overrides=ChipOverrides(...))` queries `jax.experimental.pallas.tpu.get_tpu_info_for_chip` and `jax.numpy` directly for per-TensorCore HBM capacity/bandwidth, Vector Memory (VMEM) capacity, Matrix Multiply Unit (MXU) peaks (`bf16`, `fp8`, `int8`, `int4`), tiling (`mxu_column_size`, `get_sublane_tiling`, `num_lanes`, `num_sublanes`), `smem_capacity_bytes // 2`, `sparse_core.dma_granule_size_bytes`, and `ChipVersion.num_physical_tensor_cores_per_chip`.

### Declarative JSON Profile (`vllm_torchtpu.roofline.chip_spec/v1`)

Pass `--chip-spec <chip.json>` to any CLI subcommand (`inspect`, `solve`, `emit`, `ceiling`, `plan`, `explain`, or `--assumptions`), or call `load_chip_spec(path_or_mapping)` in Python (optionally pairing with `register_chip(chip, topologies)`):

- **Required metadata**: `schema_version` (`"vllm_torchtpu.roofline.chip_spec/v1"`), non-empty `name`, and `provenance` (one of `"public_spec"`, `"jax_tpu_info"`, `"measured_microbench"`, `"projected"`).
- **Core physical fields** (strictly required `> 0` unless inheriting from `"base"`, such as `"tpu7x"`): `hbm_bytes`, `hbm_bw`, `bf16_flops`, `vmem_bytes`, `cores_per_chip`, `chips_per_host`. Both `vector_flops` and `vector_to_bf16_ratio` are optional (defaulting to the `base` chip's vector-to-BF16 ratio or `DEFAULT_VECTOR_TO_BF16_RATIO = 1/128` when omitted).
- **Capability and interconnect traits**: quantized MXU rates (`fp8_flops`, `int8_flops`, `fp4_flops`, `int4_flops`, defaulting to the `base` chip's value when `"base"` is set or `0.0` otherwise; unsupported `int8` falls back to `bf16_flops`, while sub-byte weights with `< 1` byte per element charge vector unpack work unless a native FP4/INT4 MXU peak is non-zero), ICI traits (`ici_links_per_chip`, `ici_link_bw`, `d2d_bw`, `ici_bw`, `ici_hop_latency_s`), Data Center Network (DCN) and Peripheral Component Interconnect Express (PCIe) bandwidths (`dcn_host_gbit_per_s`, `dcn_bw`, `pcie_bw`), `mxu_m_tile`, `mxu_lhs_rows_bf16`, `mxu_lhs_rows_fp8`, `num_lanes`, `num_sublanes`, `physical_alignment_elems`, `smem_page_table_bytes`, `dma_granule_bytes`, `subchannel_block_k_floor`, `min_torus_dim`, and `has_sparse_core`.
- **Bandwidth inheritance (`base` set) and standalone defaults (`base` omitted)**: when `"base"` is set, `d2d_bw` inherits `base_chip.d2d_bw` unless `ici_links_per_chip` or `ici_link_bw` is provided (in which case it defaults to `ici_links_per_chip * ici_link_bw`); `ici_bw` inherits `base_chip.ici_bw` unless `ici_links_per_chip`, `ici_link_bw`, or `cores_per_chip` is provided (in which case it defaults to `ici_links_per_chip * ici_link_bw / cores_per_chip`, decoupled from `d2d_bw`); and `dcn_bw` inherits `base_chip.dcn_bw` unless `dcn_host_gbit_per_s` or `chips_per_host` is provided (in which case it defaults to `(dcn_host_gbit_per_s * 1e9 / 8) / chips_per_host`). Without `"base"`, omitted hardware capability defaults (`mxu_m_tile`, `mxu_lhs_rows_bf16`, `mxu_lhs_rows_fp8`, `num_lanes`, `num_sublanes`, `physical_alignment_elems`, `smem_page_table_bytes`, `has_sparse_core`, `dma_granule_bytes`) hydrate live from `jax.experimental.pallas.tpu.get_tpu_info_for_chip` (`mxu_column_size`, `get_sublane_tiling`, `num_lanes`, `num_sublanes`, `smem_capacity_bytes // 2`, `sparse_core.dma_granule_size_bytes`), while omitted interconnect/tiling assumptions fall back to module constants (`DEFAULT_ICI_LINKS_PER_CHIP`, `DEFAULT_ICI_LINK_BW`, `DEFAULT_ICI_HOP_LATENCY_S`, `DEFAULT_DCN_HOST_GBIT_PER_S`, `DEFAULT_PCIE_BW`, `SUBCHANNEL_BLOCK_K_FLOOR`, `MIN_TORUS_DIM`).
- **Slice shapes (`topologies`)**: list of `{"name": str, "chip_axes": [x, y, z], "wrapped_axes": [bool, bool, bool]}`. When omitted, `"base"` inherits the base chip's slice shapes updated with the resolved `cores_per_chip` and `min_torus_dim`, while standalone specs default to a single `"1x1"` slice.

Example minimal 1-core 2D torus profile (`chip.json`):

```json
{
  "schema_version": "vllm_torchtpu.roofline.chip_spec/v1",
  "provenance": "public_spec",
  "name": "example_1core_2d",
  "cores_per_chip": 1,
  "chips_per_host": 4,
  "hbm_bytes": 32000000000,
  "hbm_bw": 1600000000000.0,
  "bf16_flops": 500000000000000.0,
  "fp8_flops": 1000000000000000.0,
  "vmem_bytes": 67108864,
  "ici_link_bw": 50000000000.0,
  "ici_links_per_chip": 4,
  "ici_hop_latency_s": 1e-06,
  "dcn_host_gbit_per_s": 200.0,
  "topologies": [
    {"name": "1x1", "chip_axes": [1, 1, 1], "wrapped_axes": [false, false, false]},
    {"name": "2x2", "chip_axes": [2, 2, 1], "wrapped_axes": [false, false, false]}
  ]
}
```

Run serving layout optimization against that custom profile on CPU:

```bash
python -m vllm_torchtpu.roofline solve \
  --chip-spec /path/to/chip.json \
  --config /path/to/config.json \
  --allow-unlisted-model-type \
  --example-scenario chat-8k1k \
  --max-chips 4 \
  --json
```

---

## 3. Four-Step Agentic Onboarding Checklist

Execute this closed-CPU verification loop (each CLI command completes in a few seconds without TPU hardware):

1. **Stage inputs**: obtain the Hugging Face `config.json` (or synthesize one from published tech-report dimensions) and, if targeting unlisted silicon, write a `vllm_torchtpu.roofline.chip_spec/v1` JSON profile.
2. **Audit structural coverage**: run `inspect --json` (adding `--allow-unlisted-model-type` for unseen `model_type` values) and verify `onboarding_tier == "tier0_ready"`, `unmapped_structural_keys == []`, and `blocking_reasons == []`.
3. **Check parameter conservation**: verify `param_counts.total_params` and `param_counts.active_params_per_token` match published specs; add missing `ffn_num_matrices`, `tie_word_embeddings`, `dsa_layers`, or `linear_attn_config` fields if mismatched.
4. **Solve, emit, and plan**: run `solve` (and `emit` or `ceiling`) with `--chip-spec` and `--max-chips` against a target `--example-scenario` to inspect L1-feasible layout families (`bound_*` pruning and `est_*` heuristic shortlisting), or run `plan` (or `explain`) with `--chip-spec`, `--config`, and a Tensor Parallelism (TP) degree `--tp` with zero runtime errors.
