# Qwen3.8-2.4T TP32 x DCP8 NVFP4 Megakernel (`qwen38_2_4t`)

Self-contained megakernel model folder for Qwen3.8-2.4T (`num_hidden_layers == 92`,
`hidden_size == 8192`, `3:1` GDN + Full Attention, `512` routed experts top-`8` +
`1` shared expert) under `TP=32 x DCP=8` and `NVFP4` quantization.

## Runtime Contract (`SPEC`)

- **Package**: `vllm_torchtpu.models.qwen38_2_4t.megakernel`
- **Plugin Selection**: `plugin=vllm_torchtpu.models.qwen38_2_4t.megakernel`
  (or `{"megakernel": {"plugin": "vllm_torchtpu.models.qwen38_2_4t.megakernel"}}`)
- **Collective domain**: `DOMAIN_TP` (`tp`) so `ProgramDriver` startup agreement runs
  uniformly across the 32-rank TP group while the kernel performs `DCP=8` indexing inside
  the TP mesh.
- **Hooks**:
  - `create = ".adapter:create"`
  - `layout_request = ".adapter:layout_request"` (`Qwen38LayoutRequest(block_size=128, cp_kv_cache_interleave_size=128, mamba_slot_tokens=640)`)
- **Plugin Options (`OPTION_KEYS`)**:
  - `buckets`: comma-separated or sequence decode bucket sizes (default `(1, 2, 4, 8, 16)`)
  - `compile_hlo_check`: fail if lowered StableHLO contains `all-gather` (default `False`)
  - `allow_mini_geometry`: allow 4- or 8-layer mini test geometry (`H=128`, default `False`)
  - `interpret`: `bool` or `pltpu.InterpretParams` override for CPU/debug execution

## Quickstart

### 1. Copy / Register (for a new model)

Copy `src/vllm_torchtpu/models/qwen38_2_4t/megakernel/` to your model package, define a
`SPEC = TpuMonoSpec(...)` in `__init__.py` (keeping `__init__.py` stdlib-only at import),
and select the module via `VLLM_TPU_MEGAKERNEL=plugin=<module>` or
`additional_config["megakernel"]["plugin"]`.

### 2. Test (CPU, no TPU)

```bash
JAX_PLATFORMS=cpu pytest -m "not tpu_test and not multichip and not nightly" \
  tests/models/qwen38_2_4t/
```

### 3. Serve (TP32 x DCP8, TPU)

```bash
vllm serve /path/to/qwen38-checkpoint \
  --tensor-parallel-size 32 \
  --decode-context-parallel-size 8 \
  --enable-expert-parallel \
  --additional-config '{"megakernel": {"plugin": "vllm_torchtpu.models.qwen38_2_4t.megakernel"}}'
```

### 4. Benchmark

```bash
vllm bench serve \
  --backend vllm \
  --model /path/to/qwen38-checkpoint \
  --dataset-name sonnet \
  --num-prompts 100
```

## Composed Substrate Primitives

- **Hierarchical TP32 All-Reduce (`dma.Qwen38HierAllReduce`)**: Composes
  `PeerTopology(rank=me, world_size=32, group_size=4)` and
  `make_remote_copy_batch` from `vllm_torchtpu.megakernel.core.collectives` for a 2-stage
  `4x8` reduction (4-rank intra-tray ring followed by 8-way XOR reduction).
- **DCP=8 Query & LSE Exchanges**: Uses `make_remote_copy_batch` across the
  8-rank DCP group (`(me // 8) * 8 + peer`).
- **Stage Scratchpad Reuse (`plan.build_stage_scratch_plan`)**: Uses `StageScratchPlan`
  and `pool_alias.view(pool_ref, shape, dtype)` from `vllm_torchtpu.megakernel.core.dma`
  to pack stage-disjoint down-projection buffers (`wo_buf`, `wout_buf`, `wse2_buf`)
  onto a shared scratchpad interval.

## Limitations

- **Decode-only (`T = 1`)**: Serves uniform `T = 1` decode steps (`B in buckets`);
  prefill and multi-token speculative verify steps fall back cleanly to the stock
  vLLM model graph.
- **Target Topology & Cache Layout**: Scoped to `TP = 32 x DCP = 8` on TPU v7x
  (`TPU7x`, `tp_sizes=()` on `SPEC` so TP mismatches decline rank-uniformly in
  `adapter.create`). Uses a `128`-token kernel KV page size (`block_size = 128`
  in `Qwen38LayoutRequest`) while requiring `cache_block_size >= 640` (`5`
  `128`-token pages per Mamba state slot) on unified runner-owned region pools.
- **VMEM Budget Headroom**: Explicit Pallas scratchpad buffers occupy
  `38.48 MiB` (`B = 1`), `45.67 MiB` (`B = 8`), and `61.01 MiB` (`B = 16`) against
  the `62 MiB` `TPU7x` VMEM limit (`23.1 KiB` scalar SMEM against `1 MiB`) before
  Mosaic compiler temporaries; pass `buckets="1,2,4,8"` if compiler temporaries
  exceed VMEM at `B = 16`.
- **Quantization**: Requires 16-element group NVFP4 quantized checkpoints (`nvfp4`,
  `fp4`, or `compressed-tensors` / `modelopt` FP4).
