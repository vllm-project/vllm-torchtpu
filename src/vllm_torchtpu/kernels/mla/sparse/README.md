# Sparse MLA Attention Kernel Package

This directory implements TPU-friendly Multi-Head Latent Attention (MLA)
ragged paged attention with top-k sparse KV gathering. It supports both
TensorCore (`uint8`) and native SparseCore (`uint32`) KV cache layouts,
decode and mixed/prefill buckets, capped top-k windows with in-kernel
online-softmax overflow fix-up, and Distributed Context Parallelism (DCP)
with optional send-layout packing, fused small all-to-all exchange, and
in-kernel partial merging.

## Module Boundaries

The package is structured as a strict acyclic dependency graph so each layer
can be understood, modified, and unit-tested independently without breaking
Pallas/Mosaic lowering:

```text
kernel.py               Public facade, backward-compat exports, chunk orchestration
├── gather_dispatch.py  PreparedIndices, GatheredChunk, and KV gather strategies
│   └── index_prep.py   Deduplicated page-table lookup, window & atom-block prep
├── attention_kernel.py Mode-specialized Pallas kernels (decode vs. mixed/prefill)
│   ├── attention_body.py  Stateless Q/KV load, math, 4 strategies, overflow fix-up
│   ├── dma.py             Stateless KV/Q/O/LSE DMA fetches/sends & width ladders
│   ├── dcp_exchange.py    Send-layout packing, LSE words, RDMA barrier & merge
│   └── kernel_context.py  Frozen KernelGeometry and DoubleBufferState
├── ref_registry.py     Declarative PallasRefRegistry, BoundsRef & DmaSemaphores
├── config.py           StrEnums, frozen validated dataclasses & layout predicates
└── testing_faults.py   Isolated test-only FaultHooks (zero production fault checks)
```

Supporting gather and cache-layout primitives live alongside these modules in
`dsa_gather.py`, `native_sc_gather.py`, and `native_sc_cache_layout.py`.

## Configuration Contracts (`config.py`)

All static kernel configuration flows through frozen, hashable dataclasses
safe for `jax.jit(static_argnames=...)`:

- `SparseMlaConfig`: top-level composite configuration grouping
  `GatherConfig`, `CappedWindowConfig`, `LseConfig`, and `DcpConfig`, plus
  single-source-of-truth `validate_caches` and `validate_queries`.
- `BodyOptimizationConfig`: replaces raw lever string tuples with typed
  boolean flags and a single mutually exclusive `BodyExecutionStrategy`
  (`TOKEN_PIPELINED`, `PAIR`, `QUAD`, `TRANSPOSED_QK`), making conflicting
  scheduling modes unrepresentable.
- `SendExchangeConfig`: replaces positional 2–5 element tuples with named
  fields (`axis`, `lane_axis`, `vmem_limit_bytes`, `hbm_pin`,
  `in_kernel_merge`) and typed `HbmPinRole` matching.
- `AttentionKernelConfig`: resolved per-launch Pallas kernel configuration
  whose `__post_init__` enforces all cross-flag invariants before tracing:
  - `preserve_suffix` requires TensorCore layout without LSE.
  - `q_rope` requires `split_q`; `narrow_q_rope_dma` (`"q128"`) requires
    `q_rope`.
  - `has_group_tab` and `row_widths` are mutually exclusive; `live_widths`
    requires `has_group_tab`; `row_widths` requires `capped_windows` on the
    SparseCore layout.
  - `send_enabled` requires `lse_fill`, `split_q`, and `not preserve_suffix`;
    `xchg_enabled` requires `send_enabled`; `recv_in` requires `xchg_enabled`
    and `not xchg_merge`.
  - Any active `body` optimization requires `sparsecore_layout`, either
    `has_group_tab` or `row_widths`, and passes mode-specific validation
    (e.g., decode `zero_nope_words_only` requires `pv_nope_only`).

Public entrypoints (`sparse_ragged_paged_attention` and
`sparse_attention_chunk`) accept `config=SparseMlaConfig(...)` and continue
supporting legacy keyword arguments via a hashable `UNSET` sentinel. Passing
both `config` and any legacy config keyword argument fails fast with
`ValueError`.

## Buffer, Bounds, and Semaphore Contracts (`ref_registry.py`)

All Pallas `PrefetchScalarGridSpec` slots are declared by name through
`PallasRefRegistry`:

- `build_attention_ref_registry` and `build_live_fused_prep_ref_registry`
  register prefetches, inputs (with `HbmPinRole` and by-name `alias_output`),
  outputs/output groups, and VMEM/SMEM/DMA-semaphore scratches in deterministic
  order.
- `registry.layout` produces an array-free, frozen, hashable `RefLayout`
  passed via `functools.partial` into `pl.pallas_call` (never closing over
  traced `jax.Array`s). Inside the kernel, `AttentionRefs.from_bound(...)`
  and `LiveFusedPrepRefs.from_bound(...)` bind every slot by name and fail
  fast on arity mismatch.
- `input_output_aliases` is derived automatically from `alias_output` names,
  eliminating manual operand-offset arithmetic.
- `BoundsBuilder` / `BoundsRef` name every slot of the scalar bounds prefetch
  (`start_seq_idx`, `end_seq_idx`, optional `prefix_end`, `peer_rank`,
  `peer_lane`, `real_tokens`), raising `KeyError` on unregistered accesses.
- `DmaSemaphores` wraps the core DMA semaphore array with named accessors
  (`.kv_nope`, `.q`, `.o`, `.kv_rope`, `.lse`).

All registry, geometry, and hook abstractions resolve strictly at Python
trace time and emit zero additional XLA/Mosaic ops or VMEM allocations.

## Mode-Specialized Pallas Kernels (`attention_kernel.py`)

`select_attention_kernel(cfg)` routes each `pl.pallas_call` invocation to one
of two concise, mode-specialized kernels sharing common step and epilogue
pipeline helpers:

- `_decode_attention_kernel`: handles decode-specialized executions
  (`lse_fill`, `row_widths`, `send_world`, `send_exchange`, `xchg_merge`, and
  decode DCP mode) when `not cfg.has_group_tab and not cfg.preserve_suffix`.
- `_mixed_prefill_attention_kernel`: handles mixed/prefill executions
  (`has_group_tab` / `live_widths`, DCP `kv_rungs`, standard TensorCore and
  SparseCore prefill, `preserve_suffix`, `out_pieces`, and single-chunk
  mixed `lse_fill`).

## Fault-Injection Isolation (`testing_faults.py`)

Positive-control fault injections used by the test suite live exclusively in
`testing_faults.py` as typed `StrEnum`s and a frozen, hashable `FaultHooks`
dataclass (`NO_FAULT_HOOKS` default). Production kernel code contains zero
inline `if fault == ...` string checks, and `NO_FAULT_HOOKS` short-circuits
in pure Python at trace time with zero runtime overhead.

## Testing

CPU contract tests (no TPU required):

```bash
pytest tests/kernels/test_sparse_mla_contracts.py
```

TPU numerical parity tests:

```bash
pytest \
  tests/kernels/test_mla_native_sc_attention.py \
  tests/kernels/test_mla_decode_prologue.py \
  tests/kernels/test_mla_decode_row_widths.py \
  tests/kernels/test_mla_decode_body_levers.py \
  tests/kernels/test_dcp_mla_send_layout.py \
  tests/kernels/test_dcp_mla_token_split.py \
  tests/kernels/test_mla_fused_small_a2a.py \
  tests/kernels/test_mla_kv_write_in_gather.py \
  tests/kernels/test_mla_sparse_pad_hash.py \
  tests/kernels/test_sparse_mla_padding.py \
  tests/kernels/test_t3_final_composition.py \
  tests/kernels/test_t4_combined_composition.py \
  tests/kernels/test_t4_stack2_composition.py
```
