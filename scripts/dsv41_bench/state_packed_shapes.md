# Compressor state-packed shape experiment

**Post-integration update:** the loader now executes the integrated packed source
without rewriting its guard. Production supports the bounded new T values;
T288 and combined-addresses remain unchanged. The original exploration notes
below describe the pre-integration experiment, not current production dispatch.
See [accepted production evidence](state_packed_bridge.md) for actual Torch A/B.
Both standalone CPU entrypoints were rerun successfully with local JAX0.11.2.

Historical benchmark-only invocation: run
`python scripts/dsv41_bench/state_packed_shapes.py --cpu-test` locally, or run
`python scripts/dsv41_bench/state_packed_shapes.py --out /tmp/state-packed --samples 100 --profile`
on an existing TPU7 environment. No jobs are submitted by this script.

## Exact scope and source contracts

- Tokens: 96, 192, 240, 256; original fused 288 control. B48, table48x8,
  capacity16, stride16, offset0, replay window128. All five inputs INT32,
  shapes `(48,8), (49,), (49,), (T,), (48,)`.
- Production `supports_compressor_state_packed` remains exact T288 and exact
  geometry. The production packed entry additionally checks actual shapes and
  dtype. `_compressor_state_addresses` derives T and request count, but hardcodes
  geometry16/16/0 and window128.
- `preprocess_ops.compress_rope_packed` is TPU7-only and dispatches non288 to
  `_compress_native_metadata`; its separate combined-address entry is out of scope.
- Candidate loads the actual packed source in a private namespace and changes
  only its expected positions shape. The external benchmark gate restricts it to
  the listed cases and geometry; all other shape/dtype checks remain intact.
  At T288 the harness invokes the **original unmodified packed function**.
- Baseline uses unchanged Torch native helper extraction from attention_state,
  then the actual `compress_rope._compact`, including its original T288 fast path.
  Timed outputs on both sides are the six-element flags-and-tables tuple.
  No rewritten JAX baseline replaces native preparation.

## Independent correctness and consumer offsets

The NumPy scalar-loop oracle maps each token with right-boundary request lookup,
clamped to request47, independently for state and KV boundaries. `has_previous`
means token row differs from its request start. Valid compression means an active
KV token, odd position, and not in `[minimum, minimum+128)` when minimum is positive.
State slot is `block_table[request,0]*16 + position%16`; only active nonnegative
slots in the request's last16 rows are written. Gather uses `(position-1)%16`
only at request starts with a nonnegative active current slot. Stable compaction
drops addresses outside `[0,total_records)` and emits token/address pairs, zero
suffix, and INT32 `[0,count]` offsets. Tests cover uniform, ragged/zero-length,
padding, empty, distinct state/KV boundaries, long requests, negative/out-of-range
blocks and bank-boundary addresses. Both raw addresses and compact outputs are
compared; no repeated-block DMA safety claim is made from address-only tests.

The harness executes source-extracted **actual select_config** and tuning table
with TPU version7, hidden5120, ratio2. T96/192/256 default to whole-T; T240 and
T288 select tuned whole-T, weight_chunk512. Thus each has one tile and requires
exactly two offsets, so these tuples are structurally safe for that consumer.
The actual consumer checks flat pair shape `(2*T,)`, offsets `(tiles+1,)`, INT32,
and unpadded T, then accesses offsets[tile] and offsets[tile+1]. T384 defaults to
192-row tiles and is excluded: `[0,count]` cannot describe multiple tiles.
T288 without the tested tuning (different hidden/runtime) defaults to144, so a
future production broad guard must retain the TPU7/hidden5120/ratio2 envelope or
verify `selected_config.token_block == padded_T`, not simply allow T<=288.

## Evidence and limitations

CPU mode verifies 35 source-expression/oracle cases and whole-T selection. It
uses NumPy for source address and compaction expressions (NumPy cumsum replaces
prefix_sum), not TPU Pallas compilation. Device mode additionally checks actual
native and packed tuples, performs synchronized ABBA samples after warmup, and
optionally records16 calls per side per shape. Compact JSON includes per-file
SHA256, geometry, correctness, and raw timing samples. CPU output has no latency
claims.

**Must-follow integration:** actual `fused_compress_rope` consumer with identical
weights, inputs, RoPE, initial state and both bank outputs must be checked and
benchmarked before production adoption. This local environment has no JAX,
Torch or TPU runtime; no device or end-to-end result was measured. Address-table
correctness plus source-proven offsets is not numerical consumer validation or
a speedup claim. Consumer integration must use well-formed contiguous request
positions and unique valid write records; synthetic address boundary stress
cases are not necessarily safe DMA workloads. Preserve broad native fallback
and the original T288 target when proposing any later production gate change.
