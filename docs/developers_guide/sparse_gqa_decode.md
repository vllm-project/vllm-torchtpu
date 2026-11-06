# Fused sparse GQA decode

QSA uses fused decode when the shared row ABI and resource budgets fit,
and bounded batched RPA otherwise. Prefill layouts without aligned TC DMA segments also use bounded RPA.
[Eligibility][eligibility] requires BF16
Q/gate, divisible positive Q/KV heads, lane-aligned dimensions, native
`HEAD_ALONG_SUBLANE` BF16/FP8 cache, nonempty metadata and int32-bounded
addresses, positive finite K/V scales, and a positive ratio dividing page and tile.
Callers guarantee valid prefix IDs/lengths and a cache containing the current
token. The direct entry rejects unsupported inputs; dispatch falls back to RPA.

## Geometry and launch

Physical page size is `S = cache.shape[1]`; a virtual selected group has
`r = ratio` tokens. With `a = S/r`, group `g` maps to `table[g//a]*a + g%a`.
A compute tile holds `bkv_tokens/r` groups. Duplicates retain multiplicity;
partial tails are included and inactive padding is masked.

The 256-token tile and eight lanes balance scheduling work and live buffers;
this measured policy is not a hardware law or universal optimum. [Planning][planning]
derives capacity from shapes, schedule/scalar bytes and live buffers against
SMEM/VMEM budgets; conservative compiler headroom is not measured peak use.
The actual batch is tried before smaller power-of-two chunks. No fitting
single-query plan means RPA fallback.
Benchmark maxima do not define eligibility.

## DMA, ownership and rounding

The shared RPA wait ABI uses 128 uint32 words per row: `128*4 = 512` bytes.
[Schedule copies][abi] preserve complete rows: zero start, length aligned to
128 words and capped at capacity; zero work skips DMA. Start/wait slices match.
This is the supported row-preserving policy, not a silicon DMA minimum.
Unknown fields fail closed. Selected KV is not materialized in an HBM gather;
the returned cache aliases its input and preserves the caller's donation.
The schedule resolves physical page IDs before the shared KV buffers consume it.
Decode scalar prefetch holds query lengths and offsets; the RPA fallback accounts
for both page-table DMA windows in its SMEM bound.

The visible RPA pipeline retains input/output slots 3/2 and LAST_TILE scratch
carry/reset. Sigmoid, explicit BF16 rounding and multiplication precede output
DMA; integer RNE, zeroing and fences are required arithmetic/ownership boundaries.

## Output projection and preparation

Projected decode derives the BF16 weight tiles, logical output width and HC
branches from the supplied arrays. The final output-column tile is padded
internally and sliced back to the logical width. Each dot reduces the complete
input width with FP32 accumulation and the existing BF16 output rounding.

The planner counts the resident full weight, padded attended/residual/gate/output
buffers and projection working planes alongside the attention buffers. It tries
the default pipeline concurrency, then smaller powers of two when necessary to
fit that combined budget. If fusion cannot fit, QSA attention followed by the
separate projection and HC operations remains available. Packing also checks the
resident-weight lower bound before creating a persistent tiled copy. The fused
HC entry requires an unsharded projection input because it has no cross-shard
sum before adding the residual.

Decode preparation derives head grouping and rotary extents from the arrays;
state storage retains its eight-row ring/tag padding ABI. Index scoring groups
whole pages into lane-aligned key tiles, with a named target key count used only
for tuning. A minimum aligned tile that cannot fit VMEM is rejected explicitly.
These resource checks do not establish a speedup for every admitted geometry.

## Checks and benchmark

```bash
pytest tests/kernels/test_sparse_gqa_decode.py \
  tests/kernels/test_sparse_gqa_decode_contracts.py
python scripts/benchmark_sparse_gqa_decode.py --help
```

Use `tpu_test` markers for hardware tests. Fixture correctness and synchronized
samples do not establish full-model or every-shape performance.

[eligibility]: ../../src/vllm_torchtpu/kernels/sparse_gqa_decode/eligibility.py
[planning]: ../../src/vllm_torchtpu/kernels/sparse_gqa_decode/planning.py
[abi]: ../../src/vllm_torchtpu/kernels/sparse_gqa_decode/abi.py
