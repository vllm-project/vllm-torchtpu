# Fused sparse GQA decode

QSA uses fused decode on supported TPU7x inputs and retains batched RPA
otherwise; prefill is unchanged. [Eligibility][eligibility] requires BF16
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
No fitting single-query plan means RPA fallback; larger batches are chunked.
Benchmark maxima do not define eligibility.

## DMA, ownership and rounding

The shared RPA wait ABI uses 128 uint32 words per row: `128*4 = 512` bytes.
[Schedule copies][abi] preserve complete rows: zero start, length aligned to
128 words and capped at capacity; zero work skips DMA. Start/wait slices match.
This is the supported row-preserving policy, not a silicon DMA minimum.
Unknown fields fail closed. Selected KV is not materialized in an HBM gather;
the returned cache aliases its input and preserves the caller's donation.

The visible RPA pipeline retains input/output slots 3/2 and LAST_TILE scratch
carry/reset. Sigmoid, explicit BF16 rounding and multiplication precede output
DMA; integer RNE, zeroing and fences are required arithmetic/ownership boundaries.

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
