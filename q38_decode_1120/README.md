# Qwen3.8-Flash decode-leg configs (2026-11-20)

Decode-leg tuning of sheet `q38bpb1119-1pyd-exos-pp8-tierkr400-blcfg-1p1d-fp8cache1119-ixccfg-blmatch-bestband-noxprof-d1200-qwen38-pd-hc2-i6`
(dash `d-5234ec7a-1f25-470f-acdc`): 1P1D, PP8 prefill -> TP8 + main-KV DCP4 decode, MTP k=3, `max_num_seqs` 64.
The sheet ran this branch's parent, `2c6ceeca2`. Microbench dash: `d-632b8507-c933-44a4-8961`.
No FP4 anywhere: `--quantization=fp8`, `--kv-cache-dtype=fp8`, `VLLM_TPU_OFFLOAD_FP4` unset.

## c1 (recommended): env only, on `2c6ceeca2` (no code from this branch needed)

Add to the decode leg (`DECODE_EXTRA_SERVER_ENV`, and the recipe's `DECODE_TPU_TOKEN_BUCKET_EXTRA`):

```
DECODE_TPU_TOKEN_BUCKET_EXTRA=48,80,96,112,160,192,224      # served: 96,112,160,192
TPU_QWEN4_HC_DECODE_COMPARE=1                               # hc_decode_cmp.py (summation order differs, ulp-level)
VLLM_TPU_DECODE_TP_AR=h8x                                   # hierarchical TP8 all-reduce, f32 wire (bitwise)
VLLM_TPU_QSA_UNION_LEAN_GLUE=1                              # "same results"
VLLM_TPU_QSA_UNION_CHAINED_CALLS=1                          # "same results"
VLLM_TPU_QSA_INDEXER_META_SELECT=1                          # "same integers"
VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD=1                     # "same selection"
VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD_MIN_ROWS=128
```

| conc (decode running) | microbench step (sheet `mbhc-q38pd1119-decode-c1`) | served TPOT p50 | served tok/s/chip | served TTFT p50 |
|---|---|---|---|---|
| 4 (1) | +9.6% (host-noise run) | +0.3% | -1.2% | +0.7% |
| 8 (2) | -5.8% | +0.6% | -0.5% | +1.3% |
| 32 (7) | -6.0% | -5.2% | +0.8% | +0.8% |
| 64 (16) | ~-6% | -5.6% | +0.4% | +0.9% |
| 128 (47) | -8.4% | -8.1% | +3.7% | -7.8% |
| 160 (52) | -10.2% | -6.0% | +6.6% | -27.3% |
| 192 (44) | -7.9% | -11.6% | -0.9% | -3.8% |
| 224 (18) | -5.8% | -4.2% | +0.1% | -22.6% |
| 256 (20) | -6.3% | -7.5% | +2.2% | -13.5% |

Served = sheet `q38bpb1119-blmatch-bestband-mbhc-c1` vs `q38bpb1119-blmatch-bestband-rerun-base` (same window), both on
`d-5234ec7a-1f25-470f-acdc`. Served c1 jobs: c4 j-c46ceccb-2f29-483c-89d6, c8 j-9005150a-caf6-4e88-8814,
c32 j-88304706-29bd-4e1c-9393, c64 j-3710d5a3-7b8a-4c95-873c, c128 j-0b829832-fa1a-46ec-8bd5,
c160 j-87adce29-98a5-47e4-9eb5, c192 j-574540d4-cec5-4472-9de8, c224 j-85495e4f-74ed-43b2-8d9e,
c256 j-90ba3b9c-b237-436b-a656. No accuracy run (synthetic acceptance 2.32).

## c2: the megakernel (this branch's first commit) + the c1 flags

`DECODE_CONTEXT_PARALLEL_SIZE_DECODE=2`, `DECODE_QWEN4_EXP_MAX_POOL_BYTES=48318382080`, and
`VLLM_TPU_MEGAKERNEL=whole=1,wholeshard=1,wholeunion=1,wholeloop=1,wholerows=32,wholecap=262144`,
`VLLM_TPU_MK_VMEM_CUTS=io,chunk,acc,lean,vmem64`, `VLLM_TPU_MK_TILES=rel_fetch=true`, `VLLM_TPU_MK_BUCKET_TUNE=1`,
`VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE=1`, `VLLM_TPU_MK_HC_FP8=1` (FP8 HC weights: numerics change), `VLLM_TPU_MK_PLE_TRIM=0`,
`VLLM_TPU_MK_QSEL_ASYNC=1`, `VLLM_TPU_MK_SCATTER_DEEP=1`, `VLLM_TPU_MK_STACKED_QSA_WRITES=1`,
`VLLM_TPU_MK_WHOLE_BATCHED_INPUTS=1`, `VLLM_TPU_MK_WHOLE_L45=1`, `VLLM_TPU_MK_WHOLE_L47=1`, `VLLM_TPU_QSA_DCP_MERGE=`.

Microbench step: -12.3% (c4), -19.8% (c32), worse than c1 from 16 running requests up (DCP2).
Served: NOT deployable as is. Routed PD transfer (`QWEN4_EXP_PD_ROUTED_DST`) needs decode-local prefixes on
2048-token boundaries, but DCP2 x block 512 gives 1024-token entries (`pd_route.skip_rows` raises). With routing off,
TPOT is -5.5/-6.4/-5.4% at c4/c8/c32 but TTFT +13..18%, and the prefill fails at c128+ with "Conflicting replay after
terminal Stage-3 producer registration" (tpu_connector.py). Needs a KV-connector change.

## VLLM_TPU_QSA_UNION_NOROUTE (this branch's second commit)

Diagnostic switch; within noise of the default. Not part of c1.
