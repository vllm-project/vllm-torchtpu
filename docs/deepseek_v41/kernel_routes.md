# DeepSeek V4.1 v2 kernel routing

Production routes follow input contracts and tuned tables. Deployment scripts
no longer select layout variants through environment variables.

## Scoring configuration

The indexer K cache uses upstream's FP8 record (128 E4M3 bytes plus one FP32
power-of-two scale, 132 bytes per token) in transposed native pages. Upstream's
`indexer_kv_dtype` must resolve to `fp8`; TPU has no MXFP4 or BF16 indexer.

`kernels/deepseek_v41/index_selection/fp8/tuned.py` owns the measured paged
MQA tiles and pipeline depths. `index_selection/tuned_block_sizes.py` keeps
the streaming top-k and candidate publication tables.

- Dense layers score the whole compressed context with the paged FP8 MQA
  logits kernel, then select with streaming or block-maximum top-k.
- Candidate consumers gather the published candidate blocks once per stage
  and score only those keys. Candidate membership is never discarded.

## Retired environment selectors

Remove these exports from deployment configurations; they no longer select a
production route:

| Retired variable suffix (`TPU_DEEPSEEK_V41_`) | Production policy |
| --- | --- |
| `ATTENTION` | CSA2 when its TP/dtype/head/window contract holds; native otherwise |
| `INDEX_KEY_LAYOUT` | Packed FP8 index cache |
| `PREFILL_INDEXER` | Packed FP8 paged MQA logits |
| `MOE_SCALE_ROW_PADDED` | Row-padded flat fused EP block scales |
| `PAGE_ROWS_ONEHOT` | Packed FP8 page addressing |
| `INDEX_SCORES_2D` | Packed FP8 paged MQA logits |
| `CANDIDATE_FLAT` | Packed FP8 candidate gather |
| `CANDIDATE_SKIP` | Packed FP8 candidate gather |
| `SP` | Upstream topology policy; no TP-only opt-in |
| `LOCAL_Q` | Actual token-parallel attention and supported weights |
| `ENGRAM_SPLIT` | Host Engram placement and unsharded encoder rows |

CED and host placement remain explicit choices in the
[serving policy](README.md#serving-configuration). Native attention for unsupported CSA2 inputs, pure SWA, two-bank cache
addressing, short-context all-indices selection and candidate overflow handling
are functional paths, not redundant implementations.

## Validation

Run kernel correctness tests before full serving:

```bash
python -m pytest tests/kernels/deepseek_v41/test_fp8_paged_mqa_logits.py \
  tests/kernels/deepseek_v41/test_fp8_indexer_select.py \
  tests/models/vllm/test_dsv41_v2_decode_indexer.py
```

Use `scripts/dsv41_bench/server_steps/run.py` for paired `encoder_extend` and
`decode` serving runs with identical frozen inputs, dependencies and hardware.
