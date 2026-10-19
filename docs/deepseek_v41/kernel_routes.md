# DeepSeek V4.1 v2 kernel routing

Production routes follow input contracts and tuned tables. Deployment scripts
no longer select layout variants through environment variables.

## Scoring configuration

`kernels/deepseek_v41/index_selection/tuned_block_sizes.py` owns the measured
request/key tiles. Exact matches take precedence; missing entries log the
lookup key and select bounded, aligned defaults within the scorer's input
contract; they are not tuned for every workload. Candidate table
misses use a 4096-key tile (reduced by GCD for alignment) and at most 16 whole
requests per score buffer, further bounded by the measured score-buffer volume.

- Decode uses the common paged score body. Aligned request groups write 2D
  scores directly. Other groups use the same body with a singleton output
  axis, preserving all query rows without padding the model batch.
- Candidate scoring uses one dot/ReLU/head-reduction/membership-mask body.
  The measured eight-request group retains its 2D output. The basic config
  scores one whole request per group; unaligned query counts use a singleton
  output axis. Empty union tails are skipped in production.
- Candidate cache reuse accepts uniform request-major batches with 1–8 queries
  per request, including one request. It is not specific to DSpark K5.
  Irregular request ownership, non-divisible batches and union overflow retain
  the exact generic paged indexer. Candidate membership is never discarded.
- Single-request prefill retains resident-query scoring and streaming top-k.
  The explicit `prefill_mode` kernel argument remains for standalone comparisons;
  serving uses the streaming default.

The measured C384 entries (48 requests × 6 queries on each logical device) are
unchanged, including packed halfplane candidate keys. This change does not
alter runner compilation buckets. In particular, it does not pad 24 target
rows to 48 to obtain a regular request-major view.

## Retired environment selectors

Remove these exports from deployment configurations; they no longer select a
production route:

| Retired variable suffix (`TPU_DEEPSEEK_V41_`) | Production policy |
| --- | --- |
| `ATTENTION` | CSA2 when its TP/dtype/head/window contract holds; native otherwise |
| `INDEX_KEY_LAYOUT` | Halfplane index cache |
| `PREFILL_INDEXER` | Streaming when eligible; generic paged indexer otherwise |
| `MOE_SCALE_ROW_PADDED` | Row-padded flat fused EP block scales |
| `PAGE_ROWS_ONEHOT` | Lane-dense indices when page shape permits |
| `INDEX_SCORES_2D` | Derived from request/query alignment |
| `CANDIDATE_FLAT` | Eight-request measured group or one-request basic config |
| `CANDIDATE_SKIP` | Skip empty union tails |
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
python -m pytest tests/kernels/test_dsv41_decode_tuning.py \
  tests/kernels/test_dsv41_decode_index_scores.py \
  tests/models/vllm/test_dsv41_decode_indexer_route.py
python scripts/dsv41_bench/bench_kernel_routes.py \
  --baseline-root /workspace/baseline --out /tmp/route-ab
```

The standalone A/B uses the same device/runtime and alternating synchronized
samples after warmup. It checks exact final indices, random and tied scores,
measured C384 configurations and untuned request/query counts. It bypasses
serving registration; it does not replace model integration tests.

Use `scripts/dsv41_bench/server_steps/run.py` for paired `encoder_extend` and
`decode` serving runs with identical frozen inputs, dependencies and hardware.
