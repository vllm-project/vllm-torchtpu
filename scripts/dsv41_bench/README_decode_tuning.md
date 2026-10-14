# Decode request-tile tuning

`decode_indexer(..., request_tile=None)` selects a static entry from
`index_selection/tuned_block_sizes.py`. The lookup key includes device kind,
per-device request count, queries per request, heads, score capacity, page size,
key tile, top-k count and cache bank count. It does not read dynamic lengths
back to the host. Tables are calibrated with the default score/top-k options.
Pass an explicit `request_tile` when ablating other kernel parameters.

On a miss, the fallback is 16, or the largest divisor of the request count below
16. A warning includes the complete missing key. Explicit tiles must divide the
request count; the kernel does not pad requests. No runtime tuning or JSON file
loading occurs in the inference path.

Run the standalone benchmark in a TPU environment:

```bash
python scripts/dsv41_bench/bench_decode_overlap.py \
  --out /tmp/decode-tuning --requests 48 --queries 6 \
  --width 131072 --length 98816 --repeats 41 \
  --tune-request-tiles 8 12 16 24 48
```

This sweeps only `request_tile`, checks output equality, and times complete
score/top-k pipelines in rotating order after compilation and warmup. All other
kernel options retain their defaults. `metrics.jsonl` preserves individual
samples, runtime, seed and memory statistics. `tuned_request_tile.py.txt` contains
a suggested Python table entry. Confirm the leading tiles with another seed
(`--seed 519`) and shuffled pages (`--shuffled`) before installing an entry.
The script never edits the runtime table automatically.

To validate table dispatch against an explicit tile, use a variants file:

```json
{"explicit": {"request_tile": 16}, "configured": {}}
```

Pass it with `--variants variants.json`. An empty configuration uses the same
automatic lookup as the kernel caller. `--variants` and `--tune-request-tiles`
are mutually exclusive. The standalone harness bypasses serving registration
and vLLM logging; this does not validate serving integration.

The initial TPU7x entries target 48 requests × 6 queries, 32 heads, two cache
banks, page size 128 and key tile 4096: score capacity 65536 uses request tile
12; capacity 131072 uses 16. These were measured with JAX/JAXLIB 0.10.2 and
libtpu 0.0.47. They do not imply the same choice for different batch shapes or
TPU generations.

## Serving validation

The decode route requires canonical request-major rows, BF16 Q with 32 heads
and head dimension 128, k=512, and compatible packed-cache pages. It supports
candidate publication, including the last-visible-block rule. Candidate-filtered
indexers and unsupported or ragged/mixed batches retain the existing route.
Pages are padded through the compact page table; score rows are grouped by the
tuned request tile. Scoring a later group can overlap SparseCore selection of
an earlier group through the existing pipeline helper.

A matched 96K/C384 DSpark K5 run on 2026-10-14 used two independent TPU7x hosts,
four chips/eight logical devices each, TP1/DP8/EP8, FP8 KV, automatic KV sizing
at 0.85 utilization and max model length 98816. Both versions included #1954
and #1958 with `[288,2048]` compilation buckets. Each ran one warmup, three
formal rounds and a separate synchronized six-step profile.

| Metric | Baseline | Grouped decode pipeline |
| --- | ---: | ---: |
| Accepted + bonus output tokens/s | 11693.09 | 12800.43 (+9.47%) |
| Host step median, ms | 130.992 | 119.817 (-8.53%) |
| Host step P90, ms | 131.738 | 120.836 |
| Accepted output / request / step | 4.0403 | 4.0455 |
| Valid full-DP intervals | 239 | 244 |
| Two main target graphs, summed device median, ms | 108.621 | 98.864 |
| Draft graph device median, ms | 4.220 | 4.225 |

Throughput divides total accepted+bonus tokens by the total common steady
window across three rounds. Step percentiles pool their valid intervals. Each
round completed 384 requests with 512 output tokens, zero preemptions, expected
prefix hits and no logged compilation. All summaries were recomputed from raw
worker observations. Per-rank steady shapes were 48 requests, 288 target rows
and 240 draft rows padded to 288. Eight-device profiles confirmed execution of
the grouped scorer; device timings exclude auxiliary graphs and host gaps.

Follow `server_steps/README.md` to prepare the frozen input and runtime, then
run each version with a separate empty cache and output directory:

```bash
python scripts/dsv41_bench/server_steps/run.py --scenario decode \
  --input /path/to/inputs.json --out /path/to/results \
  --cache-dir /path/to/empty-cache --cache-mode cold \
  --rounds 3 --startup-timeout 5400 --profile
```

The E2E runtime used torch 2.13.0+cpu, torch-tpu 0.1.1.dev20260924085621,
JAX/JAXLIB 0.10.2, libtpu 0.0.47 and runai-model-streamer 0.17.0, with vLLM
revision `e37e51dd246cf421c554e7d6f53e178e3fd29085`. These numbers describe
colocated serving steady decode, excluding preparation and drain; they are not
whole-request wall-clock throughput or a model-quality evaluation. Prefill and
mixed-batch performance were not remeasured for this decode-only change. The
profile does not quantify a physical TC/SC overlap percentage.
