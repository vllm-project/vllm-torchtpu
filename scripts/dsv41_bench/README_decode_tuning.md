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
indexers use the separate request-union route described below when supported;
unsupported or ragged/mixed batches retain the existing route.
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
run each version with a separate output directory. The current launcher disables
persistent caches and uses buckets `[288,2048]` with a 2048-token scheduler
input budget. Historical results above retain their recorded cache configuration:

```bash
python scripts/dsv41_bench/server_steps/run.py --scenario decode \
  --input /path/to/inputs.json --out /path/to/results \
  --rounds 3 --startup-timeout 5400 --profile
```

The E2E runtime used torch 2.13.0+cpu, torch-tpu 0.1.1.dev20260924085621,
JAX/JAXLIB 0.10.2, libtpu 0.0.47 and runai-model-streamer 0.17.0, with vLLM
revision `e37e51dd246cf421c554e7d6f53e178e3fd29085`. These numbers describe
colocated serving steady decode, excluding preparation and drain; they are not
whole-request wall-clock throughput or a model-quality evaluation. Prefill and
mixed-batch performance were not remeasured for this decode-only change. The
profile does not quantify a physical TC/SC overlap percentage.

## Candidate-filtered reindex

The candidate route shares request-union index keys within one decoder stage.
Its initial envelope is six queries per request, H32/D128, 2048 candidate
blocks of eight records, FP8 KV, TP1, and record widths above 49152 and at most
131072. It gathers at most 6144 union blocks per request and unpacks native
words once into BF16. Each reindex applies its own query membership and causal
mask before top512, then maps only selected slots to ascending logical IDs.
The source publication and index-K write versions invalidate cached keys;
entering a new stage resets all request-scoped entries. Any union overflow or
noncanonical active request mapping falls back without dropping candidates.

`prepare_candidate_cache(..., packed=True)` retains native int32 words instead
of allocating the separate BF16 unpack output. The scorer unpacks each key
tile in VMEM. Both representations store two bytes per key element; packed
mode reduces preparation temporaries rather than halving the persistent cache.
Serving currently uses BF16. Packed mode has a distinct tuning key and uses
the bounded default until calibrated.

The candidate pipeline uses the same whole-request prepare/score/select/write
helper as full-context decode, with an independent tuning table. On TPU7x,
48 requests, capacities 40960/49152 and key tile 4096, request tile 48 beat smaller
groups. This choice has one request group, so it does not claim cross-group
score/top-k overlap. Two-seed measurements used JAX/JAXLIB 0.10.2 and libtpu
0.0.47 with synthetic 70% shared candidate cores, shuffled native two-bank
pages, and four independent Q/weight pairs. In seed 825, median four-layer
latency including one union preparation was 8.504 ms, versus 23.675 ms for
the native scorer and production direct SparseCore top-k, including logical-ID
remapping and sorting. Replaying the highest-union rank from the 40K C384 serving
capture at capacity 49152 gave 9.920/9.832 ms versus 23.947/23.930 ms across
seeds 825/826. Queries, weights and physical pages were synthetic. These are
kernel measurements, not the E2E results in the preceding table.

The 40K serving capture had a maximum union of 5138 blocks, exceeding its
5120-block limit. In six profiled steps across eight devices, 12 of 48
rank-steps fell back. Serving therefore uses the measured 48K capacity;
overflow still falls back correctly without truncation.

Reproduce the candidate calibration separately from serving:

```bash
python scripts/dsv41_bench/bench_candidate_reindex.py \
  --out /tmp/candidate-tuning --seed 825 --repeats 41 --tiles 16 24 48
```

Pass `--candidate-file rank-0.npz --width 98816 --capacity 49152` to replay
captured candidate IDs with an explicit union capacity. Queries, weights and
physical page assignments remain synthetic; this is not a full model replay.
The benchmark rejects a capacity that would truncate any request's union.

Actual candidate cardinalities, fallback frequency, allocator capacity and
model throughput must be checked with the matched serving harness. Synthetic
candidate overlap is not a production routing guarantee.

### Matched candidate serving validation

A new C384 measurement on 2026-10-14 compared fixed #1972 revision
`37dd24362251e786ab1cbe37f1d4f192a0ada16a` with the request-union implementation
`d0dd8823b05370f6a8e37dab483e73b324ad1ebd`. It used the same runtime and decode
configuration above, separate empty compile caches, one warmup, three
unprofiled rounds and an independent synchronized six-step profile.

| Metric | Fixed #1972 | 48K request-union cache |
| --- | ---: | ---: |
| Accepted + bonus output tokens/s | 12359.31 | 14107.18 (+14.14%) |
| Host step median, ms | 123.419 | 108.019 (-12.48%) |
| Host step P90, ms | 125.073 | 109.353 |
| Accepted output / request / step | 4.0433 | 4.0415 |
| Valid full-DP intervals | 244 | 244 |
| Two main target graphs, summed device median, ms | 99.016 | 84.555 |
| Draft graph device median, ms | 4.215 | 4.218 |
| Automatic KV capacity, tokens per rank | 7200555 | 7200555 |
| Observed formal-round HBM peak, GiB | 85.81 | 85.66 |

Candidate round rates were 14149.43, 14106.77 and 14065.12 tokens/s.
Every formal round completed all 384 requests with 512 output tokens each,
zero preemptions, expected prefix hits and no logged compilation. Raw worker
records reproduced the summaries. HBM was sampled externally approximately
every 2.8 seconds, rounded to 0.01 GiB; these observations can miss brief peaks.

All 48 profiled device-steps used one candidate preparation and four cached
scores, with no fallback. The four selection conditionals together had a
median duration of 4.737 ms; the gather/unpack conditional took 2.867 ms.
These intervals exclude preceding union metadata/address construction and
must not be treated as the complete reindex cost. Selected attention-KV reads
remained eight page-ID gathers and sixteen bank-gather launches per device
step, matching baseline. Async launch/done event durations are not payload
latency. The post-profile raw candidate snapshot had a maximum union of 5328
blocks, within the 6144-block capacity; one snapshot does not establish the
fallback rate of every formal step.

An optional benchmark-only allocator-layout diagnostic failed after the
six-step trace exports and raw candidate NPZ saves. The original harness
failure flag is retained. All eight exports and fresh NPZ files were checked
independently; the three earlier formal rounds were complete. Runtime
allocator-layout fields and lifetime allocation counters were not recovered.
This failure was in the external diagnostic extension, not the production
candidate route. Exact selected-ID checks passed against the existing TPU
route; full completion strings also vary between baseline rounds, so these
runs do not establish bitwise generation equivalence or model quality.

### Long-context functional coverage

The candidate-filtered route now accepts index widths up to 1048576 records.
Above 131072 records, metadata sorts the bounded candidate list by logical
block and query ownership, then deduplicates it into union IDs and membership
bits. Long contexts also allow one request group and mask inactive query rows.
Widths through 131072 retain the existing dense metadata and routing branches;
96K and 128K metadata StableHLO matched revision `0cf22f90` exactly. This is a
metadata graph check, not a new whole-model performance regression measurement.

The cache still holds 49152 keys (6144 blocks) and uses the existing overflow
fallback without truncation. The full-context publisher's grouped-decode
128K limit is unchanged. A post-profile 1M snapshot exceeded cache capacity on
two of eight ranks (maximum 7046 blocks); this snapshot does not measure the
fallback frequency across the run.

On 2026-10-15, the unchanged TP1/DP8/EP8 serving configuration, max sequences
48, compile buckets `[288, 2048]` and automatic KV budget 0.85 passed all eight
1M/C8 needle requests, each with 1048064 input tokens, with zero request errors.
Input beyond the 1048576-token model limit returned HTTP 400 as expected.
Sampled HBM peaked at 91.11 of 94.74 GiB; approximately 5-6 second sampling can
miss transient peaks. Automatic KV capacity remained 12687547 tokens per rank.
Exact selected-ID route tests also passed at 1M for one and four request groups,
including overflow, invalid candidates and reordered request mappings.

Three candidate-only warm-cache decode rounds completed eight requests each
with 512 output tokens and zero preemptions. Their accepted-plus-bonus rates
were 131.62, 140.35 and 140.68 tokens/s; pooled host-step median was 265.836 ms.
Inputs were eight adjacent windows from one LongBench-v2 code document, not
eight independent documents. Work was closed after functional validation:
the matched 1M baseline and new 96K regression series were not completed, so
these results establish neither 1M speedup nor a new 96K no-regression result.
C32 serving was not tested.
