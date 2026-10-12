# DP8 cached-extend and startup benchmark

This harness qualifies `encoder_extend` on one v7x host: four physical chips,
eight logical devices, TP1/DP8/EP8/PP1. It observes normal serving completion;
it adds no device synchronization and never calls `get_output()` early.

The workload uses eight distinct 65536-token cached prefixes, 16384 new input
tokens per request, and one greedy output token. Each DP engine admits one
request with a 2048-token budget. The model uses CED, FP8 KV, CPU Engram offload
and DSpark K5. KV is automatically sized with memory utilization 0.85, rather
than capped at 4 GiB. Compile buckets are `[128,256,2048]`.
Automatic FP8 KV budgets round down to binary steps smaller than 1/32 of the
measured budget (1 GiB at 32–64 GiB), before the RoPE companion reserve.
This stabilizes shapes within a bin; crossing a bin boundary still recompiles.

This is a prefill benchmark; one-token output comparison does not qualify
long-generation accuracy or steady decode performance.

## Prepare inputs

Install the repository-pinned TPU runtime, `requests`, `prometheus-client` and
`transformers`. Supply a local DeepSeek-V4.1-Flash checkpoint and the pinned
LongBench-v2 dataset:

```bash
python scripts/dsv41_bench/server_steps/prepare_inputs.py \
  --dataset /path/to/longbench-v2.json \
  --tokenizer /models/DeepSeek-V4.1-Flash \
  --out /path/to/dsv41-code-c128.json
```

The client verifies the frozen input SHA256:
`865b183381ae43367214d9dd0870f94c578b3d581e613ddc7872d1e48f391a11`.
The first eight deterministic code windows are used for this scenario. They
have distinct prefixes, but come from adjacent windows of the same document;
they are not eight unrelated documents. Prime requests branch immediately
after the prefix. Prime, warmup and generated tokens are excluded from the
new-input token count.

The corpus methodology follows SGL-JAX DSA's `bench_dsa_cache_hit.py`:
freeze natural-code inputs and verify actual cache behavior. This implementation
uses vLLM worker-extension RPCs and DeepSeek tokenizer IDs.

## Run cold startup and serving

Run on an isolated benchmark server; the harness uses development endpoints
for pause/resume, prefix reset and worker RPC. No other traffic may run on the
server. Reserve port 8000 and use fresh output directories.

```bash
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
export ARTIFACT_LOCAL_DIR=/path/to/artifacts
python scripts/dsv41_bench/server_steps/run.py \
  --input /path/to/dsv41-code-c128.json \
  --out /path/to/results/candidate-cold \
  --cache-dir /path/to/persistent-cache/candidate \
  --cache-mode cold --rounds 5 --profile
```

`run.py` starts the memory monitor and server, waits for HTTP health readiness,
runs a separate warmup and five formal rounds, optionally collects a separate
profile wave, and stops its server process group. The startup timeout defaults
to 3000 seconds. Compiler events in formal serving windows fail the run.
A cold run rejects a nonempty cache directory. Keep the cache
on persistent storage, and use separate directories for baseline and candidate.
Both public and internal native Tier-2 shared-memory caches are disabled;
vLLM, JAX and native Tier-3 paths are recorded in the run manifest.

A cache-reuse restart uses the same code, model, hardware, runtime, compile
buckets and cache directory, with a new result directory:

```bash
python scripts/dsv41_bench/server_steps/run.py \
  --input /path/to/dsv41-code-c128.json \
  --out /path/to/results/candidate-reuse \
  --cache-dir /path/to/persistent-cache/candidate \
  --cache-mode reuse --rounds 5
```

A populated cache does not prove a cache hit. Check load attempts, recompiles,
artifact warnings and timings. Do not drop cache-key fields to force reuse.
Keep `ARTIFACT_LOCAL_DIR` consistent across an A/B/restart series; the profile
is stored under its `encoder_extend/profile` directory. Do not enable compiler
dumps during timed startup, and never pool profiled and unprofiled rounds.

For a manually managed server, use `serve.sh encoder_extend`, then invoke
`client.py --scenario encoder_extend --input INPUT --out OUTPUT --rounds 5`.

## Timing and audit contract

- `startup.json` measures launch through HTTP health readiness. This includes
  model loading, engine initialization, compilation, cache work and warmup.
- `summarize_startup.py` extracts per-rank engine-init and reported-compilation
  times from the server log. Parallel rank times are never summed. These logged
  durations are not pure XLA compilation time; serialization and cache I/O can
  be included. Repeated stage-warmup records retain their separate scopes.
- `encoder_middle` requires consecutive full-DP actual2044/padded2048 chunks,
  matching contexts, no SWA replay and no decoder stage. Its numerator is
  16352 new input tokens per global step. It covers encoder layers 0–19 plus
  downstream global KV projection/write. First and tail steps remain separate.
- Natural throughput divides useful work by consecutive maximum-rank output
  observation intervals. A fully discarded asynchronous output can return
  without waiting for device completion. Therefore host cadence alone is not
  device capacity: summaries retain `capacity_qualified=false` and
  `device_seconds=null`. Use a separate device profile for completed-graph
  time, and report the exact captured graph count and coverage.
- Also report aggregate encoder throughput including first and tail steps;
  an asynchronous boundary can shift work from middle to tail. Do not label
  middle cadence as client request throughput or isolated decode capacity.
- Each formal round must complete all eight requests, hit 524288 cached
  tokens, have zero preemptions, and contain no compilation. Invalid shapes,
  missing ranks, interrupted windows and inconsistent counters fail closed.

Report node totals, values divided by eight logical devices, and values divided
by four physical chips explicitly. DSpark K5 is not a token-count multiplier.
CPU stage-call timestamps describe dispatch, not device execution duration.

## Reprocess and validate

```bash
python scripts/dsv41_bench/server_steps/summarize.py \
  /path/to/results/candidate-cold/fixed/round-1-steps.json \
  --scenario encoder_extend --out /path/to/recomputed-summary.json
python scripts/dsv41_bench/server_steps/summarize_startup.py \
  /path/to/results/candidate-cold/server.log \
  --out /path/to/startup-phases.json
python -m pytest tests/scripts/test_dsv41_cached_extend.py -q
```

Preserve input provenance, source revisions and diffs, runtime package versions,
responses, metrics, raw worker records, startup logs and memory snapshots.
`monitor_memory.py` records host/cgroup/shared-memory and process RSS/HWM every
five seconds. Sampled peaks can miss transients; cgroup peaks can include earlier
runs in the same container. Zero cgroup OOM counters do not rule out node-pressure
eviction. Do not delete live communication buffers to free memory.

## Recorded validation: 2026-10-13

Production source `ce69313ec`; all 366 tracked production files on the benchmark
host matched the submitted source. The four-bucket baseline uses the same
rounded-KV source with only `serve.sh` restored to `[16,128,256,2048]`. Both cold
runs use independent empty caches; restart reuses the three-bucket cache.

| Case | Ready (s) | Engine init (s) | Logged compilation (s) | FX compiles | Artifact loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rounded KV, four buckets, cold | 1683.17 | 1023.10 | 910.47 | 80 | 0 |
| Rounded KV, three buckets, cold | 1479.24 | 825.46 | 713.98 | 64 | 0 |
| Same three-bucket cache, restart | 1045.10 | 194.60 | 156.70 | 0 | 64 |

Removing the 16-token bucket reduced serving main graphs from 64 to 48
(80 to 64 FX compiles including initial profiling), logged compilation by 21.6%,
and launch-to-ready by 12.1%. Restart loaded all 64 artifacts with zero new FX
compiles; cache inventory stayed at 162 files / 20,687,965,159 bytes. Logged
compilation includes cache replay/backend warmup and is not pure XLA time.
Engine initialization fell from 825.46 to 194.60 seconds on restart. Model
loading varies between runs, so the ready-time gain is smaller.

All 24 rank-start observations had the same capacity: 6,575,165 tokens per rank,
from a 32-GiB budget including RoPE. This retains the two-bank path. Bin crossings
still change shapes and trigger recompilation; cache-key checks remain intact.

| Case | Middle host tokens/s | Whole encoder-window tokens/s |
| --- | ---: | ---: |
| Rounded KV, four buckets, cold | 108,588 | 88,346 |
| Rounded KV, three buckets, cold | 107,272 | 87,447 |
| Same three-bucket cache, restart | 107,377 | 87,114 |

Each case passed five formal rounds / 35 middle samples, with 524288 prefix hits
per round, zero preemptions and no compiler events after readiness, including
prime/warmup. All 40 outputs from each three-bucket start matched the four-bucket
baseline; the baseline also matched the preceding unrounded-KV candidate.
These are single-token consistency checks, not long-generation accuracy tests.
Cgroup OOM counters stayed at zero.

A separate restart profile contains 6 complete rank-0 encoder graphs with
20 MoE calls each. Median device time is **160.851 ms**, versus the previous
160.690 ms (about 0.1% difference). At 16352 useful tokens per step, its normalized
device-equivalent rate is **101,659 tokens/s**. Profiled rounds are
excluded from host throughput. The roughly 1% lower host cadence does not prove
a device-speed change; there is only one startup/profile observation per case.
Neither metric independently qualifies serving capacity or steady decode.

Falcon experiment: `exp-hyxe77erhh`, results subtree `pr1848-startup`;
device analysis: `an-1fughtzu5e` (`topk-census.json`).
The new budget test and seven consolidated harness tests pass, as do pre-commit
checks. Full-dependency type checking retains the previous 61 diagnostics,
with no added or removed diagnostics; this is not a clean full type-check pass.
