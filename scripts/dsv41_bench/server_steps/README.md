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

## Recorded validation: 2026-10-12

Production commit `7ee60b940`, harness `bd0fe4db6`; all 366 tracked production
files on the benchmark host matched the submitted source. The later formal-window
compiler-log guard was also applied offline to these records. The measurements below used the original four buckets
`[16,128,256,2048]`, before KV budget rounding.

The baseline already includes the host metadata snapshot, packed RoPE selection
and gather lookahead. The only production difference in this series is the
static shared-prefill indexer gate. Both cold cases use independent empty
compilation caches; reuse uses the candidate's unchanged cache directory.
There is one startup observation per case, so small differences are not a
statistical speedup claim. Weight-file/system caches are not controlled.

| Case | Launch to healthy (s) | Max engine init (s) | Max logged compilation (s) | FX compile calls | Artifact load attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before static indexer, cold | 1633.56 | 1043.25 | 929.87 | 80 | 0 |
| Integrated candidate, cold | 1689.14 | 1032.43 | 921.18 | 80 | 0 |
| Same candidate, cache reuse | 1663.41 | 949.42 | 911.46 | 64 | 16 |

Cache reuse was partial: 16 artifact load attempts, but 64 new FX compile calls.
Auto-sized KV token capacities changed across restarts (for example rank 0:
6,667,507 to 6,666,237 tokens), changing the KV-block-count cache key for serving
graphs. The initial profiling graphs were reusable. Stable serving-graph reuse
needs a stable large-KV allocation plan; this run preserves the existing
shape-sensitive keys.

| Case | Middle tokens/s, node | Whole encoder window, node | Middle / logical device | Middle / physical chip |
| --- | ---: | ---: | ---: | ---: |
| Before static indexer, cold | 108,399 | 87,949 | 13,550 | 27,100 |
| Integrated candidate, cold | 108,659 | 88,410 | 13,582 | 27,165 |
| Same candidate, cache reuse | 108,397 | 87,845 | 13,550 | 27,099 |

Each case passed five formal rounds and 35 middle-step samples: 524288 cache
hits per round, zero preemptions and no compiler events in formal windows.
All 40 one-token candidate responses matched the baseline, for both starts.
KV remained auto-sized at about 32.4 GiB per rank; this is not a capacity stress
test. The separate candidate profile contains six complete rank-0 encoder
graphs with 20 MoE calls each; truncated graphs are excluded. Median device
time is **160.690 ms**, versus **160.694 ms** in the earlier lookahead run.
Normalized device-equivalent rate is 101761 new tokens/s at 16352 tokens per
step; this is not client throughput or an independently qualified node capacity.

The static gate's decoder-2048 compile median across ranks fell from
145.400 to 131.275 seconds, but total logged compilation fell only 0.9%, and
cold launch-to-health increased 3.4%. Middle host cadence increased 0.24% and
whole-window cadence 0.52%; these small differences do not establish a new
end-to-end speedup. The isolated indexer benefit does not apply uniformly to
all encoder/decoder compilation work.

Earlier matched large-KV ablations used the same frozen workload:

| Change accumulated | Middle host tokens/s | Whole encoder-window tokens/s |
| --- | ---: | ---: |
| Initial large-KV baseline | 58,989 | 55,941 |
| Host metadata snapshot | 88,529 | 72,658 |
| Plus packed RoPE selection | 106,563 | 86,641 |
| Plus gather lookahead | 108,452 | 88,325 |

Those are separate mechanism-isolation runs, not extra multipliers on this
series. The historical 4-GiB result used different input texts and source and
is not an exact A/B. No full recovery of that historical device rate is claimed.

Pre-commit checks and focused CPU/TPU regressions pass. Full-dependency type
checking retains the same 61 diagnostics as the original PR head, with no new
diagnostics. Full TPU CI still requires a maintainer's `ready` label; long
accuracy, mixed-load and steady decode qualification remain outside this run.
