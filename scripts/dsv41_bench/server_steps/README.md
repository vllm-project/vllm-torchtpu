# DP8 cached-extend and startup benchmark

This harness qualifies `encoder_extend` on one v7x host: four physical chips,
eight logical devices, TP1/DP8/EP8/PP1. It observes normal serving completion;
it adds no device synchronization and never calls `get_output()` early.

The workload uses eight distinct 98304-token cached prefixes, 16384 new input
tokens per request, and one greedy output token. Each DP engine admits one
request with a 2048-token budget. The model uses CED, FP8 KV, CPU Engram offload
and DSpark K5. KV is automatically sized with memory utilization 0.85, rather
than capped at 4 GiB. Compile buckets are `[128,256,2048]`. The server uses the model's default
context limit; it does not cap `max_model_len` to this benchmark's 114688-token
input. The 96K prefix and 16K extension are client workload sizes.
Automatic FP8 KV budgets round down to binary steps smaller than 1/32 of the
measured budget (1 GiB at 32–64 GiB), before the RoPE companion reserve.
This stabilizes shapes within a bin; crossing a bin boundary still recompiles.

This is a prefill benchmark; one-token output comparison does not qualify
long-generation accuracy or steady decode performance.

The harness has three scripts:

- `run.py`: launch configuration, cold/reuse cache lifecycle and cleanup.
- `bench_cached_extend.py`: input preparation, client, throughput and startup audit.
- `server_step_worker.py`: opt-in worker observations and shared workload definition.

Normal serving does not load this worker extension unless explicitly configured.

## Prepare inputs

Install the repository-pinned TPU runtime, `requests`, `prometheus-client` and
`transformers`. Supply a local DeepSeek-V4.1-Flash checkpoint and the pinned
LongBench-v2 dataset:

```bash
python scripts/dsv41_bench/server_steps/bench_cached_extend.py prepare \
  --dataset /path/to/longbench-v2.json \
  --tokenizer /models/DeepSeek-V4.1-Flash \
  --out /path/to/dsv41-code-96k-c128.json
```

The client verifies the frozen input SHA256:
`9ca4086c68c610b9205e1e46e1a51d3d18bb10495ed88d341c63b725ccca4064`.
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
  --input /path/to/dsv41-code-96k-c128.json \
  --out /path/to/results/candidate-cold \
  --cache-dir /path/to/persistent-cache/candidate \
  --cache-mode cold --rounds 5 --profile
```

`run.py` starts the server, waits for HTTP health readiness,
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
  --input /path/to/dsv41-code-96k-c128.json \
  --out /path/to/results/candidate-reuse \
  --cache-dir /path/to/persistent-cache/candidate \
  --cache-mode reuse --rounds 5
```

A populated cache does not prove a cache hit. Check load attempts, recompiles,
artifact warnings and timings. Do not drop cache-key fields to force reuse.
Keep `ARTIFACT_LOCAL_DIR` consistent across an A/B/restart series; the profile
is stored under its `encoder_extend/profile` directory. Do not enable compiler
dumps during timed startup, and never pool profiled and unprofiled rounds.

For an already running server with `server_step_worker.ServerStepWorker` loaded,
use `bench_cached_extend.py client --input INPUT --out OUTPUT --rounds 5`.
The server must use the same DP8 configuration as `run.py`.

## Timing and audit contract

- `startup.json` measures launch through HTTP health readiness. This includes
  model loading, engine initialization, compilation, cache work and warmup.
- `bench_cached_extend.py startup` extracts per-rank engine-init and reported-compilation
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
- Each formal round must complete all eight requests, hit 786432 cached
  tokens, have zero preemptions, and contain no compilation. Invalid shapes,
  missing ranks, interrupted windows and inconsistent counters fail closed.

Report node totals, values divided by eight logical devices, and values divided
by four physical chips explicitly. DSpark K5 is not a token-count multiplier.
CPU stage-call timestamps describe dispatch, not device execution duration.

## Reprocess and validate

```bash
python scripts/dsv41_bench/server_steps/bench_cached_extend.py summarize \
  /path/to/results/candidate-cold/fixed/round-1-steps.json \
  --out /path/to/recomputed-summary.json
python scripts/dsv41_bench/server_steps/bench_cached_extend.py startup \
  /path/to/results/candidate-cold/server.log \
  --out /path/to/startup-phases.json
python -m pytest tests/scripts/test_dsv41_cached_extend.py -q
```

Preserve input provenance, source revisions and diffs, runtime package versions,
responses, metrics, raw worker records and startup logs. `run.py` also writes
`startup-phases.json` automatically. Memory monitoring used during the original
investigation is not part of this benchmark entry point.

## Validated 96K scenario: 2026-10-13

The reference workload is 96K cached prefix + 16K new input, with the model's
1048576-token serving limit, automatic KV allocation, and three compile buckets.
The historical 64K workload used an 82944-token serving limit and is not a
matched performance baseline.

At the default 1M limit, the original full-query indexer materialized an
8-GiB FP32 score buffer before masking/top-k copies. Startup failed with a
31.40-GiB decoder temporary region. The bounded implementation completes
score/top-k selection in query tiles. The optimized implementation additionally
uses 65536 compressed key columns and 128 query rows when active lengths fit;
longer contexts retain the full-width, 32-query fallback. Candidate publication
and filtering keep their existing contracts. This changes internal working
shapes, not the serving context limit or KV budget.

### Reproduction

Use the preparation and cold-run commands above. For separate serving/client
processes (without the cold-cache startup audit):

```bash
# Terminal 1: serve with benchmark observations enabled.
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
export PYTHONPATH="$PWD/scripts/dsv41_bench/server_steps${PYTHONPATH:+:$PYTHONPATH}"
TP=1 DP=8 CHUNK=2048 MAX_SEQS=1 KV_BYTES=auto \
  bash scripts/dsv41_bench/serve_cached_prefill.sh \
  --worker-extension-cls server_step_worker.ServerStepWorker

# Terminal 2: after /health returns success.
python scripts/dsv41_bench/server_steps/bench_cached_extend.py client \
  --input /path/to/dsv41-code-96k-c128.json \
  --out /path/to/results/client --rounds 5
```

Measured source: bounded baseline `c88845921`, optimized `ebbc20148`, both
based on `c4f20a56e`. Both use one four-chip v7x host (eight logical devices),
TP1/DP8/EP8, BF16 execution with FP8 KV, CED, CPU Engram offload and DSpark K5.
Runtime: torch 2.13.0+cpu, torch-tpu 0.1.1.dev20260924085621,
JAX/JAXLIB 0.10.2, libtpu 0.0.47, and vLLM revision
`e37e51dd246cf421c554e7d6f53e178e3fd29085`.

### Results

| Metric | Bounded baseline | Context bucket + query tile 128 |
| --- | ---: | ---: |
| Middle host-observed tokens/s | 64264.96 | 84609.21 (+31.7%) |
| Middle host interval median (ms) | 287.81 | 218.90 |
| Complete encoder device median (ms) | 283.76 | 213.74 (-24.7%) |
| Device-equivalent new-input tokens/s | 57626.02 | 76503.70 |
| All observed encoder steps, tokens/s | 50320.88 | 64348.47 |
| Cold HTTP readiness (s) | 1253.06 | 1463.08 |
| Maximum engine initialization (s) | 896.11 | 905.50 |
| Maximum logged compilation (s) | 784.22 | 794.28 |
| Encoder / 2048 temporary HBM (GiB) | 5.33 | 5.34 |
| Decoder / 2048 temporary HBM (GiB) | 5.47 | 5.47 |

Each run has five unprofiled rounds and 35 middle chunks. Each round completes
all eight requests, records 786432 prefix hits and zero preemptions, and passes
the no-compilation audit. Host cadence is not client request throughput.
The separate rank-0 profiles contain four baseline and five optimized complete
encoder chunks, with 20 MoE calls each; capture-boundary partial chunks are
excluded. Each full chunk accounts for 16352 new input tokens across the node.
Startup and compilation scopes overlap and must not be summed. Startup is one
observation per version; the larger total startup increase is outside the logged
compilation scope. Compiled temporary HBM is not total process HBM.

### Validation limits

The related attention kernel module passes 55 CPU tests, including query-tile,
bank, bucket-boundary and long-context fallback cases. Full serving completed
with the production context limit and automatic KV allocation.

Numerical equivalence is not established: a TPU microtest at compressed length
54269 found 138 differing selected-index positions out of 1048576 (0.0132%).
This difference was not investigated in this performance-focused run. The
benchmark's single output token does not establish model accuracy. Long-context
TPU fallback accuracy, TP>1, steady decode and mixed workloads remain unqualified;
no performance or accuracy claim is made for those modes.

## Steady decode (C384)

`bench_decode.py` measures steady decode on the same single-host TP1/DP8/EP8
configuration, with DSpark K5, 40 layers and SWA128. Prepare a dedicated input file with
`bench_cached_extend.py prepare --scenario decode`; its SHA256 is recorded.

The workload uses 384 distinct 98,304-token prompts and 512 output tokens each.

Prepare inputs for this workload, then use the shared launcher:

```bash
python scripts/dsv41_bench/server_steps/bench_cached_extend.py prepare \
  --scenario decode --dataset /path/to/longbench-v2.json \
  --tokenizer /models/DeepSeek-V4.1-Flash --out /data/dsv41-inputs.json
```

```bash
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
python scripts/dsv41_bench/server_steps/run.py --scenario decode \
  --input /data/dsv41-inputs.json --out /results/decode \
  --cache-dir /cache/dsv41-decode --cache-mode cold
```

This compiles buckets `[288,2048]`, omitting 384 to reduce precompile work.
Steady target verification remains exactly 48 requests x (K + 1) = 288 rows.
Draft inputs (240 rows) still pad to 288; chunked prefill uses 2048.
Replay/mixed target batches above 288 rows now use 2048. These transitional
batches are excluded from the steady decode metric, but can affect preparation
time and window availability. Startup savings have not yet been measured;
the historical results below retain their original bucket configurations.
The larger prefill budget admits the 48 requests per rank before early requests
finish: exact-prefix cache hits leave 256 tokens per request to compute.
With a 512-token budget, admission overlapped request draining and produced no
common full-DP steady window in the C384 workload. It keeps
automatic KV sizing, disables both Tier2 caches and persists the JAX/vLLM/Tier3
caches. Startup readiness, phase timings, compile/load counts and package/source
identities are saved. Future runs may use `--cache-mode reuse` with that directory;
the launcher rejects mismatched source/runtime/model/scenario identities. A
populated cache is not proof of a hit; inspect `startup-phases.json` and logs.
Do not reuse earlier cache directories without an identity file.

For manual operation, start a dedicated server (terminal 1):

```bash
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
export PYTHONPATH="$PWD/scripts/dsv41_bench/server_steps${PYTHONPATH:+:$PYTHONPATH}"
TP=1 DP=8 CHUNK=2048 MAX_SEQS=48 MAX_MODEL_LEN=98816 KV_BYTES=auto \
  bash scripts/dsv41_bench/serve_cached_prefill.sh \
  --worker-extension-cls server_step_worker.ServerStepWorker \
  --compilation-config '{"compile_sizes":[288,2048]}' \
  2>&1 | tee /results/decode-server.log
```

After readiness, run the client in terminal 2 with a fresh output directory:

```bash
python scripts/dsv41_bench/server_steps/bench_decode.py client \
  --input /data/dsv41-inputs.json --out /results/decode \
  --server-log /results/decode-server.log
```

The default is one warmup followed by **one** formal window. `--rounds 0` runs
warmup only; `--single-window` measures an already warmed server without another
warmup. `--reuse-prefix` additionally skips priming and requires the same inputs
to be cached. Formal windows reuse prefixes primed for warmup. The client rejects
incomplete responses, missing ranks, unexpected prefix hits, preemption and
logged compilation in formal windows. Compilation
log checks cannot detect silent runtime compilation; device attribution requires
a separate profile. Run on an otherwise idle server.

The recorder observes native output consumption without additional device waits.
The summary retains only consecutive full-C decode observations: 48 requests,
288 target rows and 240 draft rows per rank, with five scheduled drafts/request.
Mixed phases, membership changes, gaps and drain observations split/exclude
windows. At least eight intervals in a segment of ten observations are required.
Node cadence uses the latest rank observation, never the sum of rank times.
Throughput counts accepted output events, **including bonus tokens**, within a
common host-clock window; do not multiply it by K. Host cadence is distinct from
isolated target/draft device duration. Contexts derived from accepted outputs
are reported separately from asynchronous scheduler metadata.

Reprocess saved observations without running the model:

```bash
python scripts/dsv41_bench/server_steps/bench_decode.py summarize \
  /results/decode/round-1-steps.json --scenario decode \
  --out /results/decode/reprocessed.json
```

CPU regression tests cover rank alignment, window boundaries, output accounting
and native output capture in `tests/scripts/test_dsv41_decode_*.py`.

Use `run.py --scenario decode --profile` to run one warmup and one formal
benchmark window, then a separate synchronized six-step profile on the same
server. Profile output is written locally and XPlane files are copied after
shutdown. Timing summaries exclude profiling. All eight ranks capture at the
same DP boundaries; validate XPlane coverage instead of the optional Chrome JSON.

### Validated 96k/C384 result (indexer tiles + 288 bucket)

Same host and recipe, epic/dsv41 `c7f2aea` plus the request-row indexer tiles,
the decode-shaped query-tile bypass and the `[288,384,2048]` buckets; one formal
window after warmup:

| Metric | `c7f2aea` | This change |
| --- | ---: | ---: |
| Steady host-observation step median | 309.19 ms | 139.93 ms |
| Steady step P90 (linear interpolation) | 309.92 ms | 141.49 ms |
| Accepted + bonus output throughput | 4,954.18 tok/s | 10,885.18 tok/s |
| Valid full-DP step intervals | 79 | 83 |
| Accepted outputs / request / step | 4.034 | 4.030 |

### Earlier validated 96k/C384 result (#1890, buckets `[384,2048]`)

On four physical v7x chips (eight logical devices), TP1/DP8/EP8, DSpark K5,
40 layers and SWA128, one formal window after warmup produced:

| Metric | Result |
| --- | ---: |
| Steady host-observation step median | 210.4713 ms |
| Steady step P90 (linear interpolation) | 212.2321 ms |
| Accepted + bonus output throughput | 7,291.18 tok/s |
| Valid full-DP step intervals | 80 |
| Common throughput window | 16.86161153 s |
| Accepted + bonus tokens in that window | 122,941 |

Each rank had 48 requests, 288 target rows and 240 draft rows. The formal
window had zero preemptions and no logged compilation. Throughput already
includes accepted drafts and bonus tokens; do not multiply it by K.

A separate synchronized capture recorded six main-model and six draft-module
executions on each of eight device planes. The maximum device span was
1.299657 seconds; all ranks captured ticks 32 through 37. Raw XPlane validation
found no warnings or errors. Chrome conversion's event cap does not establish
raw XPlane truncation. Full layer-by-layer coverage and model quality remain
unqualified; the profile lacks embedded HLO protos for static-memory analysis.
