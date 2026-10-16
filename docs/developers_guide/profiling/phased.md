# Phased profiling

A serving benchmark mixes prefill and decode work in a single stream, so one
continuous trace blends both. The phased profiler instead classifies every
engine step by its batch composition and captures a **separate, bounded trace
the first time each phase appears**.

Whichever way you run it, two things have to be true: the server process must
see `USE_PHASED_PROFILER=true` and a trace directory, and something must call
`/start_profile` and `/stop_profile` around the traffic. The difference between
the two options below is only *who* makes those two calls.

## Option 1: through the benchmark harness

`run_benchmarks.sh` makes the profile calls for you, so this is the whole
command — no `curl` needed:

```bash
rm -rf /tmp/vllm_phased_profile

USE_PHASED_PROFILER=true \
EXTRA_SERVE_ARGS='--profiler-config.profiler torch --profiler-config.torch_profiler_dir /tmp/vllm_phased_profile --profiler-config.max_iterations 5' \
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3.5-35b-fp8-tp4-ep
```

The harness makes both calls itself: it infers `CAPTURE_PROFILE=1` from the
trace directory in `EXTRA_SERVE_ARGS`
([as above](server.md#through-the-benchmark-harness)) and passes `--profile` to
`vllm bench serve` for the extra profile run, which POSTs to the two endpoints
around it.

## Option 2: against your own `vllm serve`

Start the server yourself, then bracket your own traffic with the profile
endpoints:

```bash
rm -rf /tmp/vllm_phased_profile

USE_PHASED_PROFILER=true vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --attention-backend CUSTOM \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir /tmp/vllm_phased_profile \
  --profiler-config.max_iterations 5 \
  --profiler-config.ignore_frontend true
```

Once the server is up:

```bash
curl -X POST http://localhost:8000/start_profile
#  ... send the traffic you want to trace ...
curl -X POST http://localhost:8000/stop_profile
```

`USE_PHASED_PROFILER` must be exported into the server's own environment — it is
read by the TPU workers, so setting it only in the shell that runs `curl` has no
effect.

The traffic you send between those two calls decides which phases you get. One
request at a time yields `prefill_only` and `decode_only`; the mixed phases need
concurrency. See
[a phase is missing](troubleshooting.md#the-run-produces-no-or-unexpected-traces) before running
this against an expensive model.

Two flags in that command are easy to miss. `--attention-backend CUSTOM` selects
the batched-RPA Pallas kernel; without it vLLM falls back to `FLASH_ATTN` with
`block_size=16`, so you would be profiling a backend you do not serve with.
`--profiler-config.ignore_frontend true` suppresses the AsyncLLM CPU profiler,
which otherwise ignores the iteration limits and warns about the added overhead.

Clear the trace directory between runs. Phases accumulate into per-phase
subdirectories, and traces from an earlier run are easy to mistake for the
current one.

## Phases

Each step's phase comes from `prefill_tokens / total_scheduled_tokens` for that
batch:

| Phase | Ratio | Meaning |
|---|---|---|
| `prefill_only` | `1.0` | Nothing but prompt ingestion in the batch |
| `prefill_heavy` | `>= 0.9` | Mostly prefill, with a few decodes riding along |
| `balanced` | `0.4` – `0.6` | Chunked prefill and decode mixed evenly |
| `decode_heavy` | `<= 0.2` | Steady-state decode with a little chunked prefill |
| `decode_only` | `0.0` | Pure token generation |

Ratios that fall in the gaps (`0.2` – `0.4` and `0.6` – `0.9`) are classified
`ambiguous` and are never captured — they are transitional batches that would
not represent either regime.

Token accounting, from
[`get_batch_composition_stats`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/runner/utils.py):

* A request with no computed tokens yet is prefill.
* An ongoing request scheduled for more than one token is chunked prefill.
* An ongoing request scheduled for one token is decode.
* Speculative-decode draft tokens count as decode, not chunked prefill — they
  are draft tokens being verified.

Each phase is captured **once per profiling session** — the window between
`/start_profile` and `/stop_profile` — for `max_iterations` steps, and then not
again within that session. A new session (for example, the next benchmark cell
run with `--profile`) starts with fresh phase tracking and captures each phase
again. A phase that the workload never produces simply yields no subdirectory.

## Configuration

| Setting | Purpose |
|---|---|
| `USE_PHASED_PROFILER=true` | Env var. Selects the phased profiler over the standard continuous one. This is the only on/off switch. |
| `--profiler-config.profiler torch` | Required alongside the trace directory; `ProfilerConfig` rejects the directory without it. |
| `--profiler-config.torch_profiler_dir <dir>` | Root directory for traces. The same field the standard profiler uses — the env var only decides which profiler consumes it. |
| `--profiler-config.max_iterations N` | Engine steps captured per phase. Upstream's default of `0` means "no limit", which is meaningless here, so the phased profiler falls back to **15** when it is unset. |
| `--profiler-config.delay_iterations N` | Number of `decode_heavy` steps to skip before starting that phase's capture. Useful for stepping past compilation and warmup. Default `0`. Only affects `decode_heavy`. |
| `--additional-config '{"phased_profiler_decode_only_kv_len_threshold": N, "phased_profiler_prefill_only_kv_len_threshold": M}'` | Skip `decode_only` / `prefill_only` steps until the batch's minimum KV length reaches the matching threshold. The decode one makes the trace show steady-state context lengths rather than the first few decodes; the prefill one traces a chunked prefill part-way through a long sequence rather than on its first chunk, where the KV length is still `0`. Both default to `-1` (disabled) and are excluded from the TPU compile cache key, so setting them does not force a recompile. |

`EXTRA_SERVE_ARGS` is an environment variable read by `run_benchmarks.sh` and
appended to the `vllm serve` command line. Each per-config `.sh` merges an
exported value ahead of its own arguments
(`${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }...`), so profiling flags can be
injected without editing the config file.
