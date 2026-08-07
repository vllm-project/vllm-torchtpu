# Profiling

`vllm-torchtpu` captures TPU traces with the PyTorch/XLA profiler. Every worker
process drives its own profiler session and can only trace the chips it owns, so
a whole-slice trace is assembled from one capture per worker (see
[`src/vllm_torchtpu/profiler_trace.py`](../src/vllm_torchtpu/profiler_trace.py)).
The output is XPlane protobuf (`*.xplane.pb`), readable in
[XProf](https://github.com/openxla/xprof) / TensorBoard.

There are three ways to profile a workload:

| Approach | Also called | Best for |
|---|---|---|
| [`examples/tpu_profiling.py`](#offline-profiling-with-examplestpu_profilingpy) | non-phased, on-demand profiling | Isolating a single shape — one prefill or one decode batch, offline |
| [Server-side capture](#server-side-capture-vllm-serve) | end-to-end (e2e) profiling | A real serving window under `vllm bench serve` or your own client |
| [Phased profiling](#phased-profiling) | phased profiling | One trace per inference phase from a single mixed benchmark run |

## Prerequisites

Install a viewer for the XPlane files. The traces themselves need nothing extra
at capture time:

```bash
pip install xprof            # or: pip install tensorboard-plugin-profile
```

## Offline profiling with `examples/tpu_profiling.py`

[`examples/tpu_profiling.py`](../examples/tpu_profiling.py) drives the offline
`LLM` API with a synthetic batch, warms up, and then captures a trace of
`--num-iters` generate calls. Use it when you want a clean trace of one shape
with no scheduler noise.

```bash
python3 examples/tpu_profiling.py --model <model> [OPTIONS]
```

Key arguments:

* `--model` — model name or path.
* `--input-len` — prompt tokens per request.
* `--output-len` — tokens generated per request.
* `--batch-size` — number of requests.
* `--num-iters-warmup` / `--num-iters` — warmup iterations (not traced) and
  profiled iterations. Warmup matters: the first iterations pay XLA compilation.
* `--profile-result-dir` — where the trace lands.
* All standard vLLM `EngineArgs` are accepted (`--tensor-parallel-size`,
  `--max-model-len`, `--enable-expert-parallel`, ...).

**Profile a prefill.** Long prompt, one output token, small batch:

```bash
python3 examples/tpu_profiling.py \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --input-len 1024 \
  --output-len 1 \
  --batch-size 1 \
  --profile-result-dir profiles/prefill_1k
```

**Profile a decode.** Single-token requests, large batch:

```bash
python3 examples/tpu_profiling.py \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --input-len 1 \
  --output-len 1 \
  --batch-size 256 \
  --profile-result-dir profiles/decode_bs256
```

[`examples/run_profiling_suite.sh`](../examples/run_profiling_suite.sh) sweeps a
matrix of these; uncomment the experiments you want and run it.

## Server-side capture (`vllm serve`)

Start the server with a trace directory, then bracket the window you care about
with the profile endpoints. `--profiler-config.profiler torch` is required —
vLLM's `ProfilerConfig` validator rejects a trace directory without it.

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir /tmp/vllm_profile \
  --profiler-config.ignore_frontend true
```

```bash
curl -X POST http://localhost:8000/start_profile
#  ... send the traffic you want to trace ...
curl -X POST http://localhost:8000/stop_profile
```

`vllm bench serve --profile` fires those two endpoints around its main run for
you.

Those are two separate steps, and the rest of this page keeps them distinct:

* **Configured** — the server was started with a profiler kind and a trace
  directory, so the profile endpoints will work. Nothing is being recorded yet.
* **Armed** — `/start_profile` has been called, so the profiler is now watching
  and will write a trace. `/stop_profile` disarms it.

A server can be configured but never armed, which is the most common reason a
run finishes having written nothing.

`--profiler-config.ignore_frontend true` skips the AsyncLLM frontend's CPU
profiler; only the TPU XPlane trace is of interest here, and the CPU trace is
pure overhead.

Note that the standard TPU capture always spans the whole window between
`/start_profile` and `/stop_profile`: the `--profiler-config.delay_iterations`
and `--profiler-config.max_iterations` limits are honored only by the
[phased profiler](#phased-profiling), not by this mode.

> [!WARNING]
> Profiling is not free. In benchmark runs, `--profile` costs roughly +11%
> TTFT and −4% throughput, so never compare profile-on numbers against a
> non-profile baseline.

### Through the benchmark harness

[`scripts/vllm/benchmarking/run_benchmarks.sh`](../scripts/vllm/benchmarking/run_benchmarks.sh)
wires all of this up. Setting `CAPTURE_PROFILE=1` adds the profiler flags to the
server and `--profile` to the bench client, writing traces to
`<results-dir>/profile`:

```bash
CAPTURE_PROFILE=1 ./scripts/vllm/benchmarking/run_benchmarks.sh \
  --config qwen3.5-35b-fp8-tp4-ep
```

`CAPTURE_PROFILE=1` is *inferred* whenever `EXTRA_SERVE_ARGS` contains
`--profiler-config.torch_profiler_dir`, which then also chooses the destination.
Without that inference the server would be configured but never armed — the
client would not send `--profile`, so nothing would call `/start_profile` and
the run would silently produce no traces. Set `CAPTURE_PROFILE=0` to opt back
out.

Warmup passes are never profiled. Prefer a single ISL/OSL case so the trace maps
to one workload shape.

## Phased profiling

A serving benchmark mixes prefill and decode work in a single stream, so one
continuous trace blends both. The phased profiler instead classifies every
engine step by its batch composition and captures a **separate, bounded trace
the first time each phase appears**.

Whichever way you run it, two things have to be true: the server process must
see `USE_PHASED_PROFILER=true` and a trace directory, and something must call
`/start_profile` and `/stop_profile` around the traffic. The difference between
the two options below is only *who* makes those two calls.

### Option 1: through the benchmark harness

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
([as above](#through-the-benchmark-harness)) and passes `--profile` to
`vllm bench serve`, which POSTs to the two endpoints around the measured run.
Warmup passes are excluded.

### Option 2: against your own `vllm serve`

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

Two flags in that command are easy to miss. `--attention-backend CUSTOM` selects
the batched-RPA Pallas kernel; without it vLLM falls back to `FLASH_ATTN` with
`block_size=16`, so you would be profiling a backend you do not serve with.
`--profiler-config.ignore_frontend true` suppresses the AsyncLLM CPU profiler,
which otherwise ignores the iteration limits and warns about the added overhead.

Clear the trace directory between runs. Phases accumulate into per-phase
subdirectories, and traces from an earlier run are easy to mistake for the
current one.

### Phases

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
[`get_batch_composition_stats`](../src/vllm_torchtpu/runner/utils.py):

* A request with no computed tokens yet is prefill.
* An ongoing request scheduled for more than one token is chunked prefill.
* An ongoing request scheduled for one token is decode.
* Speculative-decode draft tokens count as decode, not chunked prefill — they
  are draft tokens being verified.

Each phase is captured **once per profiling session** — the window between
`/start_profile` and `/stop_profile` — for `max_iterations` steps, and then not
again within that session. A new session (for example, the next benchmark case
run with `--profile`) starts with fresh phase tracking and captures each phase
again. A phase that the workload never produces simply yields no subdirectory.

### Configuration

| Setting | Purpose |
|---|---|
| `USE_PHASED_PROFILER=true` | Env var. Selects the phased profiler over the standard continuous one. This is the only on/off switch. |
| `--profiler-config.profiler torch` | Required alongside the trace directory; `ProfilerConfig` rejects the directory without it. |
| `--profiler-config.torch_profiler_dir <dir>` | Root directory for traces. The same field the standard profiler uses — the env var only decides which profiler consumes it. |
| `--profiler-config.max_iterations N` | Engine steps captured per phase. Upstream's default of `0` means "no limit", which is meaningless here, so the phased profiler falls back to **15** when it is unset. |
| `--profiler-config.delay_iterations N` | Number of `decode_heavy` steps to skip before starting that phase's capture. Useful for stepping past compilation and warmup. Default `0`. Only affects `decode_heavy`. |
| `--additional-config '{"phased_profiler_decode_only_kv_len_threshold": N}'` | Skip `decode_only` steps until the batch's minimum KV length reaches `N`, so the trace shows steady-state context lengths rather than the first few decodes. Default `-1` (disabled). This knob has no `profiler_config` equivalent, so it stays in `additional_config`; it is excluded from the TPU compile cache key, so setting it does not force a recompile. |

`EXTRA_SERVE_ARGS` is an environment variable read by `run_benchmarks.sh` and
appended to the `vllm serve` command line. Each per-config `.sh` merges an
exported value ahead of its own arguments
(`${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }...`), so profiling flags can be
injected without editing the config file.

## Output layout

### Phased runs

Traces are grouped by phase, and within a phase all ranks merge into one
timestamped run directory:

```text
/tmp/vllm_phased_profile/
├── prefill_only/
│   ├── plugins/profile/2026_08_03_10_15_42/
│   │   ├── rank0_t1v-n-abccf5c4-w-0.xplane.pb
│   │   ├── rank1_t1v-n-abccf5c4-w-0.xplane.pb
│   │   ├── rank2_t1v-n-abccf5c4-w-0.xplane.pb
│   │   └── rank3_t1v-n-abccf5c4-w-0.xplane.pb
│   ├── rank_0/batch_composition_stats_<timestamp>.json
│   ├── rank_1/...
│   └── ...
├── prefill_heavy/
│   └── ...
└── decode_only/
    └── ...
```

At TP=4 that is 4 trace files per captured phase.

The `rank_<N>/` directories are the per-rank capture sandboxes. Traces are moved
out of them into the shared `plugins/profile/<timestamp>/` directory with a
`rank<N>_` filename prefix; the sandbox survives only because the batch
composition stats are left behind in it. Those JSON files record, for every
profiled step, the batch id, scheduled/prefill/decode token counts, request
count, minimum KV length, and the phase that was assigned — which is what lets
you tie a trace back to the batch that produced it.

### Standard (non-phased) runs

Same merge, without the phase level:

```text
<torch_profiler_dir>/
├── plugins/profile/<timestamp>/
│   ├── rank0_<host>.xplane.pb
│   └── ... (one .xplane.pb per rank)
├── rank_0/
│   └── <host>_<pid>.pt.trace.json.gz
├── rank_1/...
└── <host>_<pid>.async_llm.pt.trace.json.gz
```

### Why the rank prefix exists

XPlane files are named after the *host*, so ranks co-located on one host would
overwrite each other if they wrote into a shared directory. Each rank therefore
captures into its own sandbox first, and the prefix is added on the way out. The
prefix is applied when the slice-global world size (TP × PP × DP) is greater
than 1 — so it is present on every file of a run or on none of them.

The rank is deliberately slice-global rather than the TP×PP-scoped rank from
`parallel_config`: with DP > 1, every replica would otherwise call itself rank 0
and their traces would collide on merge.

> [!WARNING]
> **Multi-host slices.** Rank 0 publishes the canonical run timestamp through a
> marker file that the other ranks poll for (5 s timeout). On a multi-host
> slice, the trace directory must be on a filesystem that every rank shares with
> rank 0. If a rank never sees the marker it falls back to its own timestamp —
> no trace is lost, but that rank lands in a separate run directory. A
> `did not find rank 0's canonical-ts marker` warning in the logs is the signal.

## Viewing traces

### On the TPU VM

The simplest option is to leave the traces where they were written and serve
them from the VM:

```bash
xprof --logdir /tmp/vllm_phased_profile
```

If you are connected through VS Code's Remote-SSH, it detects the listening port
and forwards it automatically, so the URL it prints opens directly in your local
browser. Otherwise, forward that port yourself when connecting (6006 by
default):

```bash
ssh -L 6006:localhost:6006 <tpu-vm-host>
```

### On your local machine

To keep traces after the VM is torn down, or to share them, copy them down
first. Preserve the directory hierarchy — XProf reads the
`plugins/profile/<timestamp>/` layout, not just the individual file:

```bash
rsync -avzm \
  --include='*/' \
  --include='*.xplane.pb' \
  --include='*.pt.trace.json.gz' \
  --exclude='*' \
  <tpu-vm-host>:/tmp/vllm_phased_profile ~/traces/
```

```bash
xprof --logdir ~/traces/vllm_phased_profile
```

### Reading the result

Open the printed URL and pick a run from the dropdown. Each phase is a separate
run. The Trace Viewer shows kernel execution timelines, and the Memory Profile
and Op Profile views cover HBM usage and MXU utilization.

### Seeing every rank at once

Because all of a run's ranks are merged into one `plugins/profile/<timestamp>/`
directory, pointing XProf at the run directory loads them together — use the
host picker to switch between ranks, or view them side by side, without merging
anything by hand. This is the reason the capture goes through the rank-prefix
merge described above; a per-rank directory layout would force you to open each
rank as a separate session.

Note that by migrating to the native `torch.profiler`, vllm-torchtpu concurrently generates multiple trace formats:
* **`plugins/profile/<timestamp>/*xplane.pb`**: Raw TPU hardware traces  (viewable via TensorBoard's XProf plugin, so a viewer that asks you to choose a file type wants **XPlane**).
* **`rank_<N>/*.pt.trace.json.gz`**: Detailed PyTorch/ATen host traces + Kineto TPU traces in standard Chrome Trace Event Format. These can be dragged directly into native viewers like [Perfetto](https://ui.perfetto.dev) or Chrome Tracing UI.
* **`*.async_llm.*.pt.trace.json.gz`**: The vLLM AsyncLLM CPU frontend traces.

## Troubleshooting

### The server fails to start

Both messages below are assertions raised during startup validation, quoted
verbatim so you can match them against what the server printed.

**`Legacy additional_config['phased_profiling_dir'] is no longer supported`** —
the run is using a stale script written against the old configuration. Switch it
to `USE_PHASED_PROFILER=true` plus `--profiler-config.torch_profiler_dir`.

**`USE_PHASED_PROFILER is set but there is nowhere to write traces`** —
`USE_PHASED_PROFILER=true` was set, but `torch_profiler_dir` was left empty, so
the profiler has no destination to write to. Rather than start up and silently
record nothing, the server refuses. Supply the directory:
`--profiler-config.profiler torch --profiler-config.torch_profiler_dir <dir>`.

### The run produces no or unexpected traces

**The run finishes but the directory is empty** — the server was configured but
never armed. Something has to call `/start_profile`; through the benchmark
harness that means `CAPTURE_PROFILE` must resolve to `1`.

**A phase is missing** — the workload never produced a batch with that
composition. `prefill_only` and `decode_only` need batches that are entirely one
or the other, which a heavily overlapped serving run may never schedule. Check
the `batch_composition_stats_*.json` files to see which phases the scheduler
actually produced. `decode_only` additionally needs at least two concurrent
requests: the profiler skips one-token steps (a guard against capturing the
initial request), and a single decoding request schedules exactly one token per
step.

**Traces are split across two timestamped directories** — ranks disagreed on the
canonical timestamp; see the multi-host warning above.

**Per-expert MoE timings look skewed** — with expert parallelism, a single
device's profile reflects only the experts routed to it. `FORCE_MOE_RANDOM_ROUTING=1`
routes tokens to uniformly random experts so one device's profile represents the
whole EP mesh. It produces meaningless model output; use it for profiling only,
never for serving.
