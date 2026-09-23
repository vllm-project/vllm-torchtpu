# Profiling

`vllm-torchtpu` captures TPU traces with the PyTorch/XLA profiler. Every worker
process drives its own profiler session and can only trace the chips it owns, so
a whole-slice trace is assembled from one capture per worker (see
[`src/vllm_torchtpu/profiler_trace.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/profiler_trace.py)).
Every mode writes XPlane protobuf (`*.xplane.pb`), readable in
[XProf](https://github.com/openxla/xprof) / TensorBoard; the two non-phased
modes also write Chrome-format JSON. See [Trace formats](#trace-formats) for
which mode produces what.

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

[`examples/tpu_profiling.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/examples/tpu_profiling.py) drives the offline
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

[`examples/run_profiling_suite.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/examples/run_profiling_suite.sh) sweeps a
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

### Dynamic profiling options (`profiler_kwargs` and `profile_prefix`)

Both the standard server-side profiler and the [phased profiler](#phased-profiling)
resolve per-session options through
[`src/vllm_torchtpu/tracing/options.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/tracing/options.py).
You can override standard tracer levels (`host_tracer_level`,
`device_tracer_level`, `python_tracer_level`) and pass advanced TPU
`experimental_options` (such as `tpu_trace_mode`,
`tpu_num_sparse_cores_to_trace`, `tpu_num_sparse_core_tiles_to_trace`, or
firmware event flags like `e2e_enable_fw_throttle_event`,
`e2e_enable_fw_power_level_event`, and `e2e_enable_fw_thermal_event`) dynamically
at `/start_profile` time without restarting `vllm serve`:

* **Online serving (`POST /start_profile`)** — pass a JSON payload (or query
  parameters). `profile_prefix` names the output subdirectory under
  `torch_profiler_dir`, and all other keys are forwarded via `profiler_kwargs`:

```bash
curl -X POST http://localhost:8000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "profile_prefix": "fw_events_run",
    "host_tracer_level": 3,
    "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
    "e2e_enable_fw_throttle_event": true,
    "e2e_enable_fw_power_level_event": true,
    "e2e_enable_fw_thermal_event": true
  }'
```

* **Offline Python API (`LLM.start_profile`)** — pass `profile_prefix` and
  `profiler_kwargs` directly:

```python
llm.start_profile(
    profile_prefix="fw_events_run",
    profiler_kwargs={
        "host_tracer_level": 3,
        "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
        "e2e_enable_fw_throttle_event": True,
        "e2e_enable_fw_power_level_event": True,
        "e2e_enable_fw_thermal_event": True,
    },
)
```

* **Legacy string format (`profile_prefix`)** — semicolon-delimited `key:value`
  pairs in `profile_prefix` (e.g.,
  `"host_tracer_level:3;e2e_enable_fw_throttle_event:true"`) remain supported
  and are merged with any options supplied in `profiler_kwargs` (with
  `profiler_kwargs` taking precedence on key collisions).

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

[`scripts/vllm/benchmarking/run_benchmarks.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/run_benchmarks.sh)
wires all of this up. Setting `CAPTURE_PROFILE=1` adds the profiler flags to the
server and, after each cell's measured run, runs that cell once more with
`--profile`, writing traces to `<results-dir>/profile`:

```bash
CAPTURE_PROFILE=1 ./scripts/vllm/benchmarking/run_benchmarks.sh \
  --config qwen3.5-35b-fp8-tp4-ep
```

Neither the warmup passes nor the measured run are profiled, so the numbers the
harness reports stay comparable to a baseline. The extra run's
`<cell>.profile.json` is ignored by the regression check and the results upload,
and `PROFILE_NUM_PROMPTS` sizes it — one full concurrency wave by default.

`CAPTURE_PROFILE=1` is *inferred* whenever `EXTRA_SERVE_ARGS` contains
`--profiler-config.torch_profiler_dir`, which then also chooses the destination.
Without that inference the server would be configured but never armed — the
client would not send `--profile`, so nothing would call `/start_profile` and
the run would silently produce no traces. Set `CAPTURE_PROFILE=0` to opt back
out.

Prefer a single ISL/OSL cell so the trace maps to one workload shape.

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
`vllm bench serve` for the extra profile run, which POSTs to the two endpoints
around it.

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

The traffic you send between those two calls decides which phases you get. One
request at a time yields `prefill_only` and `decode_only`; the mixed phases need
concurrency. See
[a phase is missing](#the-run-produces-no-or-unexpected-traces) before running
this against an expensive model.

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

### Configuration

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
│   ├── rank_0/
│   │   ├── batch_composition_stats_<timestamp>.json   (one per profiled step)
│   │   └── <host>_<pid>.pt.trace.json.gz
│   ├── rank_1/...
│   └── ...
├── prefill_heavy/
│   └── ...
└── decode_only/
    └── ...
```

At TP=4 that is 4 XPlane files and 4 Chrome-format traces per captured phase,
plus one `batch_composition_stats_*.json` per rank per profiled step — so a run
capturing three phases with `max_iterations=5` writes 12, 12, and 72 files
respectively.

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
`plugins/profile/<timestamp>/` layout, not just the individual file. This copies
both formats; the JSON pattern simply matches nothing on a phased run:

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

All of a run's **XPlane** files are merged into one
`plugins/profile/<timestamp>/` directory, so pointing XProf at the run directory
loads them together — use the host picker to switch between ranks, or view them
side by side, without merging anything by hand. This is the reason the capture
goes through the rank-prefix merge described above; a per-rank directory layout
would force you to open each rank as a separate session.

The Chrome-format JSON traces do **not** go through that merge. They are already
uniquely named per worker, so they never collide, but each rank keeps its own
file under `rank_<N>/`. A viewer that opens a single file therefore shows one
rank at a time.

### Combining a run into a single file

Some consumers take exactly one file rather than a directory — Google-internal
XProf's offline upload, or a bug attachment. `merge_xplane` combines a session's
per-rank `.xplane.pb` captures into one XSpace:

```bash
python -m vllm_torchtpu.tools.merge_xplane /tmp/vllm_phased_profile
```

```text
Found 3 capture sessions
A plane is one timeline inside a capture: one per TPU chip, one per host,
plus a shared metadata block. Merging combines planes, not whole files.

  decode_only  2026_08_20_22_14_43
    input   4 captures (ranks 0-3), 205 MB
    planes  20 read -> 17 written (3 duplicate /host:metadata dropped)
    output  106 MB
    -> /tmp/vllm_phased_profile/decode_only/decode_only_2026_08_20_22_14_43.xplane.pb
  ...

Summary
  3 sessions merged, 1164 MB -> 572 MB, 50% smaller
  /tmp/vllm_phased_profile/decode_only/decode_only_2026_08_20_22_14_43.xplane.pb
  /tmp/vllm_phased_profile/prefill_heavy/prefill_heavy_2026_08_20_22_14_46.xplane.pb
  /tmp/vllm_phased_profile/prefill_only/prefill_only_2026_08_20_22_14_31.xplane.pb
```

Each rank's capture holds five planes on a v6e TP=4 run — its chip
(`/device:TPU:0`), `/host:metadata`, `/device:CUSTOM:Megascale Trace`,
`Task Environment`, and `/host:CPU`. Four captures therefore contribute 20
planes, and the merged file holds 17: the three redundant `/host:metadata`
copies are collapsed into one. The `Summary` block repeats every output path,
since the per-session detail scrolls away on a long run.

Each phase is an independent capture window anchored at its own zero, so a
phased run yields one merged file per phase rather than one for the whole run.

The merged file lands in the **phase directory** — the one holding `plugins/`.
That location follows the capture, not the command, so every way of pointing at
a phase produces the same file in the same place:

```bash
cd /tmp/vllm_phased_profile/decode_only
python -m vllm_torchtpu.tools.merge_xplane .                        # here
python -m vllm_torchtpu.tools.merge_xplane plugins/profile          # or here
python -m vllm_torchtpu.tools.merge_xplane plugins/profile/2026_*   # or here
# all three write ./decode_only_2026_08_20_22_14_43.xplane.pb
```

Re-running is safe: a merged file already sitting in a phase directory is
recognised as output rather than picked up as another capture to merge.

Pass `--dry-run` to see what would be written, `-o` to name a single output
(only when the inputs resolve to one session), or `-d` to choose the output
directory.

The output is roughly half the size of the inputs combined: every rank ships a
full copy of the `/host:metadata` plane, which carries the HLO protos, and the
merge keeps one. Merging on the TPU VM before copying traces down therefore
moves considerably less data than copying the per-rank files.

Two things worth knowing:

* **You do not need this for `xprof --logdir`.** OSS XProf already reads the
  whole session directory as one distributed session, as described above.
* **Merged files must not live in `plugins/profile/<timestamp>/`.** XProf builds
  its host list from the filenames it finds there, so a merged file sitting
  beside the per-rank ones appears as an extra host holding a second copy of
  every rank — a 4-rank capture reports 8 TPU cores and twice its real event
  count, with nothing to indicate anything is wrong. The phase directory is
  safe because XProf enumerates runs only at `*/plugins/profile/*` and ignores
  loose files above that; `-d` and `-o` are still refused if they point inside
  a session directory.

## Trace formats

What lands on disk depends on which profiler ran:

| Mode | `*.xplane.pb` | `*.pt.trace.json.gz` |
|---|---|---|
| [Offline script](#offline-profiling-with-examplestpu_profilingpy) | yes | yes |
| [Server-side capture](#server-side-capture-vllm-serve) | yes | yes |
| [Phased profiling](#phased-profiling) | yes | yes, one per phase per rank |

Every mode writes both formats. A phased run writes a *set* of each: one
`.xplane.pb` and one `.pt.trace.json.gz` per rank per captured phase. At TP=4
with three phases captured that is 12 of each.

* **`plugins/profile/<timestamp>/*.xplane.pb`** — raw TPU hardware traces.
  Viewable in XProf or TensorBoard's XProf plugin; a viewer that asks you to
  choose a file type wants **XPlane**.
* **`rank_<N>/*.pt.trace.json.gz`** — PyTorch/ATen host traces plus Kineto TPU
  traces, in standard Chrome Trace Event Format. Drag one straight into
  [Perfetto](https://ui.perfetto.dev) or Chrome Tracing.
* **`*.async_llm.*.pt.trace.json.gz`** — the vLLM AsyncLLM CPU frontend traces.
  These are not written when `--profiler-config.ignore_frontend true` is set,
  which the examples on this page do set.

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
or the other, which a heavily overlapped serving run may never schedule. The
mixed phases are the mirror image: `prefill_heavy`, `balanced`, and
`decode_heavy` each need prefill and decode tokens in the *same* batch, so a run
that only ever has one request in flight yields `prefill_only` and `decode_only`
and nothing else. Check the `batch_composition_stats_*.json` files to see which
phases the scheduler actually produced.

**Traces are split across two timestamped directories** — ranks disagreed on the
canonical timestamp; see the multi-host warning above.

**Per-expert MoE timings look skewed** — with expert parallelism, a single
device's profile reflects only the experts routed to it.
`VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random` routes tokens to uniformly
random experts so one device's profile represents the whole EP mesh. It produces
meaningless model output; use it for profiling only, never for serving. The
variable is upstream vLLM's routing-simulation selector, so any strategy
registered with `RoutingSimulator` can be named here.
