# Server-side capture (`vllm serve`)

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

## Dynamic profiling options (`profiler_kwargs` and `profile_prefix`)

Both the standard server-side profiler and the [phased profiler](phased.md)
resolve per-session options through
[`src/vllm_torchtpu/tracing/options.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/tracing/options.py).

You can override standard tracer levels, advanced/kernel/SparseCore TPU
`experimental_options`, and firmware event flags dynamically at `/start_profile`
time without restarting `vllm serve`:

| Category | Option (`profiler_kwargs` key) | Env Var Default | Values / Description |
|---|---|---|---|
| **Standard Tracer** | `host_tracer_level` | `HOST_TRACER_LEVEL` (`2`) | `0` (disabled), `1` (user `TraceMe`), `2` (XLA + user `TraceMe`), `3` (verbose low-level XLA). |
| **Standard Tracer** | `device_tracer_level` | `DEVICE_TRACER_LEVEL` (`1`) | `0` (disabled), `1` (enable TPU hardware execution tracing). |
| **Standard Tracer** | `python_tracer_level` | `PYTHON_TRACER_LEVEL` (`1`) | `0` (disabled), `1` (enable Python function call tracing). |
| **Kernel & SparseCore** | `tpu_trace_mode` | `TPU_TRACE_MODE` (`"TRACE_COMPUTE"`) | `"TRACE_ONLY_STEP"`, `"TRACE_ONLY_HOST"`, `"TRACE_COMPUTE"`, `"TRACE_COMPUTE_AND_SYNC"`. |
| **Kernel & SparseCore** | `tpu_num_sparse_cores_to_trace` | `TPU_NUM_SPARSE_CORES_TO_TRACE` (`1`) | Integer count of SparseCores to trace per chip. |
| **Kernel & SparseCore** | `tpu_num_sparse_core_tiles_to_trace` | `TPU_NUM_SPARSE_CORE_TILES_TO_TRACE` (`1`) | Integer count of SparseCore tiles (`spmem`) to trace. |
| **Firmware Events** | `e2e_enable_fw_throttle_event` | *(unset / `false`)* | `true` / `false` — capture firmware throttling events in XPlane. |
| **Firmware Events** | `e2e_enable_fw_power_level_event` | *(unset / `false`)* | `true` / `false` — capture firmware power-level transition events. |
| **Firmware Events** | `e2e_enable_fw_thermal_event` | *(unset / `false`)* | `true` / `false` — capture firmware thermal events. |

- **Online serving (`POST /start_profile`)** — pass a JSON body matching vLLM's
  `StartProfileRequest` schema (`profile_prefix`, `delay_iterations`,
  `max_iterations`, and `profiler_kwargs`). `profile_prefix` names the output
  subdirectory under `torch_profiler_dir`, and per-session TPU profiler flags
  are nested inside `profiler_kwargs`:

```bash
curl -X POST http://localhost:8000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "profile_prefix": "fw_events_run",
    "profiler_kwargs": {
      "host_tracer_level": 3,
      "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
      "tpu_num_sparse_cores_to_trace": 2,
      "tpu_num_sparse_core_tiles_to_trace": 2,
      "e2e_enable_fw_throttle_event": true,
      "e2e_enable_fw_power_level_event": true,
      "e2e_enable_fw_thermal_event": true
    }
  }'
```

- **Offline Python API (`LLM.start_profile`)** — pass `profile_prefix` and
  `profiler_kwargs` directly:

```python
llm.start_profile(
    profile_prefix="fw_events_run",
    profiler_kwargs={
        "host_tracer_level": 3,
        "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
        "tpu_num_sparse_cores_to_trace": 2,
        "tpu_num_sparse_core_tiles_to_trace": 2,
        "e2e_enable_fw_throttle_event": True,
        "e2e_enable_fw_power_level_event": True,
        "e2e_enable_fw_thermal_event": True,
    },
)
```

- **Legacy string format (`profile_prefix`)** — upstream vLLM validates that the
  top-level `profile_prefix` contains only `[a-zA-Z0-9._-]` characters to
  prevent path traversal, so per-session options on `/start_profile` and
  `LLM.start_profile` should be passed via `profiler_kwargs`. For backward
  compatibility with direct worker calls or `profiler_kwargs["profile_prefix"]`,
  semicolon-delimited `key:value` strings (e.g.,
  `"host_tracer_level:3;e2e_enable_fw_throttle_event:true"`) remain supported in
  `resolve_profile_dir_and_opts` and are merged with `profiler_kwargs` (with
  explicit `profiler_kwargs` keys taking precedence).

Those are two separate steps, and the rest of these guides keep them distinct:

- **Configured** — the server was started with a profiler kind and a trace
  directory, so the profile endpoints will work. Nothing is being recorded yet.
- **Armed** — `/start_profile` has been called, so the profiler is now watching
  and will write a trace. `/stop_profile` disarms it.

A server can be configured but never armed, which is the most common reason a
run finishes having written nothing.

`--profiler-config.ignore_frontend true` skips the AsyncLLM frontend's CPU
profiler; only the TPU XPlane trace is of interest here, and the CPU trace is
pure overhead.

Note that the standard TPU capture always spans the whole window between
`/start_profile` and `/stop_profile`: the `--profiler-config.delay_iterations`
and `--profiler-config.max_iterations` limits are honored only by the
[phased profiler](phased.md), not by this mode.

> [!WARNING]
> Profiling is not free. In benchmark runs, `--profile` costs roughly +11%
> TTFT and −4% throughput, so never compare profile-on numbers against a
> non-profile baseline.

## Through the benchmark harness

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
