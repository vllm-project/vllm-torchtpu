# Troubleshooting

## The server fails to start

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

## The run produces no or unexpected traces

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
