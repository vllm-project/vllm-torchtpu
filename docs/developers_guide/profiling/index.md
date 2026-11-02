# Profiling

`vllm-torchtpu` captures TPU traces with the PyTorch/XLA profiler. Every worker
process drives its own profiler session and can only trace the chips it owns, so
a whole-slice trace is assembled from one capture per worker (see
[`src/vllm_torchtpu/profiler_trace.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/profiler_trace.py)).
Every mode writes XPlane protobuf (`*.xplane.pb`), readable in
[XProf](https://github.com/openxla/xprof) / TensorBoard; the two non-phased
modes also write Chrome-format JSON. See [Trace formats](#trace-formats) for
which mode produces what.

There are four ways to profile a workload:

| Approach | Also called | Best for |
|---|---|---|
| [`examples/tpu_profiling.py`](offline.md) | non-phased, on-demand profiling | Isolating a single shape — one prefill or one decode batch, offline |
| [Server-side capture](server.md) | end-to-end (e2e) profiling | A real serving window under `vllm bench serve` or your own client |
| [Phased profiling](phased.md) | phased profiling | One trace per inference phase from a single mixed benchmark run |
| [Disaggregated serving (Raiden3)](disaggregated.md) | PD disagg / KV cache transfer profiling | Correlating Prefill (`kv_producer`) and Decode (`kv_consumer`) traces across pods |

## Prerequisites

Install a viewer for the XPlane files. The traces themselves need nothing extra
at capture time:

```bash
pip install xprof            # or: pip install tensorboard-plugin-profile
```

## Trace formats

What lands on disk depends on which profiler ran:

| Mode | `*.xplane.pb` | `*.pt.trace.json.gz` |
|---|---|---|
| [Offline script](offline.md) | yes | yes |
| [Server-side capture](server.md) | yes | yes |
| [Phased profiling](phased.md) | yes | yes, one per phase per rank |

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
  which the examples in these guides do set.
