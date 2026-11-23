# Engram Host gather benchmark

Compares ways of fetching Engram embedding rows from Host memory into HBM on
TPU v7x-8, one TorchTPU process per core (DP8). The design and results are in
[TPU 上 Engram Host 取数方案对比](../../docs/deepseek_v41/engram_host_gather_options.md).

Each rank holds a 48M x 256 B table (12.3 GB) in CPU memory and fetches 6912
random rows per step. Every step runs one graph:

```text
Prepare -> Mat_A(a) -> start -> Mat_B(b) -> finish -> Mat_C(c)
```

Mat_B is independent of the rows and hides the fetch. Sweeping `b` finds how
much compute each backend needs to hide it, and what remains once it is hidden.

| Backend | Doc | Method |
| --- | --- | --- |
| `E` | baseline | Compute only |
| `B` | scheme 1 | C++ worker polls a request, gathers, writes HBM via PJRT raw H2D; SparseCore polls the response, TensorCore copies |
| `C` | scheme 2 | XLA HostExecute FFI gathers on the Host inside the graph; one DMA to HBM |
| `G` | scheme 3 | SparseCore DMAs each row's 4 KiB pinned Host page to HBM, no Host thread |
| `A` | | As `B`, but SparseCore copies the rows from pinned staging |
| `E2` | | `B`'s graph with no worker, to isolate the kernels' own cost |

## Files

- `bench.py`: graphs, phases (validate, calibrate, sweep, tail, profile) and
  the summary.
- `kernels.py`: SparseCore request/receive and TensorCore publish kernels for
  `A`, `B` and `E2`.
- `native/`: the polling worker (`channel.cc`) and the TorchTPU/PJRT buffer
  shim (`buffer.cc`).
- `reference/`: kernels, HLO rewrite and FFI handler for `C` and `G`, vendored
  from an earlier Engram adapter with imports flattened.
- `xprof_summary.py`: kernel, module and op spans from the XProf captures.

## Running

`buffer.cc` uses TorchTPU's unsupported tensor-buffer header, so it must be
built against the XLA commit of the installed TorchTPU wheel:

```bash
python build_native.py --xla-source /path/to/xla-<XLA_COMMIT> \
  --bazel-root /path/to/bazel-cache --out /path/to/native.so
bash run.sh /path/to/out /path/to/native.so /path/to/torch_tpu-*.whl
```

Measured with torch 2.13.0, torch_tpu 0.2.0rc1, jax 0.11.2 and libtpu 0.0.51.
`C` needs libtpu's PJRT FFI extension, which 0.0.50 lacks. `G` pins another
12.3 GB per rank; the measured run gave the container 256 GiB of shared memory.

Outputs: `summary.json` (validation, calibration, sweep, tail),
`trace_summary.json` (XProf spans), `rank-*/benchmark/metrics.jsonl` (per-step
records) and `profiling/xprof/<backend>-b<b>/`.
