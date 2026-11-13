# Pallas HBM DMA reproduction

Standalone HBM-to-VMEM experiments on a single TPU7x-8 host: eight devices,
four chips with two TensorCores per chip. No model, weights, vLLM installation,
or multi-host service is required. The measured mesh is `(8,)`; every device
independently reads its own allocation. There is no collective communication.

## Contents and provenance

These scripts preserve the kernels and configuration grids from the experiments
on 2026-11-13. Separate snapshots deliberately preserve the original contiguous
kernel's validation layout and the paged kernel's scalar-prefetch interface.
Packaging adds an output-directory CLI, environment recording, formatting and a
trace-collector drain interval outside device timing. It does not change the DMA
loops. The reproduction branch is based on public `main`; the experiments depend
on the standalone JAX environment, not on a serving branch.

| Script | Configurations | Coverage |
| --- | ---: | --- |
| `bench_contiguous.py` | 131 | 4 KiB–32 MiB single-DMA latency; 4 KiB–8 MiB window sweep |
| `bench_paged.py` | 100 | Six address patterns; 32 KiB–1 MiB pages; fragmented 1 MiB blocks |
| `bench_small_pages.py` | 270 | 1, 2, 4, 8, 16 KiB; six patterns; Q=1,8,16,32,64,128,256,512 |

`results/` contains all 501 measured configurations, including per-device,
per-call durations, configuration manifests and host wall times. Use device
`scope_us`, not host wall time, for bandwidth calculations. Raw XPlane/Chrome
traces are not committed (the small-page run alone generated about 7.4 GiB).
Every new run writes its own traces for independent verification.

## Environment

Use Linux with an idle, accessible TPU7x-8 and Python 3.12. The measured packages
are pinned in `requirements.txt`: JAX/JAXLIB 0.11.1, libtpu 0.0.46.3, NumPy 2.5.4.
Install in an isolated environment; avoid changing a running serving environment.

```bash
python3.12 -m venv /tmp/hbm-repro-venv
source /tmp/hbm-repro-venv/bin/activate
python -m pip install -r scripts/hbm_repro/requirements.txt
export JAX_PLATFORMS=tpu
export LIBTPU_INIT_ARGS="--xla_tpu_profile_traceme_level=10 --xla_enable_custom_call_region_trace=true"
python -c 'import jax; print(jax.devices())'
```

If TPU discovery needs explicit topology because metadata lookup is unavailable,
set the following **before importing JAX**. This describes the measured single
host; use your allocation's actual topology if different. `TPU_WORKER_HOSTNAMES`
must be the host's TPU-network address, not the example machine's address.

```bash
export TPU_SKIP_MDS_QUERY=true
export TPU_ACCELERATOR_TYPE=tpu7x-8
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_WORKER_ID=0
export TPU_WORKER_HOSTNAMES="<this-host-TPU-network-IP>"
```

The scripts assert eight devices. They use a 60 MiB compiler VMEM budget to match
the measured experiment; this is a benchmark budget, not a claim about hardware
VMEM capacity. Peak bandwidth is normalized to **3.69e12 bytes/s per device**.
Other hardware needs an explicitly reviewed topology, budget and normalization.

## Run and analyze

From the repository root, start with smoke tests. Output directories must be
empty. Do not redirect shell logs inside them before launching the script.

```bash
python scripts/hbm_repro/bench_contiguous.py --smoke --output /tmp/hbm-contiguous-smoke
python scripts/hbm_repro/analyze.py /tmp/hbm-contiguous-smoke
python scripts/hbm_repro/bench_paged.py --smoke --output /tmp/hbm-paged-smoke
python scripts/hbm_repro/analyze.py /tmp/hbm-paged-smoke
python scripts/hbm_repro/bench_small_pages.py --smoke --output /tmp/hbm-small-smoke
python scripts/hbm_repro/analyze.py /tmp/hbm-small-smoke
```

Omit `--smoke` to run each complete grid, using fresh directories:

```bash
python scripts/hbm_repro/bench_contiguous.py --output /tmp/hbm-contiguous
python scripts/hbm_repro/analyze.py /tmp/hbm-contiguous
python scripts/hbm_repro/bench_paged.py --output /tmp/hbm-paged
python scripts/hbm_repro/analyze.py /tmp/hbm-paged
python scripts/hbm_repro/bench_small_pages.py --output /tmp/hbm-small
python scripts/hbm_repro/analyze.py /tmp/hbm-small
```

Allow tens of minutes and at least 15 GiB of free disk for the small-page grid.
The scripts print `COMPILED_VALIDATED`, `PROFILE_BATCH_DONE` and finally `DONE`.
`run.json` must end with `status: passed`; the analyzer must pass every assertion.
A compile/validation failure or missing trace is a failed run, not a timing sample.

Each run records `environment.json`, `devices.json`, `run.json` and `profile_*`.
Analysis writes `analysis.json`. Open the profile directories with your XProf
installation if you want to inspect the individual DMA scopes and module bounds.

Browse [all 501 measured configurations](MEASUREMENTS.md), or regenerate the
tables and latency fits without a TPU:

```bash
python scripts/hbm_repro/summarize.py scripts/hbm_repro/results > /tmp/hbm-results.md
```

For a new run, place its `analysis.json` in an appropriately named subdirectory
of a results directory and pass that directory to `summarize.py`.

## Measurement contract

- FP32 source rows have 128 elements (512 bytes). Each window transfers exactly
  32 MiB of useful payload per device. Contiguous sources occupy 32 MiB/device;
  paged sources occupy 64 MiB/device. Eight devices run concurrently.
- Single-DMA latency: 64 sequential `start(); wait()` pairs per call, each with
  its own `DMA_` scope. Average the 64 durations to obtain one call sample.
- Window: prime Q slots, wait/recycle slots, then drain. The steady loop is
  unrolled eight times. One `DMA_` scope covers the whole operation, including
  scalar indexing, issue/wait overhead and fill/drain.
- Group mode: each logical 1 MiB block starts all F fragment DMAs, then waits
  for all F before recycling the block. Logical Q uses Q MiB of payload buffers;
  physical outstanding capacity is Q times F. Do not confuse the two Q values.
- Ten profiled calls per configuration; discard the first two. Average eight
  samples per device, then average the eight devices. MBU is the arithmetic mean
  of per-device `32 MiB / duration / 3.69e12`.
- Compilation and validation run before profiling. The contiguous suite checks
  the first 128 values of each final ring slot; paged suites check its first and
  last 128 values. These are sampled checks, not a checksum of all traffic.
  For a 1 KiB fragment the two samples cover the entire retained fragment.
- Page IDs are prefetched to SMEM before the timed scope. The window includes
  SMEM lookups but excludes the HBM-to-SMEM page-table preload. Validation stores,
  host dispatch, barriers and the 0.2-second inter-call drain interval are excluded.
- Profiler settings are held fixed. Fast small-DMA loops can overflow the trace
  collector. The small-page suite reduces batch sizes and all packaged suites
  pause outside timed calls. The analyzer requires ten complete module calls and
  64 latency scopes or one window scope per call on every device. Never derive
  throughput from a truncated trace.
- Compilers can reuse executable names across page-table patterns. The analyzer
  attributes scopes by module boundaries and recorded call order, not scope names.

| Pattern | Address selection |
| --- | --- |
| direct | Sequential addresses without a page table |
| offset | Sequential addresses shifted by half a logical page |
| table | Sequential page IDs in SMEM |
| stride2 | Every other page in a 64 MiB source |
| random | Permutation of pages within 32 MiB |
| spread | Unique random pages selected from 64 MiB |

The half-page offset stays 512-byte aligned even at 1 KiB. These are application
logical pages inside one JAX allocation: this does not measure OS page faults,
control physical HBM mappings, or isolate TLB effects. It also does not reproduce
the full QSA access pattern or include QSA compute and page-table loading.

See [CONCLUSIONS.md](CONCLUSIONS.md) for the measured findings and limitations.

## Packaging verification

The packaged scripts were smoke-tested on TPU7x-8: four contiguous, four paged,
and one 1 KiB random-page configuration, all on eight devices. Output samples
and complete trace counts passed. Representative device-0 timings were
10.640 us for a single 32 MiB DMA, 85.42% MBU for 1 MiB/Q8 contiguous reads,
81.64% MBU for fragmented 64 KiB random pages at logical Q8, and 3.90% MBU
for 1 KiB/Q512 random pages. These are smoke observations, not replacements
for the eight-device averages in the archived full sweeps.
