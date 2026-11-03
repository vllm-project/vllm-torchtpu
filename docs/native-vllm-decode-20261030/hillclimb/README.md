<!-- SPDX-License-Identifier: Apache-2.0 -->

# Decode hill-climb: layouts and knobs for the P131072/D1024 synthetic-KV benchmark

Extends the 2026-10-30 reproduction in the parent directory. Same four hosts, same
`env.sh`, same synthetic-KV connector, same dataset and request construction. Two files
replace `serve.sh` and `benchmark.py` so that the engine layout and the serving knobs are
parameters instead of constants:

| File | Role |
|---|---|
| [layout.sh](layout.sh) | Layout presets: engine count, parallel flags per host, `--headless`, compile sizes |
| [serve_sweep.sh](serve_sweep.sh) | Launcher: `LAYOUT=... GLOBAL_SEQS=... MTP_K=... bash serve_sweep.sh <host-rank>` |
| [bench_sweep.py](bench_sweep.py) | Client for any preset: spreads `GLOBAL_SEQS` requests over the engines, same steady-window statistics as `../benchmark.py`, records spec-decode counters |
| [ARMS.md](ARMS.md) | The arms to run, in order, with the exact environment of each and what it isolates |

The baseline (`LAYOUT=dp8tp4 GLOBAL_SEQS=96 MTP_K=3 KV_BYTES=25769803776`) launches the
2026-10-30 configuration flag for flag; `diff <(bash -x ...)` against `../serve.sh` if in doubt.

## Layouts

All presets use the whole 2x2x4 slice (16 chips, 32 TensorCores) and shard the 512 experts
32 ways; they differ in how the attention side is cut.

| LAYOUT | Engines | Per engine | KV pages of a request | Where the per-layer collectives run |
|---|---|---|---|---|
| `dp8tp4` | 8 (2 per host) | TP4 = 2 chips | one core per KV head | TP all-reduce inside 2 chips; EP exchange across 4 hosts; DP lockstep across 8 engines |
| `dp4tp8dcp2` | 4 (1 per host) | TP8 = 1 host | striped over 2 cores | TP all-reduce inside a host; EP across 4 hosts; DP lockstep across 4 engines |
| `tp32dcp8` | 1 | TP32 = 4 hosts | striped over 8 cores | TP all-reduce and EP across 4 hosts; no DP lockstep |
| `dp2tp16dcp4` | 2 (2 hosts each) | TP16 | striped over 4 cores | experimental launch form: `--nnodes 4` with `--data-parallel-size-local 1`, hosts 1 and 3 headless |

Single-host presets (one 2x2x1 host, 8 TensorCores; `NATIVE_HOST_IPS` holds one address and
`env.sh` sets the 2x2x1 topology). Experts shard 8 ways, so a 23-layer model holds the same
expert bytes per core as the 92-layer model does on the 32-core presets.

| LAYOUT | Engines | Per engine | KV pages of a request | Where the per-layer collectives run |
|---|---|---|---|---|
| `tp8` | 1 | TP8 = 1 host | one core per KV-head replica | TP all-reduce and EP inside the host |
| `tp8xdcp2` | 1 | TP8 = 1 host | striped over 2 cores | TP all-reduce and EP inside the host |
| `dp2xtp4` | 2 | TP4 = 2 chips | one core per KV head | TP all-reduce inside 2 chips; EP and DP lockstep across the host |

Why these: the model's 4 KV heads make TP4 the largest tensor-parallel degree without
KV-head replication; above it, decode context parallel (`--decode-context-parallel-size`)
turns the replicas into page stripes, so a request's 131K-token KV is read by 2, 4 or 8
cores in parallel instead of one. Fewer, wider engines also stop replicating the 42 GiB of
fp8 dense weights (10.6 GiB per core at TP4, 1.3 GiB at TP32), which is KV capacity: at
131K context the slice holds about 150 requests as `dp8tp4` and about 200 as `tp32dcp8`.

## Launch

```bash
export NATIVE_HOST_IPS=... MODEL_PATH=... PYTHON_BIN=... VLLM_BIN=...   # as in ../README.md
export LAYOUT=tp32dcp8 GLOBAL_SEQS=96 MTP_K=3
bash serve_sweep.sh <host-rank>      # on every host, same environment, same second
```

Wait for `/health` on the API hosts (host 0 only for `tp32dcp8` and `dp2tp16dcp4`; every
host otherwise), then from the client environment:

```bash
export NATIVE_BENCH_DATASET=... NATIVE_BENCH_OUTPUT_DIR=$PWD/results/<arm>
LAYOUT=tp32dcp8 GLOBAL_SEQS=96 "$PYTHON_BIN" bench_sweep.py --stage <arm> --input 131072 --output 1024 --arrival-interval 0.25
```

`bench_sweep.py` asserts that every request produced exactly `--output` tokens and prints a
`SUMMARY` line; the `steady` block carries the comparable numbers (`tokens_per_second`,
`mean_request_tpot_ms`). `cached_tokens_hist` must show `131071` for every request.

## Knobs

| Env | Default | Effect |
|---|---|---|
| `GLOBAL_SEQS` | 96 | total concurrency; `--max-num-seqs` = `GLOBAL_SEQS / engines`; must fit the KV pool (`dp8tp4` holds at most about 18 per engine at 131K) |
| `MTP_K` | 3 | draft tokens; `0` removes `--speculative-config`. With dummy weights the draft is never accepted, so every K > 0 is pure cost |
| `MNBT` | 1024 | `--max-num-batched-tokens`; must be >= `max-num-seqs * (MTP_K + 1)` |
| `KV_BYTES` | auto | explicit `--kv-cache-memory-bytes`; the baseline used 25769803776 (24 GiB) |
| `ASYNC_SCHED` | 1 | `--async-scheduling` on/off |
| `FUSED_EP` | 0 | `1` = `MOE_FUSED_EP_KERNEL_IMPL=adaptive MOE_FUSED_EP_KERNEL_MIN_TOKENS=1`: one fused program per MoE layer instead of all-gather + grouped matmul + reduce-scatter |
| `BLOCK_SIZE` | 384 | `--block-size` |
| `EXTRA_ARGS` | | appended to `vllm serve` |

Compile sizes are derived: `[max-num-seqs, max-num-seqs * (MTP_K + 1), MNBT]`.
