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
| `FUSED_EP` | 0 | `v2` or `adaptive`: one fused program per MoE layer (dispatch, expert matmuls, combine) instead of all-gather + grouped matmul + reduce-scatter, armed for every step. `adaptive` does not compile on torch_tpu 0928; it needs dev20261017 or later |
| `W4A8` | 0 | `1` = fp8 activations in the fused EP kernel, so 4-bit experts feed the MXU as fp8 instead of being dequantized to bf16 (needs `FUSED_EP`) |
| `BLOCK_SIZE` | 384 | `--block-size` |
| `HOT_SET` | | simulated routing for profiling: `uniform` = balanced; `16:0.3` = 30% of tokens route only into 16 fixed experts (per-expert load CV 1.67, one EP rank 10x loaded); `16:0.3:spread` = the same skew with one hot expert per rank (balanced control). Dummy weights route near-uniformly, so expert imbalance is invisible without this |
| `EXPERT_BLOCK` | | `256` or `512`: requantize the 4-bit expert weights to one scale per 256/512 elements (`MOE_REQUANTIZE_BLOCK_SIZE`), so the grouped matmul applies the scale after the MXU instead of dequantizing every 16-element block on the vector unit first; changes numerics, check accuracy on real weights |
| `ATTN_BLOCK` | | `long`: decode attention block of 8192 KV tokens, 4 sequences, 2 buffers (`TPU_BATCHED_RPA_DECODE_BLOCK`) instead of the kernel default; for >= 64K tokens per sequence (one KV head per core, one core: -2..-9% per layer at 64-128K, +6% at 24 x 32K); also the DCP cache pass; multi-token verify passes keep the default |
| `MOE_A2A` | `0` | `1`: under DP + EP, exchange MoE tokens with a ragged all-to-all inside each DP group (`TPU_MOE_RAGGED_A2A`): each token goes once to each rank that hosts one of its experts, and one partial output per token comes back, instead of the all-gather + reduce-scatter of the whole group's batch; expert math unchanged (the per-token sum over DP members may add in a different order) |
| `A2A_SC_OFFLOAD` | | `0`: keep the ragged all-to-all off the SparseCore (`--xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false`); libtpu <= 0.0.49 misroutes the offloaded collective on groups that do not start at device 0 |
| `FUSED_DRAFT` | 0 | `1` = draft steps 1..K-1 as one compiled program instead of two programs per step; single-engine layouts only (the switch does not support EP across data-parallel engines) |
| `FAST_SUBST` | 0 | `1` = steady-state fast token substitution (skips the substitution kernel and per-step index uploads) |
| `PACKED_DISPATCH` | 0 | `1`: under DP + EP, gather each token's hidden state and router logits in one MoE dispatch all-gather per layer instead of two (`TPU_MOE_PACKED_DISPATCH`); exact |
| `FUSED_COMBINE` | 0 | `1`: under DP + EP, finish each MoE layer with one all-reduce over the EP group (shared expert folded in, each rank keeps its own tokens' rows) instead of the DP reduce-scatter plus the TP all-reduce (`TPU_MOE_FUSED_COMBINE`); same values, different addition order |
| `WEIGHTS` | dummy | `real` loads the checkpoint at `MODEL_PATH` |
| `KV_MODE` | synthetic | `normal` = real prefill with prefix caching (required with real weights; measure with `../benchmark_window.py`) |
| `EXTRA_ARGS` | | appended to `vllm serve` |

Compile sizes are derived: `[max-num-seqs, max-num-seqs * (MTP_K + 1), MNBT]`.
