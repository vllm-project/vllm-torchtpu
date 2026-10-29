# DeepSeek-V4.1-Flash 1P1D (Stage-3 Raiden) on two v7x hosts

Deployment and benchmark scripts used with this branch (`pd/dsv41-1p1d` = epic/dsv41 + PD
commits). Paths are those of the original setup (`/zzlfs/temu/...`); adjust before reuse.

Topology: P and D are each one 4-chip v7x host, DP8 / TP1 / EP8. A proxy on the P host
(`:8000`) renders the prompt once and sends token IDs to P (`:8100`) and D (`:8200`).

## Environment

- `requirements-epic.txt`: runtime for this branch (torch-tpu 0.2.0, jax 0.11.2, libtpu 0.0.50,
  vLLM e37e51dd). The KV transfer additionally needs tpu-sync with the `GetBlockChunks` fix
  (google/tpu-sync#1382), built against the same torch-tpu.
- `apply_venv_patches.py` (run with the venv's python, idempotent) patches third-party code:
  1. runai_model_streamer_gcs: `bucket()` instead of `get_bucket()`.
  2. vLLM scheduler: loads the connector marks with `skips_prefix_replay(request)` are neither
     rounded down to a block nor replayed (D keeps prefix caching under `--swa-bounded-replay`).
  3. vLLM detokenizer: prime `DecodeStream` with the prompt tail (was the whole prompt; ~0.5 s
     per hop at 557k tokens).
  4. vLLM DP engine: `VLLM_DP_WAVE_START_DELAY_MS` holds a new wave's first step so requests
     that arrive within a few ms on other EP-DP ranks join it.
  5. starlette `Request.json` via orjson (10^6-token-id bodies).

## Serving

- `start_1p1d_epic.sh prefill|decode`: one role. Knobs: `ASYNC_SCHED`, `MAX_NUM_SEQS`,
  `PREFILL_COMPILE_SIZES`, `DECODE_COMPILE_SIZES`, `API_SERVERS`, `VLLM_DP_WAVE_START_DELAY_MS`,
  `TP_SIZE`/`DP_SIZE`, `NO_KV_TRANSFER`.
- `restart_epic.sh` / `stop_vllm_local.sh` / `wait_ready.sh`: stop both roles (and stray
  workers), confirm `/dev/vfio` is free, start, wait for health.
- `start_proxy.sh` + `toy_proxy_server.py`: proxy with local, process-pool tokenization
  (`PROXY_TOKENIZER`) and orjson bodies.

Configuration used for the reported numbers: P `[128,256,2048]` compile sizes, no async
scheduling, wave delay 40 ms; D async scheduling, `MAX_NUM_SEQS=32`,
`[16,48,96,192,2048]`, wave delay 20 ms; 4 API servers each.

## Benchmarks

- `bench_cached_prefill.py`, `run_table.sh`, `quick_set.sh`: "cached + new" TTFT / prefill
  (prefill = P's `request_prefill_time`, TTFT = client to first token).
- `tpot_bench.py`: TPOT (CLI / Server) over the full-batch window, shared cached prefix.
- `pd_logprob_check.py`: PD vs direct-D first-token logprobs on fresh text.
- `transfer_stats.py`, `d_breakdown.py`: KV transfer latency and D-side timing breakdown.
