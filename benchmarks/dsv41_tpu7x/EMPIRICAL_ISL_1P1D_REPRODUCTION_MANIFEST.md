# DeepSeek-V4.1-Flash FP8 PD Disaggregation (`1P1D`, `DP=8`) — Empirical ISL (`50:20:12:9:9`) Reproduction Manifest & Commands

## 1. 1P1D Server & Proxy Launch ([`run_pd.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/run_pd.py))

```bash
python3 benchmarks/dsv41_tpu7x/scripts/pd/run_pd.py \
  --cluster bodaborg \
  --name emp_isl_95_1p1d \
  --pod-suffix e95 \
  --keep-pod
```

This provisions (via [`in_pod_serve.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/in_pod_serve.sh) and [`in_pod_setup.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/in_pod_setup.sh)):
- **Prefill Pod (`1P`)**: `2x2x1` TPU7x (`DP=8, TP=1, EP=8`), port `8400`, `--kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_role":"kv_producer"}'` with `--enable-prefix-caching --prefix-cache-retention-interval 1024 --max-num-batched-tokens 16384 --max-num-seqs 8 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice`.
- **Decode Pod (`1D`)**: `2x2x1` TPU7x (`DP=8, TP=1, EP=8`), port `9400`, `--kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_role":"kv_consumer"}'` with `ENABLE_DSPARK=1` (`method="dflash"`, `num_speculative_tokens=5`), `--async-scheduling --enable-prefix-caching --prefix-cache-retention-interval 1024 --max-num-batched-tokens 96 --max-num-seqs 16 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice`.
- **Disaggregated Proxy ([`toy_proxy_server.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/examples/disagg/toy_proxy_server.py))**: Port `8000` on the prefill pod (`--prefill-max-inflight 8` during trunk warmup and multi-session sweeps, forwarding `/reset_prefix_cache` to both `1P` and `1D`).

---

## 2. Empirical ISL (`50:20:12:9:9`) Benchmark Execution ([`bench_empirical_isl_dsv41.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/bench_empirical_isl_dsv41.py))

```bash
# 1. 95% Cache Hit — Standard 5-or-N Trunks Sweep (n = 8, 16, 32, 64, 128 -> 40, 40, 40, 64, 128 trunks)
python3 -u /tmp/bench_empirical_isl_dsv41.py \
  --base-url http://127.0.0.1:8000 \
  --prefill-url http://127.0.0.1:8400 \
  --decode-url http://<DECODE_POD_IP>:9400 \
  --dataset /data/requests_mocked.jsonl.xz \
  --sessions 8,16,32,64,128 \
  --duration-per-n 60 \
  --cache-hit-rate 0.95 \
  --max-tokens 1024 \
  --ignore-eos \
  --dp-size 8 \
  --trunks-above-n16 0 \
  --warmup-pass-s 8 \
  --output-json /tmp/dsv41_pd_empirical_isl_95_summary.json

# 2. 95% Cache Hit — Pinned 40-Trunk High-Concurrency Run (n = 128 -> 40 trunks = 5 per DP rank)
python3 -u /tmp/bench_empirical_isl_dsv41.py \
  --base-url http://127.0.0.1:8000 \
  --prefill-url http://127.0.0.1:8400 \
  --decode-url http://<DECODE_POD_IP>:9400 \
  --dataset /data/requests_mocked.jsonl.xz \
  --sessions 128 \
  --duration-per-n 60 \
  --cache-hit-rate 0.95 \
  --max-tokens 1024 \
  --ignore-eos \
  --dp-size 8 \
  --trunks-above-n16 40 \
  --warmup-pass-s 0 \
  --output-json /tmp/dsv41_pd_empirical_isl_95_n128_tr40.json
```

---

## 3. Public GitHub & GCS Provenance Links

- **Unified Branch (`cpk/dsv41-combined`)**: [`vllm-project/vllm-torchtpu` (`cpk/dsv41-combined`)](https://github.com/vllm-project/vllm-torchtpu/tree/cpk/dsv41-combined/benchmarks/dsv41_tpu7x)
- **Benchmark Harness & Orchestration Scripts**:
  - [`bench_empirical_isl_dsv41.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/bench_empirical_isl_dsv41.py)
  - [`run_pd.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/run_pd.py)
  - [`in_pod_serve.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/in_pod_serve.sh)
  - [`pd_gates.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/pd_gates.py)
  - [`toy_proxy_server.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/examples/disagg/toy_proxy_server.py)
- **Benchmark Reports & Consolidated JSON Summaries (GitHub)**:
  - [`EMPIRICAL_ISL_1P1D_REPORT.md`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/EMPIRICAL_ISL_1P1D_REPORT.md)
  - [`EMPIRICAL_ISL_1P1D_REPRODUCTION_MANIFEST.md`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/EMPIRICAL_ISL_1P1D_REPRODUCTION_MANIFEST.md)
  - [`dsv41_pd_empirical_isl_95_summary.json`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/dsv41_pd_empirical_isl_95_summary.json)
- **GCS Archive (`gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/`)**:
  - `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/empirical_isl_95_1p1d_20261029/dsv41_pd_empirical_isl_95_summary.json`
  - `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/empirical_isl_95_1p1d_20261029/bench_empirical_isl_95_full.log`
  - `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/empirical_isl_95_1p1d_20261029/bench_empirical_isl_95_n128_tr128_pass2.log`
  - `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/empirical_isl_95_1p1d_20261029/bench_empirical_isl_95_n128_tr40.log`
  - `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/empirical_isl_95_1p1d_20261029/prefill__pd_gates.json`
  - `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/reports/`
