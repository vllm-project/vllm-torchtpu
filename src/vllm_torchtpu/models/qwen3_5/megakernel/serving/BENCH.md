# Benchmarking the Qwen3.5 megakernel

## Acceptance benchmark (`serving/acceptance_benchmark.py`)

`acceptance_benchmark.py` runs interleaved `A/B` serving sweeps (`c8`, `c16`,
`c32`, `900 s` per sweep, `>= 3` pairs) across separate `vllm serve` processes
on one `v7x-8` TPU allocation:

- **Print per-arm commands & environment:**
  ```bash
  python3 src/vllm_torchtpu/models/qwen3_5/megakernel/serving/acceptance_benchmark.py \
    print-commands --reference-mode stock
  ```
- **Latency / throughput sweeps (synthetic MTP acceptance `3.39`):**
  ```bash
  python3 src/vllm_torchtpu/models/qwen3_5/megakernel/serving/acceptance_benchmark.py \
    run --reference-mode stock --concurrencies c8,c16,c32 --pairs 3 \
    --out-dir /tmp/mk_accept \
    --client-cmd-template '<client benchmark command writing {out_json}>'
  ```
- **Accuracy gate (`GSM8K`, separate pass with real rejection sampling + greedy decoding):**
  Synthetic MTP acceptance fixes draft acceptance length for deterministic
  throughput/ITL comparison and does **not** produce the model's real tokens.
  Always run `GSM8K` (`n=1319`) as a separate pass with real rejection sampling
  (`MTP=3` on, synthetic acceptance off, greedy `temperature=0.0`) on both arms:
  ```bash
  python3 src/vllm_torchtpu/models/qwen3_5/megakernel/serving/acceptance_benchmark.py \
    run-gsm8k --reference-mode stock --concurrency c32 \
    --out-dir /tmp/mk_accept_gsm8k \
    --eval-cmd-template '<gsm8k eval command writing {out_json}>'
  ```
- **Offline re-analysis from saved sweep JSON:**
  ```bash
  python3 src/vllm_torchtpu/models/qwen3_5/megakernel/serving/acceptance_benchmark.py \
    analyze --sweeps-json /tmp/mk_accept/sweeps.json
  ```

## Quick manual comparison & in-process A/B calibrator

- Serve with `serve.sh` (`MODE=decode` or `MODE=mtp`) and drive it with
  `vllm bench serve` against `--served-model-name qwen35`.
- Compare against the stock path by serving with `VLLM_TPU_MEGAKERNEL=0`.
- `VLLM_TPU_MEGAKERNEL=ab=1` alternates megakernel and stock steps inside one
  server and logs their per-bucket latencies every `stats_period_s`.
