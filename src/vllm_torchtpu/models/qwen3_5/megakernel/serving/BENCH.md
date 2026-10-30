# Benchmarking the Qwen3.5 megakernel

The megakernel benchmark tool is maintained separately (W3's benchmark
harness, to land as `serving/bench.py`). Until then:

- serve with `serve.sh` (`MODE=decode` or `MODE=mtp`) and drive it with
  `vllm bench serve` against `--served-model-name qwen35`;
- compare against the stock path by serving with `VLLM_TPU_MEGAKERNEL=0`;
- `VLLM_TPU_MEGAKERNEL=ab=1` alternates megakernel and stock steps and logs
  their per-bucket latencies every `stats_period_s` (the A/B calibrator).
