# Benchmarking & Evaluation Guide

`vllm-torchtpu` provides an automated benchmarking and accuracy evaluation harness under `scripts/vllm/benchmarking/` that powers both local performance testing and CI regression gates (`.buildkite/pipeline_perf.yml`).

To install the required benchmarking and evaluation dependencies (`lm-eval`, `datasets`, `runai-model-streamer`), run:

```bash
uv pip install --prerelease=allow -e ".[benchmarking,runai]" --group vllm
```

Multimodal evals (`--run-mm-eval`) also need the `evalscope` CLI. Its dependencies downgrade `openai` and `filelock`, so install it in its own environment and put it on `PATH`. From the repository root:

```bash
uv venv --python 3.12 ~/evalscope-venv
uv pip install --python ~/evalscope-venv/bin/python --group evalscope
ln -s ~/evalscope-venv/bin/evalscope ~/.local/bin/evalscope
```

---

## Automated Benchmark Harness (`run_benchmarks.sh`)

The documentation below is included from `scripts/vllm/benchmarking/README.md`:

---8<--- "scripts/vllm/benchmarking/README.md:3"

---

## Standalone Server & Client Benchmark Recipe

If you prefer to run `vllm serve` and `vllm bench serve` manually in separate terminals (for example, while iterating on a kernel or capturing an XProf trace), use the recipe below from `scripts/vllm/benchmarking/unified_benchmark_script.md`:

---8<--- "scripts/vllm/benchmarking/unified_benchmark_script.md:3"
