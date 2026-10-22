# Reproduce the sparse prefill coefficient fit

The retained [calibration data](calibration/results.json), [fitter](fit.py),
[expected fit](expected-fit.json) and [runtime record](runtime.json) preserve
CPU reproduction of all 14 coefficients. The fitter needs NumPy and SciPy;
it neither imports JAX nor measures hardware. See the [cost model guide][guide]
for feature units, calibration scope and limitations.

Use an existing environment; the recorded versions are NumPy 2.5.4 and
SciPy 1.18.1. Run from this directory with a separate output directory:

```bash
REPRO_DIR="$PWD"
FIT_RUN_DIR="$(mktemp -d /tmp/sparse-prefill-fit.XXXXXX)"
mkdir "$FIT_RUN_DIR/calibration"
cp "$REPRO_DIR/calibration/results.json" "$FIT_RUN_DIR/calibration/results.json"
cd "$FIT_RUN_DIR"
python "$REPRO_DIR/fit.py"
```

Compare `fit.json` with `expected-fit.json`: both dtypes, all four fits,
`coefficients_ms`, prediction errors and per-case predictions. The fitter skips
a dtype unless it contains exactly 30 cases. Report any differences with the
environment; do not overwrite saved inputs, expected output or production
coefficients. An offline refit does not establish term necessity or validate
current dispatch performance.

Historical held-out/full-call/context-edge timings and original packaging
hashes remain in the [fixed source snapshot][archive]. They are not runtime
inputs and are not required by the fitter. The exact historical TPU harness
is not packaged here. Use `scripts/benchmark_qsa_prefill.py` for current
repository kernels; this CLI does not regenerate the historical samples.

[guide]: ../../docs/developers_guide/sparse_gqa_prefill.md#cost-and-choice
[archive]: https://github.com/vllm-project/vllm-torchtpu/tree/b56b6ec6928edef4972793d4651e84454938d927/scripts/sparse_gqa_prefill_cost_model
