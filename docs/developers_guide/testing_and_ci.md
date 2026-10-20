# Testing & CI Guide

`vllm-torchtpu` runs continuous integration on Buildkite (`.buildkite/pipeline_tests.yml`), with each step in a Kubernetes pod: CPU pods for unmarked tests and TPU v7x slices (1x1x1 and 2x2x1) for `tpu_test` and `multichip` tests.

---

## Running Tests Locally

Install the `test` dependency group before running `pytest`:

```bash
uv pip install --prerelease=allow -e . --group vllm --group test
```

The commands below match the exact marker expressions and file filters executed in `.buildkite/pipeline_tests.yml`:

### 1. Fast CPU Unit Tests (No TPU Required)

Unmarked unit tests run on CPU using the JAX CPU backend (`CPU_ONLY=1` / `JAX_PLATFORMS=cpu`) and mocked TPU devices:

```bash
JAX_PLATFORMS=cpu pytest tests/ -sv --timeout=600 \
  -m "not tpu_test and not multichip and not nightly"
```

### 2. General Single-Host TPU Tests (2-Chip Queue)

On a TPU VM, run the general `tpu_test` suite (excluding `test_speculative_decoding.py`, which runs in its own dedicated CI step):

```bash
pytest tests/ -sv --timeout=600 \
  --ignore=tests/entrypoints/llm/test_speculative_decoding.py \
  -m "tpu_test and not nightly and not multichip"
```

### 3. Multi-Chip TPU Tests (8-Chip Queue)

On an 8-chip (or 8-device TPU v7x) host, run multi-chip distributed and kernel tests:

```bash
pytest tests/ -sv --timeout=600 \
  --ignore=tests/entrypoints/llm/test_speculative_decoding.py \
  -m "not nightly and multichip"
```

### 4. Speculative Decoding TPU Tests

Speculative decoding e2e tests (`tests/entrypoints/llm/test_speculative_decoding.py`) carry a module-level `pytestmark = pytest.mark.tpu_test` and are split across 2-chip and 8-chip steps:

```bash
# 2-chip speculative decoding PR suite
pytest tests/entrypoints/llm/test_speculative_decoding.py -sv --timeout=1800 \
  -m "not nightly and not multichip"

# 8-chip speculative decoding multichip PR suite (e.g. DP + MTP)
pytest tests/entrypoints/llm/test_speculative_decoding.py -sv --timeout=3600 \
  -m "not nightly and multichip"
```

### 5. Omni Multimodal & Diffusion TPU Tests (`tests_omni/`)

With `--extra omni` installed, run the `vllm-omni` test suite:

```bash
pytest tests_omni/ -sv --timeout=1200 -m "not nightly and omni"
```

---

## Pytest Markers Reference

Every test in `tests/` and `tests_omni/` is routed to the appropriate CI hardware using the markers registered in `[tool.pytest.ini_options]` in `pyproject.toml`:

| Marker | CI hardware (kube pod) | Description |
| :--- | :--- | :--- |
| *(unmarked)* | CPU pod (`JAX_PLATFORMS=cpu`) | Fast unit tests that do not require physical TPU devices. |
| `@pytest.mark.tpu_test` | TPU v7x 1x1x1 (1 chip) | Tests that require physical TPU hardware (`torch.device("tpu")`). |
| `@pytest.mark.multichip` | TPU v7x 2x2x1 (4 chips) | Tests that require more than 2 TPU chips (e.g., TP/EP/DP collectives). |
| `@pytest.mark.spawns_tpu_workers` | Multi-chip TPU slice | Tests that spawn a `torch.distributed.run` worker group spanning the TPU slice; ordered by `tests/conftest.py` to run before any in-process TPU test in the same `pytest` session. |
| `@pytest.mark.nightly` | Nightly CI workflow (`NIGHTLY=1`) | Long-running or full-checkpoint tests excluded from per-PR CI gates. Must still be combined with `tpu_test` or `multichip` if TPU hardware is required. |
| `@pytest.mark.omni` | TPU v7x 1x1x1 (`-omni` image) | Tests that run `vllm-omni` multimodal and diffusion models (`tests_omni/`). |

---

## Buildkite Pipeline Architecture

The details below are included directly from `.buildkite/README.md`:

---8<--- ".buildkite/README.md:3"

---

## Pre-Commit & Static Analysis Gates

Every pull request is gated by `.pre-commit-config.yaml`. Run all checks locally before pushing:

```bash
pre-commit run --all-files
```

Key repository hooks:

- **`ruff-check` & `ruff-format`**: Enforces PEP 8 formatting (88-column line width) and import sorting.
- **`pyrefly`**: Runs static type checking (`[tool.pyrefly]` in `pyproject.toml`)—see [Type Checking](type_checking.md).
- **`pymarkdown`**: Automatically formats Markdown files.
- **`check-env-reads`**: Verifies that all environment variable reads go through `src/vllm_torchtpu/envs.py`.
- **`check-b023-baseline`**: Verifies that Pallas kernel files under `src/vllm_torchtpu/kernels/` do not introduce unintended loop-variable closures.
