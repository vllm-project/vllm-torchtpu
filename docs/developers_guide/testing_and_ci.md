# Testing & CI Guide

`vllm-torchtpu` runs continuous integration on Buildkite across CPU agents (`cpu_64_core`), single-host TPU agents, and multi-chip/multi-host TPU slices (`TPU v7x` and `TPU v6e`).

---

## Running Tests Locally

Install the `test` dependency group before running `pytest`:

```bash
uv pip install --prerelease=allow -e . --group vllm --group test
```

### 1. Fast CPU Unit Tests (No TPU Required)

Unmarked unit tests run on CPU using the JAX CPU backend and mocked TPU devices:

```bash
JAX_PLATFORMS=cpu pytest tests/ -m "not tpu_test and not multichip and not nightly"
```

### 2. Single-Host TPU Tests

On a TPU VM, run tests marked with `tpu_test`:

```bash
pytest tests/ -m "tpu_test and not multichip and not nightly"
```

### 3. Multi-Chip TPU Tests

On an 8-chip (or 8-device TPU v7x) host, run multi-chip distributed and kernel tests:

```bash
pytest tests/ -m "multichip and not nightly"
```

---

## Pytest Markers Reference

Every test in `tests/` and `tests_omni/` is routed to the appropriate CI queue using the markers defined in `pyproject.toml`:

| Marker | Hardware / CI Queue | Description |
| :--- | :--- | :--- |
| *(unmarked)* | `cpu_64_core` (`JAX_PLATFORMS=cpu`) | Fast unit tests that do not require physical TPU devices. |
| `@pytest.mark.tpu_test` | Single-host TPU agent | Tests that require physical TPU hardware (`torch.device("tpu")`). |
| `@pytest.mark.multichip` | 8-chip TPU agent | Tests that require more than 2 TPU chips (e.g., TP/EP/DP collectives). |
| `@pytest.mark.spawns_tpu_workers` | Multi-device TPU agent | Tests that spawn a `torch.distributed.run` worker group spanning the TPU slice; must run before any in-process TPU test in the same `pytest` session. |
| `@pytest.mark.nightly` | Nightly CI workflow only | Long-running or full-checkpoint tests excluded from per-PR CI gates. Must still be combined with `tpu_test` or `multichip` if TPU hardware is required. |
| `@pytest.mark.omni` | Omni TPU CI workflow | Tests that run `vllm-omni` multimodal and diffusion models (`tests_omni/`). |

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
