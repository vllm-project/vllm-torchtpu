# Model directories

A model directory is a package directly under `src/vllm_torchtpu/models/`, such
as `models/qwen4_exp/`. It holds one model family's code and registers that
model with vLLM itself through a `register()` function in its `__init__.py`, so
adding a model does not edit a central list. The legacy package
`models/vllm/` is not a model directory: `register_models()` there registers its
models centrally, and new code does not go under it. Its entries move into the
directories' `register()` one model at a time (Qwen4Exp first); once it is empty
it is removed, leaving one registration API.

## When `register()` runs

`vllm_torchtpu.models.register_model_directories()` is called from
`register_layers()` right after `models/vllm.register_models()`, at plugin load.
For each model directory it imports the package and calls `register()`.

- `register()` runs in every process (API server, engine core, workers), before
  vLLM selects a model, whatever model is served.
- It is unconditional and takes no arguments: there is no `vllm_config` yet.
- A package under `models/` without a callable `register()` raises `TypeError`
  at plugin load.

## What `register()` may do

`register()` may:

- register architectures with vLLM's `ModelRegistry`, as lazy `"module:Class"`
  paths, so the model code is imported only when vLLM loads that model;
- register config hooks in `MODELS_CONFIG_MAP` (a `VerifyAndUpdateConfig`
  subclass with `verify_and_update_config(vllm_config)`).

```python
def register() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_torchtpu.models.my_model.config import TPUMyModelConfig

    ModelRegistry.register_model(
        "MyModelForCausalLM", "vllm_torchtpu.models.my_model.model:MyModelForCausalLM"
    )
    MODELS_CONFIG_MAP["MyModelForCausalLM"] = TPUMyModelConfig
```

## Imports

Because `register()` runs in every process for every model:

- The package `__init__.py` imports only the standard library. It must not
  import JAX, `torch_tpu`, torch, vLLM or model code at module level.
- `register()` imports vLLM's registries inside the function body, and a
  lightweight config module if it registers a config hook. It must not import
  JAX, `torch_tpu` or model code; reference the model class by its
  `"module:Class"` path instead.

`tests/models/test_model_directories.py` imports every model directory with
these frameworks blocked and checks that it defines a callable `register()`.

## Every package under `models/` needs `register()`

The `detect-missing-init --create` pre-commit hook (`.pre-commit-config.yaml`)
adds an `__init__.py` to any folder that holds a Python file. A new folder under
`models/` therefore becomes a package, and so a model directory, as soon as it
holds a `.py` file. Give it a `register()`; if the model is still registered
elsewhere, a no-op `register()` whose docstring says where (as
`models/qwen4_exp/__init__.py` does) keeps the order unchanged. The test above
fails in CI for a folder without one.

## Optional behaviour

`register()` cannot see the served config, so do not switch it with a new
environment variable. Register unconditionally, and put optional behaviour in
the config hook it registers: `verify_and_update_config(vllm_config)` runs for
the selected architecture with the full `vllm_config` (see `_TPUQwen2Config` in
`models/vllm/__init__.py`).

vLLM calls the hook once, in the process that builds `VllmConfig`
(`try_verify_and_update_config` returns once `model_config.config_updated` is
set). The engine core and the workers, started with `spawn` by default, get a
copy of the config and do not call it again. So the hook must write its result
into `vllm_config`, as `_TPUQwen2Config` does, and the model code reads it from
`vllm_config`. A hook that changes process state instead, for example a second
`ModelRegistry.register_model`, has no effect in the process that loads the
model.

## Order and overrides

Model directories run after `register_models()`, in package-name order. A later
registration of the same architecture replaces an earlier one, and vLLM logs
that at debug level only. So a model directory overrides `register_models()` and
every model directory whose name sorts before it. When a directory must win over
another registration, check the names sort that way, or keep the other
registration in `register_models()`.

## `register()` or a `Patch`

- Use `register()` to make vLLM find the model: architectures and config hooks
  that vLLM looks up by architecture name.
- Use a `Patch` (see [TPU patch mechanism](tpu_patch_mechanism.md)) to change
  the behaviour of vLLM or library code at a lifecycle stage, for example a
  model-specific workaround that checks `model_config`. Its callback can live in
  the model directory, for example `models/<model>/patch.py`.
