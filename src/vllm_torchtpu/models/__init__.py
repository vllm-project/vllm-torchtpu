# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model directories: `models/<model>/` packages that register themselves.

The contract is in `docs/developers_guide/model_directories.md`.
"""

import importlib

# Model directories whose `register()` runs at plugin load, in this order. A
# directory is listed here when it takes over its model's registration; until
# then `models/vllm.register_models()` registers the model. Later entries win
# when two register the same architecture.
MODEL_DIRECTORIES: tuple[str, ...] = ()


def register_model_directories() -> None:
    """Call `register()` of each package in `MODEL_DIRECTORIES`, in list order.

    A `register()` can register architectures with vLLM's
    `ModelRegistry` (as lazy `"module:Class"` paths) and config hooks in
    `MODELS_CONFIG_MAP`, so the registration lives in the model's own
    directory.

    `register()` takes no arguments and runs unconditionally at plugin load in
    every process, before vLLM selects a model; the package `__init__` and
    `register()` must not import JAX, torch_tpu or model code. Optional
    behaviour goes through the config hook it registers
    (`verify_and_update_config(vllm_config)`), not a new environment variable.

    Order: this runs after `models/vllm.register_models()`, and calls the
    directories in `MODEL_DIRECTORIES` order. A later registration of the same
    architecture replaces an earlier one (vLLM logs that at debug level only),
    so a directory overrides `register_models()` and every directory listed
    before it.

    A listed package that does not define a callable `register()` is an error.
    """
    for module_name in MODEL_DIRECTORIES:
        name = f"{__name__}.{module_name}"
        package = importlib.import_module(name)
        register = package.__dict__.get("register")
        if not callable(register):
            raise TypeError(f"model directory {name} must define a callable register()")
        register()
