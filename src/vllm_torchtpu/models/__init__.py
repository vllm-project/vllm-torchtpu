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
import pkgutil
from collections.abc import Iterator

# The legacy package: `models/vllm.register_models()` registers its models
# centrally; it is not a model directory and has no `register()`.
LEGACY_PACKAGE = "vllm"


def iter_model_directories() -> Iterator[str]:
    """Yield the package name of every model directory, in name order.

    A model directory is a package directly under `vllm_torchtpu.models`; the
    legacy `vllm` package is not one.
    """
    for module in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if module.ispkg and module.name != LEGACY_PACKAGE:
            yield module.name


def register_model_directories() -> None:
    """Call `register()` of every model directory under `vllm_torchtpu.models`.

    A `register()` can register architectures with vLLM's
    `ModelRegistry` (as lazy `"module:Class"` paths) and config hooks in
    `MODELS_CONFIG_MAP`, so adding a model does not edit a central list.

    `register()` takes no arguments and runs unconditionally at plugin load in
    every process, before vLLM selects a model; the package `__init__` and
    `register()` must not import JAX, torch_tpu or model code. Optional
    behaviour goes through the config hook it registers
    (`verify_and_update_config(vllm_config)`), not a new environment variable.

    Order: this runs after `models/vllm.register_models()`, and calls the
    directories in package-name order. A later registration of the same
    architecture replaces an earlier one (vLLM logs that at debug level only),
    so a directory overrides `register_models()` and any directory whose name
    sorts before it.

    A package that does not define a callable `register()` is an error, not a
    silent skip.
    """
    for module_name in iter_model_directories():
        name = f"{__name__}.{module_name}"
        package = importlib.import_module(name)
        register = package.__dict__.get("register")
        if not callable(register):
            raise TypeError(f"model directory {name} must define a callable register()")
        register()
