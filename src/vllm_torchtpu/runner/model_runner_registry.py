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
"""Optional model-owned runner class, picked once when the worker builds it.

A model directory registers a `TPUModelRunner` subclass for its architectures
with `register_model_runner`; the worker asks `select_model_runner` which class
to build. With no registration (the default) the worker builds its own runner
class, so every other model keeps the ordinary runner.

A class may be given as "module:Class" so that registration, which runs in
every process, does not import the runner.
"""

import importlib
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_RUNNERS: dict[str, Any] = {}


def register_model_runner(cls: type | str, *architectures: str) -> None:
    """Build `cls` (a `TPUModelRunner` subclass) for `architectures`.

    Called from a model directory's registration hook. A later registration
    for the same architecture replaces the earlier one. Nothing is imported
    here: a "module:Class" path is resolved when the worker selects it.

    Raises:
        TypeError: `cls` is neither a class nor a string, or an architecture
            is not a string.
        ValueError: `cls` is a string not of the form "module:Class", or no
            architecture is given, or one is empty.
    """
    if isinstance(cls, str):
        module, sep, name = cls.partition(":")
        if not (
            sep
            and all(part.isidentifier() for part in module.split("."))
            and name.isidentifier()
        ):
            raise ValueError(
                f"model runner path {cls!r} is not of the form 'module:Class'"
            )
    elif not isinstance(cls, type):
        raise TypeError(
            f"model runner must be a class or a 'module:Class' path, not {cls!r}"
        )
    if not architectures:
        raise ValueError(f"register_model_runner({cls!r}) names no architecture")
    for arch in architectures:
        if not isinstance(arch, str):
            raise TypeError(
                f"architecture {arch!r} is not a string; pass architectures as "
                "separate arguments"
            )
        if not arch:
            raise ValueError("architecture names must not be empty")
    for arch in architectures:
        previous = _RUNNERS.get(arch)
        if previous is not None and previous != cls:
            logger.warning("Model runner for %s: %r replaces %r", arch, cls, previous)
        _RUNNERS[arch] = cls


def _resolve(cls: type | str) -> type:
    if isinstance(cls, str):
        module, _, name = cls.partition(":")
        return getattr(importlib.import_module(module), name)
    return cls


def select_model_runner(architecture: str, default: type) -> type:
    """The runner class registered for `architecture`, else `default`.

    A registered class must subclass `default`, the class the worker would
    build otherwise.
    """
    registered = _RUNNERS.get(architecture)
    if registered is None:
        return default
    cls = _resolve(registered)
    if not (isinstance(cls, type) and issubclass(cls, default)):
        raise TypeError(
            f"model runner {registered!r} registered for {architecture} does "
            f"not subclass {default.__name__}"
        )
    logger.info("Model runner for %s: %s", architecture, cls.__qualname__)
    return cls
