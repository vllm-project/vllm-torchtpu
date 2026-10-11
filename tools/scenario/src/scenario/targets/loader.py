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
"""Which target a command uses: a built-in name, or a Python file.

`--target` wins, then SCENARIO_TARGET. A value that contains `/` or ends in
.py is a file, relative to the current directory. The file defines
make_target(), or exactly one target class, which is built with no
arguments. A fleet's target lives in a file of its own, so the tool never
names a fleet.
"""

import hashlib
import importlib.util
import inspect
import pathlib
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from scenario.target import Target
from scenario.targets import local
from scenario.targets.kubectl import KubectlTarget

BUILTINS = ("local", "kubectl")
ENV_TARGET = "SCENARIO_TARGET"


class TargetLoadError(ValueError):
    """A --target value that names no usable target."""


def target_spec(flag: str | None, env: Mapping[str, str]) -> str | None:
    """--target, else SCENARIO_TARGET, else None."""
    return flag or env.get(ENV_TARGET) or None


def load_target(spec: str, cwd: pathlib.Path) -> Target:
    """The target `spec` names.

    Raises:
        TargetLoadError: no built-in has that name, or the file defines no
            single target.
    """
    if spec == "local":
        return local.make_target(cwd)
    if spec == "kubectl":
        return KubectlTarget()
    if "/" in spec or spec.endswith(".py"):
        return target_from_file(cwd / spec)
    raise TargetLoadError(
        f"no target {spec!r}: name {' or '.join(BUILTINS)}, or a .py file that "
        "defines one"
    )


def load_module(path: pathlib.Path) -> ModuleType:
    """`path` imported under a name of its own; the same file imports once.

    Raises:
        TargetLoadError: the file is missing or fails to import.
    """
    path = path.resolve()
    if not path.is_file():
        raise TargetLoadError(f"no target file {path}")
    name = "scenario_target_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TargetLoadError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        del sys.modules[name]
        raise TargetLoadError(f"importing {path} failed: {error!r}") from error
    return module


def _is_target_class(value: Any, module: ModuleType) -> bool:
    return (
        inspect.isclass(value)
        and value.__module__ == module.__name__
        and not inspect.isabstract(value)
        and all(
            callable(getattr(value, name, None))
            for name in ("info", "reserved_env", "submit")
        )
    )


def target_from_file(path: pathlib.Path) -> Target:
    """The target a file defines: make_target(), or its one target class.

    Raises:
        TargetLoadError: the file defines none, several, or one that fails.
    """
    module = load_module(path)
    factory = getattr(module, "make_target", None)
    if factory is None:
        classes = [v for v in vars(module).values() if _is_target_class(v, module)]
        if len(classes) != 1:
            names = ", ".join(c.__name__ for c in classes) or "none"
            raise TargetLoadError(
                f"{path} defines {len(classes)} target classes ({names}); define "
                "exactly one, or make_target()"
            )
        factory = classes[0]
    try:
        target = factory()
    except Exception as error:
        raise TargetLoadError(
            f"{path}: building the target failed: {error!r}"
        ) from error
    if not isinstance(target, Target):
        raise TargetLoadError(
            f"{path}: {target!r} is not a target (info, reserved_env, submit)"
        )
    return target
