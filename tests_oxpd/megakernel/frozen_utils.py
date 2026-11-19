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
"""Shared SHA-256-pinned loader for the frozen Qwen3.5 reference package."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import sys
import types
from typing import Any

import pytest

_REPO_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_FROZEN_DIR = os.path.join(_REPO_ROOT_DIR, "frozen_qwen35")
FROZEN_MK_SHA256 = "db407b8fb41a9365bdb54c8bb4a9c7992167f642d75dcd04bab0d7d18293125a"
FROZEN_PIO_SHA256 = "5acd73afccb9cea48382138f1428af2d1f1df5b43faa8817e0a8ed962d28bef6"
FROZEN_PA_SHA256 = "25935a68f06a3231da24e92bcd7659d54fbf94d7a707b9d91854643466d7fad8"


def load_frozen_qwen35(
    frozen_dir: str | os.PathLike[str] | None = None,
) -> tuple[Any, Any]:
    """Verify SHA-256 of all 3 frozen files and load `(frozen_mk, frozen_pio)`."""
    resolved_dir = pathlib.Path(
        frozen_dir
        if frozen_dir is not None
        else os.environ.get("MK_FROZEN_QWEN35_DIR", DEFAULT_FROZEN_DIR)
    )
    if not resolved_dir.is_dir():
        pytest.skip(
            f"Set MK_FROZEN_QWEN35_DIR to frozen qwen35 path (checked {resolved_dir})"
        )
    mk_path = resolved_dir / "megakernel.py"
    pio_path = resolved_dir / "pool_io.py"
    pa_path = resolved_dir / "pool_alias.py"
    assert hashlib.sha256(mk_path.read_bytes()).hexdigest() == FROZEN_MK_SHA256
    assert hashlib.sha256(pio_path.read_bytes()).hexdigest() == FROZEN_PIO_SHA256
    assert hashlib.sha256(pa_path.read_bytes()).hexdigest() == FROZEN_PA_SHA256

    pkg_name = "_frozen_qwen35_pkg"
    pkg: Any = types.ModuleType(pkg_name)
    pkg.__path__ = [str(resolved_dir)]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    spec_pa = importlib.util.spec_from_file_location(f"{pkg_name}.pool_alias", pa_path)
    assert spec_pa is not None and spec_pa.loader is not None
    frozen_pa = importlib.util.module_from_spec(spec_pa)
    frozen_pa.__package__ = pkg_name
    sys.modules[spec_pa.name] = frozen_pa
    pkg.pool_alias = frozen_pa
    spec_pa.loader.exec_module(frozen_pa)

    spec_f = importlib.util.spec_from_file_location(f"{pkg_name}.megakernel", mk_path)
    assert spec_f is not None and spec_f.loader is not None
    frozen_mk = importlib.util.module_from_spec(spec_f)
    frozen_mk.__package__ = pkg_name
    sys.modules[spec_f.name] = frozen_mk
    pkg.megakernel = frozen_mk
    spec_f.loader.exec_module(frozen_mk)

    spec_p = importlib.util.spec_from_file_location(f"{pkg_name}.pool_io", pio_path)
    assert spec_p is not None and spec_p.loader is not None
    frozen_pio = importlib.util.module_from_spec(spec_p)
    frozen_pio.__package__ = pkg_name
    sys.modules[spec_p.name] = frozen_pio
    pkg.pool_io = frozen_pio
    spec_p.loader.exec_module(frozen_pio)
    return frozen_mk, frozen_pio
