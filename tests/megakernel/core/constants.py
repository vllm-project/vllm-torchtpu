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
"""Test constants for megakernel core runtime tests."""

from __future__ import annotations

from pathlib import Path

REPO_SRC: Path = Path(__file__).resolve().parents[3] / "src"

IMPORT_PROBE: str = """
import sys
for name in {blocked!r}:
    sys.modules[name] = None
import vllm_torchtpu.megakernel.core.runtime.{module}
print("OK")
"""

STARTUP_OK: dict[str, int] = {"num_layers": 60, "expected_num_layers": 60}

__all__ = ["IMPORT_PROBE", "REPO_SRC", "STARTUP_OK"]
