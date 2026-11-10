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
"""Environment knobs for Kimi-K3 megakernel tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm_torchtpu.envs import env_str

if TYPE_CHECKING:
    SLURM_NODEID: str = "0"

environment_variables = {
    "SLURM_NODEID": env_str("SLURM_NODEID", "0"),
}


def __getattr__(name: str) -> Any:
    """Read a knob lazily."""
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the knobs."""
    return list(environment_variables.keys())
