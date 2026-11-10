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

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm_torchtpu.envs import env_int, env_str

if TYPE_CHECKING:
    JOBID: str = ""
    K3_VENV_DIR: str = ""
    LIBTPU_INIT_ARGS: str = ""
    MEGAKERNEL_TOOL_WORKER: str = ""
    SLURM_JOB_ID: str = ""
    SLURM_LOCALID: str = ""
    SLURM_NODEID: str = ""
    SLURM_PROCID: int = 0
    TORCH_TPU_BASE_PORT: int = 8070
    TP32_HOST_IPS: str = ""
    TP32_SESSION: str = ""
    TRACE_RANKS: str = "0"


def _required(name: str) -> Callable[[], str]:
    """A string knob that must be set; reading it unset raises ``KeyError``."""
    read = env_str(name, "")

    def _get() -> str:
        """Return the value, or raise when the variable is unset or empty."""
        value = read()
        if not value:
            raise KeyError(name)
        return value

    return _get


environment_variables: dict[str, Callable[[], Any]] = {
    # Four-host allocation the bring-up tools launch on.
    "JOBID": env_str("JOBID", ""),
    # Python environment the launched workers run; defaults to this one.
    "K3_VENV_DIR": env_str("K3_VENV_DIR", sys.prefix),
    # libtpu flags the profile command extends with its LLO dump flags.
    "LIBTPU_INIT_ARGS": env_str("LIBTPU_INIT_ARGS", ""),
    # "1" in a process the harness launched as a worker.
    "MEGAKERNEL_TOOL_WORKER": env_str("MEGAKERNEL_TOOL_WORKER", ""),
    # Launcher variables, set per worker.
    "SLURM_JOB_ID": env_str("SLURM_JOB_ID", ""),
    "SLURM_LOCALID": _required("SLURM_LOCALID"),
    "SLURM_NODEID": env_str("SLURM_NODEID", "0"),
    "SLURM_PROCID": env_int("SLURM_PROCID", 0),
    "TORCH_TPU_BASE_PORT": env_int("TORCH_TPU_BASE_PORT", 8070),
    "TP32_HOST_IPS": _required("TP32_HOST_IPS"),
    "TP32_SESSION": _required("TP32_SESSION"),
    # Ranks that save an xprof trace in `bench`, comma separated.
    "TRACE_RANKS": env_str("TRACE_RANKS", "0"),
}


def __getattr__(name: str) -> Any:
    """Read a knob lazily."""
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the knobs."""
    return list(environment_variables.keys())
