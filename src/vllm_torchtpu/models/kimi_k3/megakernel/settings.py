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
"""Environment knobs of the Kimi-K3 megakernel and its bring-up tools.

Each name reads its variable lazily through the shared helpers in
``vllm_torchtpu.envs``, so the value is the one in the environment at the time
of access. The launcher variables are the ones the bring-up harness sets per
worker (Slurm-style ranks, host list, session, base port).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from vllm_torchtpu.envs import env_int, env_str

if TYPE_CHECKING:
    from collections.abc import Callable

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
    """Build a lazy reader for an environment variable that must be non-empty.

    Args:
        name: Environment variable name.

    Returns:
        Zero-argument callable returning the non-empty environment string.
    """
    read = env_str(name, "")

    def _get() -> str:
        """Return the environment variable value, or raise when unset or empty.

        Returns:
            Non-empty string value read from ``name``.

        Raises:
            KeyError: If the environment variable is unset or empty.
        """
        value = read()
        if not value:
            raise KeyError(name)
        return value

    return _get


environment_variables: dict[str, Callable[[], object]] = {
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


def __getattr__(name: str) -> object:
    """Read an environment knob lazily at attribute access (PEP 562).

    Args:
        name: Knob name, a key of ``environment_variables``.

    Returns:
        Current value of the knob's environment variable.

    Raises:
        AttributeError: If ``name`` is not a knob of this module.
    """
    if name in environment_variables:
        return environment_variables[name]()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """List the environment knobs of this module.

    Returns:
        Names of every key in ``environment_variables``.
    """
    return list(environment_variables)
