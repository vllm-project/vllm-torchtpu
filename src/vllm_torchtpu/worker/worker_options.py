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
"""Options of the TPU worker, from ``additional_config["tpu_worker"]``.

    --additional-config '{"tpu_worker": {"numa_bind": true}}'

Every option defaults to main's behaviour without it. The options only place
the worker process on the host and never change a compiled program, so the
section stays out of the compile-cache key
(``HASH_IGNORED_ADDITIONAL_CONFIG_KEYS``). Parse once at init with :func:`get`.
"""

from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import config_section

SECTION = "tpu_worker"


@dataclass(frozen=True)
class WorkerOptions:
    numa_bind: bool = False
    """Before the TPU runtime starts, narrow each worker's CPU set to the NUMA
    node its chip is attached to and prefer that node's memory
    (``worker.numa_bind``). The inherited CPU set is only ever narrowed, and a
    rank that does not map onto a chip is left alone. Off by default."""


def parse(raw: Any) -> WorkerOptions:
    """The ``tpu_worker`` section as options (see ``config_section.parse``)."""
    return config_section.parse(WorkerOptions, SECTION, raw)


def get(vllm_config: Any) -> WorkerOptions:
    """The engine's worker options."""
    return config_section.get(WorkerOptions, SECTION, vllm_config)
