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

Every option defaults to off, which leaves the worker where its launcher put
it. The options only place the worker process on the host and never change a
compiled program, so the section stays out of the compile-cache key
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
    node its chip is attached to and prefer that node's memory (see
    ``vllm_torchtpu.worker.numa_bind``). The node of each TPU device is read
    from sysfs, so no per-host node list is needed. Ranks are spread evenly
    over the host's TPU devices, which assumes this engine and its host-local
    DP engines use the whole host. A rank that does not map onto a device
    (an engine on part of a host, or a dense-model DP engine, which vLLM runs
    as an independent engine) is left alone, and so is an engine whose ranks
    are moved to other devices (TPU_LOCAL_RANK_OFFSET,
    DEBUG_TPU_LOCAL_RANK_OFFSET, VLLM_TPU_DEVICE_ORDER, or
    TPU_MULTIHOST_BACKEND=ray, which runs each worker on the chip Ray assigned
    it). The inherited CPU set
    is only ever narrowed. Cannot be combined with vLLM's ``--numa-bind``,
    which binds the same workers through numactl. Off by default."""


def parse(raw: Any) -> WorkerOptions:
    """The ``tpu_worker`` section as options (see ``config_section.parse``)."""
    return config_section.parse(WorkerOptions, SECTION, raw)


def get(vllm_config: Any) -> WorkerOptions:
    """The engine's worker options."""
    return config_section.get(WorkerOptions, SECTION, vllm_config)


def check(vllm_config: Any) -> WorkerOptions:
    """The engine's worker options, refusing ``numa_bind`` together with
    vLLM's ``--numa-bind``: vLLM starts each worker under ``numactl
    --membind``, and this bind would then replace that strict memory policy
    with a preferred one, possibly for another node."""
    options = get(vllm_config)
    if options.numa_bind and vllm_config.parallel_config.numa_bind:
        raise ValueError(
            f"additional_config.{SECTION}.numa_bind and --numa-bind both bind "
            "each TPU worker to a NUMA node; turn one of them off."
        )
    return options
