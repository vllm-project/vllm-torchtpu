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
"""The TPU attention backends that --attention-backend <name> selects.

register_attention_backend adds a backend under a name: the name joins vLLM's
AttentionBackendEnum (vllm#41091), so --attention-backend <name> passes vLLM's
check, and TpuPlatform.get_attn_backend_cls resolves it to the backend's vLLM
attention backend class instead of overriding it to FLASH_ATTN. With no
registered name selected the layers run RPA (FLASH_ATTN), or batched
RPA (CUSTOM).
"""

from typing import TYPE_CHECKING

from vllm import envs as vllm_envs
from vllm.ray.lazy_utils import is_in_ray_actor
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_REGISTERED_BACKENDS: set[str] = set()


def register_attention_backend(name: str, *, vllm_backend: str) -> None:
    """Make --attention-backend <name> run vllm_backend, the class path of the
    vLLM attention backend the attention layers get, a PallasAttentionBackend
    subclass.

    Call it from a vllm.general_plugins entry point, which vLLM runs in every
    process before it parses --attention-backend, and may run more than once
    in one process."""
    name = name.lower()
    member = AttentionBackendEnum.__members__.get(name.upper())
    if member is None:
        AttentionBackendEnum.register(name.upper(), vllm_backend)
    elif member.get_path() != vllm_backend:
        raise ValueError(
            f"vLLM already has an attention backend named {name.upper()}, at "
            f"{member.get_path()}."
        )
    _REGISTERED_BACKENDS.add(name)


def is_registered_attention_backend(backend: AttentionBackendEnum | None) -> bool:
    """Whether vLLM's backend is one of register_attention_backend."""
    return backend is not None and backend.name.lower() in _REGISTERED_BACKENDS


def check_attention_backend_config(vllm_config: "VllmConfig") -> None:
    """Raise if a registered backend is selected for an engine whose processes
    cannot rebuild its name.

    Enum members pickle by value, so a spawned or Ray process, which unpickles
    vllm_config before it loads the plugins, cannot rebuild a registered
    member. A forked process inherits it.

    TODO: support spawn and Ray."""
    backend = vllm_config.attention_config.backend
    if not is_registered_attention_backend(backend):
        return
    assert backend is not None
    name = backend.name.lower()
    parallel_config = vllm_config.parallel_config
    starts_processes = (
        vllm_envs.VLLM_ENABLE_V1_MULTIPROCESSING
        or parallel_config.distributed_executor_backend != "uni"
    )
    reasons = []
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        reasons.append("TPU_MULTIHOST_BACKEND=ray")
    if is_in_ray_actor():
        reasons.append("the engine runs in a Ray actor")
    if starts_processes and vllm_envs.VLLM_WORKER_MULTIPROC_METHOD == "spawn":
        reasons.append(
            "VLLM_WORKER_MULTIPROC_METHOD=spawn (vllm serve sets it when unset; "
            "set VLLM_WORKER_MULTIPROC_METHOD=fork)"
        )
    if reasons:
        raise ValueError(
            f"--attention-backend {name} supports a single host with forked "
            f"processes only, not: {', '.join(reasons)}."
        )
