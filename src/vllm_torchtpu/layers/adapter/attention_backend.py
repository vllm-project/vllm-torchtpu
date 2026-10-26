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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from vllm import envs as vllm_envs
from vllm.ray.lazy_utils import is_in_ray_actor
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class HostStepInputs:
    """The host arrays TPUModelRunner._prepare_inputs filled for one step,
    requests in kernel order (decode first); block_tables and page_sizes are
    keyed by attention KV cache group id."""

    num_decode: int
    # i32[num_reqs + 1] query starts; i32[num_reqs] kv lengths incl. this step.
    query_start_loc: np.ndarray
    seq_lens: np.ndarray
    # i32[num_reqs, pages_per_seq] kernel page ids.
    block_tables: Mapping[int, np.ndarray]
    page_sizes: Mapping[int, int]
    num_tokens: int
    max_num_seqs: int
    # The largest token bucket. One backbone trace serves every bucket, so a
    # backend array's shape may follow max_num_tokens, never num_tokens.
    max_num_tokens: int


class TPUAttentionBackend:
    """The runner side of an attention backend. This base does nothing and is
    what FLASH_ATTN and CUSTOM get; a backend with per-step metadata sets
    has_step_metadata and overrides prepare_step and warmup_metadata."""

    has_step_metadata: ClassVar[bool] = False

    def check_config(self, vllm_config: "VllmConfig") -> None:
        """Raise for a configuration the backend cannot serve; it may adjust
        it, e.g. the block size."""

    def conv1d_gdn_kernel(self) -> Callable[..., Any] | None:
        """The fused conv1d + GDN kernel the compact pool's GDN layers run, in
        GDN v3's signature; None runs GDN v3."""
        return None

    def prepare_step(self, step: HostStepInputs) -> Mapping[int, tuple[Any, ...]]:
        """The backend's host arrays of one step per KV cache group id. The
        runner uploads them as AttentionMetadata.backend_metadata, which the
        attention impl appends to its kernel call."""
        raise NotImplementedError

    def warmup_metadata(
        self, page_sizes: Mapping[int, int], *, max_num_seqs: int, max_num_tokens: int
    ) -> Mapping[int, tuple[Any, ...]]:
        """prepare_step's output for a warmup step: same structure and shapes,
        doing no work."""
        raise NotImplementedError


_REGISTERED_BACKENDS: dict[str, str | None] = {}


def register_attention_backend(
    name: str, *, vllm_backend: str, attention: str | None = None
) -> None:
    """Register a new backend for `--attention-backend <name>`.

    vllm_backend is the class path of the vLLM attention backend the attention
    layers get, a PallasAttentionBackend subclass. attention is the class path
    of the runner's TPUAttentionBackend, if not the base. Both load only once
    the name is selected.

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
    _REGISTERED_BACKENDS[name] = attention


def is_registered_attention_backend(backend: AttentionBackendEnum | None) -> bool:
    """Whether vLLM's backend is one of register_attention_backend."""
    return backend is not None and backend.name.lower() in _REGISTERED_BACKENDS


def create_attention_backend(vllm_config: "VllmConfig") -> TPUAttentionBackend:
    """The runner's TPUAttentionBackend for the selected backend."""
    backend = vllm_config.attention_config.backend
    if not is_registered_attention_backend(backend):
        return TPUAttentionBackend()
    assert backend is not None
    path = _REGISTERED_BACKENDS[backend.name.lower()]
    return TPUAttentionBackend() if path is None else resolve_obj_by_qualname(path)()


def check_attention_backend_config(vllm_config: "VllmConfig") -> None:
    """Raise if a registered backend is selected for an engine whose processes
    cannot rebuild its name, or its check_config rejects the configuration.

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
    if parallel_config.data_parallel_backend == "ray":
        reasons.append("--data-parallel-backend ray")
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
    create_attention_backend(vllm_config).check_config(vllm_config)
