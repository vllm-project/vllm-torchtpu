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
"""Raiden (`tpu_sync`) in-place RL weight synchronization for `TPUWorker`.

Binds live post-load PyTorch TPU `nn.Parameter` buffers to
`tpu_sync.api.torch.weight_synchronizer.WeightSynchronizer`, publishes unified
`(dp, tp)` / `(ep,)` `WorkUnitMetadata` with NUMA-local NIC striping, and
commits incoming weight generations in place on `h2d(uuid)`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import torch
from vllm.utils.network_utils import get_ip

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.rl.param_layout import (
    ALLOWED_SYNC_DTYPES,
    DOMAIN_EP,
    DOMAIN_REPLICATED,
    DOMAIN_TP,
    WEIGHT_FLIPPED_ATTR,
    BoundVariable,
    canonical_param_name,
    classify_parameter,
    collect_variables,
)
from vllm_torchtpu.utils import synchronize_tensors

__all__ = [
    "ALLOWED_SYNC_DTYPES",
    "BoundVariable",
    "DOMAIN_EP",
    "DOMAIN_REPLICATED",
    "DOMAIN_TP",
    "RaidenWorkerSync",
    "WEIGHT_FLIPPED_ATTR",
    "canonical_param_name",
    "classify_parameter",
    "collect_variables",
    "is_parallel_h2h_enabled",
    "local_ip",
]

logger = init_logger(__name__)

_ws_lib: Any = None
_RAIDEN_IMPORT_ERROR: Exception | None = None

_DP_AXIS = "dp"
_TP_AXIS = "tp"
_EP_COMBINED_AXIS = "dp,tp"
_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "127.0.0.1", "::", "[::]", "::1", "[::1]"})


def _get_ws_lib() -> Any:
    """Import `tpu_sync` lazily so CPU imports do not initialize PJRT."""
    global _ws_lib, _RAIDEN_IMPORT_ERROR
    if _ws_lib is None:
        try:
            from tpu_sync.api.torch import weight_synchronizer as ws_mod

            _ws_lib = ws_mod
            _RAIDEN_IMPORT_ERROR = None
        except (ImportError, OSError) as exc:  # pragma: no cover
            _RAIDEN_IMPORT_ERROR = exc
    return _ws_lib


def is_parallel_h2h_enabled() -> bool:
    return bool(envs.WEIGHT_SYNC_PARALLEL_H2H)


def _format_host(host: str) -> str:
    host = host.strip()
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        return f"[{host}]"
    return host


def local_ip() -> str:
    return _format_host(envs.RAIDEN_BIND_IP or get_ip())


def _routable_endpoint(addr: str, fallback_host: str | Callable[[], str]) -> str:
    host, sep, port = addr.rpartition(":")
    if not sep:
        return addr
    if host in _WILDCARD_HOSTS:
        resolved = fallback_host() if callable(fallback_host) else fallback_host
        return f"{_format_host(resolved)}:{port}"
    return f"{_format_host(host)}:{port}"


def _endpoint_host(addr: str) -> str:
    host, sep, _ = addr.rpartition(":")
    return "" if (not sep or host in _WILDCARD_HOSTS) else _format_host(host)


def _validate_topology(tp_rank: int, tp_size: int, dp_rank: int, dp_size: int) -> None:
    if tp_size < 1 or dp_size < 1:
        raise ValueError(
            f"tp_size and dp_size must be >= 1, got ({tp_size}, {dp_size})."
        )
    if not (0 <= tp_rank < tp_size):
        raise ValueError(
            f"tp_rank must satisfy 0 <= tp_rank < tp_size ({tp_size}), got {tp_rank}."
        )
    if not (0 <= dp_rank < dp_size):
        raise ValueError(
            f"dp_rank must satisfy 0 <= dp_rank < dp_size ({dp_size}), got {dp_rank}."
        )


def _check_bindable(var: BoundVariable) -> None:
    t = var.tensor
    if t.device.type != "tpu":
        raise ValueError(f"{var.name}: expected TPU tensor, got device={t.device}")
    if t.dtype not in ALLOWED_SYNC_DTYPES:
        raise ValueError(f"{var.name}: unsupported tensor dtype {t.dtype}.")
    if not t.is_contiguous() or t.storage_offset() != 0:
        raise ValueError(
            f"{var.name}: expected contiguous root view (offset={t.storage_offset()})"
        )
    if t.dim() < 1 or t.shape[0] == 0:
        raise ValueError(f"{var.name}: empty or 0-D tensor {tuple(t.shape)}")
    if t.data_ptr() == 0:
        raise ValueError(f"{var.name}: unmaterialized tensor (data_ptr=0)")


class RaidenWorkerSync:
    """Per-worker live parameter binding and metadata publisher for Raiden."""

    def __init__(
        self,
        job_name: str,
        *,
        worker_index: int = 0,
        tp_rank: int = 0,
        tp_size: int = 1,
        dp_rank: int = 0,
        dp_size: int = 1,
        parallelism: int = 4,
        bind_ip: str | None = None,
        auto_h2d: bool | None = None,
    ) -> None:
        self.job_name = job_name
        self.worker_index = int(worker_index)
        if self.worker_index < 0:
            raise ValueError(f"worker_index must be >= 0, got {self.worker_index}.")
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.dp_rank = int(dp_rank)
        self.dp_size = int(dp_size)
        _validate_topology(self.tp_rank, self.tp_size, self.dp_rank, self.dp_size)

        raw_bind_ip = (bind_ip if bind_ip is not None else envs.RAIDEN_BIND_IP).strip()
        self._bind_ip: str | None = raw_bind_ip or None
        self._parallelism = self.resolve_parallelism(parallelism)

        self._auto_h2d = (
            (not is_parallel_h2h_enabled()) if auto_h2d is None else bool(auto_h2d)
        )
        self._effective_auto_h2d = self._auto_h2d
        self.variables: list[BoundVariable] = []
        self._bound_ptrs: list[int] = []
        self._bound_specs: list[tuple[tuple[int, ...], torch.dtype]] = []
        self._sync: Any = None

    @staticmethod
    def resolve_parallelism(parallelism: int) -> int:
        env_par = envs.RAIDEN_PARALLELISM
        effective = int(env_par) if env_par is not None else int(parallelism)
        if effective < 1:
            raise ValueError(f"parallelism must be >= 1, got {effective}.")
        return effective

    @property
    def parallelism(self) -> int:
        return self._parallelism

    def matches_binding(
        self,
        *,
        tp_rank: int,
        tp_size: int,
        parallelism: int,
    ) -> bool:
        return (self.tp_rank, self.tp_size, self._parallelism) == (
            int(tp_rank),
            int(tp_size),
            self.resolve_parallelism(parallelism),
        )

    @property
    def bound(self) -> bool:
        return self._sync is not None and bool(self.variables)

    def _validate_variable_shards(self, variables: list[BoundVariable]) -> None:
        ep_size_phys = self.dp_size * self.tp_size
        for v in variables:
            if v.shard_dim is None:
                continue
            expected = ep_size_phys if v.shard_domain == DOMAIN_EP else self.tp_size
            if v.num_shards != expected:
                raise NotImplementedError(
                    f"{v.name}: {v.shard_domain.upper()} shard count {v.num_shards} "
                    f"does not match worker size {expected} "
                    f"(dp={self.dp_size}, tp={self.tp_size})."
                )

    def bind(self, model: torch.nn.Module, *, auto_h2d: bool | None = None) -> None:
        """Bind model parameters on this TPU worker to a 1-chip synchronizer."""
        _validate_topology(self.tp_rank, self.tp_size, self.dp_rank, self.dp_size)
        ws_lib = _get_ws_lib()
        if ws_lib is None:
            raise RuntimeError(
                f"{self.job_name}: tpu_sync is not importable: {_RAIDEN_IMPORT_ERROR}"
            )
        if auto_h2d is not None:
            requested = bool(auto_h2d)
            if self._sync is not None and requested != self._effective_auto_h2d:
                logger.warning(
                    "[raiden_sync] %s: ignoring auto_h2d=%s on rebind "
                    "(already bound with auto_h2d=%s).",
                    self.job_name,
                    requested,
                    self._effective_auto_h2d,
                )
            else:
                self._auto_h2d = requested

        variables = collect_variables(model, self.tp_size)
        for v in variables:
            _check_bindable(v)
        self._validate_variable_shards(variables)
        tensors = [v.tensor for v in variables]
        if tensors and tensors[0].device.type == "tpu":
            synchronize_tensors(tensors)

        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        ptrs = [t.data_ptr() for t in tensors]
        specs = [(tuple(t.shape), t.dtype) for t in tensors]
        if self._sync is not None and specs != self._bound_specs:
            self.shutdown()

        if self._sync is None:
            logger.info(
                "[raiden_sync] %s[dp=%d/%d,tp=%d/%d]: binding %d weight tensors "
                "(%.2f GiB local).",
                self.job_name,
                self.dp_rank,
                self.dp_size,
                self.tp_rank,
                self.tp_size,
                len(tensors),
                nbytes / 2**30,
            )
            self._sync = ws_lib.WeightSynchronizer(
                [[t] for t in tensors],
                local_port=0,
                parallelism=self._parallelism,
                listener_port=0,
                bind_ip=self._bind_ip.strip("[]") if self._bind_ip else None,
                unsafe_skip_buffer_lock=True,
                auto_h2d=self._auto_h2d,
                global_shard_indices=[0],
            )
            self._effective_auto_h2d = self._auto_h2d
        elif ptrs != self._bound_ptrs:
            logger.info(
                "[raiden_sync] %s[dp=%d/%d,tp=%d/%d]: re-binding %d tensors "
                "(%.2f GiB).",
                self.job_name,
                self.dp_rank,
                self.dp_size,
                self.tp_rank,
                self.tp_size,
                len(tensors),
                nbytes / 2**30,
            )
            self._sync.bind_weights([[t] for t in tensors])
        self.variables = variables
        self._bound_ptrs = ptrs
        self._bound_specs = specs

    def shutdown(self) -> None:
        sync = self._sync
        self._sync = None
        self.variables = []
        self._bound_ptrs = []
        self._bound_specs = []
        if sync is not None:
            sync.shutdown()

    def _require(self, op: str) -> Any:
        if self._sync is None:
            raise RuntimeError(f"{self.job_name}: bind() must run before {op}.")
        return self._sync

    def h2d(self, uuid: int | None = None) -> None:
        sync = self._require("h2d()")
        if (
            not isinstance(uuid, int)
            or isinstance(uuid, bool)
            or not (0 < uuid < 2**64)
        ):
            raise ValueError(
                f"{self.job_name}: h2d() requires integer uuid in (0, 2**64), "
                f"got {uuid!r}."
            )
        t0 = time.monotonic()
        sync.wait_for_transfer_completion(uuid)
        if not self._effective_auto_h2d:
            sync.h2d()
        torch.accelerator.synchronize()
        logger.info(
            "[raiden_sync] %s: h2d complete in %.2fs (uuid=%s).",
            self.job_name,
            time.monotonic() - t0,
            uuid,
        )

    def metrics(self) -> dict[str, Any]:
        return dict(self._sync.get_metrics()) if self._sync is not None else {}

    def checksums(self, sample: int = 3) -> dict[str, float | int]:
        if not self.variables:
            return {
                "__grand_total__": 0.0,
                "__tensor_count__": 0,
                "__element_count__": 0,
            }
        per_tensor = (
            torch.stack(
                [
                    v.tensor.detach().abs().sum(dtype=torch.float32)
                    for v in self.variables
                ]
            )
            .cpu()
            .tolist()
        )
        out: dict[str, float | int] = {
            v.name: float(val)
            for v, val in zip(self.variables[:sample], per_tensor[:sample])
        }
        out["__grand_total__"] = float(math.fsum(per_tensor))
        out["__tensor_count__"] = len(self.variables)
        out["__element_count__"] = int(sum(v.tensor.numel() for v in self.variables))
        return out

    def describe(self) -> str:
        lines = [
            f"{self.job_name}[dp={self.dp_rank}/{self.dp_size},"
            f"tp={self.tp_rank}/{self.tp_size}] {len(self.variables)} variables"
        ]
        for v in self.variables:
            lines.append(
                f"  {v.name}: local={v.local_shape} global={v.global_shape} "
                f"domain={v.shard_domain} dim={v.shard_dim} "
                f"flipped={v.flipped} dtype={v.tensor.dtype} ({v.owner_cls})"
            )
        return "\n".join(lines)

    def param_layouts(self) -> dict[str, dict[str, Any]]:
        self._require("param_layouts()")
        return {
            v.name: v.layout_dict(layer_idx=idx) for idx, v in enumerate(self.variables)
        }

    def variable_dicts(self) -> list[dict[str, Any]]:
        self._validate_variable_shards(self.variables)
        ep_rank = self.dp_rank * self.tp_size + self.tp_rank
        ep_axis = _EP_COMBINED_AXIS if self.dp_size > 1 else _TP_AXIS
        out: list[dict[str, Any]] = []
        for idx, v in enumerate(self.variables):
            ndim = len(v.global_shape)
            mesh_shape = [1] * ndim
            sharding_spec = [""] * ndim
            global_shard_indices = [0]
            if v.shard_dim is not None:
                mesh_shape[v.shard_dim] = v.num_shards
                if v.shard_domain == DOMAIN_EP:
                    sharding_spec[v.shard_dim] = ep_axis
                    global_shard_indices = [ep_rank]
                else:
                    sharding_spec[v.shard_dim] = _TP_AXIS
                    global_shard_indices = [self.tp_rank]
            out.append(
                {
                    "name": v.name,
                    "shape": list(v.global_shape),
                    "mesh_shape": mesh_shape,
                    "layout": list(reversed(range(ndim))),
                    "item_size": v.tensor.element_size(),
                    "layer_idx": idx,
                    "sharding_spec": sharding_spec,
                    "global_shard_indices": global_shard_indices,
                }
            )
        return out

    def metadata_dict(self) -> dict[str, Any]:
        """Wire-safe registration metadata shaped for `WorkUnitMetadata.from_dict`."""
        sync = self._require("metadata_dict()")
        _validate_topology(self.tp_rank, self.tp_size, self.dp_rank, self.dp_size)
        num_shards = int(sync.num_shards)
        candidates_per_shard: list[list[str]] = [[] for _ in range(num_shards)]
        fallback_host: str | None = (
            _format_host(self._bind_ip) if self._bind_ip else None
        )

        def _fallback() -> str:
            nonlocal fallback_host
            if fallback_host is None:
                fallback_host = _format_host(local_ip())
            return fallback_host

        for ep in sync.get_local_endpoints():
            raw_addr = str(ep.get("endpoint", "")).strip()
            if not raw_addr:
                continue
            addr = _routable_endpoint(raw_addr, _fallback)
            for s in ep.get("shards", []):
                s_idx = int(s)
                if 0 <= s_idx < num_shards and addr not in candidates_per_shard[s_idx]:
                    candidates_per_shard[s_idx].append(addr)

        replica_rank = (
            self.worker_index * self.dp_size * self.tp_size
            + self.dp_rank * self.tp_size
            + self.tp_rank
        )
        shards_list = [
            cands[(replica_rank + s_idx) % len(cands)]
            if cands
            else f"{_fallback()}:{sync.local_port}"
            for s_idx, cands in enumerate(candidates_per_shard)
        ]
        control_host = (
            _endpoint_host(shards_list[0]) if shards_list else ""
        ) or _fallback()
        variables = self.variable_dicts()
        for var in variables:
            if len(var["global_shard_indices"]) != len(shards_list):
                raise RuntimeError(
                    f"{self.job_name}: variable {var['name']!r} publishes "
                    f"{len(var['global_shard_indices'])} global_shard_indices "
                    f"for {len(shards_list)} local shard(s)."
                )
        return {
            "unit": {"job_name": self.job_name, "job_replica_id": str(replica_rank)},
            "shards": shards_list,
            "control_plane_rpc_address": (
                f"{control_host}:{sync.listener_port}" if sync.listener_port else ""
            ),
            "mesh_shape": (
                [self.dp_size, self.tp_size] if self.dp_size > 1 else [self.tp_size]
            ),
            "mesh_axes": [_DP_AXIS, _TP_AXIS] if self.dp_size > 1 else [_TP_AXIS],
            "variables": variables,
            "auto_h2d": bool(self._effective_auto_h2d),
        }
