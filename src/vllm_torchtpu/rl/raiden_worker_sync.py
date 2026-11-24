# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Raiden (`tpu_sync`) in-place RL weight synchronization for `TPUWorker`.

Derives per-parameter shard layouts directly from the declarative
``weight_mapping`` manifest (``manifest_generator.generate_manifest_dict``),
binds live post-load PyTorch TPU ``nn.Parameter`` buffers to
``tpu_sync.api.torch.weight_synchronizer.WeightSynchronizer``, publishes
``WorkUnitMetadata`` with NUMA-local NIC striping, and commits incoming weight
generations in place on ``h2d(uuid)``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from vllm.utils.network_utils import get_ip

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.rl.manifest_generator import (
    ALLOWED_SYNC_DTYPES,
    generate_manifest_dict,
)
from vllm_torchtpu.rl.weight_conversion import TMP_KEY_PREFIX, validate_manifest

__all__ = [
    "BoundVariable",
    "DEFAULT_RAIDEN_PARALLELISM",
    "RaidenWorkerSync",
    "canonical_param_name",
    "collect_variables_from_manifest",
]

DOMAIN_EP = "ep"
DOMAIN_TP = "tp"
DOMAIN_REPLICATED = "replicated"

DEFAULT_RAIDEN_PARALLELISM = 4

# Transitional compatibility shim stripping leading `language_model.`, `base.`,
# and `model.` wrapper prefixes for external RL orchestrators that canonicalize
# source keys. Remove once orchestrators match exact manifest parameter keys.
_WRAPPER_ROOTS = ("language_model", "base", "model")

logger = init_logger(__name__)

_ws_lib: Any = None
_RAIDEN_IMPORT_ERROR: Exception | None = None

_WILDCARD_HOSTS = frozenset(
    {"", "0.0.0.0", "127.0.0.1", "::", "[::]", "::1", "[::1]", "*"}
)
_DP_AXIS = "dp"
_TP_AXIS = "tp"
_EP_COMBINED_AXIS = "dp,tp"


@dataclass
class BoundVariable:
    """Live TPU parameter bound to Raiden with its manifest-derived shard layout."""

    name: str
    tensor: torch.Tensor
    shard_dim: int | None
    num_shards: int
    shard_domain: str
    global_shape: tuple[int, ...]

    @classmethod
    def create(
        cls,
        *,
        name: str,
        tensor: torch.Tensor,
        shard_dim: int | None = None,
        num_shards: int = 1,
        shard_domain: str = DOMAIN_REPLICATED,
    ) -> BoundVariable:
        if shard_dim is None or num_shards <= 1:
            shard_dim = None
            num_shards = 1
            shard_domain = DOMAIN_REPLICATED
        global_shape = [int(d) for d in tensor.shape]
        if shard_dim is not None:
            global_shape[shard_dim] *= num_shards
        return cls(
            name=name,
            tensor=tensor,
            shard_dim=shard_dim,
            num_shards=num_shards,
            shard_domain=shard_domain,
            global_shape=tuple(global_shape),
        )


def canonical_param_name(name: str) -> str:
    """Strip leading wrapper prefixes for transitional Tunix key compatibility."""
    segments = [comp for raw in name.split(".") if (comp := raw.strip("'\""))]
    while segments and segments[0] in _WRAPPER_ROOTS:
        segments.pop(0)
    return ".".join(segments)


def collect_variables_from_manifest(
    model: torch.nn.Module,
    manifest: Mapping[str, Any] | str | None = None,
    *,
    parallel_config: Any = None,
    model_config: Any = None,
    vllm_config: Any = None,
    tp_size: int | None = None,
    dp_size: int | None = None,
    ep_size: int | None = None,
) -> list[BoundVariable]:
    """Derive bindable Raiden variables directly from the weight_mapping manifest."""
    if manifest is None:
        manifest_dict = generate_manifest_dict(
            model,
            parallel_config=parallel_config,
            model_config=model_config,
            vllm_config=vllm_config,
            tp_size=tp_size,
            dp_size=dp_size,
            ep_size=ep_size,
        )
    else:
        manifest_dict = validate_manifest(manifest)

    params_by_name = dict(model.named_parameters(remove_duplicate=False))
    stripes = manifest_dict["stripes"]
    tp_stripe = int(stripes.get("tp_size", 1))
    ep_stripe = int(stripes.get("ep_size", 1))
    moe_tp_stripe = int(stripes.get("moe_tp_size", tp_stripe))

    # Synchronize all TPU parameters once before reading data_ptr() / binding.
    # Lazy import keeps `vllm_torchtpu.rl` free of top-level JAX imports.
    tpu_tensors = [
        p.data
        for _, p in model.named_parameters(remove_duplicate=True)
        if p.data.device.type == "tpu"
    ]
    if tpu_tensors:
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors(tpu_tensors)

    out: list[BoundVariable] = []
    seen_ptrs: dict[int, str] = {}
    seen_names: dict[str, str] = {}

    for rule in manifest_dict["rules"]:
        real_targets = [
            k for k in rule["target_keys"] if not k.startswith(TMP_KEY_PREFIX)
        ]
        if not real_targets:
            continue
        sharding: list[str | None] = rule.get("sharding") or []
        sharded_axes = [(i, ax) for i, ax in enumerate(sharding) if ax is not None]
        if len(sharded_axes) > 1:
            raise NotImplementedError(
                f"{real_targets}: multi-axis sharding {sharding} is not "
                "supported by Raiden (expected at most one sharded axis)."
            )

        for target_key in real_targets:
            if target_key not in params_by_name:
                raise KeyError(
                    f"Manifest target key {target_key!r} not found in model parameters."
                )
            param = params_by_name[target_key]
            raw = param.data if isinstance(param, torch.nn.Parameter) else param
            if len(sharding) != raw.dim():
                raise ValueError(
                    f"{target_key}: manifest sharding rank {len(sharding)} != "
                    f"tensor rank {raw.dim()}."
                )

            canon = canonical_param_name(target_key)
            if not canon:
                raise ValueError(
                    f"Parameter {target_key!r} canonicalized to an empty key."
                )
            ptr = raw.data_ptr()
            if ptr != 0 and ptr in seen_ptrs:
                logger.info_once(
                    "[raiden_sync] Skipping tied parameter %s sharing buffer with %s.",
                    canon,
                    seen_ptrs[ptr],
                )
                continue
            if canon in seen_names:
                raise ValueError(
                    f"Parameters {seen_names[canon]!r} and {target_key!r} both "
                    f"canonicalize to {canon!r}."
                )

            if sharded_axes:
                shard_dim, axis_name = sharded_axes[0]
                if axis_name == DOMAIN_EP:
                    shard_domain = DOMAIN_EP
                    num_shards = ep_stripe
                elif axis_name == DOMAIN_TP:
                    shard_domain = DOMAIN_TP
                    num_shards = tp_stripe
                elif axis_name == "moe_tp":
                    shard_domain = DOMAIN_TP
                    num_shards = moe_tp_stripe
                else:
                    raise NotImplementedError(
                        f"{target_key}: unsupported manifest sharding axis "
                        f"{axis_name!r}."
                    )
            else:
                shard_dim = None
                shard_domain = DOMAIN_REPLICATED
                num_shards = 1

            var = BoundVariable.create(
                name=canon,
                tensor=raw,
                shard_dim=shard_dim,
                num_shards=num_shards,
                shard_domain=shard_domain,
            )
            if ptr != 0:
                seen_ptrs[ptr] = var.name
            seen_names[var.name] = target_key
            out.append(var)

    out.sort(key=lambda v: v.name)
    return out


def _get_ws_lib() -> Any:
    global _ws_lib, _RAIDEN_IMPORT_ERROR
    if _ws_lib is None:
        try:
            from tpu_sync.api.torch import weight_synchronizer as ws_mod

            _ws_lib = ws_mod
            _RAIDEN_IMPORT_ERROR = None
        except (ImportError, OSError) as exc:  # pragma: no cover
            _RAIDEN_IMPORT_ERROR = exc
    return _ws_lib


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
        parallelism: int = DEFAULT_RAIDEN_PARALLELISM,
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

        self._auto_h2d = self.resolve_auto_h2d(auto_h2d)
        self.variables: list[BoundVariable] = []
        self._bound_ptrs: list[int] = []
        self._bound_specs: list[tuple[tuple[int, ...], torch.dtype]] = []
        self._sync: Any = None

    @staticmethod
    def resolve_auto_h2d(auto_h2d: bool | None) -> bool:
        return (
            (not bool(envs.WEIGHT_SYNC_PARALLEL_H2H))
            if auto_h2d is None
            else bool(auto_h2d)
        )

    @staticmethod
    def resolve_parallelism(parallelism: int) -> int:
        env_par = envs.RAIDEN_PARALLELISM
        if env_par is not None:
            effective = int(env_par)
            if effective != int(parallelism):
                logger.warning_once(
                    "[raiden_sync] RAIDEN_PARALLELISM=%d overrides caller "
                    "parallelism=%d.",
                    effective,
                    int(parallelism),
                )
        else:
            effective = int(parallelism)
        if effective < 1:
            raise ValueError(f"parallelism must be >= 1, got {effective}.")
        return effective

    def matches_binding(
        self,
        *,
        job_name: str,
        worker_index: int,
        tp_rank: int,
        tp_size: int,
        dp_rank: int,
        dp_size: int,
        parallelism: int,
        auto_h2d: bool | None = None,
    ) -> bool:
        return (
            self.job_name,
            self.worker_index,
            self.tp_rank,
            self.tp_size,
            self.dp_rank,
            self.dp_size,
            self._parallelism,
            self._auto_h2d,
        ) == (
            job_name,
            int(worker_index),
            int(tp_rank),
            int(tp_size),
            int(dp_rank),
            int(dp_size),
            self.resolve_parallelism(parallelism),
            self.resolve_auto_h2d(auto_h2d),
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

    def bind(
        self,
        model: torch.nn.Module,
        *,
        manifest: Mapping[str, Any] | str | None = None,
        parallel_config: Any = None,
        model_config: Any = None,
        vllm_config: Any = None,
    ) -> None:
        """Bind model parameters on this TPU worker to a 1-chip synchronizer."""
        ws_lib = _get_ws_lib()
        if ws_lib is None:
            raise RuntimeError(
                f"{self.job_name}: tpu_sync is not importable: {_RAIDEN_IMPORT_ERROR}"
            )

        variables = collect_variables_from_manifest(
            model,
            manifest=manifest,
            parallel_config=parallel_config,
            model_config=model_config,
            vllm_config=vllm_config,
            tp_size=self.tp_size,
            dp_size=self.dp_size,
        )
        for v in variables:
            _check_bindable(v)
        self._validate_variable_shards(variables)
        tensors = [v.tensor for v in variables]

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
                # Safe only while serving is paused inside an active weight-update
                # session (enforced by TPUWorker.raiden_h2d) and synced before resume.
                unsafe_skip_buffer_lock=True,
                auto_h2d=self._auto_h2d,
                global_shard_indices=[0],
            )
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

    def h2d(self, uuid: int) -> None:
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
        if not self._auto_h2d:
            sync.h2d()
        # Whole-device sync is required: PyTorch tensor events do not track
        # out-of-band Raiden DMA writes, so synchronize_tensors would not cover them.
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
                f"  {v.name}: local={tuple(v.tensor.shape)} global={v.global_shape} "
                f"domain={v.shard_domain} dim={v.shard_dim} dtype={v.tensor.dtype}"
            )
        return "\n".join(lines)

    def variable_dicts(self) -> list[dict[str, Any]]:
        from vllm_torchtpu.utils import ep_rank as compute_ep_rank

        ep_rank = compute_ep_rank(
            dp_rank=self.dp_rank, tp_rank=self.tp_rank, tp_size=self.tp_size
        )
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
        num_shards = int(sync.num_shards)
        candidates_per_shard: list[list[str]] = [[] for _ in range(num_shards)]
        fallback_host: str | None = (
            _format_host(self._bind_ip) if self._bind_ip else None
        )

        def _fallback() -> str:
            nonlocal fallback_host
            if fallback_host is None:
                fallback_host = local_ip()
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
            "control_plane_rpc_address": (
                f"{control_host}:{sync.listener_port}" if sync.listener_port else ""
            ),
            "shards": shards_list,
            "mesh_shape": [self.dp_size, self.tp_size]
            if self.dp_size > 1
            else [self.tp_size],
            "mesh_axes": [_DP_AXIS, _TP_AXIS] if self.dp_size > 1 else [_TP_AXIS],
            "variables": variables,
        }
