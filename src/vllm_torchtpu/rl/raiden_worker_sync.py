# SPDX-License-Identifier: Apache-2.0
"""Raiden (tpu-sync) weight synchronization for RL rollouts on torch_tpu.

The destination side of a trainer -> sampler weight-sync round, bound inside
each vLLM TP worker process. In-process counterpart to tpu-inference's
``tpu_inference/rl/raiden_worker_sync.py`` (JAX) and tunix's
``RaidenSynchronizer``; driven by tunix ``RLVllmSampler`` through
``collective_rpc("bind_raiden_sync" | "get_raiden_metadata" | "raiden_h2d" |
"raiden_metrics")`` (see ``TPUWorker``).

Contract with the trainer
-------------------------
The transfer engine (``tpu_sync``) pairs source and destination variables by
*name* and moves bytes with N-D slice resharding only: no transposes, no
fusion, no dtype casts. On TPU, vllm-torchtpu stores most dense weights
transposed (``[n_in, n_out_per_rank]``), routed experts as
``[E_local, H, 2I]`` / ``[E_local, I, H]`` and keeps vLLM's fused, per-segment
TP-sharded parameters (``qkv_proj``, ``in_proj_qkvz``, ``in_proj_ba``,
``shared_expert.gate_up_proj``, ``conv1d``). tpu-sync also refuses sliced
views, so each fused parameter is one variable.

Every bound tensor is therefore published as

    global_shape = local_shape with ``shard_dim`` scaled by ``tp_size``
    sharding     = ``tp`` on ``shard_dim`` (or replicated), shard index = tp rank

i.e. the global tensor is the **rank-major concatenation of the rank-local
blocks** along ``shard_dim``. The trainer-side transform
(``maxtext.integration.tunix.weight_mapping.qwen3_5_vllm_torch``) produces
exactly these tensors, so the engine's pure slicing lands bytes in place.

Variable names are the vLLM parameter names with an optional
``language_model.`` prefix stripped (``model.layers.3.self_attn.qkv_proj.weight``,
``model.layers.0.mlp.experts.w13_weight``, ``lm_head.weight`` ...). The
``visual.*`` tower is never bound.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)

try:  # tpu_sync is a hard dependency of vllm-torchtpu, but keep import errors readable.
    from tpu_sync.api.torch import weight_synchronizer as _ws_lib

    _RAIDEN_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - environment dependent
    _ws_lib = None
    _RAIDEN_IMPORT_ERROR = e

_TRUTHY = {"1", "true", "yes", "on"}

# Prefixes that are never part of the trainer<->sampler contract.
_SKIP_PREFIXES = ("visual.", "mtp.")
# vLLM's multimodal wrapper nests the LM under `language_model.`; the trainer
# publishes LM names without it.
_STRIP_PREFIXES = ("language_model.",)
# Plain nn.Parameters that vLLM shards with `sharded_weight_loader(0)`.
_HEAD_SHARDED_PARAM_SUFFIXES = (".A_log", ".dt_bias")


def is_parallel_h2h_enabled() -> bool:
    return os.getenv("WEIGHT_SYNC_PARALLEL_H2H", "0").strip().lower() in _TRUTHY


def local_ip() -> str:
    ip = os.getenv("RAIDEN_BIND_IP") or os.getenv("TPU_RAIDEN_BIND_IP")
    if ip:
        return ip
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return socket.gethostbyname(socket.gethostname())


@dataclass
class BoundVariable:
    """One destination tensor and how it tiles the trainer's global tensor."""

    name: str
    tensor: torch.Tensor
    shard_dim: int | None  # None => replicated on every rank
    global_shape: tuple[int, ...]
    source: str = ""  # debugging aid: owning module class

    @property
    def local_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)


# vLLM 0.32 nests the expert weights one level deeper than the HF checkpoint
# (`mlp.experts.routed_experts.w13_weight`); publish them under the HF-style
# `mlp.experts.w13_weight` the trainer emits.
_NAME_REWRITES = ((".experts.routed_experts.", ".experts."),)


def _canonical_name(name: str) -> str:
    for p in _STRIP_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
    for old, new in _NAME_REWRITES:
        name = name.replace(old, new)
    return name


def _is_skipped(name: str) -> bool:
    return any(name.startswith(p) for p in _SKIP_PREFIXES)


def _owner(model: torch.nn.Module, param_name: str) -> tuple[torch.nn.Module, str]:
    mod_path, _, attr = param_name.rpartition(".")
    module = model.get_submodule(mod_path) if mod_path else model
    return module, attr


def _flipped(module: torch.nn.Module) -> bool:
    try:
        from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
    except Exception:  # pragma: no cover
        return False
    return bool(getattr(module, WEIGHT_FLIPPED_ATTR, False))


def classify_parameter(
    model: torch.nn.Module,
    param_name: str,
    param: torch.Tensor,
    tp_size: int,
) -> tuple[int | None, str]:
    """Returns (shard_dim or None, owner class name) for ``param``.

    Decides which tensor dimension holds this TP rank's block. Derived from the
    owning vLLM layer type plus vllm-torchtpu's transposed-weight marker, so it
    follows the engine's actual storage layout rather than the HF checkpoint.
    """
    module, attr = _owner(model, param_name)
    cls = type(module).__name__
    mro = {c.__name__ for c in type(module).__mro__}

    if tp_size == 1:
        return None, cls

    # --- vocab-parallel embedding / lm_head: dim 0 (no transpose on TPU) ---
    if "VocabParallelEmbedding" in mro or "ParallelLMHead" in mro:
        if int(getattr(module, "tp_size", tp_size)) == 1:
            return None, cls
        org = int(getattr(module, "org_vocab_size", param.shape[0] * tp_size))
        if param.shape[0] * tp_size != org:
            raise ValueError(
                f"{param_name}: padded vocab shard {tuple(param.shape)} x tp={tp_size}"
                f" != org_vocab_size={org}; padded vocab is not supported by weight sync"
            )
        return 0, cls

    # --- routed experts (FusedMoE / RoutedExperts): expert dim under EP ---
    if "FusedMoE" in mro or "RoutedExperts" in mro or attr in ("w13_weight", "w2_weight"):
        n_global = int(
            getattr(module, "global_num_experts", 0)
            or getattr(module, "num_experts", 0)
            or getattr(module, "n_routed_experts", 0)
            or 0
        )
        ep_size = int(getattr(module, "ep_size", 1) or 1)
        moe_tp = int(getattr(module, "tp_size", 1) or 1)
        if (n_global and param.shape[0] * tp_size == n_global) or (ep_size == tp_size and moe_tp == 1):
            return 0, cls  # rank r owns experts [E_local*r, E_local*(r+1))
        if (n_global and param.shape[0] == n_global) or (ep_size == 1 and moe_tp == tp_size):
            # TP-sharded intermediate: stored [E, H, 2I_local] / [E, I_local, H].
            return (2 if attr == "w13_weight" else 1), cls
        raise ValueError(
            f"{param_name}: unsupported MoE parallelism ep_size={ep_size} tp_size={moe_tp}"
            f" global_experts={n_global} local_shape={tuple(param.shape)} (worker tp={tp_size})"
        )

    # --- dense linears ---
    if "LinearBase" in mro:
        if "ReplicatedLinear" in mro or int(getattr(module, "tp_size", tp_size)) == 1:
            return None, cls
        flipped = _flipped(module)
        if attr != "weight":  # bias etc. (not used by Qwen3.5-35B-A3B)
            if "RowParallelLinear" in mro:
                return None, cls
            return 0, cls
        if param.dim() == 3:
            # conv1d [channels_per_rank, 1, kernel] (GdnColumnParallelLinear)
            return 0, cls
        if "RowParallelLinear" in mro:
            # vLLM [out, in_pp]; TPU transposed [in_pp, out]
            return (0 if flipped else 1), cls
        # ColumnParallel family (QKV, Merged, Gdn, plain): vLLM [out_pp, in];
        # TPU transposed [in, out_pp].
        return (1 if flipped else 0), cls

    # --- plain parameters ---
    if any(param_name.endswith(s) for s in _HEAD_SHARDED_PARAM_SUFFIXES):
        return 0, cls
    # Norm scales, gates and everything else: replicated.
    return None, cls


def collect_variables(
    model: torch.nn.Module, tp_size: int, tp_rank: int
) -> list[BoundVariable]:
    """Enumerates the model's LM parameters as destination variables."""
    del tp_rank
    out: list[BoundVariable] = []
    for name, param in model.named_parameters():
        if _is_skipped(name):
            continue
        tensor = param.data
        shard_dim, cls = classify_parameter(model, name, tensor, tp_size)
        gshape = list(tensor.shape)
        if shard_dim is not None:
            gshape[shard_dim] *= tp_size
        out.append(
            BoundVariable(
                name=_canonical_name(name),
                tensor=tensor,
                shard_dim=shard_dim,
                global_shape=tuple(gshape),
                source=cls,
            )
        )
    if not out:
        raise RuntimeError("no bindable parameters found on the model")
    # The controller pairs variables by name but stamps every push with the
    # *source's* positional layer index, which the receiver uses to pick its
    # host staging buffer. Both sides must therefore enumerate variables in
    # the same order. The trainer binds a JAX pytree, whose dict keys flatten
    # in sorted order per level; sorting the dotted path component-wise
    # reproduces that order exactly (`lm_head.weight`, `model.embed_tokens.…`,
    # `model.layers.0.…`, `model.layers.1.…`, `model.layers.10.…`, …).
    out.sort(key=lambda v: tuple(v.name.split(".")))
    return out


def _check_bindable(var: BoundVariable) -> None:
    t = var.tensor
    if t.device.type != "tpu":
        raise ValueError(f"{var.name}: not on TPU ({t.device})")
    if not t.is_contiguous():
        raise ValueError(f"{var.name}: not contiguous")
    if t.storage_offset() != 0:
        raise ValueError(f"{var.name}: sliced view (storage_offset={t.storage_offset()})")
    if t.dim() < 1 or t.shape[0] == 0:
        raise ValueError(f"{var.name}: empty / 0-d tensor {tuple(t.shape)}")


def _l1(t: torch.Tensor) -> float:
    return float(t.detach().abs().sum(dtype=torch.float32).cpu().item())


class RaidenWorkerSync:
    """One TP worker's live weights on the Raiden transport + wire-safe metadata."""

    def __init__(
        self,
        job_name: str,
        *,
        worker_index: int = 0,
        tp_rank: int = 0,
        tp_size: int = 1,
        parallelism: int = 4,
        bind_ip: str | None = None,
        auto_h2d: bool | None = None,
    ):
        self.job_name = job_name
        self.worker_index = worker_index
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.ip = bind_ip or local_ip()
        self._parallelism = int(os.getenv("RAIDEN_PARALLELISM", str(parallelism)))
        self._auto_h2d: bool = (
            (not is_parallel_h2h_enabled()) if auto_h2d is None else bool(auto_h2d)
        )
        self._effective_auto_h2d = self._auto_h2d
        self.variables: list[BoundVariable] = []
        self._bound_ptrs: list[int] = []
        self._sync: Any = None

    # ------------------------------------------------------------------ bind
    @property
    def bound(self) -> bool:
        return bool(self.variables)

    def bind(self, model: torch.nn.Module, *, auto_h2d: bool | None = None) -> None:
        if _ws_lib is None:
            raise RuntimeError(
                f"{self.job_name}: tpu_sync is not importable; cannot bind weight"
                f" synchronizer. Original error: {_RAIDEN_IMPORT_ERROR}"
            )
        if auto_h2d is not None:
            if self._sync is not None and bool(auto_h2d) != self._effective_auto_h2d:
                logger.warning(
                    "%s: ignoring auto_h2d=%s on rebind (synchronizer created with %s)",
                    self.job_name, auto_h2d, self._effective_auto_h2d,
                )
            else:
                self._auto_h2d = bool(auto_h2d)

        variables = collect_variables(model, self.tp_size, self.tp_rank)
        for v in variables:
            _check_bindable(v)
        tensors = [v.tensor for v in variables]
        # Flush deferred (DEFER_AND_FUSE) work so every buffer is materialized
        # before the C++ side takes its PjRt handle.
        try:
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors(tensors)
        except Exception:  # pragma: no cover
            torch.accelerator.synchronize()

        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        logger.info(
            "%s[tp%d/%d]: binding %d weight tensors, %.1f GiB local",
            self.job_name, self.tp_rank, self.tp_size, len(tensors), nbytes / 2**30,
        )
        ptrs = [t.data_ptr() for t in tensors]
        if self._sync is None:
            self._sync = _ws_lib.WeightSynchronizer(
                [[t] for t in tensors],
                local_port=0,
                parallelism=self._parallelism,
                listener_port=0,
                bind_ip=None,
                unsafe_skip_buffer_lock=True,
                auto_h2d=self._auto_h2d,
                global_shard_indices=[self.tp_rank],
            )
            self._effective_auto_h2d = self._auto_h2d
        elif ptrs != self._bound_ptrs or len(tensors) != len(self.variables):
            logger.info("%s: parameter buffers changed; re-binding", self.job_name)
            self._sync.bind_weights([[t] for t in tensors])
        self.variables = variables
        self._bound_ptrs = ptrs

    def refresh_if_moved(self, model: torch.nn.Module) -> bool:
        """Re-binds when any parameter has been re-materialized since bind()."""
        if self._sync is None:
            return False
        current = {
            _canonical_name(n): p.data
            for n, p in model.named_parameters()
            if not _is_skipped(n)
        }
        moved = [
            v.name for v in self.variables
            if v.name in current and current[v.name].data_ptr() != v.tensor.data_ptr()
        ]
        if moved:
            logger.warning(
                "%s: %d parameters were re-materialized since bind (e.g. %s); re-binding",
                self.job_name, len(moved), moved[:3],
            )
            self.bind(model)
            return True
        return False

    def _require(self, op: str) -> Any:
        if self._sync is None:
            raise RuntimeError(f"{self.job_name}: bind() must run before {op}")
        return self._sync

    # ------------------------------------------------------------------- h2d
    def h2d(self, uuid: int | None = None) -> None:
        """Blocks until the transfer that just landed is visible on device."""
        sync = self._require("h2d()")
        t0 = time.monotonic()
        if not self._effective_auto_h2d:
            if uuid is None or int(uuid) <= 0:
                raise ValueError(
                    f"{self.job_name}: h2d() with auto_h2d=False requires a transfer uuid > 0"
                )
            sync.wait_for_transfer_completion(int(uuid))
            sync.h2d()
        else:
            sync.wait_for_transfer_completion(uuid)
        torch.accelerator.synchronize()
        logger.info("%s: h2d complete in %.2fs (uuid=%s)", self.job_name, time.monotonic() - t0, uuid)

    def metrics(self) -> dict:
        if self._sync is None:
            return {}
        return dict(self._sync.get_metrics())

    def checksums(self, sample: int = 3) -> dict:
        out = {v.name: _l1(v.tensor) for v in self.variables[:sample]}
        out["__grand_total__"] = float(sum(_l1(v.tensor) for v in self.variables))
        out["__tensor_count__"] = len(self.variables)
        out["__element_count__"] = int(sum(v.tensor.numel() for v in self.variables))
        return out

    # -------------------------------------------------------------- metadata
    def variable_dicts(self) -> list[dict]:
        out = []
        for idx, v in enumerate(self.variables):
            nd = len(v.global_shape)
            mesh_shape = [1] * nd
            spec = [""] * nd
            gsi = [0]
            if v.shard_dim is not None:
                mesh_shape[v.shard_dim] = self.tp_size
                spec[v.shard_dim] = "tp"
                gsi = [self.tp_rank]
            out.append({
                "name": v.name,
                "shape": list(v.global_shape),
                "mesh_shape": mesh_shape,
                "layout": list(reversed(range(nd))),
                "item_size": v.tensor.element_size(),
                "layer_idx": idx,  # positional, per side
                "sharding_spec": spec,
                "global_shard_indices": gsi,
            })
        return out

    def metadata_dict(self) -> dict:
        """Wire-safe registration metadata shaped for tunix's ``dict_to_metadata``."""
        sync = self._require("metadata_dict()")
        num_shards = int(sync.num_shards)
        shards_list = [""] * num_shards
        try:
            for ep in sync.get_local_endpoints():
                addr = ep.get("endpoint", "")
                if addr.startswith(":"):
                    addr = f"{self.ip}{addr}"
                elif ":" in addr and addr.split(":")[0] in ("0.0.0.0", "127.0.0.1", ""):
                    addr = f"{self.ip}:{addr.split(':')[1]}"
                for s in ep.get("shards", []):
                    if 0 <= s < num_shards:
                        shards_list[s] = addr
        except Exception:  # pragma: no cover
            pass
        if not all(shards_list):
            shards_list = [s or f"{self.ip}:{sync.local_port}" for s in shards_list]
        control_addr = f"{self.ip}:{sync.listener_port}" if sync.listener_port else ""
        return {
            "unit": {
                "job_name": self.job_name,
                # One work unit per TP rank; keep ids unique and integer-like.
                "job_replica_id": str(self.worker_index * self.tp_size + self.tp_rank),
            },
            "shards": shards_list,
            "control_plane_rpc_address": control_addr,
            "mesh_shape": [self.tp_size],
            "mesh_axes": ["tp"],
            "host_subgrid": None,
            "variables": self.variable_dicts(),
            "auto_h2d": bool(self._effective_auto_h2d),
        }

    def describe(self) -> str:
        lines = [f"{self.job_name}[tp{self.tp_rank}/{self.tp_size}] {len(self.variables)} variables"]
        for v in self.variables:
            lines.append(
                f"  {v.name}: local={v.local_shape} global={v.global_shape}"
                f" shard_dim={v.shard_dim} dtype={v.tensor.dtype} ({v.source})"
            )
        return "\n".join(lines)
