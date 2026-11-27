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
"""Post-load parameter classification and fail-closed layout validation for RL sync.

Raiden (`tpu_sync`) writes N-D slices directly into live TPU `nn.Parameter`
buffers without re-running `model.load_weights()` or
`process_weights_after_loading()` (PWAL). This module inspects each parameter's
owning vLLM/TorchTPU module against an explicit whitelist of supported layer
types and fails closed with `NotImplementedError` on quantized, permuted,
padded, derived, or unrecognized layouts.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.parameter import BasevLLMParameter

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.dp_vocab_parallel import DP_VOCAB_ATTR
from vllm_torchtpu.layers.adapter.expert_placement import has_applied_layers
from vllm_torchtpu.layers.adapter.gdn_linear import (
    GdnColumnParallelLinear,
    GdnInterleavedColumnParallelLinear,
)
from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

DOMAIN_EP = "ep"
DOMAIN_TP = "tp"
DOMAIN_REPLICATED = "replicated"

ALLOWED_SYNC_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32})

_SKIP_PREFIXES = ("visual", "mtp")
_WRAPPER_ROOTS = ("language_model", "base", "model")
_NAME_REWRITES = ((".experts.routed_experts.", ".experts."),)
# Restrict index splitting to container prefixes so numbered layer names (e.g.
# `post_feedforward_layernorm_1`) keep their sort order relative to JAX pytrees.
_FOLDED_INDEX_RE = re.compile(r"^(layers|blocks|experts)_(\d+)$")
_HEAD_SHARDED_ATTRS = frozenset({"A_log", "dt_bias"})
_SHARDED_WEIGHT_LOADER_CODE = sharded_weight_loader(0).__code__

# Exact-class whitelists: unrecognized subclasses (e.g. hybrid-sharded or PLE
# variants) fail closed until explicitly audited for raw slice DMA safety.
_SUPPORTED_VOCAB_TYPES: frozenset[type[torch.nn.Module]] = frozenset(
    {
        VocabParallelEmbedding,
        ParallelLMHead,
    }
)
_SUPPORTED_LINEAR_TYPES: frozenset[type[LinearBase]] = frozenset(
    {
        ReplicatedLinear,
        GateLinear,
        RowParallelLinear,
        TpuRowParallelLinear,
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        GdnColumnParallelLinear,
        GdnInterleavedColumnParallelLinear,
    }
)


@dataclass
class BoundVariable:
    """Destination parameter bound to Raiden plus post-PWAL layout metadata."""

    name: str
    tensor: torch.Tensor
    shard_dim: int | None
    num_shards: int
    shard_domain: str
    global_shape: tuple[int, ...]
    flipped: bool = False
    conv_transposed: bool | None = None
    replication_factors: tuple[int, ...] | None = None
    owner_cls: str = ""

    @classmethod
    def create(
        cls,
        *,
        name: str,
        tensor: torch.Tensor,
        owner_cls: str,
        shard_dim: int | None = None,
        num_shards: int = 1,
        shard_domain: str = DOMAIN_REPLICATED,
        flipped: bool = False,
        conv_transposed: bool | None = None,
        replication_factors: tuple[int, ...] | None = None,
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
            flipped=flipped,
            conv_transposed=conv_transposed,
            replication_factors=replication_factors,
            owner_cls=owner_cls,
        )

    @property
    def local_shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self.tensor.shape)

    def layout_dict(self, *, layer_idx: int | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "local_shape": list(self.local_shape),
            "global_shape": list(self.global_shape),
            "dtype": str(self.tensor.dtype).removeprefix("torch."),
            "shard_dim": self.shard_dim,
            "num_shards": self.num_shards,
            "shard_domain": self.shard_domain,
            "flipped": self.flipped,
        }
        if layer_idx is not None:
            out["layer_idx"] = int(layer_idx)
        if self.conv_transposed is not None:
            out["conv_transposed"] = self.conv_transposed
        if self.replication_factors is not None:
            out["replication_factors"] = list(self.replication_factors)
        return out


def canonical_param_name(name: str) -> str:
    """Normalize a vLLM parameter path to its canonical dotted sync key."""
    for old, new in _NAME_REWRITES:
        name = name.replace(old, new)
    segments: list[str] = []
    for raw in name.split("."):
        comp = raw.strip("'\"")
        if not comp:
            continue
        folded = _FOLDED_INDEX_RE.match(comp)
        segments.extend(folded.groups() if folded else (comp,))
    while segments and segments[0] in _WRAPPER_ROOTS:
        segments.pop(0)
    if segments and segments[-1] == "value":
        segments.pop()
    return ".".join(segments)


def _language_model_and_skipped_roots(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, set[str]]:
    get_lm = getattr(model, "get_language_model", None)
    lm_model = get_lm() if callable(get_lm) else model
    if not isinstance(lm_model, torch.nn.Module):
        lm_model = model

    skipped_roots: set[str] = set(_SKIP_PREFIXES)
    for root in (model, lm_model):
        for tower_name in getattr(root, "_tower_model_names", ()) or ():
            skipped_roots.add(str(tower_name))
        for sub_name, sub_mod in root.named_modules():
            if sub_name and getattr(sub_mod, "_is_tower_model", False):
                skipped_roots.add(str(sub_name))
    return lm_model, skipped_roots


def _is_skipped_path(name: str, skipped_roots: set[str]) -> bool:
    cur = name
    while True:
        if any(cur == r or cur.startswith(f"{r}.") for r in skipped_roots):
            return True
        head, sep, rest = cur.partition(".")
        if not sep or head not in _WRAPPER_ROOTS:
            return False
        cur = rest


def _sharded_weight_loader_dim(loader: Any) -> int | None:
    if loader is None or isinstance(loader, partial) or inspect.ismethod(loader):
        return None
    if getattr(loader, "__code__", None) is not _SHARDED_WEIGHT_LOADER_CODE:
        return None
    axis = inspect.getclosurevars(loader).nonlocals.get("shard_axis")
    return axis if isinstance(axis, int) and axis >= 0 else None


def _check_param_safety(
    param_name: str,
    raw_tensor: torch.Tensor,
) -> None:
    # TODO: support quantized/packed weights, LoRA, expert placement, and MLA.
    if raw_tensor.dtype not in ALLOWED_SYNC_DTYPES:
        raise NotImplementedError(
            f"{param_name}: unsupported parameter dtype {raw_tensor.dtype}."
        )
    parts = param_name.split(".")
    if "base_layer" in parts or any(p.startswith("lora_") for p in parts):
        raise NotImplementedError(
            f"{param_name}: LoRA parameters (.base_layer / lora_*) are not supported."
        )


def _classify_vocab_embedding(
    module: VocabParallelEmbedding | ParallelLMHead,
    *,
    param_name: str,
    canonical_name: str,
    attr: str,
    raw_tensor: torch.Tensor,
    owner_cls: str,
) -> BoundVariable:
    qmethod = getattr(module, "quant_method", None)
    if qmethod is not None and not isinstance(qmethod, UnquantizedEmbeddingMethod):
        raise NotImplementedError(
            f"{param_name}: quantized {owner_cls} method {type(qmethod).__name__} "
            "is not supported."
        )
    if attr != "weight":
        raise NotImplementedError(
            f"{param_name}: unsupported {owner_cls} parameter {attr!r}."
        )
    mod_tp = int(module.tp_size)
    org_vocab_size = int(module.org_vocab_size)
    shard_domain = (
        DOMAIN_EP if bool(getattr(module, DP_VOCAB_ATTR, False)) else DOMAIN_TP
    )
    if raw_tensor.shape[0] * mod_tp != org_vocab_size:
        raise NotImplementedError(
            f"{param_name}: padded vocab shard {tuple(raw_tensor.shape)} * "
            f"{shard_domain}={mod_tp} != org_vocab_size={org_vocab_size}."
        )
    return BoundVariable.create(
        name=canonical_name,
        tensor=raw_tensor,
        owner_cls=owner_cls,
        shard_dim=0 if mod_tp > 1 else None,
        num_shards=mod_tp,
        shard_domain=shard_domain,
    )


def _classify_routed_experts(
    module: RoutedExperts,
    *,
    param_name: str,
    canonical_name: str,
    attr: str,
    param: torch.Tensor,
    raw_tensor: torch.Tensor,
    owner_cls: str,
) -> BoundVariable:
    qmethod = module.quant_method
    if qmethod is not None and not isinstance(qmethod, UnquantizedFusedMoEMethod):
        raise NotImplementedError(
            f"{param_name}: quantized MoE method {type(qmethod).__name__} "
            "is not supported."
        )
    if module.expert_placement_strategy != "linear":
        raise NotImplementedError(
            f"{param_name}: expert_placement_strategy="
            f"{module.expert_placement_strategy!r} is not supported."
        )
    moe_cfg = module.moe_config
    moe_parallel_cfg = moe_cfg.moe_parallel_config
    n_redundant = int(moe_cfg.num_experts) - int(moe_cfg.num_logical_experts)
    if bool(moe_parallel_cfg.enable_eplb) or n_redundant > 0:
        raise NotImplementedError(
            f"{param_name}: EPLB / redundant experts (n_redundant={n_redundant}) "
            "are not supported."
        )
    n_fused_shared = int(module.expert_map_manager.num_fused_shared_experts)
    if n_fused_shared > 0:
        raise NotImplementedError(
            f"{param_name}: fused shared experts ({n_fused_shared}) are not supported."
        )
    act_str = getattr(qmethod, "_tpu_activation_str", None) or module.activation
    if act_str is not None and "swigluoai" in str(act_str).lower():
        raise NotImplementedError(
            f"{param_name}: interleaved swigluoai MoE layout is not supported."
        )

    moe_flipped = bool(getattr(module, WEIGHT_FLIPPED_ATTR, False)) and bool(
        getattr(param, WEIGHT_FLIPPED_ATTR, False)
    )
    if moe_parallel_cfg.use_ep:
        if int(moe_parallel_cfg.tp_size) != 1:
            raise NotImplementedError(
                f"{param_name}: hybrid EP+TP RoutedExperts "
                f"(ep_size={moe_parallel_cfg.ep_size}, "
                f"tp_size={moe_parallel_cfg.tp_size}) is not supported."
            )
        if attr not in ("w13_weight", "w2_weight"):
            raise NotImplementedError(
                f"{param_name}: unsupported EP RoutedExperts parameter {attr!r}."
            )
        ep_size = int(moe_parallel_cfg.ep_size)
        return BoundVariable.create(
            name=canonical_name,
            tensor=raw_tensor,
            owner_cls=owner_cls,
            shard_dim=0 if ep_size > 1 else None,
            num_shards=ep_size,
            shard_domain=DOMAIN_EP,
            flipped=moe_flipped,
        )

    if attr == "w13_weight":
        shard_dim = 2 if moe_flipped else 1
    elif attr == "w2_weight":
        shard_dim = 1 if moe_flipped else 2
    else:
        raise NotImplementedError(
            f"{param_name}: unsupported TP-MoE parameter {attr!r}."
        )
    return BoundVariable.create(
        name=canonical_name,
        tensor=raw_tensor,
        owner_cls=owner_cls,
        shard_dim=shard_dim,
        num_shards=int(moe_parallel_cfg.tp_size),
        shard_domain=DOMAIN_TP,
        flipped=moe_flipped,
    )


def _linear_replication_factors(module: LinearBase) -> tuple[int, ...] | None:
    if isinstance(module, GdnColumnParallelLinear):
        return tuple(int(r) for r in module.replication_factors)
    if isinstance(module, GdnInterleavedColumnParallelLinear):
        kq_rep = int(module.geometry.kq_replication_factor)
        return tuple(kq_rep if shared else 1 for shared in module.replicated_segments)
    if isinstance(module, QKVParallelLinear):
        kv_rep = int(module.num_kv_head_replicas)
        return (1, kv_rep, kv_rep)
    if isinstance(module, MergedColumnParallelLinear):
        return (1,) * len(module.output_sizes)
    return None


def _classify_linear(
    module: LinearBase,
    *,
    param_name: str,
    canonical_name: str,
    attr: str,
    param: torch.Tensor,
    raw_tensor: torch.Tensor,
    owner_cls: str,
) -> BoundVariable:
    qmethod = module.quant_method
    if qmethod is not None and not isinstance(qmethod, UnquantizedLinearMethod):
        raise NotImplementedError(
            f"{param_name}: quantized LinearBase method {type(qmethod).__name__} "
            "is not supported."
        )
    if attr not in ("weight", "bias"):
        raise NotImplementedError(
            f"{param_name}: unsupported LinearBase parameter {attr!r}."
        )
    lin_loader = getattr(param, "weight_loader", None)
    if lin_loader is not None and (
        not inspect.ismethod(lin_loader) or lin_loader.__self__ is not module
    ):
        raise NotImplementedError(
            f"{param_name}: custom weight_loader override on {owner_cls} "
            "is not supported."
        )

    is_conv3d = (
        attr == "weight"
        and raw_tensor.dim() == 3
        and isinstance(module, ColumnParallelLinear)
    )
    flipped = bool(getattr(module, WEIGHT_FLIPPED_ATTR, False)) and attr == "weight"
    if isinstance(module, ReplicatedLinear) or bool(module.disable_tp):
        return BoundVariable.create(
            name=canonical_name,
            tensor=raw_tensor,
            owner_cls=owner_cls,
            flipped=flipped,
        )

    mod_tp = int(module.tp_size)
    rep_factors = _linear_replication_factors(module)
    if is_conv3d:
        return BoundVariable.create(
            name=canonical_name,
            tensor=raw_tensor,
            owner_cls=owner_cls,
            shard_dim=0,
            num_shards=mod_tp,
            shard_domain=DOMAIN_TP,
            conv_transposed=False,
            replication_factors=rep_factors,
        )
    if isinstance(module, RowParallelLinear):
        shard_dim = (0 if flipped else 1) if attr == "weight" else None
        return BoundVariable.create(
            name=canonical_name,
            tensor=raw_tensor,
            owner_cls=owner_cls,
            shard_dim=shard_dim,
            num_shards=mod_tp,
            shard_domain=DOMAIN_TP,
            flipped=flipped,
        )
    shard_dim = 1 if (flipped and attr == "weight") else 0
    return BoundVariable.create(
        name=canonical_name,
        tensor=raw_tensor,
        owner_cls=owner_cls,
        shard_dim=shard_dim,
        num_shards=mod_tp,
        shard_domain=DOMAIN_TP,
        flipped=flipped,
        replication_factors=rep_factors,
    )


def classify_parameter(
    model: torch.nn.Module,
    param_name: str,
    param: torch.Tensor,
    tp_size: int,
) -> BoundVariable:
    """Inspect owning `nn.Module` and `nn.Parameter` against the layout whitelist."""
    mod_path, _, attr = param_name.rpartition(".")
    module = model.get_submodule(mod_path) if mod_path else model
    raw_tensor = param.data if isinstance(param, torch.nn.Parameter) else param
    canonical_name = canonical_param_name(param_name)
    mod_type = type(module)
    owner_cls = mod_type.__name__

    _check_param_safety(param_name, raw_tensor)

    if mod_type in _SUPPORTED_VOCAB_TYPES:
        assert isinstance(module, (VocabParallelEmbedding, ParallelLMHead))
        return _classify_vocab_embedding(
            module,
            param_name=param_name,
            canonical_name=canonical_name,
            attr=attr,
            raw_tensor=raw_tensor,
            owner_cls=owner_cls,
        )
    if mod_type is RoutedExperts:
        assert isinstance(module, RoutedExperts)
        return _classify_routed_experts(
            module,
            param_name=param_name,
            canonical_name=canonical_name,
            attr=attr,
            param=param,
            raw_tensor=raw_tensor,
            owner_cls=owner_cls,
        )
    if mod_type in _SUPPORTED_LINEAR_TYPES:
        assert isinstance(module, LinearBase)
        return _classify_linear(
            module,
            param_name=param_name,
            canonical_name=canonical_name,
            attr=attr,
            param=param,
            raw_tensor=raw_tensor,
            owner_cls=owner_cls,
        )
    if isinstance(
        module, (VocabParallelEmbedding, ParallelLMHead, RoutedExperts, LinearBase)
    ):
        raise NotImplementedError(
            f"{param_name}: unrecognized layer subclass {owner_cls} is not supported."
        )

    mod_tp = int(getattr(module, "tp_size", tp_size))
    loader = getattr(param, "weight_loader", None)
    sharded_dim = _sharded_weight_loader_dim(loader)
    if sharded_dim is not None:
        if sharded_dim >= raw_tensor.dim():
            raise ValueError(
                f"{param_name}: sharded_weight_loader axis {sharded_dim} out of "
                f"bounds for shape {tuple(raw_tensor.shape)}."
            )
        return BoundVariable.create(
            name=canonical_name,
            tensor=raw_tensor,
            owner_cls=owner_cls,
            shard_dim=sharded_dim if tp_size > 1 else None,
            num_shards=tp_size,
            shard_domain=DOMAIN_TP,
        )
    if isinstance(module, MambaBase) and attr in _HEAD_SHARDED_ATTRS and loader is None:
        return BoundVariable.create(
            name=canonical_name,
            tensor=raw_tensor,
            owner_cls=owner_cls,
            shard_dim=0 if mod_tp > 1 else None,
            num_shards=mod_tp,
            shard_domain=DOMAIN_TP,
        )

    loader_fn = loader.__func__ if inspect.ismethod(loader) else loader
    if loader is not None and loader_fn is not default_weight_loader:
        raise NotImplementedError(
            f"{param_name}: unrecognized weight_loader {loader!r} on {owner_cls}."
        )
    if mod_tp > 1 and (
        isinstance(param, BasevLLMParameter)
        or getattr(param, "input_dim", None) is not None
        or getattr(param, "output_dim", None) is not None
    ):
        raise NotImplementedError(
            f"{param_name}: unrecognized sharded parameter ({type(param).__name__}) "
            f"on {owner_cls} under tp_size={mod_tp}."
        )
    # Outside the explicit module whitelist above, only 1-D replicated parameters
    # (such as RMSNorm/LayerNorm scales) are accepted.
    if raw_tensor.dim() != 1:
        raise NotImplementedError(
            f"{param_name}: unsupported {raw_tensor.dim()}-D parameter on {owner_cls}; "
            "only 1-D replicated parameters are supported outside whitelisted layers."
        )
    return BoundVariable.create(
        name=canonical_name,
        tensor=raw_tensor,
        owner_cls=owner_cls,
    )


def collect_variables(model: torch.nn.Module, tp_size: int) -> list[BoundVariable]:
    """Enumerate and canonicalize bindable language-model parameters."""
    # TODO: support static expert placement permutation maps in Raiden RL sync.
    if envs.TPU_EXPERT_PLACEMENT or has_applied_layers():
        raise NotImplementedError(
            "TPU_EXPERT_PLACEMENT permutes MoE experts; Raiden requires natural order."
        )
    lm_model, skipped_roots = _language_model_and_skipped_roots(model)
    named_params = [
        (name, param)
        for name, param in lm_model.named_parameters()
        if not _is_skipped_path(name, skipped_roots)
    ]
    tpu_tensors = [
        param.data for _, param in named_params if param.data.device.type == "tpu"
    ]
    if tpu_tensors:
        synchronize_tensors(tpu_tensors)

    out: list[BoundVariable] = []
    seen_ptrs: dict[int, str] = {}
    seen_names: dict[str, str] = {}

    for name, param in named_params:
        ptr = param.data.data_ptr()
        var = classify_parameter(lm_model, name, param, tp_size)
        if not var.name:
            raise ValueError(f"Parameter {name!r} canonicalized to an empty key.")
        if ptr != 0 and ptr in seen_ptrs:
            logger.info_once(
                "[raiden_sync] Skipping tied parameter %s sharing buffer with %s.",
                var.name,
                seen_ptrs[ptr],
            )
            continue
        if var.name in seen_names:
            raise ValueError(
                f"Parameters {seen_names[var.name]!r} and {name!r} both "
                f"canonicalize to {var.name!r}."
            )
        if ptr != 0:
            seen_ptrs[ptr] = var.name
        seen_names[var.name] = name
        out.append(var)

    if not out:
        raise RuntimeError("No bindable language-model parameters found on model.")

    # C++ indexes destination buffers by `src_var.layer_idx`, so sort by dotted
    # path tuple to match JAX pytree traversal on the trainer.
    out.sort(key=lambda v: tuple(v.name.split(".")))
    return out
