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
"""Config-driven composable TPU Pallas megakernel library."""

from __future__ import annotations

import importlib
import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # eager imports for type checkers and IDEs
    from . import (
        collectives,
        decls,
        derive,
        dma,
        errors,
        exec_spec,
        geometry,
        model_def,
        module,
        module_options,
        modules,
        ops,
        planner,
        pool,
        profile,
        profiles,
        reference,
        runtime,
        schedule,
        spec,
        validate,
        weights,
    )
    from .collectives import (
        AllReduceRing,
    )
    from .dma import (
        pool_alias,
        vmem,
    )
    from .dma.vmem import (
        fit_default_tiles,
        fitted_plan,
    )
    from .errors import (
        MegakernelConfigError,
        MegakernelLayoutError,
    )
    from .exec_spec import (
        DebugOptions,
        ExecSpec,
        Tiles,
        default_tiles,
    )
    from .model_def import (
        Every,
        FromHF,
        FromHFLayerTypes,
        LMHead,
        ModelDef,
        ModelGeometry,
        ResolvedSchedule,
        Schedule,
        find_model,
        get_model,
        list_models,
        register_model,
    )
    from .modules import (
        GDN,
        GQA,
        Embedding,
        EmbeddingModule,
        GDNLinInLayout,
        GDNModule,
        GDNParams,
        GQAModule,
        GQAParams,
        MoE,
        MoEDebug,
        MoEModule,
        MoEParams,
        Norm,
        NormModule,
    )
    from .planner import (
        encode_ecopy,
    )
    from .pool import (
        POOL_STATE_NAME,
        CacheExec,
        PoolBackend,
        PoolLayout,
        allocate_test_layout,
        cache_exec,
        conv_to_raw_page,
        derive_pool_layout,
        kv_to_raw_page,
        make_pool_backend,
        raw_page_to_conv,
        raw_page_to_kv,
        raw_pages_to_ssm,
        ssm_to_raw_pages,
        with_cache,
    )
    from .profile import (
        DEFAULT_PROFILE,
        Baseline,
        Profile,
        TileContext,
        TileRule,
        get_profile,
        list_profiles,
        model_default_profile,
        register_profile,
    )
    from .spec import (
        REQUIRED,
        CompareMode,
        ExtraInput,
        FrozenDict,
        HFParamDecl,
        KVCacheDecl,
        LayerType,
        ModelSpec,
        Module,
        ModuleBinding,
        OutputRole,
        ParamInit,
        ParamTransform,
        RecurrentCacheDecl,
        ResolvedModuleGeometry,
        ScratchReq,
        SmallField,
        Spec,
        StateDecl,
        SwitchSlot,
        VLLMAlias,
        WeightDecl,
    )
    from .weights import (
        ArrayOps,
        JaxArrayOps,
        WeightSource,
        default_eplb_cores,
        encode_eplb_ecopy,
        pack_spec_logical_weights,
    )

# Public names resolve on first access (PEP 562): importing a submodule
# (e.g. `runtime.contract`) doesn't import jax through this package.
_SUBMODULES = frozenset(
    {
        "collectives",
        "decls",
        "derive",
        "dma",
        "errors",
        "exec_spec",
        "geometry",
        "model_def",
        "module",
        "module_options",
        "modules",
        "ops",
        "planner",
        "pool",
        "profile",
        "profiles",
        "reference",
        "runtime",
        "schedule",
        "spec",
        "validate",
        "weights",
    }
)
_LAZY: dict[str, str] = {
    "AllReduceRing": ".collectives",
    "ArrayOps": ".weights",
    "Baseline": ".profile",
    "CacheExec": ".pool",
    "CompareMode": ".spec",
    "DEFAULT_PROFILE": ".profile",
    "DebugOptions": ".exec_spec",
    "Embedding": ".modules",
    "EmbeddingModule": ".modules",
    "Every": ".model_def",
    "ExecSpec": ".exec_spec",
    "ExtraInput": ".spec",
    "FromHF": ".model_def",
    "FromHFLayerTypes": ".model_def",
    "FrozenDict": ".spec",
    "GDN": ".modules",
    "GDNLinInLayout": ".modules",
    "GDNModule": ".modules",
    "GDNParams": ".modules",
    "GQA": ".modules",
    "GQAModule": ".modules",
    "GQAParams": ".modules",
    "HFParamDecl": ".spec",
    "JaxArrayOps": ".weights",
    "KVCacheDecl": ".spec",
    "LMHead": ".model_def",
    "LayerType": ".spec",
    "MegakernelConfigError": ".errors",
    "MegakernelLayoutError": ".errors",
    "MoE": ".modules",
    "MoEDebug": ".modules",
    "MoEModule": ".modules",
    "MoEParams": ".modules",
    "ModelDef": ".model_def",
    "ModelGeometry": ".model_def",
    "ModelSpec": ".spec",
    "Module": ".spec",
    "ModuleBinding": ".spec",
    "Norm": ".modules",
    "NormModule": ".modules",
    "OutputRole": ".spec",
    "POOL_STATE_NAME": ".pool",
    "ParamInit": ".spec",
    "ParamTransform": ".spec",
    "PoolBackend": ".pool",
    "PoolLayout": ".pool",
    "Profile": ".profile",
    "REQUIRED": ".spec",
    "RecurrentCacheDecl": ".spec",
    "ResolvedModuleGeometry": ".spec",
    "ResolvedSchedule": ".model_def",
    "Schedule": ".model_def",
    "ScratchReq": ".spec",
    "SmallField": ".spec",
    "Spec": ".spec",
    "StateDecl": ".spec",
    "SwitchSlot": ".spec",
    "TileContext": ".profile",
    "TileRule": ".profile",
    "Tiles": ".exec_spec",
    "VLLMAlias": ".spec",
    "WeightDecl": ".spec",
    "WeightSource": ".weights",
    "allocate_test_layout": ".pool",
    "cache_exec": ".pool",
    "conv_to_raw_page": ".pool",
    "default_eplb_cores": ".weights",
    "default_tiles": ".exec_spec",
    "derive_pool_layout": ".pool",
    "encode_ecopy": ".planner",
    "encode_eplb_ecopy": ".weights",
    "find_model": ".model_def",
    "fit_default_tiles": ".dma.vmem",
    "fitted_plan": ".dma.vmem",
    "get_model": ".model_def",
    "get_profile": ".profile",
    "kv_to_raw_page": ".pool",
    "list_models": ".model_def",
    "list_profiles": ".profile",
    "make_pool_backend": ".pool",
    "model_default_profile": ".profile",
    "pack_spec_logical_weights": ".weights",
    "pool_alias": ".dma",
    "raw_page_to_conv": ".pool",
    "raw_page_to_kv": ".pool",
    "raw_pages_to_ssm": ".pool",
    "register_model": ".model_def",
    "register_profile": ".profile",
    "ssm_to_raw_pages": ".pool",
    "vmem": ".dma",
    "with_cache": ".pool",
}

__all__ = [
    "AllReduceRing",
    "ArrayOps",
    "Baseline",
    "CacheExec",
    "CompareMode",
    "DEFAULT_PROFILE",
    "DebugOptions",
    "Embedding",
    "EmbeddingModule",
    "Every",
    "ExecSpec",
    "ExtraInput",
    "FromHF",
    "FromHFLayerTypes",
    "FrozenDict",
    "GDN",
    "GDNLinInLayout",
    "GDNModule",
    "GDNParams",
    "GQA",
    "GQAModule",
    "GQAParams",
    "HFParamDecl",
    "JaxArrayOps",
    "KVCacheDecl",
    "LMHead",
    "LayerType",
    "MegakernelConfigError",
    "MegakernelLayoutError",
    "MoE",
    "MoEDebug",
    "MoEModule",
    "MoEParams",
    "ModelDef",
    "ModelGeometry",
    "ModelSpec",
    "Module",
    "ModuleBinding",
    "Norm",
    "NormModule",
    "OutputRole",
    "POOL_STATE_NAME",
    "ParamInit",
    "ParamTransform",
    "PoolBackend",
    "PoolLayout",
    "Profile",
    "REQUIRED",
    "RecurrentCacheDecl",
    "ResolvedModuleGeometry",
    "ResolvedSchedule",
    "Schedule",
    "ScratchReq",
    "SmallField",
    "Spec",
    "StateDecl",
    "SwitchSlot",
    "TileContext",
    "TileRule",
    "Tiles",
    "VLLMAlias",
    "WeightDecl",
    "WeightSource",
    "allocate_test_layout",
    "cache_exec",
    "collectives",
    "conv_to_raw_page",
    "decls",
    "default_eplb_cores",
    "default_tiles",
    "derive",
    "derive_pool_layout",
    "dma",
    "encode_ecopy",
    "encode_eplb_ecopy",
    "errors",
    "exec_spec",
    "find_model",
    "fit_default_tiles",
    "fitted_plan",
    "geometry",
    "get_model",
    "get_profile",
    "kv_to_raw_page",
    "list_models",
    "list_profiles",
    "make_pool_backend",
    "model_def",
    "model_default_profile",
    "module",
    "module_options",
    "modules",
    "ops",
    "pack_spec_logical_weights",
    "planner",
    "pool",
    "pool_alias",
    "profile",
    "profiles",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "reference",
    "register_model",
    "register_profile",
    "runtime",
    "schedule",
    "spec",
    "ssm_to_raw_pages",
    "validate",
    "vmem",
    "weights",
    "with_cache",
]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        mod_obj = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod_obj
        return mod_obj
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)


# `verify_module` names both a submodule and the function `verify` exports.
# The import system binds a loaded submodule on its package, which would
# shadow the lazily resolved function; the name stays the function, as with
# the eager `from .verify import verify_module` (the submodule stays importable
# through `sys.modules`).
_FUNCTION_NAMED_SUBMODULES = frozenset({"verify_module"})


class _Package(types.ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in _FUNCTION_NAMED_SUBMODULES and isinstance(value, types.ModuleType):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Package
