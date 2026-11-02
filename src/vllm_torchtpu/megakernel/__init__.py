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
        decls,
        dma,
        errors,
        exec_spec,
        module,
        module_options,
        ops,
        planner,
        pool,
        profile,
        profiles,
        spec,
    )
    from .dma import pool_alias
    from .errors import MegakernelConfigError, MegakernelLayoutError
    from .exec_spec import DebugOptions, ExecSpec, Tiles, default_tiles
    from .ops import col_chunk, rup
    from .planner import encode_ecopy
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
        TileRule,
        get_profile,
        list_profiles,
        model_default_profile,
        register_profile,
    )
    from .spec import (
        REQUIRED,
        ExtraInput,
        FrozenDict,
        HFParamDecl,
        KVCacheDecl,
        LayerType,
        ModelSpec,
        Module,
        ModuleBinding,
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

# Public names resolve on first access (PEP 562): importing a submodule
# (e.g. `runtime.contract`) doesn't import jax through this package.
_SUBMODULES = frozenset(
    {
        "decls",
        "dma",
        "errors",
        "exec_spec",
        "module",
        "module_options",
        "ops",
        "planner",
        "pool",
        "profile",
        "profiles",
        "spec",
    }
)
_LAZY: dict[str, str] = {
    "MegakernelConfigError": ".errors",
    "MegakernelLayoutError": ".errors",
    "DebugOptions": ".exec_spec",
    "ExecSpec": ".exec_spec",
    "Tiles": ".exec_spec",
    "default_tiles": ".exec_spec",
    "col_chunk": ".ops",
    "rup": ".ops",
    "encode_ecopy": ".planner",
    "POOL_STATE_NAME": ".pool",
    "CacheExec": ".pool",
    "PoolBackend": ".pool",
    "PoolLayout": ".pool",
    "allocate_test_layout": ".pool",
    "cache_exec": ".pool",
    "conv_to_raw_page": ".pool",
    "derive_pool_layout": ".pool",
    "kv_to_raw_page": ".pool",
    "make_pool_backend": ".pool",
    "raw_page_to_conv": ".pool",
    "raw_page_to_kv": ".pool",
    "raw_pages_to_ssm": ".pool",
    "ssm_to_raw_pages": ".pool",
    "with_cache": ".pool",
    "DEFAULT_PROFILE": ".profile",
    "Baseline": ".profile",
    "Profile": ".profile",
    "TileRule": ".profile",
    "get_profile": ".profile",
    "list_profiles": ".profile",
    "model_default_profile": ".profile",
    "register_profile": ".profile",
    "REQUIRED": ".spec",
    "ExtraInput": ".spec",
    "FrozenDict": ".spec",
    "HFParamDecl": ".spec",
    "KVCacheDecl": ".spec",
    "LayerType": ".spec",
    "ModelSpec": ".spec",
    "Module": ".spec",
    "ModuleBinding": ".spec",
    "ParamTransform": ".spec",
    "RecurrentCacheDecl": ".spec",
    "ResolvedModuleGeometry": ".spec",
    "ScratchReq": ".spec",
    "SmallField": ".spec",
    "Spec": ".spec",
    "StateDecl": ".spec",
    "SwitchSlot": ".spec",
    "VLLMAlias": ".spec",
    "WeightDecl": ".spec",
    "pool_alias": ".dma",
}

__all__ = [
    "DEFAULT_PROFILE",
    "POOL_STATE_NAME",
    "REQUIRED",
    "Baseline",
    "CacheExec",
    "DebugOptions",
    "ExecSpec",
    "ExtraInput",
    "FrozenDict",
    "HFParamDecl",
    "KVCacheDecl",
    "LayerType",
    "MegakernelConfigError",
    "MegakernelLayoutError",
    "ModelSpec",
    "Module",
    "ModuleBinding",
    "ParamTransform",
    "PoolBackend",
    "PoolLayout",
    "Profile",
    "RecurrentCacheDecl",
    "ResolvedModuleGeometry",
    "ScratchReq",
    "SmallField",
    "Spec",
    "StateDecl",
    "SwitchSlot",
    "TileRule",
    "Tiles",
    "VLLMAlias",
    "WeightDecl",
    "allocate_test_layout",
    "cache_exec",
    "col_chunk",
    "conv_to_raw_page",
    "decls",
    "default_tiles",
    "derive_pool_layout",
    "dma",
    "encode_ecopy",
    "errors",
    "exec_spec",
    "get_profile",
    "kv_to_raw_page",
    "list_profiles",
    "make_pool_backend",
    "model_default_profile",
    "module",
    "module_options",
    "ops",
    "planner",
    "pool",
    "pool_alias",
    "profile",
    "profiles",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "register_profile",
    "rup",
    "spec",
    "ssm_to_raw_pages",
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
