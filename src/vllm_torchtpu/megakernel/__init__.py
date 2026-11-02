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
"""Core declarations, execution config, VMEM pool aliasing, and numeric primitives."""

from . import (
    decls,
    errors,
    exec_spec,
    module,
    module_options,
    ops,
    planner,
    pool_alias,
    profiles,
    spec,
)
from .errors import MegakernelConfigError, MegakernelLayoutError
from .exec_spec import DebugOptions, ExecSpec, Tiles, default_tiles
from .ops import rup
from .planner import encode_ecopy
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

__all__ = [
    "REQUIRED",
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
    "RecurrentCacheDecl",
    "ResolvedModuleGeometry",
    "ScratchReq",
    "SmallField",
    "Spec",
    "StateDecl",
    "SwitchSlot",
    "Tiles",
    "VLLMAlias",
    "WeightDecl",
    "decls",
    "default_tiles",
    "encode_ecopy",
    "errors",
    "exec_spec",
    "module",
    "module_options",
    "ops",
    "planner",
    "pool_alias",
    "profiles",
    "rup",
    "spec",
]
