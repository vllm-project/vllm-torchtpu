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

from . import errors, exec_spec, ops, pool_alias, profiles, spec
from .errors import MegakernelConfigError, MegakernelLayoutError
from .exec_spec import DebugOptions, ExecSpec, Tiles, default_tiles
from .ops import rup
from .spec import (
    Block,
    ExtraInput,
    LayerType,
    ModelSpec,
    ScratchReq,
    SmallField,
    Spec,
    StateDecl,
    SwitchSlot,
    WeightDecl,
)

__all__ = [
    "Block",
    "DebugOptions",
    "ExecSpec",
    "ExtraInput",
    "LayerType",
    "MegakernelConfigError",
    "MegakernelLayoutError",
    "ModelSpec",
    "ScratchReq",
    "SmallField",
    "Spec",
    "StateDecl",
    "SwitchSlot",
    "Tiles",
    "WeightDecl",
    "default_tiles",
    "errors",
    "exec_spec",
    "ops",
    "pool_alias",
    "profiles",
    "rup",
    "spec",
]
