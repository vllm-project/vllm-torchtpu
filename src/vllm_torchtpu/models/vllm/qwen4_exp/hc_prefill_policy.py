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
"""Independent HC profitability policy; kernel admission remains mandatory."""

from typing import Literal

from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware

HcPrefillMode = int | Literal["auto"]

# HC fusion needs enough token work to amortize launch and working-set costs.
# This profitability policy does not replace resource or dtype admission.
_MEASURED_HARDWARE = HcHardware("7x", 64 << 20, 128, 16, 8, 256)
_MIX_MIN_TOKENS = 1024
_COMBINE_MIN_TOKENS = 1024


def validate_hc_prefill_mode(value: object, *, full_mix: bool) -> HcPrefillMode:
    if isinstance(value, str) and value == "auto":
        return "auto"
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        message = "HC prefill mode must be 'auto' or a nonnegative integer"
        if full_mix:
            message += "; full mix must be a multiple of 128"
        raise ValueError(message)
    if full_mix and value and value % 128:
        raise ValueError("HC prefill full mix tile must be a multiple of 128")
    return value


def _measured_geometry(
    hardware: HcHardware | None,
    branches: int,
    hidden: int,
    lowrank: int,
    per_branch_norm: bool,
    eps: float,
) -> bool:
    return (
        hardware == _MEASURED_HARDWARE
        and (branches, hidden, lowrank) == (4, 2560, 320)
        and per_branch_norm
        and eps == 1e-6
    )


def full_mix_auto_min_tokens(
    hardware: HcHardware | None,
    branches: int,
    hidden: int,
    lowrank: int,
    per_branch_norm: bool,
    eps: float,
) -> int | None:
    return (
        _MIX_MIN_TOKENS
        if _measured_geometry(hardware, branches, hidden, lowrank, per_branch_norm, eps)
        else None
    )


def combine_auto_min_tokens(
    hardware: HcHardware | None,
    branches: int,
    hidden: int,
    lowrank: int,
    per_branch_norm: bool,
    eps: float,
) -> int | None:
    # Preserve the measured normalization/projection context even without mix.
    return (
        _COMBINE_MIN_TOKENS
        if _measured_geometry(hardware, branches, hidden, lowrank, per_branch_norm, eps)
        else None
    )
