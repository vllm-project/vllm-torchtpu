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
"""HLO check that a compiled pooled step makes no full-pool HBM / VMEM copy."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from jax.typing import DTypeLike

from .pool import POOL_KERNEL_NAMES, PoolLayout

# HLO primitive type names of the non-fp8 page dtypes (fp8 maps by prefix).
_HLO_DTYPES = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}


def _hlo_dtype(dtype: DTypeLike) -> str:
    """HLO primitive type name of `dtype` (e.g. `float8_e4m3fn` -> `f8e4m3fn`)."""
    name = jnp.dtype(dtype).name
    if name.startswith("float8_"):
        return "f8" + name.removeprefix("float8_")
    return _HLO_DTYPES.get(name, name)


def check_fused_pooled_step_hlo(
    hlo_text: str,
    *,
    kernel_names: tuple[str, str] | None = None,
) -> tuple[bool, str]:
    """Verify that a 1-launch `fused_pool_io` HLO has no separate gather/scatter."""
    gather_name, scatter_name = (
        POOL_KERNEL_NAMES if kernel_names is None else kernel_names
    )
    if (gather_name and gather_name in hlo_text) or (
        scatter_name and scatter_name in hlo_text
    ):
        return (
            False,
            "unexpected separate gather/scatter kernel in fused_pool_io HLO",
        )
    return True, "ok"


def check_pooled_step_hlo(
    hlo_text: str,
    page_size: int | None = None,
    *,
    kvh: int | None = None,
    hd: int | None = None,
    layout: PoolLayout | None = None,
    min_pool_pages: int = 16,
    require_scatter: bool = True,
    fused_pool_io: bool = False,
    kernel_names: tuple[str, str] | None = None,
) -> tuple[bool, str]:
    """Verify that a compiled pooled step HLO has no full-pool HBM or VMEM copies.

    The pool page shape (`page_size`, `kvh`, `hd`) defaults to `layout` (or the
    validated `PoolLayout()` page geometry); explicit arguments override it.
    `kernel_names` defaults to the generic pool kernel names.

    `min_pool_pages` (default 16) filters out small per-token KV (1-page) or
    checkpoint-slot (`pages_per_slot`-page) staging slices materialized in CPU
    interpret mode while flagging any full-pool HBM copy
    (`num_pages >= min_pool_pages`).

    `hlo_text` must be XLA HLO text (`HloModule ...`), preferably the compiled
    module (`jax.jit(...).lower(...).compile().as_text()`); StableHLO MLIR text
    (`.lower(...).as_text()`) is rejected, since its pool copies cannot be matched.
    """
    if "HloModule" not in hlo_text:
        return (
            False,
            "not XLA HLO text (no 'HloModule'); pass the compiled module "
            "text, `jax.jit(...).lower(...).compile().as_text()`",
        )
    lay = PoolLayout() if layout is None else layout
    lay = dataclasses.replace(
        lay,
        page_size=lay.page_size if page_size is None else page_size,
        kvh=lay.kvh if kvh is None else kvh,
        hd=lay.hd if hd is None else hd,
    )
    if kernel_names is None:
        kernel_names = POOL_KERNEL_NAMES
    n_tpu_cc = hlo_text.count("tpu_custom_call")
    n_cpu_cc = hlo_text.count("xla_ffi_python_cpu_callback")
    is_cpu_interp = n_cpu_cc > 0
    _, scatter_name = kernel_names
    if fused_pool_io:
        min_cc = 1
        ok_f, msg_f = check_fused_pooled_step_hlo(hlo_text, kernel_names=kernel_names)
        if not ok_f:
            return False, msg_f
    else:
        min_cc = 2 if require_scatter else 1
        if require_scatter:
            has_scatter = (
                (bool(scatter_name) and scatter_name in hlo_text)
                or n_tpu_cc >= 2
                or (is_cpu_interp and n_cpu_cc >= 2)
            )
            if not has_scatter:
                return False, f"missing {scatter_name or 'scatter'} in HLO"
    has_mk = (
        "decode_megakernel" in hlo_text
        or n_tpu_cc >= min_cc
        or (is_cpu_interp and n_cpu_cc >= min_cc)
        or ("pallas_call" in hlo_text and "_body" in hlo_text)
    )
    if not has_mk:
        msg = (
            f"missing megakernel call in HLO "
            f"(found {n_tpu_cc} tpu_custom_call, need >={min_cc})"
        )
        return False, msg
    pool_suffix = "," + ",".join(map(str, lay.kv_page_shape)) + "]"
    page_ty = f"= {_hlo_dtype(lay.page_dtype)}["
    for line in hlo_text.splitlines():
        if pool_suffix in line and (
            "copy-start(" in line or (page_ty in line and "copy(" in line)
        ):
            prefix = line.split(pool_suffix, 1)[0]
            dim_str = prefix.rsplit("[", 1)[-1].rsplit(",", 1)[-1].strip()
            if dim_str.isdigit() and int(dim_str) < min_pool_pages:
                continue
            if "S(1)" in line:
                return False, f"pool operand copied to VMEM S(1): {line.strip()}"
            return False, f"unexpected full-pool HBM copy in HLO: {line.strip()}"
    return True, "ok"
