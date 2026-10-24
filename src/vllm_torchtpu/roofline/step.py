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
"""Hard and serial step floors for standard and speculative decode."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .chips import ChipSpec
from .collectives import make
from .geometry import ModelGeometry
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    _split_matmul_flops,
    pass_costs,
)


@dataclass(frozen=True)
class StepFloor:
    """Resource times and binding bottleneck for one decode step or cycle."""

    modules: tuple[ModuleCost, ...]
    hbm_s: float
    mm_s: float
    ici_s: float
    chain_s: float
    hard_floor_s: float
    serial_floor_s: float
    binding: str


def _combine_modules(
    modules: tuple[ModuleCost, ...],
    chip: ChipSpec,
    collective_latency_s: float = 0.0,
) -> StepFloor:
    hbm_s = sum(m.hbm_bytes for m in modules) / chip.hbm_bw
    mm_s = (
        sum(m.fp8_flops for m in modules) / chip.fp8_flops
        + sum(m.bf16_flops for m in modules) / chip.bf16_flops
    )
    ici_s = (
        sum(m.ici_bytes for m in modules) / chip.ici_bw
        if chip.ici_bw > 0 and math.isfinite(chip.ici_bw)
        else 0.0
    )
    hops = sum(m.chain_hops for m in modules)
    chain_s = hops * collective_latency_s + ici_s
    hard_floor_s = max(hbm_s, mm_s, chain_s)
    serial_floor_s = (
        sum(
            max(
                m.hbm_bytes / chip.hbm_bw,
                m.fp8_flops / chip.fp8_flops + m.bf16_flops / chip.bf16_flops,
            )
            for m in modules
        )
        + chain_s
    )
    if hard_floor_s == hbm_s:
        binding = "hbm"
    elif hard_floor_s == mm_s:
        binding = "matmul"
    elif collective_latency_s > 0.0:
        binding = "chain"
    else:
        binding = "ici"
    return StepFloor(
        modules=modules,
        hbm_s=hbm_s,
        mm_s=mm_s,
        ici_s=ici_s,
        chain_s=chain_s,
        hard_floor_s=hard_floor_s,
        serial_floor_s=serial_floor_s,
        binding=binding,
    )


DEFAULT_DTYPES = Dtypes()


def step_floor(
    geom: ModelGeometry,
    chip: ChipSpec,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes = DEFAULT_DTYPES,
    collective_latency_s: float = 0.0,
) -> StepFloor:
    """Compute the floor for one non-speculative or verify forward pass."""
    return _combine_modules(
        pass_costs(geom, par, shape, dt),
        chip,
        collective_latency_s=collective_latency_s,
    )


def spec_cycle_floor(
    geom: ModelGeometry,
    chip: ChipSpec,
    par: Parallel,
    batch: int,
    context: int,
    k: int,
    dt: Dtypes = DEFAULT_DTYPES,
    collective_latency_s: float = 0.0,
) -> StepFloor:
    """Compute the floor for one speculative cycle (verify + k draft passes)."""
    verify_shape = StepShape(
        batch=batch,
        rows_per_seq=k + 1,
        context=context,
        state_writes=k + 1,
    )
    verify_mods = list(pass_costs(geom, par, verify_shape, dt))
    if k <= 0:
        return _combine_modules(
            tuple(verify_mods), chip, collective_latency_s=collective_latency_s
        )

    draft_geom = ModelGeometry(
        hidden=geom.hidden,
        num_layers=1,
        layer_types=("full_attention",),
        num_heads=geom.num_heads,
        num_kv_heads=geom.num_kv_heads,
        head_dim=geom.head_dim,
        attn_output_gate=geom.attn_output_gate,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=geom.conv_kernel,
        num_experts=geom.num_experts,
        top_k=geom.top_k,
        moe_inter=geom.moe_inter,
        shared_inter=geom.shared_inter,
        vocab=geom.vocab,
        mtp_layers=geom.mtp_layers,
        dense_inter=geom.dense_inter,
    )
    draft_shape = StepShape(
        batch=batch, rows_per_seq=1, context=context, state_writes=1
    )
    one_draft = pass_costs(draft_geom, par, draft_shape, dt)

    fc_elems = float(2 * geom.hidden * geom.hidden // par.tp)
    fc_fp8, fc_bf16 = _split_matmul_flops(batch, fc_elems, dt.w_aux, dt.act)
    fc_colls = (
        (
            make(
                "draft_fc_ag",
                "ag",
                float(batch * geom.hidden * 2),
                par.tp,
                par.ar_algo,
                par.ranks_per_chip,
            ),
        )
        * k
        if par.tp > 1
        else ()
    )
    draft_fc = ModuleCost(
        name="draft_fc",
        hbm_bytes=fc_elems * dt.w_aux * k,
        fp8_flops=fc_fp8 * k,
        bf16_flops=fc_bf16 * k,
        collectives=fc_colls,
    )
    for m in one_draft:
        verify_mods.append(
            ModuleCost(
                name=f"draft_{m.name}",
                hbm_bytes=m.hbm_bytes * k,
                fp8_flops=m.fp8_flops * k,
                bf16_flops=m.bf16_flops * k,
                collectives=m.collectives * k,
                hbm_bytes_min=(
                    None if m.hbm_bytes_min is None else m.hbm_bytes_min * k
                ),
            )
        )
    verify_mods.append(draft_fc)
    return _combine_modules(
        tuple(verify_mods), chip, collective_latency_s=collective_latency_s
    )
