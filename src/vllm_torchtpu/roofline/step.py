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
"""Hard and serial step floors for decode, prefill, mixed, and speculative cycles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from .chips import ChipSpec, SliceTopology
from .collectives import (
    Collective,
    collective_ici_hops,
    make,
    submesh_for_group,
    torus_collective_s,
)
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
    """Resource times and binding bottleneck for one step or speculative cycle."""

    modules: tuple[ModuleCost, ...]
    hbm_s: float
    mm_s: float
    ici_s: float
    chain_s: float
    hard_floor_s: float
    serial_floor_s: float
    binding: str
    lat_s: float = 0.0

    @property
    def resource_bound_s(self) -> float:
        return max(self.hbm_s, self.mm_s, self.ici_s)

    @property
    def latency_bound_s(self) -> float:
        return self.lat_s


@lru_cache(maxsize=32768)
def _unit_coll_times(
    c: Collective,
    topology: SliceTopology | None,
    ranks_per_chip: int,
    c_axes: tuple[int, int, int, int] | None,
    chip: ChipSpec,
    use_torus: bool,
    unit_hop_lat_s: float,
) -> tuple[float, float]:
    c_hops = collective_ici_hops(c, topology, ranks_per_chip, c_axes)
    hop_s = c_hops * unit_hop_lat_s
    if use_torus:
        submesh = submesh_for_group(c.group, topology, ranks_per_chip, c_axes)
        anal_s = torus_collective_s(c.kind, c.payload, submesh, chip, topology)
    else:
        wire_s = (
            c.wire_bytes * c.offchip / chip.ici_bw
            if chip.ici_bw > 0 and math.isfinite(chip.ici_bw)
            else 0.0
        )
        anal_s = max(hop_s, wire_s)
    return (hop_s, anal_s)


def _combine_modules(
    modules: tuple[ModuleCost, ...],
    chip: ChipSpec,
    collective_latency_s: float = 0.0,
    *,
    topology: SliceTopology | None = None,
    par: Parallel | None = None,
) -> StepFloor:
    has_ici = chip.ici_bw > 0 and math.isfinite(chip.ici_bw)
    tot_hbm_b = 0.0
    tot_fp8 = 0.0
    tot_bf16 = 0.0
    tot_ici_b = 0.0
    mod_serial_s = 0.0
    for m in modules:
        hb = m.hbm_bytes
        f8 = m.fp8_flops
        b16 = m.bf16_flops
        tot_hbm_b += hb
        tot_fp8 += f8
        tot_bf16 += b16
        if has_ici:
            tot_ici_b += m.ici_bytes
        m_hb_s = hb / chip.hbm_bw
        m_mm_s = f8 / chip.fp8_flops + b16 / chip.bf16_flops
        mod_serial_s += m_hb_s if m_hb_s >= m_mm_s else m_mm_s
    hbm_s = tot_hbm_b / chip.hbm_bw
    mm_s = tot_fp8 / chip.fp8_flops + tot_bf16 / chip.bf16_flops
    ici_s = tot_ici_b / chip.ici_bw if has_ici else 0.0
    mesh_axes = par.mesh_axes if par is not None else None
    ranks_per_chip = par.ranks_per_chip if par is not None else chip.cores_per_chip
    use_torus = topology is not None or mesh_axes is not None

    unit_hop_lat_s = (
        chip.ici_hop_latency_s
        if (
            chip.ici_bw > 0
            and math.isfinite(chip.ici_bw)
            and chip.ici_hop_latency_s > 0
            and math.isfinite(chip.ici_hop_latency_s)
        )
        else 0.0
    )

    coll_counts: dict[tuple[str, float, int, float, float, int], int] = {}
    coll_samples = []
    for m in modules:
        if m.unique_collectives is not None:
            for c, m_cnt in m.unique_collectives:
                if c.group > 1 and m_cnt > 0:
                    ckey = (
                        c.kind,
                        c.payload,
                        c.group,
                        c.wire_bytes,
                        c.offchip,
                        c.hops,
                    )
                    if ckey not in coll_counts:
                        coll_counts[ckey] = m_cnt
                        coll_samples.append((ckey, c))
                    else:
                        coll_counts[ckey] += m_cnt
        else:
            for c in m.collectives:
                if c.group > 1:
                    ckey = (
                        c.kind,
                        c.payload,
                        c.group,
                        c.wire_bytes,
                        c.offchip,
                        c.hops,
                    )
                    if ckey not in coll_counts:
                        coll_counts[ckey] = 1
                        coll_samples.append((ckey, c))
                    else:
                        coll_counts[ckey] += 1

    hop_lat_s = 0.0
    analytic_chain_s = 0.0
    user_chain_lat_s = 0.0
    for ckey, c in coll_samples:
        cnt = coll_counts[ckey]
        c_axes = mesh_axes if (par is not None and c.group == par.tp) else None
        hop_s, anal_s = _unit_coll_times(
            c,
            topology,
            ranks_per_chip,
            c_axes,
            chip,
            use_torus,
            unit_hop_lat_s,
        )
        hop_lat_s += cnt * hop_s
        analytic_chain_s += cnt * anal_s
        if collective_latency_s > 0.0:
            user_chain_lat_s += cnt * (c.hops * collective_latency_s)
    chain_s = analytic_chain_s + user_chain_lat_s

    resource_bound_s = max(hbm_s, mm_s, ici_s)
    latency_bound_s = hop_lat_s
    hard_floor_s = max(resource_bound_s, latency_bound_s, analytic_chain_s)
    serial_floor_s = mod_serial_s + chain_s
    if hard_floor_s == hbm_s:
        binding = "hbm"
    elif hard_floor_s == mm_s:
        binding = "matmul"
    elif hard_floor_s > ici_s or use_torus or hop_lat_s > ici_s:
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
        lat_s=hop_lat_s,
    )


@lru_cache(maxsize=32768)
def step_floor(
    geom: ModelGeometry,
    chip: ChipSpec,
    par: Parallel,
    shape: StepShape,
    dt: Dtypes,
    collective_latency_s: float = 0.0,
    topology: SliceTopology | None = None,
) -> StepFloor:
    """Compute the floor for one decode, prefill, or mixed forward pass."""
    return _combine_modules(
        pass_costs(geom, par, shape, dt, mxu_m_tile=chip.mxu_m_tile),
        chip,
        collective_latency_s=collective_latency_s,
        topology=topology,
        par=par,
    )


@lru_cache(maxsize=32768)
def spec_cycle_floor(
    geom: ModelGeometry,
    chip: ChipSpec,
    par: Parallel,
    batch: int,
    context: int,
    k: int,
    dt: Dtypes,
    collective_latency_s: float = 0.0,
    topology: SliceTopology | None = None,
) -> StepFloor:
    """Compute the floor for one speculative cycle (verify + k draft passes)."""
    verify_shape = StepShape(
        batch=batch,
        rows_per_seq=k + 1,
        context=context,
        state_writes=1,
    )
    verify_mods = list(
        pass_costs(geom, par, verify_shape, dt, mxu_m_tile=chip.mxu_m_tile)
    )
    if k <= 0:
        return _combine_modules(
            tuple(verify_mods),
            chip,
            collective_latency_s=collective_latency_s,
            topology=topology,
            par=par,
        )

    draft_layer_type = (
        "dsa"
        if (geom.kv_lora_rank > 0 and geom.index_topk > 0)
        else ("mla" if geom.kv_lora_rank > 0 else "full_attention")
    )
    use_dense_mtp = geom.mtp_inter > 0
    draft_geom = ModelGeometry(
        hidden=geom.hidden,
        num_layers=1,
        layer_types=(draft_layer_type,),
        num_heads=geom.num_heads,
        num_kv_heads=geom.num_kv_heads,
        head_dim=geom.head_dim,
        attn_output_gate=geom.attn_output_gate,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=geom.conv_kernel,
        num_experts=0 if use_dense_mtp else geom.num_experts,
        top_k=0 if use_dense_mtp else geom.top_k,
        moe_inter=0 if use_dense_mtp else geom.moe_inter,
        shared_inter=0 if use_dense_mtp else geom.shared_inter,
        vocab=geom.vocab,
        mtp_layers=geom.mtp_layers,
        dense_inter=geom.mtp_inter if use_dense_mtp else geom.dense_inter,
        q_lora_rank=geom.q_lora_rank,
        kv_lora_rank=geom.kv_lora_rank,
        qk_nope_head_dim=geom.qk_nope_head_dim,
        qk_rope_head_dim=geom.qk_rope_head_dim,
        v_head_dim=geom.v_head_dim,
        index_n_heads=geom.index_n_heads,
        index_head_dim=geom.index_head_dim,
        index_topk=geom.index_topk,
        sliding_window=geom.sliding_window,
        block_parallel=geom.block_parallel,
        mtp_inter=geom.mtp_inter,
    )
    draft_shape = StepShape(
        batch=batch, rows_per_seq=1, context=context, state_writes=1
    )
    one_draft = pass_costs(draft_geom, par, draft_shape, dt, mxu_m_tile=chip.mxu_m_tile)

    fc_elems = float(2 * geom.hidden * geom.hidden // par.tp)
    fc_fp8, fc_bf16 = _split_matmul_flops(batch, fc_elems, dt.w_aux, dt.act)
    fc_unit = (
        make(
            "draft_fc_ag",
            "ag",
            float(batch * geom.hidden * 2),
            par.tp,
            par.ar_algo,
            par.ranks_per_chip,
        )
        if par.tp > 1
        else None
    )
    draft_fc = ModuleCost(
        name="draft_fc",
        hbm_bytes=fc_elems * dt.w_aux * k,
        fp8_flops=fc_fp8 * k,
        bf16_flops=fc_bf16 * k,
        collectives=(fc_unit,) * k if fc_unit is not None else (),
        unique_collectives=((fc_unit, k),) if fc_unit is not None else (),
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
                unique_collectives=(
                    None
                    if m.unique_collectives is None
                    else tuple((c, cnt * k) for c, cnt in m.unique_collectives)
                ),
            )
        )
    verify_mods.append(draft_fc)
    return _combine_modules(
        tuple(verify_mods),
        chip,
        collective_latency_s=collective_latency_s,
        topology=topology,
        par=par,
    )
