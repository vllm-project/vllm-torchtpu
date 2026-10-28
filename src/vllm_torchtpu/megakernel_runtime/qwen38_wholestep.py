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
"""Torch side of the Qwen3.8 whole-step megakernel (`VLLM_TPU_MEGAKERNEL=...,whole=1`).

`Qwen38WholeStep` owns the packed operands of model layers 3..46 (one kernel call per step)
and the torch op around `build_wholestep_call`. The head groups (layers 0..2) and the last
QSA layer (47) stay on the segment driver's calls (`Qwen38Segments.head` / the last segment).

Bridge rules (q32/q35, measured): every operand is a base tensor (no `execute_as_strided`
copy per call), one custom call per step for layers 3..46, the pool donated and written in
place inside that call, and no per-step Python work beyond collecting the runner's tensors.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.wholestep.qsa_full_c import UNION_STATS_REQS

from .qwen38_segment_call import T_VERIFY
from .qwen38_wholestep_cache import QSALayerCache
from .qwen38_wholestep_call import indexed_options, union_stats_on, wholestep_tiles

logger = init_logger(__name__)

FIRST_LAYER, LAST_LAYER = 3, 47  # kernel covers model layers [FIRST_LAYER, LAST_LAYER)
L_WHOLE = LAST_LAYER - FIRST_LAYER


def qsa_layer_cache(attn: Any) -> QSALayerCache:
    kw = getattr(attn, "_select_keywords", {}) or {}
    scales = (attn._k_scale_float, attn._v_scale_float) if attn.fp8_kv else (1.0, 1.0)
    return QSALayerCache(
        compressed_slot=int(kw.get("compressed_slot", 0)),
        main_slot=int(kw.get("main_slot", -1)),
        k_scale=float(scales[0]),
        v_scale=float(scales[1]),
    )


class _WholeKind:
    """The torch op of the whole-step call; buckets resolve on the concrete row count."""

    def __init__(self, mesh: Any, g: Any, layout: Any, qsa: list[QSALayerCache], *,
                 budget: int, max_groups: int, norm_eps: float, rotary_dim: int, write: bool,
                 tag: str) -> None:
        self.mesh, self.g, self.layout, self.qsa = mesh, g, layout, qsa
        self.kw = dict(budget=budget, max_groups=max_groups, norm_eps=norm_eps,
                       rotary_dim=rotary_dim, write=write)
        self._calls: dict[int, Any] = {}
        name = f"qwen38_mk::{tag}"

        @torch.library.custom_op(name, mutates_args=())
        def op(hidden: torch.Tensor, args: list[torch.Tensor]) -> list[torch.Tensor]:
            return list(self.callable_for(int(hidden.shape[0]))(hidden, *args))

        @op.register_fake
        def _(hidden, args):
            out = [torch.empty_like(hidden), torch.empty_like(args[self.pool_index])]
            if union_stats_on():  # this rank's [QSA layers, UNION_STATS_REQS, 1, 128]
                # Torch sees each rank's shard (w08: a buffer sized for all ranks failed the
                # copy, [11, 16, 1, 128] -> [88, 16, 1, 128]); bucket-independent.
                shape = (len(qsa), UNION_STATS_REQS, 1, 128)
                out.append(hidden.new_empty(shape, dtype=torch.int32))
            return out

        self.op = op
        self.pool_index = -1

    def callable_for(self, B: int) -> Any:
        if B not in self._calls:
            from .exported_bridge import make_typed_step_wrapper, sharded_jax_callable
            from .qwen38_wholestep_call import build_wholestep_call

            full, in_specs, outs, donate = build_wholestep_call(
                self.mesh, self.g, B, self.layout, self.qsa, L=L_WHOLE,
                tiles=wholestep_tiles(B, self.kw["max_groups"]), **self.kw,
            )
            typed = make_typed_step_wrapper(f"qwen38_mk_whole_b{B}", full, len(in_specs), len(outs))
            self._calls[B] = sharded_jax_callable(
                f"qwen38_mk::whole_b{B}", typed, mesh=self.mesh,
                input_partition_specs=in_specs, output_partition_specs=outs,
                donate_argnums=donate,
            )
        return self._calls[B]


class Qwen38WholeStep(nn.Module):
    """Layers 3..46 in one whole-step kernel call; owns the packed operands as buffers."""

    def __init__(self, model: Any, g: Any, mesh: Any, buckets: tuple[int, ...], *,
                 budget: int, max_groups: int, write: bool = True, tag: str = "whole",
                 hold: str = "buffer") -> None:
        super().__init__()
        from vllm_torchtpu.megakernel.exec_spec import ExecSpec
        from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

        from .qwen38_wholestep_weights import pack_wholestep_operands

        self.g, self.mesh, self.buckets = g, mesh, buckets
        self.layers = list(model.model.layers)  # plain list: the model owns its layers
        ids = list(range(FIRST_LAYER, LAST_LAYER))
        self.qsa_ids = [i for i in ids if self.layers[i].layer_type == "full_attention"]
        self.gdn_ids = [i for i in ids if self.layers[i].layer_type == "linear_attention"]
        R = mesh.size
        n_att = len(self.qsa_ids)
        pps = max_groups * 4 // 128
        spec = wholestep_spec(
            g, R, L=L_WHOLE, first="qsa", pool_gdn=True, qsa=dict(budget=budget), step=True,
            indexed=dict(max_groups=max_groups, pages_per_seq=pps, carrier="fp8", n_layers=n_att, **indexed_options()),
        )
        ex = ExecSpec(R=R, B=buckets[0], T=T_VERIFY, ctx=128, tiles=wholestep_tiles(buckets[0], max_groups))
        ops = pack_wholestep_operands(model, g, spec, ex, ids)
        # Host round-trip of the small packed operands (operand_hold): w08r c1's whole-step
        # steps ran 18 per-call execute_as_strided programs more than stock (+0.85 ms/step).
        from .operand_hold import host_round_trip_small

        n_trip = host_round_trip_small(ops)
        logger.info("Qwen3.8 whole-step: %d of %d operands host round-tripped", n_trip, len(ops))
        from .operand_hold import hbm_summary

        logger.info("Qwen3.8 whole-step: packed operands %s", hbm_summary(ops, model))
        self.n_w = len(ops)
        # How the operands are held decides what the compiled program sees: registered
        # buffers are re-materialised by `execute_as_strided` every step in the served segment
        # MK (agent-1a, q32/q39), parameters passing the same op are not; the Qwen3.5 driver
        # keeps a plain list. `hold`: "buffer" | "param" | "list".
        self.hold = hold
        if hold == "list":
            object.__setattr__(self, "_ops", ops)
        for i, t in enumerate(ops if hold != "list" else ()):
            if hold == "param":
                setattr(self, f"w_{i}", nn.Parameter(t, requires_grad=False))
            else:
                self.register_buffer(f"w_{i}", t, persistent=False)
        attn0 = self.layers[self.qsa_ids[0]].self_attn
        self.norm_eps = float(attn0.q_norm.variance_epsilon)
        self.rotary_dim = int(attn0._select_keywords.get("rotary_dim", 64))
        self.budget, self.max_groups, self.write, self.tag = budget, max_groups, write, tag
        self.pool = None
        self.kind: _WholeKind | None = None

    def bind_pool(self, pool: torch.Tensor, layout: Any) -> None:
        self.pool = pool
        # Union stats land here inside the compiled call (in place, like the pool); the driver
        # reads them outside it (`log_union_stats`). Room for the largest bucket's requests.
        self.union_stats = (
            torch.zeros((len(self.qsa_ids), UNION_STATS_REQS, 1, 128),
                        dtype=torch.int32, device=pool.device)
            if union_stats_on() else None
        )
        self.kind = _WholeKind(
            self.mesh, self.g, layout,
            [qsa_layer_cache(self.layers[i].self_attn) for i in self.qsa_ids],
            budget=self.budget, max_groups=self.max_groups, norm_eps=self.norm_eps,
            rotary_dim=self.rotary_dim, write=self.write, tag=self.tag,
        )
        self.kind.pool_index = self.n_w

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, query_start_loc: torch.Tensor,
                context_lens: torch.Tensor) -> torch.Tensor:
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        meta: list[torch.Tensor] = []
        for i in self.gdn_ids:
            m = md[self.layers[i].linear_attn.prefix]
            meta += [
                m.mamba_state_indices.to(torch.int32),
                m.mamba_slot_read_offsets.to(torch.int32),
                m.mamba_ckpt_indices.to(torch.int32),
                m.query_start_loc.to(torch.int32),
                m.seq_lens.to(torch.int32),
            ]
        tables: list[torch.Tensor] = []
        norms: list[torch.Tensor] = []
        for i in self.qsa_ids:
            a = self.layers[i].self_attn
            tables += [
                md[a.prefix + ".raw_ring"].block_tables.to(torch.int32),
                md[a.main_cache_name].block_tables.to(torch.int32),
                md[a.compressed_cache_name].block_tables.to(torch.int32),
            ]
            norms += [a.k_norm.weight, a.indexer.k_layernorm.weight]
        rope = self.layers[self.qsa_ids[0]].self_attn.rope_cache
        w = self._ops if self.hold == "list" else [getattr(self, f"w_{i}") for i in range(self.n_w)]
        args = [*w, self.pool, *meta, *tables, positions.to(torch.int32),
                query_start_loc.to(torch.int32), context_lens.to(torch.int32), rope, *norms]
        outs = self.kind.op(hidden, args)
        res, pool = outs[0], outs[1]
        self.pool.copy_(pool)
        if len(outs) > 2 and self.union_stats is not None:
            # Python control flow or a host read here would be traced into the compiled
            # program (w06g j-b1a76c09: ShapeSpecializationError at bucket 32); only copy.
            self.union_stats.copy_(outs[2])
        return res

    def log_union_stats(self, calls: int) -> None:
        """How much a request's verify rows share (rank 0's rows; the selection is replicated),
        from the last whole-step call. Called by the driver outside the compiled program; the
        host read syncs that one step only."""
        if self.union_stats is None:
            return
        self._stat_calls = calls
        s = self.union_stats[: len(self.qsa_ids), :, 0, :4].to("cpu").to(torch.int64)
        union, total, mx, rows = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        live = (union > 0) & (rows > 1)
        if not bool(live.any()):
            return
        u, t, m = union[live].double(), total[live].double(), mx[live].double()
        share = t / u  # 1 = disjoint rows, rows-per-request = identical selections
        q = torch.quantile(share, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
        logger.info(
            "Qwen3.8 whole-step union stats (call %d, %d layer-requests): rows' groups / union "
            "p10 %.2f p50 %.2f p90 %.2f; union / max row p50 %.2f; mean union %.0f groups",
            self._stat_calls, int(live.sum()), q[0], q[1], q[2],
            float(torch.quantile(u / m, 0.5)), float(u.mean()),
        )
