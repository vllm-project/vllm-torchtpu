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

`Qwen38WholeStep` owns the packed operands of model layers 3..46 (one kernel call per
step)
and the torch op around `build_wholestep_call`. The head groups (layers 0..2) and the
last
QSA layer (47) stay on the segment driver's calls (`Qwen38Segments.head` / the last
segment).

Bridge rules (q32/q35, measured): every operand is a base tensor (no
`execute_as_strided`
copy per call), one custom call per step for layers 3..46, the pool donated and written
in
place inside that call, and no per-step Python work beyond collecting the runner's
tensors.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.wholestep.qsa_full_c import UNION_STATS_REQS

from .qwen38_segment_call import T_VERIFY
from .qwen38_wholestep_cache import QSALayerCache
from .qwen38_kv_scale_guard import check_kv_scales  # q38 HIX700 KV-SCALE-GUARD
from .qwen38_wholestep_call import indexed_options, union_stats_on, wholestep_tiles, live_rows_construct_line  # q38 XKL
# q38 L1-D07B1 begin
from .qwen38_wholestep_call import expand_gdn_meta, gdn_meta_shared  # q38 L1-D07B1 (+ L1-GLUEFIX)
from vllm_torchtpu import envs  # q38 L1-GLUEFIX: VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS in log lines
# q38 L1-D07B1 end
# q38 L1-L47F begin
from .qwen38_l47fold import l47f_on, sink_width, write_sink  # q38 L1-L47F
# q38 L1-L47F end

logger = init_logger(__name__)

FIRST_LAYER, LAST_LAYER = 3, 47  # kernel covers model layers [FIRST_LAYER, LAST_LAYER)
L_WHOLE = LAST_LAYER - FIRST_LAYER


def qsa_layer_cache(attn: Any) -> QSALayerCache:
    kw = getattr(attn, "_select_keywords", {}) or {}
    scales = (attn._k_scale_float, attn._v_scale_float) if attn.fp8_kv else (1.0, 1.0)
    check_kv_scales(scales[0], scales[1], rank=int(getattr(attn, "tp_rank", -1)), where=str(getattr(attn, "prefix", "")))  # q38 KV-SCALE-GUARD
    return QSALayerCache(
        compressed_slot=int(kw.get("compressed_slot", 0)),
        main_slot=int(kw.get("main_slot", -1)),
        k_scale=float(scales[0]),
        v_scale=float(scales[1]),
    )


class _WholeKind:
    """The torch op of the whole-step call; buckets resolve on the concrete row
    count.
    """

    def __init__(
        self,
        mesh: Any,
        g: Any,
        layout: Any,
        qsa: list[QSALayerCache],
        *,
        budget: int,
        max_groups: int,
        norm_eps: float,
        rotary_dim: int,
        write: bool,
        tag: str,
    ) -> None:
        self.mesh, self.g, self.layout, self.qsa = mesh, g, layout, qsa
        self.kw = dict(
            budget=budget,
            max_groups=max_groups,
            norm_eps=norm_eps,
            rotary_dim=rotary_dim,
            write=write,
        )
        self._calls: dict[int, Any] = {}
        self.tag = tag
        # q38 L1-D07B1 begin
        # q38 L1-D07B1: the operand list the module passes (VLLM_TPU_MK_GDN_META_DEDUP at bind).
        self.gdn_shared = gdn_meta_shared()
        # q38 L1-GLUEFIX: GDN layers in the shared run (set at bind); callable_for expands it below the min rows.
        self.n_gdn = 0
        # q38 L1-D07B1 end
        # q38 L1-L47F begin
        # q38 L1-L47F (VLLM_TPU_MK_L47_FOLD): the kernel covers model layers 3..47 (L 45) and the call's third output is
        # layer 47's stock `selected` [rows, token_topk + ratio - 1] (this rank's view of the step's rows).
        self.l47f = l47f_on()
        # q38 L1-L47F end
        name = f"qwen38_mk::{tag}"

        @torch.library.custom_op(name, mutates_args=())
        def op(hidden: torch.Tensor, args: list[torch.Tensor]) -> list[torch.Tensor]:
            return _g1_stash(self, list(self.callable_for(int(hidden.shape[0]))(hidden, *args)), int(hidden.shape[0]))  # q38 L1-G1

        @op.register_fake
        def _(hidden, args):
            out = [torch.empty_like(hidden), torch.empty_like(args[self.pool_index])]
            if union_stats_on():  # this rank's [QSA layers, UNION_STATS_REQS, 1, 128]
                # Torch sees each rank's shard (w08: a buffer sized for all ranks failed
                # the
                # copy, [11, 16, 1, 128] -> [88, 16, 1, 128]); bucket-independent.
                shape = (len(qsa), UNION_STATS_REQS, 1, 128)
                out.append(hidden.new_empty(shape, dtype=torch.int32))
            # q38 L1-L47F begin
            if self.l47f:  # layer 47's stock `selected` for the MTP draft's sink (bucket rows x width)
                out.append(hidden.new_empty((hidden.shape[0], sink_width(self.kw["budget"])), dtype=torch.int32))
            # q38 L1-L47F end
            return out + _side_fake(self, hidden, args)  # q38 HIX700 GDN-SIDECAR: + the 3 sidecar outputs at its bucket ([] off)

        self.op = op
        self.pool_index = -1

    def callable_for(self, B: int) -> Any:
        if B not in self._calls:
            # q38 L1-D07B1 begin
            # q38 L1-D07B1: build_wholestep_call reads the flag again; the operand list must match.
            assert gdn_meta_shared() == self.gdn_shared, "VLLM_TPU_MK_GDN_META_DEDUP changed after bind"
            # q38 L1-D07B1 end
            from .exported_bridge import make_typed_step_wrapper, sharded_jax_callable
            from .qwen38_l47fold import build_wholestep_call_l47f as build_wholestep_call  # q38 L1-L47F (itself when off)

            full, in_specs, outs, donate = build_wholestep_call(
                self.mesh,
                self.g,
                B,
                self.layout,
                self.qsa,
                L=L_WHOLE + int(self.l47f),  # q38 L1-L47F: 45 with the fold
                tiles=served_tiles(B, self.kw["max_groups"]),  # q38 S3A-ENG T5: the one served-tiles source (end of module)
                **self.kw,
            )
            typed = make_typed_step_wrapper(
                f"qwen38_mk_whole_b{B}", full, len(in_specs), len(outs)
            )
            # Program names carry the tag: the write-free shadow and the served program
            # of one bucket are different programs and must never share a name.
            typed = make_typed_step_wrapper(f"qwen38_mk_{self.tag}_b{B}", full, len(in_specs), len(outs))
            self._calls[B] = sharded_jax_callable(
                f"qwen38_mk::{self.tag}_b{B}", typed, mesh=self.mesh,
                input_partition_specs=in_specs, output_partition_specs=outs,
                donate_argnums=donate,
            )
            # q38 L1-D07B1 begin
            # q38 L1-GLUEFIX (VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS): below the min rows build_wholestep_call built the
            # flag-off program `call`, but the module passes the shared GDN operand list (its forward cannot branch on
            # the bucket); expand it here, on the host, to the flag-off list (the same tensor objects in the flag-off
            # order), so this bucket runs exactly the flag-off program on exactly the flag-off operands.
            if self.gdn_shared and not gdn_meta_shared(B):
                assert self.n_gdn > 0 and self.pool_index >= 0, (self.n_gdn, self.pool_index)
                inner, k, n = self._calls[B], self.pool_index + 1, self.n_gdn

                def expanded(hidden: Any, *args: Any) -> Any:
                    return inner(hidden, *expand_gdn_meta(args, k, n))

                self._calls[B] = expanded
                logger.info(
                    "q38 L1-GLUEFIX whole-step (%s) b%d: flag-off GDN program (VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS=%d), "
                    "shared operands expanded on the host", self.tag, B, int(envs.VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS),
                )
            # q38 L1-D07B1 end
        return self._calls[B]


class Qwen38WholeStep(nn.Module):
    """Layers 3..46 in one whole-step kernel call; owns the packed operands as
    buffers.
    """

    def __init__(
        self,
        model: Any,
        g: Any,
        mesh: Any,
        buckets: tuple[int, ...],
        *,
        budget: int,
        max_groups: int,
        write: bool = True,
        tag: str = "whole",
        hold: str = "buffer",
    ) -> None:
        super().__init__()
        from vllm_torchtpu.megakernel.exec_spec import ExecSpec
        from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

        from .qwen38_wholestep_weights import pack_wholestep_operands

        self.g, self.mesh, self.buckets = g, mesh, buckets
        self.layers = list(model.model.layers)  # plain list: the model owns its layers
        ids = list(range(FIRST_LAYER, LAST_LAYER + int(l47f_on())))  # q38 L1-L47F: + model layer 47 (the last QSA layer)
        self.qsa_ids = [i for i in ids if self.layers[i].layer_type == "full_attention"]
        self.gdn_ids = [
            i for i in ids if self.layers[i].layer_type == "linear_attention"
        ]
        # q38 L1-L47F begin
        self.l47f = len(ids) > L_WHOLE
        if self.l47f:
            assert ids[-1] == len(self.layers) - 1 and self.layers[ids[-1]].layer_type == "full_attention", (
                "q38 L1-L47F: layer 47 must be the model's last layer and a QSA layer", ids[-1], len(self.layers))
            logger.info(
                "q38 L1-L47F whole step: model layers %d..%d in one kernel (L %d: %d QSA + %d GDN layers); stock layer "
                "%d skipped, its sink written from the kernel's last QSA layer", ids[0], ids[-1], len(ids),
                len(self.qsa_ids), len(self.gdn_ids), ids[-1])
            # The sink's row update op is built at import (dynamo cannot trace its construction in the forward).
            import vllm_torchtpu.layers.adapter.custom_ops.rows_update_op  # noqa: F401
        # q38 L1-L47F end
        R = mesh.size
        n_att = len(self.qsa_ids)
        pps = max_groups * 4 // 128
        spec = wholestep_spec(
            g,
            R,
            L=len(ids),  # q38 L1-L47F: L_WHOLE when off
            first="qsa",
            pool_gdn=True,
            qsa=dict(budget=budget),
            step=True,
            indexed=dict(
                max_groups=max_groups,
                pages_per_seq=pps,
                carrier="fp8",
                n_layers=n_att,
                **indexed_options(),
            ),
        )
        ex = ExecSpec(
            R=R,
            B=buckets[0],
            T=T_VERIFY,
            ctx=128,
            tiles=wholestep_tiles(buckets[0], max_groups),
        )
        ops = pack_wholestep_operands(model, g, spec, ex, ids)
        # Host round-trip of the small packed operands (operand_hold): w08r c1's
        # whole-step
        # steps ran 18 per-call execute_as_strided programs more than stock (+0.85
        # ms/step).
        from .operand_hold import host_round_trip_small

        n_trip = host_round_trip_small(ops)
        logger.info(
            "Qwen3.8 whole-step: %d of %d operands host round-tripped", n_trip, len(ops)
        )
        from .operand_hold import hbm_summary

        logger.info("Qwen3.8 whole-step: packed operands %s", hbm_summary(ops, model))
        self.n_w = len(ops)
        # How the operands are held decides what the compiled program sees: registered
        # buffers are re-materialised by `execute_as_strided` every step in the served
        # segment
        # MK (agent-1a, q32/q39), parameters passing the same op are not; the Qwen3.5
        # driver
        # keeps a plain list. `hold`: "buffer" | "param" | "list".
        self.hold = hold
        self._ops: list[torch.Tensor] = []
        if hold == "list":
            object.__setattr__(self, "_ops", ops)
        for i, t in enumerate(ops if hold != "list" else ()):
            if hold == "param":
                setattr(self, f"w_{i}", nn.Parameter(t, requires_grad=False))
            else:
                self.register_buffer(f"w_{i}", t, persistent=False)
        attn0 = self.layers[self.qsa_ids[0]].self_attn
        self.norm_eps = float(attn0.q_norm.variance_epsilon)
        self.rotary_dim = int(attn0._select_keywords.get("rotary_dim", 64)); from vllm_torchtpu.hxsmall import construct_lines as _hxs_construct; _hxs_construct(tag, buckets, spec, R, max_groups)  # q38 HXSMALL: GLUE_TRIM / AR0ST construction lines (every engine start; none when off)
        self.budget, self.max_groups, self.write, self.tag = (
            budget,
            max_groups,
            write,
            tag,
        )
        self.pool: torch.Tensor | None = None
        from .hix700_pregate import pregate_bind_line  # q38 S3A-ENG T1
        from .qwen38_wholestep_call import s2_construct_line  # q38 S3A-ENG T2
        _rank = int(getattr(attn0, "tp_rank", -1))  # q38 S3A-ENG: this rank for the bind twins (-1 = unknown, the reader voids it)
        pregate_bind_line(_rank, buckets, max_groups)  # q38 S3A-ENG T1 + HIX-HMERGE fix: the line from the SERVED tiles; refuses an unserved PREGATE
        s2_construct_line(_rank)  # q38 S3A-ENG T2: S2 options bind twin, every engine start, every rank
        from vllm_torchtpu.hxsmall import ar0st_used_line  # q38 S3A-ENG T5
        ar0st_used_line(_rank, tag, buckets, spec, R, max_groups, served_exec)  # q38 S3A-ENG T5: AR0S-T used per bucket, every rank
        from .qwen38_kern import kern_bind_line; kern_bind_line(_rank)  # noqa: E702  q38 HIX-KERN: bind line, every engine start, every rank
        from .qwen38_k8 import k8_bind_line; k8_bind_line(_rank)  # noqa: E702  q38 HIX-KERN K8: per-rank bind line + refusals, every rank
        self.kind: _WholeKind | None = None; live_rows_construct_line(tag, buckets)  # q38 XKL: X-K-LIVE line of record (every engine start)

    def bind_pool(self, pool: torch.Tensor, layout: Any) -> None:
        # q38 L1-L47F begin
        if getattr(self, "l47f", False) and not self.write:  # wholecheck's shadow cuts at layer 47 (not composed)
            raise ValueError("q38 L1-L47F: VLLM_TPU_MK_L47_FOLD and the write-free whole-step shadow are not composed")
        assert l47f_on() == getattr(self, "l47f", False), "VLLM_TPU_MK_L47_FOLD changed after the module was built"
        # q38 L1-L47F end
        self.pool = pool
        # q38 L1-D07B1 begin
        # q38 L1-D07B1 (VLLM_TPU_MK_GDN_META_DEDUP): the call takes the GDN layers' shared read
        # offsets / query starts / seq lens once, then their state ids, then their checkpoint ids.
        self.gdn_shared = gdn_meta_shared()
        if self.gdn_shared:
            logger.info(
                "Qwen3.8 whole-step (%s): GDN metadata shared (VLLM_TPU_MK_GDN_META_DEDUP): "
                "%d GDN layers -> 3 shared + %d per-layer id operands (was %d)",
                self.tag,
                len(self.gdn_ids),
                2 * len(self.gdn_ids),
                5 * len(self.gdn_ids),
            )
            below = [b for b in self.buckets if not gdn_meta_shared(b)]
            if below:  # q38 L1-GLUEFIX: engagement line for STEP / served readers
                logger.info(
                    "q38 L1-GLUEFIX whole-step (%s): GDN metadata shared from %d rows (VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS); "
                    "buckets %s run the flag-off program",
                    self.tag,
                    int(envs.VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS),
                    ",".join(str(b) for b in below),
                )
        # q38 L1-D07B1 end
        from vllm_torchtpu import hixglue as _hxg  # q38 HIX-GLUE GM: per-rank bind line (refuses GM without the shared program)
        if _hxg.on("VLLM_TPU_HXG_GDN_META"):
            from .qwen38_wholestep_call import gm_bind_line

            gm_bind_line(int(getattr(self.layers[self.qsa_ids[0]].self_attn, "tp_rank", -1)), self.tag, self.gdn_shared, self.buckets)
        # Union stats land here inside the compiled call (in place, like the pool); the
        # driver
        # reads them outside it (`log_union_stats`). Room for the largest bucket's
        # requests.
        self.union_stats = (
            torch.zeros(
                (len(self.qsa_ids), UNION_STATS_REQS, 1, 128),
                dtype=torch.int32,
                device=pool.device,
            )
            if union_stats_on()
            else None
        )
        self.kind = _WholeKind(
            self.mesh,
            self.g,
            layout,
            [qsa_layer_cache(self.layers[i].self_attn) for i in self.qsa_ids],
            budget=self.budget,
            max_groups=self.max_groups,
            norm_eps=self.norm_eps,
            rotary_dim=self.rotary_dim,
            write=self.write,
            tag=self.tag,
        )
        self.kind.pool_index = self.n_w
        # q38 L1-D07B1 begin
        self.kind.n_gdn = len(self.gdn_ids)  # q38 L1-GLUEFIX: the shared GDN run's layer count (expand_gdn_meta)
        # q38 L1-D07B1 end
        _side_bind(self, pool, layout)  # q38 HIX700 GDN-SIDECAR: own line, after self.kind (a '; call' after a comment never ran: 00698c8a93..3b9df20142)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.forward_context import get_forward_context

        assert self.pool is not None
        assert self.kind is not None
        md: Any = get_forward_context().attn_metadata
        meta: list[torch.Tensor] = []
        for i in () if self.gdn_shared else self.gdn_ids:  # q38 L1-D07B1: shared below
            m = md[self.layers[i].linear_attn.prefix]
            meta += [
                m.mamba_state_indices.to(torch.int32),
                m.mamba_slot_read_offsets.to(torch.int32),
                m.mamba_ckpt_indices.to(torch.int32),
                m.query_start_loc.to(torch.int32),
                m.seq_lens.to(torch.int32),
            ]
        # q38 L1-D07B1 begin
        if self.gdn_shared:  # q38 L1-D07B1 (VLLM_TPU_MK_GDN_META_DEDUP)
            meta = self._gdn_meta_shared(md)
        # q38 L1-D07B1 end
        tables: list[torch.Tensor] = []
        norms: list[torch.Tensor] = []
        for i in self.qsa_ids:
            a: Any = self.layers[i].self_attn
            tables += [
                md[a.prefix + ".raw_ring"].block_tables.to(torch.int32),
                md[a.main_cache_name].block_tables.to(torch.int32),
                md[a.compressed_cache_name].block_tables.to(torch.int32),
            ]
            norms += [a.k_norm.weight, a.indexer.k_layernorm.weight]
        rope = self.layers[self.qsa_ids[0]].self_attn.rope_cache
        w = self._ops if self.hold == "list" else [getattr(self, f"w_{i}") for i in range(self.n_w)]
        args = [*w, self.pool, *_side_args(self, hidden), *meta, *tables, positions.to(torch.int32),  # q38 HIX700 GDN-SIDECAR: [] off
                query_start_loc.to(torch.int32), context_lens.to(torch.int32), rope, *norms]
        if getattr(self.layers[self.qsa_ids[0]].self_attn, "main_kv_dcp", 1) > 1:
            args.append(self.layers[self.qsa_ids[0]].self_attn.dcp_rank)  # main-KV DCP pair rank
        outs = self.kind.op(hidden, args)
        res, pool = outs[0], outs[1]; _side_store(self, hidden, outs)  # noqa: E702  q38 HIX700 GDN-SIDECAR
        self.pool.copy_(pool)
        if len(outs) > 2 and self.union_stats is not None:
            # Python control flow or a host read here would be traced into the compiled
            # program (w06g j-b1a76c09: ShapeSpecializationError at bucket 32); only
            # copy.
            self.union_stats.copy_(outs[2])
        # q38 L1-L47F begin
        if self.l47f:  # outs[2]: layer 47's stock `selected` -> the MTP draft's shared-selection sink
            write_sink(self.layers[self.qsa_ids[-1]].self_attn, outs[2])
        # q38 L1-L47F end
        return res

    # q38 L1-D07B1 begin
    # q38 L1-D07B1 (VLLM_TPU_MK_GDN_META_DEDUP) -------------------------------------------------
    # The step-level tensors every GDN layer's metadata carries (the runner's ctx tensors, one object
    # per step); only the state / checkpoint ids are per layer (one KV cache group per GDN layer).
    GDN_SHARED_FIELDS = ("mamba_slot_read_offsets", "query_start_loc", "seq_lens")

    def _gdn_meta_shared(self, md: Any) -> list[torch.Tensor]:
        """`off, qsl, seq` of the first GDN layer, then every layer's state ids, then every
        layer's checkpoint ids (`qwen38_wholestep_call.call_shared`'s operand order)."""
        ms = [md[self.layers[i].linear_attn.prefix] for i in self.gdn_ids]
        meta = [getattr(ms[0], f).to(torch.int32) for f in self.GDN_SHARED_FIELDS]
        meta += [m.mamba_state_indices.to(torch.int32) for m in ms]
        meta += [m.mamba_ckpt_indices.to(torch.int32) for m in ms]
        return meta

    def check_gdn_meta_shared(self) -> None:
        """Host side, outside the compiled program: with the shared operand list every GDN layer's
        read offsets, query starts and seq lens must be the first layer's tensors (the same
        objects), or the call would read wrong ids. Raises on any difference (the driver then
        falls back to stock). A no-op with the flag off."""
        if not self.gdn_shared:
            return
        from vllm.forward_context import get_forward_context

        md: Any = get_forward_context().attn_metadata
        ms = [md[self.layers[i].linear_attn.prefix] for i in self.gdn_ids]
        for i, m in zip(self.gdn_ids[1:], ms[1:]):
            for f in self.GDN_SHARED_FIELDS:
                if getattr(m, f) is not getattr(ms[0], f):
                    raise RuntimeError(
                        f"VLLM_TPU_MK_GDN_META_DEDUP: GDN layer {i} has its own {f} (not layer "
                        f"{self.gdn_ids[0]}'s tensor); the shared operand list would be wrong"
                    )

    # q38 L1-D07B1 end
    def log_union_stats(self, calls: int) -> None:
        """How much a request's verify rows share (rank 0's rows; the selection is
        replicated),
        from the last whole-step call. Called by the driver outside the compiled
        program; the
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
            "Qwen3.8 whole-step union stats (call %d, %d layer-requests): rows' "
            "groups / union "
            "p10 %.2f p50 %.2f p90 %.2f; union / max row p50 %.2f; mean union %.0f "
            "groups",
            self._stat_calls,
            int(live.sum()),
            q[0],
            q[1],
            q[2],
            float(torch.quantile(u / m, 0.5)),
            float(u.mean()),
        )


# q38 L1-G1 begin (GDN-COMMIT1, VLLM_TPU_MK_GDN_COMMIT1): with the flag the whole-step call returns 4 more outputs (the
# kernel's checkpoints and the step's write / state ids); the op keeps them on the kind for the runner's commit and returns
# its declared outputs (megakernel_runtime/qwen38_gdn_commit.py `stash`). Off: the list unchanged.
def _g1_stash(kind: Any, outs: list, rows: int) -> list:
    from .qwen38_gdn_commit import stash

    return stash(kind, outs, rows)
# q38 L1-G1 end


# q38 HIX700 GDN-SIDECAR begin (VLLM_TPU_MK_GDN_SIDECAR): the module's persistent sidecar buffers (per rank, allocated at bind) and the per-step
# operands / outputs of the 16-row whole-step call. The buffers are donated by the call and written in place by the kernel (the pool's
# pattern: the copy_ of the donated alias is elided); `side_on` (int32 [1], set before the forward) and `side_wb` (the write ids of the step
# that last filled the sidecar, [NGDN, NS, T]) belong to megakernel_runtime/qwen38_gdn_sidecar_rt.py. Off (or at another bucket) every helper
# is a no-op. Appended; the module's lines are edited only in place.
def _side_active(mod: Any, hidden: Any) -> bool:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import SIDE_ROWS

    return getattr(mod, "side_rec", None) is not None and int(hidden.shape[0]) == SIDE_ROWS


def _side_bind(mod: Any, pool: Any, layout: Any) -> None:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import SIDE_ROWS, side_shapes, sidecar_on

    from .qwen38_segment_call import T_VERIFY

    kind = mod.kind  # bound after the kind: the op's fake (registered on the kind) reads the kind's mirror below
    mod.side_rec = mod.side_conv = mod.side_on = mod.side_wb = kind.side_rec = kind.side_wb = None
    if not sidecar_on() or not getattr(mod, "write", True) or SIDE_ROWS not in tuple(getattr(mod, "buckets", ())):
        return
    n = len(mod.gdn_ids)
    rec_shape, conv_shape = side_shapes(n, SIDE_ROWS, SIDE_ROWS // T_VERIFY, T_VERIFY, layout.n_v, layout.d_k, layout.d_v, layout.conv_dim)
    mod.side_rec = torch.zeros(rec_shape, dtype=torch.bfloat16, device=pool.device)
    mod.side_conv = torch.zeros(conv_shape, dtype=torch.float32, device=pool.device)
    mod.side_on = torch.zeros((1,), dtype=torch.int32, device=pool.device)
    mod.side_wb = torch.zeros((n, SIDE_ROWS // T_VERIFY, T_VERIFY), dtype=torch.int32, device=pool.device)
    kind.side_rec, kind.side_wb = mod.side_rec, mod.side_wb  # _side_fake(kind, ...) declares the 3 sidecar outputs only when these are set
    logger.info("q38 GDN-SIDECAR whole-step (%s) bound: sidecar rec %s bf16, conv %s f32 per rank (%.1f MiB); 16-row bucket only",
                mod.tag, tuple(rec_shape), tuple(conv_shape),
                (mod.side_rec.numel() * 2 + mod.side_conv.numel() * 4) / 2**20)


def _side_args(mod: Any, hidden: Any) -> list:
    """The call's three sidecar operands right after the pool, at the sidecar bucket; [] otherwise."""
    if not _side_active(mod, hidden):
        return []
    return [mod.side_rec, mod.side_conv, mod.side_on]


def _side_store(mod: Any, hidden: Any, outs: Any) -> None:
    """The call's last three outputs (rec, conv: the donated buffers; wb: the step's write ids) back into the module's buffers."""
    if not _side_active(mod, hidden):
        return
    mod.side_rec.copy_(outs[-3])
    mod.side_conv.copy_(outs[-2])
    mod.side_wb.copy_(outs[-1])


def _side_fake(mod: Any, hidden: Any, args: Any) -> list:
    if not _side_active(mod, hidden):
        return []
    k = mod.pool_index
    # q38 HIX700 SIDE-FIX: the write ids output is built from side_wb's shape and dtype on the op input `hidden`. side_wb is the REAL
    # bind-time buffer (not an op input), so empty_like(side_wb) under FakeTensorMode (the aot_compile trace) raised "Please convert all
    # Tensors to FakeTensors first" and the driver disabled the megakernel (r4 j-c431f374, every rank).
    return [torch.empty_like(args[k + 1]), torch.empty_like(args[k + 2]), hidden.new_empty(tuple(mod.side_wb.shape), dtype=mod.side_wb.dtype)]
# q38 HIX700 GDN-SIDECAR end


# q38 S3A-ENG T5 begin: the tiles and ExecSpec of the served whole-step build. `_WholeKind.callable_for` passes `served_tiles` to
# build_wholestep_call, so build_wholestep_call's `tiles is None` branch (pregate_tiles) never runs on the served path (F3); the T5 used line
# (hxsmall.ar0st_used_line, called from Qwen38WholeStep.__init__) plans with exactly this ExecSpec. Appended; nothing above moves.
def served_tiles(B: int, max_groups: int) -> Any:
    """The tiles `_WholeKind.callable_for` hands build_wholestep_call for bucket B."""
    return wholestep_tiles(B, max_groups)


def served_exec(R: int, B: int, max_groups: int) -> Any:
    """The ExecSpec build_wholestep_call builds for bucket B on the served path: (R, B, T_VERIFY, ctx 128, served_tiles, live_rows by
    `_live_rows_on`'s rule without its log lines). interpret and the debug scopes do not reach the AR0S-T decision."""
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.live_rows import live_rows_mode

    from .qwen38_wholestep_call import LIVE_GDN_COMPUTE_MIN_ROWS

    mode = live_rows_mode()
    if mode >= 2 and 0 < B < LIVE_GDN_COMPUTE_MIN_ROWS:
        mode = 1
    return ExecSpec(R=R, B=B, T=T_VERIFY, ctx=128, tiles=served_tiles(B, max_groups), live_rows=mode)
# q38 S3A-ENG T5 end
