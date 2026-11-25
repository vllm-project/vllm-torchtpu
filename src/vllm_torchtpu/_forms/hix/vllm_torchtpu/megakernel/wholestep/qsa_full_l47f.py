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
"""q38 L1-L47F (L47-FOLD-S1): the MTP draft's shared-selection export from the whole step's last QSA layer.

With VLLM_TPU_MK_L47_FOLD the whole step runs model layers 3..47 (L 45; the last kernel layer is model layer 47, a QSA
layer) and the stock layer-47 path is skipped. That path's `attend` was the only writer of the draft's
`_selection_sink` (models/vllm/qwen4_exp/qsa.py `attend_with_indices`, bound by `Qwen4ExpMTP.bind_shared_selection`).
Under D08 S1 B2 (`union_lean`) the kernel builds no per-row selected ids (only per-request unions with row bit codes,
split by DCP owner), so the indexed block gains option `l47_sink`: at its LAST QSA layer (the block's last layer
ordinal, read from the plan) every owner compacts each of its rows' whole selection (`sel` over every visible group,
the step's completed group g_a included, exactly the set `select_rows` codes) ascending into `budget` slots, with MM's
union-pass arithmetic (`qsa_full_lean._union_prologue` / `_union_pass`, `UNION_CHUNK` slots per pass, so the `[slots,
S]` temporaries stay those of the union compaction), and DMAs the row to output `qsa_l47_sink_ids`
(`[rows, 1, budget]` int32, not per layer; rows = this rank's run of requests x T with the sharded indexer, all rows
without it). Slots past a row's count hold 0. The call (`megakernel_runtime/qwen38_l47fold.py`) gathers the runs
across ranks (the request shards) and expands them to the stock `selected` layout with the stock `selected_tokens`.

The export reads only state `select_rows` already holds for the request (its scores `qi_s[0:T]`, `qi_nv`, `qi_th`,
`qi_tm`) right after the request's union compaction (`_compact` / `_qsplit_compact`, the methods every selection
variant calls once per request and DCP half), so the selection itself, the unions and every other output are
unchanged. Classes here are built only when `l47_sink` is in the indexed options (`lean_step_block`); without it the
spec's block is exactly `qsa_full_idx.lean_step_block`'s.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..exec_spec import ExecSpec
from ..ops import I32
from ..spec import ScratchReq, StateDecl
from . import union_attn as ua
from .qsa_full_idx import QSAFullStepGroupIdxBlock, QSAFullStepIdxBlock
from .qsa_full_idx import lean_step_block as _idx_lean_step_block
from .qsa_full_lean import (
    QSAFullStepGroupLeanBlock,
    QSAFullStepLeanBlock,
    _LeanMixin,
    _union_pass,
    _union_prologue,
)

SINK_KEY = "l47_sink"
SINK_IDS = "qsa_l47_sink_ids"  # [rows, 1, budget] int32 (not per layer): the last QSA layer's per-row selected groups


def block_layers(sp: Any, name: str) -> int:
    """Layers of block `name` in the plan's schedule (the planner's `block_layer_counts`, partial last cycle
    included): the last QSA layer's ordinal is this minus 1."""
    b = sp.block_branch_idx.get(name)
    if b is None or len(sp.branch_counts) == 1:
        return int(sp.L)
    n, rem = divmod(int(sp.L), int(sp.period))
    c = int(sp.branch_counts[b])
    return n * c + min(c, max(0, rem - int(sum(sp.branch_counts[:b]))))


class _L47SinkMixin:
    """`l47_sink`: the last QSA layer exports every owned row's selected group ids (module docstring)."""

    def __init__(self, *, l47_sink: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self.l47_sink = bool(l47_sink)
        assert not self.l47_sink or isinstance(self, _LeanMixin), "the sink export rides the D08 S1 lean block"

    def sink_rows(self, ex: ExecSpec) -> int:
        """Rows of the export on each rank: its run of requests (sharded indexer) or every request, times T."""
        npr = self._shard_rows(ex)[0] if self.shard_index else ex.MB // ex.T
        return npr * ex.T

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        out = super().outputs(g, ex)
        if self.l47_sink:
            out = out + [StateDecl(SINK_IDS, (self.sink_rows(ex), 1, self.budget), I32, "w", per_layer=False)]
        return out

    def host_update_states(self, g, ex, pos, states, raw_outputs):  # noqa: ANN001
        out = super().host_update_states(g, ex, pos, states, raw_outputs)
        if self.l47_sink:
            out[SINK_IDS] = raw_outputs[SINK_IDS]
        return out

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        out = super().scratch(g, ex, plan)
        if self.l47_sink:
            out = out + [
                ScratchReq("qs_row", pltpu.VMEM((1, 1, self.budget), I32)),
                ScratchReq("qs_sem", pltpu.SemaphoreType.DMA((1,))),
            ]
        return out

    def _select(self, ctx: Any, l: Any, slot: Any) -> None:  # noqa: E741
        if self.l47_sink:  # read by the export inside the selection: is this the block's last QSA layer?
            sp = ctx.stream.sp
            ctx._l47_last = sp.block_ord(l, self.name) == block_layers(sp, self.name) - 1
        return super()._select(ctx, l, slot)

    def _compact(self, ctx: Any, q: Any, r: int, code: jax.Array, KU: int, npr: int) -> None:
        super()._compact(ctx, q, r, code, KU, npr)
        if self.l47_sink and r == max(self.dcp, 1) - 1:  # after the request's last union half
            self._sink_export(ctx, q, npr)

    def _qsplit_compact(self, ctx: Any, q: Any, KU: int, npr: int) -> None:
        super()._qsplit_compact(ctx, q, KU, npr)  # type: ignore[misc]  (L1-IDX classes only)
        if self.l47_sink:
            self._sink_export(ctx, q, npr)

    def _sink_export(self, ctx: Any, q: Any, npr: int) -> None:
        """Rows `q T .. q T + T - 1` (request `q`, owned here) at the last QSA layer: each row's selection (the row
        test of `select_rows`, all visible groups) compacted ascending into `budget` slots, one DMA per row to
        `SINK_IDS` at this rank's local row."""
        ex, scr = ctx.ex, ctx.scr
        T, K, S = ex.T, self.budget, self.max_groups // 128
        W = min(K, ua.UNION_CHUNK)
        assert K % W == 0, (K, W)
        consts = ua.union_consts(S, K)
        lid = jax.lax.broadcasted_iota(I32, (S, 128), 0) * 128 + jax.lax.broadcasted_iota(I32, (S, 128), 1)
        i = q - self._run(ctx) * npr if self.shard_index else q

        @pl.when(ctx._l47_last)
        def _():
            for t in range(T):
                r = q * T + t
                nvr = scr["qi_nv"][r][:, :1]
                b = jnp.where(lid < nvr, pltpu.bitcast(scr["qi_s"][t].reshape(S, 128), I32), -1)
                th, tm = scr["qi_th"][r][:, :1], scr["qi_tm"][r][:, :1]
                ov = nvr > K
                sel = (ov & ((b > th) | ((b == th) & (lid <= tm)))) | (~ov & (b >= 0))
                pro = _union_prologue(sel.astype(I32), consts)
                for j0 in range(0, K, W):
                    ids, _ = _union_pass(pro, consts, j0)
                    scr["qs_row"][0, :, pl.ds(j0, W)] = ids
                c = pltpu.make_async_copy(scr["qs_row"], ctx.O[SINK_IDS].at[pl.ds(i * T + t, 1)], scr["qs_sem"].at[0])
                c.start()
                c.wait()


class QSAFullStepL47FLeanBlock(_L47SinkMixin, QSAFullStepLeanBlock):
    """MM's stage-C block (DCP 1 / 2) with the D08 S1 stages and the layer-47 sink export."""


class QSAFullStepGroupL47FLeanBlock(_L47SinkMixin, QSAFullStepGroupLeanBlock):
    """D14's DCP-4 group block with the D08 S1 stages and the layer-47 sink export."""


class QSAFullStepL47FIdxBlock(_L47SinkMixin, QSAFullStepIdxBlock):
    """The L1-IDX block (DCP 1 / 2) with the layer-47 sink export."""


class QSAFullStepGroupL47FIdxBlock(_L47SinkMixin, QSAFullStepGroupIdxBlock):
    """The L1-IDX DCP-4 group block with the layer-47 sink export."""


L47F_CLASSES = {
    QSAFullStepLeanBlock: QSAFullStepL47FLeanBlock,
    QSAFullStepGroupLeanBlock: QSAFullStepGroupL47FLeanBlock,
    QSAFullStepIdxBlock: QSAFullStepL47FIdxBlock,
    QSAFullStepGroupIdxBlock: QSAFullStepGroupL47FIdxBlock,
}


def lean_step_block(indexed: dict, qsa: dict | None) -> Any:
    """`qsa_full_idx.lean_step_block` (so D08 S1 / L1-IDX routing is unchanged) unless `l47_sink` is set: then the
    same block class with the sink export. Refuses the export without a D08 S1 lean block (MM's / D14's own blocks
    keep their per-row path or none; the fold is built on S1)."""
    if not indexed.get(SINK_KEY):
        rest = {k: v for k, v in indexed.items() if k != SINK_KEY}
        return _idx_lean_step_block(rest if SINK_KEY in indexed else indexed, qsa)
    rest = {k: v for k, v in indexed.items() if k != SINK_KEY}
    base = _idx_lean_step_block(dict(rest), qsa)
    cls = L47F_CLASSES.get(type(base)) if base is not None else None
    if cls is None:
        raise ValueError(
            "q38 L1-L47F: the layer-47 sink export needs the D08 S1 lean block (VLLM_TPU_MK_UNION_LEAN / _BOUNDED / "
            f"_SLOT_BATCH); the indexed options build {type(base).__name__ if base is not None else 'MM / D14 block'}"
        )
    return _hxp1a_build(cls, rest, qsa)  # q38 HXP1A: cls(**rest, **(qsa or {}), l47_sink=True) with L1-SELLOOP's select_owned engaged on it (end of module)


# q38 HXP1A begin (P1a: the layer-47 fold composed with L1-SELLOOP and L1-QX; VLLM_TPU_MK_L47_FOLD stays the only gate)
# `lean_step_block` routes the indexed options through qsa_full_idx.lean_step_block (which engages `select_owned` on its block and
# picks the L1-QX subclass when `qx_early` is present), then rebuilds that block's class with the sink export. Two compositions the
# l01 decode env needs were refused (L1-DECIDE12 ALL48 rule-16 leg j-a28706f2):
#   * L1-SELLOOP: `select_owned` is not a constructor option; `_hxp1a_build` takes it out and engages it on the rebuilt block, as
#     qsa_full_idx.lean_step_block does. Bitwise by construction: the owned-slot loop runs the same owned slots in the same order
#     with the same first / last flags, and the export rides `_compact` / `_qsplit_compact` of exactly those slots.
#   * L1-QX: the QX subclasses of the S1 lean / L1-IDX blocks get their fold classes below. The export's scratch (`qs_row`,
#     `qs_sem`) and the early q exchange's buffers / semaphores (`qd_q` / `qdg_q`, `qd_[sr]sem0` / `qdg_[sr]sem0`) are disjoint;
#     same descriptors, same values.
import logging as _hxp1a_logging  # noqa: E402

from .qsa_full_idx import (  # noqa: E402
    QSAFullStepGroupIdxQxBlock,
    QSAFullStepGroupLeanQxBlock,
    QSAFullStepIdxQxBlock,
    QSAFullStepLeanQxBlock,
)


class QSAFullStepL47FLeanQxBlock(_L47SinkMixin, QSAFullStepLeanQxBlock):
    """MM's stage-C block (DCP 1 / 2) with the D08 S1 stages, X-DCP-QX-EARLY and the layer-47 sink export."""


class QSAFullStepGroupL47FLeanQxBlock(_L47SinkMixin, QSAFullStepGroupLeanQxBlock):
    """D14's DCP-4 group block with the D08 S1 stages, X-DCP-QX-EARLY and the layer-47 sink export."""


class QSAFullStepL47FIdxQxBlock(_L47SinkMixin, QSAFullStepIdxQxBlock):
    """The L1-IDX block (DCP 1 / 2) with X-DCP-QX-EARLY and the layer-47 sink export."""


class QSAFullStepGroupL47FIdxQxBlock(_L47SinkMixin, QSAFullStepGroupIdxQxBlock):
    """The L1-IDX DCP-4 group block with X-DCP-QX-EARLY and the layer-47 sink export (the l01 decode form's class)."""


L47F_CLASSES.update({
    QSAFullStepLeanQxBlock: QSAFullStepL47FLeanQxBlock,
    QSAFullStepGroupLeanQxBlock: QSAFullStepGroupL47FLeanQxBlock,
    QSAFullStepIdxQxBlock: QSAFullStepL47FIdxQxBlock,
    QSAFullStepGroupIdxQxBlock: QSAFullStepGroupL47FIdxQxBlock,
})
HXP1A_TRACES: list = []  # (class name, select_owned, qx_early) per fold block built (tests / engagement)
_HXP1A_LOGGED: set = set()


def _hxp1a_build(cls: Any, rest: dict, qsa: dict | None) -> Any:
    """`cls(**rest, **qsa, l47_sink=True)` with `select_owned` (L1-SELLOOP) taken out of `rest` and engaged on the built block."""
    owned = int(rest.get("select_owned", 0) or 0)
    blk = cls(**{k: v for k, v in rest.items() if k != "select_owned"}, **(qsa or {}), l47_sink=True)
    if owned:
        assert hasattr(blk, "engage_select_owned"), ("q38 HXP1A: select_owned needs the D08 S1 lean block", type(blk))
        blk.engage_select_owned(owned)
    rec = (type(blk).__name__, int(getattr(blk, "select_owned", 0) or 0), bool(getattr(blk, "qx_early", False)))
    HXP1A_TRACES.append(rec)
    if rec not in _HXP1A_LOGGED:
        _HXP1A_LOGGED.add(rec)
        _hxp1a_logging.getLogger("vllm." + __name__).info(
            "q38 HXP1A L47F composed: block %s select_owned=%d qx_early=%s (layer-47 fold with L1-SELLOOP / L1-QX)", *rec)
    return blk
# q38 HXP1A end
