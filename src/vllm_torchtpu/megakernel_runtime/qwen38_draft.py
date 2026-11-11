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
"""The Qwen3.8 MTP draft step as one whole-step kernel call (`VLLM_TPU_MK_DRAFT`).

One draft step is the draft decoder layer (QSA attention at the served stage-C packing:
in-kernel indexer, selection and step K/V; HC(attention); HC(MoE) + MoE), the draft's
final HC mixer, `lm_head` and the greedy top token: `wholestep_spec(L=1, first="qsa",
step=True, tail=...)` with `T` query rows per request (1 for a draft loop step, 4 for
draft step 0 over the verify rows). The step's K/V rows go to the draft layer's caches
through the whole-step cache writer, after the kernel (the pool is read-only inside).

Returns the residual after the layer (the next step's hidden carry, as the stock
`Qwen4ExpMTP.forward` second output) and the greedy tokens (the stock
`compute_logits(...).argmax(-1)`). The draft's weights keep their shipped precision:
the HC mix weights stay BF16 (`hc_fp8=False`), the experts as checkpointed.
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torch import nn

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.exec_spec import DebugOptions, ExecSpec, Tiles
from vllm_torchtpu.megakernel.models.qwen38_step import make_flat_segment_step
from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
from vllm_torchtpu.megakernel.wholestep.qsa_full_c import IR_NEW, KR_NEW
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
from vllm_torchtpu.megakernel.wholestep.tail import TOP, greedy_tokens
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import main_kv_dcp

from .qwen38_wholestep_cache import QSALayerCache, wholestep_qsa_writes
from .qwen38_wholestep_call import (
    LONG_CONTEXT,
    QSA_TABLES,
    compressed_dcp,
    indexed_options,
    wholestep_tiles,
)
from .qwen38_wholestep_inputs import qsa_step_inputs

logger = init_logger(__name__)

STEP0_T, LOOP_T = 4, 1  # query rows per request: draft step 0 (verify rows), loop steps
# Skip idle request slots in the served request loop (`QSAFullIndexedBlock.skip_idle`).
SKIP_IDLE = True


def draft_spec(g: Any, R: int, *, budget: int, max_groups: int, vocab_shard: int) -> Any:
    """The one-layer draft-step kernel spec (BF16 HC: the draft keeps its precision).
    With the served request loop, idle request slots are skipped (`skip_idle`: at one
    row per request a 16-row tile holds 16 request slots)."""
    opts = indexed_options()
    return wholestep_spec(
        g,
        R,
        L=1,
        first="qsa",
        pool_gdn=True,
        qsa=dict(budget=budget),
        step=True,
        indexed=dict(
            max_groups=max_groups,
            pages_per_seq=max_groups * 4 // 128,
            carrier="fp8",
            n_layers=1,
            skip_idle=SKIP_IDLE and bool(opts.get("loop_requests")),
            **opts,
        ),
        tail=dict(vocab_shard=vocab_shard),
        hc_fp8=False,
    )


def build_draft_call(
    mesh: Any,
    g: Any,
    B: int,
    T: int,
    layout: Any,
    layer: QSALayerCache,
    *,
    budget: int,
    max_groups: int,
    norm_eps: float,
    rotary_dim: int,
    vocab_shard: int,
    write: bool = True,
    interpret: bool = False,
    tiles: Tiles | None = None,
) -> tuple[Any, tuple, tuple, tuple]:
    """`call(hidden, *weights, pool, ring, main, comp, positions, starts, context, trig,
    k_norm, index_k_norm[, dcp_rank]) -> (residual [B, hc*H], tokens [B] int32, pool)`
    at `B` rows (`B // T` requests of `T` rows each), the pool donated."""
    del layout  # the draft layer has no GDN state; kept for the whole-step signature
    R = mesh.size
    assert B % T == 0, (B, T)
    ex = ExecSpec(
        R=R, B=B, T=T, ctx=128,
        tiles=wholestep_tiles(B, max_groups) if tiles is None else tiles,
        interpret=interpret,
        # VLLM_TPU_MK_SCOPES=N: per-phase named scopes in the kernel (profiling only).
        debug=DebugOptions(scopes=int(os.environ.get("VLLM_TPU_MK_SCOPES", "0") or 0)),
    )
    spec = draft_spec(g, R, budget=budget, max_groups=max_groups, vocab_shard=vocab_shard)
    fn, k_specs, _, n_w = make_flat_segment_step(
        mesh, spec, ex, extra_outs=(KV_NEW, KR_NEW, IR_NEW, TOP)
    )
    M = ex.M
    tp, rep = P("tp"), P()
    _, info = kernel.make_step(mesh, spec, ex)
    targets: list[tuple[int, ...] | None] = []
    for name in info["plan"].w_names:
        sh = info["wshape"][name]
        if kernel.is_tupled_weight(sh):
            targets += [None] * len(sh)
        else:
            targets.append((R * sh[0][0],) + tuple(sh[0][1:]))
    x_specs = k_specs[2 : 2 + len(fn.extra_names)]
    hbm = {x for x, s in zip(fn.extra_names, x_specs) if s == tp}
    pps = max_groups * 4 // 128
    names = (
        "qsa_pool", "qsa_ck", "qsa_cktab", "qsa_nvis", "qsa_tail", "qsa_mtab", "qsa_off",
        "qsa_scales", "qsa_rope", "qsa_ctx", "qsa_ring", "qsa_rope_ck",
    )
    pad = jax.shard_map(
        lambda h: jnp.pad(h, ((0, M - B), (0, 0))), mesh=mesh, in_specs=tp,
        out_specs=tp, check_vma=False,
    )
    unpad = jax.shard_map(
        lambda r: r[:B], mesh=mesh, in_specs=tp, out_specs=tp, check_vma=False
    )
    pos0 = jnp.zeros((B,), jnp.int32)

    def per_rank_inputs(pool, ring, main, comp, positions, starts, context, trig):
        d = qsa_step_inputs(
            pool, [(ring, main, comp)], [(layer.main_slot, layer.compressed_slot)], trig,
            positions, starts, context, M=M, T=T, pages_per_seq=pps, max_groups=max_groups,
            k_scale=layer.k_scale, v_scale=layer.v_scale,
            layer_major_tables=LONG_CONTEXT["hbm_tables"],
        )
        return tuple(d[n] for n in names)

    inputs = jax.shard_map(
        per_rank_inputs, mesh=mesh, in_specs=(tp,) + (rep,) * 7,
        out_specs=tuple(tp if n in hbm else rep for n in names), check_vma=False,
    )
    dcp_size = main_kv_dcp()

    def per_rank_writes(pool, kv, kr, ir, ring, main, comp, positions, starts, context, trig,
                        k_norm, ik_norm, *rank):
        if not write:
            return pool
        Bq = len(positions)
        rows = [(kr[0, :Bq], kv[0, :Bq, kv.shape[-1] // 2 :], ir[0, :Bq])]
        dcp = (rank[0].reshape(1), dcp_size) if dcp_size > 1 else None
        return wholestep_qsa_writes(
            pool, rows, [(k_norm, ik_norm)], [(ring, main, comp)], [layer], positions, trig,
            starts, context, compress_ratio=4, norm_eps=norm_eps, rotary_dim=rotary_dim,
            max_query_len=T, num_requests=M // T, dcp=dcp,
            compressed_dcp=dcp if compressed_dcp() else None,
        )

    writes = jax.shard_map(
        per_rank_writes, mesh=mesh,
        in_specs=(tp, tp, tp, tp) + (rep,) * 9 + ((tp,) if dcp_size > 1 else ()),
        out_specs=tp, check_vma=False,
    )

    def call(*args):
        hidden = args[0]
        w = [
            x if t is None or tuple(x.shape) == t else x.reshape(t)
            for x, t in zip(args[1 : 1 + n_w], targets, strict=True)
        ]
        i = 1 + n_w
        pool = args[i]
        ring, main, comp, positions, starts, context, trig, k_norm, ik_norm = args[
            i + 1 : i + 10
        ]
        rank = args[i + 10 :]
        xs = dict(zip(names, inputs(pool, ring, main, comp, positions, starts, context, trig)))
        outs = fn(pos0, pad(hidden), *(xs[k] for k in fn.extra_names), *w)
        res, kv, kr, ir, top = outs[:5]
        pool = writes(pool, kv, kr, ir, ring, main, comp, positions, starts, context, trig,
                      k_norm, ik_norm, *rank)
        tokens = greedy_tokens(top, R, M)[:B]
        return unpad(res), tokens, pool

    in_specs = (
        (tp,) + (tp,) * n_w + (tp,) + (rep,) * 9 + ((tp,) if dcp_size > 1 else ())
    )
    return call, in_specs, (tp, rep, tp), (1 + n_w,)


class DraftKind:
    """The draft-step torch op for one `T`; resolves the rows count on the input."""

    def __init__(self, mesh: Any, g: Any, layout: Any, layer: QSALayerCache, T: int,
                 **kw: Any) -> None:
        self.mesh, self.g, self.layout, self.layer, self.T, self.kw = (
            mesh, g, layout, layer, T, kw,
        )
        self._calls: dict[int, Any] = {}
        self.pool_index = -1
        tag = f"draft_t{T}"
        self.tag = tag

        @torch.library.custom_op(f"qwen38_mk::{tag}", mutates_args=())
        def op(hidden: torch.Tensor, args: list[torch.Tensor]) -> list[torch.Tensor]:
            return list(self.callable_for(int(hidden.shape[0]))(hidden, *args))

        @op.register_fake
        def _(hidden, args):
            return [
                torch.empty_like(hidden),
                hidden.new_empty((hidden.shape[0],), dtype=torch.int32),
                torch.empty_like(args[self.pool_index]),
            ]

        self.op = op

    def callable_for(self, B: int) -> Any:
        if B not in self._calls:
            from .exported_bridge import make_typed_step_wrapper, sharded_jax_callable

            full, in_specs, outs, donate = build_draft_call(
                self.mesh, self.g, B, self.T, self.layout, self.layer,
                tiles=wholestep_tiles(B, self.kw["max_groups"]), **self.kw,
            )
            typed = make_typed_step_wrapper(
                f"qwen38_mk_{self.tag}_b{B}", full, len(in_specs), len(outs)
            )
            self._calls[B] = sharded_jax_callable(
                f"qwen38_mk::{self.tag}_b{B}", typed, mesh=self.mesh,
                input_partition_specs=in_specs, output_partition_specs=outs,
                donate_argnums=donate,
            )
            logger.info("Qwen3.8 MK draft op %s built at %d rows", self.tag, B)
        return self._calls[B]


class MkDraft(nn.Module):
    """The draft layer + final mixer + lm_head + greedy pick as one op per step.

    Owns the packed draft operands (one set, shared by the step-0 and loop kinds: the
    weights do not depend on `T`). `step(...)` mirrors `Qwen4ExpMTP.forward` followed
    by `compute_logits(...).argmax(-1)`.
    """

    def __init__(self, draft: Any, g: Any, mesh: Any, layout: Any, pool: torch.Tensor,
                 *, budget: int, max_groups: int, norm_eps: float, rotary_dim: int) -> None:
        super().__init__()
        from .qwen38_wholestep import qsa_layer_cache
        from .qwen38_wholestep_weights import pack_wholestep_operands

        mtp_model = draft.model
        layer = mtp_model.layers[0]
        attn = layer.self_attn
        lm = draft.lm_head
        R = mesh.size
        V = int(lm.weight.shape[0])
        vocab = int(getattr(lm, "org_vocab_size", V * R) or V * R)
        if V * R != vocab:
            raise ValueError(f"draft lm_head shards {V} x {R} != vocab {vocab}")
        self.attn_prefix = attn.prefix
        self.raw_ring_name = attn.prefix + ".raw_ring"
        self.main_name, self.comp_name = attn.main_cache_name, attn.compressed_cache_name
        cache = qsa_layer_cache(attn)
        kw = dict(budget=budget, max_groups=max_groups, norm_eps=norm_eps,
                  rotary_dim=rotary_dim, vocab_shard=V)
        object.__setattr__(self, "kinds", {
            t: DraftKind(mesh, g, layout, cache, t, **kw) for t in (STEP0_T, LOOP_T)
        })
        spec = draft_spec(g, R, budget=budget, max_groups=max_groups, vocab_shard=V)
        ex = ExecSpec(R=R, B=4, T=LOOP_T, ctx=128, tiles=wholestep_tiles(4, max_groups))
        ops = pack_wholestep_operands(
            draft, g, spec, ex, [0], tail=(mtp_model.hyper_connection_mixer, lm, V)
        )
        self.n_w = len(ops)
        for i, t in enumerate(ops):
            self.register_buffer(f"w_{i}", t, persistent=False)
        object.__setattr__(self, "pool", pool)
        self.k_norm = attn.k_norm.weight
        self.ik_norm = attn.indexer.k_layernorm.weight
        self.rope = attn.rope_cache
        self.dcp_rank = attn.dcp_rank if getattr(attn, "main_kv_dcp", 1) > 1 else None
        for kind in self.kinds.values():
            kind.pool_index = self.n_w

    def step(self, hidden: torch.Tensor, positions: torch.Tensor, T: int,
             metadata: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """`(residual, tokens)` of one draft step on the fused input `hidden`
        `[rows, hc*H]` (`Qwen4ExpMTP.fuse_inputs`), `T` rows per request."""
        md = metadata[self.attn_prefix]
        starts = md.query_start_loc.to(torch.int32)
        seq = md.seq_lens.to(torch.int32)
        context = seq - (starts[1:] - starts[:-1])
        args = [getattr(self, f"w_{i}") for i in range(self.n_w)] + [
            self.pool,
            metadata[self.raw_ring_name].block_tables.to(torch.int32),
            metadata[self.main_name].block_tables.to(torch.int32),
            metadata[self.comp_name].block_tables.to(torch.int32),
            positions.to(torch.int32),
            starts,
            context,
            self.rope,
            self.k_norm,
            self.ik_norm,
        ]
        if self.dcp_rank is not None:
            args.append(self.dcp_rank)  # main-KV DCP pair rank
        res, tokens, pool = self.kinds[T].op(hidden, args)
        self.pool.copy_(pool)  # as Qwen38WholeStep: the donated pool aliases in place
        return res, tokens
