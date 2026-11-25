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
"""q38 L1-L47F (L47-FOLD-S1, VLLM_TPU_MK_L47_FOLD, default off, hashed): model layer 47 inside the whole step.

* `Qwen38WholeStep` packs model layers 3..47 (L 45: 12 QSA + 33 GDN layers; the planner's partial last cycle is the
  cycle's QSA layer) and `forward_wholestep` skips layer 47's stock HC mix + `attend` + last segment kernel, going
  straight to the final HC mixer. Layer 47's KV / index-key writes are the post-kernel writer's, like the other 11
  in-kernel QSA layers' (`qwen38_wholestep_cache.wholestep_qsa_writes` over all 12 `qsa_layers`).
* The MTP draft's shared-selection sink (`_selection_sink` on layer 47's attention, `Qwen4ExpMTP.bind_shared_selection`)
  is written from the kernel's last QSA layer: the kernel exports each owned row's selected group ids
  (`wholestep/qsa_full_l47f.py`, output `qsa_l47_sink_ids`, this rank's run of requests), `wrap_call` all-gathers the
  runs over TP (the request shards; rank order = row order, as the stock request-shard `request_shard_blocks`) and
  expands them with the stock `selected_tokens` to the stock `selected` layout `[tokens, token_topk + ratio - 1]`, and
  `write_sink` puts it into the sink with the stock `update_rows_`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs

RATIO = 4  # the indexer's compress ratio (the whole step's writers take compress_ratio=4)
_LOGGED: set = set()


def l47f_on() -> bool:
    return bool(envs.VLLM_TPU_MK_L47_FOLD)


def _log_once(key: str, msg: str, *args: Any) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    import logging

    logging.getLogger("vllm." + __name__).info(msg, *args)


def l47f_options(L: int, att_every: int) -> dict:
    """The indexed-block option of the fold ({} when off). On, the kernel must be the 3..47 whole step."""
    if not l47f_on():
        return {}
    assert L % att_every == 1, f"VLLM_TPU_MK_L47_FOLD=1: the whole step must end on layer 47's QSA layer (L {L})"
    return {"l47_sink": True}


def sink_width(budget: int, ratio: int = RATIO) -> int:
    """Columns of the stock `selected` / the sink: token_topk + ratio - 1 (token_topk = budget groups x ratio)."""
    return budget * ratio + ratio - 1


def stock_lengths(B: int, starts: jax.Array, context: jax.Array) -> jax.Array:
    """Per-token causal lengths exactly as the stock pooled path derives them (`pooled_qsa.qsa_cache_writes`: request
    slots active iff their query range is non-empty, `qsa._token_metadata`, `lengths = valid ? logical + 1 : 0`)."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa import _token_metadata

    starts, context = starts.astype(jnp.int32), context.astype(jnp.int32)
    active = starts[1:] > starts[:-1]
    slots = jnp.where(active, jnp.arange(len(context), dtype=jnp.int32), -1)
    _, logical, valid = _token_metadata(B, starts, context, slots)
    return jnp.where(valid, logical + 1, 0).astype(jnp.int32)


def sink_rows(ids: jax.Array, starts: jax.Array, context: jax.Array, *, B: int, budget: int,
              ratio: int = RATIO) -> jax.Array:
    """`ids [N >= B, budget]` (row t: its selected group ids ascending; slots past its count arbitrary) -> the stock
    `selected` `[B, budget * ratio + ratio - 1]` int32: the first min(len // ratio, budget) slots are the row's
    blocks, the rest -1, then `qsa_rpa.selected_tokens` (block j at columns [j r, (j + 1) r), the partial group's
    tokens after the last block, -1 elsewhere; rows with length 0 all -1)."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens

    lengths = stock_lengths(B, starts, context)
    count = jnp.minimum(lengths // ratio, budget)
    blocks = jnp.where(jnp.arange(budget, dtype=jnp.int32)[None, :] < count[:, None], ids[:B].astype(jnp.int32), -1)
    return selected_tokens(blocks, lengths, budget * ratio, ratio).astype(jnp.int32)


def gather_runs(local: jax.Array, axis_name: str = "tp", gather: bool = True) -> jax.Array:
    """Inside a shard_map: this rank's export `[rows, 1, budget]` -> every rank's runs `[R rows, budget]` in rank order
    (= row order: run r holds requests [r npr, (r + 1) npr)); without the sharded indexer every rank holds all rows."""
    x = local.reshape(local.shape[0], local.shape[-1])
    return jax.lax.all_gather(x, axis_name, axis=0, tiled=True) if gather else x


def wrap_call(call: Any, in_specs: tuple, out_specs: tuple, donate: tuple, *, mesh: Any, B: int, n_att: int,
              dcp: int, gather: bool, budget: int) -> tuple:
    """`build_wholestep_call`'s `call` (whose third output is the kernel's `qsa_l47_sink_ids`) -> the same call whose
    third output is the stock `selected` of layer 47, `[R B, width]` tp-sharded (each rank's shard: the whole step's
    `[B, width]`, as the stock op returns it on every rank). The step's `starts` / `context` are the call's own
    operands (tail: positions, starts, context, trig, 2 x n_att norms, the DCP rank when DCP > 1)."""
    from jax.sharding import PartitionSpec as P

    tp, rep = P("tp"), P()
    assert tuple(out_specs)[:3] == (tp, tp, tp) and all(x == tp for x in tuple(out_specs)[3:]), out_specs  # q38 HIX700 GDN-SIDECAR: + its 3 outputs (rec, conv, write ids)
    k = 2 * n_att + int(dcp > 1)

    def expand(local, starts, context):
        return sink_rows(gather_runs(local, gather=gather), starts, context, B=B, budget=budget)

    expand_sm = jax.shard_map(expand, mesh=mesh, in_specs=(tp, rep, rep), out_specs=tp, check_vma=False)

    def call_l47f(*args):
        res, pool, exp, *rest = call(*args)  # q38 HIX700 GDN-SIDECAR: rest = the sidecar buffers (empty without it)
        return (res, pool, expand_sm(exp, args[-(k + 3)], args[-(k + 2)]), *rest)

    call_l47f.__name__ = f"{getattr(call, '__name__', 'call')}_l47f"
    return call_l47f, in_specs, out_specs, donate


def build_wholestep_call_l47f(mesh: Any, g: Any, B: int, layout: Any, qsa_layers: list, *, L: int = 44,
                              budget: int, **kw: Any) -> tuple:
    """`qwen38_wholestep_call.build_wholestep_call` itself when the fold is off; on, its call wrapped by `wrap_call`."""
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import main_kv_dcp

    from . import qwen38_wholestep_call as wsc

    out = wsc.build_wholestep_call(mesh, g, B, layout, qsa_layers, L=L, budget=budget, **kw)
    if not l47f_on() or not l47f_options(L, g.att_every):  # q38 L1-V5: off -> the call itself, `g` not read (glue32 / D07-b1 stubs)
        return out
    n_att = -(-L // g.att_every)
    gather = bool(wsc.indexed_options().get("shard_index"))
    _log_once("call", "q38 L1-L47F whole step: layers 3..%d in one kernel (L %d, %d QSA layers), layer-47 sink export "
              "(%s), stock selected width %d", 3 + L - 1, L, n_att, "gathered over TP" if gather else "every rank",
              sink_width(budget))
    return wrap_call(*out, mesh=mesh, B=B, n_att=n_att, dcp=main_kv_dcp(), gather=gather, budget=budget)


def write_sink(attn: Any, selected: Any) -> bool:
    """Torch side: layer 47's attention `attn` -> its `_selection_sink` (if bound) gets the step's rows, through the
    stock donating row update (`attend_with_indices`' write). Returns whether it wrote."""
    sink = getattr(attn, "_selection_sink", None)
    if sink is None:
        return False
    from vllm_torchtpu.layers.adapter.custom_ops.rows_update_op import update_rows_

    from vllm_torchtpu.hixglue import pad_cols  # q38 HIX-GLUE SL: no-op pad when widths match

    update_rows_(sink, pad_cols(selected, sink.shape[-1]))
    return True
