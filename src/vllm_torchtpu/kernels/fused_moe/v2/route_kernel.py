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
"""The fused-EP v2 router epilogue as one Pallas pass over softmax scores.

After a softmax scoring, the layer runs `router_ops.pallas_select` (top-k by
repeated max-and-mask over a row-major [rows, experts] tile: two cross-lane
reductions per pick), XLA's `_row_routes` guard and the per-expert pair count
the routing exchange carries (`pair_block_hist(ids).sum(0)`). This kernel
returns the same three things from one read of the scores:

  * the top-k by the same rule -- the row maximum, the lowest expert equal to
    it, that expert removed, `topk` times. Each tile of 128 tokens is
    transposed in VMEM to [experts, tokens], so a pick is one lexicographic
    (value, lowest expert) tree across the vreg rows plus a sublane reduction
    instead of lane reductions, and the removal is a mask of the previous
    pick in the tree's leaves. Two tiles are interleaved per grid step where
    the token count allows, which overlaps one tile's tree with the other's
    reductions;
  * the route of every token, `_row_routes(scores, gate)`;
  * the number of selected pairs per expert over the routed tokens, in the
    selector's expert numbering (the caller applies the mesh relabel).

It relies on the scoring being a softmax (`jax.nn.softmax`), which makes every
row either all finite or all NaN: a NaN or +inf logit poisons the row's max and
sum, all -inf logits give 0/0, and a -inf logit next to a finite one scores 0.
Hence a row routes iff its first score is finite (an all non-finite gate always
yields an all-NaN row, so the gate test in `_row_routes` is implied); a routed
row has 512 finite scores above NEG, so its 10 picks are distinct and are
exactly the entries the masks leave at NEG, which is what the count reads once
per tile. Rows that do not route compute values nothing reads: the layer
replaces their weights by 0 and their ids by e_total exactly as before.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs

NEG = -3.0e38  # router_ops.NEG: below any softmax score
TOKENS = 128  # tokens per tile: the lane width of the transposed tile
OUT_SUBLANES = 16


def route_kernel_supported(tokens, experts, topk):
    return tokens % TOKENS == 0 and experts % 8 == 0 and 0 < topk <= OUT_SUBLANES


def _precondition_waived():
    # Test mutant: the softmax precondition check removed (it must be caught).
    return envs.MOE_FUSED_EP_V2_TEST_MUTANT == "re_gate"


def route_kernel_applies(*, plan_k2, scoring_fn, has_hash_ids, has_score_bias, tokens, experts, topk):
    """Whether the layer may use this kernel: the plain selector path of the plan
    kernel, a softmax scoring (the precondition above), at a supported shape."""
    return bool(
        plan_k2
        and (scoring_fn == "softmax" or _precondition_waived())
        and not has_hash_ids
        and not has_score_bias
        and route_kernel_supported(tokens, experts, topk)
    )


def _sub_tiles(tokens):
    return 2 if tokens % (2 * TOKENS) == 0 else 1


def _lex(a, b, ties_high=False):
    """Combine (value, expert) pairs; `b` holds the higher experts and wins only
    on a strictly larger value, so ties keep the lower expert."""
    take = (b[0] >= a[0]) if ties_high else (b[0] > a[0])
    return jnp.where(take, b[0], a[0]), jnp.where(take, b[1], a[1])


def _pairwise(leaves, ties_high):
    """Balanced reduction of the leaves in expert order, few values live."""
    stack = []
    for leaf in leaves:
        node, lvl = leaf, 0
        while stack and stack[-1][0] == lvl:
            node, lvl = _lex(stack.pop()[1], node, ties_high), lvl + 1
        stack.append((lvl, node))
    node = stack.pop()[1]
    while stack:
        node = _lex(stack.pop()[1], node, ties_high)
    return node


def _body(s_ref, w_ref, i_ref, r_ref, c_ref, x_ref, cnt_ref, e_ref, *, topk, n, steps, sub, mutant):
    step = pl.program_id(0)
    rows = n // 8

    @pl.when(step == 0)
    def _():
        cnt_ref[...] = jnp.zeros_like(cnt_ref)
        e_ref[...] = lax.broadcasted_iota(jnp.int32, (n, TOKENS), 0)

    routes = []
    for u in range(sub):
        xt = s_ref[pl.ds(u * TOKENS, TOKENS), :].T  # [n, TOKENS]
        routes.append(jnp.isfinite(xt[0:1, :]))
        x_ref[u] = xt
    ws = [[] for _ in range(sub)]
    ids = [[] for _ in range(sub)]
    prev = [None] * sub
    for j in range(topk + 1):
        for u in range(sub):
            leaves = []
            for r in range(rows):
                xr = x_ref[u, pl.ds(8 * r, 8), :]
                er = e_ref[pl.ds(8 * r, 8), :]
                if prev[u] is not None:
                    xr = jnp.where(er == prev[u], NEG, xr)
                    x_ref[u, pl.ds(8 * r, 8), :] = xr
                leaves.append((xr, er))
            if j == topk:
                continue
            hi = mutant == "re_hi" and j == 0
            v8, e8 = _pairwise(leaves, hi)
            m = jnp.max(v8, axis=0, keepdims=True)
            if hi:
                idx = jnp.max(jnp.where(v8 == m, e8, 0), axis=0, keepdims=True)
            else:
                idx = jnp.min(jnp.where(v8 == m, e8, n - 1), axis=0, keepdims=True)
            ws[u].append(m)
            ids[u].append(idx)
            prev[u] = jnp.broadcast_to(idx, (8, TOKENS))
    pad = OUT_SUBLANES - topk
    for u in range(sub):
        live = routes[u] if mutant != "re_cnt" else jnp.ones_like(routes[u])
        for r in range(rows):
            picked = (x_ref[u, pl.ds(8 * r, 8), :] == NEG) & live
            cnt_ref[pl.ds(8 * r, 8), :] += jnp.where(picked, 1, 0)
        wt = jnp.concatenate(ws[u] + [jnp.zeros((pad, TOKENS), jnp.float32)] * min(pad, 1), axis=0)
        it = jnp.concatenate(ids[u] + [jnp.zeros((pad, TOKENS), jnp.int32)] * min(pad, 1), axis=0)
        w_ref[pl.ds(u * TOKENS, TOKENS), :] = wt.T[:, :topk]
        i_ref[pl.ds(u * TOKENS, TOKENS), :] = it.T[:, :topk]
        r_ref[:, pl.ds(u * TOKENS, TOKENS)] = routes[u].astype(jnp.int32)

    @pl.when(step == steps - 1)
    def _():
        c_ref[...] = jnp.sum(cnt_ref[...].T, axis=0, keepdims=True)


def route_epilogue(scores, *, topk, scoring_fn, mutant=None, interpret=None):
    """Softmax scores [t, n] f32 -> (weights [t, topk] f32, ids [t, topk] i32,
    routes [1, t] i32 (1 = routes), counts [1, n] i32).

    weights and ids are pallas_select's on every routed row.
    """
    t, n = scores.shape
    if scoring_fn != "softmax" and not _precondition_waived():
        raise ValueError(f"route kernel: needs softmax scores (all finite or all NaN per row); got {scoring_fn!r}")
    if scores.dtype != jnp.float32 or not route_kernel_supported(t, n, topk):
        raise ValueError(f"route kernel: unsupported scores {scores.shape} {scores.dtype}, topk={topk}")
    if mutant is None:
        m = envs.MOE_FUSED_EP_V2_TEST_MUTANT
        mutant = m if m.startswith("re_") else ""
    if interpret is None:
        interpret = envs.MOE_FUSED_EP_V2_INTERPRET
    sub = _sub_tiles(t)
    tb = sub * TOKENS
    steps = t // tb
    return pl.pallas_call(
        functools.partial(_body, topk=topk, n=n, steps=steps, sub=sub, mutant=mutant),
        grid=(steps,),
        in_specs=[pl.BlockSpec((tb, n), lambda i: (i, 0))],
        out_specs=(pl.BlockSpec((tb, topk), lambda i: (i, 0)),
                   pl.BlockSpec((tb, topk), lambda i: (i, 0)),
                   pl.BlockSpec((1, tb), lambda i: (0, i)),
                   pl.BlockSpec((1, n), lambda i: (0, 0))),
        out_shape=(jax.ShapeDtypeStruct((t, topk), jnp.float32),
                   jax.ShapeDtypeStruct((t, topk), jnp.int32),
                   jax.ShapeDtypeStruct((1, t), jnp.int32),
                   jax.ShapeDtypeStruct((1, n), jnp.int32)),
        scratch_shapes=[pltpu.VMEM((sub, n, TOKENS), jnp.float32),
                        pltpu.VMEM((n, TOKENS), jnp.int32),
                        pltpu.VMEM((n, TOKENS), jnp.int32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=(pltpu.InterpretParams() if interpret else False),
        name="fused_ep_route_epilogue" + (("_mut" + mutant) if mutant else ""),
    )(scores)


def relabel_counts(counts, *, g_local, mesh_ep_ranks):
    """counts [n] in the selector's numbering -> the mesh numbering that
    `layer._relabel_expert_ids_to_mesh_order` gives the ids."""
    if mesh_ep_ranks is None:
        return counts
    blocks = {mesh_index: counts[ep_rank * g_local:(ep_rank + 1) * g_local]
              for mesh_index, ep_rank in enumerate(mesh_ep_ranks)}
    return jnp.concatenate([blocks[i] for i in range(len(mesh_ep_ranks))])
