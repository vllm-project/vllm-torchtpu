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
"""One Qwen3.8 segment call as a single SPMD JAX function (framework-free).

`build_segment_call` returns `(fn, in_specs, out_specs, donate_argnums)` for
one segment kind (`L` kernel layers) at one row count `B` (= runner-padded
tokens, `B / T_VERIFY` requests). `fn(hidden, attn, *weights[, pool, *meta])`
pads rows to the kernel's `M`, derives the GDN read / write block ids from the
runner's metadata, gathers the GDN states from the unified pool, runs the
segment kernel, scatters the per-token checkpoints back into the (donated)
pool and returns `(residual[, pool])`. A bridge wraps it as a torch op.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_step import make_flat_segment_step

from .qwen38_gdn_state import gather_gdn_states, scatter_gdn_ckpts
from .qwen38_scatter import scatter_gdn_ckpts_fast

T_VERIFY = 4


def segment_tiles(B: int, hier: int | None = None) -> Tiles:
    """Expert ring on the software-pipelined EP path (e13 in 2 tiles, e2 in 1).

    Ring depths shrink with the bucket so the 64-row scratch fits VMEM; the
    bank sizes (hence every weight's tiling) are the same for all buckets.
    """
    nbe, nbd = (12, 10) if B <= 16 else ((6, 10) if B <= 32 else (3, 4))
    h = 1 if hier is None else hier
    return Tiles(ebank=2 << 20, nbe=nbe, nbd=nbd, vmem_mib=62, hier=h).with_blocks(
        {"moe": {"pipe_ep": True, "moe_unroll": True}}
    )


GDN_META = 5  # per GDN layer: state ids, slot read offsets, ckpt ids, query start loc, seq lens


def _fit(x: Any, n: int) -> Any:
    """First `n` rows of `x`, zero-padded when the runner's array is shorter."""


    x = x[:n]
    if x.shape[0] < n:
        x = jnp.pad(x, ((0, n - x.shape[0]),) + ((0, 0),) * (x.ndim - 1))
    return x


def gdn_blocks(meta: Any, NS: int, write: bool) -> tuple[Any, Any, Any]:
    """`(read [NL,NS], write [NL,NS,T], has_state [NL,NS])` block ids from GDN metadata.

    Requests past the live ones (and every write when `write` is False) point
    at the null block 0.
    """


    rbs, wbs, his = [], [], []
    for i in range(0, len(meta), GDN_META):
        st, off, ck, qsl, seq = meta[i : i + GDN_META]
        st = _fit(st, NS)
        ck = _fit(ck[:, :T_VERIFY], NS)
        qsl = _fit(qsl, NS + 1)
        q_len = qsl[1:] - qsl[:-1]
        live = q_len > 0
        rb = jnp.take_along_axis(ck, off[st][:, None], axis=1)[:, 0]
        rbs.append(jnp.where(live, rb, 0))
        wbs.append(jnp.where(live[:, None] & write, ck, 0))
        his.append(live & ((_fit(seq, NS) - q_len) > 0))
    return jnp.stack(rbs), jnp.stack(wbs), jnp.stack(his)


def build_segment_call(
    mesh: Any,
    g: Any,
    L: int,
    B: int,
    layout: Any,
    write: bool = True,
    *,
    interpret: bool = False,
    tiles: Tiles | None = None,
    head: bool = False,
    pool_gdn: bool = True,
    fast_scatter: bool = True,
) -> tuple[Any, tuple, tuple, tuple]:
    """`(fn, in_specs, out_specs, donate_argnums)` of segment kind `L` at `B` rows.

    `fn(hidden[, attn], *weights[, pool, *meta])`: no `attn` for a head segment
    (`head=True`); `pool` / `meta` when the segment has GDN layers. With
    `pool_gdn` the kernel reads the GDN states from the pool itself; otherwise
    they are gathered into per-layer arrays first. The per-token checkpoints
    are scattered into the (donated) pool after the kernel either way, with
    `qwen38_scatter` (only the changed bytes) or, `fast_scatter=False`, the
    region adapter.
    `interpret` / `tiles` are for CPU tests.
    """
    R = mesh.size
    ex = ExecSpec(
        R=R,
        B=B,
        T=T_VERIFY,
        ctx=128,
        tiles=segment_tiles(B) if tiles is None else tiles,
        interpret=interpret,
    )
    has_gdn = head or L > 1
    pool_gdn = pool_gdn and has_gdn
    spec = qwen38.segment_spec(g, R, L=L, head=head, pool_gdn=pool_gdn)
    fn, _, _, n_w = make_flat_segment_step(mesh, spec, ex)
    M, NS = ex.M, B // T_VERIFY
    tp, rep = P("tp"), P()
    pad = jax.shard_map(
        lambda h: jnp.pad(h, ((0, M - B), (0, 0))),
        mesh=mesh,
        in_specs=tp,
        out_specs=tp,
        check_vma=False,
    )
    unpad = jax.shard_map(lambda r: r[:B], mesh=mesh, in_specs=tp, out_specs=tp, check_vma=False)
    pos = jnp.zeros((B,), jnp.int32)
    n_lead = 1 if head else 2
    n_meta = GDN_META * (L if head else L - 1) if has_gdn else 0

    gather = jax.shard_map(
        lambda pool, rb, hi: gather_gdn_states(
            pool, rb, layout=layout, T=T_VERIFY, has_initial_state=hi, interpret=interpret
        ),
        mesh=mesh,
        in_specs=(tp, rep, rep),
        out_specs=(tp, tp),
        check_vma=False,
    )
    scatter_fn = scatter_gdn_ckpts_fast if fast_scatter else scatter_gdn_ckpts
    scatter = jax.shard_map(
        lambda pool, c, r, wb: scatter_fn(pool, c, r, wb, layout=layout, interpret=interpret),
        mesh=mesh,
        in_specs=(tp, tp, tp, rep),
        out_specs=tp,
        check_vma=False,
    )

    def call(*args):
        hidden = args[0]
        attn = None if head else args[1]
        w = args[n_lead : n_lead + n_w]
        x = pad(hidden)
        named = {} if head else {"qsa_attn": pad(attn)}
        states: tuple = ()
        if has_gdn:
            pool, meta = args[n_lead + n_w], args[n_lead + n_w + 1 :]
            rb, wb, hi = gdn_blocks(meta, NS, write)
            if pool_gdn:
                named.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32))
            else:
                states = gather(pool, rb, hi)
        xs = tuple(named[k] for k in fn.extra_names)
        outs = fn(pos, x, *xs, *w, *states)
        if not has_gdn:
            return (unpad(outs[0]),)
        res, conv_ck, rec_ck = outs
        return unpad(res), scatter(pool, conv_ck, rec_ck, wb)

    in_specs = (tp,) * n_lead + (tp,) * n_w
    if has_gdn:
        in_specs += (tp,) + (rep,) * n_meta
        return call, in_specs, (tp, tp), (n_lead + n_w,)
    return call, in_specs, (tp,), ()
