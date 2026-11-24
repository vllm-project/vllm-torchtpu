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

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel.blocks.moe.params import MoETunables
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_step import make_flat_segment_step
from vllm_torchtpu.megakernel.ops import rup

from .qwen38_gdn_state import gather_gdn_states, scatter_gdn_ckpts
from .qwen38_scatter import scatter_gdn_ckpts_fast

T_VERIFY = 4


# All-reduce mode and (expert, dense) ring slots when the caller passes none
# (`VLLM_TPU_MEGAKERNEL=...,hier=N,rings=E:D` sets them; rings for every bucket).
DEFAULT_HIER: int = 2
DEFAULT_RINGS: tuple[int, int] | None = None
# `pipeep=0`: the MoE experts on the non-pipelined EP path (A/B of a served factor).
PIPE_EP: bool = True
# `VLLM_TPU_MEGAKERNEL=...,gdnclamp=1`: block ids outside the pool go to the null block
# 0.
CLAMP_BLOCKS: bool = False


def ring_depths(B: int) -> tuple[int, int]:
    """(expert, dense) ring slots per bucket.

    The expert ring must hold more than one expert or expert streaming and
    compute run serially (2.55 vs 1.49 us per expert measured at 48 rows); with
    the HC all-reduce on `hier=2` the deeper rings below fit VMEM up to 64 rows
    (mk/kernel_work/mk-row-slope/report.md, cut C1).
    """
    if B <= 16:
        return 12, 10
    if B <= 48:
        return 6, 10
    return (6, 4) if B <= 64 else (2, 3)


def _safe_valid_rows(qsl: Any, B: int) -> Any:
    """Return active row count `[1] int32` from `query_start_loc`, or `[B]` if empty.

    Warmup dummy tensors (e.g. `c32 DCP=2 mns=12`) may pass an empty 0-element
    `query_start_loc` (`shape == (0,)`), where indexing `qsl[-1]` would raise
    `IndexError`.
    """
    shape = getattr(qsl, "shape", ())
    if shape and shape[0] > 0:
        return jnp.clip(qsl[-1], 0, B).reshape(1).astype(jnp.int32)
    return jnp.asarray([B], jnp.int32)


def moe_e7_hier(B: int, default_hier: int) -> int:
    """Resolve E7 MoE collective hierarchy: HIER=4 (AR2D) >= min_tokens, else HIER=5 (exact_width)."""
    ar2d_on = bool(envs.VLLM_TPU_MK_MOE_E7 or envs.VLLM_TPU_MK_MOE_AR2D)
    exact_on = bool(envs.VLLM_TPU_MK_MOE_E7 or envs.VLLM_TPU_MK_MOE_EXACT_WIDTH)
    min_tok = int(envs.VLLM_TPU_MK_MOE_AR2D_MIN_TOKENS)
    mr = rup(B, 8)
    ar2d_rows_ok = (
        (mr >= 32 and mr % 16 == 0)
        if envs.VLLM_TPU_MK_MOE_AR2D_ANY16
        else (mr % 64 == 0)
    )
    if ar2d_on and min_tok > 0 and B >= min_tok and ar2d_rows_ok:
        return 4
    if exact_on:
        return 5
    return default_hier


def segment_tiles(
    B: int, hier: int | None = None, rings: tuple[int, int] | None = None
) -> Tiles:
    """Expert ring on the software-pipelined EP path (e13 in 2 tiles, e2 in 1).

    Ring depths follow `ring_depths`; the bank sizes (hence every weight's
    tiling) are the same for all buckets. `hier=2` by default (C1: lower
    all-reduce growth per row, frees VMEM for the expert ring).
    """
    nbe, nbd = rings or DEFAULT_RINGS or ring_depths(B)
    h = moe_e7_hier(B, DEFAULT_HIER) if hier is None else hier
    skip_pad = bool(envs.VLLM_TPU_MK_MOE_E7 or envs.VLLM_TPU_MK_MOE_SKIP_PAD)
    exact_width = bool(envs.VLLM_TPU_MK_MOE_E7 or envs.VLLM_TPU_MK_MOE_EXACT_WIDTH)
    moe_cfg: dict[str, Any] = {
        "pipe_ep": PIPE_EP,
        "moe_unroll": True,
        "skip_pad": skip_pad,
        "exact_width": exact_width,
    }
    if envs.VLLM_TPU_MK_MOE_DIRECT_STORE:
        moe_cfg["r2_direct_store"] = True
    if envs.VLLM_TPU_MK_MOE_TOPK_LANES:
        moe_cfg["topk_lanes"] = True
    if envs.VLLM_TPU_MK_MOE_LOCAL_WN:
        moe_cfg["local_wn"] = True
    blocks: dict[str, dict[str, Any]] = {"moe": moe_cfg}
    if envs.VLLM_TPU_MK_GDN_VEC_REC:
        blocks["gdn"] = {"vec_rec": True}
    return Tiles(
        ebank=2 << 20,
        nbe=nbe,
        nbd=nbd,
        vmem_mib=62,
        hier=h,
        ar2d_any16=bool(envs.VLLM_TPU_MK_MOE_AR2D_ANY16),
        ar2d_single_slot=bool(envs.VLLM_TPU_MK_AR2D_SINGLE_SLOT),
        xlayer_prefetch=bool(envs.VLLM_TPU_MK_XLAYER_PREFETCH),
    ).with_blocks(blocks)


# Per GDN layer: state ids, slot read offsets, ckpt ids, query start loc, seq lens
GDN_META = 5


def _fit(x: Any, n: int) -> Any:
    """First `n` rows of `x`, zero-padded when the runner's array is shorter."""

    x = x[:n]
    if x.shape[0] < n:
        x = jnp.pad(x, ((0, n - x.shape[0]),) + ((0, 0),) * (x.ndim - 1))
    return x


def gdn_blocks(
    meta: Any, NS: int, write: bool, n_blocks: int | None = None
) -> tuple[Any, Any, Any]:
    """`(read [NL,NS], write [NL,NS,T], has_state [NL,NS])` block ids from GDN metadata.

    Requests past the live ones (and every write when `write` is False) point
    at the null block 0; with `n_blocks`, so does any id outside `[0, n_blocks)`.
    """

    def inside(x):
        return x if n_blocks is None else jnp.where((x >= 0) & (x < n_blocks), x, 0)

    rbs, wbs, his = [], [], []
    for i in range(0, len(meta), GDN_META):
        st, off, ck, qsl, seq = meta[i : i + GDN_META]
        st = _fit(st, NS)
        ck = _fit(ck[:, :T_VERIFY], NS)
        qsl = _fit(qsl, NS + 1)
        q_len = qsl[1:] - qsl[:-1]
        live = q_len > 0
        rb = jnp.take_along_axis(ck, off[st][:, None], axis=1)[:, 0]
        rbs.append(inside(jnp.where(live, rb, 0)))
        wbs.append(inside(jnp.where(live[:, None] & write, ck, 0)))
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
    flat_smalls: bool = False,
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
    has_gdn = head or L > 1
    ex = ExecSpec(
        R=R,
        B=B,
        # Without GDN layers the token grouping is irrelevant (the MTP draft
        # tail can see any row count).
        T=T_VERIFY if has_gdn or B % T_VERIFY == 0 else 1,
        ctx=128,
        tiles=segment_tiles(B) if tiles is None else tiles,
        interpret=interpret,
    )
    pool_gdn = pool_gdn and has_gdn
    skip_pad = bool(ex.resolved_tiles.block("moe", MoETunables).skip_pad)
    spec = qwen38.segment_spec(g, R, L=L, head=head, pool_gdn=pool_gdn)
    fn, _, _, n_w = make_flat_segment_step(mesh, spec, ex, extra_outs=_ink_extra_outs(spec, interpret, layout))  # q38 X-GLUE-1
    # `flat_smalls`: the packer passed "smalls" as `[L * S / 128, 128]` (no
    # size-1 tiling dim, so torch_tpu does not re-lay it out on every call).
    smalls_at = None
    if flat_smalls:
        flat_names = [n for n, c in fn.weight_layout for _ in range(max(c, 1))]
        smalls_at = flat_names.index("smalls") if "smalls" in flat_names else None
    M, NS = ex.M, B // T_VERIFY
    tp, rep = P("tp"), P()
    pad = jax.shard_map(
        lambda h: jnp.pad(h, ((0, M - B), (0, 0))),
        mesh=mesh,
        in_specs=tp,
        out_specs=tp,
        check_vma=False,
    )
    unpad = jax.shard_map(
        lambda r: r[:B], mesh=mesh, in_specs=tp, out_specs=tp, check_vma=False
    )
    pos = jnp.zeros((B,), jnp.int32)
    n_lead = 1 if head else 2
    n_meta = GDN_META * (L if head else L - 1) if has_gdn else 0

    gather = jax.shard_map(
        lambda pool, rb, hi: gather_gdn_states(
            pool,
            rb,
            layout=layout,
            T=T_VERIFY,
            has_initial_state=hi,
            interpret=interpret,
        ),
        mesh=mesh,
        in_specs=(tp, rep, rep),
        out_specs=(tp, tp),
        check_vma=False,
    )
    scatter_fn = scatter_gdn_ckpts_fast if fast_scatter else scatter_gdn_ckpts
    scatter = jax.shard_map(
        lambda pool, c, r, wb: _g1_scatter(scatter_fn, write and head)(  # q38 L1-G1: no head scatter with VLLM_TPU_MK_GDN_COMMIT1
            pool, c, r, wb, layout=layout, interpret=interpret
        ) if (c.shape, c.dtype) != (pool.shape, pool.dtype) else (c if interpret else pool),  # q38 X-GLUE-1: in-kernel checkpoints pass the pool through
        mesh=mesh,
        in_specs=(tp, tp, tp, rep),
        out_specs=tp,
        check_vma=False,
    )

    def call(*args):
        hidden = args[0]
        attn = None if head else args[1]
        w = args[n_lead : n_lead + n_w]
        if smalls_at is not None:
            w = list(w)
            w[smalls_at] = w[smalls_at].reshape(-1, 1, fn.smalls_width)
            w = tuple(w)
        x = pad(hidden)
        named = {} if head else {"qsa_attn": pad(attn)}
        states: tuple = ()
        pool: Any = None
        wb: Any = None
        if has_gdn:
            pool, meta = args[n_lead + n_w], args[n_lead + n_w + 1 :]
            if skip_pad:
                named["moe_valid_rows"] = (
                    _safe_valid_rows(meta[3], B)
                    if len(meta) > 3
                    else jnp.asarray([B], jnp.int32)
                )
            rb, wb, hi = gdn_blocks(
                meta,
                NS,
                write,
                layout.pool_shape[0] // layout.split if CLAMP_BLOCKS else None,
            )
            if pool_gdn:
                named.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32)); named.update(_ink_xs(fn, wb))  # noqa: E702  q38 X-GLUE-1
            else:
                states = gather(pool, rb, hi)
        elif skip_pad:
            named["moe_valid_rows"] = jnp.asarray([B], jnp.int32)
        xs = tuple(named[k] for k in fn.extra_names)
        outs = fn(pos, x, *xs, *w, *states); outs = _ink_outs(fn, outs, pool, interpret)  # noqa: E702  q38 X-GLUE-1
        if not has_gdn:
            return (unpad(outs[0]),)
        res, conv_ck, rec_ck = outs
        return _g1_ret((unpad(res), scatter(pool, conv_ck, rec_ck, wb)), write and head, conv_ck, rec_ck, wb, meta[0::GDN_META], NS, R)  # q38 L1-G1

    in_specs = (tp,) * n_lead + (tp,) * n_w
    if has_gdn:
        in_specs += (tp,) + (rep,) * n_meta
        return call, in_specs, _g1_specs((tp, tp), write and head, tp, rep), (n_lead + n_w,)  # q38 L1-G1
    return call, in_specs, (tp,), ()
# q38 L1-D07B1 begin


# q38 L1-D07B1 (VLLM_TPU_MK_GDN_META_DEDUP): the whole step's GDN block ids in one pass. Appended after
# `build_segment_call` so the lines `gdn_blocks` and the segment call trace from are unchanged.
GDN_SHARED = 3  # read offsets, query start loc, seq lens: one copy for every GDN layer


def _fit_requests(x: Any, n: int) -> Any:
    """`_fit` along axis 1 (the request axis of stacked `[NL, rows, ...]` ids)."""

    x = x[:, :n]
    if x.shape[1] < n:
        x = jnp.pad(x, ((0, 0), (0, n - x.shape[1])) + ((0, 0),) * (x.ndim - 2))
    return x


def gdn_blocks_shared(
    off: Any,
    qsl: Any,
    seq: Any,
    sts: Any,
    cks: Any,
    NS: int,
    write: bool,
    n_blocks: int | None = None,
) -> tuple[Any, Any, Any]:
    """`gdn_blocks` for `len(sts)` layers whose metadata share `off` / `qsl` / `seq`.

    The runner's GDN metadata objects differ per layer only in their state and
    checkpoint ids (one KV cache group per GDN layer under the GCD group-size rule
    with the MTP draft in the target's buckets); the slot read offsets, query start
    locations and sequence lengths are the step's own tensors, the same objects in
    every layer's metadata (`Qwen38WholeStep.check_gdn_meta_shared` checks that on
    the host). So the per-request terms (`live`, `has_state`) are computed once and
    the per-layer read / write ids in one vectorised gather over `[NL, NS]`, instead
    of `NL` chains of small ops. Integer ops only: bit-identical to
    `gdn_blocks([st_l, off, ck_l, qsl, seq for each l])`.
    """
    assert len(sts) == len(cks) > 0, (len(sts), len(cks))

    def inside(x):
        return x if n_blocks is None else jnp.where((x >= 0) & (x < n_blocks), x, 0)

    if len({tuple(s.shape) for s in sts}) == 1 and len({tuple(c.shape) for c in cks}) == 1:
        # One concatenate per id kind, then the request fit on the stacked array.
        st = _fit_requests(jnp.stack(sts), NS)
        ck = _fit_requests(jnp.stack(cks)[:, :, :T_VERIFY], NS)
    else:
        st = jnp.stack([_fit(s, NS) for s in sts])
        ck = jnp.stack([_fit(c[:, :T_VERIFY], NS) for c in cks])
    qsl = _fit(qsl, NS + 1)
    q_len = qsl[1:] - qsl[:-1]
    live = q_len > 0
    rb = jnp.take_along_axis(ck, off[st][:, :, None], axis=2)[:, :, 0]
    rb = inside(jnp.where(live[None, :], rb, 0))
    wb = inside(jnp.where(live[None, :, None] & write, ck, 0))
    hi = live & ((_fit(seq, NS) - q_len) > 0)
    return rb, wb, jnp.broadcast_to(hi[None, :], rb.shape)
# q38 L1-D07B1 end
# q38 X-GLUE-1 begin
# VLLM_TPU_MK_GDN_CKPT_INK=1: the pool GDN kernel writes its checkpoints into the pool (qwen38_gdn_ckpt_ink); the call
# hands it the write ids and the scatter passes the pool through. Appended; the call's lines are edited only at their ends.


def _ink_extra_outs(spec: Any, interpret: Any, layout: Any) -> tuple:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_ckpt_ink import ink_extra_outs

    return ink_extra_outs(spec, interpret, layout)


def _ink_xs(fn: Any, wb: Any) -> dict:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_ckpt_ink import ink_xs

    return ink_xs(fn, wb)


def _ink_outs(fn: Any, outs: Any, pool: Any, interpret: Any) -> Any:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_ckpt_ink import ink_outs

    return ink_outs(fn, outs, pool, interpret)
# q38 X-GLUE-1 end


# q38 L1-G1 begin (GDN-COMMIT1, VLLM_TPU_MK_GDN_COMMIT1): with the flag a writing HEAD call (the head group of the whole step
# and of the segment program) skips its checkpoint scatter and returns conv_ckpt / rec_ckpt / write ids / state ids for the
# runner's commit (megakernel_runtime/qwen38_gdn_commit.py); body segments and the flag off keep the scatter (each helper
# hands back its first argument). Appended last; the call's lines are edited only in place.
def _g1_scatter(scatter: Any, write: bool) -> Any:
    from .qwen38_gdn_commit import scatter_fn

    return scatter_fn(scatter, write)


def _g1_ret(ret: tuple, write: bool, conv_ck: Any, rec_ck: Any, wb: Any, sts: Any, NS: int, R: int) -> tuple:
    from .qwen38_gdn_commit import call_outputs

    return call_outputs(ret, write, conv_ck, rec_ck, wb, sts, NS, R)


def _g1_specs(specs: tuple, write: bool, tp: Any, rep: Any) -> tuple:
    from .qwen38_gdn_commit import out_specs

    return out_specs(specs, write, tp, rep)
# q38 L1-G1 end
