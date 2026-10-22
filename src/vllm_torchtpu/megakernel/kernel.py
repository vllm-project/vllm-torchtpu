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
"""Generic slot-driven megakernel builder (`KernelContext`, `build`, `make_step`)."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from .comm import AllReduceRing
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .ops import BF16, F32, I32, rmsnorm
from .planner import (
    extra_inputs,
    kernel_out_shapes,
    state_shapes,
    weight_shapes,
)
from .pool.config import POOL_STATE_NAME, cache_exec
from .spec import ModelSpec, SwitchSlot
from .stream import TwoRing
from .vmem import CorePlan, build_scratch, fit_default_tiles

ABLATIONS = ("dense", "experts", "kv", "gdn_state", "attn", "gdn", "comm")
COLLECTIVE_ID = 0


@dataclass
class KernelContext:
    """In-kernel execution context passed to `Block` methods."""

    g: Any
    ex: ExecSpec
    core_plan: CorePlan
    pos_ref: Any
    hid_ref: Any
    W: dict[str, Any]
    ST: dict[str, Any]
    O: dict[str, Any]  # noqa: E741
    out_ref: Any
    extra: dict[str, Any]  # `Block.extra_inputs` refs by name
    scr: dict[str, Any]
    me: Any
    live: Any
    stream: TwoRing
    comm: AllReduceRing
    skip: frozenset[str]
    sc: Any
    norm_x: Any = None
    raw_refs: Any = None

    def block_params(self, name: str) -> Any:
        """Return the build-time parameters computed by `Block.plan_params`."""
        return self.core_plan.block(name)

    def sm(self, slot: Any, name: str, width: int | None = None) -> Any:
        """Read `[1, width]` slice `name` from double-buffered `scr['sm'][slot]`."""
        o, w = self.core_plan.stream.lay[name]
        n = w if width is None else width
        return self.scr["sm"][slot, :, o : o + n]

    def apply_input_norm(self, norm_field: str | None, slot: Any) -> None:
        """Normalize `scr['res']` with `sm[slot, norm_field]` into `scr['xb']`."""
        if norm_field is None:
            self.norm_x = None
            return
        sc_name = "norm_in" if norm_field == "in_norm" else norm_field
        with self.sc(sc_name):
            x = rmsnorm(
                self.scr["res"][...],
                self.sm(slot, norm_field),
                self.g.eps,
                zero_centered=bool(self.g.zero_centered_norm),
            )
            self.scr["xb"][...] = x.astype(BF16)
            self.norm_x = x


def build(
    model_spec: ModelSpec,
    ex: ExecSpec,
) -> tuple[Any, dict[str, Any]]:
    """Build the Pallas megakernel `(call, info)` from `model_spec` and `ex`.

    `ex.tiles=None` selects the default tiles fitted to the model's VMEM budget
    (`vmem.fit_default_tiles`); `info["exec"]` is the `ExecSpec` actually built.
    """
    g = model_spec.config
    if hasattr(g, "check"):
        g.check(ex)
    ex = fit_default_tiles(model_spec, ex)
    skip = ex.skip
    assert set(skip) <= set(ABLATIONS), skip
    tiles = ex.resolve_tiles(g)
    scopes = ex.scopes

    def sc(name: str, level: int = 1):
        return (
            jax.named_scope(name) if int(scopes) >= level else contextlib.nullcontext()
        )

    core_plan, scratch = build_scratch(model_spec, ex)
    p = core_plan.plan
    sp = core_plan.stream
    cp = core_plan.comm
    s_names = list(scratch.keys())

    w_names = p.w_names
    st_names = p.s_names
    o_names = p.o_names
    n_w, n_s = len(w_names), len(st_names)
    blocks = model_spec.all_blocks()
    extras = extra_inputs(model_spec, ex)
    x_names = [x.name for x in extras]

    def kernel(*refs):
        pos_ref, hid_ref = refs[:2]
        W = dict(zip(w_names, refs[2 : 2 + n_w]))
        ST = dict(zip(st_names, refs[2 + n_w : 2 + n_w + n_s]))
        k_ = 2 + n_w + n_s
        extra = dict(zip(x_names, refs[k_ : k_ + len(x_names)]))
        k_ += len(x_names)
        O = dict(zip(o_names, refs[k_ : k_ + len(o_names)]))  # noqa: E741
        out_ref = O["hidden"]
        scr = dict(zip(s_names, refs[k_ + len(o_names) :]))
        pool_key = POOL_STATE_NAME
        raw_refs = (
            (O[pool_key] if bool(ex.interpret) else ST[pool_key])
            if cache_exec(ex).FUSED
            else (ST.get(pool_key) if cache_exec(ex).paged_kv else None)
        )

        me = lax.axis_index("tp").astype(I32)
        row_m = lax.broadcasted_iota(I32, (ex.MB, 1), 0)
        live = row_m < ex.B

        stream = TwoRing(sp, W, scr, me, interpret=bool(ex.interpret), skip=skip, sc=sc)
        comm = AllReduceRing(cp, scr, me, skip=skip, sc=sc)
        ctx = KernelContext(
            g=g,
            ex=ex,
            core_plan=core_plan,
            pos_ref=pos_ref,
            hid_ref=hid_ref,
            W=W,
            ST=ST,
            O=O,
            out_ref=out_ref,
            extra=extra,
            scr=scr,
            me=me,
            live=live,
            stream=stream,
            comm=comm,
            skip=skip,
            sc=sc,
            raw_refs=raw_refs,
        )

        # ------------------------------ layer loop ------------------------------
        def layer(l, carry):  # noqa: E741
            slot = l % 2
            first = l == 0
            with sc("smalls_sync"):
                pltpu.make_async_copy(
                    W["smalls"].at[0], scr["sm"].at[slot], scr["sm_sem"].at[slot]
                ).wait()

                @pl.when(l + 1 < g.L)
                def _():
                    pltpu.make_async_copy(
                        W["smalls"].at[l + 1],
                        scr["sm"].at[1 - slot],
                        scr["sm_sem"].at[1 - slot],
                    ).start()

                for b in blocks:
                    b.on_layer_start(ctx, l, slot)

            stream.begin_layer(l)

            for slot_entry, ar_slot in zip(
                model_spec.layer_type.slots, p.slot_ar_index
            ):
                if isinstance(slot_entry, SwitchSlot):
                    active = slot_entry.active_branches
                    norm_names = {b.input_norm for b in active}
                    assert len(norm_names) == 1, norm_names
                    ctx.apply_input_norm(active[0].input_norm, slot)
                    if len(active) == 1:
                        part = active[0].emit(ctx, l, slot)
                        if ar_slot is not None:
                            comm.add_residual(comm.allreduce(ar_slot, part, first))
                    elif len(active) == 2:
                        c0 = slot_entry.active_counts[0]
                        is_b1 = (l % slot_entry.period) >= c0
                        b0, b1 = active[0], active[1]

                        # Emit allreduce + add_residual inside each branch so
                        # the lowered `scf.if` yields `()` instead of returning
                        # a `vector<MBxHxf32>` tensor across the branch boundary
                        # (which would keep 64 VRegs live across `scf.if`).
                        def _emit_branch(blk, ar_slot=ar_slot):
                            part = blk.emit(ctx, l, slot)
                            if ar_slot is not None:
                                comm.add_residual(comm.allreduce(ar_slot, part, first))

                        stream.cond_branches(
                            is_b1,
                            lambda b1=b1: _emit_branch(b1),
                            lambda b0=b0: _emit_branch(b0),
                            true_branch=1,
                            false_branch=0,
                        )
                    else:
                        raise NotImplementedError(
                            f"SwitchSlot with {len(active)} > 2 active branches "
                            "not yet supported"
                        )
                else:
                    ctx.apply_input_norm(slot_entry.input_norm, slot)
                    part = slot_entry.emit(ctx, l, slot)
                    if ar_slot is not None:
                        comm.add_residual(comm.allreduce(ar_slot, part, first))

            return carry

        # ------------------------------ step prologue ------------------------------
        for i in range(sp.NBD):
            stream.fetch(jnp.int32(i))
        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).start()

        for b in blocks:
            b.prologue(ctx)

        if ex.R > 1:
            barrier = pltpu.get_barrier_semaphore()
            for off in range(1, ex.R):
                pl.semaphore_signal(
                    barrier,
                    1,
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, ex.R - 1)

        scr["res"][...] = jnp.where(live, hid_ref[: ex.MB].astype(F32), 0.0)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        for b in blocks:
            b.init_buffers(ctx)

        lax.fori_loop(0, g.L, layer, 0)

        # ------------------------------ step epilogue ------------------------------
        comm.drain_sends()
        for b in blocks:
            b.epilogue(ctx)

        fn_cp = pltpu.make_async_copy(W["final_norm"], scr["fn"], scr["fn_sem"])
        fn_cp.start()
        fn_cp.wait()
        out_ref[: ex.MB] = rmsnorm(
            scr["res"][...],
            scr["fn"][...],
            g.eps,
            zero_centered=bool(g.zero_centered_norm),
        ).astype(BF16)
        if ex.MB < ex.M:
            out_ref[ex.MB :] = jnp.zeros((ex.M - ex.MB, g.H), BF16)

    wsh = weight_shapes(model_spec, ex, p)
    ssh = state_shapes(model_spec, ex, p)
    osh = kernel_out_shapes(model_spec, ex, p)

    def _mem_space(ms_name: str):
        if ms_name == "smem":
            return pltpu.SMEM
        if ms_name == "vmem":
            return pltpu.VMEM
        return pltpu.HBM

    w_specs = [
        tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in wsh[k])
        if isinstance(wsh[k], tuple) and hasattr(wsh[k][0], "shape")
        else pl.BlockSpec(memory_space=pltpu.HBM)
        for k in w_names
    ]
    s_specs = [
        tuple(
            pl.BlockSpec(memory_space=_mem_space(p.state_decls[k].memory_space))
            for _ in ssh[k]
        )
        if p.state_decls[k].tupled
        else pl.BlockSpec(memory_space=_mem_space(p.state_decls[k].memory_space))
        for k in st_names
    ]
    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.VMEM)]
        + w_specs
        + s_specs
        + [pl.BlockSpec(memory_space=_mem_space(x.memory_space)) for x in extras]
    )
    out_shape = [
        tuple(jax.ShapeDtypeStruct(s.shape, s.dtype) for s in osh[k])
        if (k != "hidden" and p.output_decls[k].tupled)
        else jax.ShapeDtypeStruct(*osh[k])
        for k in o_names
    ]
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM)
        if k == "hidden"
        else tuple(
            pl.BlockSpec(memory_space=_mem_space(p.output_decls[k].memory_space))
            for _ in osh[k]
        )
        if p.output_decls[k].tupled
        else pl.BlockSpec(memory_space=_mem_space(p.output_decls[k].memory_space))
        for k in o_names
    ]

    n_w_leaves = sum(
        len(wsh[k])
        if (isinstance(wsh[k], tuple) and hasattr(wsh[k][0], "shape"))
        else 1
        for k in w_names
    )
    pool_key = POOL_STATE_NAME
    aliases: dict[int, int] = {}
    if (
        cache_exec(ex).FUSED
        and bool(ex.interpret)
        and pool_key in st_names
        and pool_key in o_names
    ):
        raw_s_offset = sum(
            len(ssh[k]) if p.state_decls[k].tupled else 1
            for k in st_names[: st_names.index(pool_key)]
        )
        raw_o_offset = sum(
            len(osh[k]) if (k != "hidden" and p.output_decls[k].tupled) else 1
            for k in o_names[: o_names.index(pool_key)]
        )
        n_raw = len(ssh[pool_key]) if p.state_decls[pool_key].tupled else 1
        for a_i in range(n_raw):
            aliases[2 + n_w_leaves + raw_s_offset + a_i] = raw_o_offset + a_i
    elif tiles.inplace:
        assert ex.R == 1, (
            "tiles.inplace cannot alias rw states inside a communicating "
            "R > 1 pallas_call"
        )
        s_leaf_offset = 0
        for k in st_names:
            st = p.state_decls[k]
            n_s_leaves = len(ssh[k]) if st.tupled else 1
            if st.access == "rw" and k in o_names:
                aliases[2 + n_w_leaves + s_leaf_offset] = o_names.index(k)
            s_leaf_offset += n_s_leaves

    call = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID if ex.R > 1 else None,
            vmem_limit_bytes=tiles.vmem_mib * 1024 * 1024,
            disable_bounds_checks=not tiles.bounds_checks,
        ),
        interpret=(
            ex.interpret
            if isinstance(ex.interpret, pltpu.InterpretParams)
            else pltpu.InterpretParams()
            if ex.interpret
            else False
        ),
        name=f"{model_spec.name}_decode_megakernel",
    )

    info = {
        "exec": ex,
        "plan": p,
        "core_plan": core_plan,
        "T_TOTAL": sp.T_TOTAL,
        "vmem_scratch_bytes": core_plan.vmem_bytes,
        "vmem_shapes": {
            k: (sh, jnp.dtype(dt).name) for k, (sh, dt) in core_plan.vmem_shapes.items()
        },
        "smem_scratch_bytes": core_plan.smem_bytes,
        "smem_shapes": {
            k: (sh, jnp.dtype(dt).name)
            for k, (sh, dt) in (core_plan.smem_shapes or {}).items()
        },
        "s_names": s_names,
        "wshape": wsh,
        "sshape": ssh,
        "oshape": osh,
    }
    return call, info


def fold(x):
    """`[R, N0, ...] -> [R * N0, ...]`: rank-folded state layout for `make_step`."""
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def unfold(x, R: int):
    return x.reshape((R, x.shape[0] // R) + x.shape[1:])


def is_tupled_weight(wsh_entry: Any) -> bool:
    """Return True if a `wshape[k]` entry describes a per-layer weight tuple."""
    return isinstance(wsh_entry, tuple) and hasattr(wsh_entry[0], "shape")


def target_weight_shape(wsh_entry: Any) -> tuple[int, ...]:
    """Return the per-rank target tensor shape for a `wshape[k]` entry."""
    if is_tupled_weight(wsh_entry):
        return wsh_entry[0].shape
    return wsh_entry[0]


def unpack_local_weight(
    x: jax.Array,
    target_sh: tuple[int, ...],
    *,
    name: str = "",
    allow_expert_reshape: bool = False,
) -> jax.Array:
    """Unpack a shard_map-local weight shard to `target_sh`, validating its shape."""
    if x.shape == (1, *target_sh):
        return x[0]
    if x.shape == target_sh:
        return x
    if allow_expert_reshape and len(target_sh) >= 4:
        unpacked_sh = (*target_sh[:-3], target_sh[-3] * target_sh[-2], target_sh[-1])
        if x.shape == (1, *unpacked_sh):
            return x[0].reshape(target_sh)
        if x.shape == unpacked_sh:
            return x.reshape(target_sh)
    raise MegakernelConfigError(
        f"Weight {name!r} local shard has shape {x.shape}, expected "
        f"{(1, *target_sh)} or {target_sh}"
    )


def make_step(
    mesh,
    model_spec: ModelSpec,
    ex: ExecSpec,
    *,
    donate_states: bool = False,
):
    """`jax.jit` `shard_map` step wrapping `build(model_spec, ex)` and state updates."""
    g = model_spec.config
    call, info = build(model_spec, ex)
    ex = info["exec"]
    p = info["plan"]
    w_names = p.w_names
    st_names = p.s_names
    o_names = p.o_names
    wsh = info["wshape"]
    blocks = model_spec.all_blocks()
    extras = extra_inputs(model_spec, ex)

    def local(pos, hidden, weights, states, *extra):
        w = [
            tuple(
                unpack_local_weight(
                    x,
                    target_weight_shape(wsh[k]),
                    name=k,
                    allow_expert_reshape=k in p.dynamic_kinds,
                )
                for x in weights[k]
            )
            if isinstance(weights[k], tuple)
            else unpack_local_weight(
                weights[k],
                target_weight_shape(wsh[k]),
                name=k,
                allow_expert_reshape=k in p.dynamic_kinds,
            )
            for k in w_names
        ]
        s = [
            tuple(
                x.squeeze(0) if x.ndim == len(p.state_decls[k].shape) + 1 else x
                for x in states[k]
            )
            if isinstance(states[k], tuple)
            else (
                states[k].squeeze(0)
                if (
                    p.state_decls[k].memory_space == "smem"
                    and states[k].ndim == len(p.state_decls[k].shape) + 1
                )
                else states[k]
            )
            for k in st_names
        ]
        out_vals = call(pos, hidden, *w, *s, *extra)
        if not isinstance(out_vals, (tuple, list)):
            out_vals = (out_vals,)
        hid = out_vals[0]
        raw_outs = dict(zip(o_names[1:], out_vals[1:]))
        new_states = {k: v for k, v in states.items() if _is_mutable_state(k)}
        for b in blocks:
            new_states.update(b.host_update_states(g, ex, pos, new_states, raw_outs))
        return hid, new_states

    def _is_mutable_state(k: str) -> bool:
        return k not in p.state_decls or (
            not p.state_decls[k].tupled and p.state_decls[k].memory_space == "hbm"
        )

    def _state_pspec(k: str, val: Any):
        if isinstance(val, tuple):
            return tuple(P("tp") for _ in val)
        if k in p.state_decls and p.state_decls[k].memory_space == "smem":
            exp_sh = p.state_decls[k].shape
            if val.shape == exp_sh or val.shape == (1, *exp_sh):
                return P()
            if ex.R > 1 and (
                val.shape == (ex.R, *exp_sh)
                or (
                    val.ndim == len(exp_sh)
                    and val.shape[0] == ex.R * exp_sh[0]
                    and val.shape[1:] == exp_sh[1:]
                )
            ):
                return P("tp")
            raise ValueError(
                f"SMEM state {k!r} has shape {val.shape}, expected {exp_sh} "
                f"or ({ex.R * exp_sh[0]}, *{exp_sh[1:]})"
            )
        return P("tp")

    def _raw_step(pos, hidden, weights, mut_states, ro_states, *extra):
        states = {**mut_states, **ro_states}
        ws = {
            k: tuple(P("tp") for _ in weights[k])
            if isinstance(weights[k], tuple)
            else P("tp")
            for k in weights
        }
        ss = {k: _state_pspec(k, states[k]) for k in states}
        in_specs = (P(), P("tp"), ws, ss) + tuple(
            P("tp") if x.memory_space == "hbm" else P() for x in extras
        )
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=(P("tp"), P("tp")),
            check_vma=False,
        )(pos, hidden, weights, states, *extra)

    donate_argnums = (3,) if donate_states else ()
    _jitted_step = jax.jit(_raw_step, donate_argnums=donate_argnums)

    def _split_states(
        states: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mut = {k: v for k, v in states.items() if _is_mutable_state(k)}
        ro = {k: v for k, v in states.items() if not _is_mutable_state(k)}
        return mut, ro

    class _StepFn:
        def __call__(self, pos, hidden, weights, states, *extra):
            mut, ro = _split_states(states)
            return _jitted_step(pos, hidden, weights, mut, ro, *extra)

        def lower(self, pos, hidden, weights, states, *extra):
            mut, ro = _split_states(states)
            return _jitted_step.lower(pos, hidden, weights, mut, ro, *extra)

    return _StepFn(), info
