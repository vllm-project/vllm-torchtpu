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
"""Two-ring weight streamer (`TwoRing`: static `dpool` + dynamic routed `epool`)."""

from __future__ import annotations

import contextlib
from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import pool_alias
from .ops import F32
from .vmem import StreamParams

NN = (((1,), (0,)), ((), ()))


class TwoRing:
    """Dual weight streaming ring: `dpool` for static, `epool` for dynamic weights.

    Maintains `self.cursor` across slots inside a layer and supports scoped
    consumption + optional lazy slot refill via `with stream.segment(lazy=...):`.
    Note: `dmatmul` and `segment` mutate Python-level cursor/offset tracking at trace
    time and must only be called at the top level of `Block.emit` (or inside branches
    dispatched via `stream.cond_branches`), never inside a `lax.fori_loop` or `pl.when`.
    """

    def __init__(
        self,
        sp: StreamParams,
        W: dict[str, Any],
        scr: dict[str, Any],
        me: Any,
        *,
        interpret: bool = False,
        skip: frozenset[str] = frozenset(),
        sc: Any = None,
    ):
        self.sp = sp
        self.W = W
        self.scr = scr
        self.me = me
        self.interpret = interpret
        self.skip = skip
        self.sc = sc if sc is not None else (lambda *_: contextlib.nullcontext())
        self.cursor: Any = jnp.int32(0)
        self._lazy: bool = False
        self._seg_tiles: int = 0
        self._cursor_delta: int = 0
        self._cond_depth: int = 0
        self._in_segment: bool = False
        self._branch_offsets: list[int] | None = None
        self._active_branches: tuple[int, ...] = tuple(range(len(sp.branch_counts)))
        self._cur_layer: Any = None
        self._ering_busy: bool = False

    def begin_layer(self, lyr: Any) -> Any:
        """Reset `self.cursor` and per-branch segment offsets at layer `lyr`."""
        assert not self._in_segment, (
            "begin_layer() cannot be called inside stream.segment()"
        )
        self._cur_layer = lyr
        self.cursor = self.sp.layer_start_tile(lyr)
        self._cursor_delta = 0
        self._branch_offsets = [0 for _ in self.sp.branch_counts]
        self._active_branches = tuple(range(len(self.sp.branch_counts)))
        return self.cursor

    def cond_branches(
        self,
        pred: Any,
        true_fn: Any,
        false_fn: Any,
        *,
        true_branch: int = 0,
        false_branch: int = 1,
    ) -> Any:
        """Execute `lax.cond(pred, true_fn, false_fn)` restoring `self.cursor`."""
        assert not self._in_segment, (
            "cond_branches() cannot be called inside stream.segment()"
        )
        g0 = jnp.asarray(self.cursor, dtype=jnp.int32)
        prev_active = self._active_branches
        prev_delta = self._cursor_delta
        dt_box = [0]
        df_box = [0]

        def _t():
            self.cursor = g0
            self._cursor_delta = 0
            if len(self.sp.branch_counts) >= 2:
                self._active_branches = (true_branch,)
            out = true_fn()
            dt_box[0] = self._cursor_delta
            return out

        def _f():
            self.cursor = g0
            self._cursor_delta = 0
            if len(self.sp.branch_counts) >= 2:
                self._active_branches = (false_branch,)
            out = false_fn()
            df_box[0] = self._cursor_delta
            return out

        self._cond_depth += 1
        try:
            out = lax.cond(pred, _t, _f)
        finally:
            self._cond_depth -= 1
            self._active_branches = prev_active
            self._cursor_delta = prev_delta
        dt, df = dt_box[0], df_box[0]
        step = jnp.int32(dt) if dt == df else jnp.where(pred, dt, df)
        self.cursor = g0 + step
        if dt == df:
            self._cursor_delta += dt
        else:
            assert self._cond_depth == 0, (
                f"Nested cond_branches() requires equal tile counts across "
                f"branches, got {dt} vs {df}"
            )
        return out

    @contextlib.contextmanager
    def segment(self, *, lazy: bool = False):
        """Scope a block's dense weight stream consumption and optional lazy refill."""
        assert not self._in_segment, (
            "Nested stream.segment() is not allowed; call segment() only "
            "at the top level of Block.emit()"
        )
        g_start = self.cursor
        seg_offsets = (
            {b_idx: self._branch_offsets[b_idx] for b_idx in self._active_branches}
            if self._branch_offsets is not None
            else None
        )
        prev_lazy = self._lazy
        prev_tiles = self._seg_tiles
        self._in_segment = True
        self._lazy = lazy
        self._seg_tiles = 0
        try:
            yield g_start
        finally:
            n_tiles = self._seg_tiles
            self._in_segment = False
            self._lazy = prev_lazy
            self._seg_tiles = prev_tiles
            if lazy and n_tiles > 0:
                with self.sc("refill"):
                    if (
                        self.sp.rel_fetch
                        and self._cur_layer is not None
                        and seg_offsets is not None
                    ):
                        sp = self.sp
                        blk = (
                            self._cur_layer
                            if sp.period == 1
                            else self._cur_layer // sp.period
                        )
                        if len(self._active_branches) == 1:
                            b_idx = self._active_branches[0]
                            r0 = seg_offsets[b_idx]
                            l_start = sum(sp.branch_counts[:b_idx])
                            c_b = sp.branch_counts[b_idx]
                            if c_b == 1:
                                self.flush_rel(g_start, blk, l_start, r0, n_tiles)
                            else:
                                lt_tr = self._cur_layer % sp.period
                                for lt_s in range(l_start, l_start + c_b):
                                    pl.when(lt_tr == lt_s)(
                                        lambda lt_s=lt_s, r0=r0: self.flush_rel(
                                            g_start, blk, lt_s, r0, n_tiles
                                        )
                                    )
                        else:
                            lt_tr = self._cur_layer % sp.period
                            for b_idx in self._active_branches:
                                r0 = seg_offsets[b_idx]
                                l_start = sum(sp.branch_counts[:b_idx])
                                c_b = sp.branch_counts[b_idx]
                                for lt_s in range(l_start, l_start + c_b):
                                    pl.when(lt_tr == lt_s)(
                                        lambda lt_s=lt_s, r0=r0: self.flush_rel(
                                            g_start, blk, lt_s, r0, n_tiles
                                        )
                                    )
                    else:
                        self.flush(g_start, n_tiles)

    def mm(self, x, w, dims=NN):
        if self.interpret and x.dtype != w.dtype:
            w = w.astype(x.dtype)
        return lax.dot_general(x, w, dims, preferred_element_type=F32)

    def dview(self, kind: str, bank):
        sp = self.sp
        if sp.typed:
            return self.scr["dring_" + kind].at[bank]
        shp, dt = sp.vshape[kind], jnp.dtype(sp.vdtype[kind])
        return pool_alias.view(
            self.scr["dpool"].at[
                pl.ds(bank * sp.BROWS, pool_alias.storage_size(shp, dt) // 128)
            ],
            shp,
            dt,
        )

    @contextlib.contextmanager
    def expert_stream(self):
        """Mark the expert ring (`epool`) as owned by dynamic expert streaming.

        Tracked at trace time: `Block.emit` bodies are traced sequentially, so a
        `borrow_epool()` traced inside this window would alias in-flight expert
        DMAs and is rejected.
        """
        assert not self._ering_busy, "expert_stream() is not re-entrant"
        self._ering_busy = True
        try:
            yield
        finally:
            self._ering_busy = False

    def borrow_epool(self):
        """Return `scr['epool']` for non-overlapping block use (e.g. GQA `KVE`)."""
        assert not self._ering_busy, (
            "borrow_epool() cannot be called while dynamic expert streaming is active"
        )
        assert "epool" in self.scr, "borrow_epool() requires 'epool' in scratch"
        return self.scr["epool"]

    def eview(self, kind: str, bank):
        sp = self.sp
        if sp.typed:
            return self.scr["ering_" + kind].at[bank]
        shp, dt = sp.vshape[kind], jnp.dtype(sp.vdtype[kind])
        return pool_alias.view(
            self.scr["epool"].at[
                pl.ds(bank * sp.EBROWS, pool_alias.storage_size(shp, dt) // 128)
            ],
            shp,
            dt,
        )

    def dcopy(self, kind: str, src, bank):
        return pltpu.make_async_copy(
            src, self.dview(kind, bank), self.scr["dsem"].at[bank]
        )

    def ecopy(self, kind: str, src, bank):
        return pltpu.make_async_copy(
            src, self.eview(kind, bank), self.scr["esem"].at[bank]
        )

    def _seg_issue(self, segs, r, ords, bank):
        for kind, lo, hi, ordi in segs:

            @pl.when(jnp.logical_and(r >= lo, r < hi))
            def _(kind=kind, lo=lo, ordi=ordi):
                self.dcopy(kind, self.W[kind].at[ords[ordi], r - lo], bank).start()

    def fetch(self, g):
        """Issue dense stream tile `g` into bank `g % NBD`."""
        sp = self.sp
        if sp.T_TOTAL == 0:
            return

        @pl.when(jnp.logical_and(g < sp.T_TOTAL, "dense" not in self.skip))
        def _():
            bank = g % sp.NBD
            if len(sp.branch_counts) == 1:
                t0 = sp.branch_tiles[0]
                layer = g // t0
                r = g - layer * t0
                self._seg_issue(sp.branch_segs[0], r, (layer, layer), bank)
                return

            cyc = g // sp.t_cycle
            sub = g - cyc * sp.t_cycle
            tile_lo = 0
            layer_lo = 0
            for c_b, t_b, segs_b in zip(
                sp.branch_counts, sp.branch_tiles, sp.branch_segs
            ):
                tile_hi = tile_lo + c_b * t_b

                @pl.when(jnp.logical_and(sub >= tile_lo, sub < tile_hi))
                def _(
                    c_b=c_b,
                    t_b=t_b,
                    segs_b=segs_b,
                    tile_lo=tile_lo,
                    layer_lo=layer_lo,
                ):
                    sub_b = sub - tile_lo
                    lt_b = jnp.int32(0) if c_b == 1 else (sub_b // t_b)
                    r_b = sub_b if c_b == 1 else (sub_b - lt_b * t_b)
                    layer = cyc * sp.period + layer_lo + lt_b
                    ord_b = cyc * c_b + lt_b
                    self._seg_issue(segs_b, r_b, (ord_b, layer), bank)

                tile_lo = tile_hi
                layer_lo += c_b

    def fetch_rel(self, g, blk, lt, r: int, dt=None, dt_vals=(0,), lts=None):
        """Issue dense stream tile `g` using trace-time `(layer, kind, row)` decode."""
        sp = self.sp
        if sp.T_TOTAL == 0:
            return
        if not isinstance(lt, int):
            for lt_s in range(sp.period) if lts is None else lts:
                pl.when(lt == lt_s)(
                    lambda lt_s=lt_s: self.fetch_rel(g, blk, lt_s, r, dt, dt_vals)
                )
            return
        if dt is None:
            dt_vals = (0,)
        groups: dict[tuple[int, int, int, int, str, int, int], list[int]] = {}
        for v in dt_vals:
            groups.setdefault(sp.rel_decode(lt, r + v), []).append(v)
        for (b_off, lt_, b_idx, lt_in_b, kind, ordi, lo), vs in groups.items():
            b = blk + b_off
            c_b = sp.branch_counts[b_idx]
            ord_b = b if c_b == 1 else b * c_b + lt_in_b
            ord_l = b if sp.period == 1 else b * sp.period + lt_
            ords = (ord_b, ord_l)
            row = r - lo if dt is None else dt + (r - lo)
            cond = jnp.logical_and(b < sp.L // sp.period, "dense" not in self.skip)
            if len(groups) > 1:
                hit = dt == vs[0]
                for v in vs[1:]:
                    hit = jnp.logical_or(hit, dt == v)
                cond = jnp.logical_and(cond, hit)

            @pl.when(cond)
            def _(kind=kind, ords=ords, ordi=ordi, row=row):
                self.dcopy(
                    kind, self.W[kind].at[ords[ordi], row], lax.rem(g, sp.NBD)
                ).start()

    def flush_rel(self, g0, blk, lt, r0: int, n: int):
        """Refill `n` lazy tiles starting at `g0` (`r0` in layer type `lt`)."""
        NBD = self.sp.NBD
        for t in range(n):
            self.fetch_rel(g0 + t + NBD, blk, lt, r0 + t + NBD)

    def _give_tile(self, g0, t, u: int, t_vals, dmm_ctx):
        """Refill tile `g0 + t + u` via `fetch_rel` when `dmm_ctx` is known."""
        NBD = self.sp.NBD
        if dmm_ctx is None:
            self.dgive(g0 + t + u)
            return
        blk, lt, r0, lts = dmm_ctx
        if isinstance(r0, dict):
            for lt_s, r0_s in r0.items():
                pl.when(lt == lt_s)(
                    lambda lt_s=lt_s, r0_s=r0_s: self._give_tile(
                        g0, t, u, t_vals, (blk, lt_s, r0_s, (lt_s,))
                    )
                )
            return
        if isinstance(t, int):
            self.fetch_rel(g0 + t + u + NBD, blk, lt, r0 + t + u + NBD, lts=lts)
        else:
            self.fetch_rel(
                g0 + t + u + NBD,
                blk,
                lt,
                r0 + u + NBD,
                dt=t,
                dt_vals=t_vals,
                lts=lts,
            )

    def dtake(self, kind: str, g):
        bank = g % self.sp.NBD
        if "dense" not in self.skip:
            with self.sc("dwait", 2):
                self.dcopy(kind, self.W[kind].at[0, 0], bank).wait()
        return self.dview(kind, bank)

    def dgive(self, g):
        self.fetch(g + self.sp.NBD)

    def flush(self, g0, n: int):
        """Refill the slots of tiles `g0 .. g0+n-1`, consumed lazily (no refill yet)."""
        NBD = self.sp.NBD
        for t in range(n):
            self.fetch(g0 + t + NBD)

    def dmatmul_pair(
        self,
        kind: str,
        g0,
        xsrc,
        N: int,
        *,
        lazy: bool = False,
        dmm_ctx=None,
    ):
        """Paired/unrolled dense matmul: `acc[:, :N] = xsrc[:, :K] @ W(kind)`."""
        sp = self.sp
        n_t, t_k = sp.nt[kind], sp.tk[kind]
        acc = self.scr["acc"]
        DMU = min(sp.dmm_unroll, sp.NBD)
        DCW = 512
        U = n_t if n_t <= DMU else (2 if n_t % 2 == 0 and N > DCW else 1)
        chunks = [(c0, min(DCW, N - c0)) for c0 in range(0, N, DCW)]
        n_s = n_t // U

        def step(t, first):
            ws, xs = [], []
            for u in range(U):
                ws.append(self.dtake(kind, g0 + t + u))
                k0 = (t + u) * t_k
                if not isinstance(k0, int):
                    k0 = pl.multiple_of(k0, t_k)
                xs.append(xsrc[:, pl.ds(k0, t_k)])
            for c0, cw in chunks:
                p = self.mm(xs[0], ws[0][:, c0 : c0 + cw])
                for u in range(1, U):
                    p = p + self.mm(xs[u], ws[u][:, c0 : c0 + cw])
                if first:
                    acc[:, c0 : c0 + cw] = p
                else:
                    acc[:, c0 : c0 + cw] += p
            if not lazy:
                for u in range(U):
                    self._give_tile(
                        g0, t, u, tuple(i * U for i in range(1, n_s)), dmm_ctx
                    )

        step(0, True)
        if n_s == 2:
            step(U, False)
        elif n_s > 2:

            def body_pair(i, carry):
                step(i * U, False)
                return carry

            lax.fori_loop(1, n_s, body_pair, 0)
        return acc[:, :N]

    def dmatmul(
        self,
        kind: str,
        xsrc,
        N: int,
        *,
        g0=None,
        lazy: bool | None = None,
    ):
        """`acc[:, :N] = xsrc[:, :K] @ W(kind)`, advancing `self.cursor`."""
        sp = self.sp
        n_t, t_k = sp.nt[kind], sp.tk[kind]
        dmm_ctx = None
        if g0 is None:
            if self._branch_offsets is not None:
                if self.sp.rel_fetch and self._cur_layer is not None:
                    blk = (
                        self._cur_layer
                        if sp.period == 1
                        else self._cur_layer // sp.period
                    )
                    if len(self._active_branches) == 1:
                        b_idx = self._active_branches[0]
                        r0 = self._branch_offsets[b_idx]
                        l_start = sum(sp.branch_counts[:b_idx])
                        c_b = sp.branch_counts[b_idx]
                        lt = l_start if c_b == 1 else (self._cur_layer % sp.period)
                        lts = tuple(range(l_start, l_start + c_b))
                        dmm_ctx = (blk, lt, r0, lts)
                    else:
                        r0_map = {
                            lt_s: self._branch_offsets[b]
                            for b in self._active_branches
                            for lt_s in range(
                                sum(sp.branch_counts[:b]),
                                sum(sp.branch_counts[: b + 1]),
                            )
                        }
                        r0_vals = set(r0_map.values())
                        lt = self._cur_layer % sp.period
                        lts = tuple(r0_map.keys())
                        r0_arg: Any = (
                            next(iter(r0_vals)) if len(r0_vals) == 1 else r0_map
                        )
                        dmm_ctx = (blk, lt, r0_arg, lts)
                for b_idx in self._active_branches:
                    cur_off = self._branch_offsets[b_idx]
                    matched = [
                        (s_kind, s_lo, s_hi)
                        for s_kind, s_lo, s_hi, _ in sp.branch_segs[b_idx]
                        if s_lo == cur_off
                    ]
                    assert (
                        len(matched) == 1
                        and matched[0][0] == kind
                        and matched[0][2] == cur_off + n_t
                    ), (
                        f"Stream consumption order mismatch on branch {b_idx}: "
                        f"dmatmul({kind!r}) called at layer tile offset {cur_off}, "
                        f"expected {matched} in schedule {sp.branch_segs[b_idx]}"
                    )
                    self._branch_offsets[b_idx] = cur_off + n_t
            g0 = self.cursor
            self.cursor = g0 + n_t
            self._seg_tiles += n_t
            self._cursor_delta += n_t
        use_lazy = self._lazy if lazy is None else lazy
        DMU = min(sp.dmm_unroll, sp.NBD)
        if n_t <= DMU or sp.dense_acc_pair:
            return self.dmatmul_pair(kind, g0, xsrc, N, lazy=use_lazy, dmm_ctx=dmm_ctx)
        acc = self.scr["acc"]
        acc[:, :N] = jnp.zeros((sp.MB, N), F32)

        def body(t, carry):
            g = g0 + t
            w = self.dtake(kind, g)
            k0 = pl.multiple_of(t * t_k, t_k)
            acc[:, :N] += self.mm(xsrc[:, pl.ds(k0, t_k)], w[...])
            if not use_lazy:
                self._give_tile(g0, t, 0, tuple(range(n_t)), dmm_ctx)
            return carry

        lax.fori_loop(0, n_t, body, 0)
        return acc[:, :N]
