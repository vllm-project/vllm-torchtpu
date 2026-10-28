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
"""Qwen3.8 (`Qwen4ExpForCausalLM`) segment-megakernel driver.

Uniform MTP verify steps (every request verifies `T = K + 1 = 4` tokens, at
most 16 requests) run `Qwen4ExpSegmentModel`, a second compiled program in
which the layers between two QSA attention cores are one megakernel call each
(`megakernel.models.qwen38.segment_spec`). Every other step runs the stock
model. Per segment call, in one SPMD op: gather the segment's three GDN
layers' states from the unified pool, run the kernel, scatter the per-token
GDN checkpoints back (pool donated, updated in place).

Enable with `VLLM_TPU_MEGAKERNEL=1`; comma-separated options `buckets=4:8:16:32:64`,
`head=0` (keep the layers before the first QSA layer stock), `check=N` (shadow
the first N eligible steps against the stock model).
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen38

from .driver import (
    MegakernelDriver,
    _build_tp_rank_mesh,
    _resolve_tp_size,
    register_megakernel_driver,
)
from .exported_bridge import make_typed_step_wrapper, sharded_jax_callable
from .qwen38_mtp_tail import MtpTailKind
from .qwen38_segment_call import GDN_META, T_VERIFY, build_segment_call, segment_tiles
from .qwen38_weights import head_groups, pack_segment_operands, segment_layers

logger = init_logger(__name__)

# With the C1 tiles the kernel beats stock through 64 rows (mk/kernel_work/
# mk-row-slope/report.md vs mk/stock-segment-vs-rows.md); 64 is the margin case.
# 24 and 48 are serving token buckets too (TPU_TOKEN_BUCKET_EXTRA); without them
# 24-/48-row verify steps (6 or 12 requests) run stock.
# 96 rows (24 requests, the default seat cap; a serving token bucket) runs as row
# chunks of at most KERNEL_ROWS. 128+ is left out: the segment program shares the
# stock hyper-connection code, which branches at its 128-row prefill tile.
DEFAULT_BUCKETS = (4, 8, 16, 24, 32, 48, 64, 96)
# The segment kernel's row limit (Qwen38InnerGeometry.check: B <= M <= 64).
KERNEL_ROWS = 64


def _parse_buckets(raw: str) -> tuple[int, ...]:
    for part in raw.split(","):
        if part.startswith("buckets="):
            return tuple(sorted(int(b) for b in part[len("buckets=") :].split(":")))
    return DEFAULT_BUCKETS


def _parse_flag(raw: str, key: str, default: bool) -> bool:
    for part in raw.split(","):
        if part.startswith(key + "="):
            return part[len(key) + 1 :].strip().lower() in ("1", "true", "on", "yes")
    return default


def _parse_check(raw: str) -> int:
    """`check=N`: shadow the first N eligible steps against the stock model."""
    for part in raw.split(","):
        if part.startswith("check="):
            return int(part[len("check=") :])
    return 0


def _parse_int(raw: str, key: str) -> int | None:
    for part in raw.split(","):
        if part.startswith(key + "="):
            return int(part[len(key) + 1 :])
    return None


def _real_traffic(kw: dict) -> bool:
    """A served step rather than a warmup dummy: some request has more than 64 tokens of context."""
    ctx = kw.get("context_lens")
    if not isinstance(ctx, torch.Tensor) or ctx.numel() == 0:
        return True
    return int(ctx.max()) > 64


def _diff_stats(a: torch.Tensor, b: torch.Tensor, n: int, hc: int) -> dict:
    """`a` vs reference `b` over the first `n` rows: rel, per-stream rel/cos/norm ratio, per-row rel."""
    a, b = a.float()[:n], b.float()[:n]

    def rel(x, y):
        return float((x - y).abs().mean() / y.abs().mean().clamp_min(1e-12))

    out = {"rel": round(rel(a, b), 5), "nan_a": int(torch.isnan(a).any(-1).sum()), "nan_b": int(torch.isnan(b).any(-1).sum())}
    if a.shape[-1] % hc == 0 and hc > 1:
        a3, b3 = a.unflatten(-1, (hc, -1)), b.unflatten(-1, (hc, -1))
        out["stream_rel"] = [round(rel(a3[:, k], b3[:, k]), 4) for k in range(hc)]
        out["stream_cos"] = [round(float(torch.nn.functional.cosine_similarity(a3[:, k], b3[:, k], dim=-1).mean()), 4) for k in range(hc)]
        out["stream_ratio"] = [round(float((a3[:, k].norm(dim=-1) / b3[:, k].norm(dim=-1).clamp_min(1e-12)).mean()), 4) for k in range(hc)]
    out["row_rel"] = [round(float((a[r] - b[r]).abs().mean() / b[r].abs().mean().clamp_min(1e-12)), 3) for r in range(min(n, 8))]
    return out


def _zeros_like(t: torch.Tensor) -> torch.Tensor:
    """`zeros_like` that also works for FP8 on torch_tpu (fills a same-size int view)."""
    if t.dtype.itemsize == 1 and t.dtype not in (torch.uint8, torch.int8, torch.bool):
        return torch.zeros(t.shape, dtype=torch.uint8, device=t.device).view(t.dtype)
    return torch.zeros_like(t)


def _operand_names(spec: Any, ex: Any, L: int) -> list[str]:
    """Names of the flattened operands `pack_segment_operands` returns."""
    from vllm_torchtpu.megakernel import planner

    names: list[str] = []
    for n in planner.plan(spec, ex).w_names:
        names += [f"{n}[{j}]" for j in range(L)] if n in ("e13", "e2") else [n]
    return names


def _cpu_proxy(m: Any, memo: dict | None = None) -> Any:
    """A CPU copy of module `m` with the attributes `pack_segment_operands` reads."""
    import types

    from vllm_torchtpu.layers.adapter.quantization.unquantized import WEIGHT_FLIPPED_ATTR

    memo = {} if memo is None else memo
    if id(m) in memo:
        return memo[id(m)]
    ns = memo[id(m)] = types.SimpleNamespace()
    for k, v in vars(m).items():
        if k.startswith("_") and k != WEIGHT_FLIPPED_ATTR:
            continue
        if v is None or isinstance(v, (bool, int, float, str)):
            setattr(ns, k, v)
        elif isinstance(v, nn.Module) and k == "routed_experts":
            setattr(ns, k, _cpu_proxy(v, memo))
    for k, v in m._parameters.items():
        setattr(ns, k, None if v is None else v.detach().cpu())
    for k, v in m._buffers.items():
        setattr(ns, k, None if v is None else v.detach().cpu())
    for k, v in m._modules.items():
        setattr(ns, k, None if v is None else _cpu_proxy(v, memo))
    return ns


# `VLLM_TPU_MEGAKERNEL=...,nodonate=1`: the GDN pool is not donated to the segment op; it
# writes a fresh pool that `_run` copies back into the live cache (bisecting a served hang).
NO_DONATE = False
# Largest whole-step bucket: its VMEM plan fits up to 64 rows (`wholestep_tiles`).
WHOLE_MAX_ROWS = 64
# Model steps after warmup (verify steps; any step once disabled) by which a `whole=1`
# server must have made a whole-step call.
WHOLE_SELF_CHECK_STEPS = 200
STEP_MIX_EVERY = 5000  # steps between periodic step-mix log lines
STEP_MIX_INTERVAL_S = float(os.environ.get("VLLM_TPU_MK_STEP_MIX_INTERVAL_S", "10") or 0)

_UNSET = object()


class _StackedContexts:
    """Enter the given context factories in order, exit in reverse (an ExitStack
    with nothing to resolve per use)."""

    __slots__ = ("_factories", "_entered")

    def __init__(self, factories: tuple) -> None:
        self._factories = factories
        self._entered: list = []

    def __enter__(self) -> None:
        try:
            for make in self._factories:
                cm = make()
                cm.__enter__()
                self._entered.append(cm)
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc: Any) -> bool:
        suppressed = False
        while self._entered:
            cm = self._entered.pop()
            if cm.__exit__(*(exc if not suppressed else (None, None, None))):
                suppressed = True
        return suppressed


def _vllm_env_override(**values: Any) -> Any:
    """Override `vllm.envs` attributes inside the context.

    Workers cache `vllm.envs` lookups after init (`enable_envs_cache`), so the
    process environment cannot be changed then; a module attribute takes
    precedence over the module's (cached) `__getattr__`.
    """
    import contextlib

    import vllm.envs as venvs

    @contextlib.contextmanager
    def ctx():
        saved = {k: venvs.__dict__.get(k, _UNSET) for k in values}
        for k, v in values.items():
            setattr(venvs, k, v)
        try:
            yield
        finally:
            for k, old in saved.items():
                if old is _UNSET:
                    delattr(venvs, k)
                else:
                    setattr(venvs, k, old)

    return ctx()


def _pallas_tp_mesh(runner: Any) -> Any:
    """The 8-device `tp` mesh every Pallas kernel in the served program uses.

    torch_tpu binds partition p to TP rank p and requires all Pallas kernels in
    one compiled program to share a device assignment; the runner's mesh can
    list the devices in ICI ring order instead (e.g. [0, 1, 4, 5, 6, 7, 2, 3]),
    which put the megakernel's remote DMAs on the wrong peers. Fall back to the
    runner's mesh only without torch_tpu (CPU tests).
    """
    try:
        import numpy as np
        from torch_tpu._internal.pallas import get_pallas_mesh

        mesh = get_pallas_mesh(axis_names=("tp",))
        if int(mesh.devices.size) == 8:
            runner_mesh = getattr(runner, "mesh", None)
            logger.info(
                "Qwen3.8 megakernel mesh %s (runner mesh %s)",
                [getattr(d, "id", d) for d in mesh.devices.flat],
                None if runner_mesh is None else [getattr(d, "id", d) for d in np.asarray(runner_mesh.devices).flat],
            )
            return mesh
    except Exception as exc:
        logger.debug("get_pallas_mesh unavailable (%s); using the runner mesh", exc)
    return _build_tp_rank_mesh(8, getattr(runner, "mesh", None))


def pool_layout_of(pool: torch.Tensor) -> Any:
    """`Qwen38GdnPoolLayout` of a torch pool tensor (shape and dtype name only)."""
    from .qwen38_gdn_state import Qwen38GdnPoolLayout

    return Qwen38GdnPoolLayout(
        pool_shape=tuple(int(d) for d in pool.shape),
        pool_dtype=str(pool.dtype).removeprefix("torch."),
    )


class _SegmentKind:
    """One torch op for a segment kind (`L` layers); buckets resolve on the concrete rows.

    The torch side passes the runner-padded `hidden` / `attention` and the raw
    GDN metadata; padding to the kernel's `M` rows, the block-id arithmetic,
    the pool gather / scatter and the kernel run in one SPMD JAX call per rows
    count, so the compiled torch program never branches on the token count.
    """

    def __init__(
        self, mesh: Any, g: Any, L: int, layout: Any, write: bool, head: bool = False,
        suffix: str = "",
    ) -> None:
        self.mesh, self.g, self.L, self.layout, self.write = mesh, g, L, layout, write
        self.head = head
        self.has_gdn = head or L > 1
        self._calls: dict[int, Any] = {}
        self.tag = f"{'head' if head else 'seg'}_l{L}" + ("" if write else "_shadow") + suffix
        name = f"qwen38_mk::{self.tag}"
        has_gdn = self.has_gdn

        @torch.library.custom_op(name, mutates_args=())
        def op(
            hidden: torch.Tensor,
            attn: torch.Tensor,
            weights: list[torch.Tensor],
            gdn_io: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            B = int(hidden.shape[0])
            if B > KERNEL_ROWS:
                return self._chunked(hidden, attn, weights, gdn_io, B)
            call = self.callable_for(B)
            if head:
                return list(call(hidden, *weights, *gdn_io))
            return list(call(hidden, attn, *weights, *gdn_io))

        @op.register_fake
        def _(hidden, attn, weights, gdn_io):
            out = [torch.empty_like(hidden)]
            if has_gdn:
                out.append(torch.empty_like(gdn_io[0]))
            return out

        self.op = op

    def _chunked(self, hidden: Any, attn: Any, weights: list, gdn_io: list, B: int) -> list:
        """`B > KERNEL_ROWS` rows as consecutive kernel calls over request-aligned
        row chunks. Rows are independent and a request's T_VERIFY rows never
        straddle a chunk, so this is exact; the GDN pool is threaded through the
        calls (each donates it to the next) and the per-request metadata sliced.
        """
        outs: list = []
        pool = gdn_io[0] if self.has_gdn else None
        meta = list(gdn_io[1:])
        for r0 in range(0, B, KERNEL_ROWS):
            b = min(KERNEL_ROWS, B - r0)
            s0, ns = r0 // T_VERIFY, b // T_VERIFY
            io: list = []
            if self.has_gdn:
                io = [pool]
                for i in range(0, len(meta), GDN_META):
                    st, off, ck, qsl, seq = meta[i : i + GDN_META]
                    io += [st[s0 : s0 + ns], off, ck[s0 : s0 + ns], qsl[s0 : s0 + ns + 1], seq[s0 : s0 + ns]]
            call = self.callable_for(b)
            h = hidden[r0 : r0 + b]
            res = call(h, *weights, *io) if self.head else call(h, attn[r0 : r0 + b], *weights, *io)
            outs.append(res[0])
            if self.has_gdn:
                pool = res[1]
        return [torch.cat(outs, dim=0)] + ([pool] if self.has_gdn else [])

    def callable_for(self, B: int) -> Any:
        if B not in self._calls:
            full, in_specs, outs, donate = build_segment_call(
                self.mesh, self.g, self.L, B, self.layout, self.write, head=self.head,
                flat_smalls=bool(envs.VLLM_TPU_MK_OWN_OPERANDS),
            )
            # JaxCallable needs explicit, jax.Array-annotated positional parameters.
            typed = make_typed_step_wrapper(
                f"qwen38_mk_{self.tag}_b{B}", full, len(in_specs), len(outs)
            )
            self._calls[B] = sharded_jax_callable(
                f"qwen38_mk::{self.tag}_b{B}",
                typed,
                mesh=self.mesh,
                input_partition_specs=in_specs,
                output_partition_specs=outs,
                donate_argnums=() if NO_DONATE else donate,
            )
        return self._calls[B]


class Qwen38Segments(nn.Module):
    """Per-segment kernel call; owns the packed operands as buffers."""

    def __init__(
        self,
        model: Any,
        g: Any,
        mesh: Any,
        buckets: tuple[int, ...],
        *,
        shared: Qwen38Segments | None = None,
        head: bool = True,
    ) -> None:
        super().__init__()
        self.write_ckpts = shared is None
        self.g = g
        self.mesh = mesh
        self.buckets = buckets
        # A plain list, not a submodule: the model's layers stay owned by the model.
        self.layers = list(model.model.layers)
        self.n_seg = g.L // g.att_every
        self.head_groups = head_groups(model, g) if head else []
        self.kinds: dict[tuple[int, bool], _SegmentKind] = {}
        if shared is not None:
            # A shadow view: same operands, GDN checkpoints not written.
            self.n_w, self.n_hw = shared.n_w, shared.n_hw
            for name, buf in shared.named_buffers(recurse=False):
                self.register_buffer(name, buf, persistent=False)
            # Plain reference (not a submodule), like `layers`.
            object.__setattr__(self, "_shared", shared)
            self.pool = self.layout = None
            return
        object.__setattr__(self, "_shared", None)
        R = mesh.size
        self.n_w: list[int] = []
        for s in range(self.n_seg):
            L = len(segment_layers(g, s))
            ex = ExecSpec(R=R, B=buckets[0], T=T_VERIFY, ctx=128, tiles=segment_tiles(buckets[0]))
            ops = pack_segment_operands(
                model, g, s, qwen38.segment_spec(g, R, L=L), ex,
                boundary=bool(envs.VLLM_TPU_MK_OWN_OPERANDS),
            )
            ops = self._own(ops, f"segment {s}")
            self.n_w.append(len(ops))
            for i, t in enumerate(ops):
                self.register_buffer(f"w{s}_{i}", t, persistent=False)
        self.n_hw: list[int] = []
        for h, group in enumerate(self.head_groups):
            L = len(group)
            ex = ExecSpec(R=R, B=buckets[0], T=T_VERIFY, ctx=128, tiles=segment_tiles(buckets[0]))
            spec = qwen38.segment_spec(g, R, L=L, head=True)
            ops = pack_segment_operands(
                model, g, -1, spec, ex, layer_ids=group, head=True,
                boundary=bool(envs.VLLM_TPU_MK_OWN_OPERANDS),
            )
            ops = self._own(ops, f"head group {h}")
            self.n_hw.append(len(ops))
            for i, t in enumerate(ops):
                self.register_buffer(f"h{h}_{i}", t, persistent=False)
        self.pool = None
        self.layout = None
        logger.info(
            "Qwen3.8 segment megakernel: packed operands %s (buckets %s)",
            hbm_summary(self.buffers(), model), buckets,
        )

    @staticmethod
    def _own(ops: list[torch.Tensor], what: str) -> list[torch.Tensor]:
        """Debug: with VLLM_TPU_MK_OPERAND_DUMP, log every operand's layout."""
        if os.environ.get("VLLM_TPU_MK_OPERAND_DUMP"):
            for i, t in enumerate(ops):
                logger.info(
                    "qwen38 megakernel operand %s[%d]: %s %s stride=%s offset=%d view=%s",
                    what, i, tuple(t.shape), t.dtype, t.stride(), t.storage_offset(),
                    t._base is not None,
                )
        return ops

    def bind_pool(self) -> None:
        from .qwen38_gdn_state import Qwen38GdnPoolLayout

        if self._shared is not None:
            self.pool, self.layout = self._shared.pool, self._shared.layout
        else:
            pools = [
                layer.linear_attn.kv_cache[0]
                for layer in self.layers
                if layer.layer_type == "linear_attention"
            ]
            base = pools[0]
            for p in pools:
                if p.data_ptr() != base.data_ptr():
                    raise RuntimeError("Qwen3.8 megakernel needs one unified GDN pool tensor")
            if getattr(self.layers[0].linear_attn, "_bm_kernel", False):
                raise RuntimeError("Qwen3.8 megakernel does not support block-major pools yet")
            if envs.VLLM_TPU_MK_OWN_OPERANDS and base._base is not None:
                # The unified pool is a view of the runner's raw KV buffer. A
                # detached alias keeps the same storage (in-place writes still
                # land in the cache) without view metadata.
                logger.info(
                    "qwen38 megakernel GDN pool: view %s (offset %d, contiguous %s) aliased",
                    tuple(base.shape), base.storage_offset(), base.is_contiguous(),
                )
                base = base.detach()
            self.pool = base
            self.layout = pool_layout_of(base)
        for L in sorted({len(segment_layers(self.g, s)) for s in range(self.n_seg)}):
            self.kinds[(L, False)] = _SegmentKind(
                self.mesh, self.g, L, self.layout, self.write_ckpts
            )
        for L in sorted({len(gr) for gr in self.head_groups}):
            self.kinds[(L, True)] = _SegmentKind(
                self.mesh, self.g, L, self.layout, self.write_ckpts, head=True
            )

    def _gdn_meta(self, layer_ids: list[int]) -> list[torch.Tensor]:
        from vllm.forward_context import get_forward_context

        md_all = get_forward_context().attn_metadata
        meta: list[torch.Tensor] = []
        for i in layer_ids:
            md = md_all[self.layers[i].linear_attn.prefix]
            meta += [
                md.mamba_state_indices.to(torch.int32),
                md.mamba_slot_read_offsets.to(torch.int32),
                md.mamba_ckpt_indices.to(torch.int32),
                md.query_start_loc.to(torch.int32),
                md.seq_lens.to(torch.int32),
            ]
        return meta

    def _run(self, kind, hidden, lead, w, meta):
        """One `kind.op` call; the GDN pool is written back after it.

        No branch on the token count here: this runs inside a vLLM-compiled
        program, and a traced graph that branches on it is rejected
        (ShapeSpecializationError). The op resolves the bucket itself.
        """
        if kind.has_gdn:
            res, pool = kind.op(hidden, lead, w, [self.pool, *meta])
            self.pool.copy_(pool)
            return res
        (res,) = kind.op(hidden, lead, w, [])
        return res

    def forward(self, s: int, hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        ids = segment_layers(self.g, s)
        kind = self.kinds[(len(ids), False)]
        w = [getattr(self, f"w{s}_{i}") for i in range(self.n_w[s])]
        meta = self._gdn_meta(ids[1:]) if kind.has_gdn else []
        return self._run(kind, hidden, attention, w, meta)

    def head(self, h: int, hidden: torch.Tensor) -> torch.Tensor:
        """Head group `h` (layers before the first QSA layer) in one kernel call."""
        ids = self.head_groups[h]
        kind = self.kinds[(len(ids), True)]
        w = [getattr(self, f"h{h}_{i}") for i in range(self.n_hw[h])]
        return self._run(kind, hidden, hidden, w, self._gdn_meta(ids))


class MtpTail(nn.Module):
    """The MTP draft layer after attention (o_proj, both HCs, MoE) as one tail-kernel call.

    Its own op (`MtpTailKind`): the draft reaches the tail at every rows count it
    is traced at, and counts past the buckets run in largest-bucket chunks.
    Owns the draft layer's packed operands as buffers.
    """

    def __init__(self, draft: Any, g: Any, mesh: Any, kind: MtpTailKind, buckets: tuple[int, ...]):
        super().__init__()
        object.__setattr__(self, "kind", kind)
        self.buckets = buckets
        R = mesh.size
        ex = ExecSpec(R=R, B=buckets[0], T=T_VERIFY, ctx=128, tiles=segment_tiles(buckets[0]))
        ops = pack_segment_operands(draft, g, -1, qwen38.segment_spec(g, R, L=1), ex, layer_ids=[0])
        ops = Qwen38Segments._own(ops, "MTP tail")  # MTP tail keeps the plain layout
        self.n_w = len(ops)
        for i, t in enumerate(ops):
            self.register_buffer(f"w_{i}", t, persistent=False)

    def forward(self, hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        w = [getattr(self, f"w_{i}") for i in range(self.n_w)]
        (res,) = self.kind.op(hidden, attention, w)
        return res


class Qwen38SegmentDriver:
    """Routes uniform verify steps to `Qwen4ExpSegmentModel`."""

    takes_model_kwargs = True

    def __init__(self, runner: Any, buckets: tuple[int, ...], head: bool = True) -> None:
        self.buckets = buckets
        self.g = qwen38.FULL
        self.mesh = _pallas_tp_mesh(runner)
        self.segments = Qwen38Segments(runner.model, self.g, self.mesh, buckets, head=head)
        self.program = None
        self.check_left = 0
        self.shadow = None
        self.h0check_left = 0
        self.h0kind = None
        self.segx_left = 0
        self.segx_set: list[int] = []
        self.segx_audited: set[int] = set()
        self.private_kinds: dict = {}
        self.stats: dict[str, int] = {}
        self.mtp = False
        self.cfg = None
        self.seq = 0  # try_forward calls, to compare the step sequence across ranks
        self.mk_calls = 0
        self.sync = False  # `sync=1`: wait for the first calls' outputs and log it (bisecting)
        self.warm = False  # `warm=1`: host barriers around the first (compiling) call
        self.noaot = False  # `noaot=1`: no AOT serialization / compile cache for the segment program
        # `whole=1`: layers 3..46 in one whole-step call (`Qwen4ExpWholeStepModel`) on steps
        # whose every request's context is <= `wholecap=N` tokens (others: segment program).
        self.whole = False
        self.whole_cap = 32768
        self.whole_hold = "buffer"  # `wholehold=param|list|buffer`: how the operands are held
        # `wholerows=N`: whole-step only up to N rows (its in-kernel indexer / attention run row
        # by row, ~0.25 ms per row: it loses to the segment / stock path at high concurrency).
        self.whole_max_rows = 16
        self.whole_above = "stock"  # `wholeabove=stock|seg`: steps above `wholerows`
        self.whole_program = None
        self.whole_calls = 0
        self.whole_dry = False  # `wholedry=1`: decide and count as whole=1, run stock (AL control)

    @classmethod
    def maybe_create(cls, runner: Any) -> Qwen38SegmentDriver | None:
        raw = str(envs.VLLM_TPU_MEGAKERNEL or "").strip()
        if not raw or raw.lower() in ("0", "false", "off"):
            return None
        reasons = []
        if _resolve_tp_size(runner) != 8:
            reasons.append("needs TP=8")
        spec_cfg = getattr(runner, "speculative_config", None)
        if int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0) != T_VERIFY - 1:
            reasons.append("needs MTP with 3 speculative tokens")
        t = runner.model_config.hf_text_config
        for field, want in (("hidden_size", 2560), ("num_hidden_layers", 48), ("hc_count", 4)):
            if getattr(t, field, None) != want:
                reasons.append(f"{field}={getattr(t, field, None)} != {want}")
        if getattr(getattr(runner.model, "model", None), "host_ple_index", None) is not None:
            # PLE CPU offload (PLE_CPU_OFFLOAD, auto-on at max concurrency >= 32): the MK programs need
            # the PLE table on HBM, so no driver at all - building one would still pack the segment
            # operands into HBM before KV-cache sizing and shrink the pool of an all-stock run
            # (c32 j-f34a09a4: prefix hit 0.9403 vs 0.9454 MK-unset).
            reasons.append("PLE CPU offload is on (the segment program needs the PLE table on HBM)")
        if reasons:
            logger.warning("Qwen3.8 megakernel disabled: %s", "; ".join(reasons))
            return None
        from . import qwen38_segment_call

        hier = _parse_int(raw, "hier")
        if hier is not None:
            qwen38_segment_call.DEFAULT_HIER = hier
        qwen38_segment_call.CLAMP_BLOCKS = _parse_flag(raw, "gdnclamp", False)
        qwen38_segment_call.PIPE_EP = _parse_flag(raw, "pipeep", True)
        for part in raw.split(","):
            if part.startswith("rings="):
                e, d = part[len("rings=") :].split(":")
                qwen38_segment_call.DEFAULT_RINGS = (int(e), int(d))
        drv = cls(runner, _parse_buckets(raw), head=_parse_flag(raw, "head", True))
        drv.check_left = _parse_check(raw)
        # `h0check=N`: every eligible step runs stock; the first N real ones also
        # compare the head-0 (layer 0) kernel against stock layer 0 on the same
        # input, the kernel reading a private copy of layer 0's state blocks.
        drv.h0check_left = _parse_int(raw, "h0check") or 0
        # `h0rows=96`: h0check only steps of that many rows (e.g. the chunked path).
        drv.h0rows = _parse_int(raw, "h0rows") or 0
        # `h0dump=1`: the first h0check step also writes each rank's operands, input,
        # private state snapshot, metadata, kernel output and stock stages to
        # $CDK_OUTPUT_DIR/h0dump/rank<r>.pt (offline reference replay).
        drv.h0dump = bool(_parse_int(raw, "h0dump") or 0)
        # `segxcheck=N,segx=8:2`: like h0check for the named segments (each segment
        # kernel vs its stock layers on the same input, private GDN state copies).
        drv.segx_left = _parse_int(raw, "segxcheck") or 0
        for part in raw.split(","):
            if part.startswith("segx="):
                drv.segx_set = [int(x) for x in part[5:].split(":")]
        for part in raw.split(","):
            # `segset=1:5` / `headset=0`: only these segments / head groups run
            # their kernels, the rest stock (bisection).
            if part.startswith("segset="):
                drv.segments.mk_segment_set = {int(x) for x in part[7:].split(":")}
            if part.startswith("headset="):
                drv.segments.mk_head_set = {int(x) for x in part[8:].split(":") if x != ""}
        segs = _parse_int(raw, "segs")
        if segs is not None:
            drv.segments.mk_segments = segs
        drv.mtp = _parse_flag(raw, "mtp", False)
        drv.sync = _parse_flag(raw, "sync", False)
        drv.warm = _parse_flag(raw, "warm", False)
        drv.noaot = _parse_flag(raw, "noaot", False)
        drv.whole = _parse_flag(raw, "whole", False)
        drv.whole_dry = _parse_flag(raw, "wholedry", False)
        cap = _parse_int(raw, "wholecap")
        if cap is not None:
            drv.whole_cap = cap
        if _parse_flag(raw, "wholeshard", False):
            from . import qwen38_wholestep_call

            qwen38_wholestep_call.SHARD_INDEX = True
        if _parse_flag(raw, "wholeunion", False):
            from . import qwen38_wholestep_call

            qwen38_wholestep_call.UNION_ATTN = True
        if _parse_flag(raw, "wholephys", False):
            from . import qwen38_wholestep_call

            qwen38_wholestep_call.UNION_PHYS = True
        if _parse_flag(raw, "wholeloop", False):
            from . import qwen38_wholestep_call

            qwen38_wholestep_call.LOOP_REQUESTS = True
        every = _parse_int(raw, "wholestats")
        if every:
            from . import qwen38_wholestep_call

            qwen38_wholestep_call.UNION_STATS_EVERY = every
        rows_max = _parse_int(raw, "wholerows")
        if rows_max is not None:
            drv.whole_max_rows = rows_max
        for part in raw.split(","):
            if part.startswith("wholeabove="):
                drv.whole_above = part[len("wholeabove=") :]
        for part in raw.split(","):
            if part.startswith("wholehold="):
                drv.whole_hold = part[len("wholehold=") :]
        global NO_DONATE
        NO_DONATE = _parse_flag(raw, "nodonate", False)
        if not _parse_flag(raw, "replay", False):
            # The segment ops lower per rank (the kernel's mesh position), but the
            # traced graph is the same on every rank, so a peer would replay rank
            # 0's program on all ranks (q39 j-698bb3ee: "Replayed ... rank_0_0/
            # model_mk_segments"). `replay=1` restores the old behaviour.
            from vllm_torchtpu.compilation.tpu_compiler import mark_rank_dependent

            mark_rank_dependent("qwen38_mk")
        logger.info(
            "Qwen3.8 segment megakernel enabled, verify buckets %s, shadow checks %d, hier %s, rings %s, head %s",
            drv.buckets,
            drv.check_left,
            qwen38_segment_call.DEFAULT_HIER,
            f"{qwen38_segment_call.DEFAULT_RINGS or 'per bucket'}, gdnclamp {qwen38_segment_call.CLAMP_BLOCKS}, nodonate {NO_DONATE}",
            bool(drv.segments.head_groups),
        )
        return drv

    def _segment_config(self, runner: Any) -> Any:
        """The runner's config with `compile_sizes` cut to the megakernel buckets.

        vLLM traces a compiled module once and builds that graph for every
        compile size. The segment program shares the stock model's
        hyper-connection code, which branches at its prefill tile (128 rows),
        so a trace at a megakernel bucket cannot serve 128+. The program only
        ever runs at the buckets, so it compiles only those (the drafter
        likewise compiles under its own copy of the config).
        """
        import copy

        cfg = copy.copy(runner.vllm_config)
        cc = copy.copy(runner.vllm_config.compilation_config)
        paddings = set(int(x) for x in (cc.compile_sizes or []))
        cc.compile_sizes = sorted(b for b in self.buckets if not paddings or b in paddings)
        object.__setattr__(cfg, "compilation_config", cc)
        return cfg

    def _contexts(self, runner: Any, tag: str) -> Any:
        """The contexts the runner builds its main model under (config, mesh, tag)."""
        import contextlib

        from vllm.compilation.backends import set_model_tag
        from vllm.config import set_current_vllm_config

        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        stack = contextlib.ExitStack()
        stack.enter_context(set_model_tag(tag))
        cfg = (getattr(self, "whole_cfg", None) if tag == "model_mk_whole" else None) or (
            getattr(self, "cfg", None) or runner.vllm_config
        )
        stack.enter_context(set_vllm_model_wrapper_context(mesh=runner.mesh, vllm_config=cfg))
        stack.enter_context(set_current_vllm_config(cfg))
        if getattr(self, "noaot", False):
            # `noaot=1`: the segment program is compiled without vLLM's AOT
            # serialization and compile cache (as the Qwen3.5 megakernel step,
            # torch.compile(..., serializable=False)); the main model keeps both.
            stack.enter_context(
                _vllm_env_override(VLLM_USE_AOT_COMPILE=False, VLLM_DISABLE_COMPILE_CACHE=True)
            )
        return stack

    def _start_watchdog(self, runner: Any, period_s: float = 300.0) -> None:
        """Self-check: a segment MK that silently stops serving reads like a clean
        stock run. Every `period_s` after warmup, log an ERROR if the driver saw
        decode steps but made 0 MK calls (unless auto-disabled), and a WARNING
        once if the driver was not called at all in the period."""
        import threading
        import time

        def loop() -> None:
            while not getattr(runner, "warmup_done", False):
                time.sleep(10.0)
            last_calls, last_mk = getattr(self, "calls_total", 0), self.mk_calls
            quiet_logged = False
            while True:
                time.sleep(period_s)
                calls, mk = getattr(self, "calls_total", 0), self.mk_calls
                if calls == last_calls:
                    if not quiet_logged:
                        logger.warning(
                            "qwen38 megakernel self-check: driver not called in the last %.0f s "
                            "(idle server, or forward steps bypass the megakernel hook)", period_s,
                        )
                        quiet_logged = True
                else:
                    quiet_logged = False
                    # `wholedry=1` routes every MK-eligible step to stock on purpose.
                    if (
                        mk == last_mk
                        and getattr(self, "disabled_reason", None) is None
                        and not getattr(self, "whole_dry", False)
                    ):
                        logger.error(
                            "qwen38 megakernel self-check: 0 MK calls in %d driver steps over %.0f s; "
                            "step mix %s", calls - last_calls, period_s, dict(sorted(self.stats.items())),
                        )
                last_calls, last_mk = calls, mk

        self._watchdog = threading.Thread(target=loop, name="qwen38-mk-selfcheck", daemon=True)
        self._watchdog.start()

    def _serve_contexts(self, runner: Any) -> Any:
        """`_contexts(runner, "model_mk_segments")` for the per-step call, without
        the per-step imports, attribute lookups and ExitStack: the three context
        factories and their arguments are resolved once. Still entered and exited
        around every call, so nothing leaks into the stock path."""
        fast = getattr(self, "_serve_ctx", None)
        if fast is None or getattr(self, "noaot", False):
            return self._contexts(runner, "model_mk_segments")
        return _StackedContexts(fast)

    def bind_and_warmup(self, runner: Any) -> None:
        from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpSegmentModel

        if getattr(runner.model.model, "host_ple_index", None) is not None:
            # forward_segments needs the PLE table on HBM; with PLE CPU offload
            # (PLE_CPU_OFFLOAD, auto-on at max concurrency >= 32) the first MK
            # call would fail to trace and fail closed. Say so once, up front.
            logger.warning(
                "Qwen3.8 megakernel disabled: PLE CPU offload is on (host_ple_index=%s); "
                "the segment program needs the PLE table on HBM. Every step runs stock.",
                runner.model.model.host_ple_index,
            )
            self.disabled_reason = "stock_host_ple"
            return
        self._start_watchdog(runner)
        self.segments.bind_pool()
        self.cfg = self._segment_config(runner)
        from vllm.compilation.backends import set_model_tag
        from vllm.config import set_current_vllm_config

        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        cfg, mesh = self.cfg, runner.mesh
        self._serve_ctx = (
            lambda: set_model_tag("model_mk_segments"),
            lambda: set_vllm_model_wrapper_context(mesh=mesh, vllm_config=cfg),
            lambda: set_current_vllm_config(cfg),
        )
        with self._contexts(runner, "model_mk_segments"):
            self.program = Qwen4ExpSegmentModel(
                vllm_config=self.cfg,
                prefix="model_mk_segments",
                backbone=runner.model.model,
                segments=self.segments,
            )
        if self.whole:
            self._attach_whole(runner)
        if self.mtp:
            self._attach_mtp_tail(runner)
        if self.check_left > 0:
            shadow_seg = Qwen38Segments(
                runner.model,
                self.g,
                self.mesh,
                self.buckets,
                shared=self.segments,
                head=bool(self.segments.head_groups),
            )
            shadow_seg.bind_pool()
            with self._contexts(runner, "model_mk_segments_shadow"):
                self.shadow = Qwen4ExpSegmentModel(
                    vllm_config=self.cfg,
                    prefix="model_mk_segments_shadow",
                    backbone=runner.model.model,
                    segments=shadow_seg,
                )

    def _h0check(self, runner: Any, input_ids: Any, positions: Any, kw: dict, rows: int, n_req: int) -> Any:
        from vllm.forward_context import get_forward_context

        from .qwen38_gdn_state import Qwen38GdnPoolLayout

        segs, bb = self.segments, runner.model.model
        layer = bb.layers[0]
        hc = bb.hc_count
        NS, T = rows // T_VERIFY, T_VERIFY
        md = get_forward_context().attn_metadata[layer.linear_attn.prefix]
        st = md.mamba_state_indices.to(torch.int64)[:NS]
        off_tab = md.mamba_slot_read_offsets.to(torch.int64)
        ck = md.mamba_ckpt_indices.to(torch.int64)[:NS, :T]
        qsl = md.query_start_loc.to(torch.int32)[: NS + 1]
        seq = md.seq_lens.to(torch.int32)[:NS]
        w = [getattr(segs, f"h0_{i}") for i in range(segs.n_hw[0])]
        with torch.no_grad():
            h_in = bb.embed_tokens(input_ids).repeat(1, hc)
            ids0 = list(segs.head_groups[0])
            R = self.mesh.size
            ex = ExecSpec(R=R, B=segs.buckets[0], T=T_VERIFY, ctx=128, tiles=segment_tiles(segs.buckets[0]))
            spec = qwen38.segment_spec(self.g, R, L=len(ids0), head=True)
            names = _operand_names(spec, ex, len(ids0))
            if not getattr(self, "h0_audited", False):
                self.h0_audited = True
                self._audit_operands(runner, w, names, spec, ex, ids0, "head0", True)
            # Zero-weight splits: no MoE (routed + shared expert weights 0, so the
            # MoE adds exactly 0) and no GDN (out_proj 0, so the GDN adds exactly 0).
            no_moe = [_zeros_like(t) if n.split("[")[0] in ("e13", "e2", "sh13", "sh2") else t for n, t in zip(names, w)]
            no_gdn = [_zeros_like(t) if n == "lin_out" else t for n, t in zip(names, w)]
            def ok(tag, t):
                # Materialize `t` so an async failure names its stage.
                logger.info("qwen38 h0check stage %s ok: sum %.6g", tag, float(t.float().sum()))

            snap0 = self._snap(ids0, NS)
            ok("snap", snap0[0])
            # The kernel donates the pool operand: copy the snapshot first.
            snap_cpu = [t.detach().cpu() for t in snap0] if getattr(self, "h0dump", False) else None
            mk_out = self._mk_run(len(ids0), True, w, h_in, None, snap0)
            ok("mk", mk_out)
            ok("zeros", no_moe[names.index("e13[0]")])
            mk_nomoe = self._mk_run(len(ids0), True, no_moe, h_in, None, self._snap(ids0, NS))
            ok("mk_nomoe", mk_nomoe)
            mk_nogdn = self._mk_run(len(ids0), True, no_gdn, h_in, None, self._snap(ids0, NS))
            ok("mk_nogdn", mk_nogdn)
            # Stock layer 0, staged like Qwen4ExpDecoderLayer.forward.
            from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp

            block, residual = layer.attn_hyper_connection.mix(h_in)
            la = layer.linear_attn
            cap: dict = {}
            hooks = []
            if getattr(self, "h0dump", False):
                # Stock GDN internals for the offline replay: projections, the
                # gated norm's inputs/output and out_proj's input/output.
                cap["qkvz"] = la.in_proj_qkvz(block)[0]
                cap["ba"] = la.in_proj_ba(block)[0]

                def grab(tag):
                    def hook(mod, args, out):
                        cap[tag + "_in"] = [a for a in args if isinstance(a, torch.Tensor)]
                        cap[tag + "_out"] = out[0] if isinstance(out, tuple) else out
                    return hook

                hooks = [la.norm.register_forward_hook(grab("norm")), la.out_proj.register_forward_hook(grab("out_proj"))]
            try:
                g = sp.reduce(la(block), False)
            finally:
                for hk in hooks:
                    hk.remove()
            mixed2, residual2 = layer.attn_hyper_connection.combine_mix(g, residual, layer.mlp_hyper_connection)
            valid = kw["valid_tokens"].to(torch.bool)
            mixed2 = torch.where(valid[:, None], mixed2, 0)
            moe = layer.mlp(mixed2)
            st_out = layer.mlp_hyper_connection.combine(moe, residual2)
            # Stock counterparts of the splits (no GDN: block output 0).
            st_nomoe = residual2[0] if isinstance(residual2, tuple) else residual2
            m0, r0 = layer.attn_hyper_connection.combine_mix(torch.zeros_like(g), residual, layer.mlp_hyper_connection)
            m0 = torch.where(valid[:, None], m0, 0)
            st_nogdn = layer.mlp_hyper_connection.combine(layer.mlp(m0), r0)
            # Stock layer 0 above is this step's served layer 0 (its GDN writes are the
            # real ones); the rest of the model runs stock here and serves the step.
            args = (
                positions.to(torch.int64), input_ids.to(torch.int64), kw["query_start_loc"],
                kw["ngram_context"], kw["state_slots"], kw["context_lens"], kw["valid_tokens"],
                kw["main_block_table"], kw["compressed_block_table"],
            )
            h = st_out
            for later in bb.layers[1:]:
                h = later(h, *args)
            mixed = bb.hyper_connection_mixer.mix(h)[0]
        # Stats over the live requests' rows only: padded rows are masked to 0
        # before the stock MoE but not in the kernel.
        n = n_req * T
        if getattr(self, "h0dump", False):
            self.h0dump = False
            try:
                import os

                from vllm.distributed import get_tensor_model_parallel_rank

                cpu = lambda t: t.detach().cpu()  # noqa: E731
                r2 = residual2[0] if isinstance(residual2, tuple) else residual2
                d = {
                    "names": names, "rows": rows, "n_req": n_req,
                    "w": [cpu(t) for t in w], "h_in": cpu(h_in),
                    "snap": snap_cpu, "mk_out": cpu(mk_out), "st_out": cpu(st_out),
                    "stock_block": cpu(block), "stock_g": cpu(g), "stock_res2": cpu(r2),
                    "stock_mixed2": cpu(mixed2), "stock_moe": cpu(moe),
                    "mk_nomoe": cpu(mk_nomoe), "mk_nogdn": cpu(mk_nogdn),
                    "st_nogdn": cpu(st_nogdn), "valid": cpu(kw["valid_tokens"]),
                    "stock_gdn": {k: ([cpu(t) for t in v] if isinstance(v, list) else cpu(v)) for k, v in cap.items()},
                }
                out = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"), "h0dump")
                os.makedirs(out, exist_ok=True)
                path = os.path.join(out, f"rank{get_tensor_model_parallel_rank()}.pt")
                torch.save(d, path)
                logger.info("qwen38 h0check dump written: %s", path)
            except Exception:
                logger.exception("qwen38 h0check dump failed")
        logger.info(
            "qwen38 h0check inputs: rows=%d n_req=%d h_in_norm=%.4f st=%s off=%s ck0=%s seq=%s",
            rows, n_req, float(h_in.float()[:n].norm()), st.tolist()[:8], off_tab[st].tolist()[:8],
            ck[:2].tolist(), seq.tolist()[:8],
        )
        logger.info(
            "qwen38 h0check layer0_out MK vs stock: %s | stock residual-after-GDN(norm)=%.4f moe(norm)=%.4f (%d left)",
            _diff_stats(mk_out, st_out, n, hc), float((residual2[0] if isinstance(residual2, tuple) else residual2).float()[:n].norm()),
            float(moe.float()[:n].norm()), self.h0check_left,
        )
        logger.info("qwen38 h0check split no-MoE (GDN + attn HC) MK vs stock: %s", _diff_stats(mk_nomoe, st_nomoe, n, hc))
        logger.info("qwen38 h0check split no-GDN (MoE + HC) MK vs stock: %s", _diff_stats(mk_nogdn, st_nogdn, n, hc))
        return (mixed, h) if bb.emit_mtp_hidden else mixed

    def _audit_operands(
        self, runner: Any, w: list, names: list, spec: Any, ex: Any, ids: list, tag: str, head: bool
    ) -> None:
        """Each served operand vs the same packing done on CPU from CPU copies of
        the modules (torch_tpu op check), one log line per operand; then what the
        JAX side sees (position-weighted sums inside a shard_map) vs the torch side."""
        try:
            import types

            layers = [None] * (max(ids) + 1)
            for i in ids:
                layers[i] = _cpu_proxy(runner.model.model.layers[i])
            proxy = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
            ref = pack_segment_operands(
                proxy, self.g, -1, spec, ex, layer_ids=ids, head=head,
                boundary=bool(envs.VLLM_TPU_MK_OWN_OPERANDS),
            )
            for n, a, b in zip(names, w, ref):
                a = a.detach().cpu()
                if tuple(a.shape) != tuple(b.shape):
                    logger.info("qwen38 audit %s %s: SHAPE served %s cpu %s", tag, n, tuple(a.shape), tuple(b.shape))
                    continue
                af, bf = a.float(), b.float()
                d = (af - bf).abs()
                logger.info(
                    "qwen38 audit %s %s %s %s: maxabs %.4g rel %.4g |cpu| %.4g nz served %d cpu %d",
                    tag, n, tuple(a.shape), str(a.dtype).removeprefix("torch."), float(d.max()),
                    float(d.mean() / bf.abs().mean().clamp_min(1e-30)), float(bf.abs().mean()),
                    int((af != 0).sum()), int((bf != 0).sum()),
                )
            self._audit_jax_view(w, ref, names, tag)
        except Exception:
            logger.exception("qwen38 audit %s failed", tag)

    def _audit_jax_view(self, w: list, ref: list, names: list, tag: str) -> None:
        import jax
        import jax.numpy as jnp
        from jax.sharding import PartitionSpec as P

        def stats(*xs):
            out = []
            for x in xs:
                f = x.astype(jnp.float32).reshape(-1)
                pos = (jnp.arange(f.shape[0], dtype=jnp.int32) % 7 + 1).astype(jnp.float32)
                out.append(jnp.stack([jnp.sum(jnp.abs(f)), jnp.sum(f * pos)])[None])
            return tuple(out)

        n = len(w)
        body = jax.shard_map(stats, mesh=self.mesh, in_specs=(P("tp"),) * n, out_specs=(P("tp"),) * n, check_vma=False)
        typed = make_typed_step_wrapper(f"qwen38_audit_{tag}", body, n, n)
        call = sharded_jax_callable(
            f"qwen38_mk::audit_{tag}", typed, mesh=self.mesh,
            input_partition_specs=(P("tp"),) * n, output_partition_specs=(P("tp"),) * n,
        )
        outs = call(*w)
        for name, o, b in zip(names, outs, ref):
            f = b.float().reshape(-1)
            pos = (torch.arange(f.numel()) % 7 + 1).float()
            cpu = (float(f.abs().sum()), float((f * pos).sum()))
            o = o.detach().cpu().reshape(-1).tolist()
            logger.info(
                "qwen38 audit-jax %s %s: jax abs %.6g pos %.6g | cpu abs %.6g pos %.6g",
                tag, name, o[0], o[1], cpu[0], cpu[1],
            )

    def _snap(self, gdn_ids: list, NS: int) -> list:
        """A private copy of `gdn_ids`' GDN state blocks plus remapped metadata.

        Fixed capacity (null block + 32 requests x 4 checkpoint columns per GDN
        layer) so one compiled kind serves every bucket. The real pool is only
        read; take it before the stock layers write this step's checkpoints.
        """
        from vllm.forward_context import get_forward_context

        if not gdn_ids:
            return []
        segs = self.segments
        T, CAP = T_VERIFY, 32
        md_all = get_forward_context().attn_metadata
        pool = segs.pool
        n_blk = 1 + len(gdn_ids) * CAP * T
        dev = pool.device
        new_off = torch.zeros(n_blk, dtype=torch.int32, device=dev)
        # One gather builds the private pool (block 0 = the real null block, unused
        # slots also block 0): no FP8 fill/index_put, which fails on torch_tpu
        # ("a required argument is in a failed state" downstream).
        src = [torch.zeros(1, dtype=torch.int64, device=dev)]
        meta = []
        for k, i in enumerate(gdn_ids):
            md = md_all[segs.layers[i].linear_attn.prefix]
            st = md.mamba_state_indices.to(torch.int64)[:NS]
            ck = md.mamba_ckpt_indices.to(torch.int64)[:NS, :T]
            base = 1 + k * CAP * T
            src += [ck.reshape(-1), torch.zeros(CAP * T - NS * T, dtype=torch.int64, device=dev)]
            new_ck = (base + torch.arange(NS * T, device=dev).reshape(NS, T)).to(torch.int32)
            new_ck = torch.where(ck > 0, new_ck, torch.zeros_like(new_ck))
            new_off[new_ck[:, 0].long()] = md.mamba_slot_read_offsets.to(torch.int32)[st]
            meta += [new_ck[:, 0], None, new_ck, md.query_start_loc.to(torch.int32)[: NS + 1], md.seq_lens.to(torch.int32)[:NS]]
        small = pool[torch.cat(src)]
        return [small, *[new_off if m is None else m for m in meta]]

    def _mk_run(self, L: int, head: bool, w: list, h: Any, att: Any, gdn_io: list) -> Any:
        """The `L`-layer kernel on `gdn_io` from `_snap` (checkpoints not written)."""
        from .qwen38_gdn_state import Qwen38GdnPoolLayout

        key = (L, head, int(gdn_io[0].shape[0]) if gdn_io else 0)
        if key not in self.private_kinds:
            lay = self.segments.layout
            if gdn_io:
                lay = Qwen38GdnPoolLayout(
                    pool_shape=tuple(int(d) for d in gdn_io[0].shape), pool_dtype=lay.pool_dtype
                )
            sfx = f"_check{key[2]}"
            self.private_kinds[key] = _SegmentKind(self.mesh, self.g, L, lay, write=False, head=head, suffix=sfx)
        return self.private_kinds[key].op(h, h if att is None else att, w, gdn_io)[0]

    def _segxcheck(self, runner: Any, input_ids: Any, positions: Any, kw: dict, rows: int) -> Any:
        """Serve the step with eager stock layers; on the named segments also run
        the kernel on the same input (private GDN state) and log the diff."""
        from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp
        from vllm_torchtpu.models.vllm.qwen4_exp.qsa import QSAMetadata

        segs, bb = self.segments, runner.model.model
        layers, hc = bb.layers, bb.hc_count
        NS = rows // T_VERIFY
        pos64, ids64 = positions.to(torch.int64), input_ids.to(torch.int64)
        args = (
            pos64, ids64, kw["query_start_loc"], kw["ngram_context"], kw["state_slots"],
            kw["context_lens"], kw["valid_tokens"], kw["main_block_table"], kw["compressed_block_table"],
        )
        meta = QSAMetadata(
            kw["query_start_loc"], kw["context_lens"], kw["state_slots"],
            kw["main_block_table"], kw["compressed_block_table"],
        )
        starts = {segment_layers(self.g, s)[0]: s for s in self.segx_set}
        n = rows
        with torch.no_grad():
            h = bb.embed_tokens(input_ids).repeat(1, hc)
            i = 0
            while i < len(layers):
                if i in starts:
                    s = starts[i]
                    ids = segment_layers(self.g, s)
                    q = layers[i]

                    def ok(tag, t):
                        # Materialize `t` so an async failure names its stage.
                        logger.info("qwen38 segxcheck seg%d stage %s ok: sum %.6g", s, tag, float(t.float().sum()))

                    ok("in", h)
                    block, residual = q.attn_hyper_connection.mix(h)
                    att = q.self_attn.attend(block, pos64, meta)
                    ok("attend", att)
                    w = [getattr(segs, f"w{s}_{j}") for j in range(segs.n_w[s])]
                    if s not in self.segx_audited:
                        self.segx_audited.add(s)
                        R = self.mesh.size
                        ex = ExecSpec(R=R, B=segs.buckets[0], T=T_VERIFY, ctx=128, tiles=segment_tiles(segs.buckets[0]))
                        spec = qwen38.segment_spec(self.g, R, L=len(ids))
                        self._audit_operands(runner, w, _operand_names(spec, ex, len(ids)), spec, ex, ids, f"seg{s}", False)
                    snap = self._snap(ids[1:], NS)
                    ok("snap", snap[0])
                    mk = self._mk_run(len(ids), False, w, h, att, snap)
                    ok("mk", mk)
                    snaps = [self._snap([j], NS) for j in ids[1:]]
                    ok("snaps", snaps[0][0])
                    # Stock QSA layer on the same `att` (attention and its cache
                    # writes run once), staged like Qwen4ExpDecoderLayer.forward.
                    o = sp.reduce(q.self_attn.o_proj(att), False)
                    mixed2, residual2 = q.attn_hyper_connection.combine_mix(o, residual, q.mlp_hyper_connection)
                    mixed2 = torch.where(kw["valid_tokens"].to(torch.bool)[:, None], mixed2, 0)
                    st, ins = q.mlp_hyper_connection.combine(q.mlp(mixed2), residual2), [h]
                    ok("stock-qsa", st)
                    for j in ids[1:]:
                        ins.append(st)
                        st = layers[j](st, *args)
                    logger.info(
                        "qwen38 segxcheck seg%d layers %d-%d rows=%d in_norm=%.4f MK vs stock: %s (%d left)",
                        s, ids[0], ids[-1], rows, float(h.float().norm()), _diff_stats(mk, st, n, hc), self.segx_left,
                    )
                    # Each layer alone on its stock input: the QSA layer as an L=1
                    # segment, each GDN layer as an L=1 head (weights packed here and
                    # dropped after, to keep HBM free).
                    R = self.mesh.size
                    ex = ExecSpec(R=R, B=segs.buckets[0], T=T_VERIFY, ctx=128, tiles=segment_tiles(segs.buckets[0]))
                    try:
                        for k, j in enumerate(ids):
                            head = k > 0
                            spec = qwen38.segment_spec(self.g, R, L=1, head=head)
                            wj = pack_segment_operands(
                                runner.model, self.g, -1, spec, ex, layer_ids=[j], head=head,
                                boundary=bool(envs.VLLM_TPU_MK_OWN_OPERANDS),
                            )
                            out_j = self._mk_run(1, head, wj, ins[k], att if k == 0 else None, snaps[k - 1] if head else [])
                            ref_j = ins[k + 1] if k + 1 < len(ids) else st
                            logger.info(
                                "qwen38 segxcheck seg%d layer %d (%s) MK vs stock: %s",
                                s, j, "gdn" if head else "qsa-out", _diff_stats(out_j, ref_j, n, hc),
                            )
                            del wj, out_j
                    except Exception:
                        logger.exception("qwen38 segxcheck seg%d per-layer check failed", s)
                    h = st
                    i = ids[-1] + 1
                    continue
                h = layers[i](h, *args)
                i += 1
            mixed = bb.hyper_connection_mixer.mix(h)[0]
        return (mixed, h) if bb.emit_mtp_hidden else mixed


    def _attach_whole(self, runner: Any) -> None:
        """The whole-step program (its own compiled class); on failure it stays off."""
        from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpWholeStepModel

        from .qwen38_wholestep import Qwen38WholeStep

        try:
            attn = next(l.self_attn for l in self.segments.layers if l.layer_type == "full_attention")
            budget = int(attn.token_topk) // int(attn.compress_ratio)
            max_groups = -(-self.whole_cap // 4 // 1024) * 1024
            # vLLM compiles a module for every compile size on its first call: the whole-step
            # program compiles only the buckets it can serve (w04 j-05391f11: the 96-row bucket
            # of the segment driver ran out of VMEM, 67.9 of 63 MiB, and disabled the MK).
            buckets = tuple(b for b in self.buckets if b <= min(WHOLE_MAX_ROWS, self.whole_max_rows))
            if not buckets:
                raise ValueError(f"no whole-step bucket <= {self.whole_max_rows} rows in {self.buckets}")
            import copy

            self.whole_cfg = copy.copy(self.cfg)
            cc = copy.copy(self.cfg.compilation_config)
            cc.compile_sizes = list(buckets)
            object.__setattr__(self.whole_cfg, "compilation_config", cc)
            whole = Qwen38WholeStep(
                runner.model, self.g, self.mesh, buckets, budget=budget, max_groups=max_groups,
                hold=self.whole_hold,
            )
            whole.bind_pool(self.segments.pool, self.segments.layout)
            self.whole_module = whole
            with self._contexts(runner, "model_mk_whole"):
                self.whole_program = Qwen4ExpWholeStepModel(
                    vllm_config=self.whole_cfg,
                    prefix="model_mk_whole",
                    backbone=runner.model.model,
                    segments=self.segments,
                    whole=whole,
                )
            logger.info(
                "Qwen3.8 whole-step megakernel attached: layers 3..46, context cap %d, rows <= %d, "
                "budget %d, max_groups %d, %d operands, hold %s", self.whole_cap, self.whole_max_rows,
                budget, max_groups, whole.n_w, self.whole_hold,
            )
        except Exception:
            logger.exception("Qwen3.8 whole-step: setup failed; segment program only")
            self.whole_program = None

    def _whole_self_check(self, runner: Any) -> None:
        """Loud error if `whole=1` but no whole-step call ran in the first model steps after
        warmup: a silent fall-back to stock (w06g / w07g served GSM8K on stock after the MK
        failed at its first call) then shows within minutes instead of at the end of a run."""
        if getattr(self, "_whole_checked", False) or not getattr(runner, "warmup_done", False):
            return
        if not hasattr(self, "_whole_check_base"):
            self._whole_check_base, self._whole_check_steps = self.whole_calls, 0
        self._whole_check_steps += 1
        if self._whole_check_steps < WHOLE_SELF_CHECK_STEPS:
            return
        self._whole_checked = True
        served = self.whole_calls - self._whole_check_base
        if self.whole_dry:  # no whole-step call by design; report the would-be count instead
            logger.info(
                "Qwen3.8 whole-step self-check (wholedry): %d whole-step calls (expected 0), %d "
                "dry whole-eligible steps so far; step mix %s",
                served, sum(v for k, v in self.stats.items() if k.startswith("dry_whole_b")),
                dict(sorted(self.stats.items())),
            )
        elif served == 0:
            logger.error(
                "Qwen3.8 whole-step SELF-CHECK FAILED: whole=1 but no whole-step call in the "
                "first %d steps after warmup (megakernel %s); step mix %s",
                WHOLE_SELF_CHECK_STEPS,
                "disabled" if self.whole_program is None else "on, nothing eligible",
                dict(sorted(self.stats.items())),
            )
        else:
            logger.info(
                "Qwen3.8 whole-step self-check: %d whole-step calls in the first %d verify steps",
                served, WHOLE_SELF_CHECK_STEPS,
            )

    def _whole_eligible(self, runner: Any, n_req: int, rows: int) -> bool:
        if self.whole_program is None or rows > min(WHOLE_MAX_ROWS, self.whole_max_rows):
            return False
        try:
            ctx = runner.input_batch.num_computed_tokens_cpu[:n_req]
            return int(ctx.max()) + T_VERIFY <= self.whole_cap
        except Exception:
            return False

    def _attach_mtp_tail(self, runner: Any) -> None:
        """Route the MTP draft layer's post-attention half to the tail kernel.

        Must run before the draft's first trace; the draft keeps its stock
        layer for rows counts outside the buckets and whenever this fails.
        """
        from vllm_torchtpu.models.vllm.qwen4_exp.mtp import Qwen4ExpMTP

        if not (envs.VLLM_TPU_MTP_DRAFT_ROWS and envs.VLLM_TPU_FUSED_DRAFT_LOOP):
            # Without draft rows, step 0's all-token draft call could carry a
            # prefill-sized chunk, which the tail kernel does not take.
            logger.warning("Qwen3.8 MTP tail: needs MTP draft rows and the fused draft loop")
            return
        draft = getattr(getattr(runner, "drafter", None), "draft_model", None)
        if not isinstance(draft, Qwen4ExpMTP):
            logger.warning("Qwen3.8 MTP tail: no Qwen4ExpMTP draft model (%s)", type(draft))
            return
        o_proj = draft.model.layers[0].self_attn.o_proj.weight
        if self.g.NH // self.mesh.size * self.g.HD not in tuple(o_proj.shape):
            logger.warning("Qwen3.8 MTP tail: draft o_proj %s is not TP-sharded like the target", tuple(o_proj.shape))
            return
        try:
            kind = MtpTailKind(
                self.mesh, self.g, self.segments.layout, self.segments.write_ckpts,
                min(max(self.buckets), KERNEL_ROWS),
            )
            tail = MtpTail(draft, self.g, self.mesh, kind, self.buckets)
        except Exception:
            logger.exception("Qwen3.8 MTP tail: packing failed; draft stays stock")
            return
        draft.model.mk_tail = tail
        logger.info("Qwen3.8 MTP draft tail kernel attached, rows buckets %s", self.buckets)

    def _shadow_step(self, runner: Any, input_ids: Any, positions: Any, kw: dict) -> Any:
        """Run the kernel program (no GDN writes) then the stock model; log, return stock."""
        with self._contexts(runner, "model_mk_segments_shadow"):
            mk = self.shadow(input_ids=input_ids, positions=positions, **kw)
        stock = runner.model(input_ids=input_ids, positions=positions, **kw)
        mk_h = mk[0] if isinstance(mk, (tuple, list)) else mk
        st_h = stock[0] if isinstance(stock, (tuple, list)) else stock
        d = (mk_h.float() - st_h.float()).abs()
        rel = float(d.mean() / st_h.float().abs().mean().clamp_min(1e-6))
        top_mk = runner.model.compute_logits(mk_h).argmax(-1)
        top_st = runner.model.compute_logits(st_h).argmax(-1)
        agree = float((top_mk == top_st).float().mean())
        logger.info(
            "qwen38 megakernel shadow check rows=%d mean_rel=%.4g max_abs=%.4g "
            "top1_agree=%.4f (%d left)",
            int(input_ids.shape[0]),
            rel,
            float(d.max()),
            agree,
            self.check_left - 1,
        )
        self.check_left -= 1
        return stock

    def _tp_barrier(self, what: str) -> None:
        import time

        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group

        t0 = time.time()
        dist.barrier(group=get_tp_group().cpu_group)
        logger.info("qwen38 megakernel tp barrier %s: %.2f s", what, time.time() - t0)

    def _log_program_fingerprint(self) -> None:
        """md5 of each compiled segment-program artifact of this rank (are all ranks' programs the same?)."""
        import glob
        import hashlib

        from vllm.distributed.parallel_state import get_tp_group

        r = get_tp_group().rank_in_group
        files = sorted(
            glob.glob(f"/root/vllm-torchtpu-cache/vllm/torch_compile_cache/*/rank_{r}_*/model_mk_segments/*")
        )
        import os

        marks = []
        for f in files:
            h = hashlib.md5()
            parts = [f] if os.path.isfile(f) else sorted(
                os.path.join(d, n) for d, _, ns in os.walk(f) for n in ns
            )
            for part in parts:
                try:
                    with open(part, "rb") as fh:
                        h.update(fh.read())
                except OSError:
                    continue
            marks.append(f"{f.rsplit('/', 1)[-1]}={h.hexdigest()[:10]}")
        logger.info("qwen38 megakernel program fingerprint rank %d: %s", r, " ".join(marks) or "no artifacts")

    def _log_gdn_meta(self) -> None:
        """Host-side ranges of the first GDN layer's metadata and the pool block count."""
        try:
            from vllm.forward_context import get_forward_context

            seg = self.segments
            layer = next(l for l in seg.layers if l.layer_type == "linear_attention")
            md = get_forward_context().attn_metadata[layer.linear_attn.prefix]
            parts = []
            for name in ("mamba_state_indices", "mamba_slot_read_offsets", "mamba_ckpt_indices",
                         "query_start_loc", "seq_lens"):
                t = getattr(md, name).detach().to("cpu").to(torch.int64)
                parts.append(f"{name}{tuple(t.shape)}[{int(t.min())},{int(t.max())}]")
            lay = seg.layout
            logger.info(
                "qwen38 megakernel gdn meta: %s; pool blocks %d",
                " ".join(parts),
                lay.pool_shape[0] // lay.split,
            )
        except Exception as exc:
            logger.info("qwen38 megakernel gdn meta: unavailable (%s)", exc)

    def _count(self, key: str, n: int) -> None:
        """Add `n` to a counter logged with the step mix without counting a step (drafted
        tokens per path, for attributing accepted tokens to whole-step vs stock steps)."""
        self.counters = getattr(self, "counters", {})
        self.counters[key] = self.counters.get(key, 0) + n

    def _note(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1
        n = sum(self.stats.values())
        # Powers of two early, then every STEP_MIX_EVERY steps, so the mix can be joined in
        # time to the serving metrics (AL windows) over a whole run.
        if (n & (n - 1) == 0 and n >= 64) or n % STEP_MIX_EVERY == 0:
            logger.info("qwen38 megakernel step mix: %s", dict(sorted(self.stats.items())))
        if STEP_MIX_INTERVAL_S:
            import time

            now = time.monotonic()
            if now - getattr(self, "_mix_t", 0.0) >= STEP_MIX_INTERVAL_S:
                # Cumulative counts on a wall-clock cadence matching vLLM's spec-decode metrics
                # interval: differencing both per window attributes accepted / drafted tokens to
                # whole-step vs stock steps (tools: whole_path_al.py).
                self._mix_t = now
                logger.info(
                    "qwen38 megakernel step mix (timed): %s counters %s",
                    dict(sorted(self.stats.items())), dict(getattr(self, "counters", {})),
                )

    def try_forward(
        self,
        runner: Any,
        input_ids: Any,
        positions: Any,
        inputs_embeds: Any,
        intermediate_tensors: Any,
        model_kwargs: Any = None,
    ) -> tuple[bool, Any]:
        self.calls_total = getattr(self, "calls_total", 0) + 1
        if inputs_embeds is not None or intermediate_tensors is not None:
            # Counted like every other stock route, so a bypass shows in the mix.
            self._note("stock_embeds" if inputs_embeds is not None else "stock_intermediate")
            return False, None
        if self.program is None:  # disabled (it failed): the self-check still counts steps
            self._note(getattr(self, "disabled_reason", "stock_noprogram"))
            if self.whole:
                self._whole_self_check(runner)
            return False, None
        self.seq += 1
        ok, n_req, t_seq, reason = MegakernelDriver._inspect_step_detail(None, None, runner)
        rows = int(input_ids.shape[0])
        if not ok or t_seq != T_VERIFY:
            self._note(reason if not ok else f"stock_t{t_seq}")
            return False, None
        if self.whole:
            self._whole_self_check(runner)
        if rows not in self.buckets or n_req * T_VERIFY > rows:
            self._note(f"stock_rows{rows}_n{n_req}")
            return False, None
        if (
            self.whole_program is not None
            and self.whole_above == "stock"
            and not self._whole_eligible(runner, n_req, rows)
        ):
            # Above `wholerows` or past `wholecap`: stock, not the segment program, which only
            # compiles (all its buckets, mid-serving) on its first call (w05 j-84f89805: a c1
            # AgentX request past the 32K cap started that compile and the sampler RPC timed out).
            above = rows > min(WHOLE_MAX_ROWS, self.whole_max_rows)
            self._note(f"stock_above_whole_b{rows}" if above else "stock_whole_ctx")
            self._count("drafts_stock", n_req * (T_VERIFY - 1))
            return False, None
        if self.whole_dry and self.whole_program is not None:
            # `wholedry=1`: the same eligibility decision and per-path draft counters, but the
            # whole-step-eligible steps run stock too: the stock acceptance of exactly the
            # requests whole-step would serve (per-path AL control).
            self._note(f"dry_whole_b{rows}")
            self._count("drafts_whole", n_req * (T_VERIFY - 1))
            return False, None
        self._note(f"mk_b{rows}")
        self.mk_calls += 1
        if self.mk_calls <= 4:
            logger.info(
                "qwen38 megakernel call %d at step seq %d: rows=%d n_req=%d",
                self.mk_calls, self.seq, rows, n_req,
            )
            if self.mk_calls <= 2:
                self._log_gdn_meta()
        # Read-only below: no per-step copy.
        kw = model_kwargs if model_kwargs is not None else {}
        if self.segx_left > 0 or getattr(self, "segx_done", False):
            if self.segx_left > 0 and getattr(runner, "warmup_done", False) and _real_traffic(kw):
                self.segx_left -= 1
                self.segx_done = True
                try:
                    return True, self._segxcheck(runner, input_ids, positions, kw, rows)
                except Exception:
                    logger.exception("qwen38 megakernel segxcheck failed; segxcheck off")
                    self.segx_left = 0
            return False, None
        if self.h0check_left > 0 or getattr(self, "h0check_done", False):
            warm, real = bool(getattr(runner, "warmup_done", False)), _real_traffic(kw)
            want = not getattr(self, "h0rows", 0) or rows == self.h0rows
            seen = self.__dict__.setdefault("_h0_gate_seen", set())
            if self.h0check_left > 0 and (rows, warm, real) not in seen:
                # A check that silently never fires looks like a clean pass: log
                # each (rows, warmup_done, real traffic) gate state once.
                seen.add((rows, warm, real))
                logger.info(
                    "qwen38 h0check gate: rows=%d n_req=%d warmup_done=%s real_traffic=%s rows_wanted=%s left=%d",
                    rows, n_req, warm, real, want, self.h0check_left,
                )
            if self.h0check_left > 0 and warm and real and want:
                self.h0check_left -= 1
                self.h0check_done = True
                try:
                    return True, self._h0check(runner, input_ids, positions, kw, rows, n_req)
                except Exception:
                    logger.exception("qwen38 megakernel h0check failed; h0check off")
                    self.h0check_left = 0
            return False, None
        first_warm = self.warm and self.mk_calls == 1
        if first_warm:
            self._tp_barrier("before the first (compiling) call")
        try:
            if self.check_left > 0 and self.shadow is not None:
                out = self._shadow_step(runner, input_ids, positions, kw)
            elif self._whole_eligible(runner, n_req, rows):
                self.whole_calls += 1
                self._count("drafts_whole", n_req * (T_VERIFY - 1))
                self._note(f"whole_b{rows}")
                with self._contexts(runner, "model_mk_whole"):
                    out = self.whole_program(input_ids=input_ids, positions=positions, **kw)
                from . import qwen38_wholestep_call as wsc

                if wsc.union_stats_on() and self.whole_calls % wsc.UNION_STATS_EVERY == 0:
                    self.whole_module.log_union_stats(self.whole_calls)
            else:
                with self._serve_contexts(runner):
                    out = self.program(input_ids=input_ids, positions=positions, **kw)
        except Exception:
            # Fail closed: one failure disables the megakernel for this server;
            # this step and all later ones run the stock model.
            logger.exception("Qwen3.8 megakernel failed; disabling it, stock path from now on")
            self.program = self.shadow = self.whole_program = None
            return False, None
        if isinstance(out, (list, tuple)):
            out = tuple(out)
        if first_warm:
            from torch_tpu._internal import sync as tpu_sync

            tpu_sync.synchronize(list(out) if isinstance(out, tuple) else [out], wait=True)
            self._log_program_fingerprint()
            self._tp_barrier("after the first call finished on device")
        if self.sync and self.mk_calls <= 4:
            import time

            from torch_tpu._internal import sync as tpu_sync

            t0 = time.time()
            tpu_sync.synchronize(list(out) if isinstance(out, tuple) else [out], wait=True)
            logger.info(
                "qwen38 megakernel call %d done on device after %.2f s", self.mk_calls, time.time() - t0
            )
        return True, out


register_megakernel_driver("Qwen4ExpForCausalLM", Qwen38SegmentDriver.maybe_create)
register_megakernel_driver("qwen4_exp", Qwen38SegmentDriver.maybe_create)

__all__ = ["Qwen38SegmentDriver", "Qwen38Segments"]
