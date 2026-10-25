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
from .qwen38_segment_call import T_VERIFY, build_segment_call, segment_tiles
from .qwen38_weights import head_groups, pack_segment_operands, segment_layers

logger = init_logger(__name__)

# With the C1 tiles the kernel beats stock through 64 rows (mk/kernel_work/
# mk-row-slope/report.md vs mk/stock-segment-vs-rows.md); 64 is the margin case.
# 24 and 48 are serving token buckets too (TPU_TOKEN_BUCKET_EXTRA); without them
# 24-/48-row verify steps (6 or 12 requests) run stock.
DEFAULT_BUCKETS = (4, 8, 16, 24, 32, 48, 64)


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


# `VLLM_TPU_MEGAKERNEL=...,nodonate=1`: the GDN pool is not donated to the segment op; it
# writes a fresh pool that `_run` copies back into the live cache (bisecting a served hang).
NO_DONATE = False

_UNSET = object()


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
        self, mesh: Any, g: Any, L: int, layout: Any, write: bool, head: bool = False
    ) -> None:
        self.mesh, self.g, self.L, self.layout, self.write = mesh, g, L, layout, write
        self.head = head
        self.has_gdn = head or L > 1
        self._calls: dict[int, Any] = {}
        self.tag = f"{'head' if head else 'seg'}_l{L}" + ("" if write else "_shadow")
        name = f"qwen38_mk::{self.tag}"
        has_gdn = self.has_gdn

        @torch.library.custom_op(name, mutates_args=())
        def op(
            hidden: torch.Tensor,
            attn: torch.Tensor,
            weights: list[torch.Tensor],
            gdn_io: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            call = self.callable_for(int(hidden.shape[0]))
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

    def callable_for(self, B: int) -> Any:
        if B not in self._calls:
            full, in_specs, outs, donate = build_segment_call(
                self.mesh, self.g, self.L, B, self.layout, self.write, head=self.head
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
            ops = pack_segment_operands(model, g, s, qwen38.segment_spec(g, R, L=L), ex)
            self.n_w.append(len(ops))
            for i, t in enumerate(ops):
                self.register_buffer(f"w{s}_{i}", t, persistent=False)
        self.n_hw: list[int] = []
        for h, group in enumerate(self.head_groups):
            L = len(group)
            ex = ExecSpec(R=R, B=buckets[0], T=T_VERIFY, ctx=128, tiles=segment_tiles(buckets[0]))
            spec = qwen38.segment_spec(g, R, L=L, head=True)
            ops = pack_segment_operands(model, g, -1, spec, ex, layer_ids=group, head=True)
            self.n_hw.append(len(ops))
            for i, t in enumerate(ops):
                self.register_buffer(f"h{h}_{i}", t, persistent=False)
        self.pool = None
        self.layout = None

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
        self.stats: dict[str, int] = {}
        self.mtp = False
        self.cfg = None
        self.seq = 0  # try_forward calls, to compare the step sequence across ranks
        self.mk_calls = 0
        self.sync = False  # `sync=1`: wait for the first calls' outputs and log it (bisecting)
        self.warm = False  # `warm=1`: host barriers around the first (compiling) call
        self.noaot = False  # `noaot=1`: no AOT serialization / compile cache for the segment program

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
        cfg = getattr(self, "cfg", None) or runner.vllm_config
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

    def bind_and_warmup(self, runner: Any) -> None:
        from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpSegmentModel

        self.segments.bind_pool()
        self.cfg = self._segment_config(runner)
        with self._contexts(runner, "model_mk_segments"):
            self.program = Qwen4ExpSegmentModel(
                vllm_config=self.cfg,
                prefix="model_mk_segments",
                backbone=runner.model.model,
                segments=self.segments,
            )
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

    def _h0check(self, runner: Any, input_ids: Any, kw: dict, rows: int, n_req: int) -> None:
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
        pool = segs.pool
        flat = ck.reshape(-1)
        saved = pool[flat].clone()
        # Private pool: block 0 null, request s column t at 1 + s*T + t.
        small = torch.cat([torch.zeros_like(pool[:1]), saved], dim=0)
        new_ck = (1 + torch.arange(NS * T, device=ck.device).reshape(NS, T)).to(torch.int32)
        new_ck = torch.where(ck > 0, new_ck, torch.zeros_like(new_ck))
        new_st = new_ck[:, 0]
        new_off = torch.zeros(small.shape[0], dtype=torch.int32, device=ck.device)
        new_off[new_st.long()] = off_tab[st].to(torch.int32)
        if self.h0kind is None:
            lay = Qwen38GdnPoolLayout(pool_shape=tuple(int(d) for d in small.shape), pool_dtype=segs.layout.pool_dtype)
            self.h0kind = _SegmentKind(self.mesh, self.g, len(segs.head_groups[0]), lay, write=False, head=True)
        w = [getattr(segs, f"h0_{i}") for i in range(segs.n_hw[0])]
        meta = [new_st, new_off, new_ck, qsl, seq]
        with torch.no_grad():
            h_in = bb.embed_tokens(input_ids).repeat(1, hc)
            mk_out = self.h0kind.op(h_in, h_in, w, [small, *meta])[0]
            # Stock layer 0, staged like Qwen4ExpDecoderLayer.forward.
            block, residual = layer.attn_hyper_connection.mix(h_in)
            g = layer.linear_attn(block)
            mixed2, residual2 = layer.attn_hyper_connection.combine_mix(g, residual, layer.mlp_hyper_connection)
            valid = kw["valid_tokens"].to(torch.bool)
            mixed2 = torch.where(valid[:, None], mixed2, 0)
            moe = layer.mlp(mixed2)
            st_out = layer.mlp_hyper_connection.combine(moe, residual2)
        n = NS * T
        pool[flat] = saved  # undo stock layer 0's checkpoint writes; the stock model serves this step
        logger.info(
            "qwen38 h0check inputs: rows=%d n_req=%d h_in_norm=%.4f st=%s off=%s ck0=%s seq=%s",
            rows, n_req, float(h_in.float()[:n].norm()), st.tolist()[:8], off_tab[st].tolist()[:8],
            ck[:2].tolist(), seq.tolist()[:8],
        )
        logger.info(
            "qwen38 h0check layer0_out MK vs stock: %s | stock residual-after-GDN(norm)=%.4f moe(norm)=%.4f (%d left)",
            _diff_stats(mk_out, st_out, n, hc), float((residual2[0] if isinstance(residual2, tuple) else residual2).float()[:n].norm()),
            float(moe.float()[:n].norm()), self.h0check_left - 1,
        )

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
                self.mesh, self.g, self.segments.layout, self.segments.write_ckpts, max(self.buckets)
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

    def _note(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1
        n = sum(self.stats.values())
        if n & (n - 1) == 0 and n >= 64:
            logger.info("qwen38 megakernel step mix: %s", dict(sorted(self.stats.items())))

    def try_forward(
        self,
        runner: Any,
        input_ids: Any,
        positions: Any,
        inputs_embeds: Any,
        intermediate_tensors: Any,
        model_kwargs: Any = None,
    ) -> tuple[bool, Any]:
        if self.program is None or inputs_embeds is not None or intermediate_tensors is not None:
            return False, None
        self.seq += 1
        ok, n_req, t_seq, reason = MegakernelDriver._inspect_step_detail(None, None, runner)
        rows = int(input_ids.shape[0])
        if not ok or t_seq != T_VERIFY:
            self._note(reason if not ok else f"stock_t{t_seq}")
            return False, None
        if rows not in self.buckets or n_req * T_VERIFY > rows:
            self._note(f"stock_rows{rows}_n{n_req}")
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
        kw = dict(model_kwargs or {})
        if self.h0check_left > 0 or getattr(self, "h0check_done", False):
            if self.h0check_left > 0 and getattr(runner, "warmup_done", False) and _real_traffic(kw):
                try:
                    self._h0check(runner, input_ids, kw, rows, n_req)
                except Exception:
                    logger.exception("qwen38 megakernel h0check failed")
                self.h0check_left -= 1
                self.h0check_done = True
            return False, None
        first_warm = self.warm and self.mk_calls == 1
        if first_warm:
            self._tp_barrier("before the first (compiling) call")
        try:
            if self.check_left > 0 and self.shadow is not None:
                out = self._shadow_step(runner, input_ids, positions, kw)
            else:
                with self._contexts(runner, "model_mk_segments"):
                    out = self.program(input_ids=input_ids, positions=positions, **kw)
        except Exception:
            # Fail closed: one failure disables the megakernel for this server;
            # this step and all later ones run the stock model.
            logger.exception("Qwen3.8 megakernel failed; disabling it, stock path from now on")
            self.program = self.shadow = None
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
