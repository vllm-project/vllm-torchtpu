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
from .exported_bridge import sharded_jax_callable
from .qwen38_segment_call import T_VERIFY, build_segment_call, segment_tiles
from .qwen38_weights import head_groups, pack_segment_operands, segment_layers

logger = init_logger(__name__)

DEFAULT_BUCKETS = (4, 8, 16, 32, 64)


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
            self._calls[B] = sharded_jax_callable(
                f"qwen38_mk::{self.tag}_b{B}",
                full,
                mesh=self.mesh,
                input_partition_specs=in_specs,
                output_partition_specs=outs,
                donate_argnums=donate,
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
        self.layers = model.model.layers
        self.n_seg = g.L // g.att_every
        self.head_groups = head_groups(model, g) if head else []
        self.kinds: dict[tuple[int, bool], _SegmentKind] = {}
        if shared is not None:
            # A shadow view: same operands, GDN checkpoints not written.
            self.n_w, self.n_hw = shared.n_w, shared.n_hw
            for name, buf in shared.named_buffers():
                self.register_buffer(name, buf, persistent=False)
            self._shared = shared
            self.pool = self.layout = None
            return
        self._shared = None
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

    def forward(self, s: int, hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        ids = segment_layers(self.g, s)
        kind = self.kinds[(len(ids), False)]
        w = [getattr(self, f"w{s}_{i}") for i in range(self.n_w[s])]
        if kind.has_gdn:
            res, pool = kind.op(hidden, attention, w, [self.pool, *self._gdn_meta(ids[1:])])
            self.pool.copy_(pool)
        else:
            (res,) = kind.op(hidden, attention, w, [])
        return res

    def head(self, h: int, hidden: torch.Tensor) -> torch.Tensor:
        """Head group `h` (layers before the first QSA layer) in one kernel call."""
        ids = self.head_groups[h]
        kind = self.kinds[(len(ids), True)]
        w = [getattr(self, f"h{h}_{i}") for i in range(self.n_hw[h])]
        res, pool = kind.op(hidden, hidden, w, [self.pool, *self._gdn_meta(ids)])
        self.pool.copy_(pool)
        return res


class Qwen38SegmentDriver:
    """Routes uniform verify steps to `Qwen4ExpSegmentModel`."""

    takes_model_kwargs = True

    def __init__(self, runner: Any, buckets: tuple[int, ...], head: bool = True) -> None:
        self.buckets = buckets
        self.g = qwen38.FULL
        self.mesh = _build_tp_rank_mesh(8, getattr(runner, "mesh", None))
        self.segments = Qwen38Segments(runner.model, self.g, self.mesh, buckets, head=head)
        self.program = None
        self.check_left = 0
        self.shadow = None
        self.stats: dict[str, int] = {}

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
        drv = cls(runner, _parse_buckets(raw), head=_parse_flag(raw, "head", True))
        drv.check_left = _parse_check(raw)
        logger.info(
            "Qwen3.8 segment megakernel enabled, verify buckets %s, shadow checks %d",
            drv.buckets,
            drv.check_left,
        )
        return drv

    def bind_and_warmup(self, runner: Any) -> None:
        from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpSegmentModel

        self.segments.bind_pool()
        self.program = Qwen4ExpSegmentModel(
            vllm_config=runner.vllm_config,
            prefix="model_mk_segments",
            backbone=runner.model.model,
            segments=self.segments,
        )
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
            self.shadow = Qwen4ExpSegmentModel(
                vllm_config=runner.vllm_config,
                prefix="model_mk_segments_shadow",
                backbone=runner.model.model,
                segments=shadow_seg,
            )

    def _shadow_step(self, runner: Any, input_ids: Any, positions: Any, kw: dict) -> Any:
        """Run the kernel program (no GDN writes) then the stock model; log, return stock."""
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
        ok, n_req, t_seq, reason = MegakernelDriver._inspect_step_detail(None, None, runner)
        rows = int(input_ids.shape[0])
        if not ok or t_seq != T_VERIFY:
            self._note(reason if not ok else f"stock_t{t_seq}")
            return False, None
        if rows not in self.buckets or n_req * T_VERIFY > rows:
            self._note(f"stock_rows{rows}_n{n_req}")
            return False, None
        self._note(f"mk_b{rows}")
        kw = dict(model_kwargs or {})
        if self.check_left > 0 and self.shadow is not None:
            out = self._shadow_step(runner, input_ids, positions, kw)
        else:
            out = self.program(input_ids=input_ids, positions=positions, **kw)
        if isinstance(out, (list, tuple)):
            out = tuple(out)
        return True, out


register_megakernel_driver("Qwen4ExpForCausalLM", Qwen38SegmentDriver.maybe_create)
register_megakernel_driver("qwen4_exp", Qwen38SegmentDriver.maybe_create)

__all__ = ["Qwen38SegmentDriver", "Qwen38Segments"]
