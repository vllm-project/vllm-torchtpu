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

Enable with `VLLM_TPU_MEGAKERNEL=1` (or `buckets=4:8:16:32`).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_step import make_flat_segment_step

from .driver import (
    MegakernelDriver,
    _build_tp_rank_mesh,
    _resolve_tp_size,
    register_megakernel_driver,
)
from .exported_bridge import sharded_jax_callable
from .qwen38_weights import pack_segment_operands, segment_layers

logger = init_logger(__name__)

T_VERIFY = 4
DEFAULT_BUCKETS = (4, 8, 16, 32, 64)


def segment_tiles(B: int, hier: int | None = None) -> Tiles:
    """Expert ring on the software-pipelined EP path (e13 in 2 tiles, e2 in 1).

    Ring depths shrink with the bucket so the 64-row scratch fits VMEM; the
    bank sizes (hence every weight's tiling) are the same for all buckets.
    """
    nbe, nbd = (12, 10) if B <= 16 else ((6, 10) if B <= 32 else (3, 6))
    h = 1 if hier is None else hier
    return Tiles(ebank=2 << 20, nbe=nbe, nbd=nbd, vmem_mib=62, hier=h).with_blocks(
        {"moe": {"pipe_ep": True, "moe_unroll": True}}
    )


def _parse_buckets(raw: str) -> tuple[int, ...]:
    for part in raw.split(","):
        if part.startswith("buckets="):
            return tuple(sorted(int(b) for b in part[len("buckets=") :].split(":")))
    return DEFAULT_BUCKETS


def _parse_check(raw: str) -> int:
    """`check=N`: shadow the first N eligible steps against the stock model."""
    for part in raw.split(","):
        if part.startswith("check="):
            return int(part[len("check=") :])
    return 0


class _BucketOp:
    """SPMD gather -> segment kernel -> scatter for one `(kind, rows)` bucket."""

    def __init__(self, mesh: Any, g: Any, L: int, B: int, layout: Any) -> None:
        import jax
        from jax.sharding import PartitionSpec as P

        from .qwen38_gdn_state import gather_gdn_states, scatter_gdn_ckpts

        R = mesh.size
        self.ex = ExecSpec(R=R, B=B, T=T_VERIFY, ctx=128, tiles=segment_tiles(B))
        spec = qwen38.segment_spec(g, R, L=L)
        fn, _, out_specs, self.n_w = make_flat_segment_step(mesh, spec, self.ex)
        self.has_gdn = len(out_specs) == 3
        self.M = self.ex.M
        tp, rep = P("tp"), P()

        if self.has_gdn:
            gather = jax.shard_map(
                lambda pool, rb, hi: gather_gdn_states(
                    pool, rb, layout=layout, T=T_VERIFY, has_initial_state=hi
                ),
                mesh=mesh,
                in_specs=(tp, rep, rep),
                out_specs=(tp, tp),
                check_vma=False,
            )
            scatter = jax.shard_map(
                lambda pool, c, r, wb: scatter_gdn_ckpts(pool, c, r, wb, layout=layout),
                mesh=mesh,
                in_specs=(tp, tp, tp, rep),
                out_specs=tp,
                check_vma=False,
            )

            def full(pos, hidden, attn, *rest):
                w, (pool, rb, wb, hi) = rest[: self.n_w], rest[self.n_w :]
                conv, rec = gather(pool, rb, hi)
                res, conv_ck, rec_ck = fn(pos, hidden, attn, *w, conv, rec)
                return res, scatter(pool, conv_ck, rec_ck, wb)

            in_specs = (rep, tp, tp) + (tp,) * self.n_w + (tp, rep, rep, rep)
            outs = (tp, tp)
            donate = (3 + self.n_w,)
        else:

            def full(pos, hidden, attn, *w):
                return fn(pos, hidden, attn, *w)

            in_specs = (rep, tp, tp) + (tp,) * self.n_w
            outs = (tp,)
            donate = ()
        name = f"qwen38_mk::seg_l{L}_b{B}"
        call = sharded_jax_callable(
            name,
            full,
            mesh=mesh,
            input_partition_specs=in_specs,
            output_partition_specs=outs,
            donate_argnums=donate,
        )
        has_gdn = self.has_gdn

        @torch.library.custom_op(name, mutates_args=())
        def op(
            pos: torch.Tensor,
            hidden: torch.Tensor,
            attn: torch.Tensor,
            weights: list[torch.Tensor],
            pool_io: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            return list(call(pos, hidden, attn, *weights, *pool_io))

        @op.register_fake
        def _(pos, hidden, attn, weights, pool_io):
            out = [torch.empty_like(hidden)]
            if has_gdn:
                out.append(torch.empty_like(pool_io[0]))
            return out

        self.op = op


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
    ) -> None:
        super().__init__()
        self.write_ckpts = shared is None
        self.g = g
        self.mesh = mesh
        self.buckets = buckets
        self.layers = model.model.layers
        self.n_seg = g.L // g.att_every
        R = mesh.size
        self.n_w: list[int] = []
        if shared is not None:
            # A shadow view: same operands and ops, GDN checkpoints not written.
            self.n_w = shared.n_w
            for name, buf in shared.named_buffers():
                self.register_buffer(name, buf, persistent=False)
            self.pool, self.layout, self._ops = None, None, shared._ops
            self._shared = shared
            return
        self._shared = None
        for s in range(self.n_seg):
            L = len(segment_layers(g, s))
            ex = ExecSpec(R=R, B=buckets[0], T=T_VERIFY, ctx=128, tiles=segment_tiles(buckets[0]))
            ops = pack_segment_operands(model, g, s, qwen38.segment_spec(g, R, L=L), ex)
            self.n_w.append(len(ops))
            for i, t in enumerate(ops):
                self.register_buffer(f"w{s}_{i}", t, persistent=False)
        self.pool = None
        self.layout = None
        self._ops: dict[tuple[int, int], _BucketOp] = {}

    def bind_pool(self) -> None:
        from .qwen38_gdn_state import Qwen38GdnPoolLayout

        if self._shared is not None:
            self.pool, self.layout = self._shared.pool, self._shared.layout
            return

        pools = [self.layers[i].linear_attn.kv_cache[0] for i in range(len(self.layers))
                 if self.layers[i].layer_type == "linear_attention"]
        base = pools[0]
        for p in pools:
            if p.data_ptr() != base.data_ptr():
                raise RuntimeError("Qwen3.8 megakernel needs one unified GDN pool tensor")
        if getattr(self.layers[0].linear_attn, "_bm_kernel", False):
            raise RuntimeError("Qwen3.8 megakernel does not support block-major pools yet")
        self.pool = base
        self.layout = Qwen38GdnPoolLayout.from_pool(base)

    def _op(self, L: int, B: int) -> _BucketOp:
        key = (L, B)
        if key not in self._ops:
            self._ops[key] = _BucketOp(self.mesh, self.g, L, B, self.layout)
        return self._ops[key]

    def _gdn_blocks(self, s: int, NS: int) -> tuple[torch.Tensor, ...]:
        from vllm.forward_context import get_forward_context

        def fit(t: torch.Tensor, n: int) -> torch.Tensor:
            """First `n` rows of `t`, zero-padded when the runner's array is shorter."""
            t = t[:n]
            if t.shape[0] < n:
                t = F.pad(t, (0, 0) * (t.dim() - 1) + (0, n - t.shape[0]))
            return t

        md_all = get_forward_context().attn_metadata
        rbs, wbs, his = [], [], []
        for i in segment_layers(self.g, s)[1:]:
            la = self.layers[i].linear_attn
            md = md_all[la.prefix]
            st = fit(la._bm_state_ids(md.mamba_state_indices.to(torch.int32)), NS)
            ck = fit(md.mamba_ckpt_indices.to(torch.int32)[:, :T_VERIFY], NS)
            off = md.mamba_slot_read_offsets.to(torch.int64)[st.to(torch.int64)]
            qsl = fit(md.query_start_loc.to(torch.int32), NS + 1)
            q_len = qsl[1:] - qsl[:-1]
            live = q_len > 0
            hi = live & ((fit(md.seq_lens.to(torch.int32), NS) - q_len) > 0)
            rb = torch.gather(ck, 1, off[:, None]).squeeze(1)
            rbs.append(torch.where(live, rb, 0))
            write = live[:, None] if self.write_ckpts else torch.zeros_like(live)[:, None]
            wbs.append(torch.where(write, ck, 0))
            his.append(hi)
        return torch.stack(rbs), torch.stack(wbs), torch.stack(his)

    def forward(self, s: int, hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        B = hidden.shape[0]
        L = len(segment_layers(self.g, s))
        op = self._op(L, B)
        M = op.M
        pos = torch.zeros((B,), dtype=torch.int32, device=hidden.device)
        x = F.pad(hidden, (0, 0, 0, M - B)) if M > B else hidden
        att = F.pad(attention, (0, 0, 0, M - B)) if M > B else attention
        w = [getattr(self, f"w{s}_{i}") for i in range(self.n_w[s])]
        if op.has_gdn:
            rb, wb, hi = self._gdn_blocks(s, B // T_VERIFY)
            res, pool = op.op(pos, x, att, w, [self.pool, rb, wb, hi])
            self.pool.copy_(pool)
        else:
            (res,) = op.op(pos, x, att, w, [])
        return res[:B]


class Qwen38SegmentDriver:
    """Routes uniform verify steps to `Qwen4ExpSegmentModel`."""

    takes_model_kwargs = True

    def __init__(self, runner: Any, buckets: tuple[int, ...]) -> None:
        self.buckets = buckets
        self.g = qwen38.FULL
        self.mesh = _build_tp_rank_mesh(8, getattr(runner, "mesh", None))
        self.segments = Qwen38Segments(runner.model, self.g, self.mesh, buckets)
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
        drv = cls(runner, _parse_buckets(raw))
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
                runner.model, self.g, self.mesh, self.buckets, shared=self.segments
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
