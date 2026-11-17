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
"""`Qwen3VLProgram`: the Qwen3-VL text decoder megakernel as a `MonoProgram`.

`create` checks the checkpoint without loading anything; `load` packs the
bf16 weights of the loaded vLLM model; `bind` aliases its 64 per-layer bf16
KV caches; `build` returns one bridge step per decode bucket, which embeds the
ids, gathers each row's M-RoPE cos / sin from vLLM's rotary cache and runs
the backbone through the final norm (the runner computes the logits);
`decide` / `forward` route decode (`T=1`) steps whose requests are all past
their prompt.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.adapter import (
    build_pooled_step_op,
    execute_pooled_step_inplace,
    find_backbone,
    load_vllm_weights,
)
from vllm_torchtpu.megakernel.adapter.plugin import ForwardView, RunnerView
from vllm_torchtpu.megakernel.core import CacheExec, ExecSpec, Profile
from vllm_torchtpu.megakernel.core import runtime as mk_rt
from vllm_torchtpu.megakernel.core.runtime.contract import (
    Bucket,
    Decline,
    EligibilityResult,
    LayoutDecline,
    RunnerShape,
    StepDecision,
    WarmupCase,
)

from .model import Qwen3VLText, register

logger = init_logger(__name__)

OPTION_KEYS = frozenset({"profile", "buckets"})
KV_LAYOUT_DECLINE = "kv_cache_layout"
KV_LAYOUT_HINT = (
    "serve with VLLM_KV_CACHE_LAYOUT=HND --attention-backend CUSTOM "
    "--block-size 128 and a bf16 KV cache"
)


def _unwrap(x: Any) -> Any:
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def _kv_decline(msg: str) -> LayoutDecline:
    return LayoutDecline(
        EligibilityResult(
            False, KV_LAYOUT_DECLINE, f"Megakernel: {msg}; {KV_LAYOUT_HINT}; skipping."
        )
    )


def create(view: RunnerView, options: Mapping[str, Any]) -> Qwen3VLProgram | Decline:
    """Check the checkpoint and config; build an unloaded program, or decline.

    Runs no collective and loads no weights. The chip, TP size, parallelism,
    features and required server settings are the adapter's checks (`SPEC`).
    """
    if bad := sorted(options.keys() - OPTION_KEYS):
        raise ValueError(
            f"Qwen3VLProgram: unknown option {bad[0]!r}; known: {sorted(OPTION_KEYS)}"
        )
    register()
    cfg = mk_rt.parse_runtime_config(options)
    res = Qwen3VLText.validate_hf_config(view.hf_config)
    if not res.eligible:
        return Decline(res.code, res.message, res.level)
    try:
        backbone = find_backbone(view.model)
    except ValueError as exc:
        return Decline("backbone_not_found", f"Megakernel: {exc}; skipping.")
    n_layers = Qwen3VLText.derive_geometry().L
    if len(backbone.layers) != n_layers:
        return Decline(
            "layer_count_mismatch",
            f"Megakernel: the model has {len(backbone.layers)} decoder layers, "
            f"the geometry {n_layers}; skipping.",
        )
    profile = mk_rt.resolve_driver_profile(cfg["profile"], model_name=Qwen3VLText.name)
    policy = mk_rt.BucketPolicy.resolve(
        profile, mk_rt.explicit_profile_settings(cfg)["buckets"], ()
    )
    el = mk_rt.check_post_mesh_eligibility(tp_size=view.tp_size, bucket_policy=policy)
    if not el.eligible:
        return Decline(el.code, el.message, el.level)
    return Qwen3VLProgram(
        view.model, tp_rank=view.tp_rank, profile=profile, buckets=policy.buckets
    )


class Qwen3VLProgram:
    """The Qwen3-VL text decoder megakernel on one TP rank (a `MonoProgram`).

    `geom` / `base_exec` replace the checkpoint geometry and the derived
    `ExecSpec` (tiny CPU tests).
    """

    def __init__(
        self,
        model: Any,
        *,
        tp_rank: int,
        profile: Profile,
        buckets: Sequence[int],
        geom: Any = None,
        base_exec: ExecSpec | None = None,
    ) -> None:
        self.model_def = Qwen3VLText
        self.backbone = find_backbone(model)
        self.tp_rank = tp_rank
        self.profile = profile
        self.decode_buckets = tuple(buckets)
        self.geom = Qwen3VLText.derive_geometry() if geom is None else geom
        self._base_exec_arg = base_exec
        self.display_name = Qwen3VLText.display_name
        self.bucket_policy = mk_rt.BucketPolicy(self.decode_buckets, ())
        self.view: RunnerView | None = None
        self.raw_pools: list[torch.Tensor] = []
        self._ir_checked: set[tuple[int, int]] = set()
        self._ir_error: str | None = None
        self._writes_checked = False

    # --- load ---------------------------------------------------------------

    def load(self, mesh: Any) -> None:
        """Build the spec and pack the bf16 weights (after the adapter's
        startup agreement)."""
        self.mesh = mesh
        r_cnt = mesh.devices.size
        self.spec = self.model_def.to_model_spec(self.geom, R=r_cnt)
        b_max = max(self.decode_buckets)
        # The packed weight tiles are the buckets' (one tile rule for all).
        tiles = self.profile.tiles_for(b_max, R=r_cnt, g=self.geom)
        self.base_exec = self._base_exec_arg or ExecSpec(
            R=r_cnt, B=b_max, T=1, tiles=tiles, cache=CacheExec(paged_kv=True)
        )
        t0 = time.perf_counter()
        self.weights = load_vllm_weights(
            self.model_def,
            self.backbone,
            self.base_exec,
            g=self.geom,
            tp_rank=self.tp_rank,
        )
        self.w_flat: list[torch.Tensor] = self.weights["w_flat"]
        logger.info(
            "Packed %s megakernel weights (%d flat tensors) in %.2fs",
            self.display_name,
            len(self.w_flat),
            time.perf_counter() - t0,
        )

    # --- bind ---------------------------------------------------------------

    def bind(self, view: RunnerView) -> None:
        """Alias every layer's KV cache as one bf16 pool of the kernel's pages,
        and read vLLM's rotary cos / sin cache."""
        lay = self.spec.cache_backend.layout
        page = tuple(lay.kv_page_shape)
        pools = []
        for i, layer in enumerate(self.backbone.layers):
            raw = _unwrap(layer.self_attn.attn.kv_cache)
            if isinstance(raw, torch.Tensor) and raw.ndim == 6 and raw.shape[0] == 1:
                raw = raw[0]
            if not (
                isinstance(raw, torch.Tensor)
                and raw.dtype == torch.bfloat16
                and tuple(raw.shape[1:]) == page
                and raw.is_contiguous()
                and raw.storage_offset() == 0
            ):
                desc = (
                    f"{tuple(raw.shape)} {raw.dtype} offset {raw.storage_offset()}"
                    if isinstance(raw, torch.Tensor)
                    else type(raw).__name__
                )
                raise _kv_decline(
                    f"layer {i} KV cache is {desc}, not its own contiguous "
                    f"(pages, {', '.join(map(str, page))}) bfloat16 pages"
                )
            pools.append(raw)
        num_pages = pools[0].shape[0]
        if {p.shape[0] for p in pools} != {num_pages} or len(
            {p.data_ptr() for p in pools}
        ) != len(pools):
            raise _kv_decline("the layers' KV caches differ in size or share storage")
        rope = self.backbone.layers[0].self_attn.rotary_emb
        if not (rope.is_neox_style and rope.cos_sin_cache.shape[-1] == self.geom.ROT):
            raise LayoutDecline(
                EligibilityResult(
                    False,
                    "rope_unsupported",
                    "Megakernel: the rotary embedding is not a full neox-style "
                    f"rotation of {self.geom.ROT} dims; skipping.",
                )
            )
        self.view = view
        self.raw_pools = pools
        self.num_pages = num_pages
        self.max_pages_per_seq = view.max_pages_per_seq
        self.rope_cache = rope.cos_sin_cache
        self.attn_layer_name = self.backbone.layers[0].self_attn.attn.layer_name
        self.base_exec = mk_rt.align_cache_exec(
            self.base_exec,
            self.profile,
            self.geom,
            num_pages=num_pages,
            page_size=page[-1],
            max_pages_per_seq=self.max_pages_per_seq,
        )

    # --- buckets and build ----------------------------------------------------

    def buckets(self, token_paddings: Sequence[int]) -> list[Bucket]:
        """One decode `(1, B)` bucket per configured batch size."""
        self.bucket_policy = mk_rt.BucketPolicy(
            self.decode_buckets, (), tuple(token_paddings)
        )
        self._ir_checked.clear()
        return [
            Bucket(1, b, self.bucket_policy.out_rows(b)) for b in self.decode_buckets
        ]

    def build(self, bucket: Bucket) -> Any:
        """Check pool donation in the step's HLO; return the bridge step."""
        b = bucket.B
        ex, _, _ = mk_rt.bucket_exec_spec(
            self.spec, self.base_exec, self.profile, bucket_t=b, tokens_per_seq=1
        )
        ok, why = mk_rt.verify_pooled_step_hlo(
            self.mesh,
            self.spec,
            ex,
            self.w_flat,
            self.raw_pools,
            max_pages_per_seq=self.max_pages_per_seq,
        )
        if not ok:
            raise RuntimeError(
                f"{self.display_name} megakernel HLO pool-donation check failed "
                f"for B={b}: {why}"
            )
        op, _, _, _ = build_pooled_step_op(
            self.mesh,
            self.spec,
            ex,
            op_name=f"megakernel::{self.spec.name}_t1_b{b}",
            donate_pools=True,
        )
        embed, rope = self.backbone.embed_tokens, self.rope_cache
        m_rows, mr, half = ex.M, ex.MR, self.geom.ROT // 2
        max_pages = self.max_pages_per_seq
        out_rows = bucket.out_rows if bucket.out_rows is not None else m_rows

        def _step_impl(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            block_tables: torch.Tensor,
            seq_lens: torch.Tensor,
            weights: list[torch.Tensor],
            pools: list[torch.Tensor],
        ) -> torch.Tensor:
            # The KV index of the new token; RoPE takes its M-RoPE position.
            pos_b = (seq_lens[:b] - 1).clamp(min=0).to(torch.int32)
            rope_pos = (positions[0] if positions.ndim == 2 else positions)[:b]
            pt = block_tables.reshape(seq_lens.shape[0], -1)[:b, :max_pages]
            pt = F.pad(pt, (0, max_pages - pt.shape[1]))
            # vLLM rotates with the cache in the activation dtype (bf16).
            cs = rope[rope_pos].to(torch.bfloat16).float()
            cs = cs.reshape(b, 2, half).transpose(0, 1)
            rope_cs = F.pad(cs, (0, 0, 0, mr - b))
            hid = F.pad(embed(input_ids[:b]).to(torch.bfloat16), (0, 0, 0, m_rows - b))
            out = execute_pooled_step_inplace(
                op, pos_b, hid, pt, [rope_cs], weights, pools
            )
            return F.pad(out[:out_rows], (0, 0, 0, max(0, out_rows - m_rows)))

        return _step_impl

    def check_compiled_ir(self, debug: Any, bucket: Bucket) -> None:
        """The compiled step donates and aliases every KV pool."""
        texts = getattr(debug, "stablehlo_forward_text", None) or ()
        lay = self.spec.cache_backend.layout
        ok, why = mk_rt.verify_compiled_step_ir(
            "\n".join(str(t) for t in texts),
            num_pages=self.num_pages,
            page_size=lay.page_size,
            kvh=lay.kvh,
            hd=lay.hd,
            num_pools=len(self.raw_pools),
            page_dtype=lay.page_dtype,
        )
        if not ok:
            self._ir_error = (
                f"{self.display_name} megakernel compiled IR pool-donation check "
                f"failed for B={bucket.B}: {why}"
            )
            raise RuntimeError(self._ir_error)
        self._ir_checked.add(bucket.key)

    # --- warmup -----------------------------------------------------------------

    @contextlib.contextmanager
    def warmup_session(self) -> Iterator[None]:
        """On CPU, restore the page warmup writes; on TPU, require the compiled
        IR check of every bucket afterwards."""
        dev = self.raw_pools[0].device
        saved = [p[1].clone() for p in self.raw_pools] if dev.type != "tpu" else None
        try:
            yield
            if dev.type == "tpu" and not self.view.enforce_eager:
                buckets = self.bucket_policy.iter_buckets()
                missing = [k for k in buckets if k not in self._ir_checked]
                if self._ir_error is not None or missing:
                    raise RuntimeError(
                        self._ir_error
                        or f"{self.display_name} megakernel compiled IR check did "
                        f"not run for buckets {missing}"
                    )
        finally:
            if saved is not None:
                for p, s in zip(self.raw_pools, saved, strict=True):
                    p[1].copy_(s)

    def warmup_inputs(self, bucket: Bucket, shape: RunnerShape) -> WarmupCase:
        """One decode step at KV slot 0 of page 1 (free during warmup). The
        output must be finite, and the first call must change that page in
        every layer's KV cache (the pools alias the runner's)."""
        dev = self.raw_pools[0].device
        i32 = dict(dtype=torch.int32, device=dev)
        ids = torch.zeros((shape.tok_pad,), **i32)
        pos = torch.zeros(
            (3, shape.tok_pad) if self.view.uses_mrope else (shape.tok_pad,), **i32
        )
        bt = torch.zeros((shape.req_pad, shape.blk_pad), **i32)
        bt[0, 0] = 1
        seq_lens = torch.zeros((shape.req_pad,), **i32)
        seq_lens[: bucket.B] = 1
        check_writes, self._writes_checked = not self._writes_checked, True
        before = [
            p[1].view(torch.uint16).cpu().clone() if check_writes else None
            for p in self.raw_pools
        ]

        def check(out: torch.Tensor) -> None:
            if not bool(torch.isfinite(out[: bucket.B].float()).all()):
                raise RuntimeError(f"{self.display_name} megakernel warmup: non-finite")
            if not check_writes:
                return
            for i, (p, b0) in enumerate(zip(self.raw_pools, before, strict=True)):
                if torch.equal(p[1].view(torch.uint16).cpu(), b0):
                    raise RuntimeError(
                        f"{self.display_name} megakernel warmup wrote no K/V into "
                        f"layer {i}'s KV cache (pool aliasing broken)"
                    )

        return WarmupCase(
            (ids, pos, bt.reshape(-1), seq_lens, self.w_flat, self.raw_pools), check
        )

    # --- per step -------------------------------------------------------------

    def decide(self, ctx: Any, fwd: ForwardView) -> StepDecision | None:
        """A decode step of `B` requests, all past their prompt, on bucket `B`
        (`None`: no attention metadata, stock without counting)."""
        md = fwd.attn_metadata
        attn_md = md.get(self.attn_layer_name) if md else None
        if attn_md is None:
            return None
        bs = fwd.batch_shape(attn_md, None)
        d = mk_rt.check_step_eligibility(
            bs, self.bucket_policy, fwd.steps, input_rows=ctx.input_ids.shape[0]
        )
        if not d.eligible:
            return d
        # A 1-token final prefill chunk looks like decode, but its M-RoPE
        # rows may differ (vision tokens); the stock model serves it.
        if fwd.prompt_left is not False:
            return StepDecision(False, "stock_prompt", d.num_reqs, 1)
        # Padded rows would scatter K/V at stale positions.
        if d.bucket_t != d.num_reqs:
            return StepDecision(False, f"stock_t1_pad{d.num_reqs}", d.num_reqs, 1)
        return d

    def forward(
        self, decision: StepDecision | None, ctx: Any, fwd: ForwardView
    ) -> torch.Tensor:
        """Run the decided decode step; returns the rows the stock model would."""
        assert decision is not None and fwd.attn_metadata is not None
        attn_md = fwd.attn_metadata[self.attn_layer_name]
        step_fn = fwd.steps[(1, decision.bucket_t)]
        out = step_fn(
            ctx.input_ids,
            ctx.positions,
            attn_md.block_tables,
            attn_md.seq_lens,
            self.w_flat,
            self.raw_pools,
        )
        n = fwd.num_tokens if fwd.num_tokens is not None else ctx.input_ids.shape[0]
        return out[:n] if out.shape[0] >= n else F.pad(out, (0, 0, 0, n - out.shape[0]))


__all__ = ["OPTION_KEYS", "Qwen3VLProgram", "create"]
