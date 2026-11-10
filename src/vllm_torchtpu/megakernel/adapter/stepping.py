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
"""The per-step half of the megakernel plugin driver: step inspection, the
routing decision, `try_forward` with the A/B latency calibrator and the
step-mix log; plus the runner input shapes a bucket is warmed up with."""

from __future__ import annotations

import functools
import time
from collections.abc import Sequence
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.adapter.guards import _aux_hidden_layers
from vllm_torchtpu.megakernel.adapter.hooks import StepContext, is_suppressed
from vllm_torchtpu.megakernel.adapter.plugin import ForwardView
from vllm_torchtpu.megakernel.core.runtime import stats as _stats
from vllm_torchtpu.megakernel.core.runtime.contract import Bucket, RunnerShape, StepFn

logger = init_logger(__name__)


def get_forward_context() -> Any:
    """vLLM's forward context for the current step. vLLM is imported here, on
    first use, as in the sibling modules."""
    from vllm.forward_context import get_forward_context as current

    return current()


def _inspect_step_detail(
    attn_md: Any, m0: Any, runner: Any = None
) -> _stats.BatchShape:
    """Return `BatchShape` without host-device synchronization."""
    num_reqs: int | None = None
    has_roff = False
    comb_wt: tuple[int, int] | None = None
    qsl_host: Any = None
    if runner is not None:
        bctx = getattr(runner, "_attn_metadata_builder_ctx", None)
        if bctx is not None and getattr(bctx, "num_reqs", None) is not None:
            num_reqs = int(bctx.num_reqs)
        has_roff = getattr(runner, "mamba_slot_read_offsets", None) is not None
        comb_key = getattr(runner, "_cached_comb_key", None)
        comb_cpu: Any = getattr(runner, "_combined_request_distribution_cpu", None)
        if has_roff and isinstance(comb_key, tuple) and len(comb_key) == 3:
            comb_wt = (int(comb_key[1]), int(comb_key[2]))
        elif has_roff and isinstance(comb_cpu, torch.Tensor) and comb_cpu.numel() >= 6:
            comb_wt = (int(comb_cpu[3].item()), int(comb_cpu[5].item()))
        qsl_host = getattr(runner, "query_start_loc_np", None)
        if qsl_host is None:
            qsl_host = getattr(runner, "query_start_loc_cpu", None)
        runner_decides = qsl_host is not None and (
            num_reqs is not None or (has_roff and comb_wt is not None)
        )
        if not runner_decides and has_roff and comb_wt is not None:
            eff_n = comb_wt[1] if num_reqs is None else num_reqs
            runner_decides = comb_wt[1] <= 0 or eff_n <= 0 or comb_wt[0] != comb_wt[1]
        if runner_decides:
            return _stats.inspect_batch_shape(
                num_reqs=num_reqs,
                has_slot_read_offsets=has_roff,
                decode_and_total_reqs=comb_wt,
                query_start_loc=qsl_host,
            )

    # Fallback for unit tests passing CPU metadata without a full runner instance.
    # Never read device tensors on TPU; fail closed if no host query_start_loc.
    dist_dt: tuple[int, int] | None = None
    dist: Any = getattr(m0, "mamba_request_distribution", None)
    if dist is None:
        dist = getattr(attn_md, "mamba_request_distribution", None)
    if dist is None:
        dist = getattr(attn_md, "request_distribution", None)
    if (
        isinstance(dist, torch.Tensor)
        and dist.device.type == "cpu"
        and dist.numel() >= 3
    ):
        dist_dt = (int(dist[0].item()), int(dist[2].item()))

    fb_nseq: int | None = None
    ns_attr = getattr(attn_md, "__dict__", {}).get("num_seqs")
    if ns_attr is not None:
        if isinstance(ns_attr, torch.Tensor):
            fb_nseq = int(ns_attr.item()) if ns_attr.device.type == "cpu" else None
        else:
            fb_nseq = int(ns_attr)

    qsl_md = getattr(attn_md, "__dict__", {}).get("query_start_loc_cpu")
    if qsl_md is None:
        qsl_dev: Any = getattr(attn_md, "query_start_loc", None)
        if isinstance(qsl_dev, torch.Tensor) and qsl_dev.device.type == "cpu":
            qsl_md = qsl_dev

    return _stats.inspect_batch_shape(
        num_reqs=num_reqs,
        has_slot_read_offsets=has_roff,
        decode_and_total_reqs=comb_wt,
        query_start_loc=qsl_host,
        fallback_decode_and_total_reqs=dist_dt,
        fallback_num_seqs=fb_nseq,
        fallback_query_start_loc=qsl_md,
    )


def _default_out_rows(b: int, token_paddings: Sequence[int]) -> int:
    return next(
        (int(p) for p in token_paddings if int(p) >= b), max(16, -(-b // 16) * 16)
    )


def runner_shapes(
    bucket: Bucket,
    token_paddings: Sequence[int],
    req_block_pairs: Sequence[tuple[int, int]],
    *,
    ckpt_window: int,
    max_pages_per_seq: int,
) -> list[RunnerShape]:
    """The runner input shapes a bucket is warmed up with (one graph each)."""
    T, B = int(bucket.T), int(bucket.B)
    cols = max(T, max(1, int(ckpt_window)))
    if token_paddings and req_block_pairs:
        pads = [int(p) for p in token_paddings if int(p) >= B]
        tok = pads[0] if pads else int(token_paddings[-1])
        return [RunnerShape(tok, int(r), int(k), cols) for r, k in req_block_pairs]
    return [RunnerShape(B, B // T, int(max_pages_per_seq), cols)]


def _log_step_mix(name: str, summary: Any) -> None:
    elapsed, mk_cnt, total, pct, snap = summary
    logger.info(
        "%s megakernel step mix over %.0fs: %d/%d MK (%.1f%%) %s",
        name,
        elapsed,
        mk_cnt,
        total,
        pct,
        snap,
    )


class StepRouting:
    """Routes each runner step to the program or to the stock model, runs
    it, keeps the step-mix stats and times the A/B calibrator's arms.

    A mixin: the driver class that uses it sets the state these methods read,
    declared below (`_steps` and `_bound` once the program is bound; `_pending`,
    `_inspect` and `_inspect_runner` start as `None`).
    """

    program: Any
    tp_rank: int
    _display_name: str
    _backbone: Any
    _steps: dict[tuple[int, int], StepFn]
    _bound: bool
    _pending: tuple[Any, ForwardView, StepContext] | None
    _inspect: Any
    _inspect_runner: Any
    step_mix_tracker: _stats.StepMixTracker
    ab_calibrator: _stats.MKABCalibrator

    @classmethod
    def _inspect_step_detail(
        cls, attn_md: Any, m0: Any, runner: Any = None
    ) -> _stats.BatchShape:
        """Return `BatchShape` without host-device synchronization."""
        return _inspect_step_detail(attn_md, m0, runner)

    def _forward_view(self, runner: Any, num_tokens: int | None = None) -> ForwardView:
        md = getattr(get_forward_context(), "attn_metadata", None)
        if runner is not self._inspect_runner or self._inspect is None:
            self._inspect = functools.partial(_inspect_step_detail, runner=runner)
            self._inspect_runner = runner
        return ForwardView(
            md if isinstance(md, dict) else None,
            getattr(getattr(runner, "mrope_positions", None), "np", None),
            self._steps,
            num_tokens,
            self._inspect,
        )

    def can_run(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        runner: Any = None,
    ) -> bool:
        """Return `True` if the current forward step should run on the megakernel."""
        ctx = StepContext(input_ids, positions, inputs_embeds, intermediate_tensors)
        return self._route(ctx, runner)

    def _route(self, ctx: StepContext, runner: Any) -> bool:
        self._pending = None
        self.ab_calibrator.last_decision = ("stock_early", "stock")
        if (
            not self._bound
            or ctx.input_ids is None
            or ctx.inputs_embeds is not None
            or ctx.intermediate_tensors is not None
            or _aux_hidden_layers(self._backbone)
        ):
            return False
        bctx = getattr(runner, "_attn_metadata_builder_ctx", None)
        if getattr(bctx, "position_ids_override", None) is not None:
            return False
        fwd = self._forward_view(runner)
        decision = self.program.decide(ctx, fwd)
        if decision is None:
            return False
        if not decision.eligible:
            return self._note_step(decision.reason, False)
        run = self._note_step(decision.reason, True)
        # The A/B calibrator may send an eligible step to the stock arm.
        self._pending = (decision, fwd, ctx) if run else None
        return run

    def _note_step(self, key: str, run: bool) -> bool:
        actual_run, summary = self.step_mix_tracker.note_step(
            key, run, self.ab_calibrator
        )
        if summary is not None and self.tp_rank == 0:
            _log_step_mix(self._display_name, summary)
        return actual_run

    def record_ab_step(self, dt_ms: float, padded_tokens: int | None = None) -> None:
        res = self.ab_calibrator.record(dt_ms, padded_tokens=padded_tokens)
        if res is not None and self.tp_rank == 0:
            elapsed, win_json, cum_json = res
            logger.info(
                "%s megakernel AB latency (window %.0fs): %s",
                self._display_name,
                elapsed,
                win_json,
            )
            logger.info(
                "%s megakernel AB latency (cumulative): %s",
                self._display_name,
                cum_json,
            )

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int | None = None,
        runner: Any = None,
    ) -> Any:
        """Run the step `can_run` decided (or decide now) on the program."""
        pending, self._pending = self._pending, None
        n_tok = int(num_tokens if num_tokens is not None else input_ids.shape[0])
        if pending is None:
            decision, fwd = None, self._forward_view(runner, n_tok)
            ctx = StepContext(input_ids, positions)
        else:
            decision, fwd, ctx = pending
            fwd.num_tokens = n_tok
            if ctx.input_ids is not input_ids or ctx.positions is not positions:
                ctx = StepContext(input_ids, positions)
        return self.program.forward(decision, ctx, fwd)

    def try_forward(self, runner: Any, ctx: StepContext) -> tuple[bool, Any]:
        """Try to execute the current forward step on the megakernel.

        Returns `(True, out)` when handled by the megakernel (or timed by the
        A/B calibrator), where `out` is what the stock model would return; or
        `(False, None)` to fall through to the runner's stock model forward.
        Always `(False, None)` inside `megakernel.adapter.suppressed(runner)`
        (dummy / precompile / profile runs), so those runs compile the stock
        graphs, and when `ctx.model_kwargs` is non-empty: the program doesn't
        consume extra model inputs. The runner's stock forward passes none
        today, so this only guards a runner that starts to.
        """
        if is_suppressed(runner) or ctx.model_kwargs:
            self._pending = None
            return False, None
        input_ids, positions = ctx.input_ids, ctx.positions
        inputs_embeds = ctx.inputs_embeds
        intermediate_tensors = ctx.intermediate_tensors
        use_mk = self._route(ctx, runner)
        ab_t0 = time.perf_counter() if self.ab_calibrator.enabled else None
        if use_mk:
            assert input_ids is not None and positions is not None
            num_tok = int(positions.shape[-1])
            out = self(input_ids, positions, num_tokens=num_tok, runner=runner)
            if ab_t0 is not None:
                if getattr(getattr(runner, "device", None), "type", None) == "tpu":
                    from vllm_torchtpu.utils import synchronize_tensors

                    synchronize_tensors([out])
                self.record_ab_step(
                    (time.perf_counter() - ab_t0) * 1000.0, padded_tokens=num_tok
                )
            return True, out

        if (
            ab_t0 is not None
            and self.ab_calibrator.last_decision[0] != "stock_early"
            and positions is not None
            and intermediate_tensors is None
            and callable(getattr(runner, "model", None))
            and not _aux_hidden_layers(self._backbone)
        ):
            out = runner.model(
                input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds
            )
            if isinstance(out, (list, tuple)) and len(out) == 1:
                out = out[0]
            if getattr(
                getattr(runner, "device", None), "type", None
            ) == "tpu" and isinstance(out, torch.Tensor):
                from vllm_torchtpu.utils import synchronize_tensors

                synchronize_tensors([out])
            self.record_ab_step(
                (time.perf_counter() - ab_t0) * 1000.0,
                padded_tokens=int(positions.shape[-1]),
            )
            return True, out

        return False, None


__all__ = ["StepRouting", "runner_shapes"]
