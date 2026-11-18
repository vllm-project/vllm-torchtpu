# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The opt-in device-decode path (VLLM_TPU_DEVICE_DECODE): its startup, and
the stock steps' write-back into the decode slots.

With the flag set, every step still runs on the stock path. Startup builds
the decode slots, compiles the path's programs and, on TPU, runs the forward
of a device step once per decode shape. Each stock chunk then draws its
sampling noise from the slots' keyed RNG, as the head keys it, and writes
its outcome back into the slots, which follow the runner's requests on
every step.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.forward_context import set_forward_context

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.sequence_layout import (
    AllSequenceLayoutPlanner,
    SequenceLayoutPlan,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.runner.device_decode.head import SamplerVariant, precompile_head
from vllm_torchtpu.runner.device_decode.inputs import (
    DecodeInputs,
    DecodeShape,
    WalkGeometry,
    build_decode_attention_metadata,
    precompile_prepare,
    prepare_decode_inputs,
)
from vllm_torchtpu.runner.device_decode.rng import counter_uniform_for_rows_impl
from vllm_torchtpu.runner.device_decode.slots import (
    DecodeSlots,
    SlotConfig,
    StockChunkPlan,
)
from vllm_torchtpu.tracing.annotation import TraceAnnotation
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


def stock_uniform_key(num_rows_padded: int, vocab_size: int) -> tuple:
    return ("stock_uniform", num_rows_padded, vocab_size)


def stock_uniform_impl(
    seed_lo: torch.Tensor,
    seed_hi: torch.Tensor,
    plan: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """The plain body of `stock_uniform`."""
    return counter_uniform_for_rows_impl(
        seed_lo, seed_hi, plan[:, 0], plan[:, 1], vocab_size
    )


# The plan enters whole and is sliced inside the program, as the write-back
# slices it: its columns passed in would be strided views.
_stock_uniform_compiled = torch.compile(
    stock_uniform_impl, backend="tpu", fullgraph=True, dynamic=False
)


def stock_uniform(
    seed_lo: torch.Tensor,
    seed_hi: torch.Tensor,
    plan: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """float32 [P, vocab_size] sampling noise for the rows of a stock
    chunk's plan (`StockChunkPlan.to_device`: int32 [P, 4] whose first two
    columns are the slot and the position), each row keyed by its slot's seed
    and its position, as the head keys it."""
    fn = _stock_uniform_compiled if plan.device.type == "tpu" else stock_uniform_impl
    with torch.no_grad():
        return fn(seed_lo, seed_hi, plan, vocab_size)


@dataclass(frozen=True)
class StockChunk:
    """A stock-sampled chunk's slot write-back plan, uploaded once for both
    its sampling noise and its write-back."""

    plan: StockChunkPlan
    # int32 [P, 4]: `plan.to_device`.
    plan_dev: torch.Tensor
    slots: DecodeSlots

    def uniform(self, vocab_size: int) -> torch.Tensor:
        """float32 [P, vocab_size] sampling noise, each row keyed by its
        request's seed and the position it samples at, as the head keys it."""
        return stock_uniform(
            self.slots.seed_lo, self.slots.seed_hi, self.plan_dev, vocab_size
        )


class DeviceDecodeController:
    """Owns the decode slots: builds them at startup, follows the runner's
    requests with them on every step (`on_states_updated`) and writes each
    stock chunk's outcome back into them (`stock_chunk_plan`,
    `after_stock_sampling`). All of it is a function of the scheduler output
    and the host mirrors, so all TP ranks keep the same slots."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.disabled_reason: str | None = None
        self.slots: DecodeSlots | None = None
        self.geometry: WalkGeometry | None = None
        # Every program compiled at startup.
        self.registry: set[tuple] = set()
        self._ones = np.ones(0, dtype=np.int32)

    @classmethod
    def maybe_create(cls, runner: Any) -> DeviceDecodeController | None:
        """A controller for `runner` when VLLM_TPU_DEVICE_DECODE is set."""
        return cls(runner) if envs.VLLM_TPU_DEVICE_DECODE else None

    def static_gate_failures(self) -> list[str]:
        """The configuration features the device path does not support."""
        r = self.runner
        parallel = r.parallel_config
        checks = (
            ("pooling", r.is_pooling_model),
            ("speculative decoding", r.speculative_config is not None),
            ("pipeline parallelism", parallel.pipeline_parallel_size > 1),
            (
                "prefill context parallelism",
                r.pcp_manager is not None
                or not isinstance(r.sequence_layout_planner, AllSequenceLayoutPlanner),
            ),
            ("decode context parallelism", parallel.decode_context_parallel_size > 1),
            ("M-RoPE", r.uses_mrope),
            ("multimodal", r.supports_mm_inputs),
            ("LoRA", r.lora_config is not None),
            ("KV transfer", r.vllm_config.kv_transfer_config is not None),
            ("DP lockstep", r._dp_lockstep_enabled()),
            ("VLLM_TPU_MOST_MODEL_LEN", r.most_model_len is not None),
            ("TPU_GREEDY_TOP_TOKENS", r._greedy_head_owner is not None),
            ("TPU_DP_VOCAB_PARALLEL", r._dp_vocab_logits),
            ("enforce_eager", r.model_config.enforce_eager),
        )
        return [name for name, unsupported in checks if unsupported]

    def precompile_heads(self) -> None:
        """Compiles the head and the stock-path noise programs. Runs from
        `_precompile_sampling_subgraphs`, as the stock sampler programs
        compile: in the profile run, when there is one, the memory probe sees
        their HBM before the KV cache is sized."""
        if self.static_gate_failures():
            return
        r = self.runner
        variants = SamplerVariant.all_variants(r)
        self.registry.update(precompile_head(r, variants, r.num_reqs_paddings))
        self.registry.update(self._precompile_stock_uniform())

    def initialize_and_precompile(self) -> None:
        """Builds the slots and compiles the rest of the device-decode
        programs at the end of `capture_model`, or records why the path stays
        off."""
        failures = self.static_gate_failures()
        if failures:
            self.disabled_reason = "unsupported configuration: " + ", ".join(failures)
            logger.info(
                "VLLM_TPU_DEVICE_DECODE is set, but every step runs on the "
                "stock decode path: %s.",
                self.disabled_reason,
            )
            return
        r = self.runner
        self.slots = DecodeSlots(SlotConfig.from_runner(r), r.device)
        self.geometry = WalkGeometry.from_runner(r)
        self._ones = np.ones(r.max_num_reqs, dtype=np.int32)
        shapes = DecodeShape.all_decode_shapes(r)
        if not any(key[0] == "decode_head" for key in self.registry):
            self.precompile_heads()
        with r._precompile_timed("device_decode_prepare"):
            self.registry.update(
                precompile_prepare(r, self.slots, self.geometry, shapes)
            )
        with r._precompile_timed("device_decode_slot_writes"):
            self.registry.update(self.slots.precompile(r))
        if r.device.type == "tpu":
            self._dry_steps(shapes)

    def on_states_updated(self, scheduler_output: Any) -> None:
        """Follows the runner's requests after `_update_states`, on every
        step, also those without tokens: frees and allocates slots, then
        writes the slot rows that changed."""
        if self.slots is None:
            return
        self.slots.sync_from_runner(self.runner, scheduler_output)
        self.slots.apply_staged_writes(self.runner)

    def stock_chunk_plan(
        self,
        scheduler_output: Any,
        start: int,
        end: int,
        num_rows_padded: int,
    ) -> StockChunk:
        """The slot write-back of the stock chunk of batch rows [start, end),
        whose logits have `num_rows_padded` rows. Build it before the chunk is
        sampled: its rows key the chunk's sampling noise."""
        assert self.slots is not None
        r = self.runner
        plan = self.slots.plan_stock_chunk(
            r, scheduler_output, start, end, num_rows_padded
        )
        return StockChunk(plan, plan.to_device(r), self.slots)

    def after_stock_sampling(
        self, chunks: list[tuple[StockChunk, torch.Tensor]]
    ) -> None:
        """Writes a stock step's sampled chunks into the slots."""
        assert self.slots is not None
        for chunk, sampled in chunks:
            self.slots.write_back_stock_chunk(
                self.runner, chunk.plan, sampled, chunk.plan_dev
            )

    def _prefill_ckpt_offsets(
        self, num_reqs: int, target_rows: int
    ) -> torch.Tensor | None:
        """The GDN prefill checkpoint offsets of a decode step, as the stock
        `_prepare_inputs` derives and uploads them (zeros; reused while
        unchanged)."""
        r = self.runner
        if r._prefill_ckpt_units is None:
            return None
        offsets = r._prefill_ckpt_offsets_cpu
        r._prefill_ckpt_units.fill_offsets(
            offsets.numpy(), r.seq_lens_np[:num_reqs], self._ones[:num_reqs]
        )
        return r._upload_unless_unchanged("prefill_ckpt_offsets", offsets[:target_rows])

    def _layout_plan(
        self, scheduler_output: Any, shape: DecodeShape
    ) -> SequenceLayoutPlan:
        num_reqs = shape.num_reqs
        plan = self.runner.sequence_layout_planner.prepare_real(
            runner=self.runner,
            scheduler_output=scheduler_output,
            start_index=0,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=self._ones[:num_reqs],
            total_num_scheduled_tokens=num_reqs,
            use_max_model_len=True,
            target_num_reqs=shape.target_rows,
            padded_num_reqs=shape.num_reqs_padded,
        )
        assert plan.local_padded_num_tokens == shape.num_tokens_padded, (plan, shape)
        return plan

    def _forward(
        self,
        attn_metadata: Any,
        inputs: DecodeInputs,
        shape: DecodeShape,
        stage: contextlib.AbstractContextManager,
    ) -> torch.Tensor:
        """The stock backbone and logits programs on device-built inputs."""
        r = self.runner
        num_tokens_padded = shape.num_tokens_padded
        with (
            TraceAnnotation(
                name=f"DeviceDecodeForward: {shape.num_reqs} reqs, "
                f"{num_tokens_padded} toks",
                num_reqs=shape.num_reqs,
            ),
            set_forward_context(
                attn_metadata,
                r.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=None,
            ),
            set_vllm_model_wrapper_context(
                mesh=r.mesh,
                vllm_config=r.vllm_config,
                kv_cache_bundle=r._kv_cache_bundle,
            ),
            stage,
        ):
            hidden_states, _ = r.forward_model(
                input_ids=inputs.input_ids, positions=inputs.positions
            )
        return r.compute_selected_logits(hidden_states, inputs.logits_indices)

    def _dry_steps(self, shapes: list[DecodeShape]) -> None:
        """Runs a device step's forward and logits once per decode shape, so
        a Dynamo guard the device-built inputs miss compiles at warmup, not
        at the first device step. Every live row reads slot 0, which is free
        and at its defaults during warmup: one token at position 0 on the
        null block and state slot, as in the stock dummy runs."""
        r = self.runner
        assert self.slots is not None and self.geometry is not None
        config = self.slots.config
        idx_np = np.full(config.max_num_reqs, config.null_slot, dtype=np.int32)
        with r._precompile_timed("device_decode_dry_steps"), torch.no_grad():
            for shape in shapes:
                idx_np[: shape.num_reqs] = 0
                r._wait_input_staging_fence()
                idx_mapping = torch.from_numpy(idx_np.copy()).to(r.device)
                offsets = self._prefill_ckpt_offsets(0, shape.target_rows)
                inputs = prepare_decode_inputs(
                    self.slots, idx_mapping, shape, self.geometry, offsets
                )
                attn_metadata = build_decode_attention_metadata(
                    r, inputs, shape, offsets
                )
                r._last_sequence_layout_plan = self._layout_plan(None, shape)
                r._record_input_staging_fence()
                r._token_padding_update(shape.num_tokens_padded)
                logits = self._forward(
                    attn_metadata, inputs, shape, contextlib.nullcontext()
                )
                synchronize_tensors(logits)

    def _precompile_stock_uniform(self) -> list[tuple]:
        r = self.runner
        num_slots = r.max_num_reqs + 1
        keys = []
        with r._precompile_timed("device_decode_stock_uniform"), torch.no_grad():
            seed_lo = torch.zeros(num_slots, dtype=torch.int32).to(r.device)
            seed_hi = torch.zeros(num_slots, dtype=torch.int32).to(r.device)
            for num_rows in r.num_reqs_paddings:
                # Laid out as `StockChunkPlan.to_device`: a view of a flat upload.
                plan = (
                    torch.zeros(num_rows * 4, dtype=torch.int32)
                    .to(r.device)
                    .view(num_rows, 4)
                )
                uniform = stock_uniform(seed_lo, seed_hi, plan, r.vocab_size)
                synchronize_tensors(uniform)
                keys.append(stock_uniform_key(num_rows, r.vocab_size))
        return keys
