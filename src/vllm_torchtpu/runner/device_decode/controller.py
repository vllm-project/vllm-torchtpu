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
"""The opt-in device-decode path (VLLM_TPU_DEVICE_DECODE): which steps run on
the decode slots, the device step itself, and the stock steps' write-back.

A pure-decode step whose requests all own a decode-ready slot builds its
inputs on the device (`inputs`), runs the stock backbone and logits programs,
then samples, advances the slots and packs its outputs in one program (`head`)
that reaches the host in one copy. Any other step runs on the stock path,
draws its sampling noise from the same keyed RNG and writes its outcome back
into the slots, so a request samples the same token on either path.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.config import get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.sequence_layout import (
    AllSequenceLayoutPlanner,
    SequenceLayoutPlan,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.runner.device_decode.head import (
    PackedLayout,
    SamplerVariant,
    decode_head,
    head_key,
    precompile_head,
)
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
    MIRROR_MISMATCH,
    SLOT_NOT_READY,
    DecodeSlots,
    SlotConfig,
    StockChunkPlan,
    mrope_position_rows,
)
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncPreResults,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
    DeviceDecodeInvariantError,
)
from vllm_torchtpu.tracing.annotation import TraceAnnotation
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

# Why a step runs on the stock path, in the order they are checked; the
# slot checks (`slots.SLOT_NOT_READY`, `slots.MIRROR_MISMATCH`) come between
# ENCODER_OR_KV and MROPE_PROMPT_TOKEN.
NOT_PURE_DECODE = "not_pure_decode"
MULTI_CHUNK = "multi_chunk"
STRUCTURED_OUTPUT = "structured_output"
MIN_TOKENS = "min_tokens"
PROMPT_LOGPROBS = "prompt_logprobs"
UNSUPPORTED_SAMPLING = "unsupported_sampling"
ENCODER_OR_KV = "encoder_or_kv"
# An M-RoPE model's row whose token is a prompt token: the stock path reads
# its position from the request's precomputed table, which no slot holds.
MROPE_PROMPT_TOKEN = "mrope_prompt_token"
UNREGISTERED_SHAPE = "unregistered_shape"
# Last, on a verified step: its inputs built on the device disagree with the
# stock ones, so it runs on the stock path and device steps turn off.
VERIFY_INPUT_MISMATCH = "verify_input_mismatch"
# The other counters.
DEVICE_STEPS = "device_steps"
VERIFY_STEPS = "verify_steps"
VERIFY_MISMATCHES = "verify_mismatches"

# A request whose slot disagrees with the runner on this many checks in a
# row points at a slot bug, not at a rollback the write-back repairs. A
# check where every mirror agrees ends the row; a step that never reaches
# the check leaves it as it is.
_MAX_MIRROR_MISMATCHES = 3

# HBM that the programs `initialize_and_precompile` compiles after the KV
# cache is sized take: the prepare and slot-write programs, the int32 token
# substitution and what the dry steps compile. `hbm_reserve_bytes` adds the
# slot tensors, which it computes.
# On TPU v7x (Qwen3.5-0.8B, TP=1) flag on vs off: +5.6 MiB at max_num_seqs 32 and
# max_model_len 4352, +15.1 MiB at 256 and 16384; plus 16 MiB for noise, rounded up.
_PROGRAM_SCRATCH_BYTES = 32 << 20


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


@dataclass
class DeviceDecodeState:
    """The `execute_model_state` a device step leaves for `sample_tokens`."""

    scheduler_output: Any
    shape: DecodeShape
    variant: SamplerVariant
    idx_mapping: torch.Tensor
    idx_np: np.ndarray
    logits: torch.Tensor
    # int32 [num_reqs]: the input position of each request, before the step.
    expected_positions: np.ndarray
    verify: bool


class DeviceDecodeController:
    """Owns the decode slots and decides, step by step, whether a step runs
    on the device (`try_execute`, `sample_tokens`) or on the stock path,
    whose chunks it then writes back (`stock_chunk_plan`,
    `after_stock_sampling`). Every decision is a function of the scheduler
    output and the host mirrors, so all TP ranks take the same path.

    `enabled` allows device steps; the slots, once built, serve the stock
    steps for the rest of the run even after a check turns device steps
    off."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.enabled = False
        self.disabled_reason: str | None = None
        self.slots: DecodeSlots | None = None
        self.geometry: WalkGeometry | None = None
        # Every program compiled at startup. A step whose programs are not in
        # it runs on the stock path instead of compiling mid-serving.
        self.registry: set[tuple] = set()
        self._counters: Counter[str] = Counter()
        self._verify_every = envs.VLLM_TPU_DEVICE_DECODE_VERIFY
        # Live requests asking for logits processing the TPU samplers skip.
        self._logits_processor_reqs: set[str] = set()
        self._mirror_mismatches: tuple[str | None, int] = (None, 0)
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

    def hbm_reserve_bytes(self) -> int:
        """HBM that `initialize_and_precompile` allocates after the KV cache
        is sized, which `TPUModelRunner.post_profile_hbm_reserves` leaves
        free: the slot tensors and the program scratch; 0 when the
        configuration keeps every step on the stock path."""
        if self.static_gate_failures():
            return 0
        return self._reserve_slot_config().device_bytes + _PROGRAM_SCRATCH_BYTES

    def _reserve_slot_config(self) -> SlotConfig:
        """The slot config the reserve counts. The budget is computed before
        `initialize_kv_cache` makes the KV cache groups final, so it counts
        one block table per KV cache layer (every group holds at least one),
        each as wide as the runner's block tables are until then."""
        r = self.runner
        layers = get_layers_from_vllm_config(
            r.vllm_config,
            (AttentionLayerBase, MambaBase),  # type: ignore[type-abstract]
        )
        return SlotConfig(
            max_num_reqs=r.max_num_reqs,
            group_widths=(r.max_num_blocks_per_req,) * len(layers),
            vocab_size=r.vocab_size,
            model_seed=int(r.model_config.seed or 0),
            mrope_rows=mrope_position_rows(r),
        )

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
        if r.scheduler_config.async_scheduling:
            # A stock step after a device step substitutes the head's tokens.
            r._precompile_substitute_placeholder_token(torch.int32)
        if r.device.type == "tpu":
            self._dry_steps(shapes)
        self.enabled = True
        logger.info(
            "Device decode is on: %d programs compiled for %d decode shapes.",
            len(self.registry),
            len(shapes),
        )

    def on_states_updated(self, scheduler_output: Any) -> None:
        """Follows the runner's requests after `_update_states`, on every
        step, also those without tokens: frees and allocates slots, then
        writes the slot rows that changed."""
        if self.slots is None:
            return
        self._logits_processor_reqs.difference_update(scheduler_output.finished_req_ids)
        for new_req in scheduler_output.scheduled_new_reqs:
            params = new_req.sampling_params
            if params is not None and (params.min_p > 0.0 or params.logit_bias):
                self._logits_processor_reqs.add(new_req.req_id)
            else:
                self._logits_processor_reqs.discard(new_req.req_id)
        self.slots.sync_from_runner(self.runner, scheduler_output)
        self.slots.apply_staged_writes(self.runner)

    def try_execute(self, scheduler_output: Any, target_num_chunks: int) -> bool:
        """Dispatches the step's forward and logits from the slots and leaves
        a `DeviceDecodeState` for `sample_tokens`, or returns False to leave
        the step to the stock path."""
        if not self.enabled:
            return False
        plan = self._plan_step(scheduler_output, target_num_chunks)
        if isinstance(plan, str):
            self._count_stock_step(plan)
            return False
        shape, variant = plan
        r = self.runner
        assert self.slots is not None and self.geometry is not None
        num_reqs = shape.num_reqs
        idx_np = self.slots.idx_mapping_host(r.input_batch.req_ids[:num_reqs])
        expected_positions = r.input_batch.num_computed_tokens_cpu[:num_reqs].astype(
            np.int32
        )
        verify = (
            self._verify_every > 0
            and self._counters[DEVICE_STEPS] % self._verify_every == 0
        )
        r._wait_input_staging_fence()
        r.seq_lens_np[:num_reqs] = expected_positions + 1
        r._check_attention_schedule(num_reqs, self._ones[:num_reqs], num_reqs)
        idx_mapping = r._upload_unless_unchanged(
            "device_decode_idx", torch.from_numpy(idx_np)
        )
        offsets = self._prefill_ckpt_offsets(num_reqs, shape.target_rows)
        inputs = prepare_decode_inputs(
            self.slots, idx_mapping, shape, self.geometry, offsets
        )
        if verify:
            self._counters[VERIFY_STEPS] += 1
            mismatch = self._input_mismatch(inputs, shape, scheduler_output)
            if mismatch is not None:
                self._counters[VERIFY_MISMATCHES] += 1
                self._count_stock_step(VERIFY_INPUT_MISMATCH)
                self._disable(f"verified inputs: {mismatch}")
                return False
        attn_metadata = build_decode_attention_metadata(r, inputs, shape, offsets)
        r._last_sequence_layout_plan = self._layout_plan(scheduler_output, shape)
        r._collect_mamba_state_seed_copies(scheduler_output, 0, num_reqs)
        r._flush_mamba_state_seed_copies()
        r._record_input_staging_fence()
        r._token_padding_update(shape.num_tokens_padded)
        logits = self._forward(
            attn_metadata, inputs, shape, r._stage_timing.stage("main_forward")
        )
        self.slots.note_device_step(idx_np[:num_reqs])
        self._counters[DEVICE_STEPS] += 1
        r.execute_model_state = DeviceDecodeState(
            scheduler_output=scheduler_output,
            shape=shape,
            variant=variant,
            idx_mapping=idx_mapping,
            idx_np=idx_np,
            logits=logits,
            expected_positions=expected_positions,
            verify=verify,
        )
        return True

    def sample_tokens(
        self, state: DeviceDecodeState, grammar_output: Any
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        """The device step's head and single copy, then the bookkeeping
        `TPUModelRunner.sample_tokens` does for a plain one-chunk step."""
        assert grammar_output is None, "a structured-output step ran on the device"
        r = self.runner
        slots = self.slots
        assert slots is not None
        shape, variant = state.shape, state.variant
        with r._stage_timing.stage("sample"):
            tokens, packed = decode_head(
                state.logits,
                state.idx_mapping,
                slots.num_computed,
                slots.last_token,
                slots.temperature,
                slots.top_k,
                slots.top_p,
                slots.seed_lo,
                slots.seed_hi,
                variant=variant,
                null_slot=slots.config.null_slot,
            )
        layout = PackedLayout(shape.num_reqs_padded, variant.num_logprobs)
        with r._stage_timing.stage("sampled_token_d2h"):
            copy_state = AsyncTPUCopyState.from_packed(
                packed, layout, shape.num_reqs, state.expected_positions
            )
        next_tokens: torch.Tensor = tokens
        logprobs: LogprobsLists | None = None
        if state.verify:
            stock = self._verify_head(state, copy_state)
            if stock is not None:
                copy_state, next_tokens, logprobs = stock
        r._update_num_xla_graphs("device_decode_step")
        return self._finish_step(
            state.scheduler_output, copy_state, next_tokens, logprobs
        )

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

    def counters(self) -> dict[str, int]:
        """Device steps, stock steps by reason, verified steps and verify
        mismatches. Only steps planned while device steps are on count: once
        a check turns them off, later steps are not counted."""
        return dict(self._counters)

    def _plan_step(
        self, scheduler_output: Any, target_num_chunks: int
    ) -> str | tuple[DecodeShape, SamplerVariant]:
        """The first reason the step must run on the stock path, or its
        device shape and sampler."""
        r = self.runner
        assert self.slots is not None
        input_batch = r.input_batch
        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids[:num_reqs]
        if scheduler_output.total_num_scheduled_tokens != num_reqs:
            return NOT_PURE_DECODE
        if target_num_chunks != 1:
            return MULTI_CHUNK
        if scheduler_output.has_structured_output_requests:
            return STRUCTURED_OUTPUT
        if r._min_tokens.any_in(input_batch.req_id_to_index):
            return MIN_TOKENS
        if r.num_prompt_logprobs and not r.num_prompt_logprobs.keys().isdisjoint(
            req_ids
        ):
            return PROMPT_LOGPROBS
        if self._unsupported_sampling(input_batch, req_ids):
            return UNSUPPORTED_SAMPLING
        # The stock sampler enforces thinking_token_budget; the head does not.
        thinking_budget = r._thinking_budget
        if thinking_budget and not thinking_budget.requests.keys().isdisjoint(req_ids):
            return UNSUPPORTED_SAMPLING
        if (
            scheduler_output.scheduled_encoder_inputs
            or scheduler_output.kv_connector_metadata is not None
        ):
            return ENCODER_OR_KV
        mismatch = self.slots.mismatch_reason(r, num_reqs)
        if mismatch is not None:
            return mismatch
        self._mirror_mismatches = (None, 0)
        # A device step keeps every row's token and advances its slot, so a
        # row with tokens left to compute after this one (whose token the
        # stock path discards) runs on the stock path.
        if not np.array_equal(
            input_batch.num_tokens_no_spec[:num_reqs],
            input_batch.num_computed_tokens_cpu[:num_reqs] + 1,
        ):
            return SLOT_NOT_READY
        # A generated token's M-RoPE position is its index plus the request's
        # delta in every section, which the slots hold; a prompt token's comes
        # from the request's precomputed table. A decode-ready slot became
        # ready on a step that produced a token, so its next token is a
        # generated one and this check only guards the rule.
        if self.slots.config.mrope_rows and np.any(
            input_batch.num_computed_tokens_cpu[:num_reqs]
            < input_batch.num_prompt_tokens[:num_reqs]
        ):
            return MROPE_PROMPT_TOKEN
        shape = DecodeShape.for_batch(r, num_reqs)
        variant = SamplerVariant.for_batch(r, num_reqs)
        if (
            shape.program_key not in self.registry
            or head_key(shape.num_reqs_padded, variant) not in self.registry
        ):
            return UNREGISTERED_SHAPE
        return shape, variant

    def _unsupported_sampling(self, input_batch: Any, req_ids: list[str]) -> bool:
        """Sampling the head does not apply. It stays on the stock path, so it
        behaves as it does with the flag off."""
        max_num_logprobs = input_batch.max_num_logprobs
        return bool(
            not input_batch.no_penalties
            or not input_batch.no_allowed_token_ids
            or input_batch.bad_words_token_ids
            or input_batch.logprob_token_ids
            or (max_num_logprobs is not None and max_num_logprobs < 0)
            or not self._logits_processor_reqs.isdisjoint(req_ids)
        )

    def _count_stock_step(self, reason: str) -> None:
        self._counters[reason] += 1
        if reason != MIRROR_MISMATCH:
            return
        assert self.slots is not None
        req_id = self.slots.last_mismatch_req_id
        last_req_id, count = self._mirror_mismatches
        count = count + 1 if req_id == last_req_id else 1
        self._mirror_mismatches = (req_id, count)
        if count >= _MAX_MIRROR_MISMATCHES:
            self._disable(
                f"the slot of request {req_id!r} disagreed with the runner "
                f"on {count} checks in a row"
            )

    def _disable(self, reason: str) -> None:
        """Runs every later step on the stock path; with tensor parallelism,
        stops the engine instead, so the ranks never part. Stock steps keep
        the keyed noise and the slots, so no random stream changes source."""
        if self.runner.parallel_config.tensor_parallel_size > 1:
            raise DeviceDecodeInvariantError(
                f"Device decode failed a check ({reason}). Every tensor "
                "parallel rank must take the same decode path, so the engine "
                "stops; unset VLLM_TPU_DEVICE_DECODE to serve on the stock "
                "decode path."
            )
        logger.error(
            "Device decode is off for the rest of the run (%s); unset "
            "VLLM_TPU_DEVICE_DECODE to keep it off.",
            reason,
        )
        self.enabled = False
        self.disabled_reason = reason

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

    def _finish_step(
        self,
        scheduler_output: Any,
        copy_state: AsyncTPUCopyState,
        next_tokens: torch.Tensor,
        logprobs: LogprobsLists | None,
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        """`TPUModelRunner.sample_tokens` after sampling, for one plain
        chunk."""
        r = self.runner
        input_batch = r.input_batch
        num_reqs = input_batch.num_reqs
        req_ids = list(input_batch.req_ids[:num_reqs])
        # A device step keeps every token: `_plan_step` leaves a row whose
        # token the stock rule discards to the stock path.
        discard_sampled_tokens_req_indices: list[int] = []
        request_seq_lens = []
        for i, req_id in enumerate(req_ids):
            req_state = r.requests[req_id]
            seq_len = (
                req_state.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[req_id]
            )
            if seq_len < req_state.num_tokens:
                raise DeviceDecodeInvariantError(
                    f"Device decode sampled request {req_id!r} and advanced "
                    "its slot, but the request still has tokens to compute, so "
                    "the stock rule would discard that token. Unset "
                    "VLLM_TPU_DEVICE_DECODE to serve on the stock decode path, "
                    "and report this."
                )
            request_seq_lens.append((i, req_state, seq_len, req_id))
        async_scheduling = r.scheduler_config.async_scheduling
        if async_scheduling:
            r._modify_prev_results()
        r.maybe_wait_for_kv_save(None)
        kv_connector_output = r._build_kv_connector_output(scheduler_output)
        if async_scheduling:
            req_id_to_index_copy = r._update_placeholder(
                discard_sampled_tokens_req_indices,
                request_seq_lens,
                {i: i for i in range(num_reqs)},
            )
            r._pre_async_results = AsyncPreResults(
                req_ids=req_ids,
                next_tokens_tpu=next_tokens,
                request_seq_lens=request_seq_lens,
                discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
                req_id_to_index_copy=req_id_to_index_copy,
                copy_state=copy_state,
            )
        with r._stage_timing.host_stage("output_assembly"):
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index=dict(input_batch.req_id_to_index),
                sampled_token_ids=[],
                logprobs=logprobs,
                prompt_logprobs_dict={},
                pooler_output=[],
                kv_connector_output=kv_connector_output,
            )
            # pyrefly: ignore[missing-attribute]
            model_runner_output.other_rank_decoding = r._dp_step_other_decode
            async_output = AsyncTPUModelRunnerOutput(
                model_runner_output=model_runner_output,
                copy_state=copy_state,
                discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
            )
        if async_scheduling:
            return async_output
        output = async_output.get_output()
        for i, req_state, _seq_len, req_id in request_seq_lens:
            valid_tokens = output.sampled_token_ids[i]
            if not valid_tokens:
                continue
            req_idx = input_batch.req_id_to_index[req_id]
            start_tok_idx = input_batch.num_tokens_no_spec[req_idx]
            end_tok_idx = start_tok_idx + len(valid_tokens)
            input_batch.token_ids_cpu[req_idx, start_tok_idx:end_tok_idx] = valid_tokens
            input_batch.num_tokens_no_spec[req_idx] = end_tok_idx
            req_state.output_token_ids.extend(valid_tokens)
        return output

    def _input_mismatch(
        self, inputs: DecodeInputs, shape: DecodeShape, scheduler_output: Any
    ) -> str | None:
        """How the live part of the device-built inputs differs from what the
        stock path stages for the step, or None. Syncs the host."""
        r = self.runner
        assert self.geometry is not None
        input_batch = r.input_batch
        num_reqs = shape.num_reqs
        num_computed = input_batch.num_computed_tokens_cpu[:num_reqs]

        def host(tensor: torch.Tensor) -> np.ndarray:
            # Copy the padded tensor and trim on the host: a device slice
            # would compile a program per request count.
            return tensor.cpu().numpy()[:num_reqs]

        checks: list[tuple[str, Any, Any]]
        if inputs.positions.dim() == 1:
            checks = [("position", host(inputs.positions), num_computed)]
        else:
            # The M-RoPE positions the stock path stages for the step. Writing
            # them into the runner's host staging buffer is safe: `try_execute`
            # has waited on the input-staging fence, and a device step uploads
            # nothing from that buffer.
            r._calc_mrope_positions(scheduler_output)
            device = inputs.positions.cpu().numpy()
            checks = [
                (f"section {i} M-RoPE position", got[:num_reqs], want[:num_reqs])
                for i, (got, want) in enumerate(zip(device, r.mrope_positions.np))
            ]
        checks += [
            ("sequence length", host(inputs.seq_lens), num_computed + 1),
            ("input token", host(inputs.input_ids), self._input_tokens(num_reqs)),
        ]
        if inputs.mamba_state_indices is not None:
            req_ids = input_batch.req_ids[:num_reqs]
            mamba_slots = [r._mamba_slot_by_req_id[req_id] for req_id in req_ids]
            checks.append(
                ("mamba state slot", host(inputs.mamba_state_indices), mamba_slots)
            )
        for name, got, want in checks:
            bad = np.flatnonzero(got != np.asarray(want))
            if bad.size:
                row = int(bad[0])
                return (
                    f"request row {row} has {name} {int(got[row])} on the device "
                    f"and {int(np.asarray(want)[row])} on the host"
                )
        for gid, width in zip(self.geometry.gids, self.geometry.widths):
            table = inputs.staged_block_tables[gid].cpu().numpy()
            table = table.reshape(shape.target_rows, width)[:num_reqs]
            block_table = input_batch.block_table[gid]
            num_blocks = block_table.num_blocks_per_row[:num_reqs]
            filled = np.arange(width) < num_blocks[:, None]
            stale = (
                table != block_table.get_cpu_tensor()[:num_reqs, :width].numpy()
            ) & filled
            bad = np.flatnonzero(stale.any(axis=1))
            if bad.size:
                return (
                    f"request row {int(bad[0])} has a stale block table row in "
                    f"KV cache group {gid}"
                )
        return None

    def _input_tokens(self, num_reqs: int) -> np.ndarray:
        """Each request's input token by the stock rule: the previous step's
        sampled token where its async copy holds it, else the token buffer."""
        r = self.runner
        input_batch = r.input_batch
        rows = np.arange(num_reqs)
        positions = input_batch.num_computed_tokens_cpu[:num_reqs]
        tokens = input_batch.token_ids_cpu[rows, positions].astype(np.int64)
        previous = r._pre_async_results
        if previous is not None and previous.req_id_to_index_copy:
            sampled = _next_tokens_cpu(previous)
            for row, req_id in enumerate(input_batch.req_ids[:num_reqs]):
                index = previous.req_id_to_index_copy.get(req_id)
                if index is not None:
                    tokens[row] = int(sampled[index])
        return tokens

    def _verify_head(
        self, state: DeviceDecodeState, copy_state: AsyncTPUCopyState
    ) -> tuple[AsyncTPUCopyState, torch.Tensor, LogprobsLists | None] | None:
        """Samples the step's logits again with the stock programs and the
        same noise and compares. On a mismatch, returns the stock results,
        written back into the slots, and turns the device path off."""
        r = self.runner
        assert self.slots is not None
        shape, variant = state.shape, state.variant
        num_reqs, num_rows = shape.num_reqs, shape.num_reqs_padded
        live = np.arange(num_rows) < num_reqs
        positions = np.zeros(num_rows, dtype=np.int32)
        positions[:num_reqs] = state.expected_positions
        plan = StockChunkPlan(
            num_rows_padded=num_rows,
            slots=state.idx_np[:num_rows].copy(),
            positions=positions,
            produced=live,
            new_num_computed=np.where(live, positions + 1, 0).astype(np.int32),
        )
        chunk = StockChunk(plan, plan.to_device(r), self.slots)
        logits = state.logits
        if variant.all_greedy:
            stock_tokens = r._greedy_sample(logits)
        else:
            temperatures, top_k, top_p = r._build_padded_sampling_params(
                0, num_reqs, logits
            )
            stock_tokens = r.sample_from_logits_func(
                logits,
                temperatures,
                chunk.uniform(logits.shape[-1]),
                top_k,
                top_p,
                all_greedy=False,
                topk_candidates=variant.topk_candidates,
                apply_topk_topp=variant.apply_topk_topp,
            )
        stock_logprobs = (
            r.gather_logprobs(logits, stock_tokens)
            if variant.num_logprobs >= 0
            else None
        )
        # The padded results reach the host whole and are trimmed there: a
        # device slice would compile a program per request count.
        stock_tokens_cpu = stock_tokens.cpu()[:num_reqs]
        logprobs = None
        if stock_logprobs is not None:
            padded = stock_logprobs.tolists()
            logprobs = LogprobsLists(
                logprob_token_ids=padded.logprob_token_ids[:num_reqs],
                logprobs=padded.logprobs[:num_reqs],
                sampled_token_ranks=padded.sampled_token_ranks[:num_reqs],
            )
        mismatch = _head_mismatch(copy_state, stock_tokens_cpu, logprobs)
        if mismatch is None:
            return None
        self._counters[VERIFY_MISMATCHES] += 1
        self.slots.write_back_stock_chunk(r, plan, stock_tokens, chunk.plan_dev)
        stock_copy = AsyncTPUCopyState.from_device_chunks([stock_tokens], [num_reqs])
        self._disable(f"verified sampling: {mismatch}")
        return stock_copy, stock_tokens.view(-1), logprobs


def _next_tokens_cpu(previous: AsyncPreResults) -> np.ndarray:
    """The host copy of `previous.next_tokens_tpu`, which the stock token
    substitution indexes with `req_id_to_index_copy`: a device step's token
    column, or a stock step's chunks at their padded lengths, back to back
    (unlike the trimmed `sampled_token_ids_cpu`). Waits for the copy."""
    copy_state = previous.copy_state
    assert copy_state is not None
    copy_state.wait()
    if copy_state.packed_cpu is not None:
        return copy_state.packed_cpu[:, 0].numpy()
    return torch.cat([chunk.reshape(-1) for chunk in copy_state.chunks_cpu]).numpy()


def _head_mismatch(
    copy_state: AsyncTPUCopyState,
    stock_tokens: torch.Tensor,
    stock_logprobs: LogprobsLists | None,
) -> str | None:
    """How the device step's copied results differ from the stock sampler's
    (host copies of the live rows), or None. Waits for the copy."""
    copy_state.wait()
    got = copy_state.sampled_token_ids_cpu
    bad = torch.nonzero(got.view(-1) != stock_tokens.view(-1))
    if bad.numel():
        row = int(bad[0])
        return (
            f"request row {row} sampled token {int(got[row, 0])} on the device "
            f"and {int(stock_tokens[row, 0])} on the stock path"
        )
    if stock_logprobs is None:
        return None
    lists = copy_state.packed_logprobs()
    assert lists is not None
    cols = lists.logprob_token_ids.shape[1]
    values = stock_logprobs.logprobs[:, :cols]
    same = (
        np.array_equal(
            lists.logprob_token_ids, stock_logprobs.logprob_token_ids[:, :cols]
        )
        and np.array_equal(lists.logprobs.view(np.int32), values.view(np.int32))
        and np.array_equal(
            lists.sampled_token_ranks, stock_logprobs.sampled_token_ranks
        )
    )
    return None if same else "the logprobs differ from the stock gather_logprobs"
