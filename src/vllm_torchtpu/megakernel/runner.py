# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model runner for megakernel programs.

``MegakernelModelRunner`` owns the vLLM contract -- request lifecycle, the
KV connector, async output -- and hands every step to the program
(``megakernel.program``) as host rows: no input preparation, attention
metadata, logits, sampler or drafter. The program chooses the tokens in its
ops and keeps each request's next input on the device.

It reuses ``TPUModelRunner`` for the persistent batch, the unified
KV-cache allocation and the connector registration; the program is built
from the model's configuration after vLLM builds the (weightless) model,
whose layers still declare the KV cache spec.

Structured outputs: the grammar bitmask only reaches ``sample_tokens``, so a
decode step with a structured request is deferred -- ``execute_model`` holds
its :class:`StepInput` and ``sample_tokens`` runs the program with the masks
(``StepInput.mask``); the plain ops then choose among the allowed tokens.
With DSpark such a step runs plain too: its drafts stay on the device, so
vLLM's per-draft rows are unconstrained and only each request's first row
is used.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch_tpu._internal import execution_mode
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel import (
    backend,
    loader,  # noqa: F401  (registers the load format)
)
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3 import program as k3
from vllm_torchtpu.megakernel.kimi_k3.layout import NUM_LAYERS
from vllm_torchtpu.megakernel.program import (
    KVGroup,
    MegakernelProgram,
    RequestInput,
    SamplingSpec,
    StepInput,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncPreResults,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)

logger = init_logger(__name__)


@dataclasses.dataclass
class MegakernelStepState:
    """What ``sample_tokens`` needs from a megakernel step."""

    scheduler_output: SchedulerOutput
    tokens: torch.Tensor | None  # [>= rows, width] device int32, -1 padded
    num_draft_per_req: dict[int, int]
    # Set when the step was deferred to sample_tokens (structured outputs:
    # the grammar bitmask only arrives there, so the program runs there too).
    step_input: StepInput | None = None


def grammar_bitmask_rows(
    grammar_output: Any,
    req_id_to_index: dict[str, int],
    num_reqs: int,
    words: int,
    spec_tokens: dict[str, list[int]] | None = None,
) -> np.ndarray | None:
    """The batch's per-request grammar bitmasks ``[num_reqs, words]`` int32
    (all-ones rows unconstrained), from ``grammar_output``'s compact rows
    (ordered by ``structured_output_request_ids``). A structured request
    has one row, or with speculative decoding one per scheduled draft
    (``spec_tokens``, vLLM's ``scheduled_spec_decode_tokens``) and then the
    bonus row; its first row, the next token's given the tokens it has
    committed, is the one a step that does not verify drafts uses. None: no
    structured request of this batch has a row."""
    mask = np.full((num_reqs, words), -1, np.int32)
    mapped = False
    row = 0
    for req_id in grammar_output.structured_output_request_ids:
        index = req_id_to_index.get(req_id)
        if index is not None and index < num_reqs:
            mask[index] = grammar_output.grammar_bitmask[row]
            mapped = True
        row += 1 + len((spec_tokens or {}).get(req_id, ()))
    return mask if mapped else None


def build_program(vllm_config: Any, tp_rank: int) -> MegakernelProgram:
    """The Kimi-K3 program for the role ``vllm_config`` selects, on TP rank
    ``tp_rank``."""
    hf_config = vllm_config.model_config.hf_text_config
    if int(vllm_config.parallel_config.pipeline_parallel_size) != 1:
        raise NotImplementedError(
            "the K3 megakernel does not support pipeline parallelism"
        )
    if (
        int(hf_config.num_hidden_layers) != NUM_LAYERS
        or int(hf_config.hidden_size) != k3.HIDDEN_SIZE
    ):
        raise NotImplementedError("the K3 megakernel implements full Kimi K3 only")
    spec = vllm_config.speculative_config
    if spec is not None and spec.method != "dspark":
        raise NotImplementedError("the K3 decoder speculates with DSpark only")
    config = MegakernelEngineConfig.from_vllm_config(vllm_config)
    moe_reduce_dtype = tpu_envs.TPU_K3_MOE_REDUCE_DTYPE
    if config.attention_dp > 1:
        k3.check(config)
    if config.attention_dp > 1 or moe_reduce_dtype != "float32":
        from vllm_torchtpu.megakernel.kimi_k3 import config as kernel_config

        # Kernel modules read the config at import: install it before the
        # program imports them.
        kernel_config.install(
            kernel_config.KernelConfig(
                attention_dp=config.attention_dp,
                dcp_group=config.decode_context_parallel_size,
                moe_reduce_dtype=moe_reduce_dtype,
            )
        )
    # The prefill computes the rings of its loader config's draft; the
    # decoder drafts with the speculative config's model (and imports rings).
    if config.role == "prefill":
        draft = backend.draft(vllm_config)
    else:
        draft = Path(str(spec.model)) if spec is not None else None
    return k3.build(config, backend.presharded(vllm_config), draft, tp_rank=tp_rank)


class MegakernelModelRunner(TPUModelRunner):
    """See the module docstring."""

    def __init__(
        self, vllm_config: Any, device: Any, *args: Any, **kwargs: Any
    ) -> None:
        # The program is a handful of whole-model custom ops per step: fusing
        # the eager ops around them would compile a new device program for
        # every distinct surrounding op sequence.
        execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER
        super().__init__(vllm_config, device, *args, **kwargs)
        role = backend.role(vllm_config)
        if role is None:
            raise RuntimeError(
                "MegakernelModelRunner needs the megakernel backend "
                "(--load-format megakernel)"
            )
        self.megakernel_role = role
        self.program: MegakernelProgram | None = None
        self._pending_copies: list[tuple[int, int]] = []
        # Finished requests not yet handed to the program: the scheduler can
        # report them on a zero-token step, which returns before the program runs.
        self._finished_req_ids_pending: set[str] = set()

    # ------------------------------------------------------------ setup
    def _init_speculative_decoding(self) -> None:
        """The program drafts, verifies and accepts in its own ops."""
        self.drafter = None
        self.rejection_sampler = None

    @property
    def _is_async_drafter(self) -> bool:
        return False

    def load_model(self) -> None:
        if self.parallel_config.data_parallel_size > 1:
            raise NotImplementedError(
                "the megakernel runner does not support data parallelism"
            )
        # The weightless model (its layers declare the KV cache spec), through
        # the megakernel load format.
        super().load_model()
        tp_rank = get_tensor_model_parallel_rank()
        self._is_output_rank = tp_rank == 0
        self.program = build_program(self.vllm_config, tp_rank)
        self.program.load_weights()

    def profile_run(self, num_tokens: int) -> None:
        """The program's weights and workspace are resident after
        ``load_model``; the KV budget comes from ``kv_cache_memory_bytes``."""
        if not self.cache_config.kv_cache_memory_bytes:
            raise ValueError(
                "the megakernel runner needs an explicit KV budget "
                "(--kv-cache-memory-bytes)"
            )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """The layers' spec without the state blocks ``MambaSpec`` reserves
        per draft token: the program verifies drafts in its own ops, so no
        request's recurrent state is ever held in them."""
        return {
            name: dataclasses.replace(spec, num_speculative_blocks=0)
            if isinstance(spec, MambaSpec)
            else spec
            for name, spec in super().get_kv_cache_spec().items()
        }

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        super().initialize_kv_cache(kv_cache_config)
        self._skip_device_sampling_metadata()
        self._bind_program()

    def _skip_device_sampling_metadata(self) -> None:
        """The program samples from each request's ``SamplingSpec``, so the
        persistent batch need not rebuild its device sampling metadata and
        logits-processor state on every batch change. Unread, those
        ``temperature[:num_reqs].copy_(...)`` slice writes still run: torch_tpu
        defers them into the program's next op, which then compiles a new
        executable for every batch size."""
        batch = self.input_batch

        def refresh_metadata() -> None:
            batch.batch_update_builder.get_and_reset(batch.num_reqs)

        batch.refresh_metadata = refresh_metadata

    def _bind_program(self) -> None:
        """Hand the program vLLM's KV cache tensors and groups (as the
        attention metadata builders see them)."""
        assert self.program is not None
        context = self.vllm_config.compilation_config.static_forward_context
        caches = {
            name: layer.kv_cache
            for name, layer in context.items()
            if getattr(layer, "kv_cache", None) is not None
        }
        groups: list[list[Any]] = [
            [[], 0] for _ in self.kv_cache_config.kv_cache_groups
        ]
        for group_list in self.attn_groups:
            for group in group_list:
                builder = group.metadata_builders[0]
                groups[builder.kv_cache_group_id][0] += list(group.layer_names)
                groups[builder.kv_cache_group_id][1] = int(builder.target_block_size)
        widths = [
            int(self.input_batch.block_table[g].get_cpu_tensor().shape[1])
            for g in range(len(groups))
        ]
        self.program.bind_kv_caches(
            caches,
            [
                KVGroup(tuple(names), size, width)
                for (names, size), width in zip(groups, widths, strict=True)
            ],
        )

    def capture_model(self) -> int:
        """Compile the program's ops (it warms every shape it will run)."""
        assert self.program is not None
        self.program.warmup()
        return 0

    def _dummy_run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "the megakernel runner has no dummy runs (no data parallelism)"
        )

    # ------------------------------------------------------------ steps
    def _update_states(self, scheduler_output: SchedulerOutput) -> Any:
        # Copy-on-write block copies (a prefix hit ending inside a cached
        # block) go to the program: it owns the pools' contents.
        copies = scheduler_output.kv_cache_block_copies or []
        scheduler_output.kv_cache_block_copies = None
        self._pending_copies = [
            (int(c.src_block_id), int(c.dst_block_id)) for c in copies
        ]
        return super()._update_states(scheduler_output)

    def _prepare_step_input(self, scheduler_output: SchedulerOutput) -> StepInput:
        """The program's input for this step: every request of the persistent
        batch, the requests finished since the program last ran, and the
        pending copy-on-write block copies."""
        batch = self.input_batch
        tables = [
            batch.block_table[g].get_cpu_tensor().numpy()
            for g in range(len(self.kv_cache_config.kv_cache_groups))
        ]
        requests = []
        for i, req_id in enumerate(batch.req_ids[: batch.num_reqs]):
            state = self.requests[req_id]
            requests.append(
                RequestInput(
                    req_id=req_id,
                    computed=int(state.num_computed_tokens),
                    scheduled=int(scheduler_output.num_scheduled_tokens.get(req_id, 0)),
                    token_ids=batch.token_ids_cpu[i],
                    num_tokens=int(state.num_tokens),
                    num_prompt_tokens=int(state.num_prompt_tokens),
                    block_tables=tuple(table[i] for table in tables),
                    sampling=SamplingSpec.from_sampling_params(
                        state.sampling_params, req_id
                    ),
                )
            )
        step_input = StepInput(
            requests, self._finished_req_ids_pending, self._pending_copies
        )
        self._finished_req_ids_pending, self._pending_copies = set(), []
        return step_input

    def _num_draft_per_req(self, step_input: StepInput) -> dict[int, int]:
        """Drafts each request verifies this step, by batch index (vLLM's
        async placeholders count them)."""
        return {
            i: max(request.scheduled - 1, 0)
            if self.speculative_config is not None
            else 0
            for i, request in enumerate(step_input.requests)
        }

    @torch.no_grad()
    def execute_model(
        self, scheduler_output: SchedulerOutput, intermediate_tensors: Any = None
    ) -> Any:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )
        self._update_states(scheduler_output)
        self._finished_req_ids_pending.update(scheduler_output.finished_req_ids)
        if self.scheduler_config.async_scheduling:
            self._flush_disjoint_async_results()
        if scheduler_output.total_num_scheduled_tokens == 0:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
        with set_forward_context(None, self.vllm_config):
            raiden_inline = dist_utils.get_raiden_inline_load()
            self.maybe_setup_kv_connector(
                scheduler_output,
                wait_for_completion=raiden_inline,
                report_completion=not raiden_inline,
            )
        step_input = self._prepare_step_input(scheduler_output)
        assert self.program is not None
        if (
            self.megakernel_role == "decode"
            and scheduler_output.has_structured_output_requests
        ):
            # A structured request decodes this step: the grammar bitmask
            # only reaches sample_tokens (the engine computes it from the
            # previous step's tokens), so the program runs there, masks in
            # hand. The step's async overlap is lost (its op no longer
            # launches ahead of the previous step's host-side output
            # processing).
            self.execute_model_state = MegakernelStepState(
                scheduler_output,
                None,
                self._num_draft_per_req(step_input),
                step_input,
            )
            return None
        output = self.program.step(step_input)
        self.execute_model_state = MegakernelStepState(
            scheduler_output, output.tokens, self._num_draft_per_req(step_input)
        )
        return None

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        state = self.execute_model_state
        self.execute_model_state = None
        if not isinstance(state, MegakernelStepState):
            raise RuntimeError("sample_tokens() without a megakernel step")
        if state.step_input is not None:
            # The deferred structured-output step: run it now, masked (an
            # all-ones mask for unconstrained rows; no structured row mapped:
            # the plain unmasked step).
            assert self.program is not None
            if grammar_output is not None:
                state.step_input.mask = grammar_bitmask_rows(
                    grammar_output,
                    self.input_batch.req_id_to_index,
                    self.input_batch.num_reqs,
                    -(-self.vocab_size // 32),
                    state.scheduler_output.scheduled_spec_decode_tokens,
                )
            state.tokens = self.program.step(state.step_input).tokens
        assert state.tokens is not None
        scheduler_output = state.scheduler_output
        num_reqs = self.input_batch.num_reqs
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])
        discard: list[int] = []
        request_seq_lens = []
        for i, req_id in enumerate(req_ids):
            req_state = self.requests[req_id]
            seq_len = (
                req_state.num_computed_tokens
                + scheduler_output.num_scheduled_tokens.get(req_id, 0)
            )
            if seq_len < req_state.num_tokens:
                discard.append(i)  # a partial prefill chunk: no token
            else:
                request_seq_lens.append((i, req_state, seq_len, req_id))
        copy_state = AsyncTPUCopyState.from_device_chunks([state.tokens], [num_reqs])
        self._modify_prev_results()
        self.maybe_wait_for_kv_save()
        kv_connector_output = self._build_kv_connector_output(scheduler_output)
        materialize_token_ids = (
            not self.scheduler_config.async_scheduling
            or self._is_output_rank
            or not has_kv_transfer_group()
        )
        if materialize_token_ids:
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index=dict(self.input_batch.req_id_to_index),
                sampled_token_ids=[],
                logprobs=[],
                prompt_logprobs_dict={req_id: None for req_id in req_ids},
                pooler_output=[],
                kv_connector_output=kv_connector_output,
            )
        else:
            model_runner_output = ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                kv_connector_output=kv_connector_output,
            )
        async_output = AsyncTPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            copy_state=copy_state,
            discard_sampled_tokens_req_indices=discard,
            materialize_token_ids=materialize_token_ids,
        )
        if self.scheduler_config.async_scheduling:
            req_id_to_index_copy = self._update_placeholder(
                discard,
                request_seq_lens,
                {i: i for i in range(num_reqs)},
                state.num_draft_per_req,
            )
            self._pre_async_results = AsyncPreResults(
                req_ids=req_ids,
                # unused: the program keeps its next inputs
                next_tokens_tpu=state.tokens,
                request_seq_lens=request_seq_lens,
                discard_sampled_tokens_req_indices=discard,
                req_id_to_index_copy=req_id_to_index_copy,
                copy_state=copy_state,
                num_draft_per_req=state.num_draft_per_req,
            )
            return async_output
        # Synchronous scheduling: the tokens now, into the persistent batch.
        final_output = async_output.get_output()
        for i, req_state, _, req_id in request_seq_lens:
            tokens = final_output.sampled_token_ids[i]
            if not tokens:
                continue
            index = self.input_batch.req_id_to_index[req_id]
            start = self.input_batch.num_tokens_no_spec[index]
            self.input_batch.token_ids_cpu[index, start : start + len(tokens)] = tokens
            self.input_batch.num_tokens_no_spec[index] = start + len(tokens)
            req_state.output_token_ids.extend(tokens)
        return final_output
