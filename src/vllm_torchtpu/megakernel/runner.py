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

Pipeline parallelism: a program that declares ``pipeline`` runs one stage
(``program.PipelineStage``). Each step it takes the previous stage's
hand-off from the pipeline wave (``distributed.pp_wave``) and gives the wave
its own; only the last stage produces tokens, and the earlier stages report
just their KV-connector progress.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch_tpu._internal import execution_mode
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.forward_context import set_forward_context
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.pp_wave import PPWave
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel import (
    backend,
    loader,  # noqa: F401  (registers the load format)
)
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3 import pools
from vllm_torchtpu.megakernel.kimi_k3 import program as k3
from vllm_torchtpu.megakernel.kimi_k3.layout import NUM_LAYERS
from vllm_torchtpu.megakernel.program import (
    KVGroup,
    MegakernelProgram,
    PipelineStage,
    RequestInput,
    SamplingSpec,
    StepInput,
    StepOutput,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner, _pp_step
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
    ``tp_rank`` (under pipeline parallelism, this worker's prefill stage)."""
    hf_config = vllm_config.model_config.hf_text_config
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
    select_pack16 = tpu_envs.TPU_K3_DSA_SELECT_PACK16
    sparse_prefill = tpu_envs.TPU_K3_MLA_SPARSE_PREFILL.lower()
    host_combine = tpu_envs.TPU_K3_DCP_HOST_COMBINE
    select_sort = tpu_envs.TPU_K3_DSA_SELECT_SORT
    select_stop = tpu_envs.TPU_K3_DSA_SELECT_STOP
    select_digit_bits = tpu_envs.TPU_K3_DSA_SELECT_DIGIT_BITS
    select_count_unroll = tpu_envs.TPU_K3_DSA_SELECT_COUNT_UNROLL
    kda_branch_first = tpu_envs.TPU_K3_KDA_BRANCH_FIRST
    score_block_major = tpu_envs.TPU_K3_DSA_SCORE_BLOCK_MAJOR
    select_packed_unroll = tpu_envs.TPU_K3_DSA_SELECT_PACKED_UNROLL
    query_wire = tpu_envs.TPU_K3_DCP_QUERY_WIRE.lower()
    if config.attention_dp > 1:
        k3.check(config)
    if (
        config.attention_dp > 1
        or moe_reduce_dtype != "float32"
        or select_pack16
        or sparse_prefill != "off"
        or host_combine
        or select_sort
        or select_stop
        or kda_branch_first
        or score_block_major
        or select_digit_bits
        or select_count_unroll != 1
        or select_packed_unroll
        or query_wire != "bf16"
    ):
        from vllm_torchtpu.megakernel.kimi_k3 import config as kernel_config

        # Kernel modules read the config at import: install it before the
        # program imports them.
        kernel_config.install(
            kernel_config.KernelConfig(
                attention_dp=config.attention_dp,
                dcp_group=config.decode_context_parallel_size,
                moe_reduce_dtype=moe_reduce_dtype,
                mla_sparse_prefill=sparse_prefill,
                dsa_select_pack16=select_pack16,
                mla_sparse_fp8_gather=sparse_prefill != "off",
                dcp_host_combine=host_combine,
                dsa_select_sort=select_sort,
                dsa_select_stop=select_stop,
                kda_branch_first=kda_branch_first,
                dsa_score_block_major=score_block_major,
                dsa_select_digit_bits=select_digit_bits,
                dsa_select_count_unroll=select_count_unroll,
                dsa_select_packed_unroll=select_packed_unroll,
                dcp_query_wire=query_wire,
            )
        )
    if config.pipeline_parallel_size > 1:
        from vllm.distributed import get_pp_group
        from vllm.distributed.parallel_state import get_pcp_group
        from vllm.distributed.utils import get_pp_indices

        from vllm_torchtpu.distributed.pcp import get_or_create_pcp_mesh
        from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_stage import STAGE_LAYERS
        from vllm_torchtpu.megakernel.kimi_k3.stage import AXIS

        stage = get_pp_group().rank_in_group
        if get_pp_indices(NUM_LAYERS, stage, config.pipeline_parallel_size) != tuple(
            STAGE_LAYERS[stage]
        ):
            raise NotImplementedError(
                "the K3 prefill stages hold layers "
                f"{', '.join(f'{a}-{b - 1}' for a, b in STAGE_LAYERS)}: set "
                "VLLM_PP_LAYER_PARTITION="
                + ",".join(str(b - a) for a, b in STAGE_LAYERS)
            )
        return k3.build_stage(
            config,
            backend.presharded(vllm_config),
            stage=stage,
            mesh=get_or_create_pcp_mesh(AXIS),
            core=get_pcp_group().rank_in_group,
            draft=backend.draft(vllm_config),
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
        # Finished and preempted requests not yet handed to the program: the
        # scheduler can report them on a zero-token step, which returns before
        # the program runs. vLLM recomputes a preempted request when it resumes.
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
        # The MLA DSpark draft's latent layers: KV cache layers of both roles
        # (of a pipeline's last stage, which writes them).
        draft = backend.mla_draft(self.vllm_config)
        if draft is not None and get_pp_group().is_last_rank:
            from vllm_torchtpu.megakernel.kimi_k3.draft_layers import (
                register_draft_layers,
            )

            # Held only to keep the registered layer modules alive.
            self._draft_layers = register_draft_layers(self.vllm_config, draft)
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
        specs = {
            name: dataclasses.replace(spec, num_speculative_blocks=0)
            if isinstance(spec, MambaSpec)
            else spec
            for name, spec in super().get_kv_cache_spec().items()
        }
        self._reserve_index_cache(specs)
        return specs

    def _reserve_index_cache(self, specs: dict[str, KVCacheSpec]) -> None:
        """Leave room in the KV budget for Kimi-K3.1's index-key cache: per
        block, one index page per MLA layer beside the block's unified pages,
        which vLLM sizes by its widest group of layers (the MLA DSpark
        draft's latent layers join the target's MLA layers there)."""
        context = self.vllm_config.compilation_config.static_forward_context
        names = pools.index_cache_names(context)
        if not names or self.cache_config.num_gpu_blocks_override is not None:
            return
        spec = next(
            (s for s in specs.values() if isinstance(s, MLAAttentionSpec)), None
        )
        if spec is None:
            raise RuntimeError("Kimi-K3.1's index-key cache needs MLA layers")
        groups = get_kv_cache_groups(self.vllm_config, dict(specs))
        block_bytes = max(
            sum(
                (
                    group.kv_cache_spec.kv_cache_specs[name]
                    if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                    else group.kv_cache_spec
                ).page_size_bytes
                for name in group.layer_names
            )
            for group in groups
        )
        self.cache_config.num_gpu_blocks_override = pools.index_block_budget(
            self.cache_config.kv_cache_memory_bytes,
            block_bytes,
            len(names),
            spec.block_size,
            self.cache_config.cache_dtype,
        )

    def extra_kv_cache_tensors(self, kv_cache_config: Any) -> dict[str, torch.Tensor]:
        """Kimi-K3.1's index-key cache, one block-major tensor bound under
        every MLA layer's index name; none for Kimi-K3."""
        context = self.vllm_config.compilation_config.static_forward_context
        names = pools.index_cache_names(context)
        if not names:
            return {}
        blocks = {
            int(group.kv_cache_spec.block_size)
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, MLAAttentionSpec)
        }
        if len(blocks) != 1:
            raise RuntimeError(
                "Kimi-K3.1's index-key cache needs one MLA block size, found "
                f"{sorted(blocks)}"
            )
        (block,) = blocks
        dtype = self.cache_config.cache_dtype
        index_bytes = (
            kv_cache_config.num_blocks
            * len(names)
            * pools.index_cache_rows(block, dtype)
            * 128
        )
        if self.kv_cache_raw_tensors:
            used = sum(t.nbytes for t in self.kv_cache_raw_tensors) + index_bytes
            if used > self.cache_config.kv_cache_memory_bytes:
                raise RuntimeError(
                    f"the unified pools and Kimi-K3.1's index-key cache take {used} "
                    f"bytes, above the KV budget of "
                    f"{self.cache_config.kv_cache_memory_bytes}"
                )
        return pools.allocate_index_cache(
            kv_cache_config.num_blocks, names, block, dtype, self.device
        )

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
        """Build the pipeline wave on a stage, then compile the program's ops
        (it warms every shape it will run)."""
        assert self.program is not None
        if self.parallel_config.pipeline_parallel_size > 1 and self._pp_wave is None:
            stage = self._stage()
            self._pp_wave = PPWave(
                self.device, stage.max_rows(), dict(stage.handoff_template())
            )
            self._pp_wave.warmup()
        self.program.warmup()
        return 0

    def _stage(self) -> PipelineStage:
        assert self.program is not None
        return cast(PipelineStage, self.program)

    def profile_pipeline_chunks(self) -> dict[str, Any]:
        """The chunk granularity and this stage's step times for the pipeline
        chunk scheduler's cost model (see ``TPUModelRunner``)."""
        from vllm_torchtpu.core.pp_chunks import (
            chunk_buckets,
            chunk_granularity,
            profile_points,
        )

        capabilities = backend.capabilities(self.vllm_config)
        assert capabilities is not None
        tokens = self.scheduler_config.max_num_batched_tokens
        granularity = chunk_granularity(
            self.cache_config.block_size, tokens, capabilities.page_aligned_chunks
        )
        points = profile_points(
            [*chunk_buckets(tokens, granularity), tokens], tokens, self.max_model_len
        )
        times = self._stage().time_steps(points)
        return {
            "granularity": granularity,
            "samples": [
                (int(t), int(p), float(ms))
                for (t, p), ms in zip(points, times, strict=True)
            ],
            "schedule": None,
        }

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
        batch, the requests finished or preempted since the program last ran,
        and the pending copy-on-write block copies."""
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
    @_pp_step
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
        self._finished_req_ids_pending.update(scheduler_output.preempted_req_ids or ())
        # The KV connector before the flush: the flush waits for the previous
        # step's tokens (the device), and a prefill registers the blocks of the
        # requests it finished for their transfer in the connector's setup.
        if scheduler_output.total_num_scheduled_tokens == 0:
            if not has_kv_transfer_group():
                if self.scheduler_config.async_scheduling:
                    self._flush_disjoint_async_results()
                return EMPTY_MODEL_RUNNER_OUTPUT
            output = self.kv_connector_no_forward(scheduler_output, self.vllm_config)
            if self.scheduler_config.async_scheduling:
                self._flush_disjoint_async_results()
            return output
        with set_forward_context(None, self.vllm_config):
            raiden_inline = dist_utils.get_raiden_inline_load()
            self.maybe_setup_kv_connector(
                scheduler_output,
                wait_for_completion=raiden_inline,
                report_completion=not raiden_inline,
            )
        if self.scheduler_config.async_scheduling:
            self._flush_disjoint_async_results()
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
        output = self._run_program(step_input)
        if output is None:
            # A stage before the last: it reports only its connector progress.
            self._pp_pending_scheduler_output = scheduler_output
            return None
        self.execute_model_state = MegakernelStepState(
            scheduler_output, output.tokens, self._num_draft_per_req(step_input)
        )
        return None

    def _run_program(self, step_input: StepInput) -> StepOutput | None:
        """Run one step, between the pipeline wave's hand-offs on a stage.
        None on a stage before the last, which produces no tokens."""
        assert self.program is not None
        if self._pp_wave is None:
            return self.program.step(step_input)
        rows = self._stage().rows(step_input)
        received = self._pp_wave.start_forward(rows)
        step_input.hidden = None if received is None else received.tensors
        output = self.program.step(step_input)
        if self._pp_is_last:
            self._pp_wave.end_forward(None, rows)
            return output
        self._pp_wave.end_forward(dict(output.hidden or {}), rows)
        return None

    @_pp_step
    def sample_tokens(self, grammar_output: Any = None) -> Any:
        state = self.execute_model_state
        self.execute_model_state = None
        if state is None and not self._pp_is_last:
            return self._sample_tokens_pp_intermediate_stage()
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
