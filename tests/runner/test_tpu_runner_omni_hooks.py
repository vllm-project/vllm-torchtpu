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
"""Unit tests for the minimal named Omni extension hooks on ``TPUModelRunner``."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
)
from vllm_torchtpu.runner import min_tokens
from vllm_torchtpu.runner import repetition_penalty as rep_pen
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState


@dataclass
class _CustomRunnerOutput(ModelRunnerOutput):
    multimodal_output: dict[str, Any] | None = None


class _HookedRunner(TPUModelRunner):
    """Minimal subclass exercising all 5 named extension hooks."""

    def _build_model_inputs(
        self,
        scheduler_output,
        input_ids: torch.Tensor,
        chunk_mm_inputs,
        start_index: int,
        num_reqs: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
        del scheduler_output, chunk_mm_inputs
        embeds = torch.full(
            (input_ids.shape[0], 4), float(start_index + num_reqs), dtype=torch.float32
        )
        return None, embeds, {"omni_tag": f"chunk_{start_index}_{num_reqs}"}

    def _unwrap_forward_output(
        self, raw_output: Any, **model_kwargs: Any
    ) -> tuple[Any, Any]:
        hidden_states, mm_payload = raw_output
        self._captured_mm = (mm_payload, dict(model_kwargs))
        return hidden_states, None

    def _postprocess_model_output(
        self,
        hidden_states: Any,
        start_index: int,
        num_reqs: int,
        logits_indices: torch.Tensor | None = None,
    ) -> Any:
        self._postprocessed = (
            start_index,
            num_reqs,
            None if logits_indices is None else logits_indices.tolist(),
        )
        return hidden_states + 1.0

    def _make_model_runner_output(self, output: ModelRunnerOutput) -> ModelRunnerOutput:
        return _CustomRunnerOutput(
            req_ids=output.req_ids,
            req_id_to_index=output.req_id_to_index,
            sampled_token_ids=output.sampled_token_ids,
            logprobs=output.logprobs,
            prompt_logprobs_dict=output.prompt_logprobs_dict,
            pooler_output=output.pooler_output,
            kv_connector_output=output.kv_connector_output,
            multimodal_output={"audio": torch.tensor([1.0, 2.0])},
        )

    def _dummy_run_uses_embeds(self) -> bool:
        return True


def test_forward_model_unwrap_hook_and_intermediate_tensors_bypass():
    runner = object.__new__(_HookedRunner)

    def fake_model(input_ids, positions, inputs_embeds=None, **kwargs):
        del input_ids, inputs_embeds
        return (
            torch.ones((positions.shape[0], 4)),
            {"captured_layer": torch.tensor([9])},
        )

    runner.model = fake_model
    out, aux = runner.forward_model(
        input_ids=None,
        positions=torch.arange(3),
        inputs_embeds=torch.zeros((3, 4)),
        omni_tag="step0",
    )
    assert aux is None
    assert out.shape == (3, 4)
    assert runner._captured_mm[0]["captured_layer"].tolist() == [9]
    assert runner._captured_mm[1] == {"omni_tag": "step0"}

    # Non-last PP stage returning IntermediateTensors bypasses _unwrap_forward_output.
    inter = IntermediateTensors({"hidden_states": torch.zeros((3, 4))})
    runner.model = lambda **_kw: inter
    out_pp, aux_pp = runner.forward_model(
        input_ids=torch.zeros(3, dtype=torch.int32),
        positions=torch.arange(3),
    )
    assert out_pp is inter and aux_pp is None


def test_execute_model_calls_build_inputs_and_postprocess_hooks(monkeypatch):
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_forward_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    runner = object.__new__(_HookedRunner)
    runner._pp_wave = None
    runner._dp_step_other_decode = False
    runner._pp_is_first = True
    runner._pp_is_last = True
    runner._pp_pending_scheduler_output = None
    runner._pp_pending_kv_save_nan_check = None
    runner.execute_model_state = None
    runner._update_states = lambda _so: None
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner._mtp_prefill_only = False
    runner._reorder_batch_for_rpa = lambda _so: (0, 0)
    runner._count_input_chunks = lambda _so: (2, 1)
    runner._dp_coordinated_step = lambda *_a, **_kw: (2, 1)
    runner._dp_num_tokens_across_dp = lambda _n: None
    runner.supports_mm_inputs = False
    runner.phase_based_profiler = None
    runner.speculative_config = None
    runner._greedy_top_tokens_this_step = lambda: False
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.maybe_setup_kv_connector = lambda *_a, **_kw: None
    runner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
    )
    runner.query_start_loc_np = np.array([0, 2, 0], dtype=np.int32)
    runner._last_sequence_layout_plan = None
    runner.position_ids = torch.arange(2, dtype=torch.int32)
    runner._flush_mamba_state_seed_copies = lambda: None
    runner._reset_read_offsets_for_new_requests = lambda *_a: None
    runner.input_ids = torch.zeros(2, dtype=torch.int32)
    runner._apply_async_token_substitution = lambda ids, *_a: ids
    runner._record_input_staging_fence = lambda: None
    runner._token_padding_update = lambda _n: None
    runner.mesh = None
    runner._kv_cache_bundle = None
    runner._stage_timing = MagicMock()
    runner._stage_timing.stage.return_value = contextlib.nullcontext()
    runner.is_pooling_model = False
    runner.sequence_layout_planner = SimpleNamespace(
        maybe_select_logits_hidden_states=lambda hs, _plan, idx: hs[idx]
    )
    seen_logits_hs: list[torch.Tensor] = []

    def compute_logits(hs: torch.Tensor) -> torch.Tensor:
        seen_logits_hs.append(hs.clone())
        return torch.zeros((hs.shape[0], 8))

    runner.compute_logits_from_hidden_states = compute_logits
    runner._attn_metadata_builder_ctx = SimpleNamespace(
        mamba_state_indices=None, unified_mamba_state_indices=None
    )
    runner.mamba_slot_read_offsets = None

    empty = torch.empty(0, dtype=torch.int32)
    runner._prepare_inputs = lambda *_a: (
        None,
        torch.tensor([1], dtype=torch.int32),
        1,
        1,
        1,
        empty,
        empty,
        None,
        None,
        None,
    )

    seen_model_call: dict[str, Any] = {}

    def fake_model(input_ids, positions, inputs_embeds=None, **kwargs):
        seen_model_call["input_ids"] = input_ids
        seen_model_call["inputs_embeds"] = inputs_embeds
        seen_model_call["kwargs"] = kwargs
        return torch.full((positions.shape[0], 4), 2.0), {"mm": 1}

    runner.model = fake_model

    so = SimpleNamespace(
        total_num_scheduled_tokens=2,
        scheduled_encoder_inputs={},
        num_scheduled_tokens={"r0": 2},
    )
    ret = runner.execute_model(so)
    assert ret is None
    assert seen_model_call["input_ids"] is None
    assert seen_model_call["inputs_embeds"].shape == (2, 4)
    assert seen_model_call["kwargs"] == {"omni_tag": "chunk_0_1"}
    assert runner._postprocessed == (0, 1, [1])
    # _postprocess_model_output added +1.0 to 2.0 -> 3.0 before compute_logits!
    assert torch.allclose(seen_logits_hs[0], torch.full((1, 4), 3.0))


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_make_model_runner_output_hook_survives_sync_and_async(async_scheduling):
    runner = object.__new__(_HookedRunner)
    runner.execute_model_state = SimpleNamespace(
        scheduler_output=SimpleNamespace(
            num_scheduled_tokens={"r0": 1},
            finished_req_ids=set(),
            has_structured_output_requests=False,
        ),
        draft_chunks=[],
        logits_list=[torch.tensor([[0.1, 0.9, 0.2]])],
        num_reqs_list=[1],
        spec_decode_metadata_list=[None],
        mamba_state_indices_list=None,
        pooler_output_list=[],
        top_tokens=False,
        kv_save_nan_check=None,
    )
    runner.input_batch = SimpleNamespace(
        max_num_logprobs=None,
        num_reqs=1,
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        all_greedy=True,
        num_tokens_no_spec=[1],
        token_ids_cpu=np.zeros((1, 8), dtype=np.int32),
    )
    runner.requests = {
        "r0": SimpleNamespace(
            req_id="r0", num_computed_tokens=0, num_tokens=1, output_token_ids=[]
        )
    }
    runner._pre_async_results = None
    runner._pp_wave = None
    runner._dp_step_other_decode = False
    runner._min_tokens = min_tokens.MinTokensTracker()
    runner._repetition_penalty = rep_pen.RepetitionPenaltyTracker()
    runner.speculative_config = None
    runner._pp_is_last = True
    runner.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    runner._is_dspark_pd_producer = False
    runner.is_pooling_model = False
    runner._get_sampling_generator = lambda: None
    runner._greedy_sample = lambda x: torch.argmax(x, dim=-1, keepdim=True)
    runner._update_mamba_slot_read_offsets = lambda *_a: None
    runner._update_num_xla_graphs = lambda *_a: None
    runner._modify_prev_results = lambda: None
    runner._sync_replicated_drafts_across_tp = lambda drafts: drafts
    runner._update_placeholder = lambda *_a: {"r0": 0}
    runner._stage_timing = MagicMock()
    runner._stage_timing.take_samples.return_value = ([], [])

    def host_copy(chunks, lengths, spec_logprobs=None):
        return AsyncTPUCopyState(
            chunks_cpu=chunks,
            chunk_real_lens=lengths,
            copy_ready_event=MagicMock(),
            spec_logprobs=spec_logprobs,
        )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
    ):
        res = runner.sample_tokens(None)
    final = res.get_output() if async_scheduling else res
    assert isinstance(final, _CustomRunnerOutput)
    assert final.sampled_token_ids == [[1]]
    assert final.multimodal_output is not None
    assert final.multimodal_output["audio"].tolist() == [1.0, 2.0]


def test_dummy_run_uses_embeds_hook_when_supports_mm_inputs_is_false(monkeypatch):
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_forward_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.synchronize_tensors",
        lambda *_a, **_kw: None,
    )
    runner = object.__new__(_HookedRunner)
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.uses_mrope = False
    runner._pp_is_first = True
    runner._pp_wave = None
    runner._pp_topk_buffer = None
    runner._hidden_states_dtype = torch.float32
    runner._kv_cache_bundle = None
    runner.mesh = None
    runner.vllm_config = MagicMock()
    runner.lora_config = None
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner._has_mamba_state = False
    runner.mamba_slot_read_offsets = None
    runner._mamba_ckpt_window = 1
    runner._unified_kv_layout = False
    runner.empty_slot_mappings = {}
    runner.position_ids = torch.zeros(32, dtype=torch.int32)
    runner.num_reqs_paddings = [4, 8]
    runner.num_reqs_max_model_len = 8
    runner.num_reqs_most_model_len = None
    runner.block_table_cpu = torch.zeros((8, 4), dtype=torch.int32)
    runner.speculative_config = None
    runner._pp_is_last = True
    runner.is_pooling_model = False
    runner._record_input_staging_fence = lambda: None
    runner._input_staging_fence = None
    runner._unfenced_h2d_sources = []
    runner._fenced_h2d_sources = []
    runner._free_h2d_sources = {}
    runner._registered_h2d_sources = set()
    runner._token_padding_state = None
    runner.sequence_layout_planner = SimpleNamespace(
        prepare_dummy=lambda **_kw: SimpleNamespace(
            descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR
        )
    )
    runner.maybe_select_dummy_loras = lambda *_a, **_kw: contextlib.nullcontext()
    runner._dp_num_tokens_across_dp = lambda _n: None
    bt = MagicMock()
    bt.max_num_blocks_per_req = 4
    bt.get_cpu_tensor.return_value = torch.zeros((4, 4), dtype=torch.int32)
    runner.input_batch = SimpleNamespace(block_table=[bt])
    builder = AttentionMetadataBuilder(
        kv_cache_spec=FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        ),
        layer_names=["attn.0"],
        vllm_config=MagicMock(),
        device=runner.device,
        runner=runner,
        kv_cache_group_id=0,
    )
    group = MagicMock()
    group.metadata_builders = [builder]
    group.layer_names = ["attn.0"]
    runner.attn_groups = [[group]]

    embed_calls: list[torch.Tensor] = []
    forward_calls: list[tuple[Any, Any]] = []

    def embed_input_ids(ids: torch.Tensor) -> torch.Tensor:
        embed_calls.append(ids.clone())
        return torch.zeros((ids.shape[0], 8), dtype=torch.float32)

    runner.model = SimpleNamespace(embed_input_ids=embed_input_ids)

    def fake_forward_model(input_ids, positions, inputs_embeds=None, **_kw):
        del positions
        forward_calls.append((input_ids, inputs_embeds))
        return torch.zeros((16, 8), dtype=torch.float32), None

    runner.forward_model = fake_forward_model
    runner._dummy_run(num_tokens=16, num_reqs=4, num_blocks=4)

    assert len(embed_calls) == 1 and embed_calls[0].shape == (16,)
    assert len(forward_calls) == 1
    assert forward_calls[0][0] is None
    assert forward_calls[0][1] is not None and forward_calls[0][1].shape == (16, 8)
