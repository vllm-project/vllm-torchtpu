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
"""Tests for custom stage input/output processors across stage boundaries."""

import threading
from collections import defaultdict, deque
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.data_entry_keys import EmbeddingsStruct, OmniPayloadStruct
from vllm_omni.distributed.omni_connectors.transfer_adapter import (
    chunk_transfer_adapter,
)
from vllm_omni.engine.stage_engine_core_client import (
    StageEngineCoreClient,
    _default_process_engine_inputs,
)
from vllm_omni.engine.stage_init_utils import (
    extract_stage_metadata_from_omni_stage_config,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.diffusion_disagg import (
    diffusion_stage_handoff,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S

OmniChunkTransferAdapter = chunk_transfer_adapter.OmniChunkTransferAdapter

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

REL_L2_RATIO = 4.0
REL_L2_FLOOR = 1e-2


def _sync_next_stage_hook(
    transfer_manager: Any,
    multimodal_output: dict[str, Any],
    request: Any,
    is_finished: bool,
) -> OmniPayloadStruct:
    del transfer_manager, request, is_finished
    return OmniPayloadStruct(
        embed=EmbeddingsStruct(embedding=multimodal_output["embeds"] * 2.0)
    )


def _async_chunk_next_stage_hook(
    transfer_manager: Any,
    multimodal_output: dict[str, Any],
    request: Any,
    is_finished: bool,
) -> OmniPayloadStruct:
    del transfer_manager, request, is_finished
    return OmniPayloadStruct(
        embed=EmbeddingsStruct(embedding=multimodal_output["embeds"] + 0.5)
    )


def _custom_stage1_input_hook(
    source_outputs: list[Any],
    prompt: Any,
    requires_multimodal_data: bool,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data
    return [
        OmniTokensPrompt(
            prompt_token_ids=[tid + 10 for tid in so.outputs[0].token_ids],
            multi_modal_data=None,
        )
        for so in source_outputs
    ]


class _MemoryPutConnector:
    """Minimal in-memory connector target for OmniChunkTransferAdapter puts."""

    stage_id = 0

    def __init__(self) -> None:
        self.puts: list[tuple[str, str, str, OmniPayloadStruct]] = []

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: OmniPayloadStruct,
    ) -> tuple[bool, int, dict[str, Any]]:
        self.puts.append((from_stage, to_stage, put_key, data))
        return True, 0, {}


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a = actual.detach().cpu().double().flatten()
    r = reference.detach().cpu().double().flatten()
    return float((a - r).norm() / r.norm())


@pytest.mark.parametrize("async_chunk", [False, True])
def test_custom_stage_input_output_processors(async_chunk: bool) -> None:
    """Custom Stage Input / Output Processors

    - TPU path: resolve a 2-stage PipelineConfig with async_chunk=False/True via
      VllmOmniConfig.from_pipeline_config and
      extract_stage_metadata_from_omni_stage_config, then execute the producer
      hook in OmniChunkTransferAdapter, consumer custom_process_input_func in
      StageEngineCoreClient.process_engine_inputs, and diffusion_stage_handoff
      on bf16 tensors on torch.device("tpu").
    - Setup check: assert stage 0 selects custom_process_next_stage_input_func
      vs async_chunk_process_next_stage_input_func per async_chunk and stage 1
      StageMetadata.custom_process_input_func resolves to the imported callable.
    - Outcome check: assert producer/consumer hooks format TPU tensor payloads
      and token/prompt fields across the stage boundary, matching CPU fp32
      reference within max(RATIO * cpu_bf16_err, FLOOR) and exact tokens.
    """
    sync_path = f"{__name__}._sync_next_stage_hook"
    async_path = f"{__name__}._async_chunk_next_stage_hook"
    input_path = f"{__name__}._custom_stage1_input_hook"

    pipeline_cfg = PipelineConfig(
        model_type="custom_stage_processors_test",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="thinker",
                execution_type=StageExecutionType.LLM_AR,
                custom_process_next_stage_input_func=sync_path,
                async_chunk_process_next_stage_input_func=async_path,
                final_output=False,
            ),
            StagePipelineConfig(
                stage_id=1,
                model_stage="talker",
                execution_type=StageExecutionType.LLM_AR,
                input_sources=(0,),
                custom_process_input_func=input_path,
                final_output=True,
                final_output_type="audio",
            ),
        ),
    )

    omni_cfg = VllmOmniConfig.from_pipeline_config(
        pipeline_cfg,
        cli_overrides={"async_chunk": async_chunk},
    )
    stage0_cfg = omni_cfg.stage_configs[0]
    stage1_cfg = omni_cfg.stage_configs[1]

    expected_next_path = async_path if async_chunk else sync_path
    assert stage0_cfg.custom_process_next_stage_input_func == expected_next_path, (
        f"async_chunk={async_chunk}: expected {expected_next_path!r}, "
        f"got {stage0_cfg.custom_process_next_stage_input_func!r}"
    )

    stage1_meta = extract_stage_metadata_from_omni_stage_config(stage1_cfg)
    assert stage1_meta.custom_process_input_func is _custom_stage1_input_hook, (
        f"Stage 1 custom_process_input_func {stage1_meta.custom_process_input_func} "
        f"!= {_custom_stage1_input_hook}"
    )

    torch.manual_seed(0)
    cpu_fp32_embeds = torch.randn(2, 8, dtype=torch.float32)
    cpu_bf16_embeds = cpu_fp32_embeds.to(torch.bfloat16)
    tpu_bf16_embeds = cpu_bf16_embeds.to(torch.device("tpu"))

    expected_next_fn = (
        _async_chunk_next_stage_hook if async_chunk else _sync_next_stage_hook
    )
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter.connector = _MemoryPutConnector()
    adapter._sender_state_lock = threading.Lock()
    adapter._receiver_state_lock = threading.Lock()
    adapter._sender_tokens = {}
    adapter._registered_load_entries = {}
    adapter._active_streams = {}
    adapter.put_req_chunk = defaultdict(int)
    adapter.get_req_chunk = defaultdict(int)
    adapter.ramp_chunk_count = defaultdict(int)
    adapter.upstream_exhausted_requests = set()
    adapter.segment_finished_requests = set()
    adapter.requests_with_ready_chunks = set()
    adapter.replaced_streaming_prompt_ids = set()
    adapter.request_ids_mapping = {}
    adapter.requests_origin_status = {}
    adapter._pending_ar_prompt_updates = {}
    adapter._streaming_condition_lengths = {}
    adapter._streaming_condition_seqs = {}
    adapter.waiting_for_chunk_waiting_requests = deque()
    adapter.waiting_for_chunk_running_requests = deque()
    adapter._held_non_active = deque()
    adapter._finished_load_reqs = set()
    adapter._waiting_since = {}
    adapter.request_payload = {}
    adapter.code_prompt_token_ids = defaultdict(list)
    adapter.requests_num_chunks_sent = defaultdict(int)
    adapter._segment_generation = {}
    adapter._adaptive_states = {}
    adapter._pending_streaming_prefills = {}
    adapter._processor_accepts_step_tokens = {}
    adapter.custom_process_next_stage_input_func = (
        _async_chunk_next_stage_hook
        if stage0_cfg.custom_process_next_stage_input_func == async_path
        else _sync_next_stage_hook
    )

    adapter._send_single_request_for_generation(
        {
            "multimodal_output": {"embeds": tpu_bf16_embeds},
            "request": SimpleNamespace(request_id="req-0", external_req_id="req-0"),
            "is_finished": True,
            "is_segment_finished": True,
        }
    )
    assert len(adapter.connector.puts) == 1, (
        f"Expected 1 connector put, got {len(adapter.connector.puts)}"
    )
    put_payload = adapter.connector.puts[0][3]
    assert put_payload.embed is not None, (
        f"async_chunk={async_chunk}: expected non-None put_payload.embed, "
        f"got {put_payload}"
    )
    assert put_payload.embed.embedding is not None, (
        f"async_chunk={async_chunk}: expected non-None put_payload.embed.embedding"
    )
    tpu_payload = put_payload.embed.embedding
    cpu_fp32_embed = expected_next_fn(
        None, {"embeds": cpu_fp32_embeds}, None, True
    ).embed
    assert cpu_fp32_embed is not None, (
        f"async_chunk={async_chunk}: CPU fp32 reference embed is None"
    )
    assert cpu_fp32_embed.embedding is not None, (
        f"async_chunk={async_chunk}: CPU fp32 reference embed.embedding is None"
    )
    cpu_fp32_payload = cpu_fp32_embed.embedding
    cpu_bf16_embed = expected_next_fn(
        None, {"embeds": cpu_bf16_embeds}, None, True
    ).embed
    assert cpu_bf16_embed is not None, (
        f"async_chunk={async_chunk}: CPU bf16 reference embed is None"
    )
    assert cpu_bf16_embed.embedding is not None, (
        f"async_chunk={async_chunk}: CPU bf16 reference embed.embedding is None"
    )
    cpu_bf16_payload = cpu_bf16_embed.embedding

    cpu_bf16_err = _rel_l2(cpu_bf16_payload, cpu_fp32_payload)
    tpu_err = _rel_l2(tpu_payload, cpu_fp32_payload)
    limit = max(REL_L2_RATIO * cpu_bf16_err, REL_L2_FLOOR)
    assert tpu_err <= limit, (
        f"async_chunk={async_chunk}: producer payload rel_l2 {tpu_err:.4e} "
        f"exceeds limit {limit:.4e}"
    )

    source_output = SimpleNamespace(
        request_id="req-0",
        outputs=[SimpleNamespace(token_ids=[1, 2, 3])],
        custom_output={"latents": tpu_bf16_embeds},
    )
    prompt = {"prompt": "test prompt", "seed": 42}

    client = object.__new__(StageEngineCoreClient)
    client.stage_id = stage1_meta.stage_id
    client.engine_input_source = stage1_meta.engine_input_source
    client.requires_multimodal_data = stage1_meta.requires_multimodal_data
    client.custom_process_input_func = stage1_meta.custom_process_input_func
    client._stage_hf_config = None
    client.vllm_config = SimpleNamespace(model_config=SimpleNamespace())

    custom_inputs = client.process_engine_inputs([source_output], prompt=[prompt])
    default_inputs = _default_process_engine_inputs([source_output], [prompt], False)
    ref_inputs = _custom_stage1_input_hook([source_output], [prompt], False)

    assert custom_inputs[0]["prompt_token_ids"] == ref_inputs[0]["prompt_token_ids"], (
        f"Custom token_ids {custom_inputs[0]['prompt_token_ids']} != "
        f"reference {ref_inputs[0]['prompt_token_ids']}"
    )
    assert (
        custom_inputs[0]["prompt_token_ids"] != default_inputs[0]["prompt_token_ids"]
    ), "Custom stage input processor did not override default token_ids"

    handoff = diffusion_stage_handoff([source_output], [prompt])[0]
    assert handoff["prompt"] == "test prompt", (
        f"diffusion_stage_handoff prompt {handoff['prompt']!r} != 'test prompt'"
    )
    assert handoff["seed"] == 42, (
        f"diffusion_stage_handoff seed {handoff['seed']} != 42"
    )
    handoff_err = _rel_l2(handoff["latents"], cpu_fp32_embeds)
    handoff_limit = max(
        REL_L2_RATIO * _rel_l2(cpu_bf16_embeds, cpu_fp32_embeds), REL_L2_FLOOR
    )
    assert handoff_err <= handoff_limit, (
        f"diffusion_stage_handoff latents rel_l2 {handoff_err:.4e} "
        f"exceeds limit {handoff_limit:.4e}"
    )
