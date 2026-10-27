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
"""Real native tables and runner/adapter methods, with scheduler-row fixtures."""

from types import SimpleNamespace as NS

import torch
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.block_table import MultiGroupBlockTable, SlotMappingMode

from tests.runner.runner_input_test_utils import _make_runner, _scheduler_output
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
from vllm_torchtpu.runner.model_adapters import ModelInputSnapshot
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def prepare_native_pool_inputs(cfg, plan, block_ids, contexts, scheduled, tokens):
    # Scheduler counts and input staging owners are fixtures. The native table,
    # runner preparation, per-group builder and model adapter are unmodified.
    runner = _make_runner(
        num_computed_tokens=contexts,
        prompt_tokens=[c + n for c, n in zip(contexts, scheduled)],
        scheduled_tokens=scheduled,
        token_paddings=[tokens],
        pcp_size=1,
        block_size=cfg.cache_config.block_size,
    )
    runner.model_config = cfg.model_config
    runner.cache_config = cfg.cache_config
    runner._unified_kv_layout = True
    specs = [g.kv_cache_spec for g in plan.kv_cache_groups]
    table = MultiGroupBlockTable(
        max_num_reqs=runner.max_num_reqs,
        max_num_batched_tokens=tokens,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[s.block_size for s in specs],
        max_num_blocks=[
            s.max_num_blocks_per_req(cfg, cfg.model_config.max_model_len) for s in specs
        ],
        slot_mapping_modes=[
            SlotMappingMode.NONE
            if isinstance(s, MambaSpec)
            else SlotMappingMode.TOKEN_TO_KV_SLOT
            for s in specs
        ],
    )
    for row, ids in enumerate(block_ids):
        table.add_row(ids, row)
    runner.input_batch.block_table = table
    builders = [
        AttentionMetadataBuilder(
            g.kv_cache_spec, g.layer_names, cfg, torch.device("cpu"), runner, gid
        )
        for gid, g in enumerate(plan.kv_cache_groups)
    ]

    def build(**kwargs):
        common = NS(num_reqs=kwargs["num_reqs"])
        result = {}
        for group, builder in zip(plan.kv_cache_groups, builders):
            metadata = builder.build(0, common)
            result.update({name: metadata for name in group.layer_names})
        return result, None

    runner._build_attention_metadata = build
    metadata = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0
    )[0]
    adapter = Qwen4ExpAdapter(runner)
    kwargs = adapter.prepare_inputs(
        ModelInputSnapshot(
            adapter.begin_step(),
            0,
            len(scheduled),
            tokens,
            metadata,
            runner.input_ids_cpu[:tokens],
        )
    )
    return runner, kwargs, metadata
