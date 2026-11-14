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
"""Emitter: the parallel-config log line."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from vllm.config import ParallelConfig

from vllm_torchtpu.observability import placement_emitter as emitter


def parallel_config(**kw):
    base = dict(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        world_size=4,
        nnodes=1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def rank_placement_lines(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("rank_placement ")
    ]


def test_parallel_summary_mirrors_parallel_config():
    summary = emitter.parallel_summary(parallel_config(nnodes=2, world_size=8))
    assert summary.to_dict() == {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "world_size": 8,
        "nnodes": 2,
    }


def test_parallel_summary_reads_vllm_parallel_config_at_the_pin():
    """Every field is read from the real ParallelConfig, so a rename at a pin
    bump fails here rather than at server startup."""
    config = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
    summary = emitter.parallel_summary(config).to_dict()
    assert summary == {name: getattr(config, name) for name in summary}
    assert summary["tensor_parallel_size"] == 2
    assert summary["world_size"] == config.world_size


def test_emit_log_is_one_json_line(caplog):
    with caplog.at_level(logging.INFO, logger=emitter.logger.name):
        emitter.emit_log(emitter.parallel_summary(parallel_config()))
    lines = rank_placement_lines(caplog)
    assert len(lines) == 1
    payload = json.loads(lines[0].split(" ", 1)[1])
    assert payload["parallel"]["tensor_parallel_size"] == 4


def test_publish_logs_the_engine_config(caplog):
    config = SimpleNamespace(parallel_config=parallel_config(data_parallel_size=2))
    with caplog.at_level(logging.INFO, logger=emitter.logger.name):
        emitter.publish_placement(config)
    payload = json.loads(rank_placement_lines(caplog)[0].split(" ", 1)[1])
    assert payload["parallel"]["data_parallel_size"] == 2
