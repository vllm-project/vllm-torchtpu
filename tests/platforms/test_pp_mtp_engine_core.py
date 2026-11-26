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
"""MTP prefill KV under a pipeline: the engine core never asks for drafts.

vLLM's post_step calls take_draft_token_ids after every synchronous step, a
blocking RPC that waits for the step to leave the last stage, which keeps a
second batch out of the pipeline. Runs vLLM's real post_step on a stub core.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
from vllm.v1.engine.core import EngineCore

from vllm_torchtpu.platforms.pp_validation import (
    patch_engine_core_for_pp_mtp_prefill_kv,
)


def _config(pp: int, spec: bool):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        speculative_config=SimpleNamespace(method="mtp") if spec else None,
    )


@pytest.fixture
def stub_init(monkeypatch):
    def init(self, vllm_config):
        # What the real __init__ sets for post_step.
        self.vllm_config = vllm_config
        self.use_spec_decode = vllm_config.speculative_config is not None
        self.check_for_draft_tokens = self.use_spec_decode
        self.async_scheduling = False
        self.model_executor = mock.Mock()
        self.model_executor.take_draft_token_ids.return_value = None
        self.scheduler = mock.Mock()

    monkeypatch.setattr(EngineCore, "__init__", init)
    monkeypatch.setattr(EngineCore, "_tpu_pp_mtp_kv_patch", False, raising=False)


def _post_step_rpcs(vllm_config) -> int:
    core = EngineCore(vllm_config)
    for _ in range(3):
        core.post_step(model_executed=True)
    return core.model_executor.take_draft_token_ids.call_count


def test_pp_mtp_post_step_skips_the_draft_rpc(stub_init):
    config = _config(pp=2, spec=True)
    patch_engine_core_for_pp_mtp_prefill_kv(config)
    assert EngineCore._tpu_pp_mtp_kv_patch
    assert _post_step_rpcs(config) == 0


def test_patch_is_idempotent(stub_init):
    config = _config(pp=2, spec=True)
    patch_engine_core_for_pp_mtp_prefill_kv(config)
    patched = EngineCore.__init__
    patch_engine_core_for_pp_mtp_prefill_kv(config)
    assert EngineCore.__init__ is patched
    assert _post_step_rpcs(config) == 0


def test_without_the_patch_post_step_asks_every_step(stub_init):
    # The behaviour the patch removes: one blocking RPC per step.
    assert _post_step_rpcs(_config(pp=2, spec=True)) == 3


@pytest.mark.parametrize("pp,spec", [(1, True), (2, False)])
def test_other_configs_are_left_alone(stub_init, pp, spec):
    config = _config(pp=pp, spec=spec)
    patch_engine_core_for_pp_mtp_prefill_kv(config)
    assert not EngineCore._tpu_pp_mtp_kv_patch
    assert _post_step_rpcs(config) == (3 if spec else 0)


def test_one_stage_spec_engine_in_a_patched_process_still_asks(stub_init):
    # The patch is class-wide; an engine without a pipeline keeps its drafts.
    patch_engine_core_for_pp_mtp_prefill_kv(_config(pp=2, spec=True))
    assert _post_step_rpcs(_config(pp=1, spec=True)) == 3
