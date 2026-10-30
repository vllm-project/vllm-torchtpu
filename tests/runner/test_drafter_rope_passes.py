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
"""The rope-cache passes (TPU_ROPE_CACHE_TRUNCATE, TPU_ROPE_CACHE_ROW_MAJOR)
on the MTP draft model after ``drafter.load_model``: the modules the target's
pass handled -- vLLM's ``get_rope`` shares one rotary instance per geometry --
are skipped, a draft-only rotary module gets its own pass, and nothing
happens with the flags off."""

from collections import Counter
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from vllm_torchtpu import envs
from vllm_torchtpu.runner import utils as runner_utils
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _rope(max_pos):
    cfg = SimpleNamespace(
        custom_ops=["all"],
        enabled_custom_ops=Counter(),
        disabled_custom_ops=Counter(),
        mode=0,
        backend="eager",
    )
    with mock.patch(
        "vllm.model_executor.custom_op.get_cached_compilation_config",
        return_value=cfg,
    ):
        return RotaryEmbedding(64, 64, max_pos, 10000, True, torch.float32)


def _runner(max_model_len=1024):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            multimodal_config=None,
            is_multimodal_model=False,
            max_model_len=max_model_len,
        ),
        model=None,
    )


def _models():
    shared = _rope(4096)
    target = torch.nn.Module()
    target.rotary = shared
    draft = torch.nn.Module()
    draft.rotary = shared  # get_rope's cached instance
    draft.own = _rope(8192)  # a geometry/dtype of the draft's own
    return target, draft


def test_truncate_skips_the_shared_module_and_handles_the_drafts_own():
    target, draft = _models()
    runner = _runner()
    runner.model = target
    TPUModelRunner._truncate_rope_caches(runner)
    assert target.rotary.cos_sin_cache.shape[0] == 1024
    skip = {id(m) for m in target.modules()}
    with mock.patch("vllm_torchtpu.runner.tpu_runner.logger") as log:
        TPUModelRunner._truncate_rope_caches(
            runner, draft, skip_modules=skip, label="the draft"
        )
    assert draft.rotary is target.rotary
    assert draft.own.cos_sin_cache.shape[0] == 1024
    assert log.info.call_args.args[:2] == ("Truncated %d of %d rope caches of %s", 1)
    assert log.info.call_args.args[2:] == (1, "the draft")


def test_truncate_reports_a_draft_sharing_every_module():
    target, draft = _models()
    del draft.own
    runner = _runner()
    runner.model = target
    TPUModelRunner._truncate_rope_caches(runner)
    skip = {id(m) for m in target.modules()}
    with mock.patch("vllm_torchtpu.runner.tpu_runner.logger") as log:
        TPUModelRunner._truncate_rope_caches(
            runner, draft, skip_modules=skip, label="the draft"
        )
    assert "shared with an earlier pass" in log.info.call_args.args[0]
    assert log.info.call_args.args[1:] == ("the draft", 1)
    log.warning.assert_not_called()


def test_relayout_skips_the_shared_module(monkeypatch):
    target, draft = _models()
    relaid = []

    def fake_row_major(t):
        relaid.append(t)
        return t

    monkeypatch.setattr(runner_utils, "_to_row_major", fake_row_major)
    monkeypatch.setattr(runner_utils, "_install_entry_layout_pin", lambda: None)
    runner_utils.relayout_rope_caches(target)
    assert len(relaid) == 1
    skip = {id(m) for m in target.modules()}
    with mock.patch.object(runner_utils, "logger") as log:
        runner_utils.relayout_rope_caches(draft, skip_modules=skip)
    assert len(relaid) == 2 and relaid[1] is draft.own.cos_sin_cache
    assert log.info.call_args.args[-2:] == (1, 1)  # one changed, one shared


@pytest.mark.parametrize(
    "flag", ["TPU_ROPE_CACHE_TRUNCATE", "TPU_ROPE_CACHE_ROW_MAJOR"]
)
def test_the_flags_default_off(monkeypatch, flag):
    monkeypatch.delenv(flag, raising=False)
    assert envs.environment_variables[flag]() is False
