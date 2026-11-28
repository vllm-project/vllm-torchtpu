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
import logging

import pytest
import torch
import vllm.envs as vllm_envs

from vllm_torchtpu.runner import compile_watch


@pytest.fixture
def fresh_watch(monkeypatch):
    monkeypatch.setattr(compile_watch, "_installed", False)
    monkeypatch.setattr(torch._logging, "set_logs", lambda **kwargs: None)
    yield
    from torch._dynamo.callback import callback_handler

    callback_handler.clear()


def test_post_warmup_compile_is_off_without_the_recompilation_check(
    fresh_watch, monkeypatch
):
    monkeypatch.setattr(vllm_envs, "VLLM_XLA_CHECK_RECOMPILATION", False)
    compile_watch.install_post_warmup_compile_log()
    assert not compile_watch._installed


def test_post_warmup_compile_logs_one_warning_per_compile(
    fresh_watch, monkeypatch, caplog
):
    monkeypatch.setattr(vllm_envs, "VLLM_XLA_CHECK_RECOMPILATION", True)
    compile_watch.install_post_warmup_compile_log()

    @torch.compile(backend="eager", fullgraph=True, dynamic=False)
    def f(x):
        return x + 1

    with caplog.at_level(logging.WARNING):
        f(torch.ones(3))
        f(torch.ones(3))  # cached: no second compile
        f(torch.ones(5))  # new shape: a recompile
    lines = [
        r.getMessage()
        for r in caplog.records
        if "Post-warmup dynamo compile" in r.getMessage()
    ]
    assert len(lines) == 2, lines
    assert "#1" in lines[0] and "#2" in lines[1]
