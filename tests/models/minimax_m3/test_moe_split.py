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
"""MiniMax M3's guard on the hierarchical expert split (TPU_MOE_HIERARCHICAL_EP)."""

import types

import pytest

from vllm_torchtpu.models.minimax_m3 import moe


def _config(load_format):
    return types.SimpleNamespace(
        load_config=types.SimpleNamespace(load_format=load_format)
    )


@pytest.mark.parametrize("load_format", ["runai_streamer", "auto", "dummy"])
def test_runai_refused_only_with_the_split(monkeypatch, load_format) -> None:
    monkeypatch.setattr(moe, "get_current_vllm_config", lambda: _config(load_format))
    monkeypatch.setattr(moe, "hierarchical_split_or_none", lambda: (4, 1, 2, 0))
    if load_format == "runai_streamer":
        with pytest.raises(ValueError, match="runai_streamer"):
            moe._refuse_hierarchical_runai_load()
    else:
        moe._refuse_hierarchical_runai_load()


def test_runai_allowed_without_the_split(monkeypatch) -> None:
    monkeypatch.setattr(
        moe, "get_current_vllm_config", lambda: _config("runai_streamer")
    )
    monkeypatch.setattr(moe, "hierarchical_split_or_none", lambda: None)
    moe._refuse_hierarchical_runai_load()
