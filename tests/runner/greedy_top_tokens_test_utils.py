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
"""Test helper shared by the greedy top tokens and draft head tests."""

from vllm_torchtpu import envs


def set_greedy_top_tokens(monkeypatch) -> None:
    """Turn TPU_GREEDY_TOP_TOKENS on in the environment, where envs reads it.
    A monkeypatch.setattr(envs, ...) undo leaves the value it read in the
    module dict, which shadows envs.__getattr__ for the rest of the session
    (see tests/layers/adapter/test_mla_attention.py), so any such entry an
    earlier test left is dropped first."""
    if "TPU_GREEDY_TOP_TOKENS" in vars(envs):
        monkeypatch.delattr(envs, "TPU_GREEDY_TOP_TOKENS")
    monkeypatch.setenv("TPU_GREEDY_TOP_TOKENS", "1")
