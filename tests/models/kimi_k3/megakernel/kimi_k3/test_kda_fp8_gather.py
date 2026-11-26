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


from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import KernelConfig


def test_should_keep_bf16_window_when_env_unset(monkeypatch):
    monkeypatch.delenv("K3_PREFILL_KDA_FP8_GATHER", raising=False)

    assert KernelConfig().prefill_kda_fp8_gather is False


def test_should_select_fp8_window_when_env_set(monkeypatch):
    monkeypatch.setenv("K3_PREFILL_KDA_FP8_GATHER", "1")

    assert KernelConfig().prefill_kda_fp8_gather is True
