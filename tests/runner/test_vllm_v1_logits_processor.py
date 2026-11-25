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

import importlib.metadata

from vllm_torchtpu.runner.vllm_v1.sample import logits_processor


def test_tpu_loads_no_custom_logits_processors(monkeypatch):
    # TPUModelRunner applies none, so neither installed plugins nor configured
    # classes are loaded.
    def no_plugin_lookup(*args, **kwargs):
        raise AssertionError("installed logits processor plugins were looked up")

    monkeypatch.setattr(importlib.metadata, "entry_points", no_plugin_lookup)
    custom = ("missing.module:Proc", logits_processor.MinPLogitsProcessor)
    assert logits_processor._load_custom_logitsprocs(custom) == []
