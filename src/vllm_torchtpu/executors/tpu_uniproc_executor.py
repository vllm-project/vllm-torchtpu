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
"""TpuUniProcExecutor: single-process executor that pins hybrid KV sizing."""

from vllm.v1.executor.uniproc_executor import UniProcExecutor

from vllm_torchtpu.executors.kv_block_override import KVBlockOverrideMixin


class TpuUniProcExecutor(KVBlockOverrideMixin, UniProcExecutor):
    """Single-process TPU executor that pins the worker-sized hybrid KV block
    count on the engine config, as the multi-process executors do."""
