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
"""Runs the model folder's own CPU end-to-end test
(`src/vllm_torchtpu/models/qwen3_5/megakernel/tests/`) with this suite."""

from vllm_torchtpu.megakernel_runtime.testkit import mk_fake_runner  # noqa: F401
from vllm_torchtpu.models.qwen3_5.megakernel.tests.test_cpu_e2e import (  # noqa: F401
    test_serves_a_decode_step_on_cpu,
    test_spec_loads_without_registering_anything,
    tiny_runner,
)
