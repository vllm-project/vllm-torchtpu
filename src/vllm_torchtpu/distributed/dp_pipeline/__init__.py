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
"""Pipeline-parallel prefill across data-parallel engines (TPU_DP_PIPELINE_PREFILL).

A long prompt that arrives while other DP engines are idle is prefilled as a
pipeline over them. Every engine holds the whole model, so stage ``s`` runs the
decoder layers ``[bounds[s], bounds[s + 1])`` on each chunk of the prompt and
hands the chunk's hidden states to the engine running the next layers. The
engine running the last layers owns the request: it samples the first token
and, once the other stages' K/V pages have reached it, decodes alone.

* ``plan``: which engines run which layers, carried in the request's
  ``SamplingParams.extra_args``.
* ``kv_plan``: which K/V pages each helper sends the owner.
* ``transport``: the hand-offs through host shared memory. Dense DP engines run
  in separate torch_tpu slices and share no device collective.
* ``core/dp_lb_exact_inflight.py``: the API server picks the engines and sends
  each one its copy of the request.
* ``core/dp_pipeline_scheduler.py``: each engine schedules every chunk of the
  prompt alone, so all of them cut the same chunks.
* ``runner/dp_pipeline_stage.py``: the runner runs a stage.
* ``models/vllm/gemma4_patch.py``: the layer ranges as compiled programs.
"""
