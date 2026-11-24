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
"""vLLM Model Runner V1 pieces that TPUModelRunner builds on.

vLLM v0.34.0 removes Model Runner V1
(https://github.com/vllm-project/vllm/pull/63373). TPUModelRunner subclasses
V1's GPUModelRunner and uses its InputBatch, BlockTable, EC connector mixin
and NgramProposer, so this package keeps copies of them from vLLM v0.33.0;
utils.py holds the v0.33.0 helpers the copies call that v0.34.0 deletes. The
layout mirrors upstream's vllm/v1/ so each file diffs cleanly against its
source.

Only imports differ from upstream, except in two files. worker/gpu_model_runner.py
keeps only the GPUModelRunner methods TPUModelRunner reaches (including the
ones vLLM's WorkerBase looks up by name) and drops the GPU-only drafter,
sampler, CUDA graph and n-gram GPU code. worker/ec_connector_model_runner_mixin.py
keeps only maybe_save_ec_to_connector. Rules for edits are in
docs/developers_guide/coding_guidance.md (R-3.1). Moving TPUModelRunner onto
Model Runner V2 retires this package:
https://github.com/vllm-project/vllm-torchtpu/issues/3723
"""
