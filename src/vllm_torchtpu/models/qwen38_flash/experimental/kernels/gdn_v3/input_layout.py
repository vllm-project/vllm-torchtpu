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
# Provenance: model-owned copy of src/vllm_torchtpu/kernels/gdn/v3/input_layout.py
#   (reference at 33ee547ba947, Qwen3.8 changes through
#   e1edc3ac3e401a40ef63b69afa1e06fc80962134).
#   Owned by models/qwen38_flash from this copy on.

import dataclasses

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Conv1DInput:
    """Prepared FP32 [tokens, 1, features] input and projection output dtype.

    The dtype is static pytree metadata, so preparing a BF16 projection does
    not change the output dtype of the shared convolution/recurrence kernel.
    """

    values: jax.Array
    output_dtype: jnp.dtype = dataclasses.field(metadata={"static": True})
