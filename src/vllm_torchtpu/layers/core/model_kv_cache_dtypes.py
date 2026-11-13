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
"""KV cache dtypes that model directories add for their own caches.

A model package registers a `--kv-cache-dtype` name from its plugin
`register`, together with the architectures that accept it and the torch
dtype that the shared attention layers store for their (non model-owned)
caches under that name. The platform then admits the name for those
architectures only, and `TPU_STR_DTYPE_TO_TORCH_DTYPE` resolves it.
"""

import torch

# name -> torch dtype of the shared attention caches under that name.
MODEL_KV_CACHE_DTYPES: dict[str, torch.dtype] = {}
_ARCHITECTURES: dict[str, set[str]] = {}


def register_model_kv_cache_dtype(
    name: str, attention_dtype: torch.dtype, *architectures: str
) -> None:
    """Admit `--kv-cache-dtype name` for `architectures`."""
    registered = MODEL_KV_CACHE_DTYPES.setdefault(name, attention_dtype)
    if registered != attention_dtype:
        raise ValueError(
            f"KV cache dtype {name!r} is already registered as {registered}."
        )
    _ARCHITECTURES.setdefault(name, set()).update(architectures)


def is_model_kv_cache_dtype_supported(name: str, architectures: list[str]) -> bool:
    return any(a in _ARCHITECTURES.get(name, ()) for a in architectures)
