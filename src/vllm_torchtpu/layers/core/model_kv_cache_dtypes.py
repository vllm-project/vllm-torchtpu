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
"""TPU `--kv-cache-dtype` names, including ones model directories add.

A model package registers a `--kv-cache-dtype` name from its plugin
`register`, together with the architectures that accept it and the torch
dtype that the shared attention layers store for their (non model-owned)
caches under that name. The platform then admits the name for those
architectures only, and `TPU_KV_CACHE_DTYPES` (the one name -> dtype map
used by the attention layers, worker, runner and block sizing) resolves it.
"""

from collections import ChainMap

import torch

# Note: TPU can fp8 as storage dtype but doesn't support converting from uint8
# from to fp32 directly. That's why it has a dtype mapping different from GPU
_BUILTIN_KV_CACHE_DTYPES: dict[str, torch.dtype] = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_ds_mla": torch.uint8,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

# name -> torch dtype of the shared attention caches under that name.
MODEL_KV_CACHE_DTYPES: dict[str, torch.dtype] = {}
_ARCHITECTURES: dict[str, set[str]] = {}

# Built-in names first, then the names model directories register.
TPU_KV_CACHE_DTYPES: ChainMap[str, torch.dtype] = ChainMap(
    _BUILTIN_KV_CACHE_DTYPES, MODEL_KV_CACHE_DTYPES
)


def register_model_kv_cache_dtype(
    name: str, attention_dtype: torch.dtype, *architectures: str
) -> None:
    """Admit `--kv-cache-dtype name` for `architectures`.

    Raises:
        ValueError: `name` is not lowercase, is not a `--kv-cache-dtype` value
            vLLM accepts, is a built-in TPU name, or is already registered
            with a different dtype.
    """
    # Local import: vllm.config.cache resolves the platform at import time,
    # and tpu_platform -> tpu_block_size_utils imports this module.
    from vllm.config.cache import is_known_kv_cache_dtype

    if name in _BUILTIN_KV_CACHE_DTYPES:
        raise ValueError(f"KV cache dtype {name!r} is a built-in TPU name.")
    if name != name.lower().strip():
        raise ValueError(f"KV cache dtype {name!r} must be lowercase.")
    if not is_known_kv_cache_dtype(name):
        raise ValueError(f"vLLM does not accept --kv-cache-dtype {name!r}.")
    registered = MODEL_KV_CACHE_DTYPES.setdefault(name, attention_dtype)
    if registered != attention_dtype:
        raise ValueError(
            f"KV cache dtype {name!r} is already registered as {registered}."
        )
    _ARCHITECTURES.setdefault(name, set()).update(architectures)


def is_model_kv_cache_dtype_supported(name: str, architectures: list[str]) -> bool:
    return any(a in _ARCHITECTURES.get(name, ()) for a in architectures)
