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
"""Test helpers for megakernel plugins: a CPU stand-in for `TPUModelRunner`.

`fake_runner(...)` sets every attribute `RunnerView` and the runtime's
startup checks read, with serving-like defaults (TP=1, no draft, no
connectors, the KV cache initialized), so a plugin's `create` / `bind` /
`decide` can run on CPU. `RunnerView` reads them directly: a runner stand-in
must set them all. Each attribute is one `TPUModelRunner` (or vLLM's
`GPUModelRunner`, its base) sets.

This module ships in the package, not under `tests/`, because plugin tests
outside this repository use it (directly or through `testkit`).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def fake_runner(
    *,
    model: Any = None,
    hf_config: Any = None,
    megakernel: Any = None,
    tp_size: int = 1,
    device: Any = None,
    block_size: int = 256,
    speculative_tokens: int = 0,
    token_paddings: tuple[int, ...] = (16, 32, 64),
    req_block_pairs: tuple[tuple[int, int], ...] = ((8, 4),),
    ckpt_window: int | None = None,
    mamba_slot_read_offsets: Any = None,
    uses_mrope: bool = False,
    enforce_eager: bool = False,
    unified_kv_layout: bool = True,
    enable_prefix_caching: bool = False,
    kv_transfer_config: Any = None,
    enable_eplb: bool = False,
    enable_expert_parallel: bool = False,
    kv_cache_dtype: str = "auto",
    mamba_ssm_cache_dtype: Any = "auto",
    mamba_cache_mode: str = "none",
    supports_mm_inputs: bool = False,
    multimodal_config: Any = None,
    mesh: Any = None,
    kv_cache_initialized: bool = True,
    **extra: Any,
) -> SimpleNamespace:
    """A CPU runner stand-in; `extra` sets (or overrides) any attribute.

    `req_block_pairs` are the attention kernel's `(sequences, pages per
    sequence)` per shape class (max-model-len first; a second entry is the
    most-model-len class), with `block_size` as its page size.
    `kv_cache_initialized=False` is the runner before `initialize_kv_cache`
    (`kv_cache_config` is `None`), as `create` sees it.
    """
    import torch

    spec_cfg = (
        SimpleNamespace(num_speculative_tokens=int(speculative_tokens))
        if speculative_tokens
        else None
    )
    pairs = tuple((int(r), int(b)) for r, b in req_block_pairs)

    def attention_kernel_shapes(max_model_len: bool) -> tuple[int, int, int]:
        r, b = pairs[0] if max_model_len or len(pairs) == 1 else pairs[1]
        return r, b, int(block_size)

    model_cfg = SimpleNamespace(
        hf_config=hf_config,
        enable_sleep_mode=False,
        multimodal_config=multimodal_config,
    )
    parallel_cfg = SimpleNamespace(
        tensor_parallel_size=int(tp_size),
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_eplb=bool(enable_eplb),
        enable_expert_parallel=bool(enable_expert_parallel),
    )
    cache_cfg = SimpleNamespace(
        block_size=int(block_size),
        enable_prefix_caching=bool(enable_prefix_caching),
        cache_dtype=kv_cache_dtype,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
        mamba_cache_mode=mamba_cache_mode,
    )
    add_cfg = {} if megakernel is None else {"megakernel": megakernel}
    vllm_cfg = SimpleNamespace(
        model_config=model_cfg,
        parallel_config=parallel_cfg,
        cache_config=cache_cfg,
        speculative_config=spec_cfg,
        kv_transfer_config=kv_transfer_config,
        additional_config=add_cfg,
    )
    runner = SimpleNamespace(
        vllm_config=vllm_cfg,
        model_config=model_cfg,
        parallel_config=parallel_cfg,
        cache_config=cache_cfg,
        speculative_config=spec_cfg,
        model=model,
        device=torch.device("cpu") if device is None else device,
        mesh=mesh,
        num_tokens_paddings=list(token_paddings),
        _attention_kernel_shapes=attention_kernel_shapes,
        most_model_len=None if len(pairs) == 1 else 1,
        mamba_slot_read_offsets=mamba_slot_read_offsets,
        # `initialize_kv_cache` sets it; `None` before.
        kv_cache_config=SimpleNamespace() if kv_cache_initialized else None,
        # As TPUModelRunner: the draft's tokens + 1 with the unified KV
        # layout, else 1.
        _mamba_ckpt_window=int(
            ckpt_window
            if ckpt_window is not None
            else (int(speculative_tokens) + 1 if unified_kv_layout else 1)
        ),
        uses_mrope=bool(uses_mrope),
        enforce_eager=bool(enforce_eager),
        _unified_kv_layout=bool(unified_kv_layout),
        supports_mm_inputs=bool(supports_mm_inputs),
        lora_config=None,
        _pp_is_first=True,
        _pp_is_last=True,
        _pcp_enabled=False,
        _dp_lockstep_enabled=lambda: False,
    )
    for k, v in extra.items():
        setattr(runner, k, v)
    return runner


__all__ = ["fake_runner"]
