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

`fake_runner(...)` carries every attribute `RunnerView` and the runtime's
startup checks read, with serving-like defaults (TP=1, no draft, no
connectors), so a plugin's `create` / `bind` / `decide` can run on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def _pop_int(overrides: dict[str, object], key: str, default: int) -> int:
    """Pop an integer override from `overrides` with `default`."""
    val = overrides.pop(key, default)
    if isinstance(val, bool) or not isinstance(val, int):
        msg = f"fake_vllm_config {key} must be int, got {val!r}"
        raise TypeError(msg)
    return val


def fake_vllm_config(
    *,
    megakernel: object = None,
    block_size: int = 256,
    cp_kv_cache_interleave_size: int = 1,
    kv_cache_dtype: object = "auto",
    **overrides: object,
) -> SimpleNamespace:
    """Build a structured CPU `VllmConfig` stand-in matching `VllmConfigLike`."""
    user_specified = bool(overrides.pop("user_specified_block_size", False))
    dcp_interleave = _pop_int(
        overrides, "dcp_kv_cache_interleave_size", cp_kv_cache_interleave_size
    )
    hf_config = overrides.pop("hf_config", None)
    is_hybrid = bool(overrides.pop("is_hybrid", False))
    tp_size = _pop_int(overrides, "tp_size", 1)
    dp_size = _pop_int(overrides, "dp_size", 1)
    pp_size = _pop_int(overrides, "pp_size", 1)
    pcp_size = _pop_int(overrides, "pcp_size", 1)
    dcp_size = _pop_int(overrides, "dcp_size", 1)
    if overrides:
        unknown = sorted(overrides)
        msg = f"Unsupported fake_vllm_config overrides: {unknown}"
        raise TypeError(msg)
    model_cfg = SimpleNamespace(
        hf_config=hf_config,
        enable_sleep_mode=False,
        multimodal_config=None,
        quantization=None,
        is_hybrid=is_hybrid,
        dtype="bfloat16",
        get_num_kv_heads=lambda *_: 8,
        get_head_size=lambda: 128,
    )
    parallel_cfg = SimpleNamespace(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        pipeline_parallel_size=pp_size,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=int(cp_kv_cache_interleave_size),
        dcp_kv_cache_interleave_size=dcp_interleave,
        enable_eplb=False,
        enable_expert_parallel=False,
    )
    cache_cfg = SimpleNamespace(
        block_size=int(block_size),
        user_specified_block_size=user_specified,
        enable_prefix_caching=False,
        cache_dtype=kv_cache_dtype,
        mamba_ssm_cache_dtype="auto",
        mamba_cache_mode="none",
        mamba_block_size=int(block_size),
        mamba_page_size_padded=0,
    )
    add_cfg = {} if megakernel is None else {"megakernel": megakernel}
    return SimpleNamespace(
        model_config=model_cfg,
        parallel_config=parallel_cfg,
        cache_config=cache_cfg,
        speculative_config=None,
        kv_transfer_config=None,
        load_config=SimpleNamespace(),
        device_config=SimpleNamespace(device=SimpleNamespace(type="cpu")),
        quant_config=None,
        additional_config=add_cfg,
    )


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
    max_pages_per_seq: int | None = None,
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
    **extra: Any,
) -> SimpleNamespace:
    """A CPU runner stand-in; `extra` sets (or overrides) any attribute."""
    import torch

    spec_cfg = (
        SimpleNamespace(num_speculative_tokens=int(speculative_tokens))
        if speculative_tokens
        else None
    )
    pairs = tuple((int(r), int(b)) for r, b in req_block_pairs)
    max_pages = int(max_pages_per_seq if max_pages_per_seq is not None else pairs[0][1])

    def attention_kernel_shapes(max_model_len: bool) -> tuple[int, int, int]:
        r, b = pairs[0] if max_model_len or len(pairs) == 1 else pairs[1]
        return r, (max_pages if max_model_len else b), 0

    model_cfg = SimpleNamespace(
        hf_config=hf_config,
        enable_sleep_mode=False,
        multimodal_config=multimodal_config,
    )
    dp_size = int(extra.pop("dp_size", 1))
    pp_size = int(extra.pop("pp_size", 1))
    pcp_size = int(extra.pop("pcp_size", 1))
    dcp_size = int(extra.pop("dcp_size", 1))
    dp_lockstep = bool(extra.pop("dp_lockstep", False))
    parallel_cfg = SimpleNamespace(
        tensor_parallel_size=int(tp_size),
        data_parallel_size=dp_size,
        pipeline_parallel_size=pp_size,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
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
        uses_mrope=bool(uses_mrope),
        enforce_eager=bool(enforce_eager),
        _unified_kv_layout=bool(unified_kv_layout),
        supports_mm_inputs=bool(supports_mm_inputs),
        lora_config=None,
        _pp_is_first=True,
        _pp_is_last=pp_size == 1,
        _pcp_enabled=pcp_size > 1,
        _dcp_enabled=dcp_size > 1,
        _dp_lockstep_enabled=lambda: dp_lockstep,
        _megakernel=None,
    )
    if ckpt_window is not None:
        runner._mamba_ckpt_window = int(ckpt_window)
    for k, v in extra.items():
        setattr(runner, k, v)
    return runner


__all__ = ["fake_runner", "fake_vllm_config"]
