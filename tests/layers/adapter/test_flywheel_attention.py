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
"""--attention-backend flywheel: its configuration check, per-step host
metadata, and vLLM backend class."""

import inspect
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flywheel_tpu.flash_attn_interface import PAGE_SIZE_MULTIPLE
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_torchtpu.layers.adapter import attention_backend as ab
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend
from vllm_torchtpu.layers.adapter.flywheel_attention import (
    FlywheelAttentionBackend,
    FlywheelMetadata,
    FlywheelPallasAttentionBackend,
    FlywheelPallasAttentionBackendImpl,
)
from vllm_torchtpu.layers.core import flywheel_attention_interface as fa


def _vllm_config(**overrides):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            user_specified_block_size=False, block_size=256, cache_dtype="auto"
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            is_hybrid=False,
            dtype=torch.bfloat16,
            get_head_size=lambda: 128,
            architecture="Qwen3ForCausalLM",
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=None,
    )
    for path, value in overrides.items():
        obj = config
        *parents, attr = path.split(".")
        for parent in parents:
            obj = getattr(obj, parent)
        setattr(obj, attr, value)
    return config


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
def test_check_accepts_a_dense_model(cache_dtype):
    FlywheelAttentionBackend().check_config(
        _vllm_config(**{"cache_config.cache_dtype": cache_dtype})
    )


def test_check_accepts_qwen_gdn_on_the_compact_pool(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    config = _vllm_config(
        **{
            "model_config.is_hybrid": True,
            "model_config.architecture": "Qwen3_5MoeForConditionalGeneration",
        }
    )

    FlywheelAttentionBackend().check_config(config)


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ({"model_config.use_mla": True}, "MLA"),
        (
            {
                "model_config.is_hybrid": True,
                "model_config.architecture": "KimiLinearForCausalLM",
            },
            "hybrid models other than Qwen GDN",
        ),
        (
            {
                "model_config.is_hybrid": True,
                "model_config.architecture": "Qwen3_5MoeForConditionalGeneration",
            },
            "unified block pool",
        ),
        ({"model_config.get_head_size": lambda: 64}, "head_dim 64"),
        ({"model_config.dtype": torch.float16}, "model dtype"),
        ({"cache_config.cache_dtype": "int8"}, "kv cache dtype"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"kv_transfer_config": object()}, "KV transfer"),
        ({"parallel_config.decode_context_parallel_size": 2}, "decode context"),
        ({"parallel_config.prefill_context_parallel_size": 2}, "prefill context"),
        ({"parallel_config.pipeline_parallel_size": 2}, "pipeline"),
        (
            {
                "cache_config.user_specified_block_size": True,
                "cache_config.block_size": 64,
            },
            "block size 64",
        ),
    ],
)
def test_check_rejects(override, problem, monkeypatch):
    monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", raising=False)
    with pytest.raises(ValueError, match=problem):
        FlywheelAttentionBackend().check_config(_vllm_config(**override))


def _step(
    num_decode, q_lens, kv_lens, *, num_tokens, max_num_seqs=8, max_num_tokens=256
):
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    block_table = np.arange(len(q_lens) * 4, dtype=np.int32).reshape(len(q_lens), 4)
    return ab.HostStepInputs(
        num_decode=num_decode,
        query_start_loc=cu_q_lens,
        seq_lens=np.array(kv_lens, np.int32),
        block_tables={0: block_table},
        page_sizes={0: 128},
        num_tokens=num_tokens,
        max_num_seqs=max_num_seqs,
        max_num_tokens=max_num_tokens,
    )


def test_prepare_step_matches_the_warmup_shapes():
    backend = FlywheelAttentionBackend()

    metadata = backend.prepare_step(_step(1, [1, 40], [10, 300], num_tokens=128))[0]
    warmup = backend.warmup_metadata({0: 128}, max_num_seqs=8, max_num_tokens=256)[0]

    for real, dummy in zip(metadata, warmup, strict=True):
        assert real.shape == dummy.shape and real.dtype == dummy.dtype
    assert not warmup.kv_write_num_slices.any() and not warmup.kernel_index.any()


def test_prepare_step_shapes_do_not_follow_the_bucket():
    # One backbone trace serves every token bucket, with only token axes
    # dynamic: a table sized by the step's bucket fed the 256-token bucket's
    # executable a 16-token-bucket-shaped table.
    backend = FlywheelAttentionBackend()
    decode = backend.prepare_step(_step(2, [1, 1], [9, 300], num_tokens=16))
    mixed = backend.prepare_step(_step(1, [1, 40], [9, 300], num_tokens=256))

    assert decode[0].kv_write_slices.shape == mixed[0].kv_write_slices.shape


def test_prepare_step_writes_only_the_rows_after_the_decode_rows():
    step = _step(1, [1, 40], [10, 300], num_tokens=128)

    metadata = FlywheelAttentionBackend().prepare_step(step)[0]

    # The prefill request's 40 tokens, positions 260..299, sit in its page 2.
    num_slices = int(metadata.kv_write_num_slices[0])
    slices = metadata.kv_write_slices.reshape(3, -1)[:, :num_slices]
    assert num_slices == 1
    assert slices[:, 0].tolist() == [6 * 128 + 260 - 256, 1, 40]
    assert fa.step_kernels(128)[int(metadata.kernel_index[0])] == "extend"


def test_backend_class_pages_are_whole_kernel_pages():
    assert FlywheelPallasAttentionBackend.get_supported_kernel_block_sizes() == [
        MultipleOf(PAGE_SIZE_MULTIPLE)
    ]
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=32768))

    # RPA's long-context page size is 16 tokens; FlyWheel indexes whole pages.
    assert FlywheelPallasAttentionBackend.get_page_size(config) == 128


@pytest.mark.parametrize(
    ("dtype", "num_kv_heads", "cache_heads"),
    [
        (torch.bfloat16, 2, 4),  # [k0, v0, k1, v1]
        (torch.bfloat16, 1, 2),
        (torch.float8_e4m3fn, 2, 4),  # [k0, k1, v0, v1]
        (torch.float8_e4m3fn, 1, 4),  # one zero pad head
    ],
)
def test_backend_class_allocates_flywheel_pages_of_rpa_size(
    dtype, num_kv_heads, cache_heads
):
    args = dict(block_size=128, num_kv_heads=num_kv_heads, head_size=128)

    shape = FlywheelPallasAttentionBackend.get_kv_cache_shape(
        num_blocks=10, cache_dtype_str=dtype, **args
    )

    assert shape == (10, 128, cache_heads, 128)
    assert FlywheelPallasAttentionBackend.get_kv_cache_page_size_bytes(
        cache_dtype_str=dtype, **args
    ) == PallasAttentionBackend.get_kv_cache_page_size_bytes(
        cache_dtype_str=dtype, **args
    )


def test_kernel_entry_takes_every_argument_rpa_passes():
    # PallasAttentionBackendImpl._build_rpa_kernel binds these keywords, and
    # forward passes RPA's arrays, then the backend's, positionally.
    params = inspect.signature(
        FlywheelPallasAttentionBackendImpl._kernel_entry
    ).parameters
    keywords = {
        "mesh",
        "sliding_window",
        "sm_scale",
        "soft_cap",
        "q_scale",
        "k_scale",
        "v_scale",
        "skip_kv_update",
        "use_causal_mask",
    }
    assert keywords <= {name for name, p in params.items() if p.kind is p.KEYWORD_ONLY}
    positional = [name for name, p in params.items() if p.kind is not p.KEYWORD_ONLY]
    assert tuple(positional[9:]) == FlywheelMetadata._fields


def test_flywheel_is_registered_by_the_plugin_entry_point():
    from vllm_torchtpu.layers import register_layers

    register_layers()

    member = AttentionBackendEnum["FLYWHEEL"]
    assert member.get_class() is FlywheelPallasAttentionBackend
    config = SimpleNamespace(attention_config=SimpleNamespace(backend=member))
    assert isinstance(ab.create_attention_backend(config), FlywheelAttentionBackend)


def test_registering_loads_no_flywheel_kernels():
    code = (
        "import sys\n"
        "from vllm_torchtpu.layers import register_layers\n"
        "register_layers()\n"
        "assert 'flywheel_tpu' not in sys.modules, 'flywheel_tpu loaded'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
