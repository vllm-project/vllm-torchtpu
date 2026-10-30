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

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig
from vllm.config import CacheConfig, VllmConfig

from tests.models.vllm.test_qwen4_exp_parallel import config
from vllm_torchtpu.models.vllm.qwen4_exp.config import (
    Qwen4ExpForCausalLMConfig,
    TPUQwen4ExpConfig,
    validate_config,
)
from vllm_torchtpu.models.vllm.qwen4_exp.inputs import dummy_query_lengths
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpPooledQSABackend


@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "fp8", "fp8_e4m3"])
@pytest.mark.parametrize("dimension", [64, 128])
def test_dual_store_shape_keeps_both_kv_payloads(dtype, dimension):
    shape = Qwen4ExpPooledQSABackend.get_kv_cache_shape(3, 96, 3, dimension, dtype)
    if dtype in ("auto", "bfloat16"):
        assert shape == (3, 96, 3, 2, 128)
        assert torch.empty(shape, device="meta", dtype=torch.bfloat16).nbytes == 442368
    else:
        assert shape == (3, 96, 2, 4, 128)
        assert (
            torch.empty(shape, device="meta", dtype=torch.float8_e4m3fn).nbytes
            == 294912
        )


@pytest.mark.parametrize("prefix", [False, True])
def test_configuration_is_idempotent_after_pool_layout(monkeypatch, prefix):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.setattr(
        Qwen4ExpForCausalLMConfig,
        "verify_and_update_config",
        staticmethod(lambda cfg: None),
    )
    cfg = config()
    cfg.cache_config = CacheConfig(
        cache_dtype="bfloat16", enable_prefix_caching=prefix, mamba_cache_mode="align"
    )
    TPUQwen4ExpConfig.verify_and_update_config(cfg)
    first = (
        cfg.cache_config.block_size,
        cfg.cache_config.mamba_page_size_padded,
        cfg.cache_config.num_gpu_blocks_override,
    )
    TPUQwen4ExpConfig.verify_and_update_config(cfg)
    assert first == (
        cfg.cache_config.block_size,
        cfg.cache_config.mamba_page_size_padded,
        cfg.cache_config.num_gpu_blocks_override,
    )


def test_quantization_metadata_has_a_clear_mapping_error():
    cfg = config()
    cfg.model_config.hf_config = PretrainedConfig.from_dict(
        {"quantization_config": None}
    )
    with pytest.raises(ValueError, match="quantization_config mapping"):
        validate_config(cfg)


def test_missing_quantization_metadata_has_a_clear_mapping_error():
    cfg = config()
    cfg.model_config.hf_config = PretrainedConfig()
    with pytest.raises(ValueError, match="quantization_config mapping"):
        validate_config(cfg)


def test_hash_only_options_keep_shared_defaults():
    from vllm_torchtpu.models.model_hooks import qwen4_options

    class HashOnly:
        def compute_hash(self) -> str:
            return "hash-only"

    cfg = config()
    cfg.additional_config = HashOnly()
    assert qwen4_options(cast(VllmConfig, cfg)) == {}
    validate_config(cfg)


def test_valid_geometry_has_no_checkpoint_page_or_token_bucket_gate():
    cfg = config()
    cfg.cache_config.block_size = 256
    cfg.model_config.max_model_len = 127
    cfg.scheduler_config.max_num_batched_tokens = 8193
    validate_config(cfg)


@pytest.mark.parametrize(
    "tokens,capacities,expected",
    [(8192, [2048] * 4, [2048] * 4), (8, [3, 4], [3, 4]), (8, [1, 8], [1, 7])],
)
def test_dummy_lengths_respect_each_context_capacity(tokens, capacities, expected):
    actual = dummy_query_lengths(tokens, len(capacities), len(capacities), capacities)
    assert actual == expected


@pytest.mark.parametrize(
    "tokens,context_limit,rows,prefix",
    [(8, 3, 2, None), (8, 4, 2, 1), (8192, 2048, 16, None)],
)
def test_dummy_adapter_caps_live_rows_and_pads_the_bucket(
    monkeypatch, tokens, context_limit, rows, prefix
):
    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import inputs
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache

    captured = {}

    def context(metadata, *args, **kwargs):
        captured["metadata"] = metadata
        return nullcontext()

    def forward(**kwargs):
        captured.update(kwargs)
        return torch.zeros((tokens, 4), dtype=torch.bfloat16), None

    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    monkeypatch.setattr(inputs, "set_forward_context", context)
    monkeypatch.setattr(
        inputs, "set_vllm_model_wrapper_context", lambda **kwargs: nullcontext()
    )
    monkeypatch.setattr(inputs, "synchronize_tensors", lambda *args, **kwargs: None)
    runner = SimpleNamespace(
        device="cpu",
        _pp_is_first=True,
        _record_input_staging_fence=Mock(),
        supports_mm_inputs=False,
        input_batch=SimpleNamespace(num_reqs=0),
        model_config=SimpleNamespace(hf_text_config=q4exp_text_config()),
        cache_config=SimpleNamespace(block_size=128),
        scheduler_config=SimpleNamespace(max_num_seqs=rows),
        max_model_len=context_limit,
        most_model_len=context_limit,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        vllm_config=SimpleNamespace(),
        mesh=None,
        _dp_num_tokens_across_dp=lambda count: torch.tensor([count], dtype=torch.int32),
        forward_model=forward,
    )
    adapter = inputs.Qwen4ExpAdapter(runner)
    adapter.cache = cast(
        Qwen4ExpPooledCache,
        SimpleNamespace(
            allocate=Mock(),
            pages=1 + rows * ((context_limit + 127) // 128),
            dummy_metadata=lambda metadata, active, **kwargs: {"attention": metadata},
        ),
    )
    adapter.dummy_run(tokens, rows, profile_prefix=prefix)
    kwargs = captured["model_kwargs"]
    starts = kwargs["query_start_loc"]
    contexts = kwargs["context_lens"]
    lengths = starts[1:] - starts[:-1]
    live = min(tokens, rows * context_limit - (prefix or 0))
    assert int(starts[-1]) == live
    assert torch.all(lengths + contexts <= context_limit)
    assert kwargs["valid_tokens"].tolist() == [True] * live + [False] * (tokens - live)
    expected = [
        position
        for context, length in zip(contexts.tolist(), lengths.tolist())
        for position in range(context, context + length)
    ]
    expected += [0] * (tokens - live)
    assert captured["positions"].tolist() == expected
    assert {"state_slots", "main_block_table", "compressed_block_table"}.isdisjoint(
        kwargs
    )
