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
"""A TurboQuant KV cache on the GLM-5.2 PP8 prefill configs: the TurboQuant
config check and the pipeline-parallel validator, which
`TpuPlatform.check_and_update_config` both run, accept them, and
TurboQuant's own refusals still hold under pipeline parallelism."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter.turboquant_attention import check_turboquant_config
from vllm_torchtpu.platforms.pp_validation import validate_pipeline_parallel_config

RAIDEN_PRODUCER = SimpleNamespace(
    kv_connector="TPURaidenConnector", is_kv_producer=True
)
# `kv_role` kv_both: a producer too.
RAIDEN_OFFLOAD_BOTH = SimpleNamespace(
    kv_connector="TPURaidenOffloadingConnector", is_kv_producer=True
)


def _pp8(
    kv_transfer=RAIDEN_PRODUCER,
    cache_dtype="turboquant_4bit_nc",
    block_size=256,
    pcp=1,
    dcp=1,
):
    """The PP8 x TP1 prefill engine: MTP K=1 prefill-only, sync scheduling,
    block size 256, GLM-5.2's sparse MLA (index_topk, kv_lora_rank 512)."""
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=8,
            tensor_parallel_size=1,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
            data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        kv_transfer_config=kv_transfer,
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            block_size=block_size,
            user_specified_block_size=True,
            kv_cache_dtype_skip_layers=[],
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            hf_text_config=SimpleNamespace(
                layer_types=None, index_topk=2048, kv_lora_rank=512
            ),
        ),
    )


@pytest.mark.parametrize(
    "kv_transfer",
    [RAIDEN_PRODUCER, RAIDEN_OFFLOAD_BOTH],
    ids=["pd_producer", "offloading_kv_both"],
)
def test_pp8_prefill_config_accepts_turboquant(kv_transfer):
    config = _pp8(kv_transfer)
    check_turboquant_config(config)
    validate_pipeline_parallel_config(config)
    assert config.cache_config.block_size == 256


def test_pp8_turboquant_still_refuses_pcp_stages(monkeypatch):
    from vllm_torchtpu.platforms import pp_validation

    monkeypatch.setattr(pp_validation, "_torch_tpu_builds_stage_meshes", lambda: True)
    config = _pp8(pcp=2)
    validate_pipeline_parallel_config(config)
    with pytest.raises(NotImplementedError, match="PCP"):
        check_turboquant_config(config)


def test_pp8_turboquant_refuses_dcp():
    with pytest.raises(NotImplementedError, match="DCP"):
        check_turboquant_config(_pp8(dcp=2))


def test_pp8_turboquant_refuses_presets_other_than_4bit_nc():
    with pytest.raises(NotImplementedError, match="sparse MLA"):
        check_turboquant_config(_pp8(cache_dtype="turboquant_k8v4"))


def test_pp8_turboquant_refuses_block_sizes_off_32():
    with pytest.raises(ValueError, match="multiple of 32"):
        check_turboquant_config(_pp8(block_size=48))


@pytest.mark.parametrize(
    "name", ["TPU_SPARSE_MLA_NOPE_LAYOUT", "TPU_SPARSE_MLA_ROPE_LAYOUT"]
)
def test_pp8_turboquant_needs_sparsecore_cache_layouts(monkeypatch, name):
    monkeypatch.setenv(name, "tensorcore")
    with pytest.raises(NotImplementedError, match="sparsecore"):
        check_turboquant_config(_pp8())
