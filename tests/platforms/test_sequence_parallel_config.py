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
"""Tests for sequence-parallel MoE platform compilation configuration."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def configure_sp_moe():
    """Stub optional dependencies and yield _configure_sequence_parallel_moe."""
    stubs = {
        mod: MagicMock()
        for mod in (
            "torch_tpu",
            "torch_tpu._internal",
            "torch_tpu._internal.compile",
            "torch_tpu._internal.utils",
            "portpicker",
        )
        if mod not in sys.modules
    }
    with patch.dict(sys.modules, stubs):
        from vllm_torchtpu.platforms.tpu_platform import (
            _configure_sequence_parallel_moe,
        )

        yield _configure_sequence_parallel_moe


@pytest.fixture
def all_buckets_sequence_parallel(monkeypatch):
    """A threshold of 1 row: no bucket runs replicated, so every one must split."""
    from vllm_torchtpu.layers.adapter import sequence_parallel

    monkeypatch.setattr(sequence_parallel, "DEFAULT_MIN_ROWS", 1)


@pytest.mark.usefixtures("all_buckets_sequence_parallel")
@pytest.mark.parametrize(
    "tp,original,expected",
    [
        (8, [8, 16, 24, 32, 48, 64, 128, 256], [16, 32, 64, 128, 256]),
        (4, [4, 8, 16, 24, 32, 64], [8, 16, 32, 64]),
        (2, [2, 4, 8, 16, 32], [4, 8, 16, 32]),
    ],
)
def test_sequence_parallel_compile_sizes(configure_sp_moe, tp, original, expected):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    compilation_config = SimpleNamespace(compile_sizes=list(original))
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=compilation_config,
        speculative_config=None,
    )
    configure_sp_moe(vllm_config)
    assert compilation_config.compile_sizes == expected


def test_by_default_every_bucket_follows_mains_rule(configure_sp_moe):
    """Until a model follows each step's form, no step runs replicated, so
    every compile size must split over TP as on main."""
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    compilation_config = SimpleNamespace(
        compile_sizes=[4, 8, 16, 24, 32, 64, 96, 512, 1024]
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=compilation_config,
        speculative_config=None,
    )
    configure_sp_moe(vllm_config)
    assert compilation_config.compile_sizes == [16, 32, 64, 512, 1024]


@pytest.mark.parametrize(
    "original,expected",
    [
        # Below a threshold of 512 rows a step runs replicated, so its bucket
        # need not split over TP.
        ([4, 8, 24, 48, 96, 512, 1024], [4, 8, 24, 48, 96, 512, 1024]),
        # From the threshold on it must: 520 // 8 = 65 rows per rank.
        ([24, 48, 520, 1024], [24, 48, 1024]),
    ],
)
def test_sequence_parallel_keeps_buckets_below_min_rows(
    configure_sp_moe, monkeypatch, original, expected
):
    from vllm_torchtpu.layers.adapter import sequence_parallel

    monkeypatch.setattr(sequence_parallel, "DEFAULT_MIN_ROWS", 512)
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    compilation_config = SimpleNamespace(compile_sizes=list(original))
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=compilation_config,
        speculative_config=None,
    )
    configure_sp_moe(vllm_config)
    assert compilation_config.compile_sizes == expected


def test_sequence_parallel_buckets_follow_the_engines_layout(
    configure_sp_moe, monkeypatch
):
    """The threshold comes from the engine's own layout: the platform runs
    before any model-parallel group exists, so it must pass its config."""
    from vllm_torchtpu.layers.adapter import sequence_parallel

    monkeypatch.setitem(sequence_parallel._MIN_ROWS_BY_LAYOUT, (4, 2), 64)
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        data_parallel_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    # 72 rows split as 36 per rank: unaligned, so dropped once 72 >= 64.
    compilation_config = SimpleNamespace(compile_sizes=[8, 24, 48, 72, 96, 1024])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=compilation_config,
        speculative_config=None,
    )
    configure_sp_moe(vllm_config)
    assert compilation_config.compile_sizes == [8, 24, 48, 96, 1024]


@pytest.mark.usefixtures("all_buckets_sequence_parallel")
def test_sequence_parallel_rejects_empty_compile_sizes(configure_sp_moe):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    compilation_config = SimpleNamespace(compile_sizes=[8, 24, 48])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=compilation_config,
        speculative_config=None,
    )
    with pytest.raises(ValueError, match="left no valid compile sizes"):
        configure_sp_moe(vllm_config)


@pytest.mark.usefixtures("all_buckets_sequence_parallel")
def test_sequence_parallel_rejects_largest_size_dropped(configure_sp_moe):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    # 48 is dropped (48 // 8 = 6, not power-of-two 2, 4 or multiple of 8)
    compilation_config = SimpleNamespace(compile_sizes=[16, 32, 48])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=compilation_config,
        speculative_config=None,
    )
    with pytest.raises(ValueError, match="dropped largest compile size 48"):
        configure_sp_moe(vllm_config)


@pytest.mark.parametrize("arch", ["DeepseekV3ForCausalLM", "Qwen3NextForCausalLM"])
def test_sequence_parallel_rejects_unvalidated_arch(configure_sp_moe, arch):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=[arch])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=SimpleNamespace(compile_sizes=[16, 32, 64]),
        speculative_config=None,
    )
    with pytest.raises(
        ValueError,
        match="validated only for Qwen3.5 MoE",
    ):
        configure_sp_moe(vllm_config)


def test_sequence_parallel_admits_mtp(configure_sp_moe):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=SimpleNamespace(compile_sizes=[16, 32, 64]),
        speculative_config=SimpleNamespace(method="mtp"),
    )
    configure_sp_moe(vllm_config)
    assert vllm_config.compilation_config.compile_sizes == [16, 32, 64]


@pytest.mark.parametrize("method", ["eagle", "ngram"])
def test_sequence_parallel_rejects_non_mtp_speculative(configure_sp_moe, method):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=SimpleNamespace(compile_sizes=[16, 32, 64]),
        speculative_config=SimpleNamespace(method=method),
    )
    with pytest.raises(ValueError, match="covers only the MTP drafter"):
        configure_sp_moe(vllm_config)


def test_sequence_parallel_rejects_pipeline_parallel(configure_sp_moe):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=SimpleNamespace(compile_sizes=[16, 32, 64]),
        speculative_config=None,
    )
    with pytest.raises(
        ValueError,
        match="does not support pipeline parallelism",
    ):
        configure_sp_moe(vllm_config)


@pytest.mark.parametrize(
    "tp,dp,ep,expected",
    [
        (8, 1, True, True),  # TP8 without data parallelism
        (2, 4, True, True),  # attention data parallelism over EP ranks
        (4, 2, True, True),
        (1, 8, True, False),  # no TP: nothing to slice
        (2, 4, False, False),  # no EP: the expert sum is never reduced
    ],
)
def test_sequence_parallel_moe_admission(monkeypatch, tp, dp, ep, expected):
    from vllm.config.parallel import ParallelConfig

    import vllm_torchtpu
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    saved = ParallelConfig.__dict__.get("use_sequence_parallel_moe")
    saved_flag = ParallelConfig.__dict__.get("_tpu_no_sp_moe_patch")
    ParallelConfig._tpu_no_sp_moe_patch = False
    try:
        vllm_torchtpu._patch_sequence_parallel_moe_admission()
        admitted = ParallelConfig.__dict__["use_sequence_parallel_moe"].fget
        config = SimpleNamespace(
            enable_expert_parallel=ep,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
        )
        assert admitted(config) is expected
    finally:
        if saved is not None:
            ParallelConfig.use_sequence_parallel_moe = saved
        if saved_flag is not None:
            ParallelConfig._tpu_no_sp_moe_patch = saved_flag
