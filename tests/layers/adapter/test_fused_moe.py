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

from enum import Enum
from types import SimpleNamespace

import pytest
import torch
import vllm.config

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe
from vllm_torchtpu.layers.adapter.fused_moe import get_fused_moe_activation
from vllm_torchtpu.layers.core import fused_moe_gmm


class _Activation(Enum):
    SILU = "silu"


@pytest.mark.parametrize("activation", ["silu", _Activation.SILU])
def test_get_fused_moe_activation_preserves_standard_activations(activation):
    assert get_fused_moe_activation(activation, object()) == "silu"


@pytest.mark.parametrize(
    "linear_beta,expected",
    [
        (None, "situ:4.0:none"),
        (25.0, "situ:4.0:25.0"),
    ],
)
def test_get_fused_moe_activation_encodes_situ(linear_beta, expected):
    moe_config = SimpleNamespace(
        activation_situ_beta=4.0,
        activation_situ_linear_beta=linear_beta,
    )

    assert get_fused_moe_activation("situ", moe_config) == expected


def test_get_fused_moe_activation_requires_situ_beta():
    moe_config = SimpleNamespace(
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )

    with pytest.raises(AssertionError):
        get_fused_moe_activation("situ", moe_config)


def test_get_fused_moe_activation_compiles():
    @torch.compile(fullgraph=True)
    def compiled_fn(x, activation):
        act_str = get_fused_moe_activation(activation, None)
        if act_str == "silu":
            return x * 2
        return x

    res = compiled_fn(torch.ones(1).to("tpu"), _Activation.SILU)
    assert res.cpu() == 2.0


def _fake_vllm_config(max_num_seqs, compile_sizes, dp, draft_tokens=0):
    spec = None
    if draft_tokens:
        spec = SimpleNamespace(num_speculative_tokens=draft_tokens)
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=spec,
        compilation_config=SimpleNamespace(compile_sizes=compile_sizes),
        parallel_config=SimpleNamespace(data_parallel_size=dp),
    )


@pytest.mark.parametrize(
    "max_num_seqs,compile_sizes,dp,draft_tokens,expected",
    [
        # 48 requests per engine, 2 engines: 96 gathered tokens x top-10.
        (48, [48, 1024], 2, 0, 960),
        (12, [12, 1024], 8, 0, 960),
        # The step pads to its token bucket.
        (48, [16, 32, 64, 1024], 2, 0, 1280),
        # Verify steps carry the draft tokens.
        (48, [48, 192, 1024], 2, 3, 3840),
        # A step larger than every bucket runs in the largest one.
        (4096, [48, 1024], 1, 0, 10240),
        (48, None, 2, 0, 960),
    ],
)
def test_routed_rows_per_decode_step_follow_the_decode_bucket(
    monkeypatch, max_num_seqs, compile_sizes, dp, draft_tokens, expected
):
    cfg = _fake_vllm_config(max_num_seqs, compile_sizes, dp, draft_tokens)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: cfg)
    assert fused_moe._routed_rows_per_decode_step(10) == expected


def test_routed_rows_per_decode_step_zero_without_config(monkeypatch):
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: None)
    assert fused_moe._routed_rows_per_decode_step(10) == 0


@pytest.mark.parametrize("use_ep,expected", [(True, 960), (False, 511)])
def test_fused_moe_op_threshold_covers_the_ep_decode_step(
    monkeypatch, use_ep, expected
):
    monkeypatch.setattr(envs, "ONEHOT_MOE_PERMUTE_THRESHOLD", None, raising=False)
    sc_info = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=16)
    monkeypatch.setattr(
        fused_moe_gmm.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(sparse_core=sc_info),
    )
    cfg = _fake_vllm_config(48, [48, 1024], 2)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr(fused_moe, "_fused_moe_kernel_cache", {})
    built = {}

    class _Op:
        def register_fake(self, fn):
            pass

    def _jax_op(name, fn):
        built["fn"] = fn
        return _Op()

    monkeypatch.setattr(fused_moe.pallas, "jax_op", _jax_op)
    fused_moe._build_fused_moe_custom_op(
        topk=10, activation="silu", use_ep=use_ep, skip_padded_tokens=False
    )
    assert built["fn"].keywords["onehot_moe_permute_threshold"] == expected
