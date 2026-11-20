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
"""TPU_MINIMAX_M3_FP8_DRAFT_LINEARS: which linears keep e4m3 weights in the
target and in the DSpark draft config. Off (default) = the previous rule."""

import pytest
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu import envs
from vllm_torchtpu.models.minimax_m3 import quant

BASE = {
    "weight_block_size": [1, quant.MXFP8_BLOCK],
    "activation_scheme": "dynamic",
    "moe_quant_algo": "NVFP4",
    "moe_quant_format": "nvfp4-pack-quantized",
    "group_size": quant._NVFP4_GROUP,
}
OLD = quant._DSPARK_PREFIX[0]


class _Linear(LinearBase):
    def __init__(self):  # no weights: only the isinstance check matters
        pass


def _cfg(draft):
    ignored = [OLD + "fc"] if draft else ["lm_head"]
    return quant.MiniMaxM3QuantConfig.from_config(dict(BASE, ignored_layers=ignored))


def _keeps_fp8(cfg, prefix):
    method = cfg._get_checkpoint_quant_method(_Linear(), prefix)
    assert isinstance(method, quant.MiniMaxM3Mxfp8LinearMethod)
    return method.keep_fp8


@pytest.fixture
def env(monkeypatch):
    def set_(linears, draft):
        monkeypatch.setattr(envs, "TPU_MINIMAX_M3_FP8_LINEARS", linears, raising=False)
        monkeypatch.setattr(envs, "TPU_MINIMAX_M3_FP8_DRAFT_LINEARS", draft)

    return set_


PREFIXES = ("model.layers.0.self_attn.qkv_proj", "model.layers.0.mlp.gate_up_proj")


@pytest.mark.parametrize("linears", [False, True])
def test_off_is_previous_rule(env, linears):
    env(linears, False)
    target, draft = _cfg(False), _cfg(True)
    assert not draft.fp8_linears
    for p in PREFIXES:
        assert _keeps_fp8(target, p) == linears
        assert not _keeps_fp8(draft, p)


def test_on_keeps_only_draft_mlp(env):
    env(True, True)
    target, draft = _cfg(False), _cfg(True)
    assert draft.is_draft and not target.is_draft
    assert _keeps_fp8(draft, "model.layers.0.mlp.gate_up_proj")
    assert _keeps_fp8(draft, "model.layers.0.mlp.down_proj")
    assert not _keeps_fp8(draft, "model.layers.0.self_attn.qkv_proj")
    assert not _keeps_fp8(draft, "model.layers.0.self_attn.o_proj")
    for p in PREFIXES:  # target unchanged
        assert _keeps_fp8(target, p)


def test_on_needs_fp8_linears(env):
    env(False, True)
    assert not _keeps_fp8(_cfg(True), "model.layers.0.mlp.gate_up_proj")
