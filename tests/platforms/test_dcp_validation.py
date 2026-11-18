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
"""Static DCP admission: the two TPU contracts vLLM does not check.

A DCP group must hold one KV head, and the Raiden admission has no
dp x tp x dcp consumer layout. Each violation would be a wrong answer at
runtime, so what is pinned here is that startup refuses it with the admissible
set named, and admits every layout that keeps them, the PCP manager's
DCP = TP x PCP included.
"""

import itertools
from types import SimpleNamespace

import pytest
import torch

# Initialize vLLM's plugin registry before importing the plugin package.
import vllm  # noqa: F401

from vllm_torchtpu.platforms.dcp_validation import (
    DcpStaticSupportValidator,
    admissible_dcp_sizes,
)


def _vllm_config(
    *,
    dcp_size=1,
    tp_size=8,
    pcp_size=1,
    dp_size=1,
    kv_heads=2,
    use_mla=False,
    kv_role=None,
):
    model_config = None
    if kv_heads is not None:
        model_config = SimpleNamespace(
            get_total_num_kv_heads=lambda: kv_heads, use_mla=use_mla
        )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            tensor_parallel_size=tp_size,
            prefill_context_parallel_size=pcp_size,
            data_parallel_size=dp_size,
        ),
        kv_transfer_config=(
            None if kv_role is None else SimpleNamespace(kv_role=kv_role)
        ),
    )


def test_admissible_sizes_follow_ranks_per_head():
    # Qwen3.5-397B: 2 KV heads.
    assert admissible_dcp_sizes(8, 2) == (1, 2, 4)
    assert admissible_dcp_sizes(16, 2) == (1, 2, 4, 8)
    assert admissible_dcp_sizes(32, 2) == (1, 2, 4, 8, 16)
    # Distinct heads on every rank: no group of two ranks shares a head.
    assert admissible_dcp_sizes(1, 2) == (1,)
    assert admissible_dcp_sizes(4, 8) == (1,)
    # Unknown model geometry: no restriction beyond vLLM's own.
    assert admissible_dcp_sizes(8, 0) == (1,)
    # Over TP x PCP: the PCP ranks of one TP rank hold its heads.
    assert admissible_dcp_sizes(2, 2, pcp_size=4) == (1, 2, 4)
    assert admissible_dcp_sizes(2, 1, pcp_size=4) == (1, 2, 4, 8)
    assert admissible_dcp_sizes(4, 8, pcp_size=2) == (1, 2)


def _vllm_dcp_groups(tp, pcp, dcp):
    """vLLM's DCP groups over one TP x PCP domain, built as
    vllm.distributed.parallel_state.initialize_model_parallel builds them
    (layout ExternalDP x DP x PP x PCP x TP; DCP transposes PCP and TP)."""
    all_ranks = torch.arange(tp * pcp).reshape(1, 1, 1, pcp, tp)
    dcp_ranks = all_ranks.transpose(-1, -2) if dcp > 1 else all_ranks
    return [g.tolist() for g in dcp_ranks.reshape(-1, dcp).unbind(0)]


def _heads_of(rank, tp, heads):
    """The KV heads global rank `rank` holds: vLLM's TP sharding of the heads
    (replicated when TP exceeds them), the same on every PCP rank."""
    tp_rank = rank % tp
    if tp >= heads:
        return frozenset({tp_rank // (tp // heads)})
    per_rank = heads // tp
    return frozenset(range(tp_rank * per_rank, (tp_rank + 1) * per_rank))


@pytest.mark.parametrize(
    "tp,pcp,heads",
    [
        (tp, pcp, heads)
        for tp, pcp, heads in itertools.product(
            (1, 2, 4, 8, 16, 32), (1, 2, 4, 8), (1, 2, 4, 8, 16)
        )
        if tp % heads == 0 or heads % tp == 0
    ],
)
def test_admissible_sizes_are_vllms_groups_that_hold_one_head(tp, pcp, heads):
    one_head = {
        dcp
        for dcp in range(1, tp * pcp + 1)
        if tp * pcp % dcp == 0
        and all(
            len({_heads_of(r, tp, heads) for r in group}) == 1
            for group in _vllm_dcp_groups(tp, pcp, dcp)
        )
    }
    assert set(admissible_dcp_sizes(tp, heads, pcp)) == one_head


@pytest.mark.parametrize(
    "tp,dcp",
    [(8, 1), (8, 2), (8, 4), (16, 8), (32, 16), (32, 4)],
)
def test_plan_layouts_are_admitted(tp, dcp):
    config = DcpStaticSupportValidator.validate_platform_config(
        _vllm_config(tp_size=tp, dcp_size=dcp, kv_role="kv_consumer")
    )
    assert config.dcp_size == dcp
    assert config.enabled is (dcp > 1)


@pytest.mark.parametrize("tp,dcp", [(8, 8), (16, 16), (32, 32), (4, 4)])
def test_group_crossing_kv_heads_is_rejected(tp, dcp):
    with pytest.raises(ValueError, match="must share one KV head"):
        DcpStaticSupportValidator.validate_platform_config(
            _vllm_config(tp_size=tp, dcp_size=dcp)
        )


def test_qwen3_5_397b_at_tp8_admits_exactly_dcp_1_2_4():
    admitted = set()
    for dcp in (1, 2, 4, 8):
        try:
            DcpStaticSupportValidator.validate_platform_config(
                _vllm_config(tp_size=8, dcp_size=dcp, kv_heads=2)
            )
        except ValueError:
            continue
        admitted.add(dcp)
    assert admitted == {1, 2, 4}


@pytest.mark.parametrize("tp,pcp,dcp", [(2, 4, 8), (1, 4, 4), (4, 2, 8)])
def test_pcp_manager_layouts_are_admitted(tp, pcp, dcp):
    # The PCP manager's DCP = TP x PCP on a sparse MLA model (the docs'
    # GLM-5.2 example is TP2 / PCP4 / DCP8). MLA checkpoints report their
    # query heads as num_key_value_heads (128 for DeepSeek-V3), but the cache
    # is one latent every rank holds.
    config = DcpStaticSupportValidator.validate_platform_config(
        _vllm_config(tp_size=tp, pcp_size=pcp, dcp_size=dcp, kv_heads=128, use_mla=True)
    )
    assert config.dcp_size == dcp and config.num_kv_heads == 1


def test_mla_without_pcp_admits_every_divisor_of_tp():
    for dcp in (2, 4, 8):
        DcpStaticSupportValidator.validate_platform_config(
            _vllm_config(tp_size=8, dcp_size=dcp, kv_heads=128, use_mla=True)
        )


def test_gqa_group_over_tp_x_pcp_crossing_heads_is_rejected():
    # 2 KV heads at TP2 x PCP4: a DCP group of 8 spans both TP ranks.
    DcpStaticSupportValidator.validate_platform_config(
        _vllm_config(tp_size=2, pcp_size=4, dcp_size=4, kv_heads=2)
    )
    with pytest.raises(ValueError, match="must share one KV head"):
        DcpStaticSupportValidator.validate_platform_config(
            _vllm_config(tp_size=2, pcp_size=4, dcp_size=8, kv_heads=2)
        )


def test_dcp_must_divide_tp_x_pcp_before_the_head_rule():
    with pytest.raises(ValueError, match="divisible"):
        DcpStaticSupportValidator.validate_platform_config(
            _vllm_config(tp_size=8, dcp_size=3)
        )
    with pytest.raises(ValueError, match="divisible"):
        DcpStaticSupportValidator.validate_platform_config(
            _vllm_config(tp_size=2, pcp_size=4, dcp_size=16, use_mla=True)
        )


def test_no_model_config_keeps_only_the_vllm_rule():
    config = DcpStaticSupportValidator.validate_platform_config(
        _vllm_config(tp_size=8, dcp_size=8, kv_heads=None)
    )
    assert config.total_num_kv_heads == 0


def test_attention_dp_consumer_with_dcp_is_refused_until_admission_names_it():
    with pytest.raises(NotImplementedError, match="dp x tp x dcp"):
        DcpStaticSupportValidator.validate_platform_config(
            _vllm_config(tp_size=4, dcp_size=2, dp_size=2, kv_role="kv_consumer")
        )
    # Without a KV transfer there is no Raiden admission to disagree with.
    DcpStaticSupportValidator.validate_platform_config(
        _vllm_config(tp_size=4, dcp_size=2, dp_size=2)
    )


def test_dcp_one_is_a_no_op():
    config = DcpStaticSupportValidator.validate_platform_config(
        _vllm_config(tp_size=8, dcp_size=1, dp_size=2, kv_role="kv_consumer")
    )
    assert not config.enabled
