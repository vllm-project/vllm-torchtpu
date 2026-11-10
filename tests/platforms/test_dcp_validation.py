# SPDX-License-Identifier: Apache-2.0
"""Static DCP admission: the three TPU contracts vLLM does not check.

A DCP group must share one KV head, the Qwen3.5 megakernel follows that rule, and
the Raiden admission has no dp x tp x dcp consumer layout. Each violation
would be a wrong answer at runtime, so what is pinned here is that startup
refuses it with the admissible set named.
"""

from types import SimpleNamespace

import pytest

# Initialize vLLM's plugin registry before importing the plugin package.
import vllm  # noqa: F401

from vllm_torchtpu.platforms.dcp_validation import (
    DcpStaticSupportValidator,
    admissible_dcp_sizes,
    validate_decode_context_parallel_config,
)


def _vllm_config(
    *,
    dcp_size=1,
    tp_size=8,
    pcp_size=1,
    dp_size=1,
    kv_heads=2,
    kv_role=None,
):
    model_config = None
    if kv_heads is not None:
        model_config = SimpleNamespace(get_total_num_kv_heads=lambda: kv_heads)
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


@pytest.mark.parametrize(
    "tp,dcp",
    [(8, 1), (8, 2), (8, 4), (16, 8), (32, 16), (32, 4)],
)
def test_plan_layouts_are_admitted(tp, dcp):
    config = validate_decode_context_parallel_config(
        _vllm_config(tp_size=tp, dcp_size=dcp, kv_role="kv_consumer")
    )
    assert config.dcp_size == dcp
    assert config.enabled is (dcp > 1)


@pytest.mark.parametrize("tp,dcp", [(8, 8), (16, 16), (32, 32), (4, 4)])
def test_group_crossing_kv_heads_is_rejected(tp, dcp):
    with pytest.raises(ValueError, match="must share one KV head"):
        validate_decode_context_parallel_config(_vllm_config(tp_size=tp, dcp_size=dcp))


def test_dcp_must_divide_tp_before_the_head_rule():
    with pytest.raises(ValueError, match="divisible"):
        validate_decode_context_parallel_config(_vllm_config(tp_size=8, dcp_size=3))


def test_no_model_config_keeps_only_the_vllm_rule():
    config = validate_decode_context_parallel_config(
        _vllm_config(tp_size=8, dcp_size=8, kv_heads=None)
    )
    assert config.total_num_kv_heads == 0


def test_attention_dp_consumer_with_dcp_is_refused_until_admission_names_it():
    with pytest.raises(NotImplementedError, match="dp x tp x dcp"):
        validate_decode_context_parallel_config(
            _vllm_config(tp_size=4, dcp_size=2, dp_size=2, kv_role="kv_consumer")
        )
    # Without a KV transfer there is no Raiden admission to disagree with.
    validate_decode_context_parallel_config(
        _vllm_config(tp_size=4, dcp_size=2, dp_size=2)
    )


def test_dcp_one_is_a_no_op():
    config = DcpStaticSupportValidator.validate_platform_config(
        _vllm_config(tp_size=8, dcp_size=1, dp_size=2, kv_role="kv_consumer")
    )
    assert not config.enabled
