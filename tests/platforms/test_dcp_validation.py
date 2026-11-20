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

A DCP group must hold one KV head, and the Raiden connector has no
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
from vllm.config import KVTransferConfig

from vllm_torchtpu.platforms.dcp_validation import (
    DcpStaticSupportValidator,
    admissible_dcp_sizes,
)

_TPU_CONNECTOR = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


def _connector(name, module, role, extra=None):
    return {
        "kv_connector": name,
        "kv_connector_module_path": module,
        "kv_role": role,
        "kv_connector_extra_config": extra or {},
    }


def _multi(role, *children):
    return _connector(
        "TPUMultiConnector",
        "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector",
        role,
        {"connectors": list(children)},
    )


_RAIDEN = _connector("TPURaidenConnector", _TPU_CONNECTOR, "kv_consumer")
_OFFLOAD = _connector(
    "TPURaidenOffloadingConnector", "vllm_torchtpu.offload.raiden_connector", "kv_both"
)
# TPUConnector picks its backend: ZMQ-shm unless the Raiden one is switched on.
_TPU_ZMQ = _connector("TPUConnector", _TPU_CONNECTOR, "kv_consumer")


def _vllm_config(
    *,
    dcp_size=1,
    tp_size=8,
    pcp_size=1,
    dp_size=1,
    kv_heads=2,
    use_mla=False,
    connector=None,
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
            None if connector is None else KVTransferConfig(**connector)
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
        _vllm_config(tp_size=tp, dcp_size=dcp, connector=_RAIDEN)
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


@pytest.fixture
def no_raiden_env(monkeypatch):
    """TPUConnector's environment switches for its Raiden backend, unset."""
    monkeypatch.delenv("TPU_KV_RESHARD_TRANSPORT", raising=False)
    monkeypatch.delenv("TPU_USE_RAIDEN_CONNECTOR", raising=False)


@pytest.mark.parametrize(
    "connector,name",
    [
        (_RAIDEN, "TPURaidenConnector"),
        (_multi("kv_producer", _RAIDEN, _OFFLOAD), "TPURaidenConnector"),
        (
            _connector(
                "TPUConnector",
                _TPU_CONNECTOR,
                "kv_consumer",
                {"use_raiden_connector": True},
            ),
            "TPUConnector",
        ),
    ],
    ids=["raiden", "raiden_in_a_multi_connector", "tpu_connector_raiden_backend"],
)
def test_attention_dp_with_dcp_is_refused_where_a_connector_runs_raiden(
    no_raiden_env, connector, name
):
    with pytest.raises(NotImplementedError, match=f"{name} runs the Raiden"):
        DcpStaticSupportValidator.validate_platform_config(
            _vllm_config(tp_size=4, dcp_size=2, dp_size=2, connector=connector)
        )


@pytest.mark.parametrize(
    "connector",
    [None, _OFFLOAD, _multi("kv_both", _OFFLOAD), _TPU_ZMQ],
    ids=["no_kv_transfer", "offload", "offload_in_a_multi_connector", "zmq_shm"],
)
def test_attention_dp_with_dcp_is_admitted_without_a_raiden_connector(
    no_raiden_env, connector
):
    """None of these connectors runs the Raiden layout naming."""
    config = DcpStaticSupportValidator.validate_platform_config(
        _vllm_config(tp_size=4, dcp_size=2, dp_size=2, connector=connector)
    )
    assert config.enabled and config.dp_size == 2


def _vllm_multi(role, *children):
    """vLLM's own MultiConnector, by its registry name."""
    return {
        "kv_connector": "MultiConnector",
        "kv_role": role,
        "kv_connector_extra_config": {"connectors": list(children)},
    }


_OFFLOAD_SIZED = {
    **_OFFLOAD,
    "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
}


def _tpu_connector(use_raiden):
    return _connector(
        "TPUConnector",
        _TPU_CONNECTOR,
        "kv_consumer",
        {"use_raiden_connector": use_raiden, "late_poll": True},
    )


@pytest.mark.parametrize(
    "connector",
    [
        _multi("kv_producer", _RAIDEN, _OFFLOAD_SIZED),
        _multi("kv_consumer", _tpu_connector(True), _OFFLOAD),
        _multi("kv_consumer", _tpu_connector(False), _OFFLOAD_SIZED),
        _multi("kv_producer", _multi("kv_producer", _RAIDEN, _OFFLOAD), _OFFLOAD),
        _vllm_multi("kv_consumer", _tpu_connector(True), _OFFLOAD_SIZED),
    ],
    ids=[
        "raiden_and_sized_offload",
        "tpu_connector_raiden_on",
        "tpu_connector_raiden_off",
        "nested_multi",
        "vllm_multi_connector",
    ],
)
def test_multiconnector_children_are_the_configs_vllm_builds_them_from(connector):
    """The check reads a MultiConnector's children from the public KV transfer
    config fields. vLLM's MultiConnector builds the same child configs (in its
    private _get_connector_classes_and_configs, compared here so that a change
    in vLLM fails this test rather than the check), each with its entry's own
    extra config, where use_raiden_connector lives."""
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    from vllm_torchtpu.platforms.dcp_validation import _child_configs

    config = _vllm_config(connector=connector)
    ours = [child.kv_transfer_config for child in _child_configs(config)]
    vllms = [
        child.kv_transfer_config
        for _, child in MultiConnector._get_connector_classes_and_configs(config)
    ]
    assert ours == vllms
    entries = connector["kv_connector_extra_config"]["connectors"]
    assert [c.kv_connector_extra_config for c in ours] == [
        e["kv_connector_extra_config"] for e in entries
    ]
    assert {c.engine_id for c in ours} == {config.kv_transfer_config.engine_id}


def test_multiconnector_children_cannot_set_their_own_engine_id():
    """vLLM's MultiConnector refuses an entry with its own engine_id, and so
    does the check's reading of it."""
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    from vllm_torchtpu.platforms.dcp_validation import _child_configs

    own_id = {**_OFFLOAD, "engine_id": "offload-0"}
    config = _vllm_config(connector=_multi("kv_producer", _RAIDEN, own_id))
    with pytest.raises(TypeError, match="engine_id"):
        _child_configs(config)
    with pytest.raises(TypeError, match="engine_id"):
        MultiConnector._get_connector_classes_and_configs(config)


def test_tpu_connector_raiden_transport_switch_is_refused(no_raiden_env, monkeypatch):
    """TPU_KV_RESHARD_TRANSPORT=raiden puts every TPUConnector on Raiden."""
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    with pytest.raises(NotImplementedError, match="TPUConnector runs the Raiden"):
        DcpStaticSupportValidator.validate_platform_config(
            _vllm_config(tp_size=4, dcp_size=2, dp_size=2, connector=_TPU_ZMQ)
        )


def test_pcp_engines_leave_attention_dp_with_dcp_to_the_pcp_checks(no_raiden_env):
    """pcp_validation refuses DCP with PCP on the default path, and every
    Raiden admission topology refuses PCP > 1 under DP > 1, so this check
    does not."""
    config = DcpStaticSupportValidator.validate_platform_config(
        _vllm_config(tp_size=4, pcp_size=2, dcp_size=2, dp_size=2, connector=_RAIDEN)
    )
    assert config.enabled


def test_dcp_one_is_a_no_op():
    config = DcpStaticSupportValidator.validate_platform_config(
        _vllm_config(tp_size=8, dcp_size=1, dp_size=2, connector=_RAIDEN)
    )
    assert not config.enabled
