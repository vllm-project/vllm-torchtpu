# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_torchtpu.platforms.tpu_platform import _drop_default_expert_placement


def _pc(tp=1, dp=1, pcp=1, ep=True):
    return SimpleNamespace(
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        prefill_context_parallel_size=pcp,
        enable_expert_parallel=ep,
    )


def test_pp4_pcp2_ep2_stage_turns_the_ep8_default_off():
    env = {}
    assert _drop_default_expert_placement(_pc(pcp=2), env)
    assert env == {"TPU_EXPERT_PLACEMENT": ""}


@pytest.mark.parametrize(
    "pc", [_pc(tp=8), _pc(dp=8), _pc(tp=2, dp=4), _pc(pcp=8), _pc(pcp=2, ep=False)]
)
def test_an_ep8_group_or_no_ep_keeps_the_default(pc):
    env = {}
    assert not _drop_default_expert_placement(pc, env)
    assert env == {}


def test_an_explicit_table_is_left_alone():
    env = {"TPU_EXPERT_PLACEMENT": "/some/table.json"}
    assert not _drop_default_expert_placement(_pc(pcp=2), env)
    assert env == {"TPU_EXPERT_PLACEMENT": "/some/table.json"}
