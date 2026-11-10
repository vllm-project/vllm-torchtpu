# SPDX-License-Identifier: Apache-2.0
from vllm_torchtpu.distributed.pp_wave import drain_before_handoff


def test_one_chip_pcp2_stage_drains():
    assert drain_before_handoff(1, 2, environ={}) is True


def test_one_chip_pcp1_stage_keeps_the_pipelined_launch():
    assert drain_before_handoff(1, 1, environ={}) is False


def test_multi_chip_stage_drains():
    assert drain_before_handoff(2, 1, environ={}) is True


def test_override_wins_both_ways():
    assert drain_before_handoff(1, 2, environ={"VLLM_PP_WAVE_ALWAYS_DRAIN": "0"}) is False
    assert drain_before_handoff(1, 1, environ={"VLLM_PP_WAVE_ALWAYS_DRAIN": "1"}) is True
