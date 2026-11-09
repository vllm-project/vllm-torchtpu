# SPDX-License-Identifier: Apache-2.0
"""DSpark sampler registration binds compile-time choices outside kernel tracing."""

from unittest.mock import Mock

import jax.numpy as jnp
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import dspark_sampling


def test_sampler_option_specializes_op_and_stays_bound(monkeypatch):
    built = []

    def build(name, fn, **kwargs):
        op = Mock()
        built.append((name, fn))
        return op

    monkeypatch.setattr(dspark_sampling.pallas, "jax_op", build)
    sample = Mock(return_value="tokens")
    monkeypatch.setattr(dspark_sampling, "samples", sample)
    dspark_sampling._initialize_samples.cache_clear()
    try:
        selected = []
        for enabled in (False, True, False):
            monkeypatch.setattr(envs, "TPU_DSPARK_PIPELINED_MARKOV", enabled)
            selected.append(dspark_sampling.initialize_samples())
        assert selected[0] is selected[2]
        assert selected[0] is not selected[1]
        assert len(built) == 2 and built[0][0] != built[1][0]
        for enabled, (_, fn) in zip((False, True), built, strict=True):
            assert fn(None, None, None, None) == "tokens"
            assert sample.call_args.kwargs["pipelined"] is enabled
    finally:
        dspark_sampling._initialize_samples.cache_clear()


@pytest.mark.parametrize("pipelined", [False, True])
def test_each_slot_receives_the_registered_option(monkeypatch, pipelined):
    slot = Mock(return_value=jnp.zeros((2,), dtype=jnp.int32))
    monkeypatch.setattr(dspark_sampling, "sample_slot", slot)
    result = dspark_sampling.samples(
        jnp.zeros((10, 4)),
        jnp.zeros((4, 8)),
        jnp.zeros((4, 8)),
        jnp.zeros((2,), dtype=jnp.int32),
        pipelined=pipelined,
    )
    assert result.shape == (2, 5)
    assert [call.kwargs["slot"] for call in slot.call_args_list] == list(range(5))
    assert all(call.kwargs["pipelined"] is pipelined for call in slot.call_args_list)
