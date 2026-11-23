# SPDX-License-Identifier: Apache-2.0
"""q38 HC-HIGHROWS-2 CMP-S: TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID runs Benliang Wang's
cmp decode HC schedule with the served kernel's sigmoid in the mix term.

The numerics of record are judged by a separate locked harness; these tests pin the
switch: default off, refused without TPU_QWEN4_HC_DECODE_COMPARE or with
TPU_QWEN4_HC_DECODE_WIDE, op names gain _cmps, the auto path takes the sigmoid
schedule at cmp rows, the flag keys the compile cache, and the served kernel's
_check_bf16 tolerance holds (interpret mode).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import hc_decode_cmp
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix

BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN


def _inputs(seed, tokens):
    keys = jax.random.split(jax.random.key(seed), 8)
    bf16 = jnp.bfloat16
    residual = (4.0 * jax.random.normal(keys[0], (tokens, WIDTH))).astype(bf16)
    block = jax.random.normal(keys[1], (tokens, HIDDEN)).astype(bf16)
    injection = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (tokens, BRANCHES)))
    weights = [
        (0.3 * jax.random.normal(keys[3], (WIDTH,))).astype(bf16),
        (0.02 * jax.random.normal(keys[4], (LOWRANK, WIDTH))).astype(bf16),
        (0.05 * jax.random.normal(keys[5], (WIDTH, LOWRANK))).astype(bf16),
        (0.02 * jax.random.normal(keys[6], (BRANCHES, WIDTH))).astype(bf16),
    ]
    return residual, block, injection.astype(jnp.float32), weights


def _f32(x):
    return np.asarray(jnp.asarray(x, jnp.float32))


def _choice(monkeypatch, compare, sigmoid, wide=0):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    monkeypatch.setenv("TPU_QWEN4_HC_DECODE_COMPARE", "1" if compare else "0")
    monkeypatch.setenv("TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID", "1" if sigmoid else "0")
    monkeypatch.setenv("TPU_QWEN4_HC_DECODE_WIDE", str(wide))
    return vllm_adapter._decode_mix_choice()


def test_default_off_and_registered():
    assert "TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID" in envs.environment_variables
    assert hc_decode_cmp.SIGMOID_OPTS == hc_decode_cmp.DEFAULT_OPTS + (("sigmoid", True),)


def test_choice_follows_the_flags(monkeypatch):
    fn, tag = _choice(monkeypatch, True, False)
    assert fn is hc_decode_cmp.hc_decode_mix_auto and tag == "_cmp"
    fn, tag = _choice(monkeypatch, True, True)
    assert isinstance(fn, functools.partial) and fn.func is hc_decode_cmp.hc_decode_mix_auto
    assert fn.keywords == {"sigmoid": True} and tag == "_cmps"


@pytest.mark.parametrize("compare,wide,match", [(False, 0, "needs"), (False, 1, "exclusive"), (True, 2, "exclusive")])
def test_bad_settings_are_refused(monkeypatch, compare, wide, match):
    with pytest.raises(ValueError, match=match):
        _choice(monkeypatch, compare, True, wide)


def test_compile_factors_key_the_flag(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler

    monkeypatch.delenv("TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID", raising=False)
    unset = tpu_compiler._tpu_compile_env_factors()
    assert unset["TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID"] is False
    monkeypatch.setenv("TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID", "1")
    on = tpu_compiler._tpu_compile_env_factors()
    assert {k for k in on if on[k] != unset.get(k)} == {"TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID"}


@pytest.mark.parametrize("tokens", [32, 112])
def test_auto_sigmoid_is_the_sigmoid_schedule(tokens):
    residual, block, injection, weights = _inputs(tokens, tokens)
    static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS, block=block, prev_injection=injection)
    interpret = pltpu.InterpretParams(uninitialized_memory="nan")
    auto = hc_decode_cmp.hc_decode_mix_auto(*[residual] + weights, max_tile=512, sigmoid=True, interpret=interpret, **static)
    direct = hc_decode_cmp.hc_decode_mix_cmp(*[residual] + weights, _opts=hc_decode_cmp.SIGMOID_OPTS, interpret=interpret, **static)
    tanh = hc_decode_cmp.hc_decode_mix_cmp(*[residual] + weights, interpret=interpret, **static)
    for a, d in zip(auto, direct):
        np.testing.assert_array_equal(_f32(a), _f32(d))
    # the sigmoid schedule is a different program from the tanh one: the mixed rows differ somewhere
    assert not np.array_equal(_f32(direct[0]), _f32(tanh[0]))


def test_cmps_matches_served_kernel_to_its_tolerance():
    tokens = 48
    residual, block, injection, weights = _inputs(tokens, tokens)
    static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS, block=block, prev_injection=injection)
    interpret = pltpu.InterpretParams(uninitialized_memory="nan")
    ref = hc_decode_mix(residual, *weights, max_tile=512, interpret=interpret, **static)
    out = hc_decode_cmp.hc_decode_mix_cmp(residual, *weights, _opts=hc_decode_cmp.SIGMOID_OPTS, interpret=interpret, **static)
    a, e = _f32(out[0]), _f32(ref[0])
    d = np.abs(a - e)
    assert np.isfinite(a).all() and d.max() <= 2.0**-7 * np.abs(e).max() and (d > 0).mean() <= 0.02
    np.testing.assert_array_equal(_f32(out[2]), _f32(ref[2]))
