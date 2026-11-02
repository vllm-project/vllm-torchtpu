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
"""Inject-projecting HC mix kernel: shapes, admission, and TPU math contract."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.kernels.qwen4_exp.hc_inject_mix import (
    full_mix_inject,
    full_mix_inject_broadcast,
)
from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
    build_full_mix_inject_broadcast_op,
    build_full_mix_inject_op,
)


def inputs(tokens=256, branches=4, hidden=128, lowrank=128):
    rng = np.random.default_rng(41)
    width = branches * hidden
    shapes = (
        (tokens, width),
        (width,),
        (lowrank, width),
        (width, lowrank),
        (branches, width),
    )
    return tuple((rng.normal(size=s) * 0.1).astype(np.float32) for s in shapes)


def raw_words(array):
    """Exact BF16 storage words, signed zeros and payloads included."""
    return np.asarray(array).view(np.uint16)


def reference(x, weight, down, up, inject, branches=4):
    hidden = x.shape[1] // branches
    values = x.astype(jnp.float32).reshape(-1, branches, hidden)
    norm = values * jax.lax.rsqrt(jnp.mean(values**2, -1, keepdims=True) + 1e-6)
    norm = norm.reshape(x.shape)
    norm = (norm + norm * weight.astype(jnp.float32)).astype(jnp.bfloat16)
    mid = (norm.astype(jnp.float32) @ down.astype(jnp.float32).T).astype(jnp.bfloat16)
    mid = jax.nn.silu(mid.astype(jnp.float32) / branches).astype(jnp.bfloat16)
    logits = (norm.astype(jnp.float32) @ inject.astype(jnp.float32).T).astype(
        jnp.bfloat16
    )
    logits_up = (mid.astype(jnp.float32) @ up.astype(jnp.float32).T).astype(
        jnp.bfloat16
    )
    weighted = jax.nn.sigmoid(logits_up.astype(jnp.float32)) * norm.astype(jnp.float32)
    mixed = jnp.mean(weighted.reshape(-1, branches, hidden), axis=1).astype(
        jnp.bfloat16
    )
    return mixed, logits


@pytest.mark.parametrize(
    "tokens,branches,hidden,tile,lowrank",
    [
        (127, 4, 128, 128, 128),
        (0, 4, 128, 128, 128),
        (129, 4, 128, 128, 128),
        (257, 4, 128, 128, 128),
        (256, 4, 128, 0, 128),
        (256, 4, 128, 96, 128),
        (256, 4, 129, 128, 128),
        (256, 4, 0, 128, 128),
        (256, 0, 128, 128, 128),
    ],
)
def test_hc_inject_rejects_invalid_geometry(tokens, branches, hidden, tile, lowrank):
    shape = jax.ShapeDtypeStruct
    width = branches * hidden
    x = shape((tokens, width), jnp.bfloat16)
    weight = shape((width,), jnp.bfloat16)
    down = shape((lowrank, width), jnp.bfloat16)
    up = shape((width, lowrank), jnp.bfloat16)
    inject = shape((max(branches, 1), width), jnp.bfloat16)
    fn = functools.partial(
        full_mix_inject, branches=branches, hidden=hidden, eps=1e-6, token_tile=tile
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, x, weight, down, up, inject)


@pytest.mark.parametrize("operand", range(5))
def test_hc_inject_rejects_non_bf16_operands(operand):
    arrays = [jax.ShapeDtypeStruct(a.shape, jnp.bfloat16) for a in inputs(branches=3)]
    arrays[operand] = jax.ShapeDtypeStruct(arrays[operand].shape, jnp.float32)
    fn = functools.partial(
        full_mix_inject, branches=3, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="BF16"):
        jax.eval_shape(fn, *arrays)


def test_hc_inject_rejects_injected_weight_branch_mismatch():
    shape = jax.ShapeDtypeStruct
    arrays = [
        shape((256, 512), jnp.bfloat16),
        shape((512,), jnp.bfloat16),
        shape((128, 512), jnp.bfloat16),
        shape((512, 128), jnp.bfloat16),
        shape((3, 512), jnp.bfloat16),
    ]
    fn = functools.partial(
        full_mix_inject, branches=4, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, *arrays)


def test_hc_inject_torch_fake_shapes():
    op = build_full_mix_inject_op(4, 128, 1e-6, 128)
    with FakeTensorMode():
        x = torch.empty(256, 512, dtype=torch.bfloat16)
        weight = torch.empty(512, dtype=torch.bfloat16)
        down = torch.empty(128, 512, dtype=torch.bfloat16)
        up = torch.empty(512, 128, dtype=torch.bfloat16)
        inject = torch.empty(4, 512, dtype=torch.bfloat16)
        mixed, logits = op(x, weight, down, up, inject)
        assert mixed.shape == (256, 128)
        assert logits.shape == (256, 4)
        for tensor in (mixed, logits):
            assert tensor.dtype == torch.bfloat16
            assert tensor.device == x.device


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "tokens,branches,hidden,lowrank,tile",
    [
        (128, 4, 128, 128, 128),
        (256, 4, 128, 128, 128),
        (256, 3, 128, 128, 128),
        # lowrank off the hardware row boundary: the packed plane pads the
        # inject rows past the down rows and defines the boundary rows.
        (256, 4, 128, 40, 128),
        (256, 4, 2560, 320, 128),
        (256, 4, 2560, 320, 256),
        (512, 4, 2560, 320, 256),
        (384, 2, 512, 64, 128),
    ],
)
def test_hc_inject_mix_matches_cpu_reference(tokens, branches, hidden, lowrank, tile):
    cpu = jax.devices("cpu")[0]
    tpu = jax.devices("tpu")[0]
    with jax.default_device(cpu):
        arrays = tuple(
            jnp.asarray(a, jnp.bfloat16)
            for a in inputs(tokens, branches, hidden, lowrank)
        )
        expected_mix, expected_logits = reference(*arrays, branches=branches)
    x, weight, down, up, inject = tuple(jax.device_put(a, tpu) for a in arrays)
    # Exact storage snapshots of every read-only operand, taken before the
    # tested calls: float equality would hide signed zeros and BF16 payload
    # differences.
    before = tuple(raw_words(a).copy() for a in (x, weight, down, up, inject))
    mixed, logits = full_mix_inject(
        x,
        weight,
        down,
        up,
        inject,
        branches=branches,
        hidden=hidden,
        eps=1e-6,
        token_tile=tile,
    )
    repeated = full_mix_inject(
        x,
        weight,
        down,
        up,
        inject,
        branches=branches,
        hidden=hidden,
        eps=1e-6,
        token_tile=tile,
    )
    for first, second in zip((mixed, logits), repeated, strict=True):
        np.testing.assert_array_equal(raw_words(first), raw_words(second))
    for actual, expected in [
        (mixed, expected_mix),
        (logits, expected_logits),
    ]:
        a, e = (np.asarray(v, np.float32) for v in (actual, expected))
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01
        assert (
            np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e))
            >= 0.999
        )
    for operand, snapshot in zip((x, weight, down, up, inject), before, strict=True):
        np.testing.assert_array_equal(raw_words(operand), snapshot)


@pytest.mark.tpu_test
def test_hc_inject_rounds_logits_at_bf16_boundary():
    """Injection logits match a standalone BF16 linear over the stored norm.

    With unit rows, zero affine/down/up: the up gate is sigmoid(0)=0.5, so the
    mixed output is 0.5 * mean over branches of the unit norm = 0.5; the
    injection logit is the width sum 512 * 2^-8 = 2.0. Both are exactly
    representable BF16 words, so the kernel must store exactly what a
    standalone linear over the stored normalized value would produce before
    ``combine`` applies the FP32 sigmoid.
    """
    tpu = jax.devices("tpu")[0]
    branches, hidden, lowrank, tokens = 4, 128, 128, 128
    width = branches * hidden
    host = [
        np.ones((tokens, width), np.float32),
        np.zeros((width,), np.float32),
        np.zeros((lowrank, width), np.float32),
        np.zeros((width, lowrank), np.float32),
        np.full((branches, width), 2**-8, np.float32),
    ]
    arrays = [jax.device_put(a, tpu).astype(jnp.bfloat16) for a in host]
    before = tuple(raw_words(a).copy() for a in arrays)
    mixed, logits = full_mix_inject(
        *arrays, branches=branches, hidden=hidden, eps=1e-6, token_tile=128
    )
    # silu(0)=0 zeroes the down/up path, sigmoid(0)=0.5 scales the unit norm,
    # and every injection logit is the full-width sum 512 * 2^-8 = 2.0.
    expected_mixed = np.full((tokens, hidden), 0.5, np.asarray(mixed).dtype)
    expected_logits = np.full((tokens, branches), 2.0, np.asarray(logits).dtype)
    np.testing.assert_array_equal(raw_words(mixed), expected_mixed.view(np.uint16))
    np.testing.assert_array_equal(raw_words(logits), expected_logits.view(np.uint16))
    for operand, snapshot in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(raw_words(operand), snapshot)


@pytest.mark.parametrize(
    "tokens,branches,hidden,tile,lowrank",
    [
        (127, 4, 128, 128, 128),
        (0, 4, 128, 128, 128),
        (129, 4, 128, 128, 128),
        (257, 4, 128, 128, 128),
        (256, 4, 128, 0, 128),
        (256, 4, 128, 96, 128),
        (256, 4, 129, 128, 128),
        (256, 4, 0, 128, 128),
        (256, 0, 128, 128, 128),
    ],
)
def test_hc_inject_broadcast_rejects_invalid_geometry(
    tokens, branches, hidden, tile, lowrank
):
    shape = jax.ShapeDtypeStruct
    width = max(branches * hidden, 1)
    x = shape((tokens, hidden), jnp.bfloat16)
    weight = shape((width,), jnp.bfloat16)
    down = shape((max(lowrank, 1), width), jnp.bfloat16)
    up = shape((width, max(lowrank, 1)), jnp.bfloat16)
    inject = shape((max(branches, 1), width), jnp.bfloat16)
    fn = functools.partial(
        full_mix_inject_broadcast,
        branches=branches,
        hidden=hidden,
        eps=1e-6,
        token_tile=tile,
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, x, weight, down, up, inject)


@pytest.mark.parametrize("operand", range(5))
def test_hc_inject_broadcast_rejects_non_bf16_operands(operand):
    shape = jax.ShapeDtypeStruct
    arrays = [
        shape((256, 128), jnp.bfloat16),
        shape((384,), jnp.bfloat16),
        shape((128, 384), jnp.bfloat16),
        shape((384, 128), jnp.bfloat16),
        shape((3, 384), jnp.bfloat16),
    ]
    arrays[operand] = jax.ShapeDtypeStruct(arrays[operand].shape, jnp.float32)
    fn = functools.partial(
        full_mix_inject_broadcast, branches=3, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="BF16"):
        jax.eval_shape(fn, *arrays)


def test_hc_inject_broadcast_rejects_unexpanded_width_mismatch():
    shape = jax.ShapeDtypeStruct
    arrays = [
        shape((256, 127), jnp.bfloat16),
        shape((384,), jnp.bfloat16),
        shape((128, 384), jnp.bfloat16),
        shape((384, 128), jnp.bfloat16),
        shape((3, 384), jnp.bfloat16),
    ]
    fn = functools.partial(
        full_mix_inject_broadcast, branches=3, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, *arrays)


def test_hc_inject_broadcast_torch_fake_shapes():
    op = build_full_mix_inject_broadcast_op(4, 128, 1e-6, 128)
    with FakeTensorMode():
        x = torch.empty(256, 128, dtype=torch.bfloat16)
        weight = torch.empty(512, dtype=torch.bfloat16)
        down = torch.empty(128, 512, dtype=torch.bfloat16)
        up = torch.empty(512, 128, dtype=torch.bfloat16)
        inject = torch.empty(4, 512, dtype=torch.bfloat16)
        mixed, logits = op(x, weight, down, up, inject)
        assert mixed.shape == (256, 128)
        assert logits.shape == (256, 4)
        for tensor in (mixed, logits):
            assert tensor.dtype == torch.bfloat16
            assert tensor.device == x.device


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "tokens,branches,hidden,lowrank,tile",
    [
        (128, 4, 128, 128, 128),
        (256, 4, 128, 128, 128),
        (256, 3, 128, 128, 128),
        # lowrank off the hardware row boundary: the packed plane pads the
        # inject rows past the down rows and defines the boundary rows.
        (256, 4, 128, 40, 128),
        (256, 4, 2560, 320, 128),
        (512, 4, 2560, 320, 128),
        (256, 4, 2560, 320, 256),
        (384, 2, 512, 64, 128),
    ],
)
def test_hc_inject_broadcast_matches_cpu_reference(
    tokens, branches, hidden, lowrank, tile
):
    """The broadcast mix reproduces the inject mix over the expanded words.

    The compact operand is one branch slice of the same random fixture, and the
    explicit expanded hyper input is built from it by tiling branchwise (a BF16
    copy of the exact words). The unchanged independent reference runs over
    that expanded value with the same weights the kernel receives with the
    compact operand, so both sides see the same input. Matching through the
    L2/cosine oracle proves the broadcast normalization forms each branch's
    affine-scaled value from the same BF16 words the expanded plane would
    carry; the broadcast normalization reduces the variance over ``hidden``
    once where the expanded path reduces each branch's repeated slice, so raw
    words are not compared across the two paths.
    """
    cpu = jax.devices("cpu")[0]
    tpu = jax.devices("tpu")[0]
    with jax.default_device(cpu):
        arrays = tuple(
            jnp.asarray(a, jnp.bfloat16)
            for a in inputs(tokens, branches, hidden, lowrank)
        )
        # The compact operand the kernel actually stages: one branch slice.
        compact = arrays[0][:, :hidden]
        # The expanded plane every branch slice of which repeats it, built from
        # the exact compact words.
        expanded = jnp.tile(compact, (1, branches))
        expected_mix, expected_logits = reference(
            expanded, *arrays[1:], branches=branches
        )
    x, weight, down, up, inject = (
        jax.device_put(a, tpu) for a in (compact, *arrays[1:])
    )
    # Exact storage snapshots of every read-only operand, taken before the
    # tested calls: float equality would hide signed zeros and BF16 payload
    # differences.
    before = tuple(raw_words(a).copy() for a in (x, weight, down, up, inject))
    mixed, logits = full_mix_inject_broadcast(
        x,
        weight,
        down,
        up,
        inject,
        branches=branches,
        hidden=hidden,
        eps=1e-6,
        token_tile=tile,
    )
    repeated = full_mix_inject_broadcast(
        x,
        weight,
        down,
        up,
        inject,
        branches=branches,
        hidden=hidden,
        eps=1e-6,
        token_tile=tile,
    )
    for first, second in zip((mixed, logits), repeated, strict=True):
        np.testing.assert_array_equal(raw_words(first), raw_words(second))
    for actual, expected in [
        (mixed, expected_mix),
        (logits, expected_logits),
    ]:
        a, e = (np.asarray(v, np.float32) for v in (actual, expected))
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01
        assert (
            np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e))
            >= 0.999
        )
    for operand, snapshot in zip((x, weight, down, up, inject), before, strict=True):
        np.testing.assert_array_equal(raw_words(operand), snapshot)


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [128, 256])
def test_hc_inject_broadcast_matches_expanded_inject_words(tokens):
    """Cross-implementation evidence against the fused inject-mix control.

    Both arms run over the same values: the broadcast kernel receives the
    compact operand, and the inject-mix control receives the explicit expanded
    hyper input built from it by tiling branchwise. The hard gates are the
    original finite/L2/cosine oracles plus the same-implementation repeat and
    input storage contracts. The cross-implementation word difference is
    emitted as a diagnostic only: the broadcast normalization reduces the
    variance over ``hidden`` once where the expanded path reduces each branch's
    repeated slice — mathematically the same values, but FP32 sum order can
    legitimately change words — so no bitmatch gate and no extra rounding or
    barrier may force agreement.
    """
    cpu = jax.devices("cpu")[0]
    tpu = jax.devices("tpu")[0]
    branches, hidden, lowrank = 4, 128, 128
    with jax.default_device(cpu):
        arrays = tuple(
            jnp.asarray(a, jnp.bfloat16)
            for a in inputs(tokens, branches, hidden, lowrank)
        )
        compact = arrays[0][:, :hidden]
        expanded = jnp.tile(compact, (1, branches))
    broadcast_operands = tuple(jax.device_put(a, tpu) for a in (compact, *arrays[1:]))
    before = tuple(raw_words(a).copy() for a in broadcast_operands)
    broadcast = full_mix_inject_broadcast(
        *broadcast_operands, branches=branches, hidden=hidden, eps=1e-6, token_tile=128
    )
    repeated = full_mix_inject_broadcast(
        *broadcast_operands, branches=branches, hidden=hidden, eps=1e-6, token_tile=128
    )
    for first, second in zip(broadcast, repeated, strict=True):
        np.testing.assert_array_equal(raw_words(first), raw_words(second))
    control_operands = tuple(jax.device_put(a, tpu) for a in (expanded, *arrays[1:]))
    control = full_mix_inject(
        *control_operands, branches=branches, hidden=hidden, eps=1e-6, token_tile=128
    )
    for got, want in zip(broadcast, control, strict=True):
        a, e = (np.asarray(v, np.float32) for v in (got, want))
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01
        assert (
            np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e))
            >= 0.999
        )
        # Diagnostic only, never a gate: the count of differing raw words and
        # the maximum absolute value difference across the two implementations.
        words_a, words_e = raw_words(got), raw_words(want)
        differing = words_a != words_e
        print(
            f"broadcast-vs-expanded inject words (tokens={tokens}): "
            f"{int(differing.sum())}/{words_a.size} words differ, "
            f"max_abs={float(np.abs(a - e).max())}"
        )
    for operand, snapshot in zip(broadcast_operands, before, strict=True):
        np.testing.assert_array_equal(raw_words(operand), snapshot)
