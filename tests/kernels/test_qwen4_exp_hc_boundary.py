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
"""Combine->next-mix boundary kernel: shapes, admission, and TPU math contract."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.kernels.qwen4_exp.hc_combine_mix import (
    combine_mix_inject,
    combine_mix_inject_broadcast,
    combine_mix_terminal,
)
from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
    build_combine_mix_broadcast_op,
    build_combine_mix_op,
    build_combine_mix_terminal_op,
)


def inputs(tokens=256, branches=4, hidden=128, lowrank=128):
    rng = np.random.default_rng(43)
    width = branches * hidden
    shapes = (
        (tokens, width),
        (tokens, hidden),
        (tokens, branches),
        (width,),
        (lowrank, width),
        (width, lowrank),
        (branches, width),
    )
    return tuple((rng.normal(size=s) * 0.1).astype(np.float32) for s in shapes)


def raw_words(array):
    """Exact BF16 storage words, signed zeros and payloads included."""
    return np.asarray(array).view(np.uint16)


def reference(
    residual, block, in_logits, weight, down, up, inject, branches=4, eps=1e-6
):
    """Unfused boundary math with the checkpoint BF16 store boundaries."""
    hidden = residual.shape[1] // branches
    gates = 2.0 * jax.nn.sigmoid(in_logits.astype(jnp.float32) / branches)
    combined = (
        (
            residual.astype(jnp.float32).reshape(-1, branches, hidden)
            + block.astype(jnp.float32)[:, None] * gates[:, :, None]
        )
        .astype(jnp.bfloat16)
        .reshape(residual.shape)
    )
    values = combined.astype(jnp.float32).reshape(-1, branches, hidden)
    norm = values * jax.lax.rsqrt(jnp.mean(values**2, -1, keepdims=True) + eps)
    norm = norm.reshape(combined.shape)
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
    return combined, mixed, logits


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
def test_hc_combine_rejects_invalid_geometry(tokens, branches, hidden, tile, lowrank):
    shape = jax.ShapeDtypeStruct
    width = branches * hidden
    residual = shape((tokens, width), jnp.bfloat16)
    block = shape((tokens, hidden), jnp.bfloat16)
    in_logits = shape((tokens, max(branches, 1)), jnp.bfloat16)
    weight = shape((width,), jnp.bfloat16)
    down = shape((lowrank, width), jnp.bfloat16)
    up = shape((width, lowrank), jnp.bfloat16)
    inject = shape((max(branches, 1), width), jnp.bfloat16)
    fn = functools.partial(
        combine_mix_inject, branches=branches, hidden=hidden, eps=1e-6, token_tile=tile
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, residual, block, in_logits, weight, down, up, inject)


def test_hc_combine_rejects_residual_width_outside_branch_hidden_product():
    # Every other operand matches the residual's own 384-column width, so only
    # the branch-hidden product check can reject: without it the kernel would
    # reach device introspection and lower against mismatched geometry.
    shape = jax.ShapeDtypeStruct
    arrays = [
        shape((256, 384), jnp.bfloat16),
        shape((256, 128), jnp.bfloat16),
        shape((256, 4), jnp.bfloat16),
        shape((384,), jnp.bfloat16),
        shape((128, 384), jnp.bfloat16),
        shape((384, 128), jnp.bfloat16),
        shape((4, 384), jnp.bfloat16),
    ]
    fn = functools.partial(
        combine_mix_inject, branches=4, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, *arrays)


@pytest.mark.parametrize("operand", range(7))
def test_hc_combine_rejects_non_bf16_operands(operand):
    arrays = [jax.ShapeDtypeStruct(a.shape, jnp.bfloat16) for a in inputs(branches=3)]
    arrays[operand] = jax.ShapeDtypeStruct(arrays[operand].shape, jnp.float32)
    fn = functools.partial(
        combine_mix_inject, branches=3, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="BF16"):
        jax.eval_shape(fn, *arrays)


def test_hc_combine_rejects_mismatched_operand_shapes():
    shape = jax.ShapeDtypeStruct
    arrays = [
        shape((256, 512), jnp.bfloat16),
        shape((256, 128), jnp.bfloat16),
        shape((256, 4), jnp.bfloat16),
        shape((512,), jnp.bfloat16),
        shape((128, 512), jnp.bfloat16),
        shape((512, 128), jnp.bfloat16),
        shape((4, 512), jnp.bfloat16),
    ]
    arrays[2] = shape((256, 3), jnp.bfloat16)
    fn = functools.partial(
        combine_mix_inject, branches=4, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, *arrays)


def test_hc_combine_rejects_injected_weight_branch_mismatch():
    shape = jax.ShapeDtypeStruct
    arrays = [
        shape((256, 512), jnp.bfloat16),
        shape((256, 128), jnp.bfloat16),
        shape((256, 4), jnp.bfloat16),
        shape((512,), jnp.bfloat16),
        shape((128, 512), jnp.bfloat16),
        shape((512, 128), jnp.bfloat16),
        shape((3, 512), jnp.bfloat16),
    ]
    fn = functools.partial(
        combine_mix_inject, branches=4, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, *arrays)


def test_hc_combine_torch_fake_shapes():
    op = build_combine_mix_op(4, 128, 1e-6, 128)
    with FakeTensorMode():
        residual = torch.empty(256, 512, dtype=torch.bfloat16)
        block = torch.empty(256, 128, dtype=torch.bfloat16)
        in_logits = torch.empty(256, 4, dtype=torch.bfloat16)
        weight = torch.empty(512, dtype=torch.bfloat16)
        down = torch.empty(128, 512, dtype=torch.bfloat16)
        up = torch.empty(512, 128, dtype=torch.bfloat16)
        inject = torch.empty(4, 512, dtype=torch.bfloat16)
        combined, mixed, logits = op(
            residual, block, in_logits, weight, down, up, inject
        )
        assert combined.shape == (256, 512)
        assert mixed.shape == (256, 128)
        assert logits.shape == (256, 4)
        for tensor in (combined, mixed, logits):
            assert tensor.dtype == torch.bfloat16
            assert tensor.device == residual.device


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
        (512, 4, 2560, 320, 256),
        (384, 2, 512, 64, 128),
    ],
)
def test_hc_combine_mix_matches_cpu_reference(tokens, branches, hidden, lowrank, tile):
    cpu = jax.devices("cpu")[0]
    tpu = jax.devices("tpu")[0]
    with jax.default_device(cpu):
        arrays = tuple(
            jnp.asarray(a, jnp.bfloat16)
            for a in inputs(tokens, branches, hidden, lowrank)
        )
        expected = reference(*arrays, branches=branches)
    device_arrays = tuple(jax.device_put(a, tpu) for a in arrays)
    # Exact storage snapshots of every read-only operand, taken before the
    # tested calls: float equality would hide signed zeros and BF16 payload
    # differences.
    before = tuple(raw_words(a).copy() for a in device_arrays)
    actual = combine_mix_inject(
        *device_arrays, branches=branches, hidden=hidden, eps=1e-6, token_tile=tile
    )
    repeated = combine_mix_inject(
        *device_arrays, branches=branches, hidden=hidden, eps=1e-6, token_tile=tile
    )
    for first, second in zip(actual, repeated, strict=True):
        np.testing.assert_array_equal(raw_words(first), raw_words(second))
    for got, want in zip(actual, expected, strict=True):
        a, e = (np.asarray(v, np.float32) for v in (got, want))
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01
        assert (
            np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e))
            >= 0.999
        )
    for operand, snapshot in zip(device_arrays, before, strict=True):
        np.testing.assert_array_equal(raw_words(operand), snapshot)


@pytest.mark.tpu_test
def test_hc_combine_rounds_boundary_at_bf16_stores():
    """Combine and logits keep the standalone BF16 store boundaries.

    With unit residual and block rows, zero affine/down/up and zero prior
    logits, the gate is 2*sigmoid(0)=1.0 so every combined element is exactly
    2.0; the normalized rows return to 1.0, silu(0) zeroes the down/up path,
    and the up gate sigmoid(0)=0.5 scales the mean over branches to 0.5. The
    next injection logit is the width sum 512 * 2^-8 = 2.0. Every value here
    is exactly representable in BF16, so the kernel must store exactly the
    words the unfused sequence would.
    """
    tpu = jax.devices("tpu")[0]
    branches, hidden, lowrank, tokens = 4, 128, 128, 128
    width = branches * hidden
    host = [
        np.ones((tokens, width), np.float32),
        np.ones((tokens, hidden), np.float32),
        np.zeros((tokens, branches), np.float32),
        np.zeros((width,), np.float32),
        np.zeros((lowrank, width), np.float32),
        np.zeros((width, lowrank), np.float32),
        np.full((branches, width), 2**-8, np.float32),
    ]
    arrays = [jax.device_put(a, tpu).astype(jnp.bfloat16) for a in host]
    before = tuple(raw_words(a).copy() for a in arrays)
    combined, mixed, logits = combine_mix_inject(
        *arrays, branches=branches, hidden=hidden, eps=1e-6, token_tile=128
    )
    expected_combined = np.full((tokens, width), 2.0, np.asarray(combined).dtype)
    expected_mixed = np.full((tokens, hidden), 0.5, np.asarray(mixed).dtype)
    expected_logits = np.full((tokens, branches), 2.0, np.asarray(logits).dtype)
    for got, want in (
        (combined, expected_combined),
        (mixed, expected_mixed),
        (logits, expected_logits),
    ):
        np.testing.assert_array_equal(raw_words(got), want.view(np.uint16))
    for operand, snapshot in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(raw_words(operand), snapshot)


def terminal_inputs(tokens=256, branches=4, hidden=128, lowrank=128):
    rng = np.random.default_rng(44)
    width = branches * hidden
    shapes = (
        (tokens, width),
        (tokens, hidden),
        (tokens, branches),
        (width,),
        (lowrank, width),
        (width, lowrank),
    )
    return tuple((rng.normal(size=s) * 0.1).astype(np.float32) for s in shapes)


def terminal_reference(
    residual, block, in_logits, weight, down, up, branches=4, eps=1e-6
):
    """Unfused terminal math with the checkpoint BF16 store boundaries.

    The combined value rounds through BF16 exactly as the public ``combine``
    stores it, then the terminal ``mix`` reads those words back: the fused
    kernel must keep that boundary while never writing the plane to HBM.
    """
    hidden = residual.shape[1] // branches
    gates = 2.0 * jax.nn.sigmoid(in_logits.astype(jnp.float32) / branches)
    combined = (
        (
            residual.astype(jnp.float32).reshape(-1, branches, hidden)
            + block.astype(jnp.float32)[:, None] * gates[:, :, None]
        )
        .astype(jnp.bfloat16)
        .reshape(residual.shape)
    )
    values = combined.astype(jnp.float32).reshape(-1, branches, hidden)
    norm = values * jax.lax.rsqrt(jnp.mean(values**2, -1, keepdims=True) + eps)
    norm = norm.reshape(combined.shape)
    norm = (norm + norm * weight.astype(jnp.float32)).astype(jnp.bfloat16)
    mid = (norm.astype(jnp.float32) @ down.astype(jnp.float32).T).astype(jnp.bfloat16)
    mid = jax.nn.silu(mid.astype(jnp.float32) / branches).astype(jnp.bfloat16)
    logits_up = (mid.astype(jnp.float32) @ up.astype(jnp.float32).T).astype(
        jnp.bfloat16
    )
    weighted = jax.nn.sigmoid(logits_up.astype(jnp.float32)) * norm.astype(jnp.float32)
    mixed = jnp.mean(weighted.reshape(-1, branches, hidden), axis=1).astype(
        jnp.bfloat16
    )
    return mixed


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
def test_hc_terminal_rejects_invalid_geometry(tokens, branches, hidden, tile, lowrank):
    shape = jax.ShapeDtypeStruct
    width = branches * hidden
    residual = shape((tokens, width), jnp.bfloat16)
    block = shape((tokens, hidden), jnp.bfloat16)
    in_logits = shape((tokens, max(branches, 1)), jnp.bfloat16)
    weight = shape((width,), jnp.bfloat16)
    down = shape((lowrank, width), jnp.bfloat16)
    up = shape((width, lowrank), jnp.bfloat16)
    fn = functools.partial(
        combine_mix_terminal,
        branches=branches,
        hidden=hidden,
        eps=1e-6,
        token_tile=tile,
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, residual, block, in_logits, weight, down, up)


@pytest.mark.parametrize("operand", range(6))
def test_hc_terminal_rejects_non_bf16_operands(operand):
    arrays = [
        jax.ShapeDtypeStruct(a.shape, jnp.bfloat16) for a in terminal_inputs(branches=3)
    ]
    arrays[operand] = jax.ShapeDtypeStruct(arrays[operand].shape, jnp.float32)
    fn = functools.partial(
        combine_mix_terminal, branches=3, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="BF16"):
        jax.eval_shape(fn, *arrays)


def test_hc_terminal_rejects_mismatched_operand_shapes():
    shape = jax.ShapeDtypeStruct
    arrays = [
        shape((256, 512), jnp.bfloat16),
        shape((256, 128), jnp.bfloat16),
        shape((256, 4), jnp.bfloat16),
        shape((512,), jnp.bfloat16),
        shape((128, 512), jnp.bfloat16),
        shape((512, 128), jnp.bfloat16),
    ]
    arrays[2] = shape((256, 3), jnp.bfloat16)
    fn = functools.partial(
        combine_mix_terminal, branches=4, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, *arrays)


def test_hc_terminal_torch_fake_shapes():
    op = build_combine_mix_terminal_op(4, 128, 1e-6, 128)
    with FakeTensorMode():
        residual = torch.empty(256, 512, dtype=torch.bfloat16)
        block = torch.empty(256, 128, dtype=torch.bfloat16)
        in_logits = torch.empty(256, 4, dtype=torch.bfloat16)
        weight = torch.empty(512, dtype=torch.bfloat16)
        down = torch.empty(128, 512, dtype=torch.bfloat16)
        up = torch.empty(512, 128, dtype=torch.bfloat16)
        mixed = op(residual, block, in_logits, weight, down, up)
        assert mixed.shape == (256, 128)
        assert mixed.dtype == torch.bfloat16
        assert mixed.device == residual.device


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "tokens,branches,hidden,lowrank,tile",
    [
        (128, 4, 128, 128, 128),
        (256, 4, 128, 128, 128),
        (256, 3, 128, 128, 128),
        # lowrank off the hardware row boundary: the down plane stages alone,
        # so no packed gap rows exist on this path.
        (256, 4, 128, 40, 128),
        (256, 4, 2560, 320, 128),
        (512, 4, 2560, 320, 128),
        (256, 4, 2560, 320, 256),
        (512, 4, 2560, 320, 256),
        (384, 2, 512, 64, 128),
    ],
)
def test_hc_terminal_mix_matches_cpu_reference(tokens, branches, hidden, lowrank, tile):
    cpu = jax.devices("cpu")[0]
    tpu = jax.devices("tpu")[0]
    with jax.default_device(cpu):
        arrays = tuple(
            jnp.asarray(a, jnp.bfloat16)
            for a in terminal_inputs(tokens, branches, hidden, lowrank)
        )
        expected = terminal_reference(*arrays, branches=branches)
    device_arrays = tuple(jax.device_put(a, tpu) for a in arrays)
    # Exact storage snapshots of every read-only operand, taken before the
    # tested calls: float equality would hide signed zeros and BF16 payload
    # differences.
    before = tuple(raw_words(a).copy() for a in device_arrays)
    actual = combine_mix_terminal(
        *device_arrays,
        branches=branches,
        hidden=hidden,
        eps=1e-6,
        token_tile=tile,
    )
    repeated = combine_mix_terminal(
        *device_arrays,
        branches=branches,
        hidden=hidden,
        eps=1e-6,
        token_tile=tile,
    )
    np.testing.assert_array_equal(raw_words(actual), raw_words(repeated))
    assert actual.shape == (tokens, hidden)
    a, e = (np.asarray(v, np.float32) for v in (actual, expected))
    assert np.isfinite(a).all()
    assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01
    assert (
        np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e)) >= 0.999
    )
    for operand, snapshot in zip(device_arrays, before, strict=True):
        np.testing.assert_array_equal(raw_words(operand), snapshot)


@pytest.mark.tpu_test
def test_hc_terminal_rounds_boundary_at_bf16_stores():
    """The terminal mix keeps the standalone BF16 store boundaries.

    With unit residual and block rows, zero affine/down/up and zero prior
    logits, the gate is 2*sigmoid(0)=1.0 so every combined element is exactly
    2.0; the normalized rows return to 1.0, silu(0) zeroes the down/up path,
    and the up gate sigmoid(0)=0.5 scales the mean over branches to 0.5. Every
    value here is exactly representable in BF16, so the in-place combined
    plane must round exactly like the public combine's HBM-stored tensor.
    """
    tpu = jax.devices("tpu")[0]
    branches, hidden, lowrank, tokens = 4, 128, 128, 128
    width = branches * hidden
    host = [
        np.ones((tokens, width), np.float32),
        np.ones((tokens, hidden), np.float32),
        np.zeros((tokens, branches), np.float32),
        np.zeros((width,), np.float32),
        np.zeros((lowrank, width), np.float32),
        np.zeros((width, lowrank), np.float32),
    ]
    arrays = [jax.device_put(a, tpu).astype(jnp.bfloat16) for a in host]
    before = tuple(raw_words(a).copy() for a in arrays)
    mixed = combine_mix_terminal(
        *arrays, branches=branches, hidden=hidden, eps=1e-6, token_tile=128
    )
    expected_mixed = np.full((tokens, hidden), 0.5, np.asarray(mixed).dtype)
    np.testing.assert_array_equal(raw_words(mixed), expected_mixed.view(np.uint16))
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
def test_hc_combine_broadcast_rejects_invalid_geometry(
    tokens, branches, hidden, tile, lowrank
):
    shape = jax.ShapeDtypeStruct
    width = max(branches * hidden, 1)
    residual = shape((tokens, hidden), jnp.bfloat16)
    block = shape((tokens, hidden), jnp.bfloat16)
    in_logits = shape((tokens, max(branches, 1)), jnp.bfloat16)
    weight = shape((width,), jnp.bfloat16)
    down = shape((max(lowrank, 1), width), jnp.bfloat16)
    up = shape((width, max(lowrank, 1)), jnp.bfloat16)
    inject = shape((max(branches, 1), width), jnp.bfloat16)
    fn = functools.partial(
        combine_mix_inject_broadcast,
        branches=branches,
        hidden=hidden,
        eps=1e-6,
        token_tile=tile,
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, residual, block, in_logits, weight, down, up, inject)


@pytest.mark.parametrize("operand", range(7))
def test_hc_combine_broadcast_rejects_non_bf16_operands(operand):
    shape = jax.ShapeDtypeStruct
    arrays = [
        shape((256, 128), jnp.bfloat16),
        shape((256, 128), jnp.bfloat16),
        shape((256, 3), jnp.bfloat16),
        shape((384,), jnp.bfloat16),
        shape((128, 384), jnp.bfloat16),
        shape((384, 128), jnp.bfloat16),
        shape((3, 384), jnp.bfloat16),
    ]
    arrays[operand] = jax.ShapeDtypeStruct(arrays[operand].shape, jnp.float32)
    fn = functools.partial(
        combine_mix_inject_broadcast, branches=3, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="BF16"):
        jax.eval_shape(fn, *arrays)


def test_hc_combine_broadcast_rejects_residual_width_mismatch():
    shape = jax.ShapeDtypeStruct
    arrays = [
        shape((256, 127), jnp.bfloat16),
        shape((256, 128), jnp.bfloat16),
        shape((256, 3), jnp.bfloat16),
        shape((384,), jnp.bfloat16),
        shape((128, 384), jnp.bfloat16),
        shape((384, 128), jnp.bfloat16),
        shape((3, 384), jnp.bfloat16),
    ]
    fn = functools.partial(
        combine_mix_inject_broadcast, branches=3, hidden=128, eps=1e-6, token_tile=128
    )
    with pytest.raises(ValueError, match="HC .* requires"):
        jax.eval_shape(fn, *arrays)


def test_hc_combine_broadcast_torch_fake_shapes():
    op = build_combine_mix_broadcast_op(4, 128, 1e-6, 128)
    with FakeTensorMode():
        residual = torch.empty(256, 128, dtype=torch.bfloat16)
        block = torch.empty(256, 128, dtype=torch.bfloat16)
        in_logits = torch.empty(256, 4, dtype=torch.bfloat16)
        weight = torch.empty(512, dtype=torch.bfloat16)
        down = torch.empty(128, 512, dtype=torch.bfloat16)
        up = torch.empty(512, 128, dtype=torch.bfloat16)
        inject = torch.empty(4, 512, dtype=torch.bfloat16)
        combined, mixed, logits = op(
            residual, block, in_logits, weight, down, up, inject
        )
        assert combined.shape == (256, 512)
        assert mixed.shape == (256, 128)
        assert logits.shape == (256, 4)
        for tensor in (combined, mixed, logits):
            assert tensor.dtype == torch.bfloat16
            assert tensor.device == residual.device


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
def test_hc_combine_broadcast_matches_cpu_reference(
    tokens, branches, hidden, lowrank, tile
):
    """The broadcast residual reproduces the boundary over the expanded words.

    The compact operand is one branch slice of the same random fixture, and the
    explicit expanded residual is built from it by tiling branchwise (a BF16
    copy of the exact words). The unchanged independent reference runs over
    that expanded value with the same block/logits/weights the kernel receives
    with the compact operand, so both sides see the same input. Matching
    through the L2/cosine oracle proves the broadcast staging forms each
    branch's combined and normalized values from the same BF16 words the
    expanded plane would carry; the broadcast normalization reduces the
    variance over ``hidden`` once where the expanded path reduces each branch's
    repeated slice, so raw words are not compared across the two paths.
    """
    cpu = jax.devices("cpu")[0]
    tpu = jax.devices("tpu")[0]
    with jax.default_device(cpu):
        arrays = tuple(
            jnp.asarray(a, jnp.bfloat16)
            for a in inputs(tokens, branches, hidden, lowrank)
        )
        # The compact operand the kernel actually stages: one branch slice.
        residual = arrays[0][:, :hidden]
        # The expanded plane every branch slice of which repeats it, built from
        # the exact compact words.
        expanded = jnp.tile(residual, (1, branches))
        expected = reference(expanded, *arrays[1:], branches=branches)
    operands = (residual, *arrays[1:])
    device_arrays = tuple(jax.device_put(a, tpu) for a in operands)
    before = tuple(raw_words(a).copy() for a in device_arrays)
    actual = combine_mix_inject_broadcast(
        *device_arrays, branches=branches, hidden=hidden, eps=1e-6, token_tile=tile
    )
    repeated = combine_mix_inject_broadcast(
        *device_arrays, branches=branches, hidden=hidden, eps=1e-6, token_tile=tile
    )
    for first, second in zip(actual, repeated, strict=True):
        np.testing.assert_array_equal(raw_words(first), raw_words(second))
    for got, want in zip(actual, expected, strict=True):
        a, e = (np.asarray(v, np.float32) for v in (got, want))
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01
        assert (
            np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e))
            >= 0.999
        )
    for operand, snapshot in zip(device_arrays, before, strict=True):
        np.testing.assert_array_equal(raw_words(operand), snapshot)


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [128, 256])
def test_hc_combine_broadcast_matches_expanded_boundary_words(tokens):
    """Cross-implementation evidence against the fused boundary control.

    Both arms run over the same values: the broadcast kernel receives the
    compact operand, and the boundary control receives the explicit expanded
    residual built from it by tiling branchwise. The hard gates are the
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
        residual = arrays[0][:, :hidden]
        expanded = jnp.tile(residual, (1, branches))
    broadcast_operands = tuple(jax.device_put(a, tpu) for a in (residual, *arrays[1:]))
    before = tuple(raw_words(a).copy() for a in broadcast_operands)
    broadcast = combine_mix_inject_broadcast(
        *broadcast_operands, branches=branches, hidden=hidden, eps=1e-6, token_tile=128
    )
    repeated = combine_mix_inject_broadcast(
        *broadcast_operands, branches=branches, hidden=hidden, eps=1e-6, token_tile=128
    )
    for first, second in zip(broadcast, repeated, strict=True):
        np.testing.assert_array_equal(raw_words(first), raw_words(second))
    control_operands = tuple(jax.device_put(a, tpu) for a in (expanded, *arrays[1:]))
    control = combine_mix_inject(
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
            f"broadcast-vs-expanded boundary words (tokens={tokens}): "
            f"{int(differing.sum())}/{words_a.size} words differ, "
            f"max_abs={float(np.abs(a - e).max())}"
        )
    for operand, snapshot in zip(broadcast_operands, before, strict=True):
        np.testing.assert_array_equal(raw_words(operand), snapshot)
