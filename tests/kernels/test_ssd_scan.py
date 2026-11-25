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
"""Correctness and Mosaic lowering tests for Pallas SSD scan kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import export

from vllm_torchtpu.kernels.ssd_scan import (
    chunk_scan_pallas,
    chunk_state_pallas,
)
from vllm_torchtpu.kernels.ssd_scan.ssd_scan import _select_head_block_size


class _FakeTpuDevice:
    def __init__(self, device_kind: str):
        self.device_kind = device_kind
        self.platform = "tpu"
        self.num_cores = 1


def _make_kernel_inputs(
    n_chunks: int,
    block_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    *,
    n_groups: int | None = None,
    dtype=jnp.float32,
):
    groups = n_heads if n_groups is None else n_groups
    rng = jax.random.key(42 + n_chunks + n_heads + groups + block_len)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    X_c = jax.random.normal(k1, (n_chunks, n_heads, block_len, d_head), dtype=dtype)
    A_c = -jnp.exp(
        jax.random.normal(k2, (n_chunks, n_heads, block_len), dtype=jnp.float32)
    )
    A_cumsum = jnp.cumsum(A_c, axis=-1)
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)[:, :, None, :]
    B_c = jax.random.normal(k3, (n_chunks, groups, block_len, d_state), dtype=dtype)
    C_c = jax.random.normal(k4, (n_chunks, groups, block_len, d_state), dtype=dtype)
    states = jax.random.normal(
        k5, (n_chunks, n_heads, d_head, d_state), dtype=jnp.float32
    )
    return B_c, decay_states, X_c, C_c, states


def _chunk_state_reference(
    B_c: jax.Array,
    decay_states: jax.Array,
    X_c: jax.Array,
) -> np.ndarray:
    b_np = np.asarray(B_c.astype(jnp.float32), dtype=np.float32)
    d_np = np.asarray(decay_states.astype(jnp.float32), dtype=np.float32)[:, :, 0, :]
    x_np = np.asarray(X_c.astype(jnp.float32), dtype=np.float32)
    n_heads = x_np.shape[1]
    n_groups = b_np.shape[1]
    if n_groups < n_heads:
        b_np = np.repeat(b_np, n_heads // n_groups, axis=1)
    return np.einsum("chln,chlp,chl->chpn", b_np, x_np, d_np, optimize=True)


def _chunk_scan_reference(
    C_c: jax.Array,
    states: jax.Array,
) -> np.ndarray:
    c_np = np.asarray(C_c.astype(jnp.float32), dtype=np.float32)
    s_np = np.asarray(states.astype(jnp.float32), dtype=np.float32)
    n_heads = s_np.shape[1]
    n_groups = c_np.shape[1]
    if n_groups < n_heads:
        c_np = np.repeat(c_np, n_heads // n_groups, axis=1)
    return np.einsum("chln,chpn->chlp", c_np, s_np, optimize=True)


def test_ssd_kernels_validate_input_shapes() -> None:
    n_chunks, n_heads, block_len, d_head, d_state = 2, 4, 8, 16, 8
    X_c = jnp.zeros((n_chunks, n_heads, block_len, d_head), dtype=jnp.float32)
    B_c = jnp.zeros((n_chunks, n_heads, block_len, d_state), dtype=jnp.float32)
    C_c = jnp.zeros((n_chunks, n_heads, block_len, d_state), dtype=jnp.float32)
    decay_states = jnp.zeros((n_chunks, n_heads, 1, block_len), dtype=jnp.float32)
    states = jnp.zeros((n_chunks, n_heads, d_head, d_state), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Expected positive 4D X_c"):
        chunk_state_pallas(B_c, decay_states, X_c[0], interpret=True)
    with pytest.raises(ValueError, match="Expected positive 4D X_c"):
        chunk_state_pallas(B_c[:0], decay_states[:0], X_c[:0], interpret=True)
    with pytest.raises(ValueError, match="Expected positive 4D B_c"):
        chunk_state_pallas(B_c[:, :3], decay_states, X_c, interpret=True)
    with pytest.raises(ValueError, match="Expected decay_states shape"):
        chunk_state_pallas(B_c, jnp.swapaxes(decay_states, -1, -2), X_c, interpret=True)

    with pytest.raises(ValueError, match="Expected positive 4D C_c"):
        chunk_scan_pallas(C_c[0], states, interpret=True)
    with pytest.raises(ValueError, match="Expected positive 4D C_c"):
        chunk_scan_pallas(C_c[:0], states[:0], interpret=True)
    with pytest.raises(ValueError, match="Expected positive 4D states"):
        chunk_scan_pallas(C_c[:, :3], states, interpret=True)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(
    ("n_chunks", "block_len", "n_heads", "n_groups", "d_head", "d_state"),
    [
        (4, 4, 4, 4, 8, 8),
        (4, 16, 1, 1, 16, 8),
        (4, 16, 6, 1, 16, 8),
        (4, 16, 8, 2, 16, 8),
        (4, 16, 12, 2, 16, 8),
        (3, 32, 4, 4, 16, 8),
        (2, 64, 8, 2, 32, 16),
    ],
)
def test_ssd_kernels_correctness_interpret(
    dtype,
    n_chunks: int,
    block_len: int,
    n_heads: int,
    n_groups: int,
    d_head: int,
    d_state: int,
) -> None:
    B_c, decay_states, X_c, C_c, states = _make_kernel_inputs(
        n_chunks,
        block_len,
        n_heads,
        d_head,
        d_state,
        n_groups=n_groups,
        dtype=dtype,
    )

    states_got = chunk_state_pallas(B_c, decay_states, X_c, interpret=True)
    assert states_got.dtype == jnp.float32
    states_ref = _chunk_state_reference(B_c, decay_states, X_c)
    np.testing.assert_allclose(np.asarray(states_got), states_ref, rtol=1e-4, atol=1e-4)

    y_f32 = chunk_scan_pallas(C_c, states, out_dtype=jnp.float32, interpret=True)
    assert y_f32.dtype == jnp.float32
    y_ref = _chunk_scan_reference(C_c, states)
    np.testing.assert_allclose(np.asarray(y_f32), y_ref, rtol=1e-4, atol=1e-4)

    y_default = chunk_scan_pallas(C_c, states, interpret=True)
    assert y_default.dtype == dtype
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-4
    np.testing.assert_allclose(
        np.asarray(y_default.astype(jnp.float32)), y_ref, rtol=tol, atol=tol
    )

    if n_groups < n_heads:
        repeats = n_heads // n_groups
        B_rep = jnp.repeat(B_c, repeats, axis=1)
        C_rep = jnp.repeat(C_c, repeats, axis=1)
        states_rep = chunk_state_pallas(B_rep, decay_states, X_c, interpret=True)
        y_rep = chunk_scan_pallas(C_rep, states, out_dtype=jnp.float32, interpret=True)
        np.testing.assert_allclose(
            np.asarray(states_got), np.asarray(states_rep), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(y_f32), np.asarray(y_rep), rtol=1e-5, atol=1e-5
        )


def test_ssd_kernels_grouped_multi_blocks_per_group_interpret(monkeypatch) -> None:
    """Exercise `blocks_per_group > 1` (`head_block < n_heads // n_groups`)."""
    import vllm_torchtpu.kernels.ssd_scan.ssd_scan as ssd_mod

    # With n_heads=8, n_groups=2 (heads_per_group=4), block_len=32, d_head=16,
    # d_state=16: bc_single_bytes = 32768, per_head_bytes = 57344 -> budget
    # 147456 at vmem_capacity = 294912 forces head_block = 2 (`blocks_per_group=2`).
    monkeypatch.setattr(ssd_mod, "_INTERPRET_DEFAULT_VMEM_BYTES", 294912)
    hb_state, _ = _select_head_block_size(
        8, 2, 32, 16, 16, include_decay_tile=True, interpret=True
    )
    hb_scan, _ = _select_head_block_size(
        8, 2, 32, 16, 16, include_decay_tile=False, interpret=True
    )
    assert hb_state == 2 and (8 // 2) // hb_state == 2
    assert hb_scan == 2 and (8 // 2) // hb_scan == 2

    B_c, decay_states, X_c, C_c, states = _make_kernel_inputs(
        n_chunks=4,
        block_len=32,
        n_heads=8,
        d_head=16,
        d_state=16,
        n_groups=2,
        dtype=jnp.float32,
    )
    states_got = chunk_state_pallas(B_c, decay_states, X_c, interpret=True)
    y_got = chunk_scan_pallas(C_c, states, out_dtype=jnp.float32, interpret=True)
    np.testing.assert_allclose(
        np.asarray(states_got),
        _chunk_state_reference(B_c, decay_states, X_c),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(y_got),
        _chunk_scan_reference(C_c, states),
        rtol=1e-4,
        atol=1e-4,
    )


def test_ssd_scan_vmem_derived_head_block_and_mosaic_lowering() -> None:
    """Verify `_select_head_block_size` queries `pltpu.get_tpu_info()` when
    `not interpret` and lowers cleanly to Mosaic TPU MLIR across shapes,
    including `mamba2-2.7b` (`n_heads=80, n_groups=1, head_block=40`).
    """
    mesh_16m = jax.sharding.AbstractMesh(
        (1,), ("x",), abstract_device=_FakeTpuDevice("TPU v4")
    )
    mesh_128m = jax.sharding.AbstractMesh(
        (1,), ("x",), abstract_device=_FakeTpuDevice("TPU v6 lite")
    )

    hb_interp, limit_interp = _select_head_block_size(
        64, 64, 512, 128, 256, include_decay_tile=True, interpret=True
    )
    assert hb_interp == 4
    assert limit_interp is None

    with jax.sharding.use_abstract_mesh(mesh_16m):
        hb_16m, limit_16m = _select_head_block_size(
            64, 64, 512, 128, 256, include_decay_tile=True, interpret=False
        )
        hb_grouped, limit_grouped = _select_head_block_size(
            12, 2, 64, 64, 128, include_decay_tile=True, interpret=False
        )
        hb_27b, limit_27b = _select_head_block_size(
            80, 1, 64, 64, 128, include_decay_tile=True, interpret=False
        )
    with jax.sharding.use_abstract_mesh(mesh_128m):
        hb_128m, limit_128m = _select_head_block_size(
            64, 64, 512, 128, 256, include_decay_tile=True, interpret=False
        )

    assert hb_16m == 4
    assert hb_grouped == 6
    assert hb_27b == 40 and (80 // 1) // hb_27b == 2
    assert hb_128m > hb_16m
    assert limit_16m == limit_grouped == limit_27b == int(0.80 * 16 * 1024 * 1024)
    assert limit_128m == int(0.80 * 128 * 1024 * 1024)

    jitted_state = jax.jit(chunk_state_pallas)
    jitted_scan = jax.jit(lambda C, s: chunk_scan_pallas(C, s, out_dtype=jnp.float32))

    with jax.sharding.use_abstract_mesh(mesh_128m):
        exported_state = export.export(jitted_state, platforms=("tpu",))(
            jax.ShapeDtypeStruct((3, 2, 64, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((3, 12, 1, 64), jnp.float32),
            jax.ShapeDtypeStruct((3, 12, 64, 64), jnp.bfloat16),
        )
        exported_scan = export.export(jitted_scan, platforms=("tpu",))(
            jax.ShapeDtypeStruct((3, 2, 64, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((3, 12, 64, 128), jnp.float32),
        )
        assert "ssd_chunk_state_pallas" in exported_state.mlir_module()
        assert "ssd_chunk_scan_pallas" in exported_scan.mlir_module()

        # Large-tile config whose selected head_block exceeds a 32 MiB default
        # scoped VMEM cap on 128 MiB v6e, verifying vmem_limit_bytes propagation.
        exported_large = export.export(jitted_state, platforms=("tpu",))(
            jax.ShapeDtypeStruct((2, 64, 512, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, 64, 1, 512), jnp.float32),
            jax.ShapeDtypeStruct((2, 64, 512, 128), jnp.bfloat16),
        )
        assert "ssd_chunk_state_pallas" in exported_large.mlir_module()

    # Verify mamba2-2.7b geometry (n_heads=80, n_groups=1, d_head=64, d_state=128)
    # on 16 MiB VMEM lowers cleanly with head_block=40 (`blocks_per_group=2`).
    with jax.sharding.use_abstract_mesh(mesh_16m):
        exported_27b_state = export.export(jitted_state, platforms=("tpu",))(
            jax.ShapeDtypeStruct((2, 1, 64, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, 80, 1, 64), jnp.float32),
            jax.ShapeDtypeStruct((2, 80, 64, 64), jnp.bfloat16),
        )
        exported_27b_scan = export.export(jitted_scan, platforms=("tpu",))(
            jax.ShapeDtypeStruct((2, 1, 64, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, 80, 64, 128), jnp.float32),
        )
        assert "ssd_chunk_state_pallas" in exported_27b_state.mlir_module()
        assert "ssd_chunk_scan_pallas" in exported_27b_scan.mlir_module()


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(
    ("n_chunks", "block_len", "n_heads", "n_groups", "d_head", "d_state"),
    [
        (8, 32, 4, 4, 16, 8),
        (4, 64, 6, 6, 64, 128),
        (4, 64, 12, 2, 64, 128),
        (8, 64, 16, 4, 64, 128),
    ],
)
def test_ssd_kernels_correctness_tpu(
    dtype,
    n_chunks: int,
    block_len: int,
    n_heads: int,
    n_groups: int,
    d_head: int,
    d_state: int,
) -> None:
    B_c, decay_states, X_c, C_c, states = _make_kernel_inputs(
        n_chunks,
        block_len,
        n_heads,
        d_head,
        d_state,
        n_groups=n_groups,
        dtype=dtype,
    )

    states_got = jax.jit(chunk_state_pallas)(B_c, decay_states, X_c)
    assert states_got.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(states_got),
        _chunk_state_reference(B_c, decay_states, X_c),
        rtol=1e-3,
        atol=1e-3,
    )

    y_f32 = jax.jit(lambda C, s: chunk_scan_pallas(C, s, out_dtype=jnp.float32))(
        C_c, states
    )
    assert y_f32.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(y_f32),
        _chunk_scan_reference(C_c, states),
        rtol=1e-3,
        atol=1e-3,
    )

    y_default = jax.jit(chunk_scan_pallas)(C_c, states)
    assert y_default.dtype == dtype
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-3
    np.testing.assert_allclose(
        np.asarray(y_default.astype(jnp.float32)),
        _chunk_scan_reference(C_c, states),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_ssd_kernels_mamba2_27b_80_heads_tpu(dtype) -> None:
    """Compile and run mamba2-2.7b (`n_heads=80, head_block=40`) on TPU."""
    B_c, decay_states, X_c, C_c, states = _make_kernel_inputs(
        n_chunks=2,
        block_len=64,
        n_heads=80,
        d_head=64,
        d_state=128,
        n_groups=1,
        dtype=dtype,
    )

    states_got = jax.jit(chunk_state_pallas)(B_c, decay_states, X_c)
    assert states_got.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(states_got),
        _chunk_state_reference(B_c, decay_states, X_c),
        rtol=1e-3,
        atol=1e-3,
    )

    y_f32 = jax.jit(lambda C, s: chunk_scan_pallas(C, s, out_dtype=jnp.float32))(
        C_c, states
    )
    assert y_f32.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(y_f32),
        _chunk_scan_reference(C_c, states),
        rtol=1e-3,
        atol=1e-3,
    )
