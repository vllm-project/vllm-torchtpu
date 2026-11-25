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
"""CPU and TPU unit tests for packed GDN recurrent-state formats."""

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, tiling
from vllm_torchtpu.kernels.gdn.v3 import recurrent_state as rs

_SCALED_FORMATS = (rs.RecurrentStateFormat.FP8,)
# The FP8 decode update from stored values rounds each element to bfloat16 a
# few times, each time by under 0.4%.
_MAX_FP8_DECODE_REL_ERROR = 1e-2


def _rel_error(
    want: jax.Array, got: jax.Array, axis: int | tuple[int, ...]
) -> np.ndarray:
    want_f64 = np.asarray(want, np.float64)
    err = np.asarray(got, np.float64) - want_f64
    return np.sqrt(
        np.sum(np.square(err), axis=axis) / np.sum(np.square(want_f64), axis=axis)
    )


def _rel_l2(ref: np.ndarray, got: np.ndarray) -> float:
    r = np.asarray(ref, dtype=np.float64)
    g = np.asarray(got, dtype=np.float64)
    return float(np.linalg.norm(r - g) / (np.linalg.norm(r) + 1e-12))


def _independent_fp8_quantize_ref(state_f32: np.ndarray) -> np.ndarray:
    """Independent NumPy reference for per-column power-of-two FP8 quantization."""
    # Pack quantizes via bfloat16 first, then shifts exponents into float8_e4m3fn.
    x_bf16 = np.asarray(jnp.asarray(state_f32, jnp.bfloat16), dtype=np.float32)
    amax = np.max(np.abs(x_bf16), axis=-2, keepdims=True)
    # Match power-of-two scale keeping |x / scale| < 256 in float8_e4m3fn.
    clipped = np.clip(amax * (2.0 / 256.0), 2.0**-110, 2.0**118)
    scale = np.ldexp(1.0, np.floor(np.log2(clipped)).astype(np.int32)).astype(
        np.float32
    )
    q = jnp.asarray(x_bf16 / scale, dtype=jnp.float8_e4m3fn)
    return np.asarray(q, dtype=np.float32) * scale


def _pallas_stream_roundtrip(
    in_slot: jax.Array,
    seq_has_init_arr: jax.Array,
    *,
    fmt: rs.RecurrentStateFormat,
    n_v: int,
    d_k: int,
    d_v: int,
    window_size: int,
) -> jax.Array:
    """Runs `RecurrentStateStream` on real Pallas VMEM refs."""
    n_seq = in_slot.shape[0]
    out_shape = (n_seq, window_size, *rs.slot_window_shape(fmt, n_v, d_k, d_v))

    def _kernel(in_ref, has_init_ref, out_ref):
        stream = rs.RecurrentStateStream(
            in_ref,
            out_ref,
            fmt=fmt,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            window_size=window_size,
            seq_has_init=[has_init_ref[s] != 0 for s in range(n_seq)],
        )
        for s in range(n_seq):
            for v in range(n_v):
                h = stream.load_head(s, v)
                for w_idx in range(window_size):
                    stream.store_ckpt_head(s, v, w_idx, h * float(w_idx + 1))

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.uint32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    )(in_slot, seq_has_init_arr)


def _pallas_carry_roundtrip(
    in_slot: jax.Array,
    has_init_arr: jax.Array,
    *,
    fmt: rs.RecurrentStateFormat,
    n_v: int,
    d_k: int,
    d_v: int,
) -> jax.Array:
    """Runs `unpack_slot_to_carry` and `pack_carry_to_slot` on real VMEM refs."""
    out_shape = (1, 1, *rs.slot_window_shape(fmt, n_v, d_k, d_v))

    def _kernel(in_ref, has_init_ref, out_ref, carry_ref):
        rs.unpack_slot_to_carry(
            in_ref,
            carry_ref,
            has_init_ref[0] != 0,
            n_v=n_v,
            d_k=d_k,
            fmt=fmt,
        )
        rs.pack_carry_to_slot(
            carry_ref,
            out_ref,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            fmt=fmt,
        )

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.uint32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=(pltpu.VMEM((1, n_v, d_k, d_v), jnp.float32),),
    )(in_slot, has_init_arr)


def _normalize_idx(idx: Any) -> Any:
    if not isinstance(idx, tuple):
        idx = (idx,)
    return tuple(
        slice(k.start, k.start + k.size) if isinstance(k, pl.Slice) else k for k in idx
    )


class _FakeSlotRef:
    """In-memory stand-in for a Pallas VMEM ref supporting `pl.ds` slices."""

    def __init__(self, arr: jax.Array):
        self.arr = arr

    def __getitem__(self, idx: Any) -> jax.Array:
        return self.arr[_normalize_idx(idx)]

    def __setitem__(self, idx: Any, val: jax.Array) -> None:
        self.arr = self.arr.at[_normalize_idx(idx)].set(val)


def _to_u32_slot(state: jax.Array, fmt: rs.RecurrentStateFormat) -> jax.Array:
    packed = rs.pack_state(state, fmt)
    if not rs.is_scaled(fmt):
        packed = rs._host_bitcast_dim_m2(packed, jnp.uint32)
    return packed[:, None]


def _from_u32_slot(
    slot_u32: jax.Array, fmt: rs.RecurrentStateFormat, *, n_v: int, d_k: int
) -> jax.Array:
    if rs.is_scaled(fmt):
        return rs.unpack_state(slot_u32, fmt, n_v=n_v, d_k=d_k)
    dtype = rs._unscaled_dtype(fmt)
    return rs._host_bitcast_dim_m2(slot_u32, dtype).astype(jnp.float32)


def _wide_state(seed: int, n_v: int = 8) -> jax.Array:
    rng = np.random.default_rng(seed)
    col_scale = np.exp(rng.uniform(np.log(1e-3), np.log(1e3), (1, 1, 1, 128)))
    state = rng.standard_normal((2, n_v, 128, 128)) * col_scale
    return jnp.asarray(state, jnp.float32)


class RecurrentStateTest(parameterized.TestCase):
    @parameterized.product(fmt=_SCALED_FORMATS, n_v=(4, 6, 8, 16))
    def test_host_pack_matches_block_pack_and_unpack_head(self, fmt, n_v):
        state = _wide_state(0, n_v=n_v)
        packed = rs.pack_state(state, fmt)
        self.assertEqual(packed.dtype, jnp.uint32)
        self.assertEqual(packed.shape, (2, rs.packed_rows(fmt, n_v, 128), 128))

        rph = rs.data_rows_per_head(fmt, 128)
        data_rows = n_v * rph
        unpacked_all = rs.unpack_state(packed, fmt, n_v=n_v, d_k=128)
        unpack_head_fn = jax.jit(lambda r, s: rs.unpack_head(r, s, fmt))

        @jax.jit
        def store_all(state_in):
            slot = _FakeSlotRef(
                jnp.zeros((2, 1, rs.packed_rows(fmt, n_v, 128), 128), jnp.uint32)
            )
            for b in range(state_in.shape[0]):
                for v0 in range(0, n_v, 8):
                    blk = min(8, n_v - v0)
                    rs.store_head_block(
                        cast(jax.Ref, slot),
                        b,
                        0,
                        v0,
                        [state_in[b, v0 + i] for i in range(blk)],
                        n_v=n_v,
                        d_k=128,
                        d_v=128,
                        fmt=fmt,
                    )
            return slot.arr[:, 0]

        stored = store_all(state)
        np.testing.assert_array_equal(np.asarray(packed), np.asarray(stored))
        # Verify unpacked values accurately reconstruct the original state and
        # match an independent NumPy quantization reference (catches symmetric
        # pack/unpack bugs such as all-zero output).
        state_np = np.asarray(state, dtype=np.float32)
        unpacked_np = np.asarray(unpacked_all, dtype=np.float32)
        self.assertLess(_rel_l2(state_np, unpacked_np), 5e-2)
        ref_np = _independent_fp8_quantize_ref(state_np)
        np.testing.assert_array_equal(unpacked_np, ref_np)
        for b in range(state.shape[0]):
            for v in range(n_v):
                scale_f32 = jax.lax.bitcast_convert_type(
                    stored[b, data_rows + v : data_rows + v + 1], jnp.float32
                )
                np.testing.assert_array_equal(
                    np.asarray(unpacked_all[b, v]),
                    np.asarray(
                        unpack_head_fn(stored[b, v * rph : (v + 1) * rph], scale_f32)
                    ),
                )

    @parameterized.parameters(*rs.RecurrentStateFormat)
    def test_zero_uint32_buffer_unpacks_to_zero(self, fmt):
        n_v, d_k, d_v = 8, 128, 128
        if rs.is_scaled(fmt):
            zeros = jnp.zeros((3, rs.packed_rows(fmt, n_v, d_k), d_v), jnp.uint32)
        else:
            dtype = jnp.bfloat16 if fmt == rs.RecurrentStateFormat.BF16 else jnp.float32
            zeros = jnp.zeros((3, n_v, d_k, d_v), dtype)
        unpacked = rs.unpack_state(zeros, fmt, n_v=n_v, d_k=d_k)
        np.testing.assert_array_equal(np.asarray(unpacked), 0.0)

    @parameterized.parameters(_SCALED_FORMATS)
    def test_pack_unpack_matches_under_jit(self, fmt):
        state = _wide_state(1, n_v=8)
        eager = rs.unpack_state(rs.pack_state(state, fmt), fmt, n_v=8, d_k=128)
        jitted = jax.jit(
            lambda s: rs.unpack_state(rs.pack_state(s, fmt), fmt, n_v=8, d_k=128)
        )(state)
        np.testing.assert_array_equal(np.asarray(jitted), np.asarray(eager))

    def test_slot_bytes(self):
        n_v, d_k, d_v = 64, 128, 128
        self.assertEqual(
            rs.slot_bytes(rs.RecurrentStateFormat.F32, n_v, d_k, d_v),
            64 * 65536,
        )
        self.assertEqual(
            rs.slot_bytes(rs.RecurrentStateFormat.BF16, n_v, d_k, d_v),
            64 * 32768,
        )
        self.assertEqual(
            rs.slot_bytes(rs.RecurrentStateFormat.FP8, n_v, d_k, d_v),
            64 * 16896,
        )

    def test_slot_window_shape(self):
        n_v, d_k, d_v = 10, 128, 128
        self.assertEqual(
            rs.slot_window_shape(rs.RecurrentStateFormat.F32, n_v, d_k, d_v),
            (10, 128, 128),
        )
        self.assertEqual(
            rs.slot_window_shape(rs.RecurrentStateFormat.BF16, n_v, d_k, d_v),
            (10, 64, 128),
        )
        self.assertEqual(
            rs.slot_window_shape(rs.RecurrentStateFormat.FP8, n_v, d_k, d_v),
            (10 * 32 + 16, 128),
        )

    @parameterized.product(fmt=tuple(rs.RecurrentStateFormat), n_v=(6, 10))
    def test_recurrent_state_stream_roundtrip(self, fmt, n_v):
        d_k, d_v, window_size = 128, 128, 2
        state = _wide_state(3, n_v=n_v)
        in_slot = _to_u32_slot(state, fmt)
        # Poison sequence 1 with NaN bits to verify has_init=False masks raw bits.
        in_slot = in_slot.at[1].set(jnp.uint32(0xFFFFFFFF))
        out_shape = (2, window_size, *rs.slot_window_shape(fmt, n_v, d_k, d_v))
        expected_s0 = rs.unpack_state(
            rs.pack_state(state[:1], fmt), fmt, n_v=n_v, d_k=d_k
        )[0]

        @jax.jit
        def run(slot_in):
            in_ref = _FakeSlotRef(slot_in)
            out_ref = _FakeSlotRef(jnp.zeros(out_shape, jnp.uint32))
            stream = rs.RecurrentStateStream(
                cast(jax.Ref, in_ref),
                cast(jax.Ref, out_ref),
                fmt=fmt,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                window_size=window_size,
                seq_has_init=[jnp.bool_(True), jnp.bool_(False)],
            )
            loaded = []
            for s in range(2):
                seq_heads = []
                for v in range(n_v):
                    h = stream.load_head(s, v)
                    seq_heads.append(h)
                    for w_idx in range(window_size):
                        stream.store_ckpt_head(s, v, w_idx, h * float(w_idx + 1))
                loaded.append(jnp.stack(seq_heads, axis=0))
            return jnp.stack(loaded, axis=0), out_ref.arr

        loaded, out_u32 = run(in_slot)
        np.testing.assert_array_equal(np.asarray(loaded[0]), np.asarray(expected_s0))
        np.testing.assert_array_equal(np.asarray(loaded[1]), 0.0)
        unpacked_out = _from_u32_slot(out_u32, fmt, n_v=n_v, d_k=d_k)
        expected_w0 = rs.unpack_state(
            rs.pack_state(expected_s0, fmt), fmt, n_v=n_v, d_k=d_k
        )
        expected_w1 = rs.unpack_state(
            rs.pack_state(expected_s0 * 2.0, fmt), fmt, n_v=n_v, d_k=d_k
        )
        np.testing.assert_array_equal(
            np.asarray(unpacked_out[0, 0]), np.asarray(expected_w0)
        )
        np.testing.assert_array_equal(
            np.asarray(unpacked_out[0, 1]), np.asarray(expected_w1)
        )
        np.testing.assert_array_equal(np.asarray(unpacked_out[1]), 0.0)

    @parameterized.parameters(*rs.RecurrentStateFormat)
    def test_carry_pack_unpack_roundtrip(self, fmt):
        n_v, d_k, d_v = 6, 128, 128
        state = _wide_state(4, n_v=n_v)[:1]
        in_slot = _to_u32_slot(state, fmt)
        out_shape = (1, 1, *rs.slot_window_shape(fmt, n_v, d_k, d_v))

        @jax.jit
        def run(slot_in):
            in_ref = _FakeSlotRef(slot_in)
            carry_ref = _FakeSlotRef(jnp.zeros((1, n_v, d_k, d_v), jnp.float32))
            out_ref = _FakeSlotRef(jnp.zeros(out_shape, jnp.uint32))
            rs.unpack_slot_to_carry(
                cast(jax.Ref, in_ref),
                cast(jax.Ref, carry_ref),
                jnp.bool_(True),
                n_v=n_v,
                d_k=d_k,
                fmt=fmt,
            )
            rs.pack_carry_to_slot(
                cast(jax.Ref, carry_ref),
                cast(jax.Ref, out_ref),
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                fmt=fmt,
            )
            return carry_ref.arr, out_ref.arr

        carry, out_u32 = run(in_slot)
        expected = rs.unpack_state(rs.pack_state(state, fmt), fmt, n_v=n_v, d_k=d_k)
        np.testing.assert_array_equal(np.asarray(carry), np.asarray(expected))
        np.testing.assert_array_equal(
            np.asarray(out_u32), np.asarray(_to_u32_slot(expected, fmt))
        )

    def test_recurrent_state_stream_rejects_out_of_order_heads(self):
        n_v, d_k, d_v = 8, 128, 128
        fmt = rs.RecurrentStateFormat.FP8
        slot_shape = (2, 1, *rs.slot_window_shape(fmt, n_v, d_k, d_v))
        head = jnp.zeros((d_k, d_v), jnp.float32)

        def _make_stream():
            return rs.RecurrentStateStream(
                cast(jax.Ref, _FakeSlotRef(jnp.zeros(slot_shape, jnp.uint32))),
                cast(jax.Ref, _FakeSlotRef(jnp.zeros(slot_shape, jnp.uint32))),
                fmt=fmt,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                window_size=1,
                seq_has_init=[jnp.bool_(True), jnp.bool_(True)],
            )

        with self.assertRaisesRegex(ValueError, "ascending order"):
            _make_stream().store_ckpt_head(0, 1, 0, head)

        # Cross-sequence interleaving within an unfinished 8-head block fails.
        stream_seq = _make_stream()
        stream_seq.store_ckpt_head(0, 0, 0, head)
        with self.assertRaisesRegex(ValueError, "contiguous ascending"):
            stream_seq.store_ckpt_head(1, 1, 0, head)

        # Skipping to another 8-head block before finishing block 0 fails.
        slot_16 = (1, 1, *rs.slot_window_shape(fmt, 16, d_k, d_v))
        stream_blk = rs.RecurrentStateStream(
            cast(jax.Ref, _FakeSlotRef(jnp.zeros(slot_16, jnp.uint32))),
            cast(jax.Ref, _FakeSlotRef(jnp.zeros(slot_16, jnp.uint32))),
            fmt=fmt,
            n_v=16,
            d_k=d_k,
            d_v=d_v,
            window_size=1,
            seq_has_init=[jnp.bool_(True)],
        )
        stream_blk.store_ckpt_head(0, 0, 0, head)
        with self.assertRaisesRegex(ValueError, "contiguous ascending"):
            stream_blk.store_ckpt_head(0, 9, 0, head)

        # Out-of-bounds head index fails on both load and store.
        with self.assertRaisesRegex(ValueError, "Head index v"):
            _make_stream().load_head(0, n_v)
        with self.assertRaisesRegex(ValueError, "Head index v"):
            _make_stream().store_ckpt_head(0, n_v, 0, head)

    def test_recurrent_state_stream_carry_and_flush_cond(self):
        n_v, d_k, d_v = 6, 128, 128
        fmt = rs.RecurrentStateFormat.FP8
        state = _wide_state(7, n_v=n_v)[:1]
        in_slot = _to_u32_slot(state, fmt)
        carry_init = _wide_state(8, n_v=n_v)[:1]
        out_shape = (1, 1, *rs.slot_window_shape(fmt, n_v, d_k, d_v))

        @jax.jit
        def run(slot_in, carry_in, is_first, flush_flag):
            in_ref = jax.new_ref(slot_in)
            out_ref = jax.new_ref(jnp.zeros(out_shape, jnp.uint32))
            carry_ref = jax.new_ref(carry_in)
            stream = rs.RecurrentStateStream(
                in_ref,
                out_ref,
                fmt=fmt,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                window_size=1,
                seq_has_init=[jnp.bool_(True)],
                carry_ref=carry_ref,
                seq_is_first_tile=[is_first],
                flush_cond=flush_flag,
            )
            loaded = []
            for v in range(n_v):
                h = stream.load_head(0, v)
                loaded.append(h)
                stream.store_ckpt_head(0, v, 0, h * 0.5)
            return jnp.stack(loaded, axis=0), carry_ref[...], out_ref[...]

        # First tile reads in_slot (not carry_ref); flush_cond=False skips out_slot.
        loaded_first, carry_first, out_first_skipped = run(
            in_slot, carry_init, jnp.bool_(True), jnp.bool_(False)
        )
        want_from_slot = rs.unpack_state(in_slot[:, 0], fmt, n_v=n_v, d_k=d_k)[0]
        np.testing.assert_array_equal(
            np.asarray(loaded_first), np.asarray(want_from_slot)
        )
        np.testing.assert_array_equal(
            np.asarray(carry_first[0]), np.asarray(want_from_slot * 0.5)
        )
        np.testing.assert_array_equal(np.asarray(out_first_skipped), 0)

        # Non-first tile reads carry_ref; flush_cond=False skips out_slot write.
        loaded_c, carry_after, out_skipped = run(
            in_slot, carry_init, jnp.bool_(False), jnp.bool_(False)
        )
        np.testing.assert_array_equal(np.asarray(loaded_c), np.asarray(carry_init[0]))
        np.testing.assert_array_equal(
            np.asarray(carry_after[0]), np.asarray(carry_init[0] * 0.5)
        )
        np.testing.assert_array_equal(np.asarray(out_skipped), 0)

        # flush_cond=True flushes packed FP8 output.
        _, _, out_flushed = run(in_slot, carry_init, jnp.bool_(False), jnp.bool_(True))
        want_out = _to_u32_slot(carry_init * 0.5, fmt)
        np.testing.assert_array_equal(np.asarray(out_flushed), np.asarray(want_out))

    def test_unpack_slot_to_carry_masks_nan_when_uninitialized(self):
        n_v, d_k, d_v = 6, 128, 128
        fmt = rs.RecurrentStateFormat.FP8
        poison = jnp.full(
            (1, 1, *rs.slot_window_shape(fmt, n_v, d_k, d_v)),
            jnp.uint32(0xFFFFFFFF),
        )

        @jax.jit
        def run(slot_in):
            in_ref = _FakeSlotRef(slot_in)
            carry_ref = _FakeSlotRef(jnp.ones((1, n_v, d_k, d_v), jnp.float32))
            rs.unpack_slot_to_carry(
                cast(jax.Ref, in_ref),
                cast(jax.Ref, carry_ref),
                jnp.bool_(False),
                n_v=n_v,
                d_k=d_k,
                fmt=fmt,
            )
            return carry_ref.arr

        np.testing.assert_array_equal(np.asarray(run(poison)), 0.0)

    def test_resolve_format_and_slot_granules(self):
        self.assertEqual(
            rs.resolve_format("fp8", jnp.float32), rs.RecurrentStateFormat.FP8
        )
        self.assertEqual(rs.resolve_format(None, None), rs.RecurrentStateFormat.F32)
        self.assertEqual(
            rs.resolve_format(None, jnp.float32), rs.RecurrentStateFormat.F32
        )
        self.assertEqual(
            rs.resolve_format(None, jnp.bfloat16), rs.RecurrentStateFormat.BF16
        )
        with self.assertRaisesRegex(NotImplementedError, "Unsupported recurrent_dtype"):
            rs.resolve_format(None, jnp.uint32)
        self.assertEqual(
            rs.slot_granules(rs.RecurrentStateFormat.FP8, 8, 128, 128, 128), 264
        )
        with self.assertRaisesRegex(ValueError, "divisible by lane_bytes"):
            rs.slot_granules(rs.RecurrentStateFormat.FP8, 8, 128, 63, 128)

    def test_pallas_roundtrip_kernels_trace_for_tpu(self):
        """Verifies real Pallas VMEM helpers trace cleanly under an abstract TPU."""
        n_v, d_k, d_v = 6, 128, 128
        fmt = rs.RecurrentStateFormat.FP8
        in_slot = jnp.zeros(
            (2, 1, *rs.slot_window_shape(fmt, n_v, d_k, d_v)), jnp.uint32
        )
        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        with use_abstract_mesh(mesh):
            stream_jaxpr = str(
                jax.make_jaxpr(
                    lambda s, h: _pallas_stream_roundtrip(
                        s, h, fmt=fmt, n_v=n_v, d_k=d_k, d_v=d_v, window_size=2
                    )
                )(in_slot, jnp.array([1, 0], jnp.int32))
            )
            carry_jaxpr = str(
                jax.make_jaxpr(
                    lambda s, h: _pallas_carry_roundtrip(
                        s, h, fmt=fmt, n_v=n_v, d_k=d_k, d_v=d_v
                    )
                )(in_slot[:1], jnp.array([1], jnp.int32))
            )
        self.assertIn("pallas_call", stream_jaxpr)
        self.assertIn("pallas_call", carry_jaxpr)

    def test_tiling_vmem_footprint_with_fp8_state(self):
        n_kq, n_v, d_k, d_v = 8, 64, 128, 128
        vmem_limit = 64 * 1024 * 1024
        dec_f32, _ = tiling.get_tile_sizes(
            batch_size=16,
            num_seqs=16,
            padded_batch_size=16,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=4,
            conv_state_dim_size=2 * n_kq * d_k + n_v * d_v,
            act_out_dtype=jnp.dtype(jnp.bfloat16),
            recurrent_state_dtype=jnp.dtype(jnp.float32),
            num_lanes=128,
            decode_vmem_limit_bytes=vmem_limit,
            mixed_vmem_limit_bytes=vmem_limit,
            window_size=4,
            recurrent_state_format=rs.RecurrentStateFormat.F32,
        )
        dec_fp8, _ = tiling.get_tile_sizes(
            batch_size=16,
            num_seqs=16,
            padded_batch_size=16,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=4,
            conv_state_dim_size=2 * n_kq * d_k + n_v * d_v,
            act_out_dtype=jnp.dtype(jnp.bfloat16),
            recurrent_state_dtype=jnp.dtype(jnp.uint32),
            num_lanes=128,
            decode_vmem_limit_bytes=vmem_limit,
            mixed_vmem_limit_bytes=vmem_limit,
            window_size=4,
            recurrent_state_format=rs.RecurrentStateFormat.FP8,
        )
        self.assertGreater(dec_fp8, dec_f32)

    @parameterized.parameters(1, 4, 6, 8)
    def test_max_abs_8_heads_butterfly_matches_per_head_max(self, blk):
        state = _wide_state(2, n_v=blk)[0].astype(jnp.bfloat16)
        heads = [state[i] for i in range(blk)]
        got = np.asarray(rs._max_abs_8_heads(heads, 128), np.float32)
        self.assertEqual(got.shape, (8, 128))
        for i in range(blk):
            want_i = np.max(np.abs(np.asarray(heads[i], np.float32)), axis=0)
            np.testing.assert_array_equal(got[i], want_i)
        if blk < 8:
            np.testing.assert_array_equal(got[blk:], 0.0)

    def test_format_and_dtype_guards(self):
        dummy_ref = cast(jax.Ref, _FakeSlotRef(jnp.zeros((1, 1, 8, 128), jnp.uint32)))
        with self.assertRaisesRegex(ValueError, "requires FP8"):
            rs.load_scale_tile(
                dummy_ref, 0, 0, 0, n_v=8, d_k=128, fmt=rs.RecurrentStateFormat.F32
            )
        with self.assertRaisesRegex(ValueError, "requires FP8"):
            rs.load_head(
                dummy_ref,
                0,
                0,
                0,
                d_k=128,
                fmt=rs.RecurrentStateFormat.F32,
                scale_row=jnp.ones((1, 128), jnp.float32),
            )
        with self.assertRaisesRegex(ValueError, "uint32"):
            rs.unpack_state(
                jnp.zeros(
                    (1, rs.packed_rows(rs.RecurrentStateFormat.FP8, 8, 128), 128),
                    jnp.float32,
                ),
                rs.RecurrentStateFormat.FP8,
                n_v=8,
                d_k=128,
            )
        with self.assertRaisesRegex(ValueError, "Expected .* packed rows"):
            rs.unpack_state(
                jnp.zeros((1, 10, 128), jnp.uint32),
                rs.RecurrentStateFormat.FP8,
                n_v=8,
                d_k=128,
            )
        with self.assertRaisesRegex(ValueError, "multiple of 32"):
            rs.data_rows_per_head(rs.RecurrentStateFormat.FP8, 48)
        with self.assertRaisesRegex(ValueError, "must be even"):
            rs.data_rows_per_head(rs.RecurrentStateFormat.BF16, 127)
        with self.assertRaisesRegex(ValueError, "multiple of 8"):
            rs.load_scale_tile(
                dummy_ref, 0, 0, 4, n_v=8, d_k=128, fmt=rs.RecurrentStateFormat.FP8
            )
        with self.assertRaisesRegex(ValueError, "v0 must be in"):
            rs.load_scale_tile(
                dummy_ref, 0, 0, 8, n_v=8, d_k=128, fmt=rs.RecurrentStateFormat.FP8
            )
        with self.assertRaisesRegex(ValueError, "multiple of 8"):
            rs.store_head_block(
                dummy_ref,
                0,
                0,
                4,
                [jnp.zeros((128, 128), jnp.float32)],
                n_v=8,
                d_k=128,
                d_v=128,
                fmt=rs.RecurrentStateFormat.FP8,
            )
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            rs.store_head_block(
                dummy_ref,
                0,
                0,
                8,
                [jnp.zeros((128, 128), jnp.float32)],
                n_v=8,
                d_k=128,
                d_v=128,
                fmt=rs.RecurrentStateFormat.FP8,
            )
        for bad_len in (0, 9):
            with self.assertRaisesRegex(ValueError, "heads_blk length"):
                rs.store_head_block(
                    dummy_ref,
                    0,
                    0,
                    0,
                    [jnp.zeros((128, 128), jnp.float32)] * bad_len,
                    n_v=8,
                    d_k=128,
                    d_v=128,
                    fmt=rs.RecurrentStateFormat.FP8,
                )
        with self.assertRaisesRegex(ValueError, "load_dtype"):
            rs.RecurrentStateStream(
                dummy_ref,
                dummy_ref,
                fmt=rs.RecurrentStateFormat.FP8,
                n_v=8,
                d_k=128,
                d_v=128,
                window_size=1,
                seq_has_init=[jnp.bool_(True)],
                load_dtype=jnp.float16,
            )
        with self.assertRaisesRegex(ValueError, "carry_ref"):
            rs.RecurrentStateStream(
                dummy_ref,
                dummy_ref,
                fmt=rs.RecurrentStateFormat.FP8,
                n_v=8,
                d_k=128,
                d_v=128,
                window_size=1,
                seq_has_init=[jnp.bool_(True)],
                carry_ref=dummy_ref,
                seq_is_first_tile=[jnp.bool_(True)],
                load_dtype=jnp.bfloat16,
            )
        fp8_stream = rs.RecurrentStateStream(
            dummy_ref,
            dummy_ref,
            fmt=rs.RecurrentStateFormat.FP8,
            n_v=8,
            d_k=128,
            d_v=128,
            window_size=1,
            seq_has_init=[jnp.bool_(True)],
            load_dtype=jnp.bfloat16,
        )
        for bad_v in (-1, 8):
            with self.assertRaisesRegex(ValueError, "Head index v"):
                fp8_stream.load_head(0, bad_v)

    @parameterized.parameters((2, 16), (3, 6))
    def test_fp8_bf16_decode_matches_float32_update(self, n_kq, n_v):
        fmt = rs.RecurrentStateFormat.FP8
        n_seq, d = 2, 128
        f32 = jnp.dtype(jnp.float32)
        cfg = config.GDNConfig(
            mode=config.GDNMode.BATCHED,
            dtypes=config.Dtypes(f32, f32, f32, jnp.dtype(jnp.uint32), f32),
            batch_size=n_seq,
            dim_size=(2 * n_kq + n_v) * d,
            kernel_size=4,
            tile_size=n_seq,
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            kq_head_dim=d,
            v_head_dim=d,
            recurrent_state_format=fmt,
        )
        rng = np.random.default_rng(4)

        def normal(*shape: int) -> jax.Array:
            return jnp.asarray(rng.standard_normal(shape), jnp.float32)

        state = _wide_state(4, n_v=n_v)
        q = normal(n_seq, n_kq, 1, 1, d)
        k = normal(n_seq, n_kq, 1, 1, d)
        v = normal(n_seq, n_v, 1, 1, d)
        b = normal(n_seq, 1, 1, 1, n_v)
        a = normal(n_seq, 1, 1, 1, n_v)
        # Heads with short and with long memory.
        a_log = jnp.asarray(rng.uniform(-3.0, 0.5, n_v), jnp.float32)
        in_slot = _to_u32_slot(state, fmt)

        # Independent float64 mathematical reference on the unpacked initial state.
        init_unpacked = np.asarray(
            rs.unpack_state(in_slot[:, 0], fmt, n_v=n_v, d_k=d), dtype=np.float64
        )
        q_np = np.repeat(np.asarray(q[:, :, 0, 0, :], np.float64), n_v // n_kq, axis=1)
        k_np = np.repeat(np.asarray(k[:, :, 0, 0, :], np.float64), n_v // n_kq, axis=1)
        q_np = (q_np / np.sqrt(np.sum(q_np * q_np, axis=-1, keepdims=True) + 1e-6)) * (
            d**-0.5
        )
        k_np = k_np / np.sqrt(np.sum(k_np * k_np, axis=-1, keepdims=True) + 1e-6)
        v_np = np.asarray(v[:, :, 0, 0, :], np.float64)
        beta_np = 1.0 / (1.0 + np.exp(-np.asarray(b[:, 0, 0, 0, :], np.float64)))
        a_np = np.asarray(a[:, 0, 0, 0, :], np.float64)
        decay_np = np.exp(
            -np.exp(np.asarray(a_log, np.float64)[None, :]) * np.log1p(np.exp(a_np))
        )
        state_upd_ref = init_unpacked * decay_np[:, :, None, None]
        v_pred_ref = np.einsum("shd,shdm->shm", k_np, state_upd_ref)
        v_new_ref = beta_np[:, :, None] * (v_np - v_pred_ref)
        ref_state = state_upd_ref + np.einsum("shd,shm->shdm", k_np, v_new_ref)
        ref_out = np.einsum("shd,shdm->shm", q_np, ref_state)[:, None, :, :]

        stored_dtypes: list[jnp.dtype] = []

        def decode(slot_in: jax.Array, use_bf16: bool):
            in_ref = _FakeSlotRef(slot_in)
            out_shape = (n_seq, 1, *rs.slot_window_shape(fmt, n_v, d, d))
            out_ref = _FakeSlotRef(jnp.zeros(out_shape, jnp.uint32))
            stream = rs.RecurrentStateStream(
                cast(jax.Ref, in_ref),
                cast(jax.Ref, out_ref),
                fmt=fmt,
                n_v=n_v,
                d_k=d,
                d_v=d,
                window_size=1,
                seq_has_init=[jnp.bool_(True)] * n_seq,
                load_dtype=jnp.bfloat16 if use_bf16 else jnp.float32,
            )

            def _record_and_store(s, h, w_idx, head):
                if use_bf16:
                    stored_dtypes.append(head.dtype)
                stream.store_ckpt_head(s, h, w_idx, head)

            out, _ = compute_gdn.recurrent_gdn(
                jnp.ones((n_seq,), jnp.int32),
                q,
                k,
                v,
                b,
                a,
                None,
                a_log,
                jnp.zeros((n_v,), jnp.float32),
                cfg,
                load_head_fn=stream.load_head,
                store_ckpt_head_fn=_record_and_store,
            )
            return out, out_ref.arr[:, 0]

        want_out, want_packed = jax.jit(lambda x: decode(x, False))(in_slot)
        got_out, got_packed = jax.jit(lambda x: decode(x, True))(in_slot)
        want_state = rs.unpack_state(want_packed, fmt, n_v=n_v, d_k=d)
        got_state = rs.unpack_state(got_packed, fmt, n_v=n_v, d_k=d)

        self.assertTrue(stored_dtypes)
        self.assertTrue(all(dt == jnp.bfloat16 for dt in stored_dtypes))
        np.testing.assert_array_less(
            _rel_error(want_out, got_out, -1), _MAX_FP8_DECODE_REL_ERROR
        )
        np.testing.assert_array_less(
            _rel_error(jnp.asarray(ref_out, jnp.float32), got_out, -1),
            _MAX_FP8_DECODE_REL_ERROR,
        )
        np.testing.assert_array_less(
            _rel_error(jnp.asarray(ref_state, jnp.float32), got_state, (-2, -1)),
            6e-2,
        )
        np.testing.assert_array_less(
            _rel_error(want_state, got_state, (-2, -1)),
            6e-2,
        )

    def test_load_head_bf16_masks_nan_when_uninitialized(self):
        fmt = rs.RecurrentStateFormat.FP8
        n_v, d = 8, 128
        in_shape = (1, 1, *rs.slot_window_shape(fmt, n_v, d, d))

        @jax.jit
        def load_head(raw_slot: jax.Array, has_init: jax.Array):
            in_ref = _FakeSlotRef(raw_slot)
            out_ref = _FakeSlotRef(jnp.zeros(in_shape, jnp.uint32))
            stream = rs.RecurrentStateStream(
                cast(jax.Ref, in_ref),
                cast(jax.Ref, out_ref),
                fmt=fmt,
                n_v=n_v,
                d_k=d,
                d_v=d,
                window_size=1,
                seq_has_init=[has_init],
                load_dtype=jnp.bfloat16,
            )
            return stream.load_head(0, 0)

        # 0xFFFFFFFF poisons every float8_e4m3fn byte and the float32 scale with NaN.
        poison = jnp.full(in_shape, jnp.uint32(0xFFFFFFFF))

        head_uninit = load_head(poison, jnp.bool_(False))
        self.assertEqual(head_uninit.dtype, jnp.bfloat16)
        np.testing.assert_array_equal(
            np.asarray(head_uninit), np.zeros((d, d), jnp.bfloat16)
        )

        # Control: without masking, the poisoned slot produces NaN values.
        head_init = load_head(poison, jnp.bool_(True))
        self.assertEqual(head_init.dtype, jnp.bfloat16)
        self.assertTrue(bool(jnp.all(jnp.isnan(head_init))))

    def test_in_memory_bf16_state_prev_preserves_float32_update(self):
        n_seq, n_kq, n_v, d = 1, 2, 4, 128
        f32 = jnp.dtype(jnp.float32)
        cfg = config.GDNConfig(
            mode=config.GDNMode.BATCHED,
            dtypes=config.Dtypes(f32, f32, f32, jnp.dtype(jnp.bfloat16), f32),
            batch_size=n_seq,
            dim_size=(2 * n_kq + n_v) * d,
            kernel_size=4,
            tile_size=n_seq,
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            kq_head_dim=d,
            v_head_dim=d,
        )
        z = jnp.zeros
        _, states = compute_gdn.recurrent_gdn(
            jnp.ones((n_seq,), jnp.int32),
            z((n_seq, n_kq, 1, 1, d), jnp.float32),
            z((n_seq, n_kq, 1, 1, d), jnp.float32),
            z((n_seq, n_v, 1, 1, d), jnp.float32),
            z((n_seq, 1, 1, 1, n_v), jnp.float32),
            z((n_seq, 1, 1, 1, n_v), jnp.float32),
            z((n_seq, n_v, d, d), jnp.bfloat16),
            z((n_v,), jnp.float32),
            z((n_v,), jnp.float32),
            cfg,
        )
        self.assertEqual(states.dtype, jnp.float32)


@pytest.mark.tpu_test
class RecurrentStateTpuTest(parameterized.TestCase):
    """Real TPU Pallas VMEM roundtrip tests for `recurrent_state` helpers."""

    @parameterized.product(fmt=tuple(rs.RecurrentStateFormat), n_v=(6, 10))
    def test_pallas_stream_and_carry_roundtrip_on_tpu(self, fmt, n_v):
        d_k, d_v, window_size = 128, 128, 2
        state = _wide_state(11, n_v=n_v)
        in_slot = _to_u32_slot(state, fmt)
        # Poison sequence 1 with NaN bits to verify has_init=False on real VMEM.
        in_slot = in_slot.at[1].set(jnp.uint32(0xFFFFFFFF))

        out_stream = _pallas_stream_roundtrip(
            in_slot,
            jnp.array([1, 0], jnp.int32),
            fmt=fmt,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            window_size=window_size,
        )
        unpacked = np.asarray(
            _from_u32_slot(out_stream, fmt, n_v=n_v, d_k=d_k), dtype=np.float32
        )
        state0 = np.asarray(state[0], dtype=np.float32)
        tol = {
            rs.RecurrentStateFormat.F32: 1e-6,
            rs.RecurrentStateFormat.BF16: 1e-2,
            rs.RecurrentStateFormat.FP8: 5e-2,
        }[fmt]
        self.assertLess(_rel_l2(state0, unpacked[0, 0]), tol)
        self.assertLess(_rel_l2(state0 * 2.0, unpacked[0, 1]), tol)
        np.testing.assert_array_equal(unpacked[1], 0.0)

        out_carry = _pallas_carry_roundtrip(
            in_slot[:1],
            jnp.array([1], jnp.int32),
            fmt=fmt,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        unpacked_carry = np.asarray(
            _from_u32_slot(out_carry, fmt, n_v=n_v, d_k=d_k), dtype=np.float32
        )
        self.assertLess(_rel_l2(state0, unpacked_carry[0, 0]), tol)

        out_carry_uninit = _pallas_carry_roundtrip(
            in_slot[1:2],
            jnp.array([0], jnp.int32),
            fmt=fmt,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        unpacked_uninit = np.asarray(
            _from_u32_slot(out_carry_uninit, fmt, n_v=n_v, d_k=d_k),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(unpacked_uninit, 0.0)


if __name__ == "__main__":
    absltest.main()
