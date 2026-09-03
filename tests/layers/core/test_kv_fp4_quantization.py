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
"""Correctness tests for `_quantize_to_fp4_packed`, the KV-cache FP4 path.

`_quantize_to_fp4_packed` replaces a nearest-value search over the 16-entry
e2m1 table with seven threshold comparisons, so the properties worth pinning
are (a) that it rounds exactly like a real e2m1 cast, ties included, and
(b) that the nibble layout and scale it emits are the ones the batched RPA
kernel reads back.

The rounding oracle is `ml_dtypes.float4_e2m1fn`, which is the same dtype
object `jnp.float4_e2m1fn` exposes, so these tests compare against the same
cast the JAX-side kernel uses -- without pulling JAX (and a TPU client) into
the test process.

Run on CPU:
    pytest tests/layers/core/test_kv_fp4_quantization.py -v -p no:cacheprovider
Run on TPU:
    pytest tests/layers/core/test_kv_fp4_quantization.py -v --use-tpu
"""

import ml_dtypes
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.core.quantization import (  # isort: skip
    FP4_E2M1_MAX, _quantize_to_fp4_packed, _safe_inverse_scale,
    apply_hadamard_transform, pack_fp4_indices, quantize_kv_fp4_per_tensor,
    unpack_uint8_to_fp4)

# Value -> e2m1 code for every exact halfway point on the magnitude ladder
# (0, .5, 1, 1.5, 2, 3, 4, 6), plus the ladder entries themselves. Written out
# rather than derived so the round-to-nearest-EVEN choice is visible: 0.75
# rounds up to 1.0 and 1.25 rounds down to 1.0, because 1.0 has the even
# significand in both pairs. These are the cases the old argmin path got
# wrong (it always rounded ties toward zero).
_POSITIVE_TIE_CASES = (
    (0.0, 0),
    (0.25, 0),  # tie 0 / 0.5   -> 0
    (0.5, 1),
    (0.75, 2),  # tie 0.5 / 1.0 -> 1.0
    (1.0, 2),
    (1.25, 2),  # tie 1.0 / 1.5 -> 1.0
    (1.5, 3),
    (1.75, 4),  # tie 1.5 / 2.0 -> 2.0
    (2.0, 4),
    (2.5, 4),  # tie 2.0 / 3.0 -> 2.0
    (3.0, 5),
    (3.5, 6),  # tie 3.0 / 4.0 -> 4.0
    (4.0, 6),
    (5.0, 6),  # tie 4.0 / 6.0 -> 4.0
    (6.0, 7),
)

# Negating a value sets bit 3, including for magnitudes that round to zero
# (-0.25 becomes -0.0, code 8). Exact -0.0 input is the one exception and gets
# its own test below. Every row here contains +-6.0, which forces the
# per-row scale to exactly 1.0 so the listed values *are* the scaled values.
_NEGATIVE_TIE_CASES = tuple(
    (-value, code | 0x8) for value, code in _POSITIVE_TIE_CASES if value != 0)

_TIE_CASES = _POSITIVE_TIE_CASES + _NEGATIVE_TIE_CASES + ((0.0, 0), )
assert len(_TIE_CASES) % 2 == 0, "tie row must have an even head_dim"


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """Split packed bytes back into one e2m1 code per element."""
    low = packed & 0x0F
    high = packed >> 4
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)


def _reference_codes(tensor: torch.Tensor) -> np.ndarray:
    """Ground-truth e2m1 codes for `tensor`, via a real fp4 cast.

    The scale and reciprocal multiply are recomputed on `tensor`'s own device
    and only the resulting float -> e2m1 step is handed to `ml_dtypes`. A TPU
    reciprocal that lands one ULP away from the CPU one would otherwise show
    up here as a rounding bug, which is not what this oracle is checking.
    """
    scale = (tensor.abs().amax(dim=-1, keepdim=True).to(torch.float32) /
             FP4_E2M1_MAX)
    scaled = (tensor * _safe_inverse_scale(scale)).to(torch.float32)
    as_fp4 = scaled.cpu().numpy().astype(ml_dtypes.float4_e2m1fn)
    return as_fp4.view(np.uint8)


def _assert_codes_match_fp4_cast(tensor: torch.Tensor) -> None:
    packed, _ = _quantize_to_fp4_packed(tensor)
    actual = _unpack_codes(packed).cpu().numpy()
    expected = _reference_codes(tensor)
    np.testing.assert_array_equal(actual, expected)


class TestOutputContract:
    """Shapes, dtypes and the packed nibble layout."""

    @pytest.mark.parametrize("shape", [
        (128, ),
        (7, 128),
        (4, 8, 128),
        (2, 3, 5, 64),
        (1, 2),
    ])
    def test_shapes_and_dtypes(self, device, shape):
        tensor = torch.randn(shape, device=device)

        packed, scale = _quantize_to_fp4_packed(tensor)

        assert packed.shape == (*shape[:-1], shape[-1] // 2)
        assert packed.dtype == torch.uint8
        assert scale.shape == (*shape[:-1], 1)
        assert scale.dtype == torch.float32

    def test_even_element_goes_in_the_low_nibble(self, device):
        # Scale is 1.0 (max magnitude 6), so these values are their own codes:
        # 0.5 -> 1, 6.0 -> 7, -1.0 -> 10, 2.0 -> 4.
        tensor = torch.tensor([[0.5, 6.0, -1.0, 2.0]],
                              dtype=torch.float32,
                              device=device)

        packed, _ = _quantize_to_fp4_packed(tensor)

        expected = torch.tensor([[1 | (7 << 4), 10 | (4 << 4)]],
                                dtype=torch.uint8,
                                device=device)
        assert torch.equal(packed, expected)

    def test_layout_agrees_with_pack_fp4_indices(self, device):
        """The packing must be the one `pack_fp4_indices` documents."""
        torch.manual_seed(0)
        tensor = torch.randn(64, 128, device=device)

        packed, _ = _quantize_to_fp4_packed(tensor)

        reference_indices = torch.from_numpy(
            _reference_codes(tensor).astype(np.int8)).to(device)
        assert torch.equal(packed, pack_fp4_indices(reference_indices))

    def test_unpacks_through_the_shared_helper(self, device):
        """`unpack_uint8_to_fp4` must read back what we wrote."""
        torch.manual_seed(0)
        tensor = torch.randn(16, 64, device=device)

        packed, scale = _quantize_to_fp4_packed(tensor)
        values = unpack_uint8_to_fp4(packed)

        assert values.shape == tensor.shape
        # Every unpacked value has to be on the e2m1 ladder.
        ladder = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.],
                              device=device)
        assert torch.isin(values.abs(), ladder).all()
        # And within one scaled ladder step of the input.
        assert torch.all((values * scale - tensor).abs() <= scale * 1.000001)

    @pytest.mark.parametrize("head_dim", [1, 3, 127])
    def test_odd_head_dim_raises(self, device, head_dim):
        tensor = torch.randn(2, head_dim, device=device)

        with pytest.raises(ValueError, match="even head_dim"):
            _quantize_to_fp4_packed(tensor)


class TestRounding:
    """Round-to-nearest-even against a real e2m1 cast."""

    def test_exact_tie_points_round_to_even(self, device):
        values = torch.tensor([[value for value, _ in _TIE_CASES]],
                              dtype=torch.float32,
                              device=device)
        expected = torch.tensor([[code for _, code in _TIE_CASES]],
                                dtype=torch.uint8,
                                device=device)

        packed, scale = _quantize_to_fp4_packed(values)

        # +-6.0 is present, so the scale is exactly 1 and no reciprocal
        # rounding can move a value off its tie point.
        assert scale.item() == 1.0
        assert torch.equal(_unpack_codes(packed), expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(1, 2), (64, 128), (8, 16, 64)])
    def test_matches_fp4_cast_on_random_input(self, device, dtype, shape):
        torch.manual_seed(0)
        tensor = torch.randn(shape, device=device).to(dtype)

        _assert_codes_match_fp4_cast(tensor)

    @pytest.mark.parametrize("spread", [1e-6, 1.0, 1e6])
    def test_matches_fp4_cast_across_magnitudes(self, device, spread):
        """Rows whose absolute scale is tiny or huge round the same way."""
        torch.manual_seed(0)
        tensor = torch.randn(32, 128, device=device) * spread

        _assert_codes_match_fp4_cast(tensor)

    def test_matches_fp4_cast_on_heavy_tailed_rows(self, device):
        """One dominant outlier per row pushes most elements toward zero."""
        torch.manual_seed(0)
        tensor = torch.randn(32, 128, device=device)
        tensor[:, 0] *= 500.0

        _assert_codes_match_fp4_cast(tensor)

    def test_matches_fp4_cast_within_ulps_of_every_midpoint(self, device):
        """The decision boundaries themselves, probed a few ULPs either way."""
        midpoints = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
        values = []
        for midpoint in midpoints:
            for steps in range(-3, 4):
                probe = np.float32(midpoint)
                for _ in range(abs(steps)):
                    probe = np.nextafter(
                        probe,
                        np.float32(np.inf if steps > 0 else -np.inf),
                        dtype=np.float32)
                values += [float(probe), -float(probe)]
        values.append(6.0)  # pins the scale to exactly 1.0
        if len(values) % 2:
            values.append(0.0)
        tensor = torch.tensor([values], dtype=torch.float32, device=device)

        _assert_codes_match_fp4_cast(tensor)

    def test_row_maximum_saturates_to_six(self, device):
        """The largest element of a row must land on +-6, never wrap.

        `scale` is `amax / 6`, so the largest element scales back to ~6.0 --
        but only to within the rounding of the reciprocal multiply. This
        pins that a value landing a hair above 6.0 still yields code 7 and
        not a wrapped-around code.
        """
        torch.manual_seed(0)
        tensor = torch.randn(512, 64, device=device)
        tensor[::2] *= -1.0  # exercise both signs of the row maximum

        packed, _ = _quantize_to_fp4_packed(tensor)
        codes = _unpack_codes(packed)

        argmax = tensor.abs().argmax(dim=-1, keepdim=True)
        max_codes = codes.gather(-1, argmax)
        negative = tensor.gather(-1, argmax) < 0
        expected = torch.where(negative, torch.full_like(max_codes, 15),
                               torch.full_like(max_codes, 7))
        assert torch.equal(max_codes, expected)

    def test_negatives_that_round_to_zero_keep_their_sign(self, device):
        """A small negative becomes -0.0 (code 8), matching the fp4 cast."""
        tensor = torch.tensor([[-1e-4, -0.2, 6.0, 1e-4]],
                              dtype=torch.float32,
                              device=device)

        packed, _ = _quantize_to_fp4_packed(tensor)
        codes = _unpack_codes(packed)

        assert codes.tolist() == [[8, 8, 7, 0]]

    def test_exact_negative_zero_input_becomes_positive_zero(self, device):
        """Documents the one place this diverges from a hardware fp4 cast.

        `-0.0 < 0` is False, so the sign bit is not set and an input of
        exactly -0.0 emits code 0 where a real e2m1 cast emits code 8. Both
        dequantize to zero, so nothing downstream can observe it, but the
        codes are not bit-identical and a test asserting otherwise would be
        wrong. Everything else -- including negatives that merely *round* to
        zero, covered above -- matches the cast exactly.
        """
        tensor = torch.tensor([[-0.0, 6.0]],
                              dtype=torch.float32,
                              device=device)

        packed, _ = _quantize_to_fp4_packed(tensor)

        assert _unpack_codes(packed).tolist() == [[0, 7]]
        assert _reference_codes(tensor).tolist() == [[8, 7]]


class TestScale:
    """The per-token scale and its independence across rows."""

    def test_scale_is_row_absmax_over_six(self, device):
        tensor = torch.tensor([[1.0, -3.0, 0.5, 2.0], [0.25, 0.1, -0.2, 0.05]],
                              dtype=torch.float32,
                              device=device)

        _, scale = _quantize_to_fp4_packed(tensor)

        expected = torch.tensor([[3.0 / 6.0], [0.25 / 6.0]],
                                dtype=torch.float32,
                                device=device)
        torch.testing.assert_close(scale, expected, rtol=0, atol=0)

    def test_rows_are_quantized_independently(self, device):
        """A huge row must not disturb its neighbours' codes or scales."""
        torch.manual_seed(0)
        tensor = torch.randn(6, 64, device=device)
        tensor[3] *= 1e6

        packed, scale = _quantize_to_fp4_packed(tensor)

        for row in range(tensor.shape[0]):
            row_packed, row_scale = _quantize_to_fp4_packed(tensor[row])
            assert torch.equal(packed[row], row_packed)
            assert torch.equal(scale[row], row_scale)

    def test_leading_dims_do_not_change_results(self, device):
        """[B, H, D] must quantize exactly as B*H independent [D] rows."""
        torch.manual_seed(0)
        tensor = torch.randn(4, 3, 32, device=device)

        packed, scale = _quantize_to_fp4_packed(tensor)
        flat_packed, flat_scale = _quantize_to_fp4_packed(tensor.reshape(
            -1, 32))

        assert torch.equal(packed.reshape(-1, 16), flat_packed)
        assert torch.equal(scale.reshape(-1, 1), flat_scale)

    def test_zero_rows_stay_finite(self, device):
        """An all-zero row has scale 0; nothing may become NaN or Inf."""
        tensor = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 3.0, -4.0]],
                              dtype=torch.float32,
                              device=device)

        packed, scale = _quantize_to_fp4_packed(tensor)

        assert scale[0].item() == 0.0
        assert torch.equal(packed[0], torch.zeros_like(packed[0]))
        assert torch.isfinite(scale).all()
        assert torch.isfinite(unpack_uint8_to_fp4(packed)).all()

    def test_subnormal_row_stays_finite(self, device):
        """A row small enough that amax/6 underflows must not blow up."""
        tiny = float(torch.finfo(torch.float32).tiny)
        tensor = torch.full((1, 8), tiny, dtype=torch.float32, device=device)

        packed, scale = _quantize_to_fp4_packed(tensor)

        assert torch.isfinite(scale).all()
        assert torch.isfinite(unpack_uint8_to_fp4(packed) * scale).all()


class TestDequantization:
    """The property that actually matters downstream: reconstruction."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_roundtrip_error_within_one_ladder_step(self, device, dtype):
        """Half the widest e2m1 gap (6 - 4) is 1.0, in units of `scale`."""
        torch.manual_seed(0)
        tensor = torch.randn(128, 128, device=device).to(dtype)

        packed, scale = _quantize_to_fp4_packed(tensor)
        restored = unpack_uint8_to_fp4(packed) * scale

        error = (restored - tensor.to(torch.float32)).abs()
        assert torch.all(error <= scale * 1.000001)

    def test_row_maximum_is_reconstructed(self, device):
        torch.manual_seed(0)
        tensor = torch.randn(64, 128, device=device)

        packed, scale = _quantize_to_fp4_packed(tensor)
        restored = unpack_uint8_to_fp4(packed) * scale

        argmax = tensor.abs().argmax(dim=-1, keepdim=True)
        torch.testing.assert_close(restored.gather(-1, argmax),
                                   tensor.gather(-1, argmax),
                                   rtol=1e-6,
                                   atol=0)

    def test_exactly_representable_rows_roundtrip_bit_exactly(self, device):
        """A row already on the ladder must survive untouched."""
        ladder = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        tensor = torch.tensor([ladder + [-v for v in ladder]],
                              dtype=torch.float32,
                              device=device)

        packed, scale = _quantize_to_fp4_packed(tensor)
        restored = unpack_uint8_to_fp4(packed) * scale

        assert scale.item() == 1.0
        torch.testing.assert_close(restored, tensor, rtol=0, atol=0)


class TestKVWrapper:
    """`quantize_kv_fp4_per_tensor` is the only caller."""

    def test_delegates_to_the_packed_quantizer(self, device):
        torch.manual_seed(0)
        key = torch.randn(8, 4, 64, device=device)
        value = torch.randn(8, 4, 64, device=device)

        k_packed, v_packed, k_scale, v_scale = quantize_kv_fp4_per_tensor(
            key, value)

        expected_k, expected_k_scale = _quantize_to_fp4_packed(key)
        expected_v, expected_v_scale = _quantize_to_fp4_packed(value)
        assert torch.equal(k_packed, expected_k)
        assert torch.equal(v_packed, expected_v)
        assert k_scale.dtype == torch.bfloat16
        assert v_scale.dtype == torch.bfloat16
        assert torch.equal(k_scale, expected_k_scale.to(torch.bfloat16))
        assert torch.equal(v_scale, expected_v_scale.to(torch.bfloat16))

    @pytest.mark.parametrize("hadamard_mode", [None, "k", "kv"])
    def test_hadamard_modes_rotate_the_expected_tensors(
            self, device, hadamard_mode):
        torch.manual_seed(0)
        key = torch.randn(4, 2, 64, device=device)
        value = torch.randn(4, 2, 64, device=device)

        k_packed, v_packed, _, _ = quantize_kv_fp4_per_tensor(
            key, value, hadamard_mode=hadamard_mode)

        expected_key = key if hadamard_mode is None else (
            apply_hadamard_transform(key))
        expected_value = value if hadamard_mode != "kv" else (
            apply_hadamard_transform(value))
        assert torch.equal(k_packed, _quantize_to_fp4_packed(expected_key)[0])
        assert torch.equal(v_packed,
                           _quantize_to_fp4_packed(expected_value)[0])

    def test_rejects_unknown_hadamard_mode(self, device):
        tensor = torch.randn(2, 2, 8, device=device)

        with pytest.raises(ValueError, match="Unsupported hadamard_mode"):
            quantize_kv_fp4_per_tensor(tensor, tensor, hadamard_mode="q")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
