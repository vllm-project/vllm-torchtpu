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
"""Tests for the hierrs_tc reduce-scatter kernels."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.collectives.hierrs_tc import config as hrs_config
from vllm_torchtpu.kernels.collectives.hierrs_tc import dispatch, kernel_stream
from vllm_torchtpu.kernels.collectives.hierrs_tc import wrapper as hrs

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec

# EP rank at each mesh index of a tpu7x host's ascending-device-id EP mesh.
_TPU7X_RANK_AT_MESH_INDEX = (0, 1, 6, 7, 2, 3, 4, 5)


def _make_mesh(num_devices: int, axis_name: str) -> jax.sharding.Mesh:
    # Not jax.make_mesh: given devices, it reorders them into a ring and
    # chip ^ 2 becomes the diagonal chip.
    devices = sorted(jax.devices(), key=lambda d: d.id)[:num_devices]
    return jax.sharding.Mesh(np.asarray(devices), (axis_name,))


def _hierarchical_reduce_scatter(
    x, *, mesh, in_specs, num_micro_batches=None, **kwargs
):
    return jax.shard_map(
        lambda local_x: hrs.hierarchical_reduce_scatter_local(
            local_x,
            num_devices=mesh.devices.size,
            num_micro_batches=num_micro_batches,
            **kwargs,
        ),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=in_specs,
        check_vma=False,
    )(x)


def _reduce_scatter_stream(x, *, mesh, order=None):
    axis = mesh.axis_names[0]
    spec = P(axis, None)
    return jax.shard_map(
        lambda local_x: kernel_stream.reduce_scatter_stream_local(
            local_x, axis_name=axis, order=order
        ),
        mesh=mesh,
        in_specs=spec,
        out_specs=spec,
        check_vma=False,
    )(x)


# The kernel and psum sum in different orders, so near-zero outputs defeat
# an elementwise rtol; measured agreement is ~48 dB.
_MIN_BF16_SNR_DB = 40.0

_BF16_TRUTH_SLACK = 1.5


def _snr_db(signal: jax.Array, noise: jax.Array) -> float:
    signal_power = float(jnp.sum(signal.astype(jnp.float32) ** 2))
    noise_power = float(jnp.sum(noise.astype(jnp.float32) ** 2))
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * jnp.log10(signal_power / noise_power).item()


def _reference_reduce_scatter(
    x: jax.Array, mesh: jax.sharding.Mesh, in_specs: P
) -> jax.Array:
    axis_name = mesh.axis_names[0]
    num_devices = mesh.devices.size

    def inner(local_x):
        reduced = jax.lax.psum(local_x, axis_name=axis_name)
        idx = jax.lax.axis_index(axis_name)
        chunk = local_x.shape[0] // num_devices
        return jax.lax.dynamic_slice_in_dim(reduced, idx * chunk, chunk, axis=0)

    return jax.shard_map(
        inner,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=in_specs,
        check_vma=False,
    )(x)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
@pytest.mark.tpu_test
@pytest.mark.multichip
class HierarchicalReduceScatterTest(jtu.JaxTestCase):
    def _requires_devices(self, n: int):
        if jax.device_count() < n:
            self.skipTest(f"Need {n} devices, got {jax.device_count()}")

    @parameterized.product(
        num_micro_batches=[1, 2, 4],
        hidden=[4096, 5120],
    )
    def test_correctness_bf16(self, num_micro_batches, hidden):
        self._requires_devices(8)
        axis_name = "x"
        num_devices = jax.device_count()
        mesh = _make_mesh(num_devices, axis_name)

        seq_len = 1024
        in_specs = P(axis_name, None)

        for seed in range(3):
            x = jax.random.normal(
                jax.random.key(seed), (seq_len, hidden), dtype=jnp.bfloat16
            )
            x_sharded = jax.device_put(x, jax.sharding.NamedSharding(mesh, in_specs))

            ref = _reference_reduce_scatter(x_sharded, mesh, in_specs)
            out = _hierarchical_reduce_scatter(
                x_sharded,
                mesh=mesh,
                in_specs=in_specs,
                num_micro_batches=num_micro_batches,
            )

            ref_f32 = jnp.asarray(ref, jnp.float32)
            out_f32 = jnp.asarray(out, jnp.float32)

            snr = _snr_db(ref_f32, out_f32 - ref_f32)
            self.assertGreater(
                snr,
                _MIN_BF16_SNR_DB,
                msg=f"bf16 hidden={hidden} mb={num_micro_batches} seed={seed}: "
                f"SNR {snr:.1f} dB "
                f"< {_MIN_BF16_SNR_DB} dB vs psum reference",
            )

            truth = (
                jnp.asarray(x, jnp.float32)
                .reshape(num_devices, seq_len // num_devices, hidden)
                .sum(axis=0)
            )
            scale = float(jnp.max(jnp.abs(truth))) or 1.0
            err_kernel = float(jnp.max(jnp.abs(out_f32 - truth))) / scale
            err_ref = float(jnp.max(jnp.abs(ref_f32 - truth))) / scale
            self.assertLess(
                err_kernel,
                max(err_ref * _BF16_TRUTH_SLACK, 1e-3),
                msg=f"bf16 hidden={hidden} mb={num_micro_batches} seed={seed}: "
                f"kernel error "
                f"{err_kernel:.4g} vs f32 truth exceeds reference "
                f"{err_ref:.4g} by more than {_BF16_TRUTH_SLACK}x",
            )

            again = _hierarchical_reduce_scatter(
                x_sharded,
                mesh=mesh,
                in_specs=in_specs,
                num_micro_batches=num_micro_batches,
            )
            self.assertTrue(
                bool(jnp.array_equal(jnp.asarray(again, jnp.float32), out_f32)),
                msg=f"bf16 hidden={hidden} mb={num_micro_batches} seed={seed}: "
                f"two runs on "
                f"identical input disagree -- race, not precision",
            )

    @parameterized.product(
        num_micro_batches=[2, 4],
    )
    def test_no_race_multi_micro_batch(self, num_micro_batches):
        # The write-after-read races showed at 512 local rows, not at 128.
        self._requires_devices(8)
        axis_name = "x"
        num_devices = jax.device_count()
        mesh = _make_mesh(num_devices, axis_name)

        seq_len, hidden = 4096, 4096
        in_specs = P(axis_name, None)
        runs = 12

        x = jax.random.normal(jax.random.key(0), (seq_len, hidden), dtype=jnp.bfloat16)
        x_sharded = jax.device_put(x, jax.sharding.NamedSharding(mesh, in_specs))
        truth = (
            jnp.asarray(x, jnp.float32)
            .reshape(num_devices, seq_len // num_devices, hidden)
            .sum(axis=0)
        )
        scale = float(jnp.max(jnp.abs(truth))) or 1.0
        tol = 0.05

        first, worst = None, 0.0
        for i in range(runs):
            out = jnp.asarray(
                _hierarchical_reduce_scatter(
                    x_sharded,
                    mesh=mesh,
                    in_specs=in_specs,
                    num_micro_batches=num_micro_batches,
                ),
                jnp.float32,
            )
            worst = max(worst, float(jnp.max(jnp.abs(out - truth))) / scale)
            if first is None:
                first = out
            else:
                self.assertTrue(
                    bool(jnp.array_equal(first, out)),
                    msg=f"mb={num_micro_batches}: run {i} "
                    f"differs from run 0 on identical input -- race",
                )

        self.assertLess(
            worst,
            tol,
            msg=f"mb={num_micro_batches}: max relative error "
            f"{worst:.4g} over {runs} runs exceeds {tol}",
        )

    @parameterized.parameters(
        # Two micro-batches is where step 1 used to overwrite a step-0 landing
        # slot under skew; four is the control.
        ("bf16", 256, 2),
        ("bf16", 512, 2),
        ("bf16", 1024, 4),
    )
    def test_no_race_under_inter_chip_skew(self, wire, local_rows, num_micro_batches):
        self._requires_devices(8)
        axis_name, hidden, n = "x", 4096, 8
        mesh = _make_mesh(n, axis_name)
        in_specs = P(axis_name, None)
        sharding = jax.sharding.NamedSharding(mesh, in_specs)
        tol = 0.05
        reps = 4

        # The late chip is held at kernel entry by serial local DMAs, so every
        # other chip starts phase 2 first. Chip 1, then chip 2: the late chip is
        # a step-0 neighbour along each hypercube dim in turn. The copy counts
        # span tens of microseconds to milliseconds: larger shapes need more.
        for late_chip in (1, 2):
            for num_copies in (40, 400, 4000):
                for rep in range(reps):
                    # A fresh input per call: a stale buffer looks right on a
                    # repeat.
                    key = jax.random.key(local_rows + num_micro_batches)
                    for salt in (late_chip, num_copies, rep):
                        key = jax.random.fold_in(key, salt)
                    x = jax.random.normal(
                        key, (n * local_rows, hidden), dtype=jnp.bfloat16
                    )
                    truth = (
                        jnp.asarray(x, jnp.float32)
                        .reshape(n, local_rows, hidden)
                        .sum(axis=0)
                    )
                    out = jnp.asarray(
                        _hierarchical_reduce_scatter(
                            jax.device_put(x, sharding),
                            mesh=mesh,
                            in_specs=in_specs,
                            num_micro_batches=num_micro_batches,
                            _debug_entry_delay=(late_chip, num_copies),
                        ),
                        jnp.float32,
                    )
                    where = (
                        f"{wire} rows={local_rows} mb={num_micro_batches} "
                        f"late_chip={late_chip} copies={num_copies} rep={rep}"
                    )
                    scale = float(jnp.max(jnp.abs(truth))) or 1.0
                    err = float(jnp.max(jnp.abs(out - truth))) / scale
                    self.assertLess(
                        err,
                        tol,
                        msg=f"{where}: relative error {err:.3g} -- skew race",
                    )

    @parameterized.product(
        rows_per_device=[1024, 4096, 16384],
        order=[tuple(range(8)), _TPU7X_RANK_AT_MESH_INDEX],
    )
    def test_stream_matches_reference(self, rows_per_device, order):
        """Mesh index i gets the sum of row block order[i]."""
        self._requires_devices(8)
        axis_name, hidden, n = "x", 4096, 8
        mesh = _make_mesh(n, axis_name)
        in_specs = P(axis_name, None)
        block = rows_per_device // n
        # A fresh input per case: a stale-buffer bug looks right on a repeat.
        x = jax.random.normal(
            jax.random.key(rows_per_device + order[2]),
            (n * rows_per_device, hidden),
            dtype=jnp.bfloat16,
        )
        x_ordered = x.reshape(n, n, block, hidden)[:, list(order)].reshape(x.shape)
        sharding = jax.sharding.NamedSharding(mesh, in_specs)
        ref = _reference_reduce_scatter(
            jax.device_put(x_ordered, sharding), mesh, in_specs
        )
        out = _reduce_scatter_stream(
            jax.device_put(x, sharding), mesh=mesh, order=order
        )
        snr = _snr_db(ref, out.astype(jnp.float32) - ref.astype(jnp.float32))
        self.assertGreater(snr, _MIN_BF16_SNR_DB, msg=f"SNR {snr:.1f} dB")

    @parameterized.product(
        num_micro_batches=[1, 2, 4],
        order=[(7, 6, 5, 4, 3, 2, 1, 0), _TPU7X_RANK_AT_MESH_INDEX],
    )
    def test_micro_batched_order_matches_reference(self, num_micro_batches, order):
        """Mesh index i gets the sum of row block order[i], with no input copy."""
        self._requires_devices(8)
        axis_name, rows, hidden, n = "x", 1024, 4096, 8
        mesh = _make_mesh(n, axis_name)
        in_specs = P(axis_name, None)
        block = rows // n
        sharding = jax.sharding.NamedSharding(mesh, in_specs)
        x = jax.random.normal(
            jax.random.key(num_micro_batches * 131 + order[1]),
            (n * rows, hidden),
            dtype=jnp.bfloat16,
        )
        x_ordered = x.reshape(n, n, block, hidden)[:, list(order)].reshape(x.shape)
        ref = _reference_reduce_scatter(
            jax.device_put(x_ordered, sharding), mesh, in_specs
        )
        out = _hierarchical_reduce_scatter(
            jax.device_put(x, sharding),
            mesh=mesh,
            in_specs=in_specs,
            num_micro_batches=num_micro_batches,
            order=order,
        )
        snr = _snr_db(ref, out.astype(jnp.float32) - ref.astype(jnp.float32))
        self.assertGreater(snr, _MIN_BF16_SNR_DB, msg=f"SNR {snr:.1f} dB")

    @parameterized.product(
        # Rows per rank x hidden at the default threshold: the micro-batched
        # kernel padded (64, 160) and not (1024, 16000), the row-streaming one
        # (2048, 16384), and a hidden too wide for its VMEM (2048 x 7168).
        shape=[
            (64, 4096),
            (160, 4096),
            (1024, 4096),
            (16000, 4096),
            (2048, 4096),
            (16384, 4096),
            (2048, 7168),
        ],
        order=[tuple(range(8)), _TPU7X_RANK_AT_MESH_INDEX],
    )
    def test_dispatch_matches_rank_ordered_reduce_scatter(self, shape, order):
        """Mesh index i holds rank order[i]'s input and gets its block's sum."""
        self._requires_devices(8)
        rows, hidden = shape
        n, axis_name = 8, "x"
        mesh = _make_mesh(n, axis_name)
        sharding = jax.sharding.NamedSharding(mesh, P(axis_name, None))
        fn = jax.jit(dispatch.make_reduce_scatter_fn(mesh, order, axis_name=axis_name))
        block = rows // n
        # Two calls with new data, as serving makes one per MoE layer.
        for call in range(2):
            rng = np.random.default_rng([rows, hidden, *order, call])
            per_rank = [
                rng.standard_normal((rows, hidden), dtype=np.float32) for _ in range(n)
            ]
            per_rank = [
                np.asarray(jnp.asarray(r, jnp.bfloat16), np.float32) for r in per_rank
            ]
            x = np.concatenate([per_rank[r] for r in order], axis=0)
            out = fn(jax.device_put(jnp.asarray(x, jnp.bfloat16), sharding))
            summed = sum(per_rank)
            want = np.concatenate(
                [summed[r * block : (r + 1) * block] for r in order], axis=0
            )
            snr = _snr_db(want, np.asarray(out, np.float32) - want)
            self.assertGreater(
                snr, _MIN_BF16_SNR_DB, msg=f"call {call}: SNR {snr:.1f} dB"
            )

    def test_stream_supported_shapes(self):
        self.assertTrue(kernel_stream.supports_stream(8 * 128, 4096, 2))
        self.assertTrue(kernel_stream.supports_stream(8 * 1024, 4096, 2))
        self.assertFalse(kernel_stream.supports_stream(8 * 64, 4096, 2))
        self.assertFalse(kernel_stream.supports_stream(8 * 192, 4096, 2))
        self.assertFalse(kernel_stream.supports_stream(8 * 128 + 8, 4096, 2))
        # Its scratch is 8448 bytes per bf16 hidden column: 53.6 MiB at 6656,
        # which compiles, and 57.75 MiB at 7168, which does not.
        self.assertTrue(kernel_stream.supports_stream(8 * 1024, 6656, 2))
        self.assertFalse(kernel_stream.supports_stream(8 * 1024, 7168, 2))


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class HierarchicalReduceScatterPlanningTest(jtu.JaxTestCase):
    PRODUCTION_ROWS = (32, 64, 128, 256, 512, 1024, 2048)
    HIDDEN = 4096
    BF16_ITEMSIZE = 2

    def test_micro_batch_floor(self):
        self.assertEqual(
            hrs_config._MIN_SAFE_MICRO_BATCHES,
            2,
            msg="the mb>=2 correctness floor is not in effect",
        )
        for rows in self.PRODUCTION_ROWS:
            mb = hrs_config.pick_num_micro_batches(
                rows, self.HIDDEN, self.BF16_ITEMSIZE, num_devices=8
            )
            self.assertGreaterEqual(
                mb,
                2,
                msg=f"local_seq_len={rows} picked mb={mb}",
            )
            self.assertLessEqual(mb, hrs_config._MAX_MICRO_BATCHES)

    def test_micro_batches_split_hidden_evenly(self):
        for hidden in (2048, 4096, 5120, 7168):
            for rows in self.PRODUCTION_ROWS + (8192, 16384):
                mb = hrs_config.pick_num_micro_batches(
                    rows, hidden, self.BF16_ITEMSIZE, num_devices=8
                )
                self.assertTrue(
                    hrs_config.splits_hidden_evenly(hidden, mb, 8),
                    msg=f"hidden={hidden} rows={rows} picked mb={mb}",
                )

    def test_default_kernel_selection(self):
        for rows, hidden, stream in (
            (64, 4096, False),
            (1024, 4096, False),
            (16000, 4096, False),
            (2048, 4096, True),
            (16384, 4096, True),
            (2048, 7168, False),
        ):
            self.assertEqual(
                dispatch.uses_stream(rows, hidden, self.BF16_ITEMSIZE),
                stream,
                msg=f"{rows} x {hidden}",
            )

    def test_unsplittable_hidden_is_unsupported(self):
        self.assertTrue(hrs.is_supported_by_hierrs_tc(8, 1024, 4096))
        self.assertFalse(hrs.is_supported_by_hierrs_tc(8, 1024, 2816))

    def test_work_set_is_three_bytes_per_element(self):
        for rows in self.PRODUCTION_ROWS:
            elems = rows * self.HIDDEN
            self.assertEqual(
                hrs._work_set_bytes(rows, self.HIDDEN, self.BF16_ITEMSIZE), 3 * elems
            )

    def test_work_scratch_falls_back_when_it_cannot_fit(self):
        fake_info = mock.Mock(vmem_capacity_bytes=64 * 2**20)
        with mock.patch.object(hrs.pltpu, "get_tpu_info", return_value=fake_info):
            for rows, mb, expected in (
                (256, 1, True),
                (512, 1, True),
                (1024, 1, True),
                (2048, 2, True),
                (4096, 4, False),
            ):
                enabled, frac = hrs._plan_work_scratch(
                    rows, self.HIDDEN, self.BF16_ITEMSIZE, mb, 0.95
                )
                self.assertEqual(
                    enabled,
                    expected,
                    msg=f"local_seq_len={rows} mb={mb}: VMEM scratch "
                    f"enabled={enabled}, expected {expected}",
                )
                if enabled:
                    self.assertGreater(frac, 0.0)
                    self.assertLess(frac, 1.0)


if __name__ == "__main__":
    absltest.main()
