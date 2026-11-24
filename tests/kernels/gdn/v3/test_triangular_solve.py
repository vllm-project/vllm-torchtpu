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
"""CPU unit tests for GDN triangular solvers."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.gdn.v3 import triangular_solve


def _rel_l2(ref: jax.Array, got: jax.Array) -> float:
    r = np.asarray(ref, dtype=np.float32)
    y = np.asarray(got, dtype=np.float32)
    return float(np.linalg.norm(r - y) / (np.linalg.norm(r) + 1e-12))


class TriangularSolveTest(parameterized.TestCase):
    @parameterized.product(chunk=(32, 64, 128), block_size=(8, 16))
    def test_solve_triangular_rhs_matches_linalg_solve(
        self, chunk: int, block_size: int
    ):
        rng = np.random.default_rng(19 + chunk + block_size)
        raw = rng.standard_normal((4, chunk, chunk)).astype(np.float32) * 0.1
        strict_lower = jnp.tril(jnp.asarray(raw), k=-1)
        rhs = jnp.asarray(rng.standard_normal((4, chunk, 32)), dtype=jnp.float32)
        eye = jnp.broadcast_to(jnp.eye(chunk, dtype=jnp.float32), (4, chunk, chunk))
        ref = jnp.linalg.solve(eye + strict_lower, rhs)
        got = triangular_solve.solve_triangular_rhs(strict_lower, rhs, block_size)
        self.assertLessEqual(_rel_l2(ref, got), 1e-5)

    @staticmethod
    def _max_rel_for_dtype(fp8_dtype: jnp.dtype | None) -> float:
        if fp8_dtype is None:
            return 5e-3
        if fp8_dtype == jnp.dtype(jnp.float8_e4m3fn):
            return 1.5e-2
        return 2.5e-2

    def _run_block_doubling_case(
        self,
        *,
        chunk: int,
        fp8_dtype,
        target_block_size: int = triangular_solve.TARGET_BLOCK_SIZE,
        num_heads: int = 2,
        seed: int = 31,
        check_zero_lw_fails: bool = False,
    ) -> None:
        rng = np.random.default_rng(seed + chunk)
        heads = [0, 1]
        v_dim = 64
        dt = None if fp8_dtype is None else jnp.dtype(fp8_dtype)
        solver = triangular_solve.BlockDoublingSolver(
            chunk, dt, target_block_size=target_block_size
        )

        d_init: dict[int, jax.Array] = {}
        cs_map: dict[int, dict[int, jax.Array]] = {}
        lw_map: dict[int, jax.Array | None] = {}
        rhs_bf16_map: dict[int, jax.Array] = {}
        refs: dict[int, jax.Array] = {}

        eye = jnp.eye(chunk, dtype=jnp.float32)
        row = np.arange(chunk)[:, None]
        col = np.arange(chunk)[None, :]
        # Slow enough decay that cross-block coupling `lw` contributes ~9% to
        # the solution on chunk >= 128, so dropping it fails the tolerance.
        decay = np.exp(-0.03 * np.maximum(0, row - col)).astype(np.float32)
        for v in heads:
            raw = rng.standard_normal((chunk, chunk)).astype(np.float32) * 0.06
            strict_lower = jnp.tril(jnp.asarray(raw * decay), k=-1)
            rhs = jnp.asarray(rng.standard_normal((chunk, v_dim)), dtype=jnp.float32)
            refs[v] = jnp.linalg.solve(eye + strict_lower, rhs)

            coef = solver.select_block_diag(strict_lower)
            d_init[v] = solver.invert_base_blocks(coef)
            l_s = (strict_lower * jnp.float32(triangular_solve.S_L)).astype(
                jnp.bfloat16
            )
            cs_map[v], lw_map[v] = solver.slice_coupling(l_s)
            rhs_bf16_map[v] = rhs.astype(jnp.bfloat16)

        d_inv_map = solver.double_inverses(d_init, cs_map, heads)
        max_rel = self._max_rel_for_dtype(dt)

        if check_zero_lw_fails:
            lw_zero = {
                v: None if lw is None else jnp.zeros_like(lw)
                for v, lw in lw_map.items()
            }
            solved_zero = solver.solve_blocks(
                dict(d_inv_map),
                lw_zero,
                dict(rhs_bf16_map),
                heads,
                num_heads=num_heads,
                v_dim=v_dim,
            )
            for v in heads:
                u_zero = sum(
                    limb.astype(jnp.float32) for limb in solved_zero.pop_u_limbs(v)
                ) / jnp.float32(triangular_solve.S_D)
                self.assertGreater(_rel_l2(refs[v], u_zero), max_rel)

        solved = solver.solve_blocks(
            d_inv_map,
            lw_map,
            rhs_bf16_map,
            heads,
            num_heads=num_heads,
            v_dim=v_dim,
        )
        for v in heads:
            u_limbs = solved.pop_u_limbs(v)
            u_scaled = sum(limb.astype(jnp.float32) for limb in u_limbs)
            u_got = u_scaled / jnp.float32(triangular_solve.S_D)
            self.assertLessEqual(_rel_l2(refs[v], u_got), max_rel)

    @parameterized.product(
        chunk=(32, 64, 128, 256),
        fp8_dtype=(None, jnp.float8_e4m3fn, jnp.float8_e5m2),
    )
    def test_block_doubling_matches_linalg_solve(self, chunk: int, fp8_dtype):
        self._run_block_doubling_case(
            chunk=chunk,
            fp8_dtype=fp8_dtype,
            check_zero_lw_fails=(chunk == 128),
        )

    @parameterized.parameters(
        (128, 64, 32, None),
        (128, 64, 32, jnp.float8_e4m3fn),
        (64, 32, 2, None),
        (64, 32, 2, jnp.float8_e4m3fn),
        (128, 128, 2, None),
        (128, 128, 2, jnp.float8_e4m3fn),
    )
    def test_block_doubling_non_default_block_and_padded_paths(
        self, chunk: int, target_block_size: int, num_heads: int, fp8_dtype
    ):
        self._run_block_doubling_case(
            chunk=chunk,
            fp8_dtype=fp8_dtype,
            target_block_size=target_block_size,
            num_heads=num_heads,
            seed=73,
            check_zero_lw_fails=(chunk == 2 * target_block_size),
        )

    @parameterized.parameters(
        (16, 64, False),
        (32, 64, True),
        (48, 64, False),
        (64, 64, True),
        (96, 64, False),
        (128, 64, True),
        (160, 64, False),
        (192, 64, False),
        (256, 64, True),
        (128, 8, False),
        (128, 48, False),
    )
    def test_supports_block_doubling_chunk_matches_solver_init(
        self, chunk: int, target_block_size: int, want: bool
    ):
        self.assertEqual(
            triangular_solve.supports_block_doubling_chunk(chunk, target_block_size),
            want,
        )
        if not want:
            with self.assertRaises(ValueError):
                triangular_solve.BlockDoublingSolver(
                    chunk, None, target_block_size=target_block_size
                )

    def test_block_doubling_invalid_fp8_dtype_raises(self):
        with self.assertRaisesRegex(ValueError, "fp8_dtype"):
            triangular_solve.BlockDoublingSolver(64, jnp.dtype(jnp.float32))


if __name__ == "__main__":
    absltest.main()
