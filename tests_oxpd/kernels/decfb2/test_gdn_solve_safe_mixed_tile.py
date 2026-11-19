# SPDX-License-Identifier: Apache-2.0
"""PD DEC-FB-2: the GDN mixed tile is rounded to a power of two only where the 16-row triangular solve would drop rows
(PR 2821 ece3d83b5, conservative form). Run: python -m pytest -q tests/kernels/decfb2/test_gdn_solve_safe_mixed_tile.py

Every test goes through the real module: importing it is part of what is tested (ddd967194 put the helper under
fused_conv1d_gdn's @jax.jit decorator, and the import then failed with "Jitted function has invalid argnames").
"""
import inspect

import pytest

from vllm_torchtpu.kernels.gdn.v3 import wrapper


@pytest.mark.parametrize("tile,batch,want", [
    (64, 24, 16), (64, 40, 32), (64, 18, 16), (64, 100, 64),      # the PR's failing buckets, and the default clip
    (64, 48, 48), (64, 16, 16), (64, 12, 12), (64, 7, 7), (64, 1, 1), (32, 64, 32), (64, 64, 64),   # unchanged
])
def test_rounding(tile, batch, want):
    assert wrapper.solve_safe_mixed_tile(tile, batch) == want


def test_result_always_divisible_by_the_solve_block():
    for batch in range(1, 257):
        t = wrapper.solve_safe_mixed_tile(64, batch)
        assert t <= min(64, batch) and t % min(16, t) == 0, (batch, t)


def test_helper_is_plain_and_entry_point_is_jitted():
    # A decorator landing on the helper again makes it a jax-wrapped callable (or breaks the import above).
    assert inspect.isfunction(wrapper.solve_safe_mixed_tile)
    assert isinstance(wrapper.solve_safe_mixed_tile(64, 24), int)
    assert hasattr(wrapper.fused_conv1d_gdn, "lower")  # the jitted entry point
    assert not inspect.isfunction(wrapper.fused_conv1d_gdn)


def test_wrapper_uses_it():
    src = inspect.getsource(wrapper.fused_conv1d_gdn)
    assert "mixed_tile_size = solve_safe_mixed_tile(mixed_tile_size, batch_size)" in src
    assert "mixed_tile_size = min(mixed_tile_size, batch_size)\n" not in src
