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
"""VLLM_TPU_MK_IO_LEAN: every lever's operand prep gives the bytes of the code it replaces.

Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest --noconftest <this file>
"""

from __future__ import annotations

import sys
import types

try:
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_io_lean import (  # noqa: E402
    LEVERS,
    io_lean_levers,
    ring_columns,
)


def _bytes(x):
    return np.asarray(x).view(np.uint8)


def test_lever_parsing():
    assert io_lean_levers("") == frozenset()
    assert io_lean_levers("0") == frozenset()
    assert io_lean_levers("1") == frozenset(LEVERS)
    assert io_lean_levers("all") == frozenset(LEVERS)
    assert io_lean_levers("ring") == {"ring"}
    with pytest.raises(AssertionError):
        io_lean_levers("rign")


def test_flag_is_registered_and_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_IO_LEAN", raising=False)
    assert "VLLM_TPU_MK_IO_LEAN" in envs.environment_variables  # compile-cache key
    assert envs.environment_variables["VLLM_TPU_MK_IO_LEAN"]() == ""
    assert io_lean_levers() == frozenset()


# --- ring ---------------------------------------------------------------------------------

def test_ring_columns_are_column_zero_of_each_flat_table():
    rng = np.random.default_rng(0)
    n, W, L = 16, 32, 3
    tabs = []
    for _ in range(L):
        tabs += [jnp.asarray(rng.integers(-1, 2**31 - 1, size=n * W), jnp.int32),
                 jnp.asarray(rng.integers(0, 9, size=n * 4), jnp.int32),
                 jnp.asarray(rng.integers(0, 9, size=n * 5), jnp.int32)]
    got = jax.jit(lambda *t: ring_columns(t, n))(*tabs)
    for o in range(L):
        want = np.asarray(tabs[3 * o]).reshape(n, W)[:, 0]
        np.testing.assert_array_equal(np.asarray(got[3 * o]), want)
        # The readers' own expression gives the same column on the replaced table.
        np.testing.assert_array_equal(np.asarray(got[3 * o]).reshape(n, -1)[:, 0], want)
        for k in (1, 2):  # other tables untouched
            np.testing.assert_array_equal(np.asarray(got[3 * o + k]), np.asarray(tabs[3 * o + k]))
