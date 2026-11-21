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
"""Few-row e4m3 linear kernel vs ``bf16(bf16(x @ bf16(w)) * s)``.

Runs the kernel body in the TPU interpreter with tpu7x's TPU info; it does
not gate Mosaic lowering.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import fp8_linear as kernel

_TPU7X = jax.sharding.AbstractMesh(
    (1,),
    ("x",),
    abstract_device=jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    ),
)


def _reference(x, w, s):
    acc = np.asarray(x, np.float32) @ np.asarray(w, np.float32)
    rounded = np.asarray(jnp.asarray(acc, jnp.bfloat16), np.float32)
    return np.asarray(
        jnp.asarray(rounded * np.asarray(s, np.float32), jnp.bfloat16), np.float32
    )


@pytest.mark.parametrize("rows,k,n", [(8, 384, 256), (48, 768, 1024), (24, 1024, 512)])
@pytest.mark.parametrize("scale_dtype", [jnp.bfloat16, jnp.float32])
def test_matches_bf16_matmul(
    rows: int,
    k: int,
    n: int,
    scale_dtype: jnp.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with jax.sharding.use_abstract_mesh(_TPU7X):
        info = kernel.pltpu.get_tpu_info()
    monkeypatch.setattr(kernel.pltpu, "get_tpu_info", lambda: info)
    rng = np.random.default_rng(rows + k + n)
    x = jnp.asarray(rng.standard_normal((rows, k)), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((k, n)) * 0.5, jnp.float8_e4m3fn)
    s = jnp.asarray(np.exp2(rng.integers(-8, -2, (1, n))), scale_dtype)
    got = kernel.fp8_linear(x, w, s, interpret=True)
    assert got.shape == (rows, n) and got.dtype == jnp.bfloat16
    # The f32 accumulation order may differ from the reference's: one bf16 ulp.
    np.testing.assert_allclose(
        np.asarray(got, np.float32), _reference(x, w, s), rtol=8e-3, atol=0
    )


def test_max_rows_on_tpu7x() -> None:
    with jax.sharding.use_abstract_mesh(_TPU7X):
        assert kernel.max_rows(6144, 1536) == 156
        assert kernel.max_rows(6144, 1000) == 0
        assert kernel.max_rows(384, 6144) == 156
        assert kernel.max_rows(320, 6144) == 0
        assert kernel.max_rows(6144, 6144) == 0
