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


import jax.numpy as jnp
import pytest


@pytest.mark.parametrize("lanes,head_dim", [(128, 64), (128, 192), (64, 32)])
def test_strided_loader_rejects_partial_lane_heads(monkeypatch, lanes, head_dim):
    from types import SimpleNamespace

    from vllm_torchtpu.kernels.gdn.v3 import vmem_ldst

    monkeypatch.setattr(
        vmem_ldst.pltpu, "get_tpu_info", lambda: SimpleNamespace(num_lanes=lanes)
    )
    import jax
    from jax.experimental import pallas as pl

    def kernel(input_ref, output_ref):
        output_ref[...] = vmem_ldst.load_strided_heads(
            input_ref.at[...], num_heads=1, head_dim=head_dim
        )

    with pytest.raises(ValueError, match=f"loader lane count {lanes}"):
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1, 1, head_dim), jnp.float32),
            interpret=True,
        )(jnp.zeros((1, 1, lanes), jnp.float32))


@pytest.mark.parametrize("lanes,head_dim", [(128, 128), (128, 256), (64, 64)])
def test_strided_loader_uses_public_lane_geometry(monkeypatch, lanes, head_dim):
    from types import SimpleNamespace

    import jax
    import numpy as np
    from jax.experimental import pallas as pl

    from vllm_torchtpu.kernels.gdn.v3 import vmem_ldst

    monkeypatch.setattr(
        vmem_ldst.pltpu, "get_tpu_info", lambda: SimpleNamespace(num_lanes=lanes)
    )
    values = jnp.arange(3 * 2 * head_dim, dtype=jnp.float32).reshape(3, 1, 2 * head_dim)

    def kernel(input_ref, output_ref):
        output_ref[...] = vmem_ldst.load_strided_heads(
            input_ref.at[...], num_heads=2, head_dim=head_dim
        )

    actual = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 3, head_dim), jnp.float32),
        interpret=True,
    )(values)
    expected = np.asarray(values).reshape(3, 2, head_dim).transpose(1, 0, 2)
    np.testing.assert_array_equal(actual, expected)
