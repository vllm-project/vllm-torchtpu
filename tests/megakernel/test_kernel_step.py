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
"""Model-free unit tests for `kernel.build`, `kernel.make_step`, and helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tests.megakernel.megakernel_fakes import TINY_TILES, synth_hybrid_spec
from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import DebugOptions, ExecSpec
from vllm_torchtpu.megakernel.ops import BF16, F32
from vllm_torchtpu.megakernel.reference import spec_reference_step
from vllm_torchtpu.megakernel.weights.spec_inputs import (
    pack_spec_inputs,
    random_spec_inputs,
)


def test_kernel_helpers_fold_unfold_and_unpack_local_weight():
    """`fold`, `unfold`, and `unpack_local_weight` handle rank and tile shapes."""
    x = jnp.arange(2 * 3 * 4, dtype=F32).reshape(2, 3, 4)
    folded = kernel.fold(x)
    assert folded.shape == (6, 4)
    np.testing.assert_array_equal(kernel.unfold(folded, 2), x)

    assert not kernel.is_tupled_weight(((4, 8), F32))
    assert kernel.target_weight_shape(((4, 8), F32)) == (4, 8)
    tup_entry = (jax.ShapeDtypeStruct((4, 8), F32), jax.ShapeDtypeStruct((4, 8), F32))
    assert kernel.is_tupled_weight(tup_entry)
    assert kernel.target_weight_shape(tup_entry) == (4, 8)

    w = jnp.arange(2 * 3 * 4 * 8, dtype=BF16).reshape(2, 3, 4, 8)
    np.testing.assert_array_equal(
        kernel.unpack_local_weight(w, (2, 3, 4, 8), name="w"), w
    )
    np.testing.assert_array_equal(
        kernel.unpack_local_weight(w[None], (2, 3, 4, 8), name="w"), w
    )
    w_merged = w.reshape(2, 12, 8)
    np.testing.assert_array_equal(
        kernel.unpack_local_weight(
            w_merged, (2, 3, 4, 8), name="w", allow_merged_tiles=True
        ),
        w,
    )
    np.testing.assert_array_equal(
        kernel.unpack_local_weight(
            w_merged[None], (2, 3, 4, 8), name="w", allow_merged_tiles=True
        ),
        w,
    )
    with pytest.raises(MegakernelConfigError, match="local shard has shape"):
        kernel.unpack_local_weight(w_merged, (2, 3, 4, 8), name="w")


@pytest.mark.parametrize(("T", "B"), [(1, 4), (4, 4)], ids=["decode_t1", "verify_t4"])
def test_kernel_make_step_matches_spec_reference_step(T: int, B: int):
    """`kernel.make_step` matches `spec_reference_step` on `synth_hybrid_spec`."""
    R = 2
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    tiles = TINY_TILES.with_modules({"gqa": {"kv_blk": 128, "nkv": 2}})
    ex = ExecSpec(B=B, T=T, R=R, ctx=256, stack_e=True, tiles=tiles, interpret=True)
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))

    pos, hid, w_log, st_log = random_spec_inputs(jax.random.PRNGKey(10 + T), spec, ex)
    pos_mk, hid_mk, w_mk, st_mk = pack_spec_inputs(
        pos, hid, w_log, st_log, spec, ex, mesh=mesh
    )

    step, info = kernel.make_step(mesh, spec, ex)
    assert info["exec"].B == B and info["exec"].T == T
    assert step.lower(pos_mk, hid_mk, w_mk, st_mk) is not None

    out_hid_mk, out_st_mk = step(pos_mk, hid_mk, w_mk, st_mk)
    jax.block_until_ready(out_hid_mk)

    out_hid_r = out_hid_mk.reshape(R, ex.M, spec.config.H)[:, :B]
    np.testing.assert_array_equal(out_hid_r[0], out_hid_r[1])

    ref_hid, ref_st = spec_reference_step(
        spec, spec.config, ex, pos, hid, w_log, st_log
    )
    np.testing.assert_allclose(
        out_hid_r[0].astype(F32), ref_hid.astype(F32), rtol=2e-2, atol=2e-2
    )

    kc_mk = kernel.unfold(out_st_mk["kc"], R)
    vc_mk = kernel.unfold(out_st_mk["vc"], R)
    np.testing.assert_allclose(
        kc_mk.astype(F32), ref_st["kc"].astype(F32), rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        vc_mk.astype(F32), ref_st["vc"].astype(F32), rtol=2e-2, atol=2e-2
    )
    rec_key = "rec_ckpt" if T > 1 else "rec"
    rec_mk = kernel.unfold(out_st_mk[rec_key], R)
    np.testing.assert_allclose(
        rec_mk.astype(F32), ref_st[rec_key].astype(F32), rtol=2e-2, atol=2e-2
    )


def test_kernel_build_ablations_and_scopes():
    """`kernel.build` supports `skip` ablations and `scopes` levels."""
    R = 2
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    tiles = TINY_TILES.with_modules({"gqa": {"kv_blk": 128, "nkv": 2}})
    ex = ExecSpec(
        B=4,
        T=1,
        R=R,
        ctx=256,
        stack_e=True,
        tiles=tiles,
        interpret=True,
        debug=DebugOptions(skip=frozenset({"experts", "kv"}), scopes=2),
    )
    call, info = kernel.build(spec, ex)
    assert callable(call)
    assert info["exec"].skip == frozenset({"experts", "kv"})
