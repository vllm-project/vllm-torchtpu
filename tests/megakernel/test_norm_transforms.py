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
"""Model-free unit tests for `NormModule`, `EmbeddingModule`, and `transforms`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tests.megakernel.megakernel_fakes import FakeGeometry
from vllm_torchtpu.megakernel.decls import HFParamDecl, ParamTransform
from vllm_torchtpu.megakernel.modules.norm import EmbeddingModule, NormModule
from vllm_torchtpu.megakernel.modules.transforms import apply_hf_transforms
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, rmsnorm


def test_norm_module_bind_hf_params_and_reference():
    """`NormModule` resolves canonical/renamed params and matches `rmsnorm`."""
    cfg = {"hidden_size": 128, "rms_norm_eps": 1e-5}
    for prefix, expected in (
        ("input_layernorm", "in_norm"),
        ("layers.0.post_attention_layernorm", "post_norm"),
        ("model.norm", "final_norm"),
        ("custom_ln", "custom_ln"),
    ):
        b = NormModule.bind(prefix, zero_centered=True)
        mod = b.to_module(cfg)
        assert mod.canonical_param_name() == expected
        decls = mod.hf_params()
        assert len(decls) == 1 and decls[0].name == expected
        assert decls[0].shape == (128,) and decls[0].dtype == F32

    b_ren = NormModule.bind(
        "input_layernorm",
        zero_centered=False,
        small_field="in_norm",
        rename={"in_norm": "ln_w"},
    )
    mod_ren = b_ren.to_module(cfg)
    decl = mod_ren.hf_params()[0]
    assert decl.name == "ln_w" and decl.transform.small_field == "in_norm"

    x = jnp.linspace(-1.0, 1.0, 2 * 128, dtype=BF16).reshape(2, 128)
    w = jnp.linspace(0.5, 1.5, 128, dtype=F32)
    out_std, st = mod_ren.reference({"ln_w": w}, x, state="s0", out_dtype=BF16)
    ref_std = rmsnorm(x.astype(F32), w, 1e-5, zero_centered=False).astype(BF16)
    np.testing.assert_allclose(out_std, ref_std, rtol=0, atol=0)
    assert st == "s0"

    mod_zc = NormModule("model.norm", geometry=FakeGeometry(H=128), zero_centered=True)
    out_zc, _ = mod_zc.reference(w, x)
    ref_zc = rmsnorm(x.astype(F32), w, 1e-6, zero_centered=True)
    np.testing.assert_allclose(out_zc, ref_zc, rtol=1e-6, atol=1e-6)

    with pytest.raises(KeyError, match="could not find weight"):
        mod_ren.reference({}, x)
    with pytest.raises(ValueError, match="unknown rename"):
        NormModule.bind("input_layernorm", rename={"bad": "x"})
    with pytest.raises(ValueError, match="unknown transforms"):
        NormModule.bind(
            "input_layernorm", transforms={"bad": ParamTransform(shard_axis=0)}
        )


def test_embedding_module_bind_hf_params_and_reference():
    """`EmbeddingModule` declares sharded vocab embeddings and gathers token rows."""
    cfg = {"vocab_size": 64, "hidden_size": 128}
    b = EmbeddingModule.bind(
        "tok_emb", param_name="tok_w", dtype=F32, rename={"tok_w": "emb_table"}
    )
    mod = b.to_module(cfg)
    decls = mod.hf_params()
    assert len(decls) == 1
    assert decls[0].name == "emb_table"
    assert decls[0].hf_name == "tok_emb.weight"
    assert decls[0].shape == (64, 128) and decls[0].dtype == F32
    assert decls[0].transform.shard_axis == 0

    table = jnp.arange(64 * 128, dtype=F32).reshape(64, 128)
    ids = jnp.array([3, 17, 0], dtype=jnp.int32)
    out, st = mod.reference({"emb_table": table}, ids, state=42, out_dtype=BF16)
    np.testing.assert_array_equal(out, table[ids].astype(BF16))
    assert st == 42

    out_direct, _ = mod.reference(table, ids)
    np.testing.assert_array_equal(out_direct, table[ids])

    with pytest.raises(KeyError, match="could not find weight"):
        mod.reference({}, ids)
    with pytest.raises(ValueError, match="unknown rename"):
        EmbeddingModule.bind("tok_emb", rename={"wrong": "x"})
    with pytest.raises(ValueError, match="unknown transforms"):
        EmbeddingModule.bind(
            "tok_emb", transforms={"wrong": ParamTransform(shard_axis=0)}
        )


def test_apply_hf_transforms_shard_repeat_reduce_pad_and_fuse():
    """`apply_hf_transforms` shards, repeats KV, reduces heads, pads, and fuses."""
    R = 4
    d_q = HFParamDecl(
        name="q",
        hf_name="q.w",
        shape=(8, 16),
        dtype=F32,
        transform=ParamTransform(
            shard_axis=0, fuse_into="qkv", fuse_order=0, fuse_axis=1
        ),
    )
    d_kv = HFParamDecl(
        name="kv",
        hf_name="kv.w",
        shape=(2, 1, 16),
        dtype=F32,
        transform=ParamTransform(
            shard_axis=0, kv_repeat=True, fuse_into="qkv", fuse_order=1, fuse_axis=1
        ),
    )
    d_red = HFParamDecl(
        name="head_s",
        hf_name="hs.w",
        shape=(2, 8, 4),
        dtype=F32,
        transform=ParamTransform(
            shard_axis=1, head_reduce="mean", pad=((0, 2), (0, 8))
        ),
    )
    d_rep = HFParamDecl(
        name="ln",
        hf_name="ln.w",
        shape=(16,),
        dtype=F32,
        transform=ParamTransform(shard_axis=None, replicate_ranks=True),
    )
    d_split = HFParamDecl(
        name="split_c",
        hf_name="sc.w",
        shape=(4, 16),
        dtype=F32,
        transform=ParamTransform(channel_splits=(8, 8), flatten_tail=True),
    )
    hf = {
        "q": jnp.arange(8 * 16, dtype=F32).reshape(8, 16),
        "kv": jnp.arange(2 * 16, dtype=F32).reshape(2, 1, 16),
        "head_s": jnp.ones((2, 8, 4), dtype=F32) * 3.0,
        "ln": jnp.arange(16, dtype=F32),
        "split_c": jnp.arange(4 * 16, dtype=F32).reshape(4, 16),
    }
    out = apply_hf_transforms([d_q, d_kv, d_red, d_rep, d_split], hf, R=R)
    assert set(out) == {"qkv", "head_s", "ln", "split_c"}
    assert out["qkv"].shape == (R, 2 + 1, 16)
    np.testing.assert_array_equal(out["qkv"][0, :2], hf["q"][:2])
    np.testing.assert_array_equal(out["qkv"][0, 2], hf["kv"][0, 0])
    np.testing.assert_array_equal(out["qkv"][1, 2], hf["kv"][0, 0])
    assert out["head_s"].shape == (R, 2, 8)
    np.testing.assert_allclose(out["head_s"][:, :, :4], 3.0)
    np.testing.assert_allclose(out["head_s"][:, :, 4:], 0.0)
    assert out["ln"].shape == (R, 16)
    assert out["split_c"].shape == (R, 4 * 4)


def test_apply_hf_transforms_validation_errors():
    """`apply_hf_transforms` rejects malformed transforms, shapes, and keys."""
    arr = jnp.zeros((4, 8), dtype=F32)
    d_q = HFParamDecl(
        "w", "w", (4, 8), FP8, transform=ParamTransform(shard_axis=0, quantize="fp8")
    )
    with pytest.raises(ValueError, match="does not quantize on the fly"):
        apply_hf_transforms([d_q], {"w": arr}, R=2)

    d_sf = HFParamDecl(
        "s", "s", (4, 8), F32, transform=ParamTransform(scale_for="missing")
    )
    with pytest.raises(ValueError, match="unknown scale_for"):
        apply_hf_transforms([d_sf], {"s": arr}, R=2)

    d_bad_shape = HFParamDecl("w", "w", (4, 8), F32)
    with pytest.raises(ValueError, match="trailing canonical shape"):
        apply_hf_transforms([d_bad_shape], {"w": jnp.zeros((4, 4), dtype=F32)}, R=2)

    d_conflict = HFParamDecl("w", "w", (4, 4), F32)
    with pytest.raises(ValueError, match="conflicting declarations"):
        apply_hf_transforms([d_bad_shape, d_conflict], {"w": arr}, R=2)

    with pytest.raises(ValueError, match="missing canonical parameter key"):
        apply_hf_transforms([d_bad_shape], {}, R=2)
