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
"""Unit and CPU interpret parity tests for Module API declarations and hooks."""

from __future__ import annotations

import copy
import dataclasses
import pickle
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.megakernel as mk
from vllm_torchtpu.megakernel import (
    POOL_STATE_NAME,
    REQUIRED,
    CacheExec,
    Embedding,
    ExecSpec,
    FrozenDict,
    GDNModule,
    GQAModule,
    KVCacheDecl,
    LayerType,
    MegakernelConfigError,
    ModelSpec,
    MoEModule,
    Norm,
    ParamTransform,
    PoolLayout,
    RecurrentCacheDecl,
    Tiles,
    derive_pool_layout,
    kernel,
    planner,
)
from vllm_torchtpu.megakernel.modules.gqa import GQATunables
from vllm_torchtpu.megakernel.modules.transforms import apply_hf_transforms
from vllm_torchtpu.megakernel.ops import BF16, F32, I32, rmsnorm, rnd, rup
from vllm_torchtpu.megakernel.pool.config import resolved_max_pages_per_seq
from vllm_torchtpu.megakernel.spec import combined_hf_param_shapes
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35
from vllm_torchtpu.models.qwen3_5.megakernel.pool import (
    qwen35_layer_cache_decls,
    qwen35_pool_layout,
)


def _make_mock_hf_config(g: qwen35.Qwen35Geometry) -> SimpleNamespace:
    text_cfg = SimpleNamespace(
        hidden_size=g.H,
        num_attention_heads=g.NH,
        num_key_value_heads=g.NKV,
        head_dim=g.HD,
        partial_rotary_factor=g.ROT / g.HD,
        rope_theta=g.theta,
        rms_norm_eps=g.eps,
        linear_num_key_heads=g.KH,
        linear_num_value_heads=g.VH,
        linear_key_head_dim=g.KD,
        linear_value_head_dim=g.VD,
        linear_conv_kernel_dim=4,
        num_experts=g.E,
        num_experts_per_tok=g.TOPK,
        moe_intermediate_size=g.I,
        shared_expert_intermediate_size=g.IS,
        vocab_size=248320,
    )
    return SimpleNamespace(text_config=text_cfg)


def _cpu_mesh(R: int) -> jax.sharding.Mesh:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Requires >= {R} CPU devices, found {len(devs)}")
    return jax.sharding.Mesh(np.array(devs[:R]), ("tp",))


def test_derive_qwen35_pool_layout_from_module_cache_decls():
    """`derive_pool_layout` derives pool grouping and page counts from layer decls."""
    for g in (
        qwen35.FULL,
        qwen35.TINY,
        dataclasses.replace(qwen35.TINY, att_every=1),
    ):
        gqa = GQAModule(geometry=g)
        gdn = GDNModule(geometry=g)
        moe = MoEModule(geometry=g)
        norm = Norm("input_layernorm", geometry=g)
        emb = Embedding("model.embed_tokens", geometry=g)

        kv_decl = gqa.cache_decl(g)
        rec_decl = gdn.cache_decl(g)

        assert isinstance(kv_decl, KVCacheDecl)
        assert isinstance(rec_decl, RecurrentCacheDecl)
        assert rec_decl.rec_pages_per_slot is None
        assert moe.cache_decl(g) is None
        assert norm.cache_decl(g) is None
        assert emb.cache_decl(g) is None

        layer_decls = qwen35_layer_cache_decls(g)
        if g.att_every == 1:
            derived = derive_pool_layout(layer_decls)
            via_classmethod = PoolLayout.from_cache_decls(layer_decls)
        else:
            with pytest.raises(MegakernelConfigError, match="requires explicit R > 0"):
                derive_pool_layout(layer_decls)
            derived = derive_pool_layout(layer_decls, R=8)
            via_classmethod = PoolLayout.from_cache_decls(layer_decls, R=8)
        expected = qwen35_pool_layout(g)

        assert derived == expected
        assert via_classmethod == expected
        assert derived.num_pools == g.NATT
        assert derived.slots_per_pool == g.att_every - 1
        assert derived.rec_pages_per_slot == 4

    # Verify page count derivation scales when per-rank recurrent state exceeds 4 pages.
    wide_rec = RecurrentCacheDecl(kh=32, vh=128, KD=128, VD=128)
    wide_lay = derive_pool_layout([wide_rec, KVCacheDecl(hd=256)], R=8)
    assert wide_lay.rec_pages_per_slot == 8


def test_qwen35_hf_params_transforms_and_vllm_aliases():
    """`hf_params` transforms match `split_hf_weights` bitwise and cover aliases."""
    hf_cfg_tp = _make_mock_hf_config(qwen35.TINY)
    in_norm_mod = Norm.bind("input_layernorm", zero_centered=True).resolve(hf_cfg_tp)
    post_norm_mod = Norm.bind("post_attention_layernorm", zero_centered=True).resolve(
        hf_cfg_tp
    )
    final_norm_mod = Norm.bind("model.norm", zero_centered=True).resolve(hf_cfg_tp)
    gqa_mod = GQAModule.bind("self_attn").resolve(hf_cfg_tp)
    gdn_mod = GDNModule.bind("linear_attn").resolve(hf_cfg_tp)
    moe_mod_tp = MoEModule.bind("mlp", moe_tp=True).resolve(hf_cfg_tp)
    moe_mod_ep = MoEModule.bind("mlp", moe_tp=False).resolve(hf_cfg_tp)

    # Check real HF checkpoint names, shapes, hf_shapes, and derived pads.
    gqa_by_name = {d.name: d for d in gqa_mod.hf_params("self_attn")}
    assert gqa_by_name["qn"].shape == (qwen35.TINY.NH, qwen35.TINY.HD)
    assert gqa_by_name["qn"].hf_shape == (qwen35.TINY.HD,)
    assert gqa_by_name["kn"].shape == (qwen35.TINY.NKV, qwen35.TINY.HD)
    assert gqa_by_name["kn"].hf_shape == (qwen35.TINY.HD,)
    assert gqa_by_name["k_proj"].transform.transpose == (0, 2, 1)

    gdn_by_name = {d.name: d for d in gdn_mod.hf_params("linear_attn")}
    assert gdn_by_name["lin_q"].hf_name == "linear_attn.in_proj_qkv.weight"
    assert gdn_by_name["lin_k"].hf_name == "linear_attn.in_proj_qkv.weight"
    assert gdn_by_name["lin_v"].hf_name == "linear_attn.in_proj_qkv.weight"
    assert gdn_by_name["lin_z"].hf_name == "linear_attn.in_proj_z.weight"
    assert gdn_by_name["lin_z"].transform.split_from is None
    assert gdn_by_name["lin_b"].hf_name == "linear_attn.in_proj_b.weight"
    assert gdn_by_name["lin_a"].hf_name == "linear_attn.in_proj_a.weight"
    assert gdn_by_name["alog"].transform.pad is None
    assert gdn_by_name["dtb"].transform.pad is None
    assert gdn_by_name["conv"].transform.transpose is None

    # Every canonical parameter in every module is covered by at least one VLLMAlias.
    for mod, sub in (
        (in_norm_mod, "input_layernorm"),
        (gqa_mod, "self_attn"),
        (gdn_mod, "linear_attn"),
        (post_norm_mod, "post_attention_layernorm"),
        (moe_mod_tp, "mlp"),
        (moe_mod_ep, "mlp"),
        (final_norm_mod, "model.norm"),
    ):
        b_decls = mod.hf_params(sub)
        canon_names = {d.name for d in b_decls}
        covered: set[str] = set()
        for d in b_decls:
            for alias in d.vllm_aliases:
                assert alias.source.startswith(sub), (alias.source, sub)
                assert set(alias.targets) <= canon_names
                assert set(alias.scale_targets) <= canon_names
                covered.update(alias.targets)
                covered.update(alias.scale_targets)
        assert covered == canon_names, (sub, canon_names - covered)

    # Check combined_hf_param_shapes on MoE and verify load_logical_weights and
    # load_hf / w_flat match the base golden hashes (4d43f2936) across all 12 combos.
    import hashlib
    import json
    from pathlib import Path

    moe_fused = combined_hf_param_shapes(moe_mod_tp.hf_params("mlp"))
    assert set(moe_fused.keys()) == {"sh13", "sh13_s", "e13", "e13_s"}

    golden_path = Path(__file__).parent / "golden" / "b5_weight_hashes.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden["base_commit"] == "4d43f2936f571c305f516031038d1378e62e6295"

    def _arr_info(a: Any) -> dict[str, Any]:
        b = np.ascontiguousarray(np.asarray(a)).view(np.uint8).tobytes()
        return {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "sha256": hashlib.sha256(b).hexdigest(),
        }

    ex_base = ExecSpec(R=2, B=2, T=1, ctx=2048)
    g_eplb = dataclasses.replace(qwen35.TINY_EP, e_slots=72)
    e_cores_eplb = mk.default_eplb_cores(g_eplb, ex_base.R)
    geom_cases = [
        ("TINY", qwen35.TINY, None, ("default", "candidate30", "candidate30-fused")),
        ("TINY_EP", qwen35.TINY_EP, None, ("default", "candidate30")),
        ("TINY_EP_EPLB72", g_eplb, e_cores_eplb, ("default",)),
    ]

    def _pinned_b5_hf_weights(key: jax.Array, geom: Any) -> dict[str, jax.Array]:
        from vllm_torchtpu.megakernel.ops import FP8
        from vllm_torchtpu.megakernel.weights.logical import (
            resolve_spec_and_norms,
            switch_branch_for_layer,
        )

        r_p = max(int(getattr(geom, "NKV", 1)), 1)
        ex_p = ExecSpec(R=r_p, B=r_p, T=1, ctx=2048)
        ms, geom, pre_n, post_n, final_n = resolve_spec_and_norms(
            qwen35.Qwen35, ex_p, g=geom
        )
        L_ = int(geom.L)
        cnts: dict[str, int] = {m.name: 0 for m in ms.all_modules()}
        for s in ms.layer_type.slots:
            if isinstance(s, mk.SwitchSlot):
                for lyr in range(L_):
                    bm, _ = switch_branch_for_layer(s, lyr)
                    cnts[bm.name] = cnts.get(bm.name, 0) + 1
            else:
                cnts[s.name] = cnts.get(s.name, 0) + L_
        non_rec, rec = [], []
        for m in ms.all_modules():
            if (cnt := cnts.get(m.name, 0)) > 0:
                (
                    rec
                    if isinstance(m.cache_decl(geom), RecurrentCacheDecl)
                    else non_rec
                ).append((m, cnt))
        it = iter(jax.random.split(key, 56))

        def _draw(d: Any, lead: tuple[int, ...]) -> jax.Array:
            sh = (*lead, *d.shape)
            if d.transform.scale_for is not None:
                arr = jnp.abs(jax.random.normal(next(it), sh, dtype=F32)) * 0.02 + 0.01
            elif d.transform.quantize is not None or jnp.dtype(d.dtype) == jnp.dtype(
                FP8
            ):
                arr = (jax.random.normal(next(it), sh, dtype=F32) * 0.25).astype(FP8)
            else:
                mean = 0.0 if d.init is None else float(d.init.mean)
                std = 0.05 if d.init is None else float(d.init.std)
                raw = jax.random.normal(next(it), sh, dtype=d.dtype) * std
                arr = jnp.ones(sh, dtype=d.dtype) + raw if mean == 1.0 else raw + mean
            tp = d.transform.transpose
            if d.hf_shape == d.shape and tp is not None:
                n_l = len(lead)
                arr = jnp.transpose(arr, (*range(n_l), *(n_l + i for i in tp)))
            return arr

        pre_drawn: dict[str, jax.Array] = {}
        for mod, n_l in non_rec:
            ds = mod.hf_params(g=geom)
            by_n = {x.name: x for x in ds}
            for d in ds:
                t = by_n.get(d.transform.scale_for, d) if d.transform.scale_for else d
                if t.transform.fuse_into is not None or bool(
                    t.vllm_aliases and len(t.vllm_aliases[-1].targets) > 1
                ):
                    pre_drawn[d.name] = _draw(d, (n_l,))
        out_hf: dict[str, jax.Array] = {}
        for nm in (pre_n, post_n):
            if nm is not None:
                nd = nm.hf_params(g=geom)[0]
                out_hf[nd.name] = _draw(nd, (L_,))
        if final_n is not None:
            fd = final_n.hf_params(g=geom)[0]
            out_hf[fd.name] = _draw(fd, ())
        for mod, n_l in (*non_rec, *rec):
            for d in mod.hf_params(g=geom):
                out_hf[d.name] = (
                    pre_drawn[d.name] if d.name in pre_drawn else _draw(d, (n_l,))
                )
        return out_hf

    # Verify mk.random_hf_weights draws at d.hf_shape and emits only decl names.
    hf_decl_only = mk.random_hf_weights(jax.random.PRNGKey(0), qwen35.TINY)
    expected_decl_names = {
        d.name
        for mod in (
            in_norm_mod,
            gqa_mod,
            gdn_mod,
            post_norm_mod,
            moe_mod_tp,
            final_norm_mod,
        )
        for d in mod.hf_params()
    }
    assert set(hf_decl_only) == expected_decl_names
    for d in gqa_mod.hf_params():
        assert hf_decl_only[d.name].shape == (qwen35.TINY.NATT, *d.hf_shape)
    for d in gdn_mod.hf_params():
        assert hf_decl_only[d.name].shape == (qwen35.TINY.NGDN, *d.hf_shape)
    for d in moe_mod_tp.hf_params():
        assert hf_decl_only[d.name].shape == (qwen35.TINY.L, *d.hf_shape)

    checked_packed = 0
    for g_name, g_case, e_cores, profiles in geom_cases:
        hf = _pinned_b5_hf_weights(jax.random.PRNGKey(0), g_case)
        w_l = mk.load_logical_weights(
            qwen35.Qwen35, ex_base, hf, g=g_case, e_cores=e_cores
        )
        got_logical = {
            k: _arr_info(v) for k, v in sorted(w_l.items()) if k != "e_cores"
        }
        assert got_logical == golden["logical"][g_name], f"logical mismatch on {g_name}"

        for prof_name in profiles:
            prof = qwen35.Qwen35.get_profile(prof_name)
            for stack_e in (False, True):
                ex_c = prof.exec_for(
                    dataclasses.replace(ex_base, stack_e=stack_e),
                    B=ex_base.B,
                    T=ex_base.T,
                    g=g_case,
                )
                ex_c = mk.fit_default_tiles(g_case.to_model_spec(), ex_c)
                w_mk = mk.load_hf(qwen35.Qwen35, ex_c, hf, g=g_case, e_cores=e_cores)
                p = planner.plan(g_case.to_model_spec(), ex_c)
                w_flat = [
                    x
                    for k in p.w_names
                    for x in (
                        w_mk[k] if isinstance(w_mk[k], (tuple, list)) else (w_mk[k],)
                    )
                ]
                got_w_mk = {
                    k: [_arr_info(x) for x in v]
                    if isinstance(v, tuple)
                    else _arr_info(v)
                    for k, v in sorted(w_mk.items())
                }
                got_w_flat = [_arr_info(x) for x in w_flat]
                combo_key = f"{g_name}/{prof_name}/stack_e={stack_e}"
                exp_combo = golden["packed"][combo_key]
                assert got_w_mk == exp_combo["w_mk"], f"w_mk mismatch on {combo_key}"
                assert got_w_flat == exp_combo["w_flat"], (
                    f"w_flat mismatch on {combo_key}"
                )
                checked_packed += 1
    assert checked_packed == 12

    # Verify apply_hf_transforms rejects wrong shapes and unsupported transform fields.
    q_decl = gqa_by_name["q_proj"]
    with pytest.raises(ValueError, match="trailing canonical shape"):
        apply_hf_transforms(
            [q_decl], {"q_proj": jnp.zeros(q_decl.hf_shape, dtype=q_decl.dtype)}, R=2
        )
    with pytest.raises(ValueError, match="does not support quantize='int4'"):
        apply_hf_transforms(
            [dataclasses.replace(q_decl, transform=ParamTransform(quantize="int4"))],
            {"q_proj": jnp.zeros(q_decl.shape, dtype=q_decl.dtype)},
            R=2,
        )
    with pytest.raises(ValueError, match="does not quantize on the fly"):
        apply_hf_transforms(
            [dataclasses.replace(q_decl, transform=ParamTransform(quantize="fp8"))],
            {"q_proj": jnp.zeros(q_decl.shape, dtype=BF16)},
            R=2,
        )
    with pytest.raises(ValueError, match="does not support head_reduce='sum'"):
        apply_hf_transforms(
            [
                dataclasses.replace(
                    q_decl, transform=ParamTransform(shard_axis=0, head_reduce="sum")
                )
            ],
            {"q_proj": jnp.zeros(q_decl.shape, dtype=q_decl.dtype)},
            R=2,
        )
    with pytest.raises(ValueError, match="both split_from and split_slice"):
        apply_hf_transforms(
            [dataclasses.replace(q_decl, transform=ParamTransform(split_from="raw"))],
            {"q_proj": jnp.zeros(q_decl.shape, dtype=q_decl.dtype)},
            R=2,
        )
    with pytest.raises(ValueError, match="conflicting declarations"):
        apply_hf_transforms(
            [
                q_decl,
                dataclasses.replace(q_decl, transform=ParamTransform(shard_axis=None)),
            ],
            {"q_proj": jnp.zeros(q_decl.shape, dtype=q_decl.dtype)},
            R=2,
        )


def test_module_bind_resolve_validation_and_fail_closed():
    """Test fail-closed options, field/rename/transforms validation, and deepcopy."""
    g = qwen35.TINY
    hf_cfg = _make_mock_hf_config(g)

    # 1. Fail-closed on unimplemented options, MoE mismatch, and conv_kernel != 4
    with pytest.raises(MegakernelConfigError, match="qk_norm=True"):
        GQAModule.bind("self_attn", qk_norm=False)
    with pytest.raises(MegakernelConfigError, match="output_gate=True"):
        GQAModule(output_gate=False)
    with pytest.raises(MegakernelConfigError, match="shared_gate=True"):
        MoEModule.bind("mlp", shared_gate=False)
    with pytest.raises(MegakernelConfigError, match="conv_kernel == 4"):
        GDNModule.bind("linear_attn", conv_kernel=3).resolve(hf_cfg)
    with pytest.raises(MegakernelConfigError, match="disagrees with geometry moe_tp"):
        MoEModule(geometry=qwen35.TINY_EP, moe_tp=True)
    with pytest.raises(MegakernelConfigError, match="disagrees with geometry e_slots"):
        MoEModule(
            geometry=dataclasses.replace(qwen35.TINY_EP, e_slots=72),
            moe_tp=False,
            e_slots=0,
        )

    # 2. Explicit HF config key must exist (no silent fallback to default key)
    with pytest.raises(ValueError, match="num_attn_heads_typo"):
        GQAModule.bind("self_attn", heads="num_attn_heads_typo").resolve(hf_cfg)

    # 3. Unknown/colliding rename or transforms keys raise ValueError; overrides apply
    with pytest.raises(ValueError, match="unknown rename key"):
        GQAModule.bind("self_attn", rename={"nonexistent": "foo"})
    with pytest.raises(ValueError, match="rename target collision"):
        GQAModule.bind("self_attn", rename={"q_proj": "k_proj"})
    with pytest.raises(ValueError, match="rename target collision"):
        GQAModule.bind("self_attn", rename={"q_proj": "dup_w", "k_proj": "dup_w"})
    with pytest.raises(ValueError, match="unknown transforms key"):
        GQAModule.bind("self_attn", transforms={"nonexistent": {"shard_axis": 0}})
    with pytest.raises(ValueError, match="unknown fields"):
        GQAModule.bind("self_attn", transforms={"q_proj": {"bad_field": 1}})

    gqa_custom = GQAModule.bind(
        "self_attn",
        rotary_dim=32,
        rename={"q_proj": "custom_q"},
        transforms={"custom_q": {"shard_axis": None}},
    ).resolve(hf_cfg)
    assert gqa_custom.geometry.ROT == 32
    custom_decls = {d.name: d for d in gqa_custom.hf_params()}
    assert "custom_q" in custom_decls
    assert custom_decls["custom_q"].transform == dataclasses.replace(
        GQAModule.bind("self_attn").resolve(hf_cfg).hf_params()[0].transform,
        shard_axis=None,
    )

    # 4. HF-neutral defaults: head_dim = hidden // heads, rot = 1.0,
    #    nested text_config.rope_parameters lookup, theta/vocab required,
    #    and Norm.zero_centered defaults to False.
    minimal_att_cfg = SimpleNamespace(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        rope_theta=10000.0,
    )
    resolved_minimal = GQAModule.bind("self_attn").resolve(minimal_att_cfg)
    assert resolved_minimal.HD == 64
    assert resolved_minimal.ROT == 64
    nested_rope_cfg = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
            rope_parameters={"rope_theta": 500000.0, "partial_rotary_factor": 0.5},
        )
    )
    resolved_nested = GQAModule.bind("self_attn").resolve(nested_rope_cfg)
    assert resolved_nested.theta == 500000.0
    assert resolved_nested.ROT == 32
    with pytest.raises(ValueError, match="theta"):
        GQAModule.bind("self_attn").resolve(
            SimpleNamespace(
                hidden_size=512, num_attention_heads=8, num_key_value_heads=2
            )
        )
    with pytest.raises(ValueError, match="vocab"):
        Embedding.bind("model.embed_tokens").resolve(SimpleNamespace(hidden_size=512))
    assert Norm("input_layernorm", geometry=g).options["zero_centered"] is False

    # 5. Norm.reference and Embedding.reference look up only the canonical/renamed key
    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (4, g.H), dtype=BF16)
    w = jax.random.normal(k2, (g.H,), dtype=F32) * 0.05

    norm_zc = Norm("input_layernorm", geometry=g, zero_centered=True, eps=g.eps)
    out_zc, st_none = norm_zc.reference({"in_norm": w}, x)
    assert st_none is None
    exp_zc = rmsnorm(x.astype(F32), w, g.eps, zero_centered=True)
    np.testing.assert_array_equal(np.asarray(out_zc), np.asarray(exp_zc))
    with pytest.raises(KeyError, match="in_norm"):
        norm_zc.reference({"weight": w}, x)

    norm_std = Norm("model.norm", geometry=g, eps=1e-5)
    out_std, st_passthru = norm_std.reference(w, x, {"sentinel": 1}, out_dtype=BF16)
    exp_std = rmsnorm(x.astype(F32), w, 1e-5, zero_centered=False).astype(BF16)
    assert out_std.dtype == BF16
    assert st_passthru == {"sentinel": 1}
    np.testing.assert_array_equal(np.asarray(out_std), np.asarray(exp_std))

    emb = Embedding.bind("model.embed_tokens", vocab=128, H=64).resolve(
        SimpleNamespace()
    )
    table = jax.random.normal(k3, (128, 64), dtype=BF16)
    tok_ids = jnp.asarray([[1, 5, 19], [0, 42, 127]], dtype=I32)
    looked_up, _ = emb.reference({"embed_tokens": table}, tok_ids)
    np.testing.assert_array_equal(np.asarray(looked_up), np.asarray(table[tok_ids]))
    with pytest.raises(KeyError, match="embed_tokens"):
        emb.reference({"weight": table}, tok_ids)

    # 6. deepcopy / pickle / order-independent FrozenDict hashability
    spec = qwen35.qwen35_spec(qwen35.TINY)
    spec_copied = copy.deepcopy(spec)
    assert spec_copied.name == spec.name
    assert len(spec_copied.all_modules()) == len(spec.all_modules())
    binding = GQAModule.bind("self_attn", rename={"q_proj": "custom_q"})
    assert pickle.loads(pickle.dumps(binding)) == binding
    assert hash(binding) == hash(copy.deepcopy(binding))
    assert pickle.loads(pickle.dumps(gqa_custom)) == gqa_custom
    assert copy.deepcopy(REQUIRED) is REQUIRED
    assert pickle.loads(pickle.dumps(REQUIRED)) is REQUIRED
    fd1 = FrozenDict({"a": [1, {"x": [2, 3]}], "b": 4})
    fd2 = FrozenDict({"b": 4, "a": (1, {"x": (2, 3)})})
    assert fd1 == fd2
    assert fd1 == {"a": [1, {"x": [2, 3]}], "b": 4}
    assert fd1.__eq__(42) is NotImplemented
    assert hash(fd1) == hash(fd2)
    assert dict(MoEModule().options) == {"shared_gate": True}


def _pack_single_module_step_inputs(
    spec: ModelSpec,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    mesh: jax.sharding.Mesh,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Pack logical weights/states for a single-slot 1-layer `ModelSpec`."""
    p = planner.plan(spec, ex)
    R, B, M = ex.R, ex.B, ex.M
    L = g.L

    hid_pad = jnp.pad(hidden[:B], ((0, M - B), (0, 0)))
    hid_mk = jnp.tile(hid_pad[None, :, :], (R, 1, 1)).reshape(R * M, g.H)

    class _SingleModuleDef(mk.ModelDef):
        schedule = mk.Schedule(
            mixer=GQAModule.bind("self_attn"),
            ffn=MoEModule.bind("mlp"),
            pre_norm=(
                Norm.bind(
                    "input_layernorm",
                    zero_centered=g.zero_centered_norm,
                    small_field="in_norm",
                )
                if "in_norm" in p.smalls_layout
                else None
            ),
            post_norm=(
                Norm.bind(
                    "post_attention_layernorm",
                    zero_centered=g.zero_centered_norm,
                    small_field="post_norm",
                )
                if "post_norm" in p.smalls_layout
                else None
            ),
        )
        final_norm = Norm.bind("model.norm", zero_centered=g.zero_centered_norm)

        @classmethod
        def to_model_spec(cls, hf_config_or_geom: Any = None, **kw: Any) -> ModelSpec:
            return spec

    weights_mk = mk.pack_logical_weights(_SingleModuleDef, ex, logical_weights, g=g)

    states_mk: dict[str, Any] = {}
    for s_name in p.s_names:
        if s_name in ("kc", "vc"):
            states_mk[s_name] = kernel.fold(logical_states[s_name])
        elif s_name == POOL_STATE_NAME:
            pt, ckpt, roff, _ = qwen35.pool.default_pool_layout(g, ex)
            raw_pools = qwen35.pool.states_to_pool(
                g, ex, logical_states, pt, ckpt, roff
            )
            states_mk[POOL_STATE_NAME] = raw_pools
            states_mk["page_table"] = jnp.tile(
                pt[:, : resolved_max_pages_per_seq(ex)], (R, 1)
            )
        elif s_name == "conv":
            conv_l = logical_states["conv"]
            qkv = g.QKV(R)
            if ex.T == 1:
                cr = rup(3 * B, 8)
                conv_3b = jnp.swapaxes(conv_l, 2, 3).reshape(R, L, 3 * B, qkv)
                conv_p = jnp.pad(conv_3b, ((0, 0), (0, 0), (0, cr - 3 * B), (0, 0)))
            else:
                hr = rup(3 + ex.T, 8)
                conv_p = jnp.pad(conv_l, ((0, 0), (0, 0), (0, 0), (0, hr - 3), (0, 0)))
            states_mk["conv"] = kernel.fold(conv_p.astype(F32))
        elif s_name == "rec":
            rec_l = logical_states["rec"]
            vh = g.vh(R)
            rec_p = (
                rec_l.reshape(R, L, ex.NS * vh, g.KD, g.VD)
                if rec_l.ndim == 6
                else rec_l
            )
            states_mk["rec"] = kernel.fold(rec_p.astype(BF16))

    rep = NamedSharding(mesh, P())
    tp = NamedSharding(mesh, P("tp"))
    pos = jax.device_put(pos, rep)
    hid_mk = jax.device_put(hid_mk, tp)
    weights_mk = {
        k: tuple(jax.device_put(x, tp) for x in v)
        if isinstance(v, tuple)
        else jax.device_put(v, tp)
        for k, v in weights_mk.items()
    }
    states_mk = {
        k: tuple(jax.device_put(x, tp) for x in v)
        if isinstance(v, tuple)
        else jax.device_put(v, tp)
        for k, v in states_mk.items()
    }
    return pos, hid_mk, weights_mk, states_mk


_TINY_STEP_TILES = Tiles(
    bank=512 << 10,
    nbd=10,
    nbe=8,
    ebank=512 << 10,
    vmem_mib=64,
    modules={"gqa": GQATunables(kv_blk=128, nkv=2)},
)


@dataclasses.dataclass(frozen=True)
class _SingleLayerGDNGeometry(qwen35.Qwen35Geometry):
    """1-layer GDN-only geometry for isolated `GDNModule` step testing."""

    @property
    def NGDN(self) -> int:
        return self.L

    @property
    def NATT(self) -> int:
        return 0

    def check(self, ex: ExecSpec) -> None:
        return None


@pytest.mark.parametrize("paged_kv", [False, True])
@pytest.mark.parametrize("T", [1, 2])
def test_gqa_module_reference_matches_kernel_make_step_cpu(T: int, paged_kv: bool):
    """`GQAModule.reference` matches `kernel.make_step` on a 1-module GQA model."""
    g = dataclasses.replace(qwen35.TINY, L=1, att_every=1)
    cache_cfg = (
        CacheExec(paged_kv=True, num_pages=64)
        if paged_kv
        else CacheExec(paged_kv=False)
    )
    ex = ExecSpec(
        R=2, B=2, T=T, ctx=128, interpret=True, tiles=_TINY_STEP_TILES, cache=cache_cfg
    )
    mesh = _cpu_mesh(ex.R)

    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.PRNGKey(10 + T), g, ex)
    gqa_mod = GQAModule(geometry=g)
    in_norm_mod = Norm(
        "input_layernorm", geometry=g, zero_centered=g.zero_centered_norm
    )
    final_norm_mod = Norm("model.norm", geometry=g, zero_centered=g.zero_centered_norm)

    res = hid_l[: ex.B].astype(F32)
    x_norm, _ = in_norm_mod.reference(w_l["in_norm"][0], res, g=g)
    att_delta, ref_states = gqa_mod.reference(
        w_l,
        x_norm,
        {"kc": st_l["kc"], "vc": st_l["vc"]},
        g=g,
        ex=ex,
        pos=pos,
        att_idx=0,
        layer_idx=0,
        paged_kv=paged_kv,
    )
    res = rnd(res + att_delta)
    ref_out, _ = final_norm_mod.reference(w_l["final_norm"], res, g=g, out_dtype=BF16)

    spec = ModelSpec(
        name="gqa_single",
        config=g,
        layer_type=LayerType(name="gqa_only", slots=(gqa_mod,)),
        cache_backend=qwen35.pool.qwen35_pool_backend(g) if paged_kv else None,
    )
    pos_mk, hid_mk, w_mk, st_mk = _pack_single_module_step_inputs(
        spec, g, ex, pos, hid_l, w_l, st_l, mesh
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    mk_hid, mk_states = step(pos_mk, hid_mk, w_mk, st_mk)
    mk_hid_ranks = np.asarray(
        mk_hid.reshape(ex.R, ex.M, g.H)[:, : ex.B], dtype=np.float32
    )
    ref_out_np = np.asarray(ref_out, dtype=np.float32)
    for r in range(ex.R):
        np.testing.assert_allclose(mk_hid_ranks[r], ref_out_np, atol=2e-2, rtol=2e-2)

    if not paged_kv:
        kc_mk = np.asarray(
            mk_states["kc"].reshape(ex.R, 1, ex.NS, ex.ctx, g.HD), dtype=np.float32
        )
        vc_mk = np.asarray(
            mk_states["vc"].reshape(ex.R, 1, ex.NS, ex.ctx, g.HD), dtype=np.float32
        )
        np.testing.assert_allclose(
            kc_mk, np.asarray(ref_states["kc"], dtype=np.float32), atol=1e-2, rtol=1e-2
        )
        np.testing.assert_allclose(
            vc_mk, np.asarray(ref_states["vc"], dtype=np.float32), atol=1e-2, rtol=1e-2
        )
    else:
        knew_mk = np.asarray(
            mk_states["knew"].reshape(ex.R, 1, ex.MR, g.HD)[:, 0, : ex.B],
            dtype=np.float32,
        )
        vnew_mk = np.asarray(
            mk_states["vnew"].reshape(ex.R, 1, ex.MR, g.HD)[:, 0, : ex.B],
            dtype=np.float32,
        )
        kc_ref = np.asarray(ref_states["kc"], dtype=np.float32)
        vc_ref = np.asarray(ref_states["vc"], dtype=np.float32)
        for b in range(ex.B):
            sq = b // ex.T
            p_b = int(pos[b])
            np.testing.assert_allclose(
                knew_mk[:, b], kc_ref[:, 0, sq, p_b], atol=1e-2, rtol=1e-2
            )
            np.testing.assert_allclose(
                vnew_mk[:, b], vc_ref[:, 0, sq, p_b], atol=1e-2, rtol=1e-2
            )


@pytest.mark.parametrize("T", [1, 2])
def test_gdn_module_reference_matches_kernel_make_step_cpu(T: int):
    """`GDNModule.reference` matches `kernel.make_step` on a 1-module GDN model."""
    ex = ExecSpec(R=2, B=2, T=T, ctx=128, interpret=True, tiles=_TINY_STEP_TILES)
    mesh = _cpu_mesh(ex.R)

    pos, hid_l, w_l4, st_l4 = mk.random_logical_inputs(
        jax.random.PRNGKey(20 + T), qwen35.TINY, ex
    )
    g = _SingleLayerGDNGeometry(
        L=1,
        att_every=2,
        H=qwen35.TINY.H,
        NH=qwen35.TINY.NH,
        NKV=qwen35.TINY.NKV,
        HD=qwen35.TINY.HD,
        ROT=qwen35.TINY.ROT,
        KH=qwen35.TINY.KH,
        VH=qwen35.TINY.VH,
        KD=qwen35.TINY.KD,
        VD=qwen35.TINY.VD,
        E=qwen35.TINY.E,
        TOPK=qwen35.TINY.TOPK,
        I=qwen35.TINY.I,
        IS=qwen35.TINY.IS,
    )
    w_l: dict[str, Any] = {
        "in_norm": w_l4["in_norm"][:1],
        "final_norm": w_l4["final_norm"],
        "lin_in": w_l4["lin_in"][:, :1],
        "lin_in_s": w_l4["lin_in_s"][:, :1],
        "lin_out": w_l4["lin_out"][:, :1],
        "lin_out_s": w_l4["lin_out_s"][:, :1],
        "conv": w_l4["conv"][:, :1],
        "alog": w_l4["alog"][:, :1],
        "dtb": w_l4["dtb"][:, :1],
        "gnorm": w_l4["gnorm"][:, :1],
    }
    st_l: dict[str, Any] = {
        "conv": st_l4["conv"][:, :1],
        "rec": st_l4["rec"][:, :1],
    }
    gdn_mod = GDNModule(geometry=g)
    in_norm_mod = Norm(
        "input_layernorm", geometry=g, zero_centered=g.zero_centered_norm
    )
    final_norm_mod = Norm("model.norm", geometry=g, zero_centered=g.zero_centered_norm)

    res = hid_l[: ex.B].astype(F32)
    x_norm, _ = in_norm_mod.reference(w_l["in_norm"][0], res, g=g)
    gdn_delta, ref_states = gdn_mod.reference(
        w_l,
        x_norm,
        {"conv": st_l["conv"], "rec": st_l["rec"]},
        g=g,
        ex=ex,
        gdn_idx=0,
        layer_idx=0,
        multi_layer=True,
    )
    res = rnd(res + gdn_delta)
    ref_out, _ = final_norm_mod.reference(w_l["final_norm"], res, g=g, out_dtype=BF16)

    spec = ModelSpec(
        name="gdn_single",
        config=g,
        layer_type=LayerType(name="gdn_only", slots=(gdn_mod,)),
    )
    pos_mk, hid_mk, w_mk, st_mk = _pack_single_module_step_inputs(
        spec, g, ex, pos, hid_l, w_l, st_l, mesh
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    mk_hid, mk_states = step(pos_mk, hid_mk, w_mk, st_mk)
    mk_hid_ranks = np.asarray(
        mk_hid.reshape(ex.R, ex.M, g.H)[:, : ex.B], dtype=np.float32
    )
    ref_out_np = np.asarray(ref_out, dtype=np.float32)
    for r in range(ex.R):
        np.testing.assert_allclose(mk_hid_ranks[r], ref_out_np, atol=2e-2, rtol=2e-2)

    lay_gdn = GDNModule.lin_in_layout(g, ex)
    rec_key = "rec_ckpt" if ex.VER else "rec"
    conv_key = "conv_ckpt" if ex.VER else "conv"
    rec_mk = np.asarray(
        mk_states[rec_key].reshape(ex.R, ex.B * lay_gdn.vh, lay_gdn.KD, lay_gdn.VD),
        dtype=np.float32,
    )
    rec_ref = np.asarray(ref_states[rec_key], dtype=np.float32)
    np.testing.assert_allclose(rec_mk, rec_ref, atol=2e-2, rtol=2e-2)

    conv_ref = np.asarray(ref_states[conv_key], dtype=np.float32)
    conv_mk = np.asarray(mk_states[conv_key].reshape(conv_ref.shape), dtype=np.float32)
    np.testing.assert_allclose(conv_mk, conv_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("moe_tp", [True, False])
def test_moe_module_reference_matches_kernel_make_step_cpu(moe_tp: bool):
    """`MoEModule.reference` matches `kernel.make_step` on a 1-module MoE model."""
    g = dataclasses.replace(qwen35.TINY, L=1, att_every=1, moe_tp=moe_tp)
    ex = ExecSpec(R=2, B=2, T=1, ctx=128, interpret=True, tiles=_TINY_STEP_TILES)
    mesh = _cpu_mesh(ex.R)

    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.PRNGKey(30), g, ex)
    moe_mod = MoEModule(geometry=g, moe_tp=moe_tp)
    post_norm_mod = Norm(
        "post_attention_layernorm", geometry=g, zero_centered=g.zero_centered_norm
    )
    final_norm_mod = Norm("model.norm", geometry=g, zero_centered=g.zero_centered_norm)

    res = hid_l[: ex.B].astype(F32)
    x_moe, _ = post_norm_mod.reference(w_l["post_norm"][0], res, g=g)
    moe_delta, _ = moe_mod.reference(
        w_l,
        x_moe,
        None,
        g=g,
        ex=ex,
        layer_idx=0,
    )
    res = rnd(res + moe_delta)
    ref_out, _ = final_norm_mod.reference(w_l["final_norm"], res, g=g, out_dtype=BF16)

    spec = ModelSpec(
        name="moe_single",
        config=g,
        layer_type=LayerType(name="moe_only", slots=(moe_mod,)),
    )
    pos_mk, hid_mk, w_mk, st_mk = _pack_single_module_step_inputs(
        spec, g, ex, pos, hid_l, w_l, st_l, mesh
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    mk_hid, _ = step(pos_mk, hid_mk, w_mk, st_mk)
    mk_hid_ranks = np.asarray(
        mk_hid.reshape(ex.R, ex.M, g.H)[:, : ex.B], dtype=np.float32
    )
    ref_out_np = np.asarray(ref_out, dtype=np.float32)
    for r in range(ex.R):
        max_diff = float(np.max(np.abs(mk_hid_ranks[r] - ref_out_np)))
        assert max_diff < 2e-2, (
            f"MoE kernel vs module.reference rank {r} (moe_tp={moe_tp}) "
            f"max_diff={max_diff}"
        )


def test_slice_alias_targets_jax_array_ops_and_loader_validations():
    """Unit test `slice_alias_targets` (`JaxArrayOps`) and loader validations."""
    from vllm_torchtpu.megakernel.ops import FP8
    from vllm_torchtpu.megakernel.weights import (
        JaxArrayOps,
        resolve_spec_and_norms,
        slice_alias_targets,
    )

    ops = JaxArrayOps()
    g = qwen35.TINY
    ex = ExecSpec(R=2, B=2, T=1, ctx=128)
    moe_decls = {d.name: d for d in MoEModule(geometry=g).hf_params("mlp")}
    gdn_decls = {d.name: d for d in GDNModule(geometry=g).hf_params("linear_attn")}

    # 1. Fused 3-target alias (_fused_gate_gu_fp8_weight): router dequantized to BF16,
    #    sh_gate_proj + sh_up_proj sliced per TP rank.
    fused_alias = moe_decls["router"].vllm_aliases[0]
    t_decls = [moe_decls[tn] for tn in fused_alias.targets]
    s_decls = {sn: moe_decls[sn] for sn in fused_alias.scale_targets}
    total_cols = g.E + 2 * g.IS
    w_fused = (
        jax.random.normal(jax.random.PRNGKey(1), (g.H, total_cols), dtype=F32) * 0.25
    ).astype(FP8)
    s_fused = jnp.linspace(0.01, 0.05, total_cols, dtype=F32)
    sliced_r1 = slice_alias_targets(
        ops, fused_alias, w_fused, s_fused, t_decls, s_decls, R=2, tp_rank=1
    )
    exp_router = (w_fused[:, : g.E].astype(F32) * s_fused[: g.E][None, :]).astype(BF16)
    np.testing.assert_array_equal(
        np.asarray(sliced_r1["router"]).view(np.uint8),
        np.asarray(exp_router).view(np.uint8),
    )
    isr = g.ISR(2)
    np.testing.assert_array_equal(
        np.asarray(sliced_r1["sh_gate_proj"]).view(np.uint8),
        np.asarray(w_fused[:, g.E + isr : g.E + 2 * isr]).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(sliced_r1["sh_gate_proj_s"]),
        np.asarray(s_fused[g.E + isr : g.E + 2 * isr]),
    )

    # 2. Fast path vs padded slow path: padded GDN lin_b/lin_a keeps scales
    ba_alias = gdn_decls["lin_b"].vllm_aliases[0]
    ba_t = [gdn_decls[tn] for tn in ba_alias.targets]
    ba_s = {sn: gdn_decls[sn] for sn in ba_alias.scale_targets}
    vh = g.vh(2)
    w_ba_bf16 = jax.random.normal(jax.random.PRNGKey(2), (g.H, 2 * vh), dtype=BF16)
    ba_out = slice_alias_targets(
        ops,
        ba_alias,
        w_ba_bf16,
        None,
        ba_t,
        ba_s,
        R=2,
        tp_rank=0,
        fuse_groups={"lin_in": ("lin_b", "lin_a"), "lin_in_s": ("lin_b_s", "lin_a_s")},
    )
    assert "lin_b" in ba_out and "lin_b_s" in ba_out
    assert "lin_a" in ba_out and "lin_a_s" in ba_out
    assert ba_out["lin_b"].shape == (g.H, 64)
    assert ba_out["lin_b_s"].shape == (64,)

    # Even if already_fp8=True on a padded group, fast path is skipped so scales stay
    ba_fp8_alias = dataclasses.replace(ba_alias, already_fp8=True)
    w_ba_fp8, s_ba_fp8 = ops.quantize_fp8(w_ba_bf16)
    ba_fp8_out = slice_alias_targets(
        ops,
        ba_fp8_alias,
        w_ba_fp8,
        s_ba_fp8,
        ba_t,
        ba_s,
        R=2,
        tp_rank=0,
        fuse_groups={"lin_in": ("lin_b", "lin_a"), "lin_in_s": ("lin_b_s", "lin_a_s")},
    )
    assert ba_fp8_out["lin_b_s"].shape == (64,)
    assert ba_fp8_out["lin_a_s"].shape == (64,)

    # 3. Error checks: already_fp8 mismatch, already_sharded width mismatch, scale width
    with pytest.raises(ValueError, match="declared already_fp8=True"):
        slice_alias_targets(
            ops,
            fused_alias,
            w_fused.astype(BF16),
            s_fused,
            t_decls,
            s_decls,
            R=2,
            tp_rank=0,
        )
    with pytest.raises(ValueError, match="already_sharded=False"):
        slice_alias_targets(
            ops,
            fused_alias,
            w_fused[:, : g.E + isr],
            s_fused,
            t_decls,
            s_decls,
            R=2,
            tp_rank=0,
        )
    with pytest.raises(ValueError, match="Scale for alias .* has trailing width"):
        slice_alias_targets(
            ops,
            fused_alias,
            w_fused,
            s_fused[:-1],
            t_decls,
            s_decls,
            R=2,
            tp_rank=0,
        )

    # 4. resolve_spec_and_norms requires a ModelDef subclass
    with pytest.raises(TypeError, match="Expected a ModelDef subclass"):
        resolve_spec_and_norms(g.to_model_spec(), ex, g=g)

    # 5. _coerce_canonical_hf_entry transposes hf_shape and rejects invalid shapes
    hf_valid = mk.random_hf_weights(jax.random.PRNGKey(3), g)
    hf_bad = dict(hf_valid, sh2=jnp.zeros((g.L, g.H * g.IS, 1), dtype=FP8))
    with pytest.raises(ValueError, match="trailing shape"):
        mk.load_logical_weights(qwen35.Qwen35, ex, hf_bad, g=g)

    # Transposed 2D hf_shape (L, H, IS) on sh2 (shape=(IS, H), transpose=(1, 0)) works
    g_rect = dataclasses.replace(g, IS=512)
    hf_rect = mk.random_hf_weights(jax.random.PRNGKey(30), g_rect)
    assert hf_rect["sh2"].shape == (g_rect.L, g_rect.H, g_rect.IS)
    sh2_kn = jnp.transpose(hf_rect["sh2"], (0, 2, 1))
    assert sh2_kn.shape == (g_rect.L, g_rect.IS, g_rect.H)
    w_l_trans = mk.load_logical_weights(qwen35.Qwen35, ex, hf_rect, g=g_rect)
    w_l_canon = mk.load_logical_weights(
        qwen35.Qwen35, ex, dict(hf_rect, sh2=sh2_kn), g=g_rect
    )
    np.testing.assert_array_equal(
        np.asarray(w_l_trans["sh2"]).view(np.uint8),
        np.asarray(w_l_canon["sh2"]).view(np.uint8),
    )

    # 6. EPLB branch validates expert dtype/shape
    g_eplb = dataclasses.replace(qwen35.TINY_EP, e_slots=72)
    hf_ep = mk.random_hf_weights(jax.random.PRNGKey(4), g_eplb)
    with pytest.raises(ValueError, match="dtype"):
        mk.load_logical_weights(
            qwen35.Qwen35, ex, dict(hf_ep, e2=hf_ep["e2"].astype(BF16)), g=g_eplb
        )
