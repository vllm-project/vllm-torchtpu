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
"""Qwen3.5 `ModelDef` derivation, `ModelSpec`/`PoolLayout`, and reference tests."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vllm_torchtpu.megakernel as mk
from tests.megakernel import frozen_utils
from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.decls import RecurrentCacheDecl
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.model_def import (
    Every,
    FromHF,
    LMHead,
    ModelDef,
    Schedule,
    find_model,
    get_model,
    register_model,
)
from vllm_torchtpu.megakernel.modules import (
    GDN,
    GQA,
    Embedding,
    GDNModule,
    GQAModule,
    MoE,
    MoEModule,
    Norm,
)
from vllm_torchtpu.megakernel.ops import BF16, F32, round_bf16
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35_mod
from vllm_torchtpu.models.qwen3_5.megakernel import pool as qwen35_pool
from vllm_torchtpu.models.qwen3_5.megakernel.model import DEFAULT_HF_CONFIG, Qwen35
from vllm_torchtpu.models.qwen3_5.megakernel.profiles.baseline import (
    PROFILE,
    QWEN35_MODEL_NAME,
    TINY_TILES,
)


def assert_modules_equal(ba: mk.Module, bb: mk.Module) -> None:
    """Assert structural equality of two `Module` instances including `prefix`."""
    assert type(ba) is type(bb)
    assert (
        ba.name,
        ba.reduction,
        ba.input_norm,
        ba.prefix,
        ba.options,
    ) == (
        bb.name,
        bb.reduction,
        bb.input_norm,
        bb.prefix,
        bb.options,
    )
    if isinstance(ba, GQAModule):
        assert isinstance(bb, GQAModule)
        assert (ba.pool_layout, ba.qk_norm, ba.output_gate) == (
            bb.pool_layout,
            bb.qk_norm,
            bb.output_gate,
        )
    elif isinstance(ba, GDNModule):
        assert isinstance(bb, GDNModule)
        assert ba.prefix == bb.prefix
    elif isinstance(ba, MoEModule):
        assert isinstance(bb, MoEModule)
        assert (
            ba.shared_gate,
            ba._explicit_moe_tp,
            ba._explicit_e_slots,
            ba.moe_tp,
            ba.e_slots,
        ) == (
            bb.shared_gate,
            bb._explicit_moe_tp,
            bb._explicit_e_slots,
            bb.moe_tp,
            bb.e_slots,
        )
    elif isinstance(ba, Norm):
        assert isinstance(bb, Norm)
        assert (ba.zero_centered, ba.eps, ba.param_name) == (
            bb.zero_centered,
            bb.eps,
            bb.param_name,
        )
    elif isinstance(ba, Embedding):
        assert isinstance(bb, Embedding)
        assert (ba.param_name, ba.dtype) == (bb.param_name, bb.dtype)


def assert_pool_backends_equal(
    ba: mk.PoolBackend | None, bb: mk.PoolBackend | None
) -> None:
    """Assert structural equality of two `PoolBackend` instances including callbacks."""
    if ba is None or bb is None:
        assert ba is bb
        return
    assert type(ba) is type(bb) is mk.PoolBackend
    assert (
        ba.layout,
        ba.kernel_names,
        ba.page_table_state_name,
        ba.supports_fused_io,
        ba.supports_pool_gather,
    ) == (
        bb.layout,
        bb.kernel_names,
        bb.page_table_state_name,
        bb.supports_fused_io,
        bb.supports_pool_gather,
    )
    for attr in ("gather_states", "scatter_states", "can_pool_gather"):
        fn_a = getattr(ba, attr)
        fn_b = getattr(bb, attr)
        assert callable(fn_a) and callable(fn_b)
        assert fn_a.__code__ is fn_b.__code__
        assert fn_a.__qualname__ == fn_b.__qualname__
        nl_a = inspect.getclosurevars(fn_a).nonlocals
        nl_b = inspect.getclosurevars(fn_b).nonlocals
        assert set(nl_a) == set(nl_b)
        assert nl_a["layout"] == nl_b["layout"]
        if "has_rec" in nl_a:
            assert nl_a["has_rec"] == nl_b["has_rec"]
        if "rec_chunk" in nl_a:
            assert nl_a["rec_chunk"] == nl_b["rec_chunk"]
        if "check" in nl_a:
            chk_a, chk_b = nl_a["check"], nl_b["check"]
            assert (
                getattr(chk_a, "__func__", chk_a),
                getattr(chk_a, "__self__", None),
            ) == (
                getattr(chk_b, "__func__", chk_b),
                getattr(chk_b, "__self__", None),
            )
    for ex_probe in (
        mk.with_cache(ExecSpec(R=8, B=8, T=1, ctx=256), paged_kv=True),
        mk.with_cache(
            ExecSpec(R=8, B=24, T=4, ctx=256),
            paged_kv=True,
            pool_gather="eligible",
        ),
    ):
        assert ba.can_pool_gather(ex_probe) == bb.can_pool_gather(ex_probe)


def assert_specs_equal(spec_a: mk.ModelSpec, spec_b: mk.ModelSpec) -> None:
    """Assert structural equality of two `ModelSpec` instances."""
    assert type(spec_a) is type(spec_b) is mk.ModelSpec
    assert spec_a.name == spec_b.name
    assert type(spec_a.config) is type(spec_b.config)
    assert spec_a.config == spec_b.config
    assert_pool_backends_equal(spec_a.cache_backend, spec_b.cache_backend)
    assert spec_a.layer_type.name == spec_b.layer_type.name
    assert len(spec_a.layer_type.slots) == len(spec_b.layer_type.slots)
    for sa, sb in zip(spec_a.layer_type.slots, spec_b.layer_type.slots, strict=True):
        if isinstance(sa, mk.SwitchSlot):
            assert isinstance(sb, mk.SwitchSlot)
            assert sa.name == sb.name
            assert sa.branch_counts == sb.branch_counts
            assert sa.period == sb.period
            for ba, bb in zip(sa.branches, sb.branches, strict=True):
                assert_modules_equal(ba, bb)
        else:
            assert isinstance(sa, mk.Module) and isinstance(sb, mk.Module)
            assert_modules_equal(sa, sb)
    for ba, bb in zip(spec_a.all_modules(), spec_b.all_modules(), strict=True):
        assert_modules_equal(ba, bb)


def _independent_qwen35_pool_layout(
    g: qwen35.Qwen35Geometry, *, R: int = 8
) -> mk.PoolLayout:
    """Build expected `PoolLayout` from `cache_decl()` + `derive_pool_layout`."""
    kv_decl = GQAModule(geometry=g).cache_decl(g)
    assert kv_decl is not None
    if g.att_every == 1:
        decls: list[Any] = [kv_decl] * g.NATT
    else:
        rec_decl = GDNModule(geometry=g).cache_decl(g)
        assert rec_decl is not None
        decls = ([rec_decl] * (g.att_every - 1) + [kv_decl]) * g.NATT
    return mk.pool.derive_pool_layout(decls, R=R)


def _independent_qwen35_spec(
    g: qwen35.Qwen35Geometry, *, gdn_module: mk.Module | None = None, R: int = 8
) -> mk.ModelSpec:
    """Build expected `ModelSpec` independently of `Qwen35.to_model_spec`."""
    play = _independent_qwen35_pool_layout(g, R=R)
    gqa_mod = GQAModule(
        pool_layout=play, prefix="self_attn", qk_norm=True, output_gate=True
    )
    moe_mod = MoEModule(prefix="mlp", shared_gate=True)
    if g.att_every == 1:
        mixer = mk.SwitchSlot("mixer", branches=(gqa_mod,), branch_counts=(1,))
    else:
        gdn_mod = GDNModule(prefix="linear_attn") if gdn_module is None else gdn_module
        mixer = mk.SwitchSlot(
            "mixer", branches=(gdn_mod, gqa_mod), branch_counts=(g.att_every - 1, 1)
        )
    lt = mk.LayerType(name="qwen35_layer", slots=(mixer, moe_mod))
    backend = mk.pool.make_pool_backend(
        play, check=g.check, rec_chunk=GDNModule.chunk_rows
    )
    return mk.ModelSpec(
        name=QWEN35_MODEL_NAME,
        config=g,
        layer_type=lt,
        cache_backend=backend,
    )


def test_qwen35_model_def_line_budget_docstrings_and_runtime_registry(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """The Qwen3.5 model folder's `model.py` has no fmt: skip / noqa, keeps
    docstrings, tests driver."""
    import logging
    import types

    from vllm_torchtpu.megakernel_runtime import driver as driver_mod
    from vllm_torchtpu.models.qwen3_5.megakernel import adapter as qwen_adapter
    from vllm_torchtpu.models.qwen3_5.megakernel.profiles import (
        baseline as qwen35_profile,
    )

    def _declined(drv: Any) -> bool:
        return isinstance(drv, driver_mod.DeclinedDriver)

    monkeypatch.setattr(
        driver_mod,
        "logger",
        logging.getLogger("vllm_torchtpu.megakernel_runtime.driver"),
    )

    qwen_text = Path(qwen35.__file__).read_text(encoding="utf-8")
    qwen_lines = qwen_text.splitlines()
    assert len(qwen_lines) <= 230, f"model.py has {len(qwen_lines)} lines (> 230)"
    assert "fmt: skip" not in qwen_text, "qwen35.py contains fmt: skip"
    assert "noqa" not in qwen_text, "qwen35.py contains noqa"
    for mod in (qwen35_pool, qwen35_profile):
        p = Path(mod.__file__)
        mod_text = p.read_text(encoding="utf-8")
        n_lines = len(mod_text.splitlines())
        assert n_lines <= 400, f"{p.name} has {n_lines} lines (> 400)"
        assert "fmt: skip" not in mod_text, f"{p.name} contains fmt: skip"
        assert "noqa" not in mod_text, f"{p.name} contains noqa"

    for fn in (
        qwen35.Qwen35Geometry.to_model_spec,
        qwen35.qwen35_spec,
        qwen35.build,
        qwen35.make_step,
        qwen35.make_pooled_step,
    ):
        assert fn.__doc__ and fn.__doc__.strip(), f"{fn.__name__} missing docstring"
    assert "checkpoint slots" in (qwen35_pool.__doc__ or "")
    assert "donate_states" in inspect.signature(qwen35.make_step).parameters
    pooled_params = inspect.signature(qwen35.make_pooled_step).parameters
    for kw_name in ("donate_pools", "fused_pool_io", "max_b_low", "min_b_hi"):
        assert kw_name in pooled_params

    assert qwen35.register() is Qwen35
    assert get_model(QWEN35_MODEL_NAME) is Qwen35
    assert get_model("Qwen3_5MoeForConditionalGeneration") is Qwen35
    assert get_model("Qwen3_5ForConditionalGeneration") is Qwen35
    assert get_model("qwen3_5_moe") is Qwen35
    assert get_model("qwen3_5_moe_text") is Qwen35
    assert (
        get_model(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": DEFAULT_HF_CONFIG,
            }
        )
        is Qwen35
    )
    assert Qwen35.default_profile() is PROFILE
    assert Qwen35.get_profile("candidate30").name == "qwen35-candidate30"

    assert Qwen35.hf_architectures == (
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLM",
    )
    assert Qwen35.hf_model_type == ("qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text")
    assert Qwen35.supported_tp_sizes == (8,)
    assert Qwen35.display_name == "Qwen3.5"

    dense_cfg = {k: v for k, v in DEFAULT_HF_CONFIG.items() if k != "num_experts"}
    with pytest.raises(
        MegakernelConfigError, match="Qwen35 supports MoE checkpoints only"
    ):
        Qwen35.derive_geometry(dense_cfg)

    class _UnregisteredMissingFinalNorm(Qwen35):
        final_norm = None

    for fn_call in (
        lambda: _UnregisteredMissingFinalNorm.derive_geometry(DEFAULT_HF_CONFIG),
        lambda: _UnregisteredMissingFinalNorm.to_model_spec(qwen35.TINY),
        lambda: _UnregisteredMissingFinalNorm.reference_step(
            qwen35.TINY, ExecSpec(R=2, B=2, T=1, ctx=128), None, None, {}, {}
        ),
    ):
        with pytest.raises(
            MegakernelConfigError,
            match=r"_UnregisteredMissingFinalNorm.*final_norm is None",
        ) as fn_exc:
            fn_call()
        assert "layer " not in str(fn_exc.value)

    def dict_to_namespace(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return types.SimpleNamespace(
                **{str(k): dict_to_namespace(v) for k, v in obj.items()}
            )
        if isinstance(obj, list):
            return [dict_to_namespace(x) for x in obj]
        return obj

    def _make_runner(hf_cfg: Any) -> Any:
        from vllm_torchtpu.megakernel_runtime import testing

        return testing.fake_runner(
            hf_config=hf_cfg,
            tp_size=8,
            mamba_ssm_cache_dtype="bfloat16",
            megakernel={"enabled": True},
        )

    nested_rope_text = {
        k: v
        for k, v in DEFAULT_HF_CONFIG.items()
        if k not in ("rope_theta", "partial_rotary_factor")
    }
    nested_rope_text["rope_parameters"] = {
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.25,
    }
    nested_vlm = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": nested_rope_text,
    }
    for valid_cfg in (
        nested_vlm,
        dict_to_namespace(nested_vlm),
    ):
        assert Qwen35.validate_hf_config(valid_cfg).eligible
        assert qwen_adapter.validate_hf_config(valid_cfg).eligible

    cfg_48l = {
        **nested_vlm,
        "text_config": {
            **nested_rope_text,
            "num_hidden_layers": 48,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
        },
    }
    for cand_48l in (cfg_48l, dict_to_namespace(cfg_48l)):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert not Qwen35.validate_hf_config(cand_48l).eligible
            assert not qwen_adapter.validate_hf_config(cand_48l).eligible
            assert _declined(driver_mod.maybe_create(_make_runner(cand_48l)))
        assert "L=48 (expected 60)" in caplog.text

    cfg_missing_theta = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            k: v
            for k, v in DEFAULT_HF_CONFIG.items()
            if k not in ("rope_theta", "partial_rotary_factor")
        },
    }
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert _declined(driver_mod.maybe_create(_make_runner(cfg_missing_theta)))
    assert "missing HF key 'rope_theta'" in caplog.text
    assert "num_attention_heads" not in caplog.text

    cfg_2048h = {
        **nested_vlm,
        "text_config": {**nested_rope_text, "hidden_size": 2048},
    }
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert _declined(driver_mod.maybe_create(_make_runner(cfg_2048h)))
    assert "H=2048 (expected 4096)" in caplog.text

    dense_with_geom = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": dense_cfg,
    }
    dense_without_geom = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"model_type": "qwen3_5_text"},
    }
    for dense_cand in (
        dense_with_geom,
        dict_to_namespace(dense_with_geom),
        dense_without_geom,
        dict_to_namespace(dense_without_geom),
    ):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert not Qwen35.validate_hf_config(dense_cand).eligible
            assert not qwen_adapter.validate_hf_config(dense_cand).eligible
            assert _declined(driver_mod.maybe_create(_make_runner(dense_cand)))
        assert "Qwen35 supports MoE checkpoints only" in caplog.text

    yarn_cfg = {
        **nested_vlm,
        "text_config": {
            **nested_rope_text,
            "rope_parameters": {
                **nested_rope_text["rope_parameters"],
                "rope_type": "yarn",
                "factor": 4.0,
            },
        },
    }
    linear_cfg = {
        **DEFAULT_HF_CONFIG,
        "rope_scaling": {"rope_type": "linear", "factor": 2.0},
    }
    for scaled_cfg in (
        yarn_cfg,
        dict_to_namespace(yarn_cfg),
        linear_cfg,
        dict_to_namespace(linear_cfg),
    ):
        with pytest.raises(MegakernelConfigError, match="unsupported .* rope_type"):
            Qwen35.derive_geometry(scaled_cfg)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert not Qwen35.validate_hf_config(scaled_cfg).eligible
            assert not qwen_adapter.validate_hf_config(scaled_cfg).eligible
            assert _declined(driver_mod.maybe_create(_make_runner(scaled_cfg)))
        assert "unsupported" in caplog.text and "rope_type" in caplog.text


def test_validate_runner_hf_config_frozen_qwen35_config_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Frozen Qwen3.5 `config.json` passes `validate_hf_config` (skippable)."""
    import json

    from vllm_torchtpu.models.qwen3_5.megakernel import adapter as qwen_adapter

    frozen_cfg = frozen_utils.load_frozen_qwen35_config_json()
    for valid_cfg in (frozen_cfg, frozen_utils.dict_to_namespace(frozen_cfg)):
        assert Qwen35.validate_hf_config(valid_cfg).eligible
        assert qwen_adapter.validate_hf_config(valid_cfg).eligible

    bad_dir = tmp_path / "bad_frozen"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text("{}", encoding="utf-8")
    good_dir = tmp_path / "good_frozen"
    good_dir.mkdir()
    orig_dir = Path(
        frozen_utils.os.environ.get(
            "MK_FROZEN_QWEN35_DIR", frozen_utils.DEFAULT_FROZEN_DIR
        )
    )
    (good_dir / "config.json").write_bytes((orig_dir / "config.json").read_bytes())
    assert (
        json.loads((good_dir / "config.json").read_text(encoding="utf-8")) == frozen_cfg
    )
    monkeypatch.setattr(frozen_utils, "DEFAULT_FROZEN_DIR", str(good_dir))
    assert frozen_utils.load_frozen_qwen35_config_json(bad_dir) == frozen_cfg
    monkeypatch.setattr(frozen_utils, "DEFAULT_FROZEN_DIR", str(bad_dir))
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        frozen_utils.load_frozen_qwen35_config_json(bad_dir)


def test_qwen35_model_def_derived_geometry_spec_and_pool_layout_equivalence():
    """Derived `Qwen35` geometry, `ModelSpec`, and `PoolLayout` match ground truth."""
    nested_vlm_cfg = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": dict(DEFAULT_HF_CONFIG),
    }
    expected_full = qwen35.Qwen35Geometry()
    for cfg in (DEFAULT_HF_CONFIG, nested_vlm_cfg):
        g_derived = Qwen35.derive_geometry(cfg)
        assert type(g_derived) is qwen35.Qwen35Geometry
        assert g_derived == expected_full
        for f in dataclasses.fields(qwen35.Qwen35Geometry):
            assert getattr(g_derived, f.name) == getattr(expected_full, f.name), f.name

    tiny_hf_cfg = {
        **DEFAULT_HF_CONFIG,
        "num_hidden_layers": 4,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "linear_num_key_heads": 8,
        "linear_num_value_heads": 16,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    }
    expected_tiny = qwen35.Qwen35Geometry(
        L=4, H=1024, NH=8, KH=8, VH=16, E=128, TOPK=4, I=1024, IS=1024
    )
    g_tiny_from_hf = Qwen35.derive_geometry(tiny_hf_cfg)
    assert type(g_tiny_from_hf) is qwen35.Qwen35Geometry
    assert g_tiny_from_hf == expected_tiny
    assert expected_tiny == qwen35.TINY
    for f in dataclasses.fields(qwen35.Qwen35Geometry):
        assert getattr(g_tiny_from_hf, f.name) == getattr(expected_tiny, f.name), f.name
        assert getattr(qwen35.TINY, f.name) == getattr(expected_tiny, f.name), f.name

    g_ep_eplb = dataclasses.replace(qwen35.FULL_EP, e_slots=80)
    for r_val in (2, 4, 8):
        for base_g in (qwen35.FULL, qwen35.FULL_EP, g_ep_eplb):
            g_shrunk = base_g.shrink(R=r_val)
            assert type(g_shrunk) is qwen35.Qwen35Geometry
            assert (g_shrunk.L, g_shrunk.att_every) == (8, 4)
            g_shrunk.check(ExecSpec(R=r_val, B=r_val, T=1, ctx=256))

    att_only_tiny = dataclasses.replace(
        qwen35.TINY, att_every=1, KH=0, VH=0, KD=0, VD=0
    )
    att_only_hf_cfg = {**tiny_hf_cfg, "layer_types": ["full_attention"] * 4}
    eplb72 = dataclasses.replace(qwen35.FULL_EP, e_slots=72)
    spec_cases = [
        (qwen35.FULL, DEFAULT_HF_CONFIG, True, 0),
        (qwen35.TINY, tiny_hf_cfg, True, 0),
        (qwen35.FULL_EP, DEFAULT_HF_CONFIG, False, 0),
        (eplb72, DEFAULT_HF_CONFIG, False, 72),
        (att_only_tiny, att_only_hf_cfg, True, 0),
    ]
    for g, cfg, moe_tp_val, e_slots_val in spec_cases:
        expected_layout = _independent_qwen35_pool_layout(g, R=8)
        assert Qwen35.pool_layout(cfg, R=8) == expected_layout
        assert Qwen35.pool_layout(g, R=8) == expected_layout
        assert qwen35_pool.qwen35_pool_layout(g, R=8) == expected_layout

        expected_spec = _independent_qwen35_spec(g, R=8)
        spec_from_cfg = Qwen35.to_model_spec(
            cfg, moe_tp=moe_tp_val, e_slots=e_slots_val
        )
        spec_from_g = qwen35.qwen35_spec(g)
        assert_specs_equal(spec_from_cfg, expected_spec)
        assert_specs_equal(spec_from_g, expected_spec)

    g_att1 = dataclasses.replace(qwen35.TINY, att_every=1)
    assert qwen35_pool.qwen35_pool_layout(
        g_att1, R=8
    ) == _independent_qwen35_pool_layout(g_att1, R=8)
    assert_specs_equal(
        qwen35.qwen35_spec(g_att1), _independent_qwen35_spec(g_att1, R=8)
    )


@pytest.mark.parametrize(
    "t_val,moe_tp,pool_io",
    [
        (1, True, False),
        pytest.param(4, True, False, marks=pytest.mark.slow),
        (1, False, True),
    ],
)
def test_qwen35_model_def_composed_reference_matches_module_loop(
    t_val: int, moe_tp: bool, pool_io: bool
):
    """Composed `Qwen35.reference_step` is bitwise identical to explicit module loop."""
    g = dataclasses.replace(qwen35.TINY, moe_tp=moe_tp)
    ex = ExecSpec(R=2, B=2 * t_val, T=t_val, ctx=128)
    pos, hid, w_l, st_l = mk.random_logical_inputs(
        jax.random.PRNGKey(17 + t_val), g, ex
    )
    out_composed, st_composed = Qwen35.reference_step(
        g, ex, pos, hid, w_l, st_l, pool_io=pool_io
    )

    in_norm_mod = Norm("input_layernorm", geometry=g, zero_centered=True, eps=g.eps)
    post_norm_mod = Norm(
        "post_attention_layernorm", geometry=g, zero_centered=True, eps=g.eps
    )
    final_norm_mod = Norm("model.norm", geometry=g, zero_centered=True, eps=g.eps)
    gqa_mod = GQAModule(geometry=g)
    gdn_mod = GDNModule(geometry=g)
    moe_mod = MoEModule(geometry=g)
    lay_gdn = GDNModule.lin_in_layout(g, ex)
    cr, hr = mk.round_up(3 * ex.B, 8), mk.round_up(3 + ex.T, 8)
    conv_in = st_l["conv"].astype(BF16).astype(F32) if pool_io else st_l["conv"]
    rec_in = st_l["rec"].reshape(
        ex.R, g.NGDN, ex.NS, lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
    )
    step_state: dict[str, Any] = {
        "kc": st_l["kc"],
        "vc": st_l["vc"],
        "conv": conv_in,
        "rec": rec_in,
        "conv_out": (
            jnp.zeros((ex.R, g.NGDN, cr, lay_gdn.QKV), dtype=F32)
            if ex.T == 1
            else jnp.zeros((ex.R, g.NGDN, ex.NS, hr, lay_gdn.QKV), dtype=F32)
        ),
        "rec_out": jnp.zeros(
            (ex.R, g.NGDN, ex.B * lay_gdn.vh, lay_gdn.KD, lay_gdn.VD), dtype=BF16
        ),
    }
    res = hid[: ex.B].astype(F32)
    for lyr in range(g.L):
        x_m, _ = in_norm_mod.reference(w_l["in_norm"][lyr], res, g=g)
        if (lyr + 1) % g.att_every == 0:
            d_m, step_state = gqa_mod.reference(
                w_l,
                x_m,
                step_state,
                g=g,
                ex=ex,
                pos=pos,
                att_idx=lyr // g.att_every,
                layer_idx=lyr,
                paged_kv=False,
            )
        else:
            d_m, step_state = gdn_mod.reference(
                w_l,
                x_m,
                step_state,
                g=g,
                ex=ex,
                gdn_idx=lyr - (lyr + 1) // g.att_every,
                layer_idx=lyr,
                multi_layer=True,
            )
        res = round_bf16(res + d_m)
        x_f, _ = post_norm_mod.reference(w_l["post_norm"][lyr], res, g=g)
        d_f, step_state = moe_mod.reference(
            w_l, x_f, step_state, g=g, ex=ex, layer_idx=lyr
        )
        res = round_bf16(res + d_f)
    out_hand, _ = final_norm_mod.reference(w_l["final_norm"], res, g=g, out_dtype=BF16)
    conv_out = (
        step_state["conv_out"].astype(BF16).astype(F32)
        if pool_io
        else step_state["conv_out"]
    )
    rec_out = step_state["rec_out"]
    st_hand: dict[str, Any] = {"kc": step_state["kc"], "vc": step_state["vc"]}
    if ex.VER:
        conv_pad = (
            jnp.pad(
                conv_in,
                ((0, 0), (0, 0), (0, 0), (0, hr - conv_in.shape[-2]), (0, 0)),
            )
            if conv_in.ndim == 5 and conv_in.shape[-2] < hr
            else conv_in
        )
        st_hand["conv"] = conv_pad
        st_hand["rec"] = rec_in.reshape(
            ex.R, g.NGDN, ex.NS * lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
        )
        st_hand["conv_ckpt"] = conv_out
        st_hand["rec_ckpt"] = rec_out
    else:
        st_hand["conv"] = conv_out
        st_hand["rec"] = rec_out

    assert jnp.array_equal(out_composed, out_hand)
    assert set(st_composed) == set(st_hand)
    for k in st_hand:
        assert jnp.array_equal(st_composed[k], st_hand[k]), k


def test_synthetic_model_def_every2_cpu_kernel_vs_reference(
    monkeypatch: pytest.MonkeyPatch,
):
    """Synthetic `ModelDef` (`Every(n=2)`) HF loaders and CPU kernel vs reference."""
    from vllm_torchtpu.megakernel import model_def, profile

    monkeypatch.setattr(model_def, "_MODEL_REGISTRY", dict(model_def._MODEL_REGISTRY))
    monkeypatch.setattr(model_def, "_ARCH_REGISTRY", dict(model_def._ARCH_REGISTRY))
    monkeypatch.setattr(
        model_def, "_MODEL_TYPE_REGISTRY", dict(model_def._MODEL_TYPE_REGISTRY)
    )
    monkeypatch.setattr(profile, "_REGISTRY", dict(profile._REGISTRY))
    monkeypatch.setattr(profile, "_MODEL_DEFAULTS", dict(profile._MODEL_DEFAULTS))

    synth_hf_cfg = {
        "model_type": "synth_hybrid",
        "architectures": ["SynthHybridForCausalLM"],
        "vocab_size": 64000,
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500_000.0,
        "partial_rotary_factor": 0.25,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 256,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
    }

    @register_model(
        name="synth_hybrid",
        hf_architectures=("SynthHybridForCausalLM",),
        hf_model_type="synth_hybrid",
    )
    class SynthHybrid(ModelDef):
        default_hf_config = synth_hf_cfg
        geometry = FromHF(
            hidden="hidden_size",
            layers="num_hidden_layers",
            vocab="vocab_size",
            eps="rms_norm_eps",
            rope_theta="rope_theta",
            rope_partial=("partial_rotary_factor", 1.0),
        )
        attn = GQA.bind(
            prefix="self_attn",
            heads="num_attention_heads",
            kv_heads="num_key_value_heads",
            head_dim="head_dim",
            qk_norm=True,
            output_gate=True,
        )
        linear = GDN.bind(
            prefix="linear_attn",
            k_heads="linear_num_key_heads",
            v_heads="linear_num_value_heads",
            k_dim="linear_key_head_dim",
            v_dim="linear_value_head_dim",
            conv_kernel="linear_conv_kernel_dim",
        )
        moe = MoE.bind(
            prefix="mlp",
            experts="num_experts",
            top_k="num_experts_per_tok",
            inter="moe_intermediate_size",
            shared_inter="shared_expert_intermediate_size",
            shared_gate=True,
        )
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=linear),
            ffn=moe,
            pre_norm=Norm.bind("attn_norm", zero_centered=True),
            post_norm=Norm.bind("ffn_norm", zero_centered=True),
        )
        embed = Embedding("model.embed_tokens")
        final_norm = Norm.bind("final_ln", zero_centered=True)
        lm_head = LMHead("lm_head", tied_to=None)

    ex = ExecSpec(
        R=2,
        B=2,
        T=1,
        ctx=256,
        interpret=True,
        tiles=dataclasses.replace(TINY_TILES, bank=512 << 10, nbd=6),
    )
    g_tiny = SynthHybrid.shrink(R=ex.R)
    ms = SynthHybrid.to_model_spec(g_tiny, R=ex.R)

    hf_synth = mk.random_hf_weights(jax.random.PRNGKey(42), g_tiny)
    assert {"attn_norm", "ffn_norm", "final_ln"} <= set(hf_synth)
    assert {"in_norm", "post_norm", "final_norm"}.isdisjoint(hf_synth)

    w_l_loaded = mk.load_logical_weights(SynthHybrid, ex, hf_synth, g=g_tiny)
    assert {"attn_norm", "ffn_norm", "final_ln"} <= set(w_l_loaded)
    w_mk_packed = mk.pack_logical_weights(SynthHybrid, ex, w_l_loaded, g=g_tiny)
    for r in range(ex.R):
        w_rank = mk.load_weights(
            SynthHybrid,
            ex,
            mk.MappingWeightSource(w_l_loaded),
            g=g_tiny,
            tp_rank=r,
        )
        assert set(w_rank) == set(w_mk_packed) | {"plan", "w_flat"}
        for k in w_mk_packed:
            if isinstance(w_mk_packed[k], tuple):
                for lyr, (got_l, exp_l) in enumerate(
                    zip(w_rank[k], w_mk_packed[k], strict=True)
                ):
                    np.testing.assert_array_equal(
                        np.asarray(got_l).view(np.uint8),
                        np.asarray(exp_l[r]).view(np.uint8),
                        err_msg=f"SynthHybrid rank {r} layer {lyr} mismatch on {k}",
                    )
            else:
                np.testing.assert_array_equal(
                    np.asarray(w_rank[k]).view(np.uint8),
                    np.asarray(w_mk_packed[k][r]).view(np.uint8),
                    err_msg=f"SynthHybrid rank {r} mismatch on {k}",
                )

    pos, hid_l, _, st_l = mk.random_logical_inputs(jax.random.PRNGKey(42), g_tiny, ex)
    ref_out, ref_st = SynthHybrid.reference_step(
        g_tiny, ex, pos, hid_l, w_l_loaded, st_l
    )

    devs = jax.devices("cpu")[: ex.R]
    mesh = jax.sharding.Mesh(np.asarray(devs), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = mk.pack_inputs(
        pos, hid_l, w_l_loaded, st_l, g_tiny, ex, mesh=mesh
    )
    step_fn, _ = mk.make_step(mesh, ms, ex)
    out_mk, st_out_mk = step_fn(pos_mk, hid_mk, w_mk, st_mk)
    out_mk_unfolded = mk.unfold(out_mk, ex.R)
    for r in range(ex.R):
        out_r = out_mk_unfolded[r, : ex.B].astype(F32)
        ref_f32 = ref_out.astype(F32)
        max_diff = float(jnp.max(jnp.abs(out_r - ref_f32)))
        cos = float(
            jnp.sum(out_r * ref_f32)
            / (jnp.linalg.norm(out_r) * jnp.linalg.norm(ref_f32))
        )
        assert max_diff < 0.05, f"rank {r}: max_diff={max_diff}"
        assert cos > 0.9999, f"rank {r}: cos={cos}"

    kc_mk = mk.unfold(st_out_mk["kc"], ex.R)
    vc_mk = mk.unfold(st_out_mk["vc"], ex.R)
    conv_mk = mk.unfold(st_out_mk["conv"], ex.R)
    rec_mk = mk.unfold(st_out_mk["rec"], ex.R)
    assert jnp.allclose(
        kc_mk.astype(F32), ref_st["kc"].astype(F32), atol=2e-2, rtol=1e-2
    )
    assert jnp.allclose(
        vc_mk.astype(F32), ref_st["vc"].astype(F32), atol=2e-2, rtol=1e-2
    )
    assert jnp.allclose(conv_mk, ref_st["conv"], atol=1e-2, rtol=1e-2)
    assert jnp.allclose(
        rec_mk.astype(F32), ref_st["rec"].astype(F32), atol=5e-2, rtol=2e-2
    )


@pytest.mark.parametrize(
    ("arch", "model_type", "num_experts", "exp_found", "exp_eligible", "exp_reason"),
    [
        ("UnknownArch", "qwen3_5_moe", 512, True, True, "ok"),
        ("UnknownArch", "qwen3_5_moe_text", 512, True, True, "ok"),
        ("UnknownArch", "qwen3_5", 512, True, True, "ok"),
        ("UnknownArch", "qwen3_5", 0, True, False, "non_moe_checkpoint"),
        ("Qwen3_5MoeForConditionalGeneration", "unknown", 512, True, True, "ok"),
        ("Qwen3_5MoeForCausalLM", "unknown", 512, True, True, "ok"),
        ("Qwen3_5ForConditionalGeneration", "unknown", 512, True, True, "ok"),
        (
            "Qwen3_5ForConditionalGeneration",
            "unknown",
            0,
            True,
            False,
            "non_moe_checkpoint",
        ),
        ("Qwen3_5ForCausalLM", "unknown", 512, True, True, "ok"),
        ("Qwen3_5ForCausalLM", "unknown", None, True, False, "non_moe_checkpoint"),
        ("LlamaForCausalLM", "llama", 512, False, False, ""),
        ("Qwen2_5ForCausalLM", "qwen2_5", 512, False, False, ""),
    ],
)
def test_qwen35_seven_routing_keys_and_dense_rejection(
    arch: str,
    model_type: str,
    num_experts: int | None,
    exp_found: bool,
    exp_eligible: bool,
    exp_reason: str,
):
    """Verify all 7 `Qwen35` routing keys resolve to `Qwen35` and reject dense."""
    cfg = dict(Qwen35.default_hf_config)
    cfg["architectures"] = [arch]
    cfg["model_type"] = model_type
    if num_experts is None:
        cfg.pop("num_experts", None)
    else:
        cfg["num_experts"] = num_experts

    found = find_model(cfg)
    if not exp_found:
        assert found is None
        return
    assert found is Qwen35
    el = found.validate_hf_config(cfg)
    assert el.eligible is exp_eligible
    assert el.reason == exp_reason


@pytest.mark.slow
def test_verify_qwen35_model_hf():
    """`mk.verify_model("qwen3_5", hf=True)` passes when `transformers` is installed
    or skips cleanly with a clear reason when unavailable.
    """
    try:
        import torch
        import transformers

        has_tf = hasattr(transformers, "Qwen3_5MoeForCausalLM") and hasattr(
            torch, "Tensor"
        )
    except ImportError:
        has_tf = False

    rep = mk.verify_model("qwen3_5", hf=True, approx_profiles=False, T_values=(1,))
    by_name = {c.name: c for c in rep.checks}
    hf_c = by_name["reference_vs_hf"]
    if has_tf:
        rep.assert_passed(allow_skipped=False)
        assert hf_c.status == "passed", hf_c.reason
        assert hf_c.max_abs_err <= hf_c.tolerance["hf_hidden_atol"]
        assert hf_c.min_cos >= hf_c.tolerance["hf_hidden_min_cos"]
    else:
        assert rep.passed
        assert hf_c.status == "skipped"
        assert "unavailable" in hf_c.reason


def test_qwen35_b5_weight_hashes_and_random_hf_weights():
    """Verify `Qwen35` `load_logical_weights` and `load_hf` match `b5_weight_hashes`."""
    import hashlib
    import json
    from pathlib import Path

    from vllm_torchtpu.megakernel.ops import FP8
    from vllm_torchtpu.megakernel.weights.logical import (
        resolve_spec_and_norms,
        switch_branch_for_layer,
    )

    golden_path = (
        Path(__file__).resolve().parents[3]
        / "megakernel"
        / "golden"
        / "b5_weight_hashes.json"
    )
    if not golden_path.is_file():
        pytest.skip("b5_weight_hashes.json not present")
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
    g_eplb = dataclasses.replace(qwen35_mod.TINY_EP, e_slots=72)
    e_cores_eplb = mk.default_eplb_cores(g_eplb, ex_base.R)
    geom_cases = [
        (
            "TINY",
            qwen35_mod.TINY,
            None,
            ("default", "candidate30", "candidate30-fused"),
        ),
        ("TINY_EP", qwen35_mod.TINY_EP, None, ("default", "candidate30")),
        ("TINY_EP_EPLB72", g_eplb, e_cores_eplb, ("default",)),
    ]

    def _pinned_b5_hf_weights(key: jax.Array, geom: Any) -> dict[str, jax.Array]:
        r_p = max(int(getattr(geom, "NKV", 1)), 1)
        ex_p = ExecSpec(R=r_p, B=r_p, T=1, ctx=2048)
        ms, geom, pre_n, post_n, final_n = resolve_spec_and_norms(Qwen35, ex_p, g=geom)
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

    hf_decl_only = mk.random_hf_weights(jax.random.PRNGKey(0), qwen35_mod.TINY)
    assert "q_proj" in hf_decl_only and "lin_q" in hf_decl_only and "e2" in hf_decl_only

    checked_packed = 0
    for g_name, g_case, e_cores, profiles in geom_cases:
        hf = _pinned_b5_hf_weights(jax.random.PRNGKey(0), g_case)
        w_l = mk.load_logical_weights(Qwen35, ex_base, hf, g=g_case, e_cores=e_cores)
        got_logical = {
            k: _arr_info(v) for k, v in sorted(w_l.items()) if k != "e_cores"
        }
        assert got_logical == golden["logical"][g_name], f"logical mismatch on {g_name}"

        for prof_name in profiles:
            prof = Qwen35.get_profile(prof_name)
            for stack_e in (False, True):
                ex_c = prof.exec_for(
                    dataclasses.replace(ex_base, stack_e=stack_e),
                    B=ex_base.B,
                    T=ex_base.T,
                    g=g_case,
                )
                ex_c = mk.fit_default_tiles(g_case.to_model_spec(), ex_c)
                w_mk = mk.load_hf(Qwen35, ex_c, hf, g=g_case, e_cores=e_cores)
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
