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
"""End-to-end tests for `verify_model`, `auto_model`, and RAW-hazard guardrails."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel.models import qwen35


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


class _SynthHybridModel(mk.ModelDef):
    """Non-Qwen synthetic hybrid `ModelDef` with custom prefixes, renamed norms
    (`attn_norm`, `ffn_norm`, `final_ln`), renamed module params (`attn_q`,
    `gdn_q`, `exp_gate`), and `Every(n=2, last=attn, rest=linear)`.
    """

    name = "synth_hybrid"
    hf_architectures = ("SynthHybridForCausalLM",)
    hf_model_type = ("synth_hybrid",)
    supported_tp_sizes = (2, 8)
    default_hf_config = {
        "architectures": ["SynthHybridForCausalLM"],
        "model_type": "synth_hybrid",
        "n_vocab": 256,
        "n_layers": 4,
        "d_model": 256,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_head": 256,
        "rotary_frac": 0.25,
        "theta": 10000.0,
        "norm_eps": 1e-6,
        "ssm_k_heads": 2,
        "ssm_v_heads": 4,
        "ssm_k_dim": 128,
        "ssm_v_dim": 128,
        "ssm_conv_kernel": 4,
        "n_routed_experts": 128,
        "top_k_experts": 2,
        "d_ff_routed": 256,
        "d_ff_shared": 256,
    }
    geometry = mk.FromHF(
        hidden="d_model",
        layers="n_layers",
        vocab="n_vocab",
        eps="norm_eps",
        rope_theta="theta",
        rope_partial=("rotary_frac", 0.25),
    )
    attn = mk.GQA.bind(
        prefix="custom_attn",
        heads="n_heads",
        kv_heads="n_kv_heads",
        head_dim="d_head",
        qk_norm=True,
        output_gate=True,
        rename={"q_proj": "attn_q"},
    )
    linear = mk.GDN.bind(
        prefix="custom_ssm",
        k_heads="ssm_k_heads",
        v_heads="ssm_v_heads",
        k_dim="ssm_k_dim",
        v_dim="ssm_v_dim",
        conv_kernel="ssm_conv_kernel",
        rename={"lin_q": "gdn_q"},
    )
    moe = mk.MoE.bind(
        prefix="custom_ffn",
        experts="n_routed_experts",
        top_k="top_k_experts",
        inter="d_ff_routed",
        shared_inter="d_ff_shared",
        shared_gate=True,
        rename={"e_gate_proj": "exp_gate"},
    )
    schedule = mk.Schedule(
        mixer=mk.Every(n=2, last=attn, rest=linear),
        ffn=moe,
        pre_norm=mk.Norm.bind(
            "custom_pre_ln", zero_centered=True, param_name="attn_norm"
        ),
        post_norm=mk.Norm.bind(
            "custom_post_ln", zero_centered=True, param_name="ffn_norm"
        ),
    )
    embed = mk.Embedding("tok_emb")
    final_norm = mk.Norm.bind("final_ln_mod", zero_centered=True, param_name="final_ln")
    lm_head = mk.LMHead("out_head")


@pytest.mark.slow
@pytest.mark.parametrize("model_name", mk.list_models())
def test_verify_model_all_registered(model_name: str):
    """`mk.verify_model(model_name, hf=False)` passes all checks across default T."""
    rep = mk.verify_model(model_name, hf=False)
    rep.assert_passed(allow_skipped=False)
    by_name = {c.name: c for c in rep.checks}
    assert set(by_name) == {
        "setup",
        "derived_artifacts",
        "loader_parity",
        "kernel_vs_reference",
        "approx_profiles",
    }
    da = by_name["derived_artifacts"]
    assert da.details["target_chip"] == "tpu_v7x"
    assert da.details["max_vmem_bytes"] <= da.tolerance["vmem_limit_bytes"]
    assert da.details["max_smem_bytes"] <= da.tolerance["smem_limit_bytes"]

    lp = by_name["loader_parity"]
    assert lp.max_ulp_diff == 0 and lp.max_abs_err == 0.0

    kvr = by_name["kernel_vs_reference"]
    metrics = kvr.details["metrics"]
    assert "DirectHBM_T1" in metrics and "PallasWriter_T1" in metrics
    assert "DirectHBM_T4" in metrics and "PallasWriter_T4" in metrics


@pytest.mark.slow
@pytest.mark.parametrize("model_name", mk.list_models())
def test_verify_model_hf(model_name: str):
    """`mk.verify_model(model_name, hf=True)` passes when `transformers` is installed
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

    rep = mk.verify_model(model_name, hf=True, approx_profiles=False, T_values=(1,))
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


@pytest.mark.slow
def test_verify_model_synthetic_second_model_def():
    """Non-Qwen `_SynthHybridModel` passes `auto_model` & `verify_model` without
    code changes outside `models/`, exercising custom prefixes, renamed norms/params,
    and every reference/pool state key.
    """
    resolved = mk.auto_model(_SynthHybridModel, R=2, B=2, T=1, ctx=256)
    assert resolved.model_def is _SynthHybridModel
    assert resolved.pool_layout.num_pools == 2
    assert resolved.pool_layout.slots_per_pool == 1

    def _synth_oracle(weights: Any, inputs: Any) -> tuple[Any, dict[str, Any]]:
        assert weights
        return _SynthHybridModel.reference_step(
            inputs["g"],
            inputs["ex"],
            inputs["pos"],
            inputs["hidden"],
            inputs["w_logical"],
            dict(inputs["states"]),
            paged_kv=False,
        )

    rep = mk.verify_model(
        _SynthHybridModel,
        oracle=_synth_oracle,
        approx_profiles=False,
        T_values=(1, 2),
    )
    rep.assert_passed(allow_skipped=False)
    by_name = {c.name: c for c in rep.checks}
    assert by_name["reference_vs_hf"].status == "passed"
    assert by_name["reference_vs_hf"].max_ulp_diff == 0
    assert "step0_state_rec" in by_name["reference_vs_hf"].details
    lp = by_name["loader_parity"]
    assert lp.details["moe_tp_modes"] == (True, False)
    kvr = by_name["kernel_vs_reference"].details["metrics"]
    for k in ("kc_max_abs", "vc_max_abs", "conv_max_abs", "rec_max_abs"):
        assert k in kvr["DirectHBM_T1"], k
        assert k in kvr["PallasWriter_T1"], k
    for k in (
        "kc_max_abs",
        "vc_max_abs",
        "conv_ckpt_max_abs",
        "rec_ckpt_max_abs",
        "conv_final_max_abs",
        "rec_final_max_abs",
    ):
        assert k in kvr["DirectHBM_T2"], k
    for k in (
        "kc_max_abs",
        "vc_max_abs",
        "conv_max_abs",
        "rec_max_abs",
        "conv_final_max_abs",
        "rec_final_max_abs",
    ):
        assert k in kvr["PallasWriter_T2"], k


@pytest.mark.slow
def test_verify_model_negative_detections(monkeypatch: pytest.MonkeyPatch):
    """Negative tests: `verify_model` catches corrupted loader buffers, corrupted
    kernel/pool state outputs, and zero tolerances (`status == 'failed'`).
    """
    # 0. Setup failure -> setup check fails cleanly
    rep_setup = mk.verify_model(
        _SynthHybridModel,
        hf=False,
        approx_profiles=False,
        profile="nonexistent_profile_xyz",
    )
    assert not rep_setup.passed
    by_setup = {c.name: c for c in rep_setup.checks}
    assert by_setup["setup"].status == "failed"
    assert "nonexistent_profile_xyz" in by_setup["setup"].reason

    # 1. Zero tolerances -> kernel_vs_reference fails
    rep_zero = mk.verify_model(
        _SynthHybridModel,
        hf=False,
        approx_profiles=False,
        T_values=(1,),
        tolerances=mk.ToleranceSpec(hidden_atol=0.0, hidden_rtol=0.0),
    )
    assert not rep_zero.passed
    by_zero = {c.name: c for c in rep_zero.checks}
    assert by_zero["kernel_vs_reference"].status == "failed"
    assert "DirectHBM_T1 hidden abs=" in by_zero["kernel_vs_reference"].reason

    # 2. Corrupted loader buffer -> loader_parity fails
    orig_load_weights = mk.weights.load_weights

    def _corrupt_load_weights(*args: Any, **kwargs: Any) -> dict[str, Any]:
        res = dict(orig_load_weights(*args, **kwargs))
        wf = list(res["w_flat"])
        wf[0] = wf[0] + jnp.ones_like(wf[0])
        res["w_flat"] = tuple(wf)
        return res

    with monkeypatch.context() as m:
        m.setattr(mk.weights, "load_weights", _corrupt_load_weights)
        rep_lp = mk.verify_model(
            _SynthHybridModel, hf=False, approx_profiles=False, T_values=(1,)
        )
        assert not rep_lp.passed
        by_lp = {c.name: c for c in rep_lp.checks}
        assert by_lp["loader_parity"].status == "failed"
        assert "loader mismatch (ulp=" in by_lp["loader_parity"].reason
        assert by_lp["loader_parity"].max_ulp_diff > 0

    # 3. Corrupted DirectHBM `rec` state -> kernel_vs_reference fails naming `rec`
    orig_make_step = mk.kernel.make_step

    def _corrupt_rec_make_step(mesh: Any, spec: Any, ex: Any, **kw: Any) -> Any:
        step_fn, info = orig_make_step(mesh, spec, ex, **kw)
        if not mk.pool.config.cache_exec(ex).paged_kv:

            def _wrapped(pos: Any, hid: Any, w: Any, st: Any, *extra: Any) -> Any:
                out_h, out_st = step_fn(pos, hid, w, st, *extra)
                out_st = dict(out_st, rec=out_st["rec"] + 10.0)
                return out_h, out_st

            return _wrapped, info
        return step_fn, info

    with monkeypatch.context() as m:
        m.setattr(mk.kernel, "make_step", _corrupt_rec_make_step)
        rep_rec = mk.verify_model(
            _SynthHybridModel, hf=False, approx_profiles=False, T_values=(1,)
        )
        assert not rep_rec.passed
        by_rec = {c.name: c for c in rep_rec.checks}
        assert by_rec["kernel_vs_reference"].status == "failed"
        assert "DirectHBM_T1 rec" in by_rec["kernel_vs_reference"].reason

    # 4. Corrupted PallasWriter pool output -> kernel_vs_reference fails
    orig_pooled_step = mk.cache.make_pooled_step

    def _corrupt_pooled_step(mesh: Any, spec: Any, ex: Any, **kw: Any) -> Any:
        p_fn = orig_pooled_step(mesh, spec, ex, **kw)

        def _wrapped(*args: Any) -> Any:
            out_h, raw_out = p_fn(*args)
            raw_list = list(raw_out)
            raw_list[0] = raw_list[0] + jnp.ones_like(raw_list[0]) * 10.0
            return out_h, tuple(raw_list)

        return _wrapped

    with monkeypatch.context() as m:
        m.setattr(mk.cache, "make_pooled_step", _corrupt_pooled_step)
        rep_pw = mk.verify_model(
            _SynthHybridModel, hf=False, approx_profiles=False, T_values=(1,)
        )
        assert not rep_pw.passed
        by_pw = {c.name: c for c in rep_pw.checks}
        assert by_pw["kernel_vs_reference"].status == "failed"
        assert "PallasWriter_T1" in by_pw["kernel_vs_reference"].reason

    # 5. Missing hf_oracle (hf=True on model without hf_oracle) & corrupted oracle
    rep_no_ora = mk.verify_model(
        _SynthHybridModel, hf=True, approx_profiles=False, T_values=(1,)
    )
    assert not rep_no_ora.passed
    no_ora_c = {c.name: c for c in rep_no_ora.checks}["reference_vs_hf"]
    assert no_ora_c.status == "failed"
    assert "defines no callable hf_oracle" in no_ora_c.reason

    def _bad_oracle_state(_w: Any, inputs: Any) -> tuple[Any, dict[str, Any]]:
        out_h, out_st = _SynthHybridModel.reference_step(
            inputs["g"],
            inputs["ex"],
            inputs["pos"],
            inputs["hidden"],
            inputs["w_logical"],
            dict(inputs["states"]),
            paged_kv=False,
        )
        return out_h, dict(out_st, rec=out_st["rec"] + 10.0)

    rep_bad_ora = mk.verify_model(
        _SynthHybridModel,
        oracle=_bad_oracle_state,
        approx_profiles=False,
        T_values=(1,),
    )
    assert not rep_bad_ora.passed
    bad_ora_c = {c.name: c for c in rep_bad_ora.checks}["reference_vs_hf"]
    assert bad_ora_c.status == "failed"
    assert "reference_vs_hf max_abs=" in bad_ora_c.reason

    def _non_mapping_oracle(_w: Any, inputs: Any) -> tuple[Any, Any]:
        out_h, _ = _SynthHybridModel.reference_step(
            inputs["g"],
            inputs["ex"],
            inputs["pos"],
            inputs["hidden"],
            inputs["w_logical"],
            dict(inputs["states"]),
            paged_kv=False,
        )
        return out_h, None

    rep_non_map = mk.verify_model(
        _SynthHybridModel,
        oracle=_non_mapping_oracle,
        approx_profiles=False,
        T_values=(1,),
    )
    assert not rep_non_map.passed
    non_map_c = {c.name: c for c in rep_non_map.checks}["reference_vs_hf"]
    assert non_map_c.status == "failed"
    assert "must be a Mapping" in non_map_c.reason

    def _bad_step_shape_oracle(_w: Any, inputs: Any) -> tuple[Any, dict[str, Any]]:
        out_h, out_st = _SynthHybridModel.reference_step(
            inputs["g"],
            inputs["ex"],
            inputs["pos"],
            inputs["hidden"],
            inputs["w_logical"],
            dict(inputs["states"]),
            paged_kv=False,
        )
        if int(inputs["ex"].T) == 1:
            return out_h[:, :8], out_st
        return out_h, out_st

    rep_bad_shape = mk.verify_model(
        _SynthHybridModel,
        oracle=_bad_step_shape_oracle,
        approx_profiles=False,
        T_values=(1,),
    )
    assert not rep_bad_shape.passed
    bad_shape_c = {c.name: c for c in rep_bad_shape.checks}["reference_vs_hf"]
    assert bad_shape_c.status == "failed"
    assert "hidden shape" in bad_shape_c.reason


def test_auto_model_resolution_and_fail_closed(monkeypatch: pytest.MonkeyPatch):
    """`auto_model` resolves without calling `kernel.build` and fails closed on
    invalid configs, unsupported TP sizes, and VMEM/SMEM overflow.
    """

    def _forbid_build(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("auto_model must never invoke kernel.build()")

    monkeypatch.setattr(mk.kernel, "build", _forbid_build)

    for target in (
        "qwen3_5",
        qwen35.Qwen35,
        qwen35.Qwen35.default_hf_config,
        qwen35.Qwen35.derive_geometry(),
    ):
        res = mk.auto_model(target)
        assert res.model_def is qwen35.Qwen35
        assert res.exec_spec.R == 8
        assert res.pool_layout.num_pools == 15
        assert res.pool_layout.slots_per_pool == 3

    # 1. Unknown model name / unknown HF architecture
    with pytest.raises(mk.MegakernelConfigError, match="Unknown"):
        mk.auto_model("nonexistent_model_xyz")
    with pytest.raises(mk.MegakernelConfigError):
        mk.auto_model({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})

    # 2. Non-MoE checkpoint, geometry mismatch, and missing geometry keys
    with pytest.raises(mk.MegakernelConfigError, match="non_moe_checkpoint"):
        mk.auto_model({"architectures": ["Qwen3_5MoeForCausalLM"]})
    with pytest.raises(mk.MegakernelConfigError, match="geometry_mismatch"):
        mk.auto_model({"architectures": ["Qwen3_5MoeForCausalLM"], "num_experts": 128})
    with pytest.raises(mk.MegakernelConfigError, match="invalid_geometry"):
        mk.auto_model(
            {
                "architectures": ["Qwen3_5MoeForCausalLM"],
                "num_experts": 512,
                "hidden_size": 4096,
            }
        )

    # 3. Non-MoE checkpoint for an MoE schedule
    non_moe = dict(qwen35.Qwen35.default_hf_config, num_experts=0)
    with pytest.raises(mk.MegakernelConfigError, match="non_moe_checkpoint"):
        mk.auto_model(non_moe)

    # 4. Scaled RoPE
    scaled_rope = dict(
        qwen35.Qwen35.default_hf_config,
        rope_scaling={"rope_type": "yarn", "factor": 4.0},
    )
    with pytest.raises(mk.MegakernelConfigError, match="unsupported rope_scaling"):
        mk.auto_model(scaled_rope)

    # 5. Unsupported TP size, B < R * T, missing supported_tp_sizes, mismatched profile
    with pytest.raises(mk.MegakernelConfigError, match="supports TP sizes"):
        mk.auto_model("qwen3_5", R=3)
    with pytest.raises(mk.MegakernelConfigError, match="must be >= R \\* T"):
        mk.auto_model("qwen3_5", R=8, B=4, T=1)
    mismatched_prof = dataclasses.replace(
        qwen35.Qwen35.default_profile(), name="other-prof", model="other_model"
    )
    with pytest.raises(mk.MegakernelConfigError, match="targets model"):
        mk.auto_model("qwen3_5", profile=mismatched_prof)

    class _NoTPModel(_SynthHybridModel):
        name = "synth_no_tp"
        supported_tp_sizes = ()

    with pytest.raises(mk.MegakernelConfigError, match="no supported_tp_sizes"):
        mk.auto_model(_NoTPModel)

    # 6. VMEM / SMEM budget overflow
    base_prof = qwen35.Qwen35.default_profile()
    vmem_overflow_prof = dataclasses.replace(
        base_prof,
        name="test-vmem-overflow",
        tiles=lambda B, *, moe_tp=False, R=8: dataclasses.replace(
            base_prof.tiles_for(B, R=R, moe_tp=moe_tp), nbd=200
        ),
    )
    with pytest.raises(mk.MegakernelConfigError, match="VMEM"):
        mk.auto_model("qwen3_5", profile=vmem_overflow_prof)

    smem_overflow_prof = dataclasses.replace(
        base_prof,
        name="test-smem-overflow",
        tiles=lambda B, *, moe_tp=False, R=8: dataclasses.replace(
            base_prof.tiles_for(B, R=R, moe_tp=moe_tp), smem_mib=0
        ),
    )
    with pytest.raises(mk.MegakernelConfigError, match="SMEM"):
        mk.auto_model("qwen3_5", profile=smem_overflow_prof)


@pytest.mark.slow
@pytest.mark.parametrize("T,fused", [(1, False), (1, True), (2, False), (2, True)])
def test_detect_races_multi_layer_split_and_fused(T: int, fused: bool):
    """Run `L=4` (2+ layers) with `pltpu.InterpretParams(detect_races=True)`."""
    R, B, ctx = 2, 2 * T, 256
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = dataclasses.replace(
        qwen35.Qwen35.shrink(periods=1, R=R),
        H=256,
        I=256,
        IS=256,
        NH=4,
        NKV=2,
        KH=2,
        VH=4,
        moe_tp=False,
    )
    assert g.L == 4
    prof = mk.get_profile("qwen35-candidate30" if fused else "qwen35-baseline")
    race_ip = pltpu.InterpretParams(detect_races=True)
    base = mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        interpret=race_ip,
        cache=mk.CacheExec(paged_kv=True, num_pages=256, pool_gather=False),
    )
    ex = prof.exec_for(base, B=B, T=T, g=g)
    t = dataclasses.replace(
        ex.resolve_tiles(g),
        bank=512 << 10,
        nbd=10,
        nbe=6,
        ebank=256 << 10,
    ).with_modules(
        {"gqa": {"kv_blk": 256, "kv_sub": 0}, "gdn": {"conv_stage": not fused}}
    )
    ex = mk.with_cache(
        dataclasses.replace(ex, tiles=t, interpret=race_ip), pool_gather=False
    )
    spec = qwen35.qwen35_spec(g)
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(42 + T), g, ex)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    pos_mk, hid_mk, w_mk, st_mk = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    pools = tuple(
        jax.device_put(np.asarray(r).copy(), tp)
        for r in mk.states_to_pool(g, ex, st_l, pt, ckpt[..., :T], jnp.maximum(roff, 0))
    )
    if not fused:
        # JAX resets `ipc.races` per interpreted kernel call; verify the
        # megakernel itself in isolation before running the split pipeline.
        mk_step, mk_info = mk.make_step(mesh, spec, ex)
        assert isinstance(mk_info["exec"].interpret, pltpu.InterpretParams)
        assert mk_info["exec"].interpret.detect_races is True
        mk_out, _ = jax.block_until_ready(mk_step(pos_mk, hid_mk, w_mk, st_mk))
        assert np.all(np.isfinite(np.asarray(mk_out, dtype=np.float32)))
        assert ipc.races is not None and not ipc.races.races_found

    step, info = mk.PallasWriter(fused_pool_io=fused).make_step(
        mesh, spec, ex, donate=fused
    )
    assert isinstance(info["exec"].interpret, pltpu.InterpretParams)
    assert info["exec"].interpret.detect_races is True
    out, _ = jax.block_until_ready(step(pos, hid_mk, w_mk, pools, pt, ckpt, roff))
    assert np.all(np.isfinite(np.asarray(out, dtype=np.float32)))
    assert ipc.races is not None and not ipc.races.races_found


@pytest.mark.slow
def test_fused_t1_in_place_read_equals_write_slot():
    """Fused `T=1` with `read_slot == write_slot` (`ckpt_indices = [[[s, s]]]`,
    `roff = 0`) matches split `PallasWriter(fused_pool_io=False)` bitwise.
    """
    R, B, T, ctx = 2, 4, 1, 256
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = dataclasses.replace(
        qwen35.Qwen35.shrink(periods=1, R=R),
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=2,
        KH=2,
        VH=4,
        moe_tp=False,
        e_slots=0,
    )
    spec = qwen35.qwen35_spec(g)
    t = mk.Tiles(
        bank=512 << 10,
        nbd=10,
        nbe=6,
        ebank=256 << 10,
        vmem_mib=32,
    ).with_modules(
        {
            "gqa": {"kv_blk": 256, "nkv": 3, "kv_sub": 0, "amask": 1},
            "gdn": {"conv_stage": False, "gdn_head_major": True},
        }
    )
    ex = mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        stack_e=False,
        interpret=True,
        tiles=t,
        cache=mk.CacheExec(paged_kv=True, num_pages=256, pool_gather=False),
    )

    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(77), g, ex)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt_base, roff_base, _ = mk.default_pool_layout(g, ex)
    # Set read slot == write slot for every sequence (roff = 0)
    ckpt_same = jnp.repeat(ckpt_base[..., :1], 2, axis=-1)
    roff_same = jnp.zeros_like(roff_base)

    def _fresh_pools():
        return tuple(
            jax.device_put(np.asarray(r).copy(), tp)
            for r in mk.states_to_pool(g, ex, st_l, pt, ckpt_base, roff_same)
        )

    step_s, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    step_f, _ = mk.PallasWriter(fused_pool_io=True).make_step(
        mesh, spec, ex, donate=True
    )
    for ck in (ckpt_same, ckpt_base):
        out_s, raw_s = step_s(pos, hid_mk, w_mk, _fresh_pools(), pt, ck, roff_same)
        out_f, raw_f = step_f(pos, hid_mk, w_mk, _fresh_pools(), pt, ck, roff_same)
        np.testing.assert_array_equal(
            np.asarray(out_f).view(np.uint8), np.asarray(out_s).view(np.uint8)
        )
        for r_f, r_s in zip(raw_f, raw_s):
            np.testing.assert_array_equal(
                np.asarray(r_f).view(np.uint8), np.asarray(r_s).view(np.uint8)
            )


def test_aliased_pool_metadata_negative_checks():
    """Deliberately aliased pages/slots and out-of-range IDs are rejected by both
    `pool.check_step_pool_metadata` and `pack_step_meta`, and null-page routing
    protects `num_pages - 1`.
    """
    g = qwen35.Qwen35.shrink(periods=1, R=2)
    play = qwen35.Qwen35.pool_layout(g, R=2)
    ex_t1 = mk.ExecSpec(
        R=2, B=4, T=1, ctx=256, cache=mk.CacheExec(paged_kv=True, num_pages=128)
    )
    pt, ckpt, roff, _ = mk.pool.allocate_test_layout(play, ex_t1)
    pos = jnp.arange(ex_t1.B, dtype=jnp.int32)

    # 1. Duplicate KV write page across active sequences
    pt_dup = pt.at[1, 0].set(pt[0, 0])
    with pytest.raises(ValueError, match="duplicate KV write page"):
        mk.pool.check_step_pool_metadata(
            play, ex_t1, pt_dup, pos, ckpt_indices=ckpt, read_offsets=roff
        )

    # 2. Duplicate GDN checkpoint write slot across active sequences (2-D sub > 0)
    ckpt_dup = ckpt.at[0, 1, 0].set(ckpt[0, 0, 0])
    with pytest.raises(ValueError, match="duplicate checkpoint write slot"):
        mk.pool.check_step_pool_metadata(
            play, ex_t1, pt, pos, ckpt_indices=ckpt_dup, read_offsets=roff
        )
    # 2-D ckpt_indices where slot=8 is in bounds for sub=0 (pages [120, 125))
    # but out of bounds for sub=1 (pages [125, 130) > num_pages=128)
    ckpt_2d_sub1_oob = jnp.array([[5], [6], [7], [8]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="sub=1"):
        mk.pool.check_step_pool_metadata(
            play, ex_t1, pt, pos, ckpt_indices=ckpt_2d_sub1_oob, read_offsets=roff[0]
        )

    # 3. KV write page overlapping an active GDN checkpoint slot page (3-D & 2-D)
    gdn_page = int(ckpt[0, 0, 0]) * play.pages_per_slot + 2
    pt_overlap = pt.at[0, 0].set(gdn_page)
    with pytest.raises(ValueError, match="overlap active KV pages"):
        mk.pool.check_step_pool_metadata(
            play, ex_t1, pt_overlap, pos, ckpt_indices=ckpt, read_offsets=roff
        )
    ckpt_2d_ok = jnp.array([[2], [3], [4], [5]], dtype=jnp.int32)
    gdn_2d_sub1_page = int(play.page_base(2, 1, 2)) + 2
    pt_sub1_overlap = pt.at[0, 0].set(gdn_2d_sub1_page)
    with pytest.raises(ValueError, match="overlap active KV pages"):
        mk.pool.check_step_pool_metadata(
            play,
            ex_t1,
            pt_sub1_overlap,
            pos,
            ckpt_indices=ckpt_2d_ok,
            read_offsets=roff[0],
        )

    # 4. T > 1 with duplicate checkpoint write columns or 1D broadcast
    ex_t2 = mk.ExecSpec(
        R=2, B=4, T=2, ctx=256, cache=mk.CacheExec(paged_kv=True, num_pages=128)
    )
    pt2, ckpt2, roff2, _ = mk.pool.allocate_test_layout(play, ex_t2)
    pos2 = jnp.arange(ex_t2.B, dtype=jnp.int32)
    ckpt2_dup_col = ckpt2.at[:, :, 1].set(ckpt2[:, :, 0])
    with pytest.raises(ValueError, match="distinct positive checkpoint write slots"):
        mk.pool.check_step_pool_metadata(
            play, ex_t2, pt2, pos2, ckpt_indices=ckpt2_dup_col, read_offsets=roff2
        )
    s_raw = jnp.array([1, 2], dtype=jnp.int32)
    r_raw = jnp.zeros((4,), dtype=jnp.int32)
    with pytest.raises(ValueError, match="requires 2-D ckpt_indices"):
        mk.runtime.pack_step_meta(
            jnp.arange(4, dtype=jnp.int32),
            pos2,
            pt2,
            [(s_raw, jnp.ones((2, 1), dtype=jnp.int32), r_raw)],
            bucket_t=4,
            tokens_per_seq=2,
            max_pages_per_seq=int(pt2.shape[1]),
            has_roff=True,
            use_ckpt_2d=True,
            validate=True,
        )
    with pytest.raises(ValueError, match="distinct positive checkpoint"):
        mk.runtime.pack_step_meta(
            jnp.arange(4, dtype=jnp.int32),
            pos2,
            pt2,
            [(s_raw, jnp.array([[3, 3], [4, 5]], dtype=jnp.int32), r_raw)],
            bucket_t=4,
            tokens_per_seq=2,
            max_pages_per_seq=int(pt2.shape[1]),
            has_roff=True,
            use_ckpt_2d=True,
            validate=True,
        )

    # 5. Out-of-range page/slot rejected when check_bounds=True, and null-page 0
    #    routing never touches page num_pages - 1 on inactive sequences
    pt_oob = pt.at[0, 0].set(500)
    with pytest.raises(ValueError, match="outside valid range"):
        mk.pool.check_step_pool_metadata(
            play, ex_t1, pt_oob, pos, ckpt_indices=ckpt, read_offsets=roff
        )
    assert int(mk.pool.safe_kv_page(jnp.int32(-5), 128)) == 0
    assert int(mk.pool.safe_kv_page(jnp.int32(128), 128)) == 0
    assert int(mk.pool.safe_page_base(jnp.int32(-1), play.pages_per_slot, 128)) == 0
    assert int(mk.pool.safe_page_base(jnp.int32(125), play.pages_per_slot, 128)) == 0


def test_planner_raw_hazard_guards():
    """Planner rejects ring depth 1 (`nbd=1`, `nbe=1`, `nkv=1`, `gdn_bufs=1`) and
    schedules where MoE is not active on every layer.
    """
    g = qwen35.Qwen35.shrink(periods=1, R=2)
    spec = qwen35.qwen35_spec(g)
    base_t = mk.Tiles(
        bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=62
    ).with_modules({"gqa": {"kv_blk": 256}})

    for bad_tiles, match in (
        (dataclasses.replace(base_t, nbd=1), "ring depth must be >= 2"),
        (dataclasses.replace(base_t, nbe=1), "ring depth must be >= 2"),
        (base_t.with_modules({"gqa": {"kv_blk": 256, "nkv": 1}}), "nkv >= 2"),
    ):
        ex = mk.ExecSpec(R=2, B=4, T=1, ctx=256, tiles=bad_tiles, interpret=True)
        with pytest.raises(mk.MegakernelConfigError, match=match):
            mk.planner.plan(spec, ex)

    # Wide chunked GDN (`N_GCH > 1`) requires `gdn_bufs >= 2`
    ex_wide = mk.ExecSpec(
        R=2,
        B=24,
        T=4,
        ctx=256,
        tiles=base_t.with_modules({"gdn": {"gdn_bufs": 1}}),
        interpret=True,
    )
    with pytest.raises(mk.MegakernelConfigError, match="gdn_bufs >= 2"):
        mk.planner.plan(spec, ex_wide)

    # MoE inside a SwitchSlot with a non-MoE branch is rejected
    class _DummyNonMoE(mk.Module):
        name = "dummy_non_moe"
        reduction = "allreduce"
        input_norm = "post_norm"

        def weights(self, g, ex):
            return []

        def smalls(self, g, ex):
            return [mk.SmallField("post_norm", g.H)]

        def states(self, g, ex):
            return []

        def outputs(self, g, ex):
            return []

        def plan_params(self, g, ex, plan):
            return None

        def scratch_reqs(self, g, ex, params):
            return []

    bad_ffn_slot = mk.SwitchSlot(
        "ffn",
        branches=(spec.layer_type.slots[1], _DummyNonMoE()),
        branch_counts=(3, 1),
    )
    bad_spec = dataclasses.replace(
        spec,
        layer_type=mk.LayerType(
            "bad_layer", slots=(spec.layer_type.slots[0], bad_ffn_slot)
        ),
    )
    ex_ok = mk.ExecSpec(R=2, B=4, T=1, ctx=256, tiles=base_t, interpret=True)
    with pytest.raises(mk.MegakernelConfigError, match="every layer"):
        mk.planner.plan(bad_spec, ex_ok)


@pytest.mark.slow
@pytest.mark.parametrize("T,invalid_pos", [(1, -1), (1, 500), (2, -1), (2, 500)])
def test_negative_pos_routes_to_null_page_and_preserves_live_pages(
    T: int, invalid_pos: int
):
    """Out-of-range row (`pos < 0` or `pos >= pt_cols * page_size`) routes to
    shared null page 0 and leaves all live KV pages (`pt[0, :]`, including
    `pt[0, -1]`) untouched in both split and fused modes.
    """
    R, B, ctx = 2, 2 * T, 256
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = dataclasses.replace(
        qwen35.Qwen35.shrink(periods=1, R=R),
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=2,
        KH=2,
        VH=4,
        moe_tp=False,
        e_slots=0,
    )
    spec = qwen35.qwen35_spec(g)
    t = mk.Tiles(
        bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
    ).with_modules(
        {
            "gqa": {"kv_blk": 256, "nkv": 3, "kv_sub": 0, "amask": 1},
            "gdn": {"conv_stage": False, "gdn_head_major": True},
        }
    )
    ex = mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        stack_e=False,
        interpret=True,
        tiles=t,
        cache=mk.CacheExec(paged_kv=True, num_pages=256, pool_gather=False),
    )
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(T), g, ex)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    # Set sequence 0 (`rows 0..T-1`) to `invalid_pos` (`-1` or `>= pt_cols * page_size`)
    pos = pos.at[:T].set(invalid_pos)
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    seq0_live_pages = [int(p) for p in np.asarray(pt[0]) if int(p) > 0]
    assert seq0_live_pages

    base_pools = [
        np.asarray(r).copy()
        for r in mk.states_to_pool(g, ex, st_l, pt, ckpt[..., :T], jnp.maximum(roff, 0))
    ]
    step_s, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    step_f, _ = mk.PallasWriter(fused_pool_io=True).make_step(
        mesh, spec, ex, donate=True
    )
    pools_s = tuple(jax.device_put(r.copy(), tp) for r in base_pools)
    pools_f = tuple(jax.device_put(r.copy(), tp) for r in base_pools)
    out_s, raw_s = step_s(pos, hid_mk, w_mk, pools_s, pt, ckpt, roff)
    out_f, raw_f = step_f(pos, hid_mk, w_mk, pools_f, pt, ckpt, roff)
    np.testing.assert_array_equal(
        np.asarray(out_f).view(np.uint8), np.asarray(out_s).view(np.uint8)
    )
    for p_idx, (r_before, r_s, r_f) in enumerate(zip(base_pools, raw_s, raw_f)):
        u_before = r_before.reshape(R, -1, *r_before.shape[1:])
        u_s = np.asarray(r_s).reshape(R, -1, *r_s.shape[1:])
        u_f = np.asarray(r_f).reshape(R, -1, *r_f.shape[1:])
        for pg in seq0_live_pages:
            np.testing.assert_array_equal(
                u_s[:, pg].view(np.uint8),
                u_before[:, pg].view(np.uint8),
                err_msg=f"split modified inactive live page {pg} in pool {p_idx}",
            )
            np.testing.assert_array_equal(
                u_f[:, pg].view(np.uint8),
                u_before[:, pg].view(np.uint8),
                err_msg=f"fused modified inactive live page {pg} in pool {p_idx}",
            )


@pytest.mark.slow
@pytest.mark.parametrize("case", ["intra_seq_boundary", "cross_seq_alias"])
def test_fused_serialized_rmw_aliased_kv_write_page(case: str):
    """Fused serialized-RMW slow path (`kv_mode == 2`, `T=2`): intra-sequence
    page-boundary straddling and cross-sequence page aliasing match split
    `PallasWriter` bitwise with 0 races.
    """
    R, T, B, ctx = 2, 2, 4, 256
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = dataclasses.replace(
        qwen35.Qwen35.shrink(periods=1, R=R),
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=2,
        KH=2,
        VH=4,
        moe_tp=False,
        e_slots=0,
    )
    spec = qwen35.qwen35_spec(g)
    race_ip = pltpu.InterpretParams(detect_races=True)
    t = mk.Tiles(
        bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
    ).with_modules(
        {
            "gqa": {"kv_blk": 256, "nkv": 3, "kv_sub": 0, "amask": 1},
            "gdn": {"conv_stage": False, "gdn_head_major": True},
        }
    )
    ex = mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        stack_e=False,
        interpret=race_ip,
        tiles=t,
        cache=mk.CacheExec(paged_kv=True, num_pages=256, pool_gather=False),
    )
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(T), g, ex)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    # Sequence 0 spans page boundary (127 -> page 0, 128 -> page 1)
    pos = jnp.array([127, 128, 10, 11], dtype=jnp.int32)
    if case == "cross_seq_alias":
        # Sequence 1 (pos 10, 11 -> page 0) aliases Sequence 0's second page
        pt = pt.at[1, 0].set(pt[0, 1])

    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    base_pools = [
        np.asarray(r).copy()
        for r in mk.states_to_pool(g, ex, st_l, pt, ckpt[..., :T], jnp.maximum(roff, 0))
    ]
    step_s, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    step_f, _ = mk.PallasWriter(fused_pool_io=True).make_step(
        mesh, spec, ex, donate=True
    )
    pools_s = tuple(jax.device_put(r.copy(), tp) for r in base_pools)
    pools_f = tuple(jax.device_put(r.copy(), tp) for r in base_pools)
    out_s, raw_s = jax.block_until_ready(
        step_s(pos, hid_mk, w_mk, pools_s, pt, ckpt, roff)
    )
    out_f, raw_f = jax.block_until_ready(
        step_f(pos, hid_mk, w_mk, pools_f, pt, ckpt, roff)
    )
    assert ipc.races is not None and not ipc.races.races_found
    np.testing.assert_array_equal(
        np.asarray(out_f).view(np.uint8), np.asarray(out_s).view(np.uint8)
    )
    play = qwen35.Qwen35.pool_layout(g, R=R)
    for r_f, r_s in zip(raw_f, raw_s):
        uf = np.asarray(r_f).reshape(R, -1, *r_f.shape[1:])
        us = np.asarray(r_s).reshape(R, -1, *r_s.shape[1:])
        np.testing.assert_array_equal(
            uf[:, play.reserved_pages :].view(np.uint8),
            us[:, play.reserved_pages :].view(np.uint8),
        )
    st_f = mk.pool_to_states(g, ex, raw_f, pt, ckpt)
    st_s = mk.pool_to_states(g, ex, raw_s, pt, ckpt)
    for k in ("kc", "vc", "conv"):
        np.testing.assert_array_equal(
            np.asarray(st_f[k]).view(np.uint8), np.asarray(st_s[k]).view(np.uint8)
        )
    np.testing.assert_allclose(
        np.asarray(st_f["rec"], dtype=np.float32),
        np.asarray(st_s["rec"], dtype=np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
