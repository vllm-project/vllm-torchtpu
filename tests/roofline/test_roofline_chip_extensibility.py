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
"""Unit tests for chip-spec extensibility, registry, and hardware capabilities."""

from __future__ import annotations

import pytest
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import load_locked_config
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    FloorOptions,
    Parallel,
    SpecShape,
    StepShape,
    from_hf_config,
    pass_costs,
    spec_cycle_floor,
    step_floor,
)
from vllm_torchtpu.roofline import chips as chips_mod
from vllm_torchtpu.roofline import constants as rf_constants
from vllm_torchtpu.roofline.chips import _build_spec_chip

_SYNTH_HBM_BYTES = 32_000_000_000
_SYNTH_HBM_BW = 1.6e12
_SYNTH_BF16_FLOPS = 5.0e14
_SYNTH_FP8_FLOPS = 1.0e15
_SYNTH_FP4_FLOPS = 2.0e15
_SYNTH_INT4_FLOPS = 1.8e15
_SPEC_K_DRAFT = 3
_PACKED_INT4_WEIGHT_BYTES = 0.515625
_SYNTH_VMEM_BYTES = 67_108_864
_SYNTH_LINK_BW = 50.0e9
_SYNTH_LINKS_PER_CHIP = 4
_SYNTH_CHIPS_PER_HOST_FOUR = 4
_SYNTH_HOP_LATENCY_S = 1.0e-6
_SYNTH_HOST_DCN_GBPS = 200.0
_ONE_POINT_ZERO = 1.0
_SYNTH_TWO_HOSTS = 2
_CUSTOM_D2D_BW = 7.5e11
_CUSTOM_ICI_BW = 3.75e11
_CUSTOM_DCN_BW = 9.0e9
_PP_TWO_STAGES = 2
_CUSTOM_NUM_LANES = 256

_TINY_DENSE_CFG: dict[str, object] = {
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "hidden_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 1024,
    "vocab_size": 32000,
    "dtype": "bfloat16",
}


def _make_base_spec_payload(
    *,
    name: str = "synth_1core_2d",
    cores_per_chip: int = 1,
    chips_per_host: int = _SYNTH_CHIPS_PER_HOST_FOUR,
    fp8_flops: float = _SYNTH_FP8_FLOPS,
) -> dict[str, object]:
    """Build a valid synthetic chip_spec/v1 dictionary."""
    return {
        "schema_version": rf_constants.CHIP_SPEC_SCHEMA,
        "provenance": "public_spec",
        "name": name,
        "cores_per_chip": cores_per_chip,
        "chips_per_host": chips_per_host,
        "hbm_bytes": _SYNTH_HBM_BYTES,
        "hbm_bw": _SYNTH_HBM_BW,
        "bf16_flops": _SYNTH_BF16_FLOPS,
        "fp8_flops": fp8_flops,
        "vmem_bytes": _SYNTH_VMEM_BYTES,
        "ici_link_bw": _SYNTH_LINK_BW,
        "ici_links_per_chip": _SYNTH_LINKS_PER_CHIP,
        "ici_hop_latency_s": _SYNTH_HOP_LATENCY_S,
        "dcn_host_gbit_per_s": _SYNTH_HOST_DCN_GBPS,
        "topologies": [
            {
                "name": "1x1",
                "chip_axes": [1, 1, 1],
                "wrapped_axes": [False, False, False],
            },
            {
                "name": "2x2",
                "chip_axes": [2, 2, 1],
                "wrapped_axes": [False, False, False],
            },
            {
                "name": "8x16",
                "chip_axes": [8, 16, 1],
                "wrapped_axes": [False, True, False],
            },
        ],
    }


def test_load_chip_spec_rejects_missing_fields_and_bad_provenance() -> None:
    """Reject missing core fields, bad provenance, and non-positive vector rate."""
    bad_schema = _make_base_spec_payload()
    bad_schema["schema_version"] = "invalid.schema/v0"
    with pytest.raises(ValueError, match="schema_version"):
        rf.load_chip_spec(bad_schema)

    bad_prov = _make_base_spec_payload()
    bad_prov["provenance"] = "unverified_guess"
    with pytest.raises(ValueError, match="provenance"):
        rf.load_chip_spec(bad_prov)

    for required_field in (
        "hbm_bytes",
        "hbm_bw",
        "bf16_flops",
        "vmem_bytes",
        "cores_per_chip",
        "chips_per_host",
    ):
        missing = _make_base_spec_payload()
        del missing[required_field]
        with pytest.raises(ValueError, match=required_field):
            rf.load_chip_spec(missing)

    zero_vector = _make_base_spec_payload()
    zero_vector["vector_flops"] = 0.0
    with pytest.raises(ValueError, match="vector_flops"):
        rf.load_chip_spec(zero_vector)

    neg_fp8 = _make_base_spec_payload()
    neg_fp8["fp8_flops"] = -_ONE_POINT_ZERO
    with pytest.raises(ValueError, match="fp8_flops"):
        rf.load_chip_spec(neg_fp8)

    bad_bool = _make_base_spec_payload()
    bad_bool["has_sparse_core"] = "yes"
    with pytest.raises(TypeError, match="has_sparse_core"):
        rf.load_chip_spec(bad_bool)


def test_load_chip_spec_min_torus_dim_override() -> None:
    """Verify min_torus_dim propagates into SliceTopology wraparound checks."""
    torus2_spec = _make_base_spec_payload(name="torus_dim2_chip")
    torus2_spec["min_torus_dim"] = 2
    torus2_spec["topologies"] = [{"name": "2x2x2", "chip_axes": [2, 2, 2]}]
    _chip_dim2, topos_dim2 = rf.load_chip_spec(torus2_spec)
    assert topos_dim2["2x2x2"].has_torus_wraparound is True
    assert topos_dim2["2x2x2"].axis_wrapped(0, 2) is True

    default_torus_spec = _make_base_spec_payload(name="torus_dim4_chip")
    default_torus_spec["topologies"] = [{"name": "2x2x2", "chip_axes": [2, 2, 2]}]
    _chip_dim4, topos_dim4 = rf.load_chip_spec(default_torus_spec)
    assert topos_dim4["2x2x2"].has_torus_wraparound is False


def test_register_chip_updates_catalog_and_supported_chips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify register_chip updates CHIPS, CHIP_TOPOLOGIES, and SUPPORTED_CHIPS."""
    chip_name = "synth_registered_chip"
    reg_chip, reg_topos = rf.load_chip_spec(_make_base_spec_payload(name=chip_name))
    # Prime monkeypatch rollback entries before registration so teardown deletes them.
    monkeypatch.setitem(rf.CHIPS, chip_name, reg_chip)
    monkeypatch.delitem(rf.CHIPS, chip_name)
    monkeypatch.setitem(rf.CHIP_TOPOLOGIES, chip_name, dict(reg_topos))
    monkeypatch.delitem(rf.CHIP_TOPOLOGIES, chip_name)

    assert chip_name not in rf.CHIPS
    assert chip_name not in rf.CHIP_TOPOLOGIES
    assert chip_name not in rf.SUPPORTED_CHIPS

    rf.register_chip(reg_chip, reg_topos)
    assert rf.CHIPS[chip_name] is reg_chip
    assert rf.CHIP_TOPOLOGIES[chip_name] == reg_topos
    assert chip_name in rf.SUPPORTED_CHIPS
    assert chip_name in chips_mod.SUPPORTED_CHIPS


def test_emulated_int8_routes_to_bf16_peak() -> None:
    """Route emulated int8 matmuls to bf16 peak rather than fp8 peak."""
    tpu7x = CHIPS["tpu7x"]
    assert tpu7x.int8_flops == 0.0
    assert tpu7x.matmul_peak_kind(
        rf_constants.FP8_BYTES, "a8", weight_dtype="int8"
    ) == (
        "bf16",
        tpu7x.bf16_flops,
        False,
    )
    emu_spec = _make_base_spec_payload(name="emulated_int8_chip")
    emu_chip, _ = rf.load_chip_spec(emu_spec)
    assert emu_chip.int8_flops == 0.0
    assert emu_chip.matmul_peak_kind(
        rf_constants.FP8_BYTES, "a8", weight_dtype="compressed-tensors_w8a8_int8"
    ) == ("bf16", emu_chip.bf16_flops, False)


def test_build_spec_chip_inherits_base_bandwidths_without_overrides() -> None:
    """Verify _build_spec_chip preserves base d2d_bw, ici_bw, and dcn_bw."""
    base_payload = _make_base_spec_payload(name="custom_base_bw")
    base_payload["d2d_bw"] = _CUSTOM_D2D_BW
    base_payload["ici_bw"] = _CUSTOM_ICI_BW
    base_payload["dcn_bw"] = _CUSTOM_DCN_BW
    base_chip, _ = rf.load_chip_spec(base_payload)

    derived_raw: dict[str, object] = {
        "schema_version": rf_constants.CHIP_SPEC_SCHEMA,
        "provenance": "public_spec",
        "name": "derived_inherited_bw",
    }
    derived = _build_spec_chip(derived_raw, base_chip)
    assert derived.d2d_bw == pytest.approx(_CUSTOM_D2D_BW)
    assert derived.ici_bw == pytest.approx(_CUSTOM_ICI_BW)
    assert derived.dcn_bw == pytest.approx(_CUSTOM_DCN_BW)

    recomputed = _build_spec_chip(
        {**derived_raw, "ici_link_bw": _SYNTH_LINK_BW * _SYNTH_TWO_HOSTS},
        base_chip,
    )
    assert recomputed.d2d_bw == pytest.approx(
        rf_constants.DEFAULT_D2D_TO_ICI_LINK_MULT * _SYNTH_LINK_BW * _SYNTH_TWO_HOSTS
    )
    assert recomputed.ici_bw == pytest.approx(
        float(_SYNTH_LINKS_PER_CHIP) * _SYNTH_LINK_BW * _SYNTH_TWO_HOSTS
    )

    links_only = _build_spec_chip(
        {**derived_raw, "ici_links_per_chip": 8},
        base_chip,
    )
    assert links_only.d2d_bw == pytest.approx(_CUSTOM_D2D_BW)
    assert links_only.ici_bw == pytest.approx(8.0 * _SYNTH_LINK_BW)


def test_build_spec_chip_decouples_ici_bw_from_d2d_bw() -> None:
    """Verify _build_spec_chip keeps ici_bw independent of d2d_bw overrides."""
    base_payload = _make_base_spec_payload(name="custom_base_bw_decoupled")
    base_payload["d2d_bw"] = _CUSTOM_D2D_BW
    base_payload["ici_bw"] = _CUSTOM_ICI_BW
    base_chip, _ = rf.load_chip_spec(base_payload)

    d2d_override_chip = _build_spec_chip(
        {
            "schema_version": rf_constants.CHIP_SPEC_SCHEMA,
            "provenance": "public_spec",
            "name": "d2d_override_only",
            "d2d_bw": _CUSTOM_D2D_BW * _SYNTH_TWO_HOSTS,
        },
        base_chip,
    )
    assert d2d_override_chip.d2d_bw == pytest.approx(_CUSTOM_D2D_BW * _SYNTH_TWO_HOSTS)
    assert d2d_override_chip.ici_bw == pytest.approx(_CUSTOM_ICI_BW)

    two_core_chip = _build_spec_chip(
        {
            "schema_version": rf_constants.CHIP_SPEC_SCHEMA,
            "provenance": "public_spec",
            "name": "two_core_ici_recompute",
            "cores_per_chip": _PP_TWO_STAGES,
            "d2d_bw": _CUSTOM_D2D_BW,
            "ici_link_bw": _SYNTH_LINK_BW * _SYNTH_TWO_HOSTS,
        },
        base_chip,
    )
    expected_ici_bw = (
        float(_SYNTH_LINKS_PER_CHIP)
        * (_SYNTH_LINK_BW * _SYNTH_TWO_HOSTS)
        / float(_PP_TWO_STAGES)
    )
    assert two_core_chip.d2d_bw == pytest.approx(_CUSTOM_D2D_BW)
    assert two_core_chip.ici_bw == pytest.approx(expected_ici_bw)


def test_sparse_core_dma_granule_defaults_from_jax_and_spec() -> None:
    """Verify dma_granule_bytes resolves coherently with has_sparse_core."""
    tpu7x_granule = CHIPS["tpu7x"].dma_granule_bytes
    jax_no_sc = chips_mod.from_jax(
        "tpu7x_no_sc",
        pltpu.ChipVersion.TPU_7X,
        rf.ChipOverrides(has_sparse_core=False),
    )
    assert jax_no_sc.has_sparse_core is False
    assert jax_no_sc.dma_granule_bytes == 0

    jax_with_sc = chips_mod.from_jax(
        "tpu6e_with_sc",
        pltpu.ChipVersion.TPU_V6E,
        rf.ChipOverrides(has_sparse_core=True),
    )
    assert jax_with_sc.has_sparse_core is True
    assert jax_with_sc.dma_granule_bytes > 0

    spec_no_sc, _ = rf.load_chip_spec(
        {
            "schema_version": rf_constants.CHIP_SPEC_SCHEMA,
            "provenance": "public_spec",
            "name": "spec_no_sc",
            "base": "tpu7x",
            "has_sparse_core": False,
        }
    )
    assert spec_no_sc.has_sparse_core is False
    assert spec_no_sc.dma_granule_bytes == 0

    spec_reenabled_sc = _build_spec_chip(
        {
            "schema_version": rf_constants.CHIP_SPEC_SCHEMA,
            "provenance": "public_spec",
            "name": "spec_reenabled_sc",
            "has_sparse_core": True,
        },
        spec_no_sc,
    )
    assert spec_reenabled_sc.has_sparse_core is True
    assert spec_reenabled_sc.dma_granule_bytes == tpu7x_granule


def test_num_lanes_defaults_physical_alignment_when_omitted() -> None:
    """Verify physical_alignment_elems defaults to explicit num_lanes."""
    jax_wide = chips_mod.from_jax(
        "wide_lanes_jax",
        pltpu.ChipVersion.TPU_7X,
        rf.ChipOverrides(num_lanes=_CUSTOM_NUM_LANES),
    )
    assert jax_wide.num_lanes == _CUSTOM_NUM_LANES
    assert jax_wide.physical_alignment_elems == _CUSTOM_NUM_LANES

    base_wide, _ = rf.load_chip_spec(
        {
            "schema_version": rf_constants.CHIP_SPEC_SCHEMA,
            "provenance": "public_spec",
            "name": "wide_lanes_base",
            "base": "tpu7x",
            "num_lanes": _CUSTOM_NUM_LANES,
        }
    )
    assert base_wide.num_lanes == _CUSTOM_NUM_LANES
    assert base_wide.physical_alignment_elems == _CUSTOM_NUM_LANES

    standalone_payload = _make_base_spec_payload(name="wide_lanes_standalone")
    standalone_payload["num_lanes"] = _CUSTOM_NUM_LANES
    standalone_wide, _ = rf.load_chip_spec(standalone_payload)
    assert standalone_wide.num_lanes == _CUSTOM_NUM_LANES
    assert standalone_wide.physical_alignment_elems == _CUSTOM_NUM_LANES


def test_load_chip_spec_native_fp4_and_zero_fp8() -> None:
    """Verify native FP4/INT4 capability routing, zero unpack, and zero FP8 guard."""
    geom_moe = from_hf_config(load_locked_config("qwen3_5_397b_nvfp4.json"))
    geom_dense = from_hf_config(_TINY_DENSE_CFG)

    fp4_spec_dict = _make_base_spec_payload(name="native_fp4_chip")
    fp4_spec_dict["fp4_flops"] = _SYNTH_FP4_FLOPS
    fp4_chip, _ = rf.load_chip_spec(fp4_spec_dict)

    emu_spec_dict = _make_base_spec_payload(name="emu_fp4_chip")
    emu_spec_dict["fp4_flops"] = 0.0
    emu_chip, _ = rf.load_chip_spec(emu_spec_dict)

    dtypes_nvfp4 = Dtypes(
        w_dense=rf_constants.NVFP4_BYTES,
        w_dense_dtype="nvfp4",
        w_expert=rf_constants.NVFP4_BYTES,
        w_expert_dtype="nvfp4",
        act="a8",
    )
    parallel_single = Parallel(tp=1, ranks_per_chip=1)
    shape = StepShape(batch=16, rows_per_seq=1, context=256)

    mods_native = pass_costs(
        geom_moe, parallel_single, shape, dtypes_nvfp4, chip=fp4_chip
    )
    mods_emu = pass_costs(geom_moe, parallel_single, shape, dtypes_nvfp4, chip=emu_chip)
    moe_native = next(m for m in mods_native if m.name == "moe")
    moe_emu = next(m for m in mods_emu if m.name == "moe")
    assert moe_native.fp4_flops > 0.0
    assert moe_native.fp8_flops == 0.0
    assert moe_emu.fp4_flops == 0.0
    assert moe_emu.fp8_flops > 0.0
    assert moe_native.vector_flops < moe_emu.vector_flops

    dense_native = pass_costs(
        geom_dense, parallel_single, shape, dtypes_nvfp4, chip=fp4_chip
    )
    dense_emu = pass_costs(
        geom_dense, parallel_single, shape, dtypes_nvfp4, chip=emu_chip
    )
    ffn_native = next(m for m in dense_native if m.name == "mlp")
    ffn_emu = next(m for m in dense_emu if m.name == "mlp")
    assert ffn_native.fp4_flops > 0.0
    assert ffn_native.vector_flops < ffn_emu.vector_flops

    spec_native = spec_cycle_floor(
        geom_moe,
        fp4_chip,
        parallel_single,
        SpecShape(batch=16, context=256, spec_k=_SPEC_K_DRAFT),
        FloorOptions(dtypes=dtypes_nvfp4),
    )
    spec_emu = spec_cycle_floor(
        geom_moe,
        emu_chip,
        parallel_single,
        SpecShape(batch=16, context=256, spec_k=_SPEC_K_DRAFT),
        FloorOptions(dtypes=dtypes_nvfp4),
    )
    spec_moe_native = next(m for m in spec_native.modules if m.name == "moe")
    spec_moe_emu = next(m for m in spec_emu.modules if m.name == "moe")
    assert spec_moe_native.fp4_flops > moe_native.fp4_flops
    assert spec_moe_native.vector_flops < spec_moe_emu.vector_flops
    assert spec_native.matmul_s < spec_emu.matmul_s

    both_spec = _make_base_spec_payload(name="native_fp4_int4_chip")
    both_spec["fp4_flops"] = _SYNTH_FP4_FLOPS
    both_spec["int4_flops"] = _SYNTH_INT4_FLOPS
    both_chip, _ = rf.load_chip_spec(both_spec)
    assert both_chip.matmul_peak_kind(
        rf_constants.FP4_BYTES, "a8", weight_dtype="int4"
    ) == ("int4", _SYNTH_INT4_FLOPS, False)
    assert fp4_chip.matmul_peak_kind(
        rf_constants.FP4_BYTES, "a8", weight_dtype="int4"
    ) == ("fp8", _SYNTH_FP8_FLOPS, True)
    assert both_chip.matmul_peak_kind(
        _PACKED_INT4_WEIGHT_BYTES,
        "a8",
        weight_dtype="compressed-tensors_int4_packed",
    ) == ("int4", _SYNTH_INT4_FLOPS, False)
    assert fp4_chip.matmul_peak_kind(
        _PACKED_INT4_WEIGHT_BYTES,
        "a8",
        weight_dtype="compressed-tensors_int4_packed",
    ) == ("fp8", _SYNTH_FP8_FLOPS, True)

    zero_fp8_spec = _make_base_spec_payload(name="bf16_only_chip", fp8_flops=0.0)
    zero_fp8_chip, _ = rf.load_chip_spec(zero_fp8_spec)
    step_bf16 = step_floor(
        geom_moe,
        zero_fp8_chip,
        parallel_single,
        shape,
        FloorOptions(dtypes=Dtypes(w_dense=1.0, w_dense_dtype="fp8", act="a8")),
    )
    spec_bf16 = spec_cycle_floor(
        geom_moe,
        zero_fp8_chip,
        parallel_single,
        SpecShape(batch=16, context=256, spec_k=_SPEC_K_DRAFT),
        FloorOptions(dtypes=Dtypes(w_dense=1.0, w_dense_dtype="fp8", act="a8")),
    )
    assert step_bf16.matmul_s > 0.0
    assert spec_bf16.matmul_s > step_bf16.matmul_s


def test_unknown_chip_spec_key_rejected() -> None:
    """Verify load_chip_spec rejects misspelled chip-spec and topology keys."""
    bad_top = {
        "schema_version": rf_constants.CHIP_SPEC_SCHEMA,
        "provenance": "public_spec",
        "name": "typo_top",
        "base": "tpu7x",
        "hbm_bandwith": 1.0e12,
    }
    with pytest.raises(ValueError, match="Unknown chip spec"):
        rf.load_chip_spec(bad_top)

    bad_topo = {
        "schema_version": rf_constants.CHIP_SPEC_SCHEMA,
        "provenance": "public_spec",
        "name": "typo_topo",
        "base": "tpu7x",
        "topologies": [
            {
                "name": "2x2x1",
                "chip_axes": [2, 2, 1],
                "wrap_axes": [False, False, False],
            }
        ],
    }
    with pytest.raises(ValueError, match="Unknown topology"):
        rf.load_chip_spec(bad_topo)


def test_override_chip_ici_bw_updates_link_bw_without_coupling_d2d() -> None:
    """Verify override_chip_ici_bw derives ici_link_bw and leaves d2d_bw intact."""
    tpu7x = CHIPS["tpu7x"]
    overridden = rf.override_chip_ici_bw(tpu7x, 450.0e9)
    assert overridden.ici_bw == pytest.approx(450.0e9)
    assert overridden.ici_link_bw == pytest.approx(150.0e9)
    assert overridden.d2d_bw == pytest.approx(tpu7x.d2d_bw)

    roundtrip = rf.override_chip_ici_bw(tpu7x, 300.0e9)
    assert roundtrip.ici_link_bw == pytest.approx(100.0e9)

    for bad_bw in (0.0, -10.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ici_bw_per_core"):
            rf.override_chip_ici_bw(tpu7x, bad_bw)


def test_matmul_peak_kind_routes_a4_activations_to_fp4_or_fp8_never_bf16() -> None:
    """Verify W4A4 activations use native FP4 when available and FP8 otherwise."""
    tpu7x = CHIPS["tpu7x"]
    assert tpu7x.fp4_flops == 0.0
    assert tpu7x.matmul_peak_kind(
        rf_constants.FP4_BYTES, "a4", weight_dtype="nvfp4"
    ) == ("fp8", tpu7x.fp8_flops, True)
    assert tpu7x.matmul_peak_kind(rf_constants.FP8_BYTES, "a4", weight_dtype="fp8") == (
        "fp8",
        tpu7x.fp8_flops,
        False,
    )

    fp4_payload = _make_base_spec_payload(name="native_fp4_chip")
    fp4_payload["fp4_flops"] = 2.0e15
    fp4_chip, _ = rf.load_chip_spec(fp4_payload)
    assert fp4_chip.matmul_peak_kind(
        rf_constants.FP4_BYTES, "a4", weight_dtype="nvfp4"
    ) == ("fp4", 2.0e15, False)


def test_tpu7x_chip_catalog_exposes_provenance_brackets() -> None:
    """Verify tpu7x catalog exposes HBM, FP8, and D2D provenance brackets."""
    cat = CHIPS["tpu7x"].catalog()
    assert cat["hbm_bytes"]["bracket"] == (
        float(CHIPS["tpu7x"].hbm_bytes),
        float(rf_constants.TPU7X_HBM_BYTES_PER_CORE_MAX),
    )
    assert cat["fp8_flops"]["bracket"] == rf_constants.TPU7X_FP8_FLOPS_PER_CORE_BRACKET
    assert cat["d2d_bw"]["bracket"] == rf_constants.D2D_BW_BRACKET
