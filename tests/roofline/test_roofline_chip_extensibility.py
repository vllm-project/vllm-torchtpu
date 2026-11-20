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
"""Unit and end-to-end tests for chip-spec extensibility and capabilities."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import QWEN35_CFG, SMALL_HYBRID_MOE_CFG
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    FloorOptions,
    Parallel,
    SeqKvLayout,
    ServingRequest,
    SliceTopology,
    SolveOptions,
    SpecShape,
    StepShape,
    build_example_scenario,
    build_solve_payload,
    chip_catalog,
    cli,
    from_hf_config,
    hbm_breakdown_per_core_bytes,
    pass_costs,
    plan,
    solve_serving_plan,
    spec_cycle_floor,
    step_floor,
)
from vllm_torchtpu.roofline import chips as chips_mod
from vllm_torchtpu.roofline import constants as rf_constants
from vllm_torchtpu.roofline.chips import (
    TOPOLOGIES,
    ChipSpec,
    _build_spec_chip,
    count_hosts,
)
from vllm_torchtpu.roofline.collectives import collective_cut_bound_s, submesh_for_group
from vllm_torchtpu.roofline.crossover import (
    EpLoadSpec,
    scale_ep_moe_floor,
    scale_pp_stage_floor,
)
from vllm_torchtpu.roofline.memory import smem_max_seqs
from vllm_torchtpu.roofline.partition import partition_pp_layers, pp_stage_breakdowns
from vllm_torchtpu.roofline.report import render_html
from vllm_torchtpu.roofline.step import (
    UnpaddedFloorOptions,
    unpadded_strict_resource_times,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_SYNTH_HBM_BYTES = 32_000_000_000
_SYNTH_HBM_BW = 1.6e12
_SYNTH_BF16_FLOPS = 5.0e14
_SYNTH_FP8_FLOPS = 1.0e15
_SYNTH_FP4_FLOPS = 2.0e15
_SYNTH_INT4_FLOPS = 1.8e15
_SYNTH_INT8_FLOPS = 2.0e15
_SYNTH_VMEM_BYTES = 67_108_864
_SYNTH_LINK_BW = 50.0e9
_SYNTH_LINKS_PER_CHIP = 4
_SYNTH_CHIPS_PER_HOST_FOUR = 4
_SYNTH_CHIPS_PER_HOST_EIGHT = 8
_SYNTH_HOP_LATENCY_S = 1.0e-6
_SYNTH_HOST_DCN_GBPS = 200.0
_WRAPPED_LINK_RATIO = 2.0
_CUT_PAYLOAD_BYTES = 1_000_000.0
_SPEC_K_DRAFT = 3
_ONE_POINT_ZERO = 1.0
_PACKED_INT4_WEIGHT_BYTES = 0.515625
_CHIPS_NINE = 9
_SYNTH_TWO_HOSTS = 2
_CUSTOM_D2D_BW = 7.5e11
_CUSTOM_ICI_BW = 3.75e11
_CUSTOM_DCN_BW = 9.0e9
_UNPADDED_BATCH = 8
_UNPADDED_CTX = 128
_PP_TWO_STAGES = 2
_CUSTOM_PHYSICAL_ALIGN = 1024
_EVEN_EIGHT_LAYERS = 8
_UNEVEN_SEVEN_LAYERS = 7
_UNEVEN_ONE_LAYER = 1
_UNEVEN_FOUR_LAYERS = 4
_EP_FOUR_RANKS = 4
_EP_EXPERTS = 8
_EP_TOP_K = 2
_EP_GLOBAL_TOKENS = 6

_TINY_DENSE_CFG: dict[str, object] = {
    "model_type": "llama",
    "hidden_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 1024,
    "vocab_size": 32000,
    "dtype": "bfloat16",
}

# Frozen TPU7x + QWEN35_CFG baseline parity golden constants captured prior to
# capability-driven ChipSpec refactoring to ensure bit-identical regressions.
_PARITY_CTX = 1024
_PARITY_BATCH = 4
_PARITY_TP = 8
_PARITY_BUDGET = 8
_PARITY_SMEM_CTX = 2048
_PARITY_KV_SPLIT = 2
_PARITY_PAGE_SIZE = 16
_PARITY_SMEM_MAX_SEQS = 2048
_PARITY_SF_HARD_FLOOR_S = 0.0007175140324324324
_PARITY_SF_SERIAL_EST_S = 0.0023642279202027027
_PARITY_SF_HBM_BOUND_S = 0.0007175140324324324
_PARITY_SF_MATMUL_BOUND_S = 3.161753729132317e-05
_PARITY_SF_VECTOR_BOUND_S = 3.098632865800866e-07
_PARITY_SF_ICI_S = 8.426496e-05
_PARITY_SPEC_HARD_FLOOR_S = 0.0009615205016216216
_PARITY_SPEC_SERIAL_EST_S = 0.005899267513023403
_PARITY_SPEC_MATMUL_BOUND_S = 5.1967255349821195e-05
_PARITY_SPEC_VECTOR_BOUND_S = 1.258901167099567e-06
_PARITY_HBM_WEIGHTS = 50894368768.0
_PARITY_HBM_KV = 62914560.0
_PARITY_HBM_STATE = 96030720.0
_PARITY_HBM_TOTAL = 51053314048.0
_PARITY_REC_TOTAL_CHIPS = 8
_PARITY_REC_CONCURRENCY = 256
_PARITY_REC_TTFT_MS = 210.7532734626858
_PARITY_REC_TPOT_MS = 17.393752789120565
_PARITY_REC_TPUT_PER_CHIP = 16364.039390646305
_PARITY_REC_HARD_FLOOR_MS = 2.212448864864865


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


def _write_json(path: Path, payload: object) -> Path:
    """Serialize payload as JSON and return the written path."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _verify_synth_1core_cli_solve_and_emit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    spec_path: Path,
) -> None:
    """Verify solve HTML/JSON and emit CLI paths for the synthetic 1-core spec."""
    tiny_cfg_path = tmp_path / "tiny_dense.json"
    qwen_cfg_path = _write_json(tmp_path / "qwen_dense.json", QWEN35_CFG)
    html_out = tmp_path / "report.html"
    rc_solve = cli.main(
        [
            "solve",
            "--chip-spec",
            str(spec_path),
            "--config",
            str(tiny_cfg_path),
            "--max-chips",
            "4",
            "--example-scenario",
            "chat-8k1k",
            "--html",
            str(html_out),
            "--json",
        ]
    )
    assert rc_solve == 0
    solve_payload = json.loads(capsys.readouterr().out)
    single_host_fit = solve_payload["single_host_fit"]
    assert single_host_fit["feasible"] is True
    assert single_host_fit["topology"] == "2x2"
    assert single_host_fit["chips_per_host"] == _SYNTH_CHIPS_PER_HOST_FOUR
    assert single_host_fit["cores_per_host"] == _SYNTH_CHIPS_PER_HOST_FOUR
    html_text = html_out.read_text(encoding="utf-8")
    assert "synth_1core_2d" in html_text
    assert "2x2x1" not in html_text
    assert "8 TensorCores" not in html_text
    assert "(1 chips, 1 host, 1 cores)" in html_text
    assert "(1 chips, 1 host, 2 cores)" not in html_text

    rc_qwen = cli.main(
        [
            "solve",
            "--chip-spec",
            str(spec_path),
            "--config",
            str(qwen_cfg_path),
            "--max-chips",
            "128",
            "--example-scenario",
            "chat-8k1k",
            "--json",
        ]
    )
    assert rc_qwen == 0
    qwen_payload = json.loads(capsys.readouterr().out)
    qwen_fit = qwen_payload["single_host_fit"]
    assert qwen_fit["feasible"] is False
    assert qwen_fit["best_parallelism"] is None
    assert qwen_fit["topology"] == "2x2"

    rc_emit = cli.main(
        [
            "emit",
            "--chip-spec",
            str(spec_path),
            "--config",
            str(tiny_cfg_path),
            "--max-chips",
            "4",
            "--example-scenario",
            "chat-8k1k",
            "--json",
        ]
    )
    assert rc_emit == 0
    emit_payload = json.loads(capsys.readouterr().out)
    assert emit_payload["commands"]

    rc_disagg = cli.main(
        [
            "emit",
            "--chip-spec",
            str(spec_path),
            "--config",
            str(tiny_cfg_path),
            "--max-chips",
            "2",
            "--mode",
            "disagg",
            "--example-scenario",
            "chat-8k1k",
            "--json",
        ]
    )
    assert rc_disagg == 0
    disagg_emit = json.loads(capsys.readouterr().out)
    assert disagg_emit["mode"] == "disagg"
    assert any("1 cores" in cmd for cmd in disagg_emit["commands"])
    assert all("2 cores" not in cmd for cmd in disagg_emit["commands"])


def _verify_synth_8host_and_torus(
    chip: ChipSpec,
    topos: Mapping[str, SliceTopology],
) -> None:
    """Verify host counting on 8 chips/host and explicit 2D torus wraparound."""
    spec_8host = _make_base_spec_payload(
        name="synth_8host",
        chips_per_host=_SYNTH_CHIPS_PER_HOST_EIGHT,
    )
    chip_8host, _ = rf.load_chip_spec(spec_8host)
    assert count_hosts(_SYNTH_CHIPS_PER_HOST_EIGHT, chip=chip_8host) == 1
    assert count_hosts(_CHIPS_NINE, chip=chip_8host) == _SYNTH_TWO_HOSTS

    cut_unwrapped_x = collective_cut_bound_s(
        "allreduce", _CUT_PAYLOAD_BYTES, (1, 8, 1, 1), chip, topos["8x16"]
    )
    cut_wrapped_y = collective_cut_bound_s(
        "allreduce", _CUT_PAYLOAD_BYTES, (1, 1, 16, 1), chip, topos["8x16"]
    )
    assert cut_unwrapped_x == pytest.approx(_WRAPPED_LINK_RATIO * cut_wrapped_y)

    small_wrapped_topo = SliceTopology(
        name="2x2_wrapped_x",
        chip_axes=(2, 2, 1),
        cores_per_chip=1,
        wrapped_axes=(True, False, False),
    )
    cut_small_wrapped = collective_cut_bound_s(
        "allreduce", _CUT_PAYLOAD_BYTES, (1, 2, 1, 1), chip, small_wrapped_topo
    )
    cut_small_unwrapped = collective_cut_bound_s(
        "allreduce", _CUT_PAYLOAD_BYTES, (1, 1, 2, 1), chip, small_wrapped_topo
    )
    assert cut_small_unwrapped == pytest.approx(_WRAPPED_LINK_RATIO * cut_small_wrapped)


def test_load_chip_spec_1core_2d_torus_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify 1-core 2D torus spec works end-to-end across Python API and CLI."""
    spec_path = _write_json(tmp_path / "synth_1core.json", _make_base_spec_payload())
    tiny_cfg_path = _write_json(tmp_path / "tiny_dense.json", _TINY_DENSE_CFG)

    chip, topos = rf.load_chip_spec(spec_path)
    assert chip.cores_per_chip == 1
    assert chip.chips_per_host == _SYNTH_CHIPS_PER_HOST_FOUR
    assert chip.ici_links_per_chip == _SYNTH_LINKS_PER_CHIP
    expected_d2d = float(_SYNTH_LINKS_PER_CHIP) * _SYNTH_LINK_BW
    assert chip.d2d_bw == pytest.approx(expected_d2d)
    assert chip.ici_bw == pytest.approx(expected_d2d)
    assert set(topos) == {"1x1", "2x2", "8x16"}
    assert topos["1x1"].num_chips == 1
    assert topos["8x16"].wrapped_axes == (False, True, False)

    geom_tiny = from_hf_config(_TINY_DENSE_CFG)
    scen = build_example_scenario("chat-8k1k", slo_tier="none")
    one_chip_plan = solve_serving_plan(
        geom_tiny,
        Dtypes(),
        chip,
        scen,
        SolveOptions(max_chips=1, topologies=tuple(topos.values())),
    )
    assert one_chip_plan.recommended is not None
    assert one_chip_plan.recommended.total_chips == 1

    rows = plan(
        geom_tiny,
        chip,
        Parallel(tp=2),
        Dtypes(),
        ServingRequest(context=128, concurrencies=(1,)),
    )
    assert len(rows) == 1
    assert rows[0].hard_floor_ms > 0.0
    assert rows[0].output_tput_ceil_per_chip == pytest.approx(
        rows[0].interactivity_ceil_tok_s / 2.0
    )
    step_tp2 = step_floor(
        geom_tiny,
        chip,
        Parallel(tp=2),
        StepShape(batch=1, rows_per_seq=1, context=128),
    )
    assert step_tp2.ici_s > 0.0
    assert sum(mod.ici_bytes for mod in step_tp2.modules) > 0.0

    rc_plan = cli.main(
        [
            "plan",
            "--chip-spec",
            str(spec_path),
            "--config",
            str(tiny_cfg_path),
            "--tp",
            "2",
            "--context",
            "128",
            "--concurrency",
            "1",
            "--json",
        ]
    )
    assert rc_plan == 0
    assert len(json.loads(capsys.readouterr().out)) == 1

    rc_explain = cli.main(
        [
            "explain",
            "--chip-spec",
            str(spec_path),
            "--config",
            str(tiny_cfg_path),
            "--tp",
            "2",
            "--context",
            "128",
            "--batch",
            "1",
            "--json",
        ]
    )
    assert rc_explain == 0
    explain_mods = json.loads(capsys.readouterr().out)
    assert sum(float(m["ici_bytes"]) for m in explain_mods) > 0.0

    one_chip_html = render_html(
        build_solve_payload(
            one_chip_plan,
            chip=chip,
            dtypes=Dtypes(),
            model="tiny_llama",
            topologies=topos,
        )
    )
    assert "1 cores" in one_chip_html
    assert "2 cores" not in one_chip_html
    _verify_synth_1core_cli_solve_and_emit(tmp_path, capsys, spec_path)
    _verify_synth_8host_and_torus(chip, topos)


def test_load_chip_spec_native_fp4_and_zero_fp8() -> None:
    """Verify native FP4/INT4 capability routing, zero unpack, and zero FP8 guard."""
    geom_moe = from_hf_config(SMALL_HYBRID_MOE_CFG)
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
        SpecShape(batch=16, context=256, k=_SPEC_K_DRAFT),
        FloorOptions(dt=dtypes_nvfp4),
    )
    spec_emu = spec_cycle_floor(
        geom_moe,
        emu_chip,
        parallel_single,
        SpecShape(batch=16, context=256, k=_SPEC_K_DRAFT),
        FloorOptions(dt=dtypes_nvfp4),
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
        FloorOptions(dt=Dtypes(w_dense=1.0, w_dense_dtype="fp8", act="a8")),
    )
    spec_bf16 = spec_cycle_floor(
        geom_moe,
        zero_fp8_chip,
        parallel_single,
        SpecShape(batch=16, context=256, k=_SPEC_K_DRAFT),
        FloorOptions(dt=Dtypes(w_dense=1.0, w_dense_dtype="fp8", act="a8")),
    )
    assert step_bf16.matmul_s > 0.0
    assert spec_bf16.matmul_s > step_bf16.matmul_s


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


def test_tpu7x_parity_unchanged() -> None:
    """Verify TPU7x calculations remain bit-identical after extensibility refactor."""
    chip = CHIPS["tpu7x"]
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    expected_smem_budget = (
        int(tpu_info.smem_capacity_bytes) // rf_constants.SMEM_PAGE_TABLE_DIVISOR
    )
    assert chip.smem_page_table_bytes == expected_smem_budget
    assert (
        smem_max_seqs(_PARITY_SMEM_CTX, _PARITY_KV_SPLIT, _PARITY_PAGE_SIZE)
        == _PARITY_SMEM_MAX_SEQS
    )
    assert chip_catalog(chip)["page_table_byte_budget"]["value"] == (
        chip.smem_page_table_bytes
    )

    geom = from_hf_config(QWEN35_CFG)
    dtypes = Dtypes()
    parallel = Parallel(tp=_PARITY_TP)
    step_result = step_floor(
        geom,
        chip,
        parallel,
        StepShape(batch=_PARITY_BATCH, rows_per_seq=1, context=_PARITY_CTX),
        FloorOptions(dt=dtypes),
    )
    assert step_result.hard_floor_s == _PARITY_SF_HARD_FLOOR_S
    assert step_result.serial_estimate_s == _PARITY_SF_SERIAL_EST_S
    assert step_result.hbm_bound_s == _PARITY_SF_HBM_BOUND_S
    assert step_result.matmul_bound_s == _PARITY_SF_MATMUL_BOUND_S
    assert step_result.vector_bound_s == _PARITY_SF_VECTOR_BOUND_S
    assert step_result.ici_s == _PARITY_SF_ICI_S

    spec_result = spec_cycle_floor(
        geom,
        chip,
        parallel,
        SpecShape(batch=_PARITY_BATCH, context=_PARITY_CTX, k=_SPEC_K_DRAFT),
        FloorOptions(dt=dtypes),
    )
    assert spec_result.hard_floor_s == _PARITY_SPEC_HARD_FLOOR_S
    assert spec_result.serial_estimate_s == _PARITY_SPEC_SERIAL_EST_S
    assert spec_result.matmul_bound_s == _PARITY_SPEC_MATMUL_BOUND_S
    assert spec_result.vector_bound_s == _PARITY_SPEC_VECTOR_BOUND_S

    hbm_breakdown = hbm_breakdown_per_core_bytes(
        geom, dtypes, parallel, SeqKvLayout(_PARITY_BATCH, _PARITY_CTX)
    )
    assert (
        hbm_breakdown.weights,
        hbm_breakdown.kv,
        hbm_breakdown.state,
        hbm_breakdown.total,
    ) == (
        _PARITY_HBM_WEIGHTS,
        _PARITY_HBM_KV,
        _PARITY_HBM_STATE,
        _PARITY_HBM_TOTAL,
    )

    scen = build_example_scenario("chat-8k1k", slo_tier="interactive")
    splan = solve_serving_plan(
        geom, dtypes, chip, scen, SolveOptions(max_chips=_PARITY_BUDGET)
    )
    rec = splan.recommended
    assert rec is not None
    assert rec.total_chips == _PARITY_REC_TOTAL_CHIPS
    assert rec.concurrency == _PARITY_REC_CONCURRENCY
    assert math.isclose(rec.est_ttft_ms, _PARITY_REC_TTFT_MS, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(rec.est_tpot_ms, _PARITY_REC_TPOT_MS, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(
        rec.token_throughput_per_chip,
        _PARITY_REC_TPUT_PER_CHIP,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    assert math.isclose(
        rec.decode_floor.hard_floor_ms,
        _PARITY_REC_HARD_FLOOR_MS,
        rel_tol=0.0,
        abs_tol=0.0,
    )


def test_cli_plan_uses_int8_flops_from_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI plan preserves config int8 weight_dtype so int8_flops is used."""
    int8_cfg = dict(_TINY_DENSE_CFG)
    int8_cfg["dtype"] = "int8"
    cfg_path = _write_json(tmp_path / "tiny_int8.json", int8_cfg)

    int8_spec = _make_base_spec_payload(name="native_int8_chip")
    int8_spec["int8_flops"] = _SYNTH_INT8_FLOPS
    int8_spec_path = _write_json(tmp_path / "int8_chip.json", int8_spec)

    fp8_only_spec = _make_base_spec_payload(name="fp8_fallback_chip")
    fp8_only_spec["int8_flops"] = 0.0
    fp8_spec_path = _write_json(tmp_path / "fp8_chip.json", fp8_only_spec)

    rc_int8 = cli.main(
        [
            "plan",
            "--chip-spec",
            str(int8_spec_path),
            "--config",
            str(cfg_path),
            "--tp",
            "1",
            "--context",
            "1",
            "--concurrency",
            "2048",
            "--json",
        ]
    )
    assert rc_int8 == 0
    rows_int8 = json.loads(capsys.readouterr().out)

    rc_fp8 = cli.main(
        [
            "plan",
            "--chip-spec",
            str(fp8_spec_path),
            "--config",
            str(cfg_path),
            "--tp",
            "1",
            "--context",
            "1",
            "--concurrency",
            "2048",
            "--json",
        ]
    )
    assert rc_fp8 == 0
    rows_fp8 = json.loads(capsys.readouterr().out)

    assert rows_int8[0]["binding"] == "matmul"
    assert rows_int8[0]["hard_floor_ms"] < rows_fp8[0]["hard_floor_ms"]


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
        float(_SYNTH_LINKS_PER_CHIP) * _SYNTH_LINK_BW * _SYNTH_TWO_HOSTS
    )
    assert recomputed.ici_bw == pytest.approx(recomputed.d2d_bw)


def test_unpadded_strict_and_crossover_preserve_five_matmul_buckets() -> None:
    """Verify unpadded strict times and crossover scaling keep fp4/int8/int4 work."""
    fp4_payload = _make_base_spec_payload(name="fp4_bucket_chip")
    fp4_payload["fp4_flops"] = _SYNTH_FP4_FLOPS
    fp4_chip, topos = rf.load_chip_spec(fp4_payload)
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt_fp4 = Dtypes(
        w_dense=rf_constants.FP4_BYTES,
        w_expert=rf_constants.FP4_BYTES,
        w_dense_dtype="nvfp4",
        w_expert_dtype="nvfp4",
    )
    par = Parallel(tp=1, moe="ep", ep=_EP_FOUR_RANKS)
    shape = StepShape(batch=_UNPADDED_BATCH, rows_per_seq=1, context=_UNPADDED_CTX)
    topo = topos["2x2"]

    _hbm_s, matmul_fp4_s, _vector_s, _coll_s = unpadded_strict_resource_times(
        geom,
        fp4_chip,
        par,
        shape,
        UnpaddedFloorOptions(dt=dt_fp4, topology=topo),
    )
    no_fp4_chip, _ = rf.load_chip_spec(_make_base_spec_payload(name="bf16_fallback"))
    _hbm_fb, matmul_fb_s, _vector_fb, _coll_fb = unpadded_strict_resource_times(
        geom,
        no_fp4_chip,
        par,
        shape,
        UnpaddedFloorOptions(dt=dt_fp4, topology=topo),
    )
    assert matmul_fp4_s > 0.0
    assert matmul_fp4_s < matmul_fb_s

    sf_even = step_floor(
        geom,
        fp4_chip,
        par,
        shape,
        FloorOptions(dt=dt_fp4, topology=topo),
    )
    sf_ep = scale_ep_moe_floor(
        sf_even,
        fp4_chip,
        EpLoadSpec(
            geom=geom,
            par=par,
            global_tokens=_EP_GLOBAL_TOKENS,
        ),
        topology=topo,
    )
    moe_even = next(m for m in sf_even.modules if m.name == "moe")
    moe_ep = next(m for m in sf_ep.modules if m.name == "moe")
    assert moe_even.fp4_flops > 0.0
    assert moe_ep.fp4_flops > moe_even.fp4_flops
    assert moe_ep.fp4_flops_min == pytest.approx(moe_even.strict_fp4_flops)

    pp_par = Parallel(tp=1, pp=_PP_TWO_STAGES, moe="tp_slice")
    geom_eight = from_hf_config(
        {
            **SMALL_HYBRID_MOE_CFG,
            "num_hidden_layers": _EVEN_EIGHT_LAYERS,
            "layer_types": ["full_attention"] * _EVEN_EIGHT_LAYERS,
        }
    )
    sf_pp = step_floor(
        geom_eight,
        fp4_chip,
        pp_par,
        shape,
        FloorOptions(dt=dt_fp4, topology=topo),
    )
    sf_uneven = scale_pp_stage_floor(
        sf_pp,
        (_UNEVEN_SEVEN_LAYERS, _UNEVEN_ONE_LAYER),
        fp4_chip,
        pp_par,
        topo,
    )
    moe_pp = next(m for m in sf_pp.modules if m.name == "moe")
    moe_uneven = next(m for m in sf_uneven.modules if m.name == "moe")
    ratio = float(_UNEVEN_SEVEN_LAYERS) / float(_UNEVEN_FOUR_LAYERS)
    assert moe_uneven.fp4_flops == pytest.approx(moe_pp.fp4_flops * ratio, rel=1e-9)
    assert moe_uneven.fp4_flops_min == pytest.approx(
        moe_pp.strict_fp4_flops * ratio, rel=1e-9
    )


def test_cli_explain_text_and_reference_layout_use_resolved_chip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify explain text shows all GFLOP columns and solve handles custom ref chip."""
    cfg_path = _write_json(tmp_path / "tiny_dense.json", _TINY_DENSE_CFG)
    spec_payload = _make_base_spec_payload(name="custom_ref_chip")
    spec_payload["int8_flops"] = _SYNTH_INT8_FLOPS
    spec_path = _write_json(tmp_path / "custom_chip.json", spec_payload)

    rc_explain = cli.main(
        [
            "explain",
            "--chip-spec",
            str(spec_path),
            "--config",
            str(cfg_path),
            "--tp",
            "1",
        ]
    )
    assert rc_explain == 0
    explain_out = capsys.readouterr().out
    for col in ("int8_gflops", "fp4_gflops", "int4_gflops", "vector_gflops"):
        assert col in explain_out

    rc_solve = cli.main(
        [
            "solve",
            "--chip-spec",
            str(spec_path),
            "--config",
            str(cfg_path),
            "--max-chips",
            str(_SYNTH_CHIPS_PER_HOST_FOUR),
            "--example-scenario",
            "chat-8k1k",
            "--reference-layout",
            "TP1",
            "--json",
        ]
    )
    assert rc_solve == 0
    solve_out = json.loads(capsys.readouterr().out)
    assert solve_out["chip"] == "custom_ref_chip"
    assert solve_out["reference"] is not None
    assert solve_out["reference"]["found"] is True


def test_partition_pp_layers_uses_custom_chip_alignment() -> None:
    """Verify partition_pp_layers threads custom chip alignment into stage HBM."""
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = Dtypes()
    par = Parallel(tp=1, pp=_PP_TWO_STAGES, moe="tp_slice")
    base_payload = _make_base_spec_payload(name="wide_align_chip")
    base_payload["physical_alignment_elems"] = _CUSTOM_PHYSICAL_ALIGN
    wide_chip, _ = rf.load_chip_spec(base_payload)

    _part_base, bd_base, _, _ = pp_stage_breakdowns(
        geom,
        dt,
        par,
        (_UNPADDED_CTX, _UNPADDED_BATCH),
        chip=CHIPS["tpu7x"],
    )
    part_wide, bd_wide, _, ok_wide = pp_stage_breakdowns(
        geom,
        dt,
        par,
        (_UNPADDED_CTX, _UNPADDED_BATCH),
        chip=wide_chip,
    )
    assert ok_wide is True
    assert len(part_wide) == _PP_TWO_STAGES
    assert bd_wide.kv > bd_base.kv
    assert (
        partition_pp_layers(
            geom,
            dt,
            par,
            (_UNPADDED_CTX, _UNPADDED_BATCH),
            chip=wide_chip,
        )
        == part_wide
    )


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


def test_plan_cli_uses_custom_chip_topologies_for_interchip_ici(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify plan --chip-spec selects 1-core ICI (1, 2, 1, 1) over TPU7x D2D."""
    slow_ici_link_bw = 1.0e9
    spec_payload = _make_base_spec_payload(name="slow_ici_1core_chip")
    spec_payload["d2d_bw"] = _CUSTOM_D2D_BW
    spec_payload["ici_link_bw"] = slow_ici_link_bw
    spec_payload["ici_bw"] = float(_SYNTH_LINKS_PER_CHIP) * slow_ici_link_bw
    chip_one_core, topos_one_core = rf.load_chip_spec(spec_payload)

    assert submesh_for_group(_PP_TWO_STAGES, topos_one_core["2x2"], 1) == (
        1,
        _PP_TWO_STAGES,
        1,
        1,
    )
    assert submesh_for_group(_PP_TWO_STAGES, TOPOLOGIES["2x2x1"], 1) == (
        _PP_TWO_STAGES,
        1,
        1,
        1,
    )

    cfg_path = _write_json(tmp_path / "tiny_dense.json", _TINY_DENSE_CFG)
    spec_path = _write_json(tmp_path / "slow_ici_1core.json", spec_payload)
    rc_plan = cli.main(
        [
            "plan",
            "--chip-spec",
            str(spec_path),
            "--config",
            str(cfg_path),
            "--tp",
            str(_PP_TWO_STAGES),
            "--concurrency",
            "1",
            "--context",
            str(_UNPADDED_CTX),
            "--json",
        ]
    )
    assert rc_plan == 0
    cli_rows = json.loads(capsys.readouterr().out)

    geom = from_hf_config(_TINY_DENSE_CFG)
    request = ServingRequest(context=_UNPADDED_CTX, concurrencies=[1])
    parallel = Parallel(tp=_PP_TWO_STAGES, moe="tp_slice")
    dt = Dtypes()
    expected_rows = plan(
        geom=geom,
        chip=chip_one_core,
        parallel=parallel,
        dt=dt,
        request=request,
        topologies=topos_one_core,
    )
    selected_2x2_rows = plan(
        geom=geom,
        chip=chip_one_core,
        parallel=parallel,
        dt=dt,
        request=request,
        topologies={"2x2": topos_one_core["2x2"]},
    )
    tpu7x_topo_rows = plan(
        geom=geom,
        chip=chip_one_core,
        parallel=parallel,
        dt=dt,
        request=request,
        topologies=TOPOLOGIES,
    )
    assert cli_rows[0]["hard_floor_ms"] == pytest.approx(expected_rows[0].hard_floor_ms)
    assert expected_rows[0].hard_floor_ms == pytest.approx(
        selected_2x2_rows[0].hard_floor_ms
    )
    assert cli_rows[0]["hard_floor_ms"] > tpu7x_topo_rows[0].hard_floor_ms


_CUSTOM_NUM_LANES = 256
_CUSTOM_SMEM_BYTES = 262_144


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
        "tpu5e_with_sc",
        pltpu.ChipVersion.TPU_V5E,
        rf.ChipOverrides(has_sparse_core=True),
    )
    assert jax_with_sc.has_sparse_core is True
    assert jax_with_sc.dma_granule_bytes == tpu7x_granule

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


def test_assumptions_and_solve_payload_use_custom_chip_smem_budget(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify custom smem_page_table_bytes matches across catalog and assumptions."""
    spec_payload = _make_base_spec_payload(name="custom_smem_chip")
    spec_payload["smem_page_table_bytes"] = _CUSTOM_SMEM_BYTES
    spec_path = _write_json(tmp_path / "custom_smem_chip.json", spec_payload)
    rc = cli.main(["--chip-spec", str(spec_path), "--assumptions", "--json"])
    assert rc == 0
    assump_json = json.loads(capsys.readouterr().out)
    assert (
        assump_json["catalog"]["page_table_byte_budget"]["value"] == _CUSTOM_SMEM_BYTES
    )
    cli_smem = next(
        entry
        for entry in assump_json["solver_assumptions"]
        if entry["name"] == "smem_page_table_budget"
    )
    assert cli_smem["value"] == _CUSTOM_SMEM_BYTES

    custom_chip, custom_topos = rf.load_chip_spec(spec_payload)
    geom = from_hf_config(_TINY_DENSE_CFG)
    scen = build_example_scenario("chat-8k1k", slo_tier="none")
    splan = solve_serving_plan(
        geom,
        Dtypes(),
        custom_chip,
        scen,
        SolveOptions(
            max_chips=_SYNTH_CHIPS_PER_HOST_FOUR,
            topologies=tuple(custom_topos.values()),
        ),
    )
    payload = build_solve_payload(
        splan,
        chip=custom_chip,
        topologies=custom_topos,
    )
    emit_smem = next(
        entry
        for entry in payload["solver_assumptions"]
        if entry["name"] == "smem_page_table_budget"
    )
    assert emit_smem["value"] == _CUSTOM_SMEM_BYTES
