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

"""Unit tests for megakernel advisor regime sweeps, reports, and CLI."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.abc
import importlib.machinery
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

import pytest

try:
    from jax.experimental.pallas import tpu as pltpu
except (ImportError, RuntimeError):
    pltpu = None

from tests.megakernel.test_advisor_bounds import (
    MIXTRAL_8X7B_CFG,
    QWEN35_08B_CFG,
)
from tests.megakernel.test_advisor_specs import KIMI_K3_PRO_CFG, QWEN35_397B_CFG
from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import chips, constants
from vllm_torchtpu.megakernel.advisor.chips import resolve_chip_spec
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import Tiles

KERNEL_NAME_DENYLIST = re.compile(
    r"gmm|ragged_dot|megablox|sparsecore|mxu|tensorcore|vpu|\bllo\b", re.IGNORECASE
)


class _SvgTextCollector(HTMLParser):
    """Collect the viewBox and every <text> element's attributes and content."""

    def __init__(self) -> None:
        """Start with no viewBox, no collected text, and no open element."""
        super().__init__()
        self.view_box = ""
        self.texts: list[tuple[dict[str, str], str]] = []
        self._open: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record the svg viewBox and open a text element."""
        attr = {k: v or "" for k, v in attrs}
        if tag == "svg":
            self.view_box = attr["viewbox"]
        elif tag == "text":
            self._open = attr

    def handle_data(self, data: str) -> None:
        """Attach character data to the open text element."""
        if self._open is not None:
            self.texts.append((self._open, data))
            self._open = None


class _JaxBlocker(importlib.abc.MetaPathFinder):
    """MetaPathFinder that blocks importing `jax` or any `jax.*` submodule."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Raise ModuleNotFoundError when `jax` or `jax.*` is requested."""
        _ = (path, target)
        if fullname == "jax" or fullname.startswith("jax."):
            msg = "No module named 'jax'"
            raise ModuleNotFoundError(msg)
        return None


def _regime(
    model: str, prefix: str, options: advisor.RegimeSweepOptions | None = None
) -> advisor.RegimeEntry:
    """Return the regime row of `model` on tpu7x whose label starts with `prefix`."""
    sweep = advisor.advise_regimes(model, "tpu7x", options=options)
    return next(r for r in sweep.regimes if r.label.startswith(prefix))


@pytest.fixture
def without_declared_megakernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip deployment `megakernel` blocks to test core-limit mechanics."""
    for model_key, preset in list(advisor.DEPLOYMENT_PRESETS.items()):
        stripped_sat = (
            dataclasses.replace(preset.saturation, megakernel=None)
            if preset.saturation is not None
            else None
        )
        monkeypatch.setitem(
            advisor.DEPLOYMENT_PRESETS,
            model_key,
            dataclasses.replace(
                preset,
                decode=dataclasses.replace(preset.decode, megakernel=None),
                prefill=dataclasses.replace(preset.prefill, megakernel=None),
                saturation=stripped_sat,
            ),
        )


def test_advisor_submodules_and_static_web_assets_packaged() -> None:
    """Verify advisor submodule and static asset files exist and are rendered."""
    advisor_dir = Path(chips.__file__).resolve().parent
    assert (advisor_dir / "specs.py").is_file()
    assert (advisor_dir / "assets" / "report.css").is_file()
    css_content = advisor.assets.load_asset("report.css")
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    sweep = advisor.advise_regimes(
        model_profile, chip, options=advisor.RegimeSweepOptions(tp=8)
    )
    html_out = advisor.render_advisor_html(sweep)
    assert css_content in html_out


@pytest.mark.parametrize(
    "case",
    [
        ("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", 24, 6, 18, 0, "fp8", "fp8"),
        ("qwen3.5-397b", "Qwen/Qwen3.5-397B-A17B-FP8", 60, 15, 45, 60, "fp8", "fp8"),
        ("qwen3.8-2.4t", "Qwen/Qwen3.8-2.4T-A95B", 92, 23, 69, 92, "fp8", "nvfp4"),
        ("kimi-k3", "moonshotai/Kimi-K3", 93, 24, 69, 92, "bf16", "mxfp4"),
        ("kimi-k3.1-pro", "moonshotai/Kimi-K3.1-Pro", 93, 24, 69, 92, "fp8", "mxfp4"),
        ("deepseek-v3", "deepseek-ai/DeepSeek-V3", 61, 61, 0, 58, "fp8", "fp8"),
        ("glm-5.2", "zai-org/GLM-5.2-FP8", 78, 78, 0, 75, "fp8", "fp8"),
        (
            "mistral-large",
            "mistralai/Mistral-Large-Instruct-2411",
            88,
            88,
            0,
            0,
            "fp8",
            "fp8",
        ),
        (
            "mixtral-8x22b",
            "mistralai/Mixtral-8x22B-v0.1",
            56,
            56,
            0,
            56,
            "bf16",
            "bf16",
        ),
        (
            "llama-3.1-405b",
            "meta-llama/Llama-3.1-405B-Instruct-FP8",
            126,
            126,
            0,
            0,
            "fp8",
            "fp8",
        ),
        (
            "llama-4-maverick",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            48,
            48,
            0,
            48,
            "fp8",
            "fp8",
        ),
        ("gpt-oss-120b", "openai/gpt-oss-120b", 36, 36, 0, 36, "fp8", "mxfp4"),
    ],
)
def test_builtin_model_presets_and_hf_aliases(
    case: tuple[str, str, int, int, int, int, str, str],
) -> None:
    """Verify all 13 model presets across architectures."""
    slug, repo_id, layers, attn_l, lin_l, moe_l, wd, ed = case
    model_profile = advisor.resolve_model_profile(slug)
    assert advisor.resolve_model_profile(repo_id) == model_profile
    assert advisor.resolve_model_profile(model_profile) == model_profile
    assert (
        model_profile.num_layers,
        model_profile.num_attention_layers,
        model_profile.num_linear_layers,
        model_profile.num_moe_layers,
        model_profile.weight_dtype,
        model_profile.expert_dtype,
    ) == (layers, attn_l, lin_l, moe_l, wd, ed)
    report = advisor.advise(
        model_profile, advisor.WorkloadSpec(tp=8, batch=16), "tpu7x"
    )
    assert report.model == slug
    expected_num_scopes = 4  # per_op, single_layer, layer_group, whole_step
    assert len(report.scopes) == expected_num_scopes


def test_kimi_k3_linear_attn_config_normalization() -> None:
    """Verify KimiLinear / Kimi-K3.1-Pro nested `linear_attn_config` and latent MoE."""
    model_profile = advisor.model_profile_from_hf_config(
        KIMI_K3_PRO_CFG, name="kimi-k3.1-pro"
    )
    preset = advisor.MODEL_PRESETS["kimi-k3.1-pro"]
    assert model_profile == dataclasses.replace(
        preset,
        dtype_source=model_profile.dtype_source,
    )
    assert (
        model_profile.routed_expert_hidden_size
        == KIMI_K3_PRO_CFG["routed_expert_hidden_size"]
    )
    # Kimi-K3 and Kimi-K3.1-Pro share one public geometry.
    k3 = advisor.MODEL_PRESETS["kimi-k3"]
    assert (k3.num_experts, k3.top_k) == (
        model_profile.num_experts,
        model_profile.top_k,
    )
    expected_kda = len(KIMI_K3_PRO_CFG["linear_attn_config"]["kda_layers"])
    assert (
        advisor.resolve_model_profile(KIMI_K3_PRO_CFG).num_linear_layers == expected_kda
    )


@pytest.mark.parametrize(
    "bad_cfg",
    [
        {},
        {
            "model_type": "bert",
            "num_hidden_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
        },
        {
            "model_type": "mamba2",
            "num_hidden_layers": 24,
            "hidden_size": 2048,
            "num_attention_heads": 16,
        },
        {
            "model_type": "gemma3_unregistered",
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
        },
        {"text_config": "invalid_not_a_dict"},
        {"vision_config": {}, "text_config": {"num_hidden_layers": 12}},
        {
            "model_type": "qwen3_5",
            "num_hidden_layers": 4,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "layer_types": ["full_attention", "unknown_layer_token", "gdn", "gdn"],
        },
        {
            "model_type": "qwen3_5",
            "num_hidden_layers": 4,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "full_attention_interval": 4,
        },
    ],
)
def test_unregistered_and_malformed_model_rejection(bad_cfg: dict[str, object]) -> None:
    """Verify unregistered model_types raise UnsupportedArchitectureError."""
    with pytest.raises(advisor.UnsupportedArchitectureError):
        advisor.model_profile_from_hf_config(bad_cfg)


def test_chip_specs_match_jax_pallas_tpu_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify public TPU chips (v6e, tpu7x) and 'auto' match 'tpu_info'."""
    assert advisor.__name__ == "vllm_torchtpu.megakernel.advisor"
    chips = (
        ("v6e", "v6e"),
        ("tpu7x", "7x"),
    )
    for name, cv_str in chips:
        cs = advisor.resolve_chip_spec(name)
        assert advisor.resolve_chip_spec(cs) is cs
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion(cv_str), 1)
        assert cs.vmem_capacity_bytes == info.vmem_capacity_bytes
        assert cs.hbm_capacity_bytes == info.hbm_capacity_bytes
        assert cs.hbm_bw_bytes_per_s == float(info.mem_bw_bytes_per_second)
        assert cs.bf16_flops_per_s == float(info.bf16_ops_per_second)

    with pytest.raises(MegakernelConfigError, match="requires TPU"):
        advisor.resolve_chip_spec("auto")
    fake_7x = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion("7x"), 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: fake_7x)
    auto_cs = advisor.resolve_chip_spec("auto")
    assert auto_cs.name == "tpu7x"
    assert auto_cs.cores_per_device == 1

    with pytest.raises(MegakernelConfigError, match="Megacore"):
        advisor.resolve_chip_spec("tpu7x", megacore=True)


def test_bare_minimum_auto_parallelism_and_html_regime_report(tmp_path: Path) -> None:
    """Verify bare-minimum Model + TPUv7x 12-regime sweep and HTML SVG export."""
    sweep_397 = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(deployment_preset=False),
    )
    assert sweep_397.parallelism.auto_selected is True
    assert (sweep_397.parallelism.tp, sweep_397.parallelism.ep) == (8, 8)
    assert "2x2x1" in sweep_397.parallelism.topology
    expected_regimes_count = 12  # 3 phases x 4 context depths
    assert len(sweep_397.regimes) == expected_regimes_count
    by_label = {r.label: r for r in sweep_397.regimes}
    assert by_label["Decode Small (B=4, T=1, 4K)"].recommended_scope == "whole_step"
    assert (
        by_label["Decode Small (B=4, T=1, 4K)"].recommended_moe_mode == "masked_on_chip"
    )
    assert by_label["Decode High (B=64, T=1, 32K)"].recommended_scope == "whole_step"
    assert (
        by_label["Decode High (B=64, T=1, 32K)"].recommended_moe_mode
        == "grouped_on_chip"
    )
    assert by_label["Chunked Prefill (B=1, T=256)"].fastest_scope == "whole_step"
    assert (
        by_label["Chunked Prefill (B=1, T=256)"].recommended_moe_mode
        == "grouped_on_chip"
    )
    assert by_label["Long Prefill (B=1, T=4096)"].recommended_scope == "per_op"
    assert (
        by_label["Long Prefill (B=1, T=4096)"].recommended_moe_mode == "grouped_in_hbm"
    )

    sweep_dcp = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(dcp=4, pcp=2),
    )
    assert (sweep_dcp.parallelism.dcp, sweep_dcp.parallelism.pcp) == (4, 2)
    by_label_dcp = {r.label: r for r in sweep_dcp.regimes}
    assert (
        by_label_dcp["Long Prefill (B=1, T=4096)"].active_tokens
        == by_label["Long Prefill (B=1, T=4096)"].active_tokens // 2
    )

    sweep_k3 = advisor.advise_regimes("kimi-k3.1-pro", "tpu7x")
    assert (sweep_k3.parallelism.tp, sweep_k3.parallelism.ep) == (32, 32)
    assert "2x2x4" in sweep_k3.parallelism.topology
    assert sweep_k3.regimes[0].fastest_scope == "whole_step"
    assert sweep_k3.regimes[-1].recommended_scope == "per_op"

    html_file = tmp_path / "advisor_report.html"
    advisor.write_advisor_html(sweep_397, html_file)
    html_text = html_file.read_text("utf-8")
    assert "<svg" in html_text
    assert "advisor-data" in html_text
    assert "qwen3.5-397b" in html_text
    assert "per_op" in html_text
    assert "masked_on_chip" in html_text
    assert "grouped_on_chip" in html_text
    assert "grouped_in_hbm" in html_text
    table_txt = advisor.format_regime_table(sweep_397)
    assert "Decode Small (B=4, T=1, 4K)" in table_txt
    assert "Scope (where the kernel boundary falls):" in table_txt
    assert "Tailoring Guide:" in table_txt


def test_qwen38_24t_decode_regime_crossover_and_moe_modes() -> None:
    """Verify Qwen3.8-2.4T TP32xEP32 recommends whole_step for B=4, 8, 16 decode."""
    sweep_24t = advisor.advise_regimes("qwen3.8-2.4t", "tpu7x")
    assert (sweep_24t.parallelism.tp, sweep_24t.parallelism.ep) == (32, 32)
    min_vmem_tokens = 32  # 16 requests x 2 tokens
    assert sweep_24t.max_vmem_tokens >= min_vmem_tokens
    assert sweep_24t.single_layer_vmem_tokens > sweep_24t.max_vmem_tokens
    by_label = {r.label: r for r in sweep_24t.regimes}

    for small_lbl in (
        "Decode Small (B=4, T=1, 4K)",
        "Decode Low (B=8, T=1, 4K)",
        "Decode Med (B=16, T=1, 4K)",
        "Spec Verify Small (B=4, K=3)",
    ):
        entry = by_label[small_lbl]
        assert entry.fastest_scope == "whole_step"
        assert entry.recommended_moe_mode == "masked_on_chip"
        assert entry.num_vmem_splits == 1

    for wide_lbl in (
        "Decode High (B=64, T=1, 32K)",
        "Spec Verify Med (B=16, K=3)",
        "Chunked Prefill (B=1, T=256)",
    ):
        entry = by_label[wide_lbl]
        assert entry.fastest_scope == "whole_step"
        assert entry.recommended_moe_mode == "grouped_on_chip"

    long_pre = by_label["Long Prefill (B=1, T=4096)"]
    assert long_pre.recommended_scope == "per_op"
    assert long_pre.recommended_moe_mode == "grouped_in_hbm"


@pytest.mark.parametrize(
    "bad_extra",
    [
        ["--prefix-hit-rate", "1.0"],
        ["--ep", "0"],
        ["--kv-split", "0"],
        ["--op-overhead-us", "-1.0"],
        ["--dma-buffers", "4"],
        ["--vmem-budget-mib", "1.0"],
    ],
)
def test_cli_parameterized_validation_errors(
    tmp_path: Path, bad_extra: list[str]
) -> None:
    """Verify parameterized CLI validation boundaries raise `MegakernelConfigError`."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(QWEN35_08B_CFG), encoding="utf-8")
    with pytest.raises(MegakernelConfigError):
        advisor.main(["--config", str(cfg_file), *bad_extra])


def test_cli_json_presets_and_validation_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI JSON/table/HTML output, bare-minimum auto-TP, and error wrapping."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(QWEN35_08B_CFG), encoding="utf-8")
    rc = advisor.main(
        ["--config", str(cfg_file), "--chip", "tpu7x", "--tp", "8", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended_scope"] == "whole_step"
    assert payload["recommended_moe_mode"] == "dense_mlp"
    assert payload["build_megakernel"] is True
    assert payload["vmem_budget_mib"] == resolve_chip_spec(
        "tpu7x"
    ).vmem_usable_bytes / (1024 * 1024)
    html_out = tmp_path / "regimes.html"
    assert (
        advisor.main(
            [
                "--model",
                "qwen3.5-397b",
                "--regime-sweep",
                "--vmem-budget-mib",
                "60.0",
                "--dma-buffers",
                "2",
                "--html",
                str(html_out),
            ]
        )
        == 0
    )
    sweep_stdout = capsys.readouterr().out
    assert "Decode Parallelism: TP8 x EP8 x DCP4" in sweep_stdout
    assert "usable=60.0 MiB" in sweep_stdout
    assert html_out.is_file()

    assert (
        advisor.main(
            [
                "--model",
                "moonshotai/Kimi-K3.1-Pro",
                "--tp",
                "16",
                "--ep",
                "16",
                "--vmem-fit-headroom",
                "--no-vmem-alias",
            ]
        )
        == 0
    )
    assert "Model: kimi-k3.1-pro" in capsys.readouterr().out

    for bad_argv in (
        ["--config", str(tmp_path / "missing.json")],
        ["--model", str(tmp_path / "missing_dir")],
        ["--model", "org/unknown-preset"],
    ):
        with pytest.raises(MegakernelConfigError):
            advisor.main(bad_argv)

    bad_dense = dataclasses.replace(
        advisor.model_profile_from_hf_config(QWEN35_08B_CFG), top_k=2
    )
    with pytest.raises(MegakernelConfigError):
        advisor.advise(bad_dense, advisor.WorkloadSpec(tp=8))


@pytest.mark.parametrize("model", ["qwen3.5-397b", "kimi-k3"])
def test_regime_dot_plot_labels_once_and_inside_viewbox(model: str) -> None:
    """Verify each regime label renders once and no text leaves the viewBox."""
    sweep = advisor.advise_regimes(model, "tpu7x")
    parser = _SvgTextCollector()
    parser.feed(advisor.report.render_regime_dot_plot_svg(sweep))
    _, _, width, height = (float(v) for v in parser.view_box.split())
    for entry in sweep.regimes:
        assert sum(text == entry.label for _, text in parser.texts) == 1
    for attr, text in parser.texts:
        size = float(attr["font-size"])
        assert size >= advisor.report.SVG_MIN_FONT_PX
        span = advisor.report.svg_text_width(text, size)
        x, y = float(attr["x"]), float(attr["y"])
        left = {"start": x, "middle": x - span / 2, "end": x - span}[
            attr.get("text-anchor", "start")
        ]
        assert left >= 0.0
        assert left + span <= width
        assert y - size >= 0.0
        assert y <= height


@pytest.mark.parametrize("model", ["qwen3.5-397b", "kimi-k3", "qwen3.8-2.4t"])
def test_reports_name_mechanisms_not_kernels(model: str) -> None:
    """Verify the text table and HTML name no kernel or hardware unit."""
    sweep = advisor.advise_regimes(model, "tpu7x")
    rendered = advisor.format_regime_table(sweep) + advisor.render_advisor_html(sweep)
    assert KERNEL_NAME_DENYLIST.search(rendered) is None


def test_unregistered_model_type_estimates_with_architecture_unverified_flag() -> None:
    """Verify unseen model_type fails closed on HF dict.

    Direct ModelProfile flags unverified arch.
    """
    future_cfg: dict[str, object] = {
        "model_type": "falcon_future_causal",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    with pytest.raises(advisor.UnsupportedArchitectureError):
        advisor.model_profile_from_hf_config(future_cfg, name="falcon-future")

    model_profile = advisor.ModelProfile(
        name="falcon-future",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        num_attention_layers=32,
        num_moe_layers=0,
        weight_dtype="bf16",
        expert_dtype="bf16",
        kv_dtype="bf16",
        model_type="falcon_future_causal",
        architecture_unverified=True,
        layer_families=("gqa", "dense_mlp"),
    )
    assert model_profile.architecture_unverified is True
    assert "gqa" in model_profile.layer_families
    assert "dense_mlp" in model_profile.layer_families
    report = advisor.advise(
        model_profile, advisor.WorkloadSpec(tp=8, batch=16), "tpu7x"
    )
    assert "architecture_unverified" in report.flags


def test_unknown_chip_requires_chip_spec_fields(tmp_path: Path) -> None:
    """Verify unknown chip names require `--chip-spec` and load valid JSON specs."""
    with pytest.raises(MegakernelConfigError, match="--chip-spec") as exc_info:
        advisor.resolve_chip_spec("future_xpu_v99")
    err_text = str(exc_info.value)
    for req_key in (
        "vmem_capacity_bytes",
        "hbm_capacity_bytes",
        "hbm_bw_bytes_per_s",
        "bf16_flops_per_s",
    ):
        assert req_key in err_text

    spec_file = tmp_path / "future_xpu.json"
    spec_payload = {
        "name": "future_xpu_v99",
        "cores_per_device": 1,
        "vmem_capacity_bytes": 67108864,
        "vmem_usable_bytes": 65011712,
        "hbm_capacity_bytes": 103000000000,
        "hbm_bw_bytes_per_s": 3.7e12,
        "bf16_flops_per_s": 1.155e15,
        "fp8_flops_per_s": 2.3e15,
        "native_ocp_fp8": True,
        "ici_egress_bw_bytes_per_s": 3.0e11,
        "ici_link_bw_bytes_per_s": 1.0e11,
        "collective_latency_us_bracket": [1.0, 4.0],
        "sources": {"spec": "custom user chip spec JSON"},
    }
    spec_file.write_text(json.dumps(spec_payload), encoding="utf-8")
    cs = advisor.resolve_chip_spec("future_xpu_v99", chip_spec_path=spec_file)
    assert cs.name == "future_xpu_v99"
    assert cs.peak_flops("fp8") == resolve_chip_spec("tpu7x").peak_flops("fp8")
    rc = advisor.main(
        [
            "--model",
            "qwen3.5-0.8b",
            "--chip",
            "future_xpu_v99",
            "--chip-spec",
            str(spec_file),
            "--tp",
            "8",
        ]
    )
    assert rc == 0


def test_every_output_number_has_provenance_tag() -> None:
    """Verify every metric in single-point and sweep dicts carries a valid tag."""
    valid_tags = {"DERIVED", "ASSUMED", "MEASURED"}
    required_keys = (
        "fits_hbm",
        "tokens_per_core",
        "onchip_ceiling",
        "compute_bound_at",
        "step_floor_us",
        "throughput_tok_s_per_chip",
        "boundary_cost_us",
        "fusion_gain_max",
        "step_coverage",
        "expected_gain_max",
    )
    report = advisor.advise(
        "qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=16, tp=8), "tpu7x"
    )
    d_rep = report.to_dict()
    metrics = d_rep["metrics"]
    assert isinstance(metrics, dict)
    for k in required_keys:
        assert k in metrics
    for entry in metrics.values():
        assert entry["tag"] in valid_tags
        assert bool(entry["source"])
        assert bool(entry["unit"])
        band = entry["band"]
        assert band[0] <= band[1]

    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    d_sweep = sweep.to_dict()
    for reg in d_sweep["regimes"]:
        reg_metrics = reg["metrics"]
        for k in required_keys:
            assert k in reg_metrics
        for entry in reg_metrics.values():
            assert entry["tag"] in valid_tags
            assert entry["band"][0] <= entry["band"][1]


def test_prefill_stage_floor_stays_below_upper_ceiling() -> None:
    """Verify Qwen3.5-397B PP8 8K prefill stage floor is strictly below ceiling."""
    p397 = advisor.model_profile_from_hf_config(QWEN35_397B_CFG, name="qwen3.5-397b")
    workload = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=8192,
        chunk_size=8192,
        context=95232,
        pp=8,
        tp=1,
        ep=1,
        moe_mode="grouped_on_chip",
    )
    report = advisor.advise(p397, workload, "tpu7x")
    lg = next(s for s in report.scopes if s.scope == "layer_group")
    assert lg.moe_flop_factor == 1.0
    floor_min_us = 30_000.0
    floor_max_us = 60_000.0
    ceiling_us = 75_000.0
    assert floor_min_us <= lg.step_bound_us <= floor_max_us
    assert lg.step_bound_us < ceiling_us
    assert lg.step_upper_us >= lg.step_bound_us
    assert lg.step_upper_us < ceiling_us


def test_shared_assumptions_keep_decode_megakernel_verdict() -> None:
    """Verify fewer launches with equal collectives win at every corner, so no tie."""
    for model in ("qwen3.5-397b", "kimi-k3"):
        row = _regime(model, "Decode Small")
        assert row.fastest_scope == "whole_step"
        assert row.fastest_verdict == "megakernel_likely"
        # The assumed band is still shown even though the winner never changes.
        assert row.step_upper_us > row.step_bound_us


@pytest.mark.usefixtures("without_declared_megakernels")
def test_tie_only_when_winner_flips_between_corners() -> None:
    """Verify VMEM-split decode boundary flips from single_layer to whole_step."""
    workload = advisor.WorkloadSpec(
        phase="decode", batch=512, tp=8, dp=2, ep=16, dcp=4, context=16384
    )
    report = advisor.advise("qwen3.5-397b", workload, "tpu7x")
    assert report.fastest_scope == "single_layer"
    assert report.fastest_verdict == "tie(measure: launch_gap_us)"
    assert "vs runner-up whole_step" in report.rationale
    assert "whole_step wins at launch gap 8.0 us" in report.rationale
    knob = report.fastest_verdict.removeprefix("tie(measure: ").removesuffix(")")
    scope_inputs = {
        u["name"] for u in report.unresolved_assumptions if u["affects"] == "scope"
    }
    assert knob in scope_inputs


@pytest.mark.usefixtures("without_declared_megakernels")
def test_measurements_narrow_scope_band_and_settle_verdict(tmp_path: Path) -> None:
    """Verify measured launch gap and collective latency collapse the assumed box."""
    m_file = tmp_path / "measurements.json"
    m_payload = {
        "schema": "advisor.measurements.v1",
        "source": {"kind": "xplane", "file": "trace.xplane.pb", "tool": "calibrate"},
        "launch_gap_us": {"p50": 0.5, "p90": 0.5, "n": 40},
        "collectives": [
            {
                "op": "all_reduce",
                "group_size": 8,
                "bytes": 32768,
                "latency_us": 9.0,
                "n": 20,
            }
        ],
    }
    m_file.write_text(json.dumps(m_payload), encoding="utf-8")
    cliff_wl = advisor.WorkloadSpec(
        phase="decode", batch=512, tp=8, dp=2, ep=16, dcp=4, context=16384
    )
    prior = advisor.advise("qwen3.5-397b", cliff_wl, "tpu7x")
    assert prior.fastest_verdict.startswith("tie(measure: ")
    row = advisor.advise("qwen3.5-397b", cliff_wl, "tpu7x", measurements=m_file)
    assert row.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert row.metrics["collective_latency_us"]["tag"] == "MEASURED"
    # Decode has no attention dtype band, so a degenerate box gives a zero-width band.
    rec_scope = next(s for s in row.scopes if s.scope == row.fastest_scope)
    assert rec_scope.step_upper_us == rec_scope.step_bound_us
    assert not row.fastest_verdict.startswith("tie")


def test_single_layer_is_fused_layer_not_megakernel() -> None:
    """Verify one call per layer is reported as a fused layer, not a megakernel."""
    row = _regime("kimi-k3", "Decode Saturated")
    assert row.fastest_scope == "single_layer"
    assert row.fastest_verdict == "fused_layer_likely"
    assert _regime("kimi-k3", "Preset Prefill Window (W=2048)").fastest_verdict == (
        "megakernel_likely"
    )


def test_throughput_counts_all_window_tokens_per_physical_chip() -> None:
    """Verify tok/s/chip uses the whole PCP window and 2 devices per v7x chip."""
    row = _regime("kimi-k3", "Preset Prefill Window (W=2048)")
    # 64 tokens per core x PCP32 = 2048 tokens per step over 16 chips (32 cores).
    expected = 2048 / 16 / (row.step_bound_us * 1e-6)
    got = float(row.metrics["throughput_tok_s_per_chip"]["value"])
    assert row.active_tokens == 2048 // 32
    assert got == pytest.approx(expected, rel=1e-3)
    dec = _regime("kimi-k3", "Decode Small")
    # 4 decode tokens per step over 16 chips.
    exp_dec = 4 / 16 / (dec.step_bound_us * 1e-6)
    got_dec = float(dec.metrics["throughput_tok_s_per_chip"]["value"])
    assert got_dec == pytest.approx(exp_dec, rel=1e-3)


def test_kimi_k3_preset_uses_drafter_verify8_and_dp2_saturation() -> None:
    """Verify Kimi-K3 rows use its shipped draft length and high-c DP2 layout."""
    sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    labels = [r.label for r in sweep.regimes]
    assert "Spec Verify Small (B=4, K=7)" in labels
    verify = next(r for r in sweep.regimes if r.label.startswith("Spec Verify Small"))
    assert verify.active_tokens == 4 * 8
    sat = next(r for r in sweep.regimes if r.label.startswith("Decode Saturated"))
    assert sat.layout == "TP16 x DP2 x EP32 x DCP16"
    assert sat.label == "Decode Saturated (B=512 over DP2, T=1, 32K)"
    # Qwen keeps 3 draft tokens.
    q = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    assert "Spec Verify Small (B=4, K=3)" in [r.label for r in q.regimes]


def test_rationale_reports_margin_percent() -> None:
    """Verify the rationale states the runner-up margin in us and percent."""
    row = _regime("qwen3.5-397b", "Decode Small")
    assert "vs runner-up single_layer" in row.rationale
    assert re.search(r"\+\d+\.\d us, \d+\.\d%\)", row.rationale)


def test_kimi_k3_preset_matches_public_hf_config() -> None:
    """Verify Kimi-K3 uses the public HF expert count and routing width."""
    k3 = advisor.MODEL_PRESETS["kimi-k3"]
    # HF moonshotai/Kimi-K3 config.json: num_experts 896, num_experts_per_token 16.
    assert (k3.num_experts, k3.top_k, k3.num_moe_layers) == (896, 16, 92)
    assert "kimi-k3-pro" not in advisor.MODEL_PRESETS
    qwen_24t = advisor.resolve_model_profile("qwen3.8-2.4t-a95b")
    assert qwen_24t.num_experts == advisor.MODEL_PRESETS["qwen3.8-2.4t"].num_experts


def test_step_coverage_never_flags_or_decides_scope() -> None:
    """Verify coverage scales expected gain only: never a flag and never a tie knob."""
    for model in ("qwen3.5-397b", "kimi-k3"):
        for row in advisor.advise_regimes(model, "tpu7x").regimes:
            assert "coverage_assumed" not in row.flags
            assert "step_coverage" not in row.verdict
            cov = [
                u for u in row.unresolved_assumptions if u["name"] == "step_coverage"
            ]
            assert all(u["affects"] == "gain" for u in cov)


def test_boundary_cost_uses_sourced_launch_gap() -> None:
    """Verify default `WorkloadSpec` prices non-zero boundary cost with ASSUMED tag."""
    report = advisor.advise(
        "qwen3.5-0.8b",
        advisor.WorkloadSpec(phase="decode", batch=4, tp=8),
        "tpu7x",
    )
    assert report.metrics["boundary_cost_us"]["value"] > 0.0
    assert report.metrics["launch_gap_us"]["tag"] == "ASSUMED"
    assert report.metrics["launch_gap_us"]["value"] > 0.0


@pytest.mark.usefixtures("without_declared_megakernels")
def test_ring_plan_changes_onchip_ceiling() -> None:
    """Verify deeper ring-buffer slot plans lower the on-chip token ceiling."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    r_two = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            tp=8,
            ep=8,
            ring_plan=advisor.RingPlan(dense_slots=2, expert_slots=2),
        ),
        "tpu7x",
    )
    r_deep = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            tp=8,
            ep=8,
            ring_plan=advisor.RingPlan(dense_slots=4, expert_slots=3),
        ),
        "tpu7x",
    )
    assert (
        r_deep.metrics["onchip_ceiling"]["value"]
        < r_two.metrics["onchip_ceiling"]["value"]
    )


def test_measurements_json_promotes_assumed_to_measured(tmp_path: Path) -> None:
    """Verify `measurements.json` upgrades assumed inputs to MEASURED."""
    m_file = tmp_path / "measurements.json"
    m_payload = {
        "schema": "advisor.measurements.v1",
        "source": {"kind": "xplane", "file": "trace.xplane.pb", "tool": "calibrate"},
        "launch_gap_us": {"p50": 3.5, "p90": 5.0, "n": 40},
        "collectives": [
            {
                "op": "all_reduce",
                "group_size": 8,
                "bytes": 32768,
                "latency_us": 12.0,
                "n": 20,
            }
        ],
        "step_coverage": {
            "fused_eligible_fraction": 0.85,
            "basis": "trace",
        },
    }
    m_file.write_text(json.dumps(m_payload), encoding="utf-8")
    report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
        measurements=m_file,
    )
    assert report.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert report.metrics["step_coverage"]["tag"] == "MEASURED"
    measured_coverage = 0.85
    assert report.metrics["step_coverage"]["value"] == measured_coverage
    assert "coverage_assumed" not in report.flags


def test_import_without_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify advisor imports and runs without jax."""
    mk_init = Path(mk.__file__)
    adv_dir = mk_init.parent / "advisor"
    lazy_jax_bridges = frozenset({"jax_bridge.py", "spec_bridge.py"})
    for py_path in (mk_init, *sorted(adv_dir.glob("*.py"))):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        is_bridge = py_path.name in lazy_jax_bridges
        for ast_node in ast.walk(tree):
            if isinstance(ast_node, ast.Import):
                for alias in ast_node.names:
                    root = alias.name.split(".", 1)[0]
                    assert root != "torch", f"Unexpected torch import in {py_path}"
                    if not is_bridge:
                        assert root != "jax", (
                            f"Unexpected jax import in non-bridge {py_path}"
                        )
            elif isinstance(ast_node, ast.ImportFrom) and ast_node.module:
                root = ast_node.module.split(".", 1)[0]
                assert root != "torch", f"Unexpected torch import in {py_path}"
                if not is_bridge:
                    assert root != "jax", (
                        f"Unexpected jax import in non-bridge {py_path}"
                    )

    blocker = _JaxBlocker()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    for mod_name in list(sys.modules):
        if mod_name == "jax" or mod_name.startswith(
            ("jax.", "vllm_torchtpu.megakernel")
        ):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    mk_fresh = importlib.import_module("vllm_torchtpu.megakernel")
    adv_fresh = importlib.import_module("vllm_torchtpu.megakernel.advisor")
    assert "jax" not in sys.modules
    assert mk_fresh.advisor is adv_fresh
    report = adv_fresh.advise(
        "qwen3.5-0.8b",
        adv_fresh.WorkloadSpec(phase="decode", batch=4, tp=8),
        "tpu7x",
    )
    assert report.recommended_scope == "whole_step"


def test_objective_both_reports_disagreement() -> None:
    """Verify `objective='both'` emits `tie(objective_split: ...)` on split."""
    pk3 = advisor.resolve_model_profile("kimi-k3")
    report = advisor.compare_parallelism_candidates(
        pk3,
        (
            advisor.ParallelLayout(tp=1, ep=32, dp=1, dcp=32),
            advisor.ParallelLayout(tp=1, ep=32, dp=2, dcp=16),
        ),
        chip="tpu7x",
        base_workload=advisor.WorkloadSpec(batch=48, context=32768, objective="both"),
    )
    assert str(report["verdict"]).startswith("tie(objective_split:")
    assert report["best_step_layout"] != report["best_throughput_layout"]


def test_tpu7x_topologies_start_at_2x2x1_and_tpu6e_alias() -> None:
    """Verify TPU7X_TOPOLOGIES starts at 2x2x1 (8 cores) and tpu6e resolves."""
    min_torus_cores = 8  # 2x2x1 torus geometry = 8 cores
    assert min(advisor.TPU7X_TOPOLOGIES) == min_torus_cores
    assert advisor.TPU7X_TOPOLOGIES[8].startswith("2x2x1")
    for sub_host in (1, 2, 4):
        assert sub_host not in advisor.TPU7X_TOPOLOGIES
    c6e = advisor.resolve_chip_spec("tpu6e")
    assert c6e.name == "v6e"
    assert min(c6e.topologies) == 1
    par = advisor.recommend_parallelism(
        advisor.resolve_model_profile("qwen3.5-0.8b"), "tpu7x"
    )
    assert par.tp * par.dp * par.pp >= min_torus_cores


def test_pd_disagg_alignment_qwen35_and_kimi_k3_and_dcn_handoff(
    tmp_path: Path,
) -> None:
    """Verify Qwen3.5-397B and Kimi-K3 P/D scope recommendations and DCN handoff."""
    q_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    q_by_lbl = {r.label: r for r in q_sweep.regimes}
    q_sat = next(r for r in q_sweep.regimes if r.label.startswith("Decode Saturated"))
    assert q_by_lbl["Decode Med (B=16, T=1, 4K)"].recommended_scope == "whole_step"
    assert q_by_lbl["Decode High (B=64, T=1, 32K)"].recommended_scope == "whole_step"
    assert q_sat.layout == "TP2 x DP4 x EP8"
    assert q_sat.fastest_scope == "whole_step"
    assert q_by_lbl["Preset Prefill Window (W=2176)"].layout == "PP8"
    assert q_by_lbl["Preset Prefill Window (W=2176)"].recommended_scope == "per_op"
    assert q_sweep.disagg_handoff is not None
    assert q_sweep.disagg_handoff.dcn_provenance == "ASSUMED"
    default_dcn_bw_gbps = 100.0  # default 100 Gbps DCN
    assert q_sweep.disagg_handoff.dcn_bw_gbps == default_dcn_bw_gbps
    assert q_sweep.disagg_handoff.dcn_bw_gbps_bracket == (50.0, 200.0)

    k_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    k_by_lbl = {r.label: r for r in k_sweep.regimes}
    assert k_by_lbl["Decode Med (B=16, T=1, 4K)"].fastest_scope == "whole_step"
    assert k_by_lbl["Preset Prefill Window (W=2048)"].layout == "TP32 x EP16 x PCP32"
    assert k_by_lbl["Preset Prefill Window (W=2048)"].fastest_scope == "layer_group"
    assert k_by_lbl["Preset Prefill Window (W=8192)"].fastest_scope == "layer_group"

    m_file = tmp_path / "dcn_meas.json"
    m_file.write_text(
        json.dumps(
            {
                "schema": "advisor.measurements.v1",
                "source": {"kind": "xplane", "file": "dcn.xplane.pb"},
                "dcn_bw_gbps": 160.0,
            }
        ),
        encoding="utf-8",
    )
    q_meas = advisor.advise_regimes("qwen3.5-397b", "tpu7x", measurements=m_file)
    assert q_meas.disagg_handoff is not None
    assert q_meas.disagg_handoff.dcn_provenance == "MEASURED"
    measured_dcn_bw_gbps = 160.0
    assert q_meas.disagg_handoff.dcn_bw_gbps == measured_dcn_bw_gbps


def test_chip_spec_optional_fields_sysfs_auto_int8_quant_and_refs_main(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify optional `--chip-spec` keys, sysfs `auto`, int8, and `--objective`."""
    spec_file = tmp_path / "custom_chip.json"
    spec_file.write_text(
        json.dumps(
            {
                "name": "custom_2tc",
                "vmem_capacity_bytes": 64 << 20,
                "hbm_capacity_bytes": 96 * (10**9),
                "hbm_bw_bytes_per_s": 3690.0e9,
                "bf16_flops_per_s": 1153.0e12,
                "devices_per_chip": 2,
                "min_slice_cores": 8,
                "dcn_bw_gbps": 200.0,
                "dcn_bw_gbps_bracket": [100.0, 400.0],
                "topologies": {"8": "2x2x1 custom"},
                "matmul_row_tile_by_dtype": {"int8": 16, "bf16": 8},
            }
        ),
        encoding="utf-8",
    )
    loaded = advisor.load_chip_spec_file(spec_file)
    assert loaded.devices_per_chip == constants.MEGACORE_CORES
    expected_min_cores = 8
    expected_dcn_bw = 200.0
    assert loaded.min_slice_cores == expected_min_cores
    assert loaded.dcn_bw_gbps == expected_dcn_bw
    assert loaded.dcn_bw_gbps_bracket == (100.0, 400.0)
    assert loaded.topologies == {8: "2x2x1 custom"}
    assert loaded.matmul_row_tile("int8") == constants.DEFAULT_ROW_TILE_WIDE

    bad_spec = tmp_path / "bad_chip.json"
    bad_spec.write_text(
        json.dumps(
            {
                "vmem_capacity_bytes": 64 << 20,
                "hbm_capacity_bytes": 96 * (10**9),
                "hbm_bw_bytes_per_s": 3690.0e9,
                "bf16_flops_per_s": 1153.0e12,
                "typo_field": 123,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MegakernelConfigError, match="unrecognized keys"):
        advisor.load_chip_spec_file(bad_spec)

    pci_dev = tmp_path / "pci" / "0000:00:04.0"
    pci_dev.mkdir(parents=True)
    (pci_dev / "vendor").write_text("0x1ae0\n", encoding="utf-8")
    (pci_dev / "device").write_text("0x006f\n", encoding="utf-8")
    detected = advisor.detect_tpu_from_sysfs(tmp_path / "pci")
    auto_v6e = advisor.resolve_chip_spec(detected)
    assert auto_v6e.name == "v6e"

    int8_cfg = {
        **MIXTRAL_8X7B_CFG,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "int"}}},
        },
    }
    p_int8 = advisor.model_profile_from_hf_config(int8_cfg, name="mixtral-int8")
    assert p_int8.weight_dtype == "int8"

    hf_cache = tmp_path / "hf_home"
    repo_root = hf_cache / "hub" / "models--acme--test-qwen"
    snap_old = repo_root / "snapshots" / "zzzz_older_lex_last"
    snap_new = repo_root / "snapshots" / "aaaa_main_ref"
    snap_old.mkdir(parents=True)
    snap_new.mkdir(parents=True)
    (repo_root / "refs").mkdir(parents=True)
    (repo_root / "refs" / "main").write_text("aaaa_main_ref\n", encoding="utf-8")
    expected_hidden = 2048
    (snap_old / "config.json").write_text(
        json.dumps({**QWEN35_397B_CFG, "hidden_size": 1024}), encoding="utf-8"
    )
    (snap_new / "config.json").write_text(
        json.dumps({**QWEN35_397B_CFG, "hidden_size": expected_hidden}),
        encoding="utf-8",
    )
    prof_ref = advisor.resolve_model_profile(
        advisor.resolve_hf_snapshot_config("acme/test-qwen", hf_home=hf_cache)
    )
    assert prof_ref.hidden_size == expected_hidden

    rc = advisor.main(
        [
            "--model",
            "kimi-k3",
            "--chip",
            "tpu7x",
            "--phase",
            "decode",
            "--batch",
            "48",
            "--context",
            "32768",
            "--tp",
            "1",
            "--ep",
            "32",
            "--dp",
            "1",
            "--dcp",
            "32",
            "--objective",
            "both",
        ]
    )
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "tie(objective_split:" in out_text


def test_compute_all_reduce_transfer_cost_single_host() -> None:
    """Verifies single-host (TP <= 8) collective uses pure optical ICI parameters."""
    chip = advisor.resolve_chip_spec("tpu7x")
    message_bytes = 1024 * 1024  # 1 MiB (bulk payload)
    transfer_s, latency_s, hops, vol = advisor.compute_all_reduce_transfer_cost(
        message_bytes=message_bytes,
        effective_tp=8,
        chip_spec=chip,
    )
    expected_vol = 2.0 * (8 - 1) / 8 * message_bytes
    assert vol == expected_vol
    assert transfer_s == expected_vol / chip.ici_link_bw_bytes_per_s
    assert (hops, latency_s) == (3, 3 * chip.collective_latency_s)


def test_compute_all_reduce_transfer_cost_optical_tree_hops() -> None:
    """Verifies optical ICI collective scales logarithmically with rank count."""
    chip = advisor.resolve_chip_spec("tpu7x")
    _, _, hops_tp8, _ = advisor.compute_all_reduce_transfer_cost(
        message_bytes=65536,
        effective_tp=8,
        chip_spec=chip,
    )
    _, _, hops_tp4, _ = advisor.compute_all_reduce_transfer_cost(
        message_bytes=65536,
        effective_tp=4,
        chip_spec=chip,
    )
    assert (hops_tp8, hops_tp4) == (3, 2)


def test_fit_vmem_budget_bytes_across_chip_specs() -> None:
    """Verifies VMEM usable budget dynamically derives from target chip specs."""
    v7x = advisor.resolve_chip_spec("tpu7x")
    v6e = advisor.resolve_chip_spec("tpu6e")
    rp = advisor.default_double_buffer_ring_plan()

    expected_v7x = max(0, (62 << 20) - rp.compiler_spill_reserve_bytes)
    assert advisor.fit_vmem_budget_bytes(chip_spec=v7x) == expected_v7x

    expected_v6e = max(0, (126 << 20) - rp.compiler_spill_reserve_bytes)
    assert advisor.fit_vmem_budget_bytes(chip_spec=v6e) == expected_v6e

    assert advisor.fit_vmem_budget_bytes() == expected_v7x


def test_kimi_k3_deployment_preset_registered_and_evaluated() -> None:
    """Verifies kimi-k3 preset is registered and validly evaluated."""
    preset = advisor.DEPLOYMENT_PRESETS.get("kimi-k3")
    assert preset is not None
    assert (
        preset.decode.tp,
        preset.decode.ep,
        preset.decode.dcp,
        preset.prefill.tp,
        preset.prefill.ep,
        preset.prefill.pcp,
    ) == (32, 32, 32, 32, 16, 32)


def test_measured_collective_latency_moves_step_bound() -> None:
    """Measured collective latency must reach the all-reduce base latency term."""
    workload = advisor.WorkloadSpec(phase="decode", batch=8, isl=128, osl=1, tp=8, ep=1)
    rep_default = advisor.advise("qwen3.5-397b", workload, chip="tpu7x")
    meas = advisor.MeasurementsData(
        source_desc="test_trace",
        collective_hop_us=3.0,
        collective_n=10,
    )
    rep_calibrated = advisor.advise(
        "qwen3.5-397b", workload, chip="tpu7x", measurements=meas
    )
    assert (
        rep_calibrated.metrics["step_floor_us"]["value"]
        > rep_default.metrics["step_floor_us"]["value"]
    )


def test_custom_chip_spec_vmem_usable_bytes_preserved(tmp_path: Path) -> None:
    """Custom chip spec vmem_usable_bytes must not be reset by _scale_megacore."""
    spec_file = tmp_path / "custom_chip.json"
    spec_file.write_text(
        json.dumps(
            {
                "name": "custom_chip",
                "vmem_capacity_bytes": 64 * 1024 * 1024,
                "vmem_usable_bytes": 48 * 1024 * 1024,
                "hbm_capacity_bytes": 34359738368,
                "hbm_bw_bytes_per_s": 3700000000000.0,
                "bf16_flops_per_s": 1155000000000000.0,
                "fp8_flops_per_s": 2300000000000000.0,
                "int8_flops_per_s": 2300000000000000.0,
                "native_ocp_fp8": True,
                "ici_egress_bw_bytes_per_s": 300000000000.0,
                "ici_link_bw_bytes_per_s": 100000000000.0,
                "collective_latency_s": 0.000001,
                "collective_latency_us_bracket": [1.0, 5.0],
                "dcn_bw_gbps": 100.0,
                "dcn_bw_gbps_bracket": [50.0, 200.0],
            }
        )
    )
    spec = resolve_chip_spec("custom_chip", chip_spec_path=str(spec_file))
    assert spec.vmem_usable_bytes == 48 * 1024 * 1024
    assert spec.vmem_capacity_bytes == 64 * 1024 * 1024


def test_vmem_budget_no_double_spill_subtraction() -> None:
    """Verify fit_vmem_budget_bytes and vmem_fit_headroom keep 62.0 MiB usable VMEM."""
    expected_bytes = constants.CORE_LIMITS.vmem_mib << 20
    assert advisor.fit_vmem_budget_bytes() == expected_bytes
    assert advisor.fit_vmem_budget_bytes(Tiles(vmem_mib=62)) == expected_bytes
    rep_fit = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=64, tp=8, vmem_fit_headroom=True),
        "tpu7x",
    )
    rep_std = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=64, tp=8, vmem_fit_headroom=False),
        "tpu7x",
    )
    assert rep_fit.vmem_budget_mib == resolve_chip_spec("tpu7x").vmem_usable_bytes / (
        1024 * 1024
    )
    assert rep_fit.vmem_budget_mib == rep_std.vmem_budget_mib


@pytest.mark.usefixtures("without_declared_megakernels")
def test_adaptive_ring_contraction_single_pass_wide_batch() -> None:
    """Verify adaptive ring shrinks from (14, 10) to (2, 2) before splitting passes."""
    rep_small = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
    )
    ws_small = next(s for s in rep_small.scopes if s.scope == "whole_step")
    assert ws_small.selected_dense_slots == constants.WIDE_DENSE_RING_SLOTS
    assert ws_small.selected_expert_slots == constants.WIDE_EXPERT_RING_SLOTS
    assert ws_small.n_splits == 1

    rep_wide = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=448, tp=8, ep=8),
        "tpu7x",
    )
    ws_wide = next(s for s in rep_wide.scopes if s.scope == "whole_step")
    assert ws_wide.selected_dense_slots == constants.DOUBLE_BUFFER_SLOTS
    assert ws_wide.selected_expert_slots == constants.DOUBLE_BUFFER_SLOTS
    assert ws_wide.n_splits == 1

    rep_llama = advisor.advise(
        "llama-3.1-405b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8),
        "tpu7x",
    )
    ws_llama = next(s for s in rep_llama.scopes if s.scope == "whole_step")
    assert ws_llama.t_vmem_max > 0
    assert ws_llama.n_splits == 1
    assert ws_llama.selected_expert_slots == 0


def test_kimi_k3_dense_prefix_moe_and_qwen35_attn_gate() -> None:
    """Verify Kimi-K3 dense_prefix_moe flag and Qwen3.5 attention output gate size."""
    k3 = advisor.resolve_model_profile("kimi-k3")
    assert k3.is_periodic is False
    assert "dense_prefix_moe" in k3.non_homogeneous_reasons

    q35 = advisor.resolve_model_profile("qwen3.5-397b")
    assert q35.has_attn_output_gate is True
    q35_ungated = dataclasses.replace(q35, has_attn_output_gate=False)
    workload = advisor.WorkloadSpec(phase="decode", batch=4, tp=8, ep=8)
    rep_gated = advisor.advise(q35, workload, "tpu7x")
    rep_ungated = advisor.advise(q35_ungated, workload, "tpu7x")
    w_diff = round(rep_gated.scopes[0].w_step_gb - rep_ungated.scopes[0].w_step_gb, 4)
    expected_attn_gate_delta_gb = round((15 * 4096 * (4 * 256)) / 1e9, 4)
    assert w_diff == pytest.approx(expected_attn_gate_delta_gb, abs=2e-4)


def test_pcp_ring_attention_ici_and_validate_workload() -> None:
    """Verify PCP > 1 prices RingAttention ICI volume and validate_workload flags."""
    wl_pcp1 = advisor.WorkloadSpec(
        phase="prefill", batch=1, isl=4096, chunk_size=512, tp=8, ep=8, pcp=1
    )
    wl_pcp8 = advisor.WorkloadSpec(
        phase="prefill", batch=1, isl=4096, chunk_size=512, tp=8, ep=8, pcp=8
    )
    rep_pcp1 = advisor.advise("kimi-k3", wl_pcp1, "tpu7x")
    rep_pcp8 = advisor.advise("kimi-k3", wl_pcp8, "tpu7x")
    ws_pcp1 = next(s for s in rep_pcp1.scopes if s.scope == "per_op")
    ws_pcp8 = next(s for s in rep_pcp8.scopes if s.scope == "per_op")
    assert ws_pcp8.step_bound_us != ws_pcp1.step_bound_us

    bad_ep = advisor.WorkloadSpec(phase="decode", batch=4, tp=8, ep=7)
    reasons_ep = advisor.validate_workload("qwen3.5-397b", "tpu7x", bad_ep)
    assert "experts_not_divisible_by_ep" in reasons_ep

    bad_tp = advisor.WorkloadSpec(phase="decode", batch=4, tp=6, ep=1)
    reasons_tp = advisor.validate_workload("qwen3.5-397b", "tpu7x", bad_tp)
    assert "tp_not_power_of_two" in reasons_tp

    oom_wl = advisor.WorkloadSpec(phase="decode", batch=512, context=131072, tp=8, ep=8)
    reasons_oom = advisor.validate_workload("qwen3.5-397b", "tpu7x", oom_wl)
    assert "hbm_pool_overflow" in reasons_oom


def test_dispatch_regimes_and_compiler_proof_gain_split(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify zero_floor, compiled_xla, and unfused_eager dispatch regime metrics."""
    rep_zero = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode", batch=16, tp=8, ep=8, dispatch_regime="zero_floor"
        ),
        "tpu7x",
    )
    rep_xla = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode", batch=16, tp=8, ep=8, dispatch_regime="compiled_xla"
        ),
        "tpu7x",
    )
    rep_eager = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode", batch=16, tp=8, ep=8, dispatch_regime="unfused_eager"
        ),
        "tpu7x",
    )
    assert rep_zero.metrics["launch_gap_us"]["tag"] == "DERIVED"
    assert rep_zero.metrics["dispatch_bubble_gain_us"]["value"] == 0.0
    assert rep_zero.metrics["compiler_proof_gain_us"]["value"] > 0.0
    assert rep_xla.metrics["launch_gap_us"]["tag"] == "ASSUMED"
    assert (
        rep_eager.metrics["dispatch_bubble_gain_us"]["value"]
        > rep_xla.metrics["dispatch_bubble_gain_us"]["value"]
    )
    rc = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--tp",
            "8",
            "--ep",
            "8",
            "--batch",
            "16",
            "--dispatch-regime",
            "compiled_xla",
            "--json",
        ]
    )
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["metrics"]["launch_gap_us"]["tag"] == "ASSUMED"
    assert "compiler_proof_gain_us" in cli_payload["metrics"]


def test_no_fused_recommendation_outside_core_limits() -> None:
    """Fused scopes outside core or declared limits stay gated to per_op."""
    for model_name in (
        "kimi-k3.1-pro",
        "glm-5.2",
        "qwen3.8-2.4t",
        "deepseek-v4",
        "qwen4-exp",
    ):
        sweep = advisor.advise_regimes(model_name, "tpu7x")
        assert all(r.recommended_scope == "per_op" for r in sweep.regimes)
        dec_small = sweep.regimes[0]
        assert dec_small.fastest_scope == "whole_step"
        assert dec_small.hypothetical_best is not None

    k3_declared_max_rows = 128
    k3_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    for entry in k3_sweep.regimes:
        if (
            entry.phase in ("decode", "verify")
            and entry.active_tokens <= k3_declared_max_rows
        ):
            assert entry.recommended_scope == "whole_step"
            assert entry.evidence == "declared"
        else:
            assert entry.recommended_scope == "per_op"

    q_single_call_rows = 112
    q_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    for entry in q_sweep.regimes:
        if (
            entry.layout == "TP8 x EP8 x DCP4"
            and entry.active_tokens <= q_single_call_rows
        ):
            assert entry.recommended_scope == "whole_step"
            assert entry.evidence in ("within_core_limits", "declared")
        else:
            assert entry.recommended_scope == "per_op"


def test_cli_invalid_sharding_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid sharding exits with code 2 and prints invalid_config verdict."""
    rc = advisor.main(["--model", "qwen3.5-397b", "--chip", "tpu7x", "--tp", "7"])
    assert rc == constants.INVALID_CONFIG_EXIT_CODE
    out = capsys.readouterr().out
    assert "invalid_config(" in out


def test_cli_prints_evidence_blockers_and_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI summary prints Evidence, Blocked by, and Inputs."""
    rc = advisor.main(["--model", "deepseek-v4", "--tp", "16", "--explain"])
    assert rc == 0
    out = capsys.readouterr().out
    for token in (
        "Recommended: per_op",
        "Evidence:",
        "Blocked by:",
        "Inputs:",
    ):
        assert token in out


def test_spec_verify_saturated_regime_matches_dp4_mtp3_crossover() -> None:
    """Spec Verify Saturated (TP2 x DP4 x EP8, K=3) recommends per_op at 41582.9 us."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    by_label = {r.label: r for r in sweep.regimes}
    label = "Spec Verify Saturated (B=600 over DP4, K=3, 32K)"
    assert label in by_label
    entry = by_label[label]
    assert entry.layout == "TP2 x DP4 x EP8"
    assert entry.recommended_scope == "per_op"
    per_op_sc = next(s for s in entry.scopes if s["scope"] == "per_op")
    # Per-op pays 120 all-reduces x 32.768 us more on one link than on the egress sum.
    sat_verify_floor_us = 41582.94
    assert per_op_sc["step_bound_us"] == pytest.approx(sat_verify_floor_us, abs=0.2)


def test_cli_max_rows_per_call_prices_microbatches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --max-rows-per-call reaches the estimate and the regime sweep."""
    base = ["--model", "qwen3.5-397b", "--chip", "tpu7x"]
    mesh = ["--tp", "8", "--dcp", "4", "--ep", "8"]
    verify = ["--phase", "verify", "--batch", "28", "--spec-tokens", "3", "--json"]
    limit = ["--max-rows-per-call", str(constants.DEFAULT_ROW_LIMIT)]
    assert advisor.main([*base, *mesh, *verify, *limit]) == 0
    split = json.loads(capsys.readouterr().out)
    whole = next(s for s in split["scopes"] if s["scope"] == "whole_step")
    assert whole["t_vmem_max"] == constants.DEFAULT_ROW_LIMIT
    assert whole["n_splits"] > 1
    assert "microbatch_rerun" in split["flags"]

    assert advisor.main([*base, *mesh, *verify]) == 0
    assert "microbatch_rerun" not in json.loads(capsys.readouterr().out)["flags"]

    assert advisor.main([*base, *mesh, "--regime-sweep", "--json", *limit]) == 0
    assert (
        json.loads(capsys.readouterr().out)["max_vmem_tokens"]
        == constants.DEFAULT_ROW_LIMIT
    )


@pytest.mark.usefixtures("without_declared_megakernels")
def test_regime_entry_uses_recommended_scope_not_hypothetical() -> None:
    """Verify RegimeEntry tracks recommended_scope with hypothetical kept separate."""
    k3_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    dec_small = next(r for r in k3_sweep.regimes if r.label.startswith("Decode Small"))
    assert dec_small.recommended_scope == "per_op"
    assert dec_small.fastest_scope == "whole_step"
    per_op_sc = next(s for s in dec_small.scopes if s["scope"] == "per_op")
    whole_sc = next(s for s in dec_small.scopes if s["scope"] == "whole_step")
    assert dec_small.step_bound_us == per_op_sc["step_bound_us"]
    assert dec_small.step_upper_us == per_op_sc["step_upper_us"]
    assert dec_small.max_vmem_tokens == per_op_sc["t_vmem_max"]
    assert dec_small.hypothetical_best is not None
    assert dec_small.hypothetical_best["step_bound_us"] == whole_sc["step_bound_us"]

    q_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    expected_q35_rows = 448
    assert q_sweep.max_vmem_tokens == expected_q35_rows


def test_recommend_parallelism_raises_when_no_layout_fits() -> None:
    """Verify recommend_parallelism rejects models that OOM at max TP candidate."""
    model_profile = advisor.resolve_model_profile("qwen3.8-2.4t")
    tiny_chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        hbm_capacity_bytes=1024 * 1024,
    )
    with pytest.raises(MegakernelConfigError, match="fits in HBM"):
        advisor.recommend_parallelism(model_profile, tiny_chip)

    v6e_spec = advisor.resolve_chip_spec("v6e")
    par_v6e = advisor.recommend_parallelism(
        advisor.resolve_model_profile("qwen4-exp"), v6e_spec
    )
    v6e_cores = par_v6e.tp * par_v6e.dp * par_v6e.pp
    assert v6e_cores in v6e_spec.topologies
    v6e_sweep = advisor.advise_regimes("qwen4-exp", "v6e")
    assert not v6e_sweep.regimes[0].verdict.startswith("invalid_config(")

    dp2_sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(tp=4, dp=2),
    )
    expected_regimes_count = 12  # 3 phases x 4 context depths
    assert len(dp2_sweep.regimes) == expected_regimes_count
    for reg in dp2_sweep.regimes:
        assert not reg.verdict.startswith("invalid_config(")
        assert int(str(reg.workload["batch"])) >= int(str(reg.workload["dp"]))


def test_no_private_underscore_module_constants() -> None:
    """Verify advisor/*.py defines zero private `_`-prefixed module constants."""
    adv_dir = Path(advisor.__file__).resolve().parent
    offenders: list[str] = []
    for py_path in sorted(adv_dir.glob("*.py")):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                    and target.id.startswith("_")
                    and not target.id.startswith("__")
                ]
                offenders.extend(f"{py_path.name}:{name}" for name in names)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id.startswith("_")
                and not node.target.id.startswith("__")
            ):
                offenders.append(f"{py_path.name}:{node.target.id}")
    assert offenders == []


NO_LAYER_GROUPING_REASON = (
    "No layer grouping: needs a model preset, pp > 1, or segment_layers"
)


@pytest.mark.parametrize(
    "grouping",
    [
        {"layer_groups": (8, 8, 8, 8, 8, 8, 8, 4)},
        {"segment_layers": 15},
        {"pp": 2},
    ],
    ids=["layer_groups", "segment_layers", "pp2"],
)
def test_layer_group_infeasible_without_declared_grouping(
    grouping: dict[str, object],
) -> None:
    """layer_group is infeasible with no grouping and feasible once one is declared."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    base = advisor.WorkloadSpec(
        phase="decode", batch=16, context=4096, tp=8, dcp=4, ep=8
    )
    ungrouped = advisor.advise(model_profile, base, "tpu7x")
    ungrouped_scope = next(s for s in ungrouped.scopes if s.scope == "layer_group")
    assert ungrouped_scope.feasible is False
    assert ungrouped_scope.reason == NO_LAYER_GROUPING_REASON

    grouped = advisor.advise(
        model_profile, dataclasses.replace(base, **grouping), "tpu7x"
    )
    grouped_scope = next(s for s in grouped.scopes if s.scope == "layer_group")
    assert grouped_scope.feasible is True
    assert grouped_scope.reason != NO_LAYER_GROUPING_REASON


@pytest.mark.usefixtures("without_declared_megakernels")
@pytest.mark.parametrize(
    "layer_groups", [(), (8, 8, 8, 8, 8, 8, 8, 4)], ids=["ungrouped", "groups_of_8"]
)
def test_max_rows_per_call_caps_whole_step_only(
    layer_groups: tuple[int, ...],
) -> None:
    """max_rows_per_call lowers whole_step rows per call; other scopes stay as is."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    uncapped_wl = advisor.WorkloadSpec(
        phase="verify",
        batch=32,
        spec_tokens=3,
        context=32768,
        tp=8,
        dcp=4,
        ep=8,
        layer_groups=layer_groups,
    )
    capped_wl = dataclasses.replace(
        uncapped_wl, max_rows_per_call=constants.DEFAULT_ROW_LIMIT
    )
    uncapped = {
        s.scope: s for s in advisor.advise(model_profile, uncapped_wl, "tpu7x").scopes
    }
    capped_rep = advisor.advise(model_profile, capped_wl, "tpu7x")
    capped = {s.scope: s for s in capped_rep.scopes}

    assert uncapped["whole_step"].t_vmem_max > constants.DEFAULT_ROW_LIMIT
    assert capped["whole_step"].t_vmem_max == constants.DEFAULT_ROW_LIMIT
    assert capped_rep.active_tokens == 32 * (1 + 3)
    expected_splits = 2
    assert capped["whole_step"].n_splits == expected_splits
    assert capped["whole_step"].n_splits > uncapped["whole_step"].n_splits
    for scope in ("per_op", "single_layer", "layer_group"):
        assert capped[scope].t_vmem_max == uncapped[scope].t_vmem_max
        assert capped[scope].n_splits == uncapped[scope].n_splits
        assert capped[scope].step_bound_us == uncapped[scope].step_bound_us
        assert capped[scope].feasible == uncapped[scope].feasible


def test_advise_both_objective_keeps_not_comparable() -> None:
    """advise() reports the layout comparison when the chip counts differ."""
    # TP16 x EP16 needs 8 chips; the TP2 x DP4 x EP8 saturation preset needs 4.
    workload = advisor.WorkloadSpec(
        phase="decode", batch=16, context=4096, tp=16, ep=16, objective="both"
    )
    report = advisor.advise("qwen3.5-397b", workload, "tpu7x")
    assert "layout_not_comparable" in report.flags
    assert "objective_split" not in report.flags
    assert (
        "; step not comparable: TP16 x EP16 uses 8 chips, TP2 x DP4 x EP8 uses "
        "4 chips; best_throughput_layout=TP2 x DP4 x EP8"
    ) in report.rationale
    # The fusion-scope verdict matches the step objective, which skips the layout check.
    step_rep = advisor.advise(
        "qwen3.5-397b", dataclasses.replace(workload, objective="step"), "tpu7x"
    )
    assert (report.verdict, report.fastest_verdict, report.recommended_scope) == (
        step_rep.verdict,
        step_rep.fastest_verdict,
        step_rep.recommended_scope,
    )


def test_recommend_parallelism_multi_axis_dcp() -> None:
    """Verify recommend_parallelism auto-derives dcp=1 when viable.

    Engages dcp>1 when required.
    """
    profile = advisor.resolve_model_profile("qwen3.5-397b")
    expected_tp = 8
    expected_ep = 8
    expected_dcp_default = 1
    rec_default = advisor.recommend_parallelism(profile, "tpu7x", dcp=0)
    assert rec_default.tp == expected_tp
    assert rec_default.ep == expected_ep
    assert rec_default.dcp == expected_dcp_default

    expected_dcp_explicit = 4
    rec_explicit = advisor.recommend_parallelism(
        profile, "tpu7x", dcp=expected_dcp_explicit
    )
    assert rec_explicit.tp == expected_tp
    assert rec_explicit.ep == expected_ep
    assert rec_explicit.dcp == expected_dcp_explicit

    kv2_profile = dataclasses.replace(
        profile,
        num_kv_heads=2,
    )
    rec_kv2 = advisor.recommend_parallelism(kv2_profile, "tpu7x", dcp=0)
    assert rec_kv2.dcp >= 1


def test_cli_and_html_render_layer_group_search(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Verify `--search-layer-groups` CLI table/JSON and HTML card render."""
    rc = advisor.main(
        [
            "--model",
            "kimi-k3",
            "--phase",
            "prefill",
            "--isl",
            "512",
            "--tp",
            "32",
            "--ep",
            "16",
            "--pcp",
            "32",
            "--search-layer-groups",
        ]
    )
    assert rc == 0
    stdout_text = capsys.readouterr().out
    assert "Layer-Group Partition Search: kimi-k3" in stdout_text
    assert "one fused kernel per group" in stdout_text

    json_out = tmp_path / "lg_search.json"
    rc_json = advisor.main(
        [
            "--model",
            "kimi-k3",
            "--phase",
            "prefill",
            "--isl",
            "512",
            "--tp",
            "32",
            "--ep",
            "16",
            "--pcp",
            "32",
            "--search-layer-groups",
            "--json",
            str(json_out),
        ]
    )
    assert rc_json == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["best"]["num_groups"] == 1

    sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    html_text = advisor.render_advisor_html(sweep)
    assert "7. Layer-Group Partition Search" in html_text
