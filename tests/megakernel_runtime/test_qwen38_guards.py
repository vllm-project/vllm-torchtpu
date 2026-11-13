# SPDX-License-Identifier: Apache-2.0
"""whole=1 under main-KV DCP is refused until the whole-step program has a DCP path. CPU only."""

from vllm_torchtpu.megakernel_runtime.qwen38_guards import whole_step_allowed


def test_whole_refused_under_dcp():
    ok, why = whole_step_allowed(True, 4)
    assert not ok and "DCP" in why and "VLLM_TPU_QSA_MAIN_KV_DCP=4" in why


def test_whole_allowed_without_dcp_or_when_capable():
    assert whole_step_allowed(True, 0) == (True, None)
    assert whole_step_allowed(True, 1) == (True, None)
    assert whole_step_allowed(True, 4, dcp_capable=True) == (True, None)


def test_not_requested():
    assert whole_step_allowed(False, 4) == (False, None)


def test_driver_passes_dcp2_union_as_capable():
    """The driver marks whole-step DCP-capable only at main-KV DCP-2 with wholeunion=1."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).parents[2] / "src/vllm_torchtpu/megakernel_runtime/qwen38_driver.py"
    text = src.read_text()
    assert 'dcp_capable=main_kv_dcp == 2 and _parse_flag(raw, "wholeunion", False)' in text
    ast.parse(text)
    assert whole_step_allowed(True, 2, dcp_capable=True) == (True, None)
    assert not whole_step_allowed(True, 4, dcp_capable=False)[0]
