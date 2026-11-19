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


def test_pair_only_dcp_path_imported_whole_step():
    """The imported (MM 37b960ea9) whole step: DCP 2 with union runs; DCP 2 without union and DCP 4 are refused."""
    assert whole_step_allowed(True, 2, dcp_capable=True, pair_only=True, union=True) == (True, None)
    assert whole_step_allowed(True, 1, dcp_capable=True, pair_only=True, union=False) == (True, None)
    assert whole_step_allowed(True, 0, dcp_capable=True, pair_only=True, union=False) == (True, None)
    ok, why = whole_step_allowed(True, 2, dcp_capable=True, pair_only=True, union=False)
    assert not ok and "wholeunion=1" in why and "VLLM_TPU_QSA_MAIN_KV_DCP=2" in why
    ok, why = whole_step_allowed(True, 4, dcp_capable=True, pair_only=True, union=True)
    assert not ok and "DCP 2 only" in why and "VLLM_TPU_QSA_MAIN_KV_DCP=4" in why
    assert whole_step_allowed(False, 2, dcp_capable=True, pair_only=True, union=True) == (False, None)


def test_driver_applies_the_pair_only_guard():
    """The driver passes pair_only=True with wholeunion's UNION_ATTN (source check: CPU only, no TPU)."""
    import inspect

    from vllm_torchtpu.megakernel_runtime import qwen38_driver

    src = inspect.getsource(qwen38_driver)
    assert "dcp_capable=True, pair_only=True, union=_q38_wsc.UNION_ATTN" in src


def test_group_dcp_whole_step_runs_at_dcp4():
    """q38 L1-D14: with the DCP-4 group path (group_dcp), DCP 4 with union runs; DCP 4 without union, or with
    compressed-key DCP (pair-only), is refused; DCP 2 keeps its pair rules; without group_dcp DCP 4 is still
    refused, and DCP 8 is refused either way."""
    kw = dict(dcp_capable=True, pair_only=True)
    assert whole_step_allowed(True, 4, **kw, union=True, group_dcp=True) == (True, None)
    ok, why = whole_step_allowed(True, 4, **kw, union=False, group_dcp=True)
    assert not ok and "wholeunion=1" in why and "VLLM_TPU_QSA_MAIN_KV_DCP=4" in why
    ok, why = whole_step_allowed(True, 4, **kw, union=True, group_dcp=True, compressed=True)
    assert not ok and "VLLM_TPU_QSA_COMPRESSED_DCP" in why and "VLLM_TPU_QSA_MAIN_KV_DCP=4" in why
    assert whole_step_allowed(True, 2, **kw, union=True, group_dcp=True) == (True, None)
    assert whole_step_allowed(True, 2, **kw, union=True, group_dcp=True, compressed=True) == (True, None)
    ok, why = whole_step_allowed(True, 2, **kw, union=False, group_dcp=True)
    assert not ok and "VLLM_TPU_QSA_MAIN_KV_DCP=2" in why
    ok, why = whole_step_allowed(True, 4, **kw, union=True)
    assert not ok and "DCP 2 only" in why
    ok, why = whole_step_allowed(True, 8, **kw, union=True, group_dcp=True)
    assert not ok and "VLLM_TPU_QSA_MAIN_KV_DCP=8" in why
    assert whole_step_allowed(True, 1, **kw, union=False, group_dcp=True) == (True, None)
    assert whole_step_allowed(False, 4, **kw, union=True, group_dcp=True) == (False, None)


def test_driver_applies_the_group_dcp_guard():
    """q38 L1-D14: the driver passes group_dcp=True and the compressed-key DCP flag (source check: CPU only)."""
    import inspect

    from vllm_torchtpu.megakernel_runtime import qwen38_driver

    src = inspect.getsource(qwen38_driver)
    assert "group_dcp=True, compressed=_q38_wsc.compressed_dcp()" in src
