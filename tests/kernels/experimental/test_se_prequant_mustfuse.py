"""TPU_SE_PREQUANT_MUSTFUSE: mode 4's per-layer must-fuse x input, with MUST_FUSE
pushed onto the ops inside the xla_metadata_call callee of the exported module."""

import functools
import re

import jax
import jax.numpy as jnp
import pytest
from jax import export

from vllm_torchtpu.kernels.experimental.se_prequant import serving as S
from vllm_torchtpu.models.vllm import qwen3_next_patch as patch_mod

T, H, NI, NG = 512, 4096, 1024, 512
FP8, BF16, F32 = jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32
SE_SPECS = [((T, H), BF16), ((H,), F32), ((H, NG + 2 * NI), FP8), ((NG + 2 * NI,), F32),
            ((H,), BF16), ((NI, H), FP8), ((H,), F32)]


def _export_mode4(monkeypatch, fuse_id=7):
    monkeypatch.setenv("TPU_SE_PREQUANT_MUSTFUSE", "1")
    fn = functools.partial(S.norm_fp8_gate_and_se, eps=1e-6, mode=4, fuse_id=fuse_id)
    return export.export(jax.jit(fn), platforms=["tpu"],
                         disabled_checks=[export.DisabledSafetyCheck.custom_call("ALL")])(
        *[jax.ShapeDtypeStruct(s, d) for s, d in SE_SPECS])


def _callee_ops(text, callee):
    """Lines of the private func `callee` in an MLIR module text."""
    lines, on = [], False
    for ln in text.splitlines():
        if ln.strip().startswith("func.func") and f"@{callee}(" in ln:
            on = True; continue
        if on:
            if ln.strip().startswith("}"):
                break
            lines.append(ln.strip())
    return lines


def test_x_input_flag(monkeypatch):
    monkeypatch.delenv("TPU_SE_PREQUANT_MUSTFUSE", raising=False)
    off = S._x_input(4, 7, False)
    assert "xin" not in off and off["vmem_limit"] == S.MODE4_VMEM  # bf16 x, as before
    monkeypatch.setenv("TPU_SE_PREQUANT_MUSTFUSE", "1")
    on = S._x_input(4, 7, False)
    assert on == {"vmem_limit": S.MODE4_VMEM, "xin": "mustfuse_cvt", "fuse_x": True, "fuse_id": "7"}
    assert S._x_input(4, None, False) == off  # no layer id: unchanged
    assert S._x_input(2, 7, False) == {"xin": "mustfuse_cvt", "fuse_x": True, "fuse_id": "7"}  # mode 2 as before


def test_push_must_fuse_inward_tags_every_callee_op(monkeypatch):
    exp = _export_mode4(monkeypatch)
    text0 = exp.mlir_module()
    assert text0.count("MUST_FUSE") == 1  # only the call site, which torch_tpu drops
    new = patch_mod._push_must_fuse_inward(exp)
    text = new.mlir_module()
    assert new.out_avals == exp.out_avals and new.in_avals == exp.in_avals
    callee = [re.search(r"= call @([\w.]+)\(", ln).group(1) for ln in text.splitlines()
              if re.search(r"= call @[\w.]+\(", ln) and "MUST_FUSE" in ln]
    assert len(callee) == 1, callee
    name = callee[0]
    ops = [ln for ln in _callee_ops(text, name) if ln.startswith("%")]
    kinds = {ln.split("=", 1)[1].strip().split(" ")[0].strip('"') for ln in ops}
    assert {"stablehlo.convert", "stablehlo.custom_call"} <= kinds, kinds
    assert ops and all('MUST_FUSE = "7"' in ln for ln in ops), ops
    assert text.count("MUST_FUSE") == 1 + len(ops)


def test_push_must_fuse_inward_fails_closed_without_a_group(monkeypatch):
    monkeypatch.delenv("TPU_SE_PREQUANT_MUSTFUSE", raising=False)
    fn = functools.partial(S.norm_fp8_gate_and_se, eps=1e-6, mode=4, fuse_id=7)  # bf16 x: no group
    exp = export.export(jax.jit(fn), platforms=["tpu"],
                        disabled_checks=[export.DisabledSafetyCheck.custom_call("ALL")])(
        *[jax.ShapeDtypeStruct(s, d) for s, d in SE_SPECS])
    assert "MUST_FUSE" not in exp.mlir_module()
    with pytest.raises(RuntimeError, match="no MUST_FUSE call site"):
        patch_mod._push_must_fuse_inward(exp)


def test_context_wraps_export_and_restores(monkeypatch):
    import jax.export as jexp
    orig = jexp.export
    seen = {}
    monkeypatch.setattr(patch_mod, "_push_must_fuse_inward", lambda e: ("pushed", e))
    with patch_mod._must_fuse_inner_export():
        assert jexp.export is not orig
        f = jexp.export(jax.jit(lambda x: x + 1), platforms=["cpu"])
        seen["out"] = f(jax.ShapeDtypeStruct((4,), F32))
    assert jexp.export is orig
    assert seen["out"][0] == "pushed"
    with pytest.raises(ValueError):
        with patch_mod._must_fuse_inner_export():
            raise ValueError("boom")
    assert jexp.export is orig


@pytest.mark.parametrize("flag", [False, True])
def test_builder_wraps_only_per_layer_ops_with_the_flag(monkeypatch, flag):
    import jax.export as jexp
    orig = jexp.export
    built = []

    class FakeOp:
        def register_fake(self, f):
            pass

    def fake_jax_op(name, fn):
        built.append((name, jexp.export is not orig))
        return FakeOp()

    monkeypatch.setattr(patch_mod.pallas, "jax_op", fake_jax_op)
    monkeypatch.setattr(patch_mod, "_norm_fp8_gate_and_se_ops", {})
    if flag:
        monkeypatch.setenv("TPU_SE_PREQUANT_MUSTFUSE", "1")
    else:
        monkeypatch.delenv("TPU_SE_PREQUANT_MUSTFUSE", raising=False)
    monkeypatch.setenv("TPU_SE_PREQUANT_KERNEL", "4")
    patch_mod._get_norm_fp8_gate_and_se_op(1e-6, 7)              # mode 4, per layer
    patch_mod._get_norm_fp8_gate_and_se_op(1e-6, 7, prequant=2)  # its mode-2 fallback, per layer
    patch_mod._get_norm_fp8_gate_and_se_op(1e-6, None)           # no layer id
    names = [n for n, _ in built]
    wrapped = [w for _, w in built]
    if flag:
        assert wrapped == [True, True, False]
        assert names[0].endswith("_pq4_f7_mfi") and names[1].endswith("_pq2_f7_mfi")
        assert not names[2].endswith("_mfi")
    else:
        assert wrapped == [False, False, False]
        assert not any(n.endswith("_mfi") for n in names)
    assert jexp.export is orig
