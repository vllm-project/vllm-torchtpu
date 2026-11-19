# SPDX-License-Identifier: Apache-2.0
"""q38-l4-vfy-v5 (VFY-ONE x DRAFT-ASM composed) flag off, AST for AST (torch-free; CDK CPU pod only). VFY-BUILD.

With VLLM_TPU_DRAFT_ASM_FOLD and VLLM_TPU_VPS_VFY_ONE both off, eagle3.py and tpu_runner.py resolve (DRAFT-ASM's
tests/q38_draftasm/draftasm_flagoff_ast.off, then tests/l4vfy/vfy_flagoff_ast.off) to the composed base's
q38-l4-vfy-v5base 8344dce32 (= q38-L1-v3c b551e9b84 + D83S + q38-l4-stack-d1, i.e. the composed tree minus the
DRAFT-ASM and VFY-ONE commits), AST for AST; the check bites on an unguarded VFY-ONE hook and on an unguarded
DRAFT-ASM hook. Replaces DRAFT-ASM's own 3d2f53ec9 identity tests on this tree (L4-E).

q38 L4 AGREE (q38-l4-agree): the runner also carries the agreement clock's three guarded statements (import,
VFY-ONE's registration, execute_model's tick); tests/q38_l4agree/l4agree_flagoff_ast.off resolves them between the
two lever resolvers. Assertions unchanged."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "q38_draftasm"))
sys.path.insert(0, os.path.join(HERE, "..", "q38_l4agree"))
import draftasm_flagoff_ast as da  # noqa: E402
import l4agree_flagoff_ast as ag  # noqa: E402
import vfy_flagoff_ast as fo  # noqa: E402
import vfy_harness as H  # noqa: E402

EAGLE = os.path.join(H.PKG, "spec_decode", "eagle3.py")
RUNNER = os.path.join(H.PKG, "runner", "tpu_runner.py")
H.LOADED["v5flagoff_eagle3"] = os.path.realpath(EAGLE)
H.LOADED["v5flagoff_runner"] = os.path.realpath(RUNNER)
EAGLE_SRC, RUNNER_SRC = open(EAGLE).read(), open(RUNNER).read()
EAGLE_REF = open(os.path.join(HERE, "ref_v5base_eagle3.py.txt")).read()
RUNNER_REF = open(os.path.join(HERE, "ref_v5base_tpu_runner.py.txt")).read()


def _da_ag(src):
    """DRAFT-ASM's resolver, then L4 AGREE's (VFY-ONE's runs last, in fo.check)."""
    return ag.off(src, extra=da.off)


def test_eagle3_both_flags_off_is_the_composed_base():
    assert EAGLE_SRC != EAGLE_REF
    d = fo.check(EAGLE_SRC, EAGLE_REF, extra=_da_ag)
    print(f"\nQ38_L4VFY_V5_FLAGOFF eagle3 {'IDENTICAL' if not d else d}")
    assert not d, d


def test_tpu_runner_both_flags_off_is_the_composed_base():
    assert RUNNER_SRC != RUNNER_REF
    d = fo.check(RUNNER_SRC, RUNNER_REF, extra=_da_ag)
    print(f"\nQ38_L4VFY_V5_FLAGOFF tpu_runner {'IDENTICAL' if not d else d}")
    assert not d, d


def test_composed_check_bites():
    vfy = ('                if self.__dict__.get("_vfy") is not None:\n'
           "                    # q38 L4 VFY-ONE: its verify programs beside the D83 chain's.\n"
           "                    self._vfy_precompile()\n")
    assert RUNNER_SRC.count(vfy) == 1
    assert fo.check(RUNNER_SRC.replace(vfy, "                self._vfy_precompile()\n"), RUNNER_REF, extra=_da_ag) == [
        "differs: TPUModelRunner.capture_model"]
    asm = "        if self._draft_asm is not None:\n            self._draft_asm.precompile()\n"
    assert EAGLE_SRC.count(asm) == 1
    assert fo.check(EAGLE_SRC.replace(asm, asm + "        self._precompile_draft_seed()\n"), EAGLE_REF, extra=_da_ag) == [
        "differs: Eagle3Proposer.precompile"]
    # each resolver alone leaves the other lever's code: both are needed on this tree
    assert fo.check(RUNNER_SRC, RUNNER_REF) and da.check(RUNNER_SRC, RUNNER_REF)
