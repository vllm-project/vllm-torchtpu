# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VFY-ONE flag off, AST for AST (torch-free; CDK CPU pod only). VFY-BUILD.

eagle3.py and tpu_runner.py with VLLM_TPU_VPS_VFY_ONE's code resolved to its flag-off arm (tests/l4vfy/
vfy_flagoff_ast.py) are q38-l4-vfy f24edcc94's (= q38-L1-v3c b551e9b84 + D83S), AST for AST; the check bites on an
unguarded hook. Q38MM_SRC_ROOT points at a mutated copy."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vfy_flagoff_ast as fo  # noqa: E402
import vfy_harness as H  # noqa: E402

EAGLE = os.path.join(H.PKG, "spec_decode", "eagle3.py")
RUNNER = os.path.join(H.PKG, "runner", "tpu_runner.py")
H.LOADED["flagoff_eagle3"] = os.path.realpath(EAGLE)  # the origin plugin prints what was read
H.LOADED["flagoff_runner"] = os.path.realpath(RUNNER)
EAGLE_SRC, RUNNER_SRC = open(EAGLE).read(), open(RUNNER).read()
EAGLE_REF = open(os.path.join(HERE, "ref_f24edcc94_eagle3.py.txt")).read()
RUNNER_REF = open(os.path.join(HERE, "ref_f24edcc94_tpu_runner.py.txt")).read()


def test_eagle3_flag_off_is_f24edcc94():
    assert EAGLE_SRC != EAGLE_REF
    d = fo.check(EAGLE_SRC, EAGLE_REF)
    print(f"\nQ38_L4VFY_FLAGOFF eagle3 {'IDENTICAL' if not d else d}")
    assert not d, d


def test_tpu_runner_flag_off_is_f24edcc94():
    assert RUNNER_SRC != RUNNER_REF
    d = fo.check(RUNNER_SRC, RUNNER_REF)
    print(f"\nQ38_L4VFY_FLAGOFF tpu_runner {'IDENTICAL' if not d else d}")
    assert not d, d


def test_flag_off_check_bites():
    guarded = ('                if self.__dict__.get("_vfy") is not None:\n'
               "                    # q38 L4 VFY-ONE: its verify programs beside the D83 chain's.\n"
               "                    self._vfy_precompile()\n")
    assert RUNNER_SRC.count(guarded) == 1
    unguarded = RUNNER_SRC.replace(guarded, "                self._vfy_precompile()\n")
    assert fo.check(unguarded, RUNNER_REF) == ["differs: TPUModelRunner.capture_model"]
    route = "if not isinstance(local_logits, torch.Tensor) and self._vfy is not None:"
    assert RUNNER_SRC.count(route) == 1
    bare = RUNNER_SRC.replace(route, "if not isinstance(local_logits, torch.Tensor):")
    assert fo.check(bare, RUNNER_REF) == ["differs: TPUModelRunner._sample_spec_verify_vps"]
    block = 'if self.runner.__dict__.get("_vfy_seed") is not None:'
    assert EAGLE_SRC.count(block) == 1
    assert fo.check(EAGLE_SRC.replace(block, "if self.runner is not None:"), EAGLE_REF) == [
        "differs: Eagle3Proposer._prepare_draft_inputs"]
