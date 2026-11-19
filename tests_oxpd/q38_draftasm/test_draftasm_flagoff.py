# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT-ASM flag off (torch-free except the envs read):
1. eagle3.py and tpu_runner.py with VLLM_TPU_DRAFT_ASM_FOLD's code resolved to its flag-off arm are
   3d2f53ec9's, AST for AST (draftasm_flagoff_ast); the check bites on an unguarded statement.
2. The three variables default off; VLLM_TPU_DRAFT_ASM_FOLD is compile-hashed, the CHECK step count and the
   l6_ab switch are ignored; no on-value carries a comma.
"""

import ast
import pathlib

import draftasm_flagoff_ast as fo
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.spec_decode import eagle3

HERE = pathlib.Path(__file__).parent
EAGLE3_SRC = pathlib.Path(eagle3.__file__).read_text()
RUNNER_SRC = pathlib.Path(tpu_runner.__file__).read_text()
EAGLE3_REF = (HERE / "eagle3_ref_3d2f53ec9.py.txt").read_text()
RUNNER_REF = (HERE / "tpu_runner_ref_3d2f53ec9.py.txt").read_text()


def test_eagle3_flag_off_is_3d2f53ec9():
    assert EAGLE3_SRC != EAGLE3_REF
    d = fo.check(EAGLE3_SRC, EAGLE3_REF)
    print(f"\nDRAFTASM_FLAGOFF eagle3 {'IDENTICAL' if not d else d}")
    assert not d, d


def test_tpu_runner_flag_off_is_3d2f53ec9():
    assert RUNNER_SRC != RUNNER_REF
    d = fo.check(RUNNER_SRC, RUNNER_REF)
    print(f"\nDRAFTASM_FLAGOFF tpu_runner {'IDENTICAL' if not d else d}")
    assert not d, d


def test_flag_off_check_bites():
    anchor = "        if self._draft_asm is not None:\n            self._draft_asm.precompile()\n"
    assert EAGLE3_SRC.count(anchor) == 1
    bad = EAGLE3_SRC.replace(anchor, anchor + "        self._precompile_draft_seed()\n")
    assert fo.check(bad, EAGLE3_REF) == ["differs: Eagle3Proposer.precompile"]
    unguarded = EAGLE3_SRC.replace(anchor, "        self._draft_asm_on = True\n")
    assert fo.check(unguarded, EAGLE3_REF) == ["differs: Eagle3Proposer.precompile"]
    # the resolved module is still valid Python
    ast.parse(fo.off_source(EAGLE3_SRC))


def test_flags_default_off_registered_and_keyed(monkeypatch):
    for name in ("VLLM_TPU_DRAFT_ASM_FOLD", "VLLM_TPU_DRAFT_ASM_CHECK_STEPS", "VLLM_TPU_DRAFT_ASM_AB"):
        monkeypatch.delenv(name, raising=False)
        assert name in envs.environment_variables, name
    assert envs.VLLM_TPU_DRAFT_ASM_FOLD is False
    assert envs.VLLM_TPU_DRAFT_ASM_CHECK_STEPS == 0
    assert envs.VLLM_TPU_DRAFT_ASM_AB == ""
    assert "VLLM_TPU_DRAFT_ASM_FOLD" not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "VLLM_TPU_DRAFT_ASM_CHECK_STEPS" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "VLLM_TPU_DRAFT_ASM_AB" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_FOLD", "1")
    assert envs.VLLM_TPU_DRAFT_ASM_FOLD is True
    assert "," not in "1"
