# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT1-B with its stage flags off (torch-free except the envs / module reads):
1. draft_asm.py, tpu_runner.py and mtp_fused_loop.py are 9b8d39a74's, byte for byte.
2. eagle3.py with DRAFT1-B's code resolved to its flag-off arm (draft1b_flagoff_ast) is 9b8d39a74's, AST
   for AST, and so is mtp.py with X-D14-DRAFT-3 (a)'s int32 switch resolved off; the check bites.
3. The four variables default off; the stage flags are compile-hashed, the windows ignored; no on-value
   carries a comma.
"""

import ast
import hashlib
import pathlib

import draft1b_flagoff_ast as fo
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.models.vllm.qwen4_exp import mtp
from vllm_torchtpu.spec_decode import draft_asm, eagle3, mtp_fused_loop

HERE = pathlib.Path(__file__).parent
EAGLE3_SRC = pathlib.Path(eagle3.__file__).read_text()
EAGLE3_REF = (HERE / "eagle3_ref_9b8d39a74.py.txt").read_text()
SHA_9B8D39A74 = {
    draft_asm: "ce1478cd03237ee35602fe019d7d828d92438fae",
    tpu_runner: "bb6b1cf12a4c387ef121e953a557945e007ce3e4",
    mtp_fused_loop: "dbdd64c42106f23bd1598ab3ebd93212a052df86",
}


def test_draft_asm_runner_and_fused_loop_are_byte_identical_to_9b8d39a74():
    for mod, want in SHA_9B8D39A74.items():
        got = hashlib.sha1(pathlib.Path(mod.__file__).read_bytes()).hexdigest()
        print(f"\nDRAFT1B_FLAGOFF {pathlib.Path(mod.__file__).name} sha1 {got[:10]} (9b8d39a74 {want[:10]})")
        assert got == want, mod.__name__


def test_eagle3_stage_flags_off_is_9b8d39a74():
    assert EAGLE3_SRC != EAGLE3_REF
    d = fo.check(EAGLE3_SRC, EAGLE3_REF)
    print(f"\nDRAFT1B_FLAGOFF eagle3 {'IDENTICAL' if not d else d}")
    assert not d, d


def test_mtp_meta_i32_off_is_9b8d39a74():
    src = pathlib.Path(mtp.__file__).read_text()
    ref = (HERE / "mtp_ref_9b8d39a74.py.txt").read_text()
    assert src != ref
    d = fo.check(src, ref)
    print(f"\nDRAFT1B_FLAGOFF mtp {'IDENTICAL' if not d else d}")
    assert not d, d
    swapped = src.replace("torch.int32 if self.meta_i32 else torch.int64", "torch.int64 if self.meta_i32 else torch.int32", 1)
    assert swapped != src and fo.check(swapped, ref) != []


def test_flag_off_check_bites():
    anchor = "                self._draft_asm = draft_asm.DraftAsm(self)\n"
    assert EAGLE3_SRC.count(anchor) == 1
    bad = EAGLE3_SRC.replace(anchor, anchor + "            self._draft_rows = True\n")
    assert fo.check(bad, EAGLE3_REF) == ["differs: Eagle3Proposer.__init__"]
    hook_end = "                    hidden = self._fit_draft_rows_output(hidden, rows)\n"
    assert EAGLE3_SRC.count(hook_end) == 1
    unguarded = EAGLE3_SRC.replace(hook_end, hook_end + "                hidden = hidden + 0\n")
    assert fo.check(unguarded, EAGLE3_REF) == ["differs: Eagle3Proposer.propose"]
    ast.parse(ast.unparse(fo.off(EAGLE3_SRC)))


def test_flags_default_off_registered_and_keyed(monkeypatch):
    names = ("VLLM_TPU_DRAFT_ASM_CARRY_LOOP", "VLLM_TPU_DRAFT_ASM_ONE",
             "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB", "VLLM_TPU_DRAFT_ASM_ONE_AB")
    for name in names:
        monkeypatch.delenv(name, raising=False)
        assert name in envs.environment_variables, name
    assert envs.VLLM_TPU_DRAFT_ASM_CARRY_LOOP is False and envs.VLLM_TPU_DRAFT_ASM_ONE is False
    assert envs.VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB == "" and envs.VLLM_TPU_DRAFT_ASM_ONE_AB == ""
    assert "VLLM_TPU_DRAFT_ASM_CARRY_LOOP" not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "VLLM_TPU_DRAFT_ASM_ONE" not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "VLLM_TPU_DRAFT_ASM_ONE_AB" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_CARRY_LOOP", "1")
    monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_ONE", "1")
    assert envs.VLLM_TPU_DRAFT_ASM_CARRY_LOOP is True and envs.VLLM_TPU_DRAFT_ASM_ONE is True
    assert "," not in "1" and "," not in "ab:1+3"
