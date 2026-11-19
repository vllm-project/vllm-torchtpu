# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT-CAP with its flag off (torch-free except the module reads):
1. eagle3.py, draft_asm.py, mtp_fused_loop.py, tpu_runner.py, mtp.py and tpu_compiler.py are 29ea1e98d's, byte for
   byte; envs.py is 29ea1e98d's byte for byte once the variable's two insertions are removed.
2. draft_one.py with DRAFT-CAP's code resolved to its flag-off arm (draftcap_flagoff_ast) is 29ea1e98d's, AST for
   AST; the edit touches exactly DraftOne.__init__ and DraftOne._one_ready; the check bites.
3. The variable defaults off, is registered and compile-hashed; the on-value carries no comma.
"""

import ast
import hashlib
import pathlib

import draftcap_flagoff_ast as fo
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.models.vllm.qwen4_exp import mtp
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.spec_decode import draft_asm, draft_one, eagle3, mtp_fused_loop

HERE = pathlib.Path(__file__).parent
D1_SRC = pathlib.Path(draft_one.__file__).read_text()
D1_REF = (HERE / "draft_one_ref_29ea1e98d.py.txt").read_text()
SHA_29EA1E98D = {
    eagle3: "a9b78766468516f62fe032fc4f025af83cb8a4de",
    draft_asm: "ce1478cd03237ee35602fe019d7d828d92438fae",
    mtp_fused_loop: "dbdd64c42106f23bd1598ab3ebd93212a052df86",
    tpu_runner: "bb6b1cf12a4c387ef121e953a557945e007ce3e4",
    mtp: "da1718403a31a3221ea955989cbedfacca3ae9a3",
    tpu_compiler: "1152ee7543098ef38a69a2e63aa8f72c0c8a1c1b",
}
ENVS_SHA_29EA1E98D = "4a7dd54ad90bcefddb730f64ab96f18061424e33"
ENVS_ANNOT = "    VLLM_TPU_DRAFT_ASM_CAP: bool = False\n"
ENVS_ENTRY_END = '    "VLLM_TPU_DRAFT_ASM_CAP": env_bool("VLLM_TPU_DRAFT_ASM_CAP"),\n'


def _sha1(text_or_bytes):
    b = text_or_bytes.encode() if isinstance(text_or_bytes, str) else text_or_bytes
    return hashlib.sha1(b).hexdigest()


def test_untouched_modules_are_byte_identical_to_29ea1e98d():
    for mod, want in SHA_29EA1E98D.items():
        got = _sha1(pathlib.Path(mod.__file__).read_bytes())
        print(f"\nDRAFTCAP_FLAGOFF {pathlib.Path(mod.__file__).name} sha1 {got[:10]} (29ea1e98d {want[:10]})")
        assert got == want, mod.__name__


def test_envs_is_29ea1e98d_plus_the_variable():
    src = pathlib.Path(envs.__file__).read_text()
    assert src.count(ENVS_ANNOT) == 1 and src.count(ENVS_ENTRY_END) == 1
    end = src.index(ENVS_ENTRY_END) + len(ENVS_ENTRY_END)
    start = src.rindex("    # q38 L4 DRAFT-CAP inside DH-D", 0, end)
    block = src[start:end]
    assert block.count("\n") == 8, block
    stripped = src.replace(ENVS_ANNOT, "", 1).replace(block, "", 1)
    got = _sha1(stripped)
    print(f"\nDRAFTCAP_FLAGOFF envs minus the variable sha1 {got[:10]} (29ea1e98d {ENVS_SHA_29EA1E98D[:10]})")
    assert got == ENVS_SHA_29EA1E98D


def test_draft_one_flag_off_is_29ea1e98d():
    assert D1_SRC != D1_REF
    d = fo.check(D1_SRC, D1_REF)
    t = fo.touched(D1_SRC, D1_REF)
    print(f"\nDRAFTCAP_FLAGOFF draft_one {'IDENTICAL' if not d else d}; functions touched {t}")
    assert not d, d
    assert t == ["DraftOne.__init__", "DraftOne._one_ready"], t


def test_flag_off_check_bites():
    anchor = "        bufs = self._one_buffers(md0)\n"
    assert D1_SRC.count(anchor) == 1
    bad = D1_SRC.replace(anchor, anchor + "        bufs = tuple(bufs)\n")
    assert fo.check(bad, D1_REF) == ["differs: DraftOne._one_ready"]
    unguarded = D1_SRC.replace("        if self.one_cap:\n", "        if self.one_cap_x:\n", 1)
    assert fo.check(unguarded, D1_REF) == ["differs: DraftOne._one_ready"]
    stray = D1_SRC.replace("        self.one_cap = ", "        self.one_cap_seen = 0\n        self.one_cap = ", 1)
    assert fo.check(stray, D1_REF) == ["differs: DraftOne.__init__"]
    ast.parse(ast.unparse(fo.off(D1_SRC)))


def test_flag_default_off_registered_and_keyed(monkeypatch):
    name = "VLLM_TPU_DRAFT_ASM_CAP"
    monkeypatch.delenv(name, raising=False)
    assert name in envs.environment_variables
    assert envs.VLLM_TPU_DRAFT_ASM_CAP is False
    assert name not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.setenv(name, "1")
    assert envs.VLLM_TPU_DRAFT_ASM_CAP is True
    assert "," not in "1"
