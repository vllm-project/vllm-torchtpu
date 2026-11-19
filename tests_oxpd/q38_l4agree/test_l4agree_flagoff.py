# SPDX-License-Identifier: Apache-2.0
"""q38 L4 AGREE flag-off identity and import origin (CDK CPU pod only). AGREE-BUILD (helper of L4-E).

1. AST identity. With L4 AGREE's three guarded runner statements resolved off (tests/q38_l4agree/
   l4agree_flagoff_ast.py) and KEYLITE's one block in _vfy_verify put back to the full-key test it replaces,
   tpu_runner.py is q38-l4-vfy-v5 2e166829a's, AST for AST; eagle3.py is 2e166829a's byte for
   byte (sha256 pinned). With DRAFT-ASM's, L4 AGREE's and VFY-ONE's resolvers (both flags off), tpu_runner.py and
   eagle3.py are the composed base q38-l4-vfy-v5base 8344dce32's (tests/l4vfy/ref_v5base_*.txt). The check bites on an
   unguarded tick and on an unguarded registration.
2. Behaviour with the flags off. The runner's real statements (ast-extracted): with no VFY-ONE controller the
   registration builds no Agreement, and with no Agreement execute_model's tick does nothing; with a controller they
   build one Agreement holding "vfyone", and the tick runs once per execute_model.
3. Import origin (lead rule): the run imports the tree under test (Q38_L4AGREE_EXPECT_ROOT), not an installed copy.
"""

import ast
import hashlib
import os
import pathlib
import sys
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "l4vfy"))
sys.path.insert(0, os.path.join(HERE, "..", "q38_draftasm"))
import draftasm_flagoff_ast as da  # noqa: E402
import l4agree_flagoff_ast as ag  # noqa: E402
import vfy_flagoff_ast as fo  # noqa: E402

ROOT = os.environ.get("Q38MM_SRC_ROOT") or os.path.join(HERE, "..", "..", "src")
PKG = os.path.join(ROOT, "vllm_torchtpu")
RUNNER = os.path.join(PKG, "runner", "tpu_runner.py")
EAGLE = os.path.join(PKG, "spec_decode", "eagle3.py")
RUNNER_SRC, EAGLE_SRC = open(RUNNER).read(), open(EAGLE).read()
REF_2E = open(os.path.join(HERE, "ref_2e166829a_tpu_runner.py.txt")).read()
V5_RUNNER = open(os.path.join(HERE, "..", "l4vfy", "ref_v5base_tpu_runner.py.txt")).read()
V5_EAGLE = open(os.path.join(HERE, "..", "l4vfy", "ref_v5base_eagle3.py.txt")).read()
EAGLE_2E_SHA256 = "16d9c9bdb620083a18b470e08739e3c995d6a972da427576e773540b958e3c6c"

KEYLITE_NEW = (
    "        # q38 L4 KEYLITE (VLLM_TPU_VPS_VFY_ONE_KEYLITE): the warm verdict from the scheduler integers.\n"
    "        lite = None\n"
    "        if vfy.keylite:\n"
    "            lite = _vfy_one.lite_key(chunk.hidden, chunk.indices, padded_num_reqs, md, fallback_rows, ro[0], spec)\n"
    "        if not vfy.warm_check(call, lite, mode == \"check\"):\n")
KEYLITE_OLD = "        if _vfy_one.key(*call[1:]) not in vfy.warm:\n"


def _keylite_off(src):
    """KEYLITE's block in _vfy_verify, exactly once, put back to the full-key test it replaces."""
    assert src.count(KEYLITE_NEW) == 1, src.count(KEYLITE_NEW)
    return ast.parse(src.replace(KEYLITE_NEW, KEYLITE_OLD))


TICK = ('        _agree = self.__dict__.get("_l4_agree")\n'
        "        if _agree is not None:\n")
REG = ("        if self._vfy is not None:\n"
       "            # q38 L4 AGREE (runner/l4_agree.py): VFY-ONE agrees on the runner's engine-step clock.\n"
       "            _l4agree.register(self, self._vfy)\n")


def _chain(src):
    return ag.off(src, extra=da.off)


def test_tpu_runner_agree_off_is_2e166829a():
    assert RUNNER_SRC != REF_2E
    d = ag.check(RUNNER_SRC, REF_2E, extra=_keylite_off)
    print(f"\nQ38_L4AGREE_FLAGOFF tpu_runner vs 2e166829a {'IDENTICAL' if not d else d}")
    assert not d, d


def test_eagle3_is_2e166829a_byte_for_byte():
    got = hashlib.sha256(EAGLE_SRC.encode()).hexdigest()
    print(f"\nQ38_L4AGREE_FLAGOFF eagle3 sha256 {got[:16]} (2e166829a {EAGLE_2E_SHA256[:16]})")
    assert got == EAGLE_2E_SHA256


def test_all_flags_off_is_the_composed_base():
    dr = fo.check(RUNNER_SRC, V5_RUNNER, extra=_chain)
    de = fo.check(EAGLE_SRC, V5_EAGLE, extra=_chain)
    print(f"\nQ38_L4AGREE_FLAGOFF composed tpu_runner {'IDENTICAL' if not dr else dr} eagle3 {'IDENTICAL' if not de else de}")
    assert not dr and not de, (dr, de)


def test_the_check_bites_on_an_unguarded_tick_or_registration():
    assert RUNNER_SRC.count(TICK) == 1 and RUNNER_SRC.count(REG) == 1
    unguarded_tick = RUNNER_SRC.replace(TICK, '        _agree = self.__dict__.get("_l4_agree")\n        if True:\n')
    assert ag.check(unguarded_tick, REF_2E, extra=_keylite_off) == ["differs: TPUModelRunner.execute_model"]
    unguarded_reg = RUNNER_SRC.replace(REG, "        _l4agree.register(self, self._vfy)\n")
    assert ag.check(unguarded_reg, REF_2E, extra=_keylite_off) == ["differs: TPUModelRunner._init_speculative_decoding"]
    # and KEYLITE's block is its only change to _vfy_verify
    assert ag.check(RUNNER_SRC, REF_2E) == ["differs: TPUModelRunner._vfy_verify"]
    # without the agree resolver the composed identity fails exactly on the agree statements
    assert fo.check(RUNNER_SRC, V5_RUNNER, extra=da.off) == ["differs: TPUModelRunner.execute_model",
                                                             "module-level statements differ"]


# ---------------------------------------------------------------- 2. behaviour, flags off / on


def _stmts(fn_name, pred):
    tree = ast.parse(RUNNER_SRC)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "TPUModelRunner")
    fn = next(n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    nodes = [n for n in fn.body if pred(n)]
    src = "".join(textwrap.dedent(ast.get_source_segment(RUNNER_SRC, n, padded=True)) + "\n" for n in nodes)
    return len(nodes), compile(src, f"<{fn_name}>", "exec")


N_REG, REG_CODE = _stmts("_init_speculative_decoding", lambda n: isinstance(n, ast.If)
                         and ast.unparse(n.test) == "self._vfy is not None"
                         and "_l4agree.register" in ast.unparse(n))
N_TICK, TICK_CODE = _stmts("execute_model", lambda n: (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "_agree")
                           or (isinstance(n, ast.If) and ast.unparse(n.test) == "_agree is not None"))


def test_flags_off_build_no_agreement_and_the_tick_does_nothing():
    from vllm_torchtpu.runner import l4_agree, vfy_one

    assert N_REG == 1 and N_TICK == 2
    me = types.SimpleNamespace(_vfy=None)
    exec(REG_CODE, {"_l4agree": l4_agree}, {"self": me})
    assert "_l4_agree" not in me.__dict__
    exec(TICK_CODE, {}, {"self": me})  # no agreement: nothing runs, nothing is created
    assert "_l4_agree" not in me.__dict__
    # importing the module creates no agreement and no per-process state
    assert not [k for k, v in vars(l4_agree).items() if isinstance(v, l4_agree.Agreement)]
    # flag on: one Agreement with VFY-ONE, ticked once per execute_model
    me = types.SimpleNamespace(_vfy=vfy_one.VfyOne(1, 0, ""), warmup_done=True)
    exec(REG_CODE, {"_l4agree": l4_agree}, {"self": me})
    agree = me._l4_agree
    assert isinstance(agree, l4_agree.Agreement) and [s.name for s in agree.slots] == ["vfyone"]
    assert me._vfy.agreement is agree
    ticks = []
    agree.tick = lambda runner: ticks.append(runner)
    exec(TICK_CODE, {}, {"self": me})
    assert ticks == [me]


# ---------------------------------------------------------------- 3. import origin


def test_imports_the_staged_tree():
    """Lead rule: the run imports the tree under test, not an installed copy."""
    want = os.environ.get("Q38_L4AGREE_EXPECT_ROOT", "")
    assert want, "Q38_L4AGREE_EXPECT_ROOT must name the src root under test"
    import vllm_torchtpu
    from vllm_torchtpu.runner import l4_agree, tpu_runner, vfy_one
    from vllm_torchtpu.spec_decode import draft_asm

    root = pathlib.Path(want).resolve()
    for mod in (vllm_torchtpu, l4_agree, vfy_one, draft_asm, tpu_runner):
        got = pathlib.Path(mod.__file__).resolve()
        assert root in got.parents, (mod.__name__, str(got), str(root))
    print(f"\nQ38_L4AGREE_ORIGIN 5/5 inside {root}")
