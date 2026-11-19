# SPDX-License-Identifier: Apache-2.0
"""q38 L4 T1-K1FUSE flag off (CDK CPU pod only).

1. AST identity: with VLLM_TPU_K1FUSE's code resolved to its off arm (tests/q38_k1fuse/k1fuse_flagoff_ast.py),
   eagle3.py and tpu_runner.py are the composed base q38-l4-k1fuse 7867a18ef's (= q38-l4-vfy-v5 2e166829a + q38-l4-
   draft1b 29ea1e98d), AST for AST; the check bites on each hook left unguarded.
2. Flags: registered, default off (K1FUSE False, _AB "", _CHECK 256), VLLM_TPU_K1FUSE hashed in the compile key,
   _AB and _CHECK host-side (ignored).
3. Construction: eagle3 builds DraftK1 only with the flag; DraftK1 is a DraftOne; without DH-D it stays unconfigured.
4. The HOSTSUBS bracket of the sampled-token D2H wait: `_modify_prev_results` charges ~k1_mp_head and ~k1_nt_wait
   around wait_for_copy when VFY-ONE is on, and makes no sub-mark without it.
"""

import os
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k1fuse_flagoff_ast as fo  # noqa: E402

SRC = pathlib.Path(os.environ.get("Q38_K1FUSE_SRC") or HERE.parent.parent / "src") / "vllm_torchtpu"
EAGLE, RUNNER = SRC / "spec_decode" / "eagle3.py", SRC / "runner" / "tpu_runner.py"
EAGLE_SRC, RUNNER_SRC = EAGLE.read_text(), RUNNER.read_text()
EAGLE_REF = (HERE / "ref_7867a18ef_eagle3.py.txt").read_text()
RUNNER_REF = (HERE / "ref_7867a18ef_tpu_runner.py.txt").read_text()


def test_eagle3_flag_off_is_the_composed_base():
    assert EAGLE_SRC != EAGLE_REF
    d = fo.check(EAGLE_SRC, EAGLE_REF)
    print(f"\nK1FUSE_FLAGOFF eagle3 {'IDENTICAL' if not d else d}")
    assert not d, d


def test_tpu_runner_flag_off_is_the_composed_base():
    assert RUNNER_SRC != RUNNER_REF
    d = fo.check(RUNNER_SRC, RUNNER_REF)
    print(f"\nK1FUSE_FLAGOFF tpu_runner {'IDENTICAL' if not d else d}")
    assert not d, d


def test_flag_off_check_bites_on_each_hook():
    # the take renamed so the resolver does not know it (an unguarded hook): only propose differs
    assert EAGLE_SRC.count("k1_result") == 2
    assert fo.check(EAGLE_SRC.replace("k1_result", "k1_take"), EAGLE_REF) == ["differs: Eagle3Proposer.propose"]
    build = "                if envs.VLLM_TPU_K1FUSE:\n"
    assert EAGLE_SRC.count(build) == 1
    assert fo.check(EAGLE_SRC.replace(build, "                if envs.VLLM_TPU_DRAFT_ASM_ONE:\n"), EAGLE_REF) == [
        "differs: Eagle3Proposer.__init__"]
    hook = '"_draft_asm", None), "k1_verify", None)'
    assert RUNNER_SRC.count(hook) == 1
    assert fo.check(RUNNER_SRC.replace(hook, '"_draft_asm", None), "precompile", None)'), RUNNER_REF) == [
        "differs: TPUModelRunner._vfy_verify"]
    sub = 'host_phases.sub("k1_nt_wait")'
    assert RUNNER_SRC.count(sub) == 1
    assert fo.check(RUNNER_SRC.replace(sub, 'host_phases.sub("st_nt_wait")'), RUNNER_REF) == [
        "differs: TPUModelRunner._modify_prev_results"]
    print("\nK1FUSE_FLAGOFF bite: each of the 4 hooks unguarded is located (propose, __init__, _vfy_verify, "
          "_modify_prev_results)")


FLAGS = [("VLLM_TPU_K1FUSE", "1", False, True), ("VLLM_TPU_K1FUSE_AB", "ab:1", "", False),
         ("VLLM_TPU_K1FUSE_CHECK", "0", 256, False)]


@pytest.mark.parametrize("name,value,default,hashed", FLAGS)
def test_flags_registered_default_off_and_keyed(monkeypatch, name, value, default, hashed):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    monkeypatch.delenv(name, raising=False)
    getter = envs.environment_variables[name]
    assert getter() == default
    monkeypatch.setenv(name, value)
    assert getter() != default
    assert (name not in _TPU_COMPILE_ENV_IGNORED) == hashed


def test_eagle3_builds_draft_k1_only_with_the_flag(monkeypatch):
    import k1fuse_harness as kh
    from vllm_torchtpu.spec_decode import draft_one, eagle3, k1fuse

    src = EAGLE_SRC
    assert "self._draft_asm = k1fuse.DraftK1(self)" in src and issubclass(k1fuse.DraftK1, draft_one.DraftOne)
    kh.install(monkeypatch, k1=True)
    monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_ONE", "0")
    p = kh.h1.make_proposer(fold=False, cap=kh.CAP)

    def fresh():  # q38 L4 AGREE port (v6 composition, V6COMP-BUILD; tests only): one controller per runner clock
        p.runner.__dict__.pop("_l4_agree", None)
        return k1fuse.DraftK1(p)

    ctl = fresh()
    assert not ctl.k1_conf and ctl.warm["k1"] == set()
    monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_ONE", "1")
    monkeypatch.setenv("VLLM_TPU_K1FUSE_AB", "0")
    assert not fresh().k1_conf
    monkeypatch.setenv("VLLM_TPU_K1FUSE_AB", "")
    assert fresh().k1_conf
    assert eagle3 is not None


def test_nt_wait_sub_marks_only_with_vfy(monkeypatch):
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    subs = []
    monkeypatch.setattr(tpu_runner.host_phases, "sub", lambda name: subs.append(name))
    for vfy, want in ((object(), ["k1_mp_head", "k1_nt_wait"]), (None, [])):
        r = object.__new__(TPUModelRunner)
        r._vfy = vfy
        waited = []
        r._pre_async_results = SimpleNamespace(
            req_ids=[], request_seq_lens=[], discard_sampled_tokens_req_indices=[], num_draft_per_req=None,
            wait_for_copy=lambda: waited.append(1) or torch.zeros((0, 4), dtype=torch.int32))
        subs.clear()
        r._modify_prev_results()
        assert [x for x in subs if x.startswith("k1_")] == want and waited == [1], (vfy, subs)  # q38 L1-V6: L6 P-METER's mpr_* subs share the bracket
    print("\nK1FUSE_SUBS _modify_prev_results: ~k1_mp_head / ~k1_nt_wait around wait_for_copy with VFY-ONE on; none off")
