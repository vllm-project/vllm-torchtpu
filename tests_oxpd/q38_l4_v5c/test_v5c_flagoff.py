# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v5 composition (q38-l4-v5c) with every L4 v5 flag off: the chained flag-off resolvers give q38-L1-v4
ecf4cc34a's AST, file for file (tests/q38_l4_v5c/v5c_flagoff.py; torch-free). CDK CPU pod only. V5COMP-BUILD.

The base src is read from Q38_V5C_BASE_SRC (run_cpu.sh points it at the bundle's base/src, src hash pinned there).
Every other src file is byte-identical to the base (run_cpu.sh P0's exact diff set).

O1  each changed file resolves to ecf4cc34a's AST through its chain; the allowances are counted exactly
    (A1 DRAFT-2 (c)'s flag-off load-time agreement call, A2 mtp's logger, A3 three docstring sentences).
O2  786f7bac2 (the v5 composition fix): its lever branches sit inside `_vfy*` methods only, which VFY-ONE's resolver
    drops, so the flag-off runner has no trace of them.
O3  the check bites: an unguarded hook of each piece is named, and each resolver of a chain is needed.
v6 (q38-l4-v6c2, V6COMP-BUILD): T1-K1FUSE and L1-H3 / ATTNWIRE join the chains (k1fuse_flagoff_ast, v5c_flagoff.H3Off);
O3 also bites on K1's runner sub-marks unguarded, K1's DraftK1 build unguarded, H3's attach unguarded and H3's mtp
pass unguarded, and needs H3's resolver on tpu_runner and mtp.
Run: Q38_V5C_BASE_SRC=<base>/src python -m pytest -q -p no:cacheprovider --noconftest -s tests/q38_l4_v5c/test_v5c_flagoff.py
"""

import ast
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v5c_flagoff as fo  # noqa: E402

HEAD = os.path.join(HERE, "..", "..", "src", "vllm_torchtpu")
BASE = os.path.join(os.environ.get("Q38_V5C_BASE_SRC", "/nonexistent"), "vllm_torchtpu")
ALLOW = {
    "runner/tpu_runner.py": {"doc_p1": 1, "p1_log_fn": 1, "p1_inlined": 2, "p1_return_rows": 1},
    "models/vllm/qwen4_exp/mtp.py": {"A1_load_agreement": 1, "A2_logger": 1, "A2_logger_import": 1, "doc_emb": 2},
    "layers/adapter/sample/vp_sampler.py": {"doc_vrng": 2},
    "platforms/tpu_platform.py": {"sfplat_gate": 1, "sfplat_elif": 1},
}


def _read(root, rel):
    with open(os.path.join(root, rel)) as f:
        return f.read()


def test_o0_base_is_present():
    assert os.path.isdir(BASE), f"Q38_V5C_BASE_SRC must point at ecf4cc34a's src ({BASE})"


@pytest.mark.parametrize("rel", sorted(fo.CHAINS))
def test_o1_flag_off_resolves_to_ecf4cc34a(rel):
    head, base = _read(HEAD, rel), _read(BASE, rel)
    assert head != base
    counts = {}
    d = fo.check(head, base, fo.CHAINS[rel], counts)
    print(f"\nV5C_FLAGOFF {rel} chain={'>'.join(fo.CHAINS[rel])} {'IDENTICAL' if not d else d} allowances={counts}")
    assert not d, d
    assert counts == ALLOW.get(rel, {}), (rel, counts)


def test_o2_the_composition_fix_resolves_away_with_vfy():
    src = _read(HEAD, "runner/tpu_runner.py")
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    sites = []
    for name in ("_vfy_verify", "_vfy_chain", "_vfy_fb_tick"):
        for n in ast.walk(fns[name]):
            if isinstance(n, ast.If) and ("_l6_ab.on('sampsnap')" in ast.unparse(n.test)
                                         or "_l6_ab.on('statsasync')" in ast.unparse(n.test)):
                sites.append(f"{name}:{n.lineno}")
    assert len(sites) == 4, sites  # params + num_active (_vfy_verify), num_active (_vfy_chain), statsasync (tick)
    off, _ = fo.resolve(src, fo.CHAINS["runner/tpu_runner.py"])
    left = sorted({n.name for n in ast.walk(off) if isinstance(n, ast.FunctionDef) and n.name.startswith("_vfy")})
    assert left == [], left
    print(f"\nV5C_FLAGOFF O2 786f7bac2 lever branches at tpu_runner.py {' '.join(sites)}; none left with flags off")


def _bites(rel, old, new, want, chain=None):
    src = _read(HEAD, rel)
    assert src.count(old) >= 1, (rel, old)
    bad = src.replace(old, new, 1)
    got = fo.check(bad, _read(BASE, rel), chain or fo.CHAINS[rel])
    assert got == want, (rel, got, want)


def test_o3_the_check_bites_on_each_piece():
    r, e, m, i = "runner/tpu_runner.py", "spec_decode/eagle3.py", "models/vllm/qwen4_exp/mtp.py", "__init__.py"
    # P1 sampsnap and statsasync branches unguarded
    _bites(r, '        if _l6_ab.on("sampsnap"):\n            # q38 L4 T-2 stage 1 SAMP-SNAP: one upload per num_active value.\n',
           '        if self._vps_fb_chunks >= 0:\n            # q38 L4 T-2 stage 1 SAMP-SNAP: one upload per num_active value.\n',
           ["differs: TPUModelRunner._vps_verify_fb"])
    _bites(r, '            if _l6_ab.on("statsasync"):\n', '            if self._vps_fb_chunks < 0:\n',
           ["differs: TPUModelRunner._vps_verify_fb"])
    # VFY-ONE warm-up hook unguarded (tests/l4vfy/test_vfy_v5_flagoff.py's bite)
    _bites(r, '                if self.__dict__.get("_vfy") is not None:\n'
              "                    # q38 L4 VFY-ONE: its verify programs beside the D83 chain's.\n"
              "                    self._vfy_precompile()\n",
           "                self._vfy_precompile()\n", ["differs: TPUModelRunner.capture_model"])
    # DRAFT-ASM precompile hook followed by an unguarded seed precompile
    asm = "        if self._draft_asm is not None:\n            self._draft_asm.precompile()\n"
    _bites(e, asm, asm + "        self._precompile_draft_seed()\n", ["differs: Eagle3Proposer.precompile"])
    # DRAFT1-B hook followed by an unguarded statement
    hook = "                    hidden = self._fit_draft_rows_output(hidden, rows)\n"
    _bites(e, hook, hook + "                hidden = hidden + 0\n", ["differs: Eagle3Proposer.propose"])
    # DRAFT-2 (c): the embed branch unguarded; EMB-REP-TGT: the target table read under another name
    _bites(m, 'if getattr(self, "embed_repl", False) and int(getattr(emb, "tp_size", 1)) > 1:',
           'if int(getattr(emb, "tp_size", 1)) > 1:', ["differs: Qwen4ExpMultiTokenPredictor.embed"])
    src = _read(HEAD, i)
    bad = src.replace('getattr(self, "_q38_repl_table", None)', 'getattr(self, "_q38_repl_tbl", None)', 1)
    assert bad != src and fo.check(bad, _read(BASE, i), fo.CHAINS[i]) != []
    # A1 is allowed exactly once: a second agreement call is caught by the allowance count
    msrc = _read(HEAD, m)
    a1 = "        self.bind_replicated_embedding()\n"
    assert msrc.count(a1) == 1
    counts = {}
    assert fo.check(msrc.replace(a1, a1 + a1), _read(BASE, m), fo.CHAINS[m], counts) == []
    assert counts["A1_load_agreement"] == 2 != ALLOW[m]["A1_load_agreement"]
    # each resolver is needed where its piece left code the others do not resolve (DRAFT1-B's eagle3 hooks read
    # self._draft_asm, so DRAFT-ASM's resolver also drops them; its resolver stays in the chain, redundant there)
    needed = {r: ("p1", "agree", "samp2", "sf", "lmh", "da", "vfy"), e: ("da", "vfy"), m: ("emb", "d1")}
    for rel, steps in needed.items():
        chain = fo.CHAINS[rel]
        for step in steps:
            part = tuple(x for x in chain if x != step)
            assert fo.check(_read(HEAD, rel), _read(BASE, rel), part) != [], (rel, part)
    assert fo.check(_read(HEAD, e), _read(BASE, e), ("da", "vfy")) == []
    # v6: K1FUSE's runner sub-mark unguarded (a bare statement; the VFY resolver drops only the guarded `if`)
    sub = '        if self.__dict__.get("_vfy") is not None:\n            host_phases.sub("k1_nt_wait")\n'
    _bites(r, sub, '        host_phases.sub("k1_nt_wait")\n', ["differs: TPUModelRunner._modify_prev_results"])
    # v6: K1FUSE's DraftK1 build behind another flag (not K1's). DRAFT-ASM's resolver drops the whole fold block either
    # way, so this bite is K1's resolver alone, against the K1-resolved head
    esrc = _read(HEAD, e)
    k1b = "                if envs.VLLM_TPU_K1FUSE:\n"
    assert esrc.count(k1b) == 1
    import k1fuse_flagoff_ast as k1fo
    assert k1fo.check(esrc.replace(k1b, "                if envs.VLLM_TPU_DRAFT_ASM_ONE:\n"), k1fo.ast.unparse(
        k1fo.off(esrc))) != []
    # v6: H3's attach unguarded (capture_model) and H3's mtp pass unguarded
    att = "            if envs.VLLM_TPU_MK_DRAFT_ATTN:\n"
    got = fo.check(_read(HEAD, r).replace(att, "            if self._megakernel is not None:\n", 1), _read(BASE, r),
                   fo.CHAINS[r])
    assert got == ["differs: TPUModelRunner.capture_model"], got
    mp = "            elif self.mk_attn is not None:  # q38 L1-H3: the fused loop's pass\n"
    got = fo.check(_read(HEAD, m).replace(mp, "            elif self.mk_tail is not None:  # q38 L1-H3\n", 1),
                   _read(BASE, m), fo.CHAINS[m])
    assert got == ["differs: Qwen4ExpMultiTokenPredictor.forward"], got
    for rel in (r, m):
        part = tuple(x for x in fo.CHAINS[rel] if x != "h3")
        assert fo.check(_read(HEAD, rel), _read(BASE, rel), part) != [], (rel, "h3 needed")
    print("\nV5C_FLAGOFF O3 v6 bites: K1 runner sub-mark, K1 DraftK1 build, H3 attach (capture_model), H3 mtp pass "
          "(forward); H3's resolver needed on tpu_runner and mtp")
    print("\nV5C_FLAGOFF O3 bites: P1 sampsnap, P1 statsasync, VFY hook, DRAFT-ASM hook, DRAFT1-B hook, DRAFT-2 (c) "
          "branch, EMB-REP-TGT table, A1 twice; needed resolvers tpu_runner p1+da+vfy, eagle3 da+vfy (d1 redundant), "
          "mtp emb+d1; tpu_runner also agree")
