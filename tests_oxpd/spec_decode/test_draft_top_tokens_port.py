"""Port checks of MM DSTOCK's VLLM_TPU_DRAFT_TOP_TOKENS draft sites (6dcff75dc) onto the P-c / lever-3 chain, with
q38 D-MK's inline check (97c9664ed's _CHECK) as VLLM_TPU_DRAFT_TOP_TOKENS_CHECK (q38 DEC-FB-3 lever 2). CPU only.
tests/q38mm_dstock/test_vps_wiring.py covers the three token sites, the fused loop, the flag and the exact get_top_tokens
reduction (ties included); this adds:
1. the inline check counts mismatches against the full-logits argmax, runs only while checks are left, and logs one
   DRAFT_TOP_TOKENS_CHECK line per check;
2. step 0's token (from the separate propose program or the fused carry) is what the check compares, right after it is
   appended;
3. load_model turns the sites on only through draft_top_tokens_ok, arms the check only for a draft that qualifies, and
   the fused loop is built with the same switch;
4. Qwen4ExpMTP opts in with plain_compute_logits, and its compute_logits is exactly lm head + logits processor (the
   condition under which get_top_tokens is the same token);
5. with the switch off, eagle3.py is 066323583's, AST for AST (dtt_flagoff_ast);
6. the flags default off (the compile key: tests/compilation/test_tpu_compile_key_registration.py).
"""
import ast
import contextlib
import logging
import pathlib
import textwrap
import types

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.spec_decode import eagle3

EAGLE3_SRC = pathlib.Path(eagle3.__file__).read_text()
MTP_SRC = (pathlib.Path(eagle3.__file__).parents[1] / "models" / "vllm" / "qwen4_exp" / "mtp.py").read_text()


def _method(src, cls, name, ns):
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
    return ns[name]


# ---------------------------------------------------------------- 1. the inline check


def _check_fn(log):
    ns = {"torch": torch, "set_model_tag": lambda tag: contextlib.nullcontext(), "logger": log}
    return _method(EAGLE3_SRC, "Eagle3Proposer", "_check_draft_top_tokens", ns)


def test_check_counts_mismatches_while_checks_are_left(caplog):
    log = logging.getLogger("t-dtt")
    check = _check_fn(log)
    ref = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    seen = []
    me = types.SimpleNamespace(_draft_top_checks_left=2,
                               _draft_propose_token_full=lambda h: (seen.append(h), ref)[1])
    h = torch.zeros(4, 3)
    with caplog.at_level(logging.INFO, logger="t-dtt"):
        check(me, h, torch.tensor([5, 6, 0, 8], dtype=torch.int32))   # one mismatch
        check(me, h, ref.clone())                                     # none
        check(me, h, torch.zeros(4, dtype=torch.int32))               # no checks left: not run
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("DRAFT_TOP_TOKENS_CHECK")]
    assert lines == ["DRAFT_TOP_TOKENS_CHECK rows=4 mismatches=1 left=1",
                     "DRAFT_TOP_TOKENS_CHECK rows=4 mismatches=0 left=0"], lines
    assert me._draft_top_checks_left == 0 and len(seen) == 2 and all(x is h for x in seen)


def test_check_is_a_noop_without_checks():
    check = _check_fn(logging.getLogger("t-dtt0"))
    me = types.SimpleNamespace(_draft_propose_token_full=lambda h: (_ for _ in ()).throw(AssertionError("ran")))
    check(me, torch.zeros(2, 3), torch.zeros(2, dtype=torch.int32))   # no _draft_top_checks_left attribute
    me._draft_top_checks_left = 0
    check(me, torch.zeros(2, 3), torch.zeros(2, dtype=torch.int32))


def test_full_argmax_reference_is_the_stock_token():
    ns = {"torch": torch}
    full = _method(EAGLE3_SRC, "Eagle3Proposer", "_draft_propose_token_full", ns)
    W = torch.randn(3, 11, generator=torch.Generator().manual_seed(2))
    me = types.SimpleNamespace(draft_model=types.SimpleNamespace(compute_logits=lambda h: h @ W))
    h = torch.randn(5, 3, generator=torch.Generator().manual_seed(3))
    got = full(me, h)
    assert got.dtype == torch.int32 and torch.equal(got, (h @ W).argmax(dim=-1).to(torch.int32))


# ---------------------------------------------------------------- 2. the check's wiring at step 0


def test_check_runs_on_step0_token_right_after_it_is_appended():
    want = (
        "            if fused_token0 is not None:\n"
        "                draft_tokens_per_chunk.append([fused_token0])\n"
        "            else:\n"
        "                with set_model_tag(\"eagle_head\"):\n"
        "                    draft_tokens_per_chunk.append(\n"
        "                        [self._draft_propose_token(last_hidden_carry)]\n"
        "                    )\n"
        "            if getattr(self, \"_draft_top_checks_left\", 0) > 0:\n"
        "                self._check_draft_top_tokens(\n"
        "                    last_hidden_carry, draft_tokens_per_chunk[-1][0]\n"
        "                )\n"
    )
    assert EAGLE3_SRC.count(want) == 1


# ---------------------------------------------------------------- 3. load-time switch


def test_load_turns_sites_on_only_through_the_eligibility_check():
    want = (
        "        if vps_wiring.draft_top_tokens_enabled():\n"
        "            self._dvp = vps_wiring.draft_top_tokens_ok(self.draft_model)\n"
        "            if self._dvp:\n"
    )
    assert EAGLE3_SRC.count(want) == 1
    i = EAGLE3_SRC.index(want)
    block = EAGLE3_SRC[i:EAGLE3_SRC.index("        self._build_mtp_fused_loop()\n", i)]
    # the check is armed inside the qualifying branch only, from its own flag
    armed = "                self._draft_top_checks_left = int(envs.VLLM_TPU_DRAFT_TOP_TOKENS_CHECK)\n"
    assert block.count(armed) == 1 and block.index(armed) < block.index("            else:\n")
    # the sites read the same switch (or its absence) and the loop is built with it
    assert EAGLE3_SRC.count("                    dvp=self._dvp,\n") == 1
    assert EAGLE3_SRC.count("        self._dvp = False\n") == 1


# ---------------------------------------------------------------- 4. the draft opts in


def test_qwen4_exp_mtp_opts_in_with_plain_compute_logits():
    tree = ast.parse(MTP_SRC)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpMTP")
    flags = [n for n in cls.body if isinstance(n, ast.Assign)
             and any(getattr(t, "id", None) == "plain_compute_logits" for t in n.targets)]
    assert len(flags) == 1 and isinstance(flags[0].value, ast.Constant) and flags[0].value.value is True
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_logits")
    body = [ast.unparse(s) for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert body == ["return self.logits_processor(self.lm_head, hidden_states)"], body


# ---------------------------------------------------------------- 5. switch off = 066323583


def test_switch_off_eagle3_is_066323583():
    import importlib.util

    here = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location("dtt_flagoff_ast", here / "dtt_flagoff_ast.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # q38 L4 DRAFT-ASM sits on top: resolve its flag to off first (tests/q38_draftasm/draftasm_flagoff_ast.py).
    spec_a = importlib.util.spec_from_file_location(
        "draftasm_flagoff_ast", here.parent / "q38_draftasm" / "draftasm_flagoff_ast.py"
    )
    asm = importlib.util.module_from_spec(spec_a)
    spec_a.loader.exec_module(asm)
    diffs = mod.check(asm.off_source(EAGLE3_SRC), (here / "eagle3_ref_066323583.py.txt").read_text())
    assert not diffs, diffs


# ---------------------------------------------------------------- 6. defaults


def test_flags_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_DRAFT_TOP_TOKENS", raising=False)
    monkeypatch.delenv("VLLM_TPU_DRAFT_TOP_TOKENS_CHECK", raising=False)
    assert envs.VLLM_TPU_DRAFT_TOP_TOKENS is False
    assert envs.VLLM_TPU_DRAFT_TOP_TOKENS_CHECK == 0
