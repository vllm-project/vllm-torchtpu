# SPDX-License-Identifier: Apache-2.0
"""q38 L4 FOLDFIX: flag-off structural identity of runner/speculative_decoding_manager.py (torch-free). FOLDFIX-BUILD.

Resolving the fix's two constructs gives the base manager's AST exactly (each counted exactly once):
  * in propose_draft_token_ids' return_device branch,
        if not isinstance(result, torch.Tensor | list):
            result = self._fold(result, next_tokens_per_chunk, scheduler_output)
    becomes the old `assert isinstance(result, torch.Tensor | list)` (the same test expression);
  * the staticmethod SpeculativeDecodingManager._fold is dropped.
So on a tensor or list result (the only kind without VLLM_TPU_DRAFT_ASM_FOLD) the branch evaluates the old assert's
isinstance test and returns the result, as before; nothing else in the file changed.
Base: $Q38_FOLDFIX_BASE_SRC, else $Q38_V5C_BASE_SRC (a src dir holding the base vllm_torchtpu; run_cpu.sh sets the
latter to q38-L1-v4 ecf4cc34a, whose manager is byte-identical to q38-l4-v5c 743886475's).
Run: python -m pytest -q -p no:cacheprovider -p no:logging -s tests/q38_l4_foldfix/test_foldfix_flagoff.py
"""

import ast
import os
import pathlib
from collections import Counter

REL = "vllm_torchtpu/runner/speculative_decoding_manager.py"
HEAD = pathlib.Path(__file__).resolve().parents[2] / "src" / REL
BASE_SRC = os.environ.get("Q38_FOLDFIX_BASE_SRC") or os.environ.get("Q38_V5C_BASE_SRC") or ""
IF_TEST = "isinstance(result, torch.Tensor | list)"
IF_BODY = "result = self._fold(result, next_tokens_per_chunk, scheduler_output)"


class FoldOff(ast.NodeTransformer):
    def __init__(self):
        self.counts = Counter()

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.name == "SpeculativeDecodingManager":
            keep = [b for b in node.body if not (isinstance(b, ast.FunctionDef) and b.name == "_fold")]
            self.counts["fold_method"] += len(node.body) - len(keep)
            node.body = keep
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        t = node.test
        if (isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not) and ast.unparse(t.operand) == IF_TEST
                and not node.orelse and len(node.body) == 1 and ast.unparse(node.body[0]) == IF_BODY):
            self.counts["fold_if"] += 1
            return ast.copy_location(ast.Assert(test=t.operand, msg=None), node)
        return node


def resolve(src: str):
    r = FoldOff()
    tree = r.visit(ast.parse(src))
    return ast.dump(tree), {k: v for k, v in r.counts.items() if v}  # constructs found (a zero count is none)


def test_o0_base_is_present():
    assert BASE_SRC and (pathlib.Path(BASE_SRC) / REL).is_file(), f"base manager missing under {BASE_SRC!r}"


def test_o1_the_fix_resolves_to_the_base_manager():
    head = HEAD.read_text()
    base = (pathlib.Path(BASE_SRC) / REL).read_text()
    got, counts = resolve(head)
    assert counts == {"fold_method": 1, "fold_if": 1}, counts
    assert got == ast.dump(ast.parse(base)), "the manager differs from the base beyond the fix's two constructs"
    assert resolve(base)[1] == {}, "the base already has a fix construct"
    print(f"\nFOLDFIX_FLAGOFF manager resolves to the base ({BASE_SRC}) with fold_if=1 fold_method=1; head != base "
          f"textually: {head != base}")


def test_o2_the_check_bites():
    head = HEAD.read_text()
    base = ast.dump(ast.parse((pathlib.Path(BASE_SRC) / REL).read_text()))
    bites = {
        "if body changed": head.replace(IF_BODY, "result = self._fold(result, None, scheduler_output)", 1),
        "extra statement": head.replace("            assert isinstance(result, list)\n",
                                        "            assert isinstance(result, list)\n            result = list(result)\n", 1),
        "sync assert dropped": head.replace("            assert isinstance(result, list)\n", "", 1),
    }
    for name, src in bites.items():
        assert src != head, f"bite '{name}' did not apply"
        assert resolve(src)[0] != base, f"bite '{name}' was not caught"
    print(f"\nFOLDFIX_FLAGOFF bites caught {len(bites)}/{len(bites)}")
