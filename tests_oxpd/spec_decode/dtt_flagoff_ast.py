"""Switch-off structural identity of the DRAFT_TOP_TOKENS port (q38 DEC-FB-3 lever 2; torch-free).

MM DSTOCK's 6dcff75dc draft sites and q38 D-MK's inline check only add code behind `self._dvp` (set only when
VLLM_TPU_DRAFT_TOP_TOKENS is on and the draft qualifies), behind `vps_wiring.draft_top_tokens_enabled()`, or behind
`self._draft_top_checks_left > 0`. `off(src)` resolves that code to its switch-off arm and returns the module AST:
  * an `if` whose test reads `_dvp`, `_draft_top_checks_left` or `draft_top_tokens_enabled` becomes its `else` body
    (none if absent);
  * `self._dvp = False`, the `dvp=self._dvp` keyword of the fused-loop build and the `vps_wiring` import are dropped;
  * the new methods may exist only in ALLOWED_NEW (they are dropped).
`check(new_src, ref_src)` returns a list of differences against the reference module (066323583's eagle3.py); empty =
identical with the switch off. `off_source(src)` returns the switch-off module as source (ast.unparse), so another
flags-off check (tests/runner/crf_flagoff_ast.py) can run on top of it."""
import ast
import copy

NAMES = ("_dvp", "_draft_top_checks_left", "draft_top_tokens_enabled")
ALLOWED_NEW = {"_check_draft_top_tokens", "_draft_propose_token_full"}


def _reads_switch(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in NAMES:
            return True
        if isinstance(n, ast.Constant) and n.value in NAMES:  # getattr(self, "_dvp", False)
            return True
    return False


class _SwitchOff(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        if _reads_switch(node.test):
            return node.orelse or None
        return node

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Attribute) and t.attr == "_dvp":
                return None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "vllm_torchtpu.runner" and [a.name for a in node.names] == ["vps_wiring"]:
            return None
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        node.keywords = [k for k in node.keywords if not (k.arg == "dvp" and _reads_switch(k.value))]
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.name == "Eagle3Proposer":
            node.body = [n for n in node.body if not (isinstance(n, ast.FunctionDef) and n.name in ALLOWED_NEW)]
        return node


def off(src):
    tree = _SwitchOff().visit(copy.deepcopy(ast.parse(src)))
    ast.fix_missing_locations(tree)
    return tree


def off_source(src):
    return ast.unparse(off(src))


def check(new_src, ref_src):
    diffs = []
    new, ref = off(new_src), ast.parse(ref_src)
    tops = lambda m: [n for n in m.body if not (isinstance(n, ast.ClassDef) and n.name == "Eagle3Proposer")]  # noqa
    if [ast.dump(n) for n in tops(new)] != [ast.dump(n) for n in tops(ref)]:
        diffs.append("module level (outside Eagle3Proposer) differs")
    cls = lambda m: next(n for n in m.body if isinstance(n, ast.ClassDef) and n.name == "Eagle3Proposer")  # noqa
    a, b = cls(new), cls(ref)
    fa = {n.name: n for n in a.body if isinstance(n, ast.FunctionDef)}
    fb = {n.name: n for n in b.body if isinstance(n, ast.FunctionDef)}
    if set(fa) != set(fb):
        diffs.append("methods differ: extra %s, missing %s" % (sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))))
    for name in sorted(set(fa) & set(fb)):
        if ast.dump(fa[name]) != ast.dump(fb[name]):
            diffs.append("method %s differs with the switch off" % name)
    oa = [ast.dump(n) for n in a.body if not isinstance(n, ast.FunctionDef)]
    ob = [ast.dump(n) for n in b.body if not isinstance(n, ast.FunctionDef)]
    if oa != ob:
        diffs.append("class-level statements differ")
    return diffs


if __name__ == "__main__":
    import sys
    d = check(open(sys.argv[1]).read(), open(sys.argv[2]).read())
    print("Q38_DTT_SWITCHOFF %s %s" % ("IDENTICAL" if not d else "DIFFERS", d))
    sys.exit(1 if d else 0)
