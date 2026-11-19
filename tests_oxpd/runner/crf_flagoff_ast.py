"""Flags-off structural identity of the CARRY_ROWS_FRESH port (q38 DEC-FB-3; torch-free, so it also runs on the Mac).

dongruizi's 48745d9b0 (VLLM_TPU_DRAFT_ROWS_FRESH) and 7a0a3958e (VLLM_TPU_DRAFT_CARRY_ROWS_FRESH) only add code behind
their flags. Resolve that code to its flags-off arm and the module must be 68e6e31c9's, AST for AST:
  * an `if` whose test reads either flag, or tests `untrimmed is not None`, becomes its `else` body (none if absent);
  * `untrimmed = ...` (only read inside those branches) and `self._draft_rows_fresh_logged = False` are dropped;
  * the one refactor: `_draft_propose_token` now returns `self._propose_token_body(hidden)`, whose body (docstring
    aside) must be the old `_draft_propose_token` body;
  * the new methods may exist only in ALLOWED_NEW.
`check(new_src, ref_src)` returns a list of differences; empty = identical with the flags off."""
import ast
import copy

FLAGS = ("VLLM_TPU_DRAFT_ROWS_FRESH", "VLLM_TPU_DRAFT_CARRY_ROWS_FRESH")
ALLOWED_NEW = {"_draft_token_and_rows", "_propose_token_body", "_draft_rows_carry_token_fresh", "_carry_and_token_body"}


def _reads_flag(node):
    return any(isinstance(n, ast.Attribute) and n.attr in FLAGS for n in ast.walk(node))


def _tests_untrimmed(node):
    t = node.test
    while isinstance(t, ast.BoolOp):
        t = t.values[0]
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "untrimmed"
            and isinstance(t.ops[0], ast.IsNot))


class _FlagsOff(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        if _reads_flag(node.test) or _tests_untrimmed(node):
            return node.orelse or None
        return node

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "untrimmed":
                return None
            if isinstance(t, ast.Attribute) and t.attr == "_draft_rows_fresh_logged":
                return None
        return node


def _body_sans_doc(fn):
    b = list(fn.body)
    if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
            and isinstance(b[0].value.value, str):
        b = b[1:]
    return b


def _dump(x):
    if isinstance(x, list):
        return [ast.dump(n, annotate_fields=False) for n in x]
    return ast.dump(x, annotate_fields=False)


def _class(mod, name="Eagle3Proposer"):
    return next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == name)


def check(new_src, ref_src):
    diffs = []
    new, ref = ast.parse(new_src), ast.parse(ref_src)
    off = _FlagsOff().visit(copy.deepcopy(new))
    ast.fix_missing_locations(off)
    # module level: everything except the proposer class
    m_new = [n for n in off.body if not (isinstance(n, ast.ClassDef) and n.name == "Eagle3Proposer")]
    m_ref = [n for n in ref.body if not (isinstance(n, ast.ClassDef) and n.name == "Eagle3Proposer")]
    if _dump(m_new) != _dump(m_ref):
        diffs.append("module level (outside Eagle3Proposer) differs")
    c_new, c_ref = _class(off), _class(ref)
    f_new = {n.name: n for n in c_new.body if isinstance(n, ast.FunctionDef)}
    f_ref = {n.name: n for n in c_ref.body if isinstance(n, ast.FunctionDef)}
    other_new = [n for n in c_new.body if not isinstance(n, ast.FunctionDef)]
    other_ref = [n for n in c_ref.body if not isinstance(n, ast.FunctionDef)]
    if _dump(other_new) != _dump(other_ref):
        diffs.append("class-level statements differ")
    extra = set(f_new) - set(f_ref)
    if extra - ALLOWED_NEW:
        diffs.append("unexpected new methods %s" % sorted(extra - ALLOWED_NEW))
    if set(f_ref) - set(f_new):
        diffs.append("methods removed %s" % sorted(set(f_ref) - set(f_new)))
    for name in sorted(set(f_ref) & set(f_new)):
        a, b = f_new[name], f_ref[name]
        if name == "_draft_propose_token":
            want = ast.parse("return self._propose_token_body(hidden)").body
            if _dump(_body_sans_doc(a)) != _dump(want):
                diffs.append("_draft_propose_token is not `return self._propose_token_body(hidden)`")
            if "_propose_token_body" not in f_new or \
                    _dump(_body_sans_doc(f_new["_propose_token_body"])) != _dump(_body_sans_doc(b)):
                diffs.append("_propose_token_body is not 68e6e31c9's _draft_propose_token body")
            if _dump(a.args) != _dump(b.args) or _dump(a.decorator_list) != _dump(b.decorator_list):
                diffs.append("_draft_propose_token signature / decorators differ")
            continue
        if _dump(a) != _dump(b):
            diffs.append("method %s differs with the flags off" % name)
    return diffs


if __name__ == "__main__":
    import sys
    d = check(open(sys.argv[1]).read(), open(sys.argv[2]).read())
    print("Q38_CRF_FLAGOFF %s %s" % ("IDENTICAL" if not d else "DIFFERS", d))
    sys.exit(1 if d else 0)
