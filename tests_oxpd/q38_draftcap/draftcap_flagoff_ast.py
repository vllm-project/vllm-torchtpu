"""Flag-off structural identity of q38 L4 DRAFT-CAP (VLLM_TPU_DRAFT_ASM_CAP; torch-free).

DRAFT-CAP adds to draft_one.py only (spec_decode/draft_cap.py is new): ``self.one_cap = ...`` and a warning under
``if envs.VLLM_TPU_DRAFT_ASM_CAP and ...`` in ``DraftOne.__init__``, and the R_d branch under ``if self.one_cap:`` in
``DraftOne._one_ready``. `off(src)` resolves that code to its flag-off arm (DRAFT-ASM's / DRAFT1-B's resolver pattern):
  * an `if` whose test reads ``VLLM_TPU_DRAFT_ASM_CAP`` or the attribute ``one_cap`` becomes its `else` body;
  * an assignment to the attribute ``one_cap`` is dropped.
`check(new_src, ref_src)` returns the differences against the reference (29ea1e98d's draft_one.py; empty = identical
with the flag off)."""
import ast
import copy

FLAG = "VLLM_TPU_DRAFT_ASM_CAP"
ATTR = "one_cap"


def _reads(node):
    return any(isinstance(n, ast.Attribute) and n.attr in (FLAG, ATTR) for n in ast.walk(node))


class _FlagOff(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        if _reads(node.test):
            return node.orelse or None
        return node

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Attribute) and t.attr == ATTR for t in node.targets):
            return None
        return self.generic_visit(node)


def off(src):
    tree = _FlagOff().visit(copy.deepcopy(ast.parse(src)))
    ast.fix_missing_locations(tree)
    return tree


def _dump(x):
    return ast.dump(x, annotate_fields=False)


def _defs(body, prefix=""):
    out = {}
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[prefix + n.name] = n
            if isinstance(n, ast.ClassDef):
                out.update(_defs(n.body, prefix + n.name + "."))
    return out


def check(new_src, ref_src):
    new, ref = off(new_src), ast.parse(ref_src)
    if _dump(new) == _dump(ref):
        return []
    d_new, d_ref = _defs(new.body), _defs(ref.body)
    diffs = [f"only in new: {k}" for k in sorted(set(d_new) - set(d_ref))]
    diffs += [f"only in ref: {k}" for k in sorted(set(d_ref) - set(d_new))]
    diffs += [f"differs: {k}" for k in sorted(set(d_new) & set(d_ref))
              if not isinstance(d_new[k], ast.ClassDef) and _dump(d_new[k]) != _dump(d_ref[k])]
    top_new = [_dump(n) for n in new.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    top_ref = [_dump(n) for n in ref.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    if top_new != top_ref:
        diffs.append("module-level statements differ")
    return diffs or ["differs (unlocated)"]


def touched(new_src, ref_src):
    """The functions whose source differs from the reference before resolution (the edit's footprint)."""
    d_new, d_ref = _defs(ast.parse(new_src).body), _defs(ast.parse(ref_src).body)
    return sorted(k for k in set(d_new) & set(d_ref)
                  if not isinstance(d_new[k], ast.ClassDef) and _dump(d_new[k]) != _dump(d_ref[k]))
