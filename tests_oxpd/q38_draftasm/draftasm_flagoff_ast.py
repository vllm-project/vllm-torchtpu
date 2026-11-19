"""Flag-off structural identity of q38 L4 DRAFT-ASM (VLLM_TPU_DRAFT_ASM_FOLD; torch-free).

DRAFT-ASM only adds code behind its flag: eagle3's class attribute ``_draft_asm = None``, its construction
under ``if envs.VLLM_TPU_DRAFT_ASM_FOLD:`` in ``__init__``, and hooks guarded by ``self._draft_asm`` (None
unless the flag is on); tpu_runner's assembler hooks guarded by a local ``draft_asm`` read from the drafter's
``_draft_asm``. `off(src)` resolves that code to its flag-off arm:
  * an `if` whose test reads ``VLLM_TPU_DRAFT_ASM_FOLD``, the attribute ``_draft_asm`` or the name
    ``draft_asm`` becomes its `else` body (none if absent);
  * an assignment to the name / attribute ``_draft_asm`` or the name ``draft_asm`` is dropped.
`check(new_src, ref_src)` returns the differences against the reference module (empty = identical with the
flag off); `off_source(src)` returns the flag-off module as source, so other flags-off checks
(tests/runner/crf_flagoff_ast.py, tests/spec_decode/dtt_flagoff_ast.py) can run on top of it."""
import ast
import copy

FLAG = "VLLM_TPU_DRAFT_ASM_FOLD"
ATTR = "_draft_asm"
LOCAL = "draft_asm"


def _reads(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in (FLAG, ATTR):
            return True
        if isinstance(n, ast.Name) and n.id == LOCAL:
            return True
    return False


def _target_is_ours(t):
    return (isinstance(t, ast.Name) and t.id in (ATTR, LOCAL)) or (isinstance(t, ast.Attribute) and t.attr == ATTR)


class _FlagOff(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        if _reads(node.test):
            return node.orelse or None
        return node

    def visit_Assign(self, node):
        if any(_target_is_ours(t) for t in node.targets):
            return None
        return self.generic_visit(node)


def off(src):
    tree = _FlagOff().visit(copy.deepcopy(ast.parse(src)))
    ast.fix_missing_locations(tree)
    return tree


def off_source(src):
    return ast.unparse(off(src))


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
    cls_new = {k: [_dump(n) for n in v.body if not isinstance(n, ast.FunctionDef)]
               for k, v in d_new.items() if isinstance(v, ast.ClassDef)}
    cls_ref = {k: [_dump(n) for n in v.body if not isinstance(n, ast.FunctionDef)]
               for k, v in d_ref.items() if isinstance(v, ast.ClassDef)}
    diffs += [f"class-level statements differ: {k}" for k in sorted(set(cls_new) & set(cls_ref))
              if cls_new[k] != cls_ref[k]]
    return diffs or ["differs (unlocated)"]


if __name__ == "__main__":
    import sys

    d = check(open(sys.argv[1]).read(), open(sys.argv[2]).read())
    print("Q38_DRAFTASM_FLAGOFF %s %s" % ("IDENTICAL" if not d else "DIFFERS", d))
    sys.exit(1 if d else 0)
