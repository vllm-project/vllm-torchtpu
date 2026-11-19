"""Flag-off structural identity of q38 L4 T1-K1FUSE (VLLM_TPU_K1FUSE; torch-free).

K1FUSE adds three hooks and one log-only bracket to existing files (spec_decode/k1fuse.py is new; draft_one.py,
draft_asm.py and vfy_one.py are unchanged):
  * eagle3.py: DraftK1's construction under ``if envs.VLLM_TPU_K1FUSE:`` (else: DraftOne), and the take in propose,
    an ``if`` whose test reads the controller's ``k1_result`` (absent on DraftOne / DraftAsm);
  * tpu_runner.py: in ``_vfy_verify``, an ``if`` whose test reads the controller's ``k1_verify`` (else: the line it
    replaces), and in ``_modify_prev_results`` two ``if``s whose bodies are only ``host_phases.sub("k1_...")``.
`off(src)` resolves them to the flag-off arm: an ``if`` whose test reads ``VLLM_TPU_K1FUSE``, ``k1_result`` or
``k1_verify`` (attribute, name or string) becomes its ``else`` body; an ``if`` whose body is only
``host_phases.sub("k1_*")`` calls is dropped. `check(new_src, ref_src)` lists the differences against the reference
(the composed base 7867a18ef's file; empty = identical with the flag off)."""
import ast
import copy

NAMES = {"VLLM_TPU_K1FUSE", "k1_result", "k1_verify"}


def _reads(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in NAMES:
            return True
        if isinstance(n, ast.Name) and n.id in NAMES:
            return True
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in NAMES:
            return True
    return False


def _k1_subs_only(body):
    for st in body:
        if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)):
            return False
        f = st.value.func
        if not (isinstance(f, ast.Attribute) and f.attr == "sub" and isinstance(f.value, ast.Name)
                and f.value.id == "host_phases"):
            return False
        a = st.value.args
        if not (len(a) == 1 and isinstance(a[0], ast.Constant) and str(a[0].value).startswith("k1_")):
            return False
    return bool(body)


class _Off(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        if _reads(node.test):
            return node.orelse or None
        if _k1_subs_only(node.body) and not node.orelse:
            return None
        return node


def off(src):
    tree = _Off().visit(copy.deepcopy(ast.parse(src)))
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
