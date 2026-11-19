"""Flag-off structural identity of q38 L4 LMH-STREAM stage 0 (VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT / _VERIFY;
torch-free). After L4 VFY-BUILD's tests/l4vfy/vfy_flagoff_ast.py and DRAFT-ASM's draftasm_flagoff_ast.py (credited).

The lever only adds code that runs with a flag on. `off(src)` resolves it to its flag-off arm:
  * an `if` whose test reads VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT or _VERIFY becomes its `else` body (tpu_runner);
  * the new runner/utils.py definitions (PR 2821's HiddenMajorLMHeadMethod / relayout_lm_head and q38's
    hidden_major_refusal / _hidden_major_weight / hidden_major_head_copy / _head_census /
    arm_lm_head_hidden_major) are dropped, and in that module only, their two imports (`import copy`,
    ParallelLMHead / UnquantizedEmbeddingMethod);
  * envs.py's two annotations and two environment_variables entries are dropped.
`check(new_src, ref_src, extra=None)` returns the differences against the reference module (empty = identical with
the flags off); `extra` is another resolver applied first (to chain with DRAFT-ASM's / VFY-ONE's)."""
import ast
import copy

FLAGS = {"VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT", "VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY"}
DEFS = {
    "HiddenMajorLMHeadMethod",
    "hidden_major_refusal",
    "_hidden_major_weight",
    "relayout_lm_head",
    "hidden_major_head_copy",
    "_head_census",
    "arm_lm_head_hidden_major",
}
VPE = "vllm.model_executor.layers.vocab_parallel_embedding"


def _reads(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in FLAGS:
            return True
        if isinstance(n, ast.Name) and n.id in FLAGS:
            return True
    return False


class _Off(ast.NodeTransformer):
    def __init__(self, imports):
        # The two imports are dropped only in the module that defines the lever (runner/utils.py);
        # tpu_runner.py has its own `import copy`.
        self.imports = imports

    def visit_If(self, node):
        self.generic_visit(node)
        if _reads(node.test):
            return node.orelse or None
        return node

    def visit_FunctionDef(self, node):
        if node.name in DEFS:
            return None
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        if node.name in DEFS:
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in FLAGS:
            return None
        # envs.py's `environment_variables: dict[...] = {...}` is an annotated assignment: visit its dict.
        return self.generic_visit(node)

    def visit_Dict(self, node):
        self.generic_visit(node)
        keep = [
            (k, v)
            for k, v in zip(node.keys, node.values)
            if not (isinstance(k, ast.Constant) and k.value in FLAGS)
        ]
        node.keys = [k for k, _ in keep]
        node.values = [v for _, v in keep]
        return node

    def visit_Import(self, node):
        if self.imports:
            node.names = [a for a in node.names if a.name != "copy"]
        return node if node.names else None

    def visit_ImportFrom(self, node):
        if self.imports and node.module == VPE:
            node.names = [
                a for a in node.names if a.name not in ("ParallelLMHead", "UnquantizedEmbeddingMethod")
            ]
        return node if node.names else None


def off(src, extra=None):
    tree = copy.deepcopy(ast.parse(src)) if extra is None else extra(src)
    defines = any(isinstance(n, ast.FunctionDef) and n.name == "arm_lm_head_hidden_major" for n in tree.body)
    tree = _Off(defines).visit(tree)
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


def check(new_src, ref_src, extra=None):
    new, ref = off(new_src, extra), ast.parse(ref_src)
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
