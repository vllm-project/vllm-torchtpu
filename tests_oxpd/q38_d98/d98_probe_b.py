# SPDX-License-Identifier: Apache-2.0
"""q38 L4 D98 option (b) feasibility probe (design evidence, no build).

For real decode helpers compiled with the aot_eager stand-in, print how the
Dynamo-transformed frame feeds the compiled graph: which locals / attribute
chains it loads before calling ``__compiled_fn_*`` (the graph-input mapping a
deeper bypass would have to reproduce), what the compiled fn unwraps to
(AOTAutograd's forward -> codegen'd runtime wrapper -> inner compiled fn), and
whether the runtime wrapper's source / mutation metadata is reachable from
Python (it decides whether a generic call below AOTAutograd can be proven
equivalent).
"""

import dis
import inspect
import sys

import torch
from torch._C._dynamo import eval_frame as E


def probe(name, fn, args):
    c = torch.compile(fn, backend="aot_eager", fullgraph=True, dynamic=False)
    with torch.no_grad():
        c(*args)
    e = E._debug_get_cache_entry_list(fn.__code__)[0]
    ins = list(dis.get_instructions(e.code))
    gidx = next(i for i, x in enumerate(ins)
                if x.opname == "LOAD_GLOBAL" and str(x.argval).startswith("__compiled_fn"))
    # Every local / attribute / constant loaded between the compiled-fn load
    # and the store of its result (graph_out_*): the graph's input sources.
    loads = []
    for x in ins[gidx + 1:]:
        if x.opname == "STORE_FAST" and str(x.argval).startswith("graph_out"):
            break
        if x.opname.startswith("LOAD_FAST") or x.opname in ("LOAD_DEREF", "LOAD_ATTR",
                                                             "BINARY_SUBSCR", "LOAD_CONST"):
            v = x.argval if not isinstance(x.argval, tuple) else "+".join(map(str, x.argval))
            if not str(v).startswith(("tmp_", "record_")):
                loads.append(f"{x.opname}:{v}")
    cf = fn.__globals__[ins[gidx].argval]
    inner = cf
    while hasattr(inner, "_torchdynamo_orig_callable"):
        inner = inner._torchdynamo_orig_callable
    cells = dict(zip(inner.__code__.co_freevars, inner.__closure__ or ()))
    rt = cells.get("compiled_fn")
    rt = rt.cell_contents if rt is not None else None
    rcells = dict(zip(rt.__code__.co_freevars, rt.__closure__ or ())) if rt else {}
    w = rcells.get("_codegen_runtime_wrapper")
    src = "n/a"
    if w is not None:
        try:
            inspect.getsource(w.cell_contents)
            src = "available"
        except (OSError, TypeError):
            src = "NOT available (codegen)"
    print(f"D98B {name}: guards={len(e.guard_manager.code_parts)} "
          f"graph_inputs_from=[{', '.join(loads)}] compiled_fn->{getattr(inner, '__qualname__', inner)} "
          f"runtime_wrapper_closure={sorted(rcells)} codegen_source={src}")


def main():
    from vllm_torchtpu.runner import tpu_runner as r

    g = torch.Generator().manual_seed(0)
    ro = torch.randint(0, 4, (64,), dtype=torch.int32, generator=g)
    groups = [torch.randperm(64, generator=g)[:16].to(torch.int32) for _ in range(2)]
    nt = torch.randint(0, 100, (12, 4), generator=g)
    probe("_update_read_offsets_impl (in-place, list arg)", r._update_read_offsets_impl,
          (ro, groups, nt, torch.tensor(9)))
    probe("_spec_seed_rows_impl (pure, int arg)", r._spec_seed_rows_impl,
          (torch.randint(0, 3, (16,), dtype=torch.int32, generator=g),
           torch.randint(0, 8, (4, 7), generator=g), 2))
    print(f"D98B python {sys.version.split()[0]} torch {torch.__version__}")


if __name__ == "__main__":
    main()
