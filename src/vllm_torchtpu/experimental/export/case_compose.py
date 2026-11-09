"""Composes per-request-count variants of one module into a single module.

PROTOTYPE (exp/draft-cond): TorchTPU's trace cannot lower `torch.cond`, so a
drafter that computes only the step's live requests is lowered once per slot
count `N` and the variants are merged here, at the StableHLO text level, into
one `@main` that dispatches on the step's request count with `stablehlo.case`.
The merged module keeps the signature of the largest variant (its argument and
result attributes, donation and aliasing included), so the artifact contract is
unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_FUNC = re.compile(r"func\.func (?:public |private )?@([\w.$-]+)")
_MAIN = re.compile(r"func\.func public @main\((?P<args>.*?)\) -> \((?P<rets>.*?)\) \{", re.S)
_ARG = re.compile(r"(%arg\d+): (tensor<[^>]*>)")
_TENSOR = re.compile(r"tensor<[^>]*>")
_ATTR_DICT = re.compile(r" \{[^{}]*\}")


def _module_header(text: str) -> str:
    start = text.index("module")
    return text[start : text.index("\n", start)]


def _branch_body(text: str, n: int) -> str:
    """The module's functions with `@main` renamed `@branch_{n}`, every other
    symbol prefixed, all made private and stripped of argument attributes."""
    for name in sorted(set(_FUNC.findall(text)), key=len, reverse=True):
        new = f"branch_{n}" if name == "main" else f"b{n}_{name}"
        text = re.sub(rf"@{re.escape(name)}\b", f"@{new}", text)
    start = text.index("\n", text.index("module")) + 1
    body = text[start : text.rindex("}")]
    body = body.replace("func.func public", "func.func private")
    # Argument and result attributes (donation, aliasing, result info) belong to
    # the composed `@main` only.
    lines = []
    for line in body.splitlines():
        if line.lstrip().startswith("func.func private @"):
            line = _ATTR_DICT.sub("", line[: line.rindex("{")]) + "{"
        lines.append(line)
    return "\n".join(lines) + "\n"


def compose_by_request_count(
    variants: Mapping[int, str],
    *,
    req_dist_arg: int,
    num_seqs_index: int = 2,
) -> str:
    """Merges `variants` (`N` -> MLIR text whose `@main` computes `N` request
    slots) into one module whose `@main` runs the smallest `N` that holds the
    step's request count, read from element `num_seqs_index` of argument
    `req_dist_arg` (`i32[k]`). Every variant must have the same `@main`
    signature; the largest `N` must hold every step.
    """
    ns = sorted(variants)
    largest = variants[ns[-1]]
    sig = _MAIN.search(largest)
    if sig is None:
        raise ValueError("variant has no public @main")
    arg_types = dict(_ARG.findall(sig.group("args")))
    names = list(arg_types)
    types = [arg_types[a] for a in names]
    ret_types = _TENSOR.findall(_ATTR_DICT.sub("", sig.group("rets")))
    for n in ns[:-1]:
        other = _MAIN.search(variants[n])
        if other is None or [t for _, t in _ARG.findall(other.group("args"))] != types:
            raise ValueError(f"variant {n} has another @main signature")

    rd_type = types[req_dist_arg]
    k = len(ns)
    rt = ", ".join(ret_types)
    call = f"{', '.join(names)}) : ({', '.join(types)}) -> ({rt})"
    regions = "    }, {\n".join(
        f"      %r:{len(ret_types)} = func.call @branch_{n}({call}\n"
        f"      stablehlo.return {', '.join(f'%r#{i}' for i in range(len(ret_types)))} : {rt}\n"
        for n in ns
    )
    main = f"""  func.func public @main({sig.group("args")}) -> ({sig.group("rets")}) {{
    %s1 = stablehlo.slice %{names[req_dist_arg][1:]} [{num_seqs_index}:{num_seqs_index + 1}] : ({rd_type}) -> tensor<1xi32>
    %s = stablehlo.reshape %s1 : (tensor<1xi32>) -> tensor<i32>
    %ns = stablehlo.constant dense<[{", ".join(map(str, ns))}]> : tensor<{k}xi32>
    %sb = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<i32>) -> tensor<{k}xi32>
    %lt = stablehlo.compare LT, %ns, %sb : (tensor<{k}xi32>, tensor<{k}xi32>) -> tensor<{k}xi1>
    %one = stablehlo.constant dense<1> : tensor<{k}xi32>
    %zero = stablehlo.constant dense<0> : tensor<{k}xi32>
    %below = stablehlo.select %lt, %one, %zero : tensor<{k}xi1>, tensor<{k}xi32>
    %z = stablehlo.constant dense<0> : tensor<i32>
    %count = stablehlo.reduce(%below init: %z) applies stablehlo.add across dimensions = [0] : (tensor<{k}xi32>, tensor<i32>) -> tensor<i32>
    %last = stablehlo.constant dense<{k - 1}> : tensor<i32>
    %idx = stablehlo.minimum %count, %last : tensor<i32>
    %out:{len(ret_types)} = "stablehlo.case"(%idx) ({{
{regions}    }}) : (tensor<i32>) -> ({rt})
    return {", ".join(f"%out#{i}" for i in range(len(ret_types)))} : {rt}
  }}
"""
    bodies = "".join(_branch_body(variants[n], n) for n in ns)
    return f"{_module_header(largest)}\n{bodies}{main}}}\n"
