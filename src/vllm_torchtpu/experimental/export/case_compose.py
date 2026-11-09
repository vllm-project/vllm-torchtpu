"""Composes per-request-count variants of one module into a single module.

TorchTPU's trace cannot lower `torch.cond`, so a drafter that computes only the
step's live requests is lowered once per request count `N`
(`exporter.draft_slot_counts`) and the variants are merged here, at the
StableHLO text level, into one `@main` that runs the smallest `N` holding the
step's requests with `stablehlo.case`. The merged `@main` keeps the signature of
the largest variant, argument and result attributes (sharding, donation,
aliasing) included, so the artifact contract is unchanged: a runtime sees one
`draft_bucket{B}` with the inputs and outputs it always had.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_SYMBOL = re.compile(r"(?:func\.func|sdy\.mesh)\s+(?:public\s+|private\s+)?@([\w.$-]+)")
_MAIN_START = "func.func public @main("


def _close(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index just past the bracket closing `text[start]` (an `open_ch`),
    skipping quoted strings and nested brackets of every kind."""
    depth = 0
    i = start
    pairs = {"(": ")", "{": "}", "[": "]", "<": ">"}
    stack: list[str] = []
    while i < len(text):
        c = text[i]
        if c == '"':
            i += 1
            while text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c in pairs and not (c == "<" and text[i - 1 : i + 1] == "-<"):
            stack.append(pairs[c])
            depth += 1
        elif stack and c == stack[-1]:
            stack.pop()
            depth -= 1
            if depth == 0:
                assert text[start] == open_ch and c == close_ch
                return i + 1
        i += 1
    raise ValueError(f"unbalanced {open_ch!r} at {start}")


def _split_top(text: str) -> list[str]:
    """Splits on commas outside quotes and brackets."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(text):
        c = text[i]
        if c == '"':
            j = i + 1
            while text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            cur.append(text[i : j + 1])
            i = j + 1
            continue
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _type_of(entry: str) -> str:
    """The type of an argument (`%arg0: T {attrs}`) or result (`T {attrs}`)."""
    body = entry.split(":", 1)[1].strip() if entry.startswith("%") else entry.strip()
    if body.startswith("tensor<"):
        return body[: _close(body, len("tensor"), "<", ">")]
    return body.split(" ", 1)[0]


def _module_open(text: str) -> tuple[int, int]:
    """`(start, end)` of the module header: `module @name attributes {...} {`,
    `end` just past its body's opening brace."""
    start = text.index("module")
    brace = text.index("{", start)
    attr_at = text.find("attributes", start)
    if 0 <= attr_at < brace:
        brace = text.index("{", _close(text, text.index("{", attr_at), "{", "}"))
    return start, brace + 1


class _Main:
    """The parsed `@main` of a module: its signature and where it sits."""

    def __init__(self, text: str) -> None:
        start = text.find(_MAIN_START)
        if start < 0:
            raise ValueError("variant has no public @main")
        args_open = start + len(_MAIN_START) - 1
        args_close = _close(text, args_open, "(", ")")
        self.args = _split_top(text[args_open + 1 : args_close - 1])
        rest = text[args_close:]
        m = re.match(r"\s*->\s*", rest)
        if m is None:
            raise ValueError("@main has no results")
        pos = args_close + m.end()
        if text[pos] == "(":
            rets_close = _close(text, pos, "(", ")")
            self.rets = _split_top(text[pos + 1 : rets_close - 1])
        else:
            rets_close = pos + len(_type_of(text[pos:]))
            self.rets = [text[pos:rets_close]]
        self.attrs_and_body = text[rets_close:]
        self.start = start
        self.header_end = rets_close
        self.names = [a.split(":", 1)[0].strip() for a in self.args]
        self.arg_types = [_type_of(a) for a in self.args]
        self.ret_types = [_type_of(r) for r in self.rets]


def _branch_functions(text: str, n: int) -> str:
    """The module's top-level ops with `@main` renamed `@branch_{n}` (private,
    without argument or result attributes) and every other symbol prefixed."""
    main = _Main(text)
    sig = (
        f"func.func private @branch_{n}("
        + ", ".join(f"{a}: {t}" for a, t in zip(main.names, main.arg_types, strict=True))
        + f") -> ({', '.join(main.ret_types)})"
    )
    body = main.attrs_and_body
    # Function attributes after the results (`attributes {...}`) stay off a branch.
    body = body[body.index("{") :]
    text = text[: main.start] + sig + " " + body
    for name in sorted(set(_SYMBOL.findall(text)) - {f"branch_{n}"}, key=len, reverse=True):
        text = re.sub(rf"@{re.escape(name)}\b", f"@b{n}_{name}", text)
    text = text.replace("func.func public", "func.func private")
    _, start = _module_open(text)
    return text[start : text.rindex("}")].strip("\n") + "\n"


def compose_by_request_count(
    variants: Mapping[int, str],
    *,
    req_dist_arg: int,
    num_seqs_index: int = 2,
) -> str:
    """Merges `variants` (`N` -> MLIR text whose `@main` computes `N` request
    slots) into one module whose `@main` runs the smallest `N` that holds the
    step's request count, read from element `num_seqs_index` of argument
    `req_dist_arg` (`i32[k]`), or the largest. Every variant must have the same
    `@main` argument and result types; the largest `N` must hold every step.
    """
    ns = sorted(variants)
    largest_text = variants[ns[-1]]
    largest = _Main(largest_text)
    for n in ns[:-1]:
        other = _Main(variants[n])
        if (other.arg_types, other.ret_types) != (largest.arg_types, largest.ret_types):
            raise ValueError(f"variant {n} has another @main signature")
    names, types = largest.names, largest.arg_types
    rets = largest.ret_types
    rt = ", ".join(rets)
    k = len(ns)
    rd = names[req_dist_arg]
    rd_type = types[req_dist_arg]
    call = f"{', '.join(names)}) : ({', '.join(types)}) -> ({rt})"
    regions = "    }, {\n".join(
        f"      %r:{len(rets)} = func.call @branch_{n}({call}\n"
        f"      stablehlo.return {', '.join(f'%r#{i}' for i in range(len(rets)))} : {rt}\n"
        for n in ns
    )
    head = largest_text[largest.start : largest.header_end]
    attrs = largest.attrs_and_body[: largest.attrs_and_body.index("{")]
    main = f"""  {head}{attrs}{{
    %s1 = stablehlo.slice {rd} [{num_seqs_index}:{num_seqs_index + 1}] : ({rd_type}) -> tensor<1xi32>
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
    %out:{len(rets)} = "stablehlo.case"(%idx) ({{
{regions}    }}) : (tensor<i32>) -> ({rt})
    return {", ".join(f"%out#{i}" for i in range(len(rets)))} : {rt}
  }}
"""
    module_start, body_start = _module_open(largest_text)
    module_head = largest_text[module_start:body_start]
    bodies = "".join(_branch_functions(variants[n], n) for n in ns)
    return f"{module_head}\n{bodies}{main}}}\n"
