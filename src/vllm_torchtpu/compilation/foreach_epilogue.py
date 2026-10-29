# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One ``torch._foreach_copy_`` for the AOTAutograd input-mutation epilogue
(TPU_FOREACH_MUTATION_EPILOGUE).

The TPU backend compiles with ``keep_inference_input_mutations=False``, so
every input a compiled graph mutates (the KV caches written by each attention
layer: ~156 per GLM-5.3 forward) comes back as an output, and AOTAutograd's
generated epilogue writes it back with one ``orig.copy_(updated)`` per input,
each a separate Python -> dispatcher -> torch_tpu call on the serving thread.

With the flag on, an epilogue made only of such plain write-backs is
regenerated as one ``torch._foreach_copy_(dsts, srcs)`` over the same pairs in
the same order (torch_tpu registers a native PrivateUse1 kernel for it). The
original epilogue is kept and used whenever any destination or source requires
grad, so the autograd branch of the original stays exact. Epilogues with any
other kind of line (metadata mutations, ``set_``, stream errors) are left
untouched.
"""

from __future__ import annotations

import re

from vllm.logger import init_logger

logger = init_logger(__name__)

_HEADER = "def _apply_mutations(orig_inputs, updated_inputs):"
_PAT_IF = re.compile(
    r"^    if orig_inputs\[(\d+)\]\.requires_grad: "
    r"orig_inputs\[(\d+)\]\.detach\(\)\.copy_\(updated_inputs\[(\d+)\]\)$"
)
_PAT_ELSE = re.compile(
    r"^    else: orig_inputs\[(\d+)\]\.copy_\(updated_inputs\[(\d+)\]\)$"
)
_PAT_PLAIN = re.compile(r"^    orig_inputs\[(\d+)\]\.copy_\(updated_inputs\[(\d+)\]\)$")

STATS = {"rewritten": 0, "kept": 0, "pairs": 0}


def parse_pairs(source: str) -> list[tuple[int, int]] | None:
    """(orig index, updated index) per write-back if the epilogue is only
    plain write-backs of distinct inputs, else None."""
    lines = source.split("\n")
    if not lines or lines[0] != _HEADER:
        return None
    pairs: list[tuple[int, int]] = []
    i = 1
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = _PAT_IF.match(line)
        if m:
            o1, o2, u = m.groups()
            m2 = _PAT_ELSE.match(lines[i + 1]) if i + 1 < len(lines) else None
            if o1 != o2 or m2 is None or m2.groups() != (o1, u):
                return None
            pairs.append((int(o1), int(u)))
            i += 2
            continue
        m = _PAT_PLAIN.match(line)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
            i += 1
            continue
        return None
    if len({o for o, _ in pairs}) != len(pairs):
        return None
    return pairs


def rewrite(source: str) -> str | None:
    """The foreach epilogue for `source`, or None to keep the original."""
    pairs = parse_pairs(source)
    if pairs is None or len(pairs) < 2:
        return None
    dsts = ", ".join(f"orig_inputs[{o}]" for o, _ in pairs)
    srcs = ", ".join(f"updated_inputs[{u}]" for _, u in pairs)
    return "\n".join(
        [
            _HEADER,
            f"    _d = [{dsts}]",
            f"    _s = [{srcs}]",
            "    for _t in _d:",
            "        if _t.requires_grad:",
            "            return _h5b_original_apply_mutations("
            "orig_inputs, updated_inputs)",
            "    for _t in _s:",
            "        if _t.requires_grad:",
            "            return _h5b_original_apply_mutations("
            "orig_inputs, updated_inputs)",
            "    torch._foreach_copy_(_d, _s)",
        ]
    )


_installed = False


def install() -> bool:
    """Patch AOTAutograd's codegen hook. Idempotent; True if installed."""
    global _installed
    if _installed:
        return True
    from torch._functorch._aot_autograd import subclass_codegen as scg

    original = scg._compile_and_exec_source

    def compile_and_exec_source(source, globals_dict, fn_name, artifact_name, *a, **k):
        if fn_name != "_apply_mutations":
            return original(source, globals_dict, fn_name, artifact_name, *a, **k)
        new_source = rewrite(source)
        if new_source is None:
            STATS["kept"] += 1
            return original(source, globals_dict, fn_name, artifact_name, *a, **k)
        orig_fn = original(source, dict(globals_dict), fn_name, artifact_name, *a, **k)
        g = dict(globals_dict)
        g["_h5b_original_apply_mutations"] = orig_fn
        STATS["rewritten"] += 1
        STATS["pairs"] += len(parse_pairs(source) or ())
        return original(new_source, g, fn_name, artifact_name + "_foreach", *a, **k)

    compile_and_exec_source._h5b_original = original  # type: ignore[attr-defined]
    scg._compile_and_exec_source = compile_and_exec_source
    _installed = True
    logger.info("TPU_FOREACH_MUTATION_EPILOGUE: AOTAutograd epilogue codegen patched")
    return True
