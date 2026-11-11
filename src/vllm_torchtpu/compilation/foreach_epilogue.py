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
"""Batch AOTAutograd input-mutation write-backs into one `torch._foreach_copy_`."""

from __future__ import annotations

import re
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

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

_installed = False
_active = False


def parse_pairs(source: str) -> list[tuple[int, int]] | None:
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
    if len({o for o, _ in pairs}) != len(pairs) or len({u for _, u in pairs}) != len(
        pairs
    ):
        return None
    return pairs


def _can_foreach(dsts: list[torch.Tensor], srcs: list[torch.Tensor]) -> bool:
    for t in dsts:
        if t.requires_grad:
            return False
    for t in srcs:
        if t.requires_grad:
            return False
    if dsts and dsts[0].device.type == "tpu":
        dev0 = dsts[0].device
        return all(t.device == dev0 for t in dsts) and all(
            t.device == dev0 for t in srcs
        )
    by_ptr: dict[int, list[tuple[int, int]]] = {}
    for t in (*dsts, *srcs):
        if t.numel() == 0:
            continue
        if any(s < 0 for s in t.stride()):
            return False
        ptr = t.untyped_storage().data_ptr()
        if ptr == 0:
            continue
        item = int(t.element_size())
        lo = int(t.storage_offset()) * item
        span = sum(
            (int(sz) - 1) * int(st) for sz, st in zip(t.shape, t.stride()) if sz > 1
        )
        by_ptr.setdefault(ptr, []).append((lo, lo + (span + 1) * item))
    for spans in by_ptr.values():
        if len(spans) > 1:
            spans.sort()
            if any(spans[k][0] < spans[k - 1][1] for k in range(1, len(spans))):
                return False
    return True


def rewrite(source: str) -> str | None:
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
            "    if not _tpu_can_foreach(_d, _s):",
            "        return _tpu_original_apply_mutations(orig_inputs, updated_inputs)",
            "    torch._foreach_copy_(_d, _s)",
        ]
    )


def set_active(active: bool) -> None:
    global _active
    _active = active


def install() -> bool:
    global _installed, _active
    _active = True
    if _installed:
        return True
    from torch._functorch._aot_autograd import subclass_codegen as scg

    original = scg._compile_and_exec_source

    def compile_and_exec_source(
        source: str,
        globals_dict: dict[str, Any],
        fn_name: str,
        artifact_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not _active or fn_name != "_apply_mutations":
            return original(
                source, globals_dict, fn_name, artifact_name, *args, **kwargs
            )
        pairs = parse_pairs(source)
        new_source = rewrite(source) if pairs is not None else None
        if new_source is None or pairs is None:
            return original(
                source, globals_dict, fn_name, artifact_name, *args, **kwargs
            )
        orig_fn = original(
            source, dict(globals_dict), fn_name, artifact_name, *args, **kwargs
        )
        scoped_globals = dict(globals_dict)
        scoped_globals["_tpu_original_apply_mutations"] = orig_fn
        scoped_globals["_tpu_can_foreach"] = _can_foreach
        return original(
            new_source,
            scoped_globals,
            fn_name,
            artifact_name + "_foreach",
            *args,
            **kwargs,
        )

    compile_and_exec_source._tpu_original = original  # type: ignore[attr-defined]
    scg._compile_and_exec_source = compile_and_exec_source
    _installed = True
    logger.info("Installed batched foreach AOTAutograd mutation epilogue hook")
    return True
