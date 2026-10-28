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
"""Every global a megakernel runtime function reads is defined in its module.

Most driver paths (`Qwen38Segments.__init__`, the whole-step attach, warmup) only run with a
served model on TPU, so a missing import there passes every unit test and fails at
`load_model` (760dfe6cb: `hbm_summary` used in qwen38_driver.py without an import). This
check reads the symbol tables (stdlib `symtable`, no import of the module, so it runs on any
CPU): a name a function references as a global must be bound at module level or a builtin.
"""

import builtins
import pathlib
import symtable

import pytest

RUNTIME = pathlib.Path(__file__).resolve().parents[2] / "src" / "vllm_torchtpu" / "megakernel_runtime"
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__spec__", "__package__", "__loader__"}


def unresolved_globals(source: str, filename: str) -> list[tuple[str, str]]:
    """`(scope, name)` of every global a nested scope references that nothing binds."""
    top = symtable.symtable(source, filename, "exec")
    bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    bound |= set(dir(builtins)) | MODULE_DUNDERS
    bound |= {c.get_name() for c in top.get_children()}  # module-level def / class

    out = []

    def walk(table):
        for child in table.get_children():
            for s in child.get_symbols():
                if s.is_referenced() and s.is_global() and s.get_name() not in bound:
                    out.append((child.get_name(), s.get_name()))
            walk(child)

    walk(top)
    return out


@pytest.mark.parametrize("path", sorted(RUNTIME.glob("*.py")), ids=lambda p: p.name)
def test_runtime_functions_read_only_bound_globals(path):
    assert RUNTIME.is_dir()
    assert unresolved_globals(path.read_text(), str(path)) == []


def test_check_catches_a_missing_import():
    """Positive control: the 760dfe6cb shape (a function calling an un-imported helper)."""
    source = "from .x import a\n\nclass C:\n    def __init__(self):\n        a()\n        hbm_summary()\n"
    assert unresolved_globals(source, "m.py") == [("__init__", "hbm_summary")]
