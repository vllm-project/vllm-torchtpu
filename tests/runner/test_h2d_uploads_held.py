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
"""Non-blocking host-to-device uploads go through the runner's `_upload`.

torch_tpu reads a non-blocking upload's host memory when the transfer
executes, behind whatever the device is still running. `_upload` holds the
source until a staging fence recorded after it has been waited on; any other
non-blocking upload, or vLLM's helpers that upload host temporaries, would
race the rewrite or the free of its source.
"""

import ast
import pathlib

import vllm_torchtpu

_PACKAGE = pathlib.Path(vllm_torchtpu.__file__).parent
_HELPERS = {"async_tensor_h2d", "copy_slice", "copy_to_gpu"}


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _is_device_to_host(call: ast.Call) -> bool:
    targets = [*call.args[:1], *(k.value for k in call.keywords if k.arg == "device")]
    return any(isinstance(t, ast.Constant) and t.value == "cpu" for t in targets)


class _Scan(ast.NodeVisitor):
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.functions: list[str] = []
        self.offenders: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_Call(self, node: ast.Call) -> None:
        non_blocking = [k.value for k in node.keywords if k.arg == "non_blocking"]
        flagged = _called_name(node) in _HELPERS or (
            bool(non_blocking)
            and not (
                isinstance(non_blocking[0], ast.Constant)
                and non_blocking[0].value is False
            )
            and not _is_device_to_host(node)
        )
        in_upload = self.path.name == "tpu_runner.py" and self.functions[-1:] == [
            "_upload"
        ]
        if flagged and not in_upload:
            self.offenders.append(f"{self.path.relative_to(_PACKAGE)}:{node.lineno}")
        self.generic_visit(node)


def test_non_blocking_uploads_only_inside_upload():
    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        scan = _Scan(path)
        scan.visit(ast.parse(path.read_text()))
        offenders += scan.offenders
    assert offenders == []
