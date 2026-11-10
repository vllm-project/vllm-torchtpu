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
"""Main's fused Conv1D-GDN v3 kernel tests, run against the Qwen3.5 copy.

The copy in models/qwen3_5/kernels/gdn starts from kernels/gdn/v3, and the
Qwen3.5 changes keep its interface. Each test class of
tests/kernels/test_gdn_attention_v3.py is collected here a second time with
the kernel modules swapped for the copy's, so the copy is held to the same
checks (the TPU-only classes run in TPU CI, the rest on CPU). The tile
selection class is left out: it pins main's tiler picks, and the copy runs
the Qwen3.5 tiler.
"""

import pathlib
import types

_MAIN_TESTS = (
    pathlib.Path(__file__).resolve().parents[2] / "kernels" / "test_gdn_attention_v3.py"
)
_MAIN_IMPORT = (
    "from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, tiling, wrapper"
)
_COPY_IMPORT = (
    "from vllm_torchtpu.models.qwen3_5.kernels.gdn import "
    "compute_gdn, config, tiling, wrapper"
)


def _load_against_copy() -> types.ModuleType:
    source = _MAIN_TESTS.read_text()
    assert source.count(_MAIN_IMPORT) == 1, "main's GDN v3 test imports changed"
    module = types.ModuleType(f"{__name__}._against_copy")
    module.__file__ = str(_MAIN_TESTS)
    code = compile(source.replace(_MAIN_IMPORT, _COPY_IMPORT), str(_MAIN_TESTS), "exec")
    exec(code, module.__dict__)  # noqa: S102
    return module


_suite = _load_against_copy()
assert _suite.wrapper.__name__.startswith("vllm_torchtpu.models.qwen3_5.")

# Classes whose expectations are main's tuning, not kernel behaviour: the copy
# runs the Qwen3.5 tiler, whose picks test_qwen3_5_gdn_tiling.py checks.
_NOT_APPLICABLE = {"GDNAttentionTileSelectionTest"}

for _name, _obj in vars(_suite).items():
    if (
        isinstance(_obj, type)
        and _name.startswith("GDN")
        and _name not in _NOT_APPLICABLE
    ):
        globals()[f"Copy{_name}"] = type(f"Copy{_name}", (_obj,), {})
del _name, _obj
