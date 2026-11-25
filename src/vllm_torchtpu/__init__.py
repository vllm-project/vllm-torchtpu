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
"""Form gate for the shared q38-best-per-band branch (q38 MERGE-LEAD, 2026-11-19).

The package carries two complete forms of the plugin and picks one per process,
once, when ``vllm_torchtpu`` is first imported, from ``Q38_OXPD_FORM``:

* unset or ``0`` (the default): Benliang Wang's form. Its original ``__init__``
  body is stored byte for byte in ``_bl_init.py`` and executes here, in this
  module's namespace; every submodule loads from this directory as before.
* ``1``: the q38 oxpd form, the ``src/vllm_torchtpu`` tree of q38 commit
  2bc75911f5 ("w3"), stored byte for byte in ``_forms/oxpd/vllm_torchtpu/``.
  The package ``__path__`` is pointed there, so every ``vllm_torchtpu.*``
  submodule loads from it, and that tree's ``__init__`` executes here. The
  extra ``_forms/oxpd/`` level keeps w3's own path logic exact: its
  tpu_compiler hashes kernel sources relative to ``Path(__file__).parents[2]``,
  which is ``_forms/oxpd/`` here and ``src/`` in a w3 checkout.

Any other value is refused. Nothing from ``_forms/`` is imported when the gate
is unset, and nothing from this directory except this file runs when it is set.
The two forms cannot be mixed in one process.

Named q38 forms (q38 MERGE-1124, 2026-11-24), from ``Q38_FORM``: unset or empty
changes nothing above. ``Q38_FORM=<name>`` runs ``_forms/<name>/vllm_torchtpu/``
exactly the way ``Q38_OXPD_FORM=1`` runs the oxpd form. Each is the
``vllm_torchtpu`` package of the tree a q38 dash point ran, byte for byte
(``_forms/<name>/__init__.py`` names the commit; the point table is
``q38_bands/merge_1124/README.md``): ``k_cut1``, ``k_pkg``, ``k_sup``, ``oxk``,
``d1121``, ``k_sb11``; and the LIVING forms ``b1``, ``hix`` and ``b3``, where new q38 levers land env-keyed
and default-off (``_forms/<form>/__init__.py`` names the lineage). ``q38f`` (Phase 3, 2026-11-25) is the oxpd
package of shagarw/q38f-1124-opt 022524d309 (the q38f-1124-final c4-c64 points), byte for byte
(``_forms/q38f/__init__.py``). Any other name is refused, and so is ``Q38_FORM`` together with
``Q38_OXPD_FORM=1``.
"""

import os as _q38_os

_q38_form = _q38_os.environ.get("Q38_OXPD_FORM", "").strip()
if _q38_form not in ("", "0", "1"):
    raise RuntimeError(
        f"Q38_OXPD_FORM must be unset, 0 or 1 (got {_q38_form!r}); refusing to guess a form"
    )
_q38_named = _q38_os.environ.get("Q38_FORM", "").strip()
if _q38_named:
    if _q38_named not in ("k_cut1", "k_pkg", "k_sup", "oxk", "d1121", "k_sb11", "b1", "hix", "b3", "q38f"):
        raise RuntimeError(
            f"Q38_FORM must be unset or one of k_cut1, k_pkg, k_sup, oxk, d1121, k_sb11, b1, hix, b3, q38f (got {_q38_named!r});"
            " refusing to guess a form"
        )
    if _q38_form == "1":
        raise RuntimeError(
            f"Q38_FORM={_q38_named} and Q38_OXPD_FORM=1 select two forms; set only one"
        )
    __path__[:] = [_q38_os.path.join(_q38_os.path.dirname(__file__), "_forms", _q38_named, "vllm_torchtpu")]
    _q38_body = _q38_os.path.join(__path__[0], "__init__.py")
    __file__ = _q38_body
    import sys as _q38_sys

    print(f"q38 form gate: Q38_FORM={_q38_named}, vllm_torchtpu runs the {_q38_named} form from {__path__[0]}",
          file=_q38_sys.stderr, flush=True)
    del _q38_sys
elif _q38_form == "1":
    __path__[:] = [_q38_os.path.join(_q38_os.path.dirname(__file__), "_forms", "oxpd", "vllm_torchtpu")]
    _q38_body = _q38_os.path.join(__path__[0], "__init__.py")
    __file__ = _q38_body
    import sys as _q38_sys

    print(f"q38 form gate: Q38_OXPD_FORM=1, vllm_torchtpu runs the oxpd form from {__path__[0]}",
          file=_q38_sys.stderr, flush=True)
    del _q38_sys
else:
    _q38_body = _q38_os.path.join(_q38_os.path.dirname(__file__), "_bl_init.py")
with open(_q38_body, "rb") as _q38_f:
    _q38_code = compile(_q38_f.read(), _q38_body, "exec")
del _q38_os, _q38_form, _q38_named, _q38_body, _q38_f
exec(_q38_code)
del _q38_code
