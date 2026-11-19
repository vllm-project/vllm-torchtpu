# SPDX-License-Identifier: Apache-2.0
"""Load the real GatedResidual for the HCX tests (q38 L3-HCX, port of q38mm 20e6fcf23).

Two modes (``Q38_L3_HCX_MODE``; inferred when unset):

* ``image`` (the CDK CPU image): the real packages. ``vllm_torchtpu`` must resolve
  under ``Q38_L3_HCX_SRC_ROOT`` (default: this tree's ``src``), so a mutated copy
  is what runs; the loader refuses otherwise.
* ``mac`` (no vLLM installed): package stubs whose ``__path__`` points into
  ``Q38_L3_HCX_SRC_ROOT``, so the real ``envs.py``, ``hc_decode.py``,
  ``hc_engagement.py`` and ``hyperconnection.py`` run without executing the
  packages' ``__init__`` files; ``vllm.models.qwen4_exp.common.hyperconnection``
  is vLLM v0.31.0's pure-torch file at ``Q38_L3_HCX_VLLM_HC``.

In both modes the FINAL 5d368f166 ``hyperconnection.py`` (``data/``, sha256
pinned) is loaded beside it as ``base`` for the flags-off identity tests.
"""

import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_HC = os.path.join(HERE, "data", "hyperconnection_5d368f166.py.txt")
# git rev-parse 5d368f166:src/vllm_torchtpu/models/vllm/qwen4_exp/hyperconnection.py = 63ded874c
BASE_HC_SHA256 = "74b76118d2e14a72f832c576d1b6708fdca0d746b74c25b2912ffa7dd8f0c50c"
DEFAULT_VLLM_HC = (
    "/Users/shriharim/repos/q38/scratch/decpc/src_vllm/vllm-v0.31.0/vllm/models/qwen4_exp/common/hyperconnection.py"
)
HC_MOD = "vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection"

_CACHE: dict = {}


def src_root() -> str:
    return os.path.abspath(os.environ.get("Q38_L3_HCX_SRC_ROOT") or os.path.join(TREE, "src"))


def mode() -> str:
    m = os.environ.get("Q38_L3_HCX_MODE", "").strip()
    if m:
        if m not in ("mac", "image"):
            raise RuntimeError(f"Q38_L3_HCX_MODE must be mac or image, got {m!r}")
        return m
    return "image" if importlib.util.find_spec("vllm") is not None else "mac"


def _file_module(name: str, path: str):
    # An explicit source loader: the base fixture's .txt suffix has no default one.
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str, path: str) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [path]
    sys.modules[name] = package


def _load_mac(root: str):
    vllm_hc = os.environ.get("Q38_L3_HCX_VLLM_HC") or DEFAULT_VLLM_HC
    if not os.path.isfile(vllm_hc):
        raise RuntimeError(f"mac mode needs vLLM v0.31.0's common hyperconnection.py at {vllm_hc}")
    pkg = os.path.join(root, "vllm_torchtpu")
    for name, rel in (
        ("vllm_torchtpu", ""),
        ("vllm_torchtpu.kernels", "kernels"),
        ("vllm_torchtpu.kernels.qwen4_exp", "kernels/qwen4_exp"),
        ("vllm_torchtpu.models", "models"),
        ("vllm_torchtpu.models.vllm", "models/vllm"),
        ("vllm_torchtpu.models.vllm.qwen4_exp", "models/vllm/qwen4_exp"),
    ):
        _package(name, os.path.join(pkg, rel))
    vdir = os.path.dirname(vllm_hc)
    for name in ("vllm", "vllm.models", "vllm.models.qwen4_exp", "vllm.models.qwen4_exp.common"):
        _package(name, vdir)
    if "vllm.models.qwen4_exp.common.hyperconnection" not in sys.modules:
        _file_module("vllm.models.qwen4_exp.common.hyperconnection", vllm_hc)
    sys.modules["vllm_torchtpu"].envs = importlib.import_module("vllm_torchtpu.envs")
    return importlib.import_module(HC_MOD)


def load():
    """Namespace: hc (new hyperconnection), base (5d368f166 hyperconnection), hcd (hc_decode),
    envs, mode, root."""
    root = src_root()
    if root in _CACHE:
        return _CACHE[root]
    m = mode()
    hc = _load_mac(root) if m == "mac" else importlib.import_module(HC_MOD)
    if not os.path.abspath(hc.__file__).startswith(root + os.sep):
        raise RuntimeError(f"{HC_MOD} resolved to {hc.__file__}, not under {root} (set PYTHONPATH)")
    with open(BASE_HC, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != BASE_HC_SHA256:
        raise RuntimeError(f"base fixture {BASE_HC} sha256 {digest} != {BASE_HC_SHA256}")
    base = _file_module("q38_l3_hcx_base_hyperconnection", BASE_HC)
    hcd = importlib.import_module("vllm_torchtpu.kernels.qwen4_exp.hc_decode")
    envs = importlib.import_module("vllm_torchtpu.envs")
    for module in (hcd, envs):
        if not os.path.abspath(module.__file__).startswith(root + os.sep):
            raise RuntimeError(f"{module.__name__} resolved to {module.__file__}, not under {root}")
    ns = types.SimpleNamespace(hc=hc, base=base, hcd=hcd, envs=envs, mode=m, root=root)
    _CACHE[root] = ns
    return ns
