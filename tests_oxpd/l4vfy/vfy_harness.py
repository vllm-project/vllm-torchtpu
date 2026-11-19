# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VFY-ONE test harness (VFY-BUILD): loads the tree's runner/vfy_one.py, core/l6_ab.py and
megakernel_runtime/served_ab.py as private module copies (one copy per simulated TP rank where a test
needs per-rank module state), with a stub vllm_torchtpu.logger. Q38MM_SRC_ROOT points at a mutated copy,
as in tests/q38mm_dstock/test_vps_wiring.py. Torch CPU only; nothing here imports the real package."""

import contextlib
import importlib.util
import logging
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("Q38MM_SRC_ROOT") or os.path.join(HERE, "..", "..", "src")
PKG = os.path.join(ROOT, "vllm_torchtpu")
VFY = os.path.join(PKG, "runner", "vfy_one.py")
L6AB = os.path.join(PKG, "core", "l6_ab.py")
SAB = os.path.join(PKG, "megakernel_runtime", "served_ab.py")
AGREE = os.path.join(PKG, "runner", "l4_agree.py")  # q38 L4 AGREE: the runner's engine-step agreement clock
LOADED: dict = {}  # private name -> file (the origin check reads it)


def _stub_logger():
    m = types.ModuleType("vllm_torchtpu.logger")
    m.init_logger = lambda name: logging.getLogger(name)
    return m


def _pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []
    return m


@contextlib.contextmanager
def modules(extra: dict):
    """Install `extra` into sys.modules for the duration, then restore exactly what was there."""
    saved = {n: sys.modules.get(n) for n in extra}
    sys.modules.update(extra)
    try:
        yield
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def _load(name: str, path: str, extra: dict | None = None):
    mods = {"vllm_torchtpu.logger": _stub_logger()}
    if "vllm_torchtpu" not in sys.modules:
        mods["vllm_torchtpu"] = _pkg("vllm_torchtpu")
    mods.update(extra or {})
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # registered under its private name (dataclasses look the defining module up in sys.modules)
    sys.modules[name] = mod
    with modules(mods):
        spec.loader.exec_module(mod)
    LOADED[name] = os.path.realpath(path)
    return mod


def load_vfy(tag: str = "main"):
    return _load(f"q38_l4vfy_vfy_one_{tag}", VFY)


def load_agree(tag: str = "main"):
    """A private copy of runner/l4_agree.py (q38 L4 AGREE; torch + logger only)."""
    return _load(f"q38_l4vfy_l4_agree_{tag}", AGREE)


def load_sab(tag: str = "main"):
    return _load(f"q38_l4vfy_served_ab_{tag}", SAB)


def l6_modules(sab) -> dict:
    """What l6_ab's lazy `from vllm_torchtpu.megakernel_runtime import served_ab` needs."""
    mr = _pkg("vllm_torchtpu.megakernel_runtime")
    mr.served_ab = sab
    return {"vllm_torchtpu.megakernel_runtime": mr, "vllm_torchtpu.megakernel_runtime.served_ab": sab,
            "vllm_torchtpu.logger": _stub_logger()}


def load_l6(tag: str, sab):
    """A private copy of core/l6_ab.py resolved from the current os.environ."""
    return _load(f"q38_l4vfy_l6ab_{tag}", L6AB, l6_modules(sab))


def resolve_l6(l6, sab):
    with modules(l6_modules(sab)):
        l6._resolve()


def l6_begin(l6, sab, fp="fp"):
    """One l6_ab step. begin_step imports nothing, so no sys.modules change (simulated ranks run it from threads)."""
    if l6.ACTIVE:
        l6.begin_step(fp_fn=lambda: fp)
