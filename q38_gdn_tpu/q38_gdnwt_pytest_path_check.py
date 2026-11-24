#!/usr/bin/env python3
"""q38_gdnwt_pytest_path_check.py <tree> [--expect-fail]: DEC-FB-2 GDNWT relaunch proof (CDK CPU job). Runs the driver's own
ensure_pytest() and then timing()'s in-process import (`import test_gdn_v3_window_tile`) in an interpreter where pytest is
HIDDEN (a meta-path finder refuses the system pytest; only the driver's private --target dir may provide it), as on the
bench image (T71 j-d342e536: "ModuleNotFoundError: No module named 'pytest'" at test_gdn_v3_window_tile.py:62 via
q38_gdn_wt_tpu.py:122). PASS = the import succeeds and pytest comes from the private dir; with --expect-fail (the
fix-reverted mutant) PASS = ModuleNotFoundError naming pytest."""
import importlib.abc
import importlib.util
import os
import shutil
import sys

tree = os.path.abspath(sys.argv[1])
expect_fail = "--expect-fail" in sys.argv
PRIVATE = "/tmp/q38_gdn_pytest"
shutil.rmtree(PRIVATE, ignore_errors=True)


class HidePytest(importlib.abc.MetaPathFinder):
    """Refuse pytest / _pytest / pluggy etc. from anywhere except the driver's private dir (the bench image has none)."""
    NAMES = ("pytest", "_pytest")

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in self.NAMES:
            return None
        if os.path.isdir(PRIVATE) and PRIVATE in sys.path:
            spec = importlib.machinery.PathFinder.find_spec(name, [PRIVATE] if path is None else path)
            if spec is not None and spec.origin and spec.origin.startswith(PRIVATE):
                return spec
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)


# Where the system pytest lives (before hiding): a fallback source if this pod cannot reach PyPI for ensure_pytest's pip.
SYS_PKGS = {}
for n in ("pytest", "_pytest", "pluggy", "iniconfig"):
    sp = importlib.util.find_spec(n)
    if sp is not None and sp.origin:
        SYS_PKGS[n] = os.path.dirname(sp.origin) if sp.origin.endswith("__init__.py") else sp.origin
for n in list(sys.modules):
    if n.split(".")[0] in HidePytest.NAMES:
        del sys.modules[n]
sys.meta_path.insert(0, HidePytest())
try:
    import pytest  # noqa: F401
    print("Q38_PYTEST_PATH hide FAILED: pytest importable before ensure_pytest"); sys.exit(2)
except ModuleNotFoundError:
    print("Q38_PYTEST_PATH pytest hidden before ensure_pytest: ok", flush=True)

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.join(tree, "src"))
sys.path.insert(0, os.path.join(tree, "tests", "kernels"))
spec = importlib.util.spec_from_file_location("q38_drv", os.path.join(tree, "q38_gdn_tpu", "q38_gdn_wt_tpu.py"))
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)
import subprocess as _sp
_real_run = _sp.run


def _pip_or_copy(cmd, *a, **k):
    """ensure_pytest's own pip call; if this pod cannot install (no PyPI), populate the private dir by copying the
    system packages instead and say so -- the property under test is the driver's sys.path handling, not pip."""
    try:
        return _real_run(cmd, *a, **k)
    except Exception as e:  # noqa: BLE001
        if "pip" not in cmd:
            raise
        os.makedirs(PRIVATE, exist_ok=True)
        for n, src_ in SYS_PKGS.items():
            dst = os.path.join(PRIVATE, os.path.basename(src_))
            (shutil.copytree if os.path.isdir(src_) else shutil.copy)(src_, dst)
        print(f"Q38_PYTEST_PATH pip unavailable ({type(e).__name__}); private dir populated by copy of {sorted(SYS_PKGS)}",
              flush=True)
        return _sp.CompletedProcess(cmd, 0)


drv.subprocess.run = _pip_or_copy
src = drv.ensure_pytest()
print(f"Q38_PYTEST_PATH ensure_pytest -> {src}; private dir on sys.path: {PRIVATE in sys.path}", flush=True)
try:
    import test_gdn_v3_window_tile as tw  # noqa: F401  (timing()'s import, q38_gdn_wt_tpu.py:122)
    import pytest as p2
    ok = (p2.__file__ or "").startswith(PRIVATE)
    print(f"Q38_PYTEST_PATH timing import OK; pytest from {p2.__file__}", flush=True)
    res = "PASS" if (ok and not expect_fail) else "FAIL"
except ModuleNotFoundError as e:
    print(f"Q38_PYTEST_PATH timing import raised ModuleNotFoundError: {e}", flush=True)
    res = "PASS" if (expect_fail and "pytest" in str(e)) else "FAIL"
print(f"Q38_PYTEST_PATH {'mutant (fix reverted)' if expect_fail else 'fixed'} -> {res}", flush=True)
sys.exit(0 if res == "PASS" else 1)
