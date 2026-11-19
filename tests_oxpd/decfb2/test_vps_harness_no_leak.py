# SPDX-License-Identifier: Apache-2.0
"""The VPS test harnesses leave sys.modules as they found it (torch CPU, no TPU).

tests/q38mm_dstock/test_vp_sampler.py and test_vps_wiring.py (and tests/decfb2/test_vps_fallback.py and
test_vps_fb_wiring.py, which build on them) stub vllm / vllm_torchtpu so that neither has to be installed. The stubs may
exist only while a harness loads and while its own tests run: INT-4's union job i5 (j-f861c524) died with 1165 setup
errors because they stayed in sys.modules and the conftest's autouse `import vllm.logger` failed for every later test.

Each case is one fresh pytest process (--noconftest) over the four harness files plus a probe file collected last, with
a plugin fixture that does what tests/conftest.py's `_reset_warning_once` does around every test:
  * clean:    vllm is not imported before collection; the probe imports vllm.logger at module scope.
  * imported: vllm.logger is imported before collection (tests/conftest.py imports vllm first), so the harnesses replace
              the real entries while they load and must put them back; the whole harness suite runs, then the probe.
The probe asserts that no entry under vllm / vllm_torchtpu was added or replaced by collection or by the harness tests,
and that vllm.logger resolves through the real package. "Real" is the installed vllm when there is one, else a stand-in
package (vllm/__init__.py, vllm/logger.py) on sys.path.
Run: python -m pytest -q -p no:cacheprovider --noconftest tests/decfb2/test_vps_harness_no_leak.py
Mutation-checked by tests/decfb2/q38_decfb2_vps_leak_mutrun.py (Q38_VPS_HARNESS_TESTS points at a mutated copy).
"""

import json
import os
import re
import subprocess
import sys

import pytest

TESTS = os.environ.get("Q38_VPS_HARNESS_TESTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FILES = [os.path.join(TESTS, *p) for p in (("q38mm_dstock", "test_vp_sampler.py"), ("q38mm_dstock", "test_vps_wiring.py"),
                                           ("decfb2", "test_vps_fallback.py"), ("decfb2", "test_vps_fb_wiring.py"))]

DRIVER = r'''
import importlib.util, json, sys
fake, mode, files, probe = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4]
if importlib.util.find_spec("vllm") is None:
    sys.path.append(fake)                   # stand-in vllm package; the installed one wins when present
import pytest

TREES = ("vllm", "vllm_torchtpu")
if mode == "imported":
    import vllm.logger  # noqa: F401        tests/conftest.py imports vllm before any test module
BEFORE = {n: id(m) for n, m in list(sys.modules.items()) if n.split(".")[0] in TREES}


class LikeConftest:
    @pytest.fixture(autouse=True)
    def _q38_like_reset_warning_once(self):
        try:                                # verbatim shape of tests/conftest.py::_reset_warning_once
            import vllm.logger
        except ImportError:
            yield
            return
        vllm.logger._print_warning_once.cache_clear()
        yield
        vllm.logger._print_warning_once.cache_clear()


args = ["-q", "-p", "no:cacheprovider", "--noconftest", *files, probe]
if mode == "clean":
    args += ["-k", "q38leakprobe"]
sys.exit(pytest.main(args, plugins=[LikeConftest()]))
'''

PROBE = r'''
import os
import sys

TREES = ("vllm", "vllm_torchtpu")


def _now():
    return {n: id(m) for n, m in list(sys.modules.items()) if n.split(".")[0] in TREES}


def _changed(ref):
    now = _now()
    return sorted(n for n in set(now) | set(ref) if now.get(n) != ref.get(n))


def _ran(name):
    with open(os.environ["Q38_LEAKPROBE_OUT"], "a") as f:
        f.write(name + "\n")


AFTER_COLLECTION = _changed(sys.modules["__main__"].BEFORE)
import vllm.logger  # noqa: E402            the import every later vllm test (and the conftest fixture) does
SETTLED = _now()


def test_q38leakprobe_collection_left_nothing():
    _ran("collection")
    assert AFTER_COLLECTION == [], AFTER_COLLECTION


def test_q38leakprobe_vllm_logger_is_real():
    _ran("logger")
    import vllm.logger as lg
    assert getattr(sys.modules["vllm"], "__file__", None), "sys.modules['vllm'] is a stub package"
    assert sys.modules["vllm"].logger is lg and lg.__file__


def test_q38leakprobe_tests_left_nothing():
    _ran("tests")
    assert _changed(SETTLED) == [], _changed(SETTLED)
'''

FAKE_LOGGER = '''import functools


@functools.lru_cache(maxsize=None)
def _print_warning_once(msg):
    return None
'''


@pytest.mark.parametrize("mode", ["clean", "imported"])
def test_harnesses_leave_sys_modules_as_found(mode, tmp_path):
    fake = tmp_path / "fakevllm"
    (fake / "vllm").mkdir(parents=True)
    (fake / "vllm" / "__init__.py").write_text("")
    (fake / "vllm" / "logger.py").write_text(FAKE_LOGGER)
    (tmp_path / "q38_leak_driver.py").write_text(DRIVER)
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "test_q38leakprobe.py").write_text(PROBE)
    out = tmp_path / "ran.txt"
    env = dict(os.environ, Q38_LEAKPROBE_OUT=str(out), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(tmp_path / "q38_leak_driver.py"), str(fake), mode, json.dumps(FILES),
                        str(probe_dir / "test_q38leakprobe.py")], capture_output=True, text=True, timeout=900,
                       cwd=tmp_path, env=env)
    log = (r.stdout + r.stderr)[-6000:]
    assert r.returncode == 0, log
    assert out.exists() and sorted(out.read_text().split()) == ["collection", "logger", "tests"], log
    summary = r.stdout.strip().splitlines()[-1]
    assert not re.search(r"\d+ (failed|errors?)\b", summary), summary
    passed = int(re.search(r"(\d+) passed", summary).group(1))
    if mode == "clean":
        assert passed == 3, summary
    else:
        assert passed > 50, summary         # the whole harness suite ran under the conftest-like fixture too
