# SPDX-License-Identifier: Apache-2.0
"""q38 MERGE-1124: the Q38_FORM package gate and the Q38_PROXY_FORM proxy gate.

Each case runs in a fresh interpreter, because the package gate acts once, at
the first import of vllm_torchtpu.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PKG = SRC / "vllm_torchtpu"
PROXY = ROOT / "examples" / "disagg" / "toy_proxy_server.py"
NAMED = ("k_cut1", "k_pkg", "k_sup", "oxk", "d1121", "k_sb11", "b1", "hix", "b3")
PROXY_FORMS = ("w3", "k", "opener")
_PROBE = "import vllm_torchtpu as v; print('Q38PATH', v.__path__[0])"


def _env(**over):
    env = {k: v for k, v in os.environ.items() if k not in ("Q38_FORM", "Q38_OXPD_FORM", "Q38_PROXY_FORM")}
    env["PYTHONPATH"] = str(SRC)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["COLUMNS"] = "10000"  # argparse: no usage wrapping, so --help texts compare line for line
    env.update(over)
    return env


def _run(code, **over):
    return subprocess.run([sys.executable, "-c", code], env=_env(**over), capture_output=True, text=True, timeout=600)


def _path(res):
    lines = [x for x in res.stdout.splitlines() if x.startswith("Q38PATH ")]
    assert lines, res.stdout + res.stderr
    return Path(lines[-1].split(" ", 1)[1]).resolve()


def test_default_form_unchanged():
    res = _run(_PROBE)
    assert res.returncode == 0, res.stderr
    assert _path(res) == PKG.resolve()
    assert "q38 form gate" not in res.stderr


def test_oxpd_form_line_unchanged():
    res = _run(_PROBE, Q38_OXPD_FORM="1")
    assert res.returncode == 0, res.stderr
    want = (PKG / "_forms" / "oxpd" / "vllm_torchtpu").resolve()
    assert _path(res) == want
    assert "q38 form gate: Q38_OXPD_FORM=1, vllm_torchtpu runs the oxpd form from" in res.stderr


@pytest.mark.parametrize("name", NAMED)
def test_named_form_selected(name):
    res = _run(_PROBE, Q38_FORM=name)
    assert res.returncode == 0, res.stderr
    want = (PKG / "_forms" / name / "vllm_torchtpu").resolve()
    assert _path(res) == want
    assert f"q38 form gate: Q38_FORM={name}, vllm_torchtpu runs the {name} form from" in res.stderr


@pytest.mark.parametrize("name", NAMED)
def test_named_form_with_oxpd_zero_is_allowed(name):
    res = _run(_PROBE, Q38_FORM=name, Q38_OXPD_FORM="0")
    assert res.returncode == 0, res.stderr
    assert _path(res) == (PKG / "_forms" / name / "vllm_torchtpu").resolve()


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"Q38_FORM": "bogus"}, "Q38_FORM must be unset or one of"),
        ({"Q38_FORM": "oxpd"}, "Q38_FORM must be unset or one of"),
        ({"Q38_FORM": "oxk", "Q38_OXPD_FORM": "1"}, "select two forms; set only one"),
        ({"Q38_OXPD_FORM": "2"}, "Q38_OXPD_FORM must be unset, 0 or 1"),
    ],
)
def test_gate_refusals(over, message):
    res = _run(_PROBE, **over)
    assert res.returncode != 0
    assert "RuntimeError" in res.stderr and message in res.stderr, res.stderr


def _proxy_help(**over):
    return subprocess.run([sys.executable, str(PROXY), "--help"], env=_env(**over), capture_output=True, text=True,
                          timeout=300)


def test_proxy_default_unchanged():
    res = _proxy_help()
    assert res.returncode == 0, res.stderr
    assert "q38 proxy form gate" not in res.stderr


@pytest.mark.parametrize("name", PROXY_FORMS)
def test_proxy_form_runs_stored_copy(name):
    stored = PROXY.parent / "_q38_proxy_forms" / f"toy_proxy_server_{name}.py"
    res = _proxy_help(Q38_PROXY_FORM=name)
    assert res.returncode == 0, res.stderr
    assert f"q38 proxy form gate: Q38_PROXY_FORM={name} runs {stored}" in res.stderr
    direct = subprocess.run([sys.executable, str(stored), "--help"], env=_env(), capture_output=True, text=True,
                            timeout=300)
    assert direct.returncode == 0, direct.stderr
    # runpy.run_path sets sys.argv[0] to the copy it runs, so argparse names the copy in both runs.
    assert res.stdout == direct.stdout


def test_proxy_form_refuses_unknown_name():
    res = _proxy_help(Q38_PROXY_FORM="bogus")
    assert res.returncode != 0
    assert "Q38_PROXY_FORM must be unset or one of w3, k, opener" in res.stderr


def test_proxy_form_refuses_module_import():
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('toy_proxy_server', {str(PROXY)!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"
    )
    res = _run(code, Q38_PROXY_FORM="k")
    assert res.returncode != 0
    assert "ImportError" in res.stderr and "do not import it" in res.stderr, res.stderr
