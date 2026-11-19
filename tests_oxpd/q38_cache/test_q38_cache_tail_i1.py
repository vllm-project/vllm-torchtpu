# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the q38-cache-tail-i1 harness: the verdict logic and the mutant's import hook."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "q38_cache" / "q38_cache_tail_i1.py"
SITE = ROOT / "q38_cache" / "mutant_site"


def _load(out_dir, monkeypatch):
    monkeypatch.setenv("Q38_TAIL_OUT", str(out_dir))
    spec = importlib.util.spec_from_file_location("_tail_i1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arm(mod, name, cached2, tok_shift=0, tok_shift_turn=None):
    def rec(i, cached, turn):
        shift = tok_shift if tok_shift_turn in (None, turn) else 0
        return {"tokens": [i + shift] * mod.GEN, "cached": cached, "prompt": mod.LENS[i] + (mod.EXT if turn > 1 else 0),
                "ttft_ms": 100.0 + i}
    return {"arm": name,
            "turn1": [rec(i, 0, 1) for i in range(len(mod.LENS))],
            "turn2": [rec(i, cached2[i], 2) for i in range(len(mod.LENS))],
            "turn3": [rec(i, cached2[i], 3) for i in range(len(mod.LENS))], "turn3_wall_s": 1.0}


def _write(tmp_path, arms):
    for a, d in arms.items():
        (tmp_path / f"arm_{a}.json").write_text(json.dumps(d))


def test_verdict_passes_on_identity_engagement_and_biting_mutant(tmp_path, monkeypatch):
    mod = _load(tmp_path, monkeypatch)
    eB = [(L - 1) // mod.BLOCK * mod.BLOCK for L in mod.LENS]
    eT = [L // mod.UNIT * mod.UNIT for L in mod.LENS]
    _write(tmp_path, {"B": _arm(mod, "B", eB), "T": _arm(mod, "T", eT),
                      "X": _arm(mod, "X", eT, tok_shift=7, tok_shift_turn=2)})
    assert mod.verdict() == 0
    v = json.loads((tmp_path / "verdict.json").read_text())
    assert v["identity"] and v["engaged"] and v["mutant_bites"] and v["baseline_as_expected"]
    assert v["computed_T"] < v["computed_B"]


def test_verdict_fails_on_any_token_difference(tmp_path, monkeypatch):
    mod = _load(tmp_path, monkeypatch)
    eB = [(L - 1) // mod.BLOCK * mod.BLOCK for L in mod.LENS]
    eT = [L // mod.UNIT * mod.UNIT for L in mod.LENS]
    for turn in (1, 2, 3):
        _write(tmp_path, {"B": _arm(mod, "B", eB), "T": _arm(mod, "T", eT, tok_shift=1, tok_shift_turn=turn),
                          "X": _arm(mod, "X", eT, tok_shift=7, tok_shift_turn=2)})
        assert mod.verdict() == 1


def test_verdict_fails_when_the_tail_did_not_engage(tmp_path, monkeypatch):
    mod = _load(tmp_path, monkeypatch)
    eB = [(L - 1) // mod.BLOCK * mod.BLOCK for L in mod.LENS]
    _write(tmp_path, {"B": _arm(mod, "B", eB), "T": _arm(mod, "T", eB),
                      "X": _arm(mod, "X", eB, tok_shift=7, tok_shift_turn=2)})
    assert mod.verdict() == 1
    assert json.loads((tmp_path / "verdict.json").read_text())["engaged"] is False


def test_verdict_fails_when_the_mutant_does_not_bite(tmp_path, monkeypatch):
    mod = _load(tmp_path, monkeypatch)
    eB = [(L - 1) // mod.BLOCK * mod.BLOCK for L in mod.LENS]
    eT = [L // mod.UNIT * mod.UNIT for L in mod.LENS]
    _write(tmp_path, {"B": _arm(mod, "B", eB), "T": _arm(mod, "T", eT), "X": _arm(mod, "X", eT)})
    assert mod.verdict() == 1


def test_cold_arm_drops_caching_and_align_only(tmp_path, monkeypatch):
    mod = _load(tmp_path, monkeypatch)
    import inspect
    src = inspect.getsource(mod.child)
    assert 'serve.remove("--enable-prefix-caching")' in src and "del serve[i:i + 2]" in src
    assert mod.SERVED[mod.SERVED.index("--mamba-cache-mode") + 1] == "align"
    assert mod.ARMS["T"][0] == ["--prefix-match-unit=128"] and mod.ARMS["X"][1] == {"Q38_TAIL_MUTANT": "drop_cow"}


def test_lens_avoid_block_and_unit_boundaries(tmp_path, monkeypatch):
    mod = _load(tmp_path, monkeypatch)
    assert all(L % mod.UNIT and L % mod.BLOCK for L in mod.LENS)
    assert all(L + 2 * mod.EXT < 262144 for L in mod.LENS)


def _fake_runner_pkg(tmp_path):
    pkg = tmp_path / "pkg" / "vllm_torchtpu" / "runner"
    pkg.mkdir(parents=True)
    (tmp_path / "pkg" / "vllm_torchtpu" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "tpu_runner.py").write_text(textwrap.dedent("""
        CALLS = []
        class TPUModelRunner:
            def _apply_kv_cache_block_copies(self, copies):
                CALLS.append(copies)
    """))
    return tmp_path / "pkg"


def _run(pkg, mutant):
    env = {k: v for k, v in os.environ.items() if k != "Q38_TAIL_MUTANT"}
    env["PYTHONPATH"] = os.pathsep.join([str(SITE), str(pkg)])
    if mutant:
        env["Q38_TAIL_MUTANT"] = "drop_cow"
    code = ("from vllm_torchtpu.runner import tpu_runner as m; m.TPUModelRunner()._apply_kv_cache_block_copies([1]);"
            "print(len(m.CALLS))")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout


def test_mutant_hook_drops_copies_only_when_asked(tmp_path):
    pkg = _fake_runner_pkg(tmp_path)
    assert _run(pkg, mutant=False).strip().splitlines()[-1] == "1"
    out = _run(pkg, mutant=True).strip().splitlines()
    assert out[-1] == "0" and any("q38 mutant" in line for line in out)
