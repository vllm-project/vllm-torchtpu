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
"""Unit tests for `.buildkite/scripts/select_pr_tests.py` and its config.

The consistency tests read the real `.buildkite/ci_paths.yaml` and pipeline
files, so a renamed step or script fails here instead of silently changing
what a PR runs.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import pathlib
import re
import sys
import tempfile
import textwrap
import unittest
from typing import Any
from unittest import mock

import yaml  # pyrefly: ignore[missing-import]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / ".buildkite" / "scripts"

with mock.patch.object(sys, "path", [str(_SCRIPTS_DIR), *sys.path]):
    sp: Any = importlib.import_module("select_pr_tests")

_INTEGRATION_SCRIPT_RE = re.compile(
    r"(?:scripts/vllm/integration|examples/disagg)/[\w.\-]+"
)


def _pipeline_steps(lane: str) -> list[dict[str, Any]]:
    """Steps of `.buildkite/pipeline_<lane>.yml`, flattened out of groups."""
    doc = yaml.safe_load(
        (_REPO_ROOT / f".buildkite/pipeline_{lane}.yml").read_text(encoding="utf-8")
    )
    out: list[dict[str, Any]] = []

    def walk(steps: list[Any]) -> None:
        for step in steps:
            if not isinstance(step, dict):
                continue
            if isinstance(step.get("steps"), list):
                walk(step["steps"])
            else:
                out.append(step)

    walk(doc["steps"])
    return out


def _runs_on_prs(step: dict[str, Any]) -> bool:
    return 'NIGHTLY") == "1"' not in str(step.get("if", ""))


def _commands(step: dict[str, Any]) -> str:
    cmds = step.get("commands", step.get("command", []))
    return " ".join([cmds] if isinstance(cmds, str) else cmds)


class _TempRepo:
    """A throwaway repo root holding a ci_paths config and some files."""

    def __init__(self, files: dict[str, str]) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        for rel, text in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(text), encoding="utf-8")

    def close(self) -> None:
        self._tmp.cleanup()


_TEMP_CONFIG = """\
    selectable_steps:
      tests:
        - unit_tests_general_pr
        - unit_tests_multichip_pr
        - unit_tests_spec_decode_pr
        - unit_tests_spec_decode_multichip_pr
        - unit_tests_omni_pr
      integration:
        - integration_a
        - integration_b
    cpu_only:
      - mkdocs.yml
      - .github/
    integration_scripts:
      scripts/vllm/integration/run_a.sh:
        - integration_a
    component_configs:
      - src/pkg/early/ci_paths.yaml
      - src/pkg/late/ci_paths.yaml
"""

_EARLY_COMPONENT = """\
    name: early
    phase: phase1
    paths:
      - src/pkg/early/
      - tests_early/
    modules:
      - pkg.early
    run_steps:
      - unit_tests_omni_pr
"""

_LATE_COMPONENT = """\
    name: late
    phase: phase2
    paths:
      - src/pkg/late/
    run_steps:
      - unit_tests_general_pr
      - integration_b
"""


class SelectPrTestsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = _TempRepo(
            {
                ".buildkite/ci_paths.yaml": _TEMP_CONFIG,
                "src/pkg/early/ci_paths.yaml": _EARLY_COMPONENT,
                "src/pkg/late/ci_paths.yaml": _LATE_COMPONENT,
                "tests/foo/test_cpu.py": "def test_x():\n    pass\n",
                "tests/foo/test_tpu.py": "@pytest.mark.tpu_test\ndef test_x(): ...\n",
                "tests/foo/test_both.py": (
                    "pytestmark = [pytest.mark.tpu_test, pytest.mark.multichip]\n"
                ),
                "tests/foo/helpers.py": "X = 1\n",
            }
        )
        self.addCleanup(self.repo.close)
        self.config = sp.load_config(self.repo.root)

    def _select(self, files: list[str], mode: str = "phase2") -> Any:
        return sp.select(files, self.config, mode=mode, repo_root=self.repo.root)

    def test_parser_matches_pyyaml_and_rejects_the_unsupported(self) -> None:
        for rel in [sp.CONFIG_PATH, *self._real_component_configs()]:
            text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertEqual(sp.parse_yaml_subset(text), yaml.safe_load(text), rel)
        for text in (_TEMP_CONFIG, _EARLY_COMPONENT, _LATE_COMPONENT):
            text = textwrap.dedent(text)
            self.assertEqual(sp.parse_yaml_subset(text), yaml.safe_load(text))
        for bad in (
            "a:\n  - b: c\n",  # list of mappings
            "a: [b, c]\n",  # flow list
            "a: {b: c}\n",  # flow mapping
            "a: &x b\n",  # anchor
            "a:\n",  # no value
            "a: b\na: c\n",  # duplicate key
            "a:\n  - b\n   - c\n",  # odd indentation
            "- a\n",  # top-level list
            "a:\n\t- b\n",  # tab
        ):
            with self.assertRaises(sp.ConfigError, msg=bad):
                sp.parse_yaml_subset(bad)

    def test_load_config_validates(self) -> None:
        cfg = self.config
        self.assertEqual(
            cfg.selectable,
            {
                "unit_tests_general_pr",
                "unit_tests_multichip_pr",
                "unit_tests_spec_decode_pr",
                "unit_tests_spec_decode_multichip_pr",
                "unit_tests_omni_pr",
                "integration_a",
                "integration_b",
            },
        )
        self.assertEqual([c.name for c in cfg.components], ["early", "late"])
        bad_cases = {
            "unknown step": (
                _TEMP_CONFIG.replace(
                    "- integration_a\n    component", "- nope\n    component"
                )
            ),
            "script step outside integration": _TEMP_CONFIG.replace(
                "run_a.sh:\n        - integration_a",
                "run_a.sh:\n        - unit_tests_general_pr",
            ),
        }
        for name, text in bad_cases.items():
            repo = _TempRepo(
                {
                    ".buildkite/ci_paths.yaml": text,
                    "src/pkg/early/ci_paths.yaml": _EARLY_COMPONENT,
                    "src/pkg/late/ci_paths.yaml": _LATE_COMPONENT,
                }
            )
            self.addCleanup(repo.close)
            with self.assertRaises(sp.ConfigError, msg=name):
                sp.load_config(repo.root)
        for name, component in {
            "bad phase": _LATE_COMPONENT.replace("phase2", "phase3"),
            "unknown run step": _LATE_COMPONENT.replace("integration_b", "nope"),
            "unknown key": _LATE_COMPONENT + "    owners:\n      - x\n",
            "no paths": _LATE_COMPONENT.replace(
                "paths:\n      - src/pkg/late/\n", "paths: []\n"
            ),
        }.items():
            repo = _TempRepo(
                {
                    ".buildkite/ci_paths.yaml": _TEMP_CONFIG,
                    "src/pkg/early/ci_paths.yaml": _EARLY_COMPONENT,
                    "src/pkg/late/ci_paths.yaml": component,
                }
            )
            self.addCleanup(repo.close)
            with self.assertRaises(sp.ConfigError, msg=name):
                sp.load_config(repo.root)

    def test_classify_each_kind_of_file(self) -> None:
        def steps(path: str) -> Any:
            return sp.classify(path, self.config, self.repo.root)[0]

        general, multichip = sp.GENERAL_STEP, sp.MULTICHIP_STEP
        self.assertEqual(steps("scripts/vllm/integration/run_a.sh"), {"integration_a"})
        self.assertEqual(steps("mkdocs.yml"), frozenset())
        self.assertEqual(steps(".github/workflows/x.yml"), frozenset())
        self.assertEqual(steps("src/pkg/early/x.py"), {"unit_tests_omni_pr"})
        self.assertEqual(steps("tests_early/test_x.py"), {"unit_tests_omni_pr"})
        self.assertEqual(steps("tests/foo/test_cpu.py"), frozenset())
        self.assertEqual(steps("tests/foo/test_tpu.py"), {general})
        self.assertEqual(steps("tests/foo/test_both.py"), {general, multichip})
        self.assertEqual(steps("tests/foo/test_deleted.py"), frozenset())
        self.assertEqual(steps(sp.SPEC_DECODE_E2E), sp.SPEC_DECODE_STEPS)
        # Anything else needs the full build: shared source, test helpers,
        # unmapped integration helpers, and the selector's own files.
        for path in (
            "src/vllm_torchtpu/envs.py",
            "tests/foo/helpers.py",
            "tests/conftest.py",
            "examples/disagg/launch.sh",
            ".buildkite/ci_paths.yaml",
            ".buildkite/scripts/select_pr_tests.py",
            "mkdocs.yml.bak",
        ):
            self.assertIsNone(steps(path), path)

    def test_select_takes_the_union_and_any_full_build_file_wins(self) -> None:
        sel = self._select(
            ["scripts/vllm/integration/run_a.sh", "tests/foo/test_tpu.py"]
        )
        self.assertEqual(sel.needed, {"integration_a", sp.GENERAL_STEP})
        self.assertEqual(
            set(sel.skipped),
            self.config.selectable - {"integration_a", sp.GENERAL_STEP},
        )
        self.assertEqual(sel.full_build_file, "")

        sel = self._select(["tests/foo/test_tpu.py", "src/vllm_torchtpu/envs.py"])
        self.assertIsNone(sel.needed)
        self.assertEqual(sel.skipped, ())
        self.assertEqual(sel.would_skip, ())
        self.assertEqual(sel.full_build_file, "src/vllm_torchtpu/envs.py")

        # CPU-only changes need no TPU step at all.
        sel = self._select(["mkdocs.yml", ".github/CODEOWNERS"])
        self.assertEqual(set(sel.skipped), self.config.selectable)

    def test_modes(self) -> None:
        files = ["src/pkg/late/x.py"]
        # shadow: measures what phase2 would skip, skips nothing.
        shadow = self._select(files, "shadow")
        self.assertEqual(shadow.skipped, ())
        self.assertEqual(shadow.skip_reasons(), {})
        self.assertEqual(shadow.omni_image_skip, "")
        self.assertEqual(
            set(shadow.would_skip),
            self.config.selectable - {"unit_tests_general_pr", "integration_b"},
        )
        # phase1: a phase2 folder is not active yet, so it runs everything,
        # while the record still says what phase2 would skip.
        phase1 = self._select(files, "phase1")
        self.assertEqual(phase1.skipped, ())
        self.assertEqual(phase1.would_skip, shadow.would_skip)
        self.assertEqual(
            self._select(["src/pkg/early/x.py"], "phase1").needed,
            {"unit_tests_omni_pr"},
        )
        self.assertNotEqual(self._select(["src/pkg/early/x.py"], "phase1").skipped, ())
        # phase2: every rule is active.
        phase2 = self._select(files, "phase2")
        self.assertEqual(phase2.skipped, shadow.would_skip)
        self.assertEqual(phase2.omni_image_skip, sp.OMNI_IMAGE_SKIP)
        self.assertEqual(self._select(["src/pkg/early/x.py"]).omni_image_skip, "")

        self.assertEqual(sp.parse_mode(None), "shadow")
        self.assertEqual(sp.parse_mode(""), "shadow")
        self.assertEqual(sp.parse_mode(" Phase1 "), "phase1")
        self.assertEqual(sp.parse_mode("on"), "shadow")
        self.assertEqual(sp.parse_mode("off"), "off")

    def test_record_summary_and_reasons(self) -> None:
        self.assertLessEqual(len(sp.SKIP_REASON), 70)
        self.assertLessEqual(len(sp.OMNI_IMAGE_SKIP), 70)
        sel = self._select(["src/pkg/late/x.py", "mkdocs.yml"], "shadow")
        record = sel.record()
        self.assertEqual(record["mode"], "shadow")
        self.assertEqual(record["files"], 2)
        self.assertEqual(record["rules"], ["CPU-only file", "late folder"])
        self.assertEqual(record["skipped"], [])
        self.assertEqual(record["would_skip"], list(sel.would_skip))
        self.assertIn("shadow mode, nothing skipped", sel.summary())
        self.assertIn("would skip 5 of 7", sel.summary())
        enforced = self._select(["src/pkg/late/x.py"], "phase2")
        self.assertIn("skipped 5 of 7", enforced.summary())
        self.assertTrue(all(len(r) <= 70 for r in enforced.skip_reasons().values()))
        full = self._select(["src/x.py"], "phase2")
        self.assertIn("`src/x.py` needs the full build", full.summary())

    def test_mark_skipped(self) -> None:
        steps: list[Any] = [
            "wait",
            {"key": "a", "command": "x"},
            {"group": "g", "steps": [{"key": "b"}, {"key": "c", "skip": "kept"}]},
            {"key": "d"},
        ]
        marked = sp.mark_skipped(steps, {"a": "ra", "b": "rb", "c": "rc"})
        self.assertEqual(marked, 2)
        self.assertEqual(steps[1]["skip"], "ra")
        self.assertEqual(steps[2]["steps"][0]["skip"], "rb")
        self.assertEqual(steps[2]["steps"][1]["skip"], "kept")
        self.assertNotIn("skip", steps[3])

    def test_cli(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(sp.main(["mkdocs.yml", "src/vllm_torchtpu/envs.py"]), 0)
        self.assertIn(
            "mkdocs.yml: CPU-only file (phase1): no TPU steps", out.getvalue()
        )
        self.assertIn("src/vllm_torchtpu/envs.py: full build", out.getvalue())

    # -- Consistency with the real config and pipelines ----------------------

    @staticmethod
    def _real_component_configs() -> list[str]:
        raw = yaml.safe_load((_REPO_ROOT / sp.CONFIG_PATH).read_text(encoding="utf-8"))
        return list(raw.get("component_configs") or [])

    def test_real_config_loads(self) -> None:
        config = sp.load_config(_REPO_ROOT)
        self.assertEqual(
            len(config.selectable),
            len(config.test_steps) + len(config.integration_steps),
        )

    def test_selectable_steps_are_pr_steps_of_their_lane(self) -> None:
        config = sp.load_config(_REPO_ROOT)
        for lane, keys in (
            ("tests", config.test_steps),
            ("integration", config.integration_steps),
        ):
            pr_keys = {s.get("key") for s in _pipeline_steps(lane) if _runs_on_prs(s)}
            for key in keys:
                self.assertIn(key, pr_keys, f"{key} is not a PR step of {lane}")
        # The built-in unit-test rule names steps that must be selectable.
        for key in (sp.GENERAL_STEP, sp.MULTICHIP_STEP, *sp.SPEC_DECODE_STEPS):
            self.assertIn(key, config.test_steps)
        self.assertIn(sp.OMNI_TEST_STEP, config.test_steps)

    def test_no_kept_step_depends_on_a_selectable_step(self) -> None:
        selectable = sp.load_config(_REPO_ROOT).selectable
        for lane in ("build", "cpu", "tests", "integration", "verify"):
            for step in _pipeline_steps(lane):
                deps = step.get("depends_on") or []
                deps = [deps] if isinstance(deps, str) else deps
                names = {d if isinstance(d, str) else d.get("step") for d in deps}
                if step.get("key") not in selectable:
                    self.assertFalse(names & selectable, step.get("key"))

    def test_only_the_omni_test_step_uses_the_omni_image(self) -> None:
        users = {
            s.get("key")
            for lane in ("tests", "integration", "verify")
            for s in _pipeline_steps(lane)
            if _runs_on_prs(s)
            and (
                s.get("depends_on") == "build_docker_omni"
                or "-omni" in str(s.get("env", {}).get("DOCKER_IMAGE_SUFFIX", ""))
            )
        }
        self.assertEqual(users, {sp.OMNI_TEST_STEP})

    def test_integration_scripts_map_to_exactly_their_steps(self) -> None:
        config = sp.load_config(_REPO_ROOT)
        refs: dict[str, set[str]] = {}
        for lane in ("tests", "integration", "verify"):
            for step in _pipeline_steps(lane):
                if not _runs_on_prs(step):
                    continue
                for script in _INTEGRATION_SCRIPT_RE.findall(_commands(step)):
                    refs.setdefault(script, set()).add(step.get("key"))
        for script, steps in config.integration_scripts.items():
            self.assertTrue((_REPO_ROOT / script).is_file(), script)
            self.assertEqual(refs.get(script), set(steps), script)

    def test_cpu_only_paths_are_not_read_by_pr_tpu_steps(self) -> None:
        config = sp.load_config(_REPO_ROOT)
        lane_text = "".join(
            (_REPO_ROOT / f".buildkite/pipeline_{lane}.yml").read_text(encoding="utf-8")
            for lane in ("build", "tests", "integration", "verify")
        )
        for entry in config.cpu_only:
            self.assertTrue((_REPO_ROOT / entry).exists(), entry)
            self.assertNotIn(entry, lane_text, entry)

    def test_unit_test_rule_matches_the_pipeline_markers(self) -> None:
        cmds = {
            s.get("key"): str(s.get("plugins"))
            for s in _pipeline_steps("tests")
            if _runs_on_prs(s)
        }
        self.assertIn(
            "-m 'tpu_test and not nightly and not multichip'", cmds[sp.GENERAL_STEP]
        )
        self.assertIn("-m 'not nightly and multichip'", cmds[sp.MULTICHIP_STEP])
        for key in sp.SPEC_DECODE_STEPS:
            self.assertIn(f"'test-file-pattern': '{sp.SPEC_DECODE_E2E}'", cmds[key])
        # A test file with neither marker runs only in the CPU step, which
        # always runs, so the rule gives it no TPU step.
        cpu_cmd = next(
            str(s.get("plugins"))
            for s in _pipeline_steps("cpu")
            if s.get("key") == "unit_tests_cpu_pr"
        )
        self.assertIn("-m 'not tpu_test and not multichip and not nightly'", cpu_cmd)
        # The rule reads markers from the changed file itself, so none may be
        # applied from a conftest.
        for conftest in (_REPO_ROOT / "tests").rglob("conftest.py"):
            text = conftest.read_text(encoding="utf-8")
            self.assertIsNone(
                re.search(r"\b(tpu_test|multichip)\b", text), str(conftest)
            )


if __name__ == "__main__":
    unittest.main()
