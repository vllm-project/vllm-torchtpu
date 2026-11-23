"""Tests for the selector prototype, against a throwaway git repo.

    python3 -m unittest discover -s .buildkite/scripts/selector_prototype -p 'test_*.py'
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_table  # noqa: E402
import changed_funcs  # noqa: E402
import decide  # noqa: E402
import shard_files  # noqa: E402

GENERAL = "unit_tests_general_pr"
MOD = "src/vllm_torchtpu/mod.py"

V1 = textwrap.dedent(
    """\
    X = 1


    def called():
        return 1


    def not_called():
        return 2


    class Thing:
        def method(self):
            return 3
    """
)


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


class Repo:
    """A tiny repo with our layout, one commit per `commit` call."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        root.mkdir(parents=True)
        git(root, "init", "-q")
        git(root, "config", "user.email", "t@example.com")
        git(root, "config", "user.name", "t")

    def write(self, path: str, text: str) -> None:
        f = self.root / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)

    def commit(self) -> str:
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "c")
        return git(self.root, "rev-parse", "HEAD")


def fake_recording(out: pathlib.Path, step: str, shard: int, shards: int,
                   recorded: list[tuple[str, str]], *, rc: int = 0,
                   passed: int = 5, failed: int = 0) -> None:
    """Shard output shaped like record_step.sh and fnrec write it."""
    d = out / "fnrec" / step / f"shard{shard}"
    d.mkdir(parents=True)
    root = "/workspace/vllm-torchtpu/src/vllm_torchtpu"
    lines = [
        f"#start\tpid=1\tppid=0\troot={root}/",
        f"#root\t{root}/",
        "#tests\t/workspace/vllm-torchtpu/tests/",
    ]
    for path, name in recorded:
        abs_path = (
            f"{root}/{path[len('src/vllm_torchtpu/'):]}"
            if path.startswith("src/")
            else f"/workspace/vllm-torchtpu/{path}"
        )
        lines.append(f"{abs_path}\t{name}\t1")
    lines += ["/usr/lib/python3.12/os.py\tgetenv\t1", "#end\troot=1\terrors=0"]
    (d / "fn.host.abcd.1.txt").write_text("\n".join(lines) + "\n")
    (d / "pytest.1.jsonl").write_text(json.dumps(
        {"event": "session", "selected": passed + failed, "passed": passed,
         "failed": failed, "errors": 0}) + "\n")
    (d / "meta.json").write_text(json.dumps(
        {"step": step, "shard": shard, "shards": shards, "commit": "x",
         "exit_code": rc}))


class PrototypeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self.tmp.name)
        self.repo = Repo(base / "repo")
        self.repo.write(MOD, V1)
        self.repo.write("tests/test_mod.py", "def test_called():\n    assert True\n")
        self.recorded_at = self.repo.commit()
        self.artifacts = base / "artifacts"
        self.config = decide.rules.load_config(decide.REPO_ROOT)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def table(self, **kwargs) -> decide.Table:
        recorded = [(MOD, "<module>"), (MOD, "called"), (MOD, "Thing"),
                    ("tests/test_mod.py", "test_called")]
        for shard in range(3):
            fake_recording(self.artifacts, GENERAL, shard, 3, recorded, **kwargs)
        data = build_table.build(self.artifacts, self.repo.root, self.recorded_at)
        return decide.Table(json.loads(json.dumps(data)), self.repo.root)

    def change(self, old: str, new: str, path: str = MOD) -> list[changed_funcs.FileQuery]:
        text = (self.repo.root / path).read_text() if old else ""
        self.repo.write(path, text.replace(old, new) if old else new)
        head = self.repo.commit()
        return changed_funcs.build(self.repo.root, f"{head}^", head)

    def general(self, queries, table) -> decide.StepDecision:
        return decide.decide(queries, table, self.config)[GENERAL]

    # -- table ---------------------------------------------------------------

    def test_table_maps_paths_and_reads_health(self) -> None:
        row = self.table().rows[GENERAL]
        self.assertEqual(row["functions"][MOD], ["<module>", "Thing", "called"])
        self.assertIn("tests/test_mod.py", row["functions"])
        self.assertEqual(sorted(row["import_time"][MOD]), ["<module>", "Thing"])
        self.assertEqual(row["health"]["thin"], [])
        self.assertEqual(row["health"]["shards_seen"], [0, 1, 2])

    def test_failed_or_missing_shards_make_a_row_thin(self) -> None:
        self.assertIn("tests failed", self.table(failed=1).rows[GENERAL]["health"]["thin"])
        fake_recording(self.artifacts, "unit_tests_multichip_pr", 0, 2, [(MOD, "called")])
        table = build_table.build(self.artifacts, self.repo.root, self.recorded_at)
        self.assertIn("shards missing: [1]",
                      table["rows"]["unit_tests_multichip_pr"]["health"]["thin"])

    # -- changed functions ---------------------------------------------------

    def test_changed_function_names(self) -> None:
        (q,) = self.change("return 2", "return 22")
        self.assertEqual(q.names, {"not_called"})
        self.assertEqual(q.import_time, frozenset())
        (q,) = self.change("X = 1", "X = 2")
        self.assertEqual(q.import_time, {"<module>"})

    # -- decisions -----------------------------------------------------------

    def test_skips_when_the_recording_never_ran_the_changed_function(self) -> None:
        d = self.general(self.change("return 2", "return 22"), self.table())
        self.assertFalse(d.run, d.reasons)
        self.assertTrue(d.used_recording)

    def test_runs_when_the_recording_ran_the_changed_function(self) -> None:
        d = self.general(self.change("return 1", "return 11"), self.table())
        self.assertTrue(d.run)
        self.assertIn("called", d.reasons[0])

    def test_steps_without_a_recording_follow_the_rules(self) -> None:
        decisions = decide.decide(self.change("return 2", "return 22"), self.table(),
                                  self.config)
        self.assertTrue(decisions["unit_tests_multichip_pr"].run)
        self.assertIn("no recording", decisions["unit_tests_multichip_pr"].reasons[0])

    def test_import_time_change_runs_every_importer(self) -> None:
        self.assertTrue(self.general(self.change("X = 1", "X = 2"), self.table()).run)

    def test_thin_row_never_skips(self) -> None:
        d = self.general(self.change("return 2", "return 22"), self.table(failed=1))
        self.assertTrue(d.run)
        self.assertIn("too weak", d.reasons[0])

    def test_new_file_falls_back_to_the_rules(self) -> None:
        q = self.change("", "def f():\n    return 1\n", path="src/vllm_torchtpu/new.py")
        self.assertTrue(self.general(q, self.table()).run)

    def test_new_function_in_an_imported_file_runs(self) -> None:
        q = self.change("class Thing:", "def added():\n    return 4\n\n\nclass Thing:")
        self.assertTrue(self.general(q, self.table()).run)

    def test_cpu_only_file_needs_no_tpu_step(self) -> None:
        q = self.change("", "site_name: x\n", path="mkdocs.yml")
        self.assertFalse(self.general(q, self.table()).run)

    def test_summary_counts_chips(self) -> None:
        decisions = decide.decide(self.change("return 2", "return 22"), self.table(),
                                  self.config)
        summary = decide.summarize(decisions)
        self.assertEqual(summary["chips_total"], 68)
        self.assertEqual(summary["chips_skipped"], decide.CHIPS[GENERAL])


class ShardFilesTest(unittest.TestCase):
    def test_deals_every_file_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for name in ("tests/a/test_1.py", "tests/b_test.py", "tests/test_2.py",
                         "tests/helper.py", "tests/test_skip.py"):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text("")
            shards = [shard_files.shard_files(root, i, 2, {"tests/test_skip.py"})
                      for i in range(2)]
            self.assertEqual(sorted(sum(shards, [])),
                             ["tests/a/test_1.py", "tests/b_test.py", "tests/test_2.py"])


if __name__ == "__main__":
    unittest.main()
