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
"""Provenance headers, the scaffold on a small template, and listing the
copies of a changed source in a throwaway git checkout."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel_runtime import new, provenance

_LICENSE = "# Copyright 2026 Google LLC\n#\n# Licensed under the Apache License.\n"
_FILES = {
    "__init__.py": '''"""A template; its SPEC is "tpl"."""
from vllm_torchtpu.megakernel_runtime.plugin import TpuMonoSpec

SPEC = TpuMonoSpec(name="tpl", version="1", create=".adapter:create")
''',
    "adapter.py": '''"""Profile "tpl-fast" is the default."""
from .profiles import FAST, MODEL_NAME


def create(view, options):
    # "tpl-fast" in a comment stays.
    name = f"tpl-{options.get('x', 'fast')}"
    return (name, FAST, MODEL_NAME, "tpl", "tpl-slow")
''',
    "model.py": """from .profiles import MODEL_NAME


class Tpl:
    name = MODEL_NAME
    hf_architectures = ("TplForCausalLM",)
    hf_model_type = ("tpl",)
""",
    "profiles/__init__.py": """MODEL_NAME = "tpl_model"
FAST = dict(name="tpl-fast", model=MODEL_NAME)
SLOW = dict(name='tpl-slow', model="tpl_model")
""",
}


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git checkout with the template at `src/pkg/models/tpl/megakernel`."""
    root = tmp_path / "repo"
    tpl = root / "src" / "pkg" / "models" / "tpl" / "megakernel"
    for rel, body in _FILES.items():
        (tpl / rel).parent.mkdir(parents=True, exist_ok=True)
        (tpl / rel).write_text(_LICENSE + body, encoding="utf-8")
    (tpl / "README.md").write_text("# tpl\n", encoding="utf-8")
    for d in (root / "src" / "pkg", root / "src" / "pkg" / "models", tpl.parent):
        (d / "__init__.py").write_text(_LICENSE, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "tpl")
    return root


def test_header_insert_replace_and_read() -> None:
    text = _LICENSE + '"""Doc."""\n'
    out = provenance.with_header(text, "src/a/b.py", "abc123")
    assert out.splitlines()[3] == "# Copied from src/a/b.py @ abc123"
    assert provenance.read_provenance(out) == ("src/a/b.py", "abc123")
    again = provenance.with_header(out, "src/a/c.py", "def456")
    assert again.count("# Copied from") == 1
    assert provenance.read_provenance(again) == ("src/a/c.py", "def456")
    long_src = "src/" + "x" * 90 + ".py"
    line = provenance.header_line(long_src, "abc")
    assert line.endswith("# noqa: E501")
    assert provenance.read_provenance(line + "\n") == (long_src, "abc")
    # A header below the leading comment block doesn't count.
    assert provenance.read_provenance('"""Doc."""\n# Copied from a @ b\n') is None


def test_scaffold_renames_registry_names_and_writes_headers(repo: Path) -> None:
    tpl = repo / "src" / "pkg" / "models" / "tpl" / "megakernel"
    dest = repo / "src" / "pkg" / "models" / "neu" / "megakernel"
    (dest.parent).mkdir()
    (dest.parent / "__init__.py").write_text(_LICENSE, encoding="utf-8")
    head = _git(repo, "rev-parse", "HEAD")
    res = new.scaffold(tpl, dest, name="neu", model_name="neu_model")
    assert res.package == "pkg.models.neu.megakernel" and res.commit == head
    assert res.renames == {
        "tpl": "neu",
        "tpl-fast": "neu-fast",
        "tpl-slow": "neu-slow",
        "tpl_model": "neu_model",
    }
    assert (dest / "README.md").read_text(encoding="utf-8") == "# tpl\n"
    for rel in _FILES:
        text = (dest / rel).read_text(encoding="utf-8")
        ast.parse(text)
        src_rel = f"src/pkg/models/tpl/megakernel/{rel}"
        assert provenance.read_provenance(text) == (src_rel, head)
        assert text.startswith(_LICENSE)
    init = (dest / "__init__.py").read_text(encoding="utf-8")
    assert 'SPEC = TpuMonoSpec(name="neu"' in init
    assert '"""A template; its SPEC is "tpl"."""' in init  # docstrings stay
    adapter = (dest / "adapter.py").read_text(encoding="utf-8")
    assert '# "tpl-fast" in a comment stays.' in adapter
    assert "f\"tpl-{options.get('x', 'fast')}\"" in adapter  # f-strings stay
    assert '(name, FAST, MODEL_NAME, "tpl", "tpl-slow")' not in adapter
    assert '(name, FAST, MODEL_NAME, "tpl", "neu-slow")' in adapter
    profiles = (dest / "profiles" / "__init__.py").read_text(encoding="utf-8")
    assert 'MODEL_NAME = "neu_model"' in profiles
    assert 'FAST = dict(name="neu-fast", model=MODEL_NAME)' in profiles
    assert "SLOW = dict(name='neu-slow', model=\"neu_model\")" in profiles
    assert 'hf_model_type = ("tpl",)' in (dest / "model.py").read_text("utf-8")
    # The template is untouched.
    for rel, body in _FILES.items():
        assert (tpl / rel).read_text(encoding="utf-8") == _LICENSE + body

    # A change to a template file lists its copies.
    adapter_src = tpl / "adapter.py"
    assert provenance.list_copies(adapter_src) == [(dest / "adapter.py", head)]
    assert provenance.list_copies(
        "src/pkg/models/tpl/megakernel/model.py", repo_root=repo
    ) == [(dest / "model.py", head)]
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "neu")
    adapter_src.write_text(adapter_src.read_text("utf-8") + "# edit\n", "utf-8")
    changed = provenance.changed_sources_with_copies("HEAD", repo)
    assert changed == {
        "src/pkg/models/tpl/megakernel/adapter.py": [(dest / "adapter.py", head)]
    }


def test_scaffold_cli_and_errors(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tpl = repo / "src" / "pkg" / "models" / "tpl" / "megakernel"
    dest = tmp_path / "out" / "neu_pkg"
    assert new.main(["--from", str(tpl), "--dir", str(dest), "--name", "neu"]) == 0
    out = capsys.readouterr().out
    assert "tpl -> neu" in out and "tpl_model -> neu" in out
    assert f"python -m pytest {dest}/tests" in out
    assert '"plugin": "neu_pkg"' in out
    with pytest.raises(new.ScaffoldError, match="not empty"):
        new.scaffold(tpl, dest, name="neu")
    with pytest.raises(new.ScaffoldError, match="lowercase"):
        new.scaffold(tpl, tmp_path / "x", name="Neu")
    with pytest.raises(new.ScaffoldError, match="not a package or a directory"):
        new.resolve_template("argparse")
    assert (
        provenance.main(["--root", str(repo), "src/pkg/models/tpl/megakernel/x.py"])
        == 0
    )


def test_repo_relative_outside_a_checkout(tmp_path: Path) -> None:
    pkg = tmp_path / "site" / "top" / "sub"
    pkg.mkdir(parents=True)
    for d in (pkg.parent, pkg):
        (d / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "m.py").write_text("", encoding="utf-8")
    assert provenance.repo_relative(pkg / "m.py", None) in (
        "src/top/sub/m.py",
        # tmp_path inside a git checkout: relative to that checkout.
        (pkg / "m.py").relative_to(provenance.git_toplevel(pkg) or tmp_path).as_posix(),
    )
