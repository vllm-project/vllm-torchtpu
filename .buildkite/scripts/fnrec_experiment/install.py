# EXPERIMENT COPY. From vllm-project/ci-infra (Apache-2.0),
# buildkite/ci_selector/recorders/fnrec/install.py at ec20a876f0b9. Change:
# FNREC_ROOT is printed for FNREC_PACKAGE (default `vllm`) instead of vllm.
import importlib.util
import os
import pathlib
import site

PACKAGE = os.environ.get("FNREC_PACKAGE") or "vllm"

# ci_setup.sh fetches these next to this script.
HERE = pathlib.Path(__file__).parent
FNREC_SOURCE = (HERE / "fnrec.py").read_text()
# Optional: a missing plugin costs the pytest record, not the recorder.
PYTEST_FILE = HERE / "fnrec_pytest.py"
PYTEST_SOURCE = PYTEST_FILE.read_text() if PYTEST_FILE.is_file() else None


def site_dir():
    """Where a .pth will actually be executed.

    Under an editable install vllm's parent is the repo root, and a .pth there
    is never read, so only accept it if it is a site directory.
    """
    dirs = [p for p in site.getsitepackages() if os.path.isdir(p)]
    spec = importlib.util.find_spec("vllm")
    if spec is not None and spec.origin:
        parent = str(pathlib.Path(spec.origin).parent.parent)
        if parent in dirs:
            return parent
    return dirs[0]


target = pathlib.Path(site_dir())
(target / "fnrec.py").write_text(FNREC_SOURCE)
if PYTEST_SOURCE is not None:
    (target / "fnrec_pytest.py").write_text(PYTEST_SOURCE)
    # Marks the plugin as installed, so a job with no session file is not
    # confused with one whose plugin never installed.
    out = os.environ.get("FNREC_OUT")
    if out:
        # Best effort: this file is optional everywhere it is read, and the
        # install must not fail for it. Without FNREC_ROOT nothing records.
        try:
            marker = pathlib.Path(out) / "pytest.installed"
            marker.write_text("")
            os.chmod(marker, 0o666)
        except OSError:
            pass
# Loads fnrec at every interpreter start, including subprocesses.
(target / "fnrec.pth").write_text("import fnrec\n")

# FNREC_ROOT is where the recorded package's code is, not where we installed to.
spec = importlib.util.find_spec(PACKAGE)
print(
    pathlib.Path(spec.origin).parent
    if spec is not None and spec.origin
    else target / PACKAGE
)
