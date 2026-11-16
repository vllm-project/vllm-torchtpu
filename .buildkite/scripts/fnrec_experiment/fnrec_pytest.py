"""Record what each pytest session ran, beside the function record.

The table builder needs to know a job actually ran tests: a run that collected
them and executed none recorded only its imports, and dropping a step on a row
like that would be wrong. Reading this out of the Buildkite job log with a
regex broke whenever the log format changed.

Two lines per session. The first is written at the end of collection, before
any test runs, so a killed session still leaves a count with no outcome, which
tells the builder the row is incomplete. Leaving nothing at all would instead
look like a step that ran no pytest.

One file per process, so parallel sessions cannot interleave lines. Loaded
through PYTEST_PLUGINS. Nothing here may fail a test run.
"""

import json
import os
import time

# Keys the table builder reads. pytest spells errors `error` in its stats, so
# that one is read separately below.
_OUTCOMES = ("passed", "failed", "skipped", "deselected", "xfailed", "xpassed")


def _path():
    out = os.environ.get("FNREC_OUT")
    return os.path.join(out, f"pytest.{os.getpid()}.jsonl") if out else None


def _append(row):
    path = _path()
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        # Root writes it under docker; the agent user has to be able to clean it.
        os.chmod(path, 0o666)
    except Exception:
        pass


def pytest_collection_finish(session):
    """Before a single test runs, so a killed session still leaves a count."""
    if _is_worker(session.config):
        return
    try:
        # testscollected is not set until collection returns, after this hook.
        selected = len(getattr(session, "items", ()) or ())
        stats = _stats(session.config)
        _append(
            {
                "event": "collected",
                "selected": selected,
                "deselected": len(stats.get("deselected") or ()),
                "pid": os.getpid(),
                "t": round(time.time(), 3),
            }
        )
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
    if _is_worker(session.config):
        return
    try:
        stats = _stats(session.config)
        row = {name: len(stats.get(name) or ()) for name in _OUTCOMES}
        row["errors"] = len(stats.get("error") or ())
        row["event"] = "session"
        row["selected"] = getattr(session, "testscollected", 0) or 0
        # Survives a runner that turns a failing exit status into a passing one.
        row["testsfailed"] = getattr(session, "testsfailed", 0) or 0
        row["exitstatus"] = int(exitstatus)
        row["pid"] = os.getpid()
        row["t"] = round(time.time(), 3)
        _append(row)
    except Exception:
        pass


def _is_worker(config):
    """True for an xdist worker, whose session would double every count."""
    return hasattr(config, "workerinput")


def _stats(config):
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    return getattr(reporter, "stats", None) or {}
