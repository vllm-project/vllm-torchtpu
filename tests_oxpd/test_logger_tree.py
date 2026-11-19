# SPDX-License-Identifier: Apache-2.0
"""Module loggers must sit under vLLM's "vllm" logger, or their INFO lines vanish.

vLLM's logging config puts the only handler on the "vllm" logger and sets
propagate=False there. A logger named "vllm_torchtpu.x" (vllm.logger's
init_logger(__name__), or logging.getLogger(__name__)) is outside that tree:
in the served engine its records fall to the root logger at WARNING, so INFO
and DEBUG lines are dropped while WARNING lines still print in root format
("WARNING:vllm_torchtpu.core.eagle_block_drop:..."). pytest's caplog listens at
the root, so tests that use it pass either way; these tests listen on "vllm".

vllm_torchtpu.logger.init_logger prepends "vllm." and is the fix.
"""

import ast
import importlib
import logging
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "vllm_torchtpu"

# Modules fixed by this change; each emits an engagement INFO line in serving.
# omni.platform is fixed too, but importing it needs vllm_omni; the static
# test covers it.
FIXED = (
    "vllm_torchtpu.core.eagle_block_drop",
    "vllm_torchtpu.core.mamba_checkpoint_retention",
    "vllm_torchtpu.core.divergent_hybrid_hits",
    "vllm_torchtpu.core.tpu_scheduler",
    "vllm_torchtpu.distributed.pp_push",
)

_QUIET = {"info", "info_once", "debug", "debug_once"}


def _outside_tree(call, vllm_init, logging_names):
    """True if `call` builds a logger whose name is not "vllm" or "vllm.*"."""
    f = call.func
    builds = (isinstance(f, ast.Name) and f.id in vllm_init) or (
        isinstance(f, ast.Attribute)
        and f.attr == "getLogger"
        and isinstance(f.value, ast.Name)
        and f.value.id in logging_names
    )
    if not builds or not call.args:
        return False
    arg = call.args[0]
    if isinstance(arg, ast.Name) and arg.id == "__name__":
        return True
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return not (arg.value == "vllm" or arg.value.startswith("vllm."))
    return False


def _offenders(path):
    """(logger, line) pairs: loggers outside the vllm tree that log INFO/DEBUG.

    A logger bound in an `except ImportError` fallback is exempt (the module
    was loaded by path, without the package). WARNING-only loggers are
    exempt: their lines reach the root handler.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    vllm_init, logging_names = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "vllm.logger":
            vllm_init |= {a.asname or a.name for a in n.names if a.name == "init_logger"}
        if isinstance(n, ast.Import):
            logging_names |= {a.asname or a.name for a in n.names if a.name == "logging"}
    fallback = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler):
            for m in ast.walk(n):
                fallback.add(id(m))
    bad = {}
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and id(n) not in fallback
            and _outside_tree(n.value, vllm_init, logging_names)
        ):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    bad[t.id] = n.lineno
    out = []
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in _QUIET
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id in bad
        ):
            out.append((n.func.value.id, n.lineno))
    return out


def test_no_info_logger_outside_vllm_tree():
    found = {}
    for path in sorted(SRC.rglob("*.py")):
        hits = _offenders(path)
        if hits:
            found[str(path.relative_to(SRC.parent))] = hits
    assert not found, (
        "these modules log INFO/DEBUG through a logger outside the 'vllm' tree; "
        "serving drops those lines. Use vllm_torchtpu.logger.init_logger: "
        f"{found}"
    )


@pytest.fixture
def vllm_handler():
    """Listen on the 'vllm' logger as the served engine does (level INFO)."""
    pytest.importorskip("vllm.logger")  # applies vLLM's logging config
    records = []
    handler = logging.Handler(logging.INFO)
    handler.emit = records.append
    root = logging.getLogger("vllm")
    old_level, old_disable = root.level, logging.root.manager.disable
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    try:
        yield records
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        logging.disable(old_disable)


@pytest.mark.parametrize("name", FIXED)
def test_info_reaches_vllm_handler(name, vllm_handler):
    module = importlib.import_module(name)
    module.logger.info("q38 logaudit probe from %s", name)
    probes = [r for r in vllm_handler if r.getMessage().startswith("q38 logaudit probe")]
    assert len(probes) == 1, f"{name}: INFO did not reach the 'vllm' handler"
    assert probes[0].name == "vllm." + name
