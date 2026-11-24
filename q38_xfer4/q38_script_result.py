"""q38_script_result.py — the result.json a BENCH=script command must leave so the bench recipe scores it.

The infx-qwen38-flash-bench recipe scores every job by reading ``$RUN_DIR/run01_<ARM>/result.json`` (``ARM`` defaults
to ``A``; ``RUN_DIR = ${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38f-bench/script`` for BENCH=script) and needs ``ok`` and a
non-empty ``cases`` list; without it a script that completed still scores Failed (b1k, G2K). ``clear()`` at start, so
a later process in the same BENCH_CMD that dies cannot inherit an earlier one's result; ``write()`` at the end.
"""
import json
import os


def path(arm: str = "A") -> str:
    root = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"), "q38f-bench", "script")
    return os.path.join(root, "run01_%s" % os.environ.get("ARMS", arm).split(",")[0], "result.json")


def clear() -> None:
    try:
        os.remove(path())
    except FileNotFoundError:
        pass


def write(ok: bool, summary: dict) -> str:
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"ok": bool(ok), "cases": [summary]}, f)
    print("Q38_SCRIPT_RESULT " + json.dumps({"path": p, "ok": bool(ok)}), flush=True)
    return p
