# SPDX-License-Identifier: Apache-2.0
"""Probe: does libtpu's --xla_tpu_msa_tensor_overrides keep the PLE history out of VMEM?

Compiles `pooled_short_conv_speculative` (XLA gathers, flag off) at the served
320-row geometry in one child process per LIBTPU_INIT_ARGS variant and prints,
per variant, the memory space of every XLA reader's [R, 9, 10240] operand and
the VMEM values of 16 MiB or more. A variant whose flag libtpu rejects prints
the child's error tail. One PLEMSA line per variant.
"""

import gzip
import json
import os
import re
import subprocess
import sys

HIST = "=[[:space:]]bf16[[][0-9]+,9,10240[]]"
VARIANTS = {
    "base": "",
    "size_exact": (
        "--xla_tpu_msa_tensor_overrides=overrides{hlo_position_matcher"
        "{size_gte:23592960,size_lte:23592960},keep_in_default_memory{}}"
    ),
    "regex": (
        "--xla_tpu_msa_tensor_overrides=overrides{hlo_position_matcher"
        '{instruction_regex:"' + HIST + '"},keep_in_default_memory{}}'
    ),
    "regex_full": (
        "--xla_tpu_msa_tensor_overrides=overrides{hlo_position_matcher"
        '{instruction_regex:".*' + HIST + '.*"},keep_in_default_memory{}}'
    ),
    "cap16m": (
        "--xla_tpu_msa_tensor_overrides=overrides{hlo_position_matcher"
        "{size_gte:16777216},keep_in_default_memory{}}"
    ),
}

BYTES = {"pred": 1, "s8": 1, "u8": 1, "f8e4m3fn": 1, "bf16": 2, "f16": 2, "s16": 2,
         "u16": 2, "f32": 4, "s32": 4, "u32": 4, "s64": 8, "u64": 8}


def _entry(hlo):
    lines = hlo.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("ENTRY "))
    out = {}
    for l in lines[start + 1:]:
        if l == "}":
            break
        m = re.match(r"^\s+(?:ROOT\s+)?([\w.\-]+) = (\S+)\s+([\w\-]+)\((.*)$", l)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3), m.group(4))
    return out


def summarize(hlo):
    # Compiled.as_text() prefixes names with "%"; the captures' text does not.
    defs = _entry(hlo.replace("%", ""))
    hist = {n for n, (s, _, _) in defs.items() if re.match(r"bf16\[\d+,9,10240\]", s)}
    readers = []
    for name, (shape, op, rest) in defs.items():
        if op in ("bitcast", "copy", "copy-start", "copy-done", "get-tuple-element", "tuple"):
            continue
        for o in re.findall(r"[\w.\-]+", rest.split("), ")[0]):
            if o in hist:
                readers.append(
                    (name, shape.split("{")[0], o, "S(1)" if "S(1)" in defs[o][0] else "HBM")
                )
    big = []
    for name, (shape, op, _) in defs.items():
        m = re.match(r"(\w+)\[([\d,]*)\](\{[^}]*\})", shape)
        if not m or "S(1)" not in m.group(3):
            continue
        n = 1
        for d in filter(None, m.group(2).split(",")):
            n *= int(d)
        mib = n * BYTES.get(m.group(1), 4) / 2**20
        if mib >= 16:
            big.append((name, op, shape.split("{")[0], round(mib, 2)))
    return readers, big


def _pytest_stub():
    """The byte test module decorates with pytest marks; the image has no pytest."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        import types

        class _Mark:
            def __getattr__(self, name):
                return lambda *a, **k: (lambda f: f)

        sys.modules["pytest"] = types.SimpleNamespace(mark=_Mark())


def child():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # jax first: its etils import probes for pytest and must see the real state.
    import jax  # noqa: F401

    _pytest_stub()
    import test_qwen4_exp_ple_history_rows_tpu as T

    args, _ = T._inputs("decode80_t320", seed=5)
    hlo = T._step(False).lower(*args).compile().as_text()
    out = os.environ.get("CDK_OUTPUT_DIR")
    if out:
        os.makedirs(os.path.join(out, "msaprobe"), exist_ok=True)
        path = os.path.join(out, "msaprobe", sys.argv[-1] + ".hlo.txt.gz")
        with gzip.open(path, "wt") as f:
            f.write(hlo)
    readers, big = summarize(hlo)
    print("PLEMSA_CHILD " + json.dumps(dict(readers=readers, big=big)), flush=True)


def verdict(rows):
    """(exit code, lines) for the variants' rows; nonzero when the base row cannot judge the flag."""
    base = next((r for r in rows if r["variant"] == "base"), None)
    if base is None or base.get("rc") != 0 or "readers" not in base:
        return 2, ["PLEMSA_FAIL base variant did not compile; nothing is judged"]
    if not base["readers"]:
        return 3, ["PLEMSA_FAIL base variant found no history reader in the compiled HLO (parser or program)"]
    if not any(r[3] == "S(1)" for r in base["readers"]) or not base["big"]:
        return 4, ["PLEMSA_FAIL base variant keeps the history out of VMEM, so this program cannot show the flag's effect"]
    lines = []
    for r in rows:
        if r["variant"] == "base" or "readers" not in r:
            continue
        vmem = sorted({x[0] for x in r["readers"] if x[3] == "S(1)"})
        lines.append(
            f"PLEMSA_VERDICT {r['variant']}: history readers in VMEM {len(vmem)} of "
            f"{len(r['readers'])}; VMEM values >= 16 MiB {len(r['big'])} (base {len(base['big'])})"
        )
    return 0, lines


def main():
    base = os.environ.get("LIBTPU_INIT_ARGS", "")
    rows = []
    for name, extra in VARIANTS.items():
        env = dict(os.environ, LIBTPU_INIT_ARGS=(base + " " + extra).strip())
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", name],
            env=env, capture_output=True, text=True, timeout=1800,
        )
        line = next((l for l in r.stdout.splitlines() if l.startswith("PLEMSA_CHILD ")), None)
        row = dict(variant=name, flag=extra, rc=r.returncode)
        if line:
            row.update(json.loads(line[len("PLEMSA_CHILD "):]))
        else:
            row["error_tail"] = (r.stderr or r.stdout)[-3000:]
        print("PLEMSA " + json.dumps(row), flush=True)
        rows.append(row)
    code, lines = verdict(rows)
    for line in lines:
        print(line, flush=True)
    sys.exit(code)


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
