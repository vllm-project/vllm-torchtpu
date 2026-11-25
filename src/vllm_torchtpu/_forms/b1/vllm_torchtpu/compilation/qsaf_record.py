# SPDX-License-Identifier: Apache-2.0
"""q38 QSAF-INT: the persisted trace record of the QSAF fused-union decode route.

``VLLM_TPU_QSA_DECODE_FUSED_UNION`` (kernels/qwen4_exp/qsa_fused_union_keys.py) routes
``qsa_rpa.union_decode_attention`` to the fused kernel while the decode program is
TRACED, so a trace-time line exists only on a cold compile: a warm start (compile-cache
hit) traces nothing and prints nothing. This module is the IDXXLINE pattern
(compilation/idxx_record.py, design I-1a, credited) for that one route:

1. RECORD. The route calls ``note`` with its trace key (rows, request slots, covered
   requests, issue mode) into one CUMULATIVE set per process.
2. PERSIST. When ``TpuCompilerAdaptor`` saves an artifact, it writes that set next to
   the artifact, ``<cache_dir>/<artifact key>.q38qsaf.json``, inside vLLM's
   compile-cache dir ``<root>/<hash>/rank_<r>_<dp>/<prefix>`` (nothing anywhere else).
3. FORMAT. The record carries the cache hash (the ``<hash>`` dir name), the artifact it
   sits next to, the QSAF key values, the entries and a sha256 over all of that. Read
   back, it must match the dir it sits in, the artifact, this process's key values and
   its checksum, and every entry must be well formed with the issue mode the keys give;
   anything else is a MISMATCH.
4. Every artifact this process uses (compiled and saved, loaded from the cache, or
   replayed from a peer rank) registers its verified record; the runner prints the
   per-rank warm-safe 'engaged' lines from the union (runner/qsaf_engagement.py).

Honest label: a record says what was TRACED while this cache hash was built (cumulative
per process, as IDXXLINE). That the loaded executable runs the fused kernel rests on the
compile key (which byte-hashes kernels/ and carries the QSAF keys when on) -- inferred,
not observed on device.

Everything here is a no-op, reads and writes nothing, when the key is off.
Python only: nothing here enters a program. No file is ever removed.
"""

import hashlib
import json
import os
import threading
from typing import Any

from vllm_torchtpu.compilation.idxx_record import (
    _canon,
    _inside,
    cache_identity,
    process_rank_fields,
)

ENV = "VLLM_TPU_QSA_DECODE_FUSED_UNION"
ISSUE_ENV = "VLLM_TPU_QSA_DECODE_FUSED_UNION_ISSUE"
FORMAT = "q38-qsaf-record/1"
SUFFIX = ".q38qsaf.json"
FIELDS = ("rows", "slots", "covered", "issue", "path")

__all__ = ["process_rank_fields", "cache_identity", "note", "snapshot", "verify",
           "write", "on_saved", "on_loaded", "on_peer", "used", "active", "key_values"]

_LOCK = threading.Lock()
# canonical entry -> entry (one per distinct trace key)
_ENTRIES: dict[str, dict] = {}
# artifact path -> what this process used it as, and its record's verdict
_USED: dict[str, dict] = {}


def active() -> bool:
    """The key is on (unset, "" and "0" are off; any other value is on here and is
    validated by ``key_values``, as the compile factors validate it)."""
    return os.environ.get(ENV, "") not in ("", "0")


def key_values() -> dict[str, str]:
    """The QSAF key values exactly as tpu_compiler._tpu_compile_env_factors adds them."""
    from vllm_torchtpu.kernels.qwen4_exp import qsa_fused_union_keys

    return dict(qsa_fused_union_keys.compile_factors())


def _logger():
    from vllm_torchtpu.logger import init_logger

    return init_logger("vllm_torchtpu.q38_qsaf")


def log_info(msg: str, *args) -> None:
    """Plain logger.info (never a once-per-node form)."""
    _logger().info(msg, *args)


# ---- the per-process cumulative set


def note(rows: int, slots: int, covered: int, issue: str) -> bool:
    """Record one traced fused-union call. True the first time this key is noted in the
    process."""
    entry = {"rows": int(rows), "slots": int(slots), "covered": int(covered),
             "issue": str(issue), "path": "fused"}
    k = _canon(entry)
    with _LOCK:
        if k in _ENTRIES:
            return False
        _ENTRIES[k] = entry
        return True


def snapshot() -> list[dict]:
    """Every fused-union trace key noted so far in this process, in a stable order."""
    with _LOCK:
        entries = list(_ENTRIES.values())
    return sorted(entries, key=_canon)


def reset_for_tests() -> None:
    with _LOCK:
        _ENTRIES.clear()
        _USED.clear()


def entry_ok(entry: Any, keys: dict) -> str | None:
    """None when ``entry`` is a well-formed record entry under ``keys``; else why not."""
    if not isinstance(entry, dict) or tuple(sorted(entry)) != tuple(sorted(FIELDS)):
        return f"malformed entry {entry!r}"[:300]
    for f in ("rows", "slots", "covered"):
        v = entry[f]
        if not isinstance(v, int) or isinstance(v, bool) or v < (1 if f != "covered" else 0):
            return f"entry field {f}={v!r} is not a valid count"
    if entry["issue"] != keys.get(ISSUE_ENV):
        return f"entry issue {entry['issue']!r} != the keys' issue {keys.get(ISSUE_ENV)!r}"
    if entry["path"] != "fused":
        return f"entry path {entry['path']!r} != 'fused'"
    return None


# ---- the record file


def record_path(artifact_path: str) -> str:
    return artifact_path + SUFFIX


def _payload(artifact_path: str, ident: tuple[str, str, str]) -> dict:
    body = {
        "format": FORMAT,
        "cache_hash": ident[0],
        "rank_dir": ident[1],
        "prefix": ident[2],
        "artifact": os.path.basename(artifact_path),
        "keys": key_values(),
        "entries": snapshot(),
    }
    body["sha256"] = hashlib.sha256(_canon(body).encode()).hexdigest()
    return body


def _write_text(dst: str, text: str, cache_dir: str) -> str | None:
    """Write ``text`` at ``dst`` (tmp + os.replace); None on success, else why not."""
    if not _inside(dst, cache_dir):
        return f"refused: {dst} is not inside the compile-cache dir {cache_dir}"
    tmp = f"{dst}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, dst)
    except Exception as e:  # noqa: BLE001 - never fatal to the compile
        return f"write failed: {type(e).__name__}: {e}"
    return None


def write(artifact_path: str, cache_dir: str) -> str | None:
    """Write this process's cumulative set next to ``artifact_path``. None on success,
    else the reason nothing was written."""
    ident = cache_identity(os.path.dirname(artifact_path))
    if ident is None:
        return f"refused: {artifact_path} is not in a <hash>/rank_<r>_<dp> cache dir"
    return _write_text(
        record_path(artifact_path), _canon(_payload(artifact_path, ident)), cache_dir
    )


def verify(artifact_path: str) -> dict:
    """Read and check the record next to ``artifact_path``:
    {'status': 'ok' | 'missing' | 'mismatch', 'reason', 'cache_hash', 'entries'}."""
    path = record_path(artifact_path)
    ident = cache_identity(os.path.dirname(artifact_path))
    dir_hash = ident[0] if ident else None
    out: dict[str, Any] = {"status": "mismatch", "reason": "", "cache_hash": dir_hash,
                           "entries": []}
    if not os.path.isfile(path):
        out.update(status="missing", reason=f"no record at {path}")
        return out
    try:
        with open(path) as f:
            body = json.load(f)
        assert isinstance(body, dict)
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"unreadable record: {type(e).__name__}"
        return out
    stored = body.pop("sha256", None)
    computed = hashlib.sha256(_canon(body).encode()).hexdigest()
    entries = body.get("entries")
    keys = key_values()
    if body.get("format") != FORMAT:
        out["reason"] = f"format {body.get('format')!r} != {FORMAT}"
    elif stored != computed:
        out["reason"] = f"checksum {stored} != computed {computed}"
    elif dir_hash is None or body.get("cache_hash") != dir_hash:
        out["reason"] = f"cache hash {body.get('cache_hash')} != the dir it sits in {dir_hash}"
    elif body.get("artifact") != os.path.basename(artifact_path):
        out["reason"] = f"artifact {body.get('artifact')} != {os.path.basename(artifact_path)}"
    elif body.get("keys") != keys:
        out["reason"] = f"QSAF keys {body.get('keys')} != this process's {keys}"
    elif not isinstance(entries, list):
        out["reason"] = "no entry list"
    else:
        bad = next((why for why in (entry_ok(e, keys) for e in entries) if why), None)
        if bad is not None:
            out["reason"] = bad
        elif len({_canon(e) for e in entries}) != len(entries):
            out["reason"] = "duplicate entries"
        else:
            out.update(status="ok", reason="ok", entries=entries)
    return out


# ---- hooks (TpuCompilerAdaptor) and the registry of artifacts this process used


def _register(artifact_path: str, source: str, verdict: dict) -> None:
    with _LOCK:
        _USED[artifact_path] = dict(verdict, source=source, artifact=artifact_path)


def on_saved(artifact_path: str, cache_dir: str, source: str = "compiled") -> None:
    """After an artifact was saved (a cold compile, or load's self-heal recompile):
    write the record next to it, read it back and register it."""
    if not active():
        return
    why = write(artifact_path, cache_dir)
    verdict = verify(artifact_path)
    if why is not None and verdict["status"] == "ok":
        verdict = dict(verdict, status="mismatch", reason=f"stale record: {why}")
    elif why is not None:
        verdict = dict(verdict, reason=f"{why}; {verdict['reason']}")
    _register(artifact_path, source, verdict)


def on_loaded(artifact_path: str) -> None:
    """After an artifact was replayed from the cache: verify and register the record
    next to it (the artifact this process actually loaded)."""
    if not active():
        return
    _register(artifact_path, "loaded", verify(artifact_path))


def on_peer(peer_path: str, save_path: str, cache_dir: str, linked: bool) -> None:
    """After a bucket was replayed from a peer rank's artifact: the record next to the
    PEER's artifact describes it. Registered, and, when the artifact was linked into this
    rank's dir, copied next to that link (our cache dir only)."""
    if not active():
        return
    verdict = verify(peer_path)
    _register(peer_path, "peer", verdict)
    if linked and verdict["status"] == "ok":
        try:
            with open(record_path(peer_path)) as f:
                text = f.read()
        except Exception as e:  # noqa: BLE001
            why = f"peer record unreadable: {type(e).__name__}"
        else:
            why = _write_text(record_path(save_path), text, cache_dir)
        if why is not None:
            log_info("q38 QSAF record copy for %s not made: %s", save_path, why)


def used() -> list[dict]:
    """Every artifact this process used, with its record's verdict."""
    with _LOCK:
        return [dict(v) for _, v in sorted(_USED.items())]
