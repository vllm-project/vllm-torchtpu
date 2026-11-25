# SPDX-License-Identifier: Apache-2.0
"""q38 IDXXLINE: the persisted trace record of the IDXX members' decisions.

The three IDXX members (VLLM_TPU_QSA_INDEXER_KEY_LANES, VLLM_TPU_QSA_DCP_SC_BOUNDED_COPY,
VLLM_TPU_QSA_DCP_FUSED_MERGE) decide while the QSA ops are traced, per call shape. A
trace runs only on a cold compile, and only on the first trace of a shape in a process
(nested ``jax.jit`` caches, the pallas ops' per-shape lowering cache, chunked decode
windows), so a trace-time line is absent on a warm start and a per-artifact record of
"what this artifact traced" would be empty for most buckets.

So (main-approved design I-1a):

1. RECORD. Each member's trace path calls ``note`` with its ACTUAL decision, keyed by
   call shape, into one CUMULATIVE set per process.
2. PERSIST. When ``TpuCompilerAdaptor`` saves an artifact, it writes that set next to
   the artifact, ``<cache_dir>/<artifact key>.q38idxx.json``, inside vLLM's
   compile-cache dir ``<root>/<hash>/rank_<r>_<dp>/<prefix>`` (the dir the artifact
   itself is in; nothing is written anywhere else).
3. FORMAT. The record carries the cache hash (the ``<hash>`` dir name), the artifact it
   sits next to, the IDXX env values, the entries and a sha256 over all of that. Read
   back, it must match the dir it sits in, the artifact, the env and its checksum, and
   every entry must agree with its member's rule (SC_BOUNDED_COPY through the same
   ``bounded_copy_decision`` the kernel decides with); anything else is a MISMATCH.
4. Every artifact this process uses (compiled and saved, loaded from the cache, or
   replayed from a peer rank) registers the verified record next to it; the runner
   prints the union per rank after the warmup ladder (runner/idxx_engagement.py).

Honest label: a record says what was TRACED while this cache hash was built. That the
loaded executable makes the same decisions rests on the compile key (which hashes the
kernel sources and the set IDXX keys) -- inferred, not observed on device.

Everything here is a no-op, reads and writes nothing, when no IDXX key is set.
Python only: nothing here enters a program. No file is ever removed.
"""

import hashlib
import json
import os
import re
import sys
import threading
from typing import Any

KEYS = (
    "VLLM_TPU_QSA_INDEXER_KEY_LANES",
    "VLLM_TPU_QSA_DCP_SC_BOUNDED_COPY",
    "VLLM_TPU_QSA_DCP_FUSED_MERGE",
)
MEMBERS = ("KEY_LANES", "SC_BOUNDED_COPY", "FUSED_MERGE")
FORMAT = "q38-idxx-record/1"
SUFFIX = ".q38idxx.json"
_RANK_DIR = re.compile(r"rank_\d+_\d+")

_LOCK = threading.Lock()
# canonical (member, shape) -> list of distinct decisions traced for it (one, normally)
_DECISIONS: dict[str, list[dict]] = {}
# artifact path -> what this process used it as, and its record's verdict
_USED: dict[str, dict] = {}


def key_values() -> dict[str, str]:
    """The set IDXX keys, as tpu_compiler._tpu_compile_env_factors adds them."""
    out = {}
    for name in KEYS:
        raw = os.environ.get(name, "").strip()
        if raw:
            out[name] = raw
    return out


def active() -> bool:
    """Any IDXX key set (the same test that puts it into the compile factors)."""
    return any(os.environ.get(name, "").strip() for name in KEYS)


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _logger():
    from vllm_torchtpu.logger import init_logger

    return init_logger("vllm_torchtpu.q38_idxx")


def log_info(msg: str, *args) -> None:
    """Plain logger.info (never a once-per-node form)."""
    _logger().info(msg, *args)


def process_rank_fields() -> tuple[int, int, int]:
    """(rank, tp_rank, pid) from modules this process already imported; -1 where
    unknown. Imports nothing (callable at trace time)."""
    rank = tp_rank = -1
    dist = sys.modules.get("torch.distributed")
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
    except Exception:  # noqa: BLE001 - a log field
        pass
    ps = sys.modules.get("vllm.distributed.parallel_state")
    try:
        if ps is not None:
            tp_rank = int(ps.get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - a log field
        pass
    return rank, tp_rank, os.getpid()


# ---- member rules (the decision each recorded entry must agree with)


def key_lanes_rule(lanes: int, width: int, decode: bool) -> tuple[bool, str]:
    """KEY_LANES engages on a decode / verify pass when the key is below the key
    row width (pooled_qsa._select_blocks; prefill keeps whole rows)."""
    if not lanes:
        return False, "key lanes 0"
    if not decode:
        return False, "not a decode / verify pass (prefill keeps whole rows)"
    if lanes >= width:
        return False, f"lanes={lanes} >= row width {width}"
    return True, f"decode pass, lanes={lanes} < row width {width}"


def expected(entry: dict, keys: dict[str, str]) -> tuple[bool, str] | None:
    """(engaged, reason) that ``entry``'s member rule gives for its shape, or None
    when the entry is malformed."""
    try:
        member, shape = entry["member"], entry["shape"]
        if member == "SC_BOUNDED_COPY":
            from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
                bounded_copy_decision,
            )

            return bounded_copy_decision(int(shape["b"]), int(shape["n"]), True)
        if member == "KEY_LANES":
            return key_lanes_rule(
                int(shape["lanes"]), int(shape["width"]), bool(shape["decode"])
            )
        if member == "FUSED_MERGE":
            on = keys.get("VLLM_TPU_QSA_DCP_FUSED_MERGE") == "1"
            return on, "VLLM_TPU_QSA_DCP_FUSED_MERGE=1" if on else "flag off"
    except Exception:  # noqa: BLE001 - malformed entry
        return None
    return None


def path_ok(entry: dict) -> bool:
    """The recorded path agrees with the recorded decision (the kernel / trace path names)."""
    try:
        member, shape, engaged, path = entry["member"], entry["shape"], bool(entry["engaged"]), entry["path"]
        if member == "SC_BOUNDED_COPY":
            return path == "bounded" if engaged else str(path).startswith("partitioned:p=")
        if member == "KEY_LANES":
            return path == (f"lanes:{shape['lanes']}/{shape['width']}" if engaged else "whole_rows")
        if member == "FUSED_MERGE":
            return path == "fused"
    except Exception:  # noqa: BLE001 - malformed entry
        return False
    return False


# ---- the per-process cumulative set


def note(member: str, shape: dict, engaged: bool, path: str, reason: str) -> bool:
    """Record one traced decision. True the first time this (member, shape) gets this
    decision in the process (callers log then)."""
    assert member in MEMBERS, member
    entry = {
        "member": member,
        "shape": dict(shape),
        "engaged": bool(engaged),
        "path": str(path),
        "reason": str(reason),
    }
    k = _canon([member, entry["shape"]])
    with _LOCK:
        seen = _DECISIONS.setdefault(k, [])
        if entry in seen:
            return False
        seen.append(entry)
        return True


def snapshot() -> list[dict]:
    """Every decision traced so far in this process, in a stable order."""
    with _LOCK:
        entries = [e for v in _DECISIONS.values() for e in v]
    return sorted(entries, key=_canon)


def reset_for_tests() -> None:
    with _LOCK:
        _DECISIONS.clear()
        _USED.clear()


# ---- the record file


def cache_identity(cache_dir: str) -> tuple[str, str, str] | None:
    """(hash, rank dir, prefix) of a vLLM compile-cache dir
    ``.../<hash>/rank_<r>_<dp>[/<prefix>]``; None for any other layout."""
    parts = os.path.normpath(os.path.abspath(cache_dir)).split(os.sep)
    for i in range(len(parts) - 1, 0, -1):
        if _RANK_DIR.fullmatch(parts[i]):
            return parts[i - 1], parts[i], "/".join(parts[i + 1 :])
    return None


def record_path(artifact_path: str) -> str:
    return artifact_path + SUFFIX


def _inside(path: str, root: str) -> bool:
    real, top = os.path.realpath(path), os.path.realpath(root)
    return real.startswith(top.rstrip(os.sep) + os.sep)


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
    """Write this process's cumulative set next to ``artifact_path``. None on
    success, else the reason nothing was written."""
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
    if body.get("format") != FORMAT:
        out["reason"] = f"format {body.get('format')!r} != {FORMAT}"
    elif stored != computed:
        out["reason"] = f"checksum {stored} != computed {computed}"
    elif dir_hash is None or body.get("cache_hash") != dir_hash:
        out["reason"] = (
            f"cache hash {body.get('cache_hash')} != the dir it sits in {dir_hash}"
        )
    elif body.get("artifact") != os.path.basename(artifact_path):
        out["reason"] = (
            f"artifact {body.get('artifact')} != {os.path.basename(artifact_path)}"
        )
    elif body.get("keys") != key_values():
        out["reason"] = f"IDXX keys {body.get('keys')} != this process's {key_values()}"
    elif not isinstance(entries, list):
        out["reason"] = "no entry list"
    else:
        seen: dict[str, dict] = {}
        for e in entries:
            want = expected(e, body["keys"]) if isinstance(e, dict) else None
            if want is None:
                out["reason"] = f"malformed entry {e!r}"[:300]
                break
            if bool(e.get("engaged")) != want[0]:
                out["reason"] = (
                    f"{e['member']} {_canon(e['shape'])} recorded engaged="
                    f"{bool(e.get('engaged'))} but its rule gives {want[0]} ({want[1]})"
                )
                break
            if not path_ok(e):
                out["reason"] = (
                    f"{e['member']} {_canon(e['shape'])} path {e.get('path')!r} disagrees with "
                    f"engaged={bool(e.get('engaged'))}"
                )
                break
            k = _canon([e["member"], e["shape"]])
            if k in seen and seen[k] != e:
                out["reason"] = f"conflicting decisions for {e['member']} {_canon(e['shape'])}"
                break
            seen[k] = e
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
    PEER's artifact is the one that describes it. Registered, and, when the artifact was
    linked into this rank's dir, copied next to that link (our cache dir only)."""
    if not active():
        return
    verdict = verify(peer_path)
    _register(peer_path, "peer", verdict)
    if linked and verdict["status"] == "ok":
        try:
            with open(record_path(peer_path)) as f:
                text = f.read()
        except Exception as e:  # noqa: BLE001
            text, why = None, f"peer record unreadable: {type(e).__name__}"
        else:
            why = _write_text(record_path(save_path), text, cache_dir)
        if why is not None:
            log_info("q38 IDXX record copy for %s not made: %s", save_path, why)


def used() -> list[dict]:
    """Every artifact this process used, with its record's verdict."""
    with _LOCK:
        return [dict(v) for _, v in sorted(_USED.items())]
