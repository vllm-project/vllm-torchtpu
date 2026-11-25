#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q38cc.py v1 -- INT-CACHE (Q38 PD decode program), 2026-11-15.

Pull-only warm compile cache for Q38 PD served cars (recipe infx-qwen38-flash-agentx-disagg).  Design:
reports/int_cache_design.md.  Nothing in the installed recipe calls this file today; a recipe opt-in (render var
Q38CC, default off) would embed it in the server-leg script.  Python stdlib only.

Subcommands
  key       --render R --role decode|prefill --tree T --runtime RT.json [--payload-out F]
            Offline key of one server leg, from the render (leg env + serve args + image) and the car's tree.
  pull      --role --cache-dir --src --image --env-names F --args F [--tree auto|DIR] [--expect-src H]
            [--runtime auto|RT.json] [--host-rank N] [--status F]
            Runtime, in the pod, before `vllm serve`.  Never blocks the start: every failure prints one loud
            '[Q38CC] WARNING COLD reason=...' line, leaves CACHE_DIR in its fresh state, and exits 0.
  publish   (populate cars only; refused unless --mode populate and Q38CC_MODE=populate)
            --role --cache-dir --dst --image --env-names --args [--tree] [--runtime] [--host-rank N]
            Streams CACHE_DIR to <dst>/<key>/host<N>.tar.{zst,gz} with no local staging, then writes
            host<N>.payload.json and, last, host<N>.READY.json (create-once).
  keyline   as pull's key inputs; prints the key line only (Q38CC=keyonly: a free config marker).

The key (role-<24 hex of sha256 over a canonical JSON payload) covers:
  env_compile  every string leg env value EXCEPT keys the car's own tree declares compile-ignored (registered in
               envs.environment_variables AND listed in compilation/tpu_compiler.py _TPU_COMPILE_ENV_IGNORED).
               Unregistered keys stay in (conservative: QWEN4_EXP_MAX_POOL_BYTES, TPU_TOKEN_BUCKET_LINEAR_*,
               LIBTPU_INIT_ARGS, TPU_TOPOLOGY ... are unregistered on F2's tree and reach programs).
  bucket       compile-ignored keys that decide WHICH programs are compiled at start-up (FORCED_RX below, minus
               the log-only exemptions), forced back in, plus the bucket-deciding serve args.
  args         the full `vllm serve` argument list as rendered.
  tree         src_hash (the launchers' SRC formula: LC_ALL=C sorted sha1sum of src/**/*.py, 10 hex) and
               tree_hash (sha256 over every regular file under src/, __pycache__ / *.pyc excluded).
  runtime      installed dist versions (libtpu, jax, jaxlib, torch, torch-tpu, vllm, ...) and a content
               fingerprint of the installed vllm / torch_tpu / tpu_sync_torch packages.
  image, role, schema.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

TOOL_VERSION = "q38cc-v1"  # v1.1 (2026-11-15): parallel gzip via the recipe helper when zstd is absent; key schema unchanged
MUTANT = os.environ.get("Q38CC_MUTANT", "")  # selftest tool mutants only; empty in any real use

FORCED_RX = re.compile(r"PRECOMPILE|WARM|BUCKET|PADDING|COMPILE_SIZES|SERVED_ONLY|CUDAGRAPH|CAPTURE_SIZES")
LOG_ONLY_EXEMPT = frozenset({"VLLM_TPU_LOG_STEP_BUCKETS", "VLLM_TPU_LOG_STEP_BUCKETS_PERIOD"})
EXCLUDE_PREFIXES = ("Q38CC_", "_Q38CC", "_CDK_GCS_B64")
BUCKET_ARG_FLAGS = ("--max-num-seqs", "--max-num-batched-tokens", "--max-model-len", "--speculative-config",
                    "--compilation-config", "--block-size", "--tensor-parallel-size", "--pipeline-parallel-size",
                    "--data-parallel-size")
RUNTIME_DISTS = ("vllm", "vllm-torchtpu", "torch", "torch-tpu", "torch_xla", "jax", "jaxlib", "libtpu",
                 "tpu-sync-torch")
RUNTIME_MODULES = ("vllm", "torch_tpu", "tpu_sync_torch")
# v1.2 (2026-11-15): real CDK job ids are j-<8>-<4>-<4>-<4> (e.g. j-3154a28f-1eaf-451a-8229); v1.1 demanded a 5-group UUID and
# would have refused every real populate source. Both forms are accepted.
SRC_PIN_RX = re.compile(r"^gs://cloud-devkit/jobs/j-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}(-[0-9a-f]{12})?/q38cc$")
SRC_PIN_RX_TEST = re.compile(r"^file:///[A-Za-z0-9_./-]+/jobs/j-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}(-[0-9a-f]{12})?/q38cc$")
ARCHIVE_EXCLUDED_TOP = frozenset({"tmp", "tpu_logs"})
MANIFEST_NAME = ".q38cc_manifest.json"


def log(msg: str) -> None:
    sys.stdout.write("[Q38CC] " + msg + "\n")
    sys.stdout.flush()


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# 1. The car's tree: compile-key rules and content hashes.
# ---------------------------------------------------------------------------------------------------------------------

def _block(src: str, start_re: str) -> str:
    m = re.search(start_re, src, re.M)
    if not m:
        raise ValueError("block not found: " + start_re)
    e = re.search(r"^\}", src[m.end():], re.M)
    if not e:
        raise ValueError("block end not found: " + start_re)
    return src[m.end():m.end() + e.start()]


def parse_tree_rules(tree_root: str) -> dict:
    base = os.path.join(tree_root, "src", "vllm_torchtpu")
    with open(os.path.join(base, "envs.py"), encoding="utf-8") as f:
        envs_src = f.read()
    with open(os.path.join(base, "compilation", "tpu_compiler.py"), encoding="utf-8") as f:
        comp_src = f.read()
    registered = set(re.findall(r'^    "([A-Z][A-Z0-9_]*)"\s*:',
                                _block(envs_src, r"^environment_variables\b[^\n]*=\s*\{\s*$"), re.M))
    ignored = set(re.findall(r'^\s+"([A-Z][A-Z0-9_]*)",',
                             _block(comp_src, r"^_TPU_COMPILE_ENV_IGNORED\s*=\s*\{\s*$"), re.M))
    if not registered or not ignored:
        raise ValueError("empty env registry or ignore list in %s" % tree_root)
    forced = {k for k in ignored if FORCED_RX.search(k)} - LOG_ONLY_EXEMPT
    if MUTANT == "no_forced":
        forced = set()
    return {"registered": registered, "ignored": ignored, "forced": forced}


def is_compile_ignored(name: str, rules: dict) -> bool:
    if MUTANT == "ignore_all":
        return True
    if MUTANT == "no_filter":
        return False
    return name in rules["registered"] and name in rules["ignored"] and name not in rules["forced"]


def src_hash(tree_root: str) -> str:
    """The launchers' SRC hash: find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha1sum |
    sha1sum | cut -c1-10, run from the tree root."""
    paths = []
    for cur, dirs, files in os.walk(os.path.join(tree_root, "src")):
        for fn in files:
            full = os.path.join(cur, fn)
            if fn.endswith(".py") and os.path.isfile(full) and not os.path.islink(full):
                paths.append(os.path.relpath(full, tree_root).replace(os.sep, "/"))
    paths.sort(key=lambda p: p.encode("utf-8", "surrogateescape"))
    lines = []
    for rel in paths:
        with open(os.path.join(tree_root, rel), "rb") as f:
            lines.append("%s  %s\n" % (hashlib.sha1(f.read()).hexdigest(), rel))
    return hashlib.sha1("".join(lines).encode("utf-8", "surrogateescape")).hexdigest()[:10]


def tree_hash(tree_root: str) -> str:
    """sha256 over every regular file and symlink under src/ (path, size, content); __pycache__ and *.pyc/*.pyo
    excluded, so a running engine's byte-code never changes it."""
    entries = []
    for cur, dirs, files in os.walk(os.path.join(tree_root, "src")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith((".pyc", ".pyo")):
                continue
            if MUTANT == "py_only" and not fn.endswith(".py"):
                continue
            full = os.path.join(cur, fn)
            entries.append(os.path.relpath(full, tree_root).replace(os.sep, "/"))
    entries.sort(key=lambda p: p.encode("utf-8", "surrogateescape"))
    h = hashlib.sha256()
    for rel in entries:
        full = os.path.join(tree_root, rel)
        if os.path.islink(full):
            h.update(b"L\0" + rel.encode() + b"\0" + os.readlink(full).encode() + b"\0")
        elif os.path.isfile(full):
            with open(full, "rb") as f:
                data = f.read()
            h.update(b"F\0" + rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()[:16]


def tree_root_from_import() -> str:
    """In the pod: the repo root of the vllm_torchtpu the engine will import (find_spec executes nothing)."""
    import importlib.util
    spec = importlib.util.find_spec("vllm_torchtpu")
    if spec is None or not spec.origin:
        raise RuntimeError("vllm_torchtpu not importable")
    pkg = os.path.dirname(os.path.realpath(spec.origin))
    root = os.path.dirname(os.path.dirname(pkg))
    if os.path.basename(os.path.dirname(pkg)) != "src":
        raise RuntimeError("vllm_torchtpu resolves outside a src/ tree: %s" % pkg)
    return root


# ---------------------------------------------------------------------------------------------------------------------
# 2. Render parsing (no PyYAML: the sanitized cdk render's container env is a flat list of single-line scalars).
# ---------------------------------------------------------------------------------------------------------------------

def _scalar(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return json.loads(raw)
    if raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
        return raw[1:-1].replace("''", "'")
    return raw


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_serve_args(cmd_lines: list) -> list:
    joined = " ".join(ln.strip().rstrip("\\").strip() for ln in cmd_lines)
    toks = shlex.split(joined, posix=True)
    if toks[:2] != ["vllm", "serve"]:
        raise ValueError("not a vllm serve block")
    out = []
    for t in toks[2:]:
        if t in (">", "2>&1", "&", "|") or t.startswith(">") or t.startswith("2>"):
            break
        out.append(t)
    return out


def render_legs(text: str) -> dict:
    """role -> {env (value entries), valuefrom (names), image, args, serve_line (0-based), container_line}."""
    lines = text.splitlines()
    legs = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^( +)name: vllm-tpu-([a-z0-9]+)$", ln)
        if not m:
            continue
        ind, role = len(m.group(1)), m.group(2)
        start = None
        for j in range(i - 1, -1, -1):
            if lines[j].startswith(" " * (ind - 2) + "- ") and _indent(lines[j]) == ind - 2:
                start = j
                break
        if start is None:
            raise ValueError("container start not found for %s" % role)
        env, valuefrom, image, serve = {}, [], None, None
        j = start
        while j < i:
            s = lines[j]
            if _indent(s) == ind and s.strip() == "env:":
                k = j + 1
                cur = None
                while k < i and _indent(lines[k]) >= ind:
                    t = lines[k]
                    if _indent(t) == ind and not t.strip().startswith("- "):
                        break
                    mm = re.match(r"^ {%d}- name: (\S+)$" % ind, t)
                    if mm:
                        cur = mm.group(1)
                    else:
                        mv = re.match(r"^ {%d}value: (.*)$" % (ind + 2), t)
                        if mv and cur is not None:
                            env[cur] = _scalar(mv.group(1))
                        elif re.match(r"^ {%d}valueFrom:" % (ind + 2), t) and cur is not None:
                            valuefrom.append(cur)
                    k += 1
                j = k
                continue
            mi = re.match(r"^ {%d}image: (\S+)$" % ind, s)
            if mi:
                image = mi.group(1)
            if re.match(r"^\s+vllm serve \\$", s) and serve is None:
                serve = j
            j += 1
        if serve is None:
            continue  # not a server leg (proxy, router, bench sidecar)
        if image is None:
            raise ValueError("no image for server leg %s" % role)
        cmd = [lines[serve]]
        k = serve
        while lines[k].rstrip().endswith("\\"):
            k += 1
            cmd.append(lines[k])
        legs[role] = {"env": env, "valuefrom": valuefrom, "image": image, "args": parse_serve_args(cmd),
                      "serve_line": serve, "container_line": i, "start_line": start}
    if not legs:
        raise ValueError("no vllm-tpu-* server container in the render")
    return legs


# ---------------------------------------------------------------------------------------------------------------------
# 3. The key.
# ---------------------------------------------------------------------------------------------------------------------

def bucket_args(args: list) -> dict:
    out = {}
    for i, a in enumerate(args):
        for flag in BUCKET_ARG_FLAGS:
            if a == flag and i + 1 < len(args):
                out[flag] = args[i + 1]
            elif a.startswith(flag + "="):
                out[flag] = a.split("=", 1)[1]
    return out


def compute_payload(role: str, env: dict, env_names, args: list, image: str, tree_root: str, runtime: dict,
                    rules: dict = None) -> dict:
    rules = rules or parse_tree_rules(tree_root)
    env_compile, ignored_seen, missing = {}, [], []
    names = sorted(set(env_names))
    if MUTANT == "all_environ":
        names = sorted(set(env))
    for k in names:
        if k.startswith(EXCLUDE_PREFIXES):
            continue
        if k not in env:
            missing.append(k)
            continue
        if is_compile_ignored(k, rules):
            ignored_seen.append(k)
            continue
        env_compile[k] = str(env[k])
    payload = {
        "schema": TOOL_VERSION,
        "role": role,
        "image": image,
        "args": list(args),
        "env_compile": env_compile,
        "env_missing": missing,
        "bucket": {"env": {k: str(env[k]) for k in sorted(rules["forced"]) if k in env and k in set(names)},
                   "args": bucket_args(args)},
        "tree": {"src_hash": src_hash(tree_root), "tree_hash": tree_hash(tree_root)},
        "runtime": runtime,
    }
    if MUTANT == "no_tree":
        payload.pop("tree")
    if MUTANT == "no_versions":
        payload.pop("runtime")
    payload["_meta"] = {"ignored_count": len(ignored_seen), "compile_count": len(env_compile)}
    return payload


def payload_for_hash(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def key_of(payload: dict) -> str:
    return "%s-%s" % (payload["role"], sha256_hex(canon(payload_for_hash(payload)))[:24])


def keyline(payload: dict, mode: str) -> str:
    k = key_of(payload)
    t = payload.get("tree", {})
    b = sha256_hex(canon(payload["bucket"]))[:8]
    r = sha256_hex(canon(payload.get("runtime", {})))[:8]
    m = payload.get("_meta", {})
    return ("role=%s mode=%s Key=%s src_hash=%s tree_hash=%s env_compile=%d env_ignored=%d bucket=%s runtime=%s "
            "image=%s schema=%s" % (payload["role"], mode, k, t.get("src_hash"), t.get("tree_hash"),
                                    m.get("compile_count", -1), m.get("ignored_count", -1), b, r,
                                    payload["image"].rsplit(":", 1)[-1], TOOL_VERSION))


def payload_diff(a: dict, b: dict, limit: int = 12) -> list:
    """Field-level differences between two payloads (for the loud MISS line)."""
    out = []

    def walk(x, y, path):
        if len(out) >= limit:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k.startswith("_"):
                    continue
                if k not in x:
                    out.append("%s.%s: car=<absent> published=%r" % (path, k, y[k]))
                elif k not in y:
                    out.append("%s.%s: car=%r published=<absent>" % (path, k, x[k]))
                else:
                    walk(x[k], y[k], path + "." + k)
        elif x != y:
            out.append("%s: car=%r published=%r" % (path, x, y))

    walk(a, b, "payload")
    return [s[:240] for s in out]


# ---------------------------------------------------------------------------------------------------------------------
# 4. Runtime fields (versions, package fingerprint).
# ---------------------------------------------------------------------------------------------------------------------

def runtime_fields_live() -> dict:
    import importlib.metadata as md
    import importlib.util
    versions = {}
    for d in RUNTIME_DISTS:
        try:
            versions[d] = md.version(d)
        except Exception:  # noqa: BLE001
            versions[d] = None
    h = hashlib.sha256()
    for mod in RUNTIME_MODULES:
        try:
            spec = importlib.util.find_spec(mod)
        except Exception:  # noqa: BLE001
            spec = None
        if spec is None or not spec.origin:
            h.update(("absent:%s\n" % mod).encode())
            continue
        root = os.path.dirname(os.path.realpath(spec.origin))
        for cur, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for fn in sorted(files):
                if not (fn.endswith(".py") or fn.endswith(".so") or ".so." in fn):
                    continue
                full = os.path.join(cur, fn)
                rel = os.path.relpath(full, root)
                try:
                    st = os.stat(full)
                    with open(full, "rb") as f:
                        if st.st_size <= 16 * 1024 * 1024:
                            data = f.read()
                        else:
                            data = f.read(1 << 20)
                            f.seek(st.st_size - (1 << 20))
                            data += f.read(1 << 20)
                except OSError:
                    continue
                h.update(("%s/%s:%d\0" % (mod, rel, st.st_size)).encode())
                h.update(hashlib.sha256(data).digest())
    return {"versions": versions, "pkg_fp": h.hexdigest()[:20]}


def load_runtime(spec: str) -> dict:
    if spec == "auto":
        return runtime_fields_live()
    with open(spec, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------------
# 5. Object store backends: gs:// through the recipe's own /tmp/cdk_gcs.py CLI; file:// for CPU tests.
# ---------------------------------------------------------------------------------------------------------------------

class FileBackend:
    def __init__(self, prefix: str):
        self.root = prefix[len("file://"):]

    def _p(self, rel):
        return os.path.join(self.root, rel)

    def exists(self, rel):
        return os.path.isfile(self._p(rel))

    def get_bytes(self, rel):
        with open(self._p(rel), "rb") as f:
            return f.read()

    def open_stream(self, rel):
        return open(self._p(rel), "rb"), None

    def put_bytes_once(self, rel, data):
        p = self._p(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return 2
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return 0

    def put_stream_once(self, rel):
        p = self._p(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.exists(p):
            return None, None
        f = open(p, "xb")
        return f, None

    def list_keys(self):
        try:
            return sorted(os.listdir(self.root))
        except OSError:
            return []


class GcsBackend:
    """Reads and create-once writes through `python3 /tmp/cdk_gcs.py` (the recipe's own helper; GCE metadata creds)."""

    def __init__(self, prefix: str, helper: str = "/tmp/cdk_gcs.py"):
        self.prefix = prefix.rstrip("/")
        self.helper = helper

    def _u(self, rel):
        return self.prefix + "/" + rel

    def exists(self, rel):
        return subprocess.run([sys.executable, self.helper, "exists", self._u(rel)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    def get_bytes(self, rel):
        r = subprocess.run([sys.executable, self.helper, "stream_get", self._u(rel), "4", "8"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise IOError("stream_get failed for %s" % self._u(rel))
        return r.stdout

    def open_stream(self, rel):
        p = subprocess.Popen([sys.executable, self.helper, "stream_get", self._u(rel), "16", "32"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return p.stdout, p

    def put_bytes_once(self, rel, data):
        r = subprocess.run([sys.executable, self.helper, "stream_put", self._u(rel), "--if-generation-match=0"],
                           input=data, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return r.returncode

    def put_stream_once(self, rel):
        p = subprocess.Popen([sys.executable, self.helper, "stream_put", self._u(rel), "--if-generation-match=0"],
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return p.stdin, p

    def list_keys(self):
        r = subprocess.run([sys.executable, self.helper, "ls_subdirs", self.prefix], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, text=True)
        return sorted(x.rstrip("/").rsplit("/", 1)[-1] for x in r.stdout.split() if x.strip())


def backend_for(prefix: str):
    if prefix.startswith("file://"):
        return FileBackend(prefix)
    if prefix.startswith("gs://"):
        return GcsBackend(prefix)
    raise ValueError("unsupported prefix %s" % prefix)


def src_pinned(src: str) -> bool:
    if MUTANT == "src_any":
        return bool(src)
    if SRC_PIN_RX.match(src or ""):
        return True
    return os.environ.get("Q38CC_TEST_BACKEND") == "1" and bool(SRC_PIN_RX_TEST.match(src or ""))


# ---------------------------------------------------------------------------------------------------------------------
# 6. Cache-dir manifest, fresh-state check, wipe.
# ---------------------------------------------------------------------------------------------------------------------

def _archive_members(cache_dir: str) -> list:
    out = []
    for cur, dirs, files in os.walk(cache_dir):
        rel_cur = os.path.relpath(cur, cache_dir)
        if rel_cur == ".":
            dirs[:] = sorted(d for d in dirs if d not in ARCHIVE_EXCLUDED_TOP)
        else:
            dirs.sort()
        for fn in sorted(files):
            if fn == MANIFEST_NAME or ".tmp" in fn or fn.endswith(".lock"):
                continue
            full = os.path.join(cur, fn)
            if os.path.islink(full) or not os.path.isfile(full):
                continue
            out.append(os.path.relpath(full, cache_dir).replace(os.sep, "/"))
    return out


def build_manifest(cache_dir: str, key: str, payload_sha: str) -> dict:
    files = {}
    for rel in _archive_members(cache_dir):
        h, n = hashlib.sha256(), 0
        with open(os.path.join(cache_dir, rel), "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b:
                    break
                h.update(b)
                n += len(b)
        files[rel] = [n, h.hexdigest()]
    return {"schema": TOOL_VERSION, "key": key, "payload_sha256": payload_sha, "file_count": len(files),
            "unpacked_bytes": sum(v[0] for v in files.values()), "files": files}


def snapshot(cache_dir: str) -> dict:
    snap = {}
    for rel in _archive_members(cache_dir):
        st = os.stat(os.path.join(cache_dir, rel))
        snap[rel] = (st.st_size, st.st_mtime_ns, st.st_ino)
    return snap


def fresh_state(cache_dir: str) -> tuple:
    """(ok, n_files): a cold-start CACHE_DIR holds directories only (tmp/ and tpu_logs/ may hold files)."""
    if MUTANT == "no_fresh_check":
        return True, 0
    n = len(_archive_members(cache_dir))
    return n == 0, n


def dir_skeleton(cache_dir: str) -> list:
    out = []
    for cur, dirs, files in os.walk(cache_dir):
        for d in dirs:
            out.append(os.path.relpath(os.path.join(cur, d), cache_dir))
    return sorted(out)


def wipe_to_skeleton(cache_dir: str, skeleton: list) -> None:
    if MUTANT == "no_wipe":
        return
    for name in os.listdir(cache_dir):
        if name in ARCHIVE_EXCLUDED_TOP:
            continue
        p = os.path.join(cache_dir, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.unlink(p)
            except OSError:
                pass
    for rel in skeleton:
        os.makedirs(os.path.join(cache_dir, rel), exist_ok=True)


# ---------------------------------------------------------------------------------------------------------------------
# 7. Compression (zstd binary when present, else gzip from the stdlib) with on-the-fly sha256.
# ---------------------------------------------------------------------------------------------------------------------

def have_zstd() -> bool:
    return shutil.which("zstd") is not None and os.environ.get("Q38CC_FORCE_GZIP") != "1"


def gcs_helper() -> str:
    """The recipe's own /tmp/cdk_gcs.py (pgzip / pgunzip: multi-member parallel gzip, the CDK uploader's no-zstd path)."""
    h = os.environ.get("Q38CC_GCS_HELPER", "/tmp/cdk_gcs.py")
    if os.environ.get("Q38CC_NO_PGZIP") == "1" or not os.path.isfile(h):
        return ""
    return h


def compressor() -> str:
    if have_zstd():
        return "zstd"
    return "pgzip" if gcs_helper() else "gzip"


def _workers() -> str:
    return str(max(2, min(16, int(os.environ.get("Q38CC_WORKERS", "16")))))


def decompress_cmd(ext: str):
    if ext == "tar.zst":
        return ["zstd", "-d", "-q", "-c"]
    h = gcs_helper()
    return [sys.executable, h, "pgunzip", _workers()] if h else None


def compress_cmd(comp: str):
    if comp == "zstd":
        return ["zstd", "-1", "-T4", "-q", "-c"]
    if comp == "pgzip":
        return [sys.executable, gcs_helper(), "pgzip", _workers(), "1"]
    return None


class HashingReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw, self.h, self.n = raw, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.raw.read(len(b))
        if not data:
            return 0
        self.h.update(data)
        self.n += len(data)
        b[:len(data)] = data
        return len(data)


class HashingWriter(io.RawIOBase):
    def __init__(self, raw):
        self.raw, self.h, self.n = raw, hashlib.sha256(), 0

    def writable(self):
        return True

    def write(self, b):
        b = bytes(b)
        self.h.update(b)
        self.n += len(b)
        self.raw.write(b)
        return len(b)


def safe_extract(tf: tarfile.TarFile, dest: str) -> int:
    n = 0
    for m in tf:
        name = m.name
        if name.startswith("/") or ".." in name.split("/") or not (m.isfile() or m.isdir()):
            raise ValueError("unsafe tar member %r" % name)
        target = os.path.join(dest, name)
        if m.isdir():
            os.makedirs(target, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        src = tf.extractfile(m)
        with open(target, "wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        n += 1
    return n


# ---------------------------------------------------------------------------------------------------------------------
# 8. pull (measured cars) and publish (populate cars only).
# ---------------------------------------------------------------------------------------------------------------------

def own_sha16() -> str:
    with open(os.path.abspath(__file__), "rb") as f:
        return sha256_hex(f.read())[:16]


def _inputs_from_args(a) -> tuple:
    if getattr(a, "tool_sha", "") and own_sha16() != a.tool_sha:
        raise RuntimeError("tool-mangled: q38cc.py sha256 %s != expected %s" % (own_sha16(), a.tool_sha))
    with open(a.env_names, encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    with open(a.args, encoding="utf-8") as f:
        serve_args = json.load(f)
    tree = tree_root_from_import() if a.tree == "auto" else a.tree
    if getattr(a, "check_import", False) and MUTANT != "no_import_check":
        got = os.path.realpath(tree_root_from_import())
        if got != os.path.realpath(tree):
            raise RuntimeError("tree-not-imported: vllm_torchtpu resolves under %s, not the car tree %s" % (got, tree))
    runtime = load_runtime(a.runtime)
    env = dict(os.environ)
    payload = compute_payload(a.role, env, names, serve_args, a.image, tree, runtime)
    return payload, tree


def _write_status(path: str, doc: dict) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, sort_keys=True)
    except OSError:
        pass


def cmd_pull(a) -> int:
    t0 = time.time()
    status = {"result": "COLD", "reason": "init"}
    cache_dir = a.cache_dir
    skeleton = dir_skeleton(cache_dir) if os.path.isdir(cache_dir) else []

    def cold(reason, detail=""):
        status.update(result="COLD", reason=reason, detail=detail)
        log("WARNING COLD reason=%s key=%s %s-> COLD COMPILE (fail-closed)" % (reason, status.get("key", "?"),
                                                                              (detail + " ") if detail else ""))
        _write_status(a.status, status)
        return 1 if MUTANT == "cold_rc1" else 0

    try:
        payload, tree = _inputs_from_args(a)
    except Exception as exc:  # noqa: BLE001
        r = "key-error"
        for tag in ("tree-not-imported", "tool-mangled"):
            if tag in str(exc):
                r = tag
        return cold(r, repr(exc)[:200])
    key = key_of(payload)
    psha = sha256_hex(canon(payload_for_hash(payload)))
    status.update(key=key, payload_sha256=psha)
    log(keyline(payload, "pull"))
    if not os.path.isdir(cache_dir):
        return cold("no-cache-dir", cache_dir)
    if a.expect_src and payload.get("tree", {}).get("src_hash") != a.expect_src:
        return cold("tree-mismatch", "in-pod src_hash=%s expected=%s tree=%s"
                    % (payload.get("tree", {}).get("src_hash"), a.expect_src, tree))
    if payload["env_missing"]:
        return cold("env-missing", ",".join(payload["env_missing"][:8]))
    if not src_pinned(a.src):
        return cold("no-src", "src %r is not a pinned populate-job prefix" % a.src)
    ok, n = fresh_state(cache_dir)
    if not ok:
        return cold("cache-dir-not-fresh", "%d files already in %s" % (n, cache_dir))
    be = backend_for(a.src)
    rank = "host%d" % a.host_rank
    lookup_key = key
    if MUTANT == "any_key":
        ks = be.list_keys()
        lookup_key = ks[0] if ks else key
    ready_rel = "%s/%s.READY.json" % (lookup_key, rank)
    try:
        present = be.exists(ready_rel)
    except Exception as exc:  # noqa: BLE001
        return cold("store-error", repr(exc)[:200])
    if not present:
        diffs = []
        for other in be.list_keys()[:4]:
            try:
                pub = json.loads(be.get_bytes("%s/%s.payload.json" % (other, rank)))
                diffs.append("vs %s: %s" % (other, "; ".join(payload_diff(payload_for_hash(payload), pub, 6))))
            except Exception:  # noqa: BLE001
                continue
        return cold("miss", "no archive for this key under %s%s" % (a.src, (" | " + " | ".join(diffs)) if diffs else ""))
    try:
        ready = json.loads(be.get_bytes(ready_rel))
    except Exception as exc:  # noqa: BLE001
        return cold("ready-unreadable", repr(exc)[:200])
    if MUTANT != "any_key" and (ready.get("key") != key or ready.get("payload_sha256") != psha):
        return cold("payload-mismatch", "READY key=%s payload=%s" % (ready.get("key"), ready.get("payload_sha256")))
    max_bytes = int(float(os.environ.get("Q38CC_MAX_UNPACKED_GB", "400")) * (1 << 30))
    if int(ready.get("unpacked_bytes", 1 << 62)) > max_bytes:
        return cold("too-big", "unpacked_bytes=%s" % ready.get("unpacked_bytes"))
    arc_rel = "%s/%s.%s" % (lookup_key, rank, ready.get("ext", "tar.gz"))
    proc = None
    dec = None
    try:
        raw, proc = be.open_stream(arc_rel)
        hr = HashingReader(raw)
        dcmd = decompress_cmd(ready.get("ext", "tar.gz"))
        status["decompress"] = "zstd" if (dcmd and dcmd[0] == "zstd") else ("pgunzip" if dcmd else "stdlib-gzip")
        if dcmd:
            dec = subprocess.Popen(dcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            import threading

            def pump():
                try:
                    while True:
                        b = hr.read(1 << 20)
                        if not b:
                            break
                        dec.stdin.write(b)
                finally:
                    dec.stdin.close()
            th = threading.Thread(target=pump, daemon=True)
            th.start()
            with tarfile.open(fileobj=dec.stdout, mode="r|") as tf:
                nfiles = safe_extract(tf, cache_dir)
            th.join()
            if dec.wait() != 0:
                raise IOError("%s rc=%s" % (" ".join(dcmd[-2:]), dec.returncode))
        else:
            with tarfile.open(fileobj=io.BufferedReader(hr, 1 << 20), mode="r|gz") as tf:
                nfiles = safe_extract(tf, cache_dir)
        while hr.read(1 << 20):
            pass
        if proc is not None and proc.wait() != 0:
            raise IOError("stream_get rc=%s" % proc.returncode)
        raw.close()
        arc_sha = hr.h.hexdigest()
        if MUTANT != "skip_sha" and arc_sha != ready.get("archive_sha256"):
            raise ValueError("archive sha256 %s != READY %s" % (arc_sha[:16], str(ready.get("archive_sha256"))[:16]))
        if MUTANT != "skip_manifest":
            with open(os.path.join(cache_dir, MANIFEST_NAME), encoding="utf-8") as f:
                man = json.load(f)
            if MUTANT != "any_key" and (man.get("key") != key or man.get("payload_sha256") != psha):
                raise ValueError("manifest key/payload differs from the car's")
            if man.get("file_count") != ready.get("file_count"):
                raise ValueError("manifest file_count %s != READY %s" % (man.get("file_count"), ready.get("file_count")))
            actual = build_manifest(cache_dir, key, psha)["files"]
            if actual != man.get("files"):
                miss = sorted(set(man.get("files", {})) - set(actual))[:3]
                extra = sorted(set(actual) - set(man.get("files", {})))[:3]
                chg = sorted(k for k in set(actual) & set(man.get("files", {})) if actual[k] != man["files"][k])[:3]
                raise ValueError("manifest mismatch missing=%s extra=%s changed=%s" % (miss, extra, chg))
    except Exception as exc:  # noqa: BLE001
        for p in (proc, dec):
            if p is not None and p.poll() is None:
                p.kill()
        wipe_to_skeleton(cache_dir, skeleton)
        return cold("verify-failed", repr(exc)[:240])
    status.update(result="HIT", reason="", files=nfiles, archive_sha256=arc_sha, pull_s=round(time.time() - t0, 1),
                  src=a.src)
    log("HIT key=%s files=%d unpacked_bytes=%s pull_s=%.1f decompress=%s src=%s" % (
        key, nfiles, ready.get("unpacked_bytes"), time.time() - t0, status.get("decompress"), a.src))
    _write_status(a.status, status)
    return 0


def cache_quiet(cache_dir: str, quiet_s: float, max_s: float) -> bool:
    t_end = time.time() + max_s
    last = snapshot(cache_dir)
    quiet_since = time.time()
    while True:
        if time.time() - quiet_since >= quiet_s:
            return True
        if time.time() > t_end:
            return False
        time.sleep(min(5.0, max(0.05, quiet_s / 10)))
        cur = snapshot(cache_dir)
        if cur != last:
            last, quiet_since = cur, time.time()


def mem_headroom_gib() -> float:
    try:
        cur = int(open("/sys/fs/cgroup/memory.current").read())
        mx = open("/sys/fs/cgroup/memory.max").read().strip()
        if mx.isdigit():
            return (int(mx) - cur) / float(1 << 30)
    except (OSError, ValueError):
        pass
    return float("inf")


def cmd_publish(a) -> int:
    if MUTANT != "publish_any_mode" and (a.mode != "populate" or os.environ.get("Q38CC_MODE") != "populate"):
        log("REFUSED publish: only a populate car may write (mode=%s Q38CC_MODE=%s)" % (a.mode,
                                                                                         os.environ.get("Q38CC_MODE")))
        return 3
    payload, _tree = _inputs_from_args(a)
    key = key_of(payload)
    psha = sha256_hex(canon(payload_for_hash(payload)))
    log(keyline(payload, "populate"))
    if not cache_quiet(a.cache_dir, a.quiet_s, a.quiet_max_s):
        log("PUBLISH SKIPPED key=%s: CACHE_DIR still changing after %ss" % (key, a.quiet_max_s))
        return 4
    head = mem_headroom_gib()
    if head < a.min_headroom_gib:
        log("PUBLISH SKIPPED key=%s: cgroup headroom %.1f GiB < %.1f GiB" % (key, head, a.min_headroom_gib))
        return 5
    be = backend_for(a.dst)
    rank = "host%d" % a.host_rank
    comp_name = compressor()
    ext = "tar.zst" if comp_name == "zstd" else "tar.gz"
    if be.exists("%s/%s.READY.json" % (key, rank)):
        log("PUBLISH SKIPPED key=%s: already published under %s" % (key, a.dst))
        return 0
    man = build_manifest(a.cache_dir, key, psha)
    pre = snapshot(a.cache_dir)
    mpath = os.path.join(a.cache_dir, MANIFEST_NAME)
    with open(mpath + ".%d.tmp" % os.getpid(), "w", encoding="utf-8") as f:
        json.dump(man, f, sort_keys=True, separators=(",", ":"))
    os.replace(mpath + ".%d.tmp" % os.getpid(), mpath)
    t0 = time.time()
    sink, proc = be.put_stream_once("%s/%s.%s" % (key, rank, ext))
    if sink is None:
        log("PUBLISH SKIPPED key=%s: archive object already exists (create-once)" % key)
        return 0
    hw = HashingWriter(sink)
    try:
        ccmd = compress_cmd(comp_name)
        if ccmd:
            comp = subprocess.Popen(ccmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            import threading

            def drain():
                while True:
                    b = comp.stdout.read(1 << 20)
                    if not b:
                        break
                    hw.write(b)
            th = threading.Thread(target=drain, daemon=True)
            th.start()
            with tarfile.open(fileobj=comp.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tf:
                for rel in sorted(man["files"]) + [MANIFEST_NAME]:
                    tf.add(os.path.join(a.cache_dir, rel), arcname=rel, recursive=False)
            comp.stdin.close()
            th.join()
            if comp.wait() != 0:
                raise IOError("%s rc=%s" % (comp_name, comp.returncode))
        else:
            import gzip
            gz = gzip.GzipFile(fileobj=hw, mode="wb", compresslevel=1, mtime=0)
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tf:
                for rel in sorted(man["files"]) + [MANIFEST_NAME]:
                    tf.add(os.path.join(a.cache_dir, rel), arcname=rel, recursive=False)
            gz.close()
        sink.close()
        if proc is not None:
            rc = proc.wait()
            if rc != 0:
                raise IOError("stream_put rc=%s" % rc)
    except Exception as exc:  # noqa: BLE001
        log("PUBLISH FAILED key=%s: %r (no READY written; pulls of this key will MISS)" % (key, exc))
        return 6
    if snapshot(a.cache_dir) != pre:
        log("PUBLISH FAILED key=%s: CACHE_DIR changed during the pack (no READY written)" % key)
        return 7
    pub = payload_for_hash(payload)
    be.put_bytes_once("%s/%s.payload.json" % (key, rank), canon(pub))
    ready = {"schema": TOOL_VERSION, "key": key, "payload_sha256": psha, "archive_sha256": hw.h.hexdigest(),
             "archive_bytes": hw.n, "file_count": man["file_count"], "unpacked_bytes": man["unpacked_bytes"], "ext": ext,
             "compressor": comp_name,
             "runtime": payload.get("runtime"), "job_id": os.environ.get("CDK_JOB_ID", ""),
             "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rc = be.put_bytes_once("%s/%s.READY.json" % (key, rank), canon(ready))
    if rc != 0:
        log("PUBLISH FAILED key=%s: READY create-once rc=%s" % (key, rc))
        return 8
    log("PUBLISHED key=%s files=%d unpacked_bytes=%d archive_bytes=%d archive_sha256=%s publish_s=%.1f comp=%s dst=%s"
        % (key, man["file_count"], man["unpacked_bytes"], hw.n, hw.h.hexdigest(), time.time() - t0, comp_name, a.dst))
    return 0


def cmd_key(a) -> int:
    with open(a.render, encoding="utf-8") as f:
        legs = render_legs(f.read())
    leg = legs[a.role]
    runtime = load_runtime(a.runtime)
    payload = compute_payload(a.role, leg["env"], list(leg["env"]), leg["args"], leg["image"], a.tree, runtime)
    if a.payload_out:
        with open(a.payload_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
    print(key_of(payload))
    log(keyline(payload, "offline"))
    return 0


def cmd_keyline(a) -> int:
    try:
        payload, _ = _inputs_from_args(a)
        log(keyline(payload, "keyonly"))
    except Exception as exc:  # noqa: BLE001
        log("WARNING keyline failed: %r" % exc)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="q38cc.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("key")
    k.add_argument("--render", required=True)
    k.add_argument("--role", required=True)
    k.add_argument("--tree", required=True)
    k.add_argument("--runtime", required=True)
    k.add_argument("--payload-out", default="")
    for name in ("pull", "publish", "keyline"):
        p = sub.add_parser(name)
        p.add_argument("--role", required=True)
        p.add_argument("--image", required=True)
        p.add_argument("--env-names", required=True)
        p.add_argument("--args", required=True)
        p.add_argument("--tree", default="auto")
        p.add_argument("--runtime", default="auto")
        p.add_argument("--host-rank", type=int, default=0)
        p.add_argument("--check-import", action="store_true")
        p.add_argument("--tool-sha", default="")
        if name in ("pull", "publish"):
            p.add_argument("--cache-dir", required=True)
        if name == "pull":
            p.add_argument("--src", default="")
            p.add_argument("--expect-src", default="")
            p.add_argument("--status", default="")
        if name == "publish":
            p.add_argument("--dst", required=True)
            p.add_argument("--mode", default="")
            p.add_argument("--quiet-s", type=float, default=120.0)
            p.add_argument("--quiet-max-s", type=float, default=900.0)
            p.add_argument("--min-headroom-gib", type=float, default=25.0)
    a = ap.parse_args(argv)
    return {"key": cmd_key, "pull": cmd_pull, "publish": cmd_publish, "keyline": cmd_keyline}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
