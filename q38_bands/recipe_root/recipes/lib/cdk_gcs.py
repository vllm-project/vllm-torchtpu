#!/usr/bin/env python3
"""Dependency-free GCS helper using GCE Workload Identity + curl subprocesses.

Works inside lean server images (e.g. torchtpu-vllm-prod) that do not ship
google-cloud-sdk (gcloud/gsutil) or google-cloud-storage Python packages.
"""

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

_TOKEN_LOCK = threading.Lock()
_TOKEN_CACHE = {"token": "", "exp": 0.0}


def get_token() -> str:
    with _TOKEN_LOCK:
        now = time.time()
        if _TOKEN_CACHE["token"] and now < _TOKEN_CACHE["exp"] - 120:
            return _TOKEN_CACHE["token"]
        last_err = None
        for attempt in range(6):
            try:
                req = urllib.request.Request(
                    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                    headers={"Metadata-Flavor": "Google"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                _TOKEN_CACHE["token"] = data["access_token"]
                _TOKEN_CACHE["exp"] = now + float(data.get("expires_in", 1800))
                return _TOKEN_CACHE["token"]
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Failed to fetch GCE metadata token: {last_err}")


def parse_gs(uri: str):
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    rest = uri[5:]
    bucket, _, obj = rest.partition("/")
    return bucket, obj


def api_get_json(url: str):
    last_err = None
    for attempt in range(5):
        tok = get_token()
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return 200, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return 404, None
            if err.code == 401:
                with _TOKEN_LOCK:
                    _TOKEN_CACHE["exp"] = 0.0
            last_err = err
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GCS GET {url} failed: {last_err}")


def cmd_exists(uri: str) -> int:
    bucket, obj = parse_gs(uri)
    obj = obj.rstrip("/")
    enc = urllib.parse.quote(obj, safe="")
    code, _ = api_get_json(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{enc}")
    return 0 if code == 200 else 1


def cmd_ls_subdirs(uri: str) -> int:
    bucket, prefix = parse_gs(uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    enc = urllib.parse.quote(prefix, safe="")
    code, data = api_get_json(
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={enc}&delimiter=/"
    )
    if code != 200 or not data:
        return 1
    for p in data.get("prefixes", []):
        print(f"gs://{bucket}/{p.rstrip('/')}")
    return 0


def curl_download(bucket: str, obj: str, dst: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = f"{dst}.part.{os.getpid()}.{threading.get_ident()}"
    enc = urllib.parse.quote(obj, safe="")
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{enc}?alt=media"
    for attempt in range(4):
        tok = get_token()
        res = subprocess.run(
            [
                "curl",
                "-fsSL",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "-H",
                f"Authorization: Bearer {tok}",
                "-o",
                tmp,
                url,
            ],
            check=False,
        )
        if res.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, dst)
            return 0
        try:
            os.remove(tmp)
        except OSError:
            pass
        time.sleep(1.5 * (attempt + 1))
    return 1


def api_post_json(url: str, if_gen_zero: bool = False):
    for attempt in range(4):
        tok = get_token()
        req = urllib.request.Request(
            url,
            data=b"{}",
            method="POST",
            headers={
                "Authorization": f"Bearer {tok}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return 200, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if if_gen_zero and err.code == 412:
                return 412, None
            if err.code == 404:
                return 404, None
            if err.code == 401:
                with _TOKEN_LOCK:
                    _TOKEN_CACHE["exp"] = 0.0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.5 * (attempt + 1))
    return 500, None


def _gcs_rewrite(src: str, dst: str, if_gen_zero: bool = False) -> int:
    """Server-side GCS-to-GCS copy via rewriteTo API (zero pod download/upload bandwidth)."""
    src_b, src_o = parse_gs(src)
    dst_b, dst_o = parse_gs(dst)
    src_enc = urllib.parse.quote(src_o.rstrip("/"), safe="")
    dst_enc = urllib.parse.quote(dst_o.rstrip("/"), safe="")
    base_url = (
        f"https://storage.googleapis.com/storage/v1/b/{src_b}/o/{src_enc}"
        f"/rewriteTo/b/{dst_b}/o/{dst_enc}"
    )
    rewrite_token = ""
    while True:
        params = []
        if if_gen_zero:
            params.append("ifGenerationMatch=0")
        if rewrite_token:
            params.append(f"rewriteToken={urllib.parse.quote(rewrite_token, safe='')}")
        url = f"{base_url}?{'&'.join(params)}" if params else base_url
        code, data = api_post_json(url, if_gen_zero=if_gen_zero)
        if if_gen_zero and code == 412:
            return 2
        if code != 200 or not data:
            return 1
        if data.get("done"):
            return 0
        rewrite_token = data.get("rewriteToken", "")
        if not rewrite_token:
            return 1


def cmd_cp(src: str, dst: str, if_gen_zero: bool = False) -> int:
    if src.startswith("gs://") and dst.startswith("gs://"):
        return _gcs_rewrite(src, dst, if_gen_zero=if_gen_zero)
    if src.startswith("gs://") and not dst.startswith("gs://"):
        bucket, obj = parse_gs(src)
        return curl_download(bucket, obj, dst)
    if not src.startswith("gs://") and dst.startswith("gs://"):
        bucket, obj = parse_gs(dst)
        enc = urllib.parse.quote(obj, safe="")
        url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={enc}"
        if if_gen_zero:
            url += "&ifGenerationMatch=0"
        resp_file = f"/tmp/cdk_gcs_up_{os.getpid()}.json"
        for attempt in range(4):
            tok = get_token()
            res = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "-o",
                    resp_file,
                    "-w",
                    "%{http_code}",
                    "-X",
                    "POST",
                    "-H",
                    f"Authorization: Bearer {tok}",
                    "-H",
                    "Content-Type: application/octet-stream",
                    "-T",
                    src,
                    url,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            http_code = (res.stdout or "").strip()
            try:
                os.remove(resp_file)
            except OSError:
                pass
            if http_code in ("200", "201"):
                return 0
            if if_gen_zero and http_code == "412":
                return 2
            time.sleep(1.5 * (attempt + 1))
        return 1
    raise ValueError(f"Unsupported cp direction: {src} -> {dst}")


def cmd_stream_get(
    uri: str,
    workers: int = 16,
    chunk_bytes: int = 32 * 1024 * 1024,
    out_stream=None,
    range_fetcher=None,
    total_size_override: int = -1,
) -> int:
    """Download a GCS object via parallel HTTP Range requests and stream ordered bytes to stdout."""
    from collections import deque
    import http.client

    out = out_stream if out_stream is not None else sys.stdout.buffer
    bucket, obj = parse_gs(uri)
    obj = obj.rstrip("/")
    enc = urllib.parse.quote(obj, safe="")
    if total_size_override >= 0:
        total_size = total_size_override
    else:
        code, meta = api_get_json(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{enc}")
        if code != 200 or not meta:
            return 1
        total_size = int(meta.get("size", 0))
    if total_size <= 0:
        out.flush()
        return 0

    tls_local = threading.local()
    path = f"/storage/v1/b/{bucket}/o/{enc}?alt=media"

    def _default_fetch(start: int, end: int) -> bytes:
        expected = end - start + 1
        last_err = None
        for attempt in range(5):
            conn = getattr(tls_local, "conn", None)
            try:
                if conn is None:
                    conn = http.client.HTTPSConnection("storage.googleapis.com", timeout=45)
                    tls_local.conn = conn
                tok = get_token()
                conn.request(
                    "GET",
                    path,
                    headers={
                        "Authorization": f"Bearer {tok}",
                        "Range": f"bytes={start}-{end}",
                        "Connection": "keep-alive",
                    },
                )
                resp = conn.getresponse()
                data = resp.read()
                if resp.status in (200, 206) and len(data) == expected:
                    return data
                if resp.status == 401:
                    with _TOKEN_LOCK:
                        _TOKEN_CACHE["exp"] = 0.0
                last_err = RuntimeError(f"HTTP {resp.status} len={len(data)}/{expected}")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            try:
                if getattr(tls_local, "conn", None) is not None:
                    tls_local.conn.close()
            except Exception:  # noqa: BLE001
                pass
            tls_local.conn = None
            time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Range GET {uri} [{start}-{end}] failed: {last_err}")

    fetch_fn = range_fetcher if range_fetcher is not None else _default_fetch
    ranges = [
        (off, min(off + chunk_bytes, total_size) - 1)
        for off in range(0, total_size, chunk_bytes)
    ]
    workers = max(1, min(workers, len(ranges)))
    max_inflight = min(len(ranges), max(4, min(24, workers + 4)))
    q = deque()
    next_idx = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while next_idx < len(ranges) and len(q) < max_inflight:
            s, e = ranges[next_idx]
            q.append(pool.submit(fetch_fn, s, e))
            next_idx += 1
        while q:
            chunk = q.popleft().result()
            if next_idx < len(ranges):
                s, e = ranges[next_idx]
                q.append(pool.submit(fetch_fn, s, e))
                next_idx += 1
            out.write(chunk)
    out.flush()
    return 0


def cmd_stream_put(dst: str, if_gen_zero: bool = False) -> int:
    """Stream stdin directly to GCS uploadType=media via chunked curl -T - without /tmp staging."""
    if not dst.startswith("gs://"):
        raise ValueError(f"Expected gs:// destination, got {dst!r}")
    bucket, obj = parse_gs(dst)
    enc = urllib.parse.quote(obj, safe="")
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={enc}"
    if if_gen_zero:
        url += "&ifGenerationMatch=0"
    resp_file = f"/tmp/cdk_gcs_sup_{os.getpid()}.json"
    tok = get_token()
    res = subprocess.run(
        [
            "curl",
            "-sS",
            "--connect-timeout",
            "20",
            "-o",
            resp_file,
            "-w",
            "%{http_code}",
            "-X",
            "POST",
            "-H",
            f"Authorization: Bearer {tok}",
            "-H",
            "Content-Type: application/octet-stream",
            "-H",
            "Transfer-Encoding: chunked",
            "-T",
            "-",
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    http_code = (res.stdout or "").strip()
    try:
        os.remove(resp_file)
    except OSError:
        pass
    if http_code in ("200", "201"):
        return 0
    if if_gen_zero and http_code == "412":
        return 2
    if res.stderr:
        sys.stderr.write(res.stderr)
    return 1


def _compute_prefetch_budget_bytes(
    reserve_gib: float = 180.0,
    meminfo_path: str = "/proc/meminfo",
    cgroup_max_path: str = "/sys/fs/cgroup/memory.max",
    cgroup_stat_path: str = "/sys/fs/cgroup/memory.stat",
) -> int:
    gib = 1024 ** 3
    offload_gib = 0.0
    kv_off = os.environ.get("KV_OFFLOAD", "").strip().lower() in ("1", "true", "yes", "on")
    pd_off = os.environ.get("OFFLOAD_MODE", "").strip().lower() in ("1", "true", "yes", "on")
    if kv_off or pd_off:
        for k in ("KV_OFFLOAD_GB", "KV_CPU_SPACE_GB", "DRAM_POOL_GB", "RAIDEN_DRAM_GIB"):
            raw = os.environ.get(k, "").strip()
            if raw:
                try:
                    offload_gib = max(offload_gib, float(raw))
                except ValueError:
                    pass
        if offload_gib <= 0.0:
            offload_gib = 128.0
    for k_bytes in ("TPU_RAIDEN_MAX_CPU_BYTES", "TPU_RAIDEN_PP_MAX_CPU_BYTES"):
        raw_b = os.environ.get(k_bytes, "").strip()
        if raw_b:
            try:
                b_val = float(raw_b)
                if b_val > 0:
                    offload_gib = max(offload_gib, b_val / gib)
            except ValueError:
                pass
    reserve_bytes = int((reserve_gib + offload_gib) * gib)

    usable = None
    if os.path.isfile(meminfo_path):
        try:
            with open(meminfo_path, "r", encoding="utf-8", errors="replace") as f:
                for ln in f:
                    if ln.startswith("MemAvailable:"):
                        usable = int(ln.split()[1]) * 1024
                        break
        except (OSError, ValueError, IndexError):
            pass

    if os.path.isfile(cgroup_max_path):
        try:
            raw_max = open(cgroup_max_path, "r", encoding="utf-8").read().strip()
            if raw_max.isdigit():
                cg_max = int(raw_max)
                non_reclaim = 0
                if os.path.isfile(cgroup_stat_path):
                    with open(cgroup_stat_path, "r", encoding="utf-8", errors="replace") as sf:
                        for ln in sf:
                            parts = ln.split()
                            if len(parts) == 2 and parts[0] in ("anon", "shmem"):
                                non_reclaim += int(parts[1])
                cg_usable = max(0, cg_max - non_reclaim)
                usable = cg_usable if usable is None else min(usable, cg_usable)
        except (OSError, ValueError):
            pass

    if usable is None:
        return 0
    return max(0, usable - reserve_bytes)


def cmd_prefetch_weights(
    model_dir: str,
    workers: int = 8,
    reserve_gib: float = 180.0,
    budget_override: int = -1,
) -> int:
    """Warm Linux OS page cache for *.safetensors shards in natural order within safe RAM headroom."""
    real_dir = os.path.realpath(model_dir)
    if not os.path.isdir(real_dir):
        return 0
    evict_pc = os.environ.get("TPU_EVICT_WEIGHTS_PAGE_CACHE", "").strip().lower() in ("1", "true", "yes", "on")
    st_strat = os.environ.get("SAFETENSORS_LOAD_STRATEGY", "").strip().lower()
    if budget_override < 0 and (evict_pc or st_strat == "prefetch"):
        print(
            f"[WeightPrefetch] Skipped for {real_dir} (vLLM page-cache eviction or safetensors prefetch active)",
            flush=True,
        )
        return 0
    budget = budget_override if budget_override >= 0 else _compute_prefetch_budget_bytes(reserve_gib=reserve_gib)
    if budget <= 0:
        print(f"[WeightPrefetch] Skipped for {real_dir} (available RAM within reserve headroom)", flush=True)
        return 0

    shards = []
    for cur, dirs, files in os.walk(real_dir):
        dirs.sort()
        for fname in sorted(files):
            if not fname.endswith(".safetensors"):
                continue
            fpath = os.path.join(cur, fname)
            try:
                sz = os.path.getsize(fpath)
            except OSError:
                continue
            if sz > 0:
                shards.append((os.path.relpath(fpath, real_dir), fpath, sz))
    import re as _re

    shards.sort(key=lambda x: [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", x[0])])
    if not shards:
        return 0

    selected = []
    acc = 0
    for _, fpath, sz in shards:
        if acc + sz > budget:
            break
        selected.append((fpath, sz))
        acc += sz
    if not selected:
        return 0

    try:
        os.nice(15)
    except OSError:
        pass

    t0 = time.monotonic()
    fadvise = getattr(os, "posix_fadvise", None)
    willneed = getattr(os, "POSIX_FADV_WILLNEED", 3)

    def _read_shard(item):
        fpath, sz = item
        buf = bytearray(8 * 1024 * 1024)
        try:
            with open(fpath, "rb", buffering=0) as f:
                if fadvise is not None:
                    try:
                        fadvise(f.fileno(), 0, sz, willneed)
                    except OSError:
                        pass
                while f.readinto(buf):
                    pass
        except OSError:
            pass

    workers = max(1, min(workers, len(selected)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(_read_shard, selected):
            pass
    dt = max(0.001, time.monotonic() - t0)
    gb = acc / (1024 ** 3)
    print(
        f"[WeightPrefetch] Warmed {len(selected)}/{len(shards)} safetensors ({gb:.1f} GiB) into OS page cache in {dt:.1f}s ({gb / dt:.2f} GiB/s)",
        flush=True,
    )
    return 0


def list_objects(bucket: str, prefix: str):
    items = []
    page_token = ""
    while True:
        enc = urllib.parse.quote(prefix, safe="")
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={enc}&maxResults=1000"
        if page_token:
            url += f"&pageToken={urllib.parse.quote(page_token, safe='')}"
        code, data = api_get_json(url)
        if code != 200 or not data:
            break
        items.extend(data.get("items", []))
        page_token = data.get("nextPageToken", "")
        if not page_token:
            break
    return items


def cmd_sync_dir(uri: str, local_dir: str, workers: int = 24, match_glob: str = "") -> int:
    import fnmatch

    bucket, prefix = parse_gs(uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    items = list_objects(bucket, prefix)
    tasks = []
    for it in items:
        name = it.get("name", "")
        if not name or name.endswith("/"):
            continue
        rel = name[len(prefix) :] if name.startswith(prefix) else os.path.basename(name)
        if not rel or rel.startswith(".cache/") or "/.cache/" in rel:
            continue
        if match_glob and not fnmatch.fnmatch(os.path.basename(rel), match_glob):
            continue
        dst = os.path.join(local_dir, rel)
        sz = int(it.get("size", -1))
        if sz >= 0 and os.path.exists(dst) and os.path.getsize(dst) == sz:
            continue
        tasks.append((name, dst, sz))

    if not tasks:
        return 0

    def _worker(t):
        obj_name, dst_path, _ = t
        if curl_download(bucket, obj_name, dst_path) != 0:
            raise RuntimeError(f"Failed downloading gs://{bucket}/{obj_name}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_worker, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
    return 0


def _gzip_member(chunk: bytes, level: int = 1) -> bytes:
    import zlib

    c = zlib.compressobj(level, zlib.DEFLATED, 31)
    return c.compress(chunk) + c.flush(zlib.Z_FINISH)


def cmd_pgzip(workers: int = 24, level: int = 1, chunk_bytes: int = 8 * 1024 * 1024) -> int:
    from collections import deque

    inp = sys.stdin.buffer
    out = sys.stdout.buffer
    max_inflight = max(4, workers * 2)
    q = deque()
    wrote_any = False

    def _read_exact(n: int) -> bytes:
        parts = []
        rem = n
        while rem > 0:
            b = inp.read(rem)
            if not b:
                break
            parts.append(b)
            rem -= len(b)
        return b"".join(parts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            buf = _read_exact(chunk_bytes)
            if buf:
                q.append(pool.submit(_gzip_member, buf, level))
                wrote_any = True
            while len(q) >= max_inflight or (not buf and q):
                out.write(q.popleft().result())
            if not buf:
                break
    if not wrote_any:
        out.write(_gzip_member(b"", level))
    out.flush()
    return 0


def _try_gunzip_exact(raw: bytes):
    import zlib

    d = zlib.decompressobj(31)
    try:
        out = d.decompress(raw)
    except zlib.error:
        return None
    if not d.eof or d.unused_data:
        return None
    return out + d.flush()


def cmd_pgunzip(
    workers: int = 16,
    inp_stream=None,
    out_stream=None,
    max_member_bytes: int = 24 * 1024 * 1024,
) -> int:
    import zlib
    from collections import deque

    inp = inp_stream if inp_stream is not None else sys.stdin.buffer
    out = out_stream if out_stream is not None else sys.stdout.buffer
    hdr = b"\x1f\x8b\x08\x00\x00\x00\x00\x00"
    max_inflight = max(4, workers * 2)
    q = deque()
    buf = bytearray()
    seq_d = None
    saw_any = False

    def _feed_seq(data: bytes) -> None:
        nonlocal seq_d
        mv = memoryview(data)
        while len(mv) > 0:
            if seq_d is None:
                seq_d = zlib.decompressobj(31)
            out.write(seq_d.decompress(mv))
            if seq_d.eof:
                unused = seq_d.unused_data
                out.write(seq_d.flush())
                seq_d = None
                if unused:
                    mv = memoryview(unused)
                    continue
            break

    def _drain_one() -> None:
        raw, fut = q.popleft()
        dec = fut.result()
        if dec is not None and seq_d is None:
            out.write(dec)
        else:
            _feed_seq(raw)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            while True:
                chunk = inp.read(4 * 1024 * 1024)
                if chunk:
                    saw_any = True
                    buf.extend(chunk)
                while True:
                    s_from = 10 if buf[:8] == hdr else 1
                    idx = buf.find(hdr, s_from)
                    if idx < 0 or (chunk and idx + 10 > len(buf)):
                        break
                    raw = bytes(buf[:idx])
                    del buf[:idx]
                    q.append((raw, pool.submit(_try_gunzip_exact, raw)))
                    while len(q) >= max_inflight:
                        _drain_one()
                if chunk and len(buf) > max_member_bytes:
                    while q:
                        _drain_one()
                    keep = min(len(buf), 16)
                    flush_n = len(buf) - keep
                    if flush_n > 0:
                        _feed_seq(bytes(buf[:flush_n]))
                        del buf[:flush_n]
                if not chunk:
                    break
            if buf:
                raw = bytes(buf)
                buf.clear()
                q.append((raw, pool.submit(_try_gunzip_exact, raw)))
            while q:
                _drain_one()
    except zlib.error:
        return 1
    if not saw_any or seq_d is not None:
        return 1
    out.flush()
    return 0


def _hash_file_or_link(h, path: str, max_inline_bytes: int = 64 * 1024 * 1024) -> None:
    try:
        if os.path.islink(path):
            h.update(b"L\0" + os.readlink(path).encode("utf-8", errors="surrogateescape") + b"\0")
            return
        st = os.stat(path)
        if st.st_size > max_inline_bytes:
            h.update(f"S:{st.st_size}:{st.st_mtime_ns}\0".encode("utf-8"))
            return
        h.update(f"F:{st.st_size}\0".encode("utf-8"))
        with open(path, "rb") as f:
            while True:
                chunk = f.read(262144)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        h.update(b"ERR\0")


_SKIP_DIRS = frozenset({
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
})


def _hash_tree(root_dir: str, max_files: int = 20000) -> str:
    import hashlib

    if not root_dir or not os.path.exists(root_dir):
        return ""
    h = hashlib.sha256()
    count = 0
    for cur, dirs, files in os.walk(root_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info"))
        for fname in sorted(files):
            if fname.endswith((".pyc", ".pyo")) or fname.startswith(".tmp"):
                continue
            full = os.path.join(cur, fname)
            rel = os.path.relpath(full, root_dir)
            h.update(b"P\0" + rel.encode("utf-8", errors="surrogateescape") + b"\0")
            _hash_file_or_link(h, full)
            count += 1
            if count >= max_files:
                h.update(b"TRUNC\0")
                return h.hexdigest()[:20]
    return h.hexdigest()[:20] if count > 0 else "empty"


def git_dirty_sha(repo_dir: str) -> str:
    """Compute an offline content SHA-256 of modified, untracked, and deleted files in a git tree.

    Unlike `git diff HEAD`, `git status --porcelain -z` works completely offline on `cdk --map-dir`
    partial clones (`--filter=blob:none`) without fetching base blobs from origin, and includes
    untracked files (`??`).
    """
    import hashlib

    if not repo_dir or not os.path.isdir(repo_dir):
        return ""
    try:
        top_res = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return ""
    if top_res.returncode != 0:
        return ""
    toplevel = top_res.stdout.strip()
    if not toplevel:
        return ""
    try:
        st_res = subprocess.run(
            ["git", "-C", toplevel, "status", "--porcelain", "-z"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return ""
    if st_res.returncode != 0 or not st_res.stdout:
        return ""

    tokens = [t for t in st_res.stdout.split(b"\0") if t]
    if not tokens:
        return ""
    h = hashlib.sha256()
    idx = 0
    while idx < len(tokens):
        entry = tokens[idx]
        xy = entry[:2].decode("utf-8", errors="replace")
        rel_bytes = entry[3:]
        rel = rel_bytes.decode("utf-8", errors="surrogateescape")
        h.update(xy.encode("utf-8") + b"\0" + rel_bytes + b"\0")
        full = os.path.join(toplevel, rel)
        if os.path.isdir(full) and not os.path.islink(full):
            h.update(b"D\0" + _hash_tree(full).encode("utf-8") + b"\0")
        elif os.path.lexists(full):
            _hash_file_or_link(h, full)
        else:
            h.update(b"MISSING\0")
        if xy and xy[0] in ("R", "C") and idx + 1 < len(tokens):
            idx += 1
            h.update(b"ORIG\0" + tokens[idx] + b"\0")
        idx += 1
    return h.hexdigest()[:16]


def _decode_mountinfo_path(raw: str) -> str:
    return (
        raw.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _discover_external_mounts(mountinfo_path: str = "/proc/self/mountinfo"):
    map_dirs = []
    nfs_dirs = []
    if os.path.isfile(mountinfo_path):
        try:
            with open(mountinfo_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7 or "-" not in parts:
                        continue
                    sep = parts.index("-")
                    root = _decode_mountinfo_path(parts[3])
                    mnt = _decode_mountinfo_path(parts[4])
                    fstype = parts[sep + 1] if sep + 1 < len(parts) else ""
                    if "cdk-map-" in root:
                        map_dirs.append(mnt)
                    elif fstype in ("nfs", "nfs4") and mnt != "/root/wheels":
                        nfs_dirs.append(mnt)
        except OSError:
            pass
    for item in os.environ.get("EXOS_MAP_DIRS", "").split(";"):
        if ":" in item:
            _, _, target = item.rpartition(":")
            if target.strip():
                map_dirs.append(target.strip())
    for extra in os.environ.get("CDK_EXTRA_HASH_DIRS", "").split(":"):
        if extra.strip():
            map_dirs.append(extra.strip())
    return sorted(set(map_dirs)), sorted(set(nfs_dirs))


def compute_overlay_sha(mountinfo_path: str = "/proc/self/mountinfo") -> str:
    """Return a SHA-256 digest of any active --map-dir, dirty git checkout, or wheel/spec override.

    Returns "" on unmodified container image runs so clean warm cache keys remain unchanged.
    """
    import hashlib
    import importlib.util
    import pathlib

    overlay = {}
    map_dirs, nfs_dirs = _discover_external_mounts(mountinfo_path=mountinfo_path)
    for mdir in map_dirs:
        if os.path.exists(mdir):
            overlay[f"map:{mdir}"] = _hash_tree(mdir)

    candidate_roots = []
    for pkg in ("vllm_torchtpu", "vllm"):
        try:
            spec = importlib.util.find_spec(pkg)
            if spec and spec.origin:
                candidate_roots.append(str(pathlib.Path(spec.origin).resolve().parent))
        except Exception:  # noqa: BLE001
            pass
    for fallback in ("/workspace/vllm-torchtpu", "/root/cloud-devkit/vllm-torchtpu", "/root/torchtpu-vllm"):
        if os.path.isdir(fallback):
            candidate_roots.append(fallback)
    for py_entry in os.environ.get("PYTHONPATH", "").split(":"):
        p = py_entry.strip()
        if not p or p in ("/tmp/cdk_sitecustomize", "/tmp/cc-site") or not os.path.isdir(p):
            continue
        candidate_roots.append(p)
        if any(p == nd or p.startswith(nd.rstrip("/") + "/") for nd in nfs_dirs):
            overlay[f"nfs_py:{p}"] = _hash_tree(p)

    seen_toplevels = set()
    for root in candidate_roots:
        try:
            res = subprocess.run(
                ["git", "-C", root, "rev-parse", "--show-toplevel", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except OSError:
            continue
        if res.returncode != 0:
            continue
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        if not lines:
            continue
        top = lines[0]
        head = lines[1] if len(lines) > 1 else ""
        if top in seen_toplevels:
            continue
        seen_toplevels.add(top)
        dirty = git_dirty_sha(top)
        is_nfs = any(top == nd or top.startswith(nd.rstrip("/") + "/") for nd in nfs_dirs)
        if dirty or is_nfs:
            overlay[f"git:{top}"] = f"{head}:{dirty}"

    for env_key in ("RAIDEN_WHEEL", "LIBTPU_VERSION", "VLLM_SPEC", "MINI_LAYERS"):
        val = os.environ.get(env_key, "").strip()
        if val:
            overlay[f"env:{env_key}"] = val

    prov_dirty = os.environ.get("PROV_DIRTY_SHA", "").strip()
    if prov_dirty and prov_dirty.lower() not in ("none", "clean"):
        overlay["env:PROV_DIRTY_SHA"] = prov_dirty
    prov_pending = os.environ.get("PROV_PENDING", "").strip()
    if prov_pending and prov_pending not in ("0", "false", "no"):
        overlay["env:PROV_PENDING"] = prov_pending

    if not overlay:
        return ""
    return hashlib.sha256(json.dumps(overlay, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def compute_installed_pkg_fingerprint(
    module_names: tuple = ("vllm_torchtpu", "vllm", "tpu_sync_torch"),
    dist_names: tuple = (
        "vllm",
        "vllm-torchtpu",
        "torch",
        "torch-tpu",
        "torch_xla",
        "jax",
        "jaxlib",
        "libtpu",
        "tpu-sync-torch",
    ),
    search_paths=None,
) -> str:
    """Fingerprint installed vLLM/TorchTPU Python sources, .so extensions, and wheel versions."""
    import hashlib
    import importlib.machinery
    import importlib.metadata
    import importlib.util
    import pathlib

    h = hashlib.sha256()
    found = False
    if search_paths is None:
        for dist in dist_names:
            try:
                ver = importlib.metadata.version(dist)
            except Exception:
                continue
            h.update(f"dist:{dist}={ver}\n".encode("utf-8"))
            found = True
    seen_roots = set()
    for mod in module_names:
        try:
            spec = (
                importlib.machinery.PathFinder.find_spec(mod, list(search_paths))
                if search_paths is not None
                else importlib.util.find_spec(mod)
            )
        except Exception:
            spec = None
        origin = getattr(spec, "origin", None) if spec else None
        if not origin:
            continue
        origin_path = pathlib.Path(origin).resolve()
        if origin_path.is_file() and (origin_path.name.endswith(".so") or ".so." in origin_path.name):
            try:
                st = origin_path.stat()
                h.update(f"so:{mod}/{origin_path.name}:{st.st_size}\0".encode("utf-8"))
                with origin_path.open("rb") as fp:
                    if st.st_size <= 16 * 1024 * 1024:
                        h.update(fp.read())
                    else:
                        h.update(fp.read(1024 * 1024))
                        fp.seek(max(0, st.st_size - 1024 * 1024))
                        h.update(fp.read(1024 * 1024))
                found = True
            except OSError:
                pass
            continue
        pkg_root = origin_path.parent
        if not pkg_root.is_dir() or str(pkg_root) in seen_roots:
            continue
        seen_roots.add(str(pkg_root))
        root_str = str(pkg_root)
        for cur_dir, dirs, files in os.walk(root_str):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith(".git"))
            for fname in sorted(files):
                is_py = fname.endswith(".py")
                is_so = (not is_py) and (fname.endswith(".so") or ".so." in fname)
                if not (is_py or is_so):
                    continue
                fpath = os.path.join(cur_dir, fname)
                rel = os.path.relpath(fpath, root_str).replace(os.sep, "/")
                try:
                    if is_py:
                        with open(fpath, "rb") as fp:
                            data = fp.read()
                        h.update(f"py:{mod}/{rel}\0".encode("utf-8"))
                        h.update(data)
                        found = True
                    else:
                        st = os.stat(fpath)
                        h.update(f"so:{mod}/{rel}:{st.st_size}\0".encode("utf-8"))
                        with open(fpath, "rb") as fp:
                            if st.st_size <= 16 * 1024 * 1024:
                                h.update(fp.read())
                            else:
                                h.update(fp.read(1024 * 1024))
                                fp.seek(max(0, st.st_size - 1024 * 1024))
                                h.update(fp.read(1024 * 1024))
                        found = True
                except OSError:
                    continue
    return h.hexdigest()[:20] if found else ""


def compute_compile_cache_key(
    recipe: str,
    image: str,
    role: str,
    key_spec_b64: str,
    mountinfo_path: str = "/proc/self/mountinfo",
    pkg_search_paths=None,
) -> str:
    import base64
    import hashlib

    prov_commit = os.environ.get("PROV_COMMIT", "")
    payload = {
        "schema": "v2",
        "recipe": recipe,
        "image": image,
        "role": role,
        "key_spec": base64.b64decode(key_spec_b64).decode("utf-8"),
        "model": os.environ.get("MODEL_NAME", ""),
        "prov_commit": prov_commit,
        "prov_dirty_sha": os.environ.get("PROV_DIRTY_SHA", ""),
        "vllm_ver": os.environ.get("PROV_VLLM", ""),
        "libtpu_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "tag": os.environ.get("COMPILE_CACHE_TAG", "v1"),
    }
    # Always fingerprint installed Python sources, .so extensions, and wheel versions so
    # re-pushed mutable tags (even non-:latest developer tags) never serve stale caches.
    pkg_fp = compute_installed_pkg_fingerprint(search_paths=pkg_search_paths)
    if pkg_fp:
        payload["pkg_fp"] = pkg_fp
    overlay_sha = compute_overlay_sha(mountinfo_path=mountinfo_path)
    if overlay_sha:
        payload["overlay_sha"] = overlay_sha
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return f"{role}-{digest}"


CACHE_MANIFEST_NAME = ".cdk_cache_manifest.json"
_CACHE_EXCLUDED_TOP_DIRS = frozenset({"tmp", "tpu_logs"})


class _CacheMutatedError(RuntimeError):
    """Raised when a file inside CACHE_DIR changes size/mtime/inode while being archived."""


def _is_excluded_cache_file(fname: str) -> bool:
    return (
        fname == CACHE_MANIFEST_NAME
        or ".tmp" in fname
        or fname.endswith(".lock")
    )


def _clean_ephemeral_cache_files(cache_dir: str) -> None:
    import stat

    for cur_dir, dirs, files in os.walk(cache_dir):
        if os.path.abspath(cur_dir) == os.path.abspath(cache_dir):
            dirs[:] = [d for d in dirs if d not in _CACHE_EXCLUDED_TOP_DIRS]
        for fname in files:
            if fname == CACHE_MANIFEST_NAME:
                continue
            fpath = os.path.join(cur_dir, fname)
            try:
                st = os.lstat(fpath)
                if ".tmp" in fname or fname.endswith(".lock") or (stat.S_ISREG(st.st_mode) and st.st_size == 0):
                    os.unlink(fpath)
            except OSError:
                pass


def _scan_cache_tree(cache_dir: str, compute_hashes: bool = True):
    import hashlib
    import stat

    root_abs = os.path.abspath(cache_dir)
    manifest_files = {}
    snapshot = {}
    for cur_dir, dirs, files in os.walk(root_abs):
        if cur_dir == root_abs:
            dirs[:] = sorted(d for d in dirs if d not in _CACHE_EXCLUDED_TOP_DIRS)
        else:
            dirs.sort()
        for fname in sorted(files):
            if _is_excluded_cache_file(fname):
                continue
            fpath = os.path.join(cur_dir, fname)
            rel = os.path.relpath(fpath, root_abs).replace(os.sep, "/")
            try:
                st = os.lstat(fpath)
            except OSError as exc:
                raise _CacheMutatedError(f"disappeared during scan: {rel}") from exc
            if stat.S_ISREG(st.st_mode):
                if compute_hashes:
                    h = hashlib.sha256()
                    nread = 0
                    try:
                        with open(fpath, "rb") as fp:
                            while True:
                                chunk = fp.read(1024 * 1024)
                                if not chunk:
                                    break
                                h.update(chunk)
                                nread += len(chunk)
                        st2 = os.lstat(fpath)
                    except OSError as exc:
                        raise _CacheMutatedError(f"error reading {rel}") from exc
                    if (
                        nread != st.st_size
                        or (st2.st_size, st2.st_mtime_ns, st2.st_ino)
                        != (st.st_size, st.st_mtime_ns, st.st_ino)
                    ):
                        raise _CacheMutatedError(f"mutated mid-hash: {rel}")
                    manifest_files[rel] = {"size": nread, "sha256": h.hexdigest()}
                snapshot[rel] = ("reg", st.st_size, st.st_mtime_ns, st.st_ino)
            elif stat.S_ISLNK(st.st_mode):
                try:
                    lnk_target = os.readlink(fpath)
                except OSError as exc:
                    raise _CacheMutatedError(f"broken symlink scan: {rel}") from exc
                if compute_hashes:
                    manifest_files[rel] = {"symlink": lnk_target}
                snapshot[rel] = ("lnk", lnk_target)
    return manifest_files, snapshot


def _verify_compressed_archive_frame(archive_path: str, ext: str) -> bool:
    import zlib

    if not os.path.isfile(archive_path) or os.path.getsize(archive_path) <= 0:
        return False
    if ext == "tar.zst":
        chk = subprocess.run(["zstd", "-t", "-q", archive_path], capture_output=True, check=False)
        return chk.returncode == 0
    # Verify gzip stream (full stream for <=64 MiB; first complete member for multi-GiB local archives):
    try:
        fsize = os.path.getsize(archive_path)
        if fsize < 18:
            return False
        full_scan = fsize <= 64 * 1024 * 1024
        d = zlib.decompressobj(31)
        saw_member = False
        with open(archive_path, "rb") as fp:
            while True:
                buf = fp.read(1024 * 1024)
                if not buf:
                    return saw_member and d.eof
                while buf:
                    d.decompress(buf)
                    if d.eof:
                        saw_member = True
                        if not full_scan:
                            return True
                        buf = d.unused_data
                        if buf:
                            d = zlib.decompressobj(31)
                    else:
                        buf = b""
    except Exception:
        return False


def cmd_pack_cache(
    cache_dir: str,
    out_archive: str,
    ext: str = "tar.zst",
    max_retries: int = 2,
    retry_sleep_s: float = 3.0,
    _after_scan_hook=None,
) -> int:
    """Stage a verified, manifest-signed compile-cache archive locally before any GCS upload."""
    from collections import deque

    if not os.path.isdir(cache_dir):
        return 1
    manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_NAME)
    manifest_tmp = f"{manifest_path}.{os.getpid()}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(out_archive)) or ".", exist_ok=True)

    for attempt in range(max_retries + 1):
        for p in (out_archive, manifest_tmp):
            try:
                os.unlink(p)
            except OSError:
                pass
        _clean_ephemeral_cache_files(cache_dir)
        try:
            manifest_files, pre_snap = _scan_cache_tree(cache_dir, compute_hashes=True)
        except _CacheMutatedError as exc:
            if attempt < max_retries:
                sys.stderr.write(f"[CompileCache] Cache mutated during hash scan ({exc}); retry {attempt + 1}/{max_retries}\n")
                time.sleep(retry_sleep_s)
                continue
            sys.stderr.write(f"[CompileCache] Aborting archive pack: {exc}\n")
            return 1

        if not manifest_files:
            return 3

        manifest_doc = {
            "version": 2,
            "file_count": len(manifest_files),
            "files": manifest_files,
        }
        try:
            with open(manifest_tmp, "w", encoding="utf-8") as fp:
                json.dump(manifest_doc, fp, sort_keys=True, separators=(",", ":"))
            os.replace(manifest_tmp, manifest_path)
        except OSError as exc:
            sys.stderr.write(f"[CompileCache] Failed writing manifest: {exc}\n")
            return 1

        if callable(_after_scan_hook):
            _after_scan_hook(attempt)

        # q38 FASTSTART fix (INT-4 2026-11-12): archive exactly the files the pre-scan hashed (+ the manifest), so files a live
        # server adds during the tar (lazy compiles after healthy) cannot fail the pack; XLA writes entries as *.tmp* then renames.
        list_path = f"{out_archive}.filelist"
        try:
            with open(list_path, "w", encoding="utf-8") as lf:
                for rel in sorted(pre_snap):
                    lf.write("./" + rel + "\n")   # "./" keeps the original layout and no entry can start with "-"
                lf.write("./" + CACHE_MANIFEST_NAME + "\n")
        except OSError as exc:
            sys.stderr.write(f"[CompileCache] Failed writing tar file list: {exc}\n")
            return 1
        tar_cmd = [
            "tar",
            "-C",
            cache_dir,
            "--no-recursion",
            "-T",
            list_path,
            "-cf",
            "-",
        ]
        tar_rc = 1
        comp_rc = 1
        try:
            if ext == "tar.zst":
                tar_proc = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                comp_proc = subprocess.Popen(
                    ["zstd", "-1", "-T8", "-q", "-f", "-o", out_archive],
                    stdin=tar_proc.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                assert tar_proc.stdout is not None
                tar_proc.stdout.close()
                comp_rc = comp_proc.wait()
                tar_rc = tar_proc.wait()
            else:
                workers = min(16, max(4, os.cpu_count() or 8))
                chunk_bytes = 8 * 1024 * 1024
                tar_proc = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                assert tar_proc.stdout is not None
                q = deque()
                wrote_any = False
                with open(out_archive, "wb") as out_fp, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    while True:
                        parts = []
                        rem = chunk_bytes
                        while rem > 0:
                            b = tar_proc.stdout.read(rem)
                            if not b:
                                break
                            parts.append(b)
                            rem -= len(b)
                        buf = b"".join(parts)
                        if buf:
                            q.append(pool.submit(_gzip_member, buf, 1))
                            wrote_any = True
                        while len(q) >= workers * 2 or (not buf and q):
                            out_fp.write(q.popleft().result())
                        if not buf:
                            break
                    if not wrote_any:
                        out_fp.write(_gzip_member(b"", 1))
                    out_fp.flush()
                tar_proc.stdout.close()
                tar_rc = tar_proc.wait()
                comp_rc = 0
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[CompileCache] Archive subprocess failed: {exc}\n")
            tar_rc = 1

        try:
            os.unlink(list_path)
        except OSError:
            pass
        mutated = False
        try:
            _, post_snap = _scan_cache_tree(cache_dir, compute_hashes=False)
            # q38 FASTSTART fix: only a listed (archived) file that changed or disappeared invalidates the archive; additions do not.
            mutated = any(post_snap.get(rel) != meta for rel, meta in pre_snap.items())
        except _CacheMutatedError:
            mutated = True

        if tar_rc != 0 or comp_rc != 0 or mutated:
            for p in (out_archive, manifest_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            if attempt < max_retries:
                sys.stderr.write(
                    f"[CompileCache] Cache changed or tar failed (tar_rc={tar_rc}, comp_rc={comp_rc}, mutated={mutated}); "
                    f"retry {attempt + 1}/{max_retries} in {retry_sleep_s}s\n"
                )
                time.sleep(retry_sleep_s)
                continue
            sys.stderr.write(
                f"[CompileCache] Aborting archive pack after {max_retries + 1} attempts "
                f"(tar_rc={tar_rc}, comp_rc={comp_rc}, mutated={mutated})\n"
            )
            return 1

        if not _verify_compressed_archive_frame(out_archive, ext):
            try:
                os.unlink(out_archive)
            except OSError:
                pass
            sys.stderr.write(f"[CompileCache] Compressed frame verification failed for {out_archive}\n")
            return 1

        print(len(manifest_files))
        return 0
    return 1


def cmd_verify_cache(cache_dir: str) -> int:
    """Verify every restored file in cache_dir against .cdk_cache_manifest.json (size + SHA-256)."""
    manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        sys.stderr.write(f"[CompileCache] Missing {CACHE_MANIFEST_NAME} in {cache_dir}\n")
        return 1
    try:
        with open(manifest_path, "r", encoding="utf-8") as fp:
            manifest = json.load(fp)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[CompileCache] Corrupt {CACHE_MANIFEST_NAME}: {exc}\n")
        return 1
    if not isinstance(manifest, dict) or manifest.get("version") != 2:
        sys.stderr.write("[CompileCache] Unsupported cache manifest version\n")
        return 1
    expected_files = manifest.get("files")
    if (
        not isinstance(expected_files, dict)
        or not expected_files
        or len(expected_files) != int(manifest.get("file_count", -1))
    ):
        sys.stderr.write("[CompileCache] Invalid cache manifest file map\n")
        return 1
    for rel in expected_files:
        if rel.startswith("/") or ".." in rel.split("/"):
            sys.stderr.write(f"[CompileCache] Unsafe path in cache manifest: {rel}\n")
            return 1
    try:
        actual_files, _ = _scan_cache_tree(cache_dir, compute_hashes=True)
    except _CacheMutatedError as exc:
        sys.stderr.write(f"[CompileCache] Cache tree unstable during verify: {exc}\n")
        return 1
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        changed = sorted(
            k for k in (set(expected_files) & set(actual_files)) if expected_files[k] != actual_files[k]
        )
        sys.stderr.write(
            f"[CompileCache] Manifest mismatch (missing={missing[:3]}, extra={extra[:3]}, changed={changed[:3]})\n"
        )
        return 1
    print(len(expected_files))
    return 0



def cmd_ring_tee(log_path: str, max_bytes: int = 8 * 1024 * 1024) -> int:
    """Tee stdin to stdout intact while keeping a bounded 2-file local ring buffer at log_path."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    rot_path = f"{log_path}.1"
    try:
        os.remove(rot_path)
    except OSError:
        pass
    stdin_buf = sys.stdin.buffer
    stdout_buf = sys.stdout.buffer
    read_fn = getattr(stdin_buf, "read1", stdin_buf.read)
    fp = open(log_path, "wb")
    written = 0
    stdout_alive = True
    try:
        while True:
            chunk = read_fn(65536)
            if not chunk:
                break
            if stdout_alive:
                try:
                    stdout_buf.write(chunk)
                    stdout_buf.flush()
                except BrokenPipeError:
                    stdout_alive = False
            if written > 0 and written + len(chunk) > max_bytes:
                fp.flush()
                fp.close()
                try:
                    os.replace(log_path, rot_path)
                except OSError:
                    pass
                fp = open(log_path, "wb")
                written = 0
            fp.write(chunk)
            fp.flush()
            written += len(chunk)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            fp.flush()
            fp.close()
        except OSError:
            pass
    return 0


def cmd_aiperf_mmap_key(replay_cmd: str) -> int:
    """Compute aiperf's MANIFEST_VERSION + 32-hex dataset_mmap cache key in-process (<0.5s)."""
    import contextlib
    import shlex
    import types

    saved_env = {}
    key = None
    mv = 0
    try:
        toks = shlex.split(replay_cmd)
        i = 0
        if i < len(toks) and toks[i] == "env":
            i += 1
        while i < len(toks) and "=" in toks[i] and not toks[i].startswith("-"):
            k, v = toks[i].split("=", 1)
            if k.isidentifier():
                if k not in saved_env:
                    saved_env[k] = os.environ.get(k)
                os.environ[k] = v
                i += 1
            else:
                break
        idx = toks.index("profile", i)
        with contextlib.redirect_stdout(sys.stderr):
            from aiperf.cli_commands.profile import app as profile_app
            from aiperf.config.flags.resolver import resolve_config
            from aiperf.config.loader import build_benchmark_plan
            import aiperf.dataset.mmap_cache as mmap_cache

            _, bound, _ = profile_app.parse_args(toks[idx + 1 :])
            cli_cfg = bound.kwargs["cli_config"]
            plan = build_benchmark_plan(resolve_config(cli_cfg, cli_cfg.config_file))
            try:
                from aiperf.cli_runner import _make_benchmark_run
                from aiperf.config.resolution.resolvers import build_default_resolver_chain
                from aiperf.orchestrator.orchestrator import resolve_run_seed

                seed = (
                    resolve_run_seed(plan, plan.variations[0])
                    if getattr(plan, "variations", None)
                    else None
                )
                run = _make_benchmark_run(
                    plan.configs[0],
                    random_seed=seed,
                    variables=getattr(plan, "variables", None),
                )
                build_default_resolver_chain().resolve_all(run)
            except Exception:  # noqa: BLE001
                run = types.SimpleNamespace(
                    cfg=plan.configs[0],
                    random_seed=getattr(plan, "random_seed", None),
                )
            key = mmap_cache.compute_cache_key_from_run(run)
            mv = getattr(mmap_cache, "MANIFEST_VERSION", 0)
        if key:
            print(f"mv{mv} {key}")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[DatasetPrewarm] mmap key probe skipped: {exc}\n")
    finally:
        for k, prev in saved_env.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev
    return 0


def rewrite_prewarm_cmd(replay_cmd: str) -> str:
    """Rewrite REPLAY_CMD into a short local dummy-endpoint run with an identical dataset_mmap key."""
    import shlex

    toks = shlex.split(replay_cmd)
    out = []
    i = 0
    has_unsafe = False
    while i < len(toks):
        t = toks[i]
        if t == "--url" and i + 1 < len(toks):
            out.extend(["--url", "http://127.0.0.1:19999"])
            i += 2
        elif t.startswith("--url="):
            out.extend(["--url", "http://127.0.0.1:19999"])
            i += 1
        elif t == "--server-metrics":
            i += 1
            while i < len(toks) and not toks[i].startswith("-"):
                i += 1
        elif t.startswith("--server-metrics="):
            i += 1
        elif t == "--benchmark-duration" and i + 1 < len(toks):
            out.extend(["--benchmark-duration", "5"])
            i += 2
        elif t.startswith("--benchmark-duration="):
            out.extend(["--benchmark-duration", "5"])
            i += 1
        elif t == "--warmup-requests-per-lane" and i + 1 < len(toks):
            out.extend(["--warmup-requests-per-lane", "1"])
            i += 2
        elif t.startswith("--warmup-requests-per-lane="):
            out.extend(["--warmup-requests-per-lane", "1"])
            i += 1
        else:
            if t == "--unsafe-override":
                has_unsafe = True
            out.append(t)
            i += 1
    if not has_unsafe:
        out.append("--unsafe-override")
    return shlex.join(out)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: cdk_gcs.py <exists|ls_subdirs|cp|copy_obj|sync_dir|pgzip|pgunzip|ring_tee|git_dirty_sha|overlay_sha|cache_key|pack_cache|verify_cache|aiperf_mmap_key|prewarm_cmd> ...",
            file=sys.stderr,
        )
        return 2
    cmd = sys.argv[1]
    if cmd == "pgzip":
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else min(32, max(4, (os.cpu_count() or 8)))
        level = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        return cmd_pgzip(workers=workers, level=level)
    if cmd == "pgunzip":
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else min(32, max(4, (os.cpu_count() or 8)))
        return cmd_pgunzip(workers=workers)
    if cmd == "ring_tee":
        if len(sys.argv) < 3:
            print("Usage: cdk_gcs.py ring_tee <log_path> [max_bytes]", file=sys.stderr)
            return 2
        max_b = int(sys.argv[3]) if len(sys.argv) > 3 else 8 * 1024 * 1024
        return cmd_ring_tee(sys.argv[2], max_bytes=max_b)
    if cmd == "git_dirty_sha":
        repo_dir = sys.argv[2] if len(sys.argv) > 2 else ""
        print(git_dirty_sha(repo_dir))
        return 0
    if cmd == "overlay_sha":
        mountinfo = sys.argv[2] if len(sys.argv) > 2 else "/proc/self/mountinfo"
        print(compute_overlay_sha(mountinfo_path=mountinfo))
        return 0
    if cmd == "cache_key":
        if len(sys.argv) < 6:
            print("Usage: cdk_gcs.py cache_key <recipe> <image> <role> <key_spec_b64>", file=sys.stderr)
            return 2
        print(compute_compile_cache_key(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))
        return 0
    if cmd == "pack_cache":
        if len(sys.argv) < 4:
            print("Usage: cdk_gcs.py pack_cache <cache_dir> <out_archive> [ext]", file=sys.stderr)
            return 2
        ext = sys.argv[4] if len(sys.argv) > 4 else "tar.zst"
        return cmd_pack_cache(sys.argv[2], sys.argv[3], ext=ext)
    if cmd == "verify_cache":
        if len(sys.argv) < 3:
            print("Usage: cdk_gcs.py verify_cache <cache_dir>", file=sys.stderr)
            return 2
        return cmd_verify_cache(sys.argv[2])
    if cmd == "aiperf_mmap_key":
        return cmd_aiperf_mmap_key(sys.argv[2] if len(sys.argv) > 2 else "")
    if cmd == "prewarm_cmd":
        print(rewrite_prewarm_cmd(sys.argv[2] if len(sys.argv) > 2 else ""))
        return 0
    if len(sys.argv) < 3:
        print(
            "Usage: cdk_gcs.py <exists|ls_subdirs|cp|copy_obj|stream_get|stream_put|sync_dir|prefetch_weights|pgzip|ring_tee|git_dirty_sha|overlay_sha|cache_key|pack_cache|verify_cache|aiperf_mmap_key|prewarm_cmd> ...",
            file=sys.stderr,
        )
        return 2
    if cmd == "exists":
        return cmd_exists(sys.argv[2])
    if cmd == "ls_subdirs":
        return cmd_ls_subdirs(sys.argv[2])
    if cmd in ("cp", "copy_obj"):
        args = [a for a in sys.argv[2:] if a != "--if-generation-match=0"]
        if_gen_zero = "--if-generation-match=0" in sys.argv[2:]
        return cmd_cp(args[0], args[1], if_gen_zero=if_gen_zero)
    if cmd == "stream_get":
        uri = sys.argv[2]
        workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
        chunk_mb = int(sys.argv[4]) if len(sys.argv) > 4 else 32
        return cmd_stream_get(uri, workers=workers, chunk_bytes=max(1, chunk_mb) * 1024 * 1024)
    if cmd == "stream_put":
        args = [a for a in sys.argv[2:] if a != "--if-generation-match=0"]
        if_gen_zero = "--if-generation-match=0" in sys.argv[2:]
        return cmd_stream_put(args[0], if_gen_zero=if_gen_zero)
    if cmd == "prefetch_weights":
        model_dir = sys.argv[2]
        workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        reserve_gib = float(sys.argv[4]) if len(sys.argv) > 4 else 180.0
        return cmd_prefetch_weights(model_dir, workers=workers, reserve_gib=reserve_gib)
    if cmd == "sync_dir":
        uri = sys.argv[2]
        local_dir = sys.argv[3]
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 24
        match_glob = sys.argv[5] if len(sys.argv) > 5 else ""
        return cmd_sync_dir(uri, local_dir, workers=workers, match_glob=match_glob)
    raise ValueError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    sys.exit(main())

