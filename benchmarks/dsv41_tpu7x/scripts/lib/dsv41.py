"""Shared helpers for DeepSeek-V4.1-Flash repro runs (prefill + decode).

Multi-thread safety model (see COORDINATION.md):
  * Every run is tagged with a THREAD id (e.g. "prefill", "decode", "combined").
  * Pod names are always  <user>-dsv41-<thread>-<suffix>  and carry k8s labels
    owner=<user>, dsv41-thread=<thread>, so one thread can never delete another
    thread's pod by name.
  * A node "claim" is a lock file in dsv41/locks/ (created atomically with
    O_EXCL); a second thread asking for the same pod name fails fast.
  * Each run writes only under dsv41/runs/<thread>/<UTC timestamp>/.
  * Code stacks are content-addressed tarballs in dsv41/runs/_stacks/, so two
    threads building the same stack share it and different stacks never collide.
"""

from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(os.environ.get("DSV41_ROOT", Path(__file__).resolve().parents[2]))  # .../dsv41 or .../benchmarks/dsv41_tpu7x
REPOS = ROOT / "repos"
DEFAULT_VTT = (
    REPOS / "vtt-combined"
    if (REPOS / "vtt-combined").exists()
    else (
        ROOT.parents[1]
        if len(ROOT.parents) > 1 and (ROOT.parents[1] / "src" / "vllm_torchtpu").exists()
        else REPOS / "vtt-combined"
    )
)
SUPPORT = ROOT / "support"
WHLS = ROOT / "whls"
RUNS = ROOT / "runs"
LOCKS = ROOT / "locks"
STACKS = RUNS / "_stacks"
USER = os.environ.get("DSV41_USER", getpass.getuser())

IMAGE = "vllm/vllm-tpu:nightly"
GCS_BUCKET = os.environ.get("DSV41_GCS_BUCKET", "rsinghc-storage")
GCS_PREFIX = os.environ.get("DSV41_GCS_PREFIX", "DeepSeek-V4.1-Flash/")

SYS_NS = {
    "kube-system", "gmp-system", "gke-managed-cim", "kueue-system", "jobset-system",
    "lws-system", "gke-managed-networking-dra-driver", "gke-managed-dpv2-observability",
    "gke-managed-mldiagnostics", "gke-gmp-system", "gke-managed-system",
}


def log(msg: str) -> None:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --------------------------------------------------------------------------- auth / kubectl
def get_token() -> str:
    for cmd in (
        ["gcloud", "auth", "application-default", "print-access-token"],
        ["gcloud", "auth", "print-access-token"],
    ):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
            if len(out) > 50:
                return out
        except Exception:
            pass
    raise RuntimeError(
        "No gcloud access token. Run `gcloud auth login --update-adc` (CAA: go/gcloud-caa-error)."
    )


class Kube:
    def __init__(self, ctx: str, pod: str):
        self.ctx, self.pod = ctx, pod

    def _base(self) -> list[str]:
        return ["kubectl", f"--context={self.ctx}"]

    def run(self, args: list[str], **kw) -> subprocess.CompletedProcess:
        return subprocess.run(self._base() + args, **kw)

    def out(self, args: list[str], timeout: int = 60) -> str:
        return subprocess.check_output(self._base() + args, text=True, timeout=timeout,
                                       stderr=subprocess.DEVNULL)

    def exec_sh(self, script: str, capture: bool = False, timeout: int = 1800, retries: int = 4) -> str:
        """Run a bash script inside the pod via stdin (no quoting issues)."""
        last = None
        for att in range(retries):
            try:
                cmd = self._base() + ["exec", "-i", self.pod, "--", "bash", "-s"]
                if capture:
                    return subprocess.check_output(cmd, input=script.encode(), timeout=timeout).decode(
                        "utf-8", errors="replace")
                subprocess.run(cmd, input=script.encode(), check=True, timeout=timeout)
                return ""
            except Exception as e:  # transient apiserver / exec errors
                last = e
                time.sleep(3 * (att + 1))
        raise last  # type: ignore[misc]

    def cp_in(self, local: str | Path, remote: str, retries: int = 4) -> None:
        for att in range(retries):
            r = self.run(["cp", str(local), f"{self.pod}:{remote}"], capture_output=True, text=True)
            if r.returncode == 0:
                return
            log(f"kubectl cp retry {att}: {r.stderr.strip()[:200]}")
            time.sleep(3 * (att + 1))
        raise RuntimeError(f"kubectl cp {local} -> {remote} failed")

    def cp_out(self, remote: str, local: str | Path) -> bool:
        r = self.run(["cp", f"{self.pod}:{remote}", str(local)], capture_output=True, text=True)
        return r.returncode == 0 and Path(local).exists() and Path(local).stat().st_size > 0

    def foreign_running_pods(self) -> list[str]:
        pods = json.loads(self.out(["get", "pods", "-A", "-o", "json"]))
        return [
            f"{p['metadata']['namespace']}/{p['metadata']['name']}"
            for p in pods.get("items", [])
            if p["metadata"].get("namespace") not in SYS_NS
            and p.get("status", {}).get("phase") == "Running"
            and not p["metadata"]["name"].startswith(USER)
        ]

    def apply(self, manifest: dict) -> None:
        subprocess.run(self._base() + ["apply", "-f", "-"], input=json.dumps(manifest), text=True, check=True)

    def wait_ready(self, timeout_s: int = 1800) -> str:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                p = json.loads(self.out(["get", "pod", self.pod, "-o", "json"]))
                phase = p.get("status", {}).get("phase")
                if phase in ("Failed", "Succeeded"):
                    raise RuntimeError(f"pod {self.pod} ended in phase {phase}")
                if any(c.get("type") == "Ready" and c.get("status") == "True"
                       for c in p.get("status", {}).get("conditions", [])):
                    return p.get("spec", {}).get("nodeName", "?")
            except subprocess.CalledProcessError:
                pass
            time.sleep(10)
        raise TimeoutError(f"pod {self.pod} not Ready after {timeout_s}s")

    def delete(self) -> None:
        self.run(["delete", "pod", self.pod, "--ignore-not-found=true", "--grace-period=0", "--force"],
                 capture_output=True)


# --------------------------------------------------------------------------- naming / locks / run dirs
def pod_name(thread: str, suffix: str) -> str:
    name = f"{USER}-dsv41-{thread}-{suffix}".lower().replace("_", "-")
    assert len(name) <= 63, name
    return name


def run_dir(thread: str, tag: str) -> Path:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    d = RUNS / thread / f"{ts}_{tag}"
    d.mkdir(parents=True, exist_ok=False)
    return d


class Claim:
    """Atomic lock on (ctx, pod). Prevents two threads from driving the same pod."""

    def __init__(self, ctx: str, pod: str, thread: str, info: dict):
        LOCKS.mkdir(parents=True, exist_ok=True)
        self.path = LOCKS / f"{ctx.split('_')[-1]}__{pod}.lock"
        payload = dict(ctx=ctx, pod=pod, thread=thread, pid=os.getpid(), host=os.uname().nodename,
                       started_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(), **info)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            raise SystemExit(f"LOCKED: {self.path} exists:\n{self.path.read_text()}\n"
                             "Another thread owns this pod. Use a different --suffix, or remove the lock "
                             "only if you are sure that run is dead (check its pid).")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- stack building
def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _git_dirty(repo: Path) -> str:
    """Hash of uncommitted diff (empty string when clean)."""
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD"], text=False)
    untracked = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"], text=True)
    if not diff and not untracked.strip():
        return ""
    h = hashlib.sha256(diff)
    for f in sorted(untracked.split()):
        p = repo / f
        if p.is_file():
            h.update(f.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:10]


def build_stack(vtt_repo: Path, vllm_repo: Path, sitecustomize: Path, extra_files: dict[str, Path] | None = None) -> tuple[Path, dict]:
    """Tar of /workspace/{whls,vllm,vllm-torchtpu,sitecustomize_bench,mock_metadata.py,metadata_hook.c}.

    Content-addressed by (vtt HEAD+dirty, vllm HEAD+dirty, support files), so concurrent threads
    reuse identical stacks and never overwrite each other's.
    """
    prov = {
        "vllm_torchtpu": {"path": str(vtt_repo), "head": _git_head(vtt_repo), "dirty": _git_dirty(vtt_repo)},
        "vllm": {"path": str(vllm_repo), "head": _git_head(vllm_repo), "dirty": _git_dirty(vllm_repo)},
    }
    h = hashlib.sha256(json.dumps(prov, sort_keys=True).encode())
    support_files = {
        "workspace/sitecustomize_bench/sitecustomize.py": sitecustomize,
        "workspace/sitecustomize_bench/decode_bench_seeding.py": SUPPORT / "sitecustomize_bench/decode_bench_seeding.py",
        "workspace/mock_metadata.py": SUPPORT / "mock_metadata.py",
        "workspace/metadata_hook.c": SUPPORT / "metadata_hook.c",
        **(extra_files or {}),
    }
    for arc, p in sorted(support_files.items()):
        h.update(arc.encode()); h.update(Path(p).read_bytes())
    key = h.hexdigest()[:16]
    STACKS.mkdir(parents=True, exist_ok=True)
    out = STACKS / f"stack_{key}.tar"
    prov["stack_key"] = key
    if out.exists():
        log(f"reusing stack {out.name}")
        return out, prov

    def _filt(ti: tarfile.TarInfo):
        parts = Path(ti.name).parts
        if any(p in (".git", "__pycache__", ".ruff_cache", ".mypy_cache", "build", "dist", "runs", "repos", "locks") for p in parts):
            return None
        if ti.name.endswith(".egg-info") or ".egg-info/" in ti.name or ti.name.endswith(".whl"):
            return None
        return ti

    tmp = out.with_suffix(f".tar.tmp{os.getpid()}")
    log(f"building stack {out.name} ...")
    with tarfile.open(tmp, "w") as tf:
        for whl in sorted(WHLS.glob("*.whl")):
            tf.add(whl, arcname=f"workspace/whls/{whl.name}")
        tf.add(vllm_repo, arcname="workspace/vllm", filter=_filt)
        tf.add(vtt_repo, arcname="workspace/vllm-torchtpu", filter=_filt)
        for arc, p in support_files.items():
            tf.add(p, arcname=arc)
        info = tarfile.TarInfo("workspace/STACK_PROVENANCE.json")
        data = json.dumps(prov, indent=2).encode()
        info.size = len(data)
        import io
        tf.addfile(info, io.BytesIO(data))
    os.replace(tmp, out)  # atomic: concurrent builders end up with one identical file
    log(f"stack ready: {out} ({out.stat().st_size/1e9:.2f} GB)")
    return out, prov


# --------------------------------------------------------------------------- in-pod common staging
STAGE_BASE_SH = r"""
set -e
mkdir -p /workspace
tar --overwrite --keep-directory-symlink -C / -xf /tmp/stack.tar && rm -f /tmp/stack.tar
cd /workspace
(gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl) \
  || (apt-get update -qq >/dev/null && apt-get install -y -qq gcc >/dev/null && \
      gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl)
git config --global --add safe.directory '*' || true
echo STAGE_BASE_OK
"""

INSTALL_SH = r"""
set -e
pip uninstall -y tpu-inference torchax >/dev/null 2>&1 || true
pip install --no-deps jax==0.10.2 jaxlib==0.10.2 libtpu==0.0.44 portpicker==1.6.0 compressed-tensors==0.15.0 sympy >/dev/null
pip install --no-deps --force-reinstall /workspace/whls/*.whl >/dev/null
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.29.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0 VLLM_TARGET_DEVICE=empty \
  pip install --no-deps --no-build-isolation -e /workspace/vllm >/dev/null
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install --no-deps --no-build-isolation -e /workspace/vllm-torchtpu >/dev/null
python3 -c 'import importlib.metadata as m; v=m.version("libtpu"); assert v=="0.0.44", v; print("libtpu", v, "vllm", m.version("vllm"), "torch_tpu", m.version("torch_tpu"))'
echo INSTALL_OK
"""

# Range-GET downloader (same algorithm as the handoff scripts). NEEDED=None -> all shards.
DOWNLOAD_PY = r"""
import os, sys, json, time, glob, urllib.request, urllib.parse, urllib.error, concurrent.futures
BUCKET, PREFIX = os.environ['DSV41_GCS_BUCKET'], os.environ['DSV41_GCS_PREFIX']
HF_REPO = os.environ.get('DSV41_HF_REPO', '').strip()
DST = '/dev/shm/DeepSeek-V4.1-Flash'
SUBSET = os.environ.get('DSV41_SHARDS', 'all')   # 'prefill' or 'all'
CHUNK, BUF = 2 << 30, 16 << 20
NEEDED = None
if SUBSET == 'prefill':
    NEEDED = {f"model-{i:05d}-of-00048.safetensors" for i in list(range(2, 24)) + [47, 48]}
os.makedirs(DST, exist_ok=True)
repo_marker = f'{DST}/.hf_repo'
prev_repo = open(repo_marker).read().strip() if os.path.exists(repo_marker) else ''
if prev_repo != HF_REPO:
    print(f'Checkpoint source changed ({prev_repo!r} -> {HF_REPO!r}); clearing stale safetensors & markers', flush=True)
    for p in glob.glob(f'{DST}/.done_*') + glob.glob(f'{DST}/*.safetensors'):
        try: os.remove(p)
        except OSError: pass
marker = f'{DST}/.done_{SUBSET}'
if os.path.exists(marker):
    print(f'CKPT_ALREADY_READY (hf_repo={HF_REPO or "gcs-default"})', flush=True); sys.exit(0)
token = os.environ.get('GCS_TOKEN', '')
def _vm_token():
    req = urllib.request.Request('http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token',
                                 headers={'Metadata-Flavor': 'Google'})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())['access_token']
def _probe(tok):
    url = f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?maxResults=1&prefix=' + urllib.parse.quote(PREFIX)
    try:
        urllib.request.urlopen(urllib.request.Request(url, headers={'Authorization': f'Bearer {tok}'}), timeout=30).read()
        return True
    except Exception as e:
        print(f'token probe failed: {e}', flush=True); return False
if not HF_REPO:
    if not (token and _probe(token)):
        print('falling back to node service-account token', flush=True)
        token = _vm_token()
        assert _probe(token), 'no working GCS credential in pod'
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
_no_redir_opener = urllib.request.build_opener(_NoRedirect)
def _resolve_hf_url(rel):
    base_url = f'https://huggingface.co/{HF_REPO}/resolve/main/{urllib.parse.quote(rel, safe="/")}'
    if not rel.endswith('.safetensors'):
        return base_url
    for r in range(5):
        try:
            with _no_redir_opener.open(urllib.request.Request(base_url, method='HEAD', headers={'User-Agent': 'vllm-bench/1.0'}), timeout=20):
                return base_url
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                loc = e.headers.get('Location')
                if loc:
                    return ('https://huggingface.co' + loc) if loc.startswith('/') else loc
            if r == 4:
                return base_url
            time.sleep(1 + r)
        except Exception:
            if r == 4:
                return base_url
            time.sleep(1 + r)
    return base_url
def list_blobs():
    if HF_REPO:
        api_url = f'https://huggingface.co/api/models/{HF_REPO}?blobs=true'
        for r in range(5):
            try:
                d = json.loads(urllib.request.urlopen(urllib.request.Request(api_url, headers={'User-Agent': 'vllm-bench/1.0'}), timeout=30).read())
                break
            except Exception:
                if r == 4: raise
                time.sleep(2)
        out = []
        for it in d.get('siblings', []):
            rel = it['rfilename']
            if rel.endswith(('.pdf', '.png', '.jpeg', '.patch', '.md', '/', '.tail', '.gitattributes')) or '/eval_cache/' in rel or '/perf_cache/' in rel:
                continue
            sz = int((it.get('lfs') or {}).get('size', it.get('size', 0)))
            out.append((rel, sz))
        return out
    items, pt = [], None
    while True:
        url = f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix=' + urllib.parse.quote(PREFIX)
        if pt: url += '&pageToken=' + urllib.parse.quote(pt)
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'}), timeout=30).read())
        items += d.get('items', []); pt = d.get('nextPageToken')
        if not pt: break
    return [(it['name'][len(PREFIX):], int(it['size'])) for it in items
            if not it['name'].endswith(('.pdf', '.png', '.jpeg', '.patch', '.md', '/', '.tail'))
            and '/eval_cache/' not in it['name'] and '/perf_cache/' not in it['name']]
def dl(a):
    url, dst, s, e, is_hf = a
    for r in range(6):
        try:
            hdrs = {'User-Agent': 'vllm-bench/1.0', 'Range': f'bytes={s}-{e-1}'}
            if not is_hf:
                hdrs['Authorization'] = f'Bearer {token}'
            req = urllib.request.Request(url, headers=hdrs)
            fd = os.open(dst, os.O_WRONLY)
            try:
                pos = s
                with urllib.request.urlopen(req, timeout=180) as resp:
                    while pos < e:
                        b = resp.read(min(BUF, e - pos))
                        if not b: break
                        os.pwrite(fd, b, pos); pos += len(b)
            finally:
                os.close(fd)
            return e - s
        except Exception:
            if r == 5: raise
            time.sleep(2 * (r + 1))
t0 = time.time(); tasks = []; total = 0
blobs = list_blobs()
to_dl = []
for rel, sz in blobs:
    if NEEDED is not None and rel.endswith('.safetensors') and rel not in NEEDED: continue
    dst = os.path.join(DST, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == sz: continue
    to_dl.append((rel, dst, sz))
if HF_REPO and to_dl:
    print(f'Resolving {len(to_dl)} files from HuggingFace {HF_REPO} ...', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as tex:
        urls = list(tex.map(lambda x: _resolve_hf_url(x[0]), to_dl))
else:
    urls = [f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{urllib.parse.quote(PREFIX + rel, safe="")}?alt=media' for rel, _, _ in to_dl]
for (rel, dst, sz), url in zip(to_dl, urls):
    fd = os.open(dst, os.O_CREAT | os.O_WRONLY, 0o644); os.ftruncate(fd, sz); os.close(fd)
    total += sz
    for s in range(0, max(1, sz), CHUNK): tasks.append((url, dst, s, min(s + CHUNK, sz), bool(HF_REPO)))
print(f'Downloading {total/1e9:.1f} GB in {len(tasks)} chunks (source={HF_REPO or (BUCKET + "/" + PREFIX)})', flush=True)
with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex: list(ex.map(dl, tasks))
if NEEDED is not None:
    ip = f'{DST}/model.safetensors.index.json'; io_ = json.load(open(ip))
    io_['weight_map'] = {k: v for k, v in io_['weight_map'].items() if v in NEEDED}
    json.dump(io_, open(ip, 'w'), indent=2)
with open(repo_marker, 'w') as rf: rf.write(HF_REPO)
open(marker, 'w').close()
print(f'CKPT_READY in {time.time()-t0:.1f}s (source={HF_REPO or "gcs-default"})', flush=True)
"""


def download_ckpt_sh(subset: str, token: str, hf_repo: str = "") -> str:
    return (
        "set -e\nmkdir -p /dev/shm/DeepSeek-V4.1-Flash\n"
        f"export GCS_TOKEN='{token}' DSV41_GCS_BUCKET='{GCS_BUCKET}' DSV41_GCS_PREFIX='{GCS_PREFIX}' DSV41_SHARDS='{subset}' DSV41_HF_REPO='{hf_repo}'\n"
        "cat > /tmp/dsv41_dl.py <<'PYEOF'\n" + DOWNLOAD_PY + "\nPYEOF\npython3 -u /tmp/dsv41_dl.py\n"
    )


def preflight(ctx: str) -> None:
    get_token()
    r = subprocess.run(["kubectl", f"--context={ctx}", "get", "nodes",
                        "-l", "cloud.google.com/gke-tpu-topology=2x2x1", "--no-headers"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        sys.exit(f"preflight: cannot reach {ctx}: {r.stderr.strip()[:300]}")


# --------------------------------------------------------------------------- GCS stack staging
# `kubectl cp` through the API server runs at ~0.2 MB/s from a cloudtop, so a 0.74 GB stack takes
# over an hour. Instead: upload once to GCS (content-addressed, shared by all threads), pull in-pod.
STAGING = os.environ.get("DSV41_STAGING", "gs://chaitanyapk-dsv41-staging")


def upload_stack(stack: Path) -> str:
    uri = f"{STAGING}/stacks/{stack.name}"
    r = subprocess.run(["gcloud", "storage", "ls", uri], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"uploading {stack.name} -> {uri}")
        subprocess.run(["gcloud", "storage", "cp", str(stack), uri + ".tmp"], check=True, capture_output=True)
        subprocess.run(["gcloud", "storage", "mv", uri + ".tmp", uri], check=True, capture_output=True)
    return uri


def gcs_fetch_sh(uri: str, dst: str, token: str) -> str:
    bucket, obj = uri[len("gs://"):].split("/", 1)
    return f"""set -e
export GCS_TOKEN='{token}'
python3 - <<'PYEOF'
import os, json, urllib.request, urllib.parse, shutil
def vm():
    r = urllib.request.Request('http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token', headers={{'Metadata-Flavor': 'Google'}})
    return json.loads(urllib.request.urlopen(r, timeout=10).read())['access_token']
url = 'https://storage.googleapis.com/storage/v1/b/{bucket}/o/' + urllib.parse.quote('{obj}', safe='') + '?alt=media'
last = None
for tok in (os.environ.get('GCS_TOKEN'), None):
    try:
        tok = tok or vm()
        with urllib.request.urlopen(urllib.request.Request(url, headers={{'Authorization': 'Bearer ' + tok}}), timeout=600) as r, open('{dst}', 'wb') as f:
            shutil.copyfileobj(r, f, 16 << 20)
        print('FETCHED', os.path.getsize('{dst}'), flush=True); break
    except Exception as e:
        last = e; print('fetch attempt failed:', e, flush=True)
else:
    raise SystemExit(f'cannot fetch {uri}: {{last}}')
PYEOF
"""


def stage_stack(k: Kube, stack: Path) -> None:
    uri = upload_stack(stack)
    k.exec_sh(gcs_fetch_sh(uri, "/tmp/stack.tar", get_token()), timeout=1800)


def install_signal_handlers() -> None:
    """SIGTERM/SIGINT/SIGHUP -> KeyboardInterrupt so `finally:` deletes the pod and releases the lock."""
    import signal

    def _h(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _h)


def start_detached(k: Kube, cmd: str, pgrep_pat: str) -> None:
    """Launch a long-running in-pod process WITHOUT retries (a retry would start a 2nd copy).

    `kubectl exec` can stay attached to a backgrounded child even with all fds redirected, so we
    use a short timeout and then confirm via pgrep that the process is alive.
    """
    full = k._base() + ["exec", "-i", k.pod, "--", "bash", "-s"]
    script = f"setsid nohup bash -c {json.dumps(cmd)} > /dev/null 2>&1 < /dev/null &\ndisown || true\necho STARTED\n"
    try:
        subprocess.run(full, input=script.encode(), timeout=60, check=False)
    except subprocess.TimeoutExpired:
        log("kubectl exec did not return within 60s (expected with detached children); verifying ...")
    time.sleep(5)
    alive = k.exec_sh(f"pgrep -f {json.dumps(pgrep_pat)} >/dev/null && echo Y || echo N", capture=True).strip()
    if alive != "Y":
        raise RuntimeError(f"detached process {pgrep_pat!r} not running in {k.pod}")
    log(f"detached process running: {pgrep_pat}")
