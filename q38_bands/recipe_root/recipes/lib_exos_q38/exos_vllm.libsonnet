// The exos-vllm recipe library: provides `recipe(name, overrides)` for
// single-host and multi-host (Ray) vLLM serving + benchmarking, and
// `pdRecipe(name, overrides)` for multi-host Prefill/Decode disaggregated
// serving + benchmarking with TPURaidenConnector on Exos Borg TPU runners.
// Every default can be overridden with the cdk variable of the same name.

local cdk = import 'cdk.common';
local exos = import 'cdk.exos';

{
  defaults:: {
    PLATFORM: 'ghostlite_pod',
    SLICE: '2x2',
    IMAGE: 'us-central1-docker.pkg.dev/cloud-tpu-inference-test/inferencex/vllm@sha256:0e9c2b99fd2a8cfa2f1b87a5d03aec6acd9ebb70e84b8613ceb5188d83f1f637',
    CLIENT_IMAGE: 'us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/inferencex-agentx:latest',
    MODEL: '',
    SERVED_MODEL_NAME: 'model',
    TP: '',
    MAX_MODEL_LEN: '2048',
    VLLM_EXTRA_ARGS: '',
    MULTIHOST_BACKEND: '',
    RAY_PORT: '26379',
    // Workload mode: 'smoke' (single completion request), '8k1k'
    // (benchmark_serving.py random 8K/1K), or 'agentx' (AgentX multi-turn
    // trace replay via benchmark_lib.sh / aiperf).
    BENCH_MODE: 'smoke',
    CONCURRENCY: '32',
    ISL: '8192',
    OSL: '1024',
    RANGE_RATIO: '0.8',
    NUM_PROMPTS: '',
    NUM_WARMUPS: '',
    FAST: 'true',
    DURATION: '600',
    USE_RUNAI_STREAMER: 'true',
    LOAD_FORMAT: '',
    EXTRA_ENV: {},
  },

  // pyPatchSnippet configures the container for Borg's IPv6-only network,
  // patches RunaiModelStreamerLoader for MTP draft weight streaming from GCS,
  // and installs a fast pure-Python GCS downloader (/tmp/exos_gcs_cp.py) backed
  // by /exos/gcs and tpu_runner's GCE_METADATA_HOST server.
  //
  // IPv6 handling: components that take an address (vLLM, torch, Ray, gRPC)
  // are pointed at IPv6 through exos.ipv6Env and the per-host env set in the
  // scripts. Code that hardcodes IPv4 (libzmq's IPV6_V6ONLY=0, Ray's
  // find_free_port) is covered by the exos_ipv6_preload.so LD_PRELOAD shim,
  // which exos.ipv6PreloadSnippet enables.
  // pyPatchBody carries the Python bootstrap payload in container env
  // (_EXOS_PYPATCH via pyPatchEnv) rather than inline in argv[2] so multi-leg
  // P/D scripts stay well below Linux MAX_ARG_STRLEN (128 KiB per argv/envp
  // entry); pyPatchSnippet unsets _EXOS_PYPATCH immediately after execution.
  pyPatchBody:: |||
    import glob, os, site, sys, sysconfig, textwrap

    dirs = set()
    for key in ('purelib', 'platlib'):
        try:
            p = sysconfig.get_path(key)
            if p:
                dirs.add(p)
        except Exception:
            pass
    for d in getattr(site, 'getsitepackages', lambda: [])():
        dirs.add(d)
    for d in sys.path:
        if d:
            dirs.add(d)
    for pat in (
        '/opt/venv/lib/python*/site-packages',
        '/root/aiperf-venv/lib/python*/site-packages',
        '/usr/local/lib/python*/site-packages',
        '/usr/local/lib/python*/dist-packages',
        '/usr/lib/python*/site-packages',
        '/usr/lib/python*/dist-packages',
        '/workspace/vllm-torchtpu/src',
        '/root/cloud-devkit/vllm-torchtpu/src',
        '/root/*/vllm-torchtpu/src',
        '/root/torchtpu-vllm/src',
        '/opt/vllm-torchtpu/src',
        '/workspace/sglang/python',
    ):
        for d in glob.glob(pat):
            dirs.add(d)

    rsl_patch = '''
        # --- EXOS_MTP_RUNAI_STREAMER_PATCH ---
        import builtins as _exos_builtins
        import json as _exos_json
        import os as _exos_os
        import re as _exos_re
        import urllib.parse as _exos_up
        import urllib.request as _exos_ur

        _EXOS_INDEX_CACHE = {}

        def _exos_read_fresh_token() -> str:
            host = _exos_os.environ.get("GCE_METADATA_HOST") or "metadata.google.internal"
            try:
                treq = _exos_ur.Request(
                    f"http://{host}/computeMetadata/v1/instance/service-accounts/default/token",
                    headers={"Metadata-Flavor": "Google"},
                )
                return _exos_json.loads(_exos_ur.urlopen(treq, timeout=10).read().decode())["access_token"]
            except Exception:
                pass
            return ""

        def _exos_get_weight_map(local_or_gcs_path):
            gcs_model = _exos_os.environ.get("MODEL", "")
            key = gcs_model or str(local_or_gcs_path)
            if key in _EXOS_INDEX_CACHE:
                return _EXOS_INDEX_CACHE[key]
            for cand_dir in (str(local_or_gcs_path),):
                idx_p = _exos_os.path.join(cand_dir, "model.safetensors.index.json")
                if _exos_os.path.isfile(idx_p):
                    try:
                        wm = _exos_json.load(open(idx_p)).get("weight_map", {})
                        _EXOS_INDEX_CACHE[key] = wm
                        return wm
                    except Exception:
                        pass
            if gcs_model.startswith("gs://"):
                try:
                    import fcntl as _exos_fcntl
                    rest = gcs_model[5:].rstrip("/")
                    bkt, _, pfx = rest.partition("/")
                    fuse_idx = f"/exos/gcs/{bkt}/{pfx}/model.safetensors.index.json"
                    if _exos_os.path.isfile(fuse_idx):
                        wm = _exos_json.load(open(fuse_idx)).get("weight_map", {})
                        _EXOS_INDEX_CACHE[key] = wm
                        return wm
                    cache_dir = "/tmp/workspace/.exos_hdr_cache"
                    _exos_os.makedirs(cache_dir, exist_ok=True)
                    cache_idx = _exos_os.path.join(cache_dir, _exos_up.quote(f"{bkt}_{pfx}_index.json", safe=""))
                    with open(_exos_os.path.join(cache_dir, ".index.lock"), "w") as lf:
                        _exos_fcntl.flock(lf, _exos_fcntl.LOCK_EX)
                        if _exos_os.path.isfile(cache_idx) and _exos_os.path.getsize(cache_idx) > 2:
                            wm = _exos_json.load(open(cache_idx)).get("weight_map", {})
                            _EXOS_INDEX_CACHE[key] = wm
                            return wm
                        tok = _exos_read_fresh_token()
                        hdrs = {"Authorization": f"Bearer {tok}"} if tok else {}
                        obj = f"{pfx}/model.safetensors.index.json"
                        url = f"https://storage.googleapis.com/storage/v1/b/{bkt}/o/{_exos_up.quote(obj, safe='')}?alt=media"
                        ireq = _exos_ur.Request(url, headers=hdrs)
                        raw_idx = _exos_ur.urlopen(ireq, timeout=30).read()
                        try:
                            tmp_i = f"{cache_idx}.tmp.{_exos_os.getpid()}"
                            with open(tmp_i, "wb") as wf:
                                wf.write(raw_idx)
                            _exos_os.replace(tmp_i, cache_idx)
                        except Exception:
                            pass
                        wm = _exos_json.loads(raw_idx.decode()).get("weight_map", {})
                        _EXOS_INDEX_CACHE[key] = wm
                        return wm
                except Exception:
                    pass
            _EXOS_INDEX_CACHE[key] = {}
            return {}

        _orig_prepare_weights = RunaiModelStreamerLoader._prepare_weights
        def _exos_prepare_weights(self, model_name_or_path, *args, **kwargs):
            orig_arg = model_name_or_path
            if not str(model_name_or_path).startswith("gs://") and _exos_os.environ.get("MODEL", "").startswith("gs://"):
                model_name_or_path = _exos_os.environ["MODEL"]
            files = _orig_prepare_weights(self, model_name_or_path, *args, **kwargs)
            flt = getattr(_exos_builtins, "_EXOS_MODEL_LOAD_FILTER", None)
            if flt is not None and files:
                is_mtp = flt[0]
                wm = _exos_get_weight_map(orig_arg)
                if wm:
                    from runai_model_streamer._impl import _exos_should_skip_tensor
                    needed = set()
                    for tname, shard in wm.items():
                        if not _exos_should_skip_tensor(tname, flt):
                            needed.add(_exos_os.path.basename(shard))
                    if needed:
                        filtered = [f for f in files if _exos_os.path.basename(f) in needed]
                        if filtered:
                            print(f"[exos-shard-filter] is_mtp={is_mtp}: keeping {len(filtered)}/{len(files)} safetensors shards")
                            return filtered
            return files
        RunaiModelStreamerLoader._prepare_weights = _exos_prepare_weights

        _orig_load_weights = RunaiModelStreamerLoader.load_weights
        def _exos_load_weights(self, model, model_config):
            # EXOS_WEIGHT_LOAD_FILTER=0 streams every tensor to every rank, as vLLM does on Cloud,
            # instead of skipping other pipeline stages' layers, MTP tensors and non-local experts.
            # Needed where the tensors a rank consumed are part of a cross-leg identity: Qwen
            # PD's loaded-weight digest would otherwise differ per pipeline stage and from the
            # decode ("Layout fingerprint mismatch between source and destination").
            if _exos_os.environ.get("EXOS_WEIGHT_LOAD_FILTER", "1") == "0":
                return _orig_load_weights(self, model, model_config)
            is_mtp = "MTP" in type(model).__name__
            pp_missing = tuple(f"{n}." for n, m in model.named_modules() if type(m).__name__ == "PPMissingLayer")
            num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
            if not isinstance(num_layers, int):
                num_layers = getattr(getattr(model_config, "hf_text_config", None), "num_hidden_layers", None)
            mtp_prefix = f"model.layers.{num_layers}." if isinstance(num_layers, int) else None
            local_expert_ids = None
            if not is_mtp:
                try:
                    from vllm.config import get_current_vllm_config as _get_vc
                    from vllm.distributed import get_ep_group as _get_ep
                    _vc = _get_vc()
                    _mc = _vc.model_config
                    _pc = _vc.parallel_config
                    if getattr(_mc, "is_moe", False) and getattr(_pc, "enable_expert_parallel", False) and not getattr(_pc, "enable_eplb", False) and not _exos_os.environ.get("TPU_EXPERT_PLACEMENT", "").strip():
                        _ep = _get_ep()
                        _ep_size = _ep.world_size
                        _ep_rank = _ep.rank_in_group
                        _num_exp = _mc.get_num_experts() if hasattr(_mc, "get_num_experts") else None
                        if _ep_size > 1 and _num_exp:
                            try:
                                from vllm.model_executor.layers.fused_moe.layer import compute_local_expert_ids as _clei
                                _, _lids = _clei(_ep_rank, _ep_size, _num_exp, getattr(_pc, "expert_placement_strategy", "linear"))
                                local_expert_ids = set(_lids)
                            except Exception:
                                if _num_exp % _ep_size == 0:
                                    _strat = getattr(_pc, "expert_placement_strategy", "linear")
                                    if _strat == "round_robin":
                                        local_expert_ids = set(range(_ep_rank, _num_exp, _ep_size))
                                    else:
                                        _per = _num_exp // _ep_size
                                        local_expert_ids = set(range(_ep_rank * _per, (_ep_rank + 1) * _per))
                except Exception:
                    pass
            _exos_builtins._EXOS_MODEL_LOAD_FILTER = (is_mtp, pp_missing, mtp_prefix, local_expert_ids)
            try:
                return _orig_load_weights(self, model, model_config)
            finally:
                _exos_builtins._EXOS_MODEL_LOAD_FILTER = None
                try:
                    import ctypes as _exos_ctypes, gc as _exos_gc
                    _exos_gc.collect()
                    _exos_ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
                try:
                    import ctypes as _exos_ctypes
                    _exos_ctypes.CDLL("libtcmalloc.so.4").MallocExtension_ReleaseFreeMemory()
                except Exception:
                    pass
        RunaiModelStreamerLoader.load_weights = _exos_load_weights
    '''

    mlp_patch = '''
        # --- EXOS_MTP_PP_SKIP_PATCH ---
        import builtins as _exos_builtins
        import ctypes as _exos_mlp_ctypes
        import gc as _exos_mlp_gc
        import re as _exos_re
        _EXOS_MLP_EXPERT_RE = _exos_re.compile(r"\\.experts\\.(\\d+)\\.")
        def _exos_mlp_trim_mem():
            try:
                _exos_mlp_gc.collect()
                _exos_mlp_ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            try:
                _exos_mlp_ctypes.CDLL("libtcmalloc.so.4").MallocExtension_ReleaseFreeMemory()
            except Exception:
                pass
        if "patch_moe_expert_write_staging" in globals():
            _orig_pmews = globals()["patch_moe_expert_write_staging"]
            def _exos_pmews(*a, **kw):
                res = _orig_pmews(*a, **kw)
                try:
                    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader as _DML
                    if hasattr(_DML, "load_weights") and not getattr(_DML.load_weights, "_exos_trim_wrapped", False):
                        _orig_dml_lw = _DML.load_weights
                        def _exos_dml_lw(self, *la, **lkw):
                            try:
                                return _orig_dml_lw(self, *la, **lkw)
                            finally:
                                _exos_mlp_trim_mem()
                        _exos_dml_lw._exos_trim_wrapped = True
                        _DML.load_weights = _exos_dml_lw
                except Exception:
                    pass
                return res
            globals()["patch_moe_expert_write_staging"] = _exos_pmews
        _orig_exos_should_skip = _should_skip
        def _should_skip(name: str, local_expert_ids: set[int]) -> bool:
            flt = getattr(_exos_builtins, "_EXOS_MODEL_LOAD_FILTER", None)
            if flt is not None:
                is_mtp, pp_missing, mtp_prefix, flt_expert_ids = flt
                if is_mtp:
                    if not (name.startswith("mtp.") or (mtp_prefix and name.startswith(mtp_prefix)) or name.startswith(("model.embed_tokens.", "model.language_model.embed_tokens.", "lm_head."))):
                        return True
                else:
                    if name.startswith("mtp.") or (mtp_prefix and name.startswith(mtp_prefix)) or (pp_missing and name.startswith(pp_missing)):
                        return True
                    if flt_expert_ids is not None and not name.endswith(".input_scale"):
                        m = _EXOS_MLP_EXPERT_RE.search(name)
                        if m is not None and int(m.group(1)) not in flt_expert_ids:
                            return True
                    if local_expert_ids is None and flt_expert_ids is not None:
                        local_expert_ids = flt_expert_ids
            return _orig_exos_should_skip(name, local_expert_ids)
    '''

    kbo_patch = '''
        # --- EXOS_KV_BLOCK_OVERRIDE_PATCH ---
        _NUM_BLOCKS_OVERRIDE_TOL = 4
        _NUM_BLOCKS_OVERRIDE_REL_TOL = 0.10
    '''

    rde_patch = '''
        # --- EXOS_RAY_DISTRIBUTED_EXECUTOR_PATCH ---
        import ipaddress as _exos_ipaddress
        import os as _exos_os
        _EXOS_LOCAL_HOST_IP = _exos_os.environ.get("VLLM_HOST_IP", "").strip()
        _EXOS_NODE_LOCAL_ENV_KEYS = (
            "GCE_METADATA_HOST",
            "GCE_METADATA_IP",
            "GCE_METADATA_ROOT",
            "TPU_WORKER_ID",
            "TPU_NODE_ID",
            "CLOUD_TPU_TASK_ID",
            "TPU_VISIBLE_CHIPS",
            "TPU_VISIBLE_DEVICES",
            "VLLM_HOST_IP",
            "VLLM_LOOPBACK_IP",
        )
        def _exos_ip_to_host(ip_val) -> str:
            raw = str(ip_val).strip().strip("[]")
            if ":" not in raw:
                return raw
            try:
                norm = _exos_ipaddress.ip_address(raw).compressed
            except Exception:
                norm = raw
            try:
                if _exos_os.path.exists("/etc/hosts"):
                    job_id = _exos_os.environ.get("CDK_JOB_ID", "").strip()
                    candidates = []
                    with open("/etc/hosts") as f:
                        for line in f:
                            line = line.split("#", 1)[0].strip()
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    entry_ip = _exos_ipaddress.ip_address(parts[0]).compressed
                                except Exception:
                                    entry_ip = parts[0]
                                if entry_ip == norm:
                                    candidates.extend(parts[1:])
                    if candidates:
                        if job_id:
                            for cand in candidates:
                                if cand.startswith(job_id):
                                    return cand
                        for cand in candidates:
                            if cand.startswith("j-") and "borgtask.google.com" not in cand:
                                return cand
                        for cand in candidates:
                            if "borgtask.google.com" not in cand:
                                return cand
                        return candidates[0]
            except Exception:
                pass
            return f"[{norm}]"

        def _exos_bracket_sb(sb: str) -> str:
            if not sb:
                return sb
            parts = []
            for item in sb.split(","):
                h, sep, p = item.rpartition(":")
                if sep and ":" in h:
                    parts.append(f"{_exos_ip_to_host(h)}:{p}")
                else:
                    parts.append(item)
            return ",".join(parts)

        def _exos_clean_env_dict(d, strip_node_local=True):
            if not isinstance(d, dict):
                return d
            d.pop("VLLM_LOOPBACK_IP", None)
            if _exos_os.path.exists("/var/tmp/exos_ipv6_preload.so"):
                _cur_ld = str(d.get("LD_PRELOAD") or _exos_os.environ.get("LD_PRELOAD") or "").strip()
                _ld_parts = [p for p in _cur_ld.split(":") if p and p != "/var/tmp/exos_ipv6_preload.so"]
                d["LD_PRELOAD"] = ":".join(["/var/tmp/exos_ipv6_preload.so"] + _ld_parts)
            d["MALLOC_ARENA_MAX"] = "2"
            d["MALLOC_TRIM_THRESHOLD_"] = "65536"
            d["MALLOC_MMAP_THRESHOLD_"] = "65536"
            d["TCMALLOC_RELEASE_RATE"] = "10"
            if d.get("MASTER_ADDR") in ("::1", "127.0.0.1", "localhost"):
                head_ip = _exos_os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",")[0].strip().strip("[]")
                if head_ip:
                    d["MASTER_ADDR"] = head_ip
            if strip_node_local:
                for k in _EXOS_NODE_LOCAL_ENV_KEYS:
                    d.pop(k, None)
            else:
                for k in ("GCE_METADATA_HOST", "GCE_METADATA_IP", "GCE_METADATA_ROOT", "VLLM_HOST_IP"):
                    d.pop(k, None)
            if "TORCH_TPU_SLICEBUILDER_ADDRESSES" in d:
                d["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = _exos_bracket_sb(
                    d.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "")
                )
            return d

        if "get_driver_env_vars" in globals():
            _orig_get_driver_env_vars = globals()["get_driver_env_vars"]
            if not getattr(_orig_get_driver_env_vars, "_exos_patched", False):
                def _exos_get_driver_env_vars(*args, **kwargs):
                    res = _orig_get_driver_env_vars(*args, **kwargs)
                    return _exos_clean_env_dict(res, strip_node_local=True)
                _exos_get_driver_env_vars._exos_patched = True
                globals()["get_driver_env_vars"] = _exos_get_driver_env_vars

        if "RayWorkerProc" in globals():
            _exos_rwp = globals()["RayWorkerProc"]
            if hasattr(_exos_rwp, "initialize_worker") and not getattr(_exos_rwp.initialize_worker, "_exos_patched", False):
                _orig_rwp_init = _exos_rwp.initialize_worker
                def _exos_rwp_initialize_worker(self, local_rank, env_vars, *args, **kwargs):
                    _exos_os.environ.pop("VLLM_LOOPBACK_IP", None)
                    _exos_clean_env_dict(env_vars, strip_node_local=False)
                    if _EXOS_LOCAL_HOST_IP and _EXOS_LOCAL_HOST_IP not in ("::1", "127.0.0.1", "localhost"):
                        if isinstance(env_vars, dict):
                            env_vars["VLLM_HOST_IP"] = _EXOS_LOCAL_HOST_IP
                        _exos_os.environ["VLLM_HOST_IP"] = _EXOS_LOCAL_HOST_IP
                    new_args = []
                    for a in args:
                        if isinstance(a, dict):
                            _exos_clean_env_dict(a, strip_node_local=True)
                        new_args.append(a)
                    if isinstance(kwargs.get("driver_env_vars"), dict):
                        _exos_clean_env_dict(kwargs["driver_env_vars"], strip_node_local=True)
                    return _orig_rwp_init(self, local_rank, env_vars, *new_args, **kwargs)
                _exos_rwp_initialize_worker._exos_patched = True
                _exos_rwp.initialize_worker = _exos_rwp_initialize_worker

        for _exos_cls_name in ("RayDistributedExecutor", "RayDistributedExecutorV2", "RayExecutorV2"):
            _exos_cls = globals().get(_exos_cls_name)
            if _exos_cls is None:
                continue
            if hasattr(_exos_cls, "_get_env_vars_to_be_updated"):
                _orig_get_env = _exos_cls._get_env_vars_to_be_updated
                def _exos_make_get_env(_orig):
                    def _wrapped(self, *args, **kwargs):
                        res = _orig(self, *args, **kwargs)
                        if isinstance(res, list):
                            for d in res:
                                _exos_clean_env_dict(d, strip_node_local=False)
                        elif isinstance(res, dict):
                            _exos_clean_env_dict(res, strip_node_local=False)
                        return res
                    return _wrapped
                _exos_cls._get_env_vars_to_be_updated = _exos_make_get_env(_orig_get_env)
    '''

    rms_impl_code = '''
        import builtins
        import collections
        import concurrent.futures
        import fcntl
        import fnmatch
        import glob
        import http.client
        import json
        import os
        import re
        import shutil
        import struct
        import threading
        import time
        import urllib.parse
        import urllib.request
        import torch

        _TOKEN_CACHE = {"token": "", "ts": 0.0}

        def _gcs_get_bytes(bucket: str, obj: str, range_header=None, timeout=30, max_attempts=10) -> bytes:
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
            for attempt in range(max_attempts):
                try:
                    extra = {"Range": range_header} if range_header else None
                    req = urllib.request.Request(url, headers=_headers(extra))
                    with urllib.request.urlopen(req, timeout=timeout) as r:
                        return r.read()
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
                    _TOKEN_CACHE["ts"] = 0.0
                    time.sleep(min(10.0, 1.0 * (attempt + 1)))

        def _get_token() -> str:
            now = time.time()
            if _TOKEN_CACHE["token"] and (now - _TOKEN_CACHE["ts"] < 240.0):
                return _TOKEN_CACHE["token"]
            host = os.environ.get("GCE_METADATA_HOST") or "metadata.google.internal"
            for _ in range(3):
                try:
                    req = urllib.request.Request(
                        f"http://{host}/computeMetadata/v1/instance/service-accounts/default/token",
                        headers={"Metadata-Flavor": "Google"},
                    )
                    with urllib.request.urlopen(req, timeout=10) as r:
                        tok = json.loads(r.read().decode())["access_token"]
                        _TOKEN_CACHE["token"] = tok
                        _TOKEN_CACHE["ts"] = now
                        return tok
                except Exception:
                    time.sleep(0.5)
            return _TOKEN_CACHE["token"]

        def _headers(extra=None):
            tok = _get_token()
            h = {"Authorization": f"Bearer {tok}"} if tok else {}
            if extra:
                h.update(extra)
            return h

        def _list_gcs(gcs_uri: str):
            rest = gcs_uri[5:].rstrip("/")
            bucket, _, prefix = rest.partition("/")
            items = []
            page_token = ""
            while True:
                q = f"prefix={urllib.parse.quote(prefix + '/')}&maxResults=1000"
                if page_token:
                    q += f"&pageToken={urllib.parse.quote(page_token)}"
                url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?{q}"
                for attempt in range(4):
                    try:
                        req = urllib.request.Request(url, headers=_headers())
                        with urllib.request.urlopen(req, timeout=30) as r:
                            data = json.loads(r.read().decode())
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        _TOKEN_CACHE["ts"] = 0.0
                        time.sleep(0.5 * (attempt + 1))
                items.extend(data.get("items", []))
                page_token = data.get("nextPageToken", "")
                if not page_token:
                    break
            return bucket, prefix, items

        def pull_files(model_path: str, dst_dir: str, allow_pattern=None, ignore_pattern=None):
            os.makedirs(dst_dir, exist_ok=True)
            lock_path = os.path.join(dst_dir, ".exos_pull.lock")
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                local_dir = model_path
                if str(model_path).startswith("gs://"):
                    bkt, _, pfx = str(model_path)[5:].rstrip("/").partition("/")
                    fuse_dir = os.path.join("/exos/gcs", bkt, pfx) if pfx else os.path.join("/exos/gcs", bkt)
                    if os.path.isdir(fuse_dir):
                        local_dir = fuse_dir
                if not str(local_dir).startswith("gs://"):
                    if os.path.isdir(local_dir):
                        for fname in os.listdir(local_dir):
                            src_f = os.path.join(local_dir, fname)
                            if not os.path.isfile(src_f):
                                continue
                            if allow_pattern and not any(fnmatch.fnmatch(fname, p) for p in allow_pattern):
                                continue
                            if ignore_pattern and any(fnmatch.fnmatch(fname, p) for p in ignore_pattern):
                                continue
                            dst_f = os.path.join(dst_dir, fname)
                            if not os.path.exists(dst_f):
                                shutil.copy2(src_f, dst_f)
                    return
                bucket, prefix, items = _list_gcs(model_path)
                to_fetch = []
                for it in items:
                    name = it["name"]
                    if name.endswith("/"):
                        continue
                    rel = name[len(prefix):].lstrip("/")
                    if not rel or "/" in rel:
                        continue
                    if allow_pattern and not any(fnmatch.fnmatch(rel, p) for p in allow_pattern):
                        continue
                    if ignore_pattern and any(fnmatch.fnmatch(rel, p) for p in ignore_pattern):
                        continue
                    local_p = os.path.join(dst_dir, rel)
                    if os.path.exists(local_p) and os.path.getsize(local_p) > 0:
                        continue
                    to_fetch.append((name, local_p))
                if not to_fetch:
                    return

                def _dl_one(pair):
                    name, local_path = pair
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        return
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    dl_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe='')}?alt=media"
                    tmp_path = f"{local_path}.tmp.{os.getpid()}"
                    for attempt in range(4):
                        try:
                            req = urllib.request.Request(dl_url, headers=_headers())
                            with urllib.request.urlopen(req, timeout=120) as r, open(tmp_path, "wb") as out:
                                while True:
                                    chunk = r.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    out.write(chunk)
                            os.replace(tmp_path, local_path)
                            return
                        except Exception:
                            if attempt == 3:
                                raise
                            _TOKEN_CACHE["ts"] = 0.0
                            time.sleep(0.5 * (attempt + 1))

                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
                    list(ex.map(_dl_one, to_fetch))

        def list_safetensors(path: str) -> list[str]:
            if not str(path).startswith("gs://"):
                local_files = sorted(glob.glob(os.path.join(str(path), "*.safetensors")))
                if local_files:
                    return local_files
                if os.environ.get("MODEL", "").startswith("gs://"):
                    path = os.environ["MODEL"]
                else:
                    return []
            cache_dir = "/tmp/workspace/.exos_hdr_cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_list = os.path.join(cache_dir, urllib.parse.quote(str(path), safe="") + ".list.json")
            with open(os.path.join(cache_dir, ".list.lock"), "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                if os.path.isfile(cache_list) and os.path.getsize(cache_list) > 2:
                    try:
                        with open(cache_list, "r") as rf:
                            return json.load(rf)
                    except Exception:
                        pass
                bucket, prefix, items = _list_gcs(str(path))
                res = []
                for it in items:
                    name = it["name"]
                    rel = name[len(prefix):].lstrip("/")
                    if rel.endswith(".safetensors") and "/" not in rel:
                        res.append(f"gs://{bucket}/{name}")
                res = sorted(res)
                try:
                    tmp_l = f"{cache_list}.tmp.{os.getpid()}"
                    with open(tmp_l, "w") as wf:
                        json.dump(res, wf)
                    os.replace(tmp_l, cache_list)
                except Exception:
                    pass
                return res

        class ObjectStorageModel:
            def __init__(self, model_path: str, dst: str = "", *args, **kwargs):
                self.model_path = str(model_path)
                self.dst = str(dst)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def pull_files(self, allow_pattern=None, ignore_pattern=None):
                if isinstance(allow_pattern, str):
                    allow_pattern = [allow_pattern]
                if isinstance(ignore_pattern, str):
                    ignore_pattern = [ignore_pattern]
                pull_files(self.model_path, self.dst, allow_pattern=allow_pattern, ignore_pattern=ignore_pattern)

        class FileChunks:
            def __init__(self, id: int, path: str, offset: int, chunks: list[int]):
                self.id = int(id)
                self.path = str(path)
                self.offset = int(offset)
                self.chunks = [int(c) for c in chunks]

            @classmethod
            def contiguous(cls, id: int, path: str, offset: int, chunks: list[int]):
                return cls(id, path, offset, chunks)

        class TensorMetadata:
            def __init__(self, name: str, info: dict):
                self.name = name
                self.shape = [int(x) for x in info["shape"]]
                self.dtype = str(info["dtype"])
                self.data_offsets = [int(x) for x in info["data_offsets"]]
                self.nbytes = self.data_offsets[1] - self.data_offsets[0]

        class SafetensorsMetadata:
            def __init__(self, offset: int, tensors_metadata: list[TensorMetadata], read_sizes: list[int]):
                self.offset = offset
                self.tensors_metadata = tensors_metadata
                self.read_sizes = read_sizes

            @staticmethod
            def _read_one_header(path: str):
                if str(path).startswith("gs://"):
                    rest = path[5:]
                    bucket, _, obj = rest.partition("/")
                    cache_dir = "/tmp/workspace/.exos_hdr_cache"
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_f = os.path.join(cache_dir, urllib.parse.quote(f"{bucket}_{obj}", safe="") + ".hdr")
                    hdr_bytes = None
                    if os.path.isfile(cache_f) and os.path.getsize(cache_f) > 8:
                        try:
                            with open(cache_f, "rb") as cf:
                                raw = cf.read()
                            header_size = struct.unpack("<Q", raw[:8])[0]
                            if len(raw) == 8 + header_size:
                                hdr_bytes = raw[8:]
                        except Exception:
                            hdr_bytes = None
                    if hdr_bytes is None:
                        first = _gcs_get_bytes(bucket, obj, range_header="bytes=0-65535", timeout=15)
                        header_size = struct.unpack("<Q", first[:8])[0]
                        if 8 + header_size <= len(first):
                            hdr_bytes = first[8 : 8 + header_size]
                        else:
                            hdr_bytes = _gcs_get_bytes(
                                bucket, obj, range_header=f"bytes=8-{8 + header_size - 1}", timeout=20
                            )
                        try:
                            tmp_c = f"{cache_f}.tmp.{os.getpid()}.{threading.get_ident()}"
                            with open(tmp_c, "wb") as cf:
                                cf.write(struct.pack("<Q", header_size))
                                cf.write(hdr_bytes)
                            os.replace(tmp_c, cache_f)
                        except Exception:
                            pass
                else:
                    with open(path, "rb") as f:
                        header_size = struct.unpack("<Q", f.read(8))[0]
                        hdr_bytes = f.read(header_size)
                hdr = json.loads(hdr_bytes.decode("utf-8"))
                tensors = [TensorMetadata(k, v) for k, v in hdr.items() if k != "__metadata__"]
                tensors.sort(key=lambda t: t.data_offsets[0])
                base_offset = 8 + header_size + (tensors[0].data_offsets[0] if tensors else 0)
                read_sizes = []
                for i, t in enumerate(tensors):
                    if i + 1 < len(tensors):
                        read_sizes.append(tensors[i + 1].data_offsets[0] - t.data_offsets[0])
                    else:
                        read_sizes.append(t.data_offsets[1] - t.data_offsets[0])
                return SafetensorsMetadata(base_offset, tensors, read_sizes)

            @classmethod
            def from_files(cls, fs, hf_weights_files: list[str], credentials=None):
                cache_dir = "/tmp/workspace/.exos_hdr_cache"
                os.makedirs(cache_dir, exist_ok=True)
                with open(os.path.join(cache_dir, ".hdr.lock"), "w") as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                        return list(ex.map(cls._read_one_header, hf_weights_files))

        _DTYPE_MAP = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
            "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8),
            "F8_E5M2": getattr(torch, "float8_e5m2", torch.uint8),
            "U16": getattr(torch, "uint16", torch.int16),
            "U32": getattr(torch, "uint32", torch.int32),
            "U64": getattr(torch, "uint64", torch.int64),
        }

        def create_torch_tensor(buffer, tensor_meta: TensorMetadata) -> torch.Tensor:
            dtype = _DTYPE_MAP[tensor_meta.dtype]
            if tensor_meta.nbytes == 0:
                return torch.empty(tensor_meta.shape, dtype=dtype)
            mv = memoryview(buffer)[:tensor_meta.nbytes]
            if mv.readonly:
                mv = memoryview(bytearray(mv))
            return torch.frombuffer(mv, dtype=dtype).reshape(tensor_meta.shape)

        class FileStreamer:
            MAX_BATCH_BYTES = 16 * 1024 * 1024
            _TLS_LOCAL = threading.local()

            def __init__(self):
                self._batches = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._batches = []

            def stream_files(self, requests: list[FileChunks], credentials=None, device="cpu", is_distributed=False):
                batches = []
                for req in requests:
                    pos = req.offset
                    cur_offset = pos
                    cur_total = 0
                    cur_start_idx = 0
                    cur_chunks = []
                    for idx, csize in enumerate(req.chunks):
                        if cur_chunks and (cur_total + csize > self.MAX_BATCH_BYTES):
                            batches.append((req.id, req.path, cur_offset, cur_total, cur_start_idx, cur_chunks))
                            cur_offset = pos
                            cur_total = 0
                            cur_start_idx = idx
                            cur_chunks = []
                        if not cur_chunks:
                            cur_offset = pos
                            cur_start_idx = idx
                        cur_chunks.append(csize)
                        cur_total += csize
                        pos += csize
                    if cur_chunks:
                        batches.append((req.id, req.path, cur_offset, cur_total, cur_start_idx, cur_chunks))
                self._batches = batches

            @classmethod
            def _fetch_batch(cls, batch):
                req_id, path, offset, total_bytes, start_idx, chunk_sizes = batch
                if total_bytes == 0:
                    return req_id, start_idx, chunk_sizes, bytearray()
                if str(path).startswith("gs://"):
                    rest = path[5:]
                    bucket, _, obj = rest.partition("/")
                    req_path = f"/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
                    end_byte = offset + total_bytes - 1
                    for attempt in range(12):
                        conn = getattr(cls._TLS_LOCAL, "conn", None)
                        try:
                            if conn is None:
                                conn = http.client.HTTPSConnection("storage.googleapis.com", timeout=12)
                                cls._TLS_LOCAL.conn = conn
                            conn.request(
                                "GET",
                                req_path,
                                headers=_headers({"Host": "storage.googleapis.com", "Range": f"bytes={offset}-{end_byte}", "Connection": "keep-alive"}),
                            )
                            r = conn.getresponse()
                            if r.status not in (200, 206):
                                body = r.read(256)
                                raise IOError(f"GCS HTTP {r.status}: {body!r}")
                            buf = bytearray(total_bytes)
                            mv = memoryview(buf)
                            read_pos = 0
                            while read_pos < total_bytes:
                                n = r.readinto(mv[read_pos:])
                                if not n:
                                    break
                                read_pos += n
                            if read_pos != total_bytes:
                                raise IOError(f"Short read on {path}: got {read_pos} of {total_bytes}")
                            return req_id, start_idx, chunk_sizes, buf
                        except Exception:
                            try:
                                if conn is not None:
                                    conn.close()
                            except Exception:
                                pass
                            cls._TLS_LOCAL.conn = None
                            if attempt == 11:
                                raise
                            _TOKEN_CACHE["ts"] = 0.0
                            time.sleep(min(10.0, 1.0 * (attempt + 1)))
                else:
                    buf = bytearray(total_bytes)
                    with open(path, "rb") as f:
                        f.seek(offset)
                        f.readinto(buf)
                    return req_id, start_idx, chunk_sizes, buf

            def get_chunks(self):
                if not self._batches:
                    return
                import ctypes as _rms_ctypes, gc as _rms_gc
                def _rms_trim():
                    try:
                        _rms_gc.collect()
                        _rms_ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass
                    try:
                        _rms_ctypes.CDLL("libtcmalloc.so.4").MallocExtension_ReleaseFreeMemory()
                    except Exception:
                        pass
                max_workers = 8
                window_size = 4
                it = iter(self._batches)
                batch_cnt = 0
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                        q = collections.deque()
                        for _ in range(window_size):
                            b = next(it, None)
                            if b is None:
                                break
                            q.append(ex.submit(self._fetch_batch, b))
                        while q:
                            fut = q.popleft()
                            b = next(it, None)
                            if b is not None:
                                q.append(ex.submit(self._fetch_batch, b))
                            req_id, start_idx, chunk_sizes, buf = fut.result()
                            mv = memoryview(buf)
                            pos = 0
                            n_chunks = len(chunk_sizes)
                            for i, csize in enumerate(chunk_sizes):
                                sub = bytearray(mv[pos : pos + csize]) if n_chunks > 1 else mv[pos : pos + csize]
                                yield req_id, start_idx + i, sub
                                del sub
                                pos += csize
                            del mv
                            del buf
                            batch_cnt += 1
                            if batch_cnt % 64 == 0:
                                _rms_trim()
                finally:
                    self._batches = []
                    _rms_trim()

        _EXOS_RMS_EXPERT_RE = re.compile(r"\\.experts\\.(\\d+)\\.")

        def _exos_should_skip_tensor(name: str, flt=None) -> bool:
            if flt is None:
                flt = getattr(builtins, "_EXOS_MODEL_LOAD_FILTER", None)
            if flt is None:
                return False
            is_mtp, pp_missing, mtp_prefix, local_expert_ids = flt
            if is_mtp:
                if name.startswith("mtp.") or (mtp_prefix and name.startswith(mtp_prefix)) or name.startswith(("model.embed_tokens.", "model.language_model.embed_tokens.", "lm_head.")):
                    return False
                return True
            else:
                if name.startswith("mtp.") or (mtp_prefix and name.startswith(mtp_prefix)) or (pp_missing and name.startswith(pp_missing)):
                    return True
                if local_expert_ids is not None and not name.endswith(".input_scale"):
                    m = _EXOS_RMS_EXPERT_RE.search(name)
                    if m is not None and int(m.group(1)) not in local_expert_ids:
                        return True
                return False

        class SafetensorsStreamer:
            def __init__(self):
                self.file_streamer = FileStreamer()
                self.files_to_tensors_metadata = {}
                self._chunk_to_tensor = {}
                self._zero_tensors = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.file_streamer.__exit__(exc_type, exc_val, exc_tb)
                self.files_to_tensors_metadata = {}
                self._chunk_to_tensor = {}
                self._zero_tensors = []

            def stream_files(self, hf_weights_files: list[str], credentials=None, device="cpu", is_distributed=False):
                metas = SafetensorsMetadata.from_files(self.file_streamer, hf_weights_files, credentials)
                requests = []
                self.files_to_tensors_metadata = {}
                self._chunk_to_tensor = {}
                self._zero_tensors = []
                for path, meta in zip(hf_weights_files, metas):
                    pos = meta.offset
                    run_start = None
                    run_sizes = []
                    run_metas = []
                    kept_metas = []
                    def _flush(p=path):
                        nonlocal run_start, run_sizes, run_metas
                        if run_sizes:
                            req_id = len(requests)
                            requests.append(FileChunks(req_id, p, run_start, run_sizes))
                            for c_idx, tm in enumerate(run_metas):
                                self._chunk_to_tensor[(req_id, c_idx)] = (tm.name, tm)
                        run_start, run_sizes, run_metas = None, [], []
                    for tm, rsize in zip(meta.tensors_metadata, meta.read_sizes):
                        if _exos_should_skip_tensor(tm.name):
                            if rsize:
                                _flush()
                        elif rsize == 0:
                            self._zero_tensors.append(tm)
                            kept_metas.append(tm)
                        else:
                            if run_start is None:
                                run_start = pos
                            run_sizes.append(rsize)
                            run_metas.append(tm)
                            kept_metas.append(tm)
                        pos += rsize
                    _flush()
                    if kept_metas:
                        self.files_to_tensors_metadata[path] = kept_metas
                self.file_streamer.stream_files(requests, credentials, device, is_distributed)

            def get_tensors(self):
                for tm in self._zero_tensors:
                    yield tm.name, create_torch_tensor(memoryview(b""), tm)
                for req_id, chunk_idx, buf in self.file_streamer.get_chunks():
                    name, tm = self._chunk_to_tensor[(req_id, chunk_idx)]
                    yield name, create_torch_tensor(buf, tm)
    '''

    for d in sorted(dirs):
        if os.path.isdir(d):
            rl = os.path.join(d, 'vllm/model_executor/model_loader/runai_streamer_loader.py')
            if os.path.exists(rl):
                rcur = open(rl).read()
                if '_exos_prepare_weights' not in rcur:
                    with open(rl, 'a') as f:
                        f.write('\n' + textwrap.dedent(rsl_patch).strip() + '\n')
                    print('Patched runai_streamer_loader.py in', rl)
            mlp = os.path.join(d, 'vllm_torchtpu/model_loader_patches.py')
            if os.path.exists(mlp):
                mcur = open(mlp).read()
                if 'EXOS_MTP_PP_SKIP_PATCH' not in mcur:
                    with open(mlp, 'a') as f:
                        f.write('\n' + textwrap.dedent(mlp_patch).strip() + '\n')
                    print('Patched vllm_torchtpu/model_loader_patches.py in', mlp)
            kbo = os.path.join(d, 'vllm_torchtpu/executors/kv_block_override.py')
            if os.path.exists(kbo):
                kcur = open(kbo).read()
                if 'EXOS_KV_BLOCK_OVERRIDE_PATCH' not in kcur:
                    with open(kbo, 'a') as f:
                        f.write('\n' + textwrap.dedent(kbo_patch).strip() + '\n')
                    print('Patched vllm_torchtpu/executors/kv_block_override.py in', kbo)
            cm_p = os.path.join(d, 'vllm_torchtpu/distributed/kv_transfer/connector_metadata.py')
            if os.path.exists(cm_p):
                cm_cur = open(cm_p).read()
                if 'EXOS_SENDMETA_CLOCK_PATCH' not in cm_cur:
                    cm_patch = '''
                    # EXOS_SENDMETA_CLOCK_PATCH
                    import time as _exos_cm_time
                    import vllm_torchtpu.distributed.utils as _exos_cm_du
                    def _exos_sendmeta_setstate(self, state):
                        self.__dict__.update(state)
                        _exp = float(getattr(self, 'expiration_time', 0.0) or 0.0)
                        if 0 < _exp < 1e9:
                            self.expiration_time = _exos_cm_time.perf_counter() + float(_exos_cm_du.get_p2p_wait_pull_timeout())
                    SendMeta.__setstate__ = _exos_sendmeta_setstate
                    '''
                    with open(cm_p, 'a') as f:
                        f.write('\n' + textwrap.dedent(cm_patch).strip() + '\n')
                    print('Patched connector_metadata.py in', cm_p)
            tc_p = os.path.join(d, 'vllm_torchtpu/distributed/kv_transfer/tpu_connector.py')
            if os.path.exists(tc_p):
                tc_cur = open(tc_p).read()
                tc_changed = False
                if 'EXOS_STAGE3_EXPIRATION_CLOCK_PATCH' not in tc_cur and 'expiration_time = float(req_meta.expiration_time)' in tc_cur:
                    tc_cur = tc_cur.replace(
                        'expiration_time = float(req_meta.expiration_time)',
                        'expiration_time = (time.perf_counter() + float(dist_utils.get_p2p_wait_pull_timeout())) if 0 < float(req_meta.expiration_time) < 1e9 else float(req_meta.expiration_time)  # EXOS_STAGE3_EXPIRATION_CLOCK_PATCH',
                    )
                    tc_changed = True
                if 'EXOS_STAGE3_FAILED_RECVING_PATCH' not in tc_cur:
                    tc_fail_patch = '''
                    # EXOS_STAGE3_FAILED_RECVING_PATCH
                    if 'TPURaidenConnectorWorker' in globals():
                        _exos_orig_rsl_fail = getattr(TPURaidenConnectorWorker, '_record_stage3_load_failure', None)
                        if _exos_orig_rsl_fail is not None:
                            def _exos_record_stage3_load_failure(self, req_id, block_ids):
                                if getattr(self, '_exos_failed_recving', None) is None:
                                    self._exos_failed_recving = set()
                                self._exos_failed_recving.add(req_id)
                                return _exos_orig_rsl_fail(self, req_id, block_ids)
                            TPURaidenConnectorWorker._record_stage3_load_failure = _exos_record_stage3_load_failure

                        _exos_orig_poll_finished = getattr(TPURaidenConnectorWorker, '_poll_finished', None)
                        if _exos_orig_poll_finished is not None:
                            def _exos_poll_finished(self, engine):
                                _before = set(getattr(self, '_failed_recving', ()))
                                _res = _exos_orig_poll_finished(self, engine)
                                _newly = set(getattr(self, '_failed_recving', ())) - _before
                                if _newly:
                                    if getattr(self, '_exos_failed_recving', None) is None:
                                        self._exos_failed_recving = set()
                                    self._exos_failed_recving.update(_newly)
                                return _res
                            TPURaidenConnectorWorker._poll_finished = _exos_poll_finished

                        _exos_orig_start_load_kv = getattr(TPURaidenConnectorWorker, 'start_load_kv', None)
                        if _exos_orig_start_load_kv is not None:
                            def _exos_start_load_kv(self, *args, **kwargs):
                                _before = set(getattr(self, '_failed_recving', ()))
                                _res = _exos_orig_start_load_kv(self, *args, **kwargs)
                                _newly = set(getattr(self, '_failed_recving', ())) - _before
                                if _newly:
                                    if getattr(self, '_exos_failed_recving', None) is None:
                                        self._exos_failed_recving = set()
                                    self._exos_failed_recving.update(_newly)
                                return _res
                            TPURaidenConnectorWorker.start_load_kv = _exos_start_load_kv

                        def _exos_worker_pop_failed_recving(self):
                            _f = getattr(self, '_exos_failed_recving', None)
                            if not _f:
                                return set()
                            self._exos_failed_recving = set()
                            return _f
                        TPURaidenConnectorWorker.pop_exos_failed_recving = _exos_worker_pop_failed_recving

                    if 'TPUConnector' in globals():
                        def _exos_conn_pop_failed_recving(self):
                            _w = getattr(self, 'connector_worker', None)
                            _fn = getattr(_w, 'pop_exos_failed_recving', None)
                            return _fn() if _fn is not None else set()
                        TPUConnector.pop_exos_failed_recving = _exos_conn_pop_failed_recving
                    '''
                    tc_cur = tc_cur + '\n' + textwrap.dedent(tc_fail_patch).strip() + '\n'
                    tc_changed = True
                if tc_changed:
                    with open(tc_p, 'w') as f:
                        f.write(tc_cur)
                    print('Patched tpu_connector.py in', tc_p)
            tr_p = os.path.join(d, 'vllm_torchtpu/runner/tpu_runner.py')
            if os.path.exists(tr_p):
                tr_cur = open(tr_p).read()
                if 'EXOS_HYBRID_KV_FAILED_RECVING_PATCH' not in tr_cur and 'failed_recving = transfer_results.failed_recving' not in tr_cur:
                    tr_patch = '''
                    # EXOS_HYBRID_KV_FAILED_RECVING_PATCH
                    _EXOS_SUPPORTS_FAILED_RECVING = (
                        'failed_recving' in getattr(KVConnectorOutput, '__dataclass_fields__', {})
                    )
                    _exos_pending_failed_recving = set()

                    def _exos_pop_connector_failed_recving(conn):
                        _out = set()
                        _fn = getattr(conn, 'pop_exos_failed_recving', None)
                        if _fn is not None:
                            _out |= set(_fn() or ())
                        for _child in getattr(conn, '_connectors', ()) or ():
                            _out |= _exos_pop_connector_failed_recving(_child)
                        return _out

                    if (
                        _EXOS_SUPPORTS_FAILED_RECVING
                        and not globals().get('_KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP', False)
                        and 'TPUModelRunner' in globals()
                        and '_build_kv_connector_output' in globals()
                    ):
                        _exos_orig_get_finished_kv = TPUModelRunner.get_finished_kv_transfers
                        _exos_orig_build_kv_output = _build_kv_connector_output

                        def _exos_get_finished_kv_transfers(self, scheduler_output):
                            _res = _exos_orig_get_finished_kv(self, scheduler_output)
                            if len(_res) == 7:
                                (
                                    finished_sending,
                                    finished_recving,
                                    worker_meta,
                                    invalid_block_ids,
                                    _native_failed_recving,
                                    invalid_block_group_index,
                                    kv_connector_stats,
                                ) = _res
                                _has_7 = True
                            else:
                                (
                                    finished_sending,
                                    finished_recving,
                                    worker_meta,
                                    invalid_block_ids,
                                    invalid_block_group_index,
                                    kv_connector_stats,
                                ) = _res
                                _native_failed_recving = None
                                _has_7 = False
                            _num_groups = len(
                                getattr(getattr(self, 'kv_cache_config', None), 'kv_cache_groups', ())
                                or ()
                            )
                            if _num_groups > 1 and has_kv_transfer_group():
                                _failed_reqs = _exos_pop_connector_failed_recving(get_kv_transfer_group())
                                if _failed_reqs:
                                    if _has_7:
                                        _native_failed_recving = set(_native_failed_recving or ()) | _failed_reqs
                                    else:
                                        _exos_pending_failed_recving.update(_failed_reqs)
                                    if finished_recving is None:
                                        finished_recving = set(_failed_reqs)
                                    else:
                                        finished_recving = set(finished_recving) | _failed_reqs
                                if invalid_block_ids:
                                    invalid_block_ids = set()
                                    invalid_block_group_index = None
                            if _has_7:
                                return (
                                    finished_sending,
                                    finished_recving,
                                    worker_meta,
                                    invalid_block_ids,
                                    _native_failed_recving,
                                    invalid_block_group_index,
                                    kv_connector_stats,
                                )
                            return (
                                finished_sending,
                                finished_recving,
                                worker_meta,
                                invalid_block_ids,
                                invalid_block_group_index,
                                kv_connector_stats,
                            )

                        def _exos_build_kv_connector_output(**kwargs):
                            _out = _exos_orig_build_kv_output(**kwargs)
                            if _exos_pending_failed_recving:
                                _out.failed_recving = set(_exos_pending_failed_recving)
                                _exos_pending_failed_recving.clear()
                            return _out

                        TPUModelRunner.get_finished_kv_transfers = _exos_get_finished_kv_transfers
                        _build_kv_connector_output = _exos_build_kv_connector_output
                    '''
                    with open(tr_p, 'a') as f:
                        f.write('\n' + textwrap.dedent(tr_patch).strip() + '\n')
                    print('Patched tpu_runner.py in', tr_p)
            vtt_dir = os.path.join(d, 'vllm_torchtpu')
            if os.path.isdir(vtt_dir):
                pas_dir = os.path.join(vtt_dir, 'core')
                pas_file = os.path.join(pas_dir, 'prefix_aware_scheduler.py')
                if not os.path.exists(pas_file):
                    os.makedirs(pas_dir, exist_ok=True)
                    init_p = os.path.join(pas_dir, '__init__.py')
                    if not os.path.exists(init_p):
                        open(init_p, 'w').close()
                    with open(pas_file, 'w') as f:
                        f.write('from vllm.v1.core.sched.scheduler import Scheduler as PrefixAwareScheduler\n')
                    print('Created fallback vllm_torchtpu.core.prefix_aware_scheduler in', pas_file)
            for rde_rel in (
                'vllm_torchtpu/executors/ray_distributed_executor.py',
                'vllm_torchtpu/executors/ray_distributed_executor_v2.py',
                'vllm/v1/executor/ray_executor_v2.py',
                'vllm/v1/executor/ray_env_utils.py',
                'vllm/v1/executor/ray_distributed_executor.py',
                'vllm/executor/ray_distributed_executor.py',
            ):
                rde_p = os.path.join(d, rde_rel)
                if os.path.exists(rde_p):
                    rde_cur = open(rde_p).read()
                    if '(f"[{ip}]:{base_port + chip}" if ":" in str(ip) and not str(ip).startswith("[") else f"{ip}:{base_port + chip}")' in rde_cur:
                        rde_cur = rde_cur.replace(
                            '(f"[{ip}]:{base_port + chip}" if ":" in str(ip) and not str(ip).startswith("[") else f"{ip}:{base_port + chip}")',
                            'f"{_exos_ip_to_host(ip)}:{base_port + chip}"',
                        )
                        with open(rde_p, 'w') as f:
                            f.write(rde_cur)
                    elif 'f"{ip}:{base_port + chip}"' in rde_cur:
                        rde_cur = rde_cur.replace(
                            'f"{ip}:{base_port + chip}"',
                            'f"{_exos_ip_to_host(ip)}:{base_port + chip}"',
                        )
                        with open(rde_p, 'w') as f:
                            f.write(rde_cur)
                    if 'EXOS_RAY_DISTRIBUTED_EXECUTOR_PATCH' not in rde_cur:
                        with open(rde_p, 'a') as f:
                            f.write('\n' + textwrap.dedent(rde_patch).strip() + '\n')
                        print('Patched ray_distributed_executor in', rde_p)
            rms_dir = os.path.join(d, 'runai_model_streamer')
            if (any(d.endswith(sfx) for sfx in ('site-packages', 'dist-packages')) and (os.path.exists(os.path.join(d, 'vllm')) or os.path.exists(os.path.join(d, 'sglang')))) or os.path.exists(rms_dir):
                os.makedirs(os.path.join(rms_dir, 'file_streamer'), exist_ok=True)
                os.makedirs(os.path.join(rms_dir, 'safetensors_streamer'), exist_ok=True)
                with open(os.path.join(rms_dir, '_impl.py'), 'w') as f:
                    f.write(textwrap.dedent(rms_impl_code).lstrip())
                with open(os.path.join(rms_dir, '__init__.py'), 'w') as f:
                    f.write('from ._impl import SafetensorsStreamer, ObjectStorageModel, list_safetensors, pull_files, FileChunks, FileStreamer\n')
                with open(os.path.join(rms_dir, 'file_streamer/__init__.py'), 'w') as f:
                    f.write('from .._impl import FileChunks, FileStreamer\n')
                with open(os.path.join(rms_dir, 'file_streamer/requests_iterator.py'), 'w') as f:
                    f.write('from .._impl import FileChunks\n')
                with open(os.path.join(rms_dir, 'safetensors_streamer/__init__.py'), 'w') as f:
                    f.write('from .._impl import SafetensorsStreamer, SafetensorsMetadata, TensorMetadata, create_torch_tensor\n')
                with open(os.path.join(rms_dir, 'safetensors_streamer/safetensors_pytorch.py'), 'w') as f:
                    f.write('from .._impl import SafetensorsMetadata, TensorMetadata, create_torch_tensor\n')
                with open(os.path.join(rms_dir, 'safetensors_streamer/safetensors_streamer.py'), 'w') as f:
                    f.write('from .._impl import SafetensorsStreamer\n')
                print('Installed pure-Python runai_model_streamer in', rms_dir)
  |||,

  pyPatchEnv:: {
    _EXOS_PYPATCH: $.pyPatchBody,
  },

  // pyPatchSnippet builds the IPv6 shim (exos.ipv6BuildSnippet), then runs
  // pyPatchRunSnippet. Recipes that pass ipv6Preload=true to exos.tpuJobs()
  // already build the shim in the container preamble, so they can use
  // pyPatchRunSnippet alone.
  pyPatchSnippet:: exos.ipv6BuildSnippet + $.pyPatchRunSnippet,

  pyPatchRunSnippet:: |||
    python3 -c 'import os; exec(compile(os.environ["_EXOS_PYPATCH"], "<exos_pypatch>", "exec"), {"__name__": "__main__"})' || true
    unset _EXOS_PYPATCH

    cat > /tmp/exos_gcs_cp.py <<'PYGCS'
    import concurrent.futures, json, os, shutil, sys, threading, urllib.parse, urllib.request

    def get_token():
        host = os.environ.get('GCE_METADATA_HOST') or 'metadata.google.internal'
        req = urllib.request.Request(
            f'http://{host}/computeMetadata/v1/instance/service-accounts/default/token',
            headers={'Metadata-Flavor': 'Google'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())['access_token']

    def download_prefix(gcs_uri, dst_dir, mode='all'):
        if not gcs_uri.startswith('gs://'):
            return
        rest = gcs_uri[5:].rstrip('/')
        bucket, _, prefix = rest.partition('/')
        fuse_dir = f'/exos/gcs/{bucket}/{prefix}'
        if os.path.isdir(fuse_dir) and os.listdir(fuse_dir):
            copied = 0
            for root, dirs, files in os.walk(fuse_dir):
                rel_root = os.path.relpath(root, fuse_dir)
                if mode == 'tokenizer' and rel_root != '.':
                    dirs[:] = []
                    continue
                for fname in files:
                    if mode == 'tokenizer' and fname.endswith(('.safetensors', '.bin', '.pt', '.ot', '.msgpack')):
                        continue
                    src_p = os.path.join(root, fname)
                    rel_p = fname if rel_root == '.' else os.path.join(rel_root, fname)
                    local_path = os.path.join(dst_dir, rel_p)
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        continue
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    tmp_path = f'{local_path}.tmp.{os.getpid()}'
                    shutil.copy2(src_p, tmp_path)
                    os.replace(tmp_path, local_path)
                    copied += 1
            print(f'exos_gcs_cp: synced {copied} files from {fuse_dir} to {dst_dir} (mode={mode})')
            return
        token = get_token()
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        items = []
        page_token = ''
        while True:
            q = f'prefix={urllib.parse.quote(prefix + "/")}&maxResults=1000'
            if page_token:
                q += f'&pageToken={urllib.parse.quote(page_token)}'
            url = f'https://storage.googleapis.com/storage/v1/b/{bucket}/o?{q}'
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            items.extend(data.get('items', []))
            page_token = data.get('nextPageToken', '')
            if not page_token:
                break
        to_fetch = []
        for it in items:
            name = it['name']
            if name.endswith('/'):
                continue
            rel = name[len(prefix):].lstrip('/')
            if mode == 'tokenizer':
                if '/' in rel or rel.endswith(('.safetensors', '.bin', '.pt', '.ot', '.msgpack')):
                    continue
            to_fetch.append((name, os.path.join(dst_dir, rel)))
        def fetch_one(pair):
            name, local_path = pair
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                return
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            dl_url = f'https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe="")}?alt=media'
            req = urllib.request.Request(dl_url, headers=headers)
            tmp_path = f'{local_path}.tmp.{os.getpid()}.{threading.get_ident()}'
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp_path, 'wb') as out:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(tmp_path, local_path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(fetch_one, to_fetch))
        print(f'exos_gcs_cp: synced {len(to_fetch)} files from {gcs_uri} to {dst_dir} (mode={mode})')

    if __name__ == '__main__':
        mode = sys.argv[1]
        src = sys.argv[2]
        dst = sys.argv[3]
        download_prefix(src, dst, mode)
    PYGCS
  |||,

  // smokeBenchScript runs a single completion request after the vLLM server is
  // healthy on http://${SERVE_URL_HOST}:${SERVE_PORT} and prints EXOS_VLLM_RESULT.
  smokeBenchScript:: |||
    # Smoke completion check
        python3 -u - <<'PY'
    import time, urllib.request
  ||| + exos.stageResultPyHelper + |||

    startup_s = float(os.environ.get('EXOS_STARTUP_S', '0'))
    bench_name = f"{os.environ.get('EXOS_RECIPE', 'exos-vllm')}-smoke"
    body = json.dumps({
        'model': os.environ['SERVED_MODEL_NAME'],
        'prompt': os.environ['PROMPT'],
        'max_tokens': int(os.environ['MAX_TOKENS']),
        'temperature': 0,
    }).encode()
    req = urllib.request.Request('http://' + os.environ['SERVE_URL_HOST'] + ':' + os.environ['SERVE_PORT'] + '/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
      with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    except Exception as e:
      err_res = {'pass': False, 'error': repr(e), 'startup_s': startup_s}
      print('EXOS_VLLM_RESULT ' + json.dumps(err_res))
      publish_cdk_result({'exit_code': 1, 'bench': bench_name, 'metrics': err_res}, append=True)
      sys.exit(1)
    latency_s = round(time.time() - t0, 2)
    text = resp['choices'][0]['text']
    usage = resp.get('usage', {})
    res = {
        'pass': bool(text.strip()),
        'model': resp.get('model'),
        'prompt': os.environ['PROMPT'],
        'completion': text,
        'usage': usage,
        'startup_s': startup_s,
        'request_latency_s': latency_s,
    }
    print('EXOS_VLLM_RESULT ' + json.dumps(res))
    publish_cdk_result({'exit_code': 0 if res['pass'] else 1, 'bench': bench_name, 'metrics': res}, append=True)
    sys.exit(0 if res['pass'] else 1)
    PY
  |||,

  bench8k1kScript:: |||
    # 8k1k synthetic random benchmark
        if [ -d "/root/aiperf-venv" ]; then
          source /root/aiperf-venv/bin/activate
        fi
        TOKENIZER_DIR="${MODEL}"
        if [[ "${MODEL}" == gs://* ]]; then
          TOKENIZER_DIR="/tmp/tokenizer_local"
          mkdir -p "${TOKENIZER_DIR}"
          python3 /tmp/exos_gcs_cp.py tokenizer "${MODEL}" "${TOKENIZER_DIR}" || true
        fi
        CONC="${CONCURRENCY:-32}"
        DEFAULT_PROMPTS=$((10 * CONC))
        if [ "${DEFAULT_PROMPTS}" -lt 160 ]; then
          DEFAULT_PROMPTS=160
        fi
        PROMPTS="${NUM_PROMPTS:-${DEFAULT_PROMPTS}}"
        WARMUPS="${NUM_WARMUPS:-$((2 * CONC))}"
        BENCH_SCRIPT=""
        for candidate in \
          "/root/cloud-devkit/InferenceX/utils/bench_serving/benchmark_serving.py" \
          "/usr/local/bin/benchmark_serving.py" \
          "/workspace/vllm/benchmarks/benchmark_serving.py" \
          "/root/torchtpu-vllm/ThirdParty/vllm/benchmarks/benchmark_serving.py" \
          "/opt/vllm/benchmarks/benchmark_serving.py"; do
          if [ -f "${candidate}" ]; then
            BENCH_SCRIPT="${candidate}"
            break
          fi
        done
        if [ -z "${BENCH_SCRIPT}" ]; then
          BENCH_SCRIPT="$(find /root /workspace /opt -name benchmark_serving.py 2>/dev/null | head -n 1 || true)"
        fi
        RESULT_DIR="/tmp/bench_8k1k_results"
        RESULT_FILE="result_c${CONC}_8k1k.json"
        RESULT_JSON="${RESULT_DIR}/${RESULT_FILE}"
        rm -rf "${RESULT_DIR}" && mkdir -p "${RESULT_DIR}"
        CHAT_FLAG=""
        if [[ "${SERVED_MODEL_NAME}" == *Qwen* ]] && grep -q -- '--use-chat-template' "${BENCH_SCRIPT}" 2>/dev/null; then
          CHAT_FLAG="--use-chat-template"
        fi
        if [ -n "${BENCH_SCRIPT}" ]; then
          echo "trace-start-event: name=run-bench-${ISL:-8192}-${OSL:-1024}-${CONC} ts=$(date +'%s.%N') pid=bench tid=bench"
          python3 -u "${BENCH_SCRIPT}" \
            --backend vllm \
            --base-url "http://${SERVE_URL_HOST}:${SERVE_PORT}" \
            --model "${SERVED_MODEL_NAME}" \
            --tokenizer "${TOKENIZER_DIR}" \
            --dataset-name random \
            --random-input-len "${ISL:-8192}" \
            --random-output-len "${OSL:-1024}" \
            --random-range-ratio "${RANGE_RATIO:-0.8}" \
            --max-concurrency "${CONC}" \
            --num-prompts "${PROMPTS}" \
            --num-warmups "${WARMUPS}" \
            --request-rate inf \
            --ignore-eos \
            --trust-remote-code \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 50,90,95,99 \
            --save-result \
            --result-dir "${RESULT_DIR}" \
            --result-filename "${RESULT_FILE}" \
            ${CHAT_FLAG}
          echo "trace-end-event: name=run-bench-${ISL:-8192}-${OSL:-1024}-${CONC} ts=$(date +'%s.%N') pid=bench tid=bench"
        else
          echo "trace-start-event: name=run-bench-${ISL:-8192}-${OSL:-1024}-${CONC} ts=$(date +'%s.%N') pid=bench tid=bench"
          vllm bench serve \
            --backend vllm \
            --base-url "http://${SERVE_URL_HOST}:${SERVE_PORT}" \
            --model "${SERVED_MODEL_NAME}" \
            --tokenizer "${TOKENIZER_DIR}" \
            --dataset-name random \
            --random-input-len "${ISL:-8192}" \
            --random-output-len "${OSL:-1024}" \
            --random-range-ratio "${RANGE_RATIO:-0.8}" \
            --max-concurrency "${CONC}" \
            --num-prompts "${PROMPTS}" \
            --num-warmups "${WARMUPS}" \
            --request-rate inf \
            --ignore-eos \
            --trust-remote-code \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 50,90,95,99 \
            --save-result \
            --result-dir "${RESULT_DIR}" \
            --result-filename "${RESULT_FILE}"
          echo "trace-end-event: name=run-bench-${ISL:-8192}-${OSL:-1024}-${CONC} ts=$(date +'%s.%N') pid=bench tid=bench"
        fi
        if [ -n "${CDK_OUTPUT_DIR:-}" ]; then
          mkdir -p "${CDK_OUTPUT_DIR}" 2>/dev/null || true
          cp -f "${RESULT_DIR}"/*.json "${CDK_OUTPUT_DIR}/" 2>/dev/null || true
        fi
        python3 -u - <<'PY8K'
    import glob
  ||| + exos.stageResultPyHelper + |||

    res_dir = '/tmp/bench_8k1k_results'
    conc = int(os.environ.get('CONCURRENCY', '32'))
    bench_name = f"{os.environ.get('EXOS_RECIPE', 'exos-8k1k')}-c{conc}"
    files = sorted(f for f in glob.glob(os.path.join(res_dir, '*.json')) if not f.endswith('.meta.json'))
    if not files:
      err_res = {'pass': False, 'error': 'missing 8k1k result file'}
      print('EXOS_VLLM_RESULT ' + json.dumps(err_res))
      publish_cdk_result({'exit_code': 1, 'bench': bench_name, 'metrics': err_res}, append=True)
      sys.exit(1)
    with open(files[-1]) as f:
      d = json.load(f)
    chips = int(os.environ.get('TOTAL_BENCH_CHIPS', os.environ.get('EXPECTED_TPU_DEVICES', '4')))
    median_tpot = float(d.get('median_tpot_ms') or d.get('p50_tpot_ms') or 0.0)
    mean_tpot = float(d.get('mean_tpot_ms') or 0.0)
    total_tput = float(d.get('total_token_throughput') or 0.0)
    out_tput = float(d.get('output_throughput') or 0.0)
    req_tput = float(d.get('request_throughput') or 0.0)
    mean_e2el = float(d.get('mean_e2el_ms') or 0.0)
    median_e2el = float(d.get('median_e2el_ms') or d.get('p50_e2el_ms') or 0.0)
    osl = float(os.environ.get('OSL', '1024'))
    interactivity = (1000.0 / median_tpot) if median_tpot > 0 else 0.0
    tput_per_chip = (total_tput / chips) if chips > 0 else 0.0
    e2e_norm = ((out_tput * 1000.0) / (req_tput * mean_e2el)) if (req_tput > 0 and mean_e2el > 0) else (osl / (mean_e2el / 1000.0) if mean_e2el > 0 else 0.0)
    res = {
        'pass': int(d.get('completed', 0)) > 0 and median_tpot > 0,
        'bench_mode': '8k1k',
        'recipe': os.environ.get('EXOS_RECIPE'),
        'model': os.environ.get('SERVED_MODEL_NAME'),
        'concurrency': conc,
        'num_chips': chips,
        'completed': d.get('completed'),
        'startup_s': float(os.environ.get('EXOS_STARTUP_S', '0')),
        'duration_s': round(float(d.get('duration') or 0.0), 2),
        'total_input_tokens': d.get('total_input_tokens'),
        'total_output_tokens': d.get('total_output_tokens'),
        'request_throughput': round(req_tput, 4),
        'output_throughput': round(out_tput, 2),
        'output_throughput_per_chip': round(out_tput / chips, 2) if chips > 0 else 0.0,
        'total_token_throughput': round(total_tput, 2),
        'token_throughput_per_chip': round(tput_per_chip, 2),
        'interactivity': round(interactivity, 2),
        'e2e_normalized_interactivity': round(e2e_norm, 2),
        'mean_ttft_ms': round(float(d.get('mean_ttft_ms') or 0.0), 2),
        'median_ttft_ms': round(float(d.get('median_ttft_ms') or d.get('p50_ttft_ms') or 0.0), 2),
        'std_ttft_ms': round(float(d.get('std_ttft_ms') or 0.0), 2),
        'p90_ttft_ms': round(float(d.get('p90_ttft_ms') or 0.0), 2),
        'p95_ttft_ms': round(float(d.get('p95_ttft_ms') or 0.0), 2),
        'p99_ttft_ms': round(float(d.get('p99_ttft_ms') or 0.0), 2),
        'mean_tpot_ms': round(mean_tpot, 2),
        'median_tpot_ms': round(median_tpot, 2),
        'std_tpot_ms': round(float(d.get('std_tpot_ms') or 0.0), 2),
        'p90_tpot_ms': round(float(d.get('p90_tpot_ms') or 0.0), 2),
        'p95_tpot_ms': round(float(d.get('p95_tpot_ms') or 0.0), 2),
        'p99_tpot_ms': round(float(d.get('p99_tpot_ms') or 0.0), 2),
        'mean_itl_ms': round(float(d.get('mean_itl_ms') or 0.0), 2),
        'median_itl_ms': round(float(d.get('median_itl_ms') or d.get('p50_itl_ms') or 0.0), 2),
        'std_itl_ms': round(float(d.get('std_itl_ms') or 0.0), 2),
        'p90_itl_ms': round(float(d.get('p90_itl_ms') or 0.0), 2),
        'p95_itl_ms': round(float(d.get('p95_itl_ms') or 0.0), 2),
        'p99_itl_ms': round(float(d.get('p99_itl_ms') or 0.0), 2),
        'mean_e2el_ms': round(mean_e2el, 2),
        'median_e2el_ms': round(median_e2el, 2),
        'std_e2el_ms': round(float(d.get('std_e2el_ms') or 0.0), 2),
        'p90_e2el_ms': round(float(d.get('p90_e2el_ms') or 0.0), 2),
        'p95_e2el_ms': round(float(d.get('p95_e2el_ms') or 0.0), 2),
        'p99_e2el_ms': round(float(d.get('p99_e2el_ms') or 0.0), 2),
    }
    line = 'EXOS_VLLM_RESULT ' + json.dumps(res)
    print(line)
    publish_cdk_result({'exit_code': 0 if res['pass'] else 1, 'bench': bench_name, 'metrics': res}, append=True)
    sys.exit(0 if res['pass'] else 1)
    PY8K
  |||,

  benchAgentxScript:: |||
    # AgentX trace replay benchmark
        if [ -d "/root/aiperf-venv" ]; then
          source /root/aiperf-venv/bin/activate
        fi
        export VLLM_PLUGINS=""
        export VLLM_TARGET_DEVICE="empty"
        export JAX_PLATFORMS="cpu"
        export TPU_PJRT_EVENT_TRACE_ENABLE="0"
        export INFMAX_CONTAINER_WORKSPACE="/root/cloud-devkit/InferenceX"
        export AIPERF_DIR="/root/cloud-devkit/InferenceX/utils/aiperf"
        export AIPERF_VENV="/root/aiperf-venv"
        if [ -f "${INFMAX_CONTAINER_WORKSPACE}/benchmarks/runtime_settings.sh" ]; then
          source "${INFMAX_CONTAINER_WORKSPACE}/benchmarks/runtime_settings.sh"
        fi
        export AIPERF_EXPERIMENTAL_FAST="${FAST:-1}"
        if [ "${AIPERF_EXPERIMENTAL_FAST}" = "true" ]; then
          export AIPERF_EXPERIMENTAL_FAST="1"
        elif [ "${AIPERF_EXPERIMENTAL_FAST}" = "false" ]; then
          export AIPERF_EXPERIMENTAL_FAST="0"
        fi
        export DURATION="${DURATION:-600}"
        TOKENIZER_DIR="${MODEL}"
        if [[ "${MODEL}" == gs://* ]]; then
          TOKENIZER_DIR="/tmp/tokenizer_local"
          mkdir -p "${TOKENIZER_DIR}"
          python3 /tmp/exos_gcs_cp.py tokenizer "${MODEL}" "${TOKENIZER_DIR}" || true
        fi
        if [[ "${SERVED_MODEL_NAME}" == *Qwen* ]] || [[ "${SERVED_MODEL_NAME}" == *qwen* ]] || [[ "${WEKA_LOADER_OVERRIDE:-}" == *256k* ]]; then
          export WEKA_LOADER_OVERRIDE="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126_256k}"
          RESULT_PREFIX="qwen3.5_397b_tpu_v7x_ix_conc_"
          DATASET_GCS_SUFFIX="cc-traces-weka-062126-256k"
        elif [[ "${SERVED_MODEL_NAME}" == *GLM* ]] || [[ "${SERVED_MODEL_NAME}" == *glm* ]]; then
          export WEKA_LOADER_OVERRIDE="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126}"
          RESULT_PREFIX="glm5.2_fp8_tpu_v7x_ix_conc_"
          DATASET_GCS_SUFFIX="cc-traces-weka-062126"
        else
          export WEKA_LOADER_OVERRIDE="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126}"
          RESULT_PREFIX="${EXOS_RECIPE//-/_}_conc_"
          DATASET_GCS_SUFFIX="cc-traces-weka-062126"
        fi
        if [ ! -d "/root/.cache/huggingface/hub/datasets--semianalysisai--${DATASET_GCS_SUFFIX}" ]; then
          mkdir -p /root/.cache/huggingface
          python3 /tmp/exos_gcs_cp.py all "gs://${EXOS_BUCKET:-cloud-devkit}/datasets/hf_cache/semianalysisai/${DATASET_GCS_SUFFIX}" /root/.cache/huggingface || true
        fi
        export MODEL="${TOKENIZER_DIR}"
        # aiperf's offline _resolve_local_snapshot passes MODEL directly to
        # huggingface_hub.snapshot_download(), which rejects local directory
        # paths like /tmp/tokenizer_local with HFValidationError. Similarly,
        # BaseHFDatasetLoader._load_hf_dataset passes the repo ID directly to
        # datasets.load_dataset(), which queries the HuggingFace Hub API and
        # fails under HF_HUB_OFFLINE=1 unless resolved via snapshot_download(local_files_only=True).
        python3 - <<'PYTOK'
    p = '/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/common/tokenizer.py'
    try:
      with open(p) as f:
        s = f.read()
      target = '    def _resolve_local_snapshot(cls, name: str, revision: str) -> str:\n'
      patch = target + '        if __import__("os").path.isdir(name):\n            return name\n'
      if target in s and 'isdir(name)' not in s:
        with open(p, 'w') as f:
          f.write(s.replace(target, patch, 1))
    except Exception:
      pass
    p2 = '/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/dataset/loader/base_hf_dataset.py'
    try:
      with open(p2) as f:
        s2 = f.read()
      t2 = '        return hf_load_dataset(\n            self.hf_dataset_name,'
      r2 = (
          '        ds_path = self.hf_dataset_name\n'
          '        try:\n'
          '            from huggingface_hub import snapshot_download\n'
          '            ds_path = snapshot_download(self.hf_dataset_name, repo_type="dataset", local_files_only=True, **revision)\n'
          '        except Exception:\n'
          '            pass\n'
          '        return hf_load_dataset(\n'
          '            ds_path,'
      )
      if t2 in s2:
        with open(p2, 'w') as f:
          f.write(s2.replace(t2, r2, 1))
    except Exception:
      pass
    PYTOK
        source "${INFMAX_CONTAINER_WORKSPACE}/benchmarks/benchmark_lib.sh" 2>/dev/null || true
        ensure_aiperf_venv() { return 0; }
        setup_aiperf_venv() { return 0; }
        # CLIENT_IMAGE bakes aiperf and hf into /root/aiperf-venv, whereas newer
        # benchmark_lib.sh resets AIPERF_VENV to /tmp/inferencex-agentic-$$/venv
        # and tries to reinstall it with uv over the internet.
        if [ -x "/root/aiperf-venv/bin/aiperf" ]; then
          export AIPERF_VENV="/root/aiperf-venv"
          export AIPERF_PYTHON="/root/aiperf-venv/bin/python"
          export AIPERF_CLI="/root/aiperf-venv/bin/aiperf"
          export AIPERF_HF_CLI="/root/aiperf-venv/bin/hf"
          AIPERF_DEPS_READY=1
        fi
        resolve_trace_source 2>/dev/null || true
        if [ -f "/root/aiperf-venv/bin/activate" ]; then
          source /root/aiperf-venv/bin/activate
        fi
        export PORT="${SERVE_PORT}"
        export VLLM_HOST="${SERVE_URL_HOST}"
        export VLLM_URL="http://${SERVE_URL_HOST}:${SERVE_PORT}"
        export METRICS_URL="${METRICS_URL:-${VLLM_URL}/metrics}"
        export AIPERF_HTTP_IP_VERSION="${AIPERF_HTTP_IP_VERSION:-auto}"
        export AIPERF_SERVER_URL="${VLLM_URL}"
        export AIPERF_SERVER_METRICS_URLS="${METRICS_URL}"
        export KV_OFFLOADING="none"
        export PRECISION="fp8"
        export FRAMEWORK="vllm"
        export IS_MULTINODE="false"
        export TP="${TP:-8}"
        export EP_SIZE="${DP:-1}"
        export PP_SIZE="1"
        export PCP_SIZE="1"
        export DCP="${DECODE_CONTEXT_PARALLEL_SIZE:-1}"
        export DCP_SIZE="${DECODE_CONTEXT_PARALLEL_SIZE:-1}"
        export REQUIRE_POWER="0"
        export AIPERF_WARMUP_REQUESTS_PER_LANE="${WARMUP_REQUESTS_PER_LANE:-1}"
        export AIPERF_TRACE_IDLE_GAP_CAP_SECONDS="${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-10}"
        export AIPERF_UNSAFE_OVERRIDE="${AIPERF_UNSAFE_OVERRIDE:-false}"
        export PARENT_RESULT_DIR="/tmp/agentx_results"
        export AGENTIC_OUTPUT_DIR="${PARENT_RESULT_DIR}"
        export CONC="${CONCURRENCY:-16}"
        export RESULT_DIR="${PARENT_RESULT_DIR}/conc_${CONC}"
        export RESULT_FILENAME="${RESULT_PREFIX}${CONC}"
        rm -rf "${PARENT_RESULT_DIR}" && mkdir -p "${RESULT_DIR}"

        vllm_metric() {
          local v
          v=$(curl -sf "${METRICS_URL}" 2>/dev/null | awk -v n="vllm:$1" \
            '{k=$1; sub(/\{.*/, "", k); if (k==n) s+=$NF} END {printf "%.0f", s+0}') || v=""
          case "$v" in ''|*[!0-9]*) echo 0 ;; *) echo "$v" ;; esac
        }
        vllm_spec_metric() {
          local v
          v=$(curl -sf "${DECODE_METRICS_URL:-${METRICS_URL}}" 2>/dev/null | awk -v n="vllm:$1" \
            '{k=$1; sub(/\{.*/, "", k); if (k==n) s+=$NF} END {printf "%.0f", s+0}') || v=""
          case "$v" in ''|*[!0-9]*) echo 0 ;; *) echo "$v" ;; esac
        }
        vllm_metric_ms() {
          local v
          v=$(curl -sf "${METRICS_URL}" 2>/dev/null | awk -v n="vllm:$1" \
            '{k=$1; sub(/\{.*/, "", k); if (k==n) s+=$NF} END {printf "%.0f", (s+0)*1000}') || v=""
          case "$v" in ''|*[!0-9]*) echo 0 ;; *) echo "$v" ;; esac
        }

        pc_h0=$(vllm_metric prefix_cache_hits_total)
        pc_q0=$(vllm_metric prefix_cache_queries_total)
        sd_d0=$(vllm_spec_metric spec_decode_num_drafts_total)
        sd_dt0=$(vllm_spec_metric spec_decode_num_draft_tokens_total)
        sd_at0=$(vllm_spec_metric spec_decode_num_accepted_tokens_total)
        qt_s0=$(vllm_metric_ms request_queue_time_seconds_sum)
        qt_c0=$(vllm_metric request_queue_time_seconds_count)
        pt_s0=$(vllm_metric_ms request_prefill_time_seconds_sum)
        pt_c0=$(vllm_metric request_prefill_time_seconds_count)
        it_s0=$(vllm_metric_ms request_inference_time_seconds_sum)
        it_c0=$(vllm_metric request_inference_time_seconds_count)

        build_replay_cmd "${RESULT_DIR}"
        REPLAY_CMD="$(printf '%s' "$REPLAY_CMD" | sed -E "s#(https?://)(localhost|127\.0\.0\.1|0\.0\.0\.0):[0-9]+#\1${VLLM_HOST}:${PORT}#g")"
        REPLAY_CMD=$(echo "$REPLAY_CMD" | sed -E "s/--benchmark-duration [0-9]+/--benchmark-duration ${DURATION}/")
        REPLAY_CMD="${REPLAY_CMD/--trajectory-start-min-ratio 0.25/--trajectory-start-min-ratio ${TRAJECTORY_START_MIN_RATIO:-0.25}}"
        REPLAY_CMD="${REPLAY_CMD/--trajectory-start-max-ratio 0.75/--trajectory-start-max-ratio ${TRAJECTORY_START_MAX_RATIO:-0.75}}"
        REPLAY_CMD="${REPLAY_CMD/--warmup-requests-per-lane 2/--warmup-requests-per-lane ${AIPERF_WARMUP_REQUESTS_PER_LANE}}"
        if [ -n "${MAX_CONTEXT_LENGTH:-}" ]; then
          REPLAY_CMD="${REPLAY_CMD} --max-context-length ${MAX_CONTEXT_LENGTH}"
        fi
        if [ -n "${FAILED_REQUEST_THRESHOLD:-}" ]; then
          case "$REPLAY_CMD" in
            *--failed-request-threshold*)
              REPLAY_CMD="$(printf '%s' "$REPLAY_CMD" | sed -E "s/--failed-request-threshold[= ][0-9.]+/--failed-request-threshold ${FAILED_REQUEST_THRESHOLD}/g")" ;;
            *)
              REPLAY_CMD="${REPLAY_CMD} --failed-request-threshold ${FAILED_REQUEST_THRESHOLD}" ;;
          esac
        fi
        if [ "${DURATION}" -lt 900 ] || [ "${AIPERF_UNSAFE_OVERRIDE}" = "true" ]; then
          if [[ "$REPLAY_CMD" != *"--unsafe-override"* ]]; then
            REPLAY_CMD="$REPLAY_CMD --unsafe-override"
          fi
        fi
        # run_agentic_replay_and_write_outputs re-invokes build_replay_cmd internally,
        # which would clobber the URL, duration, warmup, and --unsafe-override rewrites above.
        build_replay_cmd() { return 0; }
        # Same coverage relaxations as the Cloud agentx recipes: a lower ratio only for short or
        # overridden runs, and coverage errors become non-fatal only with AIPERF_UNSAFE_OVERRIDE.
        if [ "${AIPERF_UNSAFE_OVERRIDE}" = "true" ] || [ "${DURATION}" -lt 3600 ]; then
          sed -i 's/minimum_profile_metric_coverage_ratio=0.95/minimum_profile_metric_coverage_ratio=0.75/g' /root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/common/scenario/inferencex_agentx_mvp.py 2>/dev/null || true
        fi
        if [ "${AIPERF_UNSAFE_OVERRIDE}" = "true" ]; then
          sed -i 's/fatal_errors.append(ErrorDetails(type="ProfileMetricCoverageError"/# fatal_errors.append(ErrorDetails(type="ProfileMetricCoverageError"/g' /root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/records/records_manager.py 2>/dev/null || true
        fi
        echo "=== ${EXOS_RECIPE}: running AgentX REPLAY_CMD=${REPLAY_CMD} ==="
        set +e
        echo "trace-start-event: name=benchmark-agentx-c${CONC} ts=$(date +'%s.%N') pid=bench tid=bench"
        run_agentic_replay_and_write_outputs "${RESULT_DIR}"
        bench_exit=$?
        export BENCH_EXIT="${bench_exit}"
        echo "trace-end-event: name=benchmark-agentx-c${CONC} ts=$(date +'%s.%N') pid=bench tid=bench"

        export PREFIX_CACHE_HITS_DELTA=$(( $(vllm_metric prefix_cache_hits_total) - pc_h0 ))
        export PREFIX_CACHE_QUERIES_DELTA=$(( $(vllm_metric prefix_cache_queries_total) - pc_q0 ))
        export SPEC_DECODE_DRAFTS_DELTA=$(( $(vllm_spec_metric spec_decode_num_drafts_total) - sd_d0 ))
        export SPEC_DECODE_DRAFT_TOKENS_DELTA=$(( $(vllm_spec_metric spec_decode_num_draft_tokens_total) - sd_dt0 ))
        export SPEC_DECODE_ACCEPTED_TOKENS_DELTA=$(( $(vllm_spec_metric spec_decode_num_accepted_tokens_total) - sd_at0 ))
        export QUEUE_TIME_SUM_MS_DELTA=$(( $(vllm_metric_ms request_queue_time_seconds_sum) - qt_s0 ))
        export QUEUE_TIME_COUNT_DELTA=$(( $(vllm_metric request_queue_time_seconds_count) - qt_c0 ))
        export PREFILL_TIME_SUM_MS_DELTA=$(( $(vllm_metric_ms request_prefill_time_seconds_sum) - pt_s0 ))
        export PREFILL_TIME_COUNT_DELTA=$(( $(vllm_metric request_prefill_time_seconds_count) - pt_c0 ))
        export INFERENCE_TIME_SUM_MS_DELTA=$(( $(vllm_metric_ms request_inference_time_seconds_sum) - it_s0 ))
        export INFERENCE_TIME_COUNT_DELTA=$(( $(vllm_metric request_inference_time_seconds_count) - it_c0 ))

        python3 -u - "${AGENTIC_OUTPUT_DIR}/${RESULT_FILENAME}.json" "${RESULT_DIR}/aiperf_artifacts" <<'PYAX'
    import math, shutil
  ||| + exos.stageResultPyHelper + |||

    def load(path):
      try:
        with open(path) as f:
          return json.load(f)
      except Exception:
        return {}

    def dig(d, *keys):
      cur = d
      for k in keys:
        if not isinstance(cur, dict):
          return None
        cur = cur.get(k)
      return cur

    def num(v):
      if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
      return v if math.isfinite(v) else None

    def put(out, key, value, scale=None):
      v = num(value)
      if v is None:
        return
      out[key] = round(v * scale, 4) if scale else v

    def resolve_aiperf(artifact_dir):
      direct = os.path.join(artifact_dir, 'profile_export_aiperf.json')
      if os.path.isfile(direct):
        return direct
      try:
        for child in sorted(os.listdir(artifact_dir)):
          nested = os.path.join(artifact_dir, child, 'profile_export_aiperf.json')
          if os.path.isfile(nested):
            return nested
      except OSError:
        pass
      return direct

    agg = load(sys.argv[1])
    aiperf = load(resolve_aiperf(sys.argv[2])) if len(sys.argv) > 2 else {}
    out = {
        'bench_mode': 'agentx',
        'recipe': os.environ.get('EXOS_RECIPE'),
        'model': os.environ.get('SERVED_MODEL_NAME'),
        'startup_s': float(os.environ.get('EXOS_STARTUP_S', '0')),
    }
    for key in ('conc', 'num_requests_total', 'num_requests_successful', 'kv_cache_pool_tokens', 'tp', 'ep'):
      put(out, key, agg.get(key))
    put(out, 'records_dropped_total', dig(agg, 'request_accounting', 'records_dropped_total'))
    put(out, 'records_error_dropped', dig(agg, 'request_accounting', 'records_error_dropped'))

    for src, dst in (('mean', 'mean'), ('p50', 'median'), ('p90', 'p90'), ('p95', 'p95')):
      for metric in ('ttft', 'tpot', 'itl', 'e2el'):
        put(out, dst + '_' + metric + '_ms', dig(agg, 'request_metrics', 'latency', metric, src), 1000.0)
      put(out, dst + '_interactivity_tok_s', dig(agg, 'request_metrics', 'latency', 'intvty', src))
      put(out, dst + '_qps', dig(agg, 'request_metrics', 'qps', src))

    for src, dst in (('mean', 'mean'), ('p50', 'median'), ('p75', 'p75'), ('p90', 'p90'), ('p95', 'p95')):
      put(out, dst + '_e2e_norm_interactivity_tok_s', dig(agg, 'request_metrics', 'latency', 'e2e_norm_intvty', src))

    tput = dig(agg, 'request_metrics', 'throughput') or {}
    put(out, 'input_throughput', dig(tput, 'input', 'tokens_per_second'))
    put(out, 'output_throughput', dig(tput, 'output', 'tokens_per_second'))
    put(out, 'total_token_throughput', dig(tput, 'total', 'tokens_per_second'))
    put(out, 'benchmark_duration_s', tput.get('duration_seconds'))
    put(out, 'request_throughput', dig(agg, 'request_metrics', 'qps', 'mean'))
    put(out, 'mean_input_tokens', dig(agg, 'request_metrics', 'tokens', 'input', 'mean'))
    put(out, 'mean_output_tokens', dig(agg, 'request_metrics', 'tokens', 'output_actual', 'mean'))
    put(out, 'theoretical_cache_hit_rate', dig(agg, 'request_metrics', 'cache', 'theoretical_cache_hit_rate'))

    kv_hit_rate = dig(agg, 'server_metrics', 'cache', 'overall_cache_hit_rate')
    if kv_hit_rate is None:
      kv_hit_rate = dig(agg, 'server_metrics', 'cache', 'gpu_cache_hit_rate')
    put(out, 'kv_cache_hit_rate', kv_hit_rate)
    put(out, 'gpu_cache_hit_rate', dig(agg, 'server_metrics', 'cache', 'gpu_cache_hit_rate'))
    put(out, 'prefix_cache_hit_rate', kv_hit_rate)
    put(out, 'prefix_cache_hits', dig(agg, 'server_metrics', 'cache', 'prefix_cache_hits'))
    put(out, 'prefix_cache_queries', dig(agg, 'server_metrics', 'cache', 'prefix_cache_queries'))

    def env_int(name):
      try:
        return int(os.environ.get(name, '') or 0)
      except ValueError:
        return 0

    hits = env_int('PREFIX_CACHE_HITS_DELTA')
    queries = env_int('PREFIX_CACHE_QUERIES_DELTA')
    if queries > 0:
      put(out, 'prefix_cache_hits', hits)
      put(out, 'prefix_cache_queries', queries)
      put(out, 'prefix_cache_hit_rate', round(hits / queries, 4))
      put(out, 'kv_cache_hit_rate', round(hits / queries, 4))

    drafts = env_int('SPEC_DECODE_DRAFTS_DELTA')
    draft_tokens = env_int('SPEC_DECODE_DRAFT_TOKENS_DELTA')
    accepted_tokens = env_int('SPEC_DECODE_ACCEPTED_TOKENS_DELTA')
    if drafts > 0:
      put(out, 'spec_decode_drafts', drafts)
      put(out, 'spec_decode_draft_tokens', draft_tokens)
      put(out, 'spec_decode_accepted_tokens', accepted_tokens)
      put(out, 'spec_decode_mean_acceptance_length', round(1 + accepted_tokens / drafts, 4))
      if draft_tokens > 0:
        put(out, 'spec_decode_acceptance_rate', round(accepted_tokens / draft_tokens, 4))

    timing_samples = 0
    for key, env_prefix in (('queue', 'QUEUE_TIME'), ('prefill', 'PREFILL_TIME'), ('inference', 'INFERENCE_TIME')):
      count = env_int(env_prefix + '_COUNT_DELTA')
      if count > 0:
        put(out, 'mean_' + key + '_time_ms', round(env_int(env_prefix + '_SUM_MS_DELTA') / count, 4))
        timing_samples = max(timing_samples, count)
    if timing_samples:
      out['request_timing_samples'] = timing_samples

    for key, path in (
        ('mean_ttft_ms', ('time_to_first_token', 'avg')),
        ('median_ttft_ms', ('time_to_first_token', 'p50')),
        ('p90_ttft_ms', ('time_to_first_token', 'p90')),
        ('p95_ttft_ms', ('time_to_first_token', 'p95')),
        ('mean_itl_ms', ('inter_token_latency', 'avg')),
        ('median_itl_ms', ('inter_token_latency', 'p50')),
        ('mean_tpot_ms', ('inter_token_latency', 'avg')),
        ('median_tpot_ms', ('inter_token_latency', 'p50')),
        ('mean_e2el_ms', ('request_latency', 'avg')),
        ('median_e2el_ms', ('request_latency', 'p50')),
        ('output_throughput', ('output_token_throughput', 'avg')),
        ('request_throughput', ('request_throughput', 'avg')),
        ('benchmark_duration_s', ('benchmark_duration', 'avg')),
    ):
      if key not in out:
        put(out, key, dig(aiperf, *path))

    if 'total_token_throughput' not in out:
      duration = num(out.get('benchmark_duration_s'))
      isl = num(dig(aiperf, 'input_sequence_length', 'avg'))
      osl = num(dig(aiperf, 'output_sequence_length', 'avg'))
      requests = num(dig(aiperf, 'request_count', 'avg'))
      if duration and requests and isl is not None and osl is not None:
        out['total_token_throughput'] = round(requests * (isl + osl) / duration, 4)

    chips = int(os.environ.get('TOTAL_BENCH_CHIPS', os.environ.get('EXPECTED_TPU_DEVICES', '4')))
    total = num(out.get('total_token_throughput'))
    output = num(out.get('output_throughput'))
    if chips > 0:
      out['num_chips'] = chips
      if total is not None:
        out['token_throughput_per_chip'] = round(total / chips, 4)
      if output is not None:
        out['output_throughput_per_chip'] = round(output / chips, 4)

    median_tpot = num(out.get('median_tpot_ms'))
    if median_tpot:
      out['interactivity'] = round(1000.0 / median_tpot, 4)

    conc = int(os.environ.get('CONC', os.environ.get('CONCURRENCY', '16')))
    out['conc'] = conc
    out['max_concurrency'] = conc
    try:
      bench_exit = int(os.environ.get('BENCH_EXIT') or '1')
    except ValueError:
      bench_exit = 1
    out['bench_exit'] = bench_exit
    out['pass'] = (bench_exit == 0) and bool(median_tpot and median_tpot > 0)

    line = 'EXOS_VLLM_RESULT ' + json.dumps(out)
    print(line)
    out_dir = os.environ.get('CDK_OUTPUT_DIR', '').strip()
    if out_dir:
      try:
        conc_out_dir = os.path.join(out_dir, f'conc_{conc}')
        os.makedirs(conc_out_dir, exist_ok=True)
        if os.path.isfile(sys.argv[1]):
          shutil.copy2(sys.argv[1], os.path.join(out_dir, os.path.basename(sys.argv[1])))
        bench_log = os.path.join(os.environ.get('RESULT_DIR', ''), 'benchmark.log')
        if os.path.isfile(bench_log):
          shutil.copy2(bench_log, os.path.join(conc_out_dir, 'benchmark.log'))
        if len(sys.argv) > 2:
          aiperf_json = resolve_aiperf(sys.argv[2])
          if os.path.isfile(aiperf_json):
            aiperf_out = os.path.join(conc_out_dir, 'aiperf_artifacts')
            os.makedirs(aiperf_out, exist_ok=True)
            shutil.copy2(aiperf_json, os.path.join(aiperf_out, 'profile_export_aiperf.json'))
      except OSError as exc:
        print(f'WARN: failed copying AgentX artifacts to {out_dir}: {exc}', file=sys.stderr)
    publish_cdk_result({'exit_code': 0 if out['pass'] else 1, 'bench': f"{os.environ.get('EXOS_RECIPE', 'exos-agentx')}-c{conc}", 'metrics': out}, append=True)
    sys.exit(0 if out['pass'] else 1)
    PYAX
  |||,

  // benchScriptFor returns only the benchmark script branch needed for `mode`
  // so container argv[2] stays well below Linux's 131,072-byte MAX_ARG_STRLEN.
  benchScriptFor(mode='')::
    if mode == 'smoke' then self.smokeBenchScript
    else if mode == '8k1k' then self.bench8k1kScript
    else if mode == 'agentx' then self.benchAgentxScript
    else
      'case "${BENCH_MODE:-smoke}" in\n  smoke)\n'
      + self.smokeBenchScript
      + '    ;;\n  8k1k)\n'
      + self.bench8k1kScript
      + '    ;;\n  agentx)\n'
      + self.benchAgentxScript
      + '    ;;\n  *)\n    echo "EXOS_VLLM_RESULT {\\"pass\\": false, \\"error\\": \\"unknown BENCH_MODE=${BENCH_MODE}\\"}"\n    exit 1\n    ;;\nesac\n',

  // benchScript runs after the vLLM server (or disagg proxy) is healthy
  // on http://${SERVE_URL_HOST}:${SERVE_PORT} and prints an EXOS_VLLM_RESULT
  // JSON line to stdout.
  benchScript:: self.benchScriptFor(''),

  // sidecarBenchContainer returns the sidecar-bench container that runs inside
  // the head worker pod alongside the vLLM container when BENCH_MODE != 'smoke'.
  sidecarBenchContainer(clientImage, commonEnv)::
    local sidecarScript = |||
      set -uo pipefail
      WORKER_ID="${TPU_WORKER_ID:-0}"
      if [ "${PD_LEG:-}" = "decode" ] || [ "${WORKER_ID}" != "0" ]; then
        echo "=== ${EXOS_RECIPE} sidecar-bench: worker=${WORKER_ID} leg=${PD_LEG:-worker} is non-head; exiting ==="
        exit 0
      fi
      trap 'rc=$?; if [ -n "${SYNC_DIR:-}" ] && [ ! -f "${SYNC_DIR}/done" ]; then echo "$rc" > "${SYNC_DIR}/done" 2>/dev/null || true; fi; echo "=== ${EXOS_RECIPE} sidecar-bench: exiting rc=$rc ==="; sleep ${EXIT_LINGER_S:-60}' EXIT
    ||| + exos.stageResultUpdaterSh + self.pyPatchSnippet + |||
      export SYNC_DIR="/tmp/workspace/sync_${JS_NAME:-run}"
      mkdir -p "${SYNC_DIR}"
      # Pre-stage tokenizer and AgentX dataset cache while the vLLM container is compiling/loading weights.
      if [[ "${MODEL:-}" == gs://* ]]; then
        mkdir -p /tmp/tokenizer_local
        python3 /tmp/exos_gcs_cp.py tokenizer "${MODEL}" /tmp/tokenizer_local || true
      fi
      if [ "${BENCH_MODE:-}" = "agentx" ]; then
        mkdir -p /root/.cache/huggingface
        if [[ "${SERVED_MODEL_NAME:-}" == *Qwen* ]] || [[ "${SERVED_MODEL_NAME:-}" == *qwen* ]] || [[ "${WEKA_LOADER_OVERRIDE:-}" == *256k* ]]; then
          DATASET_GCS_SUFFIX="cc-traces-weka-062126-256k"
        else
          DATASET_GCS_SUFFIX="cc-traces-weka-062126"
        fi
        python3 /tmp/exos_gcs_cp.py all "gs://${EXOS_BUCKET:-cloud-devkit}/datasets/hf_cache/semianalysisai/${DATASET_GCS_SUFFIX}" /root/.cache/huggingface || true
      fi
      echo "=== ${EXOS_RECIPE} sidecar-bench: waiting for ${SYNC_DIR}/ready ==="
      while [ ! -f "${SYNC_DIR}/ready" ] && [ ! -f "${SYNC_DIR}/failed" ]; do
        sleep 5
      done
      if [ -f "${SYNC_DIR}/failed" ]; then
        echo "=== ${EXOS_RECIPE} sidecar-bench: vLLM server failed before ready ==="
        echo "1" > "${SYNC_DIR}/done"
        exit 1
      fi
      set -a
      . "${SYNC_DIR}/ready"
      set +a
      echo "=== ${EXOS_RECIPE} sidecar-bench: server ready at http://${SERVE_URL_HOST}:${SERVE_PORT} (startup=${EXOS_STARTUP_S}s) ==="
    ||| + exos.ipv6PreloadSnippet + self.benchScriptFor(if std.objectHas(commonEnv, 'BENCH_MODE') then commonEnv.BENCH_MODE else '') + |||
      RC=$?
      echo "${RC}" > "${SYNC_DIR}/done"
      exit "${RC}"
    |||;
    cdk.container(
      name='sidecar-bench',
      image=clientImage,
      script=sidecarScript,
      env=cdk.jobSetNameEnv + self.pyPatchEnv + commonEnv,
      volumeMounts=[],
      pullPolicy=cdk.var('PULL_POLICY', 'Always'),
    ) + {
      workingDir: '/',
    },

  // recipe returns the JobSet for recipe `name` with `overrides` merged over
  // `defaults`.
  recipe(name, overrides={})::
    local d = self.defaults + overrides;

    local platform = cdk.var('PLATFORM', d.PLATFORM);
    local slice = cdk.var('SLICE', d.SLICE);
    local prefillRunnerGroupId = cdk.var('PREFILL_RUNNER_GROUP_ID', '');
    local runnerGroupId = if prefillRunnerGroupId != '' then prefillRunnerGroupId else cdk.var('RUNNER_GROUP_ID', cdk.var('RUNNER_GROUP', ''));
    local image = cdk.var('IMAGE', d.IMAGE);
    local clientImage = cdk.var('CLIENT_IMAGE', d.CLIENT_IMAGE);
    // Required: gs://<bucket>/<model> or a model directory in IMAGE.
    local model = cdk.var('MODEL', d.MODEL);
    local servedName = cdk.var('SERVED_MODEL_NAME', d.SERVED_MODEL_NAME);
    local maxModelLen = cdk.var('MAX_MODEL_LEN', d.MAX_MODEL_LEN);
    local extraArgs = cdk.var('VLLM_EXTRA_ARGS', d.VLLM_EXTRA_ARGS);
    local multihostBackend = cdk.var('MULTIHOST_BACKEND', d.MULTIHOST_BACKEND);
    local rayPort = cdk.var('RAY_PORT', d.RAY_PORT);
    local benchMode = cdk.var('BENCH_MODE', d.BENCH_MODE);
    local concurrency = cdk.var('CONCURRENCY', d.CONCURRENCY);
    local isl = cdk.var('ISL', d.ISL);
    local osl = cdk.var('OSL', d.OSL);
    local rangeRatio = cdk.var('RANGE_RATIO', d.RANGE_RATIO);
    local numPrompts = cdk.var('NUM_PROMPTS', d.NUM_PROMPTS);
    local numWarmups = cdk.var('NUM_WARMUPS', d.NUM_WARMUPS);
    local fast = cdk.var('FAST', d.FAST);
    local duration = cdk.var('DURATION', d.DURATION);
    local loadFormat = cdk.var('LOAD_FORMAT', d.LOAD_FORMAT);
    local defaultUseRunaiStreamer =
      if loadFormat != '' && loadFormat != 'runai_streamer' then 'false' else d.USE_RUNAI_STREAMER;
    local useRunaiStreamer = cdk.var('USE_RUNAI_STREAMER', defaultUseRunaiStreamer);
    local prompt = cdk.var('PROMPT', 'The capital of France is');
    local maxTokens = cdk.var('MAX_TOKENS', '32');
    // Borg task networks are IPv6-only: AF_INET sockets fail with EAFNOSUPPORT.
    local serveHost = cdk.var('SERVE_HOST', '::1');
    local servePort = cdk.var('SERVE_PORT', 'auto');
    // libtpu SliceBuilder listens on [base, base + chips_per_host) on every
    // host. Default below Linux's ephemeral range (32768-65535) so outbound
    // host-service sockets never collide with SliceBuilder listeners even when
    // ip_local_reserved_ports is unset.
    local sliceBuilderPortBase = cdk.var('SLICEBUILDER_PORT_BASE', '29000');
    local serveUrlHost =
      if std.length(std.findSubstr(':', serveHost)) > 0 then '[' + serveHost + ']' else serveHost;
    // Seconds to wait for /health before giving up.
    local startupTimeout = cdk.var('STARTUP_TIMEOUT', if std.objectHas(d, 'STARTUP_TIMEOUT') then d.STARTUP_TIMEOUT else '7200');
    local topo = exos.resolveTopology(platform, slice);
    // Dual-core TPU7x (ghostfish) hosts have 2 TensorCores per physical chip (8 per 4-chip host).
    local coresPerChip = if std.startsWith(platform, 'ghostfish') then 2 else 1;
    local rayTpuPerHost = topo.chipsPerHost * coresPerChip;
    local rayExpectedTpus = topo.totalChips * coresPerChip;
    local tp = cdk.var('TP', if d.TP != '' then d.TP else std.toString(rayTpuPerHost));
    local useSidecar = benchMode != 'smoke';

    local script = |||
      set -uo pipefail
      mkdir -p /workspace/vllm-torchtpu 2>/dev/null || true
      WORKER_ID="${TPU_WORKER_ID:-0}"
      export SYNC_DIR="/tmp/workspace/sync_${JS_NAME:-run}"
      if [ "${WORKER_ID}" = "0" ]; then
        rm -rf "${SYNC_DIR}" 2>/dev/null || true
        mkdir -p "${SYNC_DIR}"
      fi
      # tpu_runner's pod-log tailer is the only log path and drops the last
      # lines if the container exits right away; give it time to flush.
      trap 'rc=$?; if [ "${WORKER_ID}" = "0" ] && [ ! -f "${SYNC_DIR}/done" ]; then echo "1" > "${SYNC_DIR}/failed" 2>/dev/null || true; fi; echo "=== ${EXOS_RECIPE}: exiting rc=$rc ==="; sleep ${EXIT_LINGER_S:-60}' EXIT
      [ -d /opt/venv/bin ] && export PATH="/opt/venv/bin:$PATH"
      mkdir -p /dev/shm/triton_cache /dev/shm/tpu_logs 2>/dev/null || true
      for cdir in jax_cache vllm_cache torch_tpu_cache; do
        mkdir -p "/tmp/workspace/${cdir}" 2>/dev/null || true
        if [ ! -L "/dev/shm/${cdir}" ]; then
          rm -rf "/dev/shm/${cdir}" 2>/dev/null || true
          ln -s "/tmp/workspace/${cdir}" "/dev/shm/${cdir}" 2>/dev/null || mkdir -p "/dev/shm/${cdir}" 2>/dev/null || true
        fi
      done
      export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT:-/tmp/workspace/tier3_cache}"
      export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT:-/tmp/workspace/tier3_cache}"
      export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK="${TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK:-0}"
      mkdir -p "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}" "${TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT}" 2>/dev/null || true
      if [ -n "${CACHE_DIR:-}" ]; then
        mkdir -p "${CACHE_DIR}" "${VLLM_CACHE_ROOT:-/tmp/vllm_cache}" "${VLLM_CONFIG_ROOT:-/tmp/vllm_config}" "${VLLM_XLA_CACHE_PATH:-/tmp/xla_cache}" "${TRITON_CACHE_DIR:-/tmp/triton}" "${TORCH_TPU_CACHE_DIR:-/tmp/torch_tpu}" "${JAX_COMPILER_CACHE_DIR:-/tmp/jax}" 2>/dev/null || true
      fi
      if [ -n "${VLLM_CACHE_ROOT:-}" ]; then
        CACHE_WS="/tmp/workspace/$(basename "${VLLM_CACHE_ROOT}")"
        if [ -d "${VLLM_CACHE_ROOT}" ] && [ ! -L "${VLLM_CACHE_ROOT}" ] && [ ! -d "${CACHE_WS}" ]; then
          cp -a "${VLLM_CACHE_ROOT}" "${CACHE_WS}" 2>/dev/null || true
        fi
        if [ -d "${CACHE_WS}" ] && [ ! -L "${VLLM_CACHE_ROOT}" ]; then
          rm -rf "${VLLM_CACHE_ROOT}" 2>/dev/null || true
          mkdir -p "$(dirname "${VLLM_CACHE_ROOT}")" 2>/dev/null || true
          ln -s "${CACHE_WS}" "${VLLM_CACHE_ROOT}" 2>/dev/null || true
        fi
        mkdir -p "${VLLM_CACHE_ROOT}" 2>/dev/null || true
        find -L "${VLLM_CACHE_ROOT}" -type f -size 0 -delete 2>/dev/null || true
      fi
    ||| + exos.stageResultUpdaterSh + self.pyPatchSnippet + exos.ipv6PreloadSnippet + |||
      export MALLOC_ARENA_MAX=2
      export TCMALLOC_RELEASE_RATE=10
      echo "=== ${EXOS_RECIPE}: model=${MODEL} tp=${TP} max_model_len=${MAX_MODEL_LEN} bench_mode=${BENCH_MODE} ==="
      if [ -z "${MODEL}" ]; then
        echo 'EXOS_VLLM_RESULT {"pass": false, "error": "set MODEL=gs://... or a directory in IMAGE"}'
        exit 1
      fi
      env | grep -E '^(TPU_|EXOS_|CDK_|ENABLE_TPUNETD|JAX_|PJRT_|LIBTPU)' | sort
      SERVER_MODEL="${MODEL}"
      LOAD_ARGS=""
      _extra_has_dummy=0
      case " ${VLLM_EXTRA_ARGS:-} " in
        *" --load-format=dummy "*|*" --load-format dummy "*) _extra_has_dummy=1 ;;
      esac
      case "${MODEL}" in
        gs://*)
          echo "GCE_METADATA_HOST=${GCE_METADATA_HOST:-unset}"
          python3 - <<'PYMD' || true
      import os, urllib.request
      host = os.environ.get('GCE_METADATA_HOST')
      if host:
        req = urllib.request.Request('http://' + host + '/computeMetadata/v1/instance/service-accounts/default/email', headers={'Metadata-Flavor': 'Google'})
        print('metadata service account:', urllib.request.urlopen(req, timeout=5).read().decode())
      PYMD
          if [ "${LOAD_FORMAT:-}" = "dummy" ] || [ "${_extra_has_dummy}" = "1" ]; then
            SERVER_MODEL="/tmp/model_meta_local"
            mkdir -p "${SERVER_MODEL}"
            python3 /tmp/exos_gcs_cp.py tokenizer "${MODEL}" "${SERVER_MODEL}"
            [ "${_extra_has_dummy}" = "0" ] && LOAD_ARGS="--load-format dummy"
          elif [ "${USE_RUNAI_STREAMER:-true}" = "false" ] || [ "${USE_RUNAI_STREAMER:-true}" = "0" ]; then
            SERVER_MODEL="/tmp/workspace/model_local"
            mkdir -p "${SERVER_MODEL}"
            python3 /tmp/exos_gcs_cp.py all "${MODEL}" "${SERVER_MODEL}"
            [ -n "${LOAD_FORMAT:-}" ] && LOAD_ARGS="--load-format ${LOAD_FORMAT}"
          elif [ -n "${LOAD_FORMAT:-}" ]; then
            LOAD_ARGS="--load-format ${LOAD_FORMAT}"
          else
            # Stream safetensors straight from GCS with the Run:ai streamer.
            LOAD_ARGS="--load-format runai_streamer"
          fi
          ;;
        *)
          ls -la "${MODEL}" || { echo "EXOS_VLLM_RESULT {\"pass\": false, \"error\": \"model dir missing\"}"; exit 1; }
          if [ -n "${LOAD_FORMAT:-}" ] && [ "${_extra_has_dummy}" = "0" ]; then
            LOAD_ARGS="--load-format ${LOAD_FORMAT}"
          fi
          ;;
      esac
      df -h /tmp /tmp/workspace /dev/shm 2>/dev/null || true
      python3 -c 'import vllm; print("vllm", vllm.__version__)' || true
      cat /etc/hosts
      pick_port() { python3 -c 'import socket; s=socket.socket(socket.AF_INET6); s.bind(("::1", 0)); print(s.getsockname()[1])'; }
      # Multi-host Ray cluster formation over Pod DNS hostnames resolved via /etc/hosts.
      if [ "${MULTIHOST_BACKEND:-}" = "ray" ]; then
        export TPU_MULTIHOST_BACKEND=ray
        export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
        export RAY_memory_monitor_refresh_ms=0
        unset TPU_VISIBLE_CHIPS RAY_ADDRESS VLLM_LOOPBACK_IP
        WORKER_ID="${TPU_WORKER_ID:-0}"
        HEAD_HOST="$(echo "${TPU_WORKER_HOSTNAMES}" | cut -d, -f1)"
        MY_HOST="$(echo "${TPU_WORKER_HOSTNAMES}" | cut -d, -f$((WORKER_ID + 1)))"
        export TPU_WORKER_ID="${WORKER_ID}"
        export TPU_NODE_ID="${WORKER_ID}"
        export CLOUD_TPU_TASK_ID="${WORKER_ID}"
        # runc-proxy's slice-peer wait is best effort, so retry until the peer resolves.
        resolve_ipv6() {
          python3 -u - "$1" <<'PYRESOLVE'
      import ipaddress, socket, sys, time
      target = sys.argv[1].strip().strip('[]')
      deadline = time.time() + 300
      while True:
        try:
          ip = socket.getaddrinfo(target, None, socket.AF_INET6, socket.SOCK_STREAM)[0][4][0]
          print(ipaddress.ip_address(ip).compressed)
          sys.exit(0)
        except Exception:
          if time.time() >= deadline:
            print(f'FATAL: timed out resolving {target} via /etc/hosts', file=sys.stderr)
            sys.exit(1)
          time.sleep(1)
      PYRESOLVE
        }
        HEAD_IP="$(resolve_ipv6 "${HEAD_HOST}")" || exit 1
        MY_IP="$(resolve_ipv6 "${MY_HOST}")" || exit 1
        export MASTER_ADDR="${HEAD_HOST}"
        export VLLM_HOST_IP="${MY_IP}"
        export TORCH_TPU_SLICEBUILDER_PORT_BASE="${TORCH_TPU_SLICEBUILDER_PORT_BASE:-29000}"
        export TORCH_TPU_BASE_PORT="${TORCH_TPU_BASE_PORT:-29000}"
        export VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND:-1}"
        echo "=== ${EXOS_RECIPE}: Ray node worker=${WORKER_ID} my_host=${MY_HOST} my_ip=${MY_IP} head_host=${HEAD_HOST} head_ip=${HEAD_IP} ray_port=${RAY_PORT} ray_tpu_per_host=${RAY_TPU_PER_HOST} ==="
        if [ "${WORKER_ID}" != "0" ]; then
          for _ in $(seq 1 180); do
            python3 -c 'import socket,sys; s=socket.socket(socket.AF_INET6); s.settimeout(2); s.connect((sys.argv[1], int(sys.argv[2]))); s.close()' "${HEAD_IP}" "${RAY_PORT}" 2>/dev/null && break
            sleep 5
          done
          ray start --address="[${HEAD_IP}]:${RAY_PORT}" --node-ip-address="${MY_IP}" \
            --min-worker-port=24001 --max-worker-port=24999 \
            --node-manager-port=23001 --object-manager-port=23003 \
            --dashboard-agent-grpc-port=23007 --dashboard-agent-listen-port=23009 \
            --metrics-export-port=23011 --runtime-env-agent-port=23013 \
            --object-store-memory=4294967296 \
            --labels="ray.io/tpu-worker-id=${WORKER_ID}" \
            --num-cpus=128 --resources="{\"TPU\": ${RAY_TPU_PER_HOST}}" --disable-usage-stats || {
              echo "=== RAY WORKER LOGS ==="
              for f in /tmp/ray/session_latest/logs/*; do echo "--- $f ---"; tail -n 50 "$f" 2>/dev/null || true; done
              exit 1
            }
          export RAY_ADDRESS="[${HEAD_IP}]:${RAY_PORT}"
          while python3 -c 'import socket,sys; s=socket.socket(socket.AF_INET6); s.settimeout(2); s.connect((sys.argv[1], int(sys.argv[2]))); s.close()' "${HEAD_IP}" "${RAY_PORT}" 2>/dev/null; do
            sleep 10
          done
          ray stop --force 2>/dev/null || true
          echo 'EXOS_VLLM_RESULT {"pass": true, "role": "ray-worker", "worker_id": '"${WORKER_ID}"'}'
          exit 0
        fi
        ray start --head --node-ip-address="${MY_IP}" --port="${RAY_PORT}" \
          --min-worker-port=24001 --max-worker-port=24999 \
          --node-manager-port=23001 --object-manager-port=23003 \
          --ray-client-server-port=23005 \
          --dashboard-agent-grpc-port=23007 --dashboard-agent-listen-port=23009 \
          --metrics-export-port=23011 --runtime-env-agent-port=23013 \
          --object-store-memory=4294967296 \
          --labels="ray.io/tpu-worker-id=${WORKER_ID}" \
          --num-cpus=128 --resources="{\"TPU\": ${RAY_TPU_PER_HOST}}" --include-dashboard=false --disable-usage-stats || {
            echo "=== RAY HEAD LOGS ==="
            for f in /tmp/ray/session_latest/logs/*; do echo "--- $f ---"; tail -n 50 "$f" 2>/dev/null || true; done
            exit 1
          }
        export RAY_ADDRESS="[${MY_IP}]:${RAY_PORT}"
        for _ in $(seq 1 180); do
          READY_TPUS="$(python3 -c 'import ray; ray.init(address="auto", ignore_reinit_error=True); print(int(ray.cluster_resources().get("TPU", 0)))' 2>/dev/null || echo 0)"
          echo "Ray cluster TPUs: ${READY_TPUS}/${RAY_EXPECTED_TPUS}"
          [ "${READY_TPUS}" -ge "${RAY_EXPECTED_TPUS}" ] && break
          sleep 5
        done
      else
        if [ "${TP}" -gt 1 ]; then
          export MASTER_PORT="${MASTER_PORT:-$(pick_port)}"
          echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
        fi
      fi
      # Borg port enforcement rejects bind() to ports the task does not own
      # (EPERM), but kernel-assigned ephemeral ports are allowed.
      if [ "${SERVE_PORT}" = auto ]; then
        SERVE_PORT="$(python3 -c 'import socket,sys; s=socket.socket(socket.AF_INET6); s.bind((sys.argv[1], 0)); print(s.getsockname()[1])' "${SERVE_HOST}")" || exit 1
      fi
      export SERVE_PORT
      if [ "${ENABLE_PERFETTO:-true}" = "true" ]; then
        export TPU_VMODULE="${TPU_VMODULE:-tpu_pjrt_client=1,pjrt_stream_executor_client=1,tpu_pjrt_compiler_utils=1}"
        export TPU_STDERR_LOG_LEVEL="${TPU_STDERR_LOG_LEVEL:-0}"
        export TPU_PJRT_EVENT_TRACE_ENABLE="${TPU_PJRT_EVENT_TRACE_ENABLE:-1}"
      fi
      echo "=== ${EXOS_RECIPE}: serving on [${SERVE_HOST}]:${SERVE_PORT} ==="
      echo "trace-start-event: name=vllm-server ts=$(date +'%s.%N') pid=vllm tid=vllm"
      LOAD_ARGS_ARR=()
      [ -n "${LOAD_ARGS}" ] && read -r -a LOAD_ARGS_ARR <<< "${LOAD_ARGS}"
      readarray -d '' VLLM_EXTRA_ARGS_ARR < <(python3 -c 'import os, re, shlex, sys; s = re.sub(r"(--[A-Za-z0-9_.-]+=)(\{[^\s\x27]*\})(?=\s|$)", lambda m: m.group(1) + "\x27" + m.group(2) + "\x27", os.environ.get("VLLM_EXTRA_ARGS", "")); sys.stdout.buffer.write(b"".join(a.encode() + b"\0" for a in shlex.split(s)))')
      vllm serve "${SERVER_MODEL}" --served-model-name "${SERVED_MODEL_NAME}" \
        --tensor-parallel-size "${TP}" --max-model-len "${MAX_MODEL_LEN}" \
        --host "${SERVE_HOST}" --port "${SERVE_PORT}" \
        ${LOAD_ARGS_ARR[@]+"${LOAD_ARGS_ARR[@]}"} ${VLLM_EXTRA_ARGS_ARR[@]+"${VLLM_EXTRA_ARGS_ARR[@]}"} \
        > >(tee /tmp/vllm_serve.log) 2>&1 &
      export SERVER_PID=$!
      EXOS_STARTUP_S="$(python3 -u - <<'PYWAIT'
      import json, os, sys, time, urllib.request

      deadline = time.time() + int(os.environ['STARTUP_TIMEOUT'])
      start = time.time()
      while time.time() < deadline:
        if not os.path.exists('/proc/' + os.environ['SERVER_PID']):
          print('EXOS_VLLM_RESULT ' + json.dumps({'pass': False, 'error': 'vllm serve exited', 'startup_s': round(time.time() - start, 1)}), file=sys.stderr)
          sys.exit(1)
        try:
          with urllib.request.urlopen('http://' + os.environ['SERVE_URL_HOST'] + ':' + os.environ['SERVE_PORT'] + '/health', timeout=5) as r:
            if r.status == 200:
              print(round(time.time() - start, 1))
              sys.exit(0)
        except Exception:
          pass
        time.sleep(10)
      print('EXOS_VLLM_RESULT ' + json.dumps({'pass': False, 'error': 'server not healthy', 'startup_s': round(time.time() - start, 1)}), file=sys.stderr)
      sys.exit(1)
      PYWAIT
      )" || {
        echo "1" > "${SYNC_DIR}/failed" 2>/dev/null || true
        if [ -n "${CDK_OUTPUT_DIR:-}" ] && [ -f /tmp/vllm_serve.log ]; then
          mkdir -p "${CDK_OUTPUT_DIR}" 2>/dev/null || true
          cp -f /tmp/vllm_serve.log "${CDK_OUTPUT_DIR}/vllm_serve.log" 2>/dev/null || true
        fi
        echo "=== VLLM CRASH DIAGNOSTICS (/tmp/vllm_serve.log) ==="
        grep -a -vE '(ray_executor_v2\.py:57[0-9]|ActorHandleNotFoundError|tpu_pjrt_client\.cc|pjrt_stream_executor_client\.cc|tpu_pjrt_compiler_utils\.cc|tpu_chip_config\.cc|deepsea_compiler_)' /tmp/vllm_serve.log 2>/dev/null | grep -a -E -C 5 '(Traceback|Error|Exception|FATAL|Check failed|closed its address pipe|Engine core initialization failed)' | tail -n 105 | awk '{print; if (NR % 35 == 0) system("sleep 6")}' || true
        sleep 6
        echo "=== VLLM TAIL (/tmp/vllm_serve.log) ==="
        grep -a -vE '(ray_executor_v2\.py:57[0-9]|ActorHandleNotFoundError|tpu_pjrt_client\.cc|pjrt_stream_executor_client\.cc|tpu_pjrt_compiler_utils\.cc|tpu_chip_config\.cc|deepsea_compiler_)' /tmp/vllm_serve.log 2>/dev/null | tail -n 70 | awk '{print; if (NR % 35 == 0) system("sleep 6")}' || true
        if [ "${MULTIHOST_BACKEND:-}" = "ray" ]; then
          sleep 6
          echo "=== RAY SESSION LOGS ==="
          for f in /tmp/ray/session_latest/logs/worker*.err /tmp/ray/session_latest/logs/worker*.out /tmp/ray/session_latest/logs/gcs_server.err /tmp/ray/session_latest/logs/raylet.err; do
            if [ -s "$f" ]; then echo "--- $f ---"; tail -n 30 "$f" 2>/dev/null || true; sleep 2; fi
          done
        fi
        kill "${SERVER_PID}" 2>/dev/null || true
        [ "${MULTIHOST_BACKEND:-}" = "ray" ] && ray stop --force 2>/dev/null || true
        exit 1
      }
      export EXOS_STARTUP_S
      if [ "${USE_SIDECAR_BENCH:-0}" = "1" ]; then
        cat > "${SYNC_DIR}/ready.tmp" <<EOF
      SERVE_HOST="${SERVE_HOST}"
      SERVE_URL_HOST="${SERVE_URL_HOST}"
      SERVE_PORT="${SERVE_PORT}"
      METRICS_URL="http://${SERVE_URL_HOST}:${SERVE_PORT}/metrics"
      EXOS_STARTUP_S="${EXOS_STARTUP_S}"
      EOF
        mv "${SYNC_DIR}/ready.tmp" "${SYNC_DIR}/ready"
        echo "=== ${EXOS_RECIPE}: waiting for sidecar-bench to complete (${SYNC_DIR}/done) ==="
        while [ ! -f "${SYNC_DIR}/done" ]; do
          if [ ! -d "/proc/${SERVER_PID}" ]; then
            echo "ERROR: vllm serve exited while sidecar-bench was running"
            echo "1" > "${SYNC_DIR}/failed"
            break
          fi
          sleep 5
        done
        RC="$(cat "${SYNC_DIR}/done" 2>/dev/null || echo 1)"
      else
    ||| + (if useSidecar then '        echo "ERROR: USE_SIDECAR_BENCH=0 in sidecar recipe"\n        RC=1\n' else self.benchScriptFor(benchMode)) + |||
      RC=$?
      fi
      kill "${SERVER_PID}" 2>/dev/null || true
      if [ -n "${CDK_OUTPUT_DIR:-}" ] && [ -f /tmp/vllm_serve.log ]; then
        mkdir -p "${CDK_OUTPUT_DIR}" 2>/dev/null || true
        cp -f /tmp/vllm_serve.log "${CDK_OUTPUT_DIR}/vllm_serve.log" 2>/dev/null || true
      fi
      echo "trace-end-event: name=vllm-server ts=$(date +'%s.%N') pid=vllm tid=vllm"
      [ "${MULTIHOST_BACKEND:-}" = "ray" ] && ray stop --force 2>/dev/null || true
      exit "${RC}"
    |||;

    local jobEnv = {
      EXOS_RECIPE: name,
      MODEL: model,
      SERVED_MODEL_NAME: servedName,
      TP: tp,
      MAX_MODEL_LEN: maxModelLen,
      VLLM_EXTRA_ARGS: extraArgs,
      MULTIHOST_BACKEND: multihostBackend,
      RAY_PORT: rayPort,
      CHIPS_PER_HOST: std.toString(topo.chipsPerHost),
      RAY_TPU_PER_HOST: std.toString(rayTpuPerHost),
      RAY_EXPECTED_TPUS: std.toString(rayExpectedTpus),
      TOTAL_BENCH_CHIPS: std.toString(topo.totalChips),
      BENCH_MODE: benchMode,
      USE_SIDECAR_BENCH: if useSidecar then '1' else '0',
      CONCURRENCY: concurrency,
      ISL: isl,
      OSL: osl,
      RANGE_RATIO: rangeRatio,
      NUM_PROMPTS: numPrompts,
      NUM_WARMUPS: numWarmups,
      FAST: fast,
      DURATION: duration,
      USE_RUNAI_STREAMER: useRunaiStreamer,
      [if loadFormat != '' then 'LOAD_FORMAT']: loadFormat,
      PROMPT: prompt,
      MAX_TOKENS: maxTokens,
      STARTUP_TIMEOUT: startupTimeout,
      TORCH_TPU_SLICEBUILDER_PORT_BASE: sliceBuilderPortBase,
      TORCH_TPU_BASE_PORT: sliceBuilderPortBase,
      SERVE_HOST: serveHost,
      SERVE_PORT: servePort,
      SERVE_URL_HOST: serveUrlHost,
      HF_HUB_OFFLINE: '1',
      TRANSFORMERS_OFFLINE: '1',
      // Read by the serve script, which turns on PJRT logging (TPU_VMODULE) unless this is false.
      ENABLE_PERFETTO: if cdk.boolVar('ENABLE_PERFETTO', true) then 'true' else 'false',
    } + exos.ipv6Env
    + self.pyPatchEnv
    + (if multihostBackend == 'ray' then {} else exos.singleHostIpv6Env)
    + d.EXTRA_ENV;

    cdk.jobSet(
      name=name,
      failurePolicy=cdk.neverRestart,
      network={ enableDNSHostnames: true },
      jobs=exos.tpuJobs(
        name='worker',
        containerName='vllm',
        image=image,
        script=script,
        platform=platform,
        slice=slice,
        runnerGroupId=runnerGroupId,
        extraEnv=jobEnv,
        headExtraContainers=if useSidecar then [self.sidecarBenchContainer(clientImage, jobEnv)] else [],
      ),
    ),

  // pdDefaults holds defaults for multi-host Prefill/Decode disaggregated
  // serving with TPURaidenConnector on Exos Borg TPU runner pools.
  pdDefaults:: {
    PLATFORM: 'ghostfish_sampling',
    // A 4-host pool (SLICE=4x4, 16 chips) is split into 2 Prefill hosts
    // (ranks 0..1) and 2 Decode hosts (ranks 2..3) with TPURaidenConnector over DCN.
    SLICE: '4x4',
    IMAGE: 'us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/torchtpu-vllm-prod:nightly-20261014',
    CLIENT_IMAGE: 'us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/inferencex-agentx:latest',
    MODEL: 'gs://cloud-devkit/ckpt/hf/zai-org/GLM-5.2-FP8',
    SERVED_MODEL_NAME: 'zai-org/GLM-5.2-FP8',
    MAX_MODEL_LEN: '1048576',
    MLA_LAYOUT: 'tensorcore',
    BENCH_MODE: 'agentx',
    CONCURRENCY: '32',
    ISL: '8192',
    OSL: '1024',
    RANGE_RATIO: '0.8',
    NUM_PROMPTS: '',
    NUM_WARMUPS: '',
    FAST: 'true',
    DURATION: '600',
    USE_RUNAI_STREAMER: 'true',
    LOAD_FORMAT: '',
    PROMPT: 'The capital of France is',
    MAX_TOKENS: '32',
    ENABLE_PERFETTO: 'false',
    PREFILL_EXTRA_ARGS: '',
    DECODE_EXTRA_ARGS: '',
    PREFILL_ENV: {},
    DECODE_ENV: {},
    DEST_DCP_SIZE: '16',
    DEST_CP_INTERLEAVE: '1024',
    HOSTS_PER_LEG: '2',
    // xPyD with one prefill: DECODE_REPLICAS one-host decode slices (see pdRecipe).
    DECODE_REPLICAS: '1',
    // 2PnD (q38 TUNING-KIT5, 2026-11-18): PREFILL_REPLICAS one-host prefill slices, set only through PdBase.prefillReplicas /
    // fromPdSpec (pdRecipe reads it from the overrides, never from a recipe variable), so no existing caller changes.
    PREFILL_REPLICAS: '1',
    GLM_RAIDEN_DEFAULTS: '1',
    PREFILL_KV_CONFIG: '',
    DECODE_KV_CONFIG: '',
    PROXY_PROBE: '0',
    STARTUP_TIMEOUT: '7200',
    EXTRA_ENV_PREFILL: '',
    EXTRA_ENV_DECODE: '',
    EXTRA_ENV: {},
  },

  // pdRecipe returns the JobSet for Prefill/Decode disaggregated serving +
  // benchmarking on an Exos Borg TPU pool (supporting both 2-host/leg Ray and
  // 1-host/leg single-node P/D topologies).
  pdRecipe(name, overrides={})::
    local d = self.pdDefaults + overrides;
    local platform = cdk.var('PLATFORM', d.PLATFORM);
    local slice = cdk.var('SLICE', d.SLICE);
    local hostsPerLeg = cdk.var('HOSTS_PER_LEG', d.HOSTS_PER_LEG);
    local glmRaidenDefaults = cdk.var('GLM_RAIDEN_DEFAULTS', d.GLM_RAIDEN_DEFAULTS);
    local prefillKvConfig = cdk.var('PREFILL_KV_CONFIG', d.PREFILL_KV_CONFIG);
    local decodeKvConfig = cdk.var('DECODE_KV_CONFIG', d.DECODE_KV_CONFIG);
    local proxyProbe = cdk.var('PROXY_PROBE', d.PROXY_PROBE);
    local prefillRunnerGroupId = cdk.var('PREFILL_RUNNER_GROUP_ID', '');
    local runnerGroupId = if prefillRunnerGroupId != '' then prefillRunnerGroupId else cdk.var('RUNNER_GROUP_ID', cdk.var('RUNNER_GROUP', ''));
    local image = cdk.var('IMAGE', d.IMAGE);
    local clientImage = cdk.var('CLIENT_IMAGE', d.CLIENT_IMAGE);
    local model = cdk.var('MODEL', d.MODEL);
    local servedName = cdk.var('SERVED_MODEL_NAME', d.SERVED_MODEL_NAME);
    local maxModelLen = cdk.var('MAX_MODEL_LEN', d.MAX_MODEL_LEN);
    local mlaLayout = cdk.var('MLA_LAYOUT', d.MLA_LAYOUT);
    local benchMode = cdk.var('BENCH_MODE', d.BENCH_MODE);
    local concurrency = cdk.var('CONCURRENCY', d.CONCURRENCY);
    local isl = cdk.var('ISL', d.ISL);
    local osl = cdk.var('OSL', d.OSL);
    local rangeRatio = cdk.var('RANGE_RATIO', d.RANGE_RATIO);
    local numPrompts = cdk.var('NUM_PROMPTS', d.NUM_PROMPTS);
    local numWarmups = cdk.var('NUM_WARMUPS', d.NUM_WARMUPS);
    local fast = cdk.var('FAST', d.FAST);
    local duration = cdk.var('DURATION', d.DURATION);
    local loadFormat = cdk.var('LOAD_FORMAT', d.LOAD_FORMAT);
    local defaultUseRunaiStreamer =
      if loadFormat != '' && loadFormat != 'runai_streamer' then 'false' else d.USE_RUNAI_STREAMER;
    local useRunaiStreamer = cdk.var('USE_RUNAI_STREAMER', defaultUseRunaiStreamer);
    local prompt = cdk.var('PROMPT', d.PROMPT);
    local maxTokens = cdk.var('MAX_TOKENS', d.MAX_TOKENS);
    local enablePerfetto = cdk.var('ENABLE_PERFETTO', d.ENABLE_PERFETTO);
    local prefillExtraArgs = cdk.var('PREFILL_EXTRA_ARGS', d.PREFILL_EXTRA_ARGS);
    local decodeExtraArgs = cdk.var('DECODE_EXTRA_ARGS', d.DECODE_EXTRA_ARGS);
    local destDcpSize = cdk.var('DEST_DCP_SIZE', d.DEST_DCP_SIZE);
    local destCpInterleave = cdk.var('DEST_CP_INTERLEAVE', d.DEST_CP_INTERLEAVE);
    local extraEnvPrefill = cdk.var('EXTRA_ENV_PREFILL', d.EXTRA_ENV_PREFILL);
    local extraEnvDecode = cdk.var('EXTRA_ENV_DECODE', d.EXTRA_ENV_DECODE);
    local startupTimeout = cdk.var('STARTUP_TIMEOUT', d.STARTUP_TIMEOUT);
    // libtpu SliceBuilder listens on [base, base + chips_per_host) on every
    // host. Default below Linux's ephemeral range (32768-65535) so outbound
    // host-service sockets never collide with SliceBuilder listeners even when
    // ip_local_reserved_ports is unset.
    local sliceBuilderPortBase = cdk.var('SLICEBUILDER_PORT_BASE', '29000');
    local decodeRunnerGroupId = cdk.var('DECODE_RUNNER_GROUP_ID', cdk.var('DECODE_RUNNER_GROUP', ''));
    // TWO_SLICE_PD runs prefill and decode as separate slices, one per pool or Kueue slice.
    // Kueue and multi-host legs default to it, matching the Cloud P/D recipes (one slice per leg);
    // Ghostfish V2 slices also reject splitting one multi-host slice into separate leg sessions.
    local twoSlicePdVar = cdk.var(
      'TWO_SLICE_PD',
      if runnerGroupId == '' || decodeRunnerGroupId != '' || hostsPerLeg != '1' then 'true' else 'false'
    );
    local twoSlicePd = twoSlicePdVar == 'true' || twoSlicePdVar == '1' || decodeRunnerGroupId != '';
    local legSlice = if twoSlicePd then (if hostsPerLeg == '1' then '2x2' else '2x4') else slice;
    local topo = exos.resolveTopology(platform, legSlice);
    local perLegHosts = if twoSlicePd then topo.hosts else std.max(1, topo.hosts / 2);
    local perLegChips = topo.chipsPerHost * perLegHosts;
    local isGhostfish = std.startsWith(platform, 'ghostfish');
    local isGhostlite = std.startsWith(platform, 'ghostlite');
    local coresPerChip = if isGhostfish then 2 else 1;
    local rayTpuPerHost = topo.chipsPerHost * coresPerChip;
    local rayLegExpectedTpus = perLegChips * coresPerChip;
    local rayTpuTopology = if perLegHosts == 2 then '2x2x2' else if perLegHosts == 4 then '2x2x4' else '2x2x1';
    local rayAccelType = if isGhostfish then 'TPU-V7X' else if isGhostlite then 'TPU-V6E' else 'TPU-V5P';
    local rayPodType = (if isGhostfish then 'v7x-' else if isGhostlite then 'v6e-' else 'v5p-') + std.toString(rayLegExpectedTpus);
    local tpuAccelEnv = (if isGhostfish then 'tpu7x-' else if isGhostlite then 'v6e-' else 'v5p-') + std.toString(rayLegExpectedTpus);
    // Each leg's workers are addressed by Pod DNS name, which tpu_runner resolves to the
    // runner's DirectPath IPv6 address through the read-only /etc/hosts it mounts.
    local legHostnames(job) =
      if topo.hosts == 1 then '$(JS_NAME)-%s-0-0.$(JS_NAME)' % job
      else exos.workerHostnames(job, topo.hosts);
    // PREFILL_REPLICAS > 1 (2PnD; q38 TUNING-KIT5, 2026-11-18): one replicatedJob per prefill (prefill, prefill1, ...,
    // prefill<n-1>), each a one-host slice like the 1P1D prefill. Prefill 0 alone runs the sidecar bench and the proxy, which
    // spreads requests round robin over every prefill (and every decode), and it waits for every prefill; prefill i > 0 serves on
    // its own host, registers with Raiden as prefill-<i>-engine (as the Cloud xPyD recipes rename replica i > 0) and serves while
    // prefill 0's port is up, as the decode legs do. PREFILL_REPLICAS = 1 (the default): every rendered byte is as before.
    local prefillReplicas = std.parseJson(d.PREFILL_REPLICAS);
    assert prefillReplicas >= 1 : 'PREFILL_REPLICAS must be at least 1, got %d' % prefillReplicas;
    assert prefillReplicas == 1 || (twoSlicePd && hostsPerLeg == '1' && runnerGroupId == '' && decodeRunnerGroupId == '') :
      'PREFILL_REPLICAS > 1 needs one-host legs on separate slices (HOSTS_PER_LEG=1, TWO_SLICE_PD) and no RUNNER_GROUP_ID / DECODE_RUNNER_GROUP_ID';
    local prefillJobName(i) = if i == 0 then 'prefill' else 'prefill%d' % i;
    local defaultPrefillHostnames =
      if twoSlicePd then std.join(',', [legHostnames(prefillJobName(i)) for i in std.range(0, prefillReplicas - 1)])
      else '';
    // DECODE_REPLICAS > 1 (1PnD): one replicatedJob per decode (decode, decode1, ..., decode<n-1>),
    // each a one-host slice like the 1P1D decode, rather than replicas of one job, whose replicas
    // are the hosts of a single slice on Exos. Decode i serves on its own host and registers with
    // Raiden as decode-<i>-engine (as the Cloud xPyD recipes rename replica i > 0); the prefill
    // waits for every decode and the proxy spreads requests over all of them.
    local decodeReplicas = std.parseJson(cdk.var('DECODE_REPLICAS', d.DECODE_REPLICAS));
    assert decodeReplicas >= 1 : 'DECODE_REPLICAS must be at least 1, got %d' % decodeReplicas;
    assert decodeReplicas == 1 || (twoSlicePd && hostsPerLeg == '1' && decodeRunnerGroupId == '') :
      'DECODE_REPLICAS > 1 needs one-host legs on separate slices (HOSTS_PER_LEG=1, TWO_SLICE_PD) and no DECODE_RUNNER_GROUP_ID';
    local decodeJobName(i) = if i == 0 then 'decode' else 'decode%d' % i;
    local defaultDecodeHostnames =
      if twoSlicePd then std.join(',', [legHostnames(decodeJobName(i)) for i in std.range(0, decodeReplicas - 1)])
      else '';
    local prefillHostnames = cdk.var('PREFILL_HOSTNAMES', cdk.var('PREFILL_HOST_IPS', defaultPrefillHostnames));
    local decodeHostnames = cdk.var('DECODE_HOSTNAMES', cdk.var('DECODE_HOST_IPS', defaultDecodeHostnames));
    local decodeRunnerIndexOffset = if decodeRunnerGroupId != '' && decodeRunnerGroupId == runnerGroupId then topo.hosts else 0;
    local pdExpectedPoolHosts = if decodeRunnerGroupId != '' && decodeRunnerGroupId == runnerGroupId then 2 * topo.hosts else topo.hosts;
    local totalBenchChips = if twoSlicePd then (prefillReplicas + decodeReplicas) * topo.totalChips else topo.totalChips;
    local useSidecar = benchMode != 'smoke';

    local shellQuote(s) = "'" + std.strReplace(std.toString(s), "'", "'\"'\"'") + "'";
    local skipLegKeys = [
      'TPU_NODE_ID',
      'TPU_WORKER_ID',
      'CLOUD_TPU_TASK_ID',
      'TPU_WORKER_HOSTNAMES',
      'TPU_PROCESS_ADDRESSES',
      'TPU_HOST_BOUNDS',
      'TPU_PROCESS_BOUNDS',
      'EXPECTED_TPU_DEVICES',
      'TPU_RAIDEN_JOB_NAME',
      'TPU_RAIDEN_ADVERTISE_HOST',
      'TPU_RAIDEN_RESHARD_PORT_BASE',
      'TPU_RAIDEN_STORE_DISPATCH_PORT_BASE',
      'TPU_KV_TRANSFER_PORT',
      'TPU_SIDE_CHANNEL_PORT',
      'TPU_LOG_DIR',
    ];
    local legExportLines(envObj) =
      local lines = [
        '  export %s=%s' % [k, shellQuote(envObj[k])]
        for k in std.objectFields(envObj)
        if envObj[k] != null && !std.isObject(envObj[k]) && !std.member(skipLegKeys, k)
      ];
      if std.length(lines) == 0 then '' else std.join('\n', lines) + '\n';

    local script = |||
      set -uo pipefail
      mkdir -p /workspace/vllm-torchtpu 2>/dev/null || true
      _init_is_prefill_head=0
      if [ "${PD_LEG:-}" != "decode" ] && [ "${TPU_WORKER_ID:-0}" = "0" ]; then
        _init_is_prefill_head=1
      fi
      export SYNC_DIR="/tmp/workspace/sync_${JS_NAME:-run}"
      if [ "${_init_is_prefill_head}" = "1" ]; then
        rm -rf "${SYNC_DIR}" 2>/dev/null || true
        mkdir -p "${SYNC_DIR}"
      fi
      trap 'rc=$?; if [ "${_init_is_prefill_head}" = "1" ] && [ ! -f "${SYNC_DIR}/done" ]; then echo "1" > "${SYNC_DIR}/failed" 2>/dev/null || true; fi; echo "=== ${EXOS_RECIPE}: exiting rc=$rc ==="; sleep ${EXIT_LINGER_S:-60}' EXIT
      [ -d /opt/venv/bin ] && export PATH="/opt/venv/bin:$PATH"
      ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
    ||| + exos.stageResultUpdaterSh + self.pyPatchSnippet + exos.ipv6PreloadSnippet + |||
      export MALLOC_ARENA_MAX=2
      export TCMALLOC_RELEASE_RATE=10
      if [ "${GLM_RAIDEN_DEFAULTS:-1}" = "1" ]; then
        if [ "${MLA_LAYOUT}" = "tensorcore" ]; then
          export TPU_SPARSE_MLA_NOPE_LAYOUT=tensorcore
          export TPU_SPARSE_MLA_ROPE_LAYOUT=tensorcore
        else
          export TPU_SPARSE_MLA_NOPE_LAYOUT="${MLA_LAYOUT}"
          export TPU_SPARSE_MLA_ROPE_LAYOUT="${MLA_LAYOUT}"
        fi
      fi
      HOST1=""
      HOST3=""
      if [ "${HOSTS_PER_LEG:-2}" = "1" ]; then
        if [ "${PD_LEG:-}" = "prefill" ]; then
          WORKER_ID="0"
          P_HOSTS="${PREFILL_HOSTNAMES:-${TPU_WORKER_HOSTNAMES}}"
          D_HOSTS="${DECODE_HOSTNAMES:-}"
          HOST0="$(echo "${P_HOSTS}" | cut -d, -f1)"
          HOST2="$(echo "${D_HOSTS}" | cut -d, -f1)"
          MY_HOST="${HOST0}"
        elif [ "${PD_LEG:-}" = "decode" ]; then
          WORKER_ID="1"
          P_HOSTS="${PREFILL_HOSTNAMES:-}"
          D_HOSTS="${DECODE_HOSTNAMES:-${TPU_WORKER_HOSTNAMES}}"
          HOST0="$(echo "${P_HOSTS}" | cut -d, -f1)"
          HOST2="$(echo "${D_HOSTS}" | cut -d, -f1)"
          MY_HOST="${HOST2}"
        else
          WORKER_ID="${TPU_WORKER_ID:-0}"
          HOST0="$(echo "${TPU_WORKER_HOSTNAMES}" | cut -d, -f1)"
          HOST2="$(echo "${TPU_WORKER_HOSTNAMES}" | cut -d, -f2)"
          MY_HOST="$(echo "${TPU_WORKER_HOSTNAMES}" | cut -d, -f$((WORKER_ID + 1)))"
        fi
      else
        if [ "${PD_LEG:-}" = "prefill" ]; then
          WORKER_ID="${TPU_WORKER_ID:-0}"
          P_HOSTS="${PREFILL_HOSTNAMES:-${TPU_WORKER_HOSTNAMES}}"
          D_HOSTS="${DECODE_HOSTNAMES:-}"
          HOST0="$(echo "${P_HOSTS}" | cut -d, -f1)"
          HOST1="$(echo "${P_HOSTS}" | cut -d, -f2)"
          HOST2="$(echo "${D_HOSTS}" | cut -d, -f1)"
          HOST3="$(echo "${D_HOSTS}" | cut -d, -f2)"
          MY_HOST="$(echo "${P_HOSTS}" | cut -d, -f$((WORKER_ID + 1)))"
        elif [ "${PD_LEG:-}" = "decode" ]; then
          WORKER_ID="$(( ${TPU_WORKER_ID:-0} + 2 ))"
          P_HOSTS="${PREFILL_HOSTNAMES:-}"
          D_HOSTS="${DECODE_HOSTNAMES:-${TPU_WORKER_HOSTNAMES}}"
          HOST0="$(echo "${P_HOSTS}" | cut -d, -f1)"
          HOST1="$(echo "${P_HOSTS}" | cut -d, -f2)"
          HOST2="$(echo "${D_HOSTS}" | cut -d, -f1)"
          HOST3="$(echo "${D_HOSTS}" | cut -d, -f2)"
          MY_HOST="$(echo "${D_HOSTS}" | cut -d, -f$(( ${TPU_WORKER_ID:-0} + 1 )))"
        else
          WORKER_ID="${TPU_WORKER_ID:-0}"
          HOST0="$(echo "${TPU_WORKER_HOSTNAMES}" | cut -d, -f1)"
          HOST1="$(echo "${TPU_WORKER_HOSTNAMES}" | cut -d, -f2)"
          HOST2="$(echo "${TPU_WORKER_HOSTNAMES}" | cut -d, -f3)"
          HOST3="$(echo "${TPU_WORKER_HOSTNAMES}" | cut -d, -f4)"
          MY_HOST="$(echo "${TPU_WORKER_HOSTNAMES}" | cut -d, -f$((WORKER_ID + 1)))"
        fi
      fi
      # runc-proxy's slice-peer wait is best effort, so retry until each peer resolves.
      resolve_pd_ipv6() {
        python3 -u - "$1" <<'PYRESPD'
      import ipaddress, socket, sys, time
      target = sys.argv[1].strip().strip('[]')
      deadline = time.time() + 300
      while True:
        try:
          ip = socket.getaddrinfo(target, None, socket.AF_INET6, socket.SOCK_STREAM)[0][4][0]
          print(ipaddress.ip_address(ip).compressed)
          sys.exit(0)
        except Exception:
          if time.time() >= deadline:
            print(f'FATAL: timed out resolving {target} via /etc/hosts', file=sys.stderr)
            sys.exit(1)
          time.sleep(1)
      PYRESPD
      }
      fmt_host_port() {
        local h="${1#\[}"
        h="${h%%\]}"
        case "$h" in
          *:*) echo "[${h}]:$2" ;;
          *) echo "${h}:$2" ;;
        esac
      }
      IP0="$(resolve_pd_ipv6 "${HOST0}")" || exit 1
      IP2="$(resolve_pd_ipv6 "${HOST2}")" || exit 1
      IP1=""
      IP3=""
      if [ -n "${HOST1}" ]; then IP1="$(resolve_pd_ipv6 "${HOST1}")" || exit 1; fi
      if [ -n "${HOST3}" ]; then IP3="$(resolve_pd_ipv6 "${HOST3}")" || exit 1; fi
      MY_IP="$(resolve_pd_ipv6 "${MY_HOST}")" || exit 1
      export VLLM_HOST_IP="${MY_IP}"
      export TORCH_TPU_SLICEBUILDER_PORT_BASE="${TORCH_TPU_SLICEBUILDER_PORT_BASE:-29000}"
      export TORCH_TPU_BASE_PORT="${TORCH_TPU_BASE_PORT:-29000}"
      if [ "${HOSTS_PER_LEG:-2}" != "1" ]; then
        export VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND:-1}"
        export TPU_MULTIHOST_BACKEND=ray
        export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
        export RAY_memory_monitor_refresh_ms=0
        export RAY_DEDUP_LOGS=0
        unset TPU_VISIBLE_CHIPS RAY_ADDRESS VLLM_LOOPBACK_IP
      else
        unset TPU_MULTIHOST_BACKEND TPU_VISIBLE_CHIPS RAY_ADDRESS
      fi
      export TPU_KV_RESHARD_TRANSPORT=raiden
      export TPU_RAIDEN_RESHARD_IMPL=store
      export TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S=120
      export TPU_KV_TRANSFER_PORT=29100
      export TPU_SIDE_CHANNEL_PORT=29600
      export TPU_P2P_WAIT_PULL_TIMEOUT=600
      export TPU_KV_SHM_POOL_GB=8
      if [ "${GLM_RAIDEN_DEFAULTS:-1}" = "1" ]; then
        export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
        export TPU_RAIDEN_GLM_ADMISSION=1
        export TPU_RAIDEN_POOL_TAGS_PER_LAYER=1
        export TPU_RAIDEN_TRANSFER_PARALLELISM=16
        export VLLM_TPU_INDEXER_KV_CACHE_MODE=has
        export TPU_KV_PIN_SHM=1
        export TPU_PREMAPPED_BUFFER_SIZE=8589934592
        export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=8589934592
        export TPU_ACCELERATOR_TYPE="${TPU_ACCEL_ENV:-tpu7x-16}"
        export CLOUD_TPU_ACCELERATOR_TYPE="${TPU_ACCEL_ENV:-tpu7x-16}"
      fi
      export VLLM_TARGET_DEVICE=tpu
      export JAX_PLATFORMS="tpu,cpu"
      export WORKLOAD_NIC_PREFERRED_NUMA="0,1"
      export TPU_HBM_REPORT_ENABLE="true"
      export TPU_MIN_LOG_LEVEL=0
      export TF_CPP_MAX_VLOG_LEVEL=0
      export MEGASCALE_NUM_SLICES=""
      export MEGASCALE_SLICE_ID=""
      export MEGASCALE_PORT=""
      export MEGASCALE_COORDINATOR_ADDRESS=""
      export VLLM_WORKER_MULTIPROC_METHOD=spawn
      export VLLM_DISABLE_COMPILE_CACHE=0
      export VLLM_CACHE_ROOT="/tmp/workspace/vllm_pd_cache"
      export VLLM_XLA_CACHE_PATH="/tmp/workspace/vllm_pd_cache/xla_cache"
      export TORCH_TPU_CACHE_DIR="/tmp/workspace/torch_tpu_cache"
      export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="/tmp/workspace/torch_tpu_cache"
      export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/tmp/workspace/tier3_cache"
      export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="/tmp/workspace/tier3_cache"
      export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK=0
      export TORCHINDUCTOR_CACHE_DIR="/tmp/workspace/inductor_cache"
      export TORCHINDUCTOR_FX_GRAPH_CACHE=1
      export JAX_COMPILER_CACHE_DIR="/tmp/workspace/jax_cache"
      export JAX_COMPILATION_CACHE_DIR="/tmp/workspace/jax_cache"
      mkdir -p "${VLLM_CACHE_ROOT}" "${VLLM_XLA_CACHE_PATH}" "${TORCH_TPU_CACHE_DIR}" "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${JAX_COMPILER_CACHE_DIR}" /dev/shm/xprof 2>/dev/null || true
      find -L "${VLLM_CACHE_ROOT}" -type f -size 0 -delete 2>/dev/null || true
      export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
      export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
      export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      export TORCHINDUCTOR_AUTOGRAD_CACHE=0
      export TPU_STDERR_LOG_LEVEL=0
      if [ "${ENABLE_PERFETTO:-false}" = "true" ]; then
        export TPU_VMODULE="${TPU_VMODULE:-tpu_pjrt_client=1,pjrt_stream_executor_client=1,tpu_pjrt_compiler_utils=1}"
        export TPU_PJRT_EVENT_TRACE_ENABLE="${TPU_PJRT_EVENT_TRACE_ENABLE:-1}"
      fi
      if [ "${GLM_RAIDEN_DEFAULTS:-1}" = "1" ]; then
        export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS---xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864 --xla_tpu_enable_diagnostic_coordinator=false}"
      fi
      export IP0 IP1 IP2 IP3 HOST0 HOST1 HOST2 HOST3 MY_IP MY_HOST
      PREFILL_WORKERS=2
      [ "${HOSTS_PER_LEG:-2}" = "1" ] && PREFILL_WORKERS=1
      if [ "${WORKER_ID}" -lt "${PREFILL_WORKERS}" ]; then
        LEG="prefill"
        LEG_RANK="${WORKER_ID}"
        LEG_HEAD_HOST="${HOST0}"
        LEG_HEAD_IP="${IP0}"
        export TPU_NODE_ID="${LEG_RANK}"
        export TPU_WORKER_ID="${LEG_RANK}"
        export CLOUD_TPU_TASK_ID="${LEG_RANK}"
        if [ "${HOSTS_PER_LEG:-2}" = "1" ]; then
          if [ -z "${PD_LEG:-}" ]; then
            export ENABLE_TPUNETD_CLIENT="false"
            export TPU_TOPOLOGY="2x2x1"
            export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
            export TPU_HOST_BOUNDS="1,1,1"
            export TPU_PROCESS_BOUNDS="1,1,1"
            export TPU_WORKER_HOSTNAMES=""
            export TPU_PROCESS_ADDRESSES=""
            export TPU_PROCESS_PORT=""
          fi
          export EXPECTED_TPU_DEVICES="${CHIPS_PER_HOST:-4}"
        else
          if [ -z "${PD_LEG:-}" ]; then
            export TPU_WORKER_HOSTNAMES="${HOST0},${HOST1}"
            export TPU_PROCESS_ADDRESSES="$(fmt_host_port "${HOST0}" 8471),$(fmt_host_port "${HOST1}" 8471)"
            if [ "${TPU_CHIPS_PER_HOST_BOUNDS:-}" = "4,1,1" ]; then
              export TPU_HOST_BOUNDS="1,2,1"
              export TPU_PROCESS_BOUNDS="1,2,1"
            else
              export TPU_HOST_BOUNDS="1,1,2"
              export TPU_PROCESS_BOUNDS="1,1,2"
            fi
          fi
          export EXPECTED_TPU_DEVICES="${RAY_LEG_EXPECTED_CHIPS:-8}"
        fi
        export TPU_RAIDEN_JOB_NAME="prefill"
        export TPU_RAIDEN_ADVERTISE_HOST="${HOST0}"
        export TPU_RAIDEN_RESHARD_PORT_BASE="27000"
        export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE="27100"
        export TPU_LOG_DIR="/tmp/tpu_logs/vllm-tpu-prefill"
    ||| + legExportLines(d.PREFILL_ENV) + |||
      # Apply runtime EXTRA_ENV_PREFILL overrides when provided.
        if [ -n "${EXTRA_ENV_PREFILL:-}" ]; then
          IFS=';' read -r -a _extra_kvs <<< "${EXTRA_ENV_PREFILL}"
          for _kv in "${_extra_kvs[@]}"; do [ -n "${_kv}" ] && export "${_kv}"; done
        fi
      else
        LEG="decode"
        LEG_RANK="$((WORKER_ID - PREFILL_WORKERS))"
        LEG_HEAD_HOST="${HOST2}"
        LEG_HEAD_IP="${IP2}"
        export TPU_NODE_ID="${LEG_RANK}"
        export TPU_WORKER_ID="${LEG_RANK}"
        export CLOUD_TPU_TASK_ID="${LEG_RANK}"
        if [ "${HOSTS_PER_LEG:-2}" = "1" ]; then
          if [ -z "${PD_LEG:-}" ]; then
            export ENABLE_TPUNETD_CLIENT="false"
            export TPU_TOPOLOGY="2x2x1"
            export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
            export TPU_HOST_BOUNDS="1,1,1"
            export TPU_PROCESS_BOUNDS="1,1,1"
            export TPU_WORKER_HOSTNAMES=""
            export TPU_PROCESS_ADDRESSES=""
            export TPU_PROCESS_PORT=""
          fi
          export EXPECTED_TPU_DEVICES="${CHIPS_PER_HOST:-4}"
        else
          if [ -z "${PD_LEG:-}" ]; then
            export TPU_WORKER_HOSTNAMES="${HOST2},${HOST3}"
            export TPU_PROCESS_ADDRESSES="$(fmt_host_port "${HOST2}" 8471),$(fmt_host_port "${HOST3}" 8471)"
            if [ "${TPU_CHIPS_PER_HOST_BOUNDS:-}" = "4,1,1" ]; then
              export TPU_HOST_BOUNDS="1,2,1"
              export TPU_PROCESS_BOUNDS="1,2,1"
            else
              export TPU_HOST_BOUNDS="1,1,2"
              export TPU_PROCESS_BOUNDS="1,1,2"
            fi
          fi
          export EXPECTED_TPU_DEVICES="${RAY_LEG_EXPECTED_CHIPS:-8}"
        fi
        export TPU_RAIDEN_JOB_NAME="decode"
        export TPU_RAIDEN_ADVERTISE_HOST="${HOST2}"
        export TPU_RAIDEN_RESHARD_PORT_BASE="28000"
        export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE="28100"
        export TPU_LOG_DIR="/tmp/tpu_logs/vllm-tpu-decode"
    ||| + legExportLines(d.DECODE_ENV) + (
      if std.objectHas(d.DECODE_ENV, 'TPU_TOKEN_BUCKET_EXTRA') then '' else '  unset TPU_TOKEN_BUCKET_EXTRA\n'
    ) + |||
      # Apply runtime EXTRA_ENV_DECODE overrides when provided.
        if [ -n "${EXTRA_ENV_DECODE:-}" ]; then
          IFS=';' read -r -a _extra_kvs <<< "${EXTRA_ENV_DECODE}"
          for _kv in "${_extra_kvs[@]}"; do [ -n "${_kv}" ] && export "${_kv}"; done
        fi
        if [ "${ENABLE_PERFETTO:-false}" != "true" ]; then
          unset TPU_VMODULE TPU_PJRT_EVENT_TRACE_ENABLE
        fi
      fi
      mkdir -p "${TPU_LOG_DIR}" 2>/dev/null || true
      echo "=== ${EXOS_RECIPE}: worker=${WORKER_ID} leg=${LEG} leg_rank=${LEG_RANK} my_host=${MY_HOST} my_ip=${MY_IP} leg_head=${LEG_HEAD_HOST} (${LEG_HEAD_IP}) hosts_per_leg=${HOSTS_PER_LEG:-2} ==="
      if [ "${HOSTS_PER_LEG:-2}" = "1" ]; then
        pick_port() { python3 -c 'import socket; s=socket.socket(socket.AF_INET6); s.bind(("::1", 0)); print(s.getsockname()[1])'; }
        export MASTER_ADDR="::1"
        export VLLM_LOOPBACK_IP="::1"
        export VLLM_DP_MASTER_IP="::1"
        export TORCH_TPU_DP_MASTER_ADDR="::1"
        export PYTHONHASHSEED="0"
        export MASTER_PORT="${MASTER_PORT:-$(pick_port)}"
        export VLLM_DP_PORT="${VLLM_DP_PORT:-$(pick_port)}"
      else
        export MASTER_ADDR="${LEG_HEAD_HOST}"
        if [ "${LEG_RANK}" != "0" ]; then
          for _ in $(seq 1 180); do
            python3 -c 'import socket,sys; s=socket.socket(socket.AF_INET6); s.settimeout(2); s.connect((sys.argv[1], 26379)); s.close()' "${LEG_HEAD_IP}" 2>/dev/null && break
            sleep 5
          done
          ray start --address="[${LEG_HEAD_IP}]:26379" --node-ip-address="${MY_IP}" \
            --min-worker-port=24001 --max-worker-port=24999 \
            --node-manager-port=23001 --object-manager-port=23003 \
            --dashboard-agent-grpc-port=23007 --dashboard-agent-listen-port=23009 \
            --metrics-export-port=23011 --runtime-env-agent-port=23013 \
            --object-store-memory=4294967296 \
            --labels="ray.io/tpu-worker-id=${LEG_RANK},ray.io/tpu-topology=${RAY_TPU_TOPOLOGY:-2x2x2},ray.io/tpu-pod-type=${RAY_POD_TYPE:-v7x-16},ray.io/accelerator-type=${RAY_ACCEL_TYPE:-TPU-V7X}" \
            --resources="{\"TPU\": ${RAY_TPU_PER_HOST:-8}, \"accelerator_type:${RAY_ACCEL_TYPE:-TPU-V7X}\": 1}" --disable-usage-stats || {
              echo "=== RAY WORKER LOGS ==="
              for f in /tmp/ray/session_latest/logs/*; do echo "--- $f ---"; tail -n 50 "$f" 2>/dev/null || true; done
              exit 1
            }
          export RAY_ADDRESS="[${LEG_HEAD_IP}]:26379"
          while python3 -c 'import socket,sys; s=socket.socket(socket.AF_INET6); s.settimeout(2); s.connect((sys.argv[1], 26379)); s.close()' "${LEG_HEAD_IP}" 2>/dev/null; do
            sleep 10
          done
          ray stop --force 2>/dev/null || true
          echo 'EXOS_VLLM_RESULT {"pass": true, "role": "'"${LEG}"'-ray-worker", "worker_id": '"${WORKER_ID}"'}'
          exit 0
        fi
        ray start --head --node-ip-address="${MY_IP}" --port=26379 \
          --min-worker-port=24001 --max-worker-port=24999 \
          --node-manager-port=23001 --object-manager-port=23003 \
          --ray-client-server-port=23005 \
          --dashboard-agent-grpc-port=23007 --dashboard-agent-listen-port=23009 \
          --metrics-export-port=23011 --runtime-env-agent-port=23013 \
          --object-store-memory=4294967296 \
          --labels="ray.io/tpu-worker-id=${LEG_RANK},ray.io/tpu-topology=${RAY_TPU_TOPOLOGY:-2x2x2},ray.io/tpu-pod-type=${RAY_POD_TYPE:-v7x-16},ray.io/accelerator-type=${RAY_ACCEL_TYPE:-TPU-V7X}" \
          --resources="{\"TPU\": ${RAY_TPU_PER_HOST:-8}, \"accelerator_type:${RAY_ACCEL_TYPE:-TPU-V7X}\": 1}" --include-dashboard=false --disable-usage-stats || {
            echo "=== RAY HEAD LOGS ==="
            for f in /tmp/ray/session_latest/logs/*; do echo "--- $f ---"; tail -n 50 "$f" 2>/dev/null || true; done
            exit 1
          }
        export RAY_ADDRESS="[${MY_IP}]:26379"
        for _ in $(seq 1 180); do
          READY_TPUS="$(python3 -c 'import ray; ray.init(address="auto", ignore_reinit_error=True); print(int(ray.cluster_resources().get("TPU", 0)))' 2>/dev/null || echo 0)"
          echo "${LEG} Ray cluster TPUs: ${READY_TPUS}/${RAY_LEG_EXPECTED_TPUS:-16}"
          [ "${READY_TPUS}" -ge "${RAY_LEG_EXPECTED_TPUS:-16}" ] && break
          sleep 5
        done
      fi
      resolve_leg_model_and_load_args() {
        local leg_extra="$1"
        SERVER_MODEL="${MODEL}"
        LOAD_ARGS=""
        local extra_has_dummy=0
        case " ${leg_extra} " in
          *" --load-format=dummy "*|*" --load-format dummy "*) extra_has_dummy=1 ;;
        esac
        if [[ "${MODEL}" == gs://* ]]; then
          if [ "${LOAD_FORMAT:-}" = "dummy" ] || [ "${extra_has_dummy}" = "1" ]; then
            SERVER_MODEL="/tmp/model_meta_local"
            mkdir -p "${SERVER_MODEL}"
            python3 /tmp/exos_gcs_cp.py tokenizer "${MODEL}" "${SERVER_MODEL}"
            [ "${extra_has_dummy}" = "0" ] && LOAD_ARGS="--load-format dummy"
          elif [ "${USE_RUNAI_STREAMER:-true}" = "false" ] || [ "${USE_RUNAI_STREAMER:-true}" = "0" ]; then
            SERVER_MODEL="/tmp/workspace/model_local"
            mkdir -p "${SERVER_MODEL}"
            python3 /tmp/exos_gcs_cp.py all "${MODEL}" "${SERVER_MODEL}"
            [ -n "${LOAD_FORMAT:-}" ] && LOAD_ARGS="--load-format ${LOAD_FORMAT}"
          elif [ -n "${LOAD_FORMAT:-}" ]; then
            LOAD_ARGS="--load-format ${LOAD_FORMAT}"
          else
            LOAD_ARGS="--load-format runai_streamer"
          fi
        elif [ -n "${LOAD_FORMAT:-}" ] && [ "${extra_has_dummy}" = "0" ]; then
          LOAD_ARGS="--load-format ${LOAD_FORMAT}"
        fi
      }
      if [ "${LEG}" = "decode" ]; then
        KV_CONFIG="${DECODE_KV_CONFIG:-}"
        if [ -z "${KV_CONFIG}" ]; then
          KV_CONFIG='{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer","kv_port":14579}'
        fi
        resolve_leg_model_and_load_args "${DECODE_EXTRA_ARGS:-}"
        LOAD_ARGS_ARR=()
        [ -n "${LOAD_ARGS}" ] && read -r -a LOAD_ARGS_ARR <<< "${LOAD_ARGS}"
        readarray -d '' DECODE_EXTRA_ARGS_ARR < <(python3 -c 'import os, re, shlex, sys; s = re.sub(r"(--[A-Za-z0-9_.-]+=)(\{[^\s\x27]*\})(?=\s|$)", lambda m: m.group(1) + "\x27" + m.group(2) + "\x27", os.environ.get("DECODE_EXTRA_ARGS", "")); sys.stdout.buffer.write(b"".join(a.encode() + b"\0" for a in shlex.split(s)))')
        echo "trace-start-event: name=vllm-decode ts=$(date +'%s.%N') pid=decode tid=decode"
        vllm serve "${SERVER_MODEL}" --served-model-name "${SERVED_MODEL_NAME}" \
          --max-model-len "${MAX_MODEL_LEN}" \
          --host "${MY_IP}" --port 25100 \
          ${LOAD_ARGS_ARR[@]+"${LOAD_ARGS_ARR[@]}"} \
          ${DECODE_EXTRA_ARGS_ARR[@]+"${DECODE_EXTRA_ARGS_ARR[@]}"} \
          --kv-transfer-config "${KV_CONFIG}" \
          --no-enable-log-requests --disable-uvicorn-access-log \
          > >(tee -a /tmp/vllm_serve.log) 2>&1 &
        DECODE_PID=$!
        MONITOR_PORT=26379
        [ "${HOSTS_PER_LEG:-2}" = "1" ] && MONITOR_PORT=25200
        prefill_monitor_up() {
          python3 -c 'import socket,sys
      for ip in sys.argv[2:]:
        if not ip: continue
        try:
          s=socket.socket(socket.AF_INET6); s.settimeout(2); s.connect((ip, int(sys.argv[1]))); s.close(); sys.exit(0)
        except OSError:
          pass
      sys.exit(1)' "${MONITOR_PORT}" "${IP0}" "${IP1:-}" >/dev/null 2>&1
        }
        WAIT_ITERS="$(( ${STARTUP_TIMEOUT:-7200} / 5 ))"
        for _ in $(seq 1 "${WAIT_ITERS}"); do
          [ -d "/proc/${DECODE_PID}" ] || break
          prefill_monitor_up && break
          sleep 5
        done
        while [ -d "/proc/${DECODE_PID}" ] && prefill_monitor_up; do
          sleep 10
        done
        if [ -n "${CDK_OUTPUT_DIR:-}" ] && [ -f /tmp/vllm_serve.log ]; then
          mkdir -p "${CDK_OUTPUT_DIR}" 2>/dev/null || true
          cp -f /tmp/vllm_serve.log "${CDK_OUTPUT_DIR}/decode.log" 2>/dev/null || true
        fi
        if [ ! -d "/proc/${DECODE_PID}" ] && prefill_monitor_up; then
          echo "=== DECODE VLLM CRASH DIAGNOSTICS (/tmp/vllm_serve.log) ==="
          grep -a -vE '(ray_executor_v2\.py:57[0-9]|ActorHandleNotFoundError|tpu_pjrt_client\.cc|pjrt_stream_executor_client\.cc|tpu_pjrt_compiler_utils\.cc|tpu_chip_config\.cc|deepsea_compiler_)' /tmp/vllm_serve.log 2>/dev/null | grep -a -E -C 5 '(Traceback|Error|Exception|FATAL|Check failed|closed its address pipe|Engine core initialization failed)' | tail -n 105 | awk '{print; if (NR % 35 == 0) system("sleep 6")}' || true
          sleep 6
          echo "=== DECODE VLLM TAIL (/tmp/vllm_serve.log) ==="
          grep -a -vE '(ray_executor_v2\.py:57[0-9]|ActorHandleNotFoundError|tpu_pjrt_client\.cc|pjrt_stream_executor_client\.cc|tpu_pjrt_compiler_utils\.cc|tpu_chip_config\.cc|deepsea_compiler_)' /tmp/vllm_serve.log 2>/dev/null | tail -n 70 | awk '{print; if (NR % 35 == 0) system("sleep 6")}' || true
          if [ "${HOSTS_PER_LEG:-2}" != "1" ]; then
            sleep 6
            echo "=== DECODE RAY SESSION LOGS ==="
            for f in /tmp/ray/session_latest/logs/worker*.err /tmp/ray/session_latest/logs/worker*.out /tmp/ray/session_latest/logs/gcs_server.err /tmp/ray/session_latest/logs/raylet.err; do
              if [ -s "$f" ]; then echo "--- $f ---"; tail -n 30 "$f" 2>/dev/null || true; sleep 2; fi
            done
            ray stop --force 2>/dev/null || true
          fi
          exit 1
        fi
        kill "${DECODE_PID}" 2>/dev/null || true
        echo "trace-end-event: name=vllm-decode ts=$(date +'%s.%N') pid=decode tid=decode"
        [ "${HOSTS_PER_LEG:-2}" != "1" ] && ray stop --force 2>/dev/null || true
        echo 'EXOS_VLLM_RESULT {"pass": true, "role": "decode-head", "worker_id": '"${WORKER_ID}"'}'
        exit 0
      fi
      # Prefill head (worker 0): launch Prefill vLLM on port 25200, proxy on port 25000, and run benchmark.
      KV_CONFIG="${PREFILL_KV_CONFIG:-}"
      if [ -z "${KV_CONFIG}" ]; then
        KV_CONFIG="{\"kv_connector\":\"TPURaidenConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"kv_producer\",\"kv_port\":14579,\"kv_connector_extra_config\":{\"destination_decode_context_parallel_size\":${DEST_DCP_SIZE},\"destination_cp_kv_cache_interleave_size\":${DEST_CP_INTERLEAVE}}}"
      fi
      resolve_leg_model_and_load_args "${PREFILL_EXTRA_ARGS:-}"
      LOAD_ARGS_ARR=()
      [ -n "${LOAD_ARGS}" ] && read -r -a LOAD_ARGS_ARR <<< "${LOAD_ARGS}"
      readarray -d '' PREFILL_EXTRA_ARGS_ARR < <(python3 -c 'import os, re, shlex, sys; s = re.sub(r"(--[A-Za-z0-9_.-]+=)(\{[^\s\x27]*\})(?=\s|$)", lambda m: m.group(1) + "\x27" + m.group(2) + "\x27", os.environ.get("PREFILL_EXTRA_ARGS", "")); sys.stdout.buffer.write(b"".join(a.encode() + b"\0" for a in shlex.split(s)))')
      echo "trace-start-event: name=vllm-prefill ts=$(date +'%s.%N') pid=prefill tid=prefill"
      vllm serve "${SERVER_MODEL}" --served-model-name "${SERVED_MODEL_NAME}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --host "${MY_IP}" --port 25200 \
        ${LOAD_ARGS_ARR[@]+"${LOAD_ARGS_ARR[@]}"} \
        ${PREFILL_EXTRA_ARGS_ARR[@]+"${PREFILL_EXTRA_ARGS_ARR[@]}"} \
        --kv-transfer-config "${KV_CONFIG}" \
        --no-enable-log-requests --disable-uvicorn-access-log \
        > >(tee -a /tmp/vllm_serve.log) 2>&1 &
      export SERVER_PID=$!
      _pd_wait_out="$(python3 -u - <<'PYWAITPD'
      import json, os, sys, time, urllib.request

      deadline = time.time() + int(os.environ['STARTUP_TIMEOUT'])
      start = time.time()
      p_url = 'http://[' + os.environ['MY_IP'] + ']:25200/health'
      d_cands = [(os.environ['HOST2'], os.environ['IP2'])]
      if os.environ.get('HOST3') and os.environ.get('IP3'):
        d_cands.append((os.environ['HOST3'], os.environ['IP3']))
      while time.time() < deadline:
        if not os.path.exists('/proc/' + os.environ['SERVER_PID']):
          print('EXOS_VLLM_RESULT ' + json.dumps({'pass': False, 'error': 'prefill vllm exited'}), file=sys.stderr)
          sys.exit(1)
        p_ok = False
        try:
          with urllib.request.urlopen(p_url, timeout=5) as rp:
            p_ok = (rp.status == 200)
        except Exception:
          pass
        if p_ok:
          for d_host, d_ip in d_cands:
            try:
              with urllib.request.urlopen('http://[' + d_ip + ']:25100/health', timeout=5) as rd:
                if rd.status == 200:
                  print(f'{round(time.time() - start, 1)}|{d_host}|{d_ip}')
                  sys.exit(0)
            except Exception:
              pass
        time.sleep(10)
      print('EXOS_VLLM_RESULT ' + json.dumps({'pass': False, 'error': 'pd servers not healthy'}), file=sys.stderr)
      sys.exit(1)
      PYWAITPD
      )" || {
        echo "1" > "${SYNC_DIR}/failed" 2>/dev/null || true
        if [ -n "${CDK_OUTPUT_DIR:-}" ] && [ -f /tmp/vllm_serve.log ]; then
          mkdir -p "${CDK_OUTPUT_DIR}" 2>/dev/null || true
          cp -f /tmp/vllm_serve.log "${CDK_OUTPUT_DIR}/prefill.log" 2>/dev/null || true
        fi
        echo "=== PREFILL VLLM CRASH DIAGNOSTICS (/tmp/vllm_serve.log) ==="
        grep -a -vE '(ray_executor_v2\.py:57[0-9]|ActorHandleNotFoundError|tpu_pjrt_client\.cc|pjrt_stream_executor_client\.cc|tpu_pjrt_compiler_utils\.cc|tpu_chip_config\.cc|deepsea_compiler_)' /tmp/vllm_serve.log 2>/dev/null | grep -a -E -C 5 '(Traceback|Error|Exception|FATAL|Check failed|closed its address pipe|Engine core initialization failed)' | tail -n 105 | awk '{print; if (NR % 35 == 0) system("sleep 6")}' || true
        sleep 6
        echo "=== PREFILL VLLM TAIL (/tmp/vllm_serve.log) ==="
        grep -a -vE '(ray_executor_v2\.py:57[0-9]|ActorHandleNotFoundError|tpu_pjrt_client\.cc|pjrt_stream_executor_client\.cc|tpu_pjrt_compiler_utils\.cc|tpu_chip_config\.cc|deepsea_compiler_)' /tmp/vllm_serve.log 2>/dev/null | tail -n 70 | awk '{print; if (NR % 35 == 0) system("sleep 6")}' || true
        if [ "${HOSTS_PER_LEG:-2}" != "1" ]; then
          sleep 6
          echo "=== PREFILL RAY SESSION LOGS ==="
          for f in /tmp/ray/session_latest/logs/worker*.err /tmp/ray/session_latest/logs/worker*.out /tmp/ray/session_latest/logs/gcs_server.err /tmp/ray/session_latest/logs/raylet.err; do
            if [ -s "$f" ]; then echo "--- $f ---"; tail -n 30 "$f" 2>/dev/null || true; sleep 2; fi
          done
          ray stop --force 2>/dev/null || true
        fi
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
      }
      IFS='|' read -r EXOS_STARTUP_S ACTIVE_DECODE_HOST ACTIVE_DECODE_IP <<< "$(printf '%s\n' "${_pd_wait_out}" | tail -n 1)"
      export EXOS_STARTUP_S ACTIVE_DECODE_HOST ACTIVE_DECODE_IP
      export ACTIVE_PREFILL_HOST="${MY_HOST}" ACTIVE_PREFILL_IP="${MY_IP}"
      PROXY_PY="/root/torchtpu-vllm/examples/disagg/toy_proxy_server.py"
      for cand in \
        "/root/torchtpu-vllm/examples/disagg/toy_proxy_server.py" \
        "/workspace/vllm-torchtpu/examples/disagg/toy_proxy_server.py" \
        "/root/cloud-devkit/vllm-torchtpu/examples/disagg/toy_proxy_server.py"; do
        if [ -f "${cand}" ]; then
          PROXY_PY="${cand}"
          break
        fi
      done
      python3 - "${PROXY_PY}" > >(sed -u 's/^/[proxy] /') 2>&1 <<'PYPROXY' &
      import os, runpy, sys, uvicorn

      proxy_py = sys.argv[1]
      src = open(proxy_py, 'r', encoding='utf-8', errors='ignore').read()
      run_target = proxy_py
      old_mt = 'media_type="application/json")'
      new_mt = 'media_type=("text/event-stream" if req_data.get("stream") else "application/json"))'
      if old_mt in src and 'text/event-stream' not in src:
        run_target = '/tmp/exos_toy_proxy_server.py'
        with open(run_target, 'w', encoding='utf-8') as wf:
          wf.write(src.replace(old_mt, new_mt, 1))
      _orig_run = uvicorn.run

      def _patched_run(*a, **kw):
        kw.setdefault('timeout_keep_alive', int(os.environ.get('PROXY_CLIENT_KEEPALIVE_S', '3600')))
        return _orig_run(*a, **kw)

      uvicorn.run = _patched_run
      sys.path.insert(0, os.path.dirname(os.path.abspath(proxy_py)))
      def _url_host(name):
        h = os.environ[name].strip().strip('[]')
        return '[' + h + ']' if ':' in h else h

      prefill_host = _url_host('ACTIVE_PREFILL_HOST')
      decode_host = _url_host('ACTIVE_DECODE_HOST')
      if '--prefiller-hosts' in src:
        sys.argv = [
            proxy_py,
            '--host', '::1', '--port', '25000',
            '--prefiller-hosts', prefill_host, '--prefiller-ports', '25200',
            '--decoder-hosts', decode_host, '--decoder-ports', '25100',
        ]
      else:
        sys.argv = [
            proxy_py,
            '--host', '::1', '--port', '25000',
            '--prefiller-host', prefill_host, '--prefiller-port', '25200',
            '--decoder-host', decode_host, '--decoder-port', '25100',
        ]
      runpy.run_path(run_target, run_name='__main__')
      PYPROXY
      export PROXY_PID=$!
      for _ in $(seq 1 30); do
        [ -d "/proc/${PROXY_PID}" ] || break
        python3 -c 'import socket; s=socket.socket(socket.AF_INET6); s.settimeout(2); s.connect(("::1", 25000)); s.close()' 2>/dev/null && break
        sleep 2
      done
      if [ "${PROXY_PROBE:-0}" = "1" ]; then
        python3 -u - <<'PYPROBEPD' || {
      import json, os, sys, time, urllib.request

      body = json.dumps({
          'model': os.environ['SERVED_MODEL_NAME'],
          'messages': [{'role': 'user', 'content': 'What is 2+3? Answer with one number.'}],
          'max_tokens': 8,
          'temperature': 0,
          'stream': False,
      }).encode('utf-8')
      for attempt in range(1, 61):
        if not os.path.exists('/proc/' + os.environ['SERVER_PID']):
          print('ERROR: prefill vllm exited during proxy probe', file=sys.stderr)
          sys.exit(1)
        if not os.path.exists('/proc/' + os.environ['PROXY_PID']):
          print('ERROR: toy_proxy_server exited during proxy probe', file=sys.stderr)
          sys.exit(1)
        try:
          req = urllib.request.Request(
              'http://[::1]:25000/v1/chat/completions',
              data=body,
              headers={'Content-Type': 'application/json'},
          )
          with urllib.request.urlopen(req, timeout=300) as resp:
            payload = resp.read().decode('utf-8', errors='replace')
            if 200 <= resp.status < 300:
              print(f'proxy probe attempt={attempt} status={resp.status}: {payload[:400]}')
              sys.exit(0)
        except Exception as exc:
          print(f'proxy probe attempt={attempt} error: {exc}')
        time.sleep(10)
      print('EXOS_VLLM_RESULT ' + json.dumps({'pass': False, 'error': 'proxy probe prefill->decode failed'}), file=sys.stderr)
      sys.exit(1)
      PYPROBEPD
          echo "1" > "${SYNC_DIR}/failed" 2>/dev/null || true
          echo "=== PROXY PROBE FAILURE DIAGNOSTICS (/tmp/vllm_serve.log) ==="
          tail -n 80 /tmp/vllm_serve.log 2>/dev/null || true
          kill "${PROXY_PID}" "${SERVER_PID}" 2>/dev/null || true
          [ "${HOSTS_PER_LEG:-2}" != "1" ] && ray stop --force 2>/dev/null || true
          exit 1
        }
      fi
      export SERVE_HOST="::1" SERVE_URL_HOST="[::1]" SERVE_PORT="25000"
      export METRICS_URL="http://[${ACTIVE_PREFILL_IP}]:25200/metrics"
      export DECODE_METRICS_URL="http://[${ACTIVE_DECODE_IP}]:25100/metrics"
      if [ "${USE_SIDECAR_BENCH:-0}" = "1" ]; then
        cat > "${SYNC_DIR}/ready.tmp" <<EOF
      SERVE_HOST="${SERVE_HOST}"
      SERVE_URL_HOST="${SERVE_URL_HOST}"
      SERVE_PORT="${SERVE_PORT}"
      METRICS_URL="${METRICS_URL}"
      DECODE_METRICS_URL="${DECODE_METRICS_URL}"
      EXOS_STARTUP_S="${EXOS_STARTUP_S}"
      EOF
        mv "${SYNC_DIR}/ready.tmp" "${SYNC_DIR}/ready"
        echo "=== ${EXOS_RECIPE}: waiting for sidecar-bench to complete (${SYNC_DIR}/done) ==="
        while [ ! -f "${SYNC_DIR}/done" ]; do
          if [ ! -d "/proc/${SERVER_PID}" ]; then
            echo "ERROR: prefill vllm exited while sidecar-bench was running"
            echo "1" > "${SYNC_DIR}/failed"
            break
          fi
          sleep 5
        done
        RC="$(cat "${SYNC_DIR}/done" 2>/dev/null || echo 1)"
      else
    ||| + (if useSidecar then '        echo "ERROR: USE_SIDECAR_BENCH=0 in sidecar recipe"\n        RC=1\n' else self.benchScriptFor(benchMode)) + |||
      RC=$?
      fi
      kill "${PROXY_PID}" "${SERVER_PID}" 2>/dev/null || true
      if [ -n "${CDK_OUTPUT_DIR:-}" ] && [ -f /tmp/vllm_serve.log ]; then
        mkdir -p "${CDK_OUTPUT_DIR}" 2>/dev/null || true
        cp -f /tmp/vllm_serve.log "${CDK_OUTPUT_DIR}/prefill.log" 2>/dev/null || true
      fi
      echo "trace-end-event: name=vllm-prefill ts=$(date +'%s.%N') pid=prefill tid=prefill"
      [ "${HOSTS_PER_LEG:-2}" != "1" ] && ray stop --force 2>/dev/null || true
      exit "${RC}"
    |||;

    // The 1PnD changes to the one-host P/D script, applied only when DECODE_REPLICAS > 1 so the
    // 1P1D script stays byte-identical. Each anchor must occur exactly once.
    // Counted through strReplace (a builtin): std.findSubstr is jsonnet-level and too slow on a
    // 65 KB script.
    local replaceOnce(s, old, new) =
      local n = (std.length(s) - std.length(std.strReplace(s, old, ''))) / std.length(old);
      assert n == 1 : 'exos_vllm 1PnD patch: expected one "%s" in the P/D script, found %d' % [old, n];
      std.strReplace(s, old, new);
    local multiDecodePatches = [
      // Decode i serves on, and advertises, the i-th DECODE_HOSTNAMES entry.
      [
        'MY_HOST="${HOST2}"',
        'HOST2="$(echo "${D_HOSTS}" | cut -d, -f$(( ${PD_DECODE_INDEX:-0} + 1 )))"; MY_HOST="${HOST2}"',
      ],
      // Raiden engine ids must be unique: decode i > 0 is decode-<i>-engine.
      [
        '# Apply runtime EXTRA_ENV_DECODE overrides when provided.',
        'if [ "${PD_DECODE_INDEX:-0}" != "0" ]; then export TPU_RAIDEN_ENGINE_ID="decode-${PD_DECODE_INDEX}-engine"; fi\n'
        + '# Apply runtime EXTRA_ENV_DECODE overrides when provided.',
      ],
      // The prefill resolves every decode: PD_DECODE_ALL=host|ip,host|ip,...
      [
        'MY_IP="$(resolve_pd_ipv6 "${MY_HOST}")" || exit 1',
        |||
          MY_IP="$(resolve_pd_ipv6 "${MY_HOST}")" || exit 1
          PD_DECODE_ALL=""
          if [ "${PD_LEG:-}" = "prefill" ]; then
            IFS=',' read -r -a _pd_dhosts <<< "${D_HOSTS}"
            for _pd_h in "${_pd_dhosts[@]}"; do
              _pd_ip="$(resolve_pd_ipv6 "${_pd_h}")" || exit 1
              PD_DECODE_ALL="${PD_DECODE_ALL:+${PD_DECODE_ALL},}${_pd_h}|${_pd_ip}"
            done
            echo "=== ${EXOS_RECIPE}: ${#_pd_dhosts[@]} decodes: ${PD_DECODE_ALL} ==="
          fi
          export PD_DECODE_ALL
        |||,
      ],
      // The prefill is ready only when every decode answers /health.
      // (Inside the wait loop's body, so indented by two.)
      [
        '\n  if p_ok:\n',
        '\n' + std.join('\n', [if l == '' then l else '  ' + l for l in std.split(|||
          d_all = [tuple(x.split('|', 1)) for x in os.environ.get('PD_DECODE_ALL', '').split(',') if x]
          if p_ok and d_all:
            def _d_up(ip):
              try:
                with urllib.request.urlopen('http://[' + ip + ']:25100/health', timeout=5) as rd:
                  return rd.status == 200
              except Exception:
                return False
            d_up = [h for h, ip in d_all if _d_up(ip)]
            print(f'decodes healthy: {len(d_up)}/{len(d_all)}', file=sys.stderr)
            if len(d_up) == len(d_all):
              print(f'{round(time.time() - start, 1)}|{d_all[0][0]}|{d_all[0][1]}')
              sys.exit(0)
            p_ok = False
          if p_ok:
        |||, '\n')]),
      ],
      // The proxy spreads requests over every decode.
      [
        "decode_host = _url_host('ACTIVE_DECODE_HOST')",
        "decode_host = _url_host('ACTIVE_DECODE_HOST')\n"
        + "decode_hosts = [('[' + x.split('|', 1)[0] + ']' if ':' in x.split('|', 1)[0] else x.split('|', 1)[0]) for x in os.environ.get('PD_DECODE_ALL', '').split(',') if x] or [decode_host]",
      ],
      [
        "'--decoder-hosts', decode_host, '--decoder-ports', '25100',",
        "'--decoder-hosts', *decode_hosts, '--decoder-ports', *(['25100'] * len(decode_hosts)),",
      ],
    ];
    // The 2PnD changes to the one-host P/D script, applied only when PREFILL_REPLICAS > 1 and BEFORE the 1PnD patches (each 1PnD
    // anchor still occurs exactly once after them), so the 1P1D and 1PnD scripts stay byte-identical. Each anchor must occur once.
    local replaceOnceP(s, old, new) =
      local n = (std.length(s) - std.length(std.strReplace(s, old, ''))) / std.length(old);
      assert n == 1 : 'exos_vllm 2PnD patch: expected one "%s" in the P/D script, found %d' % [old, n];
      std.strReplace(s, old, new);
    local multiPrefillPatches = [
      // Prefill i serves on, and advertises, the i-th PREFILL_HOSTNAMES entry (HOST0 is the leg's own host in the prefill
      // branch: Raiden advertise host, leg head); PD_P0_HOST keeps prefill 0's host.
      [
        'MY_HOST="${HOST0}"',
        'PD_P0_HOST="${HOST0}"; HOST0="$(echo "${P_HOSTS}" | cut -d, -f$(( ${PD_PREFILL_INDEX:-0} + 1 )))"; MY_HOST="${HOST0}"',
      ],
      // Raiden engine ids must be unique: prefill i > 0 is prefill-<i>-engine.
      [
        '# Apply runtime EXTRA_ENV_PREFILL overrides when provided.',
        'if [ "${PD_PREFILL_INDEX:-0}" != "0" ]; then export TPU_RAIDEN_ENGINE_ID="prefill-${PD_PREFILL_INDEX}-engine"; fi\n'
        + '# Apply runtime EXTRA_ENV_PREFILL overrides when provided.',
      ],
      // Prefill 0 resolves every other prefill (PD_PREFILL_ALL=host|ip,...), which it waits for and its proxy routes to; prefill
      // i > 0 resolves prefill 0 (PD_P0_IP), whose serving port it watches.
      [
        'MY_IP="$(resolve_pd_ipv6 "${MY_HOST}")" || exit 1',
        'MY_IP="$(resolve_pd_ipv6 "${MY_HOST}")" || exit 1\n' + std.rstripChars(|||
          PD_PREFILL_ALL=""
          PD_P0_IP=""
          if [ "${PD_LEG:-}" = "prefill" ]; then
            if [ "${PD_PREFILL_INDEX:-0}" = "0" ]; then
              IFS=',' read -r -a _pd_phosts <<< "${P_HOSTS}"
              for _pd_h in "${_pd_phosts[@]:1}"; do
                _pd_ip="$(resolve_pd_ipv6 "${_pd_h}")" || exit 1
                PD_PREFILL_ALL="${PD_PREFILL_ALL:+${PD_PREFILL_ALL},}${_pd_h}|${_pd_ip}"
              done
              echo "=== ${EXOS_RECIPE}: ${#_pd_phosts[@]} prefills: ${MY_HOST}|${MY_IP}${PD_PREFILL_ALL:+,${PD_PREFILL_ALL}} ==="
            else
              PD_P0_IP="$(resolve_pd_ipv6 "${PD_P0_HOST}")" || exit 1
              echo "=== ${EXOS_RECIPE}: prefill ${PD_PREFILL_INDEX} on ${MY_HOST}|${MY_IP}; prefill 0 is ${PD_P0_HOST}|${PD_P0_IP} ==="
            fi
          fi
          export PD_PREFILL_ALL PD_P0_IP
        |||, '\n'),
      ],
      // Prefill 0 is ready only when every other prefill answers /health too (inside the wait loop's body, indented by two).
      [
        '\n  if p_ok:\n',
        '\n' + std.join('\n', [if l == '' then l else '  ' + l for l in std.split(|||
          p_all = [tuple(x.split('|', 1)) for x in os.environ.get('PD_PREFILL_ALL', '').split(',') if x]
          if p_ok and p_all:
            def _p_up(ip):
              try:
                with urllib.request.urlopen('http://[' + ip + ']:25200/health', timeout=5) as rq:
                  return rq.status == 200
              except Exception:
                return False
            p_up = [h for h, ip in p_all if _p_up(ip)]
            print(f'other prefills healthy: {len(p_up)}/{len(p_all)}', file=sys.stderr)
            p_ok = len(p_up) == len(p_all)
          if p_ok:
        |||, '\n')]),
      ],
      // Prefill i > 0 starts no proxy and no bench: it serves until prefill 0's port closes (as the decode legs do), then exits.
      [
        'export ACTIVE_PREFILL_HOST="${MY_HOST}" ACTIVE_PREFILL_IP="${MY_IP}"',
        'export ACTIVE_PREFILL_HOST="${MY_HOST}" ACTIVE_PREFILL_IP="${MY_IP}"\n' + std.rstripChars(|||
          if [ "${PD_PREFILL_INDEX:-0}" != "0" ]; then
            echo "=== ${EXOS_RECIPE}: prefill ${PD_PREFILL_INDEX} healthy on ${MY_HOST}:25200 after ${EXOS_STARTUP_S}s; prefill 0 (${PD_P0_HOST}) runs the proxy and the bench; serving until it stops ==="
            pd_p0_up() { python3 -c 'import socket,sys; s=socket.socket(socket.AF_INET6); s.settimeout(2); s.connect((sys.argv[1], 25200)); s.close()' "${PD_P0_IP}" >/dev/null 2>&1; }
            for _ in $(seq 1 "$(( ${STARTUP_TIMEOUT:-7200} / 5 ))"); do
              [ -d "/proc/${SERVER_PID}" ] || break
              pd_p0_up && break
              sleep 5
            done
            while [ -d "/proc/${SERVER_PID}" ] && pd_p0_up; do
              sleep 10
            done
            if [ -n "${CDK_OUTPUT_DIR:-}" ] && [ -f /tmp/vllm_serve.log ]; then
              mkdir -p "${CDK_OUTPUT_DIR}" 2>/dev/null || true
              cp -f /tmp/vllm_serve.log "${CDK_OUTPUT_DIR}/prefill.log" 2>/dev/null || true
            fi
            if [ ! -d "/proc/${SERVER_PID}" ] && pd_p0_up; then
              echo "=== PREFILL ${PD_PREFILL_INDEX} VLLM TAIL (/tmp/vllm_serve.log) ==="
              tail -n 80 /tmp/vllm_serve.log 2>/dev/null || true
              exit 1
            fi
            kill "${SERVER_PID}" 2>/dev/null || true
            echo "trace-end-event: name=vllm-prefill ts=$(date +'%s.%N') pid=prefill tid=prefill"
            echo 'EXOS_VLLM_RESULT {"pass": true, "role": "prefill-replica", "worker_id": '"${WORKER_ID}"'}'
            exit 0
          fi
        |||, '\n'),
      ],
      // The proxy (prefill 0) spreads requests round robin over every prefill: itself first, then PD_PREFILL_ALL in job order.
      [
        "prefill_host = _url_host('ACTIVE_PREFILL_HOST')",
        "prefill_host = _url_host('ACTIVE_PREFILL_HOST')\n"
        + "prefill_hosts = [prefill_host] + [('[' + x.split('|', 1)[0] + ']' if ':' in x.split('|', 1)[0] else x.split('|', 1)[0]) for x in os.environ.get('PD_PREFILL_ALL', '').split(',') if x]",
      ],
      [
        "'--prefiller-hosts', prefill_host, '--prefiller-ports', '25200',",
        "'--prefiller-hosts', *prefill_hosts, '--prefiller-ports', *(['25200'] * len(prefill_hosts)),",
      ],
      // A toy proxy without --prefiller-hosts cannot take several prefills: fail closed.
      [
        "'--prefiller-host', prefill_host, '--prefiller-port', '25200',",
        "'--prefiller-host', (prefill_host if len(prefill_hosts) == 1 else sys.exit('FATAL: PREFILL_REPLICAS > 1 needs a toy proxy with --prefiller-hosts')), '--prefiller-port', '25200',",
      ],
    ];
    local pdScriptP =
      if prefillReplicas == 1 then script
      else std.foldl(function(s, p) replaceOnceP(s, p[0], p[1]), multiPrefillPatches, script);
    local pdScript =
      if decodeReplicas == 1 then pdScriptP
      else std.foldl(function(s, p) replaceOnce(s, p[0], p[1]), multiDecodePatches, pdScriptP);

    local pdEnv = {
      EXOS_RECIPE: name,
      MODEL: model,
      SERVED_MODEL_NAME: servedName,
      MAX_MODEL_LEN: maxModelLen,
      MLA_LAYOUT: mlaLayout,
      HOSTS_PER_LEG: hostsPerLeg,
      GLM_RAIDEN_DEFAULTS: glmRaidenDefaults,
      [if prefillKvConfig != '' then 'PREFILL_KV_CONFIG']: prefillKvConfig,
      [if decodeKvConfig != '' then 'DECODE_KV_CONFIG']: decodeKvConfig,
      [if proxyProbe != '' && proxyProbe != '0' then 'PROXY_PROBE']: proxyProbe,
      PREFILL_EXTRA_ARGS: prefillExtraArgs,
      DECODE_EXTRA_ARGS: decodeExtraArgs,
      DEST_DCP_SIZE: destDcpSize,
      DEST_CP_INTERLEAVE: destCpInterleave,
      [if extraEnvPrefill != '' then 'EXTRA_ENV_PREFILL']: extraEnvPrefill,
      [if extraEnvDecode != '' then 'EXTRA_ENV_DECODE']: extraEnvDecode,
      CHIPS_PER_HOST: std.toString(topo.chipsPerHost),
      RAY_LEG_EXPECTED_CHIPS: std.toString(perLegChips),
      RAY_TPU_PER_HOST: std.toString(rayTpuPerHost),
      RAY_LEG_EXPECTED_TPUS: std.toString(rayLegExpectedTpus),
      RAY_TPU_TOPOLOGY: rayTpuTopology,
      RAY_POD_TYPE: rayPodType,
      RAY_ACCEL_TYPE: rayAccelType,
      TPU_ACCEL_ENV: tpuAccelEnv,
      TOTAL_BENCH_CHIPS: std.toString(totalBenchChips),
      BENCH_MODE: benchMode,
      USE_SIDECAR_BENCH: if useSidecar then '1' else '0',
      CONCURRENCY: concurrency,
      ISL: isl,
      OSL: osl,
      RANGE_RATIO: rangeRatio,
      NUM_PROMPTS: numPrompts,
      NUM_WARMUPS: numWarmups,
      FAST: fast,
      DURATION: duration,
      USE_RUNAI_STREAMER: useRunaiStreamer,
      [if loadFormat != '' then 'LOAD_FORMAT']: loadFormat,
      PROMPT: prompt,
      MAX_TOKENS: maxTokens,
      ENABLE_PERFETTO: enablePerfetto,
      STARTUP_TIMEOUT: startupTimeout,
      TORCH_TPU_SLICEBUILDER_PORT_BASE: sliceBuilderPortBase,
      TORCH_TPU_BASE_PORT: sliceBuilderPortBase,
      HF_HUB_OFFLINE: '1',
      TRANSFORMERS_OFFLINE: '1',
      GCLOUD_PROJECT: 'cloud-tpu-inference-test',
      VLLM_LOGGING_LEVEL: 'INFO',
    } + exos.ipv6Env + self.pyPatchEnv + d.EXTRA_ENV;

    cdk.jobSet(
      name=name,
      failurePolicy=cdk.neverRestart,
      network={ enableDNSHostnames: true },
      jobs=if twoSlicePd then
        std.flattenArrays([
          exos.tpuJobs(
            name=prefillJobName(i),
            containerName='vllm-pd',
            image=image,
            script=pdScript,
            platform=platform,
            slice=legSlice,
            runnerGroupId=runnerGroupId,
            expectedPoolHosts=pdExpectedPoolHosts,
            extraEnv=pdEnv { PD_LEG: 'prefill', PREFILL_HOSTNAMES: prefillHostnames, DECODE_HOSTNAMES: decodeHostnames }
                     + (if prefillReplicas > 1 then { PD_PREFILL_INDEX: std.toString(i) } else {}),
            // 2PnD: the sidecar bench rides prefill 0 only.
            headExtraContainers=if useSidecar && i == 0 then [self.sidecarBenchContainer(clientImage, pdEnv)] else [],
          )
          for i in std.range(0, prefillReplicas - 1)
        ]) + std.flattenArrays([
          exos.tpuJobs(
            name=decodeJobName(i),
            containerName='vllm-pd',
            image=image,
            script=pdScript,
            platform=platform,
            slice=legSlice,
            runnerGroupId=decodeRunnerGroupId,
            runnerIndexOffset=decodeRunnerIndexOffset,
            expectedPoolHosts=pdExpectedPoolHosts,
            extraEnv=pdEnv { PD_LEG: 'decode', PREFILL_HOSTNAMES: prefillHostnames, DECODE_HOSTNAMES: decodeHostnames }
                     + (if decodeReplicas > 1 then { PD_DECODE_INDEX: std.toString(i) } else {}),
          )
          for i in std.range(0, decodeReplicas - 1)
        ])
      else
        exos.tpuJobs(
          name='worker',
          containerName='vllm-pd',
          image=image,
          script=script,
          platform=platform,
          slice=slice,
          runnerGroupId=runnerGroupId,
          extraEnv=pdEnv,
          headExtraContainers=if useSidecar then [self.sidecarBenchContainer(clientImage, pdEnv)] else [],
        ),
    ),

  // resolveExosModelUri maps a model identifier (HuggingFace repo ID or GCS/local
  // path) to its GCS URI in gs://cloud-devkit/ckpt/hf/... so Exos Borg linked
  // runners (which have no public internet access) stream weights over IPv6.
  resolveExosModelUri(model, gcsModel='')::
    if gcsModel != '' then gcsModel
    else if std.startsWith(model, 'gs://') || std.startsWith(model, '/') then model
    else 'gs://cloud-devkit/ckpt/hf/' + model,

  // sanitizeServeArgs filters a recipe's `vllm serve` argument list down to
  // workload-level serving flags by substituting shell variable placeholders in
  // `vars` and dropping positional model paths and flags owned by `exos_vllm`
  // (`--served-model-name`, `--host`, `--port`, `--max-model-len`,
  // `--kv-transfer-config` unless `keepKvTransferConfig=true`, `--load-format`
  // except `--load-format=dummy`, and `--tensor-parallel-size` unless `keepTp=true`).
  sanitizeServeArgs(args, vars={}, keepTp=false, keepKvTransferConfig=false)::
    local ownedPrefixes = [
      '--model',
      '--served-model-name',
      '--host',
      '--port',
      '--max-model-len',
    ] + (if keepKvTransferConfig then [] else ['--kv-transfer-config'])
      + (if keepTp then [] else ['--tensor-parallel-size']);
    local isOwnedWithArg(s) = std.any([
      std.startsWith(s, p + '=') || std.startsWith(s, p + ' ')
      for p in ownedPrefixes
    ]);
    local substituteVars(s) = std.foldl(
      function(acc, k) std.strReplace(acc, '${' + k + '}', std.toString(vars[k])),
      std.objectFields(vars),
      s
    );
    local step(acc, raw) =
      if raw == null then acc
      else
        local s = std.stripChars(substituteVars(std.toString(raw)), ' \t\n');
        if s == '' then acc
        else if acc.skipNext then acc { skipNext: false }
        else if acc.checkLoadFormatNext then
          acc {
            checkLoadFormatNext: false,
            out: if s == 'dummy' then acc.out + ['--load-format=dummy'] else acc.out,
          }
        else if s == '--load-format' then acc { checkLoadFormatNext: true }
        else if std.startsWith(s, '--load-format=') || std.startsWith(s, '--load-format ') then
          local val = std.stripChars(s[std.length('--load-format') + 1:], ' \t"\'');
          if val == 'dummy' then acc { out: acc.out + ['--load-format=dummy'] } else acc
        else if std.member(ownedPrefixes, s) then acc { skipNext: true }
        else if isOwnedWithArg(s) then acc
        else if s == 'vllm serve' || std.startsWith(s, 'vllm serve ') then acc
        else if std.startsWith(s, '/') || std.startsWith(s, '$MODEL') || std.startsWith(s, '"$MODEL') || std.startsWith(s, '${MODEL') || std.startsWith(s, '"${MODEL') then acc
        else if s == '"${OFFLOAD_ARGS[@]}"' || s == '${OFFLOAD_ARGS[@]}' then acc
        else acc { out: acc.out + [s] };
    std.foldl(step, args, { skipNext: false, checkLoadFormatNext: false, out: [] }).out,

  // sanitizeServerEnv strips Cloud/GKE pod-identity, checkpoint-staging, and
  // pod-local cache-directory keys from a recipe's container `env` map while
  // preserving all TPU, XLA, vLLM, MoE, MLA, DCP, and MTP tuning variables.
  sanitizeServerEnv(env)::
    local skipKeys = [
      'POD_IP',
      'VLLM_HOST_IP',
      'JOBSET_JOB_INDEX',
      'POD_NAMESPACE',
      'JS_NAME',
      'TPU_MULTIHOST_BACKEND',
      'RAY_ADDRESS',
      'TPU_TOPOLOGY',
      'TPU_CHIPS_PER_HOST_BOUNDS',
      'TPU_HOST_BOUNDS',
      'TPU_PROCESS_BOUNDS',
      'TPU_TOPOLOGY_WRAP',
      'TPU_TOPOLOGY_ALT',
      'TPU_ACCELERATOR_TYPE',
      'CLOUD_TPU_ACCELERATOR_TYPE',
      'TPU_NAME',
      'BARE_METAL_MODE',
      'TPU_SKIP_MDS_QUERY',
      'JAX_PLATFORMS',
      'PJRT_DEVICE',
      'ENABLE_TPUNETD_CLIENT',
      'TPU_LIBRARY_PATH',
      'TPU_PROCESS_ADDRESSES',
      'TPU_PROCESS_PORT',
      'TPU_WORKER_HOSTNAMES',
      'TPU_WORKER_ID',
      'TPU_NODE_ID',
      'CLOUD_TPU_TASK_ID',
      'MODEL',
      'MODEL_PATH',
      'MODEL_NAME',
      'CUSTOM_MODEL_PATH',
      'SERVED_MODEL',
      'SERVED_MODEL_NAME',
      'WEIGHTS_FROM_GCS',
      'GCS_MODEL_SRC',
      '_CDK_GCS_B64',
      'MODEL_SOURCE',
      'SHARED_CACHE_LAYOUT',
      'HF_HOME',
      'COMPILE_CACHE_ENABLED',
      'COMPILE_CACHE_GCS',
      'COMPILE_CACHE_TAG',
      'COMPILE_CACHE_REFRESH',
      'LEGACY_COMPILE_CACHE_DIR',
      'CC_SITECUSTOMIZE_PY',
      'CACHE_DIR',
      'VLLM_CACHE_ROOT',
      'VLLM_CONFIG_ROOT',
      'VLLM_XLA_CACHE_PATH',
      'TRITON_CACHE_DIR',
      'TORCH_TPU_CACHE_DIR',
      'TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE',
      'TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT',
      'TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT',
      'TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK',
      'TORCHINDUCTOR_CACHE_DIR',
      'JAX_COMPILER_CACHE_DIR',
      'JAX_COMPILATION_CACHE_DIR',
      'TMPDIR',
      'TPU_LOG_DIR',
      'XDG_CACHE_HOME',
      'RAY_TMPDIR',
      'MEGASCALE_NUM_SLICES',
      'MEGASCALE_SLICE_ID',
      'MEGASCALE_PORT',
      'MEGASCALE_COORDINATOR_ADDRESS',
      'TPU_RAIDEN_ADVERTISE_HOST',
      'TPU_RAIDEN_JOB_NAME',
      'TPU_RAIDEN_RESHARD_PORT_BASE',
      'TPU_RAIDEN_STORE_DISPATCH_PORT_BASE',
      'TPU_KV_TRANSFER_PORT',
      'TPU_SIDE_CHANNEL_PORT',
    ];
    {
      [k]: std.toString(env[k])
      for k in std.objectFields(env)
      if env[k] != null && !std.isObject(env[k]) && !std.member(skipKeys, k)
    },

  // Base is a late-bound Jsonnet prototype object for standard (single-host or
  // multi-host Ray) Exos vLLM recipes. Standalone smoke recipes extend `Base`
  // with `+`, while dedicated Exos benchmark recipes (`<name>.exos.jsonnet`)
  // call `fromSpec(...)`.
  Base:: {
    local selfObj = self,
    name:: 'exos-vllm',
    platform:: 'ghostfish_sampling',
    chips:: 4,
    slice::
      if self.chips <= 4 then '2x2'
      else if self.chips <= 8 then '2x4'
      else '4x4',
    multihostBackend:: if self.slice == '2x2' then 'uni' else 'ray',
    serverImage:: $.defaults.IMAGE,
    clientImage:: $.defaults.CLIENT_IMAGE,
    modelName:: $.defaults.SERVED_MODEL_NAME,
    servedModelName:: self.modelName,
    gcsModel:: '',
    model:: $.resolveExosModelUri(self.modelName, self.gcsModel),
    tpSize:: if std.startsWith(self.platform, 'ghostfish') then 8 else 4,
    dpSize:: 1,
    maxModelLen:: 2048,
    startupTimeout:: 7200,
    benchMode:: 'smoke',
    concurrency:: 32,
    isl:: 8192,
    osl:: 1024,
    rangeRatio:: '0.8',
    numPrompts:: null,
    numWarmups:: null,
    // Benchmark window. null keeps the Exos defaults (FAST=true, 600 s); Cloud-derived
    // recipes pass their own values so the AgentX replay runs the same window as Cloud.
    fast:: null,
    duration:: null,
    warmupRequestsPerLane:: null,
    useRunaiStreamer:: self.loadFormat == '' || self.loadFormat == 'runai_streamer',
    loadFormat:: '',
    // Workload-level vLLM serving flags.
    serveFlags:: [],
    // Exos-specific serving flags appended automatically by the Exos base layer.
    exosServeFlags::
      (if self.slice != '2x2' || self.dpSize > 1 then ['--data-parallel-address=::1'] else [])
      + ['--no-enable-log-requests', '--disable-uvicorn-access-log'],
    // Workload-level TPU / vLLM / XLA environment variables.
    serverEnv:: {},

    local rendered = $.recipe(selfObj.name, {
      PLATFORM: selfObj.platform,
      SLICE: selfObj.slice,
      IMAGE: selfObj.serverImage,
      CLIENT_IMAGE: selfObj.clientImage,
      MODEL: selfObj.model,
      SERVED_MODEL_NAME: selfObj.servedModelName,
      MULTIHOST_BACKEND: selfObj.multihostBackend,
      TP: std.toString(selfObj.tpSize),
      MAX_MODEL_LEN: std.toString(selfObj.maxModelLen),
      [if selfObj.startupTimeout != 7200 then 'STARTUP_TIMEOUT']: std.toString(selfObj.startupTimeout),
      BENCH_MODE: selfObj.benchMode,
      CONCURRENCY: std.toString(selfObj.concurrency),
      ISL: std.toString(selfObj.isl),
      OSL: std.toString(selfObj.osl),
      RANGE_RATIO: std.toString(selfObj.rangeRatio),
      [if selfObj.numPrompts != null then 'NUM_PROMPTS']: std.toString(selfObj.numPrompts),
      [if selfObj.numWarmups != null then 'NUM_WARMUPS']: std.toString(selfObj.numWarmups),
      [if selfObj.fast != null then 'FAST']: if std.isBoolean(selfObj.fast) then (if selfObj.fast then 'true' else 'false') else std.toString(selfObj.fast),
      [if selfObj.duration != null then 'DURATION']: std.toString(selfObj.duration),
      USE_RUNAI_STREAMER: if std.isBoolean(selfObj.useRunaiStreamer) then (if selfObj.useRunaiStreamer then 'true' else 'false') else std.toString(selfObj.useRunaiStreamer),
      [if selfObj.loadFormat != '' then 'LOAD_FORMAT']: selfObj.loadFormat,
      VLLM_EXTRA_ARGS: std.join(' ', [f for f in selfObj.serveFlags + selfObj.exosServeFlags if f != null && f != '']),
      EXTRA_ENV: (
        if selfObj.multihostBackend == 'ray' && selfObj.dpSize > 1 then {
          PYTHONHASHSEED: '0',
          VLLM_DP_MASTER_IP: '::1',
          VLLM_DP_PORT: '29550',
        } else {}
      ) + (
        if selfObj.warmupRequestsPerLane != null then
          { WARMUP_REQUESTS_PER_LANE: std.toString(selfObj.warmupRequestsPerLane) }
        else {}
      ) + selfObj.serverEnv,
    }),

    apiVersion: rendered.apiVersion,
    kind: rendered.kind,
    metadata: rendered.metadata,
    spec: rendered.spec,
  },

  // fromSpec constructs a standard Exos vLLM JobSet from a recipe's local
  // bindings (accepting plain `{ key: val }` fields without requiring `::`).
  fromSpec(spec)::
    local has(k) = std.objectHasAll(spec, k);
    $.Base + {
      [if has('name') then 'name']:: spec.name,
      [if has('platform') then 'platform']:: spec.platform,
      [if has('chips') then 'chips']:: spec.chips,
      [if has('slice') then 'slice']:: spec.slice,
      [if has('multihostBackend') then 'multihostBackend']:: spec.multihostBackend,
      [if has('serverImage') then 'serverImage']:: spec.serverImage,
      [if has('clientImage') then 'clientImage']:: spec.clientImage,
      [if has('modelName') then 'modelName']:: spec.modelName,
      [if has('servedModelName') then 'servedModelName']:: spec.servedModelName,
      [if has('gcsModel') then 'gcsModel']:: spec.gcsModel,
      [if has('model') then 'model']:: spec.model,
      [if has('tpSize') then 'tpSize']:: spec.tpSize,
      [if has('dpSize') then 'dpSize']:: spec.dpSize,
      [if has('maxModelLen') then 'maxModelLen']:: spec.maxModelLen,
      [if has('startupTimeout') then 'startupTimeout']:: spec.startupTimeout,
      [if has('benchMode') then 'benchMode']:: spec.benchMode,
      [if has('concurrency') then 'concurrency']:: spec.concurrency,
      [if has('isl') then 'isl']:: spec.isl,
      [if has('osl') then 'osl']:: spec.osl,
      [if has('rangeRatio') then 'rangeRatio']:: spec.rangeRatio,
      [if has('numPrompts') then 'numPrompts']:: spec.numPrompts,
      [if has('numWarmups') then 'numWarmups']:: spec.numWarmups,
      [if has('fast') then 'fast']:: spec.fast,
      [if has('duration') then 'duration']:: spec.duration,
      [if has('warmupRequestsPerLane') then 'warmupRequestsPerLane']:: spec.warmupRequestsPerLane,
      [if has('useRunaiStreamer') then 'useRunaiStreamer']:: spec.useRunaiStreamer,
      [if has('loadFormat') then 'loadFormat']:: spec.loadFormat,
      [if has('serveFlags') then 'serveFlags']:: spec.serveFlags,
      [if has('exosServeFlags') then 'exosServeFlags']:: spec.exosServeFlags,
      [if has('serverEnv') then 'serverEnv']:: spec.serverEnv,
    },

  // PdBase is a late-bound Jsonnet prototype object for 1P1D (or, with decodeReplicas, 1PnD) Prefill/Decode
  // disaggregated Exos vLLM recipes. Dedicated Exos disaggregated recipes
  // (`<name>.exos.jsonnet`) call `fromPdSpec(...)`.
  PdBase:: {
    local selfObj = self,
    name:: 'glm52-fp8-pd-agentx',
    platform:: $.pdDefaults.PLATFORM,
    slice:: $.pdDefaults.SLICE,
    hostsPerLeg:: 2,
    // 1PnD: n one-host decode slices (needs hostsPerLeg 1).
    decodeReplicas:: 1,
    // 2PnD (q38 TUNING-KIT5): n one-host prefill slices (needs hostsPerLeg 1); prefill 0 runs the proxy and the bench.
    prefillReplicas:: 1,
    glmRaidenDefaults:: true,
    prefillKvConfig:: '',
    decodeKvConfig:: '',
    proxyProbe:: false,
    startupTimeout:: 7200,
    serverImage:: $.pdDefaults.IMAGE,
    clientImage:: $.pdDefaults.CLIENT_IMAGE,
    modelName:: $.pdDefaults.SERVED_MODEL_NAME,
    servedModelName:: self.modelName,
    gcsModel:: '',
    model:: $.resolveExosModelUri(self.modelName, self.gcsModel),
    maxModelLen:: 1048576,
    mlaLayout:: $.pdDefaults.MLA_LAYOUT,
    benchMode:: 'agentx',
    concurrency:: 32,
    isl:: 8192,
    osl:: 1024,
    rangeRatio:: '0.8',
    numPrompts:: null,
    numWarmups:: null,
    fast:: null,
    duration:: null,
    warmupRequestsPerLane:: null,
    useRunaiStreamer:: self.loadFormat == '' || self.loadFormat == 'runai_streamer',
    loadFormat:: '',
    prefillArgs:: [],
    decodeArgs:: [],
    prefillEnv:: {},
    decodeEnv:: {},
    destDcpSize:: 16,
    destCpInterleave:: 1024,
    extraEnvPrefill:: '',
    extraEnvDecode:: '',
    serverEnv:: {},

    local rendered = $.pdRecipe(selfObj.name, {
      PLATFORM: selfObj.platform,
      SLICE: selfObj.slice,
      HOSTS_PER_LEG: std.toString(selfObj.hostsPerLeg),
      DECODE_REPLICAS: std.toString(selfObj.decodeReplicas),
      PREFILL_REPLICAS: std.toString(selfObj.prefillReplicas),
      GLM_RAIDEN_DEFAULTS: if selfObj.glmRaidenDefaults then '1' else '0',
      PREFILL_KV_CONFIG: selfObj.prefillKvConfig,
      DECODE_KV_CONFIG: selfObj.decodeKvConfig,
      PROXY_PROBE: if selfObj.proxyProbe then '1' else '0',
      STARTUP_TIMEOUT: std.toString(selfObj.startupTimeout),
      IMAGE: selfObj.serverImage,
      CLIENT_IMAGE: selfObj.clientImage,
      MODEL: selfObj.model,
      SERVED_MODEL_NAME: selfObj.servedModelName,
      MAX_MODEL_LEN: std.toString(selfObj.maxModelLen),
      MLA_LAYOUT: selfObj.mlaLayout,
      BENCH_MODE: selfObj.benchMode,
      CONCURRENCY: std.toString(selfObj.concurrency),
      ISL: std.toString(selfObj.isl),
      OSL: std.toString(selfObj.osl),
      RANGE_RATIO: std.toString(selfObj.rangeRatio),
      [if selfObj.numPrompts != null then 'NUM_PROMPTS']: std.toString(selfObj.numPrompts),
      [if selfObj.numWarmups != null then 'NUM_WARMUPS']: std.toString(selfObj.numWarmups),
      [if selfObj.fast != null then 'FAST']: if std.isBoolean(selfObj.fast) then (if selfObj.fast then 'true' else 'false') else std.toString(selfObj.fast),
      [if selfObj.duration != null then 'DURATION']: std.toString(selfObj.duration),
      USE_RUNAI_STREAMER: if std.isBoolean(selfObj.useRunaiStreamer) then (if selfObj.useRunaiStreamer then 'true' else 'false') else std.toString(selfObj.useRunaiStreamer),
      [if selfObj.loadFormat != '' then 'LOAD_FORMAT']: selfObj.loadFormat,
      PREFILL_EXTRA_ARGS: std.join(' ', [f for f in selfObj.prefillArgs if f != null && f != '']),
      DECODE_EXTRA_ARGS: std.join(' ', [f for f in selfObj.decodeArgs if f != null && f != '']),
      PREFILL_ENV: selfObj.prefillEnv,
      DECODE_ENV: selfObj.decodeEnv,
      DEST_DCP_SIZE: std.toString(selfObj.destDcpSize),
      DEST_CP_INTERLEAVE: std.toString(selfObj.destCpInterleave),
      EXTRA_ENV_PREFILL: selfObj.extraEnvPrefill,
      EXTRA_ENV_DECODE: selfObj.extraEnvDecode,
      EXTRA_ENV: (
        if selfObj.warmupRequestsPerLane != null then
          { WARMUP_REQUESTS_PER_LANE: std.toString(selfObj.warmupRequestsPerLane) }
        else {}
      ) + selfObj.serverEnv,
    }),

    apiVersion: rendered.apiVersion,
    kind: rendered.kind,
    metadata: rendered.metadata,
    spec: rendered.spec,
  },

  // fromPdSpec constructs a 1P1D (or 1PnD) disaggregated Exos vLLM JobSet from a recipe's
  // local bindings (accepting plain `{ key: val }` fields without requiring `::`).
  fromPdSpec(spec)::
    local has(k) = std.objectHasAll(spec, k);
    $.PdBase + {
      [if has('name') then 'name']:: spec.name,
      [if has('platform') then 'platform']:: spec.platform,
      [if has('slice') then 'slice']:: spec.slice,
      [if has('hostsPerLeg') then 'hostsPerLeg']:: spec.hostsPerLeg,
      [if has('decodeReplicas') then 'decodeReplicas']:: spec.decodeReplicas,
      [if has('prefillReplicas') then 'prefillReplicas']:: spec.prefillReplicas,
      [if has('glmRaidenDefaults') then 'glmRaidenDefaults']:: spec.glmRaidenDefaults,
      [if has('prefillKvConfig') then 'prefillKvConfig']:: spec.prefillKvConfig,
      [if has('decodeKvConfig') then 'decodeKvConfig']:: spec.decodeKvConfig,
      [if has('proxyProbe') then 'proxyProbe']:: spec.proxyProbe,
      [if has('startupTimeout') then 'startupTimeout']:: spec.startupTimeout,
      [if has('serverImage') then 'serverImage']:: spec.serverImage,
      [if has('clientImage') then 'clientImage']:: spec.clientImage,
      [if has('modelName') then 'modelName']:: spec.modelName,
      [if has('servedModelName') then 'servedModelName']:: spec.servedModelName,
      [if has('gcsModel') then 'gcsModel']:: spec.gcsModel,
      [if has('model') then 'model']:: spec.model,
      [if has('maxModelLen') then 'maxModelLen']:: spec.maxModelLen,
      [if has('mlaLayout') then 'mlaLayout']:: spec.mlaLayout,
      [if has('benchMode') then 'benchMode']:: spec.benchMode,
      [if has('concurrency') then 'concurrency']:: spec.concurrency,
      [if has('isl') then 'isl']:: spec.isl,
      [if has('osl') then 'osl']:: spec.osl,
      [if has('rangeRatio') then 'rangeRatio']:: spec.rangeRatio,
      [if has('numPrompts') then 'numPrompts']:: spec.numPrompts,
      [if has('numWarmups') then 'numWarmups']:: spec.numWarmups,
      [if has('fast') then 'fast']:: spec.fast,
      [if has('duration') then 'duration']:: spec.duration,
      [if has('warmupRequestsPerLane') then 'warmupRequestsPerLane']:: spec.warmupRequestsPerLane,
      [if has('useRunaiStreamer') then 'useRunaiStreamer']:: spec.useRunaiStreamer,
      [if has('loadFormat') then 'loadFormat']:: spec.loadFormat,
      [if has('prefillArgs') then 'prefillArgs']:: spec.prefillArgs,
      [if has('decodeArgs') then 'decodeArgs']:: spec.decodeArgs,
      [if has('prefillEnv') then 'prefillEnv']:: spec.prefillEnv,
      [if has('decodeEnv') then 'decodeEnv']:: spec.decodeEnv,
      [if has('destDcpSize') then 'destDcpSize']:: spec.destDcpSize,
      [if has('destCpInterleave') then 'destCpInterleave']:: spec.destCpInterleave,
      [if has('extraEnvPrefill') then 'extraEnvPrefill']:: spec.extraEnvPrefill,
      [if has('extraEnvDecode') then 'extraEnvDecode']:: spec.extraEnvDecode,
      [if has('serverEnv') then 'serverEnv']:: spec.serverEnv,
    },
}
