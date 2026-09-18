"""env.py.

Centralized paths and Cloud TPU environment setup.
"""

import getpass
import os
import subprocess
import sys
from pathlib import Path

# 1. Single root location for all caches and checkpoints (zero if/else branches)
ROOT_DIR = Path(os.environ.get("CACHE_DIR", "/mnt/pd"))

# 2. All caches, checkpoints, and repositories hang off ROOT_DIR
HF_HOME = str(ROOT_DIR / "huggingface_cache")
SPARSE_CHECKPOINTS_DIR = str(ROOT_DIR / "sparse_checkpoints")
TORCHINDUCTOR_CACHE_DIR = str(ROOT_DIR / "torch_compile_cache")

VLLM_REPO_PATH = str(ROOT_DIR / "vllm")
VLLM_TORCHTPU_SRC = str(ROOT_DIR / "vllm-torchtpu/src")


def resolve_user_ldap() -> str:
    """Resolves LDAP username from environment, git email, or OS Login."""
    for ev in ["USER_LDAP", "MARC_USER_LDAP", "GOOGLE_LDAP"]:
        if val := os.environ.get(ev, "").strip():
            return val
    try:
        email = subprocess.check_output(["git", "config", "user.email"],
                                        text=True,
                                        timeout=2).strip()
        if "@google.com" in email:
            return email.split("@")[0]
    except Exception:
        pass
    raw = os.environ.get("USER") or os.environ.get(
        "LOGNAME") or getpass.getuser() or ""
    return raw[:-len("_google_com")] if raw.endswith("_google_com") else (
        raw if raw and raw.lower() != "root" else "unknown")


USER_LDAP = resolve_user_ldap()
if USER_LDAP and USER_LDAP != "unknown":
    os.environ.setdefault("USER_LDAP", USER_LDAP)


# 3. Setup function for TPU process
def setup_tpu_env() -> None:
    """Configures TPU execution flags, cache directories, and module paths."""
    os.environ.setdefault("VLLM_TARGET_DEVICE", "tpu")
    os.environ.setdefault("VLLM_PLATFORM", "tpu")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.setdefault("USE_BATCHED_RPA_KERNEL", "1")

    if USER_LDAP and USER_LDAP != "unknown":
        os.environ.setdefault("USER_LDAP", USER_LDAP)

    os.environ["HF_HOME"] = HF_HOME
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = TORCHINDUCTOR_CACHE_DIR

    current_pp = os.environ.get("PYTHONPATH", "")
    pp_entries = [p for p in current_pp.split(":") if p]
    for p in [VLLM_TORCHTPU_SRC, VLLM_REPO_PATH]:
        if p not in sys.path:
            sys.path.insert(0, p)
        if p not in pp_entries:
            pp_entries.insert(0, p)
    os.environ["PYTHONPATH"] = ":".join(pp_entries)
