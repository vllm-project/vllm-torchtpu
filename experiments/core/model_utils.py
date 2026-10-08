import os
import re

from huggingface_hub import snapshot_download, try_to_load_from_cache

# A bare Hugging Face Hub repo id: exactly one "/" separating two
# path-safe components (e.g. "google/gemma-4-31B-it"). Anything else
# (e.g. "model_A", "/abs/path", "a/b/c") is treated as a local path.
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def looks_like_hf_repo_id(model: str) -> bool:
    """Returns True if `model` is shaped like a Hub repo id ("org/name")."""
    return bool(_HF_REPO_ID_RE.match(model))


def resolve_model_path(model: str, base_dir: str = "") -> str:
    """Resolves a model repo ID or local path to an absolute directory on disk.

    1. If `model` exists locally (absolute or relative to `base_dir`),
       returns its absolute path.
    2. If `model` does not look like a Hub repo id ("org/name"), returns
       the resolved local path unchanged without touching the Hub.
    3. If already present in the Hugging Face cache, returns the cached
       snapshot directory.
    4. Otherwise downloads the full snapshot and returns the directory.

    Raises:
        FileNotFoundError: if `model` looks like a repo id but is neither
            cached nor downloadable (gated, missing, offline, timeout).
    """
    if not model:
        return ""

    cand = os.path.join(
        base_dir, model) if base_dir and not os.path.isabs(model) else model
    if os.path.exists(cand):
        return os.path.abspath(cand)

    if not looks_like_hf_repo_id(model):
        return os.path.abspath(cand)

    cfg = try_to_load_from_cache(model, "config.json")
    if cfg and isinstance(cfg, str):
        return os.path.dirname(cfg)

    print(f">>> [MODEL LOADER] Model '{model}' not found in local cache. "
          "Downloading snapshot from Hugging Face...")
    try:
        downloaded = snapshot_download(repo_id=model)
    except Exception as e:  # noqa: BLE001 - re-raised with context below
        raise FileNotFoundError(
            f"Model '{model}' is neither a local path nor a downloadable "
            f"Hugging Face repo ({type(e).__name__}: {e})") from e
    if not downloaded or not os.path.exists(downloaded):
        raise FileNotFoundError(
            f"snapshot_download for '{model}' returned no local directory "
            f"(got {downloaded!r})")
    return os.path.abspath(downloaded)
