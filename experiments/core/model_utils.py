import os

from huggingface_hub import snapshot_download, try_to_load_from_cache


def resolve_model_path(model: str, base_dir: str = "") -> str:
    """Resolves a model repo ID or local path to an absolute directory on disk.

    1. If `model` exists locally (absolute or relative to `base_dir`), returns its absolute path.
    2. If already present in the Hugging Face cache, returns the cached snapshot directory.
    3. If an uncached Hugging Face repo ID, downloads the full snapshot and returns the directory.
    4. Falls back to returning the resolved local path if not a valid HF repo ID or offline.
    """
    if not model:
        return ""

    cand = os.path.join(
        base_dir, model) if base_dir and not os.path.isabs(model) else model
    if os.path.exists(cand):
        return os.path.abspath(cand)

    try:
        cfg = try_to_load_from_cache(model, "config.json")
        if cfg and isinstance(cfg, str):
            return os.path.dirname(cfg)
        print(
            f">>> [MODEL LOADER] Model '{model}' not found in local cache. Downloading snapshot from Hugging Face..."
        )
        downloaded = snapshot_download(repo_id=model)
        if downloaded and os.path.exists(downloaded):
            return os.path.abspath(downloaded)
    except Exception:
        pass

    return os.path.abspath(cand)
