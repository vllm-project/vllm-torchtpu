# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Re-stage a ``gs://`` model's metadata on every host of a multi-host slice.

``ModelConfig.__post_init__`` calls ``maybe_pull_model_tokenizer_for_runai``,
which pulls a Run:AI URI's ``*.json`` / ``*.model`` / ``*.py`` into
``$VLLM_ASSETS_CACHE/model_streamer/<sha256(uri)[:8]>``, records the URI in
``model_weights``, and rewrites ``model`` to point at that local directory.

``__post_init__`` runs exactly once, in the process that builds the config.
Workers receive the config by cloudpickle, which does not re-run it, so on
every host except the one that built it ``model_config.model`` names a
directory that was never created. Weights are unaffected --
``RunaiModelStreamerLoader.load_weights`` prefers ``model_weights`` and streams
from object storage on every rank -- but the first reader of
``model_config.model`` fails.

On TPU that reader is ``get_quant_config``. The plugin routes every model
through it (``layers/vllm/quantization/__init__.py``), unquantized ones
included, and for a bf16 checkpoint it falls through to
``os.path.isdir(model)`` -> False -> ``snapshot_download()`` on a local path ->
``HFValidationError``. A quantized checkpoint returns early from
``hf_config.quantization_config`` and never touches the filesystem, which is
why the FP8 perf suites pass over this same code path while an unquantized
model of any size does not.

The directory name is a pure function of the URI and the URI survives pickling
in ``model_weights``, so each host can rebuild the directory for itself.
"""

import os

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Generous: the loser of the race waits for the winner's pull, which is a
# handful of JSON files but goes over the network from object storage.
_PULL_TIMEOUT_S = 600


def _is_staged(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def ensure_runai_assets_local(model_config) -> None:
    """Stage the metadata for ``model_config`` if this host is missing it.

    Safe to call from every rank. The pull is serialized by a lock file beside
    the target directory, and the check for "already staged" happens inside
    the lock: ranks sharing a host would otherwise race, and a loser can read a
    half-written ``config.json``. That is the failure documented in
    ``scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh``, where the DP
    API frontends raced on this same directory and are pinned to
    ``--api-server-count=1`` to avoid it.
    """
    # Imported lazily: this runs inside worker processes, where the vLLM import
    # graph is already warm but module import order is not ours to assume.
    import filelock
    from vllm.transformers_utils.runai_utils import (ObjectStorageModel,
                                                     is_runai_obj_uri)

    model_weights = getattr(model_config, "model_weights", None)
    if not model_weights or not is_runai_obj_uri(model_weights):
        return

    target = model_config.model
    if is_runai_obj_uri(target):
        # `model` was never rewritten, so nothing was staged to rebuild.
        return

    os.makedirs(target, exist_ok=True)
    lock_path = f"{target.rstrip(os.sep)}.lock"
    with filelock.FileLock(lock_path, timeout=_PULL_TIMEOUT_S):
        if _is_staged(target):
            return

        obj = ObjectStorageModel(url=model_weights)
        if os.path.realpath(obj.dir) != os.path.realpath(target):
            raise RuntimeError(
                f"Run:AI staging directory differs between hosts: this host "
                f"resolves {model_weights!r} to {obj.dir!r}, but the config "
                f"was built against {target!r}. VLLM_ASSETS_CACHE must be "
                f"identical on every host of the slice.")

        logger.info("Staging Run:AI model metadata for this host: %s -> %s",
                    model_weights, target)
        obj.pull_files(model_weights,
                       allow_pattern=["*.model", "*.py", "*.json"])

        # Mirror maybe_pull_model_tokenizer_for_runai: when the tokenizer was
        # served out of the same directory it got a second, wider pull (all
        # non-weight files), which picks up things the allow_pattern above
        # misses -- merges.txt being the common one.
        if getattr(model_config, "tokenizer", None) == target:
            obj.pull_files(model_weights,
                           ignore_pattern=[
                               "*.pt", "*.safetensors", "*.bin", "*.tensors",
                               "*.pth"
                           ])

        if not _is_staged(target):
            raise RuntimeError(
                f"Pulled {model_weights!r} into {target!r} but no config.json "
                f"landed there. Check that the URI names the snapshot "
                f"directory holding the checkpoint's config, not a parent "
                f"prefix or the weights alone.")
