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
"""Pull a Run:AI object-storage model once per host, not once per API server.

``ModelConfig.__post_init__`` calls
``maybe_pull_model_tokenizer_for_runai``, which mirrors a
``gs://``/``s3://``/``az://`` model's non-weight files into
``$VLLM_ASSETS_CACHE/model_streamer/<sha256(uri)[:8]>`` and rewrites
``model``/``tokenizer`` to that dir. It pulls TWICE per process: once for
``*.model``/``*.py``/``*.json``, then again for everything but the weights
when ``model == tokenizer`` (the usual case, since ``--tokenizer`` is not
passed).

That dir name is a pure function of the URI, so every process in the pod
picks the same one, and ``ObjectStorageModel.pull_files`` has no lock, no
skip-if-present and no temp+rename -- it downloads straight to the final
path. With ``--api-server-count 8`` that is sixteen concurrent rewrites of
the same ~11 MB ``tokenizer.json`` while sibling API servers are reading
it, so a reader eventually observes a half-written file::

    json.decoder.JSONDecodeError: Unterminated string starting at:
        line 449789 column 7 (char 9852170)
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in
        position 3661823: unexpected end of data

which kills that API server and takes the engine down with::

    RuntimeError: API server ApiServer_6 closed its address pipe without
        reporting its bound ZMQ addresses

The offsets differ per occurrence (char 0, 3.1 MB, 3.7 MB, 9.9 MB), which
is the signature of a torn read rather than a corrupt source object.

``worker/runai_repull.py`` already solves the same problem one level down,
for worker ranks on hosts the API server never populated, and its module
doc reasons about exactly this hazard ("upstream downloads in place (no
temp+rename) ... a sibling rank could otherwise observe a half-populated
dir"). Its head-host fast path assumes the API server's pull "finished
before any worker was spawned" -- true of workers, but there is nothing
that orders eight API servers against each other. This module closes that
remaining gap, and deliberately reuses ``runai_repull``'s completion
sentinel so the two cannot disagree about whether a dir is populated.

The fix is a lock AND a sentinel, because neither suffices alone:

* A sentinel alone loses a cold start -- all eight processes find no
  sentinel and race to create it.
* A lock alone does not separate writers from readers. The first process
  releases the lock and starts reading ``tokenizer.json`` while the second
  acquires it and rewrites the same file underneath it.

Together, exactly one process ever writes: later arrivals block until that
write is complete, then see the sentinel and skip the network entirely.
The original method still runs in every process, so the ``model`` /
``model_weights`` / ``tokenizer`` field rewrites are upstream's own and
cannot drift -- only the download is suppressed.

Why ``register_tpu_platform`` + ``_ACTIVE`` and not ``import``: the pull
happens in ``ModelConfig.__post_init__``, reached from
``create_model_config()``, while plugins are registered earlier via
``register_tpu_platform()``. Calling ``patch_runai_pull_once()`` from
``register_tpu_platform()`` guards API servers before ``create_model_config()``
without importing ``vllm`` during ``import vllm_torchtpu``, while registering
under ``_ACTIVE`` ensures spawned workers also apply it.

Inert unless the model or tokenizer is an object-storage URI, so local-disk
and HF-hub models are untouched.
"""

import functools
import os
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _sentinel_path(local_dir: str) -> str:
    from vllm_torchtpu.worker.runai_repull import COMPLETE_SENTINEL

    return os.path.join(local_dir, COMPLETE_SENTINEL)


def _mark_complete(local_dir: str, uri: str) -> None:
    """Record that ``local_dir`` is fully populated for ``uri``."""
    from vllm.model_executor.model_loader.weight_utils import atomic_writer

    path = _sentinel_path(local_dir)
    if os.path.exists(path):
        return
    try:
        with atomic_writer(path) as fh:
            fh.write(uri)
    except OSError:
        # A missing sentinel only costs a redundant pull under the lock,
        # which is still correct; never fail startup over it.
        logger.warning("[runai-pull-once] could not write %s", path)


def _install_model_config_patch() -> None:
    """Install the ModelConfig.maybe_pull_model_tokenizer_for_runai wrapper."""
    from vllm.config.model import ModelConfig
    from vllm.transformers_utils.runai_utils import is_runai_obj_uri

    if getattr(ModelConfig, "_tpu_runai_pull_once", False):
        return

    original_maybe_pull = ModelConfig.maybe_pull_model_tokenizer_for_runai

    def maybe_pull_model_tokenizer_for_runai(self, model, tokenizer) -> None:
        if isinstance(model, str) and is_runai_obj_uri(model):
            uri = model
        elif isinstance(tokenizer, str) and is_runai_obj_uri(tokenizer):
            uri = tokenizer
        else:
            return original_maybe_pull(self, model, tokenizer)

        from vllm_torchtpu.worker.runai_repull import acquire_runai_lock

        # Keyed on the model URI, matching worker/runai_repull. A tokenizer
        # served from a DIFFERENT URI lands in its own dir and is covered by
        # the same lock only because both pulls happen inside this call.
        # Uses acquire_runai_lock so nested calls from ensure_runai_aux_files
        # re-enter cleanly without deadlocking on a second FileLock fd.
        with acquire_runai_lock(uri):
            original_maybe_pull(self, model, tokenizer)
            # Upstream has now rewritten these to local dirs.
            for local_dir in {self.model, self.tokenizer}:
                if isinstance(local_dir, str) and os.path.isdir(local_dir):
                    _mark_complete(local_dir, uri)

    ModelConfig.maybe_pull_model_tokenizer_for_runai = (
        maybe_pull_model_tokenizer_for_runai
    )
    ModelConfig._tpu_runai_pull_once = True


def patch_runai_pull_once() -> None:
    """Serialize and de-duplicate the Run:AI aux-file pull across processes.

    No-op for models that do not come from object storage.

    IMPORTANT: Do NOT import ``vllm.config.model`` directly here. This function
    can run during ``register_tpu_platform()`` when ``vllm_torchtpu`` is loaded
    as a platform plugin mid-import of ``vllm.utils.torch_utils``. Importing
    ``vllm.config.model`` -> ``vllm.config.cache`` ->
    ``vllm.utils.torch_utils`` at that point causes a circular ImportError on
    ``is_quantized_kv_cache``. Instead, we patch ``ObjectStorageModel``
    immediately and hook ``maybe_model_redirect`` (called at the start of
    ``ModelConfig.__post_init__`` before
    ``maybe_pull_model_tokenizer_for_runai`` runs) to lazily install the
    ``ModelConfig`` wrapper before the first pull.
    """
    import sys

    try:
        import vllm.transformers_utils.utils as trans_utils
        from vllm.transformers_utils.runai_utils import ObjectStorageModel
    except ImportError:
        return

    if getattr(ObjectStorageModel, "_tpu_runai_pull_once", False):
        return

    original_pull_files = ObjectStorageModel.pull_files
    original_maybe_redirect = trans_utils.maybe_model_redirect

    def pull_files(self, model_path="", allow_pattern=None, ignore_pattern=None):
        # Runs under the lock taken in maybe_pull below, so the sentinel
        # cannot appear between this check and the download.
        if os.path.exists(_sentinel_path(self.dir)):
            return None
        return original_pull_files(self, model_path, allow_pattern, ignore_pattern)

    @functools.wraps(original_maybe_redirect)
    def _lazy_maybe_model_redirect(*args, **kwargs):
        _install_model_config_patch()
        return original_maybe_redirect(*args, **kwargs)

    ObjectStorageModel.pull_files = pull_files
    ObjectStorageModel._tpu_runai_pull_once = True
    trans_utils.maybe_model_redirect = _lazy_maybe_model_redirect

    model_mod: Any = sys.modules.get("vllm.config.model")
    if model_mod is not None:
        if hasattr(model_mod, "maybe_model_redirect"):
            model_mod.maybe_model_redirect = _lazy_maybe_model_redirect
        if hasattr(model_mod, "ModelConfig"):
            _install_model_config_patch()

    logger.info(
        "Applied TPU patch: Run:AI aux files are pulled once per "
        "host under a lock (API servers no longer race to rewrite "
        "the shared tokenizer/config dir)."
    )
