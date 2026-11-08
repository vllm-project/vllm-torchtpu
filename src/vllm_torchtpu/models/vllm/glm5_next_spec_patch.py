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
"""Teach an *unpatched* vLLM that ``glm5_next`` has an MTP draft head.

This is a runtime stand-in for a two-hunk change to
``vllm/config/speculative.py`` (add ``"glm5_next_mtp"`` to ``MTPModelTypes``,
add a ``Glm5Next*`` branch to ``SpeculativeConfig.hf_config_override``). The
patch belongs upstream; it lives here because the TPU image ships its own vLLM
under ``site-packages`` and this repo is NFS-mounted beside it, so a source
edit to a local vLLM checkout never reaches the pod.

Two hazards this module exists to handle, both of which bite silently:

* **The Literal is read at call time, not import time.**
  ``SpeculativeConfig.__post_init__`` decides ``method = "mtp"`` via
  ``draft_model_config.hf_config.model_type in get_args(MTPModelTypes)``,
  resolving ``MTPModelTypes`` through the module global. Rebinding that global
  therefore works -- but only if it happens before the config is built, and
  the failure if it does not is ``NotImplementedError: Unsupported speculative
  method: 'mtp'``, which names neither GLM nor the Literal.

* **The override callable must be picklable by reference.**
  ``SpeculativeConfig.compose_draft_hf_overrides`` hands
  ``SpeculativeConfig.hf_config_override`` to the draft ``ModelConfig`` as its
  ``hf_overrides``, and that config is pickled into the spawned EngineCore
  process. A closure or a ``__main__``-level function fails there with "Can't
  get local object" / a missing module. So the replacement is a plain
  module-level function in an installed package, and importing this module --
  which unpickling it does automatically -- re-applies the Literal patch in
  the child. That self-bootstrap is deliberate.

**This module is deliberately outside the ``glm5_next`` package.**
``register_models`` imports it on *every* model's boot path, and Python runs a
parent package before any submodule -- so spelling it ``glm5_next.spec_patch``
would execute ``glm5_next/__init__.py``, pulling eleven GLM modules and the
DeepSeek-V4 kernels into every Qwen and DeepSeek-V4 process and turning an
import error anywhere in GLM's tree into their boot failure.
``runner/tpu_runner.py`` refuses to import the package from its per-layer
cache predicate for the same reason. Nothing here imports from ``glm5_next``,
and ``tests/models/vllm/test_glm5_next_boot_isolation.py`` fails the build if
that stops being true.

:func:`install` is idempotent and safe to call from more than one process.
``models/vllm/__init__.register_models`` calls it, which is early enough:
``SpeculativeConfig`` is built after platform initialisation. Importing this
module also calls it, which is what covers the spawned child.
"""

from typing import Literal, get_args

from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.config import get_hf_text_config

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

#: ``model_type`` this patch stamps onto a GLM-5.3-Flash draft config. Matches
#: the name the upstream patch adds to ``MTPModelTypes``, so a vLLM that
#: already carries the change and this module agree rather than collide.
MTP_MODEL_TYPE = "glm5_next_mtp"

#: Target architectures that own a nextn layer. The real checkpoint declares
#: the multimodal name; the text-only class is here because a probe may select
#: it with ``hf_overrides={"architectures": [...]}`` -- though note vLLM keeps
#: *dict* overrides target-only, so in practice the draft sees the checkpoint's
#: own architecture string and the first entry is the one that fires.
_GLM5_NEXT_ARCHITECTURES = (
    "Glm5NextForConditionalGeneration",
    "Glm5NextForCausalLM",
)

_ORIGINAL_HF_CONFIG_OVERRIDE = None


def glm5_next_hf_config_override(hf_config):
    """``SpeculativeConfig.hf_config_override`` with a ``glm5_next`` branch.

    Module-level (not a closure) so that the draft ``ModelConfig`` carrying it
    as ``hf_overrides`` pickles into the EngineCore process; see the module
    docstring.
    """
    architectures = getattr(hf_config, "architectures", None) or []
    if architectures and architectures[0] in _GLM5_NEXT_ARCHITECTURES:
        # `num_nextn_predict_layers` lives in `text_config`: GLM-5.3-Flash's
        # outer config is multimodal, and `getattr(hf_config, ...)` on it
        # returns None -- which would then be written to `n_predict` and read
        # back as "no draft layers" much later, far from here.
        text_config = get_hf_text_config(hf_config)
        n_predict = getattr(text_config, "num_nextn_predict_layers", None)
        hf_config.model_type = MTP_MODEL_TYPE
        hf_config.update(
            {
                # Deliberately NOT `num_hidden_layers: 0`, which the sibling
                # `glm_ocr_mtp` / `glm4_moe_lite_mtp` branches do set.
                # `Glm5NextMultiTokenPredictor` derives the MTP layer index from
                # `text_config.num_hidden_layers`, so zeroing it renames the
                # ModuleDict key from "45" to "0", every `model.layers.45.*`
                # tensor lands nowhere, and the draft runs on random weights with
                # a ~0 acceptance rate and no error.
                "n_predict": n_predict,
                "architectures": ["Glm5NextMTPModel"],
            }
        )
        return hf_config

    assert _ORIGINAL_HF_CONFIG_OVERRIDE is not None, (
        "glm5_next_hf_config_override was called before install(); the "
        "original vLLM implementation was never captured."
    )
    return _ORIGINAL_HF_CONFIG_OVERRIDE(hf_config)


def install() -> bool:
    """Apply both halves of the patch. Returns True if anything changed."""
    global _ORIGINAL_HF_CONFIG_OVERRIDE
    from vllm.config import speculative as speculative_module

    changed = False

    # (1) The Literal. `get_args` is evaluated per call against the module
    # global, so rebinding the global is enough.
    known = get_args(speculative_module.MTPModelTypes)
    if MTP_MODEL_TYPE not in known:
        # A Literal built from a runtime tuple. Type checkers reject the
        # subscript because the members are not statically known, which is
        # the whole point: this reproduces at runtime an upstream edit that
        # would be written as a literal spelling.
        members = known + (MTP_MODEL_TYPE,)
        # pyrefly: ignore[invalid-literal, not-a-type]
        speculative_module.MTPModelTypes = Literal[members]
        changed = True

    # (2) The architecture branch. Only wrap once: a second wrap would make
    # the fallback chain recurse into the previous wrapper.
    current = SpeculativeConfig.hf_config_override
    if current is not glm5_next_hf_config_override:
        _ORIGINAL_HF_CONFIG_OVERRIDE = current
        SpeculativeConfig.hf_config_override = staticmethod(
            glm5_next_hf_config_override
        )
        changed = True

    if changed:
        logger.info(
            "glm5_next: patched vLLM's SpeculativeConfig for MTP "
            "(model_type=%s, draft architecture=Glm5NextMTPModel).",
            MTP_MODEL_TYPE,
        )
    return changed


# Self-bootstrap on import: unpickling `glm5_next_hf_config_override` in a
# spawned EngineCore process imports this module, and that process needs the
# Literal patch too.
install()
