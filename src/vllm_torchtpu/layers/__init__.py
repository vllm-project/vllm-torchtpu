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
# Imported for their registration side effects; the out-of-tree
# RowParallelLinear in `linear` only affects layers built after the import, so
# this module must be imported before any model is constructed.
from vllm_torchtpu.layers.adapter import attention as attention
from vllm_torchtpu.layers.adapter import linear as linear
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.tokenizers import register_tokenizers


def register_layers() -> None:
    register_models()
    register_tokenizers()
    _maybe_install_frontend_tokcache()
    _maybe_install_frontend_fix()
    _maybe_install_detok_prompt_tail()


def _maybe_install_detok_prompt_tail() -> None:
    import os

    if os.environ.get("QX_DETOK_PROMPT_TAIL", "8") in ("", "0"):
        return
    try:
        from vllm_torchtpu.frontend_fix import maybe_install_detok_prompt_tail

        maybe_install_detok_prompt_tail()
    except Exception as e:  # never break startup
        import logging

        logging.getLogger(__name__).warning(
            "[QX_DETOK_PROMPT_TAIL] install failed: %r", e
        )


def _maybe_install_frontend_tokcache() -> None:
    import os

    if os.environ.get("QX_TOKENIZE_PREFIX_CACHE", "1") != "1":
        return
    try:
        from vllm_torchtpu.frontend_tokcache import (
            maybe_install_tokenize_prefix_cache,
        )

        maybe_install_tokenize_prefix_cache()
    except Exception as e:  # never break startup
        import logging

        logging.getLogger(__name__).warning(
            "[QX_TOKCACHE] install failed: %r", e
        )


def _maybe_install_frontend_fix() -> None:
    import os

    if os.environ.get("QX_LAZY_REASONING_COUNT", "1") != "1":
        return
    try:
        from vllm_torchtpu.frontend_fix import maybe_install_lazy_reasoning_count

        maybe_install_lazy_reasoning_count()
    except Exception as e:  # never break startup
        import logging

        logging.getLogger(__name__).warning(
            "[QX_LAZY_REASONING_COUNT] install failed: %r", e
        )
