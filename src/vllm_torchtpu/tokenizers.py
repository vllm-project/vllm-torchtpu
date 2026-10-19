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
"""Tokenizer compatibility for the pinned vLLM version.

vLLM renders DeepSeek-V4 chat by calling the checkpoint's ``encode_messages``
instead of a Jinja template, and builds its encode config from ``thinking_mode``,
``drop_thinking`` and ``reasoning_effort`` alone. ``continue_final_message`` is
dropped, so a trailing assistant turn is always rendered with ``eos_token`` and
a request that prefills one asks the model to generate past end-of-sentence.

The encoder reads a per-message ``wo_eos`` flag and has a template for that
case. This module registers a tokenizer that sets the flag.
"""

import copy
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import huggingface_hub
import jinja2
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import AddedToken
from vllm.tokenizers.hf import HfTokenizer, get_cached_tokenizer
from vllm.tokenizers.protocol import TokenizerLike
from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config
from vllm.transformers_utils.repo_utils import hf_api

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _load_chat_template_fallback(
    path_or_repo_id: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    subfolder: str = "",
) -> str | None:
    """Load a standalone Jinja template when HF provides no default template."""
    try:
        model_path = Path(path_or_repo_id)
        if not model_path.is_dir():
            model_path = Path(
                hf_api().snapshot_download(
                    str(path_or_repo_id),
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=(
                        local_files_only or huggingface_hub.constants.HF_HUB_OFFLINE
                    ),
                    allow_patterns=[str(Path(subfolder) / "*.jinja")],
                )
            )

        templates = sorted((model_path / subfolder).glob("*.jinja"))
        if len(templates) != 1:
            return None
        template = templates[0].read_text(encoding="utf-8")
        return template if template.strip() else None
    except Exception:
        logger.debug(
            "Failed to load standalone chat template for %s",
            path_or_repo_id,
            exc_info=True,
        )
        return None


class TPUHfTokenizer(TokenizerLike):
    """HF loader with standalone-template discovery from gpu-vllm 3adbc54027."""

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> TokenizerLike:
        try:
            tokenizer = cast(
                HfTokenizer,
                AutoTokenizer.from_pretrained(
                    path_or_repo_id,
                    *args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    cache_dir=download_dir,
                    **kwargs,
                ),
            )
        except ValueError as e:
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI. If the "
                    "model was created with a newer version of "
                    "transformers, consider upgrading: "
                    "`uv pip install --upgrade transformers`"
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

        if not tokenizer.chat_template:
            tokenizer.chat_template = _load_chat_template_fallback(
                path_or_repo_id,
                revision=revision,
                cache_dir=download_dir,
                token=kwargs.get("token"),
                local_files_only=kwargs.get("local_files_only", False),
                subfolder=kwargs.get("subfolder", ""),
            )

        encoder_config = get_sentence_transformer_tokenizer_config(
            path_or_repo_id, revision
        )
        if isinstance(encoder_config, dict) and encoder_config.get(
            "do_lower_case", False
        ):
            special_tokens_map: dict[
                str, str | AddedToken | Sequence[str | AddedToken]
            ] = {k: v.lower() for k, v in tokenizer.special_tokens_map.items()}
            tokenizer.add_special_tokens(special_tokens_map)

        return cast(TokenizerLike, get_cached_tokenizer(tokenizer))


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    """Match variable aliases, as in vLLM vllm/renderers/hf.py."""
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname

    # `ns.rest` matches the dotted alias "ns.rest".
    if isinstance(node, jinja2.nodes.Getattr) and "." in varname:
        base, _, attr = varname.rpartition(".")
        return node.attr == attr and _is_var_access(node.node, base)

    return False


def _attr_dotted_name(node: jinja2.nodes.Node) -> str | None:
    """Render an attribute-assignment target like ``ns.rest`` as a dotted name."""
    if isinstance(node, jinja2.nodes.NSRef):
        return f"{node.name}.{node.attr}"
    parts = []
    while isinstance(node, jinja2.nodes.Getattr):
        parts.append(node.attr)
        node = node.node
    if isinstance(node, jinja2.nodes.Name):
        parts.append(node.name)
        return ".".join(reversed(parts))
    return None


def _iter_nodes_assign_var_or_elems(root: jinja2.nodes.Node, varname: str):
    from vllm.renderers.hf import _is_var_or_elems_access

    yield root, varname

    seen = {varname}
    related_varnames = deque([varname])
    while related_varnames:
        related_varname = related_varnames.popleft()

        for assign_ast in root.find_all(jinja2.nodes.Assign):
            lhs = assign_ast.target
            rhs = assign_ast.node

            if isinstance(lhs, jinja2.nodes.Name) and _is_var_or_elems_access(
                rhs, related_varname
            ):
                yield assign_ast, lhs.name

                if lhs.name != related_varname and lhs.name not in seen:
                    seen.add(lhs.name)
                    related_varnames.append(lhs.name)

            # Namespace aliases: `namespace(rest=messages)` or `ns.rest = messages`.
            aliases: list[str] = []
            if (
                isinstance(lhs, jinja2.nodes.Name)
                and isinstance(rhs, jinja2.nodes.Call)
                and isinstance(rhs.node, jinja2.nodes.Name)
                and rhs.node.name == "namespace"
            ):
                aliases = [
                    f"{lhs.name}.{kwarg.key}"
                    for kwarg in rhs.kwargs
                    if _is_var_or_elems_access(kwarg.value, related_varname)
                ]
            elif isinstance(
                lhs, (jinja2.nodes.Getattr, jinja2.nodes.NSRef)
            ) and _is_var_or_elems_access(rhs, related_varname):
                dotted = _attr_dotted_name(lhs)
                aliases = [dotted] if dotted is not None else []

            for alias in aliases:
                yield assign_ast, alias
                if alias != related_varname and alias not in seen:
                    seen.add(alias)
                    related_varnames.append(alias)


def _tpu_deepseek_v4_tokenizer(tokenizer):
    """Return a copy of ``tokenizer`` that honors ``continue_final_message``.

    The concrete class is built per instance because vLLM constructs its own
    DeepSeek-V4 tokenizer class dynamically from the checkpoint's HF backend.
    """
    base_cls = tokenizer.__class__

    class _TpuDeepseekV4Tokenizer(base_cls):
        def apply_chat_template(self, messages, tools=None, **kwargs):
            """Set ``wo_eos`` on a trailing assistant turn when
            ``continue_final_message`` is set, so the encoder omits
            ``eos_token`` and the model continues that turn."""
            if kwargs.pop("continue_final_message", False):
                # The parent renders the conversation from kwargs when it is
                # there, so the marked copy goes back the same way.
                conversation = list(kwargs.get("conversation", messages))
                if conversation and conversation[-1].get("role") == "assistant":
                    conversation[-1] = {**conversation[-1], "wo_eos": True}
                kwargs["conversation"] = conversation
            return super().apply_chat_template(messages, tools, **kwargs)

        def __reduce__(self):
            return _tpu_deepseek_v4_tokenizer, (tokenizer,)

    tpu_tokenizer = copy.copy(tokenizer)
    tpu_tokenizer.__class__ = _TpuDeepseekV4Tokenizer
    return tpu_tokenizer


class TpuDeepseekV4Tokenizer(TokenizerLike):
    """Registry entry point for ``tokenizer_mode=deepseek_v4``."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

        return _tpu_deepseek_v4_tokenizer(
            DeepseekV4Tokenizer.from_pretrained(*args, **kwargs)
        )


def register_tokenizers() -> None:
    from vllm.reasoning import ReasoningParserManager
    from vllm.renderers import hf as hf_renderer
    from vllm.tokenizers import hf as hf_tokenizer
    from vllm.tokenizers.registry import TokenizerRegistry

    from vllm_torchtpu import envs

    if not hasattr(hf_tokenizer, "_load_chat_template_fallback"):
        TokenizerRegistry.register("hf", __name__, "TPUHfTokenizer")
        hf_renderer._is_var_access = _is_var_access
        hf_renderer._iter_nodes_assign_var_or_elems = _iter_nodes_assign_var_or_elems
    ReasoningParserManager.register_lazy_module(
        "musespark1x",
        "vllm_torchtpu.models.vllm.muse_spark.reasoning",
        "MuseSpark1xReasoningParser",
    )

    if not envs.TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE:
        return

    TokenizerRegistry.register("deepseek_v4", __name__, "TpuDeepseekV4Tokenizer")
