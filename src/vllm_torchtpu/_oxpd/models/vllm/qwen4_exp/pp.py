# SPDX-License-Identifier: Apache-2.0
"""Pipeline-stage ownership of Qwen4Exp's decoder layers.

A stage owns the contiguous layer range vLLM's ``get_pp_indices`` assigns it
(``VLLM_PP_LAYER_PARTITION`` included), the same split ``make_layers`` gives
the upstream models. The first stage also owns the token embedding, the last
the final hyper-connection mixer and the LM head. Cache owners are named after
their layer (``model.layers.<i>.<role>``); a stage registers and allocates only
its own, and vLLM projects the whole-model cache groups onto them.
"""

import re

_LAYER_NAME = re.compile(r"^model\.layers\.(\d+)\.")


def pipeline_size(cfg) -> int:
    return int(getattr(cfg.parallel_config, "pipeline_parallel_size", 1) or 1)


def _rank_and_size(cfg) -> tuple[int, int]:
    size = pipeline_size(cfg)
    if size == 1:
        return 0, 1
    from vllm.distributed import get_pp_group

    group = get_pp_group()
    return group.rank_in_group, group.world_size


def stage_layers(cfg) -> range:
    """Decoder layer indices this worker's pipeline stage owns."""
    num_layers = cfg.model_config.hf_text_config.num_hidden_layers
    rank, size = _rank_and_size(cfg)
    if size == 1:
        return range(num_layers)
    from vllm.distributed.utils import get_pp_indices

    start, end = get_pp_indices(num_layers, rank, size)
    return range(start, end)


def is_first_stage(cfg) -> bool:
    return _rank_and_size(cfg)[0] == 0


def is_last_stage(cfg) -> bool:
    rank, size = _rank_and_size(cfg)
    return rank == size - 1


def layer_index(name: str) -> int | None:
    """Decoder layer of a ``model.layers.<i>.`` cache owner or parameter name."""
    match = _LAYER_NAME.match(name)
    return None if match is None else int(match.group(1))


def owns_cache(name: str, layers: range, last_stage: bool) -> bool:
    """Whether a stage holds the cache owner ``name``.

    The MTP draft's owners (``mtp.layers.<i>.``) sit with the last stage,
    which hosts the draft.
    """
    index = layer_index(name)
    if index is None:
        return last_stage
    return index in layers
