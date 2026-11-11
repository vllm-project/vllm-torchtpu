# SPDX-License-Identifier: Apache-2.0
"""Checkpoint the Mamba state where a conversation's first user turn ends
(engine core).

``VLLM_TPU_MAMBA_ROOT_CHECKPOINT=1`` (default off).

A hybrid prefix hit can only resume where a Mamba state block is cached.
With ``prefix_cache_retention_interval=0`` those are the prompt ends of
earlier requests (the replay boundaries) and upstream's shared-prefix
junction, which is recorded only once two requests have already diverged.

An agentic fork (a flat chain or a fresh sub-conversation) usually re-sends
a parent's context rendered as fewer chat messages: its first prompt is the
parent's first user message followed by more content in the *same* message.
Its tokens therefore match the parent's later prompts only up to the end of
the parent's first user message, where the parent's prompt has
``<|im_end|>\\n<|im_start|>assistant`` and the fork has user text. No cached
state sits at or just below that point unless the parent's first turn was
itself served recently: the parent's later turns resume at their previous
turn's prompt end and never touch the first turn's checkpoint, so it ages
out of the LRU while the KV before it stays hot. The fork then re-prefills
from 0 (j-e3e88acd c32: flat first turns 0.153 measured hit; the replay with
real token boundaries reproduces 0.158).

With the flag:

- A request whose prompt contains a first assistant turn, and whose prefix
  hit stops before the end of the first user message (the "root"), gets the
  root as its shared-prefix junction when no other junction was detected.
  The scheduler's Mamba split then ends a chunk there and the manager caches
  the state (upstream's junction path, block-floored; it works with or
  without the tail fold, so also under main-KV DCP where the fold is off).
- A request whose hit passes the root moves the cached root checkpoint to
  the most-recently-used end of the free queue, so it lives as long as the
  conversation's KV does.

The root is found from the chat markers ``<|im_start|>`` ``assistant`` and
``<|im_end|>`` (resolved from the tokenizer, or
``VLLM_TPU_MAMBA_ROOT_CHECKPOINT_MARKER=<im_start>,<assistant>,<im_end>``).
Nothing changes for prompts without an assistant turn.
"""

from functools import wraps

from vllm.logger import init_logger

from vllm_torchtpu import envs

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_mamba_root_checkpoint_patch"
_ATTR = "_tpu_root_checkpoint"


def _parse_marker(text: str) -> tuple[int, int, int] | None:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        return None
    try:
        im_start, assistant, im_end = (int(p) for p in parts)
    except ValueError:
        return None
    return im_start, assistant, im_end


def _marker_from_tokenizer(model_config) -> tuple[int, int, int] | None:
    from vllm.tokenizers.registry import cached_tokenizer_from_config

    tokenizer = cached_tokenizer_from_config(model_config)
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant = tokenizer.encode("assistant", add_special_tokens=False)
    unk = getattr(tokenizer, "unk_token_id", None)
    if (
        not isinstance(im_start, int)
        or not isinstance(im_end, int)
        or im_start in (None, unk)
        or im_end in (None, unk)
        or len(assistant) != 1
    ):
        return None
    return im_start, assistant[0], im_end


def root_boundary(token_ids, marker: tuple[int, int, int]) -> int:
    """Tokens shared with a fork that continues the first user message.

    The index of the ``<|im_end|>`` closing the message before the first
    ``<|im_start|>assistant``; 0 when the prompt has no assistant turn.
    """
    im_start, assistant, im_end = marker
    n = len(token_ids)
    i = -1
    while True:
        try:
            i = token_ids.index(im_start, i + 1)
        except ValueError:
            return 0
        if i + 1 >= n:
            return 0
        if token_ids[i + 1] == assistant:
            break
    j = i - 1
    while j >= 0 and token_ids[j] != im_end:
        j -= 1
    return max(j, 0)


class RootCheckpoint:
    def __init__(self, manager, marker) -> None:
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager

        coordinator = manager.coordinator
        self.pool = coordinator.block_pool
        self.marker = marker
        self.mamba = [
            m
            for m in coordinator.single_type_managers
            if isinstance(m, MambaManager) and m.mamba_cache_mode == "align"
        ]
        self.group_ids = [m.kv_cache_group_id for m in self.mamba]
        self.block_size = self.mamba[0].block_size
        self.stats = {"junctions": 0, "refreshes": 0}

    def boundary(self, request) -> int:
        cached = getattr(request, "_tpu_root_boundary", None)
        if cached is None:
            cached = root_boundary(request.prompt_token_ids or [], self.marker)
            request._tpu_root_boundary = cached
        return cached

    def _cached_checkpoint(self, request, root: int):
        """The cached state blocks of the deepest checkpoint at or below the
        root on this request's hash chain, if any."""
        hash_block_size = self.pool.hash_block_size
        hashes = request.block_hashes
        for unit in (hash_block_size, self.block_size):
            position = root // unit * unit
            idx = position // hash_block_size - 1
            if idx < 0 or idx >= len(hashes):
                continue
            blocks = self.pool.get_cached_block(hashes[idx], self.group_ids)
            if blocks:
                return blocks
        return None

    def adjust(self, request, num_hit: int, junction: int) -> int:
        root = self.boundary(request)
        if root <= 0:
            return junction
        floor = root // self.pool.hash_block_size * self.pool.hash_block_size
        if num_hit >= floor:
            blocks = self._cached_checkpoint(request, root)
            if blocks:
                queue = self.pool.free_block_queue
                for block in blocks:
                    if block.ref_cnt == 0 and not block.is_null:
                        queue.remove(block)
                        queue.append(block)
                self.stats["refreshes"] += 1
            return junction
        if junction or root // self.block_size * self.block_size <= num_hit:
            return junction
        self.stats["junctions"] += 1
        return root


def _install(manager, marker) -> None:
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    if not any(
        isinstance(m, MambaManager) and m.mamba_cache_mode == "align"
        for m in manager.coordinator.single_type_managers
    ):
        return
    setattr(manager, _ATTR, RootCheckpoint(manager, marker))


def patch_mamba_root_checkpoint() -> None:
    if not envs.VLLM_TPU_MAMBA_ROOT_CHECKPOINT:
        return
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    original_get = KVCacheManager.get_computed_blocks
    if getattr(original_get, _PATCH_FLAG, False):
        return
    original_init = KVCacheManager.__init__
    original_scheduler_init = Scheduler.__init__
    env_marker = _parse_marker(envs.VLLM_TPU_MAMBA_ROOT_CHECKPOINT_MARKER)

    @wraps(original_init)
    def __init__(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        if self.enable_caching:
            _install(self, env_marker)

    @wraps(original_scheduler_init)
    def scheduler_init(self, *args, **kwargs) -> None:
        original_scheduler_init(self, *args, **kwargs)
        root = getattr(self.kv_cache_manager, _ATTR, None)
        if root is None:
            logger.warning(
                "VLLM_TPU_MAMBA_ROOT_CHECKPOINT: no align-mode Mamba group "
                "with prefix caching; off."
            )
            return
        if root.marker is None:
            try:
                root.marker = _marker_from_tokenizer(self.vllm_config.model_config)
            except Exception as e:  # noqa: BLE001 - optional feature
                logger.warning("VLLM_TPU_MAMBA_ROOT_CHECKPOINT: tokenizer: %s", e)
        if root.marker is None:
            setattr(self.kv_cache_manager, _ATTR, None)
            logger.warning(
                "VLLM_TPU_MAMBA_ROOT_CHECKPOINT: no <|im_start|>/assistant/"
                "<|im_end|> tokens; off (set "
                "VLLM_TPU_MAMBA_ROOT_CHECKPOINT_MARKER)."
            )
            return
        logger.info(
            "VLLM_TPU_MAMBA_ROOT_CHECKPOINT: on; the Mamba state at the end of "
            "a prompt's first user message is checkpointed and kept warm "
            "(markers %s).",
            root.marker,
        )

    @wraps(original_get)
    def get_computed_blocks(self, request):
        blocks, num_hit, junction = original_get(self, request)
        root = getattr(self, _ATTR, None)
        if root is not None and root.marker is not None:
            junction = root.adjust(request, num_hit, junction)
        return blocks, num_hit, junction

    for fn in (__init__, scheduler_init, get_computed_blocks):
        setattr(fn, _PATCH_FLAG, True)
    KVCacheManager.__init__ = __init__
    Scheduler.__init__ = scheduler_init
    KVCacheManager.get_computed_blocks = get_computed_blocks
