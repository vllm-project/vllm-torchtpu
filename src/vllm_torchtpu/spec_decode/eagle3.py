# SPDX-License-Identifier: Apache-2.0
"""TPU Eagle3 and MTP speculative decoding proposer implementation.

Implements multi-step autoregressive draft token generation, KV cache slot
mapping, and device-resident greedy/rejection sampling integration.
"""

from __future__ import annotations

import bisect
import contextlib
import copy
import dataclasses
import functools
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, get_layers_from_vllm_config, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import (
    sequence_parallel,
    small_all_reduce_op,
    token_padding,
)
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilderContext,
)
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
from vllm_torchtpu.spec_decode.utils import (
    DraftChunkInputs,
    _force_draft_tp1,
    iter_mtp_shared_heads,
    maybe_share_embeddings,
    maybe_share_lm_head,
)
from vllm_torchtpu.utils import synchronize_tensors

# Sentinel for rejected / padding slots in the rejection-sampler output (and the
# async substitution tensors). Matches RejectionSampler.PLACEHOLDER_TOKEN_ID and
# the async runner's INVALID_TOKEN_ID.
INVALID_TOKEN_ID = -1


def _static_shape_args(fn):
    """Strip dynamo dynamic-dim marks from a compiled region's tensor args.

    vLLM's `@support_torch_compile` marks the model's `input_ids`/`positions`
    dynamic on every forward (`vllm/compilation/decorators.py:585`), and
    `torch._dynamo.mark_dynamic` records that by stamping a
    `_dynamo_dynamic_indices` attribute onto the tensor *object*. The stamp
    outlives the forward that applied it, and it has strictly higher precedence
    than `dynamic=False` -- `mark_static` is documented as the lower-precedence
    of the pair, so it cannot undo it. Hand such a tensor to one of the draft's
    static regions and dynamo lifts a SymInt into the graph, which torch-tpu
    rejects outright (`torch_tpu/_internal/compile/_backend.py:_raise_on_symint`
    -> "TPU backend: does not support dynamic shape").

    That is how `propose` used to die on its first draft step: `_forward_draft`
    runs the draft model on `positions`, marking it, and the very next line
    passed the same object to `_draft_gather_carries`. It presented as flaky and
    rank-dependent only because a warm compile cache short-circuits the backend
    call, so whether it fired depended on cache state rather than on shapes.

    A view is a distinct Python object and carries no stamp, so aliasing is
    enough; it copies nothing. Tensors that were never marked are passed through
    untouched.
    """

    def _unmark(a):
        if isinstance(a, torch.Tensor) and getattr(a, "_dynamo_dynamic_indices", None):
            return a[:]
        return a

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if any(
            isinstance(a, torch.Tensor) and getattr(a, "_dynamo_dynamic_indices", None)
            for a in args
        ):
            args = tuple(_unmark(a) for a in args)
        if kwargs:
            kwargs = {k: _unmark(v) for k, v in kwargs.items()}
            return fn(*args, **kwargs)
        return fn(*args)

    return wrapper


def _maybe_pad_dim0(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Right-pad a 1-D or 2-D tensor along dim 0 to `target_len` with zeros.

    If `target_len <= t.shape[0]` the tensor is returned unchanged (no-op).
    All current callers only pad upward; a caller passing a smaller target_len
    gets back the original tensor, not a truncation — callers must not rely on
    truncation behaviour.
    """
    n = t.shape[0]
    if target_len <= n:
        return t
    if t.ndim == 1:
        return torch.nn.functional.pad(t, (0, target_len - n))
    if t.ndim == 2:
        return torch.nn.functional.pad(t, (0, 0, 0, target_len - n))
    raise ValueError(f"_maybe_pad_dim0: unsupported ndim {t.ndim}, expected 1 or 2")


def _pad_dim0_repeat_last(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Right-pad dim 0 to `target_len` by repeating the last row.

    Used for scatter index/value pairs, where a repeated (index, value) row is
    an idempotent re-write. No-op if `target_len <= t.shape[0]`.
    """
    n = t.shape[0]
    if target_len <= n:
        return t
    return torch.cat((t, t[-1:].expand(target_len - n, *t.shape[1:])), dim=0)


def _maybe_pad_dim1(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad the 2nd dimension (dim=1) of a 2D tensor up to target_len."""
    n = t.shape[1]
    if target_len <= n:
        return t
    return torch.nn.functional.pad(t, (0, target_len - n))


if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


def _collect_topk_indices_buffers(mtp_model) -> list[torch.Tensor]:
    """The distinct `topk_indices_buffer` tensors of a DSA MTP draft.

    Mirrors the traversal in `DeepSeekMultiTokenPredictor.compact_topk_indices`
    (`layer.mtp_block.self_attn.mla_attn`). Layers commonly alias one buffer, so
    de-duplicate by identity to avoid saving and restoring it several times.
    """
    buffers: list[torch.Tensor] = []
    seen: set[int] = set()
    for layer in mtp_model.layers.values():
        mtp_block = getattr(layer, "mtp_block", None)
        self_attn = getattr(mtp_block, "self_attn", None)
        mla_attn = getattr(self_attn, "mla_attn", None)
        buf = getattr(mla_attn, "topk_indices_buffer", None)
        if buf is not None and id(buf) not in seen:
            seen.add(id(buf))
            buffers.append(buf)
    return buffers


class Eagle3Proposer:
    """Eagle3/MTP draft proposer for TPU.

    This class is responsible for loading the draft model and generating draft
    tokens based on the target model's outputs.

    Current limitations:
    - Greedy decoding only (temperature=0). The draft emits argmax tokens, not
      a probability distribution, so the rejection sampler cannot perform
      proper non-greedy rejection sampling. Non-greedy requests are rejected
      at add_request time in tpu_platform.py before reaching this proposer.
    - draft_tensor_parallel_size selects the draft's parallelism: 1 runs the
      eagle3/mtp head fully replicated on every TPU worker; target tp runs it
      sharded across the TP group.
    """

    def __init__(
        self,
        runner: TPUModelRunner,
        vllm_config: VllmConfig,
    ):
        self.runner = runner
        self.vllm_config = vllm_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.speculative_config = speculative_config
        draft_tp = self.speculative_config.draft_tensor_parallel_size
        target_tp = self.vllm_config.parallel_config.tensor_parallel_size
        # Default to (draft_tp == target tp) when unset. vLLM's
        # _verify_and_get_draft_tp already resolves an unset value to target tp
        # for eagle3/mtp, but set it explicitly here so self-documented.
        if draft_tp is None:
            draft_tp = target_tp
            self.speculative_config.draft_tensor_parallel_size = draft_tp
        # Only two draft parallelisms are supported: fully REPLICATED (tp=1) or
        # SHARDED across the whole TP group (draft_tp == target_tp).
        if draft_tp not in (1, target_tp):
            raise ValueError(
                f"{self.speculative_config.method} "
                f"draft_tensor_parallel_size={draft_tp} is unsupported "
                f"on TPU: it must be 1 (replicated draft) or {target_tp} "
                f"(== target tensor_parallel_size, sharded draft)."
            )
        self._draft_replicated = draft_tp == 1
        # Independent from replication: target TP1 / draft TP1 can alias.
        self._draft_tp_matches_target = draft_tp == target_tp
        # Lazily resolved by _draft_has_moe once the draft model exists.
        self._draft_has_moe_cache: bool | None = None
        logger.info(
            "%s draft parallelism: %s (draft_tp=%s).",
            self.speculative_config.method,
            "REPLICATED (tp=1)" if self._draft_replicated else "SHARDED",
            draft_tp,
        )
        self.draft_model = None
        self.draft_vllm_config: VllmConfig
        # Attention-layer names belonging to the draft model. Populated in
        # load_model by diffing the global attention-layer registry before
        # vs after the draft model loads.
        self._draft_attn_layer_names: set[str] | None = None
        # True once load_model has wired this draft's attention layers to read
        # the target's KV cache (cross-model KV sharing, Gemma-4-style MTP).
        # Resolved from the draft model's own declaration, NOT from
        # speculative_config.method: `mtp` also covers DeepSeek-V4 MTP, whose
        # draft owns its KV cache. Drives `advance_draft_positions`.
        self._draft_kv_shared: bool = False
        # Populated per-step by the runner with the per-chunk draft inputs
        # (token ids, positions, attn ctx, aux hidden states) captured
        # during the target verify forward.
        self.draft_chunks: list[DraftChunkInputs] | None = None
        # Per-propose-call cache of loop-step draft attention metadata, keyed
        # by id(chunk): (metadata, seq_lens, seq_lens_delta) of the chunk's
        # last loop step. Between loop steps everything in the metadata is
        # constant except seq_lens, which moves with seq_lens_delta, so steps
        # >= 2 swap the seq_lens field on the cached objects instead of
        # recomputing it or paying the full upstream builder walk.
        self._draft_md_cache: dict[int, tuple[dict, torch.Tensor, int]] = {}
        # Bucketed [p] pad-row index (clamped to num_reqs - 1) per
        # (p, num_reqs, device), for the async / bootstrap seed prep.
        self._async_pad_req_idx_cache: dict[tuple, torch.Tensor] = {}
        # Loop-step (query_start_loc, request_distribution) per
        # (kernel_num_reqs, num_reqs, device); both depend only on num_reqs.
        self._loop_qsl_reqdist_cache: dict[
            tuple, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        # Zero [kernel_num_reqs] rejected counts per (knr, device), for the
        # post-step-0 bridge on an all-accept sync step.
        self._zero_rej_dev_cache: dict[tuple, torch.Tensor] = {}
        # The last propose's per-chunk padded [p, K] drafts and its returned
        # device tensor; tpu_runner's async substitution reads them to skip a
        # re-pad when `drafts is _last_propose_device_out`.
        self._last_padded_drafts_per_chunk: list[torch.Tensor] | None = None
        self._last_propose_device_out: torch.Tensor | None = None
        # Set by tpu_runner for one propose: return chunk 0's slice instead of
        # concatenating chunks (the runner substitutes both chunks itself).
        self._skip_next_multi_chunk_draft_cat: bool = False
        # DSA MTP IndexShare (GLM-5.2 / DeepSeek-V3.2). Resolved in load_model.
        self._share_mtp_indices = False
        # The distinct topk_indices_buffer tensors of the draft's MTP layers.
        self._mtp_topk_buffers: list[torch.Tensor] = []
        # IndexShare's second compiled program, for draft steps 1+. Built in
        # load_model, and only when IndexShare is on.
        self._mtp_loop_model: torch.nn.Module | None = None
        # Whether that program has run once (its first run compiles it).
        self._mtp_loop_model_warm = False
        # MTP loop steps 1..K-1 as one compiled program (see
        # mtp_fused_loop.py), and whether it has run once. Built in
        # load_model; None means the per-step loop.
        self._mtp_fused_loop: torch.nn.Module | None = None
        self._mtp_fused_loop_warm = False
        # Greedy draft head for TPU_GREEDY_TOP_TOKENS: the draft's head owner,
        # or a _DraftHeadOwnerProxy over a quantized lm-head proxy
        # (TPU_DRAFT_LM_HEAD_QUANT). None: compute_logits + argmax.
        self._draft_greedy_head_owner: torch.nn.Module | None = None
        # Whether the draft's MoE layers were admitted to sequence-parallel
        # MoE (main #1562: the MTP drafter runs under SP like the target).
        # Resolved in _load_draft_model from the loaded draft's layers.
        self._draft_sequence_parallel = False
        # Dedicated TokenPaddingState for separately compiled draft loop models
        # (_mtp_fused_loop / _mtp_loop_model), so Target + Draft Step 0
        # (runner._token_padding_state) and Draft Loop Steps 1..K-1 never
        # ping-pong _last_counts or incur synchronous H2D copies per step.
        self._loop_token_padding_state: token_padding.TokenPaddingState | None = None

    def _draft_rows_sliced(
        self, num_tokens: int, step0_gather_indices: torch.Tensor | None = None
    ) -> bool:
        """Whether a draft forward over `num_tokens` rows slices them over TP.

        The same predicate as qwen3_next_patch's `_rows_are_sequence_parallel`
        on a sequence-parallel layer: rows at or above
        `sequence_parallel.min_rows()` are sliced, except under the MTP step-0
        post-attention gather, whose rows shrink mid-layer and so always run
        replicated. Because compiled draft programs (`draft_model`,
        `_mtp_fused_loop`, `_mtp_loop_model`) are traced once at the first
        padded token bucket outside `shape_variants.warmup()`, a runner whose
        smallest padded bucket is below `min_rows()` traces them replicated and
        must keep `rows_sliced=False` across every bucket of that trace.
        """
        if not self._draft_sequence_parallel or step0_gather_indices is not None:
            return False
        min_rows = sequence_parallel.min_rows()
        paddings = getattr(self.runner, "num_tokens_paddings", None)
        if (
            paddings
            and not getattr(self.runner, "enforce_eager", False)
            and int(paddings[0]) < min_rows
        ):
            return False
        return int(num_tokens) >= min_rows

    def _rank_padding_counts(
        self, num_valid_tokens: int, num_tokens: int, rows_sliced: bool
    ) -> tuple[int, int]:
        """(valid, padded) rows this rank's padding mask covers: its slice of
        the step when rows are sliced over TP, else all of them. Mirrors
        tpu_runner._token_padding_update."""
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if not rows_sliced or tp_size <= 1:
            return num_valid_tokens, num_tokens
        from vllm.distributed import get_tensor_model_parallel_rank

        chunk = num_tokens // tp_size
        rank_valid = num_valid_tokens - get_tensor_model_parallel_rank() * chunk
        return max(0, min(chunk, rank_valid)), chunk

    def _get_loop_token_padding_state(self) -> token_padding.TokenPaddingState | None:
        runner_state = getattr(self.runner, "_token_padding_state", None)
        if runner_state is None:
            return None
        if self._loop_token_padding_state is None:
            max_tokens = runner_state.local_padding_mask.shape[0]
            self._loop_token_padding_state = token_padding.TokenPaddingState.create(
                max_local_tokens=max_tokens,
                device=self.runner.device,
            )
        return self._loop_token_padding_state

    @staticmethod
    def _draft_valid_tokens(
        chunk: DraftChunkInputs, step_idx: int, num_tokens_padded: int
    ) -> int:
        num_reqs = getattr(chunk, "num_reqs", 0) or 0
        query_start_loc_np = getattr(chunk, "query_start_loc_np", None)
        if step_idx > 0:
            valid = int(num_reqs)
        elif query_start_loc_np is not None and num_reqs > 0:
            valid = int(query_start_loc_np[num_reqs])
        else:
            valid = 0
        return max(0, min(valid, int(num_tokens_padded)))

    @property
    def advance_draft_positions(self) -> bool:
        """Whether each propose step advances the draft's positions/seq_lens.

        A draft that owns its KV cache (eagle3, DeepSeek-V4 MTP) appends one
        token per step, so both advance by 1. A cross-model KV-sharing draft
        (Gemma-4 MTP) is Q-only: it writes no KV slots, and every step re-reads
        the same target prefix with only the query changing, so both stay
        fixed. Mirrors `Gemma4Speculator.advance_draft_positions` upstream.
        """
        return not self._draft_kv_shared

    def load_model(self, target_model) -> None:
        """Load the draft model and share embeddings/lm_head with target.

        Also registers the eagle3 aux hidden state layers on the target so
        its forward returns (hidden_states, aux_hidden_states) at runtime.
        """
        # Snapshot the target's attention layer names before the draft model
        # adds its own; diff after load gives us the draft layer names.
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )
        self._load_draft_model()
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase
        )
        draft_attn_layer_names = set(all_attn_layers.keys()) - target_attn_layer_names
        self._draft_attn_layer_names = draft_attn_layer_names

        if self.runner._pcp_enabled and self.runner._mtp_enabled:
            self._validate_pcp_draft_model(draft_attn_layer_names)
        self._validate_dsa_draft(all_attn_layers, draft_attn_layer_names)

        assert self.draft_model is not None
        maybe_share_embeddings(
            self.draft_model, target_model, self._draft_tp_matches_target
        )
        maybe_share_lm_head(
            self.draft_model,
            target_model,
            self._draft_replicated,
            self._draft_tp_matches_target,
            materialize_if_mismatched=False,
        )
        if self.speculative_config.method == "eagle3":
            # Lazy + guarded: only eagle3 needs this vLLM internal, so a wrong vLLM
            # checkout shouldn't break unrelated TPU runs at import time.
            try:
                from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
                    set_eagle3_aux_hidden_state_layers,
                )
            except ImportError as e:
                raise ImportError(
                    "Eagle3 speculative decoding requires a vLLM build with "
                    "vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils; the "
                    "installed vLLM lacks it — check the vLLM version/checkout."
                ) from e
            set_eagle3_aux_hidden_state_layers(target_model, self.speculative_config)

        if self.speculative_config.method == "mtp":
            self._setup_gemma4_kv_sharing(target_model, target_attn_layer_names)

        self._validate_mtp_layer_count()
        self._init_draft_greedy_head()
        self._init_mtp_index_sharing()
        if self._share_mtp_indices:
            self._build_mtp_loop_model()
        self._build_mtp_fused_loop()
        if (
            self._mtp_fused_loop is None
            and self._mtp_loop_model is None
            and envs.TPU_MTP_STEP0_POST_ATTN_GATHER
            and self.speculative_config.method == "mtp"
            and not self._share_mtp_indices
            and not self._draft_kv_shared
            and getattr(self.speculative_config, "num_speculative_tokens", 1) > 1
        ):
            # Step 0 traces with the post-attention gather baked in, so the
            # per-step loop's steps 1..K-1 need a program of their own.
            self._build_mtp_step0_gather_loop_model()

        # IndexShare assumes the draft owns its KV cache and advances
        # positions once per step; cross-model KV sharing freezes them and
        # makes every step re-read the same target prefix. Nothing today can
        # set both — a DSA draft never takes the Gemma-4 wiring path, and a
        # Gemma-4 draft has no topk_indices_buffer — but the propose loop
        # guards the two independently, so pin the invariant here rather than
        # let a future DSA+KV-sharing draft find it at runtime.
        if self._share_mtp_indices and self._draft_kv_shared:
            raise RuntimeError(
                "IndexShare and cross-model KV sharing are both enabled on "
                "this draft. IndexShare's saved top-k rows assume positions "
                "advance per draft step, which cross-model KV sharing "
                "freezes; the combination is unvalidated."
            )

    def _init_draft_greedy_head(self) -> None:
        """Resolve the draft's greedy head for TPU_GREEDY_TOP_TOKENS.

        The draft's tokens then come from the vocab-parallel argmax
        (`LogitsProcessor.get_top_tokens`, patched by greedy_top_tokens), as
        the target's do. TPU_DRAFT_LM_HEAD_QUANT picks a quantized lm-head
        proxy for the draft only (read with os.environ; unset means fp8 for a
        bf16 head, "none"/"bf16" keep the head as is; see
        greedy_top_tokens.parse_draft_lm_head_quant). It changes draft tokens,
        and so the acceptance rate, never the verified output. A draft with a
        draft->target id map, or one the argmax cannot serve, keeps
        compute_logits + argmax.
        """
        self._draft_greedy_head_owner = None
        draft_model = self.draft_model
        if (
            draft_model is None
            or not envs.TPU_GREEDY_TOP_TOKENS
            or getattr(draft_model, "draft_id_to_target_id", None) is not None
        ):
            return
        from vllm_torchtpu.runner import greedy_top_tokens

        try:
            owner = greedy_top_tokens.resolve_head_owner(
                draft_model, self.runner._dp_lockstep_enabled()
            )
        except NotImplementedError:
            return
        draft_head = owner.lm_head
        head_quant = getattr(draft_head, "_quant_mode", None)
        quant_env = os.environ.get("TPU_DRAFT_LM_HEAD_QUANT")
        quant_mode, group_size = greedy_top_tokens.parse_draft_lm_head_quant(
            quant_env
            if quant_env is not None
            else ("fp8" if head_quant is None else None)
        )
        if quant_mode is not None and head_quant != quant_mode:
            proxy_head = greedy_top_tokens._QuantizedLMHeadProxy(
                draft_head, mode=quant_mode, group_size=group_size or 128
            )
            owner = greedy_top_tokens._DraftHeadOwnerProxy(
                owner.logits_processor, proxy_head
            )
        self._draft_greedy_head_owner = owner
        logger.info(
            "TPU_GREEDY_TOP_TOKENS enabled for draft proposer (%s, quant=%s).",
            type(owner).__name__,
            getattr(owner.lm_head, "_quant_desc", "bf16"),
        )

    def _validate_mtp_layer_count(self) -> None:
        """One nextn layer only: the proposer's per-step wiring assumes it.

        A DeepSeek-style MTP checkpoint may ship several nextn layers, and
        `DeepSeekMultiTokenPredictor` then cycles them by
        `spec_step_idx % num_mtp_layers`. Two things here assume a single
        layer, and both go wrong quietly if there is more than one:

        * `_draft_propose_token` calls `compute_logits(hidden)` and takes the
          default `spec_step_idx=0`, so every step's logits come off layer 0's
          `shared_head` -- while the forward that produced `hidden` ran layer
          `step % num_mtp_layers`. Head and body disagree from step 1 on.
        * Each `DeepSeekMultiTokenPredictorLayer` allocates its *own*
          `topk_indices_buffer`, but only the layer that ran at draft step 0
          has one written. `_mtp_compact_and_save` would gather garbage rows
          out of the others, and every later step would read a buffer no
          indexer ever filled.

        Neither raises on its own, so refuse the checkpoint here instead. Every
        released DeepSeek / GLM MTP checkpoint sets this to 1.

        Both problems need a head *per nextn layer*, so that is what gates the
        refusal -- not the layer count alone. Most MTP classes (Qwen3.5 /
        Qwen3-Next, MiMo, ERNIE, Gemma4, ...) also expose `num_mtp_layers`,
        but cycle their nextn layers under one top-level `lm_head` and carry
        no indexer: the logits never depend on which layer ran, and there is
        no top-k buffer to share. They work with any layer count and must not
        be turned away here. Every DSA draft is a `DeepSeekMTP`, which does
        have per-layer heads, so the second problem is covered by the same
        test.
        """
        if self.speculative_config.method != "mtp":
            return
        mtp_model = getattr(self.draft_model, "model", None)
        num_mtp_layers = getattr(mtp_model, "num_mtp_layers", None)
        if num_mtp_layers is None or num_mtp_layers <= 1:
            return
        assert self.draft_model is not None
        if not any(True for _ in iter_mtp_shared_heads(self.draft_model)):
            return
        raise NotImplementedError(
            f"The MTP draft has num_nextn_predict_layers={num_mtp_layers} "
            "with a separate shared_head per layer. This proposer reads "
            "logits off layer 0's shared_head at every draft step, and "
            "IndexShare only ever fills the top-k buffer of the layer that "
            "ran at step 0. Use a single-layer MTP checkpoint."
        )

    def _init_mtp_index_sharing(self) -> None:
        """Enable IndexShare when a DSA MTP checkpoint asks for it.

        GLM-5.2 trains its MTP layer *with* IndexShare: draft step 0 computes
        the sparse top-k and steps 1..K-1 reuse it. That is not merely a
        speedup. Because the reused index set was computed before the draft's
        own token existed, it can never point at that token, so every draft
        step attends only to KV derived from target hidden states -- the
        property GLM calls KVShare. Recomputing the indices per step lets a
        draft step attend to its own draft-derived KV, which is the
        train/inference mismatch IndexShare was introduced to remove and costs
        acceptance rate. So this is on whenever the checkpoint asks for it,
        for any chunk count.
        """
        self._share_mtp_indices = False
        self._mtp_topk_buffers = []
        if self.speculative_config.method != "mtp":
            return
        draft_config = self.speculative_config.draft_model_config
        hf_config = getattr(draft_config, "hf_config", None)
        if not getattr(hf_config, "index_share_for_mtp_iteration", False):
            return

        mtp_model = getattr(self.draft_model, "model", None)
        if mtp_model is None or not hasattr(mtp_model, "set_skip_topk"):
            logger.warning(
                "Draft config sets index_share_for_mtp_iteration but %s "
                "exposes no set_skip_topk; drafting without IndexShare, which "
                "lowers the acceptance rate this checkpoint was trained for.",
                type(self.draft_model).__name__,
            )
            return

        self._mtp_topk_buffers = _collect_topk_indices_buffers(mtp_model)
        if not self._mtp_topk_buffers:
            logger.warning(
                "Draft config sets index_share_for_mtp_iteration but no MTP "
                "layer exposes a topk_indices_buffer; drafting without "
                "IndexShare."
            )
            return

        # The buffer is sized `max_num_batched_tokens` rows -- one per token of
        # a first-pass forward -- but the loop steps index it by *request*:
        # compaction gathers to rows [0 : p] and each loop forward reads that
        # same prefix, where p is the loop bucket for the chunk's request
        # count. p is a token-padding bucket, so it is bounded by the largest
        # bucket the runner will ever pick for `max_num_reqs` requests. Check
        # that against the rows actually allocated rather than trusting
        # max_num_batched_tokens >= max_num_seqs to hold: if it does not, the
        # compaction store and the loop-step read both run off the end of the
        # buffer.
        # `_loop_bucket` is the same function with a DP-lockstep branch that
        # is a no-op at this input (it clamps num_reqs up to max_num_reqs,
        # which is what is already being asked for), so call the padding
        # helper straight and keep load-time out of `_draft_has_moe`.
        from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

        required_rows = _get_padded_token_len(
            self.runner.num_tokens_paddings, self.runner.max_num_reqs
        )
        for buf in self._mtp_topk_buffers:
            if buf.shape[0] < required_rows:
                sched = self.vllm_config.scheduler_config
                raise RuntimeError(
                    "The MTP draft's topk_indices_buffer holds "
                    f"{buf.shape[0]} rows but IndexShare compacts and reads "
                    f"up to {required_rows} (the loop bucket for "
                    f"max_num_reqs={self.runner.max_num_reqs}). The buffer is "
                    "allocated from max_num_batched_tokens="
                    f"{sched.max_num_batched_tokens}; raise it to at least "
                    "the loop bucket, or lower max_num_seqs."
                )

        # A loop step reads the buffer without the indexer having written it
        # this step; -1 is the sparse-MLA padding sentinel (kv_len counts
        # `!= -1`), so an unwritten row degrades to an empty attend instead of
        # an out-of-range gather. Matters for the EP-DP dummy draft, which
        # replays loop-step forwards on a rank that never ran a real propose.
        for buf in self._mtp_topk_buffers:
            buf.fill_(-1)

        self._share_mtp_indices = True

    def _get_loop_vllm_config(self) -> VllmConfig:
        """A shallow copy of `draft_vllm_config` whose `compile_sizes` is capped
        at `max_loop_bucket`, since loop programs (`_mtp_fused_loop` and
        `_mtp_loop_model`) only ever execute at `p = _loop_bucket(num_reqs) <=
        max_loop_bucket`."""
        if not hasattr(self.draft_vllm_config, "compilation_config"):
            return self.draft_vllm_config
        from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

        max_loop_bucket = _get_padded_token_len(
            self.runner.num_tokens_paddings, self.runner.max_num_reqs
        )
        loop_vllm_config = copy.copy(self.draft_vllm_config)
        loop_cc = copy.copy(self.draft_vllm_config.compilation_config)
        if loop_cc.compile_sizes:
            capped = [
                s
                for s in loop_cc.compile_sizes
                if not isinstance(s, int) or s <= max_loop_bucket
            ]
            if capped:
                loop_cc.compile_sizes = capped
        loop_vllm_config.compilation_config = loop_cc
        return loop_vllm_config

    def _build_mtp_loop_model(self) -> None:
        """Give IndexShare's draft steps 1+ a compiled program of their own.

        On TPU the draft forward is compiled once and vLLM drops the guards
        that would recompile it, so the MTP attention's `skip_topk` check is
        baked in by the first trace -- a step-0 pass, with the indexer on --
        and flipping `skip_topk` later changes nothing. `MtpLoopStepModel`
        runs the same draft forward over the same weights, KV caches and top-k
        table as a second program, which `_draft_model_for_step` first runs
        with `skip_topk` True. The target needs nothing like this: each of its
        layers gets a fixed `skip_topk` when it is built.

        Only IndexShare drafts get here, so every other drafter keeps exactly
        one compiled program. If the program can't be built, IndexShare is
        switched off rather than left in place doing nothing.
        """
        assert self.draft_model is not None
        try:
            from vllm_torchtpu.spec_decode.mtp_loop_model import (
                MTP_DYNAMIC_ARG_DIMS,
                MtpLoopStepModel,
            )

            draft_dims = getattr(self.draft_model, "_dynamic_arg_dims", None)
            if draft_dims is not None and dict(draft_dims) != MTP_DYNAMIC_ARG_DIMS:
                raise ValueError(
                    f"the draft compiles with dynamic dims {dict(draft_dims)} "
                    f"but the loop-step program uses {MTP_DYNAMIC_ARG_DIMS}"
                )
            loop_vllm_config = self._get_loop_vllm_config()
            # Built the way _load_draft_model builds the draft: the compile
            # wrapper takes its config from the current one.
            with (
                set_model_tag("eagle_head"),
                set_vllm_model_wrapper_context(
                    mesh=self.runner.mesh, vllm_config=loop_vllm_config
                ),
                set_current_vllm_config(loop_vllm_config),
            ):
                self._mtp_loop_model = MtpLoopStepModel(
                    vllm_config=loop_vllm_config,
                    prefix="mtp_index_share_loop",
                    mtp=self.draft_model,
                )
        except Exception:
            logger.warning(
                "Could not build IndexShare's loop-step program; drafting "
                "without IndexShare, which lowers the acceptance rate this "
                "checkpoint was trained for.",
                exc_info=True,
            )
            self._share_mtp_indices = False
            self._mtp_topk_buffers = []
            self._mtp_loop_model = None
            return
        self._mtp_loop_model_warm = False
        logger.info(
            "MTP IndexShare enabled: draft step 0 computes the sparse top-k, "
            "steps 1+ reuse it in a separately compiled program "
            "(%d buffer(s)).",
            len(self._mtp_topk_buffers),
        )

    def _build_mtp_step0_gather_loop_model(self) -> None:
        """Build a separately compiled loop-step program for steps 1..K-1 when
        `TPU_MTP_STEP0_POST_ATTN_GATHER=1` and `VLLM_TPU_FUSED_DRAFT_LOOP=0`,
        so step 0 traces with `_step0_gather_indices` set while steps 1..K-1
        trace with `_step0_gather_indices=None` using the draft's own
        `_dynamic_arg_dims` (e.g. `positions=-1` on Qwen3.5 MRoPE)."""
        draft_model = self.draft_model
        if draft_model is None:
            return
        draft_dims = getattr(draft_model, "_dynamic_arg_dims", None)
        if draft_dims is None:
            return
        try:
            from vllm_torchtpu.models.vllm.qwen3_next_patch import (
                is_step0_post_attn_gather_installed,
            )
            from vllm_torchtpu.spec_decode.mtp_loop_model import (
                build_mtp_loop_step_model,
            )

            if not is_step0_post_attn_gather_installed(draft_model):
                return
            loop_vllm_config = self._get_loop_vllm_config()
            with (
                set_model_tag("eagle_head"),
                set_vllm_model_wrapper_context(
                    mesh=self.runner.mesh, vllm_config=loop_vllm_config
                ),
                set_current_vllm_config(loop_vllm_config),
            ):
                self._mtp_loop_model = build_mtp_loop_step_model(
                    vllm_config=loop_vllm_config,
                    prefix="mtp_step_loop",
                    mtp=draft_model,
                    dynamic_arg_dims=draft_dims,
                )
        except Exception:
            logger.warning(
                "Could not build separate MTP step-loop model for step-0 gather.",
                exc_info=True,
            )
            self._mtp_loop_model = None
            return
        self._mtp_loop_model_warm = False
        logger.info(
            "MTP step-0 post-attention gather loop program enabled (`mtp_step_loop`)."
        )

    def _build_mtp_fused_loop(self) -> None:
        """Compile an MTP draft's loop steps 1..K-1 as one program, so the
        host dispatches once per propose for them instead of twice per step.
        Opt-in with `VLLM_TPU_FUSED_DRAFT_LOOP=1`; off, the per-step loop runs.

        Only for the plain case: IndexShare restores top-k rows between steps
        on the host, a draft->target id map would need its gather in the
        program, and a cross-model KV-sharing draft (Gemma-4 MTP) keeps
        positions and seq_lens fixed across steps where the program advances
        them.

        The program also proposes step 0's token (`propose_step0`): propose
        hands it step 0's hidden carry, and it returns the stacked [p, K]
        drafts. Its lm-head is the greedy draft head when one is set.
        """
        draft_model = self.draft_model
        if (
            draft_model is None
            or not envs.VLLM_TPU_FUSED_DRAFT_LOOP
            or self.speculative_config.method != "mtp"
            or self._share_mtp_indices
            or not self.advance_draft_positions
            or getattr(draft_model, "draft_id_to_target_id", None) is not None
        ):
            return
        draft_dims = getattr(draft_model, "_dynamic_arg_dims", None)
        if draft_dims is None:
            return
        K = self.speculative_config.num_speculative_tokens
        if K < 2:
            return
        try:
            from vllm_torchtpu.spec_decode.mtp_fused_loop import build_mtp_fused_loop

            loop_vllm_config = self._get_loop_vllm_config()
            with (
                set_model_tag("eagle_head"),
                set_vllm_model_wrapper_context(
                    mesh=self.runner.mesh, vllm_config=loop_vllm_config
                ),
                set_current_vllm_config(loop_vllm_config),
            ):
                self._mtp_fused_loop = build_mtp_fused_loop(
                    vllm_config=loop_vllm_config,
                    prefix=f"mtp_fused_loop_k{K}",
                    mtp=draft_model,
                    num_steps=K - 1,
                    dynamic_arg_dims=draft_dims,
                    greedy_head_owner=self._draft_greedy_head_owner,
                    propose_step0=True,
                )
        except Exception:
            logger.warning(
                "Could not build the fused MTP loop program; draft loop steps "
                "run one program each.",
                exc_info=True,
            )
            self._mtp_fused_loop = None
            return
        self._mtp_fused_loop_warm = False
        logger.info("MTP draft loop steps 1..%d run as one compiled program.", K - 1)

    def _run_mtp_fused_loop(
        self,
        chunk: DraftChunkInputs,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        num_rejected_np: np.ndarray | None,
        num_rejected_dev: torch.Tensor | None,
        loop_query_start_loc: torch.Tensor,
        loop_request_distribution: torch.Tensor,
        precomputed_seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Draft step 0's token and steps 1..K-1 of one chunk in one dispatch.

        `positions` is step 1's and `hidden_states` is step 0's carry [p, H]
        (the program proposes step 0's token from it). `precomputed_seq_lens`
        is step 1's seq_lens when the post-step-0 bridge already built them.
        Returns the stacked [p, K] tokens, or None when this chunk's metadata
        doesn't fit the program (the caller then runs the per-step loop).
        """
        assert self._mtp_fused_loop is not None
        attn_metadata = self._build_draft_attn_metadata(
            chunk=chunk,
            step_idx=1,
            seq_lens_delta=1,
            num_rejected_np=num_rejected_np,
            num_rejected_dev=num_rejected_dev,
            loop_query_start_loc=loop_query_start_loc,
            loop_request_distribution=loop_request_distribution,
            precomputed_seq_lens=precomputed_seq_lens,
        )
        # The program bumps seq_lens in place between steps; every draft
        # layer must read that one tensor. It is fresh from the step-1 build
        # (base + 1 - rejected), so nothing else holds it -- except the
        # loop-step metadata cache, which only steps >= 2 read.
        seq_lens_set = {id(md.seq_lens): md.seq_lens for md in attn_metadata.values()}
        if len(seq_lens_set) != 1:
            return None
        (seq_lens,) = seq_lens_set.values()
        self._draft_md_cache.pop(id(chunk), None)

        num_tokens = hidden_states.shape[0]
        # Every loop step runs at this one row count, so one decision covers
        # the program; it must match the draft forward's own row gating.
        rows_sliced = self._draft_rows_sliced(num_tokens)
        sequence_parallel.set_active(rows_sliced)
        loop_padding_state = self._get_loop_token_padding_state()
        if loop_padding_state is not None:
            loop_padding_state.update(
                *self._rank_padding_counts(
                    self._draft_valid_tokens(chunk, 1, num_tokens),
                    num_tokens,
                    rows_sliced,
                )
            )
        max_loop_bucket = self._loop_bucket(self.runner.max_num_reqs)
        with (
            set_model_tag("eagle_head"),
            token_padding.use_padding_state(loop_padding_state),
            small_all_reduce_op.force_small_all_reduce(
                enabled=(
                    not rows_sliced and max_loop_bucket <= small_all_reduce_op.MAX_ROWS
                )
            ),
            set_forward_context(
                attn_metadata,
                self.draft_vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=self.runner._dp_num_tokens_across_dp(num_tokens),
            ),
            set_vllm_model_wrapper_context(
                mesh=self.runner.mesh, vllm_config=self.draft_vllm_config
            ),
        ):
            if not self._mtp_fused_loop_warm:
                # The first call traces and compiles; like the loop-step
                # program's, its outputs can carry symbolic sizes, so run it
                # again compiled. The repeat rewrites the same KV entries;
                # seq_lens is put back first because the program advanced it.
                seq_lens_step1 = seq_lens.clone()
                self._mtp_fused_loop(None, positions, hidden_states, seq_lens)
                seq_lens.copy_(seq_lens_step1)
                self._mtp_fused_loop_warm = True
            return self._mtp_fused_loop(None, positions, hidden_states, seq_lens)

    def _draft_model_for_step(self, step_idx: int) -> torch.nn.Module:
        """The compiled program draft step `step_idx` runs.

        Without a loop-step program that is always the draft itself. With
        IndexShare, step 0 runs the draft (indexer on) and steps 1+ the
        loop-step program (indexer off). `skip_topk` is set before every call,
        but it only matters on each program's first run, when it gets baked
        in. Under the MTP step-0 post-attention gather without the fused loop,
        steps 1+ likewise run their own program, traced without the gather.
        """
        assert self.draft_model is not None
        loop_model = self._mtp_loop_model
        if loop_model is None:
            return self.draft_model
        loop_step = step_idx > 0
        self._mtp_set_skip_topk(loop_step)
        return loop_model if loop_step else self.draft_model

    def _mtp_set_skip_topk(self, skip: bool) -> None:
        if self._share_mtp_indices:
            assert self.draft_model is not None
            self.draft_model.model.set_skip_topk(skip)

    def _mtp_compact_and_save(self, gather_indices: torch.Tensor) -> list[torch.Tensor]:
        """Compact step 0's per-token top-k rows to one row per request, then
        snapshot them.

        Step 0 writes one buffer row per token of the chunk's forward; the loop
        steps read row i for request i, so the rows have to be gathered to the
        front first. The snapshot is what makes this correct for more than one
        chunk: the buffer is a single tensor indexed by
        position-in-the-current-forward, so the *next* chunk's step-0 indexer
        write lands on exactly these rows and destroys them.
        """
        assert self.draft_model is not None
        self.draft_model.model.compact_topk_indices(gather_indices)
        num_rows = gather_indices.shape[0]
        return [buf[:num_rows].clone() for buf in self._mtp_topk_buffers]

    def _mtp_restore_topk(self, saved: list[torch.Tensor]) -> None:
        """Put this chunk's step-0 rows back at the front before its forward."""
        for buf, rows in zip(self._mtp_topk_buffers, saved):
            buf[: rows.shape[0]] = rows

    def _validate_kv_sharing_supported(self) -> None:
        """Fail closed on TPU paths that cannot honour `skip_kv_update`.

        A cross-model KV-sharing draft is only safe when the attention impl
        actually skips the KV write — otherwise its zeroed dummy K/V lands in
        the target's cache. Two TPU paths reject `skip_kv_update` when the
        kernel is built (`PallasAttentionBackendImpl._validate_pcp_streaming_
        support`, and the bundled block-major RPA); surface them here so the
        failure is a startup error instead of silent corruption or a
        mid-serving raise.
        """
        if envs.VLLM_TPU_BLOCK_MAJOR_KV:
            raise NotImplementedError(
                "Cross-model KV-sharing speculative decoding (Gemma-4 MTP) "
                "requires skip_kv_update, which the block-major KV bundle "
                "does not implement. Unset VLLM_TPU_BLOCK_MAJOR_KV."
            )
        if PcpStaticSupportValidator.from_vllm_config(self.vllm_config).enabled:
            raise NotImplementedError(
                "Cross-model KV-sharing speculative decoding (Gemma-4 MTP) "
                "requires skip_kv_update, which PCP streaming attention does "
                "not support."
            )

    def _setup_gemma4_kv_sharing(
        self,
        target_model,
        target_attn_layer_names: set[str],
    ) -> None:
        """Wire Q-only draft layers to read the target model's KV cache.

        Only draft layers that declare `is_kv_shared_layer` participate, so
        this is a silent no-op for an MTP draft that owns its KV cache
        (DeepSeek-V4). When any layer is wired, `self._draft_kv_shared` is set
        and every declaring layer must resolve to a target — a partially wired
        draft would write dummy K/V into the target's cache through its
        unwired layers, so an unresolvable layer raises rather than warns.
        """
        from collections import defaultdict

        # Decide participation before touching either config: this runs for
        # every `mtp` draft, and one that owns its KV cache must not be
        # required to expose a config shape it never needs.
        if not (
            hasattr(self.draft_model, "model")
            and hasattr(self.draft_model.model, "layers")
        ):
            return

        # The draft model declares whether its attention reads the target's
        # cache; `speculative_config.method` cannot distinguish Gemma-4 MTP
        # from DeepSeek-V4 MTP.
        shared_layers = [
            (idx, layer)
            for idx, layer in enumerate(self.draft_model.model.layers)
            if getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False)
        ]
        if not shared_layers:
            return
        self._validate_kv_sharing_supported()

        assert self.draft_model is not None
        draft_text_config = self.draft_model.config.get_text_config()
        target_text_config = target_model.config.get_text_config()
        target_layer_types = getattr(target_text_config, "layer_types", [])
        target_num_kv_shared = getattr(target_text_config, "num_kv_shared_layers", 0)
        num_non_shared = len(target_layer_types) - target_num_kv_shared
        type_to_target_indices: dict[str, list[int]] = defaultdict(list)
        for idx, lt in enumerate(target_layer_types[:num_non_shared]):
            type_to_target_indices[lt].append(idx)

        target_prefix = "model.layers"
        # MUST BE SORTED to prevent hash-seed desyncs across TP workers
        for name in sorted(target_attn_layer_names):
            if ".layers." in name:
                target_prefix = name.split(".layers.")[0] + ".layers"
                break

        draft_layer_types = getattr(draft_text_config, "layer_types", [])
        for draft_idx, layer in shared_layers:
            attn = getattr(layer.self_attn, "attn", None)
            if attn is None:
                raise RuntimeError(
                    f"Draft layer {draft_idx} declares is_kv_shared_layer but "
                    "exposes no self_attn.attn to wire; cannot guarantee it "
                    "will not write into the target's KV cache."
                )

            draft_layer_type = (
                draft_layer_types[draft_idx]
                if draft_idx < len(draft_layer_types)
                else "full_attention"
            )
            candidates = type_to_target_indices.get(draft_layer_type, [])
            if not candidates:
                raise RuntimeError(
                    f"No target layer of type '{draft_layer_type}' to share KV "
                    f"with for draft layer {draft_idx}. Leaving it unwired "
                    "would let it write dummy K/V into the target's cache."
                )

            # MTP layers share the LAST target layer of the matching attention type
            target_idx = candidates[-1]
            target_layer_name = f"{target_prefix}.{target_idx}.self_attn.attn"
            # Set BOTH: the module attribute is what get_kv_cache_spec reads to
            # skip allocating a cache for this layer, while the Pallas impl
            # derives `skip_kv_update` from its OWN copy (see
            # PallasAttentionBackendImpl.initialize_kernel / forward). Setting
            # only the module attr leaves the impl writing zeroed dummy K/V
            # over the target's real entries.
            attn.kv_sharing_target_layer_name = target_layer_name
            impl = getattr(attn, "impl", None)
            if impl is None:
                raise RuntimeError(
                    f"Draft attention layer {draft_idx} has no `impl`; cannot "
                    "enable skip_kv_update for cross-model KV sharing."
                )
            impl.kv_sharing_target_layer_name = target_layer_name

            logger.info(
                "Gemma4 MTP (TPU): draft layer %d (%s) -> %s",
                draft_idx,
                draft_layer_type,
                target_layer_name,
            )

        # Only now that every declaring layer is wired: the propose loop reads
        # this to freeze positions/seq_lens across draft steps.
        self._draft_kv_shared = True

    def _load_draft_model(self) -> None:
        logger.info("Loading %s draft model...", self.speculative_config.method)
        model_loader = get_model_loader(self.vllm_config.load_config)
        # Tag the draft compile with "eagle_head" so its torch.compile cache
        # lives in a separate prefix from the target's "backbone" prefix.
        draft_tp1_ctx = (
            _force_draft_tp1() if self._draft_replicated else contextlib.nullcontext()
        )
        # The draft shares the target's parallel config, so under
        # TPU_MOE_SEQUENCE_PARALLEL=1 its MoE layers are admitted to
        # sequence-parallel MoE too (main #1562); _forward_draft and the fused
        # loop then record each step's row form with sequence_parallel.
        self.draft_vllm_config = copy.copy(self.vllm_config)
        draft_model_config = copy.copy(self.speculative_config.draft_model_config)
        draft_model_config.runner_type = "draft"
        draft_compilation_config = copy.copy(self.vllm_config.compilation_config)
        draft_compilation_config.inductor_compile_config = copy.copy(
            self.vllm_config.compilation_config.inductor_compile_config
        )
        draft_compilation_config.inductor_compile_config["_vllm_model_tag"] = (
            "eagle_head"
        )
        self.draft_vllm_config.compilation_config = draft_compilation_config
        with (
            set_model_tag("eagle_head"),
            set_vllm_model_wrapper_context(
                mesh=self.runner.mesh, vllm_config=self.draft_vllm_config
            ),
            set_current_vllm_config(self.draft_vllm_config),
            draft_tp1_ctx,
        ):
            self.draft_model = model_loader.load_model(
                vllm_config=self.draft_vllm_config,
                model_config=draft_model_config,
            )
        # The same flag the Qwen3.5 / Qwen3-Next forwards gate their row
        # slicing on.
        modules = getattr(self.draft_model, "modules", None)
        self._draft_sequence_parallel = callable(modules) and any(
            getattr(m, "use_attn_reduce_scatter_for_moe", False) is True
            for m in modules()
        )

    def _validate_pcp_draft_model(self, draft_attn_layer_names: set[str]) -> None:
        """Validate draft KV-cache capabilities known only after model load."""
        kv_cache_specs = self.runner.get_kv_cache_spec()
        for layer_name in sorted(draft_attn_layer_names):
            spec = kv_cache_specs.get(layer_name)
            if not isinstance(spec, FullAttentionSpec):
                raise NotImplementedError(
                    "PCP MTP draft requires FullAttentionSpec for layer "
                    f"{layer_name}, got {type(spec).__name__}"
                )

    @staticmethod
    def _split_dsa_draft_layers(
        all_attn_layers: dict[str, AttentionLayerBase],
        draft_attn_layer_names: set[str],
    ) -> tuple[list[str], list[str]]:
        """Split a DSA draft's attention layers into (indexer K caches, MLA).

        Returns two empty lists for any draft with no lightning indexer, which
        is every non-DSA draft: eagle3, Qwen3.5 MTP, DFlash/DSpark, and a
        pre-V3.2 DeepSeek MTP.
        """
        try:
            from vllm.model_executor.layers.attention.mla_attention import MLAAttention
            from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        except ImportError:
            # A vLLM build without DSA cannot have loaded a DSA draft.
            return [], []

        indexer_names = sorted(
            name
            for name in draft_attn_layer_names
            if isinstance(all_attn_layers.get(name), DeepseekV32IndexerCache)
        )
        if not indexer_names:
            return [], []
        mla_names = sorted(
            name
            for name in draft_attn_layer_names
            if isinstance(all_attn_layers.get(name), MLAAttention)
        )
        return indexer_names, mla_names

    def _validate_dsa_draft(
        self,
        all_attn_layers: dict[str, AttentionLayerBase],
        draft_attn_layer_names: set[str],
    ) -> None:
        """Prove a sparse-attention (DSA) draft's KV caches are usable.

        A GLM-5.2 / DeepSeek-V3.2 MTP draft owns two KV caches per MTP layer:
        the MLA latent cache and the lightning indexer's fp8 K cache. Both are
        `AttentionLayerBase`s that register themselves in the *shared*
        `static_forward_context` -- the draft's compilation config is a shallow
        copy of the target's, so the two models write into one registry -- and
        both therefore reach `TPUModelRunner.get_kv_cache_spec` and, via the
        registry diff above, `_draft_attn_layer_names`. Registration needs no
        new code; what it needs is a check that the generic path really covered
        them, because every way it can fail is silent.

        A layer whose module type falls through the spec walk's trailing
        `else: continue` gets no `KVCacheSpec`. The KV cache manager then
        allocates nothing for it, its `kv_cache` stays a 0-element tensor, and
        `VllmTPUSparseAttnIndexer.forward_oot` returns the top-k buffer without
        having written it. The draft attends to whatever that buffer held --
        all -1 once `_init_mtp_index_sharing` seeds it, i.e. nothing at all --
        so every proposal comes out of an empty attend and the acceptance rate
        collapses to roughly what an unconditioned draft would get. No
        exception is raised anywhere along that path, which is exactly why it
        is worth one check at load.
        """
        indexer_names, mla_names = self._split_dsa_draft_layers(
            all_attn_layers, draft_attn_layer_names
        )
        if not indexer_names:
            return

        if not mla_names:
            raise RuntimeError(
                "The draft model registered a DSA lightning indexer "
                f"({indexer_names[0]}) but no MLA attention layer. The "
                "indexer only scores and selects KV positions; sparse MLA "
                "is what reads them. A draft with one and not the other "
                "cannot run."
            )

        # Only a draft replicated *below* the target's TP is a problem. At
        # target_tp == 1 both flags are set and nothing is sharded either way.
        if self._draft_replicated and not self._draft_tp_matches_target:
            target_tp = self.vllm_config.parallel_config.tensor_parallel_size
            raise NotImplementedError(
                "A DSA (sparse-MLA) draft cannot run replicated "
                "(draft_tensor_parallel_size=1). Its sparse-MLA and "
                "lightning-indexer Pallas ops are built against the runner's "
                "full TP mesh, and unlike the dense RPA path -- where "
                "`_initialize_attention_kernels` swaps in a LOCAL "
                "(non-shard_map) kernel for a replicated draft -- there is no "
                "single-device variant to relocate them to, so the kernels "
                "would shard head-parallel work over weights loaded "
                "unsharded. Set draft_tensor_parallel_size to the target's "
                f"tensor_parallel_size ({target_tp})."
            )

        kv_cache_specs = self.runner.get_kv_cache_spec()
        unregistered = [
            name for name in (*mla_names, *indexer_names) if name not in kv_cache_specs
        ]
        if unregistered:
            missing_types = sorted(
                {type(all_attn_layers[name]).__name__ for name in unregistered}
            )
            raise RuntimeError(
                "These DSA draft layers got no KVCacheSpec, so no KV cache "
                "would be allocated for them and the draft would silently "
                f"attend to nothing: {unregistered}. "
                "`TPUModelRunner.get_kv_cache_spec` needs a branch for their "
                f"module types ({missing_types})."
            )

        logger.info(
            "DSA draft KV caches registered: %d sparse-MLA layer(s) %s, "
            "%d lightning-indexer K cache(s) %s.",
            len(mla_names),
            mla_names,
            len(indexer_names),
            indexer_names,
        )

    @_static_shape_args
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_propose_token(self, hidden: torch.Tensor) -> torch.Tensor:
        # Draft lm-head + greedy argmax, wrapped in one torch.compile region
        # keyed on input shape only (mirrors the target's compute_selected_logits).
        # Run raw, compute_logits + argmax + the int32 cast are eager ops the
        # torch-tpu DEFER_AND_FUSE path fuses with per-step-varying neighbors in
        # the K-step propose loop -> a fresh fused program per context. Enclosing
        # them makes a fixed, bucketed program per [n, hidden] shape.
        assert self.draft_model is not None
        d2t = getattr(self.draft_model, "draft_id_to_target_id", None)
        if d2t is None:
            owner = self._draft_greedy_head_owner
            if owner is not None:
                # TPU_GREEDY_TOP_TOKENS: the vocab-parallel argmax, over the
                # quantized draft head proxy when TPU_DRAFT_LM_HEAD_QUANT
                # set one.
                tokens = owner.logits_processor.get_top_tokens(owner.lm_head, hidden)
                return tokens.to(torch.int32)
            # No draft->target mapping table. Upstream compute_logits then asserts
            # the head is already target-vocab-width, so plain argmax yields
            # target ids.
            return (
                self.draft_model.compute_logits(hidden).argmax(dim=-1).to(torch.int32)
            )
        draft_id = self.draft_model.logits_processor(
            self.draft_model.lm_head, hidden
        ).argmax(dim=-1)
        return (draft_id + d2t[draft_id]).to(torch.int32)

    @_static_shape_args
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_gather_carries(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        last_hidden: torch.Tensor,
        indices: torch.Tensor,
        step0_post_attn_gathered: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # First-pass per-request carries: gather the last-token hidden, position,
        # and pre-lm-head hidden in one compiled region. Run raw (advanced
        # indexing), these three gathers are eager ops the DEFER_AND_FUSE path
        # fuses with the step's seq_lens/positions arithmetic into a per-context
        # program. index_select is value-identical to hidden[indices].
        pos_carry = (
            torch.index_select(positions, 1, indices)
            if positions.ndim == 2
            else torch.index_select(positions, 0, indices)
        )
        if step0_post_attn_gathered:
            n = indices.shape[0]
            h_carry = hidden if hidden.shape[0] == n else hidden[:n]
            lh_carry = (
                h_carry
                if last_hidden is hidden
                else (last_hidden if last_hidden.shape[0] == n else last_hidden[:n])
            )
        else:
            h_carry = torch.index_select(hidden, 0, indices)
            lh_carry = (
                h_carry
                if last_hidden is hidden
                else torch.index_select(last_hidden, 0, indices)
            )
        return (
            h_carry,
            pos_carry,
            lh_carry,
        )

    @_static_shape_args
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_post_step0_fused(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        indices: torch.Tensor,
        base_seq_lens: torch.Tensor,
        num_rejected_dev: torch.Tensor,
        step0_post_attn_gathered: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Fused post-step-0 bridge for the fused MTP loop when last_hidden is
        # hidden and SequenceLayoutKind is ALL:
        #   1. Gather hidden_carry [p, H] and pos_carry [p] (or [3, p])
        #   2. Advance pos_step1 = pos_carry + 1 (if advance_draft_positions)
        #   3. Compute seq_lens_step1 = base_seq_lens + sl_delta - num_rejected_dev
        # Step 0's token proposal (lm_head + argmax / get_top_tokens) runs at
        # the start of _MtpFusedLoop.forward (built with propose_step0) so this
        # bridge has zero matmuls or TP collectives.
        pos_carry = (
            torch.index_select(positions, 1, indices)
            if positions.ndim == 2
            else torch.index_select(positions, 0, indices)
        )
        if step0_post_attn_gathered:
            hidden_carry = (
                hidden
                if hidden.shape[0] == indices.shape[0]
                else hidden[: indices.shape[0]]
            )
        else:
            hidden_carry = torch.index_select(hidden, 0, indices)
        pos_step1 = pos_carry + 1 if self.advance_draft_positions else pos_carry
        sl_delta = 1 if self.advance_draft_positions else 0
        seq_lens_step1 = base_seq_lens + sl_delta - num_rejected_dev
        return hidden_carry, pos_step1, seq_lens_step1

    @_static_shape_args
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_combine_hidden_states(
        self, aux0: torch.Tensor, aux1: torch.Tensor, aux2: torch.Tensor
    ) -> torch.Tensor:
        # Eagle3 always exposes exactly 3 aux hidden states. Fold the torch.cat
        # and the combine Linear into ONE compiled region keyed on
        # [num_tokens, aux_w] so torch-tpu's eager DEFER_AND_FUSE can't (a) split
        # the cat into a standalone program, (b) emit a distinct cat+mm grouping
        # per live dispatch context, or (c) fuse the combine mm forward into the
        # downstream draft_input_ids seed scatter. Same pattern as
        # _draft_propose_token / _draft_gather_carries.
        assert self.draft_model is not None
        return self.draft_model.combine_hidden_states(
            torch.cat((aux0, aux1, aux2), dim=-1)
        )

    @_static_shape_args
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_seed_input_ids(
        self,
        input_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Functional (no in-place) eagle3 first-pass input prep so fullgraph
        # holds: (1) left-shift by one (mirrors set_inputs_first_pass), (2)
        # scatter the per-request seed token at last_token_indices. Run raw, the
        # clone+shift and the index assignment are eager ops the DEFER_AND_FUSE
        # path fuses with the combine mm / metadata per context. Wrapping in one
        # compiled region keyed on [len(input_ids), len(indices)] removes them
        # from the per-context re-fusion. torch.cat((ids[1:], ids[-1:])) is
        # value-identical to clone()+[:-1]=ids[1:] (last slot keeps its own
        # value), and index_put is the functional twin of ids[idx] = val.
        shifted = input_ids
        if input_ids.shape[0] > 1:
            shifted = torch.cat((input_ids[1:], input_ids[-1:]), dim=0)
        return shifted.index_put(
            (last_token_indices,), next_token_ids.to(input_ids.dtype)
        )

    @_static_shape_args
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_async_prepare_inputs(
        self,
        input_ids: torch.Tensor,
        next_tokens_device: torch.Tensor,
        draft_lengths: torch.Tensor,
        query_start_loc: torch.Tensor,
        pad_req_idx: torch.Tensor,
        ov_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Fused on-device async first-pass input prep at bucketed shapes:
        #   input_ids: [T]
        #   next_tokens_device: [B, K+1]
        #   draft_lengths: [B] (padded rows [num_reqs:B] are 0)
        #   query_start_loc: [knr + 1]
        #   pad_req_idx: [p] (clamped to num_reqs - 1 so the padded tail
        #     repeats the last valid request idempotently)
        #   ov_tokens: optional [B] int32 override tokens for partial-prefill
        #     requests (-1 for non-overridden requests).
        # Folds the 11 eager tensor ops + unbucketed seed scatter + gather_indices
        # pad + num_rejected pad into a single bucketed compiled graph.
        B = next_tokens_device.shape[0]
        knr = query_start_loc.shape[0] - 1
        eff_b = min(B, knr)
        nt = next_tokens_device[:eff_b]
        num_valid = (nt != INVALID_TOKEN_ID).sum(dim=1).clamp(min=1)
        last_valid_col = num_valid - 1
        next_token_ids_b = (
            nt.gather(1, last_valid_col.unsqueeze(1)).squeeze(1).to(input_ids.dtype)
        )
        if ov_tokens is not None:
            ov_b = ov_tokens[:eff_b].to(input_ids.dtype)
            next_token_ids_b = torch.where(ov_b >= 0, ov_b, next_token_ids_b)
        num_draft_dev = draft_lengths[:eff_b].to(torch.int64)
        accepted_drafts = torch.minimum((num_valid - 1).clamp(min=0), num_draft_dev)
        num_rejected_b = (num_draft_dev - accepted_drafts).to(torch.int32)
        qsl_end = query_start_loc[1 : eff_b + 1].to(torch.int64)
        last_token_indices_b = qsl_end - 1 - num_rejected_b.to(torch.int64)

        gather_indices = torch.index_select(last_token_indices_b, 0, pad_req_idx)
        seed_token_ids = torch.index_select(next_token_ids_b, 0, pad_req_idx)

        shifted = input_ids
        if input_ids.shape[0] > 1:
            shifted = torch.cat((input_ids[1:], input_ids[-1:]), dim=0)
        draft_input_ids = shifted.index_put((gather_indices,), seed_token_ids)

        num_rejected_out = (
            torch.nn.functional.pad(num_rejected_b, (0, knr - eff_b))
            if eff_b < knr
            else num_rejected_b
        )
        return draft_input_ids, gather_indices, num_rejected_out

    @_static_shape_args
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_loop_seq_lens(
        self,
        base_seq_lens: torch.Tensor,
        num_rejected_dev: torch.Tensor,
        delta: int,
    ) -> torch.Tensor:
        return base_seq_lens + delta - num_rejected_dev

    @_static_shape_args
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_add_delta(
        self,
        x: torch.Tensor,
        delta: int,
    ) -> torch.Tensor:
        return x + delta

    @_static_shape_args
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_stack_tokens(
        self,
        *toks: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(toks, dim=1)

    def propose(
        self,
        sampled_token_ids: list[list[int]] | None,
        discard_sampled_tokens_req_indices: list[int],
        num_rejected_tokens_np: np.ndarray | None,
        scheduler_output,
        return_device: bool = False,
        next_tokens_per_chunk: list[torch.Tensor] | None = None,
        device_seed: torch.Tensor | None = None,
    ) -> list[list[int]] | torch.Tensor:
        """Generate K draft tokens per request via the eagle3 draft model.

        Args:
            sampled_token_ids: post-rejection sampled tokens per request.
            discard_sampled_tokens_req_indices: request indices whose sampled
                tokens should be discarded (partial-prefill case).
            num_rejected_tokens_np: per-request count of draft tokens that
                were rejected this step.
            scheduler_output: vLLM SchedulerOutput; used for partial-prefill
                next-token lookup via scheduler_output.num_scheduled_tokens.
            return_device: when True, return the raw ``[num_reqs, K]`` int32
                device tensor instead of host nested lists. The async path uses
                this to pack drafts into the next-step substitution source
                without a D2H on the critical path; the sync path keeps the
                list return.
            next_tokens_per_chunk: spec verify step (sync and async) —
                per-chunk on-device rejection-sampler output, passed to each
                chunk's ``_prepare_draft_inputs`` so the draft seed is read
                from device instead of the host ``sampled_token_ids``.
            device_seed: prefill / non-spec step (sync and async) —
                ``[total_reqs]`` device tensor of the just-sampled tokens.

        Returns (sync) a list of length num_reqs, each inner list holding K
        draft tokens; or (return_device) the ``[num_reqs, K]`` device tensor.
        """
        runner = self.runner
        self._draft_md_cache.clear()
        num_reqs = runner.input_batch.num_reqs
        if num_reqs == 0:
            return []

        K = self.speculative_config.num_speculative_tokens

        chunks = self.draft_chunks
        if not chunks:
            raise RuntimeError(
                f"{self.__class__.__name__}.propose() called but no draft chunks were "
                "captured (aux_hidden_states was None for every target chunk "
                "in eagle3, or no hidden_states were captured). "
                "Ensure target verify step correctly recorded state."
            )
        # First pass, per chunk — mirrors the target's chunked verify forward.
        # _prepare_draft_inputs returns last_token_indices padded to the chunk's
        # loop bucket p, so the per-request gathers/carries below are
        # bucket-shaped (constant) and don't recompile as num_reqs shrinks. Each
        # chunk's drafts stay [p]-shaped through the loop; we slice to the real
        # num_reqs only once at the end.
        hidden_carry_per_chunk = []
        positions_carry_per_chunk = []
        rejected_per_chunk = []
        # The same counts as host arrays for the draft metadata builds; None
        # when async, where they stay on device.
        rejected_np_per_chunk: list[np.ndarray | None] = []
        uses_aux_hidden_state = self._draft_uses_aux_hidden_state()
        draft_tokens_per_chunk = []
        is_async = next_tokens_per_chunk is not None
        # IndexShare: each chunk's step-0 top-k rows, saved right after its
        # compaction (see _mtp_compact_and_save).
        saved_topk_per_chunk: list[list[torch.Tensor]] = []
        # Per chunk, (step-1 positions, step-1 seq_lens, padded rejected
        # counts) from the post-step-0 bridge, or None without it.
        precomputed_step1_per_chunk: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None
        ] = []
        # Per chunk, whether the fused loop program runs its steps: it then
        # also proposes step 0's token from the hidden carry, which is right
        # only when the carry is the lm-head input (last_hidden is hidden) and
        # every rank holds the request's rows (ALL layout).
        fused_per_chunk: list[bool] = []
        use_fused_loop = (
            K > 1
            and self._mtp_fused_loop is not None
            and not runner._dp_lockstep_enabled()
        )
        # The post-step-0 bridge (_draft_post_step0_fused) replaces the
        # separate carry / position / seq_lens helpers, so it is skipped when
        # a test double stands in for one of them.
        inst_dict = getattr(self, "__dict__", {})
        can_use_post_step0_fused = (
            use_fused_loop
            and not self._share_mtp_indices
            and "_draft_gather_carries" not in inst_dict
            and "_draft_propose_token" not in inst_dict
            and "_draft_add_delta" not in inst_dict
            and "_draft_loop_seq_lens" not in inst_dict
        )
        for ci, chunk in enumerate(chunks):
            if uses_aux_hidden_state:
                with set_model_tag("eagle_head"):
                    target_hidden_states = self._draft_combine_hidden_states(
                        *chunk.aux_hidden_states
                    )
            else:
                assert chunk.hidden_states is not None, (
                    "DraftChunkInputs.hidden_states is required when the "
                    "draft does not use aux hidden states, but no plain "
                    "hidden state was captured for this chunk."
                )
                if self.speculative_config.method == "eagle3":
                    assert self.draft_model is not None
                    with set_model_tag("eagle_head"):
                        target_hidden_states = self.draft_model.combine_hidden_states(
                            chunk.hidden_states
                        )
                else:
                    # MTP: does not use combine_hidden_states
                    target_hidden_states = chunk.hidden_states

            (input_ids, positions, last_token_indices, num_rejected) = (
                self._prepare_draft_inputs(
                    chunk,
                    sampled_token_ids,
                    discard_sampled_tokens_req_indices,
                    num_rejected_tokens_np,
                    scheduler_output,
                    next_tokens_device=(
                        None
                        if next_tokens_per_chunk is None
                        else next_tokens_per_chunk[ci]
                    ),
                    device_seed=device_seed,
                )
            )
            rejected_per_chunk.append(num_rejected)
            rejected_np_per_chunk.append(
                None if isinstance(num_rejected, torch.Tensor) else num_rejected
            )
            is_all_layout = (
                getattr(chunk.sequence_layout_plan, "kind", None)
                is SequenceLayoutKind.ALL
            )
            if is_all_layout:
                owner_mask = None
                safe_gather_indices = last_token_indices
            else:
                owner_mask = last_token_indices.ge(0)
                safe_gather_indices = last_token_indices.clamp_min(0)
            step0_gather_indices: torch.Tensor | None = None
            if self._use_mtp_step0_post_attn_gather(
                chunk, can_use_post_step0_fused=use_fused_loop
            ):
                max_loop_p = self._loop_bucket(runner.max_num_reqs)
                step0_gather_indices = _maybe_pad_dim0(safe_gather_indices, max_loop_p)
            # Passed only when set, so a test double of _forward_draft with
            # the pre-gather signature still works.
            draft_fwd_kwargs = {}
            if step0_gather_indices is not None:
                draft_fwd_kwargs["step0_gather_indices"] = step0_gather_indices
            last_hidden, hidden = self._forward_draft(
                chunk=chunk,
                input_ids=input_ids,
                positions=positions,
                target_hidden_states=target_hidden_states,
                step_idx=0,
                seq_lens_delta=0,
                num_rejected_np=rejected_np_per_chunk[ci],
                num_tokens_padded=input_ids.shape[0],
                **draft_fwd_kwargs,
            )
            # [p, h] / [p] carries + first-pass draft token, computed in compiled
            # regions (the three gathers, then lm-head+argmax) so torch-tpu's
            # eager DEFER_AND_FUSE path can't fuse them into per-context programs.
            if self._share_mtp_indices:
                # Clamped, not raw: a PCP-localized index is -1 for a row this
                # rank does not own, and -1 would wrap to the buffer's last row.
                saved_topk_per_chunk.append(
                    self._mtp_compact_and_save(safe_gather_indices)
                )
            fused_chunk = use_fused_loop and is_all_layout and last_hidden is hidden
            fused_per_chunk.append(fused_chunk)
            if (
                fused_chunk
                and can_use_post_step0_fused
                and getattr(hidden, "device", None) is not None
                and hidden.device.type == "tpu"
            ):
                p = self._loop_bucket(chunk.num_reqs)
                knr = (
                    runner.num_reqs_max_model_len
                    if chunk.attn_ctx.use_max_model_len
                    else runner.num_reqs_most_model_len
                )
                if is_async:
                    rej_dev = (
                        _maybe_pad_dim0(num_rejected, knr)
                        if num_rejected is not None
                        else None
                    )
                elif num_rejected is not None and np.any(num_rejected):
                    padded_nr = np.zeros(knr, dtype=np.int32)
                    padded_nr[: chunk.num_reqs] = num_rejected.astype(
                        np.int32, copy=False
                    )
                    rej_dev = torch.from_numpy(padded_nr).to(runner.device)
                else:
                    rej_dev = None
                if rej_dev is not None:
                    rej_dev_for_fused = rej_dev
                else:
                    z_key = (knr, str(runner.device))
                    rej_dev_for_fused = self._zero_rej_dev_cache.get(z_key)
                    if rej_dev_for_fused is None:
                        rej_dev_for_fused = torch.from_numpy(
                            np.zeros(knr, dtype=np.int32)
                        ).to(runner.device)
                        self._zero_rej_dev_cache[z_key] = rej_dev_for_fused
                safe_gather_indices_p = _maybe_pad_dim0(safe_gather_indices, p)
                with set_model_tag("eagle_head"):
                    hidden_carry, pos_step1, seq_lens_step1 = (
                        self._draft_post_step0_fused(
                            hidden,
                            positions,
                            safe_gather_indices_p,
                            chunk.attn_ctx.seq_lens,
                            rej_dev_for_fused,
                            step0_gather_indices is not None,
                        )
                    )
                hidden_carry_per_chunk.append(hidden_carry)
                positions_carry_per_chunk.append(pos_step1)
                draft_tokens_per_chunk.append([])
                precomputed_step1_per_chunk.append((pos_step1, seq_lens_step1, rej_dev))
            else:
                gather_args = (
                    (hidden, positions, last_hidden, safe_gather_indices, True)
                    if step0_gather_indices is not None
                    else (hidden, positions, last_hidden, safe_gather_indices)
                )
                hidden_carry, positions_carry, last_hidden_carry = (
                    self._draft_gather_carries(*gather_args)
                )
                last_hidden_carry = (
                    chunk.sequence_layout_plan.aggregate_request_aligned_tensor(
                        last_hidden_carry, owner_mask
                    )
                )
                hidden_carry_per_chunk.append(hidden_carry)
                positions_carry_per_chunk.append(positions_carry)
                if fused_chunk:
                    # The fused loop proposes step 0's token itself.
                    draft_tokens_per_chunk.append([])
                else:
                    with set_model_tag("eagle_head"):
                        draft_tokens_per_chunk.append(
                            [self._draft_propose_token(last_hidden_carry)]
                        )
                precomputed_step1_per_chunk.append(None)

        if K > 1:
            # Loop steps: uniform decode shape, one query per request. The
            # carries are already padded to the chunk's bucket p, so the loop
            # draft forward hits a precompiled trace with no per-num_reqs
            # recompile.
            padded_nr_per_chunk = [self._loop_bucket(c.num_reqs) for c in chunks]
            loop_hidden = [
                _maybe_pad_dim0(h, p)
                for h, p in zip(hidden_carry_per_chunk, padded_nr_per_chunk)
            ]
            loop_positions = [
                _maybe_pad_dim1(pos, p)
                if (runner.uses_mrope and pos.ndim == 2)
                else _maybe_pad_dim0(pos, p)
                for pos, p in zip(positions_carry_per_chunk, padded_nr_per_chunk)
            ]
            # Per-chunk rejected count padded to kernel_num_reqs so the loop's
            # seq_lens subtraction is full-length (constant) — no per-num_reqs
            # recompile. Sync pads on the host (free, reused across the K-1 loop
            # steps); async pads the device tensor.
            rejected_dev_per_chunk = []
            for ci_idx, (c, nr) in enumerate(zip(chunks, rejected_per_chunk)):
                pre_step1 = precomputed_step1_per_chunk[ci_idx]
                if pre_step1 is not None:
                    rejected_dev_per_chunk.append(pre_step1[2])
                    continue
                knr = (
                    runner.num_reqs_max_model_len
                    if c.attn_ctx.use_max_model_len
                    else runner.num_reqs_most_model_len
                )
                assert knr is not None
                if is_async:
                    rejected_dev_per_chunk.append(
                        _maybe_pad_dim0(nr, knr) if nr is not None else None
                    )
                elif nr is not None and np.any(nr):
                    padded_nr = np.zeros(knr, dtype=np.int32)
                    padded_nr[: c.num_reqs] = nr.astype(np.int32, copy=False)
                    rejected_dev_per_chunk.append(
                        torch.from_numpy(padded_nr).to(runner.device)
                    )
                else:
                    rejected_dev_per_chunk.append(None)
            # Loop-step query_start_loc and request_distribution depend only on
            # num_reqs (not the step), so build them once per chunk instead of
            # rebuilding (with an H2D) on every one of the K-1 steps.
            loop_qsl_per_chunk = []
            loop_reqdist_per_chunk = []
            for c in chunks:
                knr = (
                    runner.num_reqs_max_model_len
                    if c.attn_ctx.use_max_model_len
                    else runner.num_reqs_most_model_len
                )
                assert knr is not None
                # Read-only on device, so cached across propose calls.
                loop_key = (knr, c.num_reqs, str(runner.device))
                cached_loop = self._loop_qsl_reqdist_cache.get(loop_key)
                if cached_loop is None:
                    qsl_np = np.minimum(np.arange(knr + 1, dtype=np.int32), c.num_reqs)
                    cached_loop = (
                        torch.from_numpy(qsl_np).to(runner.device),
                        torch.tensor(
                            [c.num_reqs] * 3,
                            dtype=torch.int32,
                            device=runner.device,
                        ),
                    )
                    self._loop_qsl_reqdist_cache[loop_key] = cached_loop
                loop_qsl_per_chunk.append(cached_loop[0])
                loop_reqdist_per_chunk.append(cached_loop[1])

            def run_draft_step(ci: int, step: int) -> None:
                """One separate draft step for chunk `ci`: a draft forward
                and an lm-head + argmax, each its own dispatch."""
                chunk = chunks[ci]
                if self.advance_draft_positions:
                    pos = loop_positions[ci]
                    loop_positions[ci] = (
                        self._draft_add_delta(pos, 1)
                        if pos.device.type == "tpu"
                        else pos + 1
                    )
                # Prev step's [p] tokens feed directly — already bucketed.
                loop_input_ids = draft_tokens_per_chunk[ci][-1]
                last_hidden, hidden = self._forward_draft(
                    chunk=chunk,
                    input_ids=loop_input_ids,
                    positions=loop_positions[ci],
                    target_hidden_states=loop_hidden[ci],
                    step_idx=step,
                    seq_lens_delta=(step if self.advance_draft_positions else 0),
                    num_rejected_np=rejected_np_per_chunk[ci],
                    num_tokens_padded=padded_nr_per_chunk[ci],
                    num_rejected_dev=rejected_dev_per_chunk[ci],
                    loop_query_start_loc=loop_qsl_per_chunk[ci],
                    loop_request_distribution=loop_reqdist_per_chunk[ci],
                )
                with set_model_tag("eagle_head"):
                    draft_tokens_per_chunk[ci].append(
                        self._draft_propose_token(last_hidden)
                    )
                loop_hidden[ci] = hidden

            if use_fused_loop:
                # Fused: each chunk's step-0 token + steps 1..K-1 in one
                # dispatch. Chunk-major is fine here: the separate path is
                # step-major only for IndexShare's shared top-k buffer, which
                # the fused program never runs with.
                for ci, chunk in enumerate(chunks):
                    pre_step1 = precomputed_step1_per_chunk[ci]
                    fused = None
                    if fused_per_chunk[ci]:
                        if pre_step1 is not None:
                            pos_step1, seq_lens_step1, _ = pre_step1
                        else:
                            pos = loop_positions[ci]
                            pos_step1 = (
                                self._draft_add_delta(pos, 1)
                                if pos.device.type == "tpu"
                                else pos + 1
                            )
                            seq_lens_step1 = None
                        fused = self._run_mtp_fused_loop(
                            chunk=chunk,
                            positions=pos_step1,
                            hidden_states=loop_hidden[ci],
                            num_rejected_np=rejected_np_per_chunk[ci],
                            num_rejected_dev=rejected_dev_per_chunk[ci],
                            loop_query_start_loc=loop_qsl_per_chunk[ci],
                            loop_request_distribution=loop_reqdist_per_chunk[ci],
                            precomputed_seq_lens=seq_lens_step1,
                        )
                    if fused is not None:
                        # The whole [p, K] draft block.
                        draft_tokens_per_chunk[ci] = [fused]
                        continue
                    # This chunk's metadata doesn't fit the program, or its
                    # step-0 token cannot come from the hidden carry: run the
                    # steps separately.
                    if not draft_tokens_per_chunk[ci]:
                        with set_model_tag("eagle_head"):
                            draft_tokens_per_chunk[ci].append(
                                self._draft_propose_token(loop_hidden[ci])
                            )
                    if pre_step1 is not None and self.advance_draft_positions:
                        # The bridge advanced the carry to step 1;
                        # run_draft_step advances it itself.
                        loop_positions[ci] = pre_step1[0] - 1
                    for step in range(1, K):
                        run_draft_step(ci, step)
            else:
                # Separate: one draft forward and one lm-head per step.
                for step in range(1, K):
                    for ci in range(len(chunks)):
                        if self._share_mtp_indices:
                            # The loop is step-major, so chunks interleave and
                            # each one finds the buffer holding its
                            # predecessor's rows. Restore unconditionally: one
                            # rule, no dependence on the chunk count. With a
                            # single chunk this is a self-copy of a few
                            # hundred KiB.
                            self._mtp_restore_topk(saved_topk_per_chunk[ci])
                        run_draft_step(ci, step)

        # EP-DP lockstep: pad this rank's draft-forward count up to the
        # step's coordinated chunk bound so ranks with fewer chunks still
        # run the same number of draft forwards as their peers.
        if runner._dp_lockstep_enabled():
            extra_chunks = runner._dp_step_num_chunks - len(chunks)
            if extra_chunks > 0:
                self.run_dp_dummy_draft(extra_chunks)

        # Assemble [total_num_reqs, K]. Each chunk's tokens are padded to p; slice
        # to the real num_reqs at the end. Sync slices on the host (the [p, K] D2H
        # is constant-shape, so no recompile); async keeps the device slice
        # (matches prior behaviour for the substitution source).
        per_chunk_stacked = [
            (
                toks[0]
                if len(toks) == 1 and toks[0].ndim == 2
                else self._draft_stack_tokens(*toks)
                if toks and toks[0].device.type == "tpu"
                else torch.stack(toks, dim=1)
            )
            for toks in draft_tokens_per_chunk
        ]
        self._last_padded_drafts_per_chunk = per_chunk_stacked
        skip_cat = self._skip_next_multi_chunk_draft_cat
        self._skip_next_multi_chunk_draft_cat = False
        if return_device:
            if len(per_chunk_stacked) == 1 or skip_cat:
                out = per_chunk_stacked[0][: chunks[0].num_reqs]
            else:
                out = torch.cat(
                    [s[: c.num_reqs] for s, c in zip(per_chunk_stacked, chunks)], dim=0
                )
            self._last_propose_device_out = out
            return out
        result = []
        for s, c in zip(per_chunk_stacked, chunks):
            result.extend(s.cpu().tolist()[: c.num_reqs])
        return result

    def _prepare_draft_inputs(
        self,
        chunk: DraftChunkInputs,
        sampled_token_ids: list[list[int]] | None,
        discard_sampled_tokens_req_indices: list[int],
        num_rejected_tokens_np: np.ndarray | None,
        scheduler_output,
        next_tokens_device: torch.Tensor | None = None,
        device_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray | torch.Tensor]:
        """Build draft first-pass inputs for one chunk.

        All token/request indices are chunk-local; the batch-level inputs
        (sampled_token_ids, num_rejected_tokens_np, discard indices) are
        sliced/offset by chunk.start_index.

        Args:
            next_tokens_device: async path only — the chunk's on-device
                rejection-sampler output ``[>=num_reqs, K+1]`` (accepted prefix +
                bonus, padded with INVALID_TOKEN_ID). When given, each request's
                seed token is gathered from it on-device instead of from the
                host ``sampled_token_ids`` list (which the async path lacks).
            device_seed: prefill / non-spec async bootstrap only —
                ``[total_reqs]`` device tensor of the just-sampled tokens
                (sliced ``[start:start+num_reqs]`` per chunk).

        Returns:
            input_ids: [padded_chunk_tokens] shifted + bonus-patched
            positions: [padded_chunk_tokens] chunk positions
            last_token_indices: [p] padded to the chunk's loop bucket; the
                first num_reqs entries hold the end of each request's accepted
                prefix, the padded tail holds filler indices (sliced off in
                propose()).
            num_rejected_np: [num_reqs] clamped per-request rejection count
        """
        runner = self.runner
        num_reqs = chunk.num_reqs
        start = chunk.start_index
        from vllm_torchtpu.runner.tpu_runner import _h2d_temp

        # The async (next_tokens_device) path recomputes last_token_indices and
        # num_rejected fully on-device below; only compute host arrays when
        # next_tokens_device is None.
        if next_tokens_device is None:
            query_start_loc_np = chunk.query_start_loc_np[: num_reqs + 1]
            orig_num_tokens_per_req = query_start_loc_np[1:] - query_start_loc_np[:-1]
            if num_rejected_tokens_np is None:
                num_rejected_np = np.zeros(num_reqs, dtype=np.int32)
            else:
                num_rejected_np = num_rejected_tokens_np[
                    start : start + num_reqs
                ].astype(np.int32, copy=False)
                # Clamp to orig-1, not orig: a request always retains at least
                # the bonus token, so at most orig-1 drafts can be rejected.
                num_rejected_np = np.minimum(
                    num_rejected_np, orig_num_tokens_per_req - 1
                )
            last_token_indices_np = (
                query_start_loc_np[1 : num_reqs + 1].astype(np.int64)
                - 1
                - num_rejected_np.astype(np.int64)
            )

        discard_set = (
            set(discard_sampled_tokens_req_indices)
            if discard_sampled_tokens_req_indices
            else ()
        )
        p = self._loop_bucket(num_reqs)
        if next_tokens_device is not None:
            # Async path: derive the seed token, the seed *position*
            # (last_token_indices) and the rejected count from the on-device
            # rejection-sampler output instead of the host num_rejected /
            # sampled_token_ids — avoiding a D2H on the critical path.
            ov_i = (
                [i for i in range(num_reqs) if start + i in discard_set]
                if discard_set
                else []
            )
            B = next_tokens_device.shape[0]
            if chunk.draft_lengths is not None:
                draft_lengths_dev = chunk.draft_lengths
            else:
                req_ids = runner.input_batch.req_ids[start : start + num_reqs]
                num_draft_np = np.zeros(B, dtype=np.int32)
                num_draft_np[:num_reqs] = [
                    len(scheduler_output.scheduled_spec_decode_tokens.get(rid, ()))
                    for rid in req_ids
                ]
                draft_lengths_dev = _h2d_temp(runner, torch.from_numpy(num_draft_np))
            pad_key = (p, num_reqs, str(runner.device))
            pad_req_idx = self._async_pad_req_idx_cache.get(pad_key)
            if pad_req_idx is None:
                pad_req_idx_np = np.minimum(np.arange(p, dtype=np.int64), num_reqs - 1)
                pad_req_idx = torch.from_numpy(pad_req_idx_np).to(runner.device)
                self._async_pad_req_idx_cache[pad_key] = pad_req_idx
            ov_tokens_dev = None
            if ov_i:
                req_ids = runner.input_batch.req_ids[start : start + num_reqs]
                ov_np = np.full(B, -1, dtype=np.int32)
                for i in ov_i:
                    req_state = runner.requests[req_ids[i]]
                    seq_len = (
                        req_state.num_computed_tokens
                        + scheduler_output.num_scheduled_tokens[req_ids[i]]
                    )
                    ov_np[i] = req_state.get_token_id(seq_len)
                ov_tokens_dev = _h2d_temp(runner, torch.from_numpy(ov_np))
            draft_input_ids, gather_indices, num_rejected_out = (
                self._draft_async_prepare_inputs(
                    chunk.input_ids,
                    next_tokens_device,
                    draft_lengths_dev,
                    chunk.attn_ctx.query_start_loc,
                    pad_req_idx,
                    ov_tokens_dev,
                )
            )
        elif device_seed is not None:
            # On-device prefill bootstrap: seed from the device sampled tokens
            # (no D2H). num_rejected stays 0 (num_rejected_tokens_np is None ->
            # num_rejected_np is zeros) and last_token_indices = qsl_end-1 (set
            # above) is the last prompt position -- the correct prefill seed.
            num_rejected_out = num_rejected_np
            # Partial-prefill (discard) reqs: override seed from host req_state.
            ov_i = (
                [i for i in range(num_reqs) if (start + i) in discard_set]
                if discard_set
                else []
            )
            if len(ov_i) == num_reqs:
                # All requests in the chunk are partial-prefill (discarded):
                # build both padded seed indices and override tokens on the host
                # at bucketed [p] shape — no device_seed slice, no index_put_,
                # and no unbucketed _draft_seed_input_ids recompile.
                req_ids = runner.input_batch.req_ids[start : start + num_reqs]
                ov_v = [
                    runner.requests[req_ids[i]].get_token_id(
                        runner.requests[req_ids[i]].num_computed_tokens
                        + scheduler_output.num_scheduled_tokens[req_ids[i]]
                    )
                    for i in ov_i
                ]
                lti_p_np = np.full(p, last_token_indices_np[-1], dtype=np.int64)
                lti_p_np[:num_reqs] = last_token_indices_np
                ntids_p_np = np.full(p, ov_v[-1], dtype=np.int32)
                ntids_p_np[:num_reqs] = ov_v
                last_token_indices = _h2d_temp(runner, torch.from_numpy(lti_p_np))
                next_token_ids = _h2d_temp(runner, torch.from_numpy(ntids_p_np))
            else:
                next_token_ids = device_seed[start : start + num_reqs].to(torch.int32)
                if ov_i:
                    req_ids = runner.input_batch.req_ids[start : start + num_reqs]
                    ov_np = np.full(num_reqs, -1, dtype=np.int32)
                    for i in ov_i:
                        req_state = runner.requests[req_ids[i]]
                        seq_len = (
                            req_state.num_computed_tokens
                            + scheduler_output.num_scheduled_tokens[req_ids[i]]
                        )
                        ov_np[i] = req_state.get_token_id(seq_len)
                    ov_dev = _h2d_temp(runner, torch.from_numpy(ov_np))
                    next_token_ids = torch.where(ov_dev >= 0, ov_dev, next_token_ids)
                runner_paddings = getattr(runner, "num_reqs_paddings", None)
                if (
                    num_reqs < p
                    and runner_paddings is not None
                    and num_reqs not in runner_paddings
                ):
                    pad_key = (p, num_reqs, str(runner.device))
                    pad_req_idx = self._async_pad_req_idx_cache.get(pad_key)
                    if pad_req_idx is None:
                        pad_req_idx_np = np.minimum(
                            np.arange(p, dtype=np.int64), num_reqs - 1
                        )
                        pad_req_idx = torch.from_numpy(pad_req_idx_np).to(runner.device)
                        self._async_pad_req_idx_cache[pad_key] = pad_req_idx
                    next_token_ids = torch.index_select(next_token_ids, 0, pad_req_idx)
                    lti_p_np = np.full(p, last_token_indices_np[-1], dtype=np.int64)
                    lti_p_np[:num_reqs] = last_token_indices_np
                    last_token_indices = _h2d_temp(runner, torch.from_numpy(lti_p_np))
                else:
                    last_token_indices = _h2d_temp(
                        runner, torch.from_numpy(last_token_indices_np)
                    )
        else:
            assert sampled_token_ids is not None
            req_ids = runner.input_batch.req_ids[start : start + num_reqs]
            num_rejected_out = num_rejected_np
            next_token_ids_np = np.zeros(num_reqs, dtype=np.int32)
            for i in range(num_reqs):
                batch_i = start + i  # batch-level request index
                if batch_i in discard_set:
                    req_id = req_ids[i]
                    req_state = runner.requests[req_id]
                    seq_len = (
                        req_state.num_computed_tokens
                        + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    next_token_ids_np[i] = req_state.get_token_id(seq_len)
                else:
                    ids = (
                        sampled_token_ids[batch_i]
                        if batch_i < len(sampled_token_ids)
                        else []
                    )
                    next_token_ids_np[i] = ids[-1] if ids else 0
        # Bucket the seed scatter + the returned gather index to the chunk's loop
        # bucket p so neither recompiles as num_reqs shrinks (the dominant draft
        # propose recompile). Pad by repeating the last real entry: the padded
        # rows re-write request (num_reqs-1)'s seed to its own slot (idempotent),
        # so draft_input_ids stays correct, and the padded gather tail re-reads a
        # real row that propose() slices off.
        if next_tokens_device is None and device_seed is None:
            # Pure sync: build the padded index + values on the host (free H2D,
            # no recompile), then scatter at the constant [p] shape.
            lti_p_np = np.full(p, last_token_indices_np[-1], dtype=np.int64)
            lti_p_np[:num_reqs] = last_token_indices_np
            ntids_p_np = np.full(
                p, next_token_ids_np[-1], dtype=next_token_ids_np.dtype
            )
            ntids_p_np[:num_reqs] = next_token_ids_np
            last_token_indices = _h2d_temp(runner, torch.from_numpy(lti_p_np))
            next_token_ids = _h2d_temp(runner, torch.from_numpy(ntids_p_np))
            draft_input_ids = self._draft_seed_input_ids(
                chunk.input_ids, last_token_indices, next_token_ids
            )
            gather_indices = last_token_indices
        elif device_seed is not None:
            # Bootstrap: pad the seed index + values to p as well, by repeating the
            # last real entry. Left at the real [num_reqs], the scatter missed the
            # startup-warmed [T, p] keys and dynamo re-traced it (~100-150 ms stall)
            # for every new (token bucket, num_reqs) pair while serving.
            if last_token_indices.shape[0] < p:
                last_token_indices = _pad_dim0_repeat_last(last_token_indices, p)
            if next_token_ids.shape[0] < p:
                next_token_ids = _pad_dim0_repeat_last(next_token_ids, p)
            draft_input_ids = self._draft_seed_input_ids(
                chunk.input_ids, last_token_indices, next_token_ids
            )
            padded_lti_np = np.zeros(p, dtype=np.int64)
            padded_lti_np[:num_reqs] = last_token_indices_np
            gather_indices = _h2d_temp(runner, torch.from_numpy(padded_lti_np))

        draft_input_ids, gather_indices = (
            chunk.sequence_layout_plan.localize_token_tensor_and_gather_indices(
                draft_input_ids,
                gather_indices,
                num_valid_gathers=num_reqs,
            )
        )

        return (draft_input_ids, chunk.position_ids, gather_indices, num_rejected_out)

    def _target_draft_attn_metadata(
        self, chunk: DraftChunkInputs
    ) -> dict[str, AttentionMetadata] | None:
        """The draft layers' entries of the target forward's own metadata.

        The target's walk in `_prepare_inputs` builds every KV cache group,
        the draft layers' included, from the chunk's builder context. A step-0
        rebuild would construct that same context and walk every group again
        (one block-table upload plus, on the unified pool, every mamba group's
        state-index gathers) only to keep the draft layers' entries. So these
        entries are the step-0 metadata, and a loop step differs from them
        only in seq_lens, query_start_loc and request_distribution: block
        tables cannot change mid-propose, and draft layers are attention
        layers, so no mamba field applies.

        None when the chunk carries no target metadata (EP-DP dummy chunks,
        standalone callers) or it does not cover every draft layer; the caller
        then rebuilds. Also None for a KV-sharing draft (Gemma-4 MTP): it runs
        inside the target's cache group and gets the group's full dict from
        the rebuild, not just the draft layers' entries.
        """
        if self._draft_kv_shared:
            return None
        target_md = getattr(chunk, "attn_metadata", None)
        if not target_md or chunk.attn_ctx.position_ids_override is not None:
            return None
        names = self._draft_attn_layer_names
        assert names is not None, (
            "draft attn layer names were not captured during load_model"
        )
        result = {name: md for name, md in target_md.items() if name in names}
        if len(result) != len(names) or not all(
            isinstance(md, AttentionMetadata) for md in result.values()
        ):
            return None
        return result

    def _loop_step_attn_inputs(
        self,
        chunk: DraftChunkInputs,
        seq_lens_delta: int,
        num_rejected_np: np.ndarray | None,
        num_rejected_dev: torch.Tensor | None,
        loop_query_start_loc: torch.Tensor | None,
        loop_request_distribution: torch.Tensor | None,
        precomputed_seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """seq_lens, query_start_loc and request_distribution of a loop step
        (one query per request), padded to the chunk's kernel_num_reqs.

        `precomputed_seq_lens` is the step's seq_lens already built on device
        (the post-step-0 bridge builds step 1's); it is used as is.
        """
        runner = self.runner
        num_reqs = chunk.num_reqs
        kernel_num_reqs = (
            runner.num_reqs_max_model_len
            if chunk.attn_ctx.use_max_model_len
            else runner.num_reqs_most_model_len
        )
        assert kernel_num_reqs is not None
        # query_start_loc / request_distribution depend only on num_reqs;
        # the loop hoists them across steps. Fall back to building them here
        # when not supplied (keeps this helper usable standalone).
        if loop_query_start_loc is not None:
            query_start_loc = loop_query_start_loc
        else:
            qsl_np = np.arange(kernel_num_reqs + 1, dtype=np.int32)
            qsl_np = np.minimum(qsl_np, num_reqs)
            query_start_loc = torch.from_numpy(qsl_np).to(runner.device)
        if precomputed_seq_lens is not None:
            seq_lens = precomputed_seq_lens
        else:
            # Full-length ops on the padded [kernel_num_reqs] seq_lens: the
            # tail [num_reqs:] gets the same delta / a zero subtraction but is
            # ignored by the kernel (query_start_loc caps queries at
            # num_reqs), so this stays correct while avoiding a per-num_reqs
            # slice-assign that recompiles as the batch shrinks. (`+` returns
            # a fresh tensor, so chunk.attn_ctx.seq_lens is not mutated — no
            # .clone() needed.)
            # ALWAYS subtract a (possibly all-zero) num_rejected so this
            # metadata program is ONE shape — the variant WITH a num_rejected
            # operand — regardless of whether this step actually had
            # rejections. The spec warmup's synthetic request is all-accept
            # (dummy logits -> draft==target -> zero rejections), so it only
            # ever exercised the add-only branch; the sub-with-num_rejected
            # variant then recompiled cold on the first real serving rejection
            # (the residual tt_jit_sub at the first reject step). Subtracting
            # zeros is value-identical to the add-only path, and building
            # num_rejected via from_numpy().to(device) keeps it a runtime
            # PARAMETER (matching the real-rejection program), not a folded
            # constant. num_rejected_dev (async) is already pre-padded; the
            # tail [num_reqs:] is zeros and ignored by the kernel
            # (query_start_loc caps queries at num_reqs).
            if num_rejected_dev is None:
                padded_rej = np.zeros(kernel_num_reqs, dtype=np.int32)
                if num_rejected_np is not None:
                    padded_rej[:num_reqs] = num_rejected_np
                num_rejected_dev = torch.from_numpy(padded_rej).to(runner.device)
            if chunk.attn_ctx.seq_lens.device.type == "tpu":
                # One compiled region, so the add + sub is not an eager
                # program re-fused per dispatch context.
                seq_lens = self._draft_loop_seq_lens(
                    chunk.attn_ctx.seq_lens, num_rejected_dev, int(seq_lens_delta)
                )
            else:
                seq_lens = chunk.attn_ctx.seq_lens + seq_lens_delta - num_rejected_dev
        # Pure decode distribution; matches precompile loop-step shape.
        if loop_request_distribution is not None:
            request_distribution = loop_request_distribution
        else:
            request_distribution = torch.tensor(
                [num_reqs, num_reqs, num_reqs],
                dtype=torch.int32,
                device=runner.device,
            )
        return seq_lens, query_start_loc, request_distribution

    def _build_draft_attn_metadata(
        self,
        chunk: DraftChunkInputs,
        step_idx: int,
        seq_lens_delta: int,
        num_rejected_np: np.ndarray | None,
        num_tokens_padded: int | None = None,
        num_rejected_dev: torch.Tensor | None = None,
        loop_query_start_loc: torch.Tensor | None = None,
        loop_request_distribution: torch.Tensor | None = None,
        precomputed_seq_lens: torch.Tensor | None = None,
    ) -> dict:
        """Build draft attention metadata for one chunk.

        Uses the chunk's captured ctx (chunk-local seq_lens /
        query_start_loc / request_distribution and the chunk's start_index
        for block-table slicing. num_rejected_np is already the chunk's slice.
        `precomputed_seq_lens`: a loop step's seq_lens, already built on
        device (see `_loop_step_attn_inputs`).
        """
        runner = self.runner
        num_reqs = chunk.num_reqs
        chunk_ctx = chunk.attn_ctx

        # Loop steps >= 2 reuse the chunk's previous loop step: qsl and
        # request_distribution are hoisted, block tables can't change
        # mid-propose and shapes are fixed, so only seq_lens differs. Both
        # paths below cache a step only where that holds.
        if step_idx >= 2:
            cached = self._draft_md_cache.get(id(chunk))
            if cached is not None:
                md_dict, prev_seq_lens, prev_delta = cached
                # base + prev_delta - rejected, moved by the delta's change, is
                # base + seq_lens_delta - rejected: value-identical to building
                # it. The change is +1 for a draft that appends a token per
                # step and 0 for one that re-reads the target's prefix.
                if seq_lens_delta == prev_delta:
                    return md_dict
                step_delta = int(seq_lens_delta - prev_delta)
                new_seq_lens = (
                    self._draft_add_delta(prev_seq_lens, step_delta)
                    if prev_seq_lens.device.type == "tpu"
                    else prev_seq_lens + step_delta
                )
                replaced: dict[int, object] = {}
                new_md = {}
                for lname, md in md_dict.items():
                    if id(md) not in replaced:
                        replaced[id(md)] = dataclasses.replace(
                            md, seq_lens=new_seq_lens
                        )
                    new_md[lname] = replaced[id(md)]
                self._draft_md_cache[id(chunk)] = (
                    new_md,
                    new_seq_lens,
                    seq_lens_delta,
                )
                return new_md

        target_md = self._target_draft_attn_metadata(chunk)
        if target_md is not None:
            if step_idx == 0:
                return target_md
            seq_lens, query_start_loc, request_distribution = (
                self._loop_step_attn_inputs(
                    chunk,
                    seq_lens_delta=seq_lens_delta,
                    num_rejected_np=num_rejected_np,
                    num_rejected_dev=num_rejected_dev,
                    loop_query_start_loc=loop_query_start_loc,
                    loop_request_distribution=loop_request_distribution,
                    precomputed_seq_lens=precomputed_seq_lens,
                )
            )
            replaced_target: dict[int, AttentionMetadata] = {}
            result = {}
            for name, md in target_md.items():
                if id(md) not in replaced_target:
                    # AttentionMetadata annotates these fields with the
                    # kernels' jax.Array; the runner fills them with torch
                    # tensors, as every builder of it does.
                    replaced_target[id(md)] = dataclasses.replace(
                        md,
                        # pyrefly: ignore[bad-argument-type]
                        seq_lens=seq_lens,
                        # pyrefly: ignore[bad-argument-type]
                        query_start_loc=query_start_loc,
                        # pyrefly: ignore[bad-argument-type]
                        request_distribution=request_distribution,
                    )
                result[name] = replaced_target[id(md)]
            self._draft_md_cache[id(chunk)] = (result, seq_lens, seq_lens_delta)
            return result

        # The rebuilt metadata is cached for the loop steps after it, except
        # on hybrid models and EP-DP dummy chunks (fixed positions).
        cacheable = (
            step_idx >= 1
            and not runner._has_mamba_state
            and not runner._unified_kv_layout
            and chunk_ctx.position_ids_override is None
        )

        use_max_model_len = chunk_ctx.use_max_model_len
        kernel_num_reqs = (
            runner.num_reqs_max_model_len
            if use_max_model_len
            else runner.num_reqs_most_model_len
        )
        assert kernel_num_reqs is not None

        if step_idx == 0:
            seq_lens = chunk_ctx.seq_lens
            query_start_loc = chunk_ctx.query_start_loc
            request_distribution = chunk_ctx.request_distribution
            num_tokens = int(chunk.query_start_loc_np[num_reqs])
            # Placeholder: _build_attention_metadata requires max_query_len but
            # the TPU backend never reads it. Any value is correct here.
            max_query_len = 1
        else:
            num_tokens = num_reqs
            seq_lens, query_start_loc, request_distribution = (
                self._loop_step_attn_inputs(
                    chunk,
                    seq_lens_delta=seq_lens_delta,
                    num_rejected_np=num_rejected_np,
                    num_rejected_dev=num_rejected_dev,
                    loop_query_start_loc=loop_query_start_loc,
                    loop_request_distribution=loop_request_distribution,
                    precomputed_seq_lens=precomputed_seq_lens,
                )
            )
            max_query_len = 1

        effective_num_tokens = num_tokens_padded if num_tokens_padded else num_tokens

        mamba_state_indices = (
            runner._build_mamba_state_indices(
                chunk.start_index, num_reqs, kernel_num_reqs
            )
            if runner._has_mamba_state
            else None
        )
        saved_ctx = runner._attn_metadata_builder_ctx
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=chunk.start_index,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            mamba_state_indices=mamba_state_indices,
            # None for real chunks; the EP-DP dummy-run pairing sets fixed
            # positions so the block-table build uses the zeroed-table branch
            # instead of slicing the real (idle-rank) block table.
            position_ids_override=chunk.attn_ctx.position_ids_override,
            sequence_layout_descriptor=(chunk.attn_ctx.sequence_layout_descriptor),
        )
        try:
            slot_mappings = runner.empty_slot_mappings
            per_layer_attn_metadata, _ = runner._build_attention_metadata(
                num_tokens=effective_num_tokens,
                num_reqs=kernel_num_reqs,
                max_query_len=max_query_len,
                num_tokens_padded=effective_num_tokens,
                num_reqs_padded=kernel_num_reqs,
                slot_mappings=slot_mappings,
            )
        finally:
            runner._attn_metadata_builder_ctx = saved_ctx

        assert self._draft_attn_layer_names is not None, (
            "draft attn layer names were not captured during load_model"
        )

        # Draft layers sharing the target's KV cache were added to the target's
        # KV cache group by `add_kv_sharing_layers_to_kv_cache_groups`, so
        # `_build_attention_metadata` has already emitted an entry for each of
        # them. Verify rather than re-alias: a missing entry means the layer
        # never joined a group, and aliasing it to an arbitrary entry would hide
        # that behind a plausible-looking forward.
        missing = [
            name
            for name in self._draft_attn_layer_names
            if name not in per_layer_attn_metadata
        ]
        if missing:
            raise RuntimeError(
                f"Draft attention layers {missing} have no attention metadata; "
                "they were not added to a KV cache group. Check that "
                "get_kv_cache_spec registered them in shared_kv_cache_layers."
            )

        # A KV-sharing draft runs inside the target's cache group, so pass the
        # group's entries through untouched. A draft that owns its cache keeps
        # the historical draft-only filter.
        if self._draft_kv_shared:
            result = dict(per_layer_attn_metadata)
        else:
            result = {
                name: per_layer_attn_metadata[name]
                for name in self._draft_attn_layer_names
            }

        if cacheable:
            self._draft_md_cache[id(chunk)] = (result, seq_lens, seq_lens_delta)

        return result

    @staticmethod
    def _unwrap_model_out(out) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalise a draft-model forward result to (last_hidden, carry_hidden).

        The draft model may return a bare tensor or a (last_hidden, aux_hidden)
        2-tuple (optionally wrapped in a 1-element list/tuple). A bare tensor is
        used for both outputs.
        """
        if isinstance(out, (list, tuple)) and len(out) == 1:
            out = out[0]
        if isinstance(out, (list, tuple)):
            if len(out) != 2:
                raise RuntimeError(
                    f"Draft model returned a {len(out)}-element tuple; expected "
                    "a tensor or a 2-element (last_hidden, aux_hidden) tuple."
                )
            last_hidden, hidden = out
        else:
            last_hidden = hidden = out
        return last_hidden, hidden

    def _use_mtp_step0_post_attn_gather(
        self,
        chunk: DraftChunkInputs,
        *,
        can_use_post_step0_fused: bool = True,
    ) -> bool:
        if (
            not envs.TPU_MTP_STEP0_POST_ATTN_GATHER
            or self.speculative_config.method != "mtp"
            or self._share_mtp_indices
            or self._draft_kv_shared
            or "_forward_draft" in getattr(self, "__dict__", {})
        ):
            return False
        if (
            getattr(chunk.sequence_layout_plan, "kind", SequenceLayoutKind.ALL)
            is not SequenceLayoutKind.ALL
        ):
            return False
        K = self.speculative_config.num_speculative_tokens
        if (
            K > 1
            and not can_use_post_step0_fused
            and self._mtp_loop_model is None
            and not getattr(self.runner, "enforce_eager", False)
            and not getattr(self.draft_model, "do_not_compile", True)
        ):
            return False
        from vllm_torchtpu.models.vllm.qwen3_next_patch import (
            is_step0_post_attn_gather_installed,
        )

        return is_step0_post_attn_gather_installed(self.draft_model)

    def _forward_draft(
        self,
        chunk: DraftChunkInputs,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        step_idx: int,
        seq_lens_delta: int,
        num_rejected_np: np.ndarray | None,
        num_tokens_padded: int | None = None,
        num_rejected_dev: torch.Tensor | None = None,
        loop_query_start_loc: torch.Tensor | None = None,
        loop_request_distribution: torch.Tensor | None = None,
        step0_gather_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_metadata = self._build_draft_attn_metadata(
            chunk=chunk,
            step_idx=step_idx,
            seq_lens_delta=seq_lens_delta,
            num_rejected_np=num_rejected_np,
            num_tokens_padded=num_tokens_padded,
            num_rejected_dev=num_rejected_dev,
            loop_query_start_loc=loop_query_start_loc,
            loop_request_distribution=loop_request_distribution,
        )

        num_tokens = input_ids.shape[0]
        # Record this step's row form before the compiled forward: the draft
        # runs under sequence-parallel MoE like the target (main #1562), so
        # its rows are sliced at or above min_rows and replicated below it or
        # under the step-0 post-attention gather. The MoE dispatch / combine
        # read this; a mismatch with the model's own chunking all-reduces
        # sliced rows (silently wrong output).
        rows_sliced = self._draft_rows_sliced(num_tokens, step0_gather_indices)
        sequence_parallel.set_active(rows_sliced)
        num_valid_tokens = self._draft_valid_tokens(chunk, step_idx, num_tokens)

        kwargs: dict[str, torch.Tensor | int] = {
            "input_ids": input_ids,
            "positions": positions,
            "hidden_states": target_hidden_states,
        }
        if self.speculative_config.method == "mtp":
            kwargs["spec_step_idx"] = step_idx

        model = self._draft_model_for_step(step_idx)
        if model is self._mtp_loop_model:
            loop_padding_state = self._get_loop_token_padding_state()
            if loop_padding_state is not None:
                loop_padding_state.update(
                    *self._rank_padding_counts(
                        num_valid_tokens, num_tokens, rows_sliced
                    )
                )
            padding_ctx = token_padding.use_padding_state(loop_padding_state)
            force_small_ar = (
                not rows_sliced
                and hasattr(self.runner, "max_num_reqs")
                and (
                    self._loop_bucket(self.runner.max_num_reqs)
                    <= small_all_reduce_op.MAX_ROWS
                )
            )
        else:
            if hasattr(self.runner, "_token_padding_update"):
                # The mask follows the row form recorded above: this rank's
                # slice when rows are sliced, all num_tokens rows when they
                # are replicated (the per-rank slice there would mark real
                # rows as padding and drop their experts). The step-0 gather
                # always runs replicated, over the gathered rows.
                if step0_gather_indices is not None:
                    gather_tokens = int(step0_gather_indices.shape[0])
                    gather_valid = max(
                        0, min(int(getattr(chunk, "num_reqs", 0) or 0), gather_tokens)
                    )
                    self.runner._token_padding_update(
                        gather_tokens,
                        num_valid_tokens=gather_valid,
                        rows_sliced=False,
                    )
                else:
                    self.runner._token_padding_update(
                        num_tokens,
                        num_valid_tokens=num_valid_tokens,
                        rows_sliced=rows_sliced,
                    )
            padding_ctx = contextlib.nullcontext()
            # The squash's replicated drafter always took the small
            # all-reduce kernel; a sliced step keeps main's collectives.
            force_small_ar = not rows_sliced

        if step0_gather_indices is not None:
            from vllm_torchtpu.models.vllm.qwen3_next_patch import (
                step0_post_attn_gather_context,
            )

            step0_gather_ctx = step0_post_attn_gather_context(step0_gather_indices)
        else:
            step0_gather_ctx = contextlib.nullcontext()

        with (
            set_model_tag("eagle_head"),
            padding_ctx,
            step0_gather_ctx,
            small_all_reduce_op.force_small_all_reduce(enabled=force_small_ar),
            set_forward_context(
                attn_metadata,
                self.draft_vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=self.runner._dp_num_tokens_across_dp(num_tokens),
            ),
            set_vllm_model_wrapper_context(
                mesh=self.runner.mesh, vllm_config=self.draft_vllm_config
            ),
        ):
            if model is self._mtp_loop_model and not self._mtp_loop_model_warm:
                # This call traces and compiles the loop-step program. Like
                # the draft's own first call (see the retry in
                # TPUModelRunner._warmup_spec_decode), its output can carry
                # symbolic sizes that the dynamic=False helpers after it
                # reject, so run the step once more now that it's compiled.
                # The repeat writes the same KV entries again, and a loop step
                # only reads the top-k table.
                model(**kwargs)
                self._mtp_loop_model_warm = True
            out = model(**kwargs)

        return self._unwrap_model_out(out)

    def precompile(self) -> None:
        """Precompile the draft-side @torch.compile wrapper subgraphs.

        Warms the wrappers across their bucket shapes so they don't recompile on
        the first real propose(): combine_hidden_states (token-count buckets),
        seed_input_ids ((token, index) buckets), and the lm-head/argmax
        compute_logits (num_reqs buckets). The draft forward and the per-step
        fused propose/verify programs are warmed by a real-path lifecycle
        warmup, not here (a zeros-precompile of those emits programs that never
        match the runtime eager fusion).
        """
        if self.runner.enforce_eager:
            return
        self._precompile_combine_hidden_states()
        self._precompile_draft_seed()
        self._precompile_compute_logits()

    def _draft_hidden_size(self) -> int:
        """Width of the hidden state the draft's lm_head consumes.

        Prefer the built draft's own config: that is the value every draft
        resolved today already reports, so this stays a no-op for them.

        Some drafts are loaded under a *wrapper* config that keeps the text
        dims on a nested `text_config` and exposes no top-level `hidden_size`
        at all -- Gemma-4 MTP's `Gemma4AssistantConfig` and Qwen3.5 MTP's
        `Qwen3_5MoeConfig` are both like this. Reading `hidden_size` off the
        wrapper raises AttributeError, and the one width it *does* expose
        (`backbone_hidden_size`) is the TARGET's, not the draft's. Fall back
        to vLLM's already-resolved text config for those.
        """
        assert self.draft_model is not None
        hidden_size = getattr(self.draft_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = (
                self.speculative_config.draft_model_config.hf_text_config.hidden_size
            )
        return hidden_size

    def _draft_uses_aux_hidden_state(self) -> bool:
        """Whether this draft checkpoint feeds combine_hidden_states the
        concatenation of several target aux hidden-state layers.
        """
        if self.speculative_config.method == "mtp":
            return False
        assert self.draft_model is not None
        return bool(getattr(self.draft_model.model, "use_aux_hidden_state", True))

    def _draft_combine_input_size(self) -> int:
        """Width of the per-token tensor that combine_hidden_states consumes.

        When the draft uses aux hidden states, this is the draft's own
        `fc_input_size`. Otherwise combine_hidden_states is an identity over the plain
        hidden_size-wide target hidden state, so the input width is just hidden_size.
        """
        if self._draft_uses_aux_hidden_state():
            assert self.draft_model is not None
            return self.draft_model.model.fc_input_size
        return self._draft_hidden_size()

    def _draft_has_moe(self) -> bool:
        """Whether the draft model carries MoE layers.

        A TP-replicated draft is not rank-local when it carries MoE: under
        `--enable-expert-parallel` the experts are distributed across the DP
        group, so the draft forward emits cross-DP collectives just like a
        TP-sharded draft does. `_force_draft_tp1` only collapses the TP group;
        it does not make the MoE local. (The MTP draft of an MoE target is
        itself a full decoder layer, MoE block included.) Cached after the
        first query; False until the draft model is loaded.
        """
        if self._draft_has_moe_cache is None:
            if self.draft_model is None:
                return False
            from vllm.model_executor.models.interfaces import is_mixture_of_experts

            self._draft_has_moe_cache = bool(is_mixture_of_experts(self.draft_model))
        return self._draft_has_moe_cache

    def _dp_lockstep_sharded(self) -> bool:
        """True when the draft emits cross-DP collectives under EP-DP lockstep.

        Those collectives are part of the cross-rank collective program, so
        every rank must execute the same collective trace. Two ways the draft
        can emit them: it is TP-SHARDED (its TP collectives ride the same
        program), or it carries expert-parallel MoE layers, whose experts span
        the DP group even when the draft is TP-replicated (draft_tp=1). A
        replicated, MoE-free draft is entirely local and needs none of this.
        """
        if not self.runner._dp_lockstep_enabled():
            return False
        return not self._draft_replicated or self._draft_has_moe()

    def _loop_bucket(self, num_reqs: int) -> int:
        """Loop-phase token bucket p (carries, lm-head gathers, loop forwards)."""
        runner = self.runner
        if self._dp_lockstep_sharded():
            num_reqs = runner.max_num_reqs
        req_paddings = getattr(runner, "num_reqs_paddings", None)
        min_p = req_paddings[0] if req_paddings else 1
        num_reqs = max(num_reqs, min_p)
        paddings = runner.num_tokens_paddings
        index = bisect.bisect_left(paddings, num_reqs)
        assert index < len(paddings)
        return paddings[index]

    def _precompile_combine_hidden_states(self) -> None:
        if self.speculative_config.method == "mtp":
            return  # MTP does not use combine_hidden_states

        runner = self.runner
        if not self._draft_uses_aux_hidden_state():
            # combine_hidden_states is an identity for non-aux drafts.
            return
        # Warm the compiled _draft_combine_hidden_states wrapper with 3 SEPARATE
        # aux dummies (not one pre-cat [N, 3*aux] dummy) so the compiled
        # cat+mm grouping's shape-key matches the real first-pass dispatch.
        aux_w = self._draft_combine_input_size() // 3
        with (
            set_model_tag("eagle_head"),
            runner._precompile_timed("drafter combine_hidden_states"),
        ):
            for num_tokens in runner.num_tokens_paddings:
                aux = [
                    torch.zeros(
                        (num_tokens, aux_w),
                        dtype=runner._hidden_states_dtype,
                    ).to(runner.device)
                    for _ in range(3)
                ]
                out = self._draft_combine_hidden_states(*aux)
                synchronize_tensors(out)
                logger.info("  -- drafter combine num_tokens: %d", num_tokens)

    def _precompile_draft_seed(self) -> None:
        from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

        runner = self.runner
        # input_ids length = first-pass token bucket; index length = p =
        # pad(num_reqs), bounded by the loop bucket pad(max_num_reqs). Mirror the
        # bucket selection in _precompile_compute_logits so every sync first-pass
        # (T, p) shape-key is a startup cache hit.
        max_loop_bucket = _get_padded_token_len(
            runner.num_tokens_paddings, runner.max_num_reqs
        )
        idx_sizes = sorted(
            set(runner.num_reqs_paddings)
            | {t for t in runner.num_tokens_paddings if t <= max_loop_bucket}
        )
        K = self.speculative_config.num_speculative_tokens
        knr_buckets = {runner.num_reqs_max_model_len}
        if (
            getattr(runner, "most_model_len", None) is not None
            and getattr(runner, "num_reqs_most_model_len", None) is not None
        ):
            knr_buckets.add(runner.num_reqs_most_model_len)
        req_loop_pairs = sorted(
            {
                (
                    runner.num_reqs_paddings[
                        bisect.bisect_left(runner.num_reqs_paddings, r)
                    ],
                    self._loop_bucket(r),
                )
                for r in range(1, runner.max_num_reqs + 1)
            }
            | {(B, self._loop_bucket(B)) for B in runner.num_reqs_paddings}
        )
        loop_p_buckets = sorted(
            {t for t in runner.num_tokens_paddings if t <= max_loop_bucket}
            | {p for _, p in req_loop_pairs}
        )
        with runner._precompile_timed("drafter seed_input_ids"):
            for T in runner.num_tokens_paddings:
                for p in idx_sizes:
                    if p > T:
                        continue
                    ids = torch.zeros(T, dtype=torch.int32).to(runner.device)
                    lti = torch.zeros(p, dtype=torch.int64).to(runner.device)
                    nti = torch.zeros(p, dtype=torch.int32).to(runner.device)
                    out = self._draft_seed_input_ids(ids, lti, nti)
                    synchronize_tensors(out)
                ids = torch.zeros(T, dtype=torch.int32).to(runner.device)
                for B, p in req_loop_pairs:
                    if B > T or p > T:
                        continue
                    nt = torch.zeros((B, K + 1), dtype=torch.int32).to(runner.device)
                    dl = torch.zeros(B, dtype=torch.int32).to(runner.device)
                    pad_idx = torch.zeros(p, dtype=torch.int64).to(runner.device)
                    ov_warm = torch.full(
                        (B,), -1, dtype=torch.int32, device=runner.device
                    )
                    for knr in sorted(knr_buckets):
                        qsl = torch.ones(knr + 1, dtype=torch.int32).to(runner.device)
                        out_async = self._draft_async_prepare_inputs(
                            ids, nt, dl, qsl, pad_idx
                        )
                        synchronize_tensors(out_async)
                        out_async_ov = self._draft_async_prepare_inputs(
                            ids, nt, dl, qsl, pad_idx, ov_warm
                        )
                        synchronize_tensors(out_async_ov)
            for knr in sorted(knr_buckets):
                base_sl = torch.zeros(knr, dtype=torch.int32).to(runner.device)
                rej_dev = torch.zeros(knr, dtype=torch.int32).to(runner.device)
                sl_delta = 1 if self.advance_draft_positions else 0
                out_sl = self._draft_loop_seq_lens(base_sl, rej_dev, sl_delta)
                synchronize_tensors(out_sl)
                if self.advance_draft_positions and K >= 3:
                    out_sl_add = self._draft_add_delta(base_sl, 1)
                    synchronize_tensors(out_sl_add)
            for p in loop_p_buckets:
                if self.advance_draft_positions and K >= 2:
                    pos_shape = (3, p) if runner.uses_mrope else (p,)
                    pos_p = torch.zeros(pos_shape, dtype=torch.int32).to(runner.device)
                    out_pos = self._draft_add_delta(pos_p, 1)
                    synchronize_tensors(out_pos)
                toks_p = [
                    torch.zeros(p, dtype=torch.int32).to(runner.device)
                    for _ in range(K)
                ]
                out_stack = self._draft_stack_tokens(*toks_p)
                synchronize_tensors(out_stack)

    def _precompile_compute_logits(self) -> None:
        from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

        runner = self.runner
        hidden_size = self._draft_hidden_size()
        # compute_logits is called on (a) num_reqs buckets in the first pass and
        # (b) the loop's padded carry = _get_padded_token_len(num_tokens_paddings,
        # c.num_reqs), with c.num_reqs <= max_num_reqs. So the loop can only reach
        # token buckets up to the one just past max_num_reqs — precompile those
        # plus the num_reqs buckets. (The full num_tokens_paddings also runs up to
        # max_num_batched_tokens, compiling large [n, vocab] matmuls the loop never
        # hits.) Filtering by <= max_loop_bucket keeps every intermediate bucket
        # the loop can land on, unlike sampling only at num_reqs_paddings points.
        max_loop_bucket = _get_padded_token_len(
            runner.num_tokens_paddings, runner.max_num_reqs
        )
        loop_p_buckets = sorted(
            {t for t in runner.num_tokens_paddings if t <= max_loop_bucket}
        )
        all_sizes = set(runner.num_reqs_paddings) | set(loop_p_buckets)
        knr_buckets = {runner.num_reqs_max_model_len}
        if (
            getattr(runner, "most_model_len", None) is not None
            and getattr(runner, "num_reqs_most_model_len", None) is not None
        ):
            knr_buckets.add(runner.num_reqs_most_model_len)
        with (
            set_model_tag("eagle_head"),
            runner._precompile_timed("drafter compute_logits"),
        ):
            for n in all_sizes:
                dummy_hidden = torch.zeros(
                    (n, hidden_size),
                    dtype=runner._hidden_states_dtype,
                ).to(runner.device)
                out = self._draft_propose_token(dummy_hidden)
                synchronize_tensors(out)
                logger.info("  -- drafter compute_logits n: %d", n)
            from vllm_torchtpu.models.vllm.qwen3_next_patch import (
                is_step0_post_attn_gather_installed,
            )

            step0_post_attn_gather = (
                envs.TPU_MTP_STEP0_POST_ATTN_GATHER
                and self.speculative_config.method == "mtp"
                and not self._share_mtp_indices
                and not self._draft_kv_shared
                and not runner._dp_lockstep_enabled()
                # Runtime never takes this path under PCP (the layout plan is
                # not ALL), so do not precompile it there either.
                and not getattr(runner, "_pcp_enabled", False)
                and is_step0_post_attn_gather_installed(self.draft_model)
            )
            if self._mtp_fused_loop is not None or step0_post_attn_gather:
                h_step0_gathered = (
                    torch.zeros(
                        (max_loop_bucket, hidden_size),
                        dtype=runner._hidden_states_dtype,
                    ).to(runner.device)
                    if step0_post_attn_gather
                    else None
                )
                for T in runner.num_tokens_paddings:
                    h_t = (
                        h_step0_gathered
                        if h_step0_gathered is not None
                        else torch.zeros(
                            (T, hidden_size),
                            dtype=runner._hidden_states_dtype,
                        ).to(runner.device)
                    )
                    pos_shape = (3, T) if runner.uses_mrope else (T,)
                    pos_t = torch.zeros(pos_shape, dtype=torch.int32).to(runner.device)
                    for p in loop_p_buckets:
                        if p > T:
                            continue
                        idx_p = torch.zeros(p, dtype=torch.int64).to(runner.device)
                        if self._mtp_fused_loop is not None:
                            for knr in sorted(knr_buckets):
                                base_sl = torch.zeros(knr, dtype=torch.int32).to(
                                    runner.device
                                )
                                rej_dev = torch.zeros(knr, dtype=torch.int32).to(
                                    runner.device
                                )
                                out_fused = self._draft_post_step0_fused(
                                    h_t,
                                    pos_t,
                                    idx_p,
                                    base_sl,
                                    rej_dev,
                                    step0_post_attn_gather,
                                )
                                synchronize_tensors(list(out_fused))
                        elif step0_post_attn_gather:
                            out_carries = self._draft_gather_carries(
                                h_t,
                                pos_t,
                                h_t,
                                idx_p,
                                True,
                            )
                            synchronize_tensors(list(out_carries))

    def run_dp_dummy_draft(self, num_chunks: int) -> None:
        """Replay a busy rank's DRAFT collective trace on an idle/padding rank,
        for `num_chunks` padding chunks (EP-DP lockstep).

        SHARDED draft (draft_tp == target tp): the draft's collectives are matched
        across ranks by graph-structure channel IDs, so each padding chunk must
        replay the REAL propose trace, channel IDs attach to collectives only.

        REPLICATED draft (draft_tp=1): entirely local — it emits no
        collectives, so there is nothing for an idle rank to pair with and the
        forwards would be pure wasted device time.
        """
        if not self._dp_lockstep_sharded():
            return
        self._draft_md_cache.clear()
        runner = self.runner
        K = self.speculative_config.num_speculative_tokens
        if num_chunks * K == 0:
            return
        bucket = runner._dp_target_bucket
        assert bucket is not None, "_dp_target_bucket unset in lockstep step"
        num_reqs = runner.num_reqs_max_model_len
        input_ids = torch.zeros(bucket, dtype=torch.int32, device=runner.device)
        # Two consumers, two ranks. The draft forward takes mrope-shaped
        # positions [3, N] on models that use them, and its compiled graph
        # is specialized on that rank (dynamic_arg_dims marks positions
        # dim -1), so a 1-D dummy trips the guard with
        # `IndexError: Dimension out of range`. The attn-metadata build
        # always wants the 1-D token positions, matching
        # runner.position_ids.
        token_positions = torch.zeros(bucket, dtype=torch.int32, device=runner.device)
        positions = (
            torch.zeros((3, bucket), dtype=torch.int32, device=runner.device)
            if runner.uses_mrope
            else token_positions
        )
        target_hidden_states = torch.zeros(
            (bucket, self._draft_hidden_size()),
            dtype=runner._hidden_states_dtype,
            device=runner.device,
        )

        actual_num_reqs = min(bucket, num_reqs)
        qsl_np = np.arange(num_reqs + 1, dtype=np.int32)
        attn_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones(num_reqs, dtype=torch.int32, device=runner.device),
            query_start_loc=torch.from_numpy(qsl_np).to(runner.device),
            request_distribution=torch.tensor(
                [actual_num_reqs, actual_num_reqs, actual_num_reqs],
                dtype=torch.int32,
                device=runner.device,
            ),
            # Fixed positions => the attn-metadata build takes the dummy-safe
            # zeroed-block-table branch (attention_metadata.py) instead of
            # slicing the real block table, which on an idle rank is shorter
            # than num_reqs and would fail to broadcast.
            position_ids_override=token_positions,
        )
        chunk = DraftChunkInputs(
            input_ids=input_ids,
            position_ids=positions,
            query_start_loc_np=qsl_np,
            attn_ctx=attn_ctx,
            start_index=0,
            num_reqs=num_reqs,
            aux_hidden_states=[],
        )
        # Replay the real propose trace per padding chunk.
        p = self._loop_bucket(num_reqs)
        draft_hidden = self._draft_hidden_size()
        dtype = runner._hidden_states_dtype
        # First-pass lm-head input mirrors the gathered carry [p, hidden].
        first_pass_carry = torch.zeros(
            (p, draft_hidden), dtype=dtype, device=runner.device
        )
        loop_input_ids = torch.zeros(p, dtype=torch.int32, device=runner.device)
        # Mirrors the real loop carry, which stays mrope-shaped across steps
        # (propose() pads it with _maybe_pad_dim1 when pos.ndim == 2).
        loop_positions = torch.zeros(
            (3, p) if runner.uses_mrope else (p,),
            dtype=torch.int32,
            device=runner.device,
        )
        loop_hidden = torch.zeros((p, draft_hidden), dtype=dtype, device=runner.device)
        # Loop-step metadata mirrors propose()'s per-chunk loop tensors.
        loop_qsl = torch.from_numpy(
            np.minimum(np.arange(num_reqs + 1, dtype=np.int32), actual_num_reqs)
        ).to(runner.device)
        loop_reqdist = torch.tensor(
            [actual_num_reqs] * 3, dtype=torch.int32, device=runner.device
        )
        uses_aux = self._draft_uses_aux_hidden_state()
        # Mirror propose()'s branch exactly: only eagle3 runs the plain
        # hidden state through combine_hidden_states. An MTP draft feeds the
        # target hidden state straight to the forward and has no
        # combine_hidden_states at all, so replaying one here would both
        # crash and add a step the real trace does not have.
        combines_plain = not uses_aux and (self.speculative_config.method == "eagle3")
        combine_aux: list[torch.Tensor] = []
        combine_plain: torch.Tensor | None = None
        if uses_aux:
            aux_width = self._draft_combine_input_size() // 3
            combine_aux = [
                torch.zeros((bucket, aux_width), dtype=dtype, device=runner.device)
                for _ in range(3)
            ]
        elif combines_plain:
            combine_plain = torch.zeros(
                (bucket, self._draft_combine_input_size()),
                dtype=dtype,
                device=runner.device,
            )
        for _ in range(num_chunks):
            # Mirror the real first pass: combine -> forward @bucket -> lm head.
            if uses_aux or combines_plain:
                with set_model_tag("eagle_head"):
                    if uses_aux:
                        self._draft_combine_hidden_states(*combine_aux)
                    else:
                        assert self.draft_model is not None
                        self.draft_model.combine_hidden_states(combine_plain)
            last_hidden, _ = self._forward_draft(
                chunk=chunk,
                input_ids=input_ids,
                positions=positions,
                target_hidden_states=target_hidden_states,
                step_idx=0,
                seq_lens_delta=0,
                num_rejected_np=None,
                num_tokens_padded=bucket,
            )
            with set_model_tag("eagle_head"):
                tok = self._draft_propose_token(first_pass_carry)
            # Mirror the K-1 loop steps: forward @p -> lm head @p. Under
            # IndexShare, _forward_draft routes these to the loop-step program
            # exactly as a real propose does, so this rank runs the same
            # programs, and the same collectives, as its peers. They read the
            # top-k table without having written it this step; load_model
            # seeds it with -1 so that read stays in range.
            for step in range(1, K):
                last_hidden, _ = self._forward_draft(
                    chunk=chunk,
                    input_ids=loop_input_ids,
                    positions=loop_positions,
                    target_hidden_states=loop_hidden,
                    step_idx=step,
                    # Mirror the real loop so the dummy traces the same
                    # metadata program the peer ranks' propose emits.
                    seq_lens_delta=(step if self.advance_draft_positions else 0),
                    num_rejected_np=None,
                    num_tokens_padded=p,
                    loop_query_start_loc=loop_qsl,
                    loop_request_distribution=loop_reqdist,
                )
                with set_model_tag("eagle_head"):
                    tok = self._draft_propose_token(last_hidden)
            # Force the chunk to execute so its collectives fire in lockstep
            # with the peer ranks' real propose (nothing consumes the result).
            synchronize_tensors(tok)
