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
"""MiniMax M3 (text) on TPU."""

import contextlib
import functools
from collections.abc import Callable, Iterable, Iterator

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed import pallas_all_reduce
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.quantization.fp8_lm_head import Fp8LMHeadMethod
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3 import residual_ar, sp_o_proj
from vllm_torchtpu.models.minimax_m3.attention import (
    MiniMaxM3SparseAttention,
    StepHalves,
)
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse, sp_norm
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP, MiniMaxM3MoE
from vllm_torchtpu.models.minimax_m3.ops.sp_norm import get_add_norm_quant_op
from vllm_torchtpu.models.minimax_m3.quant import CT_TO_MODELOPT_EXPERT_SUFFIX

logger = init_logger(__name__)


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, layer_id: int, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.self_attn = MiniMaxM3SparseAttention(
            config,
            quant_config,
            vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
        )
        # Dense layers name their FFN `mlp`, MoE layers `block_sparse_moe`.
        self.is_moe_layer = bool(config.moe_layer_freq[layer_id])
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3MoE(
                config, quant_config, prefix=f"{prefix}.block_sparse_moe"
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config,
                config.dense_intermediate_size,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > 1 and envs.TPU_MINIMAX_M3_SP_FUSED_NORM:
            self._add_norm_quant = get_add_norm_quant_op(config.rms_norm_eps)

    @contextlib.contextmanager
    def _layer_owned_reductions(self) -> Iterator[None]:
        """While tracing a sequence-parallel step, o_proj and the FFN return
        per-rank partial sums and the layer reduces them; every other trace
        keeps their own all-reduces, so decode programs are unchanged."""
        o_proj = self.self_attn.o_proj
        if self.is_moe_layer:
            cfg = self.block_sparse_moe.experts.moe_config
            saved = (o_proj.reduce_results, cfg.skip_final_all_reduce)
            o_proj.reduce_results, cfg.skip_final_all_reduce = False, True
        else:
            down = self.mlp.down_proj
            saved = (o_proj.reduce_results, down.reduce_results)
            o_proj.reduce_results = down.reduce_results = False
        try:
            yield
        finally:
            if self.is_moe_layer:
                o_proj.reduce_results, cfg.skip_final_all_reduce = saved
            else:
                o_proj.reduce_results, down.reduce_results = saved

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attn_rs: bool = False,
        step: StepHalves | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual-deferred layer; ``attn_rs`` (large steps) reduce-scatters
        the attention output instead of all-reducing it."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        whole = None if step is None else (step, None)
        if not attn_rs:
            hidden_states = self.self_attn(positions, hidden_states, half=whole)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
            return ffn(hidden_states), residual
        with self._layer_owned_reductions():
            return self._attn_rs(positions, hidden_states, residual, whole)

    def forward_residual_ar(
        self,
        positions: torch.Tensor,
        partial: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The layer once `residual_ar.arm` armed it: ``partial`` is the
        previous layer's FFN output before its TP all-reduce (the embedding,
        with ``residual`` None, for the first layer). Returns this layer's FFN
        partial, the residual, and the reduced residual-stream entry
        (``x + residual`` of the layer's input)."""
        op_a, op_b = self._residual_ar_ops
        w_qkv, s_qkv, nw_in, w_o, s_o, nw_post = residual_ar.residual_ar_operands(self)
        attn = self.self_attn
        if residual is None:
            entry = partial
            qkv, _ = attn.qkv_proj(self.input_layernorm(partial))
        else:
            qkv, entry = op_a(partial, residual, nw_in, w_qkv, s_qkv, self._mesh_rank)
        heads = attn(positions, None, project=False, qkv=qkv)
        normed, residual = op_b(heads, w_o, s_o, entry, nw_post, self._mesh_rank)
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        return ffn(normed), residual, entry

    def _attn_rs(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        whole: tuple[StepHalves, None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each rank gets the new residual stream of its token shard,
        normalizes it and all-gathers it as the FFN input; the attention
        partial rides the FFN all-reduce. The returned residual is the
        pre-attention one, so the next fused add + norm sees
        ``ffn + attn + residual``."""
        attn = self.self_attn(positions, hidden_states, half=whole)
        shard = tensor_model_parallel_reduce_scatter(attn + residual / self.tp_size, 0)
        normed = _all_gather_rows(
            self.post_attention_layernorm(shard), envs.TPU_MINIMAX_M3_SP_FP8_GATHER
        )
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        out = ffn(normed) + attn
        return tensor_model_parallel_all_reduce(out), residual

    def forward_sp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        shard: torch.Tensor | None,
        update: torch.Tensor | None = None,
        halves: StepHalves | None = None,
        step: StepHalves | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Sequence-parallel layer: the residual stream stays a token shard.

        ``shard`` is this rank's rows of the residual stream (None for the
        first layer, whose ``hidden_states`` is the full embedding) and
        ``update`` the previous layer's not yet added reduce-scattered FFN
        output (None when already in ``shard``). Norms run on the shard,
        all-gathers feed attention and the FFN, and their partial outputs are
        reduce-scattered back onto the shard. Returns the residual shard
        entering this layer (None for the first), the shard after attention
        and this layer's FFN update. With ``halves`` the shard holds this
        rank's block of each token half (`_halves_rows`) and attention runs
        per half (`_attention_halves`).
        """
        with self._layer_owned_reductions():
            return self._forward_sp(
                positions, hidden_states, shard, update, halves, step
            )

    def _norm_gather(
        self,
        norm: GemmaRMSNorm,
        shard: torch.Tensor,
        update: torch.Tensor | None,
        fp8: bool,
        e4m3_out: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """(shard + update, all-gathered norm(shard + update)): one fused pass
        over the shard when it feeds an e4m3 gather and an update is pending.
        ``e4m3_out`` returns the gathered e4m3 rows and their scales."""
        fused = envs.TPU_MINIMAX_M3_SP_FUSED_NORM and _fits_sp_norm(shard)
        if update is not None and fp8 and fused:
            weight = norm.weight.float() + 1.0
            shard, q, scale = self._add_norm_quant(shard, update, weight)
        else:
            if update is not None:
                shard = shard + update
            normed = norm(shard)
            if not fp8:
                return shard, tensor_model_parallel_all_gather(normed, 0)
            q, scale = _quantize_rows(normed)
        q, scale = _gather_rows_e4m3(q, scale)
        if e4m3_out:
            return shard, (q, scale)
        return shard, (q.float() * scale).to(shard.dtype)

    def _forward_sp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        shard: torch.Tensor | None,
        update: torch.Tensor | None,
        halves: StepHalves | None = None,
        step: StepHalves | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        layer_input = None
        o_proj = self.self_attn.o_proj
        whole = None if step is None else (step, None)
        if shard is None:
            normed = self.input_layernorm(hidden_states)
            if halves is not None:
                rows = [halves.take(normed, i) for i in range(2)]
                update = self._attention_halves(positions, rows, halves)
                shard = update + _halves_rows(hidden_states, self.tp_size)
                update = None
            elif sp_o_proj.applies(o_proj, hidden_states.shape[0]):
                attn, weight = self._sp_attention(positions, normed, whole)
                rows = sp_o_proj.local_rows(hidden_states)
                shard = sp_o_proj.sp_o_proj(o_proj, attn, weight=weight) + rows
            else:
                attn = self.self_attn(positions, normed, half=whole)
                shard = tensor_model_parallel_reduce_scatter(
                    attn + hidden_states / self.tp_size, 0
                )
        else:
            gather = (
                self._norm_gather_pieces
                if envs.TPU_MINIMAX_M3_SP_ATTN_PIECES
                else self._norm_gather
            )
            shard, normed = gather(
                self.input_layernorm,
                shard,
                update,
                envs.TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER,
                envs.TPU_MINIMAX_M3_SP_E4M3_QKV,
            )
            layer_input = shard
            if halves is not None:
                update = self._attention_halves(positions, normed, halves)
            elif sp_o_proj.applies(o_proj, shard.shape[0] * self.tp_size):
                attn, weight = self._sp_attention(positions, normed, whole)
                pieces = (
                    _ffn_pieces(shard.shape[0], self.tp_size)
                    if envs.TPU_MINIMAX_M3_SP_ATTN_PIECES
                    else 1
                )
                update = sp_o_proj.sp_o_proj(
                    o_proj, attn, pieces, weight=weight, weight_after=_last(normed)
                )
            else:
                attn = self.self_attn(positions, normed, half=whole)
                update = tensor_model_parallel_reduce_scatter(attn, 0)
        shard, update = self._ffn_sp(shard, update, halves is not None)
        return layer_input, shard, update

    def _sp_attention(
        self,
        positions: torch.Tensor,
        normed: torch.Tensor | tuple | list,
        whole: tuple[StepHalves, None] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """This rank's heads of an SP step's attention for o_proj's
        all-to-all path, and o_proj's gathered weight blocks when the
        attention op gathers them (`sp_o_proj.gathers_with_attention`)."""
        num_tokens = positions.shape[0]
        if sp_o_proj.gathers_with_attention(self.self_attn, num_tokens):
            return self.self_attn(
                positions, normed, project=False, half=whole, gather_o_proj=True
            )
        return self.self_attn(positions, normed, project=False, half=whole), None

    def _attention_halves(
        self, positions: torch.Tensor, rows: list, halves: StepHalves
    ) -> torch.Tensor:
        """o_proj(attention) summed over TP for this rank's block of each
        token half (the halves layout): the first half attends (and writes its
        keys) before the second; the first half's all-to-all + o_proj runs
        after the second half's attention (its all-to-all crosses under it),
        the second half's right after its own attention."""
        o_proj = self.self_attn.o_proj
        attns: list[torch.Tensor] = []
        for i, x in enumerate(rows):
            # The second half's cache writes follow the first half's reads.
            attns.append(
                self.self_attn(
                    positions,
                    x,
                    project=False,
                    half=(halves, i),
                    after=attns[-1] if attns else None,
                )
            )
        # The first half's o_proj follows the second half's attention, so its
        # all-to-all lands under that attention instead of before it.
        return torch.cat(
            [
                sp_o_proj.sp_o_proj(o_proj, attns[0], after=attns[1]),
                sp_o_proj.sp_o_proj(o_proj, attns[1]),
            ]
        )

    def _ffn_sp(
        self, shard: torch.Tensor, update: torch.Tensor | None, halves: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(shard + update, reduce-scattered FFN(all-gathered norm of it)).
        With row pieces (_ffn_pieces) each piece of the shard is normed and
        all-gathered on its own, so the FFN can start its row-local entry
        work (dequant, router) on a piece while the next one lands; the FFN
        runs once over the gathered rows in piece-major order, and its output
        is reduce-scattered per piece, so a piece's reduce-scatter can start
        when its rows are combined (TPU_MOE_TC_COMBINE_PIECES)."""
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        tp = self.tp_size
        n = (
            _ffn_pieces(shard.shape[0], tp)
            if envs.TPU_MINIMAX_M3_SP_FFN_GATHER_PIECES
            else 1
        )
        gather = self._norm_gather_pieces if n > 1 else self._norm_gather
        shard, gathered = gather(
            self.post_attention_layernorm,
            shard,
            update,
            envs.TPU_MINIMAX_M3_SP_FP8_GATHER,
            envs.TPU_MINIMAX_M3_SP_E4M3_QKV,
        )
        if n == 1:
            return shard, tensor_model_parallel_reduce_scatter(ffn(gathered), 0)
        # The MoE's padding mask is indexed by row: give it piece-major rows
        # (the halves layout gathers the step's rows in order).
        rows = (
            _identity_rows
            if halves
            else functools.partial(_piece_major_rows, shard.shape[0] * tp, tp, n)
        )
        if envs.TPU_MINIMAX_M3_SP_DENSE_FFN_PER_PIECE and not self.is_moe_layer:
            per = shard.shape[0] * tp // n
            out_updates = []
            for i, piece in enumerate(gathered):
                piece_rows = functools.partial(_rows_slice, rows, i * per, per)
                with token_padding.padding_of_rows(piece_rows):
                    out = ffn(piece)
                out_updates.append(tensor_model_parallel_reduce_scatter(out, 0))
            return shard, torch.cat(out_updates)
        with (
            contextlib.nullcontext() if halves else token_padding.padding_of_rows(rows)
        ):
            out = ffn(gathered)
        chunks = out if isinstance(out, list) else out.chunk(n)
        out_updates = [tensor_model_parallel_reduce_scatter(o, 0) for o in chunks]
        return shard, torch.cat(out_updates)

    def _norm_gather_pieces(
        self,
        norm: GemmaRMSNorm,
        shard: torch.Tensor,
        update: torch.Tensor | None,
        fp8: bool,
        e4m3_out: bool,
    ) -> tuple[torch.Tensor, object]:
        """`_norm_gather` per row piece of the shard (_ffn_pieces): the
        gathered pieces come as a list, so their consumer can start on a
        piece while the next lands; one piece gives `_norm_gather`'s result."""
        n = _ffn_pieces(shard.shape[0], self.tp_size)
        if n == 1:
            return self._norm_gather(norm, shard, update, fp8, e4m3_out)
        updates = (None,) * n if update is None else update.chunk(n)
        shards, gathered = [], []
        for piece, piece_update in zip(shard.chunk(n), updates):
            piece, normed = self._norm_gather(norm, piece, piece_update, fp8, e4m3_out)
            shards.append(piece)
            gathered.append(normed)
        return torch.cat(shards), gathered


def _sp_mode() -> str:
    return envs.TPU_MINIMAX_M3_SP_MODE if envs.TPU_MINIMAX_M3_SP_MIN_TOKENS else "off"


def _sp_step(num_tokens: int, tp_size: int) -> bool:
    """Whether a step of ``num_tokens`` runs the sequence-parallel path.

    Only inside a warmup trace, which gives each side of this branch its own
    trace (compilation/shape_variants.py); eager forwards and any other trace
    keep the plain path.
    """
    return bool(
        torch.compiler.is_compiling()
        and shape_variants.aot_cache_tag()
        and num_tokens >= envs.TPU_MINIMAX_M3_SP_MIN_TOKENS
        and num_tokens % tp_size == 0
    )


def _piece_major_rows(
    tokens: int, tp_size: int, pieces: int, x: torch.Tensor
) -> torch.Tensor:
    """A step-row tensor ``x`` (at least ``tokens`` rows) in the gathered
    pieces' order: piece i holds every rank's i-th row block of its shard."""
    x = x[:tokens].view(tp_size, pieces, -1, *x.shape[1:])
    return x.transpose(0, 1).flatten(0, 2)


def _last(gathered: object) -> torch.Tensor:
    """The rows of the last row piece a gather returned: the piece, or of
    e4m3 rows with their scales the rows (the scales take their own, smaller
    gather, which can end first)."""
    while isinstance(gathered, list):
        gathered = gathered[-1]
    return gathered[0] if isinstance(gathered, tuple) else gathered


def _identity_rows(x: torch.Tensor) -> torch.Tensor:
    return x


def _halves_rows(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    """This rank's shard of step rows ``x`` in the halves layout: its block
    of the first token half, then its block of the second."""
    x = x.view(2, tp_size, x.shape[0] // (2 * tp_size), *x.shape[1:])
    rank = pallas_all_reduce.armed_rank(get_tp_group().ranks)
    if rank is None:
        raise RuntimeError("the SP halves layout needs the armed, rank-ordered mesh")
    return torch.index_select(x, 1, rank.view(1)).flatten(0, 2)


def _gather_shards(x: torch.Tensor, halves: bool) -> torch.Tensor:
    """The step rows of a residual shard (all-gathered; the halves layout
    gathered per half, so the rows come back in step order)."""
    rows = tensor_model_parallel_all_gather(x, 0)
    if not halves:
        return rows
    # [rank][half][block] -> [half][rank][block]
    rows = rows.view(-1, 2, x.shape[0] // 2, *x.shape[1:]).transpose(0, 1)
    return rows.flatten(0, 2)


def _rows_slice(
    rows: Callable[[torch.Tensor], torch.Tensor],
    start: int,
    length: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return rows(x)[start : start + length]


def _ffn_pieces(shard_rows: int, tp_size: int) -> int:
    """Row pieces of an SP layer's gathers (TPU_MINIMAX_M3_SP_FFN_PIECES on
    steps of at least TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS tokens); 1
    unless every piece keeps whole 32-row e4m3 tiles (a piece that does not
    fill whole sp_norm blocks takes the unfused norm, as such a shard does)."""
    pieces = envs.TPU_MINIMAX_M3_SP_FFN_PIECES
    if (
        pieces <= 1
        or shard_rows * tp_size < envs.TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS
        or shard_rows % (pieces * _PIECE_ROW_TILE)
    ):
        return 1
    return pieces


_PIECE_ROW_TILE = 32


_FP8_MAX = 448.0


def _all_gather_rows(x: torch.Tensor, fp8: bool) -> torch.Tensor:
    """All-gather token rows over TP; ``fp8`` sends each bf16 row as e4m3 with
    a per-row f32 scale (gathered natively: a bitcast to a wider carrier costs
    an 8-bit relayout per call)."""
    if not fp8:
        return tensor_model_parallel_all_gather(x, 0)
    q, scale = _gather_rows_e4m3(*_quantize_rows(x))
    return (q.float() * scale).to(x.dtype)


def _fits_sp_norm(shard: torch.Tensor) -> bool:
    """Whether `sp_norm.add_norm_quant` tiles this shard's rows."""
    rows = shard.shape[0]
    return rows <= sp_norm.BLOCK_ROWS or rows % sp_norm.BLOCK_ROWS == 0


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 rows of ``x`` and their per-row f32 scales ``amax / 448``."""
    scale = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / _FP8_MAX
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def _gather_rows_e4m3(
    q: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = pallas_all_reduce.all_gather_small(get_tp_group().ranks, scale)
    if scales is None:
        scales = tensor_model_parallel_all_gather(scale, 0)
    return tensor_model_parallel_all_gather(q, 0), scales


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0}
)
class MiniMaxM3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        # Keys this model's compile-artifact cache directory.
        self.prefix = prefix
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleList(
            MiniMaxM3DecoderLayer(vllm_config, idx, prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.num_hidden_layers)
        )
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.sp_mode = _sp_mode() if self.tp_size > 1 else "off"
        # Per-half attention (StepHalves) is checked on 8-bit caches with one
        # KV head per rank (head-major queries). On 4-bit caches the second
        # half's attention reads NaN once its cache writes are ordered after
        # the first half's reads (and XLA copies the caches when they are
        # not); a DCP group over several KV heads (c64's DCP4) halted the TPU.
        attn = self.layers[0].self_attn
        self.step_halves = bool(
            envs.TPU_MINIMAX_M3_SP_HALVES
            and attn.q_head_major
            and attn.scale_cache is None
        )
        if envs.TPU_MINIMAX_M3_SP_HALVES and not self.step_halves:
            logger.warning(
                "TPU_MINIMAX_M3_SP_HALVES needs 8-bit KV caches and DCP over one "
                "KV head per rank: off"
            )
        # Residual-stream entries of these layers (k = output of layer k-1,
        # residual added, before the next norm) are also returned; set by
        # the DSpark drafter before the first trace.
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        # Set by residual_ar.arm: layers fuse their TP all-reduces.
        self.residual_ar_armed = False

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _halves_step(self, num_tokens: int) -> bool:
        """Whether this SP step runs attention per token half
        (TPU_MINIMAX_M3_SP_HALVES): on steps whose shard splits into the two
        row pieces of `_ffn_pieces`, gathered per piece on both sides, with
        o_proj's all-to-all path (whose rank order `_halves_rows` reads)."""
        return bool(
            self.step_halves
            and envs.TPU_MINIMAX_M3_SP_ATTN_PIECES
            and envs.TPU_MINIMAX_M3_SP_FFN_GATHER_PIECES
            and _ffn_pieces(num_tokens // self.tp_size, self.tp_size) == 2
            and sp_o_proj.applies(self.layers[0].self_attn.o_proj, num_tokens)
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        step = None
        if self.step_halves:
            # First thing in every trace (StepHalves).
            tokens = input_ids if inputs_embeds is None else inputs_embeds
            step = StepHalves(positions, tokens.shape[0])
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        aux_hidden_states = []
        if 0 in self.aux_hidden_state_layers:
            aux_hidden_states.append(inputs_embeds)
        sp_step = self.sp_mode != "off" and _sp_step(
            inputs_embeds.shape[0], self.tp_size
        )
        if sp_step and self.sp_mode == "full":
            shard, update = None, None
            num_tokens = inputs_embeds.shape[0]
            halves = step if step and self._halves_step(num_tokens) else None
            for idx, layer in enumerate(self.layers):
                # Layer idx materializes its input residual (entry idx).
                layer_input, shard, update = layer.forward_sp(
                    positions, inputs_embeds, shard, update, halves, step
                )
                if idx > 0 and idx in self.aux_hidden_state_layers:
                    aux_hidden_states.append(
                        _gather_shards(layer_input, halves is not None)
                    )
            shard = shard + update
            if len(self.layers) in self.aux_hidden_state_layers:
                aux_hidden_states.append(_gather_shards(shard, halves is not None))
            hidden_states = _gather_shards(self.norm(shard), halves is not None)
        elif self.residual_ar_armed and not sp_step:
            partial, residual = inputs_embeds, None
            for idx, layer in enumerate(self.layers):
                partial, residual, entry = layer.forward_residual_ar(
                    positions, partial, residual
                )
                if idx > 0 and idx in self.aux_hidden_state_layers:
                    aux_hidden_states.append(entry)
            hidden_states = tensor_model_parallel_all_reduce(partial)
            if len(self.layers) in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states, residual = inputs_embeds, None
            for idx, layer in enumerate(self.layers):
                hidden_states, residual = layer(
                    positions, hidden_states, residual, sp_step, step
                )
                if idx + 1 in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states + residual)
            hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # (param, checkpoint shard, shard id). Leading dots keep `.q_proj`
        # from matching `index_q_proj`.
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            name = name.replace(".weight_scale_inv", ".weight_scale")
            if ".experts." in name:
                for ct_suffix, modelopt_suffix in CT_TO_MODELOPT_EXPERT_SUFFIX.items():
                    name = name.replace(ct_suffix, modelopt_suffix)
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    raise ValueError(f"Unexpected expert weight {name}")
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        bound = max(layer.self_attn.score_bound() for layer in self.layers)
        on = msa_attention_lse.configure_max_free(bound)
        logger.info("MiniMax-M3 MSA score bound %.1f: max-free softmax %s", bound, on)
        msa_attention_lse.WIDE_TILES = envs.TPU_MINIMAX_M3_MSA_WIDE_TILES
        return loaded_params


class MiniMaxM3SparseForCausalLM(nn.Module):
    """Text-only MiniMax M3; also serves the vision-language architecture."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "",
            "vision_tower.": None,
            "multi_modal_projector.": None,
            "patch_merge_mlp.": None,
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.model = MiniMaxM3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        if envs.TPU_FP8_LM_HEAD:
            # A head_dtype override reads lm_head.weight unscaled.
            if getattr(self.logits_processor, "head_dtype", None) not in (
                None,
                torch.bfloat16,
            ):
                raise NotImplementedError(
                    "TPU_FP8_LM_HEAD does not support a head_dtype override."
                )
            self.lm_head.quant_method = Fp8LMHeadMethod()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        num_layers = len(self.model.layers)
        invalid = [idx for idx in layers if not 0 <= idx <= num_layers]
        if invalid:
            raise ValueError(f"Invalid MiniMax M3 aux hidden-state layers: {invalid}")
        self.model.aux_hidden_state_layers = tuple(layers)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if intermediate_tensors is not None:
            raise NotImplementedError("MiniMax M3 on TPU does not support PP")
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_shard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's vocab shard of ``compute_logits`` (pad columns
        included), without the all-gather."""
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
