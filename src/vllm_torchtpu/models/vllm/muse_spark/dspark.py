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
"""TPU Muse Spark DSpark speculative draft model on upstream vLLM v0.33.0.

Subclasses upstream ``vllm.models.muse_spark.common.dspark.MuseSparkDSparkForCausalLM``
and adapts the draft stack for TPU:
1. :class:`TPUMuseSparkDSparkDecoderLayer` builds on :class:`TPUMuseSparkDecoderLayer`
   (instead of upstream's hardcoded NVIDIA decoder layer), enforces causal
   drafting, keeps ``FullAttentionSpec`` until megakernel SWA draft support
   lands, and uses NEOX FP32 RoPE.
2. :class:`TPUMuseSparkDSparkModel` preserves the weightless FP32 input
   embedding RMSNorm used by TPU DSpark training/megakernel semantics.
3. :class:`TPUMuseSparkDSparkForCausalLM` permutes checkpoint Q/K rows from
   interleaved pairs into NEOX halves inside the ``load_weights`` iterator
   (before TP slicing or TPU ``WEIGHT_FLIPPED_ATTR`` transposes), rejects
   ``reload_weights`` post-flip, implements ``tpu_precompute_and_store_context_kv``
   (skipping RoPE on the inline zero dummy query), and applies the Metap output
   multiplier / softcap before Markov bias addition.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from typing import Any, cast

import torch
from vllm.config import VllmConfig

from vllm_torchtpu.models.vllm.muse_spark import compat

compat.apply()

import vllm.models.muse_spark.common.dspark as up_dspark  # noqa: E402
from vllm.models.muse_spark.common.dspark import (  # noqa: E402
    MuseSparkDSparkForCausalLM,
    MuseSparkDSparkModel,
)
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNorm  # noqa: E402

from vllm_torchtpu.layers.adapter.linear_common import (  # noqa: E402
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.models.vllm.muse_spark.model import (  # noqa: E402
    TPUMuseSparkDecoderLayer,
)


class TPUMuseSparkDSparkDecoderLayer(TPUMuseSparkDecoderLayer):
    """Dense Muse Spark DSpark draft decoder layer on TPU."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if getattr(self, "is_sequence_parallel", False):
            self.is_sequence_parallel = False
            self.is_sequence_parallel_input = False
            self.self_attn.is_sequence_parallel_output = False
            self.self_attn.o_proj.reduce_results = True

        cfg = vllm_config.model_config.hf_config
        dcfg = getattr(cfg, "dflash_config", None) or {}
        if dcfg.get("causal", True) is not True:
            raise NotImplementedError(
                "Muse Spark DSpark on TPU supports causal drafts only."
            )
        self.self_attn.causal = True
        if (self.self_attn.rotary_emb is not None) != bool(cfg.use_rope):
            raise ValueError(
                "Muse Spark DSpark layers must share their positional mode."
            )

        # TODO(PR-C2): keep draft KV pools as FullAttentionSpec until the
        # megakernel plugin's draft ring sync and writeback support SWA pools.
        self.self_attn.attn.sliding_window = None

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        del residual
        stream = hidden_states.to(self.residual_dtype)
        attn_out = self.self_attn(
            positions=positions,
            hidden_states=self.self_attn.pre_qkv_proj_norm(
                stream.to(self.default_dtype), out_dtype=self.default_dtype
            ),
        )
        stream = self.add_attn(stream=stream, branch=self.post_attn_norm(attn_out))
        ffn_out = self.feed_forward(self.pre_ffn_norm(stream.to(self.default_dtype)))
        stream = self.add_ffn(stream=stream, branch=self.post_ffn_norm(ffn_out))
        return stream, None


class TPUMuseSparkDSparkModel(MuseSparkDSparkModel):
    """Muse DSpark backbone with weightless FP32 input-embedding RMSNorm."""

    layers: Any
    _context_pre_norms: tuple[Any, ...]
    _stacked_kv_weight: torch.Tensor
    _num_attn_layers: int
    _kv_size: int
    _head_dim: int
    _num_kv_heads: int
    _rms_norm_eps: float
    _rotary_emb: Any
    _attn_layers: list[Any]

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        # Expose prefix before @support_torch_compile's wrapper __init__ runs
        # so TPU compile_prefix cache isolation is deterministic.
        object.__setattr__(self, "prefix", prefix)
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        self.embed_norm = MuseSpark1xRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            has_weight=False,
        )

    def _build_fused_kv_buffers(self) -> None:
        """Build stacked context K/V weights for pre- and post-flip layouts."""
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        self._context_pre_norms = tuple(attn.pre_qkv_proj_norm for attn in layers_attn)
        kv_slices: list[torch.Tensor] = []
        for attn in layers_attn:
            w = attn.qkv_proj.weight
            if getattr(attn.qkv_proj, WEIGHT_FLIPPED_ATTR, False):
                kv = w[:, attn.q_size : attn.q_size + 2 * attn.kv_size].t()
            else:
                kv = w[attn.q_size : attn.q_size + 2 * attn.kv_size]
            kv_slices.append(kv.contiguous())
        self._stacked_kv_weight = torch.stack(kv_slices, dim=0)
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.qk_norm.variance_epsilon
        self._rotary_emb = attn0.rotary_emb
        self._attn_layers = [attn.attn for attn in layers_attn]
        for attn in layers_attn[1:]:
            if (
                attn.kv_size != self._kv_size
                or attn.head_dim != self._head_dim
                or attn.num_kv_heads != self._num_kv_heads
                or attn.qk_norm.variance_epsilon != self._rms_norm_eps
            ):
                raise ValueError("Muse Spark DSpark layers must share KV geometry")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.embed_norm.forward_native(super().embed_input_ids(input_ids).float()),
        )


@contextlib.contextmanager
def _swap_upstream_dspark_classes() -> Iterator[None]:
    saved_layer = up_dspark.MuseSparkDSparkDecoderLayer
    saved_model = up_dspark.MuseSparkDSparkModel
    up_dspark.MuseSparkDSparkDecoderLayer = TPUMuseSparkDSparkDecoderLayer  # type: ignore[misc]
    up_dspark.MuseSparkDSparkModel = TPUMuseSparkDSparkModel  # type: ignore[misc]
    try:
        yield
    finally:
        up_dspark.MuseSparkDSparkDecoderLayer = saved_layer  # type: ignore[misc]
        up_dspark.MuseSparkDSparkModel = saved_model  # type: ignore[misc]


class TPUMuseSparkDSparkForCausalLM(MuseSparkDSparkForCausalLM):
    """TPU Muse Spark DSpark draft CausalLM."""

    # pyrefly: ignore[bad-override-mutable-attribute]
    model: TPUMuseSparkDSparkModel
    has_own_lm_head = True
    has_own_embed_tokens = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        assert vllm_config.speculative_config is not None
        dtp = vllm_config.speculative_config.draft_tensor_parallel_size
        if dtp not in (None, vllm_config.parallel_config.tensor_parallel_size):
            raise NotImplementedError(
                "Muse Spark DSpark on TPU requires draft_tensor_parallel_size "
                f"to match target tensor_parallel_size (got draft_tp={dtp}, "
                f"target_tp={vllm_config.parallel_config.tensor_parallel_size})."
            )
        with _swap_upstream_dspark_classes():
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if any(
            getattr(layer.self_attn.qkv_proj, WEIGHT_FLIPPED_ATTR, False)
            for layer in self.model.layers
        ):
            raise NotImplementedError(
                "reload_weights is not supported for the Muse Spark DSpark draft: "
                "qkv_proj weights are already transposed to TPU [in, out] layout."
            )

        head_dim = int(self.config.head_dim)

        def _neox_qk_weights(
            items: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in items:
                if (
                    (".self_attn.q_proj." in name or ".self_attn.k_proj." in name)
                    and name.endswith("weight")
                    and tensor.ndim == 2
                ):
                    # DSpark exports Q/K rows as interleaved pairs per head;
                    # permute to NEOX halves [even | odd] before TP slicing.
                    tensor = (
                        tensor.view(-1, head_dim // 2, 2, tensor.shape[-1])
                        .transpose(1, 2)
                        .reshape(tensor.shape)
                        .contiguous()
                    )
                yield name, tensor

        loaded = super().load_weights(_neox_qk_weights(weights))
        m = self.model
        attn0 = m.layers[0].self_attn
        expected_shape = (
            len(m.layers),
            2 * attn0.kv_size,
            int(self.config.hidden_size),
        )
        assert tuple(m._stacked_kv_weight.shape) == expected_shape, (
            f"Expected _stacked_kv_weight shape {expected_shape}, "
            f"got {tuple(m._stacked_kv_weight.shape)}."
        )
        compat.sync_residual_constants(self.model)
        return loaded

    def process_weights_after_loading(self) -> None:
        """Ensure stacked context-KV buffers exist on dummy loads and sync on TPU."""
        if not hasattr(self.model, "_num_attn_layers"):
            self.model._build_fused_kv_buffers()
        if self.model._stacked_kv_weight.device.type == "tpu":
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors([self.model._stacked_kv_weight])
        compat.sync_residual_constants(self.model)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raw_logits = super().compute_draft_logits(hidden_states)
        return self.apply_output_logit_transform(raw_logits)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return super().markov_bias(markov_embed).float() * self.output_multiplier

    def tpu_precompute_and_store_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        draft_md_tuple: tuple[Any, ...],
    ) -> torch.Tensor:
        """Fused multi-layer draft context K/V projection and RPA cache write.

        Executes inside ``@torch.compile(backend="tpu", fullgraph=True,
        dynamic=False)``: allocates the dummy zero query inline (constant-folded
        by XLA), rotates only ``K`` in FP32 so ``cos/sin`` remain FP32, and
        returns the narrow ``ctx`` tensor for warmup synchronization.
        """
        m = self.model
        if not hasattr(m, "_num_attn_layers"):
            m._build_fused_kv_buffers()
        ctx = cast(Any, self.combine_hidden_states)(target_hidden)
        kv = m._project_muse_context_kv(ctx)
        num_layers, num_tokens = m._num_attn_layers, ctx.shape[0]
        kv = kv.view(num_layers, num_tokens, 2, m._num_kv_heads, m._head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]

        kf = k.float()
        kf = kf * torch.rsqrt(kf.pow(2).mean(dim=-1, keepdim=True) + m._rms_norm_eps)
        if m._rotary_emb is not None:
            kf, _ = m._rotary_emb(
                positions.repeat(num_layers),
                kf.reshape(num_layers * num_tokens, m._num_kv_heads, m._head_dim),
                None,
            )
        k_out = kf.reshape(num_layers, num_tokens, m._num_kv_heads, m._head_dim).to(
            v.dtype
        )

        for i, attn in enumerate(m._attn_layers):
            q0 = torch.zeros(
                (num_tokens, attn.num_heads, attn.head_size),
                dtype=v.dtype,
                device=v.device,
            )
            attn.impl.forward(
                layer=attn,
                query=q0,
                key=k_out[i],
                value=v[i],
                kv_cache=attn.kv_cache,
                attn_metadata=draft_md_tuple[i],
            )
        return ctx


__all__ = [
    "TPUMuseSparkDSparkDecoderLayer",
    "TPUMuseSparkDSparkForCausalLM",
    "TPUMuseSparkDSparkModel",
]
