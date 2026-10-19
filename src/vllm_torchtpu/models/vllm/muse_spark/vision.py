# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel NaViT with variable-resolution 2D RoPE and image attention."""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import (
    should_torch_compile_mm_encoder,
    support_torch_compile,
)
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import is_vit_use_data_parallel

from .config import MuseSparkVisionConfig


def _precompute_rope_freqs_cis(
    dim: int,
    max_height: int,
    max_width: int,
    dtype: torch.dtype,
    theta: float = 10000.0,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Build width-first, one-indexed RoPE [H, W, dim//2, 2] as real cos/sin pairs."""
    assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
    half_dim = dim // 2
    freq_range = torch.arange(0, half_dim, 2, dtype=dtype, device=device)[
        : (half_dim // 2)
    ]
    inv_freq = 1.0 / (theta ** (freq_range / half_dim))

    # NaViT uses one-indexed positions.
    pos_h = torch.arange(1, max_height + 1, dtype=dtype, device=device)
    pos_w = torch.arange(1, max_width + 1, dtype=dtype, device=device)
    mesh_h, mesh_w = torch.meshgrid(pos_h, pos_w, indexing="ij")

    freq_h = torch.einsum("h w, f -> h w f", mesh_h, inv_freq)
    freq_w = torch.einsum("h w, f -> h w f", mesh_w, inv_freq)

    # Width precedes height in the frequency grid.
    freq = torch.cat([freq_w, freq_h], dim=-1)
    return torch.stack([torch.cos(freq), torch.sin(freq)], dim=-1)


def _get_pixel_shuffle_perm(
    cur_ds_idx: int,
    grid_h: int,
    grid_w: int,
    downsample_factor: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Compute permutation indices for pixel-shuffle downsampling."""
    ds_perm = torch.arange(
        cur_ds_idx,
        cur_ds_idx + grid_h * grid_w,
        device=device,
    )
    h = grid_h // downsample_factor
    w = grid_w // downsample_factor
    ds_perm = (
        ds_perm.reshape(h, downsample_factor, w, downsample_factor)
        .permute(0, 2, 1, 3)
        .reshape(-1)
    )
    assert ds_perm.shape[0] == grid_h * grid_w
    cur_ds_idx += grid_h * grid_w
    return ds_perm, cur_ds_idx


def _downsample_pixel_shuffle(
    x: torch.Tensor,
    batch_ds_perm: torch.Tensor,
    downsample_factor: int,
) -> torch.Tensor:
    """Pixel-shuffle downsample: rearrange neighbor patches, concat features."""
    ds_x = x[batch_ds_perm]
    b = x.shape[0] // (downsample_factor**2)
    r = downsample_factor**2
    d = ds_x.shape[-1]
    ds_x = ds_x.reshape(b, r, d).transpose(1, 2).reshape(b, d * r)
    return ds_x


def _get_avg_pooling_perm(
    cur_ds_idx: int,
    grid_h: int,
    grid_w: int,
    downsample_factor: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Compute permutation indices for average-pooling downsampling."""
    h = grid_h // downsample_factor
    w = grid_w // downsample_factor
    ds_perm = torch.arange(
        cur_ds_idx,
        cur_ds_idx + h * w,
        device=device,
    )
    ds_perm = (
        ds_perm.reshape(h, w)[:, None, :, None]
        .expand(h, downsample_factor, w, downsample_factor)
        .reshape(-1)
    )
    assert ds_perm.shape[0] == grid_h * grid_w
    cur_ds_idx += h * w
    return ds_perm, cur_ds_idx


def _downsample_avg_pooling(
    x: torch.Tensor,
    batch_ds_perm: torch.Tensor,
    downsample_factor: int,
) -> torch.Tensor:
    """Apply average-pooling downsampling."""
    ds_x = x.new_empty(x.shape[0] // (downsample_factor**2), x.shape[1])
    ds_x.index_reduce_(
        dim=0,
        index=batch_ds_perm,
        source=x,
        reduce="mean",
        include_self=False,
    )
    return ds_x


def _inv_perm_by_idx(perm_src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Invert a permutation given by an index tensor."""
    inv_src = torch.empty_like(perm_src)
    inv_src.index_copy_(0, idx, perm_src)
    return inv_src


def _iterate_visions(
    visions: Sequence[torch.Tensor],
    patch_temporal: int,
) -> list[torch.Tensor]:
    """Iterate over vision tensors, yielding images or temporal clips."""
    results = []
    for vision in visions:
        if vision.ndim == 3:
            results.append(vision)
        elif vision.ndim == 4:
            assert vision.shape[0] % patch_temporal == 0, (
                f"Video frame count {vision.shape[0]} is not divisible by "
                f"patch_temporal {patch_temporal}. This should have been "
                "handled by the preprocessor."
            )
            for i in range(0, vision.shape[0], patch_temporal):
                results.append(vision[i : i + patch_temporal])
    return results


def _get_sparse_attention_perm(
    cur_sp_idx: int,
    grid_h: int,
    grid_w: int,
    pos_emb_grid_height: int,
    pos_emb_grid_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute permutation and sequence lengths for sparse attention."""
    padded_grid_h = math.ceil(grid_h / pos_emb_grid_height) * pos_emb_grid_height
    padded_grid_w = math.ceil(grid_w / pos_emb_grid_width) * pos_emb_grid_width

    sp_perm = torch.arange(
        cur_sp_idx,
        cur_sp_idx + grid_h * grid_w,
        device=device,
    ).view(grid_h, grid_w)
    sp_perm = F.pad(
        sp_perm,
        (0, padded_grid_w - grid_w, 0, padded_grid_h - grid_h),
        value=-1,
    ).flatten()

    h = padded_grid_h // pos_emb_grid_height
    w = padded_grid_w // pos_emb_grid_width
    r1 = pos_emb_grid_height
    r2 = pos_emb_grid_width

    sp_perm = sp_perm.reshape(h, r1, w, r2).permute(0, 2, 1, 3).reshape(-1)
    sp_perm = sp_perm[sp_perm != -1]

    sp_slen = torch.ones((grid_h, grid_w), device=device, dtype=torch.int)
    sp_slen = F.pad(
        sp_slen,
        (0, padded_grid_w - grid_w, 0, padded_grid_h - grid_h),
        value=0,
    ).flatten()
    sp_slen = sp_slen.reshape(h, r1, w, r2).permute(0, 2, 1, 3).reshape(h * w, r1 * r2)
    sp_slen = sp_slen.sum(dim=1)
    cur_sp_idx += grid_h * grid_w

    return sp_perm, sp_slen, cur_sp_idx


def _seqlens_to_cu_seqlens(
    seqlens: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int32 cumulative lengths on device and a CPU maximum to avoid sync."""
    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    torch.cumsum(seqlens_t, dim=0, out=cu_seqlens[1:])
    max_seqlen_cpu = torch.tensor(max(seqlens), dtype=torch.int32, device="cpu")
    return cu_seqlens, max_seqlen_cpu


class NaViTAttention(nn.Module):
    """Multi-head attention with fused QKV, 2D RoPE, and MMEncoderAttention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_rope: bool = True,
        prefix: str = "",
        use_data_parallel: bool = False,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.use_data_parallel = use_data_parallel

        self.qkv_proj = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            prefix=maybe_prefix(prefix, "qkv_proj"),
            disable_tp=use_data_parallel,
            quant_config=quant_config,
        )
        # Apply row-parallel bias after the TP reduction, matching the reference.
        self.proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=True,
            skip_bias_add=True,
            prefix=maybe_prefix(prefix, "proj"),
            disable_tp=use_data_parallel,
            quant_config=quant_config,
        )

        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        self.num_heads = divide(num_heads, tp_size)
        self.scaling = self.head_dim**-0.5

        self.attn = MMEncoderAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            prefix=maybe_prefix(prefix, "attn"),
        )

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freq_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = freq_cis[..., 0].unsqueeze(1)
        sin = freq_cis[..., 1].unsqueeze(1)
        q0, q1 = q.float().chunk(2, dim=-1)
        k0, k1 = k.float().chunk(2, dim=-1)
        q_rotated = torch.cat((q0 * cos - q1 * sin, q0 * sin + q1 * cos), dim=-1)
        k_rotated = torch.cat((k0 * cos - k1 * sin, k0 * sin + k1 * cos), dim=-1)
        return q_rotated.type_as(q), k_rotated.type_as(k)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over concatenated patches with per-image sequence boundaries."""
        seq_len = x.shape[0]

        qkv, _ = self.qkv_proj(x)

        q_size = self.num_heads * self.head_dim
        q = qkv[:, :q_size].reshape(seq_len, self.num_heads, self.head_dim)
        k = qkv[:, q_size : 2 * q_size].reshape(seq_len, self.num_heads, self.head_dim)
        v = qkv[:, 2 * q_size :].reshape(seq_len, self.num_heads, self.head_dim)

        if self.use_rope and freq_cis is not None:
            q, k = self._apply_rope(q, k, freq_cis)

        out = self.attn(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        out = out.reshape(seq_len, -1)

        out, out_bias = self.proj(out)
        if out_bias is not None:
            out = out + out_bias
        return out


class NaViTMLP(nn.Module):
    """GELU MLP built from vLLM parallel layers."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        prefix: str = "",
        use_data_parallel: bool = False,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_size=dim,
            output_size=hidden_dim,
            bias=True,
            prefix=maybe_prefix(prefix, "fc1"),
            disable_tp=use_data_parallel,
            quant_config=quant_config,
        )
        # Apply bias after the TP reduction.
        self.fc2 = RowParallelLinear(
            input_size=hidden_dim,
            output_size=dim,
            bias=True,
            skip_bias_add=True,
            prefix=maybe_prefix(prefix, "fc2"),
            disable_tp=use_data_parallel,
            quant_config=quant_config,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.fc1(x)
        h = self.act(h)
        out, out_bias = self.fc2(h)
        if out_bias is not None:
            out = out + out_bias
        return out


class NaViTBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attention -> Residual -> LN -> MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        intermediate_size: int | None = None,
        use_rope: bool = True,
        norm_eps: float = 1e-5,
        prefix: str = "",
        use_data_parallel: bool = False,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = NaViTAttention(
            dim=dim,
            num_heads=num_heads,
            use_rope=use_rope,
            prefix=maybe_prefix(prefix, "attn"),
            use_data_parallel=use_data_parallel,
            quant_config=quant_config,
        )
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        # Explicit intermediate_size takes precedence over mlp_ratio.
        self.mlp = NaViTMLP(
            dim=dim,
            hidden_dim=(
                intermediate_size
                if intermediate_size is not None
                else int(mlp_ratio * dim)
            ),
            prefix=maybe_prefix(prefix, "mlp"),
            use_data_parallel=use_data_parallel,
            quant_config=quant_config,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(
            h, freq_cis=freq_cis, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
        "freq_cis": 0,
        "cu_seqlens": 0,
        "sparse_cu_seqlens": 0,
    },
    enable_if=should_torch_compile_mm_encoder,
    is_encoder=True,
)
class NaViTEncoder(nn.ModuleList):
    """Stack of NaViTBlock transformer layers."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        intermediate_size: int | None = None,
        use_rope: bool = True,
        layer_types: list[str] | None = None,
        norm_eps: float = 1e-5,
        prefix: str = "",
        use_data_parallel: bool = False,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(
            [
                NaViTBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    intermediate_size=intermediate_size,
                    use_rope=use_rope,
                    norm_eps=norm_eps,
                    prefix=maybe_prefix(prefix, str(i)),
                    use_data_parallel=use_data_parallel,
                    quant_config=quant_config,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.layer_types = layer_types or ["full_attention"] * num_layers

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sparse_cu_seqlens: torch.Tensor | None = None,
        sparse_max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for i, layer in enumerate(self):
            if self.layer_types[i] == "full_attention":
                x = layer(
                    x,
                    freq_cis=freq_cis,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            else:
                x = layer(
                    x,
                    freq_cis=freq_cis,
                    cu_seqlens=sparse_cu_seqlens,
                    max_seqlen=sparse_max_seqlen,
                )
        return x


class NaViTAdapter(nn.Module):
    """Two-layer GELU adapter from encoder to projection dimensions."""

    def __init__(
        self,
        dim: int,
        output_dim: int,
        prefix: str = "",
        use_data_parallel: bool = False,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_size=dim,
            output_size=output_dim,
            bias=False,
            prefix=maybe_prefix(prefix, "fc1"),
            disable_tp=use_data_parallel,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            input_size=output_dim,
            output_size=output_dim,
            bias=False,
            prefix=maybe_prefix(prefix, "fc2"),
            disable_tp=use_data_parallel,
            quant_config=quant_config,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.fc1(x)
        h = self.act(h)
        out, _ = self.fc2(h)
        out = self.act(out)
        return out


class _NaViTPatchEmbedder(nn.Module):
    patch_embedding: ColumnParallelLinear
    position_embedding_table: nn.Embedding


class NaViT(nn.Module):
    """NaViT patch embedding, transformer, downsampling, and adapter."""

    def __init__(
        self,
        config: MuseSparkVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.patch_height = config.patch_size
        self.patch_width = config.patch_size
        self.input_dtype = torch.get_default_dtype()
        self.patch_temporal = config.patch_temporal
        self.downsample_factor = config.merge_size
        self.downsample_strategy = "ps"
        self.sparse_attention_factor = (
            4 if "window_attention" in config.layer_types else 1
        )
        self.grid_sample_mode = "bilinear"
        self.use_cls_token = False
        self.use_rope = True
        norm_eps = config.layer_norm_eps
        use_data_parallel = is_vit_use_data_parallel(config.num_attention_heads)
        latent_dim = config.hidden_size
        output_dim = latent_dim * config.merge_size**2
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.pos_emb_grid_height = config.pos_emb_height
        self.pos_emb_grid_width = config.pos_emb_width
        self.rope_dtype = torch.float32
        self.rope_theta = config.rope_parameters["rope_theta"]
        num_heads = config.num_attention_heads
        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        self.n_heads = divide(num_heads, tp_size)
        self.rope_dim = latent_dim // num_heads

        max_num_image_tokens = 4096
        if max_num_image_tokens is not None:
            max_pre_ds_tokens = max_num_image_tokens * self.downsample_factor**2
            max_grid_per_axis = int(math.sqrt(max_pre_ds_tokens)) + 1
            max_grid_per_axis = max(
                max_grid_per_axis,
                self.pos_emb_grid_height,
                self.pos_emb_grid_width,
            )
        else:
            max_grid_per_axis = max(self.pos_emb_grid_height, self.pos_emb_grid_width)
        self.max_rope_grid_h = max_grid_per_axis
        self.max_rope_grid_w = max_grid_per_axis
        self.register_buffer(
            "precomputed_freqs_cis",
            _precompute_rope_freqs_cis(
                dim=self.rope_dim,
                max_height=max_grid_per_axis,
                max_width=max_grid_per_axis,
                dtype=self.rope_dtype,
                theta=self.rope_theta,
                # Generate on the accelerator to match training rounding.
                device=(
                    torch.accelerator.current_device_index()
                    if torch.accelerator.is_available()
                    else None
                ),
            ),
            persistent=False,
        )

        self.register_buffer(
            "zero_freq",
            torch.cat(
                [
                    torch.ones(1, self.rope_dim // 2, 1, dtype=self.rope_dtype),
                    torch.zeros(1, self.rope_dim // 2, 1, dtype=self.rope_dtype),
                ],
                dim=-1,
            ),
            persistent=False,
        )

        input_dim = 3 * self.patch_height * self.patch_width * self.patch_temporal
        self.patch_embedder = _NaViTPatchEmbedder()
        self.patch_embedder.patch_embedding = ColumnParallelLinear(
            input_size=input_dim,
            output_size=latent_dim,
            bias=False,
            gather_output=True,
            prefix=maybe_prefix(prefix, "patch_embedder.patch_embedding"),
            disable_tp=use_data_parallel,
            quant_config=quant_config,
        )

        scale = latent_dim**-0.5
        num_pos = self.pos_emb_grid_height * self.pos_emb_grid_width + int(
            self.use_cls_token
        )
        self.patch_embedder.position_embedding_table = nn.Embedding(num_pos, latent_dim)

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(scale * torch.randn(latent_dim))
        else:
            self.class_embedding = None

        self.ln_pre = nn.LayerNorm(latent_dim, eps=norm_eps)
        self.ln_post = nn.LayerNorm(latent_dim, eps=norm_eps)

        self.layers = NaViTEncoder(
            dim=latent_dim,
            num_layers=config.num_hidden_layers,
            num_heads=num_heads,
            mlp_ratio=config.intermediate_size / config.hidden_size,
            intermediate_size=config.intermediate_size,
            use_rope=self.use_rope,
            layer_types=config.layer_types,
            norm_eps=norm_eps,
            prefix=maybe_prefix(prefix, "layers"),
            use_data_parallel=use_data_parallel,
            quant_config=quant_config,
        )

    def forward(self, visions: Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode [C,H,W] images/[T,C,H,W] clips into concatenated patch embeddings."""
        if len(visions) == 0:
            return torch.empty(
                0,
                self.output_dim,
                device=self.patch_embedder.position_embedding_table.weight.device,
                dtype=self.patch_embedder.position_embedding_table.weight.dtype,
            )

        device = visions[0].device
        zero_freq = self.zero_freq.to(device=device)

        patch_list: list[torch.Tensor] = []
        pos_xy_list: list[torch.Tensor] = []
        img_slen_list: list[int] = []
        ds_perm_list: list[torch.Tensor] = []
        freq_cis_list: list[torch.Tensor] = []
        cls_tok_idx: list[int] = []

        sp_perm_list: list[torch.Tensor] = []
        sp_slen_list: list[torch.Tensor] = []

        cur_ds_idx = 0
        cur_sp_idx = 0

        for img in _iterate_visions(visions, self.patch_temporal):
            # Preserve the checkpoint's input-pixel and coordinate rounding.
            img = img.to(dtype=self.input_dtype).to(
                dtype=self.patch_embedder.position_embedding_table.weight.dtype
            )
            if img.ndim == 3:
                img = img[None]

            if img.shape[0] == 1:
                img = img.expand(self.patch_temporal, -1, -1, -1)

            assert img.shape[0] == self.patch_temporal, img.shape

            _, C, H, W = img.shape
            assert H % self.patch_height == 0 and W % self.patch_width == 0
            grid_h = H // self.patch_height
            grid_w = W // self.patch_width

            patches = (
                img.reshape(
                    self.patch_temporal,
                    C,
                    grid_h,
                    self.patch_height,
                    grid_w,
                    self.patch_width,
                )
                .permute(2, 4, 0, 1, 3, 5)
                .reshape(
                    grid_h * grid_w,
                    self.patch_temporal * C * self.patch_height * self.patch_width,
                )
            )

            inv_grid_h, inv_grid_w = 1.0 / grid_h, 1.0 / grid_w
            pos_h = torch.linspace(
                -1 + inv_grid_h,
                1 - inv_grid_h,
                grid_h,
                device=device,
                dtype=self.input_dtype,
            )
            pos_w = torch.linspace(
                -1 + inv_grid_w,
                1 - inv_grid_w,
                grid_w,
                device=device,
                dtype=self.input_dtype,
            )
            # The trained coordinate-to-patch mapping uses xy meshgrid indexing.
            pos_xy = (
                torch.stack(
                    torch.meshgrid(pos_h, pos_w, indexing="xy"),
                    dim=-1,
                )
                .reshape(-1, 2)
                .to(dtype=patches.dtype)
            )

            if grid_h > self.max_rope_grid_h or grid_w > self.max_rope_grid_w:
                freq_cis = _precompute_rope_freqs_cis(
                    dim=self.rope_dim,
                    max_height=grid_h,
                    max_width=grid_w,
                    dtype=self.rope_dtype,
                    theta=self.rope_theta,
                    device=device,
                ).reshape(-1, self.rope_dim // 2, 2)
            else:
                freq_cis = self.precomputed_freqs_cis[:grid_h, :grid_w].reshape(
                    -1, self.rope_dim // 2, 2
                )

            sp_perm, sp_slen = None, None
            if self.sparse_attention_factor > 1:
                sp_perm, sp_slen, cur_sp_idx = _get_sparse_attention_perm(
                    cur_sp_idx=cur_sp_idx,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    pos_emb_grid_height=self.pos_emb_grid_height,
                    pos_emb_grid_width=self.pos_emb_grid_width,
                    device=device,
                )

            if self.downsample_strategy == "ps":
                ds_perm, cur_ds_idx = _get_pixel_shuffle_perm(
                    cur_ds_idx, grid_h, grid_w, self.downsample_factor, device
                )
            elif self.downsample_strategy == "avg":
                ds_perm, cur_ds_idx = _get_avg_pooling_perm(
                    cur_ds_idx, grid_h, grid_w, self.downsample_factor, device
                )
            else:
                raise ValueError(
                    f"Unknown downsample strategy: {self.downsample_strategy}"
                )

            img_slen_list.append(grid_h * grid_w)
            patch_list.append(patches)
            pos_xy_list.append(pos_xy)
            freq_cis_list.append(freq_cis)
            ds_perm_list.append(ds_perm)

            if self.sparse_attention_factor > 1:
                assert sp_perm is not None and sp_slen is not None
                sp_perm_list.append(sp_perm)
                sp_slen_list.append(sp_slen)

            if self.use_cls_token:
                cls_tok_idx.append(sum(img_slen_list))
                patch_list.append(patches.new_zeros(1, patches.shape[1]))
                pos_xy_list.append(pos_xy.new_zeros(1, pos_xy.shape[1]))
                freq_cis_list.append(zero_freq)

                if self.sparse_attention_factor > 1:
                    assert sp_perm is not None and sp_slen is not None
                    sp_perm_list.append(sp_perm.new_full((1,), cur_sp_idx))
                    sp_slen_list.append(sp_slen.new_ones(1))
                    cur_sp_idx += 1

                img_slen_list[-1] += 1

        batch_patch = torch.cat(patch_list, dim=0)
        batch_pos_xy = torch.cat(pos_xy_list, dim=0)
        batch_freq_cis = torch.cat(freq_cis_list, dim=0)
        batch_ds_perm = torch.cat(ds_perm_list, dim=0)

        cu_seqlens, max_seqlen_cpu = _seqlens_to_cu_seqlens(img_slen_list, device)

        batch_sp_perm = None
        sparse_cu_seqlens = None
        sparse_max_seqlen_cpu = None
        if self.sparse_attention_factor > 1:
            batch_sp_perm = torch.cat(sp_perm_list, dim=0)
            sp_slen_cat = torch.cat(sp_slen_list, dim=0).tolist()
            sparse_cu_seqlens, sparse_max_seqlen_cpu = _seqlens_to_cu_seqlens(
                sp_slen_cat, device
            )

        x, _ = self.patch_embedder.patch_embedding(batch_patch)

        pos_emb_wei = self.patch_embedder.position_embedding_table.weight
        cls_pos_emb_wei = None
        if self.use_cls_token:
            cls_pos_emb_wei = pos_emb_wei[-1]
            pos_emb_wei = pos_emb_wei[:-1]

        pos_emb_wei = (
            pos_emb_wei.reshape(self.pos_emb_grid_height, self.pos_emb_grid_width, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        x = x + F.grid_sample(
            pos_emb_wei,
            batch_pos_xy[None, None],
            mode=self.grid_sample_mode,
            align_corners=False,
        )[0, :, 0, :].permute(1, 0)

        batch_cls_tok_idx = None
        if self.use_cls_token and self.class_embedding is not None:
            assert cls_pos_emb_wei is not None
            batch_cls_tok_idx = torch.tensor(cls_tok_idx, device=device)
            x[batch_cls_tok_idx, :] = self.class_embedding + cls_pos_emb_wei

        x = self.ln_pre(x)

        if self.sparse_attention_factor > 1 and batch_sp_perm is not None:
            x = x[batch_sp_perm]
            batch_freq_cis = batch_freq_cis[batch_sp_perm]

        x = self.layers(
            x,
            freq_cis=batch_freq_cis,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen_cpu,
            sparse_cu_seqlens=sparse_cu_seqlens,
            sparse_max_seqlen=sparse_max_seqlen_cpu,
        )

        if self.sparse_attention_factor > 1 and batch_sp_perm is not None:
            x = _inv_perm_by_idx(x, batch_sp_perm)

        x = self.ln_post(x)

        if self.use_cls_token:
            assert batch_cls_tok_idx is not None
            non_cls_msk = torch.ones(x.shape[0], dtype=torch.bool, device=device)
            non_cls_msk[batch_cls_tok_idx] = False
            x = x[non_cls_msk]

        if self.downsample_strategy == "ps":
            x = _downsample_pixel_shuffle(x, batch_ds_perm, self.downsample_factor)
        elif self.downsample_strategy == "avg":
            x = _downsample_avg_pooling(x, batch_ds_perm, self.downsample_factor)

        return x


def get_navit_patch_count(
    img_h: int,
    img_w: int,
    patch_h: int,
    patch_w: int,
    downsample_factor: int,
) -> int:
    """Count output patch tokens after spatial downsampling."""
    return (img_h // patch_h) * (img_w // patch_w) // (downsample_factor**2)


def get_navit_video_patch_count(
    video_t: int,
    video_h: int,
    video_w: int,
    patch_t: int,
    patch_h: int,
    patch_w: int,
    downsample_factor: int,
) -> int:
    """Number of output patch tokens for a video."""
    patches_per_frame = (
        (video_h // patch_h) * (video_w // patch_w) // (downsample_factor**2)
    )
    num_clips = video_t // patch_t
    return patches_per_frame * num_clips
