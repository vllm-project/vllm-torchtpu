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
"""vLLM runtime adapter (`Qwen38MoeProgram`, `layout_request`, `create`) for Qwen3.8."""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import operator
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack, cast

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel_runtime.jax_bridge import (
    jax_to_torch_cpu,
    make_typed_step_wrapper,
    sharded_jax_callable,
    torch_cpu_to_jax,
)
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket,
    Decline,
    ForwardView,
    LayoutDecline,
    LayoutRequest,
    RunnerShape,
    RunnerView,
    StepDecision,
    StepFn,
    WarmupCase,
)

from . import kernel, plan, weights
from .shapes import (
    EXPECTED_DCP_SIZE as _EXPECTED_DCP_SIZE,
)
from .shapes import (
    EXPECTED_TP_SIZE as _EXPECTED_TP_SIZE,
)
from .shapes import (
    MINI_HIDDEN_SIZE as _MINI_HIDDEN_SIZE,
)
from .shapes import (
    NVFP4_GROUP_SIZE as _NVFP4_GROUP_SIZE,
)
from .shapes import (
    PROD_HIDDEN_SIZE as _PROD_HIDDEN_SIZE,
)
from .shapes import (
    PROD_NUM_LAYERS as _PROD_NUM_LAYERS,
)
from .shapes import (
    Qwen38Config,
    mini_parity_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from jax.sharding import Mesh

    from .weights import BackboneLike

logger = init_logger(__name__)
_OP_COUNTER = itertools.count()

_ERR_POOL_UNKNOWN: str = "pool_layout_unknown"
_ERR_POOL_MISMATCH: str = "pool_layout_mismatch"
_ERR_UNSUPPORTED_ARCH: str = "unsupported_arch"
_ERR_UNSUPPORTED_GEOM: str = "unsupported_geometry"
_ERR_UNSUPPORTED_QUANT: str = "unsupported_quant"
_ERR_UNSUPPORTED_TP: str = "unsupported_tp_size"
_ERR_UNSUPPORTED_DCP: str = "unsupported_dcp_size"
_ERR_UNALIGNED_CACHE: str = "unaligned_cache_layout"
_ERR_BACKBONE_NOT_FOUND: str = "backbone_not_found"
_ERR_EMPTY_BUCKETS: str = "empty_buckets"


class Qwen38AdapterError(ValueError):
    """Raised when adapter options or configuration validation fails."""


class _KvCacheHolder(Protocol):
    """Module holding a mutable `kv_cache` attribute."""

    kv_cache: object


def _build_pallas_step_op(
    op_prefix: str,
    body_fn: Callable[..., tuple[jax.Array, ...]],
    *,
    mutated_arg_indices: tuple[int, ...],
    out_shape: tuple[int, int],
    mesh: Mesh | None = None,
) -> Callable[..., torch.Tensor]:
    """Wrap `body_fn` in a `torch.library.custom_op` preserving pool `data_ptr()`."""
    out_rows, hidden_size = out_shape
    op_id = next(_OP_COUNTER)
    qual_name = f"megakernel_runtime::{op_prefix}_{op_id}"
    num_inputs = 4 + len(mutated_arg_indices)
    num_outputs = len(mutated_arg_indices) + 1
    tpu_callable_cache: list[Callable[..., Sequence[torch.Tensor]]] = []

    def _get_tpu_callable() -> Callable[..., Sequence[torch.Tensor]]:
        """Build or return the cached sharded JAX TPU callable."""
        if not tpu_callable_cache:
            active_mesh = mesh or jax.sharding.Mesh(jax.devices(), ("tp",))
            typed_fn = make_typed_step_wrapper(
                f"{op_prefix}_{op_id}",
                body_fn,
                num_inputs,
                num_outputs,
            )
            in_specs = tuple(P() for _ in range(num_inputs))
            out_specs = tuple(P() for _ in range(num_outputs))
            tpu_callable_cache.append(
                sharded_jax_callable(
                    qual_name,
                    typed_fn,
                    mesh=active_mesh,
                    input_partition_specs=in_specs,
                    output_partition_specs=out_specs,
                    donate_argnums=mutated_arg_indices or None,
                )
            )
        return tpu_callable_cache[0]

    @torch.library.custom_op(qual_name, mutates_args=())
    def _op(operands: list[torch.Tensor]) -> list[torch.Tensor]:
        """Execute underlying Pallas body function on tensor inputs."""
        if operands[0].device.type == "cpu":
            jax_args = [
                torch_cpu_to_jax(
                    t.view(torch.uint8) if idx in mutated_arg_indices else t
                )
                for idx, t in enumerate(operands)
            ]
            jax_outs = body_fn(*jax_args)
            pool_outs = jax_outs[:-1]
            x_out = jax_outs[-1]
            torch_pools = [
                jax_to_torch_cpu(p, operands[idx].view(torch.uint8)).to(
                    device=operands[idx].device
                )
                for p, idx in zip(pool_outs, mutated_arg_indices, strict=True)
            ]
            torch_x = jax_to_torch_cpu(x_out, operands[0]).to(device=operands[0].device)
            return [*torch_pools, torch_x]

        tpu_operands = [
            t.view(torch.uint8) if idx in mutated_arg_indices else t
            for idx, t in enumerate(operands)
        ]
        tpu_outs = list(_get_tpu_callable()(*tpu_operands))
        torch_pools = [p.view(torch.uint8) for p in tpu_outs[:-1]]
        torch_x = tpu_outs[-1].to(dtype=operands[0].dtype)
        return [*torch_pools, torch_x]

    @_op.register_fake
    def _fake(operands: list[torch.Tensor]) -> list[torch.Tensor]:
        """Register fake kernel output shapes for tracing."""
        x_t = operands[0]
        fake_pools = [
            torch.empty_like(operands[idx].view(torch.uint8))
            for idx in mutated_arg_indices
        ]
        fake_x = x_t.new_empty((out_rows, hidden_size))
        return [*fake_pools, fake_x]

    def _step(*args: torch.Tensor) -> torch.Tensor:
        """Execute custom op and mutate pool buffers in place."""
        dyn_args = list(args[:num_inputs])
        outs = _op(dyn_args)
        for p_out, idx in zip(outs[:-1], mutated_arg_indices, strict=True):
            dyn_args[idx].view(torch.uint8).copy_(p_out)
        return outs[-1]

    return _step


OPTION_KEYS: frozenset[str] = frozenset(
    {
        "buckets",
        "compile_hlo_check",
        "allow_mini_geometry",
        "interpret",
    }
)
DEFAULT_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16)
SUPPORTED_ARCHS: frozenset[str] = frozenset(
    {
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
    }
)
SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)
PAGE_TOKENS: int = 128
MAMBA_SLOT_TOKENS: int = kernel.PAGES_PER_GDN_SLOT * PAGE_TOKENS
_KV_CACHE_NDIM: int = 5


def layout_request(
    vllm_config: object,
    _options: Mapping[str, object] | None = None,
) -> LayoutRequest:
    """Request 128-token pages, DCP interleave, and 640-token Mamba slots."""
    if vllm_config is None:
        msg = "vllm_config must not be None"
        raise Qwen38AdapterError(msg)
    return LayoutRequest(
        block_size=PAGE_TOKENS,
        cp_kv_cache_interleave_size=PAGE_TOKENS,
        mamba_slot_tokens=MAMBA_SLOT_TOKENS,
    )


def _parse_buckets(raw: object) -> tuple[int, ...]:
    """Parse comma-separated or sequence bucket sizes."""
    if raw is None or raw == "":
        return DEFAULT_BUCKETS
    if isinstance(raw, (list, tuple)):
        return tuple(sorted({int(x) for x in raw if int(x) > 0}))
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return tuple(sorted({int(p) for p in parts if int(p) > 0}))


def _dcp_size(view: RunnerView) -> int:
    """Extract decode context parallel size from runner view."""
    vllm_cfg = view.vllm_config
    if vllm_cfg is None or vllm_cfg.parallel_config is None:
        return 1
    val = vllm_cfg.parallel_config.decode_context_parallel_size
    return int(val) if val is not None else 1


def _extract_quant_config(text_cfg: object, hf_cfg: object) -> object:
    """Extract raw quantization config object or mapping from HuggingFace config."""
    return getattr(text_cfg, "quantization_config", None) or getattr(
        hf_cfg, "quantization_config", None
    )


def _quant_field(q_cfg: object, key: str) -> object:
    """Read `key` from a mapping or attribute-based quantization config."""
    if isinstance(q_cfg, Mapping):
        return q_cfg.get(key)
    return getattr(q_cfg, key, None)


def _is_nvfp4_scheme(q_cfg: object, q_method: str) -> bool:
    """Return True if `q_cfg` describes a 16-element group NVFP4 scheme."""
    group_size = _quant_field(q_cfg, "group_size")
    if group_size is not None and int(str(group_size)) != _NVFP4_GROUP_SIZE:
        return False
    if q_method in ("nvfp4", "fp4"):
        return True
    if q_method in ("compressed-tensors", "modelopt"):
        fmt = str(
            _quant_field(q_cfg, "format")
            or _quant_field(q_cfg, "quant_algo")
            or _quant_field(q_cfg, "config_groups")
            or ""
        ).lower()
        return "fp4" in fmt
    return False


def _effective_slot_tokens(view: RunnerView) -> int:
    """Return effective token capacity per cache slot from the runner view."""
    block_size = view.cache_block_size
    return int(block_size) if block_size else 0


class ProgramOptions(TypedDict, total=False):
    """Optional settings for Qwen38MoeProgram."""

    tp_rank: int
    decode_buckets: Sequence[int]
    compile_hlo_check: bool
    base_config: Qwen38Config | None
    interpret: bool | pltpu.InterpretParams | None


def _collect_attention_pools(
    backbone: BackboneLike, cfg: Qwen38Config
) -> tuple[list[torch.Tensor], list[str]]:
    """Collect and validate KV cache pools for attention layers."""
    pools: list[torch.Tensor] = []
    attn_names: list[str] = []
    for l_idx in range(cfg.L):
        if (l_idx + 1) % cfg.att_every != 0:
            continue
        layer = backbone.layers[l_idx]
        sa = getattr(layer, "self_attn", None)
        attn_mod = getattr(sa, "attn", sa)
        name = (
            getattr(attn_mod, "layer_name", None)
            or getattr(sa, "prefix", None)
            or f"model.layers.{l_idx}.self_attn.attn"
        )
        attn_names.append(str(name))
        raw_kv = getattr(attn_mod, "kv_cache", None)
        if isinstance(raw_kv, (list, tuple)) and raw_kv:
            raw_kv = raw_kv[0]
        if not isinstance(raw_kv, torch.Tensor):
            err_msg = f"Missing attention kv_cache tensor on layer {l_idx}"
            raise LayoutDecline(Decline(_ERR_POOL_UNKNOWN, err_msg).result)
        kv_tensor: torch.Tensor = raw_kv
        expected_tail = (2, cfg.K_R, cfg.D, cfg.page_size)
        if (
            kv_tensor.ndim != _KV_CACHE_NDIM
            or tuple(kv_tensor.shape[1:_KV_CACHE_NDIM]) != expected_tail
        ):
            err_msg = (
                f"Layer {l_idx} kv_cache shape {tuple(kv_tensor.shape)} != "
                f"(num_pages, 2, {cfg.K_R}, {cfg.D}, {cfg.page_size})"
            )
            raise LayoutDecline(Decline(_ERR_POOL_MISMATCH, err_msg).result)
        pools.append(kv_tensor)
    return pools, attn_names


def _collect_mamba_names(backbone: BackboneLike, cfg: Qwen38Config) -> list[str]:
    """Collect layer names for GDN linear attention layers."""
    mamba_names: list[str] = []
    for l_idx in range(cfg.L):
        if (l_idx + 1) % cfg.att_every == 0:
            continue
        layer = backbone.layers[l_idx]
        la = getattr(layer, "linear_attn", None)
        name = (
            getattr(la, "prefix", None)
            or getattr(la, "layer_name", None)
            or f"model.layers.{l_idx}.linear_attn"
        )
        mamba_names.append(str(name))
    return mamba_names


def _verify_gdn_pool_aliasing(
    backbone: BackboneLike, pools: list[torch.Tensor], cfg: Qwen38Config
) -> None:
    """Verify that GDN states share storage with corresponding attention regions."""
    for l_idx in range(cfg.L):
        if (l_idx + 1) % cfg.att_every == 0:
            continue
        la = getattr(backbone.layers[l_idx], "linear_attn", None)
        raw_la_kv = getattr(la, "kv_cache", None)
        if isinstance(raw_la_kv, (list, tuple)) and raw_la_kv:
            raw_la_kv = raw_la_kv[0]
        if isinstance(raw_la_kv, torch.Tensor):
            la_tensor: torch.Tensor = raw_la_kv
            reg_pool_tensor: torch.Tensor = pools[l_idx // cfg.att_every]
            if la_tensor.data_ptr() != reg_pool_tensor.data_ptr():
                err_msg = (
                    f"GDN layer {l_idx} pool is not aliased with attention region "
                    f"{l_idx // cfg.att_every}"
                )
                raise LayoutDecline(Decline(_ERR_POOL_MISMATCH, err_msg).result)


def prepare_input_tensors(
    ctx: object,
    batch_dims: tuple[int, int],
    cfg: Qwen38Config,
    dev: torch.device,
    backbone: BackboneLike,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare padded activations and position vectors for step execution."""
    n_reqs, b_dim = batch_dims
    inputs_embeds = getattr(ctx, "inputs_embeds", None)
    input_ids = getattr(ctx, "input_ids", None)
    if inputs_embeds is not None:
        x_raw: torch.Tensor = cast("torch.Tensor", inputs_embeds)[:n_reqs]
    else:
        embed_fn = getattr(backbone, "embed_tokens", None)
        if callable(embed_fn) and input_ids is not None:
            x_raw = cast("torch.Tensor", embed_fn(input_ids[:n_reqs]))
        else:
            msg = "Model is missing callable embed_tokens required to embed input_ids"
            raise RuntimeError(msg)

    if x_raw.shape[0] < b_dim:
        pad_x = torch.zeros(
            (b_dim - x_raw.shape[0], cfg.H),
            dtype=torch.bfloat16,
            device=dev,
        )
        x_in = torch.cat([x_raw.to(torch.bfloat16), pad_x], dim=0)
    else:
        x_in = x_raw[:b_dim].to(torch.bfloat16).contiguous()

    positions = getattr(ctx, "positions", None)
    if positions is None:
        msg = "StepContext is missing required positions tensor"
        raise RuntimeError(msg)
    pos_raw = positions[:n_reqs].to(device=dev, dtype=torch.int32).reshape(-1)

    if pos_raw.shape[0] < b_dim:
        pad_p = torch.full(
            (b_dim - pos_raw.shape[0],), -1, dtype=torch.int32, device=dev
        )
        pos = torch.cat([pos_raw, pad_p], dim=0)
    else:
        pos = pos_raw[:b_dim].contiguous()
    return x_in, pos


def _assign_kv_view(
    target: _KvCacheHolder | None,
    view_t: torch.Tensor,
    *,
    tensor_only_scalar: bool = False,
) -> None:
    """Rebind `target.kv_cache` to `view_t` across list, tuple, or tensor holders."""
    if target is None:
        return
    raw_kv = target.kv_cache
    if issubclass(type(raw_kv), list) and raw_kv:
        cast("list[object]", raw_kv)[0] = view_t
    elif issubclass(type(raw_kv), tuple) and raw_kv:
        target.kv_cache = (view_t, *cast("tuple[object, ...]", raw_kv)[1:])
    elif not tensor_only_scalar or issubclass(type(raw_kv), torch.Tensor):
        target.kv_cache = view_t


def _rebind_backbone_to_unified_pool(
    backbone: BackboneLike,
    raw_pools: list[torch.Tensor],
    cfg: Qwen38Config,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Pack per-region pools into one contiguous 6D tensor and rebind layer views."""
    unified = torch.stack(raw_pools, dim=0).contiguous()
    region_views = [unified[reg] for reg in range(cfg.NATT)]
    for l_idx in range(cfg.L):
        reg = l_idx // cfg.att_every
        view_t = region_views[reg]
        layer = backbone.layers[l_idx]
        if (l_idx + 1) % cfg.att_every == 0:
            sa = getattr(layer, "self_attn", None)
            attn_mod = getattr(sa, "attn", sa)
            _assign_kv_view(attn_mod, view_t)
        else:
            la = getattr(layer, "linear_attn", None)
            _assign_kv_view(la, view_t, tensor_only_scalar=True)
    return unified, region_views


def prepare_page_table(
    attn_md: Mapping[str, object],
    attn_layer_names: Sequence[str],
    dims: tuple[int, int, int],
    dev: torch.device,
) -> torch.Tensor:
    """Extract and pad page table indices across attention layers."""
    n_reqs, b_dim, max_pages = dims
    pt_rows: list[torch.Tensor] = []
    for a_name in attn_layer_names:
        md_a = attn_md.get(a_name)
        bt = getattr(md_a, "block_tables", getattr(md_a, "block_table", None))
        if bt is None:
            msg = f"Missing block_table in attention metadata for layer {a_name!r}"
            raise RuntimeError(msg)
        bt_t = bt[:n_reqs, :max_pages].to(device=dev, dtype=torch.int32)
        if bt_t.shape[0] < b_dim or bt_t.shape[1] < max_pages:
            bt_pad = torch.zeros((b_dim, max_pages), dtype=torch.int32, device=dev)
            bt_pad[: bt_t.shape[0], : bt_t.shape[1]] = bt_t
        else:
            bt_pad = bt_t.contiguous()
        pt_rows.append(bt_pad)
    return torch.stack(pt_rows, dim=0)


def prepare_mamba_slots(
    attn_md: Mapping[str, object],
    mamba_layer_names: Sequence[str],
    n_reqs: int,
    b_dim: int,
    dev: torch.device,
) -> torch.Tensor:
    """Extract and pad Mamba slot indices across GDN layers."""
    m_rows: list[torch.Tensor] = []
    for m_name in mamba_layer_names:
        md_m = attn_md.get(m_name)
        ms = getattr(md_m, "mamba_state_indices", None)
        if ms is None:
            msg = (
                "Missing mamba_state_indices in attention metadata for layer "
                f"{m_name!r}"
            )
            raise RuntimeError(msg)
        ms_t = ms[:n_reqs].to(device=dev, dtype=torch.int32).reshape(-1)
        if ms_t.shape[0] < b_dim:
            ms_pad = torch.full((b_dim,), -1, dtype=torch.int32, device=dev)
            ms_pad[: ms_t.shape[0]] = ms_t
        else:
            ms_pad = ms_t[:b_dim].contiguous()
        m_rows.append(ms_pad)
    return torch.stack(m_rows, dim=0)


class Qwen38MoeProgram:
    """`MonoProgram` implementation for Qwen3.8-2.4T TP32xDCP8 NVFP4."""

    display_name = "Qwen3.8-2.4T"

    def __init__(
        self,
        model: torch.nn.Module,
        **options: Unpack[ProgramOptions],
    ) -> None:
        """Initialize Qwen3.8-2.4T megakernel program instance."""
        self.model = model
        self.backbone = weights.find_qwen38_2_4t_backbone(model)
        self.tp_rank = int(options.get("tp_rank", 0))
        decode_buckets = options.get("decode_buckets", DEFAULT_BUCKETS)
        self.decode_buckets = tuple(int(b) for b in decode_buckets)
        self._compile_hlo_check = bool(options.get("compile_hlo_check", False))
        self.base_config = options.get("base_config") or Qwen38Config()
        self._interpret = options.get("interpret")

        self.mesh: Mesh | None = None
        self.w_dict: dict[str, torch.Tensor] = {}
        self.w_flat: list[torch.Tensor] = []
        self._jax_weights: dict[str, jax.Array] = {}
        self._per_rank_w: dict[str, jax.Array] = {}
        self._w_flat_sig: tuple[tuple[int, int], ...] = ()
        self.attn_layer_names: list[str] = []
        self.mamba_layer_names: list[str] = []
        self.region_pools: list[torch.Tensor] = []
        self.unified_pool: torch.Tensor | None = None
        self.rank_pools: list[list[torch.Tensor]] | None = None
        self._pool_snapshots: list[list[torch.Tensor]] | None = None
        self._warmed_buckets: set[tuple[int, int]] = set()

    def buckets(self, token_paddings: Sequence[int]) -> Sequence[Bucket]:
        """Compute decode buckets covering requested token padding thresholds."""
        pads = [int(p) for p in token_paddings if int(p) > 0]
        result: list[Bucket] = []
        for b in self.decode_buckets:
            out_rows = next((p for p in pads if p >= b), b)
            result.append(Bucket(T=1, B=b, out_rows=out_rows))
        return result

    def set_interpret_mode(
        self, *, interpret: bool | pltpu.InterpretParams | None = None
    ) -> None:
        """Update Pallas interpret execution mode used on subsequent `build` calls."""
        self._interpret = interpret

    def _sync_jax_weights(self) -> None:
        """Normalize and place model weights on JAX devices once at load/build."""
        version_getter = operator.attrgetter("_version")
        sig = tuple((id(t), int(version_getter(t))) for t in self.w_flat)
        if sig == self._w_flat_sig and self._jax_weights:
            return
        w_raw = {
            k: (
                torch_cpu_to_jax(t)
                if t.device.type == "cpu"
                else jax.dlpack.from_dlpack(t)
            )
            for k, t in zip(weights.WEIGHT_KEYS, self.w_flat, strict=True)
        }
        norm_w = plan.normalize_weights_for_pallas(w_raw, self.base_config)
        if (
            self.mesh is not None
            and self.base_config.R > 1
            and len(self.mesh.devices.flat) == self.base_config.R
        ):
            sharding = jax.sharding.NamedSharding(self.mesh, P("tp"))
            norm_w = {
                k: (
                    jax.device_put(v, sharding)
                    if v.shape[0] == self.base_config.R
                    else v
                )
                for k, v in norm_w.items()
            }
        self._jax_weights = norm_w
        self._per_rank_w = {
            k: v[0] if v.ndim > 0 and v.shape[0] == 1 else v for k, v in norm_w.items()
        }
        self._w_flat_sig = sig

    def load(self, mesh: Mesh | None) -> None:
        """Bind device mesh and extract normalized weights for Pallas execution."""
        self.mesh = mesh if isinstance(mesh, jax.sharding.Mesh) else None
        norm_mod = getattr(self.backbone, "norm", None)
        raw_norm_w = getattr(norm_mod, "weight", None) if norm_mod is not None else None
        device: torch.device | str = "cpu"
        if isinstance(raw_norm_w, torch.Tensor):
            norm_tensor: torch.Tensor = raw_norm_w
            device = norm_tensor.device
        self.w_dict = weights.extract_backbone_weights_torch(
            self.model, self.base_config, device=device
        )
        self.w_flat = [self.w_dict[k] for k in weights.WEIGHT_KEYS]
        self._sync_jax_weights()

    def _collect_region_pools(self, backbone: BackboneLike) -> list[torch.Tensor]:
        """Collect and validate unified KV/GDN region cache pools."""
        cfg = self.base_config
        pools, attn_names = _collect_attention_pools(backbone, cfg)
        mamba_names = _collect_mamba_names(backbone, cfg)
        _verify_gdn_pool_aliasing(backbone, pools, cfg)

        if not self.attn_layer_names:
            self.attn_layer_names = attn_names
            self.mamba_layer_names = mamba_names
        return pools

    def bind(self, view: RunnerView) -> dict[str, int]:
        """Inspect runner view cache layout and allocate region pools."""
        slot_tokens = _effective_slot_tokens(view)
        if slot_tokens < MAMBA_SLOT_TOKENS:
            err_msg = (
                f"Qwen3.8 GDN state requires >= {MAMBA_SLOT_TOKENS} tokens "
                f"(5 pages) per slot, got cache_block_size={slot_tokens}"
            )
            raise LayoutDecline(Decline(_ERR_POOL_MISMATCH, err_msg).result)

        rank_backbones = getattr(self.model, "_rank_backbones", None)
        if rank_backbones is not None:
            self.rank_pools = [
                self._collect_region_pools(cast("BackboneLike", b))
                for b in rank_backbones
            ]
            self.region_pools = self.rank_pools[0]
            self.unified_pool = None
        else:
            raw_pools = self._collect_region_pools(self.backbone)
            self.unified_pool, self.region_pools = _rebind_backbone_to_unified_pool(
                self.backbone, raw_pools, self.base_config
            )
            self.rank_pools = None

        num_pages = int(self.region_pools[0].shape[0])
        self.base_config = dataclasses.replace(
            self.base_config,
            num_pages=num_pages,
            max_pages_per_seq=int(view.max_pages_per_seq),
        )
        return {
            "num_pages": num_pages,
            "pages_per_gdn_slot": kernel.PAGES_PER_GDN_SLOT,
            "num_regions": len(self.region_pools),
        }

    def check_compiled_ir(self, debug_info: object, bucket: Bucket) -> None:
        """Verify compiled HLO contains no forbidden all-gather operations."""
        if not self._compile_hlo_check:
            return
        texts = getattr(debug_info, "stablehlo_forward_text", None) or ()
        for idx, text in enumerate(texts):
            plan.assert_no_all_gather(
                str(text), context=f"bucket=({bucket.T},{bucket.B}) graph={idx}"
            )

    @contextlib.contextmanager
    def warmup_session(self) -> Iterator[None]:
        """Context manager preserving touched page 0 of cache pools during warmup."""
        self._warmed_buckets.clear()
        if self.rank_pools is not None:
            snaps = [
                [p[:1].view(torch.uint8).clone() for p in r_pools]
                for r_pools in self.rank_pools
            ]
        else:
            snaps = [[p[:1].view(torch.uint8).clone() for p in self.region_pools]]
        self._pool_snapshots = snaps
        try:
            yield
        finally:
            self._restore_pools()
            self._pool_snapshots = None
            self._warmed_buckets.clear()

    def _restore_pools(self) -> None:
        """Restore touched page 0 of cache pool tensor buffers from snapshot."""
        if self._pool_snapshots is None:
            return
        if self.rank_pools is not None:
            for r_pools, r_snaps in zip(
                self.rank_pools, self._pool_snapshots, strict=True
            ):
                for p, s in zip(r_pools, r_snaps, strict=True):
                    p[:1].view(torch.uint8).copy_(s)
        else:
            for p, s in zip(self.region_pools, self._pool_snapshots[0], strict=True):
                p[:1].view(torch.uint8).copy_(s)

    def warmup_inputs(self, bucket: Bucket, shape: RunnerShape) -> WarmupCase | None:
        """Construct synthetic dummy inputs for compile-time trace execution."""
        if shape.tok_pad < 0:
            msg = f"RunnerShape.tok_pad must be >= 0, got {shape.tok_pad}"
            raise Qwen38AdapterError(msg)
        if self._interpret is not None or jax.devices()[0].platform == "cpu":
            return None
        bucket_key = (int(bucket.T), int(bucket.B))
        if bucket_key in self._warmed_buckets:
            return None
        self._warmed_buckets.add(bucket_key)

        cfg = self.base_config
        b_dim = int(bucket.B)
        dev = self.region_pools[0].device
        x_in = torch.zeros((b_dim, cfg.H), dtype=torch.bfloat16, device=dev)
        pos = torch.zeros((b_dim,), dtype=torch.int32, device=dev)
        page_table = torch.zeros(
            (cfg.NATT, b_dim, cfg.max_pages_per_seq), dtype=torch.int32, device=dev
        )
        mamba_slots = torch.full((cfg.NGDN, b_dim), -1, dtype=torch.int32, device=dev)

        if self.rank_pools is not None:
            flat_pools = [p for r_pools in self.rank_pools for p in r_pools]
        else:
            if self.unified_pool is None:
                msg = "Qwen38MoeProgram.unified_pool must be bound before execution"
                raise Qwen38AdapterError(msg)
            flat_pools = [self.unified_pool]

        args = (x_in, pos, page_table, mamba_slots, *flat_pools)

        def _check(_out: object) -> None:
            """Restore touched page 0 after warmup step execution."""
            self._restore_pools()

        return WarmupCase(args=args, check=_check)

    def _build_rank_step_fn(
        self,
        cfg: Qwen38Config,
        *,
        interpret: bool | pltpu.InterpretParams,
        multi_rank_host: bool,
    ) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]] | None:
        """Build single-backbone per-rank Pallas step callable wrapped in."""
        if multi_rank_host:
            return None
        per_rank_w = self._per_rank_w
        raw_rank_step_fn = kernel.make_pallas_step_fn(
            cfg,
            per_rank_w,
            has_forced_routing=False,
            interpret=interpret,
            axis_name="tp" if cfg.R > 1 else None,
        )
        if cfg.R <= 1:
            return raw_rank_step_fn
        active_mesh = self.mesh or jax.sharding.Mesh(jax.devices()[: cfg.R], ("tp",))
        w_specs = {k: P() for k in per_rank_w}
        return jax.shard_map(
            raw_rank_step_fn,
            mesh=active_mesh,
            in_specs=(P(), P(), P(), P(), P(), P(), w_specs),
            out_specs=(P(), P(), P(), P()),
            check_vma=False,
        )

    def build(self, bucket: Bucket) -> StepFn:
        """Compile and return the executable PyTorch step function for `bucket`."""
        cfg = dataclasses.replace(self.base_config, B=int(bucket.B))
        out_rows = int(bucket.out_rows or bucket.B)
        multi_rank_host = self.rank_pools is not None
        num_pool_args = cfg.R * cfg.NATT if multi_rank_host else 1
        interpret = (
            self._interpret
            if self._interpret is not None
            else jax.devices()[0].platform == "cpu"
        )
        is_cpu_backend = bool(interpret) or (
            self.mesh is not None and self.mesh.devices.flat[0].platform == "cpu"
        )
        if not multi_rank_host and cfg.R > 1 and is_cpu_backend:
            msg = (
                "Single-backbone CPU interpret execution with R > 1 requires "
                "_rank_backbones to supply per-rank shards and pools"
            )
            raise Qwen38AdapterError(msg)

        self._sync_jax_weights()
        w_norm = self._jax_weights
        per_rank_w = self._per_rank_w
        rank_step_fn = self._build_rank_step_fn(
            cfg, interpret=interpret, multi_rank_host=multi_rank_host
        )

        def _body(*jax_args: jax.Array) -> tuple[jax.Array, ...]:
            """Execute sharded Pallas megakernel step on JAX arrays."""
            x_in, pos, page_table, mamba_slots = jax_args[:4]
            pools_in = list(jax_args[4 : 4 + num_pool_args])

            if multi_rank_host:
                pools_by_reg = [
                    jnp.stack(
                        [pools_in[r * cfg.NATT + reg] for r in range(cfg.R)], axis=0
                    )
                    for reg in range(cfg.NATT)
                ]
                conv_init, rec_init = kernel.unpack_gdn_states_from_pools(
                    pools_by_reg, mamba_slots, cfg
                )
                pool_all_regs = jnp.stack(pools_by_reg, axis=1).view(jnp.float8_e4m3fn)
                prob = {
                    "x": jnp.broadcast_to(x_in[None, ...], (cfg.R, cfg.B, cfg.H)),
                    "pos": jnp.broadcast_to(pos[None, :], (cfg.R, cfg.B)),
                    "page_table": jnp.broadcast_to(
                        page_table[None, :, :, : cfg.max_pages_per_seq],
                        (cfg.R, cfg.NATT, cfg.B, cfg.max_pages_per_seq),
                    ),
                    "pool": pool_all_regs,
                    "conv": conv_init,
                    "rec": rec_init,
                    "weights": w_norm,
                }
                res = kernel.run_qwen38_dcp8_step(
                    prob,
                    cfg,
                    use_forced_routing=False,
                    return_kv_updates=True,
                    interpret=interpret,
                    mesh=self.mesh,
                )
                pool_updated_u8 = res["pool"].view(jnp.uint8)
                updated_by_reg = [pool_updated_u8[:, reg] for reg in range(cfg.NATT)]
                packed_by_reg = kernel.pack_gdn_states_into_pools(
                    updated_by_reg, res["conv"], res["rec"], mamba_slots, cfg
                )
                pools_out = [
                    packed_by_reg[reg][r]
                    for r in range(cfg.R)
                    for reg in range(cfg.NATT)
                ]
                x_res = res["x"][0]
            else:
                if rank_step_fn is None:
                    msg = "rank_step_fn must be initialized for single-backbone step"
                    raise Qwen38AdapterError(msg)
                pool_u8 = pools_in[0]
                conv_1, rec_1 = kernel.unpack_gdn_states_from_unified_pool(
                    pool_u8, mamba_slots, cfg
                )
                x_res, pool_res, conv_res, rec_res = rank_step_fn(
                    x_in,
                    pos,
                    page_table[:, :, : cfg.max_pages_per_seq],
                    pool_u8.view(jnp.float8_e4m3fn),
                    conv_1,
                    rec_1,
                    per_rank_w,
                )
                pool_packed_u8 = kernel.pack_gdn_states_into_unified_pool(
                    pool_res.view(jnp.uint8),
                    conv_res,
                    rec_res,
                    mamba_slots,
                    cfg,
                )
                pools_out = [pool_packed_u8]

            if out_rows > cfg.B:
                pad = jnp.zeros((out_rows - cfg.B, cfg.H), dtype=x_res.dtype)
                x_res = jnp.concatenate([x_res, pad], axis=0)
            return (*pools_out, x_res)

        step_body = _body if is_cpu_backend else jax.jit(_body)
        mutated_indices = tuple(range(4, 4 + num_pool_args))

        op_fn = _build_pallas_step_op(
            f"qwen38_2_4t_t1_b{cfg.B}_r{out_rows}",
            step_body,
            mutated_arg_indices=mutated_indices,
            out_shape=(out_rows, cfg.H),
            mesh=self.mesh,
        )

        def _step(
            x_in: torch.Tensor,
            pos: torch.Tensor,
            page_table: torch.Tensor,
            mamba_slots: torch.Tensor,
            *rest: torch.Tensor,
        ) -> torch.Tensor:
            """Run custom PyTorch step op for current bucket."""
            return op_fn(x_in, pos, page_table, mamba_slots, *rest)

        return _step

    def _has_layer_metadata(self, attn_md: object) -> bool:
        """Return whether attention and Mamba metadata tables are populated."""
        if not isinstance(attn_md, Mapping) or not self.attn_layer_names:
            return False
        for a_name in self.attn_layer_names:
            md_a = attn_md.get(a_name)
            if (
                getattr(md_a, "block_tables", getattr(md_a, "block_table", None))
                is None
            ):
                return False
        for m_name in self.mamba_layer_names:
            md_m = attn_md.get(m_name)
            if getattr(md_m, "mamba_state_indices", None) is None:
                return False
        return True

    def decide(self, _ctx: object, fwd: ForwardView) -> StepDecision | None:
        """Evaluate request batch eligibility and select target decode bucket."""
        attn_md = fwd.attn_metadata
        if not self._has_layer_metadata(attn_md):
            return StepDecision(
                eligible=False, reason="stock_no_metadata", num_reqs=0, tokens_per_seq=0
            )
        md_map = cast("Mapping[str, object]", attn_md)
        a0 = md_map.get(self.attn_layer_names[0])
        m0 = md_map.get(self.mamba_layer_names[0]) if self.mamba_layer_names else None
        shape = fwd.batch_shape(a0, m0)
        if not getattr(shape, "eligible", False):
            return StepDecision(
                eligible=False, reason="stock_ineligible", num_reqs=0, tokens_per_seq=0
            )

        num_reqs = int(getattr(shape, "num_reqs", 0) or 0)
        toks_per_seq = int(getattr(shape, "tokens_per_seq", 0) or 0)
        if toks_per_seq != 1 or num_reqs <= 0:
            return StepDecision(
                eligible=False,
                reason="stock_prefill",
                num_reqs=num_reqs,
                tokens_per_seq=toks_per_seq,
            )

        bucket_b = next((b for b in self.decode_buckets if b >= num_reqs), None)
        if bucket_b is None or (1, bucket_b) not in fwd.steps:
            return StepDecision(
                eligible=False,
                reason=f"stock_t1_b{num_reqs}",
                num_reqs=num_reqs,
                tokens_per_seq=1,
            )
        return StepDecision(
            eligible=True,
            reason=f"mk_t1_b{bucket_b}",
            num_reqs=num_reqs,
            tokens_per_seq=1,
            bucket_t=bucket_b,
        )

    def forward(
        self, decision: StepDecision, ctx: object, fwd: ForwardView
    ) -> torch.Tensor:
        """Execute forward step using megakernel or dispatch to fallback."""
        cfg = self.base_config
        b_dim = int(decision.bucket_t or decision.num_reqs)
        n_reqs = int(decision.num_reqs)
        attn_md = fwd.attn_metadata or {}
        dev = self.region_pools[0].device

        x_in, pos = prepare_input_tensors(ctx, (n_reqs, b_dim), cfg, dev, self.backbone)
        page_table = prepare_page_table(
            attn_md,
            self.attn_layer_names,
            (n_reqs, b_dim, cfg.max_pages_per_seq),
            dev,
        )
        mamba_slots = prepare_mamba_slots(
            attn_md, self.mamba_layer_names, n_reqs, b_dim, dev
        )

        if self.rank_pools is not None:
            flat_pools = [p for r_pools in self.rank_pools for p in r_pools]
        else:
            if self.unified_pool is None:
                msg = "Qwen38MoeProgram.unified_pool must be bound before execution"
                raise Qwen38AdapterError(msg)
            flat_pools = [self.unified_pool]

        step_fn = fwd.steps[(1, b_dim)]
        out = step_fn(x_in, pos, page_table, mamba_slots, *flat_pools)
        out_rows = int(fwd.num_tokens or n_reqs)
        return out[:out_rows]


def _validate_hf_architecture(text_cfg: object, hf_cfg: object) -> Decline | None:
    """Validate architectures and model type."""
    archs = tuple(getattr(hf_cfg, "architectures", None) or ())
    mtype = str(
        getattr(text_cfg, "model_type", None)
        or getattr(hf_cfg, "model_type", None)
        or ""
    )
    if archs and not (set(archs) & SUPPORTED_ARCHS):
        return Decline(
            _ERR_UNSUPPORTED_ARCH,
            f"Qwen3.8-2.4T plugin does not support architectures {archs}",
        )
    if mtype and mtype not in SUPPORTED_MODEL_TYPES:
        return Decline(
            _ERR_UNSUPPORTED_ARCH,
            f"Qwen3.8-2.4T plugin does not support model_type={mtype!r}",
        )
    return None


def _validate_hf_geometry(
    text_cfg: object, options: Mapping[str, object]
) -> Decline | None:
    """Validate hidden layer dimensions and expert geometry."""
    num_layers = int(
        getattr(text_cfg, "num_hidden_layers", _PROD_NUM_LAYERS) or _PROD_NUM_LAYERS
    )
    hidden_size = int(
        getattr(text_cfg, "hidden_size", _PROD_HIDDEN_SIZE) or _PROD_HIDDEN_SIZE
    )
    allow_mini = bool(options.get("allow_mini_geometry", False))
    is_prod = num_layers == _PROD_NUM_LAYERS and hidden_size == _PROD_HIDDEN_SIZE
    is_mini = allow_mini and (num_layers in (4, 8)) and hidden_size == _MINI_HIDDEN_SIZE
    if not (is_prod or is_mini):
        return Decline(
            _ERR_UNSUPPORTED_GEOM,
            f"Qwen3.8-2.4T requires num_hidden_layers=92, hidden_size=8192 "
            f"(got {num_layers}, {hidden_size})",
        )

    ref_cfg = mini_parity_config() if is_mini else Qwen38Config()
    geom_checks = (
        ("num_attention_heads", ref_cfg.Q_H),
        ("num_key_value_heads", ref_cfg.KV_H),
        ("linear_num_key_heads", ref_cfg.KH),
        ("linear_num_value_heads", ref_cfg.VH),
        ("num_experts", ref_cfg.E),
        ("n_routed_experts", ref_cfg.E),
        ("num_experts_per_tok", ref_cfg.K),
    )
    for attr, expected_val in geom_checks:
        actual_val = getattr(text_cfg, attr, None)
        if actual_val is not None and int(actual_val) != expected_val:
            return Decline(
                _ERR_UNSUPPORTED_GEOM,
                f"Qwen3.8-2.4T requires {attr}={expected_val}, got {actual_val}",
            )
    return None


def _validate_hf_config(
    view: RunnerView, options: Mapping[str, object]
) -> Decline | None:
    """Validate HuggingFace model architectures, geometry, and quantization."""
    hf_cfg = view.hf_config
    if hf_cfg is None:
        return None
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    arch_err = _validate_hf_architecture(text_cfg, hf_cfg)
    if arch_err is not None:
        return arch_err
    geom_err = _validate_hf_geometry(text_cfg, options)
    if geom_err is not None:
        return geom_err

    q_cfg = _extract_quant_config(text_cfg, hf_cfg)
    q_method = str(_quant_field(q_cfg, "quant_method") or "").lower()
    if not _is_nvfp4_scheme(q_cfg, q_method):
        return Decline(
            _ERR_UNSUPPORTED_QUANT,
            f"Qwen3.8-2.4T requires NVFP4 quantization, got {q_method!r}",
        )
    return None


def _validate_runner_view(view: RunnerView) -> Decline | None:
    """Validate parallelism topology, cache layout, and backbone availability."""
    tp_size = int(getattr(view, "tp_size", 0) or 0)
    if tp_size != _EXPECTED_TP_SIZE:
        return Decline(
            _ERR_UNSUPPORTED_TP,
            f"Qwen3.8-2.4T requires tp_size=32, got {tp_size}",
        )

    dcp_size = _dcp_size(view)
    if dcp_size != _EXPECTED_DCP_SIZE:
        return Decline(
            _ERR_UNSUPPORTED_DCP,
            f"Qwen3.8-2.4T requires decode_context_parallel_size=8, got {dcp_size}",
        )

    if not view.unified_kv_layout:
        return Decline(
            _ERR_UNALIGNED_CACHE,
            "Qwen3.8-2.4T requires unified KV/GDN layout (_unified_kv_layout=True)",
        )

    try:
        weights.find_qwen38_2_4t_backbone(view.model)
    except ValueError as exc:
        return Decline(_ERR_BACKBONE_NOT_FOUND, str(exc))
    return None


def create(
    view: RunnerView, options: Mapping[str, object]
) -> Qwen38MoeProgram | Decline:
    """Validate runner/checkpoint eligibility and construct `Qwen38MoeProgram`."""
    unknown = sorted(set(options) - OPTION_KEYS)
    if unknown:
        valid_keys = sorted(OPTION_KEYS)
        err_msg = f"unknown option {unknown[0]!r} for qwen38_2_4t (valid: {valid_keys})"
        raise Qwen38AdapterError(err_msg)

    hf_err = _validate_hf_config(view, options)
    if hf_err is not None:
        return hf_err

    runner_err = _validate_runner_view(view)
    if runner_err is not None:
        return runner_err

    buckets = _parse_buckets(options.get("buckets"))
    if not buckets:
        return Decline(_ERR_EMPTY_BUCKETS, "No valid decode buckets configured")

    hf_cfg = view.hf_config
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg) if hf_cfg is not None else None
    num_layers_cfg = int(
        getattr(text_cfg, "num_hidden_layers", _PROD_NUM_LAYERS) or _PROD_NUM_LAYERS
    )
    base_cfg = (
        dataclasses.replace(mini_parity_config(batch_size=buckets[0]), L=num_layers_cfg)
        if num_layers_cfg in (4, 8)
        else Qwen38Config(B=buckets[0])
    )
    interpret_opt = options.get("interpret")
    interpret_val: bool | pltpu.InterpretParams | None = None
    if interpret_opt is True:
        interpret_val = pltpu.InterpretParams(
            detect_races=False, out_of_bounds_reads="raise"
        )
    elif isinstance(interpret_opt, (bool, pltpu.InterpretParams)):
        interpret_val = interpret_opt

    return Qwen38MoeProgram(
        view.model,
        tp_rank=view.tp_rank,
        decode_buckets=buckets,
        compile_hlo_check=bool(options.get("compile_hlo_check", False)),
        base_config=base_cfg,
        interpret=interpret_val,
    )
