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

import dataclasses
import itertools
import operator
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict, Unpack, cast

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.adapter.jax_bridge import (
    jax_to_torch_cpu,
    make_typed_step_wrapper,
    sharded_jax_callable,
    torch_cpu_to_jax,
)
from vllm_torchtpu.megakernel.core.runtime.contract import (
    Bucket,
    Decline,
    LayoutDecline,
    RunnerShape,
    StepDecision,
    StepFn,
    WarmupCase,
)

from . import kernel, plan, weights
from .shapes import (
    EXPECTED_DCP_SIZE,
    EXPECTED_TP_SIZE,
    MINI_HIDDEN_SIZE,
    NVFP4_GROUP_SIZE,
    PROD_HIDDEN_SIZE,
    PROD_NUM_LAYERS,
    Qwen38Config,
    mini_parity_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jax.sharding import Mesh

    from vllm_torchtpu.megakernel.adapter.hooks import StepContext
    from vllm_torchtpu.megakernel.adapter.plugin import (
        ForwardView,
        RunnerView,
    )
    from vllm_torchtpu.megakernel.core.runtime.stats import BatchShape

    from .weights import BackboneLike

from .constants import (
    DEFAULT_BUCKETS,
    DEFAULT_SINGLE_SHAPE_DIM,
    ERR_BACKBONE_NOT_FOUND,
    ERR_EMPTY_BUCKETS,
    ERR_POOL_MISMATCH,
    ERR_POOL_UNKNOWN,
    ERR_UNALIGNED_CACHE,
    ERR_UNSUPPORTED_ARCH,
    ERR_UNSUPPORTED_DCP,
    ERR_UNSUPPORTED_GEOM,
    ERR_UNSUPPORTED_QUANT,
    ERR_UNSUPPORTED_TP,
    KV_CACHE_NDIM,
    MAMBA_SLOT_TOKENS,
    MINI_ALLOWED_LAYERS,
    NUM_META_ARGS,
    OPTION_KEYS,
    PAGE_TOKENS,
    SUPPORTED_ARCHS,
    SUPPORTED_MODEL_TYPES,
    TP_AXIS_NAME,
)

logger = init_logger(__name__)
_OP_COUNTER = itertools.count()


class Qwen38AdapterError(ValueError):
    """Raised when adapter options or configuration validation fails."""


def _run_custom_op_cpu(
    operands: Sequence[torch.Tensor],
    body_fn: Callable[..., tuple[jax.Array, ...]],
    mutated_arg_indices: tuple[int, ...],
) -> list[torch.Tensor]:
    """Execute JAX step body on CPU host tensors and convert outputs to PyTorch."""
    jax_args = [
        torch_cpu_to_jax(t.view(torch.uint8) if idx in mutated_arg_indices else t)
        for idx, t in enumerate(operands)
    ]
    jax_outs = body_fn(*jax_args)
    torch_pools = [
        jax_to_torch_cpu(p, operands[idx].view(torch.uint8)).to(
            device=operands[idx].device
        )
        for p, idx in zip(jax_outs[:-1], mutated_arg_indices, strict=True)
    ]
    torch_x = jax_to_torch_cpu(jax_outs[-1], operands[0]).to(device=operands[0].device)
    return [*torch_pools, torch_x]


def _run_custom_op_tpu(
    operands: Sequence[torch.Tensor],
    tpu_callable: Callable[..., Sequence[torch.Tensor]],
    mutated_arg_indices: tuple[int, ...],
) -> list[torch.Tensor]:
    """Execute cached XLA sharded callable on TPU device tensors."""
    tpu_operands = [
        t.view(torch.uint8) if idx in mutated_arg_indices else t
        for idx, t in enumerate(operands)
    ]
    tpu_outs = list(tpu_callable(*tpu_operands))
    torch_pools = [p.view(torch.uint8) for p in tpu_outs[:-1]]
    torch_x = tpu_outs[-1].to(dtype=operands[0].dtype)
    return [*torch_pools, torch_x]


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
    qual_name = f"megakernel::{op_prefix}_{op_id}"
    num_inputs = NUM_META_ARGS + len(mutated_arg_indices)
    num_outputs = len(mutated_arg_indices) + 1
    tpu_callable_cache: list[Callable[..., Sequence[torch.Tensor]]] = []

    def _get_tpu_callable() -> Callable[..., Sequence[torch.Tensor]]:
        """Build or return the cached sharded JAX TPU callable."""
        if not tpu_callable_cache:
            active_mesh = mesh or jax.sharding.Mesh(jax.devices(), (TP_AXIS_NAME,))
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
            return _run_custom_op_cpu(operands, body_fn, mutated_arg_indices)
        return _run_custom_op_tpu(operands, _get_tpu_callable(), mutated_arg_indices)

    @_op.register_fake
    def _fake(operands: list[torch.Tensor]) -> list[torch.Tensor]:
        """Register fake kernel output shapes for tracing."""
        fake_pools = [
            torch.empty_like(operands[idx].view(torch.uint8))
            for idx in mutated_arg_indices
        ]
        return [*fake_pools, operands[0].new_empty((out_rows, hidden_size))]

    def _step(*args: torch.Tensor) -> torch.Tensor:
        """Execute custom op and copy mutated pools back in place."""
        dyn_args = list(args[:num_inputs])
        outs = _op(dyn_args)
        for p_out, idx in zip(outs[:-1], mutated_arg_indices, strict=True):
            dyn_args[idx].view(torch.uint8).copy_(p_out)
        return outs[-1]

    return _step


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen38LayoutRequest:
    """Cache and page layout request returned by `qwen38_2_4t` `layout_request`."""

    block_size: int
    cp_kv_cache_interleave_size: int
    mamba_slot_tokens: int

    @property
    def slot_tokens_floor(self) -> int:
        """Minimum physical token extent per slot across KV pages and GDN state."""
        return max(self.block_size, self.mamba_slot_tokens)


LayoutRequest = Qwen38LayoutRequest


def layout_request(
    vllm_config: object,
    _options: Mapping[str, object] | None = None,
) -> Qwen38LayoutRequest:
    """Request 128-token pages, DCP interleave, and 640-token Mamba slots."""
    if vllm_config is None:
        msg = "vllm_config must not be None"
        raise Qwen38AdapterError(msg)
    return Qwen38LayoutRequest(
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
    if group_size is not None and int(str(group_size)) != NVFP4_GROUP_SIZE:
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


def _unwrap_kv_cache(holder: object) -> torch.Tensor | None:
    """Extract primary KV cache tensor from a holder list, tuple, or tensor."""
    raw = getattr(holder, "kv_cache", None)
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    return raw if isinstance(raw, torch.Tensor) else None


def _layer_name(mod: object, parent_mod: object, l_idx: int) -> str:
    """Resolve explicit vLLM layer name from `layer_name` or `prefix` fail-closed."""
    candidate = (
        getattr(mod, "layer_name", None)
        or getattr(mod, "prefix", None)
        or getattr(parent_mod, "prefix", None)
    )
    if isinstance(candidate, str) and candidate:
        return candidate
    err_msg = f"Layer {l_idx} is missing explicit vLLM layer_name or prefix"
    raise LayoutDecline(Decline(ERR_POOL_UNKNOWN, err_msg).result)


def _collect_region_pools_and_names(
    backbone: BackboneLike, cfg: Qwen38Config
) -> tuple[list[torch.Tensor], list[str], list[str], tuple[int, ...]]:
    """Collect unified KV/GDN region pools, layer names, and GDN region map."""
    pools: list[torch.Tensor] = []
    attn_names: list[str] = []
    mamba_names: list[str] = []
    gdn_tensors: list[tuple[int, torch.Tensor]] = []
    expected_tail = (2, cfg.K_R, cfg.D, cfg.page_size)

    for l_idx in range(cfg.L):
        layer = backbone.layers[l_idx]
        if (l_idx + 1) % cfg.att_every == 0:
            sa = getattr(layer, "self_attn", None)
            attn_mod = getattr(sa, "attn", sa)
            attn_names.append(_layer_name(attn_mod, sa, l_idx))
            kv_tensor = _unwrap_kv_cache(attn_mod)
            if kv_tensor is None:
                err_msg = f"Missing attention kv_cache tensor on layer {l_idx}"
                raise LayoutDecline(Decline(ERR_POOL_UNKNOWN, err_msg).result)
            if (
                kv_tensor.ndim != KV_CACHE_NDIM
                or tuple(kv_tensor.shape[1:KV_CACHE_NDIM]) != expected_tail
            ):
                err_msg = (
                    f"Layer {l_idx} kv_cache shape {tuple(kv_tensor.shape)} != "
                    f"(num_pages, 2, {cfg.K_R}, {cfg.D}, {cfg.page_size})"
                )
                raise LayoutDecline(Decline(ERR_POOL_MISMATCH, err_msg).result)
            pools.append(kv_tensor)
        else:
            la = getattr(layer, "linear_attn", None)
            mamba_names.append(_layer_name(la, la, l_idx))
            la_tensor = _unwrap_kv_cache(la)
            if la_tensor is None:
                err_msg = f"Missing linear_attn kv_cache tensor on layer {l_idx}"
                raise LayoutDecline(Decline(ERR_POOL_UNKNOWN, err_msg).result)
            gdn_tensors.append((l_idx, la_tensor))

    ptr_to_reg = {pool.data_ptr(): idx for idx, pool in enumerate(pools)}
    gdn_regions: list[int] = []
    for l_idx, la_tensor in gdn_tensors:
        reg_idx = ptr_to_reg.get(la_tensor.data_ptr())
        if reg_idx is None:
            err_msg = (
                f"GDN layer {l_idx} pool is not aliased with any attention region pool"
            )
            raise LayoutDecline(Decline(ERR_POOL_MISMATCH, err_msg).result)
        gdn_regions.append(reg_idx)
    return pools, attn_names, mamba_names, tuple(gdn_regions)


def _pad_1d_or_2d(
    tensor: torch.Tensor,
    target_shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    dev: torch.device,
    pad_value: float = 0,
) -> torch.Tensor:
    """Copy `tensor` into a padded contiguous buffer of `target_shape` on `dev`."""
    t = tensor.to(device=dev, dtype=dtype)
    if t.shape == target_shape:
        return t.contiguous()
    if len(target_shape) == DEFAULT_SINGLE_SHAPE_DIM:
        rows = target_shape[0]
        if t.shape[0] >= rows:
            return t[:rows].contiguous()
        buf = torch.full((rows,), pad_value, dtype=dtype, device=dev)
        buf[: t.shape[0]] = t
        return buf
    rows, cols = target_shape
    if t.shape[0] >= rows and t.shape[1] >= cols:
        return t[:rows, :cols].contiguous()
    buf = torch.full((rows, cols), pad_value, dtype=dtype, device=dev)
    r_copy = min(t.shape[0], rows)
    c_copy = min(t.shape[1], cols)
    buf[:r_copy, :c_copy] = t[:r_copy, :c_copy]
    return buf


def prepare_input_tensors(
    ctx: StepContext,
    batch_dims: tuple[int, int],
    cfg: Qwen38Config,
    dev: torch.device,
    backbone: BackboneLike,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare padded activations and position vectors for step execution."""
    n_reqs, b_dim = batch_dims
    inputs_embeds = ctx.inputs_embeds
    input_ids = ctx.input_ids
    if inputs_embeds is not None:
        x_raw: torch.Tensor = cast("torch.Tensor", inputs_embeds)[:n_reqs]
    else:
        embed_fn = getattr(backbone, "embed_tokens", None)
        if not callable(embed_fn) or input_ids is None:
            msg = "Model is missing callable embed_tokens required to embed input_ids"
            raise RuntimeError(msg)
        x_raw = cast("torch.Tensor", embed_fn(input_ids[:n_reqs]))

    x_in = _pad_1d_or_2d(
        x_raw, (b_dim, cfg.H), dtype=torch.bfloat16, dev=dev, pad_value=0
    )
    if ctx.positions is None:
        msg = "StepContext is missing required positions tensor"
        raise RuntimeError(msg)
    pos = _pad_1d_or_2d(
        ctx.positions[:n_reqs].reshape(-1),
        (b_dim,),
        dtype=torch.int32,
        dev=dev,
        pad_value=-1,
    )
    return x_in, pos


def prepare_page_table(
    attn_md: Mapping[str, object],
    attn_layer_names: Sequence[str],
    dims: tuple[int, int, int],
    dev: torch.device,
) -> torch.Tensor:
    """Stage and pad `(NATT, B, max_pages_per_seq)` page tables in one upload."""
    n_reqs, b_dim, max_pages = dims
    staged = torch.zeros(
        (len(attn_layer_names), b_dim, max_pages),
        dtype=torch.int32,
        device="cpu",
    )
    rows_cap = min(n_reqs, b_dim)
    cpu_cache: dict[int, torch.Tensor] = {}
    for layer_idx, a_name in enumerate(attn_layer_names):
        md_a = attn_md.get(a_name)
        bt = getattr(md_a, "block_tables", None)
        if bt is None:
            msg = f"Missing block_tables in attention metadata for layer {a_name!r}"
            raise RuntimeError(msg)
        bt_tensor = cast("torch.Tensor", bt)
        bt_cpu = cpu_cache.get(id(bt_tensor))
        if bt_cpu is None:
            bt_cpu = bt_tensor.detach().to(device="cpu", dtype=torch.int32)
            cpu_cache[id(bt_tensor)] = bt_cpu
        r_copy = min(int(bt_cpu.shape[0]), rows_cap)
        c_copy = min(int(bt_cpu.shape[1]), max_pages)
        staged[layer_idx, :r_copy, :c_copy] = bt_cpu[:r_copy, :c_copy]
    return staged if dev.type == "cpu" else staged.to(device=dev)


def prepare_mamba_slots(
    attn_md: Mapping[str, object],
    mamba_layer_names: Sequence[str],
    n_reqs: int,
    b_dim: int,
    dev: torch.device,
) -> torch.Tensor:
    """Stage and pad `(NGDN, B)` Mamba slot indices in one upload."""
    staged = torch.full(
        (len(mamba_layer_names), b_dim),
        -1,
        dtype=torch.int32,
        device="cpu",
    )
    rows_cap = min(n_reqs, b_dim)
    cpu_cache: dict[int, torch.Tensor] = {}
    for layer_idx, m_name in enumerate(mamba_layer_names):
        md_m = attn_md.get(m_name)
        ms = getattr(md_m, "mamba_state_indices", None)
        if ms is None:
            msg = (
                "Missing mamba_state_indices in attention metadata for layer "
                f"{m_name!r}"
            )
            raise RuntimeError(msg)
        ms_tensor = cast("torch.Tensor", ms)
        ms_cpu = cpu_cache.get(id(ms_tensor))
        if ms_cpu is None:
            ms_cpu = ms_tensor.detach().reshape(-1).to(device="cpu", dtype=torch.int32)
            cpu_cache[id(ms_tensor)] = ms_cpu
        r_copy = min(int(ms_cpu.shape[0]), rows_cap)
        staged[layer_idx, :r_copy] = ms_cpu[:r_copy]
    return staged if dev.type == "cpu" else staged.to(device=dev)


def _exec_multi_rank_step(
    jax_args: Sequence[jax.Array],
    cfg: Qwen38Config,
    w_norm: dict[str, jax.Array],
    *,
    step_opts: tuple[Mesh | None, bool | pltpu.InterpretParams, Sequence[int] | None],
) -> tuple[list[jax.Array], jax.Array]:
    """Run sharded `R`-rank step across split region pools on CPU or TPU."""
    mesh, interpret, gdn_region_map = step_opts
    x_in, pos, page_table, mamba_slots = jax_args[:NUM_META_ARGS]
    pools_in = list(jax_args[NUM_META_ARGS : NUM_META_ARGS + cfg.R * cfg.NATT])
    pools_by_reg = [
        jnp.stack([pools_in[r * cfg.NATT + reg] for r in range(cfg.R)], axis=0)
        for reg in range(cfg.NATT)
    ]
    conv_init, rec_init = kernel.unpack_gdn_states_from_pools(
        pools_by_reg, mamba_slots, cfg, gdn_region_map=gdn_region_map
    )
    prob: dict[str, jax.Array | dict[str, jax.Array]] = {
        "x": jnp.broadcast_to(x_in[None, ...], (cfg.R, cfg.B, cfg.H)),
        "pos": jnp.broadcast_to(pos[None, :], (cfg.R, cfg.B)),
        "page_table": jnp.broadcast_to(
            page_table[None, :, :, : cfg.max_pages_per_seq],
            (cfg.R, cfg.NATT, cfg.B, cfg.max_pages_per_seq),
        ),
        "pool": jnp.stack(pools_by_reg, axis=1).view(jnp.float8_e4m3fn),
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
        mesh=mesh,
    )
    pool_u8 = res["pool"].view(jnp.uint8)
    packed_by_reg = kernel.pack_gdn_states_into_pools(
        [pool_u8[:, reg] for reg in range(cfg.NATT)],
        (res["conv"], res["rec"]),
        mamba_slots,
        cfg,
        gdn_region_map=gdn_region_map,
    )
    pools_out: list[jax.Array] = []
    for r_idx in range(cfg.R):
        pools_out.extend(packed_by_reg[reg][r_idx] for reg in range(cfg.NATT))
    return pools_out, res["x"][0]


def _exec_single_rank_step(
    jax_args: Sequence[jax.Array],
    cfg: Qwen38Config,
    per_rank_w: dict[str, jax.Array],
    rank_step_fn: Callable[
        ..., tuple[jax.Array, tuple[jax.Array, ...], jax.Array, jax.Array]
    ],
    *,
    gdn_region_map: Sequence[int] | None = None,
) -> tuple[list[jax.Array], jax.Array]:
    """Run single-rank step in place over donated runner-owned region pools."""
    x_in, pos, page_table, mamba_slots = jax_args[:NUM_META_ARGS]
    pools_u8 = list(jax_args[NUM_META_ARGS : NUM_META_ARGS + cfg.NATT])
    conv_1, rec_1 = kernel.unpack_gdn_states_from_pools(
        pools_u8, mamba_slots, cfg, gdn_region_map=gdn_region_map
    )
    pools_fp8 = tuple(p.view(jnp.float8_e4m3fn) for p in pools_u8)
    x_res, pools_res, conv_res, rec_res = rank_step_fn(
        x_in,
        pos,
        page_table[:, :, : cfg.max_pages_per_seq],
        pools_fp8,
        conv_1,
        rec_1,
        per_rank_w,
    )
    pools_res_u8 = [p.view(jnp.uint8) for p in pools_res]
    pools_packed_u8 = kernel.pack_gdn_states_into_pools(
        pools_res_u8,
        (conv_res, rec_res),
        mamba_slots,
        cfg,
        gdn_region_map=gdn_region_map,
    )
    return pools_packed_u8, x_res


def _has_block_table(md_a: object) -> bool:
    """Return True if attention layer metadata exposes block table indices."""
    if md_a is None:
        return False
    return getattr(md_a, "block_tables", None) is not None


class _WarmupSession:
    """Context manager preserving touched page 0 of cache pools during warmup."""

    def __init__(self, program: Qwen38MoeProgram) -> None:
        """Initializes the instance."""
        self._prog = program

    def __enter__(self) -> None:
        """Clear warmed buckets and snapshot page 0 of each pool tensor."""
        self._prog.snapshot_pools_for_warmup()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Restore touched page 0 of cache pool tensor buffers from snapshot."""
        self._prog.restore_pools_after_warmup()


class Qwen38MoeProgram:
    """`MonoProgram` implementation for Qwen3.8-2.4T TP32xDCP8 NVFP4."""

    display_name = "Qwen3.8-2.4T"

    def __init__(
        self,
        model: object,
        **options: Unpack[ProgramOptions],
    ) -> None:
        """Initializes the instance."""
        self.model = model
        self.backbone = weights.find_backbone(model)
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
        self.gdn_region_map: tuple[int, ...] = ()
        self.region_pools: list[torch.Tensor] = []
        self.rank_pools: list[list[torch.Tensor]] | None = None
        self._pool_snapshots: list[list[torch.Tensor]] | None = None
        self._warmed_buckets: set[tuple[int, int]] = set()

    def buckets(self, token_paddings: Sequence[int]) -> Sequence[Bucket]:
        """Compute decode buckets covering requested token padding thresholds."""
        pads = [int(p) for p in token_paddings if int(p) > 0]
        return [
            Bucket(T=1, B=b, out_rows=next((p for p in pads if p >= b), b))
            for b in self.decode_buckets
        ]

    def set_interpret_mode(
        self, *, interpret: bool | pltpu.InterpretParams | None = None
    ) -> None:
        """Update Pallas interpret execution mode used on subsequent `build` calls."""
        self._interpret = interpret

    def load(self, mesh: Mesh | None) -> None:
        """Bind device mesh and extract normalized weights for Pallas execution."""
        self.mesh = mesh
        raw_norm_w = self.backbone.norm.weight
        device: torch.device | str = (
            raw_norm_w.device if isinstance(raw_norm_w, torch.Tensor) else "cpu"
        )
        self.w_dict = weights.extract_backbone_weights_torch(
            self.model,
            self.base_config,
            tp_rank=self.tp_rank,
            device=device,
        )
        self.w_flat = [self.w_dict[k] for k in weights.WEIGHT_KEYS]
        self._sync_jax_weights()

    def bind(self, view: RunnerView) -> dict[str, int]:
        """Inspect runner view cache layout and bind runner-owned region pools."""
        slot_tokens = _effective_slot_tokens(view)
        if slot_tokens < MAMBA_SLOT_TOKENS:
            err_msg = (
                f"Qwen3.8 GDN state requires >= {MAMBA_SLOT_TOKENS} tokens "
                f"(5 pages) per slot, got cache_block_size={slot_tokens}"
            )
            raise LayoutDecline(Decline(ERR_POOL_MISMATCH, err_msg).result)

        if isinstance(self.model, weights.MultiRankBackboneHolder):
            self.rank_pools = [
                self._collect_region_pools(cast("BackboneLike", b))
                for b in self.model.rank_backbones
            ]
            self.region_pools = self.rank_pools[0]
        else:
            self.region_pools = self._collect_region_pools(self.backbone)
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

    def warmup_session(self) -> _WarmupSession:
        """Return warmup session context manager preserving touched page 0."""
        return _WarmupSession(self)

    def snapshot_pools_for_warmup(self) -> None:
        """Clear warmed buckets and snapshot page 0 of each pool tensor."""
        self._warmed_buckets.clear()
        pool_lists = (
            self.rank_pools if self.rank_pools is not None else [self.region_pools]
        )
        self._pool_snapshots = [
            [p[:1].view(torch.uint8).clone() for p in r_pools] for r_pools in pool_lists
        ]

    def restore_pools_after_warmup(self) -> None:
        """Restore touched page 0 of cache pool tensor buffers from snapshot."""
        self._restore_pools()
        self._pool_snapshots = None
        self._warmed_buckets.clear()

    def warmup_inputs(self, bucket: Bucket, shape: RunnerShape) -> WarmupCase | None:
        """Construct synthetic trace inputs for compile-time warmup, or `None`."""
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
        args = (
            torch.zeros((b_dim, cfg.H), dtype=torch.bfloat16, device=dev),
            torch.zeros((b_dim,), dtype=torch.int32, device=dev),
            torch.zeros(
                (cfg.NATT, b_dim, cfg.max_pages_per_seq), dtype=torch.int32, device=dev
            ),
            torch.full((cfg.NGDN, b_dim), -1, dtype=torch.int32, device=dev),
            *self._flat_pool_args(),
        )
        return WarmupCase(args=args, check=lambda _out: self._restore_pools())

    def build(self, bucket: Bucket) -> StepFn:
        """Compile and return the executable PyTorch step function for `bucket`."""
        cfg = dataclasses.replace(self.base_config, B=int(bucket.B))
        out_rows = int(bucket.out_rows or bucket.B)
        multi_rank_host = self.rank_pools is not None
        num_pool_args = cfg.R * cfg.NATT if multi_rank_host else cfg.NATT
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
                "MultiRankBackboneHolder.rank_backbones to supply per-rank shards "
                "and pools"
            )
            raise Qwen38AdapterError(msg)

        self._sync_jax_weights()
        w_norm = self._jax_weights
        per_rank_w = self._per_rank_w
        gdn_map = self.gdn_region_map or None
        rank_step_fn = (
            None
            if multi_rank_host
            else self._build_rank_step_fn(cfg, interpret=interpret)
        )

        def _body(*jax_args: jax.Array) -> tuple[jax.Array, ...]:
            """Execute sharded Pallas megakernel step on JAX arrays."""
            if rank_step_fn is None:
                pools_out, x_res = _exec_multi_rank_step(
                    jax_args,
                    cfg,
                    w_norm,
                    step_opts=(self.mesh, interpret, gdn_map),
                )
            else:
                pools_out, x_res = _exec_single_rank_step(
                    jax_args,
                    cfg,
                    per_rank_w,
                    rank_step_fn,
                    gdn_region_map=gdn_map,
                )
            if out_rows > cfg.B:
                pad = jnp.zeros((out_rows - cfg.B, cfg.H), dtype=x_res.dtype)
                x_res = jnp.concatenate([x_res, pad], axis=0)
            return (*pools_out, x_res)

        step_body = _body if is_cpu_backend else jax.jit(_body)
        return _build_pallas_step_op(
            f"qwen38_2_4t_t1_b{cfg.B}_r{out_rows}",
            step_body,
            mutated_arg_indices=tuple(
                range(NUM_META_ARGS, NUM_META_ARGS + num_pool_args)
            ),
            out_shape=(out_rows, cfg.H),
            mesh=self.mesh,
        )

    def decide(self, _ctx: object, fwd: ForwardView) -> StepDecision:
        """Evaluate request batch eligibility and select target decode bucket."""
        attn_md = fwd.attn_metadata
        if not self._has_layer_metadata(attn_md):
            return StepDecision(
                eligible=False,
                reason="stock_no_metadata",
                num_reqs=0,
                tokens_per_seq=0,
            )
        md_map = cast("Mapping[str, object]", attn_md)
        a0 = md_map.get(self.attn_layer_names[0])
        m0 = md_map.get(self.mamba_layer_names[0]) if self.mamba_layer_names else None
        shape: BatchShape = fwd.batch_shape(a0, m0)
        if not shape.eligible:
            return StepDecision(
                eligible=False,
                reason=shape.reason or "stock_ineligible",
                num_reqs=int(shape.num_reqs),
                tokens_per_seq=int(shape.tokens_per_req),
            )

        num_reqs = int(shape.num_reqs)
        toks_per_req = int(shape.tokens_per_req)
        if toks_per_req != 1 or num_reqs <= 0:
            return StepDecision(
                eligible=False,
                reason="stock_prefill",
                num_reqs=num_reqs,
                tokens_per_seq=toks_per_req,
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
        self, decision: StepDecision, ctx: StepContext, fwd: ForwardView
    ) -> torch.Tensor:
        """Execute forward step using megakernel over bound unified cache pools."""
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
        step_fn = fwd.steps[(1, b_dim)]
        out = step_fn(x_in, pos, page_table, mamba_slots, *self._flat_pool_args())
        out_rows = int(fwd.num_tokens or n_reqs)
        return out[:out_rows]

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
            sharding = jax.sharding.NamedSharding(self.mesh, P(TP_AXIS_NAME))
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

    def _collect_region_pools(self, backbone: BackboneLike) -> list[torch.Tensor]:
        """Collect and validate unified KV/GDN region cache pools."""
        pools, attn_names, mamba_names, gdn_regions = _collect_region_pools_and_names(
            backbone, self.base_config
        )
        if not self.attn_layer_names:
            self.attn_layer_names = attn_names
            self.mamba_layer_names = mamba_names
            self.gdn_region_map = gdn_regions
        elif gdn_regions != self.gdn_region_map:
            err_msg = "Inconsistent GDN-to-region pool mapping across ranks"
            raise LayoutDecline(Decline(ERR_POOL_MISMATCH, err_msg).result)
        return pools

    def _restore_pools(self) -> None:
        """Restore touched page 0 of cache pool tensor buffers from snapshot."""
        if self._pool_snapshots is None:
            return
        pool_lists = (
            self.rank_pools if self.rank_pools is not None else [self.region_pools]
        )
        for r_pools, r_snaps in zip(pool_lists, self._pool_snapshots, strict=True):
            for p, s in zip(r_pools, r_snaps, strict=True):
                p[:1].view(torch.uint8).copy_(s)

    def _flat_pool_args(self) -> list[torch.Tensor]:
        """Return ordered HBM pool tensor arguments matching `build()` layout."""
        if self.rank_pools is not None:
            flat_pools: list[torch.Tensor] = []
            for rank_pool_list in self.rank_pools:
                flat_pools.extend(rank_pool_list)
            return flat_pools
        if not self.region_pools:
            msg = "Qwen38MoeProgram.region_pools must be bound before execution"
            raise Qwen38AdapterError(msg)
        return list(self.region_pools)

    def _build_rank_step_fn(
        self,
        cfg: Qwen38Config,
        *,
        interpret: bool | pltpu.InterpretParams,
    ) -> Callable[..., tuple[jax.Array, tuple[jax.Array, ...], jax.Array, jax.Array]]:
        """Build single-backbone per-rank Pallas step callable."""
        per_rank_w = self._per_rank_w
        raw_rank_step_fn = kernel.make_pallas_step_fn(
            cfg,
            per_rank_w,
            has_forced_routing=False,
            interpret=interpret,
            axis_name=TP_AXIS_NAME if cfg.R > 1 else None,
        )
        if cfg.R <= 1:
            return raw_rank_step_fn
        active_mesh = self.mesh or jax.sharding.Mesh(
            jax.devices()[: cfg.R], (TP_AXIS_NAME,)
        )
        w_specs = {k: P() for k in per_rank_w}
        pool_specs = tuple(P() for _ in range(cfg.NATT))
        return jax.shard_map(
            raw_rank_step_fn,
            mesh=active_mesh,
            in_specs=(P(), P(), P(), pool_specs, P(), P(), w_specs),
            out_specs=(P(), pool_specs, P(), P()),
            check_vma=False,
        )

    def _has_layer_metadata(self, attn_md: object) -> bool:
        """Return whether every attention and Mamba layer has metadata populated."""
        if not isinstance(attn_md, Mapping) or not self.attn_layer_names:
            return False
        has_attn = all(
            _has_block_table(attn_md.get(name)) for name in self.attn_layer_names
        )
        has_mamba = all(
            getattr(attn_md.get(name), "mamba_state_indices", None) is not None
            for name in self.mamba_layer_names
        )
        return has_attn and has_mamba


def _resolve_text_config(hf_cfg: object) -> object:
    """Return text config sub-object if present, else root hf config."""
    return getattr(hf_cfg, "text_config", hf_cfg)


def _cfg_field(obj: object, name: str, default: object = None) -> object:
    """Safely read attribute from config namespace or mapping."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _validate_hf_architecture(text_cfg: object, hf_cfg: object) -> Decline | None:
    """Validate architectures and model type."""
    raw_archs = _cfg_field(hf_cfg, "architectures")
    archs: tuple[object, ...] = (
        tuple(raw_archs) if isinstance(raw_archs, (list, tuple)) else ()
    )
    mtype = str(
        _cfg_field(text_cfg, "model_type") or _cfg_field(hf_cfg, "model_type") or ""
    )
    if archs and not (set(archs) & SUPPORTED_ARCHS):
        return Decline(
            ERR_UNSUPPORTED_ARCH,
            f"Qwen3.8-2.4T plugin does not support architectures {archs}",
        )
    if mtype and mtype not in SUPPORTED_MODEL_TYPES:
        return Decline(
            ERR_UNSUPPORTED_ARCH,
            f"Qwen3.8-2.4T plugin does not support model_type={mtype!r}",
        )
    return None


def _validate_hf_geometry(
    text_cfg: object, options: Mapping[str, object]
) -> Decline | None:
    """Validate hidden layer dimensions and expert geometry fail-closed."""
    raw_layers = _cfg_field(text_cfg, "num_hidden_layers")
    raw_hidden = _cfg_field(text_cfg, "hidden_size")
    if raw_layers is None or raw_hidden is None:
        return Decline(
            ERR_UNSUPPORTED_GEOM,
            "Qwen3.8-2.4T requires explicit num_hidden_layers and hidden_size "
            f"(got {raw_layers!r}, {raw_hidden!r})",
        )
    num_layers = int(cast("int", raw_layers))
    hidden_size = int(cast("int", raw_hidden))
    allow_mini = bool(options.get("allow_mini_geometry", False))
    is_prod = num_layers == PROD_NUM_LAYERS and hidden_size == PROD_HIDDEN_SIZE
    is_mini = (
        allow_mini
        and num_layers in MINI_ALLOWED_LAYERS
        and hidden_size == MINI_HIDDEN_SIZE
    )
    if not (is_prod or is_mini):
        return Decline(
            ERR_UNSUPPORTED_GEOM,
            f"Qwen3.8-2.4T requires num_hidden_layers={PROD_NUM_LAYERS}, "
            f"hidden_size={PROD_HIDDEN_SIZE} (got {num_layers}, {hidden_size})",
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
        actual_val = _cfg_field(text_cfg, attr)
        if actual_val is not None and int(cast("int", actual_val)) != expected_val:
            return Decline(
                ERR_UNSUPPORTED_GEOM,
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
    text_cfg = _resolve_text_config(hf_cfg)
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
            ERR_UNSUPPORTED_QUANT,
            f"Qwen3.8-2.4T requires NVFP4 quantization, got {q_method!r}",
        )
    return None


def _validate_runner_view(view: RunnerView) -> Decline | None:
    """Validate parallelism topology, cache layout, and backbone availability."""
    tp_size = int(view.tp_size)
    if tp_size != EXPECTED_TP_SIZE:
        return Decline(
            ERR_UNSUPPORTED_TP,
            f"Qwen3.8-2.4T requires tp_size={EXPECTED_TP_SIZE}, got {tp_size}",
        )

    dcp_size = _dcp_size(view)
    if dcp_size != EXPECTED_DCP_SIZE:
        return Decline(
            ERR_UNSUPPORTED_DCP,
            f"Qwen3.8-2.4T requires decode_context_parallel_size={EXPECTED_DCP_SIZE}, "
            f"got {dcp_size}",
        )

    if not view.unified_kv_layout:
        return Decline(
            ERR_UNALIGNED_CACHE,
            "Qwen3.8-2.4T requires unified KV/GDN layout (_unified_kv_layout=True)",
        )

    try:
        weights.find_backbone(view.model)
    except ValueError as exc:
        return Decline(ERR_BACKBONE_NOT_FOUND, str(exc))
    return None


def _create_program(
    view: RunnerView, options: Mapping[str, object]
) -> Qwen38MoeProgram | Decline:
    """Internal program construction and validation helper."""
    unknown = sorted(set(options) - OPTION_KEYS)
    if unknown:
        valid_keys = sorted(OPTION_KEYS)
        err_msg = f"unknown option {unknown[0]!r} for qwen38_2_4t (valid: {valid_keys})"
        raise Qwen38AdapterError(err_msg)

    hf_decline = _validate_hf_config(view, options)
    if hf_decline is not None:
        return hf_decline
    runner_decline = _validate_runner_view(view)
    if runner_decline is not None:
        return runner_decline

    buckets = _parse_buckets(options.get("buckets"))
    if not buckets:
        return Decline(ERR_EMPTY_BUCKETS, "No valid decode buckets configured")

    hf_cfg = view.hf_config
    text_cfg = _resolve_text_config(hf_cfg) if hf_cfg is not None else None
    raw_layers = (
        PROD_NUM_LAYERS
        if text_cfg is None
        else _cfg_field(text_cfg, "num_hidden_layers", PROD_NUM_LAYERS)
    )
    num_layers_cfg = int(cast("int", raw_layers))
    base_cfg = (
        dataclasses.replace(mini_parity_config(batch_size=buckets[0]), L=num_layers_cfg)
        if num_layers_cfg in MINI_ALLOWED_LAYERS
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


def create(
    view: RunnerView, options: Mapping[str, object]
) -> Qwen38MoeProgram | Decline:
    """Validate runner/checkpoint eligibility and construct `Qwen38MoeProgram`."""
    program_or_decline = _create_program(view, options)
    if isinstance(program_or_decline, Decline):
        logger.info_once(
            "Qwen3.8-2.4T megakernel declined (%s): %s",
            program_or_decline.code,
            program_or_decline.message,
        )
    return program_or_decline
