"""Drop-in `fused_moe_tpuc` matching `vllm_torchtpu` `fused_moe_gmm` on TPU v7x.

Implements `fused_moe_tpuc` with the exact function signature and tensor
contract of `vllm_torchtpu.layers.adapter.fused_moe.fused_moe_gmm`, backed by
the Ghostfish (TPU v7x) bare-metal TPU-C MoE kernel (`moe_fused_expert_ffn_gf`):

  Stage 1 (`torch_tpu._internal.pallas.jax_op`s):
    - `_ep_prep_sc_jax` (TensorCore): quantizes `hidden_states`
      (`bf16[T, H]`) once per token to `float8_e4m3fn` with per-token scale,
      filters global `topk_ids` (`int32[T, topk]`) to local experts
      `[experts_start, experts_start + E_local)`, orders pairs with a stable
      `argsort` into 32-row-aligned per-expert groups and builds `rw_b:
      f32[R, 128]`, `meta: int32[256]` and the per-pair routed row `dest`.
    - `_ep_dispatch_sc_jax`: SparseCore `ragged_gather_v2` (the kernel
      `fused_moe_gmm` uses for EP dispatch) moves packed FP8 token rows and
      scales into routed order (`xq: f8e4m3fn[R, H]`, `xs_b: f32[R, 128]`).

  Stage 2 (`torch.ops.deepsea_moe.expert_ffn_out`):
    - Zero-copy execution of `moe_fused_expert_ffn_gf` on `w1` (`f8e4m3fn[E,
      H, 2I]`) and `w2` (`f8e4m3fn[E, I, H]`), writing routing-weighted expert
      outputs into `out_tiles: bf16[R // 16, H // 128, 16, 128]`.

  Stage 3 (`_ep_combine_sc_jax`): SparseCore `ragged_gather_reduce` gathers
    each pair's routed row (`dest: int32[T * topk]`, -1 for non-local pairs)
    and sums over `topk` to return `bf16[T, H]`.

  Supported MoE dims (H, I): (4096, 1024) for Qwen3.5-397B-A17B and (2048, 512)
  for Qwen3.5-35B-A3B; each has its own kernel build.

  `TPU_MOE_STAGE_TIMES=1` records per-stage wall times in `STAGE_TIMES`.
"""

import functools
import importlib
import os
import time
from typing import Any

import torch

os.environ.setdefault("TPU_MOE_SYNC_EXECUTE", "1")

E_MAX = 80
# (hidden, moe_intermediate) pairs with a TPU-C kernel build.
SUPPORTED_DIMS = ((4096, 1024), (2048, 512))
GROUP = 32
META_WORDS = 256
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0

_BRIDGE_LOADED = False
_TPU_SYNC_MOD: Any = None
_WEIGHT_CACHE: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
_ZERO_START_CACHE: dict[torch.device, torch.Tensor] = {}
_JAX_OPS: dict[tuple[Any, ...], Any] = {}
_OUT_BUFFERS: dict[tuple[Any, ...], torch.Tensor] = {}
# Per-call wall-clock stage times in ms, recorded when TPU_MOE_STAGE_TIMES=1.
STAGE_TIMES: list[dict[str, float]] = []


def _tpu_sync(tensors: Any, *, wait: bool) -> None:
  """Synchronizes `tensors` via `torch_tpu._internal.sync.synchronize`."""
  global _TPU_SYNC_MOD
  if _TPU_SYNC_MOD is None:
    _TPU_SYNC_MOD = importlib.import_module("torch_tpu._internal.sync")
  _TPU_SYNC_MOD.synchronize(tensors, wait=wait)


@functools.cache
def _sc_kernels():
  """Returns vllm_torchtpu's SparseCore (ragged_gather, ragged_gather_reduce).

  These are the same SparseCore token-movement kernels `fused_moe_gmm` uses
  for EP dispatch/combine.
  """
  rg = importlib.import_module(
      "vllm_torchtpu.kernels.sparse_core.ragged_gather_v2"
  )
  # `ragged_gather_reduce(x, indices, topk_weights, valid_rows_mask,
  # reduce_group_size)`; v3 preferred, v1 fallback.
  for name in ("ragged_gather_reduce_v3", "ragged_gather_reduce"):
    try:
      rgr = importlib.import_module(f"vllm_torchtpu.kernels.sparse_core.{name}")
    except ImportError:
      continue
    if hasattr(rgr, "ragged_gather_reduce"):
      return rg.ragged_gather_v2, rgr.ragged_gather_reduce
  raise ImportError(
      "vllm_torchtpu.kernels.sparse_core has no ragged_gather_reduce"
  )


def _dense_compute_ctx():
  """Pins JAX ops to TensorCore (`_xla_compute_type="dense"`) to prevent SC offload."""
  xla_meta = importlib.import_module("jax.experimental.xla_metadata")
  return xla_meta.set_xla_metadata(_xla_compute_type="dense")


def _user_pkg_prefix() -> str:
  if __package__ and ".torchtpu_tpuc_kernels" in __package__:
    return __package__.rsplit(".torchtpu_tpuc_kernels", 1)[0]
  path = os.path.abspath(__file__).replace("\\", "/")
  if "/experimental/users/" in path:
    user = path.split("/experimental/users/", 1)[1].split("/", 1)[0]
    return f"google3.experimental.users.{user}.torchtpu_tpuc_cdk"
  user = os.environ.get("USER") or os.environ.get("LOGNAME") or "hning"
  return f"google3.experimental.users.{user}.torchtpu_tpuc_cdk"


def _ensure_bridge_loaded() -> None:
  """Ensures `symbol_loader` and `deepsea_torch_bridge` are loaded."""
  global _BRIDGE_LOADED
  if _BRIDGE_LOADED and hasattr(torch.ops, "deepsea_moe"):
    if hasattr(torch.ops.deepsea_moe, "expert_ffn_out"):
      return

  try:
    importlib.import_module("symbol_loader")
  except ImportError:
    pass

  prefix = _user_pkg_prefix()
  for mod_name in (
      f"{prefix}.torchtpu_tpuc_interop.deepsea_torch_bridge",
      "deepsea_torch_bridge",
      f"{prefix}.torchtpu_tpuc_interop.tpuc_plugin",
      "tpuc_plugin",
  ):
    try:
      mod = importlib.import_module(mod_name)
      if hasattr(mod, "load_tpuc_libraries"):
        mod.load_tpuc_libraries()
      break
    except ImportError:
      continue

  if not hasattr(torch.ops.deepsea_moe, "expert_ffn_out"):
    raise RuntimeError(
        "torch.ops.deepsea_moe.expert_ffn_out is not registered; ensure "
        "deepsea_torch_bridge / tpuc_plugin is available in PYTHONPATH."
    )
  _BRIDGE_LOADED = True


def _get_prepared_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
  """Prepares and caches broadcast weight scales (`s13_b`, `s2_b`) for TPU-C."""
  n_exp = int(w1.shape[0])
  if not 1 <= n_exp <= E_MAX:
    raise ValueError(f"w1.shape[0]={n_exp} must be in [1, {E_MAX}]")
  h, i = int(w1.shape[1]), int(w2.shape[1])
  if (h, i) not in SUPPORTED_DIMS:
    raise ValueError(
        f"Unsupported MoE dims hidden={h}, intermediate={i}; supported:"
        f" {SUPPORTED_DIMS}"
    )
  if tuple(w1.shape) != (n_exp, h, 2 * i) or tuple(w2.shape) != (n_exp, i, h):
    raise ValueError(
        f"Unsupported weight shapes w1={tuple(w1.shape)}, w2={tuple(w2.shape)};"
        f" expected {(n_exp, h, 2 * i)} and {(n_exp, i, h)}"
    )
  if w1.dtype != FP8 or w2.dtype != FP8:
    raise ValueError(
        f"Expected float8_e4m3fn weights, got w1={w1.dtype}, w2={w2.dtype}"
    )

  key = (
      w1.data_ptr(),
      w2.data_ptr(),
      w1_scale.data_ptr(),
      w2_scale.data_ptr(),
      w1.device,
      n_exp,
  )
  cached = _WEIGHT_CACHE.get(key)
  if cached is not None:
    return cached

  s13 = w1_scale.reshape(n_exp, 2 * i).to(torch.float32)
  s2 = w2_scale.reshape(n_exp, h).to(torch.float32)
  w13_c = w1.contiguous()
  w2_c = w2.contiguous()
  s13_b = s13[:, None, :].expand(n_exp, 8, 2 * i).contiguous()
  s2_b = s2[:, None, :].expand(n_exp, 8, h).contiguous()
  _tpu_sync([w13_c, w2_c, s13_b, s2_b], wait=True)

  prepared = {
      "w13": w13_c,
      "w2": w2_c,
      "s13_b": s13_b,
      "s2_b": s2_b,
  }
  _WEIGHT_CACHE[key] = prepared
  return prepared


def ep_rows_bound(
    tokens: int, top_k: int, n_exp: int, use_ep: bool = True
) -> int:
  """Computes the 32-row-aligned routed-row buffer bound `R` for one rank."""
  p = tokens * top_k
  if use_ep:
    # Under EP8, expected pairs per shard is p / 8. Allocating max(512, p / 4)
    # provides >19 stddevs above the Binomial(p, 1/8) mean while keeping R
    # compact across all batch sizes.
    max_pairs = min(p, max(512, p // 4))
  else:
    max_pairs = p
  return ((max_pairs + n_exp * (GROUP - 1) + GROUP - 1) // GROUP) * GROUP


def tiles_shape(rows: int, hidden: int) -> tuple[int, int, int, int]:
  return (rows // 16, hidden // 128, 16, 128)


def _get_out_buffer(
    device: torch.device, rows: int, hidden: int
) -> torch.Tensor:
  """Returns a cached, zero-initialized, materialized Stage 2 output buffer.

  Mirrors cl/998887519 (`A1V3EPLayer._get_stage2_buffers`): the TPU-C kernel
  writes `out` in place via raw PJRT, so it is allocated once per `(device,
  rows)` with `torch.zeros` (rows past the computed groups stay finite) and
  submitted to PJRT before first use instead of a fresh `torch.empty` per call.
  """
  key = (device, rows, hidden)
  out = _OUT_BUFFERS.get(key)
  if out is None:
    out = torch.zeros(
        tiles_shape(rows, hidden), dtype=torch.bfloat16, device=device
    )
    _tpu_sync([out], wait=True)
    _OUT_BUFFERS[key] = out
  return out


def _ep_prep_sc_jax(
    hidden,
    ids,
    wts,
    experts_start,
    *,
    rows: int,
    n_exp: int,
    use_ep: bool,
    skip_padded_tokens: bool,
):
  """Stage 1 ("sc"): routing on TensorCore, row dispatch via SparseCore gather.

  Pairs are ordered by a stable `argsort` on the local expert id (so pairs of
  each expert keep their original order, as with the counting sort), and each
  routed row looks up its pair with small-table TensorCore gathers. The heavy
  `[R, H]` FP8 row movement is left to `_ep_dispatch_sc_jax` (SparseCore).
  No scatters.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  with _dense_compute_ctx():
    t, k = ids.shape
    p = t * k

    amax = jnp.max(jnp.abs(hidden), axis=1).astype(jnp.float32)
    tok_scale = jnp.where(amax > 0.0, amax / FP8_MAX, 1.0)
    inv_tok_scale = 1.0 / tok_scale
    q_tok = jnp.clip(
        hidden.astype(jnp.float32) * inv_tok_scale[:, None], -FP8_MAX, FP8_MAX
    ).astype(jnp.float8_e4m3fn)
    # Gather table `[T, H / 4 + 128]` (32-bit words, viewed as f32 so the
    # SparseCore gather is a plain copy): word j of a row packs the FP8 bytes
    # of columns j, j + H/4, j + H/2, j + 3H/4 (contiguous column blocks; a
    # `reshape(T, H/4, 4)` bitcast costs a ~5 ms minor-dim relayout here), and
    # the trailing 128 words hold the token's f32 scale.
    q8 = jax.lax.bitcast_convert_type(q_tok, jnp.uint8).astype(jnp.uint32)
    hq = hidden.shape[1] // 4
    packed = (
        q8[:, :hq]
        | (q8[:, hq : 2 * hq] << 8)
        | (q8[:, 2 * hq : 3 * hq] << 16)
        | (q8[:, 3 * hq :] << 24)
    )
    scale_words = jnp.broadcast_to(
        jax.lax.bitcast_convert_type(tok_scale, jnp.uint32)[:, None], (t, 128)
    )
    table = jax.lax.bitcast_convert_type(
        jnp.concatenate([packed, scale_words], axis=1), jnp.float32
    )

    flat_ids = ids.reshape(-1).astype(jnp.int32)
    flat_wts = wts.reshape(-1).astype(jnp.float32)
    if use_ep:
      e_local = flat_ids - experts_start.reshape(()).astype(jnp.int32)
    else:
      e_local = flat_ids
    valid = (e_local >= 0) & (e_local < n_exp)
    if skip_padded_tokens:
      valid = valid & (flat_wts != 0.0)
    sort_keys = jnp.where(valid, e_local, n_exp)

    experts = jnp.arange(n_exp, dtype=jnp.int32)
    counts = jnp.sum(
        (sort_keys[:, None] == experts[None, :]).astype(jnp.int32), axis=0
    )
    groups = (counts + GROUP - 1) // GROUP
    goff = jnp.cumsum(groups) - groups
    # Never name groups past the `rows` buffer (the kernel writes every named
    # group): clamp, so an overflowing shard drops pairs instead of writing
    # out of bounds. Not hit with the worst-case bound used for serving.
    groups = jnp.clip(jnp.minimum(groups, rows // GROUP - goff), 0)
    counts = jnp.minimum(counts, groups * GROUP)
    cstart = jnp.cumsum(counts) - counts
    order = jnp.argsort(sort_keys, stable=True).astype(jnp.int32)

    # Pair -> routed row (for combine): rank within expert from the sort.
    pair_goff = jnp.sum(
        jnp.where(sort_keys[:, None] == experts[None, :], goff[None, :], 0),
        axis=1,
    )
    pair_cstart = jnp.sum(
        jnp.where(sort_keys[:, None] == experts[None, :], cstart[None, :], 0),
        axis=1,
    )
    inv_order = jnp.argsort(order).astype(jnp.int32)
    dest_raw = pair_goff * GROUP + (inv_order - pair_cstart)
    dest_safe = jnp.clip(dest_raw, 0, rows - 1)
    dest = jnp.where(valid, dest_safe, -1)

    # Routed row -> pair.
    r = jnp.arange(rows, dtype=jnp.int32)
    row_start = goff * GROUP  # [n_exp]
    row_end = row_start + groups * GROUP
    in_e = (r[:, None] >= row_start[None, :]) & (r[:, None] < row_end[None, :])
    row_e_start = jnp.sum(jnp.where(in_e, row_start[None, :], 0), axis=1)
    row_cnt = jnp.sum(jnp.where(in_e, counts[None, :], 0), axis=1)
    row_cst = jnp.sum(jnp.where(in_e, cstart[None, :], 0), axis=1)
    j = r - row_e_start
    valid_row = jnp.any(in_e, axis=1) & (j < row_cnt)
    pair_of_row = order[jnp.clip(row_cst + j, 0, p - 1)]
    row_tok_idx = jnp.where(valid_row, pair_of_row // k, 0)
    row_tok = jnp.where(valid_row, row_tok_idx, t)
    rw = jnp.where(valid_row, flat_wts[pair_of_row], 0.0)
    # Rows past the last expert group are never read by the kernel.
    used_rows = (jnp.sum(groups) * GROUP).astype(jnp.int32).reshape(1)

    head = jnp.array(
        [n_exp, 1, 0, rows // GROUP, 0] + [0] * 11, dtype=jnp.int32
    )
    if n_exp < E_MAX:
      pad_e = jnp.zeros((E_MAX - n_exp,), dtype=jnp.int32)
      g_pad = jnp.concatenate([groups, pad_e])
      off_pad = jnp.concatenate([goff, pad_e])
    else:
      g_pad = groups[:E_MAX]
      off_pad = goff[:E_MAX]
    tail = jnp.zeros((META_WORDS - 16 - 2 * E_MAX,), dtype=jnp.int32)
    meta = jnp.concatenate([head, g_pad, off_pad, tail]).astype(jnp.int32)
    rw_b = jnp.broadcast_to(rw[:, None], (rows, 128))

  return table, row_tok_idx, used_rows, rw_b, meta, row_tok, dest


def _ep_dispatch_sc_jax(table, row_tok_idx, used_rows, row_tok, *, rows: int):
  """Stage 1b ("sc"): SparseCore gather of packed FP8 token rows + scales."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  ragged_gather, _ = _sc_kernels()
  g = ragged_gather(table, row_tok_idx, jnp.zeros((1,), jnp.int32), used_rows)
  with _dense_compute_ctx():
    g = jax.lax.bitcast_convert_type(g, jnp.uint32)
    hq = table.shape[1] - 128  # table is [T, H / 4 + 128]
    pk = g[:, :hq]
    xq = jax.lax.bitcast_convert_type(
        jnp.concatenate(
            [pk & 0xFF, (pk >> 8) & 0xFF, (pk >> 16) & 0xFF, pk >> 24], axis=1
        ).astype(jnp.uint8),
        jnp.float8_e4m3fn,
    )
    # `row_tok == T` marks padding rows (and rows past `used_rows`, whose
    # gathered contents are unspecified).
    xs = jnp.where(
        row_tok < table.shape[0],
        jax.lax.bitcast_convert_type(g[:, hq], jnp.float32),
        0.0,
    )
    xs_b = jnp.broadcast_to(xs[:, None], (rows, 128))
  return xq, xs_b


def _ep_combine_sc_jax(out_tiles, dest, *, tokens: int, top_k: int):
  """Stage 3 ("sc"): SparseCore gather-reduce of routed rows over top_k."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  _, ragged_gather_reduce = _sc_kernels()
  with _dense_compute_ctx():
    valid = dest >= 0
    safe_dest = jnp.where(valid, dest, 0)
    out_2d = out_tiles.swapaxes(1, 2).reshape(-1, out_tiles.shape[1] * 128)
    ones = jnp.ones(dest.shape, jnp.float32)
  # Routing weights are already applied by the kernel (`rw_b`).
  return ragged_gather_reduce(
      out_2d, safe_dest, ones, valid, reduce_group_size=top_k
  ).astype(jnp.bfloat16)


def _get_jax_glue(kind: str, **statics):
  """Compiles and caches the Stage 1/1b/3 (`prep`/`dispatch`/`combine`) ops."""
  key = (kind,) + tuple(sorted(statics.items()))
  op = _JAX_OPS.get(key)
  if op is None:
    import jax  # pylint: disable=g-import-not-at-top
    from torch_tpu._internal import pallas  # pylint: disable=g-import-not-at-top

    tag = "_".join(
        f"{k}{int(v) if isinstance(v, bool) else v}"
        for k, v in sorted(statics.items())
    )
    if kind == "prep":

      def prep(
          hidden: jax.Array,
          ids: jax.Array,
          wts: jax.Array,
          experts_start: jax.Array,
          *,
          rows: int,
          n_exp: int,
          use_ep: bool,
          skip_padded_tokens: bool,
      ) -> tuple[jax.Array, ...]:
        return _ep_prep_sc_jax(
            hidden,
            ids,
            wts,
            experts_start,
            rows=rows,
            n_exp=n_exp,
            use_ep=use_ep,
            skip_padded_tokens=skip_padded_tokens,
        )

      fn = functools.partial(prep, **statics)
    elif kind == "dispatch":

      def dispatch(
          table: jax.Array,
          row_tok_idx: jax.Array,
          used_rows: jax.Array,
          row_tok: jax.Array,
          *,
          rows: int,
      ) -> tuple[jax.Array, jax.Array]:
        return _ep_dispatch_sc_jax(
            table, row_tok_idx, used_rows, row_tok, rows=rows
        )

      fn = functools.partial(dispatch, **statics)
    else:

      def combine(
          out_tiles: jax.Array,
          dest: jax.Array,
          *,
          tokens: int,
          top_k: int,
      ) -> jax.Array:
        return _ep_combine_sc_jax(out_tiles, dest, tokens=tokens, top_k=top_k)

      fn = functools.partial(combine, **statics)

    op = _JAX_OPS[key] = pallas.jax_op(f"deepsea_moe_tpuc::{kind}_{tag}", fn)
  return op


def fused_moe_tpuc(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor | None = None,
    topk: int = 1,
    activation: str = "silu",
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
) -> torch.Tensor:
  """Drop-in replacement for `vllm_torchtpu` `fused_moe_gmm` using TPU-C MoE.

  Args:
    hidden_states: Input token activations `bf16[T, H]`.
    w1: Local expert gate+up FP8 weights `f8e4m3fn[E_local, H, 2I]`.
    w2: Local expert down FP8 weights `f8e4m3fn[E_local, I, H]`.
    w1_scale: Gate+up per-channel scales `f32[E_local, 1, 1, 2I]` (or 2D/3D).
    w2_scale: Down per-channel scales `f32[E_local, 1, 1, H]` (or 2D/3D).
    w1_bias: Must be `None` (unsupported by Qwen 3.5 MoE).
    w2_bias: Must be `None` (unsupported by Qwen 3.5 MoE).
    topk_weights: Router weights `[T, topk]` (`bf16` or `f32`).
    topk_ids: Global router expert IDs `int32[T, topk]`.
    experts_start: 0-d `int32` tensor on TPU giving `ep_rank * E_local`, or
      `None` when `EP=1`.
    topk: Number of experts routed per token.
    activation: Activation function (must be `"silu"`).
    rhs_quant_dtype: Unused (weights are already `float8_e4m3fn`).
    skip_padded_tokens: If True, skips routes where `topk_weights == 0`.

  Returns:
    Partial MoE output tensor `bf16[T, H]` containing the weighted sum of
    local experts for each token (ready for `ep_group.reduce_scatter`).
  """
  del rhs_quant_dtype
  if w1_bias is not None or w2_bias is not None:
    raise NotImplementedError("fused_moe_tpuc does not support w1_bias/w2_bias")
  if w1_scale is None or w2_scale is None:
    raise ValueError("fused_moe_tpuc requires w1_scale and w2_scale")
  if activation != "silu":
    raise NotImplementedError(
        f"fused_moe_tpuc only supports activation='silu', got {activation!r}"
    )

  _ensure_bridge_loaded()
  w = _get_prepared_weights(w1, w2, w1_scale, w2_scale)
  top_k = int(topk_ids.shape[1]) if topk_ids.dim() == 2 else int(topk)
  return _run_tpuc(
      hidden_states,
      w,
      topk_weights,
      topk_ids,
      experts_start,
      top_k=top_k,
      skip_padded_tokens=bool(skip_padded_tokens),
      ep_rows=True,
  )


def fused_moe_tpuc_prepared(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    s13_b: torch.Tensor,
    s2_b: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor | None,
    top_k: int,
    skip_padded_tokens: bool = False,
) -> torch.Tensor:
  """`fused_moe_tpuc` with pre-broadcast scales, for serving (vLLM TPUC_MOE).

  `s13_b: f32[E, 8, 2I]` and `s2_b: f32[E, 8, H]` are the per-channel FP8
  scales broadcast over 8 sublanes (see `_get_prepared_weights`). The routed-
  row buffer is sized for the worst case (every pair local to this shard), so
  skewed routing can never overflow it.
  """
  _ensure_bridge_loaded()
  w = {"w13": w13, "w2": w2, "s13_b": s13_b, "s2_b": s2_b}
  return _run_tpuc(
      hidden_states,
      w,
      topk_weights,
      topk_ids,
      experts_start,
      top_k=int(top_k),
      skip_padded_tokens=bool(skip_padded_tokens),
      ep_rows=False,
  )


def _run_tpuc(
    hidden_states: torch.Tensor,
    w: dict[str, torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor | None,
    *,
    top_k: int,
    skip_padded_tokens: bool,
    ep_rows: bool,
) -> torch.Tensor:
  """Stages 1-3 on prepared weights; `ep_rows` uses the EP8 row bound."""
  tokens = int(hidden_states.shape[0])
  n_exp = int(w["w13"].shape[0])
  hidden = int(w["w13"].shape[1])
  if int(hidden_states.shape[1]) != hidden:
    raise ValueError(
        f"hidden_states {tuple(hidden_states.shape)} does not match w13"
        f" {tuple(w['w13'].shape)}"
    )
  use_ep = experts_start is not None
  skip_pad = skip_padded_tokens
  rows = ep_rows_bound(tokens, top_k, n_exp, use_ep=use_ep and ep_rows)

  if experts_start is None:
    dev = hidden_states.device
    experts_start_t = _ZERO_START_CACHE.get(dev)
    if experts_start_t is None:
      experts_start_t = torch.zeros((), dtype=torch.int32, device=dev)
      _ZERO_START_CACHE[dev] = experts_start_t
  else:
    experts_start_t = experts_start

  # Split graph on any deferred collective inputs (e.g. EP8 all_gather) so
  # Stage 1 (`prep_op`) compiles and executes as a single-device program.
  timed = os.environ.get("TPU_MOE_STAGE_TIMES") == "1"
  t0 = time.perf_counter() if timed else 0.0
  _tpu_sync([hidden_states, topk_ids, topk_weights, experts_start_t], wait=True)
  t1 = time.perf_counter() if timed else 0.0

  prep_op = _get_jax_glue(
      "prep",
      rows=rows,
      n_exp=n_exp,
      use_ep=use_ep,
      skip_padded_tokens=skip_pad,
  )
  prep_out = prep_op(
      hidden_states,
      topk_ids,
      topk_weights,
      experts_start_t,
  )
  table, row_tok_idx, used_rows, rw_b, meta, row_tok, dest = prep_out
  t_prep = 0.0
  if timed:
    _tpu_sync(list(prep_out), wait=True)
    t_prep = time.perf_counter()
  xq, xs_b = _get_jax_glue("dispatch", rows=rows)(
      table, row_tok_idx, used_rows, row_tok
  )

  out = _get_out_buffer(hidden_states.device, rows, hidden)
  # Materialize all Stage 1 outputs and `out` together and await completion
  # before raw PJRT C-API execution of `expert_ffn_out`.
  _tpu_sync([xq, xs_b, rw_b, meta, row_tok, dest, out], wait=True)
  t2 = time.perf_counter() if timed else 0.0

  torch.ops.deepsea_moe.expert_ffn_out(
      xq, w["w13"], w["w2"], xs_b, rw_b, w["s13_b"], w["s2_b"], meta, out
  )
  t3 = time.perf_counter() if timed else 0.0

  combine_op = _get_jax_glue("combine", tokens=tokens, top_k=top_k)
  y = combine_op(out, dest)
  if y.dtype != hidden_states.dtype:
    y = y.to(hidden_states.dtype)
  # Materialize Stage 3 (`combine_op`) while `out` and `dest` are still live
  # in scope and before any downstream collective (e.g. reduce_scatter).
  _tpu_sync([y], wait=True)
  if timed:
    t4 = time.perf_counter()
    STAGE_TIMES.append({
        "input_sync": (t1 - t0) * 1e3,
        "prep": (t_prep - t1) * 1e3,
        "dispatch": (t2 - t_prep) * 1e3,
        "kernel": (t3 - t2) * 1e3,
        "combine": (t4 - t3) * 1e3,
    })
  return y
