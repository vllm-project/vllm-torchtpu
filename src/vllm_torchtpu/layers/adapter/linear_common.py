from collections.abc import Callable
from threading import Lock

import jax
import jax.numpy as jnp
import torch
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

# Some layers have their `.weight` read raw -- bypassing the linear method's
# `apply` -- by code that assumes vLLM's `[n_out, n_in]` layout. Flipping those
# to (k, n) corrupts the consumer silently, so they keep the upstream layout.
#
# The requirement belongs to the *consumer*, not to a name list here: the
# module that reads the weight raw sets this attribute on the linear layer in
# its own `__init__`. That runs during model construction, before any
# `process_weights_after_loading`, so it is set by the time the layout decision
# is made -- and it moves with the code if a layer is renamed, which a
# name-matched list would not.
KEEP_VLLM_LAYOUT_ATTR = "_tpu_keep_vllm_weight_layout"

# Idempotence guard, set by the linear method once it has flipped a weight.
# Not how `apply` picks its contraction -- that keys off the marker above.
#
# It has to be a stored flag rather than a property of the weight: the flip
# cannot be recovered from the shape (`o_proj` and friends are square in most
# models), and encoding it in a Parameter subclass does not survive vLLM's
# `replace_parameter`, which re-wraps by type. Meanwhile
# `process_weights_after_loading` IS re-entered by vLLM's reload paths
# (`dummy_loader`, `reload/layerwise`), where a second flip would transpose
# back and silently corrupt the values.
WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"


def _get_x_q_dtype(w_q_dtype: jnp.dtype) -> jnp.dtype:
    """Return the activation quant dtype paired with a given weight dtype.

    Integer weights pair with int8 activations; floating-point (FP8) weights
    pair with float8_e4m3fn activations.
    """
    if jnp.issubdtype(w_q_dtype, jnp.integer):
        return jnp.int8
    if jnp.issubdtype(w_q_dtype, jnp.floating):
        return jnp.float8_e4m3fn
    raise ValueError(f"Unsupported quantized dtype: {w_q_dtype}")


def _quantized_matmul_jax(x: jax.Array, w_q: jax.Array, w_s: jax.Array) -> jax.Array:
    """FP8 dense matmul on the canonical (k, n) weight layout.

    `w_q` is `[n_in, n_out]`, so this is a plain `(m, k) @ (k, n)` contraction
    rather than the `(m, k) @ (n, k).T` the N-major layout forced. The
    blockwise path routes through `gmm_v2` (single group), which consumes a
    K-major rhs natively; the per-channel path uses the shared
    `xla_quantized_matmul`, which takes the same (k, n) layout.
    """
    if x.shape[1] != w_q.shape[0]:
        raise ValueError(
            f"Input hidden dim {x.shape[1]} must match weight hidden dim "
            f"{w_q.shape[0]}."
        )
    if len(w_s.shape) == 4:
        k_dim, n_out = w_q.shape
        _, sharded_num_blocks, mid, scale_n_out = w_s.shape
        if mid != 1:
            raise ValueError(
                f"Blockwise weight scale axis 2 must be 1, got {w_s.shape=}"
            )
        if scale_n_out != n_out:
            raise ValueError(
                f"Blockwise weight scale output dim {scale_n_out} must match "
                f"weight output dim {n_out}."
            )
        if sharded_num_blocks <= 0 or k_dim % sharded_num_blocks != 0:
            raise ValueError(
                f"Input hidden dim {k_dim} must be divisible by block scale "
                f"count {sharded_num_blocks}."
            )
        return gmm_v2(
            lhs=x,
            rhs=w_q[None],  # [n_in, n_out] -> [1, n_in, n_out]
            group_sizes=jnp.array([x.shape[0]], dtype=jnp.int32),
            rhs_scale=w_s,
            group_offset=jnp.array([0], dtype=jnp.int32),
            preferred_element_type=x.dtype,
            maybe_quantize_lhs=True,
            zero_initialize=True,
        )
    return xla_quantized_matmul(x, w_q, w_s)


_quantized_matmul_kernel_op: Callable | None = None
_quantized_matmul_kernel_op_lock = Lock()


def _get_quantized_matmul_op() -> Callable:
    global _quantized_matmul_kernel_op
    if _quantized_matmul_kernel_op is not None:
        return _quantized_matmul_kernel_op

    with _quantized_matmul_kernel_op_lock:
        if _quantized_matmul_kernel_op is not None:
            return _quantized_matmul_kernel_op

        op = pallas.jax_op("pallas::quantized_matmul_kernel", _quantized_matmul_jax)

        def _fake_quantized_matmul(
            x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor
        ):
            del w_s
            return torch.empty(
                x.shape[0], w_q.shape[-1], dtype=x.dtype, device=x.device
            )

        op.register_fake(_fake_quantized_matmul)
        _quantized_matmul_kernel_op = op
        return op


def quantized_matmul(
    x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor
) -> torch.Tensor:
    """Torch entry point for the runtime FP8 dense-linear matmul.

    Reshapes `x` to 2-D, dispatches through a cached `pallas.jax_op`, and
    reshapes the output back. `w_q` is the canonical (k, n) layout
    `[n_in, n_out]`. The underlying JAX function picks `gmm_v2` when `w_s` is
    rank-4 `[1, n_in_blocks, 1, n_out]`, and the per-channel dot_general path
    when `w_s` is 1-D `[n_out]`.
    """
    if x.shape[-1] != w_q.shape[0]:
        raise ValueError(
            f"Input hidden dim {x.shape[-1]} must match weight hidden dim "
            f"{w_q.shape[0]}."
        )

    orig_out_shape = (*x.shape[:-1], w_q.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = _get_quantized_matmul_op()(x_2d, w_q, w_s)
    return out_2d.reshape(orig_out_shape)


def _quantized_matmul_prequantized_jax(
    x_q: jax.Array,
    x_scale: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    *,
    output_dtype: jnp.dtype,
) -> jax.Array:
    """Contract supplied 2-D FP8 operands without quantizing either operand."""
    out = jax.lax.dot_general(
        x_q,
        w_q,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return (out * x_scale * w_scale[None, :]).astype(output_dtype)


_prequantized_output_dtypes = {
    torch.bfloat16: jnp.bfloat16,
    torch.float16: jnp.float16,
    torch.float32: jnp.float32,
}
_quantized_matmul_prequantized_ops: dict[torch.dtype, Callable] = {}
_quantized_matmul_prequantized_op_lock = Lock()


def _get_quantized_matmul_prequantized_op(output_dtype: torch.dtype) -> Callable:
    op = _quantized_matmul_prequantized_ops.get(output_dtype)
    if op is not None:
        return op

    with _quantized_matmul_prequantized_op_lock:
        op = _quantized_matmul_prequantized_ops.get(output_dtype)
        if op is not None:
            return op

        # Capture dtype as static metadata, not as a tensor operand. One
        # registration per dtype also gives fake tracing the exact output dtype.
        jax_dtype = _prequantized_output_dtypes[output_dtype]

        def kernel(
            x_q: jax.Array,
            x_scale: jax.Array,
            w_q: jax.Array,
            w_scale: jax.Array,
        ) -> jax.Array:
            return _quantized_matmul_prequantized_jax(
                x_q, x_scale, w_q, w_scale, output_dtype=jax_dtype
            )

        op = pallas.jax_op(
            f"pallas::quantized_matmul_prequantized_{jax_dtype.__name__}", kernel
        )

        def fake(x_q, x_scale, w_q, w_scale):
            del x_scale, w_scale
            return torch.empty(
                x_q.shape[0], w_q.shape[1], dtype=output_dtype, device=x_q.device
            )

        op.register_fake(fake)
        _quantized_matmul_prequantized_ops[output_dtype] = op
        return op


def quantized_matmul_prequantized(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reuse FP8 activations and FP32 row scales for a dense linear.

    Inputs are e4m3fn ``x_q[..., K]`` and canonical ``w_q[K, N]``, with
    FP32 scales ``x_scale[..., 1]`` and ``w_scale[N]``. Accumulation and
    scaling use FP32 before casting to the original floating output dtype.
    Blockwise scales and dynamic activation quantization are not supported.
    """
    if output_dtype not in _prequantized_output_dtypes:
        raise ValueError("output_dtype must be bfloat16, float16, or float32.")
    if x_q.dtype != torch.float8_e4m3fn or w_q.dtype != torch.float8_e4m3fn:
        raise ValueError("Prequantized activations and weights must be float8_e4m3fn.")
    if x_scale.dtype != torch.float32 or w_scale.dtype != torch.float32:
        raise ValueError("Prequantized activation and weight scales must be float32.")
    if x_q.ndim < 1 or w_q.ndim != 2:
        raise ValueError("Expected activations [..., K] and weights [K, N].")
    if x_q.shape[-1] != w_q.shape[0]:
        raise ValueError(
            f"Input hidden dim {x_q.shape[-1]} must match weight hidden dim "
            f"{w_q.shape[0]}."
        )
    if x_scale.shape != (*x_q.shape[:-1], 1):
        raise ValueError("Activation scales must have shape [..., 1] matching x_q.")
    if w_scale.shape != (w_q.shape[1],):
        raise ValueError("Weight scales must have shape [N] matching w_q.")

    out = _get_quantized_matmul_prequantized_op(output_dtype)(
        x_q.reshape(-1, x_q.shape[-1]), x_scale.reshape(-1, 1), w_q, w_scale
    )
    return out.reshape(*x_q.shape[:-1], w_q.shape[1])


def _quantized_matmul_fp4_jax(
    x: jax.Array, w_q: jax.Array, w_s: jax.Array
) -> jax.Array:
    """W4A16 dense matmul for NVFP4 weights via gmm_v2 (single group).

    `w_q` is the native fp4 weight in K-major layout [n_in, n_out] (unpacked at
    load); it only needs a leading group axis for the gmm_v2 rhs layout
    [1, n_in, n_out]. `w_s` is the fused fp32 block scale [1, num_blocks, 1,
    n_out]. gmm_v2 selects the regime from the block size: block-16 dequantizes
    the fp4 weight per block and keeps bf16 activations (W4A16).
    """
    rhs = w_q[None]  # [n_in, n_out] fp4 -> [1, n_in, n_out]
    return gmm_v2(
        lhs=x,
        rhs=rhs,
        group_sizes=jnp.array([x.shape[0]], dtype=jnp.int32),
        rhs_scale=w_s,
        group_offset=jnp.array([0], dtype=jnp.int32),
        preferred_element_type=x.dtype,
        maybe_quantize_lhs=True,
        zero_initialize=True,
    )


_quantized_matmul_fp4_op: Callable | None = None
_quantized_matmul_fp4_op_lock = Lock()


def _get_quantized_matmul_fp4_op() -> Callable:
    global _quantized_matmul_fp4_op
    if _quantized_matmul_fp4_op is not None:
        return _quantized_matmul_fp4_op

    with _quantized_matmul_fp4_op_lock:
        if _quantized_matmul_fp4_op is not None:
            return _quantized_matmul_fp4_op

        op = pallas.jax_op(
            "pallas::quantized_matmul_fp4_kernel", _quantized_matmul_fp4_jax
        )

        def _fake_quantized_matmul_fp4(
            x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor
        ):
            del w_q  # output width is the logical n_out carried by the scale
            return torch.empty(
                x.shape[0], w_s.shape[-1], dtype=x.dtype, device=x.device
            )

        op.register_fake(_fake_quantized_matmul_fp4)
        _quantized_matmul_fp4_op = op
        return op


def quantized_matmul_fp4(
    x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor
) -> torch.Tensor:
    """Torch entry for the NVFP4 W4A16 dense-linear matmul.

    `w_q` is the native fp4 weight in K-major layout [n_in, n_out] (unpacked at
    load), stored as ``torch.float4_e2m1fn_x2`` (2 fp4 per byte, so its torch
    last dim is packed). The fused fp32 block scale `w_s` [1, num_blocks, 1,
    n_out] carries the logical output width.
    """
    orig_out_shape = (*x.shape[:-1], w_s.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = _get_quantized_matmul_fp4_op()(x_2d, w_q, w_s)
    return out_2d.reshape(orig_out_shape)


def sharded_quantized_matmul(
    x: jax.Array, w_q: jax.Array, w_s: jax.Array, mesh: Mesh, weight_sharding: P
):
    out_axis, in_axis = weight_sharding
    x_sharding = P(None, in_axis)
    scale_sharding = P(
        out_axis,
    )
    out_sharding = P(None, out_axis)

    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, x_sharding))

    def wrapper(x, w_q, w_s):
        output = _quantized_matmul_jax(x, w_q, w_s)
        if in_axis:
            output = jax.lax.psum(output, axis_name=in_axis)
        return output

    return shard_map(
        wrapper,
        mesh=mesh,
        in_specs=(x_sharding, weight_sharding, scale_sharding),
        out_specs=(out_sharding),
        check_rep=False,
    )(x, w_q, w_s)


def reorder_concatenated_tensor_for_sharding(
    concatenated_tensor: jax.Array, split_sizes: list[int], n_shards: int, dim: int
):
    """
    Reorder a replicated concatenated tensor such that when sharded on multiple chips,
    each shard is a concatenation of the shards of the individual tensors.
    For example, let the concatenated_tensor be:
        AAAAAAAAAAAABBBBBBBBCCCC
            12 As     8 Bs  4 Cs
    and let the split_sizes = [12, 8, 4] and n_shards = 4.
    The output is:
        AAABBCAAABBCAAABBCAAABBC
    In other words, it reorders the input tensor into 4 segements, with each segment
    corresponding to a shard and being AAABBC.
    Args:
        concatenated_tensor: the tensor, concatenated on the dimension specified
            by `dim`.
        split_sizes: each individual tensor's size on the dimension specified by `dim`.
        n_shards: num of shards.
        dim: the dimension on which the concatenated_tensor is concatenated.
    """
    # Split the concatenated tensor into individual tensors.
    split_tensors = []
    start_offset = 0
    old_shape = concatenated_tensor.shape
    # New shape ensures each split_tensor[i] maps to a tensor in ith shards
    new_shape = old_shape[:dim] + (n_shards, -1) + old_shape[dim + 1 :]
    for split_size in split_sizes:
        split_tensor = jax.lax.slice_in_dim(
            concatenated_tensor, start_offset, start_offset + split_size, axis=dim
        )
        split_tensors.append(split_tensor.reshape(new_shape))
        start_offset += split_size
    # While maintaining 0th dim as a shard dim, we concatenate along 1th dim to
    # to create concatenated tnensor where 0th dim maps to shard dim.
    reordered_tensor = jnp.concatenate(split_tensors, axis=dim + 1)
    return reordered_tensor.reshape(old_shape)


def slice_sharded_tensor_for_concatenation(
    sharded_tensor: jax.Array, split_sizes: list[int], n_shards: int
):
    """
    Slice the input tensor which is sharded on multiple chips (on the last dim) into
    individual tensors with the same sharding.
    For example, let the sharded_tensor be:
        AAABBC | AAABBC | AAABBC | AAABBC
        Shard0   Shard1   Shard2   Shard3
    and let the split_sizes = [12, 8, 4] and n_shards = 4.
    The output is a list of 3 tensors:
         AAA   |  AAA   |  AAA   |  AAA
          BB   |   BB   |   BB   |   BB
           C   |    C   |    C   |    C
        Shard0   Shard1   Shard2   Shard3
    In other words, each individual tensor is a slice of the input tensor with the
    same sharding.
    Args:
        sharded_tensor: the input tensor, sharded on the last dim.
        split_sizes: each individual tensor's size on the last dim.
        n_shards: num of shards.
    """
    new_shape = sharded_tensor.shape[:-1] + (n_shards, -1)
    # New shape ensures each sharded_tensor[:, i] maps to a tensor in ith shards
    sharded_tensor = sharded_tensor.reshape(new_shape)

    split_tensors = []
    start_offset = 0
    for split_size in split_sizes:
        assert split_size % n_shards == 0
        sz = split_size // n_shards  # size of this split tensor per shard
        end_offset = start_offset + sz
        # Because we are slicing over last dim, sharding dim remains intact.
        # Therefore, splitting happens locally.
        split_tensor = sharded_tensor[..., start_offset:end_offset]
        split_tensors.append(split_tensor.reshape(new_shape[:-2] + (-1,)))
        start_offset = end_offset

    return split_tensors


MODEL_MATMUL_FUSION_TRUTH_TABLE = {
    ("Qwen/Qwen2.5-7B-Instruct", 1024, 1, "QKVParallelLinear"): True,
    ("Qwen/Qwen2.5-7B-Instruct", 1024, 1, "MergedColumnParallelLinear"): False,
    ("Qwen/Qwen2.5-7B-Instruct", 2048, 1, "QKVParallelLinear"): False,
    ("Qwen/Qwen2.5-7B-Instruct", 2048, 1, "MergedColumnParallelLinear"): False,
    ("meta-llama/Llama-3.1-8B-Instruct", 1024, 1, "QKVParallelLinear"): False,
    ("meta-llama/Llama-3.1-8B-Instruct", 1024, 1, "MergedColumnParallelLinear"): False,
    ("meta-llama/Llama-3.1-8B-Instruct", 2048, 1, "QKVParallelLinear"): False,
    ("meta-llama/Llama-3.1-8B-Instruct", 2048, 1, "MergedColumnParallelLinear"): False,
    (
        "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
        1024,
        1,
        "QKVParallelLinear",
    ): False,
    (
        "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
        1024,
        1,
        "MergedColumnParallelLinear",
    ): False,
    (
        "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
        2048,
        1,
        "QKVParallelLinear",
    ): False,
    (
        "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
        2048,
        1,
        "MergedColumnParallelLinear",
    ): False,
}


def get_model_matmul_fusion_assignment(
    model_name: str, batch_size: int, tp_size: int, layer_name: str
):
    key = (model_name, batch_size, tp_size, layer_name)
    return MODEL_MATMUL_FUSION_TRUTH_TABLE.get(key, True)
