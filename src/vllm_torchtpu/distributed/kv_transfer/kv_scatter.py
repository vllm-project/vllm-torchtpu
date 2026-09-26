"""Multi-layer HBM->HBM block scatter for the KV consumer path.

The KV consumer moves pulled blocks from shared-memory into its KV cache
in two steps: an H2D (``.to(device)`` per layer) and a scatter
(``cache[local_blocks] = src``). This module owns the scatter step only.

Interface (implementation-neutral):

  prepare_scatter_args(local_blocks, device) -> prebuilt args tuple
  multi_layer_scatter_into(srcs, dsts, local_blocks, *, prebuilt_args=None)
      -> list of output caches aliasing ``dsts``
  scatter_available() -> whether the kernel is importable
  smoke_test_multi_layer_scatter(...) -> bool

Current implementation is a single fused Pallas kernel that runs N
pallas_calls at trace time so the torch_tpu bridge is crossed just once
per scatter. If torch_tpu later exposes a lower-level scatter primitive,
this module is the only one that needs to change -- callers (see
tpu_connector._coord_scatter_shard) depend on the interface above, not
on JAX/Pallas.

Each layer's destination tensor is donated and aliased 1:1 to its output,
so the scatter is in-place on HBM with no extra allocation. Compilation
is O(num_layers) pallas_calls, so the first trace for a given num_layers
can take seconds for N=64; warmup amortizes it.
"""

from __future__ import annotations

import functools

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Lazy JAX/Pallas import so this module is safe to import on hosts without
# a working JAX install.
_jax_import_error: BaseException | None = None
try:
    import jax  # noqa: F401
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal import pallas as ttpu_pallas
except Exception as e:  # pragma: no cover
    _jax_import_error = e
    pl = None  # type: ignore[assignment]
    pltpu = None  # type: ignore[assignment]
    ttpu_pallas = None  # type: ignore[assignment]


def scatter_available() -> bool:
    """Whether the scatter kernel can be used. Callers must guard with
    this before calling into the module."""
    return _jax_import_error is None


if _jax_import_error is None:

    def _scatter_kernel_body(
        num_chunks_ref,
        src_offsets_ref,
        dest_offsets_ref,
        src_ref,  # HBM: (num_src_blocks, ...)
        dst_ref_in,  # HBM: (num_dst_blocks, ...) aliased to out
        dst_ref_out,  # HBM
    ):
        del dst_ref_in

        def body(sem):
            @pl.loop(0, num_chunks_ref[0])
            def _start(i):
                pltpu.make_async_copy(
                    src_ref.at[pl.ds(src_offsets_ref[i], 1)],
                    dst_ref_out.at[pl.ds(dest_offsets_ref[i], 1)],
                    sem,
                ).start()

            @pl.loop(0, num_chunks_ref[0])
            def _wait(i):
                pltpu.make_async_copy(
                    src_ref.at[pl.ds(src_offsets_ref[i], 1)],
                    dst_ref_out.at[pl.ds(dest_offsets_ref[i], 1)],
                    sem,
                ).wait()

        pl.run_scoped(body, pltpu.SemaphoreType.DMA)

    @functools.lru_cache(maxsize=8)
    def _make_multi_layer_scatter_kernel(num_layers: int):
        """Build (and cache) a torch_tpu-wrapped kernel that scatters
        ``num_layers`` layers in one call. The bridge further caches by
        input shapes, so varying per-call tensor shapes recompile
        transparently; varying ``num_layers`` rebuilds the Python
        wrapper (which is fine, vLLM keeps it constant per worker)."""

        def _jax_multi_layer_scatter(
            num_chunks, src_offsets, dest_offsets, *srcs_and_dsts
        ):
            if len(srcs_and_dsts) != 2 * num_layers:
                raise ValueError(
                    f"multi_layer_scatter: expected {2 * num_layers} "
                    f"src/dst tensors, got {len(srcs_and_dsts)}"
                )
            srcs = srcs_and_dsts[:num_layers]
            dsts = srcs_and_dsts[num_layers:]
            new_dsts = []
            for i in range(num_layers):
                new_dst = pl.pallas_call(
                    _scatter_kernel_body,
                    out_shape=jax.ShapeDtypeStruct(
                        shape=dsts[i].shape, dtype=dsts[i].dtype
                    ),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        grid=(1,),
                        num_scalar_prefetch=3,
                        in_specs=[
                            pl.BlockSpec(memory_space=pl.ANY),
                            pl.BlockSpec(memory_space=pl.ANY),
                        ],
                        out_specs=pl.BlockSpec(memory_space=pl.ANY),
                    ),
                    input_output_aliases={4: 0},
                )(num_chunks, src_offsets, dest_offsets, srcs[i], dsts[i])
                new_dsts.append(new_dst)
            return tuple(new_dsts)

        # Dst positions in the outer call are 3+N .. 3+2N-1; each is
        # donated and aliases output index i.
        io_aliases = {3 + num_layers + i: i for i in range(num_layers)}
        donate_argnums = tuple(3 + num_layers + i for i in range(num_layers))
        return ttpu_pallas.custom_jax_kernel(
            _jax_multi_layer_scatter,
            name=f"kv_d2d_multi_scatter_n{num_layers}",
            input_output_aliases=io_aliases,
            donate_argnums=donate_argnums,
        )

else:  # pragma: no cover
    _make_multi_layer_scatter_kernel = None  # type: ignore[assignment]


def prepare_scatter_args(
    local_blocks: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the three scalar-prefetch tensors once per scatter call
    instead of once per layer. Pass the returned tuple as
    ``prebuilt_args`` to ``multi_layer_scatter_into``."""
    num_blocks = int(local_blocks.shape[0])
    num_chunks = torch.tensor([num_blocks], dtype=torch.int32, device=device)
    src_offsets = torch.arange(num_blocks, dtype=torch.int32, device=device)
    dest_offsets = local_blocks.to(device=device, dtype=torch.int32)
    return num_chunks, src_offsets, dest_offsets


def multi_layer_scatter_into(
    srcs: list[torch.Tensor],
    dsts: list[torch.Tensor],
    local_blocks: torch.Tensor,
    *,
    prebuilt_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Scatter ``srcs[l][i]`` into ``dsts[l][local_blocks[i]]`` for all
    layers ``l`` and blocks ``i``, in one kernel dispatch.

    All tensors must be on the same device. Each ``dsts[l]`` is donated
    and the corresponding returned tensor aliases the same buffer.
    Returns a list of N output tensors in the same order as ``dsts``.
    """
    if _jax_import_error is not None:
        raise RuntimeError(f"KV scatter unavailable: {_jax_import_error!r}")
    if len(srcs) != len(dsts):
        raise ValueError(
            f"multi_layer_scatter_into: len(srcs)={len(srcs)} != len(dsts)={len(dsts)}"
        )
    num_layers = len(srcs)
    if num_layers == 0:
        return []
    num_blocks = int(local_blocks.shape[0])
    if num_blocks == 0:
        return list(dsts)
    if prebuilt_args is not None:
        num_chunks, src_offsets, dest_offsets = prebuilt_args
    else:
        num_chunks, src_offsets, dest_offsets = prepare_scatter_args(
            local_blocks, dsts[0].device
        )
    kernel = _make_multi_layer_scatter_kernel(num_layers)
    result = kernel(num_chunks, src_offsets, dest_offsets, *srcs, *dsts)
    # JAX returns a tuple matching the function's return shape.
    if isinstance(result, (list, tuple)):
        return list(result)
    return [result]


def smoke_test_multi_layer_scatter(
    device: torch.device,
    num_layers: int = 2,
    num_blocks: int = 4,
    trailing_shape: tuple[int, ...] = (16, 2, 128),
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """Round-trip a small multi-layer scatter and check contents per
    layer. ``num_layers`` defaults to 2 so the smoke test is cheap;
    compile for the real worker's num_layers happens on first invocation
    (typically during warmup)."""
    if _jax_import_error is not None:
        logger.warning("KV scatter smoke test skipped: %s", _jax_import_error)
        return False
    try:
        src_shape = (num_blocks,) + trailing_shape
        dst_shape = (num_blocks * 2,) + trailing_shape
        srcs_cpu = [torch.randn(src_shape).to(dtype) for _ in range(num_layers)]
        srcs = [s.to(device) for s in srcs_cpu]
        dsts = [
            torch.zeros(dst_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        local_blocks = torch.tensor(
            list(range(num_blocks * 2 - 1, num_blocks - 1, -1)),
            dtype=torch.int32,
            device=device,
        )
        outs = multi_layer_scatter_into(srcs, dsts, local_blocks)
        for layer in range(num_layers):
            out_cpu = outs[layer].cpu()
            expected = torch.zeros(dst_shape, dtype=dtype)
            for i, idx in enumerate(local_blocks.tolist()):
                expected[idx] = srcs_cpu[layer][i]
            if not torch.equal(out_cpu, expected):
                logger.warning("KV scatter smoke test mismatch at layer %d", layer)
                return False
        return True
    except Exception as e:
        logger.warning("KV scatter smoke test raised: %r", e)
        return False
