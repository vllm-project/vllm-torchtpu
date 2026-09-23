# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for vllm_torchtpu.distributed.kv_transfer.kv_scatter.

Split into two groups:
  - CPU tests: prepare_scatter_args and zero/empty edge cases for
    multi_layer_scatter_into that return before calling the Pallas kernel.
    These run without JAX or a TPU.
  - TPU/JAX tests: the full Pallas scatter kernel and the smoke test.
    Gated on scatter_available(); skipped automatically when JAX is absent.
    Each TPU test runs in its own `spawn`d subprocess so the libtpu lockfile
    is owned by that child and released on its exit (see
    _run_in_subprocess); otherwise the lockfile lingers in the pytest parent
    and later subprocess-spawning tests (e.g. test_async_scheduling.py)
    abort with "Internal error when accessing libtpu multi-process lockfile".
"""

import multiprocessing
import traceback

import pytest
import torch

from vllm_torchtpu.distributed.kv_transfer.kv_scatter import (
    multi_layer_scatter_into,
    prepare_scatter_args,
    scatter_available,
    smoke_test_multi_layer_scatter,
)


def _subprocess_worker(target, queue):
    try:
        target()
        queue.put(("OK", None))
    except BaseException:
        queue.put(("ERROR", traceback.format_exc()))


def _run_in_subprocess(target):
    """Run `target()` in a fresh `spawn` subprocess and re-raise any failure.

    `target` must be a top-level function so it's picklable for spawn.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=_subprocess_worker, args=(target, queue))
    p.start()
    p.join()
    if p.exitcode != 0 and queue.empty():
        raise AssertionError(f"Isolated TPU test crashed (exitcode={p.exitcode})")
    status, payload = queue.get()
    if status == "ERROR":
        raise AssertionError(f"Isolated TPU test failed:\n{payload}")


# ---------------------------------------------------------------------------
# TestScatterAvailable
# ---------------------------------------------------------------------------


class TestScatterAvailable:
    def test_returns_bool(self):
        assert isinstance(scatter_available(), bool)


# ---------------------------------------------------------------------------
# TestPrepareScatterArgs — pure torch, no TPU required
# ---------------------------------------------------------------------------


class TestPrepareScatterArgs:
    """prepare_scatter_args builds three int32 tensors from a block-id vector.
    No Pallas or JAX involved; safe to run on CPU.
    """

    def test_output_shapes(self):
        blocks = torch.tensor([3, 7, 1], dtype=torch.int64)
        num_chunks, src_offsets, dest_offsets = prepare_scatter_args(
            blocks, torch.device("cpu")
        )

        assert num_chunks.shape == (1,)
        assert src_offsets.shape == (3,)
        assert dest_offsets.shape == (3,)

    def test_all_outputs_are_int32(self):
        blocks = torch.tensor([0, 1, 2], dtype=torch.int64)
        num_chunks, src_offsets, dest_offsets = prepare_scatter_args(
            blocks, torch.device("cpu")
        )

        assert num_chunks.dtype == torch.int32
        assert src_offsets.dtype == torch.int32
        assert dest_offsets.dtype == torch.int32

    def test_num_chunks_equals_block_count(self):
        blocks = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        num_chunks, _, _ = prepare_scatter_args(blocks, torch.device("cpu"))
        assert int(num_chunks[0]) == 5

    def test_src_offsets_are_sequential_from_zero(self):
        blocks = torch.tensor([10, 20, 30], dtype=torch.int64)
        _, src_offsets, _ = prepare_scatter_args(blocks, torch.device("cpu"))
        assert src_offsets.tolist() == [0, 1, 2]

    def test_dest_offsets_mirror_input_block_ids(self):
        blocks = torch.tensor([10, 20, 30], dtype=torch.int64)
        _, _, dest_offsets = prepare_scatter_args(blocks, torch.device("cpu"))
        assert dest_offsets.tolist() == [10, 20, 30]

    def test_single_block(self):
        blocks = torch.tensor([7], dtype=torch.int64)
        num_chunks, src_offsets, dest_offsets = prepare_scatter_args(
            blocks, torch.device("cpu")
        )
        assert int(num_chunks[0]) == 1
        assert src_offsets.tolist() == [0]
        assert dest_offsets.tolist() == [7]


# ---------------------------------------------------------------------------
# TestMultiLayerScatterIntoEdgeCases
# ---------------------------------------------------------------------------


class TestMultiLayerScatterIntoEdgeCases:
    """Early-exit paths that don't invoke the Pallas kernel.

    multi_layer_scatter_into checks for JAX availability before anything else,
    so these tests are gated on scatter_available().
    """

    @pytest.mark.skipif(not scatter_available(), reason="JAX/Pallas not available")
    def test_empty_layer_list_returns_empty(self):
        result = multi_layer_scatter_into(
            srcs=[], dsts=[], local_blocks=torch.tensor([], dtype=torch.int32)
        )
        assert result == []

    @pytest.mark.skipif(not scatter_available(), reason="JAX/Pallas not available")
    def test_zero_blocks_returns_dsts_unchanged(self):
        dst = torch.zeros(8, 4, dtype=torch.float32)
        result = multi_layer_scatter_into(
            srcs=[torch.zeros(0, 4)],
            dsts=[dst],
            local_blocks=torch.tensor([], dtype=torch.int32),
        )
        assert len(result) == 1
        assert result[0] is dst

    @pytest.mark.skipif(not scatter_available(), reason="JAX/Pallas not available")
    def test_mismatched_srcs_dsts_raises_value_error(self):
        with pytest.raises(ValueError, match="multi_layer_scatter_into"):
            multi_layer_scatter_into(
                srcs=[torch.zeros(2, 4)],
                dsts=[],
                local_blocks=torch.tensor([0, 1], dtype=torch.int32),
            )


# ---------------------------------------------------------------------------
# TestMultiLayerScatterIntoKernel — TPU/Pallas required
# ---------------------------------------------------------------------------

# Shape of one KV block: (block_size, num_kv_heads, head_dim).
# Must be tile-aligned: last dim must be ≥ 128 and divisible by 128.
_TRAILING = (16, 2, 128)


def _body_single_layer_scatter_correctness():
    device = torch.device("tpu")

    num_src_blocks = 3
    num_dst_blocks = 8
    dtype = torch.bfloat16
    t = _TRAILING

    src = torch.ones((num_src_blocks, *t), dtype=dtype, device=device)
    # Fill each block with a unique value so we can distinguish them.
    for i in range(num_src_blocks):
        src[i] = float(i + 1)
    dst = torch.zeros((num_dst_blocks, *t), dtype=dtype, device=device)
    dest_ids = torch.tensor([2, 5, 7], dtype=torch.int32, device=device)

    outs = multi_layer_scatter_into([src], [dst], dest_ids)

    out_cpu = outs[0].cpu()
    src_cpu = src.cpu()
    for i, dest_idx in enumerate([2, 5, 7]):
        assert torch.equal(out_cpu[dest_idx], src_cpu[i]), (
            f"block {i} → dest[{dest_idx}] mismatch"
        )
    for untouched in [0, 1, 3, 4, 6]:
        assert torch.equal(out_cpu[untouched], torch.zeros(t, dtype=dtype))


def _body_multi_layer_scatter_all_layers_updated():
    device = torch.device("tpu")

    num_layers = 4
    num_src_blocks = 2
    num_dst_blocks = 6
    dtype = torch.bfloat16
    t = _TRAILING

    srcs = [
        torch.full((num_src_blocks, *t), float(layer_idx), dtype=dtype, device=device)
        for layer_idx in range(num_layers)
    ]
    dsts = [
        torch.zeros((num_dst_blocks, *t), dtype=dtype, device=device)
        for _ in range(num_layers)
    ]
    dest_ids = torch.tensor([1, 4], dtype=torch.int32, device=device)

    outs = multi_layer_scatter_into(srcs, dsts, dest_ids)

    for layer_idx in range(num_layers):
        out_cpu = outs[layer_idx].cpu()
        for dest_idx in [1, 4]:
            assert torch.all(out_cpu[dest_idx] == float(layer_idx)), (
                f"layer {layer_idx}, dest[{dest_idx}] not filled with {float(layer_idx)}"
            )


@pytest.mark.skipif(not scatter_available(), reason="JAX/Pallas not available")
class TestMultiLayerScatterIntoKernel:
    """Correctness test for the fused HBM→HBM scatter.

    Shapes use (num_blocks, block_size=16, kv_heads=2, head_dim=128) to match
    the TPU Mosaic tile alignment requirement (last dim must be ≥ 128 and a
    multiple of 128).  This mirrors the shape used in smoke_test_multi_layer_scatter.

    Verifies that after multi_layer_scatter_into, kv_caches[l][dest[i]] ==
    src[l][i] for each layer l and block index i, and that unwritten slots
    remain zero.

    Each test runs in a `spawn`d subprocess (see _run_in_subprocess); the
    libtpu lockfile is owned by the child and released on its exit.
    """

    def test_single_layer_scatter_correctness(self):
        _run_in_subprocess(_body_single_layer_scatter_correctness)

    def test_multi_layer_scatter_all_layers_updated(self):
        _run_in_subprocess(_body_multi_layer_scatter_all_layers_updated)


# ---------------------------------------------------------------------------
# TestSmokeTestMultiLayerScatter
# ---------------------------------------------------------------------------


def _body_smoke_test_passes_on_tpu():
    device = torch.device("tpu")
    result = smoke_test_multi_layer_scatter(device)
    assert result is True


@pytest.mark.skipif(not scatter_available(), reason="JAX/Pallas not available")
class TestSmokeTestMultiLayerScatter:
    def test_smoke_test_passes_on_tpu(self):
        _run_in_subprocess(_body_smoke_test_passes_on_tpu)
