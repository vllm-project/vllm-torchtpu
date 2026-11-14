"""Both sides of the data-parallel token-count fork lift the combine kernel's buffers: the route below the threshold
passes them to the kernel, the fused-EP route above it and vLLM's reduce-scatter fallback read them through
retain_dp_combine_inputs. The compile flow hands a bucket's executable between the two traces only when their lifted
inputs agree (compilation/shape_variants), and a read whose result is dead leaves the trace with its buffer. Traced
the way the served forward is (dynamo with a capturing backend), on the 8-core mesh; the kernel never runs."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")


def capture(fn, x):
    """The graph dynamo hands the compile flow for fn, traced with the token count dynamic as the served forward is."""
    seen = []

    def backend(gm, example_inputs):
        seen.append(gm)
        return lambda *args: [torch.zeros(0)]

    torch._dynamo.reset()
    try:
        torch.compile(fn, backend=backend, dynamic=True)(x)
    except Exception:
        pass  # the captured graph is not run here: the kernel has no CPU path
    assert seen, "dynamo captured no graph"
    return seen[0]


def buffers(gm):
    """The combine op's buffers among the placeholders, named as the compile flow lists them."""
    from vllm_torchtpu.compilation import shape_variants
    return {
        name
        for name, _ in shape_variants._handover_inputs(gm)
        if "small_all_reduce_op" in name
    }


def live_buffers(gm):
    """The same after dead-code elimination, keeping the buffers something still reads: the piecewise split gives a
    subgraph the inputs its nodes use, so a buffer read only by a dead expression is not among the executable's."""
    gm.graph.eliminate_dead_code()
    return {
        n.name
        for n in gm.graph.nodes if n.op == "placeholder"
        and "small_all_reduce_op" in n.name and n.users
    }


@pytest.fixture
def armed(monkeypatch):
    if jax.default_backend() != "tpu" or len(jax.devices()) < 8:
        pytest.skip("8 TPU cores are needed for the mesh")
    import vllm.distributed.parallel_state as ps
    from jax.sharding import Mesh

    from vllm_torchtpu.layers.vllm import small_all_reduce_op as op
    rs = op._build_reduce_scatter(Mesh(np.array(jax.devices()[:8]), ("d", )))
    monkeypatch.setattr(op, "_RS_OP", rs)
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_MY_RANK", torch.tensor([6], dtype=torch.int32))
    monkeypatch.setattr(
        op, "_CORE_OF_RANK",
        torch.tensor([0, 1, 4, 5, 6, 7, 2, 3], dtype=torch.int32))
    monkeypatch.setattr(op, "_CORES", 8)
    monkeypatch.setattr(op, "_DP_COMBINE", True)
    group = SimpleNamespace(ranks=list(range(8)), world_size=8)
    monkeypatch.setattr(ps, "get_dp_group", lambda: group)
    monkeypatch.setattr(ps, "get_ep_group", lambda: group)
    return op


def test_both_sides_of_the_fork_lift_the_same_buffers(armed):
    op = armed
    x = torch.zeros(64, 4096, dtype=torch.bfloat16)
    # called through the module attribute, as the communicator and the fused route call them
    below = capture(lambda x: op.dp_combine(x), x)
    above = capture(
        lambda x: op.retain_dp_combine_inputs(x.sum(0, keepdim=True)), x)
    fallback = capture(lambda x: op.retain_dp_combine_inputs(x[:8]), x)
    assert len(buffers(below)) == 3, buffers(below)
    assert buffers(above) == buffers(below) == buffers(fallback)
    assert live_buffers(above) == live_buffers(below) == live_buffers(
        fallback) == buffers(below)


def test_a_dead_read_does_not_reach_the_executable(armed):
    """The zero folded into the positions is dead in a layer that does not read them: dynamo lifts the buffer, the
    dead-code elimination before the split leaves it without a reader, and the subgraph's executable never takes it."""
    op = armed
    x = torch.zeros(64, 4096, dtype=torch.bfloat16)

    def dead(x):
        _positions = torch.arange(
            x.shape[0]) + op.zero_from_rank()  # computed, never used
        return x.sum(0, keepdim=True)

    gm = capture(dead, x)
    assert buffers(gm) and live_buffers(gm) == set()


def test_nothing_armed_is_the_identity():
    from vllm_torchtpu.layers.vllm import small_all_reduce_op as op
    if op._RS_OP is not None:
        pytest.skip("armed in this process")
    x = torch.ones(8, 4096, dtype=torch.bfloat16)
    assert op.retain_dp_combine_inputs(x) is x


def test_the_tensor_parallel_line_lifts_no_combine_buffers(armed, monkeypatch):
    """Where the data-parallel group is not the mesh (the tensor-parallel points), neither route reads the combine
    kernel's buffers: the fallback and the fused route retain nothing, so the traces agree without them."""
    op = armed
    monkeypatch.setattr(op, "_DP_COMBINE", False)
    x = torch.zeros(64, 4096, dtype=torch.bfloat16)
    assert op.dp_combine(x) is None
    assert buffers(
        capture(lambda x: op.retain_dp_combine_inputs(x.sum(0, keepdim=True)),
                x)) == set()
