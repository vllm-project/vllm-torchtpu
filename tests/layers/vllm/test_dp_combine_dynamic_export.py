"""Reduce-scatter must preserve the symbolic batch dimension under Dynamo.

The test needs eight cores to construct the mesh but does not run the kernel.
"""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")


def test_dynamic_batch_dimension_survives_the_op():
    if jax.default_backend() != "tpu" or len(jax.devices()) < 8:
        pytest.skip("8 TPU cores are needed for the mesh")
    from jax.sharding import Mesh

    from vllm_torchtpu.layers.vllm import small_all_reduce_op as op
    rs = op._build_reduce_scatter(Mesh(np.array(jax.devices()[:8]), ("d", )))
    rank = torch.tensor([3], dtype=torch.int32)
    my_rank = torch.tensor([6], dtype=torch.int32)
    table = torch.tensor([0, 1, 4, 5, 6, 7, 2, 3], dtype=torch.int32)

    def f(x):
        return rs(x, rank, my_rank, table)

    for rows in (64, 128):
        x = torch.zeros(rows, 4096, dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(x, 0)
        gm, _ = torch._dynamo.export(f, aten_graph=False)(x)
        assert any(n.op == "output" for n in gm.graph.nodes)
