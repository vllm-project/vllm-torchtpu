"""The data-parallel combine through the small reduce-scatter: the op is called with the block, the rank buffer,
this rank and the rank-to-core table, and its result is returned as is; the fallback stays for other groups or blocks."""
from types import SimpleNamespace
from unittest import mock

import torch

from vllm_torchtpu.layers.vllm import small_all_reduce_op as op


def _groups(ranks, rank_in_group):
    g = SimpleNamespace(ranks=list(ranks),
                        world_size=len(ranks),
                        rank_in_group=rank_in_group)
    return g


def test_calls_the_reduce_scatter_op_with_the_rank_operands():
    x = torch.arange(64 * 4096,
                     dtype=torch.float32).reshape(64, 4096).to(torch.bfloat16)
    seen = {}

    def fake_rs(block, rank, my_rank, core_of_rank):
        seen.update(block=block,
                    rank=rank,
                    my_rank=my_rank,
                    core_of_rank=core_of_rank)
        return block[:8]

    with mock.patch.object(op, "_OP", object()), mock.patch.object(op, "_RS_OP", fake_rs), mock.patch.object(op, "_RANK", torch.tensor([3])), \
         mock.patch.object(op, "_MY_RANK", torch.tensor([6])), mock.patch.object(op, "_CORE_OF_RANK", torch.tensor([0, 1, 4, 5, 6, 7, 2, 3])), \
         mock.patch.object(op, "_CORES", 8), mock.patch.object(op, "_DP_COMBINE", True), \
         mock.patch.object(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 64):
        out = op.dp_combine(x)
    assert out.shape == (8, 4096)
    assert seen["block"] is x and int(seen["rank"][0]) == 3 and int(
        seen["my_rank"][0]) == 6
    assert seen["core_of_rank"].tolist() == [0, 1, 4, 5, 6, 7, 2, 3]


def test_falls_back_when_the_groups_differ_or_the_block_does_not_fit_the_kernel(
):
    x = torch.zeros(64, 4096, dtype=torch.bfloat16)
    # the data-parallel group is not the mesh (decided at prebuild): the fallback, and the retain keeps nothing
    with mock.patch.object(op, "_OP", lambda t, r: t), mock.patch.object(op, "_RS_OP", lambda *a: a[0][:8]), mock.patch.object(op, "_CORES", 8), \
         mock.patch.object(op, "_DP_COMBINE", False), mock.patch.object(op, "_RANK", torch.tensor([3])), \
         mock.patch.object(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 64):
        assert op.dp_combine(x) is None
        assert op.retain_dp_combine_inputs(x) is x
    # a block that is not bf16 or whose width does not split over the cores falls back; the row count never decides
    with mock.patch.object(op, "_OP", lambda t, r: t), mock.patch.object(op, "_RS_OP", lambda *a: a[0][:8]), mock.patch.object(op, "_CORES", 8), \
         mock.patch.object(op, "_DP_COMBINE", True):
        assert op.dp_combine(x.float()) is None
        assert op.dp_combine(torch.zeros(64, 4000,
                                         dtype=torch.bfloat16)) is None
        assert op.dp_combine(torch.zeros(1024, 4096,
                                         dtype=torch.bfloat16)) is not None


def test_the_op_builders_traced_functions_carry_the_annotations_the_wrapper_requires(
):
    """torch_tpu's pallas.jax_op refuses a traced function without jax.Array annotations at boot; the kernel tests do
    not go through the wrapper, so this checks both builders' inner functions with the wrapper's own check."""
    import ast
    import inspect
    import textwrap

    from torch_tpu._internal.pallas import pallas as pallas_impl
    src = inspect.getsource(op)
    tree = ast.parse(src)
    for name in ("_build", "_build_reduce_scatter"):
        builder = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == name)
        inner = next(n for n in ast.walk(builder)
                     if isinstance(n, ast.FunctionDef) and n.name == "fn")
        ns = {}
        exec("import jax\n" + textwrap.dedent(ast.unparse(inner)), ns)
        pallas_impl._verify_signature(inspect.signature(ns["fn"]))


def test_the_traced_combine_does_not_log_or_write_globals():
    """dp_combine runs inside the compiled forward: dynamo refuses a Logger call there, and a global write would freeze
    at the first trace. Logging belongs to prebuild."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(op.dp_combine)))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)]
    assert not [
        n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name) and n.value.id == "logger"
    ]
