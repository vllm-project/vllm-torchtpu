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
"""Unit and multichip tests for FX graph fingerprinting across subgroups."""

import argparse
import importlib.util
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import traceback
import weakref
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed._functional_collectives
import torch.distributed.distributed_c10d as c10d
import torch.fx as fx
from vllm.config.utils import Range
from vllm.distributed import parallel_state

from vllm_torchtpu.compilation.graph_fingerprint import (
    _GROUP_ARG_NAMES,
    _SYMMETRIC_COLLECTIVE_SCHEMAS,
    _graph_fingerprint,
    _normalized_module_code,
)
from vllm_torchtpu.distributed.utils import _canonicalize_process_group_name

WORLD_SIZE = 8
SUBGROUP_SIZE = 4
ROWS = 4
HIDDEN = 8
# Stays below the multichip lane's pytest timeout, so _launch stops the group.
LAUNCH_TIMEOUT_SECONDS = 480
SKIP_EXIT_CODE = 75
_test_lib: torch.library.Library | None = None


def _ensure_test_collective_op() -> None:
    global _test_lib
    if hasattr(torch.ops.vllm_fingerprint_test, "moe_alltoall_combine"):
        return
    _test_lib = torch.library.Library("vllm_fingerprint_test", "FRAGMENT")
    _test_lib.define("moe_alltoall_combine(Tensor x, str group) -> Tensor")
    _test_lib.impl("moe_alltoall_combine", lambda x, group: x, "CPU")
    torch.library.register_fake(
        "vllm_fingerprint_test::moe_alltoall_combine",
        lambda x, group: torch.empty_like(x),
    )


def _trace_all_reduce_graph(group_name: str, reduce_op: str = "sum") -> fx.GraphModule:
    def forward(x: torch.Tensor) -> torch.Tensor:
        reduced = torch.ops._c10d_functional.all_reduce(x, reduce_op, group_name)
        waited = torch.ops._c10d_functional.wait_tensor(reduced)
        return waited + 1.0

    return fx.symbolic_trace(forward)


def _trace_multi_collective_graph(
    tp_group: str,
    ep_group: str,
    reduce_op: str = "sum",
    group_size: int = 2,
) -> fx.GraphModule:
    def forward(x: torch.Tensor) -> torch.Tensor:
        scattered = torch.ops._c10d_functional.reduce_scatter_tensor(
            x, reduce_op, group_size, tp_group
        )
        scattered_wait = torch.ops._c10d_functional.wait_tensor(scattered)
        routed = torch.ops._c10d_functional.all_to_all_single(
            scattered_wait, [2, 2], [2, 2], ep_group
        )
        routed_wait = torch.ops._c10d_functional.wait_tensor(routed)
        gathered = torch.ops._c10d_functional.all_gather_into_tensor(
            routed_wait, group_size, tp_group
        )
        gathered_wait = torch.ops._c10d_functional.wait_tensor(gathered)
        reduced = torch.ops._c10d_functional.all_reduce(
            gathered_wait, reduce_op, tp_group
        )
        return torch.ops._c10d_functional.wait_tensor(reduced)

    return fx.symbolic_trace(forward)


def install_group_stand_ins(monkeypatch):
    class _FakeDeviceGroup:
        def __init__(self, group_name: str) -> None:
            self.group_name = group_name

    class _FakeGroup:
        def __init__(
            self,
            unique_name: str,
            ranks: list[int],
            group_ranks: list[list[int]],
            device_group_name: str | None = None,
        ) -> None:
            self.unique_name = unique_name
            self.ranks = ranks
            self.world_size = len(ranks)
            self.group_ranks = group_ranks
            self.device_group = (
                _FakeDeviceGroup(device_group_name)
                if device_group_name is not None
                else None
            )

    tp_2_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    tp_strided_groups = [[0, 4], [1, 5], [2, 6], [3, 7]]
    dp_2_groups = [[0, 2], [1, 3], [4, 6], [5, 7]]
    ep_8_groups = [[0, 1, 2, 3, 4, 5, 6, 7]]
    tp_wrap_groups = [[0, 1, 6, 7], [4, 5, 2, 3]]

    live_groups = [
        _FakeGroup(unique_name="tp:0", ranks=[0, 1], group_ranks=tp_2_groups),
        _FakeGroup(unique_name="tp:3", ranks=[6, 7], group_ranks=tp_2_groups),
        _FakeGroup(unique_name="tp:4", ranks=[0, 4], group_ranks=tp_strided_groups),
        _FakeGroup(unique_name="dp:0", ranks=[0, 2], group_ranks=dp_2_groups),
        _FakeGroup(unique_name="dp:3", ranks=[5, 7], group_ranks=dp_2_groups),
        _FakeGroup(unique_name="pcp:0", ranks=[0, 1], group_ranks=tp_2_groups),
        _FakeGroup(unique_name="pcp:3", ranks=[6, 7], group_ranks=tp_2_groups),
        _FakeGroup(unique_name="dcp:0", ranks=[0, 1], group_ranks=tp_2_groups),
        _FakeGroup(unique_name="dcp:3", ranks=[6, 7], group_ranks=tp_2_groups),
        _FakeGroup(
            unique_name="ep:0",
            ranks=[0, 1, 2, 3, 4, 5, 6, 7],
            group_ranks=ep_8_groups,
        ),
        _FakeGroup(
            unique_name="ep:3",
            ranks=[0, 1, 2, 3, 4, 5, 6, 7],
            group_ranks=ep_8_groups,
        ),
        _FakeGroup(
            unique_name="tp_wrap:0",
            ranks=[0, 1, 6, 7],
            group_ranks=tp_wrap_groups,
            device_group_name="tp_wrap_c10d_0",
        ),
        _FakeGroup(
            unique_name="tp_wrap:1",
            ranks=[4, 5, 2, 3],
            group_ranks=tp_wrap_groups,
            device_group_name="tp_wrap_c10d_1",
        ),
        _FakeGroup(
            unique_name="pcp_wrap:0",
            ranks=[0, 4],
            group_ranks=tp_strided_groups,
            device_group_name="pcp_wrap_c10d_0",
        ),
    ]
    monkeypatch.setattr(
        parallel_state,
        "_groups",
        {group.unique_name: weakref.ref(group) for group in live_groups},
    )

    pg_ranks = {
        "1": [0, 1],
        "7": [6, 7],
        "13": [0, 2, 4, 6],
        "15": [1, 3, 5, 7],
        "99": [0, 4],
    }

    def _resolve_pg(name):
        key = str(name)
        if key not in pg_ranks:
            raise KeyError(key)
        return key

    monkeypatch.setattr(c10d, "_resolve_process_group", _resolve_pg)
    monkeypatch.setattr(c10d, "get_process_group_ranks", lambda pg: pg_ranks[str(pg)])
    return live_groups


@pytest.fixture
def group_stand_ins(monkeypatch):
    return install_group_stand_ins(monkeypatch)


class TestGraphFingerprint:
    def test_fingerprint_canonicalizes_c10d_subgroup_names_and_preserves_graph(
        self, group_stand_ins
    ):
        """Multi-group and role-indexed graphs match across subgroups without
        mutating the GraphModule's code or node arguments."""
        gm_rank0 = _trace_multi_collective_graph(tp_group="1", ep_group="13")
        gm_rank7 = _trace_multi_collective_graph(tp_group="7", ep_group="15")

        code_before = gm_rank0.code
        args_before = [(node.args, dict(node.kwargs)) for node in gm_rank0.graph.nodes]

        fp_rank0 = _graph_fingerprint(gm_rank0)
        fp_rank7 = _graph_fingerprint(gm_rank7)

        assert fp_rank0 is not None
        assert fp_rank0 == fp_rank7
        assert gm_rank0.code == code_before
        assert [
            (node.args, dict(node.kwargs)) for node in gm_rank0.graph.nodes
        ] == args_before

        for role in ("tp", "dp", "pcp", "dcp", "ep"):
            assert _graph_fingerprint(
                _trace_all_reduce_graph(f"{role}:0")
            ) == _graph_fingerprint(_trace_all_reduce_graph(f"{role}:3"))

    def test_fingerprint_with_vllm_parallel_state_and_c10d_stand_ins(
        self, group_stand_ins
    ):
        """Resolved vLLM GroupCoordinators and c10d ProcessGroups encode
        partition or stride geometry and differ when role, size, or layout
        differ."""
        assert (
            _canonicalize_process_group_name("tp:0", 0)
            == "vllm_pg:0:tp:size=2:group_ranks=((0, 1), (2, 3), (4, 5), (6, 7))"
        )
        assert (
            _canonicalize_process_group_name("tp:4", 0)
            == "vllm_pg:0:tp:size=2:group_ranks=((0, 4), (1, 5), (2, 6), (3, 7))"
        )
        assert (
            _canonicalize_process_group_name("1", 0) == "c10d_pg:0:size=2:strides=(1,)"
        )
        assert _graph_fingerprint(
            _trace_all_reduce_graph("tp:0")
        ) != _graph_fingerprint(_trace_all_reduce_graph("tp:4"))
        assert _graph_fingerprint(
            _trace_all_reduce_graph("tp:0")
        ) != _graph_fingerprint(_trace_all_reduce_graph("ep:0"))

        assert _graph_fingerprint(_trace_all_reduce_graph("1")) != _graph_fingerprint(
            _trace_all_reduce_graph("13")
        )
        assert _graph_fingerprint(_trace_all_reduce_graph("1")) != _graph_fingerprint(
            _trace_all_reduce_graph("99")
        )

    def test_fingerprint_matches_across_wrap_around_tpu7x_subgroups(
        self, group_stand_ins
    ):
        """Wrap-around tpu7x subgroups (e.g. tp=(0,1,6,7) and tp=(4,5,2,3))
        share the same GroupCoordinator.group_ranks partition and match whether
        referenced by unique_name or device_group.group_name."""
        assert _graph_fingerprint(
            _trace_all_reduce_graph("tp_wrap:0")
        ) == _graph_fingerprint(_trace_all_reduce_graph("tp_wrap:1"))
        assert _graph_fingerprint(
            _trace_all_reduce_graph("tp_wrap_c10d_0")
        ) == _graph_fingerprint(_trace_all_reduce_graph("tp_wrap_c10d_1"))
        assert _graph_fingerprint(
            _trace_all_reduce_graph("tp_wrap_c10d_0")
        ) != _graph_fingerprint(_trace_all_reduce_graph("pcp_wrap_c10d_0"))

    def test_fingerprint_canonicalizes_keyword_group_name_and_packet_without_default(
        self, group_stand_ins
    ):
        """Keyword group_name arguments are canonicalized without mutating
        kwargs, and OpOverloadPackets lacking a 'default' overload (such as
        torch.ops.aten.add_) do not cause _node_schema to fail."""
        assert "default" not in torch.ops.aten.add_.overloads()

        def _build_kwarg_graph(group_name: str) -> fx.GraphModule:
            graph = fx.Graph()
            x = graph.placeholder("x")
            added = graph.call_function(torch.ops.aten.add_, (x, x))
            reduced = graph.call_function(
                torch.ops._c10d_functional.all_reduce.default,
                (added,),
                {"reduce_op": "sum", "group_name": group_name},
            )
            waited = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default, (reduced,)
            )
            graph.output(waited)
            return fx.GraphModule(torch.nn.Module(), graph)

        gm_rank0 = _build_kwarg_graph("1")
        gm_rank7 = _build_kwarg_graph("7")
        gm_rank_other = _build_kwarg_graph("13")
        kwargs_before = [dict(node.kwargs) for node in gm_rank0.graph.nodes]

        fp_rank0 = _graph_fingerprint(gm_rank0)
        assert fp_rank0 is not None
        assert fp_rank0 == _graph_fingerprint(gm_rank7)
        assert fp_rank0 != _graph_fingerprint(gm_rank_other)
        assert [dict(node.kwargs) for node in gm_rank0.graph.nodes] == kwargs_before

    def test_fingerprint_shares_group_ordinals_across_submodules(self, group_stand_ins):
        """Group ordinals are shared across graph.named_modules() so submodules
        using distinct same-geometry groups do not reset to ordinal 0."""

        def _build_two_submodule_graph(group_a: str, group_b: str) -> fx.GraphModule:
            root = torch.nn.Module()
            root.sub_a = _trace_all_reduce_graph(group_a)
            root.sub_b = _trace_all_reduce_graph(group_b)
            graph = fx.Graph()
            x = graph.placeholder("x")
            y = graph.call_module("sub_a", (x,))
            z = graph.call_module("sub_b", (y,))
            graph.output(z)
            return fx.GraphModule(root, graph)

        fp_same_0 = _graph_fingerprint(_build_two_submodule_graph("1", "1"))
        fp_same_7 = _graph_fingerprint(_build_two_submodule_graph("7", "7"))
        fp_distinct_07 = _graph_fingerprint(_build_two_submodule_graph("1", "7"))
        fp_distinct_70 = _graph_fingerprint(_build_two_submodule_graph("7", "1"))

        assert fp_same_0 is not None
        assert fp_same_0 == fp_same_7
        assert fp_distinct_07 == fp_distinct_70
        assert fp_same_0 != fp_distinct_07

    def test_non_symmetric_collective_returns_none_fingerprint(self, group_stand_ins):
        """Rooted, point-to-point, or unlisted custom collectives raise
        ValueError in _normalized_module_code so _graph_fingerprint returns
        None instead of aliasing across subgroups."""

        def _broadcast_forward(x: torch.Tensor) -> torch.Tensor:
            out = torch.ops._c10d_functional.broadcast(x, 0, "1")
            return torch.ops._c10d_functional.wait_tensor(out)

        gm_broadcast = fx.symbolic_trace(_broadcast_forward)
        with pytest.raises(
            ValueError,
            match=(
                "Cannot canonicalize non-symmetric collective "
                "'_c10d_functional::broadcast'"
            ),
        ):
            _normalized_module_code(gm_broadcast)
        assert _graph_fingerprint(gm_broadcast) is None

        _ensure_test_collective_op()

        def _unlisted_custom_collective(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.vllm_fingerprint_test.moe_alltoall_combine(x, "1")

        gm_unlisted = fx.symbolic_trace(_unlisted_custom_collective)
        with pytest.raises(
            ValueError,
            match=(
                "Cannot canonicalize non-symmetric collective "
                "'vllm_fingerprint_test::moe_alltoall_combine'"
            ),
        ):
            _normalized_module_code(gm_unlisted)
        assert _graph_fingerprint(gm_unlisted) is None

    def test_symmetric_collective_schemas_cover_registered_ops(self):
        """Every registered collective schema in _c10d_functional,
        _c10d_functional_autograd, tpu, and vllm taking a group argument must
        be explicitly classified in _SYMMETRIC_COLLECTIVE_SCHEMAS or
        expected_asymmetric."""
        target_namespaces = {
            "_c10d_functional",
            "_c10d_functional_autograd",
            "tpu",
            "vllm",
        }
        expected_asymmetric = {
            "_c10d_functional::broadcast",
            "_c10d_functional::broadcast_",
            "_c10d_functional::isend",
            "_c10d_functional::irecv",
            "_c10d_functional::batch_p2p_ops",
        }
        matched_symmetric: set[str] = set()
        matched_asymmetric: set[str] = set()
        unclassified: set[str] = set()

        for schema in torch._C._jit_get_all_schemas():
            ns = schema.name.partition("::")[0]
            if ns not in target_namespaces:
                continue
            if not any(arg.name in _GROUP_ARG_NAMES for arg in schema.arguments):
                continue
            if schema.name in _SYMMETRIC_COLLECTIVE_SCHEMAS:
                matched_symmetric.add(schema.name)
            elif schema.name in expected_asymmetric:
                matched_asymmetric.add(schema.name)
            else:
                unclassified.add(schema.name)

        assert not unclassified, f"Unclassified collective schemas: {unclassified}"
        assert matched_asymmetric == expected_asymmetric
        assert {
            "_c10d_functional::all_reduce",
            "_c10d_functional::all_gather_into_tensor",
            "_c10d_functional::reduce_scatter_tensor",
            "_c10d_functional::all_to_all_single",
            "_c10d_functional_autograd::all_gather_into_tensor",
            "_c10d_functional_autograd::reduce_scatter_tensor",
            "_c10d_functional_autograd::all_to_all_single",
        }.issubset(matched_symmetric)

    def test_unresolved_group_returns_none_fingerprint(self, group_stand_ins):
        """An unresolvable group_name raises ValueError in
        _canonicalize_process_group_name and causes _graph_fingerprint to return
        None."""
        with pytest.raises(
            ValueError,
            match="Cannot resolve process group 'unknown_pg'",
        ):
            _canonicalize_process_group_name("unknown_pg", 0)

        assert _graph_fingerprint(_trace_all_reduce_graph("unknown_pg")) is None

    def test_fingerprint_negative_collision_checks(self, group_stand_ins):
        """Semantically different graphs must never collide after canonicalization."""
        fp_sum = _graph_fingerprint(_trace_all_reduce_graph("1", reduce_op="sum"))
        fp_avg = _graph_fingerprint(_trace_all_reduce_graph("1", reduce_op="avg"))
        assert fp_sum != fp_avg

        def _reduce_scatter_only(x: torch.Tensor) -> torch.Tensor:
            out = torch.ops._c10d_functional.reduce_scatter_tensor(x, "sum", 2, "1")
            return torch.ops._c10d_functional.wait_tensor(out)

        assert fp_sum != _graph_fingerprint(fx.symbolic_trace(_reduce_scatter_only))

        assert _graph_fingerprint(
            _trace_multi_collective_graph("1", "13", group_size=2)
        ) != _graph_fingerprint(_trace_multi_collective_graph("1", "13", group_size=4))

        assert _graph_fingerprint(
            _trace_multi_collective_graph(tp_group="1", ep_group="1")
        ) != _graph_fingerprint(
            _trace_multi_collective_graph(tp_group="1", ep_group="13")
        )

        assert _graph_fingerprint(
            _trace_all_reduce_graph("tp:0")
        ) != _graph_fingerprint(_trace_all_reduce_graph("dp:0"))


# ----------------------------------------------------------- multichip worker
def _launch_command(result_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--launch",
        "--result-dir",
        str(result_dir),
    ]


def _torchrun_args(result_dir: Path) -> list[str]:
    return [
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--result-dir",
        str(result_dir),
    ]


def _require_torchtpu() -> None:
    if importlib.util.find_spec("torch_tpu") is None:
        pytest.skip("TorchTPU is unavailable")


def _launch_torchrun(result_dir: Path) -> None:
    """Runs in the launcher child: a fresh eight-device environment, then torchrun."""
    for key in (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
        "LIBTPU_INIT_ARGS",
    ):
        os.environ.pop(key, None)
    try:
        from tests.distributed.tpu_test_utils import prepare_tpu_environment

        prepare_tpu_environment(world_size=WORLD_SIZE)
    except (ImportError, ValueError) as exc:
        print(f"Eight TPU devices are unavailable: {exc}", flush=True)
        sys.exit(SKIP_EXIT_CODE)
    os.environ["TPU_PRECOMPILE_PEER_REPLAY"] = "1"
    os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")
    from torch.distributed.run import main as torchrun_main

    torchrun_main(_torchrun_args(result_dir))


def _run_worker(result_dir: Path) -> None:
    import torch_tpu  # noqa: F401
    from torch import distributed as dist
    from torch_tpu._internal.compile import (
        torch_tpu_compiled_executable as tpu_compiled_exe,
    )
    from vllm.distributed.parallel_state import (
        get_world_group,
        init_distributed_environment,
        init_model_parallel_group,
    )

    from vllm_torchtpu.compilation import tpu_compiler
    from vllm_torchtpu.compilation.tpu_compiler import (
        TpuCompilationHandle,
        TpuCompilerAdaptor,
        wait_for_peer_artifact,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result_path = result_dir / f"rank_{rank}.json"
    result: dict[str, object] = {"rank": rank, "world": world}
    try:
        dist.init_process_group(backend="tpu_dist")
        device = torch.device("tpu")
        torch.empty((1,), device=device).cpu()

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        world_group = get_world_group()
        tp_group = init_model_parallel_group(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            world_group.local_rank,
            dist.get_backend(world_group.device_group),
            group_name="tp_fingerprint_test",
        )
        assert tp_group.device_group is not None
        pg_name = tp_group.device_group.group_name
        subgroup = rank // SUBGROUP_SIZE

        gm_traced = _trace_all_reduce_graph(pg_name)
        traced_fingerprint = _graph_fingerprint(gm_traced)
        assert traced_fingerprint is not None

        def _compile_and_run(
            cache_root: Path,
            wait_for_leader: bool,
            forbid_backend: bool = False,
        ) -> tuple[float, Path, bool, str]:
            compiler = TpuCompilerAdaptor()
            compiler.initialize_cache(
                cache_dir=str(cache_root / f"rank_{rank}_0" / "backbone")
            )
            backend_called = False
            exe_fp = ""
            orig_run_backend = compiler._run_backend
            orig_deserialize = tpu_compiler._deserialize_entry

            def _tracked_run_backend(graph, example_inputs):
                nonlocal backend_called, exe_fp
                if forbid_backend:
                    raise AssertionError(
                        f"rank {rank} invoked _run_backend instead of replaying peer"
                    )
                backend_called = True
                compiled_fn, entry = orig_run_backend(graph, example_inputs)
                if entry is not None:
                    exe_fp = entry.compiled_fw.result.fingerprint()
                return compiled_fn, entry

            def _tracked_deserialize(entry):
                nonlocal exe_fp
                exe_fp = entry.compiled_fw.result.fingerprint()
                return orig_deserialize(entry)

            compiler._run_backend = _tracked_run_backend  # type: ignore[method-assign]
            saved_handles: list[Any] = []

            def _adaptor_backend(graph, example_inputs):
                ctx = (
                    wait_for_peer_artifact(120.0)
                    if wait_for_leader
                    else wait_for_peer_artifact(0.0)
                )
                with (
                    ctx,
                    patch.object(
                        tpu_compiler, "_deserialize_entry", _tracked_deserialize
                    ),
                ):
                    fn, handle = compiler.compile(
                        graph,
                        example_inputs,
                        {},
                        Range(start=ROWS, end=ROWS),
                        key="all_reduce_bucket",
                    )
                assert fn is not None and handle is not None
                saved_handles.append(handle)
                return fn

            @torch.compile(backend=_adaptor_backend, fullgraph=True, dynamic=False)
            def _compiled_step(x: torch.Tensor) -> torch.Tensor:
                reduced = torch.ops._c10d_functional.all_reduce(x, "sum", pg_name)
                waited = torch.ops._c10d_functional.wait_tensor(reduced)
                return waited + 1.0

            x_tpu = torch.full(
                (ROWS, HIDDEN), float(rank + 1), dtype=torch.float32, device=device
            )
            out = _compiled_step(x_tpu).cpu()
            assert len(saved_handles) == 1
            _, saved_path, _ = saved_handles[0]
            return float(out[0, 0].item()), Path(saved_path), backend_called, exe_fp

        # 1. Independent compilation per TP subgroup: rank 0 compiles subgroup 0
        # ([0, 1, 2, 3]) and rank 4 compiles subgroup 1 ([4, 5, 6, 7]) in separate
        # cache roots.
        indep_val, indep_path, indep_compiled, exe_fingerprint = _compile_and_run(
            result_dir / f"independent_{subgroup}",
            wait_for_leader=(rank % SUBGROUP_SIZE != 0),
        )
        with (
            patch.object(
                tpu_compiled_exe,
                "_unpickle_compiled_executable",
                side_effect=lambda serialized_bytes, *_args: serialized_bytes,
            ),
            open(indep_path, "rb") as f,
        ):
            payload = pickle.load(f)
        assert isinstance(payload, TpuCompilationHandle)
        assert payload.entry is not None

        # 2. Cross-subgroup peer replay: in a shared host cache root, only rank 0
        # compiles while ranks 1..3 and subgroup 1 (ranks 4..7) replay rank 0's
        # artifact.
        torch._dynamo.reset()
        shared_val, shared_path, shared_compiled, _ = _compile_and_run(
            result_dir / "shared",
            wait_for_leader=(rank != 0),
            forbid_backend=(rank != 0),
        )
        rank0_shared_path = (
            result_dir / "shared" / "rank_0_0" / "backbone" / "all_reduce_bucket"
        )

        result.update(
            {
                "subgroup": subgroup,
                "pg_name": pg_name,
                "unique_name": tp_group.unique_name,
                "traced_fingerprint": traced_fingerprint,
                "handle_fingerprint": payload.fingerprint,
                "handle_signature": repr(payload.signature),
                "handle_was_wrapped": payload.was_wrapped,
                "executable_fingerprint": exe_fingerprint,
                "indep_compiled": indep_compiled,
                "indep_val": indep_val,
                "shared_compiled": shared_compiled,
                "shared_val": shared_val,
                "shared_samefile_as_rank0": os.path.samefile(
                    shared_path, rank0_shared_path
                ),
            }
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result_path.write_text(json.dumps(result), encoding="utf-8")
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline and not all(
            (result_dir / f"rank_{r}.json").exists() for r in range(world)
        ):
            time.sleep(0.05)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if "error" not in result else 1)


def _launch(result_dir: Path) -> subprocess.CompletedProcess[str]:
    _require_torchtpu()
    process = subprocess.Popen(
        _launch_command(result_dir),
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
    finally:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    assert process.returncode is not None
    return subprocess.CompletedProcess(process.args, process.returncode, output, "")


def _results(result_dir: Path) -> list[dict]:
    out = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
        out.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    return out


@pytest.mark.multichip
@pytest.mark.spawns_tpu_workers
def test_all_reduce_across_tp_subgroups_compiles_matching_artifacts_on_tpu(tmp_path):
    """Two TP subgroups ([0, 1, 2, 3] and [4, 5, 6, 7]) trace distinct c10d
    group names yet produce matching _graph_fingerprint and matching PJRT
    compiled executables, and subgroup 1 replays subgroup 0's compiled artifact
    on real TPU devices."""
    result_dir = tmp_path / "tp_subgroups"
    result_dir.mkdir()
    completed = _launch(result_dir)
    if completed.returncode == SKIP_EXIT_CODE:
        pytest.skip(completed.stdout.strip().splitlines()[-1])
    results = _results(result_dir)
    assert completed.returncode == 0 or any("error" in r for r in results), (
        completed.stdout[-12000:]
    )
    errors = [r for r in results if "error" in r]
    assert not errors, f"worker failures: {errors}"

    sg0_pg_names = {results[r]["pg_name"] for r in range(SUBGROUP_SIZE)}
    sg1_pg_names = {results[r]["pg_name"] for r in range(SUBGROUP_SIZE, WORLD_SIZE)}
    assert len(sg0_pg_names) == 1
    assert len(sg1_pg_names) == 1
    assert sg0_pg_names != sg1_pg_names
    assert (
        results[0]["unique_name"]
        == results[SUBGROUP_SIZE]["unique_name"]
        == "tp_fingerprint_test:0"
    )

    assert results[0]["traced_fingerprint"] is not None
    assert len({r["traced_fingerprint"] for r in results}) == 1, (
        f"mismatched traced fingerprints: {[r['traced_fingerprint'] for r in results]}"
    )
    assert len({r["handle_fingerprint"] for r in results}) == 1, (
        f"mismatched handle fingerprints: {[r['handle_fingerprint'] for r in results]}"
    )
    assert len({r["handle_signature"] for r in results}) == 1
    assert len({r["handle_was_wrapped"] for r in results}) == 1

    assert results[0]["executable_fingerprint"], "empty PJRT executable fingerprint"
    assert len({r["executable_fingerprint"] for r in results}) == 1, (
        f"mismatched PJRT executables: {[r['executable_fingerprint'] for r in results]}"
    )

    assert [r["indep_compiled"] for r in results] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert [r["shared_compiled"] for r in results] == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert all(r["shared_samefile_as_rank0"] for r in results)

    # Subgroup 0 ([0, 1, 2, 3]): (1.0 + 2.0 + 3.0 + 4.0) + 1.0 = 11.0
    # Subgroup 1 ([4, 5, 6, 7]): (5.0 + 6.0 + 7.0 + 8.0) + 1.0 = 27.0
    expected_vals = [11.0] * SUBGROUP_SIZE + [27.0] * SUBGROUP_SIZE
    assert [r["indep_val"] for r in results] == expected_vals
    assert [r["shared_val"] for r in results] == expected_vals


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.launch == args.worker:
        raise ValueError("expected exactly one of --launch and --worker")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.launch:
        _launch_torchrun(args.result_dir)
    else:
        _run_worker(args.result_dir)


if __name__ == "__main__":
    _main()
