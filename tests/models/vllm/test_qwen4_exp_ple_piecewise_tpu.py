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
"""Compile Qwen request indexing on both sides of the real host lookup op."""

import copy
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_request_indexing_across_host_lookup_compiles_all_buckets(fp8):
    from torch_tpu._internal import sync
    from vllm.compilation.backends import split_graph
    from vllm.compilation.codegen import generate_execution_code
    from vllm.compilation.piecewise_backend import create_concrete_args
    from vllm.forward_context import override_forward_context

    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import token_requests
    from vllm_torchtpu.models.vllm.qwen4_exp.ple_offload import host_lookup
    from vllm_torchtpu.models.vllm.qwen4_exp.ple_storage import HostPLETable

    buckets = (8, 16, 32, 64, 128)
    compiler = TpuCompilerAdaptor()
    compiled_pieces = []

    class Piece(torch.nn.Module):
        def __init__(self, graph):
            super().__init__()
            placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]
            symbols = [
                i
                for i, node in enumerate(placeholders)
                if isinstance(node.meta["example_value"], torch.SymInt)
                and node.meta["example_value"].node.expr.is_Symbol
            ]
            assert len(symbols) == 1
            self.shape_index = symbols[0]
            self.runs = {}
            for count in buckets:
                # Use vLLM's bucket specialization and the actual TPU adaptor,
                # including AOTAutograd and native executable creation.
                self.runs[count], _ = compiler._run_backend(
                    copy.deepcopy(graph), create_concrete_args(graph, count)
                )
            compiled_pieces.append(self)

        def forward(self, *args):
            return self.runs[int(args[self.shape_index])](*args)

    def backend(graph, inputs):
        boundaries = ["qwen4_exp::host_lookup"]
        split, pieces = split_graph(graph, boundaries)
        assert sum(piece.is_splitting_graph for piece in pieces) == len(boundaries)
        for piece in pieces:
            if piece.is_splitting_graph:
                continue
            output = next(n for n in piece.graph.graph.nodes if n.op == "output")
            assert all(
                isinstance(n.meta["example_value"], torch.Tensor)
                for n in output.args[0]
            )
            split.set_submodule(piece.submod_name, Piece(piece.graph))
        # Serving also generates Python for the split graph. Merely calling
        # split.forward misses unsupported operations on eager boundaries.
        generate_execution_code(split)
        return split.forward

    def forward(tokens, starts):
        request, position = token_requests(starts, tokens.shape[0])
        ids = (tokens + position + request)[:, None]
        raw = host_lookup(ids, tokens.shape[0], 8, 1, fp8, "ple")
        request_after, position_after = token_requests(starts, tokens.shape[0])
        return (
            raw.reshape(-1, 8).to(torch.int32).sum(-1) + request_after + position_after
        )

    bits = (torch.arange(512 * 8, device="cpu") % 97).reshape(512, 8)
    bits = bits.to(torch.int8 if fp8 else torch.int16)
    # Real byte lookup over a small host-table fixture; no checkpoint loading.
    embedding = HostPLETable.__new__(HostPLETable)
    torch.nn.Module.__init__(embedding)
    embedding._finalized = True
    embedding.storage_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    embedding.shard_indices = NS(org_vocab_start_index=0, org_vocab_end_index=512)
    embedding.embedding_dim = 8
    embedding._shared_table = None
    embedding._gather_workers = 1
    embedding._gather_pool = None
    embedding.weight = torch.nn.Parameter(
        bits.view(embedding.storage_dtype), requires_grad=False
    )
    context = NS(
        no_compile_layers={"ple": NS(ple_embedding=NS(ngram_embedding=embedding))}
    )
    starts = (torch.arange(65, device="cpu", dtype=torch.int32) * 2).to("tpu")
    run = torch.compile(forward, backend=backend, fullgraph=True)
    try:
        for count in (128, *buckets):
            for shift in (0, 5):
                positions = torch.arange(count, device="cpu", dtype=torch.int32)
                tokens_cpu = positions + shift
                tokens = tokens_cpu.to("tpu")
                torch._dynamo.mark_dynamic(tokens, 0)
                with override_forward_context(context):
                    actual = run(tokens, starts)
                sync.synchronize([actual], wait=True)
                # Requests have two tokens each; later starts are outside the
                # smaller buckets and must land in the discarded overflow slot.
                request, chunk_position = positions // 2, positions % 2
                ids = tokens_cpu + chunk_position + request
                expected = bits[ids].to(torch.int32).sum(-1) + request + chunk_position
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert len(compiled_pieces) == 2
    finally:
        torch._dynamo.reset()
        embedding.close()


@pytest.mark.tpu_test
@torch.inference_mode()
def test_native_host_lookup_through_vllm_bucket_compiler(tmp_path, monkeypatch):
    import dataclasses

    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        DeviceConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.forward_context import set_forward_context
    from vllm.platforms import current_platform

    import vllm_torchtpu
    from vllm_torchtpu.compilation import tpu_compiler
    from vllm_torchtpu.layers.adapter.custom_ops.ple_op import build_host_lookup_ops
    from vllm_torchtpu.models.vllm.qwen4_exp.ple_storage import HostPLETable

    initialized = torch.distributed.is_initialized()
    if not initialized:
        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{tmp_path}/dist", rank=0, world_size=1
        )
    try:
        rows = 128 * 8 + 64
        # Nonzero table contents must not leak through the reload marker.
        words = torch.arange(rows * 40, dtype=torch.int32) * 7919 - 777
        table = HostPLETable.__new__(HostPLETable)
        torch.nn.Module.__init__(table)
        table.storage_dtype = torch.float8_e4m3fn
        table.embedding_dim = 160
        table._shared_table = None
        table.weight = torch.nn.Parameter(
            words.view(torch.float8_e4m3fn).reshape(rows, 160), requires_grad=False
        )
        table.register_buffer("_vocab_start", torch.tensor(0, dtype=torch.int32))
        native = table.allocate_native_host(torch.device("tpu"))
        assert table._vocab_start.device.type == "tpu"
        assert table.weight.numel() == 0
        start, finish = build_host_lookup_ops(rows)
        monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
        monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")
        monkeypatch.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "0")
        # Only model admission and its hash need a real checkpoint config.
        # The entire compiler pipeline and native execution remain real.
        monkeypatch.setattr(
            tpu_compiler,
            "compute_tpu_compilation_hash",
            lambda config: "native-host-regression",
        )
        with monkeypatch.context() as admission:
            admission.setattr(
                type(current_platform),
                "check_and_update_config",
                classmethod(lambda cls, config: None),
            )
            config = VllmConfig(
                device_config=DeviceConfig(device="tpu"),
                scheduler_config=SchedulerConfig(
                    max_num_batched_tokens=8192,
                    max_model_len=8192,
                    is_encoder_decoder=False,
                    max_num_seqs=16,
                ),
                compilation_config=CompilationConfig(
                    mode=CompilationMode.VLLM_COMPILE,
                    backend="",
                    compile_sizes=[16, 8192],
                    splitting_ops=["qwen4_exp::host_lookup"],
                    cudagraph_capture_sizes=[],
                ),
            )
        for field in dataclasses.fields(config.compilation_config.pass_config):
            if isinstance(
                getattr(config.compilation_config.pass_config, field.name), bool
            ):
                setattr(config.compilation_config.pass_config, field.name, False)
        monkeypatch.setattr(
            type(config.compilation_config), "get_compile_ranges", lambda self: []
        )
        vllm_torchtpu._patch_vllm_aot_compile_cache_key()

        @support_torch_compile(dynamic_arg_dims={"ids": 0, "hidden": 0})
        class Model(torch.nn.Module):
            def __init__(self, *, vllm_config):
                super().__init__()
                self.prefix = "native_host_regression"
                self.register_buffer("table", native, persistent=False)

            def forward(self, ids, hidden):
                pending = start(self.table, ids)
                prefix = hidden * 2 + 1
                return finish(pending, ids, prefix)

        with set_current_vllm_config(config):
            compiled = Model(vllm_config=config)
        choices = torch.tensor(
            [-1, 0, 25, 51, 127, 128, rows - 1, rows], dtype=torch.int32
        )
        for tokens in (16, 8192, 16):
            cpu_ids = choices.repeat(tokens * 2).reshape(tokens, 16)
            ids = cpu_ids.to("tpu")
            hidden = torch.arange(tokens * 128, dtype=torch.float32).reshape(
                tokens, 128
            )
            hidden_tpu = hidden.to("tpu")
            torch._dynamo.mark_dynamic(ids, 0)
            torch._dynamo.mark_dynamic(hidden_tpu, 0)
            expected = torch.zeros((tokens, 16, 40), dtype=torch.int32)
            valid = (cpu_ids >= 0) & (cpu_ids < rows)
            expected[valid] = words.reshape(rows, 40)[cpu_ids[valid].long()]
            for _ in range(3):
                with set_forward_context({}, config):
                    raw, prefix = compiled(ids, hidden_tpu)
                torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(prefix.cpu(), hidden * 2 + 1, rtol=0, atol=0)
    finally:
        if not initialized:
            torch.distributed.destroy_process_group()
