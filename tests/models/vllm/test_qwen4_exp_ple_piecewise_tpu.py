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
        split, pieces = split_graph(graph, ["qwen4_exp::host_lookup"])
        assert sum(piece.is_splitting_graph for piece in pieces) == 1
        for piece in pieces:
            if piece.is_splitting_graph:
                continue
            output = next(n for n in piece.graph.graph.nodes if n.op == "output")
            assert all(
                isinstance(n.meta["example_value"], torch.Tensor)
                for n in output.args[0]
            )
            split.set_submodule(piece.submod_name, Piece(piece.graph))
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
