# SPDX-License-Identifier: Apache-2.0
"""Load production bridges without model initialization in a kernel-only runner.

Only package startup and the mesh provider are supplied by the harness. Kernel
and bridge modules are imported unchanged. Caller tests extract the unchanged
method and helpers, with publish/index compute replaced by live output captures.
"""

import ast
import importlib
import sys
import types
from types import SimpleNamespace

from attention_state import CACHE, ROOT, SOURCE, Metadata


def production_ops(mesh):
    prefixes = (
        "vllm_torchtpu",
        "vllm_torchtpu.models",
        "vllm_torchtpu.models.vllm",
        "vllm_torchtpu.models.vllm.deepseek_v41",
        "vllm_torchtpu.models.vllm.deepseek_v41.v2",
    )
    for name in prefixes:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / "src" / name.replace(".", "/"))]
            sys.modules[name] = module
    name = "vllm_torchtpu.models.vllm.vllm_model_wrapper_context"
    context = types.ModuleType(name)
    context.get_pallas_op_mesh = lambda: mesh
    sys.modules[name] = context
    return importlib.import_module(
        "vllm_torchtpu.models.vllm.deepseek_v41.v2.preprocess_ops"
    )


def bridge_candidate(ops):
    op = ops.compress_rope_packed(1e-6)
    assert op is not None

    def run(table, starts, positions, minimum, x, weight, gamma, rope, a, b):
        latent, words, scales, low, high, valid = op(
            table, starts, starts, positions, minimum, x, weight, gamma, rope, a, b
        )
        a.copy_(low)
        b.copy_(high)
        return latent, words, scales, valid

    return run


def caller_functions(ops, state_capacity=16):
    import torch

    from vllm_torchtpu.kernels.experimental.decode_preparation import (
        supports_compressor_state_packed,
    )

    # Use one actual method for both paths; None selects the native fallback.
    tree = ast.parse(SOURCE.read_text())
    methods = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_write_csa2_global_kv"
    ]
    assert len(methods) == 1
    helpers = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in (
            "_block_table",
            "_token_requests",
            "_cache_slots",
            "_window_starts",
            "_replayed",
        )
    ]
    cache_tree = ast.parse(CACHE.read_text())
    cache_class = next(
        n
        for n in cache_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PagedCacheRows"
    )
    row_indices = next(
        n
        for n in cache_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "row_indices"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module([row_indices], []), str(CACHE), "exec"), namespace)

    class Cache:
        row_indices = namespace["row_indices"]

        def __init__(self, capacity):
            self.shape = (1, capacity)
            self.page_stride = capacity
            self.row_offset = 0
            self.raw = self.raw_hi = None
            self.result = None

    class Capture:
        @staticmethod
        def publish_main(cache, records, words, scales):
            cache.result = (words, scales, records)

    ctx = SimpleNamespace(attn_metadata={}, additional_kwargs={})
    namespace.update(
        {
            "get_forward_context": lambda: ctx,
            "preprocess_ops": ops,
            "PagedCacheRows": Cache,
            "csa2_adapter": Capture,
            "supports_compressor_state_packed": supports_compressor_state_packed,
        }
    )
    exec(compile(ast.Module(helpers + methods, []), str(SOURCE), "exec"), namespace)
    write = namespace["_write_csa2_global_kv"]
    native_op = ops.compress_rope(2, 2, 1e-6)
    packed_op = ops.compress_rope_packed(1e-6)
    assert native_op is not None and packed_op is not None

    class Model:
        _group_positions = staticmethod(lambda p: p // 2 * 2)
        _write_csa2_global_kv = write

        def __init__(self, op):
            self.prefix = "main"
            self.compress_ratio = 2
            self.window_size = 128
            self._packed_compressor_op = op
            self.kv_cache = Cache(64)
            self.indexer = SimpleNamespace(
                k_cache=SimpleNamespace(
                    prefix="index", kv_cache=Cache(64), compress_ratio=2
                )
            )
            self.compressor = SimpleNamespace(
                state_cache=SimpleNamespace(
                    prefix="state", kv_cache=Cache(state_capacity)
                ),
                fused_wkv_wgate=SimpleNamespace(weight=None),
                norm=SimpleNamespace(weight=None),
                rms_norm_eps=1e-6,
            )
            self.rope = None
            self.result = None

        def _rope_rows(self, group_positions, ratio):
            return self.rope

        def _produce_index_k(self, latent, group_positions, cache, slots, valid):
            words, scales, records = self.kv_cache.result
            self.result = latent, words, scales, valid, records, slots

    def make(op):
        model = Model(op)

        def run(table, starts, positions, minimum, x, weight, gamma, rope, a, b):
            # Main/index table uses page zero for all pages: the publish/index
            # cuts are captured, not executed. State retains actual unique pages.
            output_table = torch.zeros_like(table)
            ctx.attn_metadata = {
                "state": Metadata(table, minimum, starts),
                "main": Metadata(output_table, minimum, starts),
                "index": Metadata(output_table, minimum, starts),
            }
            ctx.additional_kwargs = {"deepseek_v41_window_starts": minimum}
            model.compressor.state_cache.kv_cache.raw = a
            model.compressor.state_cache.kv_cache.raw_hi = b
            model.compressor.fused_wkv_wgate.weight = weight
            model.compressor.norm.weight = gamma
            model.rope = rope
            model._write_csa2_global_kv(x, positions)
            return model.result

        return run

    return make(None), make(packed_op)
