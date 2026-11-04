# SPDX-License-Identifier: Apache-2.0
"""Dependency-free regression checks for the CSA dispatch envelope and A/B wiring."""

import ast
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "src/vllm_torchtpu/kernels/deepseek_v41/csa2/cache.py"
BASELINE = Path(__file__).with_name("_main_records_baseline.py")
BENCHMARK = Path(__file__).with_name("main_records_shapes.py")


def dispatcher(path):
    tree = ast.parse(path.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_main_records"
    )


def guard(function):
    branches = [node for node in function.body if isinstance(node, ast.If)]
    return compile(ast.Expression(branches[1].test), "<dispatch guard>", "eval")


class MainRecordsDispatchTest(unittest.TestCase):
    def test_exact_envelope(self):
        old = guard(dispatcher(BASELINE))
        new = guard(dispatcher(PRODUCTION))
        for backend, tokens, requests, width, columns, page_size in itertools.product(
            ("cpu", "tpu"),
            (95, 96, 97, 191, 192, 193, 287, 288, 289),
            (15, 16, 17, 31, 32, 33, 47, 48, 49),
            (1, 511, 512),
            (385, 386, 387, 771, 772, 773),
            (1, 32, 64, 128, 256),
        ):
            env = dict(
                jax=SimpleNamespace(default_backend=lambda backend=backend: backend),
                selected=SimpleNamespace(shape=(tokens, width)),
                physical_pages=SimpleNamespace(shape=(requests, columns)),
                page_size=page_size,
            )
            original = (
                backend == "tpu"
                and (tokens, requests, width) == (288, 48, 512)
                and columns in (386, 772)
            )
            expanded = (
                backend == "tpu"
                and width == 512
                and (tokens, requests) in ((96, 16), (192, 32), (288, 32))
                and (columns, page_size) in ((386, 128), (772, 64))
            )
            case = (backend, tokens, requests, width, columns, page_size)
            self.assertEqual(eval(old, env), original, case)
            self.assertEqual(eval(new, env), original or expanded, case)

    def test_only_guard_changed(self):
        old = dispatcher(BASELINE)
        new = dispatcher(PRODUCTION)
        for function in (old, new):
            branches = [node for node in function.body if isinstance(node, ast.If)]
            branches[1].test = ast.Constant(True)
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_benchmark_uses_distinct_production_dispatchers(self):
        tree = ast.parse(BENCHMARK.read_text())
        imports = {
            alias.asname or alias.name: (node.module, alias.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertEqual(
            imports["baseline_prepare_main_records"],
            ("_main_records_baseline", "prepare_main_records"),
        )
        self.assertEqual(
            imports["prepare_main_records"],
            (
                "vllm_torchtpu.kernels.deepseek_v41.csa2.cache",
                "prepare_main_records",
            ),
        )
        self.assertNotIn("main_records_call", imports)
        pairs = {
            (node.elts[0].value, node.elts[1].id)
            for node in ast.walk(tree)
            if isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and isinstance(node.elts[0], ast.Constant)
            and isinstance(node.elts[1], ast.Name)
        }
        self.assertIn(("baseline", "baseline_prepare_main_records"), pairs)
        self.assertIn(("candidate", "prepare_main_records"), pairs)


if __name__ == "__main__":
    unittest.main()
