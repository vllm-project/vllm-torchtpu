# SPDX-License-Identifier: Apache-2.0
"""CPU-only source guard regression tests; no accelerator correctness claim."""

import ast
import copy
from types import SimpleNamespace

import numpy as np
import state_packed_consumer as fixtures
import state_packed_shapes as metadata


def function(tree, name):
    return next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )


def cpu_test(frozen):
    decode = ast.parse(metadata.PREP.read_text())
    preprocess = ast.parse(metadata.DISPATCH.read_text())
    original_decode = ast.parse(frozen[metadata.PREP])
    original_preprocess = ast.parse(frozen[metadata.DISPATCH])
    packed = function(decode, "compressor_state_packed")
    run = function(function(preprocess, "compress_rope_packed"), "run")
    original_run = function(
        function(original_preprocess, "compress_rope_packed"), "run"
    )
    gate = next(n for n in run.body if isinstance(n, ast.If))
    index = run.body.index(gate)
    assert ast.dump(ast.Module(run.body[index + 1 :], [])) == ast.dump(
        ast.Module(original_run.body[1:], [])
    )
    # Beyond the two bounded guards/docstring, both modules must be unchanged.
    restored = copy.deepcopy(decode)
    target = function(restored, "compressor_state_packed")
    old = function(original_decode, "compressor_state_packed")
    target.body[:4] = copy.deepcopy(old.body[:4])
    assert ast.dump(restored) == ast.dump(original_decode)
    restored = copy.deepcopy(preprocess)
    target = function(function(restored, "compress_rope_packed"), "run")
    target.body[: index + 1] = copy.deepcopy(original_run.body[:1])
    assert ast.dump(restored) == ast.dump(original_preprocess)

    compact = function(packed, "compact")
    ns = {"jnp": np, "prefix_sum": np.cumsum, "total_records": fixtures.RECORDS}
    exec(
        compile(
            ast.Module([function(decode, "_compressor_state_addresses"), compact], []),
            str(metadata.PREP),
            "exec",
        ),
        ns,
    )
    cases = 0
    for tokens in metadata.TOKENS:
        assert metadata.consumer_config(tokens).token_block == tokens
        for kind in metadata.CASES:
            arrays = fixtures.fixture(tokens, kind)
            addresses, expected = fixtures.reference_metadata(arrays)
            actual = ns["_compressor_state_addresses"](*arrays)
            metadata.check(actual, addresses)
            wp, wo = ns["compact"](actual[2])
            rp, ro = ns["compact"](actual[3])
            metadata.check(
                (
                    actual[0][:, None],
                    actual[1][:, None],
                    wp.reshape(-1),
                    wo,
                    rp.reshape(-1),
                    ro,
                ),
                expected,
            )
            cases += 1

    # Execute the actual unmodified guard expressions, stopping before Pallas.
    guard_nodes = packed.body[1:4]
    guard_code = compile(ast.Module(guard_nodes, []), str(metadata.PREP), "exec")
    dispatch_code = compile(
        ast.Module(run.body[:index], []), str(metadata.DISPATCH), "exec"
    )
    checks = 0
    for tokens in (0, 1, 95, 96, 128, 192, 240, 256, 288, 320, 384, 512):
        arrays = [
            np.zeros(s, np.int32) for s in ((48, 8), (49,), (49,), (tokens,), (48,))
        ]
        args = dict(
            zip(
                ("table", "starts", "kv_starts", "positions", "minimum"),
                arrays,
                strict=True,
            )
        )
        args.update(jnp=np)
        try:
            exec(guard_code, args)
            accepted = True
        except ValueError:
            accepted = False
        assert accepted == (tokens in metadata.TOKENS)
        for hidden, width, tile in (
            (5120, 1024, tokens),
            (4096, 1024, tokens),
            (5120, 512, tokens),
            (5120, 1024, 16),
        ):
            args.update(
                x=SimpleNamespace(shape=(tokens, hidden)),
                weight=SimpleNamespace(shape=(hidden, width)),
                compress_kernel=SimpleNamespace(
                    select_config=lambda *_, tile=tile: SimpleNamespace(
                        token_block=tile
                    )
                ),
            )
            exec(dispatch_code, args)
            assert args["packed"] == (
                tokens == 288
                or (
                    tokens in metadata.TOKENS
                    and hidden == 5120
                    and width == 1024
                    and tile == tokens
                )
            )
            checks += 1
    for i in range(5):
        for bad in (
            np.zeros((1,), np.int32),
            np.zeros(((48, 8), (49,), (49,), (96,), (48,))[i], np.int64),
        ):
            arrays = [
                np.zeros(s, np.int32) for s in ((48, 8), (49,), (49,), (96,), (48,))
            ]
            arrays[i] = bad
            args = dict(
                zip(
                    ("table", "starts", "kv_starts", "positions", "minimum"),
                    arrays,
                    strict=True,
                )
            )
            args.update(
                jnp=np,
                x=SimpleNamespace(shape=(96, 5120)),
                weight=SimpleNamespace(shape=(5120, 1024)),
                compress_kernel=SimpleNamespace(
                    select_config=lambda *_: SimpleNamespace(token_block=96)
                ),
            )
            try:
                exec(guard_code, args)
            except ValueError:
                pass
            else:
                raise AssertionError("Malformed metadata accepted")
            exec(dispatch_code, args)
            assert not args["packed"]
            checks += 1
    factory = function(preprocess, "compress_rope_packed")
    runtime_gate = next(n for n in factory.body if isinstance(n, ast.If))
    for backend, version in (("cpu", 7), ("tpu", 6), ("tpu", 7)):
        rejected = eval(
            compile(ast.Expression(runtime_gate.test), "runtime_gate", "eval"),
            {
                "jax": SimpleNamespace(default_backend=lambda backend=backend: backend),
                "get_tpu_version": lambda version=version: version,
            },
        )
        assert rejected == (backend != "tpu" or version != 7)
    ns = {}
    exec(
        compile(
            ast.Module([function(decode, "supports_compressor_state_packed")], []),
            str(metadata.PREP),
            "exec",
        ),
        ns,
    )
    support = ns["supports_compressor_state_packed"]
    geometry = [288, (48, 8), 16, 16, 0, 128]
    assert support(*geometry)
    for i, bad in enumerate((96, (48, 9), 32, 32, 1, 256)):
        changed = geometry.copy()
        changed[i] = bad
        assert not support(*changed)
    caller = function(
        ast.parse(metadata.ATTENTION.read_text()), "_write_csa2_global_kv"
    )
    support_call = next(
        n
        for n in ast.walk(caller)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "supports_compressor_state_packed"
    )
    assert ast.literal_eval(support_call.args[0]) == 288
    return {
        "caller_geometry_and_support_constant_preserved": True,
        "guards_checked": checks,
        "fixtures_exact": cases,
        "only_bounded_guards_changed": True,
        "t288_body_unchanged": True,
        "combined_addresses_unchanged": True,
        "runtime7_only": True,
    }
