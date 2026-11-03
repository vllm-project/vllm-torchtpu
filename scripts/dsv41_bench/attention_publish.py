# SPDX-License-Identifier: Apache-2.0
"""Actual caller, compressor, SC publish and packed index writer A/B.

The index projection/norm/RoPE cut is supplied with independent BF16 D128 keys.
All address, masking, bridge, donation and write paths use production bytecode.
"""

import argparse
import ast
import functools
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from attention_addresses import expected as address_reference
from attention_consumer import (
    INITIAL,
    ROWS,
    WORD,
    dense_reference,
    encode_reference_cpu,
    fixture,
)
from attention_integration import production_ops
from attention_state import CACHE, ROOT, SOURCE, Metadata, reference

BASE = SOURCE.parent


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", "") in names]
    assert len(nodes) == len(names), (path, names)
    tree = ast.Module(
        [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *nodes], []
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


def prepare(
    ops,
    mesh,
    banks,
    scale_banks,
    *,
    capture_only=False,
    fallback=False,
    writer_stage="both",
    indexer_prepare_compare=False,
):
    import torch

    from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import publish_csa2_kv
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
        PackedFP8Layout,
        write_fp8_records,
    )
    from vllm_torchtpu.kernels.experimental.decode_preparation import (
        supports_compressor_state_packed,
    )

    ns = dict(
        torch=torch,
        jax=jax,
        jnp=jnp,
        math=math,
        functools=functools,
        Any=Any,
        cast=cast,
        ROW_WORDS=128,
        ROW_BYTES=512,
        PackedFP8Layout=PackedFP8Layout,
        write_fp8_records=write_fp8_records,
        publish_csa2_kv=publish_csa2_kv,
    )
    cache_tree = ast.parse(CACHE.read_text())
    cls = next(n for n in cache_tree.body if getattr(n, "name", "") == "PagedCacheRows")
    cls.body = [
        n for n in cls.body if getattr(n, "name", "") in ("__init__", "row_indices")
    ]
    exec(compile(ast.Module([cls], []), str(CACHE), "exec"), ns)
    extract(BASE / "indexer_cache.py", ["PackedFP8IndexCacheRows"], ns)
    extract(CACHE, ["write_cache"], ns)
    extract(BASE / "indexer_select.py", ["_register"], ns)
    extract(BASE / "indexer_fp8.py", ["_write"], ns)
    write_op = ns["_register"](
        "pallas::dsv41_packed_fp8_write",
        ns["_write"],
        lambda banks, records, keys, after, layout: tuple(
            torch.empty_like(b) for b in banks
        ),
        4,
        1,
        (0,),
    )
    ns["indexer_fp8"] = SimpleNamespace(_write_op=write_op)
    candidate_index = None
    if indexer_prepare_compare:
        from indexer_prepare import write_fp8_records as candidate_write

        candidate_ns = dict(ns, write_fp8_records=candidate_write)
        extract(BASE / "indexer_fp8.py", ["_write"], candidate_ns)
        candidate_op = ns["_register"](
            "pallas::dsv41_indexer_sorted_prepare_write",
            candidate_ns["_write"],
            lambda banks, records, keys, after, layout: tuple(
                torch.empty_like(b) for b in banks
            ),
            4,
            1,
            (0,),
        )
        candidate_ns["indexer_fp8"] = SimpleNamespace(_write_op=candidate_op)
        extract(BASE / "indexer_cache.py", ["PackedFP8IndexCacheRows"], candidate_ns)
        # Preserve the actual caller's isinstance guard. Only the unchanged
        # write method uses a namespace pointing at the experimental bridge.
        candidate_index = type(
            "CandidatePackedFP8IndexCacheRows",
            (ns["PackedFP8IndexCacheRows"],),
            {"write": candidate_ns["PackedFP8IndexCacheRows"].write},
        )
    extract(
        BASE / "csa2_adapter.py",
        ["_publish_main_banked", "_manual_bridge", "publish_main"],
        ns,
    )
    from jax.sharding import PartitionSpec as P

    ns["_ops"] = {
        "publish_main_banked": ns["_register"](
            "pallas::dsv41v2_csa2_publish_main_banked",
            ns["_manual_bridge"](ns["_publish_main_banked"], mesh, 7, (P(),) * 4),
            lambda r, h, s, hs, *_: tuple(torch.empty_like(x) for x in (r, h, s, hs)),
            7,
            0,
            (0, 1, 2, 3),
        )
    }
    Paged = ns["PagedCacheRows"]
    Index = ns["PackedFP8IndexCacheRows"]
    ctx = SimpleNamespace(attn_metadata={}, additional_kwargs={})
    publish = ns["publish_main"]

    class Publisher:
        @staticmethod
        def publish_main(cache, records, words, scales):
            cache.result = (words, scales, records)
            if not capture_only and writer_stage != "index":
                publish(cache, records, words, scales)

    ns.update(
        get_forward_context=lambda: ctx,
        preprocess_ops=ops,
        csa2_adapter=Publisher,
        supports_compressor_state_packed=supports_compressor_state_packed,
    )
    extract(
        SOURCE,
        [
            "_block_table",
            "_token_requests",
            "_cache_slots",
            "_window_starts",
            "_replayed",
            "_insert",
        ],
        ns,
    )
    method = next(
        n
        for n in ast.walk(ast.parse(SOURCE.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_write_csa2_global_kv"
    )
    exec(compile(ast.Module([method], []), str(SOURCE), "exec"), ns)
    pages = sum(ROWS) // 128
    write = ns["_write_csa2_global_kv"]
    packed_op = ops.compress_rope_packed(1e-6)
    address_op = ops.compress_rope_addresses(1e-6)
    assert packed_op is not None and address_op is not None

    class Model:
        _group_positions = staticmethod(lambda p: p // 2 * 2)
        _write_csa2_global_kv = write

        def __init__(self, expanded):
            self.prefix, self.compress_ratio, self.window_size = "main", 2, 128
            self._packed_compressor_op = packed_op
            self._address_compressor_op = (
                address_op if expanded or indexer_prepare_compare else None
            )
            self.kv_cache = Paged(
                banks, ((pages, 128, 512), torch.uint8, 65536, 0), scale=scale_banks
            )
            self.indexer = SimpleNamespace(
                k_cache=SimpleNamespace(
                    prefix="index",
                    compress_ratio=2,
                    kv_cache=(
                        candidate_index
                        if indexer_prepare_compare and expanded
                        else Index
                    )(banks, ((pages, 256, 132), torch.uint8, 65536, 0)),
                )
            )
            self.compressor = SimpleNamespace(
                state_cache=SimpleNamespace(
                    prefix="state",
                    kv_cache=Paged(banks, ((pages, 16, 1024), torch.float32, 65536, 0)),
                ),
                fused_wkv_wgate=SimpleNamespace(weight=None),
                norm=SimpleNamespace(weight=None),
                rms_norm_eps=1e-6,
            )
            self.rope = self.keys = self.result = None

        def _rope_rows(self, positions, ratio):
            return self.rope

        def _produce_index_k(self, latent, group_positions, cache, slots, valid):
            if not capture_only and writer_stage != "main":
                ns["_insert"](cache, slots, self.keys, valid)
            words, scales, records = self.kv_cache.result
            # Only latent crosses the supplied index-projection cut. Captured
            # metadata must not escape into the timed writer executable.
            self.result = (
                (latent, words, scales, valid, records.int(), slots.int())
                if capture_only
                else (latent,)
            )
            self.kv_cache.result = None

    def make(expanded):
        model = Model(expanded)

        def run(
            table,
            starts,
            positions,
            minimum,
            x,
            weight,
            gamma,
            rope,
            a,
            b,
            mt,
            it,
            sa,
            sb,
            keys,
        ):
            index_starts = starts.clone() if fallback else starts
            ctx.attn_metadata = {
                "state": Metadata(table, minimum, starts),
                "main": Metadata(mt, minimum, starts),
                "index": Metadata(it, minimum, index_starts),
            }
            ctx.additional_kwargs = {"deepseek_v41_window_starts": minimum}
            for cache in (
                model.kv_cache,
                model.indexer.k_cache.kv_cache,
                model.compressor.state_cache.kv_cache,
            ):
                cache.raw, cache.raw_hi = a, b
            model.kv_cache.scale, model.kv_cache.scale_hi = sa, sb
            model.compressor.fused_wkv_wgate.weight = weight
            model.compressor.norm.weight = gamma
            model.rope, model.keys = rope, keys
            model._write_csa2_global_kv(x, positions)
            return model.result

        return run

    return make(False), make(True)


def inputs(kind):
    table, starts, pos, minimum = fixture(kind)
    # Shared physical pool; none of the three layouts overlap.
    state_pages = np.r_[np.arange(4000, 4024), np.arange(300000, 300024)]
    table[:] = state_pages[:, None]
    main_pages = np.r_[0, 261119, 261120, 484965, np.arange(2000, 2044)]
    index_pages = np.r_[1, 261118, 261121, 484964, np.arange(3000, 3044)]
    main = np.repeat(main_pages.astype(np.int32)[:, None], 386, 1)
    index = np.repeat(index_pages.astype(np.int32)[:, None], 193, 1)
    if kind == "negative":
        main[::3] = -1
        index[::3] = -1
    if kind == "page_columns":
        # Exercise every 128-column table tile without exceeding table capacity.
        for request in range(48):
            begin, end = starts[request : request + 2]
            pos[begin:end] = (request % 4) * 32768 + np.arange(end - begin)
    return table, starts, pos, minimum, main, index


def expected_updates(
    score,
    write,
    words,
    scales,
    records,
    index_slots,
    valid,
    keys,
    capture_only,
    writer_stage,
):
    write = write.astype(np.int64)
    records = records.astype(np.int64)
    row_ids = np.flatnonzero(write >= 0)
    addresses = [(write[row_ids, None] * 1024 + np.arange(1024)).reshape(-1)]
    values = [score[row_ids].view(np.int32).reshape(-1)]
    selected = np.flatnonzero(records >= 0)
    scale_ids, scale_values = records[selected], scales[selected, 0]
    if not capture_only and writer_stage != "index":
        addresses.append((records[selected, None] * 128 + np.arange(128)).reshape(-1))
        values.append(words[selected].reshape(-1))
    else:
        scale_ids, scale_values = np.empty(0, np.int64), np.empty(0, np.float32)
    if not capture_only and writer_stage != "main":
        selected = np.flatnonzero(valid & (index_slots >= 0))
        key_values = keys[selected].astype(np.float32)
        exponent = np.ceil(
            np.log2(
                np.maximum(np.abs(key_values).max(1), np.float32(1e-4))
                / np.float32(448)
            )
        ).astype(np.int32)
        scale = np.exp2(exponent).astype(np.float32)
        quantized = (key_values / scale[:, None]).astype(ml_dtypes.float8_e4m3fn)
        bits = quantized.view(np.uint8).reshape(-1, 32, 4).astype(np.uint32)
        packed = (
            bits[:, :, 0]
            | bits[:, :, 1] << 8
            | bits[:, :, 2] << 16
            | bits[:, :, 3] << 24
        )
        encoded = np.c_[packed.view(np.int32), scale.view(np.int32)]
        logical = index_slots[selected].astype(np.int64)
        addresses.append(
            (
                logical[:, None] // 256 * (128 * 128)
                + np.arange(33) * 256
                + logical[:, None] % 256
            ).reshape(-1)
        )
        values.append(encoded.reshape(-1))
    addresses, values = np.concatenate(addresses), np.concatenate(values)
    assert np.unique(addresses).size == addresses.size
    assert addresses.min(initial=0) >= 0 and addresses.max(initial=0) < sum(ROWS) * 128
    return addresses, values, scale_ids, scale_values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--capture-only", action="store_true")
    p.add_argument("--fallback", action="store_true")
    p.add_argument("--indexer-prepare-compare", action="store_true")
    p.add_argument("--writer-stage", choices=("both", "main", "index"), default="both")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import importlib.metadata

    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import sync

    torch.set_num_threads(1)
    torch.empty(1, device="tpu").cpu()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    ops = production_ops(mesh)
    banks = tuple(
        torch.full((rows, 128), WORD, dtype=torch.int32, device="tpu") for rows in ROWS
    )
    scale_banks = tuple(
        torch.full((rows,), INITIAL.item(), dtype=torch.float32, device="tpu")
        for rows in ROWS
    )
    funcs = prepare(
        ops,
        mesh,
        banks,
        scale_banks,
        capture_only=args.capture_only,
        fallback=args.fallback,
        writer_stage=args.writer_stage,
        indexer_prepare_compare=args.indexer_prepare_compare,
    )
    functions = dict(
        zip(
            ("retained", "expanded"),
            [
                torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
                for f in funcs
            ],
            strict=True,
        )
    )
    rng = np.random.default_rng(20261104)
    host = (
        torch.tensor(rng.normal(size=(288, 5120)) * 0.5, dtype=torch.bfloat16),
        torch.tensor(rng.normal(size=(5120, 1024)) * 0.02, dtype=torch.bfloat16),
        torch.tensor(rng.normal(size=(512,)), dtype=torch.bfloat16),
    )
    angles = rng.uniform(-3, 3, (288, 32)).astype(np.float32)
    rope = np.c_[np.cos(angles), np.sin(angles)]
    keys = torch.tensor(rng.normal(size=(288, 128)), dtype=torch.bfloat16)
    compute = tuple(x.to("tpu") for x in host) + (torch.from_numpy(rope).to("tpu"),)
    keys_device = keys.to("tpu")
    ref_compute = tuple(jnp.asarray(x.float().numpy(), jnp.bfloat16) for x in host)
    ref = dense_reference()
    result = {
        "scope": (
            "production caller and compressor; original SC main publish and TC "
            "index writer; index projection/norm/RoPE supplied by "
            "BF16 key input"
        ),
        "capture_only": args.capture_only,
        "fallback": args.fallback,
        "indexer_prepare_compare": args.indexer_prepare_compare,
        "writer_stage": args.writer_stage,
        "runtime": {
            n: importlib.metadata.version(n)
            for n in ("torch", "torch-tpu", "jax", "jaxlib", "libtpu", "numpy")
        },
        "geometry": {
            "raw_bank_rows": ROWS,
            "pages": sum(ROWS) // 128,
            "state": [16, 16, 0],
            "main": [128, 128, 0],
            "index": [256, 128, 0],
        },
        "source_sha256": {
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                Path(__file__),
                SOURCE,
                CACHE,
                BASE / "indexer_cache.py",
                BASE / "indexer_fp8.py",
                BASE / "indexer_select.py",
                BASE / "csa2_adapter.py",
                BASE / "preprocess_ops.py",
                ROOT / "src/vllm_torchtpu/kernels/experimental/decode_preparation.py",
            ]
        },
        "correctness": [],
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    def reset():
        for b in banks:
            b.fill_(WORD)
        for s in scale_banks:
            s.fill_(float(INITIAL))
        sync.synchronize(tensors=list(banks + scale_banks), wait=True)

    def make_values(kind):
        data = inputs(kind)
        tensors = tuple(torch.from_numpy(x).to("tpu") for x in data)
        return data, tensors[:4] + compute + banks + tensors[4:] + scale_banks + (
            keys_device,
        )

    save()
    for kind in (
        "decode",
        "padding",
        "long_request",
        "replay",
        "empty",
        "negative",
        "page_columns",
    ):
        data, values = make_values(kind)
        table, starts, pos, minimum, mt, it = data
        prev, valid, write, read = reference(
            (table, starts, starts, pos, minimum), 16, 16, 0
        )
        previous = np.zeros((288, 1024), np.float32)
        previous[read >= 0] = INITIAL
        score, latent, _, _ = ref(
            *ref_compute,
            jnp.asarray(prev[:, 0]),
            jnp.asarray(previous),
            jnp.asarray(valid[:, 0]),
            jnp.asarray(rope[:, :32]),
            jnp.asarray(rope[:, 32:]),
        )
        latent = np.asarray(latent, np.float32)
        words, scales = encode_reference_cpu(latent, rope, valid)
        records, slots = address_reference(
            (table, starts, starts, pos, minimum, mt, it)
        )[6:]
        expected = (
            (latent, words, scales, valid[:, 0] != 0, records, slots)
            if args.capture_only
            else (latent,)
        )
        ids, wanted, scale_ids, scale_wanted = expected_updates(
            np.asarray(score),
            write,
            words,
            scales,
            records,
            slots,
            valid[:, 0] != 0,
            keys.float().numpy(),
            args.capture_only,
            args.writer_stage,
        )
        checks = []
        for name, fn in functions.items():
            reset()
            outputs = fn(*values)
            sync.synchronize(
                tensors=list(outputs) + list(banks + scale_banks), wait=True
            )
            for i, (out, want) in enumerate(zip(outputs, expected, strict=True)):
                got = out.float().cpu().numpy() if i == 0 else out.cpu().numpy()
                np.testing.assert_array_equal(
                    got, want, err_msg=f"{kind}/{name}/output{i}"
                )
            cut, changed, scale_changed = 0, [], []
            for bank, scale_bank, rows in zip(banks, scale_banks, ROWS, strict=True):
                selection = (ids >= cut * 128) & (ids < (cut + rows) * 128)
                local = ids[selection] - cut * 128
                unique_rows, inverse = np.unique(local // 128, return_inverse=True)
                row_ids = torch.from_numpy(unique_rows).to("tpu")
                readback = bank.index_select(0, row_ids).cpu().numpy()
                got = readback[inverse, local % 128]
                bad = np.flatnonzero(got != wanted[selection])
                if bad.size:
                    details = {
                        "case": kind,
                        "path": name,
                        "bank_cut_rows": cut,
                        "word_ids": ids[selection][bad[:32]].tolist(),
                        "actual": got[bad[:32]].tolist(),
                        "expected": wanted[selection][bad[:32]].tolist(),
                    }
                    details["direct_first_row"] = bank[0].cpu().numpy().tolist()
                    print("BANK_MISMATCH", details, flush=True)
                    result["bank_failure"] = details
                    save()
                np.testing.assert_array_equal(
                    got, wanted[selection], err_msg=f"{kind}/{name}/bank"
                )
                total = int(torch.count_nonzero(bank != WORD).cpu())
                assert total == int(np.count_nonzero(wanted[selection] != WORD)), (
                    kind,
                    name,
                    total,
                )
                selected = (scale_ids >= cut) & (scale_ids < cut + rows)
                si = torch.from_numpy((scale_ids[selected] - cut).astype(np.int64)).to(
                    "tpu"
                )
                np.testing.assert_array_equal(
                    scale_bank.index_select(0, si).cpu().numpy(), scale_wanted[selected]
                )
                sc = int(torch.count_nonzero(scale_bank != float(INITIAL)).cpu())
                assert sc == int(np.count_nonzero(scale_wanted[selected] != INITIAL))
                changed.append(total)
                scale_changed.append(sc)
                cut += rows
            checks.append(
                {
                    "path": name,
                    "outputs_exact": len(expected),
                    "all_bank_updates_exact": True,
                    "changed_words": changed,
                    "changed_scales": scale_changed,
                }
            )
            print(kind, checks[-1], flush=True)
        result["correctness"].append({"case": kind, "paths": checks})
        save()

    data, values = make_values("decode")
    prev, valid, write, read = reference(
        (data[0], data[1], data[1], data[2], data[3]), 16, 16, 0
    )
    assert not np.intersect1d(write[write >= 0], read[read >= 0]).size
    reset()

    def run(fn):
        outputs = fn(*values)
        sync.synchronize(tensors=list(outputs) + list(banks + scale_banks), wait=True)

    for _ in range(20):
        for fn in functions.values():
            run(fn)
    samples = {name: [] for name in functions}
    for i in range(args.samples):
        order = list(functions.items())
        for name, fn in order if i % 2 == 0 else order[::-1]:
            start = time.perf_counter_ns()
            run(fn)
            samples[name].append((time.perf_counter_ns() - start) / 1000)
    result["synchronous_us"] = {
        name: {
            "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)),
            "samples": x,
        }
        for name, x in samples.items()
    }
    save()
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile")):
            for i in range(16):
                for name, fn in functions.items():
                    with jax.profiler.TraceAnnotation(name, step_num=i):
                        run(fn)
    result["passed"] = True
    save()
    print({n: v["median"] for n, v in result["synchronous_us"].items()}, flush=True)


if __name__ == "__main__":
    main()
