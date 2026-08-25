# SPDX-License-Identifier: Apache-2.0
"""Correctness and operator latency A/B for the chunked and mega K3 KDA kernels.

This script intentionally imports only the JAX kernel files, avoiding the full
vLLM serving stack in a standalone operator benchmark. A ``segment`` here is
one independent packed request with its own recurrent state; it is not a
parallel chunk of one request. Chunks within a segment remain recurrence-ordered.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DIM = 128
LOWER_BOUND = -5.0
SCALE = DIM**-0.5


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_kernels(repo_root: Path):
    src = repo_root / "src"
    package_paths = {
        "vllm_torchtpu":
        src / "vllm_torchtpu",
        "vllm_torchtpu.kernels":
        src / "vllm_torchtpu" / "kernels",
        "vllm_torchtpu.kernels.kimi_k3":
        (src / "vllm_torchtpu" / "kernels" / "kimi_k3"),
        "vllm_torchtpu.kernels.kimi_k3.mega_kda":
        (src / "vllm_torchtpu" / "kernels" / "kimi_k3" / "mega_kda"),
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    kernels_dir = src / "vllm_torchtpu" / "kernels"
    kimi_dir = kernels_dir / "kimi_k3"
    varlen = _load_module(
        "vllm_torchtpu.kernels.varlen_tiles",
        kernels_dir / "varlen_tiles.py",
    )
    sys.modules["vllm_torchtpu.kernels"].varlen_tiles = varlen
    chunked_module = _load_module(
        "vllm_torchtpu.kernels.kimi_k3.chunk_kda",
        kimi_dir / "chunk_kda.py",
    )
    mega_dir = kimi_dir / "mega_kda"
    _load_module(
        "vllm_torchtpu.kernels.kimi_k3.mega_kda.kernel",
        mega_dir / "kernel.py",
    )
    mega_module = _load_module(
        "vllm_torchtpu.kernels.kimi_k3.mega_kda",
        mega_dir / "__init__.py",
    )
    return (chunked_module.chunk_kda, mega_module.kda_forward_inference,
            mega_module._layout_supported)


def _segment_ids(lengths: tuple[int, ...], tokens: int) -> np.ndarray:
    ids = np.zeros((1, tokens), dtype=np.int32)
    offset = 0
    for segment, length in enumerate(lengths, start=1):
        ids[0, offset:offset + length] = segment
        offset += length
    if offset > tokens:
        raise ValueError(f"segment lengths {lengths} exceed T={tokens}")
    return ids


def _query_start_loc(lengths: tuple[int, ...],
                     num_segments: int) -> np.ndarray:
    starts = [0, *np.cumsum(lengths, dtype=np.int32).tolist()]
    starts.extend([starts[-1]] * (num_segments + 1 - len(starts)))
    return np.asarray(starts, dtype=np.int32)


def _inputs(*, heads: int, tokens: int, num_segments: int,
            seed: int) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    shape = (1, tokens, heads, DIM)

    def bf16_normal(scale: float = 1.0):
        value = rng.standard_normal(shape, dtype=np.float32) * scale
        return jnp.asarray(value, dtype=jnp.bfloat16)

    q = jax.nn.silu(bf16_normal().astype(jnp.float32)).astype(jnp.bfloat16)
    k = jax.nn.silu(bf16_normal().astype(jnp.float32)).astype(jnp.bfloat16)
    beta = jnp.asarray(
        rng.uniform(0.05, 0.95, (1, tokens, heads)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    return {
        "q":
        q,
        "k":
        k,
        "v":
        bf16_normal(1.5),
        "g":
        jnp.asarray(
            rng.uniform(-4.5, 4.5, shape).astype(np.float32),
            dtype=jnp.bfloat16,
        ),
        "beta":
        beta,
        "a_log":
        jnp.asarray(rng.uniform(0.2, 3.0, (heads, )).astype(np.float32)),
        "dt_bias":
        jnp.asarray(
            rng.uniform(-8.0, -1.5, (heads * DIM, )).astype(np.float32)),
        "initial_state":
        jnp.asarray(
            rng.standard_normal(
                (1, num_segments, heads, DIM, DIM), dtype=np.float32) * 0.1),
    }


def _block(result) -> None:
    for leaf in jax.tree.leaves(result):
        if leaf is not None:
            leaf.block_until_ready()


def _make_runners(chunk_kda, kda_forward_inference, layout_supported, *,
                  num_segments: int):

    def chunked_impl(q, k, v, g, beta, segment_ids, a_log, dt_bias,
                     initial_state):

        def head_first(x):
            return x.transpose(2, 0, 1, 3)

        output, final_state = chunk_kda(
            head_first(q),
            head_first(k),
            head_first(v),
            head_first(g),
            beta.transpose(2, 0, 1),
            A_log=a_log,
            dt_bias=dt_bias,
            scale=SCALE,
            initial_state=initial_state,
            output_final_state=True,
            use_gate_in_kernel=True,
            segment_ids=segment_ids,
            lower_bound=LOWER_BOUND,
            chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            N_max=num_segments,
        )
        return output.transpose(1, 2, 0, 3), final_state

    def mega_impl(q, k, v, g, beta, segment_ids, a_log, dt_bias,
                  initial_state):
        return kda_forward_inference(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=segment_ids,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=SCALE,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            chunk_size=64,
            N_max=num_segments,
        )

    @jax.jit
    def chunked(q, k, v, g, beta, segment_ids, query_start_loc, a_log, dt_bias,
                initial_state):
        del query_start_loc
        return chunked_impl(q, k, v, g, beta, segment_ids, a_log, dt_bias,
                            initial_state)

    @jax.jit
    def mega(q, k, v, g, beta, segment_ids, query_start_loc, a_log, dt_bias,
             initial_state):
        del query_start_loc
        return mega_impl(q, k, v, g, beta, segment_ids, a_log, dt_bias,
                         initial_state)

    @jax.jit
    def guarded(q, k, v, g, beta, segment_ids, query_start_loc, a_log, dt_bias,
                initial_state):
        operands = (q, k, v, g, beta, segment_ids, a_log, dt_bias,
                    initial_state)
        return jax.lax.cond(
            layout_supported(query_start_loc, q.shape[1]),
            lambda args: mega_impl(*args),
            lambda args: chunked_impl(*args),
            operands,
        )

    return chunked, mega, guarded


def _call(runner, arrays: dict[str, jax.Array], segment_ids: jax.Array,
          query_start_loc: jax.Array):
    return runner(
        arrays["q"],
        arrays["k"],
        arrays["v"],
        arrays["g"],
        arrays["beta"],
        segment_ids,
        query_start_loc,
        arrays["a_log"],
        arrays["dt_bias"],
        arrays["initial_state"],
    )


def _latencies(
    runner,
    arrays: dict[str, jax.Array],
    segment_ids: jax.Array,
    query_start_loc: jax.Array,
    *,
    warmups: int,
    iterations: int,
) -> tuple[float, list[float]]:
    start = time.perf_counter()
    first = _call(runner, arrays, segment_ids, query_start_loc)
    _block(first)
    first_ms = (time.perf_counter() - start) * 1e3
    for _ in range(warmups - 1):
        result = _call(runner, arrays, segment_ids, query_start_loc)
        _block(result)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = _call(runner, arrays, segment_ids, query_start_loc)
        _block(result)
        samples.append((time.perf_counter() - start) * 1e3)
    return first_ms, samples


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "mean_e2e_ms": statistics.fmean(samples),
        "median_e2e_ms": statistics.median(samples),
        "p90_e2e_ms": ordered[p90_index],
        "min_e2e_ms": min(samples),
        "max_e2e_ms": max(samples),
        "stddev_e2e_ms": statistics.pstdev(samples),
    }


def _compare(chunked_result, mega_result, *,
             real_segments: int) -> dict[str, float | bool]:
    chunked_output, chunked_state = jax.device_get(chunked_result)
    mega_output, mega_state = jax.device_get(mega_result)
    chunked_output = np.asarray(chunked_output, dtype=np.float32)
    mega_output = np.asarray(mega_output, dtype=np.float32)
    chunked_state = np.asarray(chunked_state[:, :real_segments],
                               dtype=np.float32)
    mega_state = np.asarray(mega_state[:, :real_segments], dtype=np.float32)
    output_delta = np.abs(chunked_output - mega_output)
    state_delta = np.abs(chunked_state - mega_state)
    return {
        "finite":
        bool(np.isfinite(mega_output).all() and np.isfinite(mega_state).all()),
        "output_close":
        bool(np.allclose(chunked_output, mega_output, rtol=0.05, atol=0.05)),
        "state_close":
        bool(np.allclose(chunked_state, mega_state, rtol=0.05, atol=0.05)),
        "output_max_abs_diff":
        float(output_delta.max(initial=0.0)),
        "state_max_abs_diff":
        float(state_delta.max(initial=0.0)),
        "output_mean_abs_diff":
        float(output_delta.mean()),
        "state_mean_abs_diff":
        float(state_delta.mean()),
    }


def _cpu_reference(arrays: dict[str, jax.Array],
                   lengths: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(arrays["q"], dtype=np.float32)
    k = np.asarray(arrays["k"], dtype=np.float32)
    v = np.asarray(arrays["v"], dtype=np.float32)
    gate = np.asarray(arrays["g"], dtype=np.float32)
    beta = np.asarray(arrays["beta"], dtype=np.float32)
    a_scale = np.exp(np.asarray(arrays["a_log"], dtype=np.float32))
    dt_bias = np.asarray(arrays["dt_bias"],
                         dtype=np.float32).reshape(q.shape[2], DIM)
    q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    q = np.asarray(jnp.asarray(q, dtype=jnp.bfloat16), dtype=np.float32)
    k = np.asarray(jnp.asarray(k, dtype=jnp.bfloat16), dtype=np.float32)
    gate = LOWER_BOUND / (1.0 + np.exp(-(a_scale[None, None, :, None] *
                                         (gate + dt_bias[None, None, :, :]))))
    state = np.asarray(arrays["initial_state"], dtype=np.float32).copy()
    output = np.zeros_like(v, dtype=np.float32)
    offset = 0
    for segment, length in enumerate(lengths):
        running = state[0, segment].copy()
        for token in range(offset, offset + length):
            running *= np.exp(gate[0, token])[:, :, None]
            prediction = np.einsum("hk,hkv->hv", k[0, token], running)
            delta = beta[0, token, :, None] * (v[0, token] - prediction)
            running += k[0, token, :, :, None] * delta[:, None, :]
            output[0, token] = SCALE * np.einsum("hk,hkv->hv", q[0, token],
                                                 running)
        state[0, segment] = running
        offset += length
    return output, state


def _write_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-steps", type=int, default=10)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    chunk_kda, mega_kda, layout_supported = _load_kernels(repo_root)
    _write_jsonl(
        args.output,
        {
            "event": "environment",
            "jax_version": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "benchmark": "chunked_vs_mega_kda",
        },
    )

    geometries = [
        (2, 128, 3, [
            ("cpu_reference", (37, 55, 24), True),
            ("k3_t128_three_segments_one_tile", (20, 20, 24), False),
        ]),
        (12, 512, 8, [
            ("k3_t512_segments1", (512, ), True),
            ("k3_t512_segments3", (130, 127, 255), True),
        ]),
        (12, 4096, 8, [
            ("k3_t4096_segments1_isl4000", (4000, ), True),
            ("k3_t4096_segments4_isl4000", (1000, 1000, 1000, 1000), True),
        ]),
    ]
    all_correct = True
    for geometry_index, (heads, tokens, num_segments,
                         cases) in enumerate(geometries):
        arrays = _inputs(
            heads=heads,
            tokens=tokens,
            num_segments=num_segments,
            seed=20260821 + geometry_index,
        )
        chunked, mega, guarded = _make_runners(
            chunk_kda,
            mega_kda,
            layout_supported,
            num_segments=num_segments,
        )
        for case_index, (case_name, lengths,
                         expected_layout_supported) in enumerate(cases):
            segment_ids = jnp.asarray(_segment_ids(lengths, tokens))
            query_start_loc = jnp.asarray(
                _query_start_loc(lengths, num_segments))
            actual_layout_supported = bool(
                layout_supported(query_start_loc, tokens))
            chunked_result = _call(chunked, arrays, segment_ids,
                                   query_start_loc)
            guarded_result = _call(guarded, arrays, segment_ids,
                                   query_start_loc)
            _block(chunked_result)
            _block(guarded_result)
            comparison = {
                "layout_guard_matches":
                actual_layout_supported == expected_layout_supported,
                **{
                    f"guarded_{key}": value
                    for key, value in _compare(
                        chunked_result,
                        guarded_result,
                        real_segments=len(lengths),
                    ).items()
                },
            }
            mega_result = None
            if expected_layout_supported:
                mega_result = _call(mega, arrays, segment_ids, query_start_loc)
                _block(mega_result)
                comparison.update({
                    f"mega_{key}": value
                    for key, value in _compare(
                        chunked_result,
                        mega_result,
                        real_segments=len(lengths),
                    ).items()
                })
            if case_name == "cpu_reference":
                reference_output, reference_state = _cpu_reference(
                    arrays, lengths)
                chunked_output, chunked_state = jax.device_get(chunked_result)
                guarded_output, guarded_state = jax.device_get(guarded_result)
                assert mega_result is not None
                mega_output, mega_state = jax.device_get(mega_result)
                comparison.update({
                    "chunked_cpu_output_close":
                    bool(
                        np.allclose(
                            np.asarray(chunked_output, dtype=np.float32),
                            reference_output,
                            rtol=0.05,
                            atol=0.05,
                        )),
                    "mega_cpu_output_close":
                    bool(
                        np.allclose(
                            np.asarray(mega_output, dtype=np.float32),
                            reference_output,
                            rtol=0.05,
                            atol=0.05,
                        )),
                    "chunked_cpu_state_close":
                    bool(
                        np.allclose(
                            np.asarray(chunked_state[:, :len(lengths)],
                                       dtype=np.float32),
                            reference_state[:, :len(lengths)],
                            rtol=0.05,
                            atol=0.05,
                        )),
                    "mega_cpu_state_close":
                    bool(
                        np.allclose(
                            np.asarray(mega_state[:, :len(lengths)],
                                       dtype=np.float32),
                            reference_state[:, :len(lengths)],
                            rtol=0.05,
                            atol=0.05,
                        )),
                    "guarded_cpu_output_close":
                    bool(
                        np.allclose(
                            np.asarray(guarded_output, dtype=np.float32),
                            reference_output,
                            rtol=0.05,
                            atol=0.05,
                        )),
                    "guarded_cpu_state_close":
                    bool(
                        np.allclose(
                            np.asarray(guarded_state[:, :len(lengths)],
                                       dtype=np.float32),
                            reference_state[:, :len(lengths)],
                            rtol=0.05,
                            atol=0.05,
                        )),
                })
            case_correct = all(
                value for key, value in comparison.items()
                if key == "layout_guard_matches" or key.endswith("_finite")
                or key.endswith("_close"))
            all_correct = all_correct and case_correct
            _write_jsonl(
                args.output, {
                    "event": "correctness",
                    "case": case_name,
                    "heads": heads,
                    "tokens": tokens,
                    "real_tokens": sum(lengths),
                    "real_segments": len(lengths),
                    "n_max": num_segments,
                    "layout_supported": actual_layout_supported,
                    "passed": case_correct,
                    **comparison,
                })

            runners = [("chunked", chunked)]
            if expected_layout_supported:
                runners.append(("mega", mega))
            runners.append(("guarded_mega", guarded))
            for kernel_name, runner in runners:
                first_ms, samples = _latencies(
                    runner,
                    arrays,
                    segment_ids,
                    query_start_loc,
                    warmups=args.warmups,
                    iterations=args.iterations,
                )
                _write_jsonl(
                    args.output, {
                        "event": "result",
                        "operator": "kimi_k3_kda_prefill",
                        "case": case_name,
                        "kernel": kernel_name,
                        "heads": heads,
                        "tokens": tokens,
                        "real_tokens": sum(lengths),
                        "real_segments": len(lengths),
                        "n_max": num_segments,
                        "first_call_ms": first_ms,
                        "warmups": args.warmups,
                        "iterations": args.iterations,
                        **_stats(samples),
                    })

            should_profile = (args.profile_dir is not None
                              and case_name == "k3_t4096_segments1_isl4000")
            if should_profile:
                with jax.profiler.trace(str(args.profile_dir),
                                        create_perfetto_trace=True):
                    for kernel_name, runner in (
                        ("chunked", chunked),
                        ("mega", mega),
                        ("guarded_mega", guarded),
                    ):
                        for step in range(args.profile_steps):
                            with jax.profiler.StepTraceAnnotation(
                                    f"{kernel_name}_{case_name}",
                                    step_num=step,
                            ):
                                result = _call(runner, arrays, segment_ids,
                                               query_start_loc)
                                _block(result)

    _write_jsonl(args.output, {"event": "complete", "passed": all_correct})
    if not all_correct:
        raise SystemExit("one or more correctness comparisons failed")


if __name__ == "__main__":
    main()
