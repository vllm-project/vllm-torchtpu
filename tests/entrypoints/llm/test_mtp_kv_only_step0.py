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
"""End to end, the MTP draft's two-pass step 0 (`TPU_MTP_KV_ONLY_STEP0`)
gives the same outputs, proposals and draft acceptance as the single pass,
and its programs compile nothing after warmup.

A one-layer GLM-5.2-shaped DSA model with one MTP layer (small dims, dummy
weights, no tokenizer) serves the same token prompts with the switch off and
on, one engine after the other, each in a spawned child process. Six short
prompts and two long ones in one batch give a step whose prefill chunk
carries many requests, and later steps that verify the short ones' drafts
while the long ones keep prefilling.

Under EP-DP lockstep (data parallel 2 with expert parallel, served: vLLM
runs data parallelism out of process), idle ranks replay the two passes;
the served outputs must match the switch-off run, and no rank may hang.

Neither test touches the TPU from the pytest process itself. Each starts its
engines without the TPU bootstrap variables an earlier test's engine may have
left in the pytest process's environment, and waits for the chips to be free
before and after every engine it starts.

Run with: pytest tests/entrypoints/llm/test_mtp_kv_only_step0.py -v
"""

import contextlib
import json
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from tpu_release import wait_for_tpu_release

pytestmark = pytest.mark.tpu_test

NUM_SPEC = 3
MAX_NUM_SEQS = 8
MAX_NUM_BATCHED_TOKENS = 1024
MAX_MODEL_LEN = 2048
VOCAB = 8192
MAX_TOKENS = 24
# Generous for one engine's build, warmup and generate.
CHILD_TIMEOUT_S = 480
# The TPU platform writes these into the environment of the process that
# builds an engine, for that engine's workers, and leaves them there. An
# earlier test that built an engine in the pytest process (on every chip,
# say) leaves its values behind; an engine started from here must work out
# its own.
TPU_BOOTSTRAP_ENV = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCH_TPU_DP_MASTER_ADDR",
    "TORCH_TPU_DP_MASTER_PORT",
    "TORCH_TPU_DP_SIZE",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_TOPOLOGY",
    "TORCH_TPU_XPROF_SESSION_ID",
)
SPEC_DRAFT_METRIC = "vllm:spec_decode_num_draft_tokens"
SPEC_ACCEPTED_METRIC = "vllm:spec_decode_num_accepted_tokens"


def _write_config(path: str) -> None:
    """GLM-5.2's architecture (MLA + lightning indexer, MoE, one MTP layer)
    at small dims; the MLA latent, RoPE and indexer head sizes are the real
    ones, which the TPU kernels are built for."""
    from transformers import GlmMoeDsaConfig

    config = GlmMoeDsaConfig(
        vocab_size=VOCAB,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=1,
        first_k_dense_replace=0,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=192,
        v_head_dim=256,
        index_topk=256,
        index_head_dim=128,
        index_n_heads=8,
        max_position_embeddings=MAX_MODEL_LEN,
        num_nextn_predict_layers=1,
        architectures=["GlmMoeDsaForCausalLM"],
        dtype="bfloat16",
    )
    config.save_pretrained(path)


def _prompts() -> list[list[int]]:
    rng = np.random.default_rng(0)
    lengths = [120, 150, 130, 140, 110, 160, 900, 700]
    return [rng.integers(10, VOCAB - 10, size=n).tolist() for n in lengths]


def _xla_cache_entries() -> int:
    return len(torch.tpu._get_cache_stats().per_entry_stats)


def _instrument(worker) -> bool:
    """From now on, record every proposal, and for every two-pass step 0
    the XLA cache entries it adds; returns whether the switch built pass 1.

    The two-pass step 0 runs pass 1, the pass-2 gather and pass-2 step 0:
    compiled programs that compile, if at all, when they are called, so an
    entry they add appears within the call."""
    drafter = worker.model_runner.drafter
    propose = drafter.propose
    drafter.recorded_drafts = []
    drafter.two_pass_graphs = []

    def recorded(*args, **kwargs):
        drafts = propose(*args, **kwargs)
        rows = drafts.cpu().tolist() if isinstance(drafts, torch.Tensor) else drafts
        drafter.recorded_drafts.append([list(row) for row in rows])
        return drafts

    drafter.propose = recorded
    if drafter._mtp_kv_only is None:
        return False
    two_pass_step0 = drafter._two_pass_step0

    def counted(*args, **kwargs):
        before = _xla_cache_entries()
        out = two_pass_step0(*args, **kwargs)
        drafter.two_pass_graphs.append(_xla_cache_entries() - before)
        return out

    drafter._two_pass_step0 = counted
    return True


def _recorded(worker) -> tuple[list, list]:
    drafter = worker.model_runner.drafter
    return drafter.two_pass_graphs, drafter.recorded_drafts


def _graphs(worker) -> int:
    return _xla_cache_entries()


def _run_in_child(model_dir: str, switch: bool) -> dict:
    """`_run` in a spawned child process. A process that has touched the TPU
    keeps its chips until it exits, so the child, not pytest, holds them, and
    they are free again for the tests after this one that start engines of
    their own."""
    # The previous engine (another test's, or the switch-off child) may still
    # hold the chips while it exits.
    wait_for_tpu_release()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_run_case, args=(model_dir, switch, child))
    process.start()
    child.close()
    try:
        result = parent.recv() if parent.poll(CHILD_TIMEOUT_S) else "timed out"
    except EOFError:
        result = "exited without a result"
    finally:
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join()
    try:
        assert isinstance(result, dict), f"child exit code {process.exitcode}: {result}"
    finally:
        wait_for_tpu_release()
    return result


def _run_case(model_dir: str, switch: bool, conn) -> None:
    """Runs one engine in this (child) process and sends back its result or
    a failure."""
    try:
        conn.send(_run(model_dir, switch))
    except BaseException:
        conn.send(traceback.format_exc())


def _run(model_dir: str, switch: bool) -> dict:
    """One engine, with the engine core in this process so `collective_rpc`
    can instrument the drafter; call it through `_run_in_child`."""
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.inputs import TokensPrompt
    from vllm.v1.metrics.reader import Counter

    for key in TPU_BOOTSTRAP_ENV:
        os.environ.pop(key, None)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "1"
    os.environ["TPU_MTP_KV_ONLY_STEP0"] = "1" if switch else "0"
    llm = LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        load_format="dummy",
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        kv_cache_memory_bytes=1 << 30,
        # The sparse MLA kernels take an fp8 latent cache only.
        kv_cache_dtype="fp8",
        enable_prefix_caching=False,
        disable_log_stats=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": NUM_SPEC},
        seed=0,
    )
    try:
        (built,) = llm.collective_rpc(_instrument)
        (graphs_before,) = llm.collective_rpc(_graphs)
        params = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            ignore_eos=True,
            detokenize=False,
        )
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in _prompts()], params
        )
        (graphs_after,) = llm.collective_rpc(_graphs)
        ((two_pass_graphs, proposals),) = llm.collective_rpc(_recorded)
        drafts = accepted = 0.0
        for metric in llm.get_metrics():
            if metric.name == SPEC_DRAFT_METRIC:
                assert isinstance(metric, Counter)
                drafts += metric.value
            elif metric.name == SPEC_ACCEPTED_METRIC:
                assert isinstance(metric, Counter)
                accepted += metric.value
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        cleanup_dist_env_and_memory()
    return {
        "built": built,
        "tokens": [list(out.outputs[0].token_ids) for out in outputs],
        "drafts": drafts,
        "accepted": accepted,
        "two_pass_steps": len(two_pass_graphs),
        "two_pass_new_graphs": sum(two_pass_graphs),
        "proposals": proposals,
        "new_graphs": graphs_after - graphs_before,
    }


# ---------------------------------------------------------------------------
# EP-DP lockstep, served
# ---------------------------------------------------------------------------

DP_SIZE = 2
SERVE_TIMEOUT_S = 1800
BUILT_LOG = "MTP draft step 0 of chunks with"
# Logged (at debug level) each time a padding chunk replays the two passes.
REPLAY_LOG = "replaying the two-pass draft step 0"
PROPOSER_LOGGER = "vllm.vllm_torchtpu.spec_decode.eagle3"


def _logging_config(path: str) -> str:
    """vLLM's logging at info, the proposer's at debug, so the server log
    shows the padding-chunk replays."""
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "format": "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] "
                "%(message)s"
            }
        },
        "handlers": {
            "vllm": {
                "class": "logging.StreamHandler",
                "formatter": "plain",
                "level": "DEBUG",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": False},
            PROPOSER_LOGGER: {"level": "DEBUG"},
        },
    }
    with open(path, "w") as fh:
        json.dump(config, fh)
    return path


def _local_chips() -> int:
    try:
        from tpu_info import device as tpu_device

        _, num_chips = tpu_device.get_local_chips()
        return num_chips
    except Exception as e:  # noqa: BLE001 - detection failure is a skip
        pytest.skip(f"cannot count the local TPU chips: {e}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _serve_and_generate(model_dir: str, switch: bool) -> dict:
    """`vllm serve` with DP 2 + EP. A lone long prompt first, so one rank
    prefills while the other is idle and replays its draft trace; then the
    e2e prompts sent concurrently, so the DP load balancer spreads them.
    Returns the generated token ids, how many ranks logged that they built
    pass 1, and how many padding-chunk replays of the two passes were
    logged."""
    # The previous engine (another test's, or this test's switch-off server)
    # may still hold the chips while it exits.
    wait_for_tpu_release()
    port = _free_port()
    base = f"http://localhost:{port}"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model_dir,
        "--port",
        str(port),
        "--skip-tokenizer-init",
        "--load-format",
        "dummy",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        str(DP_SIZE),
        "--enable-expert-parallel",
        "--kv-cache-dtype",
        "fp8",
        "--kv-cache-memory-bytes",
        str(1 << 30),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--no-enable-prefix-caching",
        "--seed",
        "0",
        "--speculative-config",
        json.dumps({"method": "mtp", "num_speculative_tokens": NUM_SPEC}),
    ]
    env = {k: v for k, v in os.environ.items() if k not in TPU_BOOTSTRAP_ENV}
    env.update(
        TPU_MTP_KV_ONLY_STEP0="1" if switch else "0",
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=str(SERVE_TIMEOUT_S),
        VLLM_ENGINE_READY_TIMEOUT_S=str(SERVE_TIMEOUT_S),
        VLLM_LOGGING_CONFIG_PATH=_logging_config(
            os.path.join(os.path.dirname(model_dir), "logging.json")
        ),
    )
    # A file, not a pipe: nothing drains a pipe during startup.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", prefix="mtp_kv_only_dp_", delete=False
    ) as log:
        server = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            start_new_session=True,
        )

    def tail() -> str:
        """The log from its first traceback or crash, then its last lines: a
        worker's root cause is logged well before the API server's traceback."""
        with open(log.name, errors="replace") as fh:
            lines = fh.readlines()
        marks = ("Traceback", "failed to start", "Fatal Python error", "Check failed")
        first = next(
            (i for i, line in enumerate(lines) if any(m in line for m in marks)),
            len(lines),
        )
        return "".join([*lines[first : first + 120], "...\n", *lines[-60:]])

    try:
        deadline = time.time() + SERVE_TIMEOUT_S
        while True:
            if server.poll() is not None:
                pytest.fail(f"vllm serve exited with {server.returncode}:\n{tail()}")
            try:
                urllib.request.urlopen(f"{base}/health", timeout=10).close()
                break
            except (urllib.error.URLError, OSError):
                pass
            if time.time() > deadline:
                pytest.fail(f"vllm serve did not become healthy:\n{tail()}")
            time.sleep(10)

        def one(ids: list[int]) -> list[int]:
            payload = json.dumps(
                {
                    "model": model_dir,
                    "prompt": ids,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "ignore_eos": True,
                    "return_token_ids": True,
                }
            ).encode()
            request = urllib.request.Request(
                f"{base}/v1/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=SERVE_TIMEOUT_S) as response:
                body = json.loads(response.read().decode())
            return body["choices"][0]["token_ids"]

        lone = one(_lone_prompt())
        with ThreadPoolExecutor(max_workers=len(_prompts())) as pool:
            tokens = list(pool.map(one, _prompts()))
        with open(log.name, errors="replace") as fh:
            text = fh.read()
        return {
            "tokens": [lone, *tokens],
            "built": text.count(BUILT_LOG),
            "replays": text.count(REPLAY_LOG),
        }
    finally:
        server.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            server.wait(timeout=180)
        # The engine processes it started may outlive it.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=60)
        wait_for_tpu_release()


def _lone_prompt() -> list[int]:
    # Longer than one step's token budget: two prefill-carrying steps.
    rng = np.random.default_rng(1)
    return rng.integers(10, VOCAB - 10, size=1500).tolist()


@pytest.mark.multichip
@pytest.mark.timeout(1200)
def test_two_pass_step0_matches_the_single_pass_under_ep_dp_lockstep(tmp_path):
    if _local_chips() < DP_SIZE:
        pytest.skip(f"needs {DP_SIZE} chips")
    model_dir = str(tmp_path / "glm-dsa-mtp-tiny")
    _write_config(model_dir)
    off = _serve_and_generate(model_dir, switch=False)
    on = _serve_and_generate(model_dir, switch=True)
    print("built: off", off["built"], "on", on["built"])
    print("replays: off", off["replays"], "on", on["replays"])
    assert off["built"] == off["replays"] == 0
    # Every DP rank builds pass 1, and the idle rank replays it while its
    # peer prefills the lone prompt.
    assert on["built"] >= DP_SIZE
    assert on["replays"] > 0
    assert on["tokens"] == off["tokens"]


# ---------------------------------------------------------------------------
# One chip, each engine in a spawned child
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
def test_two_pass_step0_matches_the_single_pass_end_to_end(tmp_path):
    model_dir = str(tmp_path / "glm-dsa-mtp-tiny")
    _write_config(model_dir)
    off = _run_in_child(model_dir, switch=False)
    on = _run_in_child(model_dir, switch=True)

    def summary(result):
        return {k: v for k, v in result.items() if k not in ("tokens", "proposals")}

    # The raw totals include programs the runner compiles after warmup with
    # the switch off too; the switch's own steps are counted separately.
    print("switch off:", summary(off), "switch on:", summary(on))
    differing = [
        step
        for step, (got, want) in enumerate(zip(on["proposals"], off["proposals"]))
        if got != want
    ]
    print(f"proposals: {len(on['proposals'])}, differing steps: {differing}")
    assert not off["built"] and off["two_pass_steps"] == 0
    # The many-request prefill chunk and the verify + prefill steps are above
    # the gate (the largest loop bucket + 1) and carry a prefill span.
    assert on["built"] and on["two_pass_steps"] > 0
    assert on["tokens"] == off["tokens"]
    assert off["drafts"] > 0
    assert (on["drafts"], on["accepted"]) == (off["drafts"], off["accepted"])
    # Every proposal, step 0's tokens and the loop steps' after them, is the
    # same: acceptance alone says little with dummy weights.
    assert len(on["proposals"]) == len(off["proposals"]) > 0
    for step, (got, want) in enumerate(zip(on["proposals"], off["proposals"])):
        assert got == want, f"proposal {step}: {got} != {want}"
    # Warmup compiled every program the two passes run: pass 1, the pass-2
    # gather and pass-2 step 0 add nothing while serving.
    assert on["two_pass_new_graphs"] == 0, on["two_pass_new_graphs"]
