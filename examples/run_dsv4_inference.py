import os

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["MODEL_IMPL_TYPE"] = "vllm"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ.setdefault("VLLM_FP4_CACHE_DIR", "/dev/shm/dsv4-flash-real-fp4")

from vllm import LLM, SamplingParams


def check_tpu_device_availability(timeout_seconds=60):
    """Wait until TPU devices /dev/accel* are free from locks before launching LLM."""
    import glob
    import shutil
    import subprocess
    import time

    start_time = time.time()
    waited = False
    while time.time() - start_time < timeout_seconds:
        accel_files = glob.glob("/dev/accel*") + glob.glob("/dev/vfio/*")
        if not accel_files:
            break
        try:
            res = subprocess.run(["lsof"] + accel_files, capture_output=True, text=True)
            if res.returncode != 0 or not res.stdout.strip():
                break
            print(
                "[PREFLIGHT_WAIT] TPU chips busy (locked by background processes). Waiting for teardown...",
                flush=True,
            )
            waited = True
        except Exception:
            break
        time.sleep(3)

    if waited:
        print(
            "[PREFLIGHT_SUCCESS] TPU chips released. Waiting 5s buffer for driver stabilization...",
            flush=True,
        )
        time.sleep(5)

    # Runs once the chips are free, so these handles are all stale by now.
    for pattern in ("/tmp/ipc*", "/tmp/psm*", "/tmp/vllm*"):
        for p in glob.glob(pattern):
            try:
                if shutil.os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    shutil.os.remove(p)
            except Exception:
                pass

    print(
        "[PREFLIGHT_CHECK] All TPU chips free. Proceeding with LLM initialization.",
        flush=True,
    )


def main():
    check_tpu_device_availability()
    # Instantiate the LLM. It will automatically use our TPU platform and patches.
    # We use TP=8 to utilize all 8 TPU chips on the v7x-8 node.
    tp_size = int(os.getenv("VLLM_TP_SIZE", "8"))
    ep_size = int(os.getenv("VLLM_EP_SIZE", "8"))
    max_model_len = int(os.getenv("DSV4_MAX_MODEL_LEN", "64"))
    # Keep batched tokens below max_model_len so long prompts prefill in
    # chunks, as the production benchmark does.
    max_num_batched_tokens = int(
        os.getenv("DSV4_MAX_NUM_BATCHED_TOKENS", str(max_model_len))
    )
    llm = LLM(
        model=os.getenv("DSV4_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash"),
        quantization="deepseek_v4_fp8",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=int(os.getenv("DSV4_MAX_NUM_SEQS", "256")),
        kv_cache_dtype="fp8_e4m3",
        gpu_memory_utilization=float(
            os.getenv("GPU_MEMORY_UTILIZATION", os.getenv("GPU_MEM_UTIL", "0.40"))
        ),
        enable_expert_parallel=ep_size > 1,
        # Off by default: it changes block allocation, which interacts with
        # the SWA/compressed-KV overlay.
        enable_prefix_caching=os.getenv("DSV4_ENABLE_PREFIX_CACHING", "0") == "1",
        enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
        # DSV4_ASYNC_SCHEDULING=0 disables vLLM's async scheduling, which
        # is otherwise auto-enabled when the executor supports it.
        async_scheduling=(False if os.getenv("DSV4_ASYNC_SCHEDULING") == "0" else None),
        # Auto-prefetch only engages for network filesystems, so a local
        # checkpoint needs this set explicitly.
        safetensors_load_strategy=os.getenv("DSV4_SAFETENSORS_LOAD_STRATEGY"),
    )

    # Comma-separated paths, each run in its own `generate` call to keep
    # single-sequence conditions, against one model load.
    prompt_files = [p for p in os.getenv("DSV4_PROMPT_FILES", "").split(",") if p]
    if prompt_files:
        prompt_contents = []
        for path in prompt_files:
            with open(path) as f:
                prompt_contents.append((os.path.basename(path), f.read()))
    else:
        prompt_contents = [
            (
                "DSV4_PROMPT",
                os.getenv(
                    "DSV4_PROMPT",
                    "Explain the difference between XLA and LLVM in one sentence.",
                ),
            )
        ]

    tokenizer = llm.get_tokenizer()
    prompt_texts = [
        (
            label,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            ),
        )
        for label, content in prompt_contents
    ]

    _temp = float(os.getenv("DSV4_TEMPERATURE", "0.0"))
    _max_tokens = int(os.getenv("DSV4_MAX_TOKENS", "50"))
    sampling_params = SamplingParams(
        temperature=_temp, max_tokens=_max_tokens, ignore_eos=False, logprobs=5
    )

    print("=== Testing formatted prompt with llm.generate ===")
    for label, prompt_text in prompt_texts:
        print(f"[SWEEP] ===== prompt: {label} =====")
        outputs = llm.generate([prompt_text], sampling_params)
        for output in outputs:
            print(
                f"[SWEEP] {label}: prompt_tokens="
                f"{len(output.prompt_token_ids)} "
                f"generated={output.outputs[0].text!r}"
            )
        _report(outputs)


def _report(outputs):
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Prompt token IDs: {output.prompt_token_ids}")
        print(f"Generated token IDs: {output.outputs[0].token_ids}")
        print(f"Generated text: {output.outputs[0].text!r}")
        if (
            output.outputs[0].logprobs is not None
            and len(output.outputs[0].logprobs) > 0
        ):
            print(f"Generated logprobs (step 1): {output.outputs[0].logprobs[0]}")
            for i, lp in enumerate(output.outputs[0].logprobs):
                print(f"[STEP_LOGPROBS] step={i} {lp}")
        print("-" * 80)


if __name__ == "__main__":
    main()
