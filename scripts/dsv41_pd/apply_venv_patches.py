"""只改 venv 里第三方包的运行时补丁（apply_patches.py 第 3、4 节），用目标 venv 的 python 运行：
    /zzlfs/temu/.venv-epic/bin/python /zzlfs/temu/scripts/apply_venv_patches.py
用 find_spec 定位文件，不 import vllm / vllm_torchtpu（避免初始化 TPU）。可重复运行。
不碰 vllm_torchtpu 源码（apply_patches.py 第 1、2 节是针对旧 PD 分支的；新代码自带 host_memory.py）。"""
import importlib.util
import os
import py_compile


def pkg_dir(name):
    spec = importlib.util.find_spec(name)
    return os.path.dirname(spec.origin)


# 3. runai_model_streamer_gcs: bucket() instead of get_bucket()（不需要 storage.buckets.get 权限）
p = os.path.join(pkg_dir("runai_model_streamer_gcs"), "files", "files.py")
txt = open(p).read()
if "gcs.get_bucket(bucket_name)" in txt:
    open(p, "w").write(txt.replace("gcs.get_bucket(bucket_name)", "gcs.bucket(bucket_name)"))
    py_compile.compile(p, doraise=True)
    print("patched", p)

# 4. vLLM scheduler: connector-declared exact-state loads are neither rounded nor replayed
p = os.path.join(pkg_dir("vllm"), "v1", "core", "sched", "scheduler.py")
txt = open(p).read()
if "_connector_skips_prefix_replay" not in txt:
    edits = [
        ("""                        if self.prefix_replay_tokens:
                            # SWA bounded replay recomputes the hit's last""",
         """                        if (
                            self.prefix_replay_tokens
                            and not self._connector_skips_prefix_replay(request)
                        ):
                            # SWA bounded replay recomputes the hit's last"""),
        ("""        num_replay_tokens = self._mark_prefix_replay(
            request, request.num_computed_tokens
        )""",
         """        num_replay_tokens = (
            0
            if self._connector_skips_prefix_replay(request)
            else self._mark_prefix_replay(request, request.num_computed_tokens)
        )"""),
        ("""    def _mark_prefix_replay(self, request: Request, num_hit_tokens: int) -> int:""",
         """    def _connector_skips_prefix_replay(self, request: Request) -> bool:
        \"\"\"Whether the connector's load restores the request's exact KV state
        (sliding-window and ring groups included), so the load is neither
        rounded down to a block nor followed by a bounded replay.\"\"\"
        skips = getattr(self.connector, "skips_prefix_replay", None)
        return bool(skips is not None and skips(request))

    def _mark_prefix_replay(self, request: Request, num_hit_tokens: int) -> int:"""),
    ]
    for old, new in edits:
        assert txt.count(old) == 1, old[:60]
        txt = txt.replace(old, new, 1)
    open(p, "w").write(txt)
    py_compile.compile(p, doraise=True)
    print("patched", p)
# 5. vLLM detokenizer: prime DecodeStream with the prompt tail only (2026-10-28)
# FastIncrementalDetokenizer primed tokenizers' DecodeStream with the whole prompt, so the first
# output token's step() decoded every prompt token: ~0.5 s for a 557k-token prompt, paid on both
# PD hops (P samples one token too). The first token's text is decode(prefix + t) - decode(prefix)
# and depends only on the last few prompt tokens (leading space / multi-byte UTF-8 across tokens);
# a window that starts inside a multi-byte character only garbles the window's head, not the diff.
p = os.path.join(pkg_dir("vllm"), "v1", "engine", "detokenizer.py")
txt = open(p).read()
old = """        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )"""
new = """        # Prime with the prompt tail only: priming with the whole prompt makes
        # the first step() decode every prompt token (~0.5 s at 557k tokens).
        # A prompt ending inside a multi-byte character keeps the full prime,
        # where DecodeStream's first emission depends on the whole prefix.
        prompt_ids = request.prompt_token_ids
        if prompt_ids is not None and len(prompt_ids) > 32 and not tokenizer._tokenizer.decode(
            prompt_ids[-8:], skip_special_tokens=self.skip_special_tokens
        ).endswith("\ufffd"):
            prompt_ids = prompt_ids[-32:]
        self.stream = tokenizers.decoders.DecodeStream(
            ids=prompt_ids,
            skip_special_tokens=self.skip_special_tokens,
        )"""
if "keeps the full prime" not in txt:
    assert txt.count(old) == 1, "detokenizer layout changed"
    open(p, "w").write(txt.replace(old, new))
    py_compile.compile(p, doraise=True)
    print("patched", p)
# 6. vLLM DP engine: hold a new wave's first step briefly (2026-10-28)
# With EP the DP ranks step in lockstep. When several requests arrive within a few ms on different
# ranks (the customer's "4P" case), the first one starts a wave immediately and the others wait a
# whole step (~180 ms at the 2048 bucket) before their prefill starts. Waiting a few ms when a wave
# starts lets them join the first step. VLLM_DP_WAVE_START_DELAY_MS (default 0 = unchanged).
p = os.path.join(pkg_dir("vllm"), "v1", "engine", "core.py")
txt = open(p).read()
old = """            was_running = self.engines_running
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()
"""
new = """            was_running = self.engines_running
            self._process_input_queue()
            if not was_running and self.engines_running:
                # A wave just started: let requests that arrive within a few ms
                # on other ranks join its first lockstep step.
                delay_ms = float(os.environ.get("VLLM_DP_WAVE_START_DELAY_MS", "0") or 0)
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000)
                    self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()
"""
if "VLLM_DP_WAVE_START_DELAY_MS" not in txt:
    assert txt.count(old) == 1, "DP busy loop layout changed"
    open(p, "w").write(txt.replace(old, new))
    py_compile.compile(p, doraise=True)
    print("patched", p)
# 7. starlette Request.json via orjson (2026-10-28)
# A rendered long prompt arrives as ~10^6 token ids; stdlib json.loads takes ~70 ms of the API
# server's event loop per request (P and D each), so concurrent long requests queue behind each
# other before reaching the engines and miss each other's first lockstep step. orjson: ~10 ms.
p = os.path.join(pkg_dir("starlette"), "requests.py")
txt = open(p).read()
old = """            body = await self.body()
            self._json = json.loads(body)"""
new = """            body = await self.body()
            try:  # orjson parses a 10^6-token-id body ~7x faster than json
                import orjson

                self._json = orjson.loads(body)
            except ImportError:
                self._json = json.loads(body)"""
if "orjson parses a 10^6-token-id body" not in txt:
    assert txt.count(old) == 1, "starlette Request.json layout changed"
    open(p, "w").write(txt.replace(old, new))
    py_compile.compile(p, doraise=True)
    print("patched", p)
print("done")
