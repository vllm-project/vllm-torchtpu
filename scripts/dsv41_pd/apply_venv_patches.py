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
# 8. vLLM renderer: long prompts tokenized in parallel chunks (2026-11-15)
# Chat requests are tokenized in P's API server (the proxy renders only
# /v1/completions). One HF fast-tokenizer call on a 250k-token AgentX prompt
# takes ~0.5 s of one core; at ~6 req/s the 4 API servers saturate and P's
# queue grows (AgentX c256: P hop 1.2 s -> 7.8 s over the run while the engine
# itself took ~0.7 s). Same split rule as the proxy (toy_proxy_server.py):
# cut only at a space between two ASCII letters, where the pre-tokenizer
# always starts a new " word" piece, so the chunks' ids concatenate to the
# whole-text ids; tokenizers.encode_batch runs the chunks on its Rust thread
# pool without the GIL. Only for plain encodes (no offsets, at most
# add_special_tokens / right-side truncation) of fast tokenizers whose
# post-processor adds no special tokens; anything else takes the original path.
# VLLM_CHUNKED_TOKENIZE=0 disables it.
p = os.path.join(pkg_dir("vllm"), "renderers", "base.py")
txt = open(p).read()
if "_chunked_token_ids" not in txt:
    old = (
        '        encoding = tokenizer(prompt["prompt"], **kwargs)\n'
        '        return self._build_tokens_prompt(\n'
        '            encoding["input_ids"],\n'
    )
    new = (
        '        if not want_offsets:\n'
        '            ids = _chunked_token_ids(tokenizer, prompt["prompt"], kwargs)\n'
        '            if ids is not None:\n'
        '                return self._build_tokens_prompt(ids, prompt, offset_mapping=None)\n'
        + old
    )
    helper = '''

# --- chunked tokenization (apply_venv_patches.py section 8) -----------------
import os as _os
import re as _re

_CHUNK_SAFE_SPLIT = _re.compile(r"(?<=[A-Za-z]) (?=[A-Za-z])")
_CHUNK_CHARS = 65536
_CHUNK_MIN_CHARS = int(_os.environ.get("VLLM_CHUNKED_TOKENIZE_MIN_CHARS", "131072"))
_CHUNK_ENABLED = _os.environ.get("VLLM_CHUNKED_TOKENIZE", "1") != "0"
_CHUNK_VERIFY = _os.environ.get("VLLM_CHUNKED_TOKENIZE_VERIFY", "0") == "1"
_CHUNK_BACKENDS: dict = {}


_CHUNK_ADDS_SPECIALS: dict = {}


def _chunk_backend(tokenizer):
    """The tokenizers.Tokenizer behind a HF fast tokenizer (else None), and
    whether its encodes add special tokens. Cached per tokenizer object."""
    key = id(tokenizer)
    if key not in _CHUNK_BACKENDS:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        adds = True
        if backend is not None:
            probe = "The capital of France is"
            # Probe the HF wrapper, not the backend: some tokenizers add BOS
            # in Python (add_bos_token) rather than in the post-processor.
            adds = tokenizer(probe, add_special_tokens=True)["input_ids"] != tokenizer(
                probe, add_special_tokens=False
            )["input_ids"]
        if backend is not None:
            # A private copy: HF's __call__ leaves enable_truncation(max_length)
            # set on its backend, which would cut every chunk.
            from tokenizers import Tokenizer as _Tokenizer

            backend = _Tokenizer.from_str(backend.to_str())
            backend.no_truncation()
            backend.no_padding()
        _CHUNK_BACKENDS[key] = backend
        _CHUNK_ADDS_SPECIALS[key] = adds
    return _CHUNK_BACKENDS[key], _CHUNK_ADDS_SPECIALS[key]


def _chunked_token_ids(tokenizer, text, kwargs):
    if not _CHUNK_ENABLED or not isinstance(text, str) or len(text) < _CHUNK_MIN_CHARS:
        return None
    if set(kwargs) - {"add_special_tokens", "truncation", "max_length"}:
        return None
    backend, adds_specials = _chunk_backend(tokenizer)
    if backend is None or (kwargs.get("add_special_tokens", True) and adds_specials):
        return None
    cuts, pos = [0], 0
    while pos + _CHUNK_CHARS < len(text):
        m = _CHUNK_SAFE_SPLIT.search(text, pos + _CHUNK_CHARS)
        if m is None:
            break
        cuts.append(m.start())
        pos = m.start()
    cuts.append(len(text))
    if len(cuts) < 3:
        return None
    ids = []
    for enc in backend.encode_batch(
        [text[a:b] for a, b in zip(cuts, cuts[1:])], add_special_tokens=False
    ):
        ids.extend(enc.ids)
    max_length = kwargs.get("max_length")
    if kwargs.get("truncation") and max_length is not None and len(ids) > max_length:
        # vLLM's tokenizer registry sets truncation_side="left" by default.
        if getattr(tokenizer, "truncation_side", "right") == "left":
            ids = ids[len(ids) - max_length:]
        else:
            ids = ids[:max_length]
    if _CHUNK_VERIFY:
        full = tokenizer(text, **kwargs)["input_ids"]
        if full != ids:
            print(f"CHUNKED_TOKENIZE MISMATCH chars={len(text)} full={len(full)} "
                  f"chunked={len(ids)}", flush=True)
            return full
        print(f"CHUNKED_TOKENIZE OK chars={len(text)} tokens={len(ids)}", flush=True)
    return ids
'''
    assert txt.count(old) == 1, "vllm renderers/base.py _tokenize_prompt layout changed"
    txt = txt.replace(old, new) + helper
    open(p, "w").write(txt)
    py_compile.compile(p, doraise=True)
    print("patched", p)
print("done")
