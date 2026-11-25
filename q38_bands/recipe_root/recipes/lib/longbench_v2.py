#!/usr/bin/env python3
"""LongBench v2 Evaluation Harness for Cloud DevKit (CDK) & vLLM OpenAI Servers.

Implements the official THUDM/LongBench-v2 evaluation protocol (pred.py + result.py)
with full configuration options for subset filtering (length, difficulty, domain,
sub_domain, ids, word-count range, max_samples), context truncation (middle, head,
tail, skip, none), prompting modes (0-shot, 2-turn CoT, no-context, system prompt),
Qwen3.5 thinking mode (enable_thinking), generation sampling parameters, parallel
request execution, automatic retry/resume, and CDK result-store metric publishing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request

# =============================================================================
# Official THUDM/LongBench-v2 Prompt Templates (prompts/*.txt)
# =============================================================================

TEMPLATE_0SHOT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""

TEMPLATE_0SHOT_NO_CONTEXT = """What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

What is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)"."""

TEMPLATE_0SHOT_COT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let's think step by step:"""

TEMPLATE_0SHOT_COT_ANS = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let's think step by step: $COT_ANS$

Based on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)"."""

HF_DATA_JSON_URL = (
    "https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json"
)


def env_bool(name: str, default: bool = False) -> bool:
  val = os.environ.get(name, "")
  if not val:
    return default
  return val.strip().lower() in ("1", "true", "yes", "on", "y")


def env_int(name: str, default: int = 0) -> int:
  val = os.environ.get(name, "").strip()
  if not val or val.lower() in ("all", "full", "none"):
    return default
  try:
    return int(val)
  except ValueError:
    return default


def env_float(name: str, default: float | None = None) -> float | None:
  val = os.environ.get(name, "").strip()
  if not val or val.lower() == "none":
    return default
  try:
    return float(val)
  except ValueError:
    return default


def strip_thinking(text: str) -> str:
  """Removes <think>...</think> reasoning blocks before answer extraction."""
  if not text:
    return ""
  if "</think>" in text:
    text = text.split("</think>")[-1]
  text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
  return text.strip()


def extract_answer(response: str, strict: bool = False) -> str | None:
  """Extracts multiple-choice answer (A/B/C/D) following LongBench v2 pred.py."""
  if not response:
    return None
  cleaned = strip_thinking(response).replace("*", "")

  # 1. Official LongBench v2 pattern: "The correct answer is (A)"
  matches = re.findall(
      r"The correct answer is \(([A-D])\)", cleaned, re.IGNORECASE
  )
  if matches:
    return matches[-1].upper()

  # 2. Official LongBench v2 pattern: "The correct answer is A"
  matches = re.findall(
      r"The correct answer is ([A-D])\b", cleaned, re.IGNORECASE
  )
  if matches:
    return matches[-1].upper()

  if strict:
    return None

  # 3. Fallback patterns when model slightly varies phrasing
  matches = re.findall(
      r"(?:correct answer|answer|choice)\s*(?:is|:)\s*\(?([A-D])\)?(?:\b|\))",
      cleaned,
      re.IGNORECASE,
  )
  if matches:
    return matches[-1].upper()

  matches = re.findall(r"^\s*\(([A-D])\)\s*$", cleaned, re.MULTILINE)
  if matches:
    return matches[-1].upper()

  matches = re.findall(r"^\s*([A-D])\s*$", cleaned)
  if matches:
    return matches[-1].upper()

  return None


def format_prompt(
    template: str, item: dict[str, Any], context_override: str | None = None
) -> str:
  ctx = (
      context_override
      if context_override is not None
      else item.get("context", "")
  )
  return (
      template.replace("$DOC$", ctx)
      .replace("$Q$", str(item.get("question", "")))
      .replace("$C_A$", str(item.get("choice_A", "")))
      .replace("$C_B$", str(item.get("choice_B", "")))
      .replace("$C_C$", str(item.get("choice_C", "")))
      .replace("$C_D$", str(item.get("choice_D", "")))
  )


def shutil_which(cmd: str) -> str | None:
  for p in os.environ.get("PATH", "").split(os.pathsep):
    full = os.path.join(p, cmd)
    if os.path.isfile(full) and os.access(full, os.X_OK):
      return full
  return None


def load_tokenizer(tokenizer_name_or_path: str, model_name: str):
  """Loads HuggingFace AutoTokenizer (preferring local cache when available)."""
  try:
    from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  except ImportError:
    print(
        "[LongBench-v2] transformers not found; installing via uv/pip...",
        flush=True,
    )
    cmd = (
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "transformers",
            "tokenizers",
        ]
        if shutil_which("uv")
        else [
            sys.executable,
            "-m",
            "pip",
            "install",
            "transformers",
            "tokenizers",
        ]
    )
    subprocess.run(cmd, check=False)
    from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  candidates = []
  if tokenizer_name_or_path:
    candidates.append(tokenizer_name_or_path)
  local_ckpt = f"/root/.cache/huggingface/{model_name}"
  if os.path.isdir(local_ckpt) and local_ckpt not in candidates:
    candidates.append(local_ckpt)
  if model_name and model_name not in candidates:
    candidates.append(model_name)

  last_err = None
  for cand in candidates:
    try:
      print(f"[LongBench-v2] Loading tokenizer from: {cand}", flush=True)
      tok = AutoTokenizer.from_pretrained(cand, trust_remote_code=True)
      return tok
    except Exception as e:  # pylint: disable=broad-except
      last_err = e
      print(
          f"[LongBench-v2] Warning: tokenizer load failed for {cand}: {e}",
          flush=True,
      )
  raise RuntimeError(f"Failed to load tokenizer from {candidates}: {last_err}")


def truncate_prompt(
    prompt: str,
    max_len: int,
    truncate_mode: str,
    tokenizer: Any | None,
    word_mult: float = 3.5,
) -> tuple[str, int, int, bool, bool]:
  """Truncates or checks prompt length according to truncate_mode.

  Returns:
    (final_prompt, raw_token_len, final_token_len, was_truncated, was_skipped)
  """
  if max_len <= 0 or truncate_mode == "none":
    if tokenizer is not None:
      n_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
    else:
      n_tok = int(len(prompt) / max(word_mult, 1.0))
    return prompt, n_tok, n_tok, False, False

  if tokenizer is not None:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    raw_len = len(input_ids)
    if raw_len <= max_len:
      return prompt, raw_len, raw_len, False, False
    if truncate_mode == "skip":
      return prompt, raw_len, raw_len, False, True
    if truncate_mode == "head":
      trunc_ids = input_ids[-max_len:]
    elif truncate_mode == "tail":
      trunc_ids = input_ids[:max_len]
    else:
      # Official LongBench v2 middle truncation: keep first half + last half
      half = max_len // 2
      trunc_ids = input_ids[:half] + input_ids[-half:]
    truncated_prompt = tokenizer.decode(trunc_ids, skip_special_tokens=True)
    return truncated_prompt, raw_len, len(trunc_ids), True, False

  # Character-based fallback when --no_use_tokenizer is specified
  max_chars = int(max_len * word_mult)
  raw_est = int(len(prompt) / max(word_mult, 1.0))
  if len(prompt) <= max_chars:
    return prompt, raw_est, raw_est, False, False
  if truncate_mode == "skip":
    return prompt, raw_est, raw_est, False, True
  if truncate_mode == "head":
    return prompt[-max_chars:], raw_est, max_len, True, False
  if truncate_mode == "tail":
    return prompt[:max_chars], raw_est, max_len, True, False
  half = max_chars // 2
  return prompt[:half] + prompt[-half:], raw_est, max_len, True, False


def load_longbench_v2_dataset(
    dataset_name: str,
    dataset_split: str,
    dataset_path: str,
) -> list[dict[str, Any]]:
  """Loads LongBench v2 from a local/GCS file, HF Hub data.json, or datasets library."""
  # 1. Explicit local or gs:// path
  if dataset_path:
    local_path = dataset_path
    if dataset_path.startswith("gs://"):
      local_path = "/tmp/longbench_v2_data.json"
      print(
          f"[LongBench-v2] Downloading dataset from {dataset_path}...",
          flush=True,
      )
      subprocess.run(
          ["gcloud", "storage", "cp", dataset_path, local_path], check=True
      )
    p = Path(local_path)
    if p.is_dir():
      for candidate in ("data.json", "train.json", "data.jsonl"):
        if (p / candidate).is_file():
          p = p / candidate
          break
    print(f"[LongBench-v2] Loading dataset from local path: {p}", flush=True)
    if p.suffix == ".jsonl":
      return [
          json.loads(line)
          for line in p.read_text(encoding="utf-8").splitlines()
          if line.strip()
      ]
    if p.suffix == ".parquet":
      from datasets import load_dataset  # pylint: disable=g-import-not-at-top

      ds = load_dataset("parquet", data_files=str(p), split="train")
      return list(ds)
    return json.loads(p.read_text(encoding="utf-8"))

  # 2. Check cached data.json under HF_HOME
  hf_home = os.environ.get("HF_HOME", "/tmp/hf_cache")
  cache_dir = Path(hf_home) / "longbench_v2"
  cache_file = cache_dir / "data.json"
  if cache_file.is_file() and cache_file.stat().st_size > 1_000_000:
    print(f"[LongBench-v2] Loading cached dataset from {cache_file}", flush=True)
    return json.loads(cache_file.read_text(encoding="utf-8"))

  # 3. If dataset is THUDM/LongBench-v2, download data.json directly via hf CLI or HTTPS
  if dataset_name in ("THUDM/LongBench-v2", "zai-org/LongBench-v2"):
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_cli = shutil_which("hf") or shutil_which("huggingface-cli")
    if hf_cli and os.path.basename(hf_cli) == "hf":
      print(
          f"[LongBench-v2] Downloading {dataset_name}/data.json via hf CLI...",
          flush=True,
      )
      res = subprocess.run(
          [
              hf_cli,
              "download",
              "--repo-type",
              "dataset",
              dataset_name,
              "data.json",
              "--local-dir",
              str(cache_dir),
          ],
          check=False,
      )
      if res.returncode == 0 and cache_file.is_file():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    # Direct HTTPS download fallback
    try:
      print(
          f"[LongBench-v2] Downloading {HF_DATA_JSON_URL} -> {cache_file}...",
          flush=True,
      )
      headers = {"User-Agent": "cloud-devkit-longbench-v2/1.0"}
      hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
          "HUGGING_FACE_HUB_TOKEN"
      )
      if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
      req = urllib.request.Request(HF_DATA_JSON_URL, headers=headers)
      with urllib.request.urlopen(req, timeout=300) as resp:
        data_bytes = resp.read()
      cache_file.write_bytes(data_bytes)
      return json.loads(data_bytes.decode("utf-8"))
    except Exception as e:  # pylint: disable=broad-except
      print(
          f"[LongBench-v2] Direct download failed ({e}); falling back to"
          " datasets.load_dataset...",
          flush=True,
      )

  # 4. Fallback to HuggingFace `datasets` library
  try:
    from datasets import load_dataset  # pylint: disable=g-import-not-at-top
  except ImportError:
    cmd = (
        ["uv", "pip", "install", "--python", sys.executable, "datasets"]
        if shutil_which("uv")
        else [sys.executable, "-m", "pip", "install", "datasets"]
    )
    subprocess.run(cmd, check=True)
    from datasets import load_dataset  # pylint: disable=g-import-not-at-top

  print(
      f"[LongBench-v2] Loading {dataset_name} (split={dataset_split}) via"
      " datasets...",
      flush=True,
  )
  ds = load_dataset(dataset_name, split=dataset_split)
  return list(ds)


def parse_csv_set(raw: str) -> list[str]:
  if not raw:
    return []
  items = [x.strip() for x in raw.split(",") if x.strip()]
  if len(items) == 1 and items[0].lower() in ("all", "full", "*", "any"):
    return []
  return items


def filter_dataset(
    dataset: list[dict[str, Any]],
    lengths: list[str],
    difficulties: list[str],
    domains: list[str],
    sub_domains: list[str],
    ids: list[str],
    min_words: int,
    max_words: int,
    shuffle: bool,
    seed: int,
    max_samples: int,
) -> list[dict[str, Any]]:
  """Applies all user-configured subset filters to LongBench v2 examples."""
  length_set = {x.lower() for x in lengths}
  diff_set = {x.lower() for x in difficulties}
  domain_lowers = [x.lower() for x in domains]
  subdomain_lowers = [x.lower() for x in sub_domains]
  id_set = set(ids)

  filtered = []
  for item in dataset:
    if id_set and str(item.get("_id", "")) not in id_set:
      continue
    if length_set and str(item.get("length", "")).lower() not in length_set:
      continue
    if diff_set and str(item.get("difficulty", "")).lower() not in diff_set:
      continue
    if domain_lowers:
      dom = str(item.get("domain", "")).lower()
      if not any(d == dom or d in dom for d in domain_lowers):
        continue
    if subdomain_lowers:
      sdom = str(item.get("sub_domain", "")).lower()
      if not any(sd == sdom or sd in sdom for sd in subdomain_lowers):
        continue
    if min_words > 0 or max_words > 0:
      wc = len(str(item.get("context", "")).split())
      if min_words > 0 and wc < min_words:
        continue
      if max_words > 0 and wc > max_words:
        continue
    filtered.append(item)

  if shuffle:
    rng = random.Random(seed)
    rng.shuffle(filtered)

  if max_samples > 0:
    filtered = filtered[:max_samples]

  return filtered


def probe_vllm_server(
    base_url: str, api_key: str
) -> tuple[list[str], int | None]:
  """Queries GET /v1/models to discover served model IDs and max_model_len."""
  url = f"{base_url.rstrip('/')}/models"
  headers = {"Authorization": f"Bearer {api_key or 'EMPTY'}"}
  try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
      data = json.loads(resp.read().decode("utf-8", errors="replace"))
    models = data.get("data") or []
    served_ids = [str(m["id"]) for m in models if m.get("id")]
    max_model_len = None
    for m in models:
      if m.get("max_model_len") is not None:
        max_model_len = int(m["max_model_len"])
        break
    return served_ids, max_model_len
  except Exception:  # pylint: disable=broad-except
    return [], None


def call_chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    enable_thinking: bool,
    timeout: int,
    max_retries: int,
) -> tuple[str, str, dict[str, Any]]:
  """Calls vLLM OpenAI /v1/chat/completions endpoint with retries.

  Returns:
    (content_text, reasoning_text, usage_dict)
  """
  url = f"{base_url.rstrip('/')}/chat/completions"
  payload: dict[str, Any] = {
      "model": model,
      "messages": messages,
      "max_tokens": max_tokens,
      "temperature": temperature,
      "chat_template_kwargs": {"enable_thinking": enable_thinking},
  }
  if top_p is not None and top_p > 0:
    payload["top_p"] = top_p
  if top_k is not None and top_k > 0:
    payload["top_k"] = top_k
  if presence_penalty is not None:
    payload["presence_penalty"] = presence_penalty
  if repetition_penalty is not None:
    payload["repetition_penalty"] = repetition_penalty

  body_bytes = json.dumps(payload).encode("utf-8")
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key or 'EMPTY'}",
  }

  last_err = None
  for attempt in range(max(1, max_retries + 1)):
    try:
      req = urllib.request.Request(url, data=body_bytes, headers=headers)
      with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
      choice = (data.get("choices") or [{}])[0]
      msg = choice.get("message") or {}
      content = msg.get("content") or ""
      reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
      # If --reasoning-parser=qwen3 placed text into reasoning_content and left
      # content empty (e.g. token limit hit or no </think> delimiter), fall back
      # to reasoning so the answer extractor can still inspect the output.
      if not content.strip() and reasoning.strip():
        content = reasoning
      usage = data.get("usage") or {}
      usage["finish_reason"] = choice.get("finish_reason")
      return content, reasoning, usage
    except urllib.error.HTTPError as e:
      err_body = ""
      try:
        err_body = e.read().decode("utf-8", errors="replace")[:500]
      except Exception:  # pylint: disable=broad-except
        pass
      last_err = RuntimeError(f"HTTP {e.code}: {err_body}")
      # Do not retry 400 Bad Request (e.g. context length exceeded on server)
      if e.code == 400:
        break
    except Exception as e:  # pylint: disable=broad-except
      last_err = e

    if attempt < max_retries:
      backoff = min(30, 2**attempt)
      time.sleep(backoff)

  raise RuntimeError(
      f"Request failed after {max_retries + 1} attempts: {last_err}"
  )


def compute_bucket_stats(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
  """Computes strict and compensated accuracy for a list of evaluated records."""
  evaluated = [
      r for r in records if not r.get("skipped_overlen") and not r.get("error")
  ]
  total = len(evaluated)
  if total == 0:
    return {
        "total": 0,
        "correct": 0,
        "unparsed": 0,
        "accuracy": None,
        "accuracy_pct": None,
        "compensated_accuracy": None,
        "compensated_accuracy_pct": None,
    }
  correct = sum(1 for r in evaluated if r.get("judge") is True)
  unparsed = sum(1 for r in evaluated if r.get("pred") is None)
  acc = correct / total
  comp_acc = (correct + 0.25 * unparsed) / total
  return {
      "total": total,
      "correct": correct,
      "unparsed": unparsed,
      "accuracy": round(acc, 6),
      "accuracy_pct": round(acc * 100.0, 2),
      "compensated_accuracy": round(comp_acc, 6),
      "compensated_accuracy_pct": round(comp_acc * 100.0, 2),
  }


def percentile(values: list[float], q: float) -> float | None:
  if not values:
    return None
  s = sorted(values)
  k = (len(s) - 1) * q
  lo = int(math.floor(k))
  hi = int(math.ceil(k))
  return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 4)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
      description="LongBench v2 Evaluation Harness with full configuration options"
  )

  # Endpoint & model
  p.add_argument(
      "--base_url",
      default=os.environ.get("LB2_BASE_URL", "http://127.0.0.1:8000/v1"),
      help=(
          "OpenAI-compatible v1 base URL (default: $LB2_BASE_URL or"
          " http://127.0.0.1:8000/v1)"
      ),
  )
  p.add_argument(
      "--api_key",
      default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
      help="API key for OpenAI endpoint",
  )
  p.add_argument(
      "--model",
      default=os.environ.get(
          "LB2_MODEL",
          os.environ.get(
              "SERVED_MODEL_NAME",
              os.environ.get(
                  "MODEL_NAME",
                  os.environ.get("MODEL", "Qwen/Qwen3.5-397B-A17B-FP8"),
              ),
          ),
      ),
      help="Served model name for API requests and tokenizer default",
  )
  p.add_argument(
      "--tokenizer",
      default=os.environ.get(
          "LB2_TOKENIZER", os.environ.get("MODEL", "")
      ),
      help="Tokenizer HF repo ID or local path (defaults to $MODEL or --model)",
  )

  # Dataset source
  p.add_argument(
      "--dataset",
      default=os.environ.get("LB2_DATASET", "THUDM/LongBench-v2"),
      help="HuggingFace dataset ID (default: THUDM/LongBench-v2)",
  )
  p.add_argument(
      "--dataset_split",
      default=os.environ.get("LB2_DATASET_SPLIT", "train"),
      help="HuggingFace dataset split (default: train)",
  )
  p.add_argument(
      "--dataset_path",
      default=os.environ.get("LB2_DATASET_PATH", ""),
      help="Optional local or gs:// path to LongBench v2 JSON/JSONL/Parquet file",
  )

  # Subset filtering options
  p.add_argument(
      "--length",
      default=os.environ.get(
          "LB2_LENGTH", os.environ.get("EVAL_LENGTH", "all")
      ),
      help=(
          "Filter by length bucket: all, short, medium, long, or"
          " comma-separated (e.g. short,medium)"
      ),
  )
  p.add_argument(
      "--difficulty",
      default=os.environ.get(
          "LB2_DIFFICULTY", os.environ.get("EVAL_DIFFICULTY", "all")
      ),
      help="Filter by difficulty: all, easy, hard",
  )
  p.add_argument(
      "--domain",
      default=os.environ.get(
          "LB2_DOMAIN", os.environ.get("EVAL_DOMAIN", "all")
      ),
      help="Filter by domain (comma-separated names or substrings, or all)",
  )
  p.add_argument(
      "--sub_domain",
      default=os.environ.get("LB2_SUB_DOMAIN", "all"),
      help="Filter by sub_domain (comma-separated names or substrings, or all)",
  )
  p.add_argument(
      "--ids",
      default=os.environ.get("LB2_IDS", ""),
      help="Filter by specific comma-separated question _id values",
  )
  p.add_argument(
      "--min_words",
      type=int,
      default=env_int("LB2_MIN_WORDS", 0),
      help="Minimum context word count filter (0 = no minimum)",
  )
  p.add_argument(
      "--max_words",
      type=int,
      default=env_int("LB2_MAX_WORDS", 0),
      help="Maximum context word count filter (0 = no maximum)",
  )
  p.add_argument(
      "--max_samples",
      type=int,
      default=env_int("LB2_MAX_SAMPLES", env_int("EVAL_LIMIT", 0)),
      help="Maximum number of filtered questions to evaluate (0 = all)",
  )
  p.add_argument(
      "--shuffle",
      action="store_true",
      default=env_bool("LB2_SHUFFLE", False),
      help="Shuffle filtered questions before applying --max_samples",
  )
  p.add_argument(
      "--seed",
      type=int,
      default=env_int("LB2_SEED", 42),
      help="Random seed when --shuffle is enabled (default: 42)",
  )

  # Context length & truncation options
  p.add_argument(
      "--max_context_length",
      type=int,
      default=env_int(
          "LB2_MAX_CONTEXT_LEN", env_int("EVAL_MAX_CONTEXT_LEN", 0)
      ),
      help=(
          "Maximum prompt length in tokens before truncation/skip (default: 0"
          " = auto-derive from server max_model_len minus completion headroom)"
      ),
  )
  p.add_argument(
      "--truncate_mode",
      choices=["middle", "head", "tail", "skip", "none"],
      default=os.environ.get(
          "LB2_TRUNCATE_MODE",
          os.environ.get("LB2_OVERLEN_POLICY", "middle"),
      ).lower(),
      help=(
          "How to handle prompts exceeding --max_context_length: middle"
          " (official), head, tail, skip, none"
      ),
  )
  p.add_argument(
      "--use_tokenizer",
      action="store_true",
      default=env_bool("LB2_USE_TOKENIZER", True),
      help="Use HF AutoTokenizer for exact token-level truncation (default: true)",
  )
  p.add_argument(
      "--no_use_tokenizer",
      action="store_false",
      dest="use_tokenizer",
      help="Disable HF AutoTokenizer and use character multiplier approximation",
  )
  p.add_argument(
      "--word_mult",
      type=float,
      default=env_float("LB2_WORD_MULT", 3.5) or 3.5,
      help="Chars-per-token multiplier when --no_use_tokenizer is set (default: 3.5)",
  )

  # Prompting & reasoning modes
  p.add_argument(
      "--cot",
      action="store_true",
      default=env_bool("LB2_COT", False),
      help=(
          "Enable official 2-turn Chain-of-Thought prompting"
          " (TEMPLATE_0SHOT_COT + TEMPLATE_0SHOT_COT_ANS)"
      ),
  )
  p.add_argument(
      "--no_context",
      action="store_true",
      default=env_bool("LB2_NO_CONTEXT", False),
      help=(
          "Evaluate without document context (TEMPLATE_0SHOT_NO_CONTEXT"
          " memorization baseline)"
      ),
  )
  default_enable_thinking = env_bool(
      "LB2_ENABLE_THINKING", env_bool("ENABLE_THINKING", True)
  )
  p.add_argument(
      "--enable_thinking",
      action="store_true",
      default=default_enable_thinking,
      help=(
          "Enable thinking mode via chat_template_kwargs.enable_thinking=true"
          " (default: true)"
      ),
  )
  p.add_argument(
      "--no_enable_thinking",
      action="store_false",
      dest="enable_thinking",
      help=(
          "Disable thinking mode"
          " (chat_template_kwargs.enable_thinking=false)"
      ),
  )
  p.add_argument(
      "--system_prompt",
      default=os.environ.get("LB2_SYSTEM_PROMPT", ""),
      help="Optional system message prepended to messages",
  )
  p.add_argument(
      "--strict_extract",
      action="store_true",
      default=env_bool("LB2_STRICT_EXTRACT", False),
      help="Only accept exact 'The correct answer is (X)' pattern without fallback regexes",
  )

  # Generation / sampling hyperparameters
  default_max_new = 32768 if default_enable_thinking else 8192
  p.add_argument(
      "--max_new_tokens",
      type=int,
      default=env_int(
          "LB2_MAX_NEW_TOKENS", env_int("EVAL_MAX_TOKENS", default_max_new)
      ),
      help=(
          "Max generated tokens for answer turn (default: 32768 when"
          " enable_thinking is on, 8192 when off)"
      ),
  )
  p.add_argument(
      "--cot_max_new_tokens",
      type=int,
      default=env_int("LB2_COT_MAX_NEW_TOKENS", default_max_new),
      help=(
          "Max generated tokens for CoT Turn 1 (default: 32768 when"
          " enable_thinking is on, 8192 when off)"
      ),
  )
  p.add_argument(
      "--temperature",
      type=float,
      default=env_float("LB2_TEMPERATURE", 0.1)
      if env_float("LB2_TEMPERATURE", 0.1) is not None
      else 0.1,
      help=(
          "Sampling temperature (default: 0.1 matching official pred.py; use"
          " 0.0 for greedy)"
      ),
  )
  p.add_argument(
      "--top_p",
      type=float,
      default=env_float("LB2_TOP_P", None),
      help="Optional top_p nucleus sampling parameter",
  )
  p.add_argument(
      "--top_k",
      type=int,
      default=env_int("LB2_TOP_K", 0) or None,
      help="Optional top_k sampling parameter",
  )
  p.add_argument(
      "--presence_penalty",
      type=float,
      default=env_float("LB2_PRESENCE_PENALTY", None),
      help="Optional presence_penalty parameter",
  )
  p.add_argument(
      "--repetition_penalty",
      type=float,
      default=env_float("LB2_REPETITION_PENALTY", None),
      help="Optional repetition_penalty parameter",
  )

  # Execution, concurrency & scoring
  p.add_argument(
      "--n_proc",
      "--concurrency",
      dest="n_proc",
      type=int,
      default=env_int(
          "LB2_CONCURRENCY",
          env_int(
              "LB2_N_PROC",
              env_int("EVAL_CONCURRENT_REQUESTS", env_int("CONC", 16)),
          ),
      ),
      help="Number of concurrent evaluation workers (default: 16)",
  )
  p.add_argument(
      "--client_timeout",
      type=int,
      default=env_int("LB2_CLIENT_TIMEOUT", 3600),
      help="Per-request HTTP timeout in seconds (default: 3600)",
  )
  p.add_argument(
      "--max_retries",
      type=int,
      default=env_int("LB2_MAX_RETRIES", 3),
      help="Number of retries per request on transient errors (default: 3)",
  )
  p.add_argument(
      "--compensated",
      action="store_true",
      default=env_bool("LB2_COMPENSATED", False),
      help=(
          "Use compensated accuracy (+0.25 for unparsed answers) as primary"
          " headline score"
      ),
  )
  p.add_argument(
      "--resume",
      action="store_true",
      default=env_bool("LB2_RESUME", True),
      help=(
          "Resume from existing predictions file in output_dir if present"
          " (default: true)"
      ),
  )
  p.add_argument(
      "--no_resume",
      action="store_false",
      dest="resume",
      help="Start fresh and overwrite any existing predictions in output_dir",
  )
  p.add_argument(
      "--min_score",
      type=float,
      default=env_float("LB2_MIN_SCORE", 0.0) or 0.0,
      help=(
          "Optional minimum accuracy threshold (0..1 or 0..100); exit non-zero"
          " if below"
      ),
  )
  p.add_argument(
      "--output_dir",
      default=os.environ.get("RESULT_DIR", "/tmp/longbench_v2_results"),
      help="Directory to write predictions and summary JSON files",
  )
  p.add_argument(
      "--aggregate_output_file",
      default=os.environ.get("LB2_AGGREGATE_FILE", ""),
      help=(
          "Optional path to write CDK aggregate JSON (e.g."
          " ${AGENTIC_OUTPUT_DIR}/${RESULT_FILENAME}.json)"
      ),
  )

  return p.parse_args()


def main() -> int:
  args = parse_args()
  t_start = time.time()

  if not args.tokenizer:
    args.tokenizer = args.model

  served_ids, server_max_model_len = probe_vllm_server(
      args.base_url, args.api_key
  )
  if served_ids and args.model not in served_ids:
    print(
        f"[LongBench-v2] Server exposes models {served_ids}; using"
        f" '{served_ids[0]}' for API requests (tokenizer='{args.tokenizer}').",
        flush=True,
    )
    args.model = served_ids[0]

  if args.max_context_length <= 0:
    base_max_len = server_max_model_len or env_int("MAX_MODEL_LEN", 262144)
    reserve = args.max_new_tokens + (
        args.cot_max_new_tokens + 2048 if args.cot else 2048
    )
    args.max_context_length = max(1024, base_max_len - reserve)
    print(
        f"[LongBench-v2] Auto-derived max_context_length={args.max_context_length}"
        f" (server max_model_len={base_max_len}, reserved={reserve}).",
        flush=True,
    )

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  pred_jsonl_path = out_dir / "longbench_v2_predictions.jsonl"
  pred_json_path = out_dir / "longbench_v2_predictions.json"
  summary_json_path = out_dir / "longbench_v2_summary.json"

  lengths = parse_csv_set(args.length)
  difficulties = parse_csv_set(args.difficulty)
  domains = parse_csv_set(args.domain)
  sub_domains = parse_csv_set(args.sub_domain)
  ids = parse_csv_set(args.ids)

  print("=" * 78, flush=True)
  print("LongBench v2 Evaluation Configuration", flush=True)
  print("=" * 78, flush=True)
  print(f"  Endpoint           : {args.base_url}", flush=True)
  print(f"  Model              : {args.model}", flush=True)
  print(
      "  Dataset            :"
      f" {args.dataset_path or f'{args.dataset} ({args.dataset_split})'}",
      flush=True,
  )
  print(
      f"  Subset Filters     : length={lengths or 'ALL'},"
      f" difficulty={difficulties or 'ALL'}, domain={domains or 'ALL'},"
      f" sub_domain={sub_domains or 'ALL'}, ids={len(ids) or 'ALL'},"
      f" words=[{args.min_words or 0}, {args.max_words or 'inf'}]",
      flush=True,
  )
  print(
      f"  Sample Cap         : max_samples={args.max_samples or 'ALL'}"
      f" (shuffle={args.shuffle}, seed={args.seed})",
      flush=True,
  )
  print(
      f"  Context & Truncate : max_context_length={args.max_context_length}"
      f" tokens, truncate_mode={args.truncate_mode},"
      f" use_tokenizer={args.use_tokenizer}",
      flush=True,
  )
  print(
      f"  Prompting Mode     : cot={args.cot}, no_context={args.no_context},"
      f" enable_thinking={args.enable_thinking},"
      f" strict_extract={args.strict_extract}",
      flush=True,
  )
  print(
      f"  Generation Knobs   : max_new_tokens={args.max_new_tokens},"
      f" cot_max_new_tokens={args.cot_max_new_tokens},"
      f" temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}",
      flush=True,
  )
  print(
      f"  Execution & Score  : concurrency(n_proc)={args.n_proc},"
      f" timeout={args.client_timeout}s, retries={args.max_retries},"
      f" compensated={args.compensated}",
      flush=True,
  )
  print(f"  Output Directory   : {out_dir}", flush=True)
  print("=" * 78, flush=True)

  # Load dataset & apply filters
  raw_dataset = load_longbench_v2_dataset(
      args.dataset, args.dataset_split, args.dataset_path
  )
  examples = filter_dataset(
      raw_dataset,
      lengths=lengths,
      difficulties=difficulties,
      domains=domains,
      sub_domains=sub_domains,
      ids=ids,
      min_words=args.min_words,
      max_words=args.max_words,
      shuffle=args.shuffle,
      seed=args.seed,
      max_samples=args.max_samples,
  )
  print(
      f"[LongBench-v2] Selected {len(examples)} / {len(raw_dataset)} questions"
      " after filtering.",
      flush=True,
  )
  if not examples:
    print(
        "[LongBench-v2] ERROR: 0 questions matched the specified filters!",
        flush=True,
    )
    return 1

  # Load tokenizer if enabled
  tokenizer = None
  if args.use_tokenizer and not args.no_context:
    tokenizer = load_tokenizer(args.tokenizer, args.model)

  # Check for existing predictions if --resume is active
  completed_by_id: dict[str, dict[str, Any]] = {}
  if args.resume and pred_jsonl_path.is_file():
    for line in pred_jsonl_path.read_text(encoding="utf-8").splitlines():
      if not line.strip():
        continue
      try:
        rec = json.loads(line)
        if rec.get("_id") and not rec.get("error"):
          completed_by_id[str(rec["_id"])] = rec
      except Exception:  # pylint: disable=broad-except
        pass
    if completed_by_id:
      print(
          f"[LongBench-v2] Resuming: found {len(completed_by_id)} completed"
          f" predictions in {pred_jsonl_path}",
          flush=True,
      )
  elif not args.resume and pred_jsonl_path.is_file():
    pred_jsonl_path.unlink()

  # Determine base template
  if args.no_context:
    base_template = TEMPLATE_0SHOT_NO_CONTEXT
  elif args.cot:
    base_template = TEMPLATE_0SHOT_COT
  else:
    base_template = TEMPLATE_0SHOT

  write_lock = threading.Lock()
  results_by_idx: dict[int, dict[str, Any]] = {}
  progress_count = 0
  total_selected = len(examples)

  def build_messages(user_prompt: str) -> list[dict[str, str]]:
    msgs = []
    if args.system_prompt:
      msgs.append({"role": "system", "content": args.system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs

  def evaluate_one(idx: int, item: dict[str, Any]) -> dict[str, Any]:
    nonlocal progress_count
    qid = str(item.get("_id", f"idx_{idx}"))
    if qid in completed_by_id:
      with write_lock:
        progress_count += 1
      return completed_by_id[qid]

    t0 = time.time()
    gold = str(item.get("answer", "")).strip().upper()
    word_count = len(str(item.get("context", "")).split())

    try:
      raw_prompt = format_prompt(base_template, item)
      prompt, raw_toks, final_toks, was_trunc, was_skipped = truncate_prompt(
          raw_prompt,
          max_len=args.max_context_length,
          truncate_mode=args.truncate_mode,
          tokenizer=tokenizer,
          word_mult=args.word_mult,
      )

      if was_skipped:
        rec = {
            "_id": qid,
            "domain": item.get("domain"),
            "sub_domain": item.get("sub_domain"),
            "difficulty": item.get("difficulty"),
            "length": item.get("length"),
            "word_count": word_count,
            "raw_prompt_tokens": raw_toks,
            "prompt_tokens": final_toks,
            "truncated": False,
            "skipped_overlen": True,
            "answer": gold,
            "pred": None,
            "judge": False,
            "latency_s": round(time.time() - t0, 3),
        }
        with write_lock:
          progress_count += 1
          with open(pred_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
          print(
              f"[{progress_count}/{total_selected}] SKIP (overlen"
              f" {raw_toks} > {args.max_context_length}) id={qid}"
              f" len={item.get('length')} diff={item.get('difficulty')}",
              flush=True,
          )
        return rec

      cot_response = None
      total_prompt_tokens = final_toks
      total_completion_tokens = 0

      if args.cot and not args.no_context:
        # Turn 1: Generate step-by-step CoT reasoning
        cot_text, _, cot_usage = call_chat_completions(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            messages=build_messages(prompt),
            max_tokens=args.cot_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            enable_thinking=args.enable_thinking,
            timeout=args.client_timeout,
            max_retries=args.max_retries,
        )
        cot_response = strip_thinking(cot_text)
        total_completion_tokens += int(cot_usage.get("completion_tokens") or 0)

        # Turn 2: Extract final choice using TEMPLATE_0SHOT_COT_ANS
        ans_raw = format_prompt(TEMPLATE_0SHOT_COT_ANS, item).replace(
            "$COT_ANS$", cot_response
        )
        ans_prompt, _, ans_toks, _, _ = truncate_prompt(
            ans_raw,
            max_len=args.max_context_length,
            truncate_mode=args.truncate_mode,
            tokenizer=tokenizer,
            word_mult=args.word_mult,
        )
        response_text, reasoning_text, usage = call_chat_completions(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            messages=build_messages(ans_prompt),
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            enable_thinking=args.enable_thinking,
            timeout=args.client_timeout,
            max_retries=args.max_retries,
        )
        total_prompt_tokens = int(usage.get("prompt_tokens") or ans_toks)
        total_completion_tokens += int(usage.get("completion_tokens") or 0)
      else:
        response_text, reasoning_text, usage = call_chat_completions(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            messages=build_messages(prompt),
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            enable_thinking=args.enable_thinking,
            timeout=args.client_timeout,
            max_retries=args.max_retries,
        )
        total_prompt_tokens = int(usage.get("prompt_tokens") or final_toks)
        total_completion_tokens = int(usage.get("completion_tokens") or 0)

      pred = extract_answer(response_text, strict=args.strict_extract)
      if pred is None and reasoning_text:
        # Fallback if the model wrote "The correct answer is (X)" inside reasoning_content
        pred = extract_answer(reasoning_text, strict=True)
      is_correct = pred == gold
      elapsed = round(time.time() - t0, 3)

      rec = {
          "_id": qid,
          "domain": item.get("domain"),
          "sub_domain": item.get("sub_domain"),
          "difficulty": item.get("difficulty"),
          "length": item.get("length"),
          "word_count": word_count,
          "raw_prompt_tokens": raw_toks,
          "prompt_tokens": total_prompt_tokens,
          "completion_tokens": total_completion_tokens,
          "truncated": was_trunc,
          "skipped_overlen": False,
          "answer": gold,
          "pred": pred,
          "judge": is_correct,
          "response": response_text,
          "reasoning_content": (
              reasoning_text[:2000] if reasoning_text else None
          ),
          "cot_response": cot_response[:2000] if cot_response else None,
          "finish_reason": usage.get("finish_reason"),
          "latency_s": elapsed,
      }
      with write_lock:
        progress_count += 1
        with open(pred_jsonl_path, "a", encoding="utf-8") as f:
          f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = (
            "OK" if is_correct else ("UNPARSED" if pred is None else "WRONG")
        )
        print(
            f"[{progress_count}/{total_selected}] {status:8s} id={qid}"
            f" len={str(item.get('length')):6s}"
            f" diff={str(item.get('difficulty')):4s} gold={gold} pred={pred}"
            f" toks={total_prompt_tokens}{'(trunc)' if was_trunc else ''}"
            f" out={total_completion_tokens} lat={elapsed:.1f}s",
            flush=True,
        )
      return rec

    except Exception as e:  # pylint: disable=broad-except
      elapsed = round(time.time() - t0, 3)
      rec = {
          "_id": qid,
          "domain": item.get("domain"),
          "sub_domain": item.get("sub_domain"),
          "difficulty": item.get("difficulty"),
          "length": item.get("length"),
          "word_count": word_count,
          "answer": gold,
          "pred": None,
          "judge": False,
          "error": str(e),
          "latency_s": elapsed,
      }
      with write_lock:
        progress_count += 1
        print(
            f"[{progress_count}/{total_selected}] ERROR    id={qid}"
            f" len={item.get('length')} diff={item.get('difficulty')}: {e}",
            flush=True,
        )
      return rec

  # Run evaluation across worker pool
  with ThreadPoolExecutor(max_workers=max(1, args.n_proc)) as pool:
    future_to_idx = {
        pool.submit(evaluate_one, i, ex): i for i, ex in enumerate(examples)
    }
    for fut in as_completed(future_to_idx):
      idx = future_to_idx[fut]
      results_by_idx[idx] = fut.result()

  ordered_results = [results_by_idx[i] for i in range(len(examples))]
  wall_time_s = round(time.time() - t_start, 2)

  # Compute overall and slice breakdowns
  overall = compute_bucket_stats(ordered_results)
  by_difficulty = {
      d: compute_bucket_stats([
          r
          for r in ordered_results
          if str(r.get("difficulty", "")).lower() == d
      ])
      for d in ("easy", "hard")
  }
  by_length = {
      l: compute_bucket_stats(
          [r for r in ordered_results if str(r.get("length", "")).lower() == l]
      )
      for l in ("short", "medium", "long")
  }
  domain_names = sorted({
      str(r.get("domain", "Unknown"))
      for r in ordered_results
      if r.get("domain")
  })
  by_domain = {
      dom: compute_bucket_stats(
          [r for r in ordered_results if str(r.get("domain", "")) == dom]
      )
      for dom in domain_names
  }

  evaluated_recs = [
      r
      for r in ordered_results
      if not r.get("skipped_overlen") and not r.get("error")
  ]
  latencies = [
      float(r["latency_s"])
      for r in evaluated_recs
      if isinstance(r.get("latency_s"), (int, float))
  ]
  prompt_toks = [
      int(r["prompt_tokens"])
      for r in evaluated_recs
      if isinstance(r.get("prompt_tokens"), (int, float))
  ]
  comp_toks = [
      int(r["completion_tokens"])
      for r in evaluated_recs
      if isinstance(r.get("completion_tokens"), (int, float))
  ]

  skipped_overlen = sum(1 for r in ordered_results if r.get("skipped_overlen"))
  truncated_count = sum(1 for r in evaluated_recs if r.get("truncated"))
  error_count = sum(1 for r in ordered_results if r.get("error"))

  primary_acc = (
      overall["compensated_accuracy"]
      if args.compensated
      else overall["accuracy"]
  )
  primary_acc_pct = (
      overall["compensated_accuracy_pct"]
      if args.compensated
      else overall["accuracy_pct"]
  )

  summary = {
      "benchmark": "longbench_v2",
      "model": args.model,
      "config": {
          "dataset": args.dataset_path or args.dataset,
          "dataset_split": args.dataset_split,
          "length_filter": lengths or "all",
          "difficulty_filter": difficulties or "all",
          "domain_filter": domains or "all",
          "sub_domain_filter": sub_domains or "all",
          "ids_filter": ids or "all",
          "min_words": args.min_words,
          "max_words": args.max_words,
          "max_samples": args.max_samples,
          "shuffle": args.shuffle,
          "seed": args.seed,
          "max_context_length": args.max_context_length,
          "truncate_mode": args.truncate_mode,
          "use_tokenizer": args.use_tokenizer,
          "cot": args.cot,
          "no_context": args.no_context,
          "enable_thinking": args.enable_thinking,
          "strict_extract": args.strict_extract,
          "max_new_tokens": args.max_new_tokens,
          "cot_max_new_tokens": args.cot_max_new_tokens,
          "temperature": args.temperature,
          "top_p": args.top_p,
          "top_k": args.top_k,
          "n_proc": args.n_proc,
          "compensated": args.compensated,
      },
      "counts": {
          "dataset_total": len(raw_dataset),
          "selected": len(ordered_results),
          "evaluated": overall["total"],
          "correct": overall["correct"],
          "unparsed": overall["unparsed"],
          "truncated": truncated_count,
          "skipped_overlen": skipped_overlen,
          "errors": error_count,
      },
      "overall": overall,
      "by_difficulty": by_difficulty,
      "by_length": by_length,
      "by_domain": by_domain,
      "performance": {
          "wall_time_s": wall_time_s,
          "mean_latency_s": (
              round(sum(latencies) / len(latencies), 3) if latencies else None
          ),
          "p50_latency_s": percentile(latencies, 0.50),
          "p90_latency_s": percentile(latencies, 0.90),
          "mean_prompt_tokens": (
              round(sum(prompt_toks) / len(prompt_toks), 1)
              if prompt_toks
              else None
          ),
          "max_prompt_tokens": max(prompt_toks) if prompt_toks else None,
          "mean_completion_tokens": (
              round(sum(comp_toks) / len(comp_toks), 1) if comp_toks else None
          ),
      },
  }

  pred_json_path.write_text(
      json.dumps(ordered_results, indent=2, ensure_ascii=False),
      encoding="utf-8",
  )
  summary_json_path.write_text(
      json.dumps(summary, indent=2, ensure_ascii=False),
      encoding="utf-8",
  )

  # Write flat CDK aggregate metrics file for cdk.agentx metricsParser / result-updater
  cdk_agg = {
      "conc": args.n_proc,
      "score": primary_acc,
      "n_eff": overall["total"],
      "num_requests_total": len(ordered_results),
      "num_requests_successful": overall["total"],
      "lb2_accuracy": overall["accuracy"],
      "lb2_accuracy_pct": overall["accuracy_pct"],
      "lb2_compensated_accuracy": overall["compensated_accuracy"],
      "lb2_compensated_accuracy_pct": overall["compensated_accuracy_pct"],
      "lb2_easy_accuracy_pct": by_difficulty["easy"]["accuracy_pct"],
      "lb2_hard_accuracy_pct": by_difficulty["hard"]["accuracy_pct"],
      "lb2_short_accuracy_pct": by_length["short"]["accuracy_pct"],
      "lb2_medium_accuracy_pct": by_length["medium"]["accuracy_pct"],
      "lb2_long_accuracy_pct": by_length["long"]["accuracy_pct"],
      "lb2_easy_compensated_pct": by_difficulty["easy"][
          "compensated_accuracy_pct"
      ],
      "lb2_hard_compensated_pct": by_difficulty["hard"][
          "compensated_accuracy_pct"
      ],
      "lb2_short_compensated_pct": by_length["short"][
          "compensated_accuracy_pct"
      ],
      "lb2_medium_compensated_pct": by_length["medium"][
          "compensated_accuracy_pct"
      ],
      "lb2_long_compensated_pct": by_length["long"]["compensated_accuracy_pct"],
      "lb2_total_questions": len(ordered_results),
      "lb2_evaluated": overall["total"],
      "lb2_correct": overall["correct"],
      "lb2_unparsed": overall["unparsed"],
      "lb2_truncated_count": truncated_count,
      "lb2_skipped_overlen": skipped_overlen,
      "lb2_errors": error_count,
      "lb2_wall_time_s": wall_time_s,
      "lb2_mean_latency_s": summary["performance"]["mean_latency_s"],
      "lb2_p50_latency_s": summary["performance"]["p50_latency_s"],
      "lb2_p90_latency_s": summary["performance"]["p90_latency_s"],
      "lb2_mean_prompt_tokens": summary["performance"]["mean_prompt_tokens"],
      "lb2_max_prompt_tokens": summary["performance"]["max_prompt_tokens"],
      "lb2_mean_completion_tokens": summary["performance"][
          "mean_completion_tokens"
      ],
  }
  if args.aggregate_output_file:
    agg_path = Path(args.aggregate_output_file)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if agg_path.is_file():
      try:
        existing = json.loads(agg_path.read_text(encoding="utf-8") or "{}")
      except Exception:  # pylint: disable=broad-except
        existing = {}
    for k, v in cdk_agg.items():
      if v is not None and k not in existing:
        existing[k] = v
    agg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

  # Print formatted summary report
  def fmt_bucket(b: dict[str, Any]) -> str:
    if not b["total"]:
      return "   N/A   (  0/  0)"
    return (
        f"{b['accuracy_pct']:6.2f}% ({b['correct']:3d}/{b['total']:3d})"
        f"  [comp: {b['compensated_accuracy_pct']:6.2f}%,"
        f" unparsed: {b['unparsed']}]"
    )

  print("\n" + "=" * 78, flush=True)
  print(f"LongBench v2 Final Results — {args.model}", flush=True)
  print("=" * 78, flush=True)
  print(
      f"  Overall               : {fmt_bucket(overall)}",
      flush=True,
  )
  print("-" * 78, flush=True)
  print(
      f"  Difficulty :: Easy    : {fmt_bucket(by_difficulty['easy'])}",
      flush=True,
  )
  print(
      f"  Difficulty :: Hard    : {fmt_bucket(by_difficulty['hard'])}",
      flush=True,
  )
  print("-" * 78, flush=True)
  print(
      f"  Length     :: Short   : {fmt_bucket(by_length['short'])}"
      "   (<32k words)",
      flush=True,
  )
  print(
      f"  Length     :: Medium  : {fmt_bucket(by_length['medium'])}"
      "   (32k-128k words)",
      flush=True,
  )
  print(
      f"  Length     :: Long    : {fmt_bucket(by_length['long'])}"
      "   (128k-2M words)",
      flush=True,
  )
  if by_domain:
    print("-" * 78, flush=True)
    for dom, b in by_domain.items():
      print(f"  Domain :: {dom[:32]:32s}: {fmt_bucket(b)}", flush=True)
  print("-" * 78, flush=True)
  print(
      f"  Wall Time             : {wall_time_s:.1f}s"
      f" ({wall_time_s / 60.0:.1f} min) | Concurrency: {args.n_proc}",
      flush=True,
  )
  print(
      "  Token Stats           :"
      f" mean_prompt={summary['performance']['mean_prompt_tokens']},"
      f" max_prompt={summary['performance']['max_prompt_tokens']},"
      f" mean_output={summary['performance']['mean_completion_tokens']},"
      f" truncated={truncated_count}, skipped={skipped_overlen},"
      f" errors={error_count}",
      flush=True,
  )
  print(f"  Saved Predictions     : {pred_jsonl_path}", flush=True)
  print(f"  Saved Summary         : {summary_json_path}", flush=True)
  print("=" * 78, flush=True)

  if overall["total"] == 0:
    print(
        "[LongBench-v2] FAIL: 0 examples were successfully evaluated.",
        flush=True,
    )
    return 1

  if args.min_score > 0:
    # Support threshold specified either as 0..1 (e.g. 0.60) or 0..100 (e.g. 60.0)
    threshold_pct = (
        args.min_score * 100.0 if args.min_score <= 1.0 else args.min_score
    )
    if (primary_acc_pct or 0.0) < threshold_pct:
      print(
          f"[LongBench-v2] FAIL: accuracy {primary_acc_pct:.2f}% <"
          f" min_score {threshold_pct:.2f}%",
          flush=True,
      )
      return 1

  return 0


if __name__ == "__main__":
  sys.exit(main())
