#!/usr/bin/env python3
"""Verify a page-sized P/D transfer and a deterministic decode result."""

import argparse
import json
import urllib.error
import urllib.request


def post_json(url: str, payload: dict, timeout: int) -> object:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(
            f"{url} returned HTTP {error.code}: {body[:8000]}") from error


def render_chat_tokens(url: str, model: str, text: str,
                       timeout: int) -> list[int]:
    rendered = post_json(
        url,
        {
            "model": model,
            "messages": [{
                "role": "user",
                "content": text
            }],
            "max_tokens": 1,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
        },
        timeout,
    )
    if not isinstance(rendered, dict):
        raise RuntimeError("chat render response was not an object")
    token_ids = rendered.get("token_ids")
    if not isinstance(token_ids, list) or not all(
            type(token_id) is int for token_id in token_ids):
        raise RuntimeError("render response did not contain integer token IDs")
    return token_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=8000)
    parser.add_argument("--prefill-port", type=int, default=8400)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=4608)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    render_url = (
        f"http://{args.host}:{args.prefill_port}/v1/chat/completions/render")
    proxy_url = f"http://{args.host}:{args.proxy_port}/v1/chat/completions"
    question = "What is two plus two? Answer only the number."
    fixture_lines: list[str] = []
    rendered_ids: list[int] = []
    while len(rendered_ids) < args.target_prompt_tokens:
        start = len(fixture_lines)
        fixture_lines.extend(
            f"Reference row {index:04d}: the archive color is blue."
            for index in range(start, start + 64))
        prompt = "\n".join((*fixture_lines, "", question))
        rendered_ids = render_chat_tokens(render_url, args.model, prompt,
                                          args.timeout)
        if len(rendered_ids) > 8191:
            raise RuntimeError(
                "chat prompt exceeded the model context before reaching the "
                "requested transfer size")

    result = post_json(
        proxy_url,
        {
            "model": args.model,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
        },
        args.timeout,
    )
    if not isinstance(result, dict):
        raise RuntimeError("completion response was not a JSON object")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("completion response contained no choices")
    message = choices[0].get("message") or {}
    text = str(message.get("content", "")).strip()
    finish_reason = choices[0].get("finish_reason")
    usage = result.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0)
    first_word = text.split(maxsplit=1)[0].rstrip(".,") if text else ""
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    ok = (first_word == "4" and prompt_tokens >= args.target_prompt_tokens
          and cached_tokens > 0)

    print(
        json.dumps({
            "target_prompt_tokens": args.target_prompt_tokens,
            "rendered_prompt_tokens": len(rendered_ids),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": finish_reason,
            "text": text,
        }),
        flush=True,
    )
    print(f"TRANSFER_PROBE_OK {int(ok)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
