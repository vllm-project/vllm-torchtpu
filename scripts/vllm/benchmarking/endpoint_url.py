#!/usr/bin/env python3
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
"""Endpoint URL normalization and `/v1/models` lookup for benchmark scripts."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from urllib.parse import urlparse


def normalize_base_url(raw: str) -> str:
    """Validate and normalize an OpenAI-compatible API root URL.

    Accepts a bare host (defaulting to ``http://<host>:8000``) or an
    ``http://`` / ``https://`` URL with an optional port and path prefix.
    Strips a trailing ``/`` and a trailing ``/v1`` from the path only, so
    clients can append their own ``/v1/...`` routes without losing a host
    named ``v1`` or doubling an endpoint route.
    """
    text = raw.strip()
    if not text or any(ch.isspace() for ch in text):
        raise ValueError(f"Invalid --base-url: {raw}")

    bare = "://" not in text
    try:
        url = urlparse("http://" + text if bare else text)
        port = url.port
    except ValueError as err:
        raise ValueError(f"Invalid --base-url: {raw}") from err

    # Credentials in the URL would be echoed into logs with every command.
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise ValueError(f"Invalid --base-url: {raw}")

    path = url.path.rstrip("/")
    if path == "/v1" or path.endswith("/v1"):
        path = path[: -len("/v1")].rstrip("/")
    if "/v1/" in f"{path}/":
        raise ValueError(
            f"Invalid --base-url: {raw} (pass the API root, not a /v1/... route)"
        )

    netloc = url.netloc
    if bare and port is None:
        netloc = f"{netloc}:8000"
    return f"{url.scheme}://{netloc}{path}"


def discover_served_model(
    base_url: str, *, api_key: str | None = None, timeout: float = 30.0
) -> str | None:
    """Return the single model id advertised at ``<base_url>/v1/models``.

    Returns ``None`` if the endpoint is unreachable, fails authentication, or
    advertises zero or multiple models.
    """
    models_url = f"{base_url}/v1/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(models_url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.load(resp)
        data = payload.get("data")
        if not isinstance(data, list):
            return None
        ids = [
            entry["id"] for entry in data if isinstance(entry, dict) and "id" in entry
        ]
    except Exception as err:
        print(
            f"Could not list the served models at {models_url}: {err}",
            file=sys.stderr,
        )
        return None
    if len(ids) == 1 and isinstance(ids[0], str):
        return ids[0]
    return None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] not in ("normalize", "discover-model"):
        print(
            "Usage: endpoint_url.py <normalize|discover-model> <url>",
            file=sys.stderr,
        )
        return 2
    command, value = args
    if command == "normalize":
        try:
            print(normalize_base_url(value))
        except ValueError as err:
            print(str(err), file=sys.stderr)
            return 1
        return 0
    model_id = discover_served_model(
        value, api_key=os.environ.get("OPENAI_API_KEY") or None
    )
    if model_id:
        print(model_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
