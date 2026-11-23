"""Download one build's selector recordings through the Buildkite API.

PROTOTYPE. Needs BUILDKITE_API_TOKEN with the read_artifacts scope.

    python3 download_artifacts.py BUILD_NUMBER --out recordings/
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import ssl
import sys
import urllib.error
import urllib.request

API = "https://api.buildkite.com/v2/organizations/vllm/pipelines/{pipeline}/builds/{build}"
PREFIX = "selector_recordings/"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _ssl_context() -> ssl.SSLContext:
    """python.org's macOS Python ships without CA certificates; certifi has them."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_CTX = _ssl_context()


def _get(url: str, token: str) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    opener = urllib.request.build_opener(
        _NoRedirect, urllib.request.HTTPSHandler(context=_CTX)
    )
    try:
        with opener.open(req) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read(), dict(err.headers)


def list_artifacts(base: str, token: str) -> list[dict]:
    artifacts, page = [], 1
    while True:
        status, body, _ = _get(f"{base}/artifacts?per_page=100&page={page}", token)
        if status != 200:
            sys.exit(f"listing artifacts failed: HTTP {status}: {body[:300]!r}")
        batch = json.loads(body)
        if not batch:
            return artifacts
        artifacts += batch
        page += 1


def download(artifact: dict, out: pathlib.Path, token: str) -> str:
    dest = out / artifact["path"]
    if dest.exists() and dest.stat().st_size == artifact.get("file_size", -1):
        return "cached"
    status, body, headers = _get(artifact["download_url"], token)
    if status in (301, 302, 303, 307, 308):
        # The file lives in object storage; fetch it without our token.
        # Header names arrive in either case; some responses carry the link
        # in a JSON body instead.
        lower = {k.lower(): v for k, v in headers.items()}
        location = lower.get("location")
        if not location:
            try:
                location = json.loads(body).get("url")
            except ValueError:
                location = None
        if not location:
            raise RuntimeError(f"{artifact['path']}: HTTP {status} with no link: {body[:200]!r}")
        with urllib.request.urlopen(location, context=_CTX) as resp:
            body = resp.read()
    elif status != 200:
        raise RuntimeError(f"{artifact['path']}: HTTP {status}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("build", type=int)
    ap.add_argument("--pipeline", default="vllm-torchtpu-dev")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("recordings"))
    args = ap.parse_args(argv)
    token = os.environ.get("BUILDKITE_API_TOKEN")
    if not token:
        sys.exit("set BUILDKITE_API_TOKEN (Buildkite > Personal Settings > API Access "
                 "Tokens, with read_artifacts)")
    base = API.format(pipeline=args.pipeline, build=args.build)
    wanted = [a for a in list_artifacts(base, token) if a["path"].startswith(PREFIX)]
    print(f"{len(wanted)} recording files in build {args.build}")
    counts: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        for result in pool.map(lambda a: download(a, args.out, token), wanted):
            counts[result] = counts.get(result, 0) + 1
    print(counts, "->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
