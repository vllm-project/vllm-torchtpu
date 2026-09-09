#!/usr/bin/env bash
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
#
# Refresh the vendored XSpace schema and regenerate its Python bindings.
#
#   ./scripts/update_xplane_proto.sh            # regenerate from the local .proto
#   ./scripts/update_xplane_proto.sh --fetch    # pull upstream first, then regenerate
#
# Run this after changing xplane.proto.
#
# The generator is pinned on purpose. Protobuf refuses to load gencode newer
# than the runtime, and the runtime differs per environment -- the CI image
# ships protobuf 6.33.6 while a local venv resolves 7.36.0. libprotoc 25.1
# predates the version gate, so its output stamps no runtime floor at all and
# loads under every runtime we build against. Regenerating with a newer protoc
# bakes in a floor and breaks CI; the check at the end of this script catches
# that before it is committed.
set -euo pipefail

PROTOC_TOOLCHAIN="grpcio-tools==1.62.3"  # bundles libprotoc 25.1
PROTOC_PYTHON="3.12"                     # newest interpreter with wheels for it

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_REL="vllm_torchtpu/tools/xplane.proto"
PROTO_PATH="$REPO_DIR/src/$PROTO_REL"
UPSTREAM_RAW="https://raw.githubusercontent.com/openxla/xla/main/third_party/tsl/tsl/profiler/protobuf/xplane.proto"
UPSTREAM_API="https://api.github.com/repos/openxla/xla/commits?path=third_party/tsl/tsl/profiler/protobuf/xplane.proto&per_page=1"

if [[ "${1:-}" == "--fetch" ]]; then
    echo "Fetching upstream xplane.proto..."
    rev="$(curl -fsSL "$UPSTREAM_API" |
        python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["sha"])')"
    body="$(curl -fsSL "$UPSTREAM_RAW")"
    # Keep our provenance header (everything above the upstream copyright
    # banner) and replace only the schema below it.
    header="$(sed -n '1,/^$/p' "$PROTO_PATH" | sed "s/^\/\/ Revision: .*/\/\/ Revision: $rev/")"
    printf '%s\n%s\n' "$header" "$body" >"$PROTO_PATH"
    echo "Updated to revision $rev"
fi

# protoc is invoked with -I src so the descriptor pool registers this file as
# `vllm_torchtpu/tools/xplane.proto`. A bare `xplane.proto` key would risk a
# "file already in pool" collision with any other library that vendors the same
# schema into the same process.
if ! command -v uv >/dev/null; then
    echo "uv is required to run the pinned generator; see the repo README." >&2
    exit 1
fi

echo "Generating bindings with $PROTOC_TOOLCHAIN..."
uv run --no-project --python "$PROTOC_PYTHON" --with "$PROTOC_TOOLCHAIN" \
    python -m grpc_tools.protoc \
    -I "$REPO_DIR/src" \
    --python_out="$REPO_DIR/src" \
    "$PROTO_REL"

GENERATED="$REPO_DIR/src/vllm_torchtpu/tools/xplane_pb2.py"
if grep -q ValidateProtobufRuntimeVersion "$GENERATED"; then
    echo "ERROR: the generated bindings carry a protobuf runtime floor, which" >&2
    echo "will fail on any environment with an older runtime. Regenerate with" >&2
    echo "$PROTOC_TOOLCHAIN rather than a newer protoc." >&2
    exit 1
fi

echo "Wrote src/vllm_torchtpu/tools/xplane_pb2.py (no runtime floor)"
