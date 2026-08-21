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
# Run this after changing xplane.proto, or when the protobuf runtime pin moves
# forward far enough that the checked-in gencode is refused at import (protobuf
# requires runtime >= gencode; the error names both versions).
set -euo pipefail

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
echo "Generating bindings..."
python3 -m grpc_tools.protoc \
    -I "$REPO_DIR/src" \
    --python_out="$REPO_DIR/src" \
    "$PROTO_REL"

echo "Wrote src/vllm_torchtpu/tools/xplane_pb2.py"
echo "Generated against protobuf runtime: $(python3 -c 'import google.protobuf as p; print(p.__version__)')"
