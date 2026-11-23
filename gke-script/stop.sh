#!/usr/bin/env bash
# 停掉 vLLM（和 P0 上的 proxy），pod 本身不动。
#   bash stop.sh [p0 p1 d]
set -uo pipefail
source "$(dirname "$0")/common.sh"
ROLES=$(roles_from_args "$@")
[[ " $ROLES " == *" p0 "* ]] && kx "$(role_pod p0)" "bash $GKE_DIR/pod_stop.sh proxy"
for r in $ROLES; do kx "$(role_pod "$r")" "bash $GKE_DIR/pod_stop.sh" & done; wait
