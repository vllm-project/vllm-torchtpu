#!/usr/bin/env bash
# 在 P0 pod 里启动 2P1D proxy（:8000）：
#   bash pod_proxy.sh <P0_IP> <P1_IP> <D_IP>
# toy_proxy_server.py 本身支持多个 prefiller（默认按前缀路由），PROXY_* 环境变量原样生效
# （AgentX 要会话亲和：PROXY_SESSION_AFFINITY_DP 等，见 scripts/dsv41_pd/REPRODUCE.zh.md 4.4）。
set -euo pipefail
P0=${1:?P0 ip}; P1=${2:?P1 ip}; D=${3:?D ip}
REPO=${REPO:-/zzlfs/temu/vllm-torchtpu-1p1d}
umask 0000
export VLLM_PLUGINS="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1
export PROXY_TOKENIZER="${PROXY_TOKENIZER-/tmp/dsv41-tokenizer}"
# 前缀 / 会话路由跨两个 P（全局 rank g = 第 g//8 个 P 的 g%8 号 rank），见 toy_proxy_server._split_prefill_rank
export PROXY_PREFILL_INSTANCES=2
echo "proxy :${PORT:-8000} prefill=$P0:8100,$P1:8100 decode=$D:8200"
exec "${PROXY_PYTHON:-/zzlfs/temu/.venv-epic/bin/python}" "$REPO/scripts/dsv41_pd/toy_proxy_server.py" \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --prefiller-hosts "$P0" "$P1" --prefiller-ports 8100 8100 \
  --decoder-hosts "$D" --decoder-ports 8200
