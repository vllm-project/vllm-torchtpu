#!/usr/bin/env bash
# 1P1D 代理（:8000）：toy_proxy_server.py，prefill 2z6z:8100 / decode clsk:8200。
# 替代 recipe_proxy.py（那份写死 185/186，且是给作业提交系统包的一层）。在 prefill 那台（2z6z）上跑：
#   ssh zzl-tpu7x-slice-mig-2z6z 'setsid nohup bash /zzlfs/temu/scripts/start_proxy.sh \
#       > /zzlfs/temu/logs/proxy-$(date +%m%d-%H%M).log 2>&1 < /dev/null &'
set -euo pipefail
PREFILL_HOST=${PREFILL_HOST:-10.128.0.175}
DECODE_HOST=${DECODE_HOST:-10.128.0.177}
# PROXY_RENDER=1（默认）：先用 P 的 /tokenize 分词一次，把 token id 发给 P 和 D，两边都不再分词。
# PROXY_TOKENIZER（默认 /tmp/dsv41-tokenizer）：在代理里本地切块并行分词（557k token 约 0.2s，/tokenize 要 2.4s）；
# 设成空串就改用 P 的 /tokenize。
# PROXY_RENDER=0 恢复原样转发原始文本（P、D 各分词一次，串行，都在 TTFT 里）。
# 代理只转发 HTTP，不需要 TPU 插件
export VLLM_PLUGINS="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1
export PROXY_TOKENIZER="${PROXY_TOKENIZER-/tmp/dsv41-tokenizer}"
# 用 .venv-epic：有 orjson（请求体序列化，见 toy_proxy_server._json_body）。
exec "${PROXY_PYTHON:-/zzlfs/temu/.venv-epic/bin/python}" /zzlfs/temu/scripts/toy_proxy_server.py \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --prefiller-host "$PREFILL_HOST" --prefiller-port 8100 \
  --decoder-host "$DECODE_HOST" --decoder-port 8200
