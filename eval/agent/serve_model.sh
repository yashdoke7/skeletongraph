#!/usr/bin/env bash
# Serve the eval model with vLLM on an AMD MI300X (ROCm).
#
#   bash eval/agent/serve_model.sh                 # default: Qwen2.5-Coder-32B
#   SG_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct bash eval/agent/serve_model.sh
#
# The harness (eval/agent/) talks to the OpenAI-compatible API this exposes.
# Tool calling REQUIRES --enable-auto-tool-choice + the right --tool-call-parser
# (hermes for Qwen2.5). Without it the ReAct loop gets no tool_calls back.
set -euo pipefail

MODEL="${SG_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
MAX_LEN="${SG_MAX_LEN:-32768}"
PORT="${SG_PORT:-8000}"

echo "Serving ${MODEL}  (max-model-len=${MAX_LEN}, port=${PORT})"

docker run -d --name vllm --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 16g \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  rocm/vllm:latest \
  vllm serve "${MODEL}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port "${PORT}"

echo "Container started. Wait for 'Application startup complete', then:"
echo "  curl http://localhost:${PORT}/v1/models"
echo "Logs:  docker logs -f vllm"
