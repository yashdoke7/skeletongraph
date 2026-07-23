# ── NIM environment setup for eval runs ──────────────────────────────────
# Usage (PowerShell):
#   1. Fill in your NIM API key below (nvapi-...)
#   2. Change RUN_TAG and MODEL as needed
#   3. Dot-source this file to export into current session:
#        . .\eval\scripts\set_nim_env.ps1
#   4. Then run:
#        python -m eval.agent.run_stage --stage baseline --workers 2 --probe

$env:SG_EVAL_RUN_TAG  = "qwen3coder_swebench"
$env:SG_EVAL_API_BASE = "https://integrate.api.nvidia.com/v1"
$env:SG_EVAL_API_KEY  = "nvapi-REPLACE_WITH_YOUR_OWN_KEY"
$env:SG_EVAL_MODEL    = "qwen/qwen3-coder-480b-a35b-instruct"   # update if model ID differs

Write-Host ""
Write-Host "=== NIM env vars set ===" -ForegroundColor Green
Write-Host "  SG_EVAL_RUN_TAG  = $env:SG_EVAL_RUN_TAG"
Write-Host "  SG_EVAL_API_BASE = $env:SG_EVAL_API_BASE"
Write-Host "  SG_EVAL_MODEL    = $env:SG_EVAL_MODEL"
Write-Host "  SG_EVAL_API_KEY  = $($env:SG_EVAL_API_KEY.Substring(0, [Math]::Min(12, $env:SG_EVAL_API_KEY.Length)))..."
Write-Host ""
Write-Host "Next: python -m eval.agent.run_stage --stage baseline --workers 2 --probe"
