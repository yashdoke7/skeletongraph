# Local 7B smoke — practice run of the full eval harness on a 4070 (8 GB VRAM).
#
# A 7B model in BF16 (~14 GB) does NOT fit 8 GB. Ollama runs it QUANTIZED
# (Q4_K_M, ~4.7 GB) — fits 8 GB with room for context. Slower than a real GPU,
# but correct; this is a rehearsal of the AMD flow, not a result.
#
# One-time setup:
#   1. Install Ollama from ollama.com (it then runs as a background service).
#   2. ollama pull qwen2.5-coder:7b
#
# Run:
#   powershell -ExecutionPolicy Bypass -File eval\agent\run_local_smoke.ps1
#
# This runs stage 1-core (sg / bm25 / grep / no-retrieval) in --probe mode
# (5 tasks) end to end: agents -> aggregate -> SUMMARY.md. No --verify (the
# SWE-bench Docker harness is heavy for a smoke) and no --push (local practice).
# The AMD run is the SAME command with the real model + --verify --push.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

# Only the endpoint differs from the AMD run — everything else is identical.
$env:SG_EVAL_MODEL    = "qwen2.5-coder:7b"
$env:SG_EVAL_API_BASE = "http://localhost:11434/v1"
$env:SG_EVAL_API_KEY  = "ollama"

Write-Host "=== SkeletonGraph local smoke ==="
Write-Host "  model:    $env:SG_EVAL_MODEL  (Ollama, Q4 quantized)"
Write-Host "  endpoint: $env:SG_EVAL_API_BASE"
Write-Host ""

# Confirm Ollama is reachable + the model is pulled.
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    if (-not ($tags.models.name -match "qwen2.5-coder:7b")) {
        Write-Host "Model not pulled. Run:  ollama pull qwen2.5-coder:7b" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "Ollama not reachable on :11434. Install Ollama and ensure it is running." -ForegroundColor Yellow
    exit 1
}

python -m eval.agent.run_all --stages 1-core --probe --workers 2

Write-Host ""
Write-Host "=== Smoke done ===  Results: eval\results\agent\SUMMARY.md"
