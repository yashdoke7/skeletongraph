# Stage 0 - one-shot GO/NO-GO eval. Pure CPU, no API, no GPU. Windows-native.
#
#   powershell -ExecutionPolicy Bypass -File eval\stage0.ps1 smoke   # 2 tasks, ~10 min
#   powershell -ExecutionPolicy Bypass -File eval\stage0.ps1         # 30 tasks, ~2-3 h
#
# Background it:
#   Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','eval\stage0.ps1' -RedirectStandardOutput eval\logs\stage0.log

param([string]$Mode = "full")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force -Path eval\logs | Out-Null

Write-Host "=== Stage 0 :: $Mode ==="

Write-Host "[1/4] dependencies"
python -m pip install -q datasets

Write-Host "[2/4] build dataset"
if ($Mode -eq "smoke") {
    python eval\make_dataset.py --smoke
} else {
    $n = if ($env:SG_STAGE0_N) { $env:SG_STAGE0_N } else { "30" }
    python eval\make_dataset.py --n $n
}

Write-Host "[3/4] retrieval + context-token eval"
python eval\run_stage0.py

Write-Host "[4/4] analytical KV-cache / serving density"
python eval\kv_cache.py --csv eval\results\stage0\kv_cache.csv

Write-Host ""
Write-Host "=== Stage 0 done ==="
Write-Host "Verdict: eval\results\stage0\SUMMARY.md"
Get-Content eval\results\stage0\SUMMARY.md
