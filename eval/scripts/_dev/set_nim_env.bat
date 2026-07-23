@echo off
REM ── NIM environment setup for eval runs ──────────────────────────────────
REM Usage:
REM   1. Fill in your NIM API key below (nvapi-...)
REM   2. Change RUN_TAG and MODEL as needed
REM   3. Run:  call eval\scripts\set_nim_env.bat
REM   4. Then: python -m eval.agent.run_stage --stage baseline --workers 2 --probe
REM
REM NOTE: must use `call` to export vars to the parent CMD session.
REM       This does NOT work from PowerShell — use set_nim_env.ps1 instead.

set SG_EVAL_RUN_TAG=gemma4_31b_swebench
set SG_EVAL_API_BASE=https://integrate.api.nvidia.com/v1
set SG_EVAL_API_KEY=nvapi-REPLACE_WITH_YOUR_OWN_KEY
set SG_EVAL_MODEL=google/gemma-4-31b-it

echo.
echo === NIM env vars set ===
echo   SG_EVAL_RUN_TAG  = %SG_EVAL_RUN_TAG%
echo   SG_EVAL_API_BASE = %SG_EVAL_API_BASE%
echo   SG_EVAL_MODEL    = %SG_EVAL_MODEL%
echo   SG_EVAL_API_KEY  = %SG_EVAL_API_KEY:~0,12%...
echo.
echo Next: python -m eval.agent.run_stage --stage baseline --workers 2 --probe
