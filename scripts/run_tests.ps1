# ARGUS test runner (Windows PowerShell)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
#
# Assumes a virtual environment is already active with the `dev-test`
# (and whichever other) dependency groups installed:
#   pip install -e ".[core,dev-test]"

$ErrorActionPreference = "Stop"

Write-Host "ARGUS: running test suite..." -ForegroundColor Cyan

python -m pytest tests/ -v

if ($LASTEXITCODE -ne 0) {
    Write-Host "ARGUS: tests FAILED" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "ARGUS: tests PASSED" -ForegroundColor Green
