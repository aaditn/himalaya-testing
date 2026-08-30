[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$hfCli = Join-Path $projectRoot ".venv\Scripts\hf.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating project virtual environment..."
    & py -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Installing the simulator, development tools, and Hugging Face cloud tooling..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $venvPython -m pip install -e "$projectRoot[dev]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $venvPython -m pip install --upgrade "huggingface_hub>=1.8.0" hf-xet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
& $hfCli --version
Write-Host "Cloud tooling is installed. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Then verify access with:"
Write-Host "  hf auth whoami"
Write-Host "  python scripts\myremote.py list --limit 5"
