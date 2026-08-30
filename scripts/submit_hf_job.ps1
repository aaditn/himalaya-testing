[CmdletBinding()]
param(
    [string]$RepoId = "himalaya-g1-uphill",
    [string]$Flavor = "h100",
    [string]$Timeout = "48h",
    [string]$Namespace = "",
    [string]$Image = "",
    [int]$TargetSlope = 15,
    [long]$TimestepsPerStage = 40000000,
    [int]$NumEnvs = 8192,
    [int]$ValidationTrials = 64,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    throw "The Hugging Face CLI is not installed. Install it with: pip install -U huggingface_hub"
}
if ($Launch -and $Image -notmatch '/himalaya(?:-g1)?-hf@sha256:[0-9a-fA-F]{64}$') {
    throw "-Image must be the digest-pinned himalaya-hf image built from Dockerfile.hf."
}
if ($Launch) {
    & python scripts/verify_training_launch_contract.py --image $Image
    if ($LASTEXITCODE -ne 0) { throw "HF training launch contract failed." }
}

$identity = (& hf auth whoami 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $identity -match "Not logged in") {
    throw "Hugging Face authentication is required. Run 'hf auth login' in a private terminal first."
}

if ($RepoId.Contains("/")) {
    $fullRepoId = $RepoId
} else {
    $usernameLine = $identity -split "`r?`n" | Where-Object { $_ -match "(?:username|user):\s*(\S+)" } | Select-Object -First 1
    if (-not $usernameLine) {
        throw "Could not determine the Hugging Face username. Pass -RepoId as 'username/repository'."
    }
    $username = ([regex]::Match($usernameLine, "(?:username|user):\s*(\S+)")).Groups[1].Value
    $fullRepoId = "$username/$RepoId"
}

Write-Host "Preparing private model repository $fullRepoId"
& hf repo create $fullRepoId --repo-type model --private --exist-ok
if ($LASTEXITCODE -ne 0) { throw "Could not create the Hugging Face repository." }

& hf upload $fullRepoId . . --repo-type model `
    --exclude ".git/**" ".venv/**" "**/__pycache__/**" "*.pyc" "runs/**" "validation/**" `
    --commit-message "Add MuJoCo-only G1 uphill training job"
if ($LASTEXITCODE -ne 0) { throw "Could not upload the training source." }

if (-not $Launch) {
    Write-Host "Source uploaded. Re-run with -Launch to start paid GPU compute."
    exit 0
}

$bootstrap = @'
hf download "$HF_REPO_ID" --repo-type model --local-dir /workspace/himalaya --exclude 'runs/**' --exclude 'validation/**' &&
cd /workspace/himalaya &&
bash scripts/hf_job.sh
'@

$namespaceArgs = @()
if ($Namespace) {
    $namespaceArgs = @("--namespace", $Namespace)
}

$targetNamespace = if ($Namespace) { $Namespace } else { "the current user" }
Write-Host "Submitting $Flavor job in $targetNamespace (timeout $Timeout)"
& hf jobs run --detach --flavor $Flavor --timeout $Timeout @namespaceArgs `
    --secrets HF_TOKEN `
    --env "HF_REPO_ID=$fullRepoId" `
    --env "IMAGE_REF=$Image" `
    --env "TARGET_SLOPE=$TargetSlope" `
    --env "TIMESTEPS_PER_STAGE=$TimestepsPerStage" `
    --env "NUM_ENVS=$NumEnvs" `
    --env "VALIDATION_TRIALS=$ValidationTrials" `
    $Image `
    bash -lc $bootstrap
if ($LASTEXITCODE -ne 0) { throw "Hugging Face job submission failed." }

Write-Host "Submitted. Inspect it with: hf jobs ps"
