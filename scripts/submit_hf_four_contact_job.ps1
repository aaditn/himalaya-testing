[CmdletBinding()]
param(
    [ValidateSet("Smoke", "Real")]
    [string]$Mode = "Smoke",
    [string]$RepoId = "himalaya-g1-four-contact",
    [string]$Image = "",
    [string]$Flavor = "",
    [string]$Timeout = "16h",
    [string]$Namespace = "",
    [string]$RunId = "",
    [long]$TrainingTimesteps30 = 40000000,
    [long]$TrainingTimesteps35 = 100000000,
    [int]$NumEnvs = 8192,
    [int]$ValidationTrials = 64,
    [double]$PromotionSuccessRate30 = 0.80,
    [double]$PromotionSuccessRate35 = 0.90,
    [int]$SmokeTimesteps = 512,
    [int]$SmokeNumEnvs = 16,
    [string]$HumanAuditApprovedBy = "",
    [string]$HumanAuditApprovalRef = "",
    [switch]$SkipSmokeGate,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$modeName = $Mode.ToLowerInvariant()

if ($TrainingTimesteps30 -le 0 -or $TrainingTimesteps35 -le 0 -or
    $NumEnvs -le 0 -or $ValidationTrials -le 0) {
    throw "Timesteps, environments, and validation trials must be positive."
}
if ($PromotionSuccessRate30 -lt 0.0 -or $PromotionSuccessRate30 -gt 1.0 -or
    $PromotionSuccessRate35 -lt 0.0 -or $PromotionSuccessRate35 -gt 1.0) {
    throw "Promotion success rates must be between 0 and 1."
}
if ($SmokeTimesteps -le 0 -or $SmokeTimesteps -gt 10000 -or
    $SmokeNumEnvs -le 0 -or $SmokeNumEnvs -gt 128) {
    throw "Smoke mode is capped at 10,000 steps and 128 environments."
}
if ($Mode -eq "Real" -and (
    $TrainingTimesteps30 -ne 40000000 -or
    $TrainingTimesteps35 -ne 100000000 -or
    $NumEnvs -ne 8192 -or $ValidationTrials -ne 64 -or
    $PromotionSuccessRate30 -ne 0.80 -or
    $PromotionSuccessRate35 -ne 0.90
)) {
    throw "Real balance mode requires 40M steps at 30 degrees, 8192 environments, 64 trials, and an 80% four-contact occupancy gate."
}
if ($Launch) {
    if ([string]::IsNullOrWhiteSpace($Flavor)) {
        throw "Pass an explicit Hugging Face GPU -Flavor."
    }
    if ($Image -notmatch '@sha256:[0-9a-fA-F]{64}$') {
        throw "-Image must be an immutable registry reference ending in @sha256:<64 hex chars>."
    }
    if ($Image -notmatch '/himalaya(?:-g1)?-hf@sha256:') {
        throw "-Image must be the digest-pinned himalaya-hf image built from Dockerfile.hf."
    }
}
if ($Launch -and $Mode -eq "Real" -and (
    [string]::IsNullOrWhiteSpace($HumanAuditApprovedBy) -or
    [string]::IsNullOrWhiteSpace($HumanAuditApprovalRef)
)) {
    throw "Real launch requires HumanAuditApprovedBy and HumanAuditApprovalRef."
}

$hfCommand = Get-Command hf -ErrorAction Stop
$hfExe = $hfCommand.Source
$identity = (& $hfExe auth whoami 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $identity -match "Not logged in") {
    throw "Hugging Face authentication is required. Run 'hf auth login'."
}
if ($RepoId.Contains("/")) {
    $fullRepoId = $RepoId
} else {
    $line = $identity -split "`r?`n" | Where-Object {
        $_ -match "(?:username|user):\s*(\S+)"
    } | Select-Object -First 1
    if (-not $line) { throw "Pass -RepoId as username/repository." }
    $username = ([regex]::Match(
        $line, "(?:username|user):\s*(\S+)"
    )).Groups[1].Value
    $fullRepoId = "$username/$RepoId"
}

# Use the interpreter that owns the hf executable, not an unrelated system Python.
$hfScripts = Split-Path $hfExe -Parent
$hfRoot = Split-Path $hfScripts -Parent
$hfPython = Join-Path $hfRoot "python.exe"
if (-not (Test-Path -LiteralPath $hfPython)) {
    $hfPython = (Get-Command python -ErrorAction Stop).Source
}
if ($Launch) {
    & $hfPython scripts/verify_training_launch_contract.py --image $Image
    if ($LASTEXITCODE -ne 0) { throw "HF training launch contract failed." }
}

$uploadOutput = @(& $hfPython scripts/hf_upload_source.py `
    --repo-id $fullRepoId --source . 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "Source upload failed:`n$($uploadOutput -join "`n")"
}
$uploadJson = $uploadOutput | ForEach-Object { "$_" } | Where-Object {
    $_ -match '^\{.*"source_revision"'
} | Select-Object -Last 1
if (-not $uploadJson) { throw "Source upload did not return a revision." }
$upload = $uploadJson | ConvertFrom-Json
$sourceRevision = $upload.source_revision
$sourceDigest = $upload.source_digest
$runtimeDigest = $upload.runtime_digest

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $suffix = [Guid]::NewGuid().ToString("N").Substring(0, 8)
    $RunId = "$modeName-$stamp-$suffix"
}
if ($RunId -notmatch '^[A-Za-z0-9._-]+$') {
    throw "RunId may contain only letters, digits, periods, underscores, and hyphens."
}

$imageHash = [Security.Cryptography.SHA256]::Create()
try {
    $imageBytes = [Text.Encoding]::UTF8.GetBytes($Image)
    $imageKey = -join ($imageHash.ComputeHash($imageBytes) | ForEach-Object {
        $_.ToString("x2")
    })
} finally {
    $imageHash.Dispose()
}
$remoteOutputPath = "runs/$modeName/$sourceRevision/$RunId"
$smokeGatePath = "smoke-gates/$runtimeDigest/$imageKey/smoke_pass.json"

if ($Launch -and $Mode -ne "Smoke" -and -not $SkipSmokeGate) {
    & $hfPython scripts/verify_smoke_gate.py `
        --repo-id $fullRepoId `
        --gate-path $smokeGatePath `
        --source-revision $sourceRevision `
        --runtime-digest $runtimeDigest `
        --image-ref $Image
    if ($LASTEXITCODE -ne 0) {
        throw "Matching smoke gate is absent or invalid; the paid job was not launched."
    }
}

Write-Host "Source revision: $sourceRevision"
Write-Host "Runtime digest:  $runtimeDigest"
Write-Host "Source branch:   $($upload.source_branch)"
Write-Host "Output path:     $remoteOutputPath"
Write-Host "Smoke gate:      $smokeGatePath"
if (-not $Launch) {
    Write-Host "Source uploaded only; add -Launch after selecting an immutable image and GPU flavor."
    exit 0
}
$approvedAt = (Get-Date).ToUniversalTime().ToString("o")
$skipSmokeValue = if ($SkipSmokeGate) { "1" } else { "0" }

$bootstrap = @'
uvx --from huggingface_hub==1.29.0 hf download "$HF_REPO_ID" --repo-type model --revision "$SOURCE_REVISION" --local-dir /workspace/himalaya --exclude 'runs/**' --exclude 'validation/**' --exclude 'smoke-gates/**' --exclude 'local_preview/**' &&
cd /workspace/himalaya &&
bash scripts/hf_four_contact_job.sh
'@
$namespaceArgs = @()
if ($Namespace) { $namespaceArgs = @("--namespace", $Namespace) }
& $hfExe jobs run --detach --flavor $Flavor --timeout $Timeout @namespaceArgs `
    --secrets HF_TOKEN `
    --env "HF_REPO_ID=$fullRepoId" `
    --env "IMAGE_REF=$Image" `
    --env "JOB_MODE=$modeName" `
    --env "RUN_ID=$RunId" `
    --env "SOURCE_REVISION=$sourceRevision" `
    --env "SOURCE_DIGEST=$sourceDigest" `
    --env "RUNTIME_DIGEST=$runtimeDigest" `
    --env "REMOTE_OUTPUT_PATH=$remoteOutputPath" `
    --env "SMOKE_GATE_PATH=$smokeGatePath" `
    --env "SKIP_SMOKE_GATE=$skipSmokeValue" `
    --env "HUMAN_AUDIT_APPROVED_BY=$HumanAuditApprovedBy" `
    --env "HUMAN_AUDIT_APPROVAL_REF=$HumanAuditApprovalRef" `
    --env "HUMAN_AUDIT_APPROVED_AT=$approvedAt" `
    --env "TRAINING_TIMESTEPS_30=$TrainingTimesteps30" `
    --env "TRAINING_TIMESTEPS_35=$TrainingTimesteps35" `
    --env "NUM_ENVS=$NumEnvs" `
    --env "VALIDATION_TRIALS=$ValidationTrials" `
    --env "PROMOTION_SUCCESS_RATE_30=$PromotionSuccessRate30" `
    --env "PROMOTION_SUCCESS_RATE_35=$PromotionSuccessRate35" `
    --env "SMOKE_TIMESTEPS=$SmokeTimesteps" `
    --env "SMOKE_NUM_ENVS=$SmokeNumEnvs" `
    $Image bash -lc $bootstrap
if ($LASTEXITCODE -ne 0) { throw "Hugging Face job submission failed." }
Write-Host "Submitted $Mode run $RunId from immutable source $sourceRevision."
