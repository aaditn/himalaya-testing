[CmdletBinding()]
param(
    [string]$RepoId = "jorshcr/himalaya-g1-uphill",
    [string]$Namespace = "iteratehack",
    [string]$Flavor = "t4-small",
    [string]$Timeout = "2h",
    [Parameter(Mandatory = $true)]
    [string]$Image,
    [string]$CheckpointRoot = "/workspace/himalaya/runs/g1_uphill_stage1/stage_00_0deg/checkpoints"
)

$ErrorActionPreference = "Stop"
if ($Image -notmatch '/himalaya(?:-g1)?-hf@sha256:[0-9a-fA-F]{64}$') {
    throw "-Image must be the digest-pinned himalaya-hf image built from Dockerfile.hf."
}
& python scripts/verify_training_launch_contract.py --image $Image
if ($LASTEXITCODE -ne 0) { throw "HF training launch contract failed." }
$bootstrap = @'
hf download "$HF_REPO_ID" --repo-type model --local-dir /workspace/himalaya &&
cd /workspace/himalaya &&
bash scripts/hf_video_job.sh
'@

& hf jobs run --detach --namespace $Namespace --flavor $Flavor --timeout $Timeout `
    --secrets HF_TOKEN `
    --env "HF_REPO_ID=$RepoId" `
    --env "IMAGE_REF=$Image" `
    --env "CHECKPOINT_ROOT=$CheckpointRoot" `
    $Image `
    bash -lc $bootstrap
if ($LASTEXITCODE -ne 0) { throw "Hugging Face video job submission failed." }

