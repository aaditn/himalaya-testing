[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ImageTag,
    [switch]$Push
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ($ImageTag -match '@sha256:') {
    throw "Build with a normal tag; this script prints the immutable digest after push."
}

if ($Push) {
    & docker buildx build --platform linux/amd64 --file Dockerfile.hf `
        --tag $ImageTag --push .
    if ($LASTEXITCODE -ne 0) { throw "Container build or push failed." }
    $inspection = (& docker buildx imagetools inspect $ImageTag 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect the pushed image." }
    $digest = ([regex]::Match($inspection, '(?m)^Digest:\s*(sha256:[0-9a-f]{64})\r?$')).Groups[1].Value
    if (-not $digest) { throw "Registry did not return an immutable image digest." }
    $repository = $ImageTag -replace ':[^/:]+$', ''
    Write-Host "Immutable image: $repository@$digest"
} else {
    & docker buildx build --platform linux/amd64 --file Dockerfile.hf `
        --tag $ImageTag --load .
    if ($LASTEXITCODE -ne 0) { throw "Local container build failed." }
    Write-Host "Local image built. Re-run with -Push to obtain the required registry digest."
}
