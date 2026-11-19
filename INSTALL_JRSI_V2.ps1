param(
    [string]$TargetRoot = "C:\work\snake_model"
)

$ErrorActionPreference = "Stop"
$PackageRoot = $PSScriptRoot
$GitDir = Join-Path $TargetRoot ".git"

if (-not (Test-Path -LiteralPath $GitDir)) {
    throw "The target is not an existing Git repository: $TargetRoot"
}
if ([IO.Path]::GetFullPath($PackageRoot).TrimEnd('\') -eq [IO.Path]::GetFullPath($TargetRoot).TrimEnd('\')) {
    throw "Extract this package to a separate directory before running the installer."
}

$Required = @("README.md", "CITATION.cff", ".zenodo.json", "snake_model\model.py", "analysis_v2.py", "results_v2\summary.json", "manuscript\manuscript.pdf")
foreach ($RelativePath in $Required) {
    if (-not (Test-Path -LiteralPath (Join-Path $PackageRoot $RelativePath))) {
        throw "The package is incomplete; missing $RelativePath"
    }
}

$Remote = (& git -C $TargetRoot remote get-url origin 2>$null)
if ($LASTEXITCODE -ne 0 -or $Remote -notmatch 'github\.com[:/]squareshorts/snake-spike-sem(\.git)?$') {
    throw "The target origin is not https://github.com/squareshorts/snake-spike-sem.git (found: $Remote)"
}
$Branch = (& git -C $TargetRoot branch --show-current)
if ($Branch -ne "main") {
    throw "The target repository is on branch '$Branch', not 'main'. No files were moved."
}
$LocalHead = (& git -C $TargetRoot rev-parse HEAD)
$RemoteMainLine = (& git -C $TargetRoot ls-remote origin refs/heads/main)
if ($LASTEXITCODE -ne 0 -or -not $RemoteMainLine) {
    throw "Could not verify the remote main branch. No files were moved."
}
$RemoteHead = ($RemoteMainLine -split '\s+')[0]
if ($LocalHead -ne $RemoteHead) {
    throw "Local main ($LocalHead) does not match origin/main ($RemoteHead). Fetch/pull and inspect the repository before installing. No files were moved."
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupRoot = "$TargetRoot.pre_jrsi_$Stamp"
New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null

Get-ChildItem -LiteralPath $TargetRoot -Force | Where-Object { $_.Name -ne ".git" } | ForEach-Object {
    Move-Item -LiteralPath $_.FullName -Destination $BackupRoot
}

Get-ChildItem -LiteralPath $PackageRoot -Force | Where-Object { $_.Name -ne ".git" } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $TargetRoot -Recurse -Force
}

Write-Host "Installed JRSI v2.0.0 working tree at $TargetRoot"
Write-Host "Previous working files were preserved at $BackupRoot"
Write-Host "Review the changes with: git -C `"$TargetRoot`" status --short"
Write-Host "Install dependencies, run tests, then run PUBLISH_GITHUB.ps1 from the target repository."
