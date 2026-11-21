# Non-destructive v2.2.1 installer for an extracted release package.
# Preview is the default. Copying requires an explicit -Install switch.
param(
    [string]$TargetRoot = 'C:\work\snake_model',
    [switch]$Install
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$PackageRoot = $PSScriptRoot
$TargetFull = [IO.Path]::GetFullPath($TargetRoot).TrimEnd('\')
$PackageFull = [IO.Path]::GetFullPath($PackageRoot).TrimEnd('\')
if ($PackageFull -ieq $TargetFull) {
    throw 'Extract the release to a separate folder. Source and target may not be the same.'
}
if (-not (Test-Path -LiteralPath (Join-Path $TargetRoot '.git'))) {
    throw "Target is not an existing Git repository: $TargetRoot"
}
$Required = @('CITATION.cff', '.zenodo.json', 'analysis_controls.py',
              'snake_model\model.py', 'manuscript\manuscript.tex',
              'results_v2\controls\coverage.csv')
foreach ($Rel in $Required) {
    if (-not (Test-Path -LiteralPath (Join-Path $PackageRoot $Rel) -PathType Leaf)) {
        throw "Release package missing: $Rel"
    }
}
$VersionText = Get-Content -LiteralPath (Join-Path $PackageRoot 'CITATION.cff') -Raw
if ($VersionText -notmatch '(?m)^version:\s*2\.2\.0\s*$') {
    throw 'This installer requires the v2.2.1 release package.'
}
$Remote = & git -C $TargetRoot remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or $Remote -notmatch 'github\.com[:/]squareshorts/snake-spike-sem(\.git)?$') {
    throw "Unexpected Git remote: $Remote"
}
$Branch = & git -C $TargetRoot branch --show-current
if ($LASTEXITCODE -ne 0 -or $Branch -ne 'main') {
    throw "Expected the main branch; found: $Branch"
}
$Files = @(Get-ChildItem -LiteralPath $PackageRoot -Recurse -File -Force | Where-Object {
    # Protect any metadata, caches or virtual environments accidentally placed in the package.
    $Rel = $_.FullName.Substring($PackageFull.Length).TrimStart('\').Replace('\','/')
    $Rel -notmatch '(^|/)(\.git|\.venv|__pycache__|\.pytest_cache)(/|$)'
})
if ($Files.Count -lt 100) { throw "Release package has only $($Files.Count) files." }
Write-Host "Release: v2.2.1"
Write-Host "Source: $PackageFull"
Write-Host "Target: $TargetFull"
Write-Host "Files available for overlay: $($Files.Count)"
Write-Host 'Existing .git, .venv and unrelated files are preserved.'
if (-not $Install) {
    Write-Host 'PREVIEW ONLY. To copy files, rerun with -Install.'
    exit 0
}

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$BackupRoot = "${TargetFull}_release_backups\installer_v221_$Stamp"
$BackedUp = 0
$Copied = 0
foreach ($File in $Files) {
    $Rel = $File.FullName.Substring($PackageFull.Length).TrimStart('\')
    $Dest = Join-Path $TargetFull $Rel
    if (Test-Path -LiteralPath $Dest -PathType Leaf) {
        $CopyOfOld = Join-Path $BackupRoot $Rel
        New-Item -ItemType Directory -Path (Split-Path -Parent $CopyOfOld) -Force | Out-Null
        Copy-Item -LiteralPath $Dest -Destination $CopyOfOld -Force
        $BackedUp++
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Dest) -Force | Out-Null
    Copy-Item -LiteralPath $File.FullName -Destination $Dest -Force
    $Copied++
}
Write-Host "Installed v2.2.1 files: $Copied"
Write-Host "Backed up existing files: $BackedUp"
Write-Host "Backup location: $BackupRoot"
Write-Host 'No Git commit, tag, push or release was created.'
