# JRSI v2.2.1 release publication. Safe by default: preview only.
# Explicit -Publish is required to stage, commit or push anything.
param([switch]$Publish)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepoRoot = $PSScriptRoot
# Reuse the independently checked 2.2.0 archive *only* as an exact file-path manifest.
$ZipPath = Join-Path $RepoRoot 'snake-spike-sem-v2.2.0.zip'
$ZipPrefix = 'snake-spike-sem-v2.2.0/'
$VersionTag = 'v2.2.1'
$ExpectedRemotePattern = 'github\.com[:/]squareshorts/snake-spike-sem(\.git)?$'

if (-not (Test-Path -LiteralPath $ZipPath -PathType Leaf)) {
    throw "Expected release ZIP not found: $ZipPath"
}

# Build a strict, exact list of release-file paths from the verified ZIP.
# Unlike 'git add -A', this never stages unrelated files in the repository.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
$PathsList = New-Object 'System.Collections.Generic.List[string]'
try {
    foreach ($Entry in $Zip.Entries) {
        $Name = $Entry.FullName.Replace('\', '/')
        if ($Name.EndsWith('/')) { continue }
        if (-not $Name.StartsWith($ZipPrefix, [System.StringComparison]::Ordinal)) {
            throw "Unexpected path in release ZIP: $Name"
        }
        $Relative = $Name.Substring($ZipPrefix.Length)
        if ([string]::IsNullOrWhiteSpace($Relative) -or
            $Relative.StartsWith('/') -or
            $Relative.Contains('../') -or
            $Relative.Contains('/./')) {
            throw "Unsafe path in release ZIP: $Name"
        }
        $FileOnDisk = Join-Path $RepoRoot ($Relative.Replace('/', '\'))
        if (-not (Test-Path -LiteralPath $FileOnDisk -PathType Leaf)) {
            throw "Release file not installed: $Relative"
        }
        $PathsList.Add($Relative)
    }
}
finally {
    $Zip.Dispose()
}
# Include one additional regression test, generated after the original 126-file ZIP.
# Every other release path must still be declared in that ZIP.
$AdditionalReleasePaths = @('tests/test_geometry_regression.py')
foreach ($Relative in $AdditionalReleasePaths) {
    $FileOnDisk = Join-Path $RepoRoot ($Relative.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $FileOnDisk -PathType Leaf)) {
        throw "Required release regression test is missing: $Relative"
    }
    $PathsList.Add($Relative)
}
$Paths = @($PathsList | Sort-Object -Unique)
if ($Paths.Count -ne 127) {
    throw "Expected 127 explicit release paths; found $($Paths.Count). Review the archive."
}


# Refuse to tag a mixed-version or incomplete release tree.
$CffText = Get-Content -LiteralPath (Join-Path $RepoRoot 'CITATION.cff') -Raw
if ($CffText -notmatch '(?m)^version:\s*2\.2\.1\s*$') { throw 'CITATION.cff does not identify 2.2.1.' }
$ZenodoMetadata = Get-Content -LiteralPath (Join-Path $RepoRoot '.zenodo.json') -Raw | ConvertFrom-Json
if ($ZenodoMetadata.version -ne '2.2.1') { throw '.zenodo.json does not identify 2.2.1.' }
$MainSource = Get-Content -LiteralPath (Join-Path $RepoRoot 'manuscript\manuscript.tex') -Raw
if ($MainSource -match 'zenodo\.24378494|zenodo\.24408778') { throw 'Manuscript references an incorrect Zenodo archive.' }
foreach ($Check in @('analysis_controls.py', 'results_v2\controls\coverage.csv', 'tests\test_geometry_regression.py')) {
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $Check) -PathType Leaf)) {
        throw "Missing new analysis file: $Check"
    }
}

$Remote = (& git -C $RepoRoot remote get-url origin 2>$null)
if ($LASTEXITCODE -ne 0 -or $Remote -notmatch $ExpectedRemotePattern) {
    throw "Unexpected Git remote: $Remote"
}
$Branch = (& git -C $RepoRoot branch --show-current)
if ($LASTEXITCODE -ne 0 -or $Branch -ne 'main') { throw "Expected branch main; found $Branch" }

Write-Host "Repository: $RepoRoot"
Write-Host "Remote: $Remote"
Write-Host "Branch: $Branch"
Write-Host "Release: $VersionTag"
Write-Host "Release paths explicitly allowed for staging: $($Paths.Count)"
Write-Host 'Unrelated ZIPs, backups and Claude outputs are excluded.'
Write-Host 'Current working-tree status:'
& git -C $RepoRoot status --short
if ($LASTEXITCODE -ne 0) { throw 'git status failed.' }

if (-not $Publish) {
    Write-Host ''
    Write-Host 'PREVIEW ONLY. No files staged, committed, tagged or pushed.'
    Write-Host 'Do not invoke -Publish until release metadata and the installation script are checked.'
    exit 0
}

$AlreadyStaged = @(& git -C $RepoRoot diff --cached --name-only)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect staged changes.' }
if ($AlreadyStaged.Count -gt 0) {
    throw 'The Git index already contains staged changes. Unstage/review them before publishing.'
}

$LocalHead = (& git -C $RepoRoot rev-parse HEAD)
$RemoteMainLine = (& git -C $RepoRoot ls-remote origin refs/heads/main)
if ($LASTEXITCODE -ne 0 -or -not $RemoteMainLine) { throw 'Could not inspect origin/main.' }
$RemoteHead = ($RemoteMainLine -split '\s+')[0]
if ($LocalHead -ne $RemoteHead) {
    throw "Local main ($LocalHead) differs from origin/main ($RemoteHead). Inspect before publishing."
}
$LocalTag = (& git -C $RepoRoot tag -l $VersionTag)
if ($LocalTag) { throw "Tag $VersionTag already exists locally." }
$RemoteTag = (& git -C $RepoRoot ls-remote --tags origin "refs/tags/$VersionTag")
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect remote tags.' }
if ($RemoteTag) { throw "Tag $VersionTag already exists on origin." }

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Required virtual-environment Python not found: $Python"
}
Write-Host 'Running pytest using the project virtual environment...'
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed. Nothing published.' }

$Answer = Read-Host "Type STAGE to stage only the $($Paths.Count) manifest paths for review"
if ($Answer -cne 'STAGE') {
    Write-Host 'Stopped without staging, committing or publishing.'
    exit 0
}

$ManifestFile = Join-Path $env:TEMP ("JRSI_v220_git_paths_$PID.txt")
try {
    $Utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
    [System.IO.File]::WriteAllLines($ManifestFile, [string[]]$Paths, $Utf8)
    & git -C $RepoRoot add "--pathspec-from-file=$ManifestFile"
    if ($LASTEXITCODE -ne 0) { throw 'Selective git add failed.' }

    $ActuallyStaged = @(& git -C $RepoRoot diff --cached --name-only)
    if ($LASTEXITCODE -ne 0 -or $ActuallyStaged.Count -eq 0) {
        throw 'No changes staged, or could not inspect staged changes.'
    }
    foreach ($File in $ActuallyStaged) {
        if ($Paths -notcontains $File) {
            throw "Unexpected staged file outside the release manifest: $File"
        }
    }
    & git -C $RepoRoot diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged diff has whitespace errors. Review before publishing.' }
    Write-Host 'Exact staged changes:'
    & git -C $RepoRoot diff --cached --stat
    Write-Host ''
    Write-Host 'Inspect the staged file list above before proceeding.'
    $Confirm = Read-Host 'Type COMMIT to commit, push main, tag v2.2.1 and push the tag'
    if ($Confirm -cne 'COMMIT') {
        & git -C $RepoRoot reset --quiet HEAD
        Write-Host 'Stopped. Release changes unstaged; all working files are preserved.'
        exit 0
    }

    & git -C $RepoRoot commit -m 'Prepare JRSI blade-array model release v2.2.1'
    if ($LASTEXITCODE -ne 0) { throw 'git commit failed.' }
    & git -C $RepoRoot push origin main
    if ($LASTEXITCODE -ne 0) { throw 'git push origin main failed.' }
    & git -C $RepoRoot tag -a $VersionTag -m 'JRSI blade-array model v2.2.1'
    if ($LASTEXITCODE -ne 0) { throw 'git tag failed.' }
    & git -C $RepoRoot push origin $VersionTag
    if ($LASTEXITCODE -ne 0) { throw 'git push tag failed.' }
    Write-Host "Pushed main and tag $VersionTag. Create a GitHub Release from the tag to trigger Zenodo."
}
finally {
    Remove-Item -LiteralPath $ManifestFile -ErrorAction SilentlyContinue
}
