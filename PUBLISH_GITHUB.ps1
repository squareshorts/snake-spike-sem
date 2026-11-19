$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$ExpectedRemotePattern = 'github\.com[:/]squareshorts/snake-spike-sem(\.git)?$'

$Remote = (& git -C $RepoRoot remote get-url origin 2>$null)
if ($LASTEXITCODE -ne 0 -or $Remote -notmatch $ExpectedRemotePattern) {
    throw "Unexpected Git remote: $Remote"
}
$Branch = (& git -C $RepoRoot branch --show-current)
if ($Branch -ne "main") { throw "Expected branch main; found $Branch" }
$LocalHead = (& git -C $RepoRoot rev-parse HEAD)
$RemoteMainLine = (& git -C $RepoRoot ls-remote origin refs/heads/main)
if ($LASTEXITCODE -ne 0 -or -not $RemoteMainLine) { throw "Could not inspect origin/main." }
$RemoteHead = ($RemoteMainLine -split '\s+')[0]
if ($LocalHead -ne $RemoteHead) {
    throw "Local main ($LocalHead) no longer matches origin/main ($RemoteHead). Fetch and inspect before publishing."
}

$ExistingTag = (& git -C $RepoRoot tag -l "v2.0.0")
if ($ExistingTag) { throw "Tag v2.0.0 already exists locally. Inspect it before continuing." }
$RemoteTag = (& git -C $RepoRoot ls-remote --tags origin "refs/tags/v2.0.0")
if ($LASTEXITCODE -ne 0) { throw "Could not inspect remote tags." }
if ($RemoteTag) { throw "Remote tag v2.0.0 already exists. Inspect it before continuing." }

$Python = Get-Command python -ErrorAction Stop
Write-Host "Running submission-bundle tests before publishing..."
& $Python.Source -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed; GitHub was not changed." }

Write-Host "Repository: $RepoRoot"
Write-Host "Remote: $Remote"
Write-Host "Branch: $Branch"
& git -C $RepoRoot status --short

$Answer = Read-Host "Type PUBLISH to commit the JRSI v2.0.0 tree, push main and push tag v2.0.0"
if ($Answer -cne "PUBLISH") {
    Write-Host "No GitHub changes were made."
    exit 0
}

& git -C $RepoRoot add -A
if ($LASTEXITCODE -ne 0) { throw "git add failed" }
& git -C $RepoRoot diff --cached --stat
& git -C $RepoRoot commit -m "Prepare JRSI blade-array model release v2.0.0"
if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
& git -C $RepoRoot push origin main
if ($LASTEXITCODE -ne 0) { throw "git push origin main failed" }
& git -C $RepoRoot tag -a v2.0.0 -m "JRSI blade-array model v2.0.0"
if ($LASTEXITCODE -ne 0) { throw "git tag failed" }
& git -C $RepoRoot push origin v2.0.0
if ($LASTEXITCODE -ne 0) { throw "git push origin v2.0.0 failed" }

Write-Host "Pushed main and tag v2.0.0. Create the GitHub Release for v2.0.0 to trigger Zenodo archiving."
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $ReleaseAnswer = Read-Host "GitHub CLI is installed. Create the GitHub Release now? Type RELEASE"
    if ($ReleaseAnswer -ceq "RELEASE") {
        & gh release create v2.0.0 --title "JRSI blade-array model v2.0.0" --notes-file CHANGELOG.md
        if ($LASTEXITCODE -ne 0) { throw "GitHub release creation failed" }
    }
} else {
    Write-Host "GitHub CLI is not installed. Create a release from tag v2.0.0 in the GitHub web interface."
}
