# Reproduce the full v2.2.0 analysis with the project's pinned virtual environment.
# Full regeneration may be computationally expensive; this script is NOT called by the publisher.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python missing: $Python. Create .venv and install requirements first."
}
if (-not $env:NPROC) { $env:NPROC = '2' }
Write-Host "Using Python: $Python"
& $Python --version
if ($LASTEXITCODE -ne 0) { throw 'Python could not be started.' }

$Stages = @('ensemble', 'numerics', 'knockouts', 'design', 'chemistry',
            'longtime', 'gsa_geometry', 'gsa_mechanism')
foreach ($Stage in $Stages) {
    Write-Host "=== $Stage ==="
    & $Python analysis_v2.py --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw "Analysis stage failed: $Stage (exit $LASTEXITCODE)" }
}
foreach ($Script in @('stats_v2.py', 'figures_v2.py', 'make_tables.py')) {
    Write-Host "=== $Script ==="
    & $Python $Script
    if ($LASTEXITCODE -ne 0) { throw "Script failed: $Script (exit $LASTEXITCODE)" }
}
foreach ($Stage in @('structural','spacing','coverage','factorial','phase','indep','matched_grid','grid')) {
    Write-Host "=== controls: $Stage ==="
    & $Python analysis_controls.py $Stage
    if ($LASTEXITCODE -ne 0) { throw "Controls failed: $Stage (exit $LASTEXITCODE)" }
}
foreach ($Script in @('tables_controls.py', 'figures_controls.py')) {
    Write-Host "=== $Script ==="
    & $Python $Script
    if ($LASTEXITCODE -ne 0) { throw "Script failed: $Script (exit $LASTEXITCODE)" }
}
Write-Host 'All computational stages completed. Rebuild the PDFs with build_manuscripts.ps1 or Overleaf.'
