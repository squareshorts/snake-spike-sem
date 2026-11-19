$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$PythonCommand = Get-Command python -ErrorAction Stop
$Python = $PythonCommand.Source
if (-not $env:NPROC) { $env:NPROC = "2" }

$Stages = @(
    "ensemble",
    "numerics",
    "knockouts",
    "design",
    "chemistry",
    "longtime",
    "gsa_geometry",
    "gsa_mechanism"
)

foreach ($Stage in $Stages) {
    Write-Host "=== $Stage ==="
    & $Python analysis_v2.py --stage $Stage
    if ($LASTEXITCODE -ne 0) { throw "Analysis stage failed: $Stage (exit $LASTEXITCODE)" }
}

Write-Host "=== Statistics ==="
& $Python stats_v2.py
if ($LASTEXITCODE -ne 0) { throw "stats_v2.py failed (exit $LASTEXITCODE)" }

Write-Host "=== Figures ==="
& $Python figures_v2.py
if ($LASTEXITCODE -ne 0) { throw "figures_v2.py failed (exit $LASTEXITCODE)" }

Write-Host "=== Manuscript tables and numerical macros ==="
& $Python make_tables.py
if ($LASTEXITCODE -ne 0) { throw "make_tables.py failed (exit $LASTEXITCODE)" }

if (Get-Command xelatex -ErrorAction SilentlyContinue) {
    Write-Host "LaTeX engine found. Rebuild the PDFs with .\build_manuscripts.ps1 when desired."
} else {
    Write-Host "No XeLaTeX executable found. The numerical outputs are complete; compile the LaTeX sources in Overleaf."
}
