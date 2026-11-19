$ErrorActionPreference = "Stop"
$XeLaTeX = Get-Command xelatex -ErrorAction Stop
$ManuscriptDir = Join-Path $PSScriptRoot "manuscript"
Push-Location $ManuscriptDir
try {
    foreach ($Tex in @("manuscript.tex", "supplement.tex", "cover_letter.tex")) {
        & $XeLaTeX.Source -interaction=nonstopmode -halt-on-error $Tex
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed for $Tex (exit $LASTEXITCODE)" }
        & $XeLaTeX.Source -interaction=nonstopmode -halt-on-error $Tex
        if ($LASTEXITCODE -ne 0) { throw "Second XeLaTeX pass failed for $Tex (exit $LASTEXITCODE)" }
    }
}
finally {
    Pop-Location
}
