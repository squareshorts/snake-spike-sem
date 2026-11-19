#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v xelatex >/dev/null 2>&1 || { echo "xelatex is required" >&2; exit 1; }
cd "$ROOT/manuscript"
for tex in manuscript.tex supplement.tex cover_letter.tex; do
  xelatex -interaction=nonstopmode -halt-on-error "$tex"
  xelatex -interaction=nonstopmode -halt-on-error "$tex"
done
