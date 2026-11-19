#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export NPROC="${NPROC:-2}"
for stage in ensemble numerics knockouts design chemistry longtime gsa_geometry gsa_mechanism; do
  echo "=== ${stage} ==="
  python3 analysis_v2.py --stage "${stage}"
done
python3 stats_v2.py
python3 figures_v2.py
python3 make_tables.py
