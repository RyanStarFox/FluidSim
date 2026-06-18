#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
ZIP_PATH="$DIST/fluidsim_final_submission.zip"

if ! command -v zip >/dev/null 2>&1; then
  echo "zip is not installed. Please install zip or create the archive manually." >&2
  exit 1
fi

mkdir -p "$DIST"
rm -f "$ZIP_PATH"

cd "$ROOT"

zip -r "$ZIP_PATH" \
  README.md \
  division_of_work.md \
  framework.py \
  plot_energy.py \
  render_all.sh \
  scripts \
  docs \
  submission \
  output/flip/ratio_970 \
  output/apic/ratio_970 \
  output/polypic/ratio_970 \
  output/comparison \
  output/benchmark_efficiency/benchmark_metadata.txt \
  output/benchmark_efficiency/efficiency_runs.csv \
  output/benchmark_efficiency/efficiency_summary.csv \
  output/benchmark_efficiency/dam_break_efficiency_ms_per_frame.png \
  output/benchmark_efficiency/liquid_pouring_efficiency_ms_per_frame.png \
  output/benchmark_efficiency/speedup_vs_flip.png \
  -x '*/__pycache__/*' \
  -x '*/frames/*' \
  -x 'output/benchmark_efficiency/raw/*' \
  -x 'dist/*'

echo "Wrote $ZIP_PATH"
