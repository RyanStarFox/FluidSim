#!/bin/bash
# Sequential render queue: all 6 FLIP ratios × 2 scenes = 12 renders
# dam_break then liquid_pouring for each ratio.

set -e
cd "$(dirname "$0")"
PY=/opt/homebrew/anaconda3/envs/fluidsim/bin/python
LOG_DIR=/tmp/render_logs
mkdir -p "$LOG_DIR"

run_scene() {
    local scene=$1 ratio=$2
    local tag="ratio$(echo "$ratio" | tr -d '.')"
    echo ""
    echo "================================================================"
    echo "  START  $scene  ratio=$ratio  $(date '+%H:%M:%S')"
    echo "================================================================"
    "$PY" framework.py "$scene" "$ratio" 2>&1 | tee "$LOG_DIR/${scene}_${tag}.log"
    echo "  DONE   $scene  ratio=$ratio  $(date '+%H:%M:%S')"
}

for ratio in 0.0 0.5 0.8 0.95 0.97 1.0; do
    run_scene dam_break      "$ratio"
    run_scene liquid_pouring "$ratio"
done

echo ""
echo "================================================================"
echo "  ALL 12 RENDERS COMPLETE  $(date '+%H:%M:%S')"
echo "================================================================"
ls -lh output/flip/ratio_*/dam_break/*.mp4 output/flip/ratio_*/liquid_pouring/*.mp4 2>/dev/null
