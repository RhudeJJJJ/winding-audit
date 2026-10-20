#!/usr/bin/env bash
# Fresh outputs go to a separate working directory; preserve bundled snapshots.
set -euo pipefail
AUDIT_PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$AUDIT_PROJECT_DIR"
export CACHE=${CACHE:-.winding_audit_cache}
CACHE=$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$CACHE")
export CACHE
AUDIT_RUN_DIR=${RUN_DIR:-reproduced-0.1.1}
mkdir -p "$AUDIT_RUN_DIR"
cd "$AUDIT_RUN_DIR"
python -m pytest -q "$AUDIT_PROJECT_DIR/tests"
python "$AUDIT_PROJECT_DIR/scripts/reproduce_chain.py" PHerc0139 20260102150214 --cache "$CACHE"
python "$AUDIT_PROJECT_DIR/scripts/reproduce_chain.py" PHerc1667 20251217075048 --cache "$CACHE"
python "$AUDIT_PROJECT_DIR/scripts/reproduce_duplicate.py" --cache "$CACHE"
python "$AUDIT_PROJECT_DIR/scripts/reproduce_planted_control.py" --cache "$CACHE"
python "$AUDIT_PROJECT_DIR/scripts/reproduce_selfcalib.py" --cache "$CACHE"
python "$AUDIT_PROJECT_DIR/scripts/reproduce_814_zones.py" --cache "$CACHE"
python "$AUDIT_PROJECT_DIR/scripts/reproduce_masked_1667.py" --cache "$CACHE"
python "$AUDIT_PROJECT_DIR/scripts/reproduce_audit_800.py" --cache "$CACHE"
python "$AUDIT_PROJECT_DIR/scripts/reproduce_placement.py"
python "$AUDIT_PROJECT_DIR/scripts/make_figures.py"
