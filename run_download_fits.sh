#!/bin/bash
set -e

PYTHON_SCRIPT=$1
CSV_FILE=$2
PROCESS=$3
OUT_FOLDER=$4

echo "SCRATCH = $_CONDOR_SCRATCH_DIR"
echo "Script = $PYTHON_SCRIPT"
echo "CSV = $CSV_FILE"
echo "Process = $PROCESS"
echo "Output folder = $OUT_FOLDER"

# Create local folder in scratch
mkdir -p "$OUT_FOLDER"

python3 "$PYTHON_SCRIPT" "$CSV_FILE" "$PROCESS" "$OUT_FOLDER"
