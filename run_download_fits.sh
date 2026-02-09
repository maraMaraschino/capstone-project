#!/bin/bash
set -e

PYTHON_SCRIPT=$1
CSV_FILE=$2
PROCESS=$3
OUT_FOLDER=$4

echo "Writing directly to $OUT_FOLDER"

mkdir -p "$OUT_FOLDER"

python3 "$PYTHON_SCRIPT" "$CSV_FILE" "$PROCESS" "$OUT_FOLDER"
