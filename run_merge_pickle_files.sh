#!/bin/bash
set -e

# positional arguments from the submit file

PYTHON_SCRIPT=$1
FILENAME=$2
SOURCE_FOLDER=$3
OUT_FOLDER=$4

mkdir -p "$OUT_FOLDER"

python3 "$PYTHON_SCRIPT" "$SOURCE_FOLDER" "FILENAME" "$OUT_FOLDER"