#!/bin/bash
set -e

# positional arguments from the submit file

PYTHON_SCRIPT=$1
SOURCE_FOLDER=$2
OUT_FOLDER=$3

mkdir -p "$OUT_FOLDER"

python3 "$PYTHON_SCRIPT" "$SOURCE_FOLDER" "$_CONDOR_SCRATCH_DIR/final_pickle"