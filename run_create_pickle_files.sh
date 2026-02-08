#!/bin/bash
set -e

PYTHON_SCRIPT=$1
SOURCE_FOLDER=$2
PROCESS=$3
OUT_FOLDER=$4

# Make a folder inside the job scratch to hold pickle files
SCRATCH_PICKLE="$OUT_FOLDER"
mkdir -p "$SCRATCH_PICKLE"

# Run Python to create the pickle files
python3 "$PYTHON_SCRIPT" "$SOURCE_FOLDER" "$PROCESS" "$SCRATCH_PICKLE"
