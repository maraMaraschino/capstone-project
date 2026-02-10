#!/bin/bash
set -e

PYTHON_SCRIPT=$1
PROCESS=$2
OUT_FOLDER=$3

# Make a folder inside the job scratch to hold pickle files
SCRATCH_DIR="$OUT_FOLDER"
mkdir -p "$SCRATCH_DIR"
mkdir -p SDSS ZOO

mv full_sdss.csv SDSS/full_sdss.csv
mv full_morphology.csv ZOO/full_morphology.csv

# Run Python to create the pickle files
python3 "$PYTHON_SCRIPT" "$PROCESS" "$OUT_FOLDER"
​
# Create tarball of outputs
tar -czf builder_pickle_files.tar builder_pickle_files/