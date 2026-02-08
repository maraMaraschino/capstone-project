#!/bin/bash
set -e

PYTHON_SCRIPT=$1
BASE_URL=$2
PROCESS=$3
OUT_FOLDER=$4

# Make a folder inside the job scratch to hold pickle files
SCRATCH_DIR="$CONDOR_SCRATCH_DIR/$OUT_FOLDER"
mkdir -p "$SCRATCH_DIR"
mkdir -p SDSS ZOO
mv full_sdss.csv SDSS/full_sdss.csv
mv full_morphology.csv ZOO/full_morphology.csv

# Run Python to create the pickle files
python3 "$PYTHON_SCRIPT" "$BASE_URL" "$PROCESS" "$SCRATCH_DIR"
