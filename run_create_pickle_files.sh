#!/bin/bash
set -e

JOB_LINE=$2           
SDSS_CSV=$3           
OUT_FOLDER=$4         
PROCESS=$5

# Make a folder inside the job scratch to hold pickle files
mkdir -p "$OUT_FOLDER"

# Run Python to create the pickle files
python3  create_pickle_files.py "$JOB_LINE" "$SDSS_CSV" "$OUT_FOLDER" "$PROCESS"