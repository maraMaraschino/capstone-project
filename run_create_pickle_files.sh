#!/bin/bash
set -e

# arguments = $(file_line) SDSS/full_sdss.csv builder_pickle_folder $(Process)           
SDSS_CSV=$1           
OUT_FOLDER=$2         
PROCESS=$3

# Make a folder inside the job scratch to hold pickle files
mkdir -p "$OUT_FOLDER"

# Run Python to create the pickle files
python3  create_pickle_files.py "$SDSS_CSV" "$OUT_FOLDER" "$PROCESS"