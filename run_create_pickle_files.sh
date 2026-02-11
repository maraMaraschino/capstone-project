#!/bin/bash
set -e

JOB_LINE=$1
SDSS_CSV=$2
OUT_FOLDER=$3
PROCESS=$4

mkdir -p "$OUT_FOLDER"

python3 create_pickle_files.py "$JOB_LINE" "$SDSS_CSV" "$OUT_FOLDER" "$PROCESS"
