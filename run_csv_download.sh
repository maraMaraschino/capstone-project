#!/bin/bash
set -e

echo "SCRATCH = $_CONDOR_SCRATCH_DIR"

mkdir -p ZOO
mkdir -p SDSS

python3 download_csv_files.py