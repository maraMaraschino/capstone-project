#!/bin/bash
set -e

# positional arguments from the submit file
PYTHON_SCRIPT=$1
CSV_FILE=$2
PROCESS=$3

echo "SCRATCH = $_CONDOR_SCRATCH_DIR"
echo "Script = $PYTHON_SCRIPT"
echo "CSV = $CSV_FILE"
echo "Process = $PROCESS"

mkdir -p FITS

# pass arguments to python
python3 "$PYTHON_SCRIPT" "$CSV_FILE" "$PROCESS" "$_CONDOR_SCRATCH_DIR/FITS"
