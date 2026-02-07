import os
import timescape_functions as tf
import sys
from pathlib import Path
import subprocess

csv_file = sys.argv[1]
process = int(sys.argv[2])
fits_dir = Path(sys.argv[3])  # convert to Path

# Always make a local directory first
fits_dir.mkdir(parents=True, exist_ok=True)

CHUNK = 50
TOTAL_ROWS = 377294

start = process * CHUNK
end   = min((process + 1) * CHUNK, TOTAL_ROWS)

try:
    print(f'Writing {end-start} FITS files to {fits_dir}... ')
    tf.download_fits_chunk(csv_file, start, end, fits_dir)
except Exception as e:
    print(f'Submission failed: {e}\n {start} until {end}')


