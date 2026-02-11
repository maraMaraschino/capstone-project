import os
import timescape_functions as tf
import sys
import pandas as pd
from pathlib import Path

csv_file = sys.argv[1]
len_csv = pd.read_csv(csv_file)
print(f'Total files to attempt to download: {len(len_csv)}')
process = int(sys.argv[2])
fits_dir = Path(sys.argv[3])  # convert to Path

# Always make a local directory first
fits_dir.mkdir(parents=True, exist_ok=True)

CHUNK = 10000
OFFSET = 50365
TOTAL_ROWS = len(len_csv)

start = (process * CHUNK) + OFFSET
end   = min(((process + 1) * CHUNK) + OFFSET, TOTAL_ROWS)

try:
    print(f'Writing {end-start} FITS files to {fits_dir}... ')
    tf.download_fits_chunk(csv_file, start, end, fits_dir)
except Exception as e:
    print(f'Submission failed: {e}\n {start} until {end}')
