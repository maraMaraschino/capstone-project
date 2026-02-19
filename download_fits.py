import os
import timescape_functions as tf
import sys
import pandas as pd
from pathlib import Path

csv_file = sys.argv[1]
len_csv = pd.read_csv(csv_file)
process = int(sys.argv[2])
fits_dir = Path(sys.argv[3])  # convert to Path

# Always make a local directory first
fits_dir.mkdir(parents=True, exist_ok=True)


print(f'csv_file: {csv_file}\nlen_csv: {len(len_csv)}\nprocess: {process}\nfits_dir: {fits_dir}')

CHUNK = 1000
TOTAL_ROWS = len(len_csv)

start = (process * CHUNK)
end   = min(((process + 1) * CHUNK), TOTAL_ROWS)

print(f'Attempting to download rows {start} to {end} of {csv_file}...')

try:
    print(f'Writing {end-start} FITS files to {fits_dir}... ')
    tf.download_fits_chunk(csv_file, start, end, fits_dir)
except Exception as e:
    print(f'Submission failed: {e}\n {start} until {end}')
