import os
import timescape_functions as tf
import sys
from pathlib import Path

csv_file = sys.argv[1]
process = int(sys.argv[2])
fits_dir = sys.argv[3]

# Always make a local directory
Path(fits_dir).mkdir(parents=True, exist_ok=True)

CHUNK = 50
TOTAL_ROWS = 377294

start = process * CHUNK
end   = min((process + 1) * CHUNK, TOTAL_ROWS)

try:
    tf.download_fits_chunk(csv_file, start, end, fits_dir)
except Exception as e:
    print(f'Submission failed: {e}\n {start} until {end}')