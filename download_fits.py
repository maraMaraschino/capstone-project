import os
import timescape_functions as tf
import sys

scratch = os.environ.get("_CONDOR_SCRATCH_DIR", ".")
fits_dir = f"{scratch}/FITS"
os.makedirs(fits_dir, exist_ok=True)

CHUNK = 500
TOTAL_ROWS = 377294

csv_file = sys.argv[1]
process = int(sys.argv[2])  # $(Process) from Condor

start = process * CHUNK
end   = min((process + 1) * CHUNK, TOTAL_ROWS)

try:
    tf.download_fits_chunk(csv_file, start, end, fits_dir)
except Exception as e:
    print(f'Submission failed: {e}\n {start} until {end}')