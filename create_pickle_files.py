import timescape_functions as tf
import os
import sys
from pathlib import Path

#scratch = os.getcwd()
source_folder = sys.argv[1]
process = int(sys.argv[2])
out_folder = sys.argv[3]

chunk = 10
total_files = len(list(Path(source_folder).iterdir()))

start = process * chunk
end = min((process + 1) * chunk, total_files)

filename = f"pickle_file_{start}_{end}.pkl"

print("SOURCE_FOLDER:", source_folder)
print("FILES FOUND:", len(list(Path(source_folder).iterdir())))

try:
    tf.save_job_pickle(source_folder, out_folder, start, end)
except Exception as e:
    print(f'Error saving pickle file:\n{e}')

