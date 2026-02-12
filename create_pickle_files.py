import sys
from pathlib import Path
import timescape_functions as tf
# create_pickle_files.py "$JOB_LINE" "$SDSS_CSV" "$OUT_FOLDER" "$PROCESS"

sdss_csv_path = sys.argv[1]
out_folder = sys.argv[2]
process = int(sys.argv[3])

# Read job_list.txt locally
with open("jobs.txt") as f:
    lines = f.readlines()

line_arg = lines[process].strip()
print(f'{line_arg}\n{sdss_csv_path}\n{out_folder}\n{process}')

tf.save_job_pickle(line_arg, sdss_csv_path, out_folder, process)
