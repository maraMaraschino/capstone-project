import sys
from pathlib import Path
import timescape_functions as tf

sdss_csv_path = sys.argv[1]
out_folder = sys.argv[2]
process = int(sys.argv[3])

# Read job_list.txt locally
with open("job_list.txt") as f:
    lines = f.readlines()

line_arg = lines[process].strip()

tf.save_job_pickle(line_arg, sdss_csv_path, out_folder, process)
