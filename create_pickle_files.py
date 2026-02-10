import timescape_functions as tf
import os
import sys
from pathlib import Path
import pandas as pd

#scratch = os.getcwd()
#base_url = sys.argv[1]
process = int(sys.argv[1])
out_folder = sys.argv[2]

chunk = 10
source_csv_file = "SDSS/full_sdss.csv"
df = pd.read_csv(source_csv_file)
total_files = len(df)

start = process * chunk
end = min((process + 1) * chunk, total_files)

filename = f"pickle_file_{start}_{end}.pkl"

#print("BASE_URL:", base_url)
print("FILES FOUND:", total_files)

try:
    tf.save_job_pickle(base_url, source_csv_file, out_folder, start, end)
except Exception as e:
    print(f'Error saving pickle file:\n{e}')

