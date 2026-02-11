import pandas as pd
import sys

valid_fits_csv_file = sys.argv[1]
job_txt_file = sys.argv[2]
chunk_size = 150

def convert_csv_to_txt(valid_fits_csv_file, job_txt_file, chunk_size):
    df = pd.read_csv(valid_fits_csv_file, header=0)
    paths = df["fits_path"].astype(str).tolist()
    osdf_prefix = "osdf:///ospool/ap40/data/adrian.fisher/FITS"

    with open(job_txt_file, "w") as f:
        for i in range(0, len(paths), chunk_size):
            batch = paths[i:i+chunk_size]
            full_paths = [osdf_prefix + name for name in batch]
            f.write(",".join(full_paths) + "\n")

convert_csv_to_txt(valid_fits_csv_file, job_txt_file, chunk_size)