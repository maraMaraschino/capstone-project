import pandas as pd

def convert_csv_to_txt(csv_file, txt_file, chunk_size):
    df = pd.read_csv(csv_file, header=0)
    paths = df["fits_path"].astype(str).tolist()
    osdf_prefix = "osdf:///ospool/ap40/data/adrian.fisher/FITS/"

    with open(txt_file, "w") as f:
        for i in range(0, len(paths), chunk_size):
            batch = paths[i:i+chunk_size]
            full_paths = [osdf_prefix + name for name in batch]
            f.write(" ".join(full_paths) + "\n")