import os
import pandas as pd
from pathlib import Path
import sys
import re

# Load csv file
# SDSS/full_sdss.csv
input_csv = sys.argv[1]
full_df = pd.read_csv(input_csv)

# SDSS/partial_sdss.csv
output_csv = sys.argv[2]

# capture plate, mjd, and fiber
full_df['plate'] = full_df['plate'].astype(int)
full_df['mjd'] = full_df['mjd'].astype(int)
full_df['fiberid'] = full_df['fiberid'].astype(int)
print(f'{len(full_df)} total FITS files.')

# Load folder of .fits files
fits_dir = Path('/mnt')
file_list = [f for f in fits_dir.iterdir() if f.is_file() and f.suffix.lower() == '.fits']
print(f'Found {len(file_list)} FITS files in {fits_dir}...')
print(f'First 10 files: {[f.name for f in file_list[:10]]}')

# capture plate, mjd, and fiber from filename
pattern = r"^spec-(\d+)-(\d+)-(\d+)\.fits$"

# build set of existing (plate, mjd, fiber)
existing = set()

for f in file_list:
    match = re.match(pattern, f.name)
    if match:
        plate, mjd, fiber = match.groups()
        existing.add((int(plate), int(mjd), int(fiber)))

print(f'Found {len(existing)} valid FITS filenames.')

# Find rows not present in folder
missing_mask = ~full_df.apply(
    lambda row: (row['plate'], row['mjd'], row['fiberid']) in existing,
    axis=1
)

missing_df = full_df[missing_mask]

print(f"Found {len(missing_df)} missing files.")

missing_df.to_csv(output_csv, index=False)

print(f'Wrote missing entries to {output_csv}')