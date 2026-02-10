# Test function to create list of valid FITS files to pickle
import timescape_functions as tf
from astropy.io import fits
import pandas as pd
import os
import sys
from pathlib import Path

fits_dir = Path("FITS")
file_list = [f for f in fits_dir.iterdir() if f.is_file()]
file_list = file_list[:20]

def grab_files(filelist, csv_file_path, output_csv):
    valid_files = []
    volume_df = pd.read_csv(csv_file_path)
    objid_all = volume_df['objid'].astype(str).values
    # Collect the index of every objid for recall
    objid_to_index = {str(objid): i for i, objid in enumerate(objid_all)}
    for filepath in filelist:
        # Skip non fits files
        try:
            hdul = fits.open(filepath, memmap=False)
            specobj = hdul[2].data
        except:
            print(f'Could not read {filepath} as .fits file...')
            objid = None
            continue
        
        # Try multiple methods of grabbing objIDs
        try:
            objid = specobj['bestObjID'][0]
        except Exception:
            try:
                objid = specobj['OBJID'][0]
            except Exception:
                objid = None
        
        # Search objid_to_index for objid, append to list if valid
        if objid not in objid_to_index:
            print(f"Failed to find objid {objid} in csv file. Skipping file...")
            continue
        else:
            valid_files.append(str(filepath))
    
    # Create dataframe
    out_df = pd.DataFrame({
        "fits_path": valid_files
    })

    # Convert to csv
    out_df.to_csv(output_csv, index=False)

grab_files(file_list, "SDSS/full_sdss.csv", "valid_fits_files.csv")


# have grab_files return csv file
# cp grab_valid_files.py /ospool/ap40/data/adrian.fisher/FITS/
# apptainer shell --bind .:/mnt /ospool/ap40/data/adrian.fisher/mycontainer2.sif
# run python script from mnt directory 
# python3 grab_valid_files.py
# grab valid files in apptainer shell - return to mnt
# new function to split files by job and return jobs.txt file
# each line on jobs.txt is a list of files that are valid with osdf:///ospool/ap40/data/adrian.fisher/FITS/ in front
# transfer_input_files = $(file_list)
# queue file_list from jobs.txt