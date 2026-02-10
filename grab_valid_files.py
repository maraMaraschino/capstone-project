import timescape_functions as tf
from astropy.io import fits
import pandas as pd
import os
import sys
import Path

fits_dir = ""
file_list = [f for f in os.listdir(fits_dir) if os.path.isfile(f)]

def grab_files(filelist, csv_file_path):
    volume_df = pd.read_csv(csv_file_path)
    objid_all = volume_df['objid'].astype(str).values
    # Collect the index of every object for recall
    objid_to_index = {str(objid): i for i, objid in enumerate(objid_all)}
    for file in filelist:
        hdul = fits.open(file, memmap=False)
        specobj = hdul[2].data
        try:
            objid = specobj['bestObjID'][0]
        except Exception as e:
            #print(f'Error on {fileid}: {e}')
            #print(f'Trying new key...')
            try:
                objid = specobj['OBJID'][0]
                #print('Success!')
            except Exception as e:
                #print(f'Failed again on {objid}: {e}')
                #print(f'Skipping object...')
                objid = None
        if objid not in objid_to_index:
            print(f"Failed to find objid {objid} in csv file...")
            continue


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