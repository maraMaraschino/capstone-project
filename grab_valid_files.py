# Test function to create list of valid FITS files to pickle
from astropy.io import fits
import pandas as pd
import sys
from pathlib import Path

input_csv = sys.argv[1]
output_csv = sys.argv[2]

fits_dir = Path("/mnt")
file_list = [f for f in fits_dir.iterdir() if f.is_file() and f.suffix.lower() == '.fits']
print(f'Found {len(file_list)} FITS files...')

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

grab_files(file_list, input_csv, output_csv)