# Test function to create list of valid FITS files to pickle
from astropy.io import fits
import pandas as pd
import sys
from pathlib import Path

input_csv = sys.argv[1]
output_csv = sys.argv[2]
start = int(sys.argv[3])
end = int(sys.argv[4])

fits_dir = Path("/mnt")
file_list = sorted(
    f for f in fits_dir.iterdir() 
    if f.is_file() and f.suffix.lower() == '.fits'
)

print(f'Processing files: {start}:{end} (out of {len(file_list)})')
file_list = file_list[start:end]

def grab_files(filelist, csv_file_path, output_csv):
    valid_files = []
    volume_df = pd.read_csv(csv_file_path)
    # Collect the index of every objid for recall
    objid_set = set(volume_df['objid'].astype(str))
    print(f'Trying to find {len(filelist)} files.')
    for filepath in filelist:
        try:
            hdul = fits.open(filepath, memmap=True)
            specobj = hdul[2].data
        except:
            print(f'Failed to open .fits file: {filepath}')
            objid = None
            continue
        try:
            objid = specobj['bestObjID'][0]
        except Exception:
            try:
                objid = specobj['OBJID'][0]
            except Exception:
                print(f'Failed to find objid for {filepath}')
                objid = None
                continue
        # Search objid_to_index for objid, append to list if valid
        if str(objid) in objid_set:
            valid_files.append(str(filepath))

    print(f'Finished finding files!')
    print(f'Found {len(valid_files)} valid FITs files ({(len(valid_files)/len(filelist)) * 100:.2f}%)')
    print(f'Writing {output_csv}...')
    
    # Create dataframe
    out_df = pd.DataFrame({
        "fits_path": valid_files
    })

    # Convert to csv
    out_df.to_csv(output_csv, index=False)
    print(f'Done!')

grab_files(file_list, input_csv, output_csv)