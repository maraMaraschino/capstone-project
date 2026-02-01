import pandas as pd
import requests
from pathlib import Path
from collections import Counter, defaultdict
from astroquery.sdss import SDSS
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as const
import pickle
import matplotlib.pyplot as plt

def sdss_chunk_query(chunk_size, last_id, file_name, folder_name):
    """
    SQL search SDSS database to return a csv file with the objid, plate, mjd, fiberid,
    and FITS file URL for all galaxies between z=0.13 to z=0.3, 50000 galaxies at a time
    to prevent timeout.
    """
    # 
    sdss_chunk = f"""
SELECT TOP {chunk_size}
p.objid, s.plate, s.mjd, s.fiberid,
dbo.fGetUrlFitsSpectrum(s.specObjID) AS spec_fits_url
FROM PhotoObj AS p
JOIN SpecObj AS s
    ON p.objid = s.bestobjid
JOIN Galaxy AS g
    ON g.objid = p.objid
    WHERE s.class = 'GALAXY'
    AND s.z BETWEEN 0.15 AND 0.3
    AND s.zWarning = 0
    AND p.objid > {last_id}
ORDER BY p.objid
    """
    table = SDSS.query_sql(sdss_chunk)
    if table is None:
        return None, None
    last_id = table[-1][0]
    new_file_name = f'{file_name}{last_id}.csv'
    table.write(f"{folder_name}/{new_file_name}", format="csv", overwrite=True)
    return last_id, new_file_name

def galaxy_zoo_chunk_query(chunk_size, last_id, file_name, folder_name):
    """
    SQL search SDSS database to return a csv file with the confidence rating for
    if a galaxy is elliptical, clockwise spiral, anticlockwise spiral, edgeon,
    unknown, or merger for every shared SDSS & Galaxy Zoo object between z=0.15 to z=0.3
    """
    galaxy_zoo_chunk = f"""
SELECT TOP {chunk_size}
p.objid,
zns.p_el as elliptical,
zns.p_cw as spiralclock,
zns.p_acw as spiralanticlock,
zns.p_edge as edgeon,
zns.p_dk as dontknow,
zns.p_mg as merger
FROM PhotoObj AS p
JOIN SpecObj AS s
    ON p.objid = s.bestobjid
JOIN Galaxy AS g
    ON g.objid = p.objid
JOIN ZooNoSpec AS zns
    ON zns.objid = g.objid
WHERE 
    s.class = 'GALAXY'
    AND s.z BETWEEN 0.15 AND 0.3
    AND s.zWarning = 0
    AND p.objid > {last_id}
ORDER BY p.objid
    """
    table = SDSS.query_sql(galaxy_zoo_chunk)
    if table is None:
        return None, None
    last_id = table[-1][0]
    new_file_name = f'{file_name}{last_id}.csv'
    table.write(f"{folder_name}/{new_file_name}", format="csv", overwrite=True)
    return last_id, new_file_name

def merge_csv(files, final_file, final_folder):
    """
    Take a list of CSV files and combine them into a single file.
    """
    outdir = Path(final_folder)
    outdir.mkdir(exist_ok=True)
    df_list = [pd.read_csv(f) for f in files]
    combined = pd.concat(df_list, ignore_index=True)
    combined.to_csv(f"{outdir}/{final_file}", index=False)
    return combined

def cleanup_files(files):
    """
    Deletes a list files, helping to conserve memory.
    """
    for f in files:
        Path(f).unlink(missing_ok=True)


def loop_galaxy_chunk(query, chunk_size, last_id, file_name, final_file, folder_name):
    """
    Loop the SQL search until exhausted, at which point concatenate all generated CSV files into a new one.
    """
    csv_file_list = []
    outdir = Path(folder_name)
    outdir.mkdir(exist_ok=True)
    while True:
        try:
            print(f'Collecting next {chunk_size} galaxies...')
            last_id, new_file_name = query(chunk_size, last_id, file_name, folder_name)
            if last_id is None:
                break
            else:
                new_file_name = f"{outdir}\{new_file_name}"
                csv_file_list.append(new_file_name)
                print("Success!")
        except Exception as e:
            print(f'Error: query failed at {last_id}\n{e}')
            raise
    print(f'Finished collecting galaxies. Merging {len(csv_file_list)} CSV files...')
    merge_csv(csv_file_list, final_file, folder_name)
    print("Deleting builder files...")
    cleanup_files(csv_file_list)
    print("Done!")

# Creating method to download data from the FITS file
def download_fits_data(df):
    """
    Uses results of create_full_galaxy_table to fill a FITS folder with the 
    downloaded FITS files of all available galaxies.
    """
    outdir = Path('FITS')
    outdir.mkdir(exist_ok=True)

    for _, row in df.iterrows():
        objid = row['objid']
        plate = row["plate"]
        mjd   = row["mjd"]
        fiber = row["fiberid"]
        url   = f""

        filename = f'spec-{plate}-{mjd}-{fiber:04d}.fits'
        filepath = outdir / filename

        if filepath.exists():
            continue

        r = requests.get(url, stream=True)
        r.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)