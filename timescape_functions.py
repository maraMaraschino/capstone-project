import pandas as pd
import requests
from pathlib import Path
from astroquery.sdss import SDSS
from astropy.io import fits
import numpy as np
import csv
from collections import defaultdict
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as const
import pickle
import matplotlib.pyplot as plt
import time
from scipy import stats
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
import random
import os
#from pelicanfs.core import OSDFFileSystem

def sdss_chunk_query(chunk_size, last_id, file_name, folder_name):
    """
    SQL search SDSS database to return a csv file with the objid, plate, mjd, fiberid,
    and FITS file URL for all galaxies between z=0.13 to z=0.3, 50000 galaxies at a time
    to prevent timeout.
    """
    sdss_chunk = f"""
SELECT TOP {chunk_size}
p.objid, s.plate, s.mjd, s.fiberid, s.z, p.ra, p.dec,
dbo.fGetUrlFitsSpectrum(s.specObjID) AS spec_fits_url
FROM PhotoObj AS p
JOIN SpecObj AS s
    ON p.objid = s.bestobjid
JOIN Galaxy AS g
    ON g.objid = p.objid
    WHERE s.class = 'GALAXY'
    AND s.z BETWEEN 0.1397816562350196 AND 0.311104966694253
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
    outdir.mkdir(parents=True, exist_ok=True)
    df_list = [pd.read_csv(f) for f in files]
    combined = pd.concat(df_list, ignore_index=True)
    combined.to_csv(f"{outdir}/{final_file}", index=False)
    return combined

def cleanup_files(files):
    """
    Deletes list files, helping to conserve memory.
    """
    for f in files:
        Path(f).unlink(missing_ok=True)

def loop_galaxy_chunk(query, chunk_size, last_id, file_name, final_file, folder_name):
    csv_file_list = []
    outdir = Path(folder_name)
    outdir.mkdir(parents=True, exist_ok=True)

    while True:
        retries = 0

        while retries <= 5:
            try:
                print(f'Collecting next {chunk_size} galaxies for {folder_name}...')
                last_id, new_file_name = query(chunk_size, last_id, file_name, folder_name)

                if last_id is None:
                    break  # exhausted — exit retry loop

                print(f'Creating builder file: {new_file_name}')
                csv_file_list.append(str(outdir / new_file_name))
                break

            except Exception as e:
                retries += 1
                if retries > 5:
                    raise
                time.sleep(3)

        if last_id is None:
            break  # exhausted — exit outer loop

    print(f'Merging {len(csv_file_list)} CSV files for {folder_name}...')
    merge_csv(csv_file_list, final_file, folder_name)
    cleanup_files(csv_file_list)
    print("Done!")

def download_fits_chunk(source_csv_file, start, end, outdir):
    """
    Uses final SDSS CSV file to fill a FITS folder with the 
    downloaded FITS files of all available galaxies in the desired redshift range (0.15-0.3).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'download_fits_chunk running from {start} to {end}...')
    print(f'Attempting to return {end-start} files to {outdir}...')

    existing = {p.name for p in outdir.glob("spec-*.fits")}

    df = pd.read_csv(source_csv_file, header=0)
    valid_df = df[(df['z'] >= 0.15) & (df['z'] <= 0.3)]

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for i in range(start, end):

        row = df.iloc[i]
        # Only select files in volume-limited range
        
        plate = row["plate"]
        mjd   = row["mjd"]
        fiber = row["fiberid"]
        url   = row["spec_fits_url"]
        url   = url.replace("http://", "https://")

        filename = f'spec-{plate:04d}-{mjd}-{fiber:04d}.fits'

        if filename in existing:
            print(f"{filename} already exists. Skipping...")
            continue

        filepath = outdir / filename

        # Handle non-existing FITS files:
        if not isinstance(url, str) or not url.strip():
            print(f"No valid url for {filename}, skipping.")
            continue
        
        print("Next: ", filepath)
        
        try:
            # Don't flood url requests
            timesleep = random.randint(10,25)
            print(f'Waiting for {timesleep} seconds before downloading row {i}...')
            time.sleep(timesleep)
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            r.raise_for_status()
            print(f'Downloading file...')
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

                existing.add(filename)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}:\n{e}")
            continue

def collect_spectrum_data(file):
    """
    Compute D4000n, collect [OII] 3727 & Hdelta flux/EW from a single SDSS FITS spectrum file.

    Negative Err values = invalid fit.

    Flux is negative for absorption spectra, positive for emission.
    """
    # Open file
    hdul = fits.open(file, memmap=False)
    
    # Assign dfs
    hdu     = hdul[0].header
    coadd   = hdul[1].data
    specobj = hdul[2].data
    spzline = hdul[3].data

    # Info
    ra  = hdu['PLUG_RA']
    dec = hdu['PLUG_DEC']

    # Label data
    plate = spzline['PLATE'][0]
    mjd   = spzline['MJD'][0]
    fiber = spzline['FIBERID'][0]
    fileid = f'spec-{plate}-{mjd}-{fiber:04d}'
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

    # Flux and wavelength
    flux       = coadd['flux']
    loglam     = coadd['loglam']
    lambda_obs = 10**loglam
    ivar       = coadd['ivar']

    # Redshift
    z     = specobj['Z'][0]
    z_err = specobj['Z_ERR'][0]

    # Shift to rest-frame
    lambda_rest = lambda_obs / (1 + z)

    # Find D4000n and uncertainty
    red_mask     = (lambda_rest >= 4000) & (lambda_rest <= 4100)
    blue_mask    = (lambda_rest >= 3850) & (lambda_rest <= 3950)
   
    F_red        = np.mean(flux[red_mask])
    red_ivar     = ivar[red_mask]

    F_blue       = np.mean(flux[blue_mask])
    blue_ivar    = ivar[blue_mask]

    D4000n       = F_red / F_blue

    # Protect against divisions by zero
    good_red = red_ivar > 0
    good_blue = blue_ivar > 0
    if good_red.sum() == 0 or good_blue.sum() == 0:
        sigma_D4000n = np.inf
    else:    
        sigma_red    = np.sqrt(1 / np.sum(red_ivar[good_red]))
        sigma_blue   = np.sqrt(1 / np.sum(blue_ivar[good_blue]))
        sigma_D4000n = D4000n * np.sqrt(
        (sigma_red / F_red)**2 +
        (sigma_blue / F_blue)**2
    )
    

    # OII 3727 Flux and EW
    oii_mask     = spzline['LINENAME']=='[O_II] 3727'
    oii_flux     = spzline['LINEAREA'][oii_mask][0]
    oii_flux_err = spzline['LINEAREA_ERR'][oii_mask][0]
    oii_EW       = spzline['LINEEW'][oii_mask][0]
    oii_EW_err   = spzline['LINEEW_ERR'][oii_mask][0]

    # H delta Flux and EW
    h_delta_mask     = spzline['LINENAME']=='H_delta'
    h_delta_flux     = spzline['LINEAREA'][h_delta_mask][0]
    h_delta_flux_err = spzline['LINEAREA_ERR'][h_delta_mask][0]
    h_delta_EW       = spzline['LINEEW'][h_delta_mask][0]
    h_delta_EW_err   = spzline['LINEEW_ERR'][h_delta_mask][0]

    # Other lines for AGN
    # OIII 5007, H beta, NII 6583, H alpha
    o_iii_mask   = spzline['LINENAME']=='[O_III] 5007'
    o_iii_flux   = spzline['LINEAREA'][o_iii_mask][0]
    h_beta_mask  = spzline['LINENAME']=='H_beta'
    h_beta_flux  = spzline['LINEAREA'][h_beta_mask][0]
    n_ii_mask    = spzline['LINENAME']=='[N_II] 6583'
    n_ii_flux    = spzline['LINEAREA'][n_ii_mask][0]
    h_alpha_mask = spzline['LINENAME']=='H_alpha'
    h_alpha_flux = spzline['LINEAREA'][h_alpha_mask][0]


    # Creating dictionary to store values
    spectrum_data_dict = {
        'objid': objid,
        'fileid': fileid,  
        'ra': ra,
        'dec': dec,                     
        'z': z,                                
        'z_err': z_err,                       
        'D4000n': D4000n, 
        'sigma_D4000n': sigma_D4000n,                    
        'oii_flux': oii_flux,                 
        'oii_flux_err': oii_flux_err,         
        'oii_EW': oii_EW,                     
        'oii_EW_err': oii_EW_err,             
        'h_delta_flux': h_delta_flux,         
        'h_delta_flux_err': h_delta_flux_err, 
        'h_delta_EW': h_delta_EW,             
        'h_delta_EW_err': h_delta_EW_err,
        'o_iii_flux': o_iii_flux,
        'h_beta_flux': h_beta_flux,
        'n_ii_flux': n_ii_flux,
        'h_alpha_flux': h_alpha_flux,    

    }

    # Return dictionary
    return spectrum_data_dict

def sort_galaxy(spectrum_data_dict):
    """
    Use spectrum values to determine the galaxy's spectral class.
    """
    # Assign variables
    h_delta_EW     = spectrum_data_dict['h_delta_EW']
    h_delta_EW_err = spectrum_data_dict['h_delta_EW_err']
    oii_EW         = spectrum_data_dict['oii_EW']
    oii_EW_err     = spectrum_data_dict['oii_EW_err']
    D4000n         = spectrum_data_dict['D4000n']
    sigma_D4000n   = spectrum_data_dict['sigma_D4000n']
    o_iii          = spectrum_data_dict['o_iii_flux']
    h_beta         = spectrum_data_dict['h_beta_flux']
    n_ii           = spectrum_data_dict['n_ii_flux']
    h_alpha        = spectrum_data_dict['h_alpha_flux']

    # Quality cuts
    if (h_delta_EW_err < 0) or (oii_EW_err < 0):
        return '?: Invalid EW value'
    elif  (D4000n / sigma_D4000n) < 2:
        return '?: D4000n quality cut'
    elif (h_delta_EW_err >= 0.8):
        return '?: H delta quality cut'
    elif (oii_EW / oii_EW_err) < 2:
        return '?: O II quality cut'
    
    # AGN before other classes
    # Avoiding division by zero/require positive values
    if (o_iii > 0) and (h_beta > 0) and (n_ii > 0) and (h_alpha > 0):

        x = np.log10(n_ii/h_alpha)
        y = np.log10(o_iii/h_beta)
        # Guard against vertical asymptote
        if not np.isclose(x - 0.05, 0.0):
            if y > (0.61 / (x - 0.05) + 1.3):
                return 'e(n)'
    
    # OII W_0 "Absent"
    if (oii_EW < -5):
        if h_delta_EW < 3:
            return 'k'
        elif (3 < h_delta_EW < 8):
            return 'k+a'
        elif h_delta_EW >= 8:
            return 'a+k'
        
    # OII EW present
    elif (oii_EW > -5):
        if (oii_EW < 40) and (h_delta_EW < 4):
            return 'e(c)'
        elif (oii_EW >= 40):
            return 'e(b)'
        elif h_delta_EW >=4:
            return 'e(a)'
        else:
            return 'e'

def determine_shape(objid, file_path="ZOO/full_morphology"):
    # Look up if objid is in full_morphology.csv
    file_path = Path(file_path)
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['objid']) == objid:
                valid_row = row
                break
        else:
            return None # objid not found
        
    shapes = {
        k: float(v)
        for k, v in valid_row.items()
        if k != 'objid'
    }

    # Sort shapes
    sorted_shapes = sorted(
        shapes.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Find S/N ratio
    (shape1, val1), (shape2, val2) = sorted_shapes[:2]
    # Reject shape2 automatically if val2 is 0
    if val2 == 0:
        return shape1
    
    ratio = val1 / val2
    if ratio >= 2:
        return shape1
    else:
        return "dontknow"

#-----------------------------------------------
# Transform search radius into theta and delta z
#-----------------------------------------------

def length_to_radius(z, mpc_radius, dis_type):
    """
    Convert a radius (Mpc) at a given redshift to an angular radius in arcminutes
    using either proper distance or comoving distance values.
    Use the redshift to return the distance to the object in Mpc as well.
    """
    # Convert z to Mpc
    if dis_type == 'proper':
        distance = cosmo.angular_diameter_distance(z)
    if dis_type == 'comoving':
        distance = cosmo.comoving_distance(z)
    # Calculate diameter theta
    theta           = np.arctan((mpc_radius * u.Mpc) / distance)
    # Convert to arcmin and radius
    radius_arcmin   = theta.to(u.arcmin).value
    return radius_arcmin

def length_to_delta_z(z, mpc_radius, dis_type):
    """
    Given a redshift and search window, find the delta z using either proper or comoving distance.
    """
    # Hubble parameter at redshift z
    if dis_type == 'proper':
        r = mpc_radius * u.Mpc
    if dis_type == 'comoving':
        r = mpc_radius * (1 + z) * u.Mpc
    Hz = cosmo.H(z) # km / s / Mpc

    # Convert H(z)/c to 1/distance units
    c = const.c.to(u.km/u.s)
    delta_z = ((Hz / c) * r).decompose()

    return delta_z.value

#-------------------------------------------------------
# Count number of neighbors and calculate number density
#-------------------------------------------------------

def count_sdss_neighbors(data_dict, mpc_radius, dis_type, ra_all, dec_all, z_all):
    ra0  = data_dict['ra']
    dec0 = data_dict['dec']
    z0   = data_dict['z']

    # Angular radius (arcmin)
    radius_arcmin = length_to_radius(z0, mpc_radius, dis_type)

    # Redshift window
    dz = length_to_delta_z(z0, mpc_radius, dis_type)
    z_min = z0 - dz
    z_max = z0 + dz

    # Redshift filter boolean mask
    z_mask = (z_all >= z_min) & (z_all <= z_max)

    # Angular separation
    dra = (ra_all - ra0) * np.cos(np.deg2rad(dec0))
    ddec = dec_all - dec0
    ang_sep_deg = np.sqrt(dra ** 2 + ddec ** 2)
    ang_sep_arcmin = ang_sep_deg * 60

    ang_mask = ang_sep_arcmin < radius_arcmin

    total_mask = z_mask & ang_mask

    count = np.sum(total_mask)

    # Subtract self if included
    return max(count - 1, 0)

def cone_slice_volume_calculator(z, mpc_radius, dis_type):
    """
    Calculate the volume of the search area using the equation of a frustrum.
    """
    if dis_type == 'proper':
        distance_center = cosmo.angular_diameter_distance(z)
    if dis_type == 'comoving':
        distance_center = cosmo.comoving_distance(z)

    theta = np.arctan((mpc_radius * u.Mpc) / distance_center) # keep in radians

    # Half-width along line of sight in redshift
    delta_z = length_to_delta_z(z, mpc_radius, dis_type)

    # Distance to near and far planes
    if dis_type == 'proper':
        near_distance = cosmo.angular_diameter_distance(z-delta_z).value
        far_distance = cosmo.angular_diameter_distance(z+delta_z).value
    if dis_type == 'comoving':
        near_distance = cosmo.comoving_distance(z-delta_z).value
        far_distance = cosmo.comoving_distance(z+delta_z).value

    # Calculate near and far radii
    r_near = np.tan(theta) * near_distance
    r_far  = np.tan(theta) * far_distance

    h = far_distance-near_distance

    return (1/3) * np.pi * h * (r_near ** 2 + (r_near * r_far) + r_far **2)

def calculate_density(n_neighbors, volume):
    """
    Use a galaxy's number of neighbors to calculate the number density for a given volume.
    """
    return n_neighbors / volume

#------------------------------------------------
# Find 5NN for both proper and comoving distances
#------------------------------------------------

def find_fifth_nearest_neighbor(ra_all, dec_all, z_all):
    """
    Use cKDTree to build a tree out of the sdss csv file of the distances between galaxies. Find the distance to the fifth nearest neighbor (5NN)
    using both the physical and the comoving distance
    """
    # Find 5NN using proper distance
    physical_distance_all = cosmo.angular_diameter_distance(z_all) # Mpc
    coords_from_phys = SkyCoord(ra=ra_all*u.deg, dec=dec_all*u.deg, distance=physical_distance_all)
    xyz_phys = np.vstack(coords_from_phys.cartesian.xyz).T
    tree_phys = cKDTree(xyz_phys)
    dis_phys, _ = tree_phys.query(xyz_phys, k=6)
    fifth_phys = dis_phys[:, 5]

    # Find 5NN using comoving distance
    comoving_distance_all = cosmo.comoving_distance(z_all) # Mpc
    coords_from_comv = SkyCoord(ra=ra_all*u.deg, dec=dec_all*u.deg, distance=comoving_distance_all)
    xyz_comv = np.vstack(coords_from_comv.cartesian.xyz).T
    tree_comv = cKDTree(xyz_comv)
    dis_comv, _ = tree_comv.query(xyz_comv, k=6)
    fifth_comv = dis_comv[:, 5]

    return fifth_phys, fifth_comv

#-------------------------------------------------------------------
# Collect all relevant values and load them into a result dictionary
#-------------------------------------------------------------------

def collect_values(files, csv_file_path):
    """
    Using the FITS files, store the objid, redshift, D4000n, sigma D4000n, Hdelta EW, Hdelta err, oii EW, oii EW err, 
    number density (for 2, 5, 10, 15, 21, and 42 Mpc search windows), galaxy class, and galaxy shape (if available)
    for every galaxy available. 
    """
    # Load full_sdss.csv for neighbor counting
    volume_df = pd.read_csv(csv_file_path, dtype={'objid': str})
    ra_all    = volume_df['ra'].values
    dec_all   = volume_df['dec'].values
    z_all     = volume_df['z'].values
    objid_all = volume_df['objid'].astype(str).values
    # Collect the index of every object for recall
    objid_to_index = {str(objid): i for i, objid in enumerate(objid_all)}    

    # Load every galaxy's 5NN
    fifth_phys_all, fifth_comv_all = find_fifth_nearest_neighbor(ra_all, dec_all, z_all)

    # Initiate type dictionaries
    class_dict = defaultdict(list)
    shape_dict = defaultdict(list)

    # Radii to calculate neighbors for
    mpc_radii = [2, 5, 10, 15, 21, 42] # cluster core to typical void radius

    # Collect values
    for file in files:
        spectrum_data_dict = collect_spectrum_data(file)
        objid = spectrum_data_dict['objid']

        # Only select data in the chosen redshift range
        z = spectrum_data_dict['z']
        if (z < 0.15) or (z > 0.3):
            continue
        
        # Skip missing objids
        if objid == None:
            print(f"Failed to find objid for {spectrum_data_dict['fileid']}.")
            continue
        if objid not in objid_to_index:
            print(f"Failed to find objid {objid} in csv file (from {spectrum_data_dict['fileid']})")
            continue
        
        idx = objid_to_index[objid]

        galaxy_class = sort_galaxy(spectrum_data_dict)
        galaxy_shape = determine_shape(objid, "ZOO/full_morphology.csv")

        # Grab proper and comoving distances from 5NN
        fifth_nn_proper = fifth_phys_all[idx]
        fifth_nn_comv = fifth_comv_all[idx]

        # Calcuate neighbor counts and density for each radius using proper and comoving distances
        proper_n_neighbors = []
        proper_densities   = []

        comoving_n_neighbors = []
        comoving_densities   = []
        for r in mpc_radii:
            #-------------------------------
            # Collect proper distance values
            #-------------------------------
            proper_nn_count = count_sdss_neighbors(spectrum_data_dict, r, 'proper', ra_all, dec_all, z_all)
            proper_n_neighbors.append(proper_nn_count)

            # Compute frustrum volume for this radius
            proper_volume = cone_slice_volume_calculator(spectrum_data_dict['z'], r, 'proper')
            proper_densities.append(calculate_density(proper_nn_count, proper_volume))

            #---------------------------------
            # Collect comoving distance values
            #---------------------------------
            comoving_nn_count = count_sdss_neighbors(spectrum_data_dict, r, 'comoving', ra_all, dec_all, z_all)
            comoving_n_neighbors.append(comoving_nn_count)

            # Compute frustrum volume for this radius
            comoving_volume = cone_slice_volume_calculator(spectrum_data_dict['z'], r, 'comoving')
            comoving_densities.append(calculate_density(comoving_nn_count, comoving_volume))

        # Store all data to sort by galaxy class
        class_dict[galaxy_class].append(
            {
                'objid': spectrum_data_dict['objid'],
                'z': spectrum_data_dict['z'],
                'ra': spectrum_data_dict['ra'],
                'dec': spectrum_data_dict['dec'],
                'D4000n': spectrum_data_dict['D4000n'], 
                'sigma_D4000n': spectrum_data_dict['sigma_D4000n'],
                'h_delta_EW': spectrum_data_dict['h_delta_EW'], 
                'h_delta_EW_err': spectrum_data_dict['h_delta_EW_err'],
                'oii_EW': spectrum_data_dict['oii_EW'],
                'oii_EW_err': spectrum_data_dict['oii_EW_err'],
                'proper_densities': proper_densities,
                'comoving_densities': comoving_densities,
                'fifth_nn_proper': fifth_nn_proper,
                'fifth_nn_comv': fifth_nn_comv,
                'galaxy_shape': galaxy_shape
            }
        )

        # Store all data to sort by galaxy shape
        shape_dict[galaxy_shape].append(
            {
                'objid': spectrum_data_dict['objid'],
                'ra': spectrum_data_dict['ra'],
                'dec': spectrum_data_dict['dec'],
                'z': spectrum_data_dict['z'],
                'ra': spectrum_data_dict['ra'],
                'dec': spectrum_data_dict['dec'],
                'D4000n': spectrum_data_dict['D4000n'], 
                'sigma_D4000n': spectrum_data_dict['sigma_D4000n'],
                'h_delta_EW': spectrum_data_dict['h_delta_EW'], 
                'h_delta_EW_err': spectrum_data_dict['h_delta_EW_err'],
                'oii_EW': spectrum_data_dict['oii_EW'],
                'oii_EW_err': spectrum_data_dict['oii_EW_err'],
                'proper_densities': proper_densities,
                'comoving_densities': comoving_densities,
                'fifth_nn_proper': fifth_nn_proper,
                'fifth_nn_comv': fifth_nn_comv,
                'galaxy_class': galaxy_class
            }
        )

    result = {
        'class_dict': class_dict,
        'shape_dict': shape_dict
    }

    return result

def construct_url_list(source_csv_file, base_url, start, end):
    df = pd.read_csv(source_csv_file, header=0)
    file_list = []
    for i in range(start, end):
        row = df.iloc[i]
        if 0.15 <= row['z'] <= 0.3:
        
            plate = row["plate"]
            mjd   = row["mjd"]
            fiber = row["fiberid"]
            url   = row["spec_fits_url"]
            filename = f'spec-{plate:04d}-{mjd}-{fiber:04d}.fits'

            # Handle non-existing FITS files:
            if not isinstance(url, str) or not url.strip():
                print(f"No valid url for {filename}, skipping.")
                continue
            
            file_url = f"{base_url}/{filename}"
            file_list.append(file_url)
            
    return file_list

def save_result(result, filename):
    """
    Quickly save result to disk after running collect_values to avoid running multiple times
    """
    tmp = str(filename) + ".tmp"

    with open(tmp, 'wb') as f:
        pickle.dump(result, f)

    os.replace(tmp, filename)

def load_result(filename):
    """
    Load pickle collect_values result after saving with save_result
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_job_pickle(file_line, sdss_csv_path, out_folder, job_index):
    """
    Process a single line from job_list.txt, save as a pickle.
    """
    # Convert line to a list of file paths
    file_list = file_line.strip().split(',')
    file_list = [os.path.basename(f) for f in file_list]

    result = collect_values(file_list, sdss_csv_path)
    out_path = Path(out_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / f"pickle_{job_index}.pkl"
    print("saving result...")
    save_result(result, filename)
    print('saved result.')

def merge_pickles(source_folder, filename, out_folder):
    out_path = Path(out_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / filename
    print(f"Source folder: {source_folder}")
    print(f'Final file name: {path}')
    print(f'Out folder: {out_folder}')

    merged_class_dict = defaultdict(list)
    merged_shape_dict = defaultdict(list)

    files = list(Path(source_folder).glob("pickle_*"))
    print(f"Files found: {len(files)}")

    bad = 0
    for file in files:
        # Skip empty files
        if file.stat().st_size == 0:
            print(f'Skipping empty file: {file}')
            bad += 1
            continue
        try:     
            result = load_result(file)
        except EOFError:
            print(f'Skipping truncated pickle: {file}')
            bad += 1
            continue
        except Exception as e:
            print(f"Skipping unreadable pickle: {file}")
            bad += 1
            continue
        
        # Merge class_dict
        for galaxy_class, entries in result['class_dict'].items():
            merged_class_dict[galaxy_class].extend(entries)

        # Merge shape_dict
        for galaxy_shape, entries in result['shape_dict'].items():
            merged_shape_dict[galaxy_shape].extend(entries)
    print(f'Bad files skipped: {bad}')

    result = {
        'class_dict': merged_class_dict,
        'shape_dict': merged_shape_dict
    }

    try:
        save_result(result, path)
    except Exception as e:
        print(f'Error saving final file:\n{e}')

    return result
    
