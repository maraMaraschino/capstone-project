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
from itertools import islice

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

def download_fits_chunk(df, start, end, outdir="FITS"):
    """
    Uses final SDSS CSV file to fill a FITS folder with the 
    downloaded FITS files of all available galaxies.
    """
    outdir = Path('FITS')
    outdir.mkdir(exist_ok=True)

    for i in range(start, end):
        row = df.iloc[i]
        
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

def collect_spectrum_data(file):
    """
    Compute D4000n, collect [OII] 3727 & Hdelta flux/EW from a single SDSS FITS spectrum file.

    Negative Err values = invalid fit.

    Flux is negative for absorption spectra, positive for emission.
    """
    # Open file
    hdul = fits.open(file)
    
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

def sort_galaxy(data_dict):
    """
    Use spectrum values to determine the galaxy's spectral class.
    """
    # Assign variables
    h_delta_EW     = data_dict['h_delta_EW']
    h_delta_EW_err = data_dict['h_delta_EW_err']
    oii_EW         = data_dict['oii_EW']
    oii_EW_err     = data_dict['oii_EW_err']
    D4000n         = data_dict['D4000n']
    sigma_D4000n   = data_dict['sigma_D4000n']
    o_iii          = data_dict['o_iii_flux']
    h_beta         = data_dict['h_beta_flux']
    n_ii           = data_dict['n_ii_flux']
    h_alpha        = data_dict['h_alpha_flux']

    # Quality cuts
    if (h_delta_EW_err < 0) or (oii_EW_err < 0):
        return '?: Invalid EW value'
    elif  (D4000n / sigma_D4000n) < 2:
        return '?: D4000n quality cut'
    elif (h_delta_EW_err >= 0.8):
        return '?: H delta quality cut'
    
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
        
def determine_shape(objid, file_path):
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
    
def physical_to_angular_radius(z, mpc_radius):
    """
    Convert a physical radius (Mpc) at a given redshift to an angular radius in arcminutes.
    Use the redshift to return the distance to the object in Mpc as well.
    """
    # Convert z to Mpc
    distance        = cosmo.angular_diameter_distance(z)
    # Calculate diameter theta
    theta           = np.arctan((mpc_radius * u.Mpc) / distance)
    # Convert to arcmin and radius
    radius_arcmin   = theta.to(u.arcmin).value/2
    return radius_arcmin

def physical_to_delta_z(z, mpc_radius):
    """
    Convert a physical line-of-sight distance (Mpc) into a redshift half-width.
    """
    # Hubble parameter at redshift z
    Hz = cosmo.H(z) # km / s / Mpc

    # Convert H(z)/c to 1/distance units
    c = const.c.to(u.km/u.s)
    delta_z = ((Hz / c) * mpc_radius * u.Mpc).decompose()

    return delta_z.value

def cone_slice_volume_calculator(z, mpc_radius):
    """
    Calculate the volume of the search area using the equation of a frustrum.
    """
    distance_center = cosmo.angular_diameter_distance(z)
    theta = np.arctan(mpc_radius / distance_center.value) # keep in radians

    # Half-width along line of sight in redshift
    delta_z = physical_to_delta_z(z, mpc_radius)

    # Distance to near and far planes
    near_distance = cosmo.comoving_distance(z-delta_z).value
    far_distance = cosmo.comoving_distance(z+delta_z).value

    # Calculate near and far radii
    r_near = np.tan(theta) * near_distance
    r_far  = np.tan(theta) * far_distance

    h = far_distance-near_distance

    return (1/3) * np.pi * h * (r_near ** 2 + (r_near * r_far) + r_far **2)

def calculate_density(n_neighbors, volume):
    return n_neighbors / volume

def build_sdss_neighbor_count_query(data_dict, mpc_radius):
    """
    Builds an SDSS SQL query that counts galaxies within a physical radius around a target galaxy.

    Returns SQL query string for SDSS CasJobs
    """
    ra = data_dict['ra']
    dec = data_dict['dec']
    z = data_dict['z']
    dz = physical_to_delta_z(data_dict['z'], mpc_radius)
    z_min = z - dz
    z_max = z + dz

    radius_arcmin = physical_to_angular_radius(z, mpc_radius)

    sql_query = f"""
SELECT COUNT(*) AS neighbor_count
FROM SpecObj as g
WHERE dbo.fDistanceEq(g.ra, g.dec, {ra}, {dec}) < {radius_arcmin}
    AND g.z BETWEEN {z_min} AND {z_max}
    """
    return sql_query

def count_sdss_neighbors(data_dict, mpc_radius):
    """
    Query SDSS and return the number of neighboring galaxies within a physical radius -1 to exclude target galaxy.
    """
    sql = build_sdss_neighbor_count_query(data_dict, mpc_radius)
    result = SDSS.query_sql(sql)
    if result[0][0] is None or 0:
        return 0
    return result[0][0]-1

def collect_values(files):
    """
    Store the redshift
    """
    # Store values by class
    class_dict = defaultdict(list)
    shape_dict = defaultdict(list)

    # Radii to calculate neighbors for
    mpc_radii = [2, 5, 10, 15]

    # Collect values
    for file in files:
        spectrum_data_dict = collect_spectrum_data(file)
        objid = spectrum_data_dict['objid']
        if objid == None:
            print(f'Failed to find objid for {spectrum_data_dict['fileid']}.')
            continue
        galaxy_class = sort_galaxy(spectrum_data_dict)
        galaxy_shape = determine_shape(objid, "ZOO/full_morphology.csv")

        # Calcuate neighbor counts and density for each radius
        n_neighbors = []
        densities = []
        for r in mpc_radii:
            count = count_sdss_neighbors(spectrum_data_dict, r)
            n_neighbors.append(count)

            # Compute frustrum volume for this radius
            volume = cone_slice_volume_calculator(spectrum_data_dict['z'], r)
            densities.append(calculate_density(count, volume))

        # Store all data to sort by galaxy class
        class_dict[galaxy_class].append(
            {
                'objid': spectrum_data_dict['objid'],
                'z': spectrum_data_dict['z'], 
                'D4000n': spectrum_data_dict['D4000n'], 
                'sigma_D4000n': spectrum_data_dict['sigma_D4000n'],
                'h_delta_EW': spectrum_data_dict['h_delta_EW'], 
                'h_delta_EW_err': spectrum_data_dict['h_delta_EW_err'],
                'oii_EW': spectrum_data_dict['oii_EW'],
                'oii_EW_err': spectrum_data_dict['oii_EW_err'],
                'n_neighbors': n_neighbors,
                'densities': densities,
                'galaxy_shape': galaxy_shape
            }
        )

        # Store all data to sort by galaxy shape
        shape_dict[galaxy_shape].append(
            {
                'objid': spectrum_data_dict['objid'],
                'z': spectrum_data_dict['z'], 
                'D4000n': spectrum_data_dict['D4000n'], 
                'sigma_D4000n': spectrum_data_dict['sigma_D4000n'],
                'h_delta_EW': spectrum_data_dict['h_delta_EW'], 
                'h_delta_EW_err': spectrum_data_dict['h_delta_EW_err'],
                'oii_EW': spectrum_data_dict['oii_EW'],
                'oii_EW_err': spectrum_data_dict['oii_EW_err'],
                'n_neighbors': n_neighbors,
                'densities': densities,
                'galaxy_class': galaxy_class
            }
        )

    result = {
        'class_dict': class_dict,
        'shape_dict': shape_dict
    }

    return result

def save_result(result, filename='result.pkl'):
    """
    Quickly save result to disk after running collect_values to avoid running multiple times
    """
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

def load_result(filename='result.pkl'):
    """
    Load pickle collect_values result after saving with save_result
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def report_env_stats_by_type(dict_key, result, radii=[2, 5, 10, 15]):
    """
    Return a dictionary with the mean, min, and max for the number of neighbors and number density of each search radius for each galaxy class, 
    as well as the percentage of galaxies that were rejected by quality cuts.
    """
    class_or_shape_dict = result[dict_key]
    total_counted = []
    total_rejected = []
    values_dict = {}
    for galaxy_type, data in class_or_shape_dict.items():
        num_counted_by_class = len(class_or_shape_dict[galaxy_type])
        if '?' in galaxy_type or 'dontknow' in galaxy_type:
            total_rejected.append(num_counted_by_class)
        #    continue
        total_counted.append(num_counted_by_class)
        values_dict[galaxy_type] = {
        'num_counted': num_counted_by_class,
        'by_radius': []
    }

        for i, radius in enumerate(radii):
            nnb_vals = []
            dens_vals = []
            for galaxy in data:
                nnb = galaxy['n_neighbors']
                dens = galaxy['densities']
                nnb_vals.append(nnb[i])
                dens_vals.append(dens[i])

            nnb_vals = np.array(nnb_vals)
            dens_vals = np.array(dens_vals)

            nnb_mean  = nnb_vals.mean()
            nnb_min   = nnb_vals.min()
            nnb_max   = nnb_vals.max()
            nnb_std   = nnb_vals.std()
            dens_mean = dens_vals.mean()
            dens_min  = dens_vals.min()
            dens_max  = dens_vals.max()
            dens_std  = dens_vals.std()
            
            values_dict[galaxy_type]['by_radius'].append(
                {
                    'radius': radius,
                    'values': {
                        'nnb_mean': nnb_mean,
                        'nnb_min': nnb_min,
                        'nnb_max': nnb_max,
                        'nnb_std': nnb_std,
                        'dens_mean': dens_mean,
                        'dens_min': dens_min,
                        'dens_max': dens_max,
                        'dens_std': dens_std
                    }
                }
            )
    values_dict['_summary'] = {
        'total_counted': sum(total_counted),
        'total_rejected':sum(total_rejected)
    }
      
    total_counted = values_dict['_summary']['total_counted']
    total_rejected = values_dict['_summary']['total_rejected']
    print(f"{total_counted} galaxies counted, {(total_rejected/(total_rejected+total_counted))*100:.2f}% or {total_rejected} rejected.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for galaxy_type, info in values_dict.items():
        if galaxy_type == '_summary':
            continue

        radii = []
        nnb_means = []
        dens_means = []
        nnb_stds = []
        dens_stds = []

        for entry in info['by_radius']:
            radii.append(entry['radius'])
            nnb_means.append(entry['values']['nnb_mean'])
            nnb_stds.append(entry['values']['nnb_std'])
            dens_means.append(entry['values']['dens_mean'])
            dens_stds.append(entry['values']['dens_std'])

        #ax1.plot(radii, nnb_means, marker='o', label=galaxy_class)
        ax1.errorbar(
            radii,
            nnb_means,
            yerr=nnb_stds,
            marker='o',
            capsize=3,
            label=galaxy_type
        )
        ax2.errorbar(
            radii, 
            dens_means, 
            yerr=dens_stds,
            marker='x', 
            label=galaxy_type
        )

    ax1.set_xlabel("Search radius (Mpc)")
    ax1.set_ylabel("Mean number of neighbors")
    #ax1.set_yscale('log')
    ax1.legend()

    ax2.set_xlabel("Search radius (Mpc)")
    ax2.set_ylabel(r"Mean number density (Galaxies/$Mpc^3$)")
    #ax2.set_yscale('log')
    fig.suptitle(f'Environment distribution by galaxy {dict_key[:-5]}.')

    plt.show()

    for galaxy_type, info in values_dict.items():
        if galaxy_type == '_summary':
            continue

        print(f"{galaxy_type} type galaxies ({info['num_counted']} total)")
        for by_radius in info['by_radius']:
            print(f"  r = {by_radius['radius']}")
            for key in by_radius['values']:
                print(f"    {key}: {by_radius['values'][key]}")
        print()

def plot_age_indicators_and_num_density_by_type(dict_key, result, radii=[2, 5, 10, 15]):
    """
    For each galaxy class, create two plots with an x-axis of the number density and a y-axis of either D4000n or Hdelta.
    Each search volume is represented as a different color
    """
    class_or_shape_dict = result[dict_key]
    for galaxy_type, data in class_or_shape_dict.items():
        if '?' in galaxy_type:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
        for i, radius in enumerate(radii):
            # Flatten spectrum values into lists
            D4000n_vals  = []
            h_delta_vals = []
            oii_vals     = []
            dens_vals    = []
            for galaxy in data:
                dens_vals.append(galaxy['densities'][i])
                D4000n_vals.append(galaxy['D4000n'])
                h_delta_vals.append(galaxy['h_delta_EW'])
                oii_vals.append(galaxy['oii_EW'])
            
            dens_vals = np.array(dens_vals)
            D4000n_vals = np.array(D4000n_vals)

            # Create plots
            ax1.scatter(D4000n_vals, dens_vals,label=f'r = {radius} Mpc', s=3)#, marker='x')
            ax2.scatter(h_delta_vals, dens_vals, label=f'r = {radius} Mpc', s=3)#, marker='x')
        ax1.set_ylabel(r'Number Density ($N/Mpc^3$)')
        ax1.set_xlabel(r'$D4000_n$')
        ax1.legend()

        ax2.set_ylabel(r'Number Density ($N/Mpc^3$)')
        ax2.set_xlabel(r'$H\delta$')
        ax2.legend()
        fig.suptitle(f'Num density to age indicators across {galaxy_type} {dict_key[:-5]} galaxies')
        fig.tight_layout() 
        plt.show()