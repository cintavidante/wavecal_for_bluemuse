"""
---------------------
Wavecal_for_BlueMUSE
---------------------

Cinta Vidante, 2025

This module contains functions to generate line list and table
in FITS tables for generate_simulation_image.

Part of my work on BlueMUSE with Peter Weilbacher, AIP.

"""

# --------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.modeling.models import Gaussian1D, Linear1D
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter

from astropy import units as u
from astropy.io import fits

from astropy.table import Table
from astropy.time import Time

from datetime import datetime, timezone

# --------------------------------------------------------------------

def get_gaussian(y, x):

    """
    Gaussian function fitting with astropy.

    Parameters:
    ----------
    y   : array-like
        y values to fit
    x   : array-like
        corresponding x values

    Returns:
    ----------
    fit: Gaussian1D
        fitted Gaussian1D object

    """

    lmfitter = LevMarLSQFitter()
    guess = Gaussian1D(amplitude=y.max(), mean=np.mean(x), stddev=5)
    fit = lmfitter(model=guess, x=x, y=y)

    return fit

# --------------------------------------------------------------------

def get_gaussian_peaks(lines, spec):

    """
    Get gaussian peaks. This uses the information from scipy's peaks.

    Parameters:
    ----------
    lines   : pandas DataFrame
            master line list of each arc lines
    spec   : pandas DataFrame
            full spectra

    Returns:
    ----------
    lines   : pandas DataFrame
            updated pandas lines with gaussian peaks and values
    
    """

    # Width of gaussian peaks
    width = int(12)
    width_scale = int(width/2)

    # Refine peaks with gaussian fitting
    new_peaks = []
    amplitudes = []
    fluxes = []

    # For every scipy pix
    for p in lines['pix']:

        # Area to find gaussian peak
        x = np.asarray(spec['pix'][p-width_scale:p+width_scale])
        y = np.asarray(spec['flux'][p-width_scale:p+width_scale])

        # Find gaussian peak
        model = get_gaussian(y, x)

        # Append the parameters of the gaussian fit: new peak and amplitude
        new_peaks.append(np.float32(np.round(model.mean.value, 6)))
        amplitudes.append(np.float32(np.round(model.amplitude.value, 6)))

        # Find the sigma
        sigma = np.float32(model.stddev.value)  
        amplitude = np.float32(model.amplitude.value)

        # Calculate flux, the area under the Gaussian
        flux = np.float32(amplitude * sigma * np.sqrt(2 * np.pi)) 
        flux = np.round(flux, 4)

        # Append the flux for this peak
        fluxes.append(flux)  
    
    # Add new information to the pandas dataframe
    lines['gaus_peaks'] = new_peaks
    lines['amplitudes'] = amplitudes
    lines['flux'] = fluxes

    # Add the normalized amplitude
    lines['norm_amp'] = lines['amplitudes'] / lines['amplitudes'].max()

    return lines

# --------------------------------------------------------------------

def get_line_for_img(lines, name, add_amp=False):

    """
    Get crucial only list for the generate_simulation_image.
    It just needs the lambda and the normalized amplitude

    Parameters:
    ----------
    lines   : pandas DataFrame
            master line list of each arc lines
    name    : str
            name of the arc
    add_amp : bool
            if add amplitude if amplitude is not known
            (FP from example)

    Returns:
    ----------
    line_list   : pandas DataFrame
                subset of lines that only has lambda and norm_amp
    
    """

    if add_amp:
        lines['norm_amp'] = 1

    line_list = lines[['lambda', 'norm_amp']]
    line_list.to_csv('data/lines_for_img/{}.csv'.format(name), index=False)

# --------------------------------------------------------------------

def get_line_for_table(lines, name, source='4MOST', add_flux=False):

    """
    From CSV/Pandas DataFrame to FITS table

    Parameters:
    ----------
    lines   : pandas DataFrame
            master line list of each arc lines
    name    : str
            name of the arc
    source : str
            source of spectra
    add_flux : bool
            if add flux (the area beneath the gaussian)

    Returns:
    ----------
    
    
    """

    # Make astropy table
    # ----------------------------

    line_fits = Table.from_pandas(lines)

    # Add the gaussian flux if needed
    if add_flux:
        flux = []
        for i in range(len(lines)):
            flux.append(np.float32(1))
        line_fits['flux'] = flux
    
    # Initialize quality
    quality = []

    # Make quality
    for i in range(len(line_fits)):
        quality.append(3)

    # Add quality to line fits
    line_fits['quality'] = quality

    # Make FITS
    # ----------------------------

    # Determine time
    ut = Time(datetime.now(tz=timezone.utc), scale='utc')

    # Main HDU
    hdu_table = fits.BinTableHDU(data=line_fits, name=name)

    # Header
    hdr = fits.Header()
    hdr['COMMENT'] = 'Table of line list'
    hdr['DATE'] = (ut.fits, 'Current time in UTC')
    hdr['METHOD'] = "Line identification"
    hdr['ARCLAMP'] = "{}".format(name)
    hdr['OBSERV'] = ('4MOST spectra', 'source of spectra')

    # Put primary header to empty header
    empty_hdr = fits.PrimaryHDU(header=hdr)

    # Make HDU list
    hdul = fits.HDUList([empty_hdr, hdu_table])

    # Generate astropy table
    hdul.writeto('output/files_from_py/line_tab_{}_{}.fits'.format(name, source), overwrite=True)

# --------------------------------------------------------------------

def get_master_peaks_for_FP(line_dict):

    """
    From CSV/Pandas DataFrame to FITS table

    Parameters:
    ----------
    line_dict  : dict
            dictionary of all lines

    Returns:
    ----------
    
    
    """

    # Get all of the lines in the dictonary
    filter = {'blue': [],
            'green': []}

    for fil, app in filter.items():

        df = []

        for key, l in line_dict.items():

            lines_df = pd.read_csv('line_lists/gaussian_peaks/peaks_{}_{}.csv'.format(key, fil))

            # Transfer to new pandas
            lines_cat = lines_df[lines_df['note'] == '4MOST & NIST']
            lines_cat = lines_cat[['gaussian peaks', 'lambda', 'element']].copy()

            df.append(lines_cat)
        
        df_full = pd.concat(df)
        df_full = df_full.reset_index(drop=True).sort_values(by='gaussian peaks', ignore_index=True).drop_duplicates(subset=['gaussian peaks'])
        df_full.to_csv('line_lists/master_peaks_{}.csv'.format(fil), index=False)

        app.append(df_full)

# --------------------------------------------------------------------

if __name__ == "__main__":

    # Main dictionary for all files
    lines = {'Cd': [],
            'Cs': [],
            'He': [],
            'Hg': [],
            # 'HgCd': [],
            # 'HgCd-LLG300': [],
            'Zn': [],
            'HgAr': [],
            'Xe': [],
            'FP': []
            }
    
    # # Create master list for FP. Uncomment if FP is not included
    # get_master_peaks_for_FP(lines)
    
    # Loop for every arc
    for key, l in lines.items():

        if key != 'FP':

            # Get line list
            line_list = pd.read_csv('line_lists/compiled_master/master_list_{}.csv'.format(key))

            # Drop rows with NaN in lambda
            line_list = line_list.dropna(subset=['lambda'])

            # Make astropy table
            get_line_for_table(line_list, key)

            # Only use line_list that has 4MOST
            lines = line_list[line_list['note'] == '4MOST & NIST'].copy().reset_index(drop=True)

            # Subset only lambda and intensity
            lines = lines[['lambda', 'intens_4MOST']]

            # Get normalized amplitude from that
            lines['norm_amp'] = lines['intens_4MOST'] / lines['intens_4MOST'].max()

            # Make file to generate arc image in generate_simulation_image
            get_line_for_img(lines, key)
        
        else:

            # Get line list
            line_list = pd.read_csv('line_lists/master_peaks_fp.csv')

            # Make file to generate arc image
            get_line_for_img(line_list, key, add_amp=True)


    
    













