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
    Gaussian function.
    """

    lmfitter = LevMarLSQFitter()
    guess = Gaussian1D(amplitude=y.max(), mean=np.mean(x), stddev=5)
    fit = lmfitter(model=guess, x=x, y=y)

    return fit

# --------------------------------------------------------------------

def get_gaussian_peaks(lines, spec):

    """
    Get gaussian peaks. This uses the information from scipy's peaks.
    
    """

    # Width of gaussian peaks
    width = int(12)
    width_scale = int(width/2)

    # Refine peaks with gaussian fitting
    new_peaks = []
    amplitudes = []
    fluxes = []

    for p in lines['pix']:

        x = np.asarray(spec['pix'][p-width_scale:p+width_scale])
        y = np.asarray(spec['flux'][p-width_scale:p+width_scale])

        model = get_gaussian(y, x)
        new_peaks.append(np.float32(np.round(model.mean.value, 6)))
        amplitudes.append(np.float32(np.round(model.amplitude.value, 6)))

        sigma = np.float32(model.stddev.value)  # Standard deviation (width)
        amplitude = np.float32(model.amplitude.value)

        flux = np.float32(amplitude * sigma * np.sqrt(2 * np.pi))  # Area under the Gaussian
        flux = np.round(flux, 4)

        fluxes.append(flux)  # Append the flux for this peak
    
    lines['gaus_peaks'] = new_peaks
    lines['amplitudes'] = amplitudes
    lines['flux'] = fluxes

    lines['norm_amp'] = lines['amplitudes'] / lines['amplitudes'].max()

    return lines

# --------------------------------------------------------------------

def get_line_for_img(lines, name, add_amp=False):

    if add_amp:
        lines['norm_amp'] = 1

    line_list = lines[['lambda', 'norm_amp']]
    line_list.to_csv('data/lines_for_img/{}.csv'.format(name), index=False)

# --------------------------------------------------------------------

def get_line_for_table(lines, name, source='4MOST', add_flux=False):

    line_fits = Table.from_pandas(lines)

    if add_flux:
        flux = []
        for i in range(len(lines)):
            flux.append(np.float32(1))
        line_fits['flux'] = flux
    
    # line_fits = line_fits[['lambda', 'flux']]
        
    quality = []
    # arc_lamp = []

    for i in range(len(line_fits)):

        quality.append(3)
    #     arc_lamp.append(name)

    # line_fits['line'] = arc_lamp
    line_fits['quality'] = quality

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

    empty_hdr = fits.PrimaryHDU(header=hdr)

    hdul = fits.HDUList([empty_hdr, hdu_table])

    hdul.writeto('output/files_from_py/{}_line_tab_{}.fits'.format(name, source), overwrite=True)
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
            'Xe': []
            #  'FP': []
            }
    
    for key, l in lines.items():

        line_list = pd.read_csv('line_lists/compiled/v3_master_list_{}.csv'.format(key))

        line_list = line_list.dropna(subset=['lambda'])

        get_line_for_table(line_list, key)

        lines = line_list[line_list['note'] == '4MOST & NIST'].copy().reset_index(drop=True)
        lines = lines[['lambda', 'intens_4MOST']]
        lines['norm_amp'] = lines['intens_4MOST'] / lines['intens_4MOST'].max()

        get_line_for_img(lines, key)
    













