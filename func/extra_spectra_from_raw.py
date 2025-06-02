"""
---------------------
Wavecal_for_BlueMUSE
---------------------

Cinta Vidante, 2025

This module contains functions and classes used to extract spectra from 4MOST images. 

There are 3 methods of extracting the spectra. Each method extract 5 peaks/dots/spectra
of the arc lamps from the images. The difference between each method is described as follows:

- esuway
    - always takes the last peak, regardless the strength of the peak
- tsuway
    - takes the maximum/strongest peak 
- average
    - average all 5 peaks

I found that the esuway creates more noise while the tsuway gives more stable spectra 
similar to the average method.


Part of my work on BlueMUSE with Peter Weilbacher, AIP.

"""

# ------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData

from astropy.modeling.models import Gaussian1D, Linear1D
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter

from astropy.table import Table, join

from scipy.signal import find_peaks

# ------------------------------------------------------------------------

# Function to get Gaussian
def get_gaussian(trace, xaxis):
    
    lmfitter = LevMarLSQFitter()
    guess = Gaussian1D(amplitude=trace.max(), mean=0, stddev=5)
    fit = lmfitter(model=guess, x=xaxis, y=trace)
    model = fit(xaxis)

    return model

#----------------------------------
# Function to get spectra
def get_spec_esuway(laser, lamp, ybox):
    
    # Get image array
    image = laser[ybox[0]:ybox[1]]
    image = np.asarray(image)

    # Get background
    background = np.median(image)

    # Get index of maximum values for each column
    # yvals = np.argmax(image, axis=0)
    xvals = np.arange(image.shape[1])

    # Gaussian off-set
    pixcut = 10

    # Get lamp
    arc = lamp[ybox[0]:ybox[1]]
    arc = np.asarray(arc)
    backlamp = np.median(arc)

    # Make a list to append
    arc_spec = []

    # Get gaussian fit for each column
    for x in xvals:

        # Get box to find the gaussian peaks
        box = image[:, x]

        # # Minimum height for the peaks
        # if (key == 'blue'):
        #     h = 25
        # else:
        #     h = 50

        # Find gaussian peaks
        peaks, _ = find_peaks(box, prominence=6, height=25, distance=5)

        # There should only be 5 gaussian peaks detected because there are only 5 spectrum from my box
        if len(peaks) != 5:
            arclamp = 0
        else:
            # Use the last gaussian (upmost right in here, but top in real image)
            # yval is the value where the peak is located
            yval = peaks[1]

            # Get cutout from yval
            cutout = image[int(yval)-pixcut:int(yval)+pixcut, x]

            # Get gaussian fit
            trace_offset = (cutout - background)
            xaxis_offset = np.arange(-pixcut, pixcut)

            model_offset = get_gaussian(trace_offset, xaxis_offset)

            arclamp = np.average(arc[int(yval)-pixcut:int(yval)+pixcut, x] - backlamp,
                            weights=model_offset)
        
        arc_spec.append(arclamp)

    # Convert list to array
    arc_spec = np.asarray(arc_spec)
    
    arc_pix = np.arange(arc_spec.shape[0])

    spec = Table([arc_pix, arc_spec],
                 names=('pix', 'flux'))
    
    return spec

# ----------------------------------------------------------------

def get_spec_tsuway(laser, lamp, ybox):
    
    # Get image array
    image = laser[ybox[0]:ybox[1]]
    image = np.asarray(image)

    # Get background
    background = np.median(image)

    # Get index of maximum values for each column
    yvals = np.argmax(image, axis=0)
    xvals = np.arange(image.shape[1])

    # Gaussian off-set
    pixcut = 10

    # Get lamp
    arc = lamp[ybox[0]:ybox[1]]
    arc = np.asarray(arc)
    backlamp = np.median(arc)

    # Make a list to append
    arc_spec = []
    
    # Get gaussian fit for each column
    for x, yval in zip(xvals,yvals):
        
        # yval is the maximum value (peak) of the gaussian
        cutout = image[int(yval)-pixcut:int(yval)+pixcut, x]

        # Get gaussian fit
        trace_offset = (cutout - background)
        xaxis_offset = np.arange(-pixcut, pixcut)

        model_offset = get_gaussian(trace_offset, xaxis_offset)
        
        # If no gaussian can be fitted due to noise, the flux value is assigned to be 0
        if np.all(model_offset):
            arclamp = np.average(arc[int(yval)-pixcut:int(yval)+pixcut, x] - backlamp,
                        weights=model_offset)
        else:
            arclamp = 0
        
        arc_spec.append(arclamp)

    # Convert list to array
    arc_spec = np.asarray(arc_spec)
    
    arc_pix = np.arange(arc_spec.shape[0])

    spec = Table([arc_pix, arc_spec],
                 names=('pix', 'flux'))
    
    return spec

# ----------------------------------------------------------------

def get_spec_average(laser, lamp, ybox):
    
    # Get image array
    image = laser[ybox[0]:ybox[1]]
    image = np.asarray(image)

    # Get background
    background = np.median(image)

    # Get index of maximum values for each column
    yvals = np.argmax(image, axis=0)
    xvals = np.arange(image.shape[1])

    # Gaussian off-set
    pixcut = 15
    cutout_offset = np.array([image[int(yval)-pixcut:int(yval)+pixcut, ii]
                    for yval, ii in zip(yvals, xvals)])
    
    # Get gaussian fit
    trace_offset = (cutout_offset - background).mean(axis=0)
    xaxis_offset = np.arange(-pixcut, pixcut)

    model_offset = get_gaussian(trace_offset, xaxis_offset)

    # Get lamp
    arc = lamp[ybox[0]:ybox[1]]
    arc = np.asarray(arc)
    background = np.median(arc)

    # Get lamp spectra
    arc_spec = np.array([np.average(arc[int(yval)-pixcut:int(yval)+pixcut, ii] - background,
                    weights=model_offset) for yval, ii in zip(yvals, xvals)])
    arc_pix = np.arange(arc_spec.shape[0])

    spec = Table([arc_pix, arc_spec],
                 names=('pix', 'flux'))
    
    return spec

# ----------------------------------------------------------------

if __name__ == "__main__":

    # Dictionary for colors
    filter = {'green': [2],
            'blue': [3]}

    # Main dictionary for all files.
    # Put the spectrum of each arc lamp in te dictionary as shown here
    files = { 
            # 'Cd': ['HP-Cd_combined_novar.fits'],
            # 'Cs': ['HP-Cs_combined_novar.fits'],
            # 'He': ['HP-He_combined_novar.fits'],
            # 'Hg': ['HP-Hg_combined_novar.fits'],
            # 'HgCd': ['HP-HgCd_combined_novar.fits'],
            # 'HgCd-LLG300': ['HP-HgCd_LLG300_combined_novar.fits'],
            # 'Zn': ['HP-Zn_combined_novar.fits'],
            # 'HgAr': ['PR-HgAr_combined_novar.fits'],
            # 'Xe': ['PR-Xe_combined_novar.fits']
            #  'FP': ['LDLS-FPE_combined_novar.fits']
            'LDLS': ['LDLS_combined_novar.fits']
            }

    # File designation for continuum laser / LDLS
    LDLS_file = 'spectra/LDLS_combined_novar.fits'

    # Define the y-axis to get spectra
    ybox = [100,500]

    # ----------------------------------------------------------------

    # Open LDLS 
    LDLS = fits.open(LDLS_file)

    for color, c in filter.items():

        # Get LDLS data for each color
        ldls = LDLS[c[0]].data 

        # Rotate 90 degree clockwise
        ldls = np.rot90(ldls)

        print('Start for filter {}'.format(color))
        print('-------')

        for key, l in files.items():

            print('Start for arc lamp: {}'.format(key))

            # Open fits and then rotate
            arc_lamp = fits.open('spectra/{}'.format(l[0]))
            arc = arc_lamp[c[0]].data
            arc = np.rot90(arc)

            # Get spectra from TSU method
            spec = get_spec_tsuway(ldls, arc, ybox)

            # Export files
            spec.to_pandas().to_csv('spectra/spectra_{}_{}.csv'.format(key, color), index=False)

            print('End for arc lamp: {}'.format(key))
            print('---')

        print('End for filter {}'.format(color))
        print('-------')









