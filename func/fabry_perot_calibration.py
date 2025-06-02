"""
---------------------
Wavecal_for_BlueMUSE
---------------------

Cinta Vidante, 2025

This module contains functions and classes used to calibrate 
Fabry-Perot (FP) spectrum from 4MOST. The main class of this module
is called Wavefp. The structure of Wavefp is written in such 
a way that the entire wavelength calibration of the FP can be conducted in 
one function automatically. 

This module is based on papers of FP by Hobston et al. (2021) 
and Bauer et al. (2015).

Part of my work on BlueMUSE with Peter Weilbacher, AIP.

"""


# ----------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.optimize import fsolve

from astropy.modeling.models import Gaussian1D, Linear1D
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter

# ----------------------------------------------------------------------------------

def get_norm(fp, ldls):

    """
    Function to return rough normalized value. 
    Norm = Fabry-Perot/LDLS

    Parameters:
    ----------
    fp  : pandas DataFrame
        spectrum of FP, with pix and flux
    ldls: pandas DataFrame
        spectrum of LDLS, with pix and flux

    Returns:
    ----------
    fp_array/ldls_array: array-like
        normalized FP spectrum against LDLS

    """
    
    # Take flux from FP dataframe and convert to numpy
    fp_array = fp['flux'].to_numpy()

    # Take flux from LDLS dataframe and convert to numpy
    ldls_array = ldls['flux'].to_numpy()

    # Return normalized array
    return fp_array/ldls_array

# ----------------------------------------------------------------------------------

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
    guess = Gaussian1D(amplitude=y.max(), mean=np.mean(x), stddev=1)
    fit = lmfitter(model=guess, x=x, y=y)

    return fit

# ----------------------------------------------------------------------------------

def get_polyfit(x, y, deg):

    """
    Function to do polynomial fitting. 
    f(x; n) = y; with f(x) is an n-th degree polynomial

    Parameters:
    ----------
    x   : array-like
        coresponding x values
    y   : array-like
        y values to fit
    deg : int
        n-th degree polynomial to fit


    Returns:
    ----------
    fit : numpy.polynomial.Polynomial
        Fitted polynomial model object of n-th degree

    """

    coef = np.polynomial.polynomial.polyfit(x, y, deg)
    fit = np.polynomial.polynomial.Polynomial(coef)

    return fit

# ----------------------------------------------------------------------------------

def pattern_matching(peaks_array, pix_peaks, peaks_theory, position_tolerance=10,
                    diff_tolerance_left=0.63, diff_tolerance_right=1.5):
    
    """
    Function to match observed FP peaks to theoretical oness. Peaks_theory is
    calculated based on:

    lam = 2 * cavity_width / wavenumber.

    Cavity_width here is assumed constant of 400 micrometer (from: 4MOST manual).

    Parameters:
    ----------
    peaks_array : array-like, flipped from reddest wavelength to bluest
        array of the detected peaks in wavelength (Angstrom) from scipy
    pix_peaks   : array-like, flipped from highest pixel to lowest
        array of the detected peaks in pixel from scipy
    peaks_theory : array-like, flipped from reddest wavelength to bluest
        array of the theoretical 
    position_tolerance : int
        tolerance of how far the peaks_theory and peaks_array are.
    diff_tolerance_left : int
        lowest spacing difference from observed to theory that can be accepted
    diff_tolerance_right : int
        highest spacing difference from observed to theory that can be accepted


    Returns:
    ----------
    filtered_peaks : pandas DataFrame
        dataframe with three columns
        lambda    : matched peaks in wavelength (Angstrom)
        wavenumber: matched wavenumber from counting
        pix       : matched peaks in pix


    """

    # Initialize array.
    # The first peak is automatically recorded.
    matched_peaks = [peaks_array[0]]
    pixs = [pix_peaks[0]]
    wavenumber = [0]

    # Initialize index
    i = 0  # index for observed
    j = 0  # index for theoretical

    # Starting status
    success = True

    # Loop
    while i < len(peaks_array) - 1 and j < len(peaks_theory) - 1:

        # Calculate observed spacing 
        observed_spacing = abs(peaks_array[i+1] - peaks_array[i])

        # Calculate theoretical spacing
        theoretical_spacing = abs(peaks_theory[j+1] - peaks_theory[j])

        # Here we compare the observed spacing to theoretical spacing.
        # If they're a perfect match, calculating from one peak to the other,
        # they should be very similar or almost identical
        spacing_rel_error = observed_spacing / theoretical_spacing

        # Compare actual positions as well
        position_diff = abs(peaks_array[i] - peaks_theory[j])

        # This program only accepts subsequent peaks with a range of spacing_rel_error

        # Explanation:
        # The cavity width is not constant in reality, so the theoretical peak and 
        # observed peak will have some difference. 
        # For example, peak_1 in theory might be closer to peak_2 in observed. 
        # This does not mean that peak_2 share the same wavenumber with peak_1. 
        # This range of value allows the program to catch cases like this. 

        # The program by default append n+1 peaks for a successful match. it then append n and n+1 peaks for prev failures.
        # For example:
        # peak 0 and 1 is a successful match. peak 0 is already logged (initialize), so program only add peak 1.
        # peak 1 and 2 is a successful match too. bc peak 1 is already logged, program adds only peak 2.
        # peak 2 and 3 is not a succesful match. peak 3 is not counted.
        # peak 3 and 4 is a succesful match. peak 4 is automatically added but peak 3 is missing. thus for case where
        # previously it was a failure, the first peak is counted too.

        if (diff_tolerance_left < spacing_rel_error < diff_tolerance_right) and position_diff < position_tolerance:

            # Add one more index to theory and observation
            i += 1
            j += 1

            if success:

                # Append the last n+1 peak because the n peak is already recorded before, assuming
                # that the peak matching is successful.
                matched_peaks.append(peaks_array[i])
                pixs.append(pix_peaks[i])

                # Count it as the wavenumber from 1
                wavenumber.append(j)
            
            else:

                # If the previous peak matching is not successful, but this one is succesful, 
                # then the n peak is not counted. Thus this part counts the n peak and then the 
                # n+1 peak.

                # Append the previous peak and the current peak
                matched_peaks.append(peaks_array[i-1])
                matched_peaks.append(peaks_array[i])

                pixs.append(pix_peaks[i-1])
                pixs.append(pix_peaks[i])

                # Add the "catch up" wavenumber too
                wavenumber.append(j-1)
                wavenumber.append(j)

            # Remember that it is successful
            success = True

        else:

            # If the spacing is not successful,
            # try to skip peak[i+1] and check peak[i] to peak[i+2]
            # This is to catch cases where scipy identify 2 peaks for 1 FP peak.
            # If the first peak is too close with smaller spacing, then try for the second peak.
            # The tolerance of 1.5 will prevent the program to identify peaks with too far spacing. 

            if i + 2 < len(peaks_array):

                # Calculate observed spacing with 2 jumps
                skip_spacing = abs(peaks_array[i+2] - peaks_array[i])

                # Compare to theoretical spacing
                skip_spacing_error = skip_spacing / theoretical_spacing

                # Check if the matching is succesfull
                if diff_tolerance_left < skip_spacing_error < diff_tolerance_right:
                        
                    # Skip i+1, move to i+2
                    i += 2
                    j += 1

                    if success:

                        # Append the n+2 peak
                        matched_peaks.append(peaks_array[i])
                        pixs.append(pix_peaks[i])

                        # Count the wavenumber.
                        wavenumber.append(j)
                    
                    else:

                        # Catching up for failure cases

                        # Append n+2 and n peaks
                        matched_peaks.append(peaks_array[i-2])
                        matched_peaks.append(peaks_array[i])
                        
                        pixs.append(pix_peaks[i-2])
                        pixs.append(pix_peaks[i])

                        # Count wavenumber
                        wavenumber.append(j-1)
                        wavenumber.append(j)


                    # Remember that it is successful
                    success = True

                    continue

            # If it still doesn't work, move on
            i += 3
            j += 2
            
            # Remember that it is not successful
            success = False
    
    # Log in into a dataframe
    filtered_peaks = pd.DataFrame({'lambda': np.array(matched_peaks),
                                    'wavenumber': np.array(wavenumber),
                                    'pix': np.array(pixs)})

    return filtered_peaks

# ----------------------------------------------------------------------------------

class Wavefp():

    def __init__(self, color, prom, height, dist, start=None, stop=None, mode='robust'):

        """
        Class to get wavelength calibration with Fabry-Perot.
        
        """

        self.color = color             # Color filter
        self.prom = prom               # Prominence for scipy
        self.height = height           # Minim height of peaks
        self.dist = dist               # Minim distance between each peaks
        self.start = start             # Pixel start
        self.stop = stop               # Pixel stop

        # Define cavity from manufacturer
        self.cavity = 400 * u.um

        # Call arc lines for a particular filter
        self.lines = pd.read_csv('line_lists/master_peaks_{}.csv'.format(self.color), index_col=None)
        
        # Fabry-perot spectrum
        self.fp = pd.read_csv('spectra/spectra_FP_{}.csv'.format(self.color), index_col=None)

        # LDLS spectrum
        self.ldls = pd.read_csv('spectra/spectra_LDLS_{}.csv'.format(self.color), index_col=None)

        # Normalized spectrum
        self.norm = get_norm(self.fp, self.ldls)

        # Determine start and end of lines
        end_pix = len(self.norm)

        # Modify spectrum for specific start and stop
        if (self.start != None) or (self.stop != None):
            self.norm = self.norm[self.start:self.stop]

        if (self.stop == None):
            self.stop = end_pix
            
        if (self.start == None):
            self.start = 0

        # Get array of pix
        self.pix = np.arange(self.start, self.stop)

        # Get peaks from scipy's find_peaks
        self.peaks, _ = find_peaks(self.norm, height=self.height, prominence=self.prom, distance=self.dist)

        # Note that at this point self.peaks count from 0 and not self.start!
        # It will be added on later!
    
    # --------------------------------------
    
    def fit_from_arc(self, array):

        """
        Function to do polynomial fitting specifically for the arc lines
        f(pix; n) = lambda. With n as 6th degree polynomial by default.

        Parameters:
        ----------
        array   : array-like
            an array of pix

        Returns:
        ----------
        ffit(array) : array-like
            array of lambda from the arc line fitting. 
            
        """

        ffit = get_polyfit(self.lines['gaussian peaks'], self.lines['lambda'], 6)

        return ffit(array)

    # --------------------------------------
        
    def get_normalized_FP(self):

        """
        Function to normalize FP peaks from 0 to 1 as much as possible.

        This takes values of self.norm, which is a rough normalization of FP/LDLS.

        In reality, this still have some features. 

        Here I used rolling average to normalize the featured norm spectrum to a more
        uniform shaped. 

        Parameters: (self, within class)
        ----------
        norm : array-like
            imperfect/normalized FP spectrum with left-over features
        peaks : array-like
            peaks identified by scipy


        Returns:
        ----------
        norm_FP : pandas DataFrame
            dataframe of newly-normalized FP with three columns
            pix : pixels of FP peaks
            lambda : lambda of FP peaks in Angstrom
            flux : flux of FP peaks
            
        """

        # Get value of peaks
        peaks_value = self.norm[self.peaks]

        # Initialize array
        matrix_value = []
        matrix_pix = []
        array_med = []

        # To get an array of peaks and their value every 10th
        # [0, .., 10]
        # [11, .., 20]
        # etc.

        # The len(see_peaks)-1 makes sure that if array has less than 10 peaks 
        # it uses the numbers as the size.
        # example: 107, the end would be [100, .., 107]

        for i in range(0, len(peaks_value), 10):

            range_size = min(10, len(peaks_value)-i)

            array_value = np.array(peaks_value[i:i+range_size])
            array_pix = np.array(self.peaks[i:i+range_size])

            matrix_value.append(array_value)
            matrix_pix.append(array_pix)
            array_med.append(np.median(array_value))
        
        # Determine the area where the 10 peaks are encompassed
        def j(matrix, i):

            return matrix[i].shape[0] - 1
            
        def get_mid_of_peaks(matrix, i):

            my_right_peak = matrix[i][j(matrix, i)]
            next_left_peak = matrix[i+1][0]

            return np.trunc((next_left_peak - my_right_peak) / 2)
        
        # Make a loop for the amount of list:
        edges = []
        
        for i in range(len(matrix_pix)):
            
            if (i == 0):

                m_right = get_mid_of_peaks(matrix_pix, i)

                left_edge = 0
                right_edge = matrix_pix[i][j(matrix_pix, i)] + m_right

                edges.append([left_edge, right_edge])
            
            elif (i != 0) and (i != (len(matrix_pix)-1)):

                m_left = get_mid_of_peaks(matrix_pix, i-1)
                m_right = get_mid_of_peaks(matrix_pix, i)

                left_edge = matrix_pix[i-1][j(matrix_pix, i-1)] + m_left + 1
                right_edge = matrix_pix[i][j(matrix_pix, i)] + m_right

                edges.append([left_edge, right_edge])
            
            elif (i == len(matrix_pix)-1):

                m_left = get_mid_of_peaks(matrix_pix, i-1)

                left_edge = matrix_pix[i-1][j(matrix_pix, i-1)] + m_left + 1
                right_edge = matrix_pix[i][j(matrix_pix, i)]

                edges.append([left_edge, right_edge])

        # Loop to get new normalized values
        for i in range(len(array_med)):

            left = int(edges[i][0])
            right = int(edges[i][1])

            self.norm[left:right] = self.norm[left:right]/array_med[i]

        # Move it to a pandas dataframe
        self.norm_FP = pd.DataFrame({
                                'pix': self.pix,
                                'lambda': self.fit_from_arc(self.pix),  # from pix to wavelength from arc lines
                                'flux': self.norm,
                                })

    # --------------------------------------

    def get_reference_wm(self):

        """ 
        Function to get the reference wavenumber.

        The reference wavelength number is the defined by the reddest peak.

        From Hobson et al. (2021) Eq 2:

        wavenumber = 2 * cavity width / lambda,

        The wavenumber is then rounded to the nearest integer.

        Returns:
        ----------
        ref_wnumber : float
                    rounded reference wavenumber determined from the reddest
                    wavelength

        """

        # Get newly-normalized FP
        self.get_normalized_FP()

        # Add self.start to self.peaks
        self.peaks = self.peaks + self.start

        # Get lambda of the reddest peak
        last_peak = self.fit_from_arc(self.peaks[-1])

        # Find the reference wavenumber
        wnumber = (2 * self.cavity.to(u.AA)) / (last_peak * u.AA)

        # Wavenumber should be an integer (from the paper), then the value is rounded
        self.ref_wnumber = np.round(wnumber.value)
    
    # --------------------------------------

    def check_pattern_and_match(self, plot=False, range_left=None, range_right=None):

        """
        Function to do the pattern matching, checking between observed peaks and
        theoretical peaks. 

        The wavelength is determined from a constant cavity width and wavenumber.

        lam = 2 * cavity width / wavenumber

        The wavenumber is then rounded to the nearest integer.

        Parameters:
        ----------
        plot    : bool
                for plotting the observed and theoretical peaks
        range_left : int
                range for plotting
        range_right : int
                range for plotting

        Returns:
        ----------
        fp_peaks : pandas DataFrame
                dataframe with three columns
                lambda    : matched peaks in wavelength (Angstrom)
                wavenumber: matched wavenumber from counting
                pix       : matched peaks in pix
        """

        # Get reference wavenumber and normalized FP
        self.get_reference_wm()

        # Make dataframe of peaks according to theory
        # lambda = 2 * d / wm

        # Create integers number from wavenumber reference 
        peaks = np.flip(np.arange(self.ref_wnumber, 3000))

        # Create dataframe of theory peaks
        peaks_theory = pd.DataFrame({'wavenumber': peaks,
                                    'lambda': ((self.cavity.to(u.AA) * 2) / peaks)})
        
        # Only include theory peaks inside of the norm_FP spectrum
        theory = peaks_theory[(peaks_theory['lambda'] > self.norm_FP['lambda'].min()) & 
                              (peaks_theory['lambda'] < self.norm_FP['lambda'].max())].copy()

        # I flip the peaks so that the algorithm starts counting from the reddest wavelength.
        # This is how the wavenumbers are calculated as well!

        # Extract theory peaks in np.array and then flip.
        theory_peaks = theory['lambda'].values
        theory_peaks = np.flip(theory_peaks)

        # Extract the observed peaks from scipy in np.array and then flip
        scipy_peaks = self.fit_from_arc(self.peaks)
        scipy_peaks = np.flip(scipy_peaks)

        # Use self.peaks as well to get the peaks in pixel
        pix_peaks = np.flip(self.peaks)

        # Run the pattern matching algorithm
        self.fp_peaks = pattern_matching(scipy_peaks, pix_peaks, theory_peaks)

        if plot:

            # Plot
            plt.plot(self.norm_FP['lambda'], self.norm_FP['flux'], color='black')
            plt.plot(theory_peaks, [1]*len(theory_peaks), 'x', color='blue', label='theory')
            plt.plot(scipy_peaks, [1]*len(scipy_peaks), '+', color='green', label='scipy')
            plt.plot(self.filtered_peaks, [1]*len(self.filtered_peaks), 'o', color='red', label='filtered')

            # Give notes on number of peaks for theory
            for i, x in enumerate(theory_peaks):
                if (x > range_left) and (x < range_right):
                    plt.text(x, 1.02, str(i), fontsize=8, ha='center', color='blue')

            # For scipy
            for i, x in enumerate(scipy_peaks):
                if (x > range_left) and (x < range_right):
                    plt.text(x, 1.05, str(i), fontsize=8, ha='center', color='green')

            plt.xlim(range_left,range_right)
            plt.ylim(0.6,1.3)

            plt.xlabel('wavelength [$\AA$]', fontsize=12)
            plt.ylabel('normalzied flux', fontsize=12)
            plt.legend(fontsize=15)

            plt.show()

    # --------------------------------------
    
    def get_gaussian_peaks(self, plot=False, xleft=None, xright=None, ydown=None, yup=None):

        """
        Function to get gaussian peaks in pixel and wavelength.

        Parameters:
        ----------
        plot  : bool
            for plotting the detected gaussian and scipy peaks
        xleft : int
            lower range for plotting, x-axis
        xright : int
            upper range for plotting, x-axis
        yup   : int
            upper range for plotting, y-axis
        ydown : int
            lower range for plotting, y-axis

        Returns:
        ----------
        fp_peaks : pandas DataFrame
                dataframe with three columns
                lambda    : gaussian peaks in wavelength (Angstrom)
                wavenumber: wavenumber from counting
                pix       : gaussian peaks in pix
        
        """
        
        # Get matched FP and normalized FP
        self.check_pattern_and_match()

        # --------------------------------------------------------------
        # This is the part of the code where I determined the gaussian peaks in wavelength directly
        # instead of doing it in pixel. This returns a smoother plot comapred to determining this from
        # the pixels, but then one cannot do the wavelength calibration. 

        # Please only uncomment if really needed!

        # ----------- uncomment here if needed --------------------------

        # # Define peaks array in 
        # new_peaks = []

        # # Check every 10th peak
        # for i, p in enumerate(self.filtered_peaks):

        #     fsr = (p ** 2) / (2 * self.cavity.to(u.AA).value)

        #     # Proper bounds
        #     check = self.norm_FP[(self.norm_FP['lambda'] > p - (fsr/1.5)) & (self.norm_FP['lambda'] < p + (fsr/1.5))]

        #     x = check['lambda'].values
        #     y = check['flux'].values

        #     # Call gaussian fitting
        #     model = get_gaussian(y, x)
        #     gaus_p = model.mean.value

        #     # If the gaussian peaks are smaller than the FSR, then take the gaussian peak
        #     if abs(gaus_p - p) < fsr/2:
        #         new_peaks.append(gaus_p)

        #     # For failed gaussian peaks, takes the scipy peaks. But this can be really inaccurate!
        #     else:
        #         new_peaks.append(p)

        # # Make new gaussian peak
        # self.gaussian_peaks = np.asarray(new_peaks)

        # --------------------------------------------------------------

        # Array to store gaussian peaks in pixels
        pix_new = []

        # Find dispersion to convert wavelength to pixel
        dlam_dpix = self.fit_from_arc(5001) - self.fit_from_arc(5000)

        # Loop for every pix in fp_peaks
        for i, p in enumerate(self.fp_peaks['pix']):

            # Calculate FSR in angstrom
            fsr = (p ** 2) / (2 * self.cavity.to(u.AA).value)

            # Convert FSR bounds in angstrom to pixel with 1 pixel to the left and right
            left_bound =  p - ((fsr/1.5)/dlam_dpix) - 1
            right_bound = p + ((fsr/1.5)/dlam_dpix) + 1

            # Fit gaussian only for the proper peak boundaries
            check = self.norm_FP[(self.norm_FP['pix'] > left_bound) & (self.norm_FP['pix'] < right_bound)]

            # Determine x and y for gaussian fitting
            x = check['pix'].values
            y = check['flux'].values

            # Call gaussian fitting
            model = get_gaussian(y, x)
            gaus_p = model.mean.value

            # If the gaussian peaks are smaller than the FSR, then take the gaussian peak
            if abs(gaus_p - p) < ((fsr/2)/dlam_dpix):
                pix_new.append(gaus_p)

            # For failed gaussian peaks, takes the scipy peaks. But this can be really inaccurate!
            else:
                pix_new.append(p)
        
        # Update values to pandas DataFrame
        self.fp_peaks['pix'] = np.asarray(pix_new)                          # gaussian peaks in pixel
        self.fp_peaks['lambda'] = np.asarray(self.fit_from_arc(pix_new))    # gaussian peaks in angstrom
        self.fp_peaks['wavenumber'] += self.ref_wnumber                     # wavenumber from reference wavenumber

        if plot:

            plt.figure(figsize=[20,12])

            plt.plot(self.pix, self.norm, color='#266489', alpha=0.7, linewidth=2)

            plt.scatter(self.pix[self.peaks], [1.1]*len(self.peaks), marker='x', s=100, 
                        color='red', label='peak detection')
            plt.scatter(self.fp_peaks['pix'], [1.1]*len(self.fp_peaks), marker='+', s=100, 
                        color='black', label='gaussian')

            plt.xlim(xleft,xright)
            plt.ylim(ydown, yup)
            plt.xlabel('Pix', fontsize=18)
            plt.ylabel('Flux', fontsize=18)

            plt.legend(fontsize=18)
    
    # --------------------------------------
    
    def get_new_cavity(self, start_deg=7, iter_poly=5, plot=False):
        
        """
        Function to get cavity width as a function of wavenumber and/or wavelength.

        This function returns a plot to determine the polynomial degree for the cavity
        width fitting. 
        
        Note that it does NOT automatically return the best degree, because the RMS
        does not soley define which degree is the best fit. Please analyze the plot
        visually as well and not only by the returned number. 

        Parameters:
        ----------
        start_deg : int
                starting degree for fitting
        iter_poly : int
                how many fitting iterations that are interested. counts from
                start_degree to start_degree+iter_poly
        plot : bool
                determine to plot or not

        Returns:
        ----------
        fp_peaks : pandas DataFrame
                updated fp_peaks dataframe with now five columns
                lambda    : gaussian peaks in wavelength (Angstrom)
                wavenumber: wavenumber from counting
                pix       : gaussian peaks in pix
                cavity    : cavity width for every wavenumber
                theory lambda : theoreticaly wavelength from constant cavity width

        """

        # Get reference wavenumber 
        self.get_gaussian_peaks()

        # Convert 'lambda' column to Quantity with units, then to microns, then extract value
        wavelength_um = (self.fp_peaks['lambda'].values * u.AA).to(u.um).value

        # Get cavity
        self.fp_peaks['cavity'] = wavelength_um * self.fp_peaks['wavenumber'] / 2

        # Get theory lambda
        self.fp_peaks['theory lambda'] = (2 * self.cavity.to(u.AA)).value / self.fp_peaks['wavenumber']

        # Flip the pandas dataframe
        self.fp_peaks = self.fp_peaks.sort_values(by='lambda').reset_index(drop=True)

        if plot:

            # Plot figure
            fig, ax = plt.subplots(2, 1, figsize=[15, 8], gridspec_kw={'height_ratios': [2, 1.5]})

            # Define color map
            cmap = plt.get_cmap('viridis')

            # Loop for the amount of poly degree iteration that is interested
            for i in range(iter_poly):

                # Get fit for wavenumber vs cavity
                ffit = get_polyfit(self.fp_peaks['wavenumber'], self.fp_peaks['cavity'], i+start_deg)

                # For double axis
                def wm_to_lam(x):
                    return (2 * 400e4) / x
                
                def lam_to_wm(x):
                    return (2 * 400e4) / x
                
                # Residual from fit
                res = ffit(self.fp_peaks['wavenumber']) - self.fp_peaks['cavity']
                rms = np.sqrt(np.sum(res**2))

                # Plot wavelength vs cavity from fit
                ax[0].plot(self.fp_peaks['wavenumber'], ffit(self.fp_peaks['wavenumber']), 
                           color=cmap(i/6), label='deg={}, rms={:.5f}'.format(i+start_deg, rms))
                
                # Secondary axis
                secax = ax[0].secondary_xaxis('top', functions=(wm_to_lam, lam_to_wm))
                secax.set_xlabel('wavelength [$\AA$]', fontsize=12)

                # Plot residual
                ax[1].scatter(self.fp_peaks['wavenumber'], res, color=cmap(i/6), s=10,
                        label='deg={}'.format(i+start_deg))

            # Plot wavenumber vs cavity from data
            ax[0].scatter(self.fp_peaks['wavenumber'], self.fp_peaks['cavity'], s=20, color='black', 
                          label='FP peaks')

            ax[0].set_ylabel('cavity width [$\mu m$]', fontsize=12)
            ax[1].set_ylabel('residuals [$\mu m$]', fontsize=12)
            ax[1].set_xlabel('wavenumber', fontsize=12)

            ax[0].legend()

            fig.suptitle('Cavity width as a function of wavenumber ({} channel)'.format(self.color), 
                         fontsize=15)
            
            plt.show()

            # plt.savefig('plots/{}_cavitywidth.png'.format(self.color))
    
    # --------------------------------------
    
    def get_wave_cal(self, deg_for_cavity=9, deg_for_cal=9, plot_check=False, plot_final=False):

        """
        Function to get wavelength calibration from FP peaks and arc lines

        Parameters:
        ----------
        deg_for_cavity : int
                poly degree for cavity vs wavenumber fitting
        deg_for_cal : int
                poly degree for wavelength calibration
        plot_check : bool
                plotting to determine poly degree for wavelength calibration
        plot_final : bool
                plotting to remove 3-sigma outliers from chosen poly degree

        Returns:
        ----------
        fp_peaks : pandas DataFrame
                updated fp_peaks dataframe with now seven columns
                    lambda    : gaussian peaks in wavelength (Angstrom)
                    wavenumber: wavenumber from counting
                    pix       : gaussian peaks in pix
                    cavity    : cavity width for every wavenumber
                    theory lambda : theoreticaly wavelength from constant cavity width
                    new cavity : cavity width from fit
                    new lambda : wavelength from new cavity width fit
        all_lines : pandas DataFrame
                dataframe of all peaks, from FP and arc lines

        """

        # Get new cavity width as a function of wavenumber
        self.get_new_cavity()

        # Get fit function
        ffit = get_polyfit(self.fp_peaks['wavenumber'], self.fp_peaks['cavity'], deg_for_cavity)

        # Get it into dataframe
        self.fp_peaks['new cavity'] = ffit(self.fp_peaks['wavenumber']) * 1e4
        self.fp_peaks['new lambda'] = (2 * self.fp_peaks['new cavity']) / self.fp_peaks['wavenumber']

        # Make dataframe of all lines (arc + fp)
        fp_df = pd.DataFrame({'gaussian peaks': self.fp_peaks['pix'].to_numpy(),
                            'lambda': self.fp_peaks['new lambda'].to_numpy(),
                            'source': ['fp']*len(self.fp_peaks)})
        
        # Add additional info to arc lines
        self.lines['source'] = ['arc']*len(self.lines)

        # Combine all lines from FP and arc lines
        self.all_lines = pd.concat([self.lines, fp_df], axis=0, ignore_index=True)
        self.all_lines = self.all_lines.sort_values(by=['gaussian peaks'], ignore_index=True)

        # Plot to determine poly degree for wavelength calibration
        if plot_check:

            fig, ax = plt.subplots(2, 1, figsize=[15,6], gridspec_kw={'height_ratios': [2, 1.5]})

            cmap = plt.get_cmap('viridis')

            for i in range(3):

                # Get polyfit function for lambda vs pix
                ffit = get_polyfit(self.all_lines['gaussian peaks'], self.all_lines['lambda'], i+deg_for_cal)

                # Plot lambda vs gaussian peaks 
                ax[0].plot(ffit(self.all_lines['gaussian peaks']), self.all_lines['gaussian peaks'], color=cmap((i+3)/6))
                ax[0].set_xlim(min(self.all_lines['lambda']), max(self.all_lines['lambda']))

                # Get residual
                res = ffit(self.all_lines['gaussian peaks']) - self.all_lines['lambda']
                rms = np.sqrt(np.sum(res**2))

                # Plot residuals
                ax[1].plot(ffit(self.all_lines['gaussian peaks']), res, color=cmap((i+3)/6), 
                           label='deg = {}, res = {:.5f}'.format(i+deg_for_cal, res))
                ax[1].set_xlim(min(self.all_lines['lambda']), max(self.all_lines['lambda']))

            # Plot lambda vs gaussian peaks from observation
            ax[0].scatter(self.all_lines['lambda'], self.all_lines['gaussian peaks'], s=10, color='black')

            # plot where the arc lamps are
            ax[1].scatter(ffit(self.lines['gaussian peaks']), ffit(self.lines['gaussian peaks']) - self.lines['lambda'], 
                            color=cmap(1/6), label='arc lines')

            ax[0].set_title('wavelength calibration arc lines + FP ({} channel)'.format(self.color), fontsize=15)
            ax[0].set_ylabel('pixels', fontsize=12)
            ax[1].set_ylabel('residual [$\AA$]', fontsize=12)
            ax[1].set_xlabel('wavelength [$\AA$]', fontsize=12)

            ax[1].legend(loc='upper right')

            plt.show()
            
            # plt.savefig('plots/{}_wavelengthcal_dotted.png'.format(self.color))

        # Plot to determine the final wavelength calibration from chosen poly degree
        if plot_final:

            # Fit polynomial
            ffit = get_polyfit(self.all_lines['gaussian peaks'], self.all_lines['lambda'], deg_for_cal)

            # Compute fit values
            self.all_lines['fit lambda'] = ffit(self.all_lines['gaussian peaks'])

            # Compute residuals and sigma
            self.all_lines['residuals'] = self.all_lines['lambda'] - self.all_lines['fit lambda']

            # Loop for 6x time to get rid of 3-sigma outliers
            for _ in range(6):
                
                # Determine std for the residuals
                std = self.all_lines['residuals'].std()

                # Mask inliers
                inliers = abs(self.all_lines['residuals']) < 3 * std

                # Keep only inliers for next iteration
                self.all_lines = self.all_lines[inliers]

            # All filtered lines
            self.all_lines = self.all_lines.reset_index(drop=True)

            # Arc lines that are filtered
            self.lines = self.all_lines[self.all_lines['source'] == 'arc'].reset_index(drop=True)
            
            # Plot
            fig, ax = plt.subplots(2, 1, figsize=[15,6], gridspec_kw={'height_ratios': [2, 1.5]})

            cmap = plt.get_cmap('viridis')

            ax[0].plot(self.all_lines['fit lambda'], self.all_lines['gaussian peaks'], color=cmap(1/2))
            ax[0].scatter(self.all_lines['lambda'], self.all_lines['gaussian peaks'], s=10, color=cmap(1/5))

            ax[1].plot(self.all_lines['fit lambda'], self.all_lines['residuals'], color=cmap(1/2))
            ax[1].scatter(self.lines['fit lambda'], self.lines['residuals'], s=10, color=cmap(1/4))

            ax[0].set_title('wavelength calibration arc lines + FP ({} channel)'.format(self.color), fontsize=15)
            ax[0].set_ylabel('pixels', fontsize=12)
            ax[1].set_ylabel('residual [$\AA$]', fontsize=12)
            ax[1].set_xlabel('wavelength [$\AA$]', fontsize=12)

            plt.show()

# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    # Dictionary for data
    filter = {'blue': [0.02, 0.1, 3],
            'green': [0.05, 0.1, 5]} # prom, height, dist, start 

    # Call class for green filter
    fil = filter['green']
    wp_g = Wavefp(color='green', prom=fil[0], height=fil[1], dist=fil[2], start=0)

    # Call class for blue filter
    fil = filter['blue']
    wp_b = Wavefp(color='blue', prom=fil[0], height=fil[1], dist=fil[2], start=1000)

    # Do wavelength calibration for both blue and green filter
    wp_b.get_wave_cal(plot_final=True)
    wp_g.get_wave_cal(plot_final=True)

    # Combine all identified lines
    master_fp_peaks = pd.concat([wp_b.all_lines, wp_g.all_lines], axis=0, ignore_index=True)

    # Only use FP peaks and not arc lines
    master_fp_peaks = master_fp_peaks[master_fp_peaks['source'] == 'fp']
    master_fp_peaks = master_fp_peaks[['fit lambda']]
    master_fp_peaks = master_fp_peaks.rename(columns={'fit lambda': 'lambda'})

    # Export to CSV
    master_fp_peaks.to_csv('line_lists/master_peaks_fp.csv', index=False)








        


        












