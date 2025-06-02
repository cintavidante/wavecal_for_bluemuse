"""
---------------------
Wavecal_for_BlueMUSE
---------------------

Cinta Vidante, 2025

This module contains functions and classes used to generate simulation 
of BlueMUSE's calibration images.

Part of my work on BlueMUSE with Peter Weilbacher, AIP.

"""

# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy import units as u
from astropy.io import fits
from astropy.time import Time

from astropy.modeling.models import Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter

from scipy.linalg import lstsq
from scipy.special import binom
from scipy.optimize import fsolve

from datetime import datetime, timezone

# ------------------------------------------------------------

def polyfit2d(x, y, z, degree=1, max_degree=None, scale=True, plot=False):
     
	"""A simple 2D polynomial fit to data x, y, z
	The polynomial can be evaluated with numpy.polynomial.polynomial.polyval2d

	Parameters
	----------
	x : array[n]
		x coordinates
	y : array[n]
		y coordinates
	z : array[n]
		data values
	degree : {int, 2-tuple}, optional
		degree of the polynomial fit in x and y direction (default: 1)
	max_degree : {int, None}, optional
		if given the maximum combined degree of the coefficients is limited to this value
	scale : bool, optional
		Wether to scale the input arrays x and y to mean 0 and variance 1, to avoid numerical overflows.
		Especially useful at higher degrees. (default: True)
	plot : bool, optional
		wether to plot the fitted surface and data (slow) (default: False)

	Returns
	-------
	coeff : array[degree+1, degree+1]
		the polynomial coefficients in numpy 2d format, i.e. coeff[i, j] for x**i * y**j
	"""

	def _get_coeff_idx(coeff):
		idx = np.indices(coeff.shape)
		idx = idx.T.swapaxes(0, 1).reshape((-1, 2))
		return idx

	def _scale(x, y):
		# Normalize x and y to avoid huge numbers
		# Mean 0, Variation 1
		offset_x, offset_y = np.mean(x), np.mean(y)
		norm_x, norm_y = np.std(x), np.std(y)
		x = (x - offset_x) / norm_x
		y = (y - offset_y) / norm_y
		return x, y, (norm_x, norm_y), (offset_x, offset_y)

	def _unscale(x, y, norm, offset):
		x = x * norm[0] + offset[0]
		y = y * norm[1] + offset[1]
		return x, y

	def polyvander2d(x, y, degree):
		A = np.polynomial.polynomial.polyvander2d(x, y, degree)
		return A

	def polyscale2d(coeff, scale_x, scale_y, copy=True):
		if copy:
			coeff = np.copy(coeff)
		idx = _get_coeff_idx(coeff)
		for k, (i, j) in enumerate(idx):
			coeff[i, j] /= scale_x ** i * scale_y ** j
		return coeff

	def polyshift2d(coeff, offset_x, offset_y, copy=True):
		if copy:
			coeff = np.copy(coeff)
		idx = _get_coeff_idx(coeff)
		# Copy coeff because it changes during the loop
		coeff2 = np.copy(coeff)
		for k, m in idx:
			not_the_same = ~((idx[:, 0] == k) & (idx[:, 1] == m))
			above = (idx[:, 0] >= k) & (idx[:, 1] >= m) & not_the_same
			for i, j in idx[above]:
				b = binom(i, k) * binom(j, m)
				sign = (-1) ** ((i - k) + (j - m))
				offset = offset_x ** (i - k) * offset_y ** (j - m)
				coeff[k, m] += sign * b * coeff2[i, j] * offset
		return coeff

	def plot2d(x, y, z, coeff):
		# regular grid covering the domain of the data
		if x.size > 500:
			choice = np.random.choice(x.size, size=500, replace=False)
		else:
			choice = slice(None, None, None)
		x, y, z = x[choice], y[choice], z[choice]
		X, Y = np.meshgrid(
			np.linspace(np.min(x), np.max(x), 20), np.linspace(np.min(y), np.max(y), 20)
		)
		Z = np.polynomial.polynomial.polyval2d(X, Y, coeff)
		fig = plt.figure()
		#ax = fig.gca(projection="3d") # deprecated
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
		ax.scatter(x, y, z, c="r", s=50)
		plt.xlabel("X")
		plt.ylabel("Y")
		ax.set_zlabel("Z")
		plt.show()

	# Flatten input
	x = np.asarray(x).ravel()
	y = np.asarray(y).ravel()
	z = np.asarray(z).ravel()

	# Remove masked values
	mask = ~(np.ma.getmask(z) | np.ma.getmask(x) | np.ma.getmask(y))
	x, y, z = x[mask].ravel(), y[mask].ravel(), z[mask].ravel()

	# Scale coordinates to smaller values to avoid numerical problems at larger degrees
	if scale:
		x, y, norm, offset = _scale(x, y)

	if np.isscalar(degree):
		degree = (int(degree), int(degree))
	degree = [int(degree[0]), int(degree[1])]
	coeff = np.zeros((degree[0] + 1, degree[1] + 1))
	idx = _get_coeff_idx(coeff)

	# Calculate elements 1, x, y, x*y, x**2, y**2, ...
	A = polyvander2d(x, y, degree)

	# We only want the combinations with maximum order COMBINED power
	if max_degree is not None:
		mask = idx[:, 0] + idx[:, 1] <= int(max_degree)
		idx = idx[mask]
		A = A[:, mask]

	# Do the actual least squares fit
	C, *_ = lstsq(A, z)

	# Reorder coefficients into numpy compatible 2d array
	for k, (i, j) in enumerate(idx):
		coeff[i, j] = C[k]

	# Reverse the scaling
	if scale:
		coeff = polyscale2d(coeff, *norm, copy=False)
		coeff = polyshift2d(coeff, *offset, copy=False)

	if plot:
		if scale:
			x, y = _unscale(x, y, norm, offset)
		plot2d(x, y, z, coeff)

	return coeff

# ------------------------------------------------------------

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

# --------------------------------------------

def read_ascii_config(file_path):

    """
    Function to read from ASCII file. The input of the main class comes
    from an ASCII file.

    Parameters:
    ----------
    file_path : str
            path to configuration file

    Returns:
    ----------
    config : dict
          dictionary containing the parameters

    """

    # Initialize dictionary
    config = {}

    with open(file_path, 'r') as file:

        for line in file:

            # Strip whitespace and ignore comments (lines starting with '#')
            line = line.strip()

            if line and not line.startswith('#'):

                # Split on the first '=' to separate keys and values
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

    return config


# --------------------------------------------

def get_poly_1d(x, coef):

    """
    1D polynomial at x given coefficients.

    The polynomial is defined as:
        f(x) = c_0 + c_1 * x + c_2 * x^2 + ... + c_n * x^n

    Parameters
    ----------
    x : array-like
      x values that want to be evaluated
    coef : array-like
      coefficients of the polynomial

    Returns
    -------
    y : array-like
      polynomial values evaluated at x

    """
    return np.polynomial.polynomial.polyval(x, coef)

# --------------------------------------------

def solve_equations(y_val, coef, x_fixed, wvl_target):

    """
    Residual function to determine the solution from scipy's fsolve.

    Two sets of equations to solve a 2D polynomial problem.
    'If z = f(x, y), and one has z (wvl), what would x and y be?

    Parameters
    ----------
    y_val   : float
            y_val that wants to be evaluated
    coef    : array-like
            coefficients of the 2D polynomial
    x-fixed : float
            x-axis pixel that is known
    wvl_target : float
              wavelength target 

    Returns
    -------
    wvl - wvl_target : float
        difference between target and current iteration.
        a root-finding algorithm minimizes this to zero.

    """

    # Equation to get wavelength from 2D polynomial.
    # This is to check what value of wavelength for every x and y.
    wvl = np.polynomial.polynomial.polyval2d(x_fixed, y_val[0], coef)

    # This function helps scipy's fsolve to numerically find solution.
    # For every iteration, they check their result to the target value.
    return wvl - wvl_target

# --------------------------------------------

def solve_for_y_wvl(coef, wvl_target, x_fixed, initial_guess=0, reso=False):

    """
    Function to find solution to a 2D polynomial, z = f(x, y). Uses scipy's fsolve.
    Also find the dispertion pix/angstrom, as in dy/dz.

    Parameters
    ----------
    coef    : array-like
            coefficients of the 2D polynomial
    x-fixed : float
            x-axis pixel that is known
    wvl_target : float
              wavelength target 
    initial_guess : float
                 initial guess of the root-finding algorithm

    Returns
    -------
    sol : float
        solution for 2D polynomial, given z and x with f(x, y) = z.
    reso : float
        finding dy/dz
    
    """

    # Find solution exactly for the target wavelength.
    sol = fsolve(solve_equations, initial_guess, args=(coef, x_fixed, wvl_target))

    # Calculate the resolution from dy/dz.
    # (y_0 - y_1) (pix) / (lambda_0 - lambda_0 + 1) (angstrom) -> pix/angstrom.
    # disper = (sol_2[1] - sol_1[1]) 
    if reso:

        sol_1 = fsolve(solve_equations, initial_guess, args=(coef, x_fixed, wvl_target+1))
        disper = (sol_1 - sol)

        return sol, reso
    
    else:
        return sol, None
    
# --------------------------------------------

def transform_to_pix(disp):

    """
    Function to transfrom from milimeter (mm) to pixel values. The BlueMUSE's
    dispersion files is given in mm as distance from the center. This code 
    transforms the X and Y position to pixels from 0.
    
    One pixel is 15 micrometer (um).

    Parameters
    ----------
    disp    : pandas DataFrame
            dataframe of BlueMUSE's dispersions. 
            the values of X and Y are in mm calculated from the center.

    Returns
    -------
    disp    : pandas DataFrame
            dataframe with X and Y position in pixels.

    """

    # Copy the pixel dispersion's dataframe 
    pix_disp = disp.copy()

    # Transform from mm to pixel
    pix_disp['CX_DET'] = ((pix_disp['CX_DET'].values * u.mm).to(u.um) / (15 * u.um)).value + 2048
    pix_disp['CY_DET'] = ((pix_disp['CY_DET'].values * u.mm).to(u.um) / (15 * u.um)).value + 2056

    return pix_disp

# --------------------------------------------

# ----------- uncomment if needed ------------

def create_subset(pix_disp, n):

    """

    Create subset for each slice. This is the format of the
    old dispersion file.

    Parameters
    ----------
    pix_disp   : pandas DataFrame
            dataframe of BlueMUSE's dispersions. 

    Returns
    -------
    X   : array-like
        X values of every center position in one slice
    Y   : array-like
        Y values for every center position in one slice
    WVL : array-like
        wavelengths at each X and Y

    """

    # Make a subset for each slice. 
    subset = pix_disp[['WVL', 'X_{}'.format(n), 'Y_{}'.format(n)]]

    # Drops any rows that have NaN 
    subset = subset.dropna()

    # Convert pandas dataframe to numpy arrays
    X = subset['X_{}'.format(n)].values
    Y = subset['Y_{}'.format(n)].values
    WVL = subset['WVL'].values

    return X, Y, WVL

# --------------------------------------------

def write_FITS(data, img, name, spec):

    """
    Class to generate all FITS files.

    I wrote the headers manually.. there should be a more
    efficient way to do this..

    Parameters
    ----------
    data   : 2D array-like
            the 2D array of the image that wants to be generated
    img    : dict
            the image dictionary with specialized info for different images.
            looks like this: 

            'flat': ["Flat image", None, other_info],
            'bias': ["Bias image", None, other_info],
            'wavemap': ["Wavelength map", None, other_info],
            'arc': ["Arc exposure", "Line identification", other_info_arc, "arc"],
            'spec': ["Full spectrum", "Interpolation", other_info_arc, "spec"]

            other_info is an array for:
            [trace_degree, wavecal_degree_1, wavecal_degree_2,
            bias_flux, flat_flux, 'BlueMUSE']

            # This is difined in the class definition
    
    name    : str
            name for file saving
    spec    : bool
            determine if there is a full-spectrum file (not just lines)

    Returns
    -------
    hdr     : Header
            primary HDU header
    hdx     : Header
            header to be put to the extension file
    hdu_img : ImageHDU 
            astropy fits image HDU of data with header = hdx
    hdul    : HDUList
            list of all HDUs that wants to be generated in the FITS

    """

    # Determine time
    ut = Time(datetime.now(tz=timezone.utc), scale='utc')
    date_obs = ut.datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] 

    # PrimaryHDU header
    hdr = fits.Header()
    hdr['HIERARCH ESO DET BINX'] = (1, "Setup binning factor along X")
    hdr['HIERARCH ESO DET BINY'] = (1, "Setup binning factor along Y")
    hdr['HIERARCH ESO DET READ CURID']  =  (1, "Used readout mode id")
    hdr['HIERARCH ESO DET READ CURNAME'] = ('SCI1.0', "Used readout mode name")
    hdr['HIERARCH ESO INS MODE'] = ('BMUSE', 'BlueMUSE mode')

    hdr['COMMENT'] = 'New dispersion and poly degree'
    hdr['INSTRUME'] = ('BlueMUSE  ', "simulated BlueMUSE data")
    hdr['DATE-OBS'] = (date_obs, 'observing date')

    if name != 'wavemap':
        if name == 'bias':
            hdr['EXPTIME'] = (0.0, '[s] exposure time')
        else:
            hdr['EXPTIME'] = (1.0, '[s] exposure time')   

    # EXTENSION header
    hdx = fits.Header()

    # If the data is spectra (arc line or full specta)
    if img[1] != None:
        hdr['METHOD'] = "{}".format(img[1])
        hdx['OBJECT'] = "{} of {}".format(img[0], name)
        hdr['HIERARCH ARC LAMP'] = "{}".format(name)

        # If there is full spec (not arc lines):
        # if spec:
        #     hdr['SOURCE'] = (other_info[5], "Source of spectra")
        # else:
        #     hdr['SOURCE'] = (other_info[5], "Source of spectra")

    else:
        hdx['OBJECT'] = "{}".format(img[0])

    # Get other info 
    other_info = img[2]

    hdr['SOURCE'] = (other_info[5], "Source of spectra")
    hdr['HIERARCH DEGREE SLICE TRACING'] = (other_info[0], '1D poly degree for slice tracing, x = f(y)')
    hdr['HIERARCH DEGREE WAVE CALIB 1'] = (other_info[1], '2D poly degree for wavelength calibration, horizontal')
    hdr['HIERARCH DEGREE WAVE CALIB 2'] = (other_info[2], '2D poly degree for wavelength calibration, vertical')

    # Write Primary HDU into empty header
    empty_hdr = fits.PrimaryHDU(header=hdr)

    # Entire hdx list
    hdx['EXTNAME'] = ('CHAN01  ', "Extension name")
    hdx['INHERIT'] =  ('T', 'Denotes the INHERIT keyword convention')
    hdx['HIERARCH ESO DET CHIP DATE']   = (ut.fits, "[YYYY-MM-DD] Date of")
    hdx['HIERARCH ESO DET CHIP ID']     = ('Cinta', "Detector chip identification")
    hdx['HIERARCH ESO DET CHIP INDEX']  = (1, 'Chip index')
    hdx['HIERARCH ESO DET CHIP LIVE']   = ('T', 'Detector alive')
    hdx['HIERARCH ESO DET CHIP NAME']   = ('CHAN01  ', 'Detector chip name')
    hdx['HIERARCH ESO DET CHIP NX']     = (4096, 'Physical active pixels in X')
    hdx['HIERARCH ESO DET CHIP NY']     = (4112, 'Physical active pixels in Y')
    hdx['HIERARCH ESO DET CHIP OVSCX']  = (0, 'Physical overscan pixels in X')
    hdx['HIERARCH ESO DET CHIP OVSCY']  = (0, 'Physical overscan pixels in Y')
    hdx['HIERARCH ESO DET CHIP PRSCX']  = (32, 'Physical prescan pixels in X')
    hdx['HIERARCH ESO DET CHIP PRSCY']  = (32, 'Physical prescan pixels in Y')
    hdx['HIERARCH ESO DET CHIP PSZX']   = (15.0, '[um] Size of pixel in X')
    hdx['HIERARCH ESO DET CHIP PSZY']   = (15.0, '[um] Size of pixel in Y')
    hdx['HIERARCH ESO DET CHIP X']      = (1, 'X location in array')
    hdx['HIERARCH ESO DET CHIP Y']      = (1, 'Y location in array')
    hdx['HIERARCH ESO DET OUT1 CONAD']  = (1.0, '[e-/ADU] Conversion ADUs to electr')
    hdx['HIERARCH ESO DET OUT1 GAIN']   = (1.0, '[ADU/e-] Conversion electrons to A')
    hdx['HIERARCH ESO DET OUT1 ID']     = ('E', 'Output ID as from manufacturer')
    hdx['HIERARCH ESO DET OUT1 INDEX']  = (1, 'Output index')
    hdx['HIERARCH ESO DET OUT1 NAME']   = ('NO1', 'Description of output')
    hdx['HIERARCH ESO DET OUT1 NX']     = (2048, 'Valid pixels along X')
    hdx['HIERARCH ESO DET OUT1 NY']     = (2056, 'Valid pixels along Y')
    hdx['HIERARCH ESO DET OUT1 OVSCX']  = (32, 'Overscan region in X')
    hdx['HIERARCH ESO DET OUT1 OVSCY']  = (32, 'Overscan region in Y')
    hdx['HIERARCH ESO DET OUT1 PRSCX']  = (32, 'Prescan region in X')
    hdx['HIERARCH ESO DET OUT1 PRSCY']  = (32, 'Prescan region in Y')
    hdx['HIERARCH ESO DET OUT1 RON']    = (other_info[3]-10, '[e-] Readout noise per output')
    hdx['HIERARCH ESO DET OUT1 X']      = (1, 'X location of output')
    hdx['HIERARCH ESO DET OUT1 Y']      = (1, 'Y location of output')
    hdx['HIERARCH ESO DET OUT2 CONAD']  = (1.0, '[e-/ADU] Conversion ADUs to electr')
    hdx['HIERARCH ESO DET OUT2 GAIN']   = (1.0, '[ADU/e-] Conversion electrons to A')
    hdx['HIERARCH ESO DET OUT2 ID']     = ('F', 'Output ID as from manufacturer')
    hdx['HIERARCH ESO DET OUT2 INDEX']  = (2, 'Output index')
    hdx['HIERARCH ESO DET OUT2 NAME']   = ('NO2', 'Description of output')
    hdx['HIERARCH ESO DET OUT2 NX']     = (2048, 'Valid pixels along X')
    hdx['HIERARCH ESO DET OUT2 NY']     = (2056, 'Valid pixels along Y')
    hdx['HIERARCH ESO DET OUT2 OVSCX']  = (32, 'Overscan region in X')
    hdx['HIERARCH ESO DET OUT2 OVSCY']  = (32, 'Overscan region in Y')
    hdx['HIERARCH ESO DET OUT2 PRSCX']  = (32, 'Prescan region in X')
    hdx['HIERARCH ESO DET OUT2 PRSCY']  = (32, 'Prescan region in Y')
    hdx['HIERARCH ESO DET OUT2 RON']    = (other_info[3], '[e-] Readout noise per output')
    hdx['HIERARCH ESO DET OUT2 X']      = (4096, 'X location of output')
    hdx['HIERARCH ESO DET OUT2 Y']      = (1, 'Y location of output')
    hdx['HIERARCH ESO DET OUT3 CONAD']  = (1.0, '[e-/ADU] Conversion ADUs to electr')
    hdx['HIERARCH ESO DET OUT3 GAIN']   = (1.0, '[ADU/e-] Conversion electrons to A')
    hdx['HIERARCH ESO DET OUT3 ID']     = ('G', 'Output ID as from manufacturer')
    hdx['HIERARCH ESO DET OUT3 INDEX']  = (3, 'Output index')
    hdx['HIERARCH ESO DET OUT3 NAME']   = ('NO3', 'Description of output')
    hdx['HIERARCH ESO DET OUT3 NX']     = (2048, 'Valid pixels along X')
    hdx['HIERARCH ESO DET OUT3 NY']     = (2056, 'Valid pixels along Y')
    hdx['HIERARCH ESO DET OUT3 OVSCX']  = (32, 'Overscan region in X')
    hdx['HIERARCH ESO DET OUT3 OVSCY']  = (32, 'Overscan region in Y')
    hdx['HIERARCH ESO DET OUT3 PRSCX']  = (32, 'Prescan region in X')
    hdx['HIERARCH ESO DET OUT3 PRSCY']  = (32, 'Prescan region in Y')
    hdx['HIERARCH ESO DET OUT3 RON']    = (other_info[3]+10, '[e-] Readout noise per output')
    hdx['HIERARCH ESO DET OUT3 X']      = (4096, 'X location of output')
    hdx['HIERARCH ESO DET OUT3 Y']      = (4112, 'Y location of output')
    hdx['HIERARCH ESO DET OUT4 CONAD']  = (1.0, '[e-/ADU] Conversion ADUs to electr')
    hdx['HIERARCH ESO DET OUT4 GAIN']   = (1.0, '[ADU/e-] Conversion electrons to A')
    hdx['HIERARCH ESO DET OUT4 ID']     = ('H', 'Output ID as from manufacturer')
    hdx['HIERARCH ESO DET OUT4 INDEX']  = (4, 'Output index')
    hdx['HIERARCH ESO DET OUT4 NAME']   = ('NO4', 'Description of output')
    hdx['HIERARCH ESO DET OUT4 NX']     = (2048, 'Valid pixels along X')
    hdx['HIERARCH ESO DET OUT4 NY']     = (2056, 'Valid pixels along Y')
    hdx['HIERARCH ESO DET OUT4 OVSCX']  = (32, 'Overscan region in X')
    hdx['HIERARCH ESO DET OUT4 OVSCY']  = (32, 'Overscan region in Y')
    hdx['HIERARCH ESO DET OUT4 PRSCX']  = (32, 'Prescan region in X')
    hdx['HIERARCH ESO DET OUT4 PRSCY']  = (32, 'Prescan region in Y')
    hdx['HIERARCH ESO DET OUT4 RON']    = (other_info[3]+20, '[e-] Readout noise per output')
    hdx['HIERARCH ESO DET OUT4 X']      = (1, 'X location of output')
    hdx['HIERARCH ESO DET OUT4 Y']      = (4112, 'Y location of output')

    # Main HDU
    hdu_img = fits.ImageHDU(data=data, header=hdx)

    # HDU list
    hdul = fits.HDUList([empty_hdr, hdu_img])

    # Export
    if img[1] != None:
        if other_info[4] == 'Pen-Ray':
            hdul.writeto('output/files_from_py/{}_{}_PR_img.fits'.format(img[3], name, overwrite=True))
        else:
            hdul.writeto('output/files_from_py/{}_{}_4MOST_img.fits'.format(img[3], name, overwrite=True))
    else:
        hdul.writeto('output/files_from_py/{}_img.fits'.format(name), overwrite=True)

# --------------------------------------------

def get_edges(allpix):

    """
    Class to check existence of overlapping slices.

    Parameters
    ----------
    allpix : 2D array
         2D array of the BlueMUSE image with slice

    Returns
    -------
    edges : list of lists
        places where edges are
    
    """
    
    # Initialize edges
    edges = []

    # Loop over the columns
    for i in range(4100):

        # Initialize y edges
        y_edges = []

        # Get y position of one column/slice
        y_array = allpix[i,:]
        
        # Loop over the rows in the columns
        for j in range(len(y_array)-1):

            # Print the x position where it is the edges, left and right
            if ((y_array[j] == 0) and (y_array[j+1] != 0)) or ((y_array[j] != 0) and (y_array[j+1] == 0)):
                
                # Append the value into an array
                y_edges.append(j)
        
        # Append the array into another array that stores everything
        edges.append(y_edges)
    
    return edges

# --------------------------------------------

def check_separation(control_pix, allpix):

    """
    Checks for the presence of overlapping slices. This program also identifies
    if the smallest gap between each slice. 

    Parameters
    ----------
    control_pix : 2D array
                BlueMUSE image with small slice width as control
    all_pix     : 2D array
                BlueMUSE image with slice width that wants to be evaluated

    """

    # Find the edges location in control_pix
    control_edges = get_edges(control_pix)

    # Find the edges location in all_pix
    edges = get_edges(allpix)

    # Initialize minimum separation
    min_sep = 1000

    # Initialize overlapping status
    overlap_detected = False

    # Loop over all edges in all_pix
    for i in range(len(edges)):

        # If there is edges in one row
        if any(edges[i]):

            # Loop for every row
            for j in range(len(edges[i])-1):

                # Calculate the gap between each edges, from one slice to the other
                gap = edges[i][j+1] - edges[i][j]

                # If it's not the same with the control edges, that means there is overlapping
                if len(edges[i]) != control_edges[i]:

                    print('overlapping slices!')
                    overlap_detected = True
                    
                    break 

                # Otherwise there is no overlapping
                else:

                    # If the gap is less than minimum separation, save it as minimum now
                    if gap < min_sep:

                        min_sep = gap
                        xpos = edges[i][j]
                        ypos = i
        
        if overlap_detected:
            break

    if overlap_detected == False:
        print('Smallest gap between neighbouring slices: {} pix'.format(min_sep))
        print('x position: ', xpos)
        print('y position: ', ypos)    

# --------------------------------------------

def combine_images(imglist):

    """
    Function to combine images to make realistic noice.

    make_realistic_noise generates noise for 4 quadrants of the BlueMUSE image
    separately.

    This program combines the 4 quadrants into one.

    Parameters
    ----------
    imglist : list of 2D array
            list containing 4 quadrants

    Returns
    ----------
    mergelist : 2D array
            combined 4 quadrants into one BlueMUSE image

    """

    # Combine q1 and q3
    uplist = np.concatenate((imglist[0], imglist[1]), axis=1)

    # Combine q2 and q4
    downlist = np.concatenate((imglist[2], imglist[3]), axis=1)

    # Combine them all
    mergelist = np.concatenate((uplist, downlist), axis=0)

    return mergelist

# --------------------------------------------

def make_realistic_noise(size, ovsc, dict, array=None, no_bias=False, just_bias=False):

    """
    Function to make realistic noise for different images.

    Parameters
    ----------
    size    : int, 2-tuple
            size of x and y
    ovsc    : int
            size of overscan
    dict    : dict
            dictionary for info on quadrants
    array   : array
            image that want to be given noise
    no_bias : bool
            adds just the quadrant without the bias
    just_bias : bool
            generate just the bias

    Returns
    ----------
    bias_img : 2D array
            bias image with quadrant noise
    raw_img  : 2D array
            image with quadrant bias noise
    
    """

    # Get the half of x-axis size and y-axis
    half_x = int(size[0]/2)
    half_y = int(size[1]/2)

    # Initialize array
    biasimg = []
    rawimg = []

    # Loop over dictionary (4 quadrants with different parameters) 
    for key, l in dict.items():

        # Make random noise
        bias = np.random.normal(loc=l[4], scale=l[5], size=[half_y+(ovsc*2),
                                                            half_x+(ovsc*2)])
        
        # If just bias image and nothing else
        if just_bias:
            biasimg.append(bias)

        # Otherwise generate 4 quadrants 
        else:

            # Get 4 quadrants of the overlaying image (e.g., flat, arc etc)
            quardx = array[l[0]:l[1],l[2]:l[3]].copy()

            # Make an array of quadrants
            raw = np.zeros((half_y+(ovsc*2), half_x+(ovsc*2)))

            # Add the raw into the quadrant
            raw[ovsc:half_y+ovsc, ovsc:half_x+ovsc] = quardx

            if no_bias:
                rawimg.append(raw)

            else:
                # Add bias to raw
                raw = raw + bias
                rawimg.append(raw)
    
    # If just bias combines the bias image
    if just_bias:
        bias_img = combine_images(biasimg)
        return bias_img

    # Else, add the raw to the bias
    else:
        raw_img = combine_images(rawimg)
        return raw_img

# ------------------------------------------------------------

class generate_images():
    """
    Class to simulate a number of BlueMUSE calibration images.
    
    """

    def __init__(self, pix_disp, size, overscan, FWHM, scale_amp, gaus_width, 
                 flat_flux, bias_flux, arc_lines=None, line_name=None, spec=False, 
                 full_spec=None, source=None, noise=False):

        self.pix_disp = pix_disp                # Get the dispersion dataframe
        self.arc_lines = arc_lines              # Arc line lists
        self.line_name = line_name              # Name of element for arc lines
        self.source = source                    # Source spectra

        self.size = size                        # A tuple of matrix size: size_x and size_y
        self.half_x = int(self.size[0]/2)       # Determine half of size x
        self.half_y = int(self.size[1]/2)       # Determine half of size y
        # self.slice_width = slice_width        # Slice width
        self.overscan = overscan                # Overscan width

        self.FWHM = FWHM                        # Determined FWHM flux for each gaussian flux for arc lines. In angstrom
        self.scale_amp = scale_amp              # Scaling amplitude. Arc line lists has normalized relative amplitude
        self.gaus_width = gaus_width            # Width of gaussian line for arc lines

        self.flat_flux = flat_flux              # Flux for slices in flat images
        self.bias_flux = bias_flux              # Flux for bias images

        self.trace_degree = 8                   # Polynomial degree for slice tracing
        self.wavecal_degree_1 = 2               # the 1st degree for 2D polynomial 
        self.wavecal_degree_2 = 11              # the 2nd degree for 2D polynomial

        self.std = self.FWHM / 2.355            # Standard deviation derived from FWHM, in angstrom
        # self.half = int(self.slice_width / 2)   # Half of slice width

        self.noise = noise                      # If noise is going to be generated
        self.spec = spec                        # If there is full-spectrum (not just arc lines)

        if self.spec:
            self.full_spec = full_spec          # Full spectra, if want to convert the spectra
            self.full_spec = self.full_spec.sort_values(by='lambda', ascending=True)    # Sort spectra by lambda
            self.full_spec['flux'] = self.full_spec['flux'] / max(self.full_spec['flux'])

            self.specpix = np.zeros((self.size[1], self.size[0]), dtype=np.float32)     # Array to store

        # Initialize array for different wavelength calibration images
        # For flat image, initially an array of zeros with a size of size_x (self.size[0]) and size_y (self.size[1])
        self.allpix = np.zeros((self.size[1], self.size[0]), dtype=np.float64)
        self.wvlpix = np.empty((self.size[1], self.size[0]), dtype=np.float64) + np.nan
        self.arcpix = np.zeros((self.size[1], self.size[0]), dtype=np.float64) 
        self.biaspix = np.empty((self.size[1], self.size[0]), dtype=np.float64) + 1000

        # Create dictionary of image types and where they're being saved
        other_info_arc = [self.trace_degree, self.wavecal_degree_1, self.wavecal_degree_2,
                      self.bias_flux, self.flat_flux, self.source]
        
        # List of other_info for non-arc images
        other_info = [self.trace_degree, self.wavecal_degree_1, self.wavecal_degree_2,
                      self.bias_flux, self.flat_flux, 'BlueMUSE']
        
        # Dictionary for information regarding different images
        self.img_dict = {
                        'flat': ["Flat image", None, other_info],
                        'bias': ["Bias image", None, other_info],
                        'wavemap': ["Wavelength map", None, other_info],
                        'arc': ["Arc exposure", "Line identification", other_info_arc, "arc"],
                        'spec': ["Full spectrum", "Interpolation", other_info_arc, "spec"]
                        }
        
        # Create dictionary for separating four quadrants
        self.quards = {'q1': [0, self.half_y, 0, self.half_x, self.bias_flux-10, 2.5],
                       'q2': [0, self.half_y, self.half_x, self.size[0], self.bias_flux, 2.3],
                       'q3': [self.half_y, self.size[1], 0, self.half_x, self.bias_flux+10, 2.8],
                       'q4': [self.half_y, self.size[1], self.half_x, self.size[0], self.bias_flux+20, 3.0]}

        # # Get gaussian peaks for arc spectra, if gaussian peaks are not yet given
        # self.arc_lines = get_gaussian_peaks(self.arc_lines, self.arc_spectra)  

        # # Get normalized relative ampltiude for each arc lines
        # self.arc_lines['norm_amp'] = self.arc_lines['amplitudes'] / self.arc_lines['amplitudes'].max()  # Get
    
    # --------------------------------------------
    
    def generate_pix_disp(self):

        """
        Generate a dataframe of the BlueMUSE dispersion table. The values here at the position of each wavelength 
        point at the center of the slice. 

        Parameters
        ----------
        pix_disp    : pandas DataFrame
                    BlueMUSE dispersion in mm

        Returns
        ----------
        pix_disp    : pandas DataFrame
                    BlueMUSE dispersion in pix
        trace_coef_list : list [48x3]
                        list of the tracing coefficients on the left edge,
                        center, and right edge of every slice
        wavecal_coef_list : list [48]
                        list of the 2D polynomial wavelength calibration
                        coefficients for 48 slices
        """

        # Transform to pix
        self.pix_disp = transform_to_pix(self.pix_disp)

        # Initialize coefficient list
        self.trace_coef_list = []  # 48 x 3
        self.wavecal_coef_list = [] # 48

        # Loop for every slice
        for j in range(48):

            # To indicate slice number
            n = j + 1

            # Loop for center, left edge, right edge
            for m in range(3):

                # For one slice and FIE (0 -> center, 1 -> slice right, 2 -> slice left)
                pix_disp_fie_in = self.pix_disp[(self.pix_disp['CONF'] == n) & 
                                        (self.pix_disp['FIE'] == m+1)].copy().dropna()

                # Get X, Y, and WVL
                X = pix_disp_fie_in['CX_DET'].values
                Y = pix_disp_fie_in['CY_DET'].values
                WVL = pix_disp_fie_in['WVL'].values

                # Slice tracing
                # ----------------
                # Get 1D polynomial coefficients
                trace_coef = np.polynomial.polynomial.polyfit(Y, X, self.trace_degree)
                self.trace_coef_list.append(trace_coef)

            # For one slice and FIE (0 -> center, 1 -> slice right, 2 -> slice left)
            pix_disp_fie = self.pix_disp[(self.pix_disp['CONF'] == n)].copy().dropna()

            # Get X, Y, and WVL of entire slice without differentiating center, left, and right
            X = pix_disp_fie['CX_DET'].values
            Y = pix_disp_fie['CY_DET'].values
            WVL = pix_disp_fie['WVL'].values

            # Wavelength of each position
            # ----------------
            # Get 2D polynomial coefficients, wvl = f(x, y)
            wavecal_coef = polyfit2d(X, Y, WVL, degree=[self.wavecal_degree_1, self.wavecal_degree_2])
            self.wavecal_coef_list.append(wavecal_coef)
    
    # --------------------------------------------

    def make_flat_wavemap_bias(self, FITS=False, just_array=False):

        """
        Generate simulated images of bias, flat, and wavemap

        Parameters
        ----------
        FITS    : bool
                generate FITS image or not
        just_array : bool
                print just the flat array

        Returns
        ----------
        allpix  : 2D array
                flat image
        biaspix : 2D array
                bias image
        wvlpix  : 2D array
                wavemap image
        
        """

        # Loop for every slice
        for j in range(48):
            
            # To count from 1
            n = j + 1

            # Create integers arrays of pixels from Y values.
            # For example, if the slice goes from y_pix = 25.6 to 78.1, I make an array of 
            # y pixels from 26 to 78. This will be the length of each slice.
            yarray = np.arange(0, self.size[1])

            # Find the left and right edge of each row 
            xright = get_poly_1d(yarray, self.trace_coef_list[(j*3)+1])
            xleft = get_poly_1d(yarray, self.trace_coef_list[(j*3)+2])        

            # Convert to integers
            yarray = yarray.astype(int)
            xright = xright.astype(int)
            xleft = xleft.astype(int)

            # Loop for every row
            for i in range(len(yarray)):

                # For each y position, determine the range of slice
                x_right = xright[i]
                x_left = xleft[i]

                # Make an array of x pixels each row
                xxx = np.arange(x_left, x_right)

                # For every x position 
                for x in xxx:

                    # Flat image, replaces values in slice range with flat_flux
                    self.allpix[yarray[i], x] = 10000

                    # Wavemap, replaces values in slice range with wavelength, wvl = f(x_center, y_center)
                    # Each slice strip for one row has the same wavelength value
                    self.wvlpix[yarray[i], x] = np.polynomial.polynomial.polyval2d(x, yarray[i], 
                                                                                 self.wavecal_coef_list[j])

                # Add noise if noise is desired
                if self.noise:

                    # Add photon noise
                    for k in range(x_right-x_left):

                        # Get array position
                        m = k + x_left
                        old_value = self.allpix[yarray[i], m]

                        # Random generator with gaussian N(old_value, sqrt(old_value))
                        mu, sigma = old_value, np.sqrt(old_value)
                        new_value = np.random.normal(mu, sigma, 1)

                        # Replace value with noise + value
                        self.allpix[yarray[i], m] = new_value[0]
    
            print('Slice {} done'.format(n),end='\r')
        
        # If no noise and want to generate FITS
        if (self.noise == False) and FITS:

            self.allpix = self.allpix.astype('int16')
            self.allpix = self.biaspix.astype('int16')

            write_FITS(self.allpix + self.bias_flux, self.img_dict['flat'], 'flat', self.spec)
            print('FITS for flat done')

            write_FITS(self.wvlpix, self.img_dict['wavemap'], 'wavemap', self.spec)
            print('FITS for wavemap done')

            write_FITS(self.biaspix, self.img_dict['bias'], 'bias', self.spec)
            print('FITS for bias done')
        
        # If noise
        if self.noise:

            # Make realistic noise for flat
            self.allpix = make_realistic_noise(self.size, self.overscan, self.quards, 
                                               self.allpix)

            # For bias
            self.biaspix = make_realistic_noise(self.size, self.overscan, self.quards, 
                                                just_bias=True)
            
            # Generate FITS for image with noise
            if FITS:
                self.allpix = self.allpix.astype('int16')
                self.biaspix = self.biaspix.astype('int16')

                write_FITS(self.allpix, self.img_dict['flat'], 'flat', self.spec)
                print('FITS for flat done')

                write_FITS(self.wvlpix, self.img_dict['wavemap'], 'wavemap', self.spec)
                print('FITS for wavemap done')

                write_FITS(self.biaspix, self.img_dict['bias'], 'bias', self.spec)
                print('FITS for bias done')
        
        if just_array:
            return self.allpix
    
    # --------------------------------------------
    
    def make_arc(self, FITS=False):

        """
        Class to make arc images.

        Parameters
        ----------
        FITS    : bool
                generate FITS image or not

        Returns
        ----------
        arcpix : 2D array
                arc image

        """

        # Loop for every slice
        for j in range(48):

            # Add one
            n = j + 1
            
            # Make int y array (all rows per slice)
            yarray = np.arange(0, self.size[1])

            # Find the center, left edge, and right edge of every row
            xcenter = get_poly_1d(yarray, self.trace_coef_list[(j*3)])
            xright = get_poly_1d(yarray, self.trace_coef_list[(j*3)+1])
            xleft = get_poly_1d(yarray, self.trace_coef_list[(j*3)+2])

            # Find the maximum x right
            av_right = int(np.max(xright))

            # Find the minimum x left
            av_left = int(np.max(xleft))

            # Make an array of x position in each row with 4 pixel buffer
            xxx = np.arange(av_left-2,av_right+2, dtype=int)

            # Determine wavelength for every y position           
            WVL = np.polynomial.polynomial.polyval2d(xcenter, yarray, self.wavecal_coef_list[j])

            # if (n % 10 == 0):
            #     print(wvl_array)
            #     print(WVL)
            #     print(Y)
            #     print(y_array)
            #     print('---')
            
            # For arc image
            # ---------------------------
            # Loop over each wavelength in arc line lists
            for idx in range(len(self.arc_lines['lambda'])):	

                # Get each lambda
                lam = self.arc_lines['lambda'][idx]

                # Only count wavelength inside the WVL
                if (lam >= min(WVL)) and (lam <= max(WVL)):

                    # For every x position in each row
                    for x in xxx:
                
                        # Get the x and y solution (sol) for each wavelength.
                        # Also the resolution, if res = True. 
                        sol, res = solve_for_y_wvl(self.wavecal_coef_list[j], wvl_target=lam, x_fixed=x,
                                                initial_guess=yarray[15], reso=True)

                        # Get y_center for the wavelength
                        y_wvl = sol[0]

                        # Get the right and left edge from y_wvl
                        xr_wvl = int(get_poly_1d(y_wvl, self.trace_coef_list[(j*3)+1]))
                        xl_wvl = int(get_poly_1d(y_wvl, self.trace_coef_list[(j*3)+2]))

                        # If x is within the range
                        if (x >= xl_wvl) and (x <= xr_wvl):

                            # if (n % 10 == 0):
                            #     print(y_wvl)
                            #     print('---')

                            # Create a pixel array with y_wvl in the center.
                            y_down = int(y_wvl - self.gaus_width)
                            y_up = int(y_wvl + self.gaus_width)
                            y = np.arange(y_down, y_up)

                            # Ampltiude of each line is scaled for the image
                            amp = self.arc_lines['norm_amp'][idx] * self.scale_amp

                            # Standard deviation in pixel. Std from FWHM (in angstrom) * resolution (pix/angstrom)
                            std = self.std * res

                            # Gaussian model with position y_wvl and standard deviation std
                            gaussian = Gaussian1D(amplitude=amp, mean=y_wvl, stddev=std)
                    
                            # Spread the gaussian values over array of y (y_down:y_up, with y_wvl in the center)
                            gaussian_values = gaussian(y)

                            # Distribute gaussian values for all x positions
                            if (y_up <= self.size[1]) and (y_down >= 0):

                                # Put the gaussian fit to the arc image
                                self.arcpix[y_down:y_up,x] = gaussian_values

                                # Add noise
                                if self.noise:

                                    # Add photon noise
                                    for k in range(y_up-y_down):

                                        # Get array position
                                        m = k + y_down
                                        old_value = self.arcpix[m,x]

                                        # Random generator with gaussian N(old_value, sqrt(old_value))
                                        mu, sigma = old_value, np.sqrt(old_value)
                                        new_value = np.random.normal(mu, sigma, 1)

                                        self.arcpix[m,x] = new_value[0]
                
            print('Slice {} done'.format(n),end='\r')

        # If no noise and generate FITS
        if (self.noise == False) and FITS:

            self.arcpix = self.arcpix.astype('int16')
            self.bias_flux = self.bias_flux.astype('int16')

            write_FITS(self.arcpix + self.bias_flux, self.img_dict['arc'], '{}'.format(self.line_name), self.spec)
            print('FITS for arc done')
        
        # If noise
        if self.noise:

            self.arcpix = make_realistic_noise(self.size, self.overscan, self.quards, 
                                               self.arcpix)

            if FITS:

                self.arcpix = self.arcpix.astype('int16')

                write_FITS(self.arcpix, self.img_dict['arc'], '{}'.format(self.line_name), self.spec)
                print('FITS for arc done')
    
    # --------------------------------------------

    # --- uncomment this if there is full spectra ---
    
    # def make_arc_from_full_spec(self, FITS=False):

    #     """
    #     Class to make arc images from full spectrum.

    #     Parameters
    #     ----------
    #     FITS    : bool
    #             generate FITS image or not

    #     Returns
    #     ----------
    #     specpix : 2D array
    #             arc image with full spectrum not just arc lines

    #     """

    #     # Loop for every slice
    #     for j in range(48):

    #         n = j + 1

    #         # yarray = np.arange(np.round(min(Y)), np.round(max(Y)))
    #         Y = np.arange(0, self.size[1])

    #         # Determine wavelength for every y position           
    #         WVL = get_poly_1d(Y, self.wavecal_coef_list[j])	
            
    #         # For full spectra image
    #         # ---------------------------
    #         # Get only within the wavelength of each slice
    #         spec = self.full_spec[(self.full_spec['lambda'] >= min(WVL)) & (self.full_spec['lambda'] <= max(WVL))].copy()

    #         # Get y_pix for all wavelength
    #         y_pixes = []

    #         for lam in spec['lambda'].values:

    #             sol = solve_for_y_wvl(self.wavecal_coef_list[j], lam, 
    #                                 initial_guess=Y[0], reso=False)[0]
    #             y_pixes.append(sol)
            
    #         spec['y_pix'] = y_pixes

    #         # Make an array of y_pix but all integers
    #         y_int = np.arange(np.round(min(spec['y_pix']))+1, np.round(max(spec['y_pix']))-1)

    #         # Interpolate to get flux for every integer y_pix
    #         fluxes = np.interp(y_int, spec['y_pix'], spec['flux'])

    #         # Create the arc image
    #         for idx in range(len(y_int)):

    #             y_wvl = int(y_int[idx])
    #             flux = fluxes[idx] * self.scale_amp

    #             x_wvl = get_poly_1d(y_wvl, self.trace_coef_list[j])
    #             x_wvl = int(x_wvl)

    #             # Determine slice position for y_wvl
    #             xleft = x_wvl - self.half
    #             xright = x_wvl + self.half

    #             self.specpix[y_wvl,xleft:xright] = flux

    #             if self.noise:

    #                 # Add photon noise
    #                 for k in range(xright-xleft):

    #                     # Get array position
    #                     m = k + xleft
    #                     old_value = self.specpix[y_wvl,m]

    #                     # Random generator with gaussian N(old_value, sqrt(old_value))
    #                     mu, sigma = old_value, np.sqrt(old_value)
    #                     new_value = np.random.normal(mu, sigma, 1)

    #                     self.specpix[y_wvl,m] = new_value[0]
            
    #         print('Slice {} done'.format(n),end='\r')

    #     if (self.noise == False) and FITS:
    #         write_FITS(self.specpix + self.bias_flux, self.img_dict['spec'], self.line_name, self.spec)
    #         print('FITS for spec done')
        
    #     if self.noise:

    #         self.specpix = make_realistic_noise(self.size, self.overscan, self.quards, 
    #                                            self.specpix)

    #         if FITS:
    #             write_FITS(self.specpix, self.img_dict['spec'], self.line_name, self.spec)
    #             print('FITS for spec done')

    
# ------------------------------------------------------------

if __name__ == "__main__":

    # Determine dispersion
    disp = pd.read_csv('data/BMUS_IFU-0236-1.53_disp.dat', sep=" ")
     
    # Load parameters from ASCII
    param = read_ascii_config('data/parameters_input_new.ascii')

    # Access parameters
    size_x = int(param['size_x'])
    size_y = int(param['size_y'])
    flat_flux = int(param['flat_flux'])
    bias_flux = int(param['bias_flux'])
    FWHM = float(param['FWHM'])
    scale_amp = int(param['scale_amp'])
    gaus_width = int(param['gaus_width'])
    overscan = 32
    noise = True
    size=[size_x, size_y]

    # Main dictionary for all files
    lines = {'Cd': [gaus_width],
            'Cs': [gaus_width],
            'He': [gaus_width],
            'Hg': [gaus_width],
            # 'HgCd': [],
            # 'HgCd-LLG300': [],
            'Zn': [gaus_width],
            'HgAr': [gaus_width],
            'Xe': [gaus_width],
            'FP': [2]
            }

    # Generate each images
    for key, l in lines.items():

        line_name = key
        arc_lines = pd.read_csv('data/lines_for_img/{}.csv'.format(key))

        gen = generate_images(pix_disp=disp, size=[size_x, size_y], overscan=overscan,
                            FWHM=FWHM, scale_amp=scale_amp, arc_lines=arc_lines,
                            line_name=line_name, gaus_width=l[0], 
                            flat_flux=flat_flux, bias_flux=bias_flux,noise=noise, source='4MOST')
        
        # Get pix dispersion
        gen.generate_pix_disp()

        if key == 'Cd':

            # First arc line gets bias, flat, and wvl
            gen.make_flat_wavemap_bias(FITS=True)

        # Make arc line images
        gen.make_arc(FITS=True)

        print('Arc {} done'.format(key))






