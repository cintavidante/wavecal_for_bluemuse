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
    Gaussian function.
    """

    lmfitter = LevMarLSQFitter()
    guess = Gaussian1D(amplitude=y.max(), mean=np.mean(x), stddev=5)
    fit = lmfitter(model=guess, x=x, y=y)

    return fit

# --------------------------------------------

def read_ascii_config(file_path):

    """
    Read from ASCII file
    
    """

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
    Returns a polynomial 1D function from a set of coefficients.

    """
    return np.polynomial.polynomial.polyval(x, coef)

# --------------------------------------------

def solve_equations(y_val, coef, x_fixed, wvl_target):
    """
    Two sets of equations to solve a 2D polynomial problem.
    'If z = f(x, y), and one has z, what would x and y be?

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
        return sol
    
# --------------------------------------------

def transform_to_pix(disp):

    pix_disp = disp.copy()

    pix_disp['CX_DET'] = ((pix_disp['CX_DET'].values * u.mm).to(u.um) / (15 * u.um)).value + 2048
    pix_disp['CY_DET'] = ((pix_disp['CY_DET'].values * u.mm).to(u.um) / (15 * u.um)).value + 2056

    return pix_disp

# --------------------------------------------

# def create_subset(pix_disp, n):

#     """
#     Create subset for each slice
#     """

#     # Make a subset for each slice. 
#     subset = pix_disp[['WVL', 'X_{}'.format(n), 'Y_{}'.format(n)]]

#     # Drops any rows that have NaN 
#     subset = subset.dropna()

#     # Convert pandas dataframe to numpy arrays
#     X = subset['X_{}'.format(n)].values
#     Y = subset['Y_{}'.format(n)].values
#     WVL = subset['WVL'].values

#     return X, Y, WVL

# --------------------------------------------

def write_FITS(data, img, name, spec):
    """
    Class to generate all FITS files
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

#   DATE-OBS= '2025-01-07T17:17:17.171' / observing date

    # hdr['DATE'] = (ut.fits, 'Current time in UTC')

    # EXTENSION header
    hdx = fits.Header()

    if img[1] != None:
        hdr['METHOD'] = "{}".format(img[1])
        hdx['OBJECT'] = "{} of {}".format(img[0], name)
        hdr['HIERARCH ARC LAMP'] = "{}".format(name)

        # if spec:
        #     hdr['SOURCE'] = (other_info[5], "Source of spectra")
        # else:
        #     hdr['SOURCE'] = (other_info[5], "Source of spectra")

    else:
        hdx['OBJECT'] = "{}".format(img[0])

    other_info = img[2]
    hdr['SOURCE'] = (other_info[5], "Source of spectra")
    hdr['HIERARCH DEGREE SLICE TRACING'] = (other_info[0], '1D poly degree for slice tracing, x = f(y)')
    hdr['HIERARCH DEGREE WAVE CALIB 1'] = (other_info[1], '2D poly degree for wavelength calibration, horizontal')
    hdr['HIERARCH DEGREE WAVE CALIB 2'] = (other_info[2], '2D poly degree for wavelength calibration, vertical')
    # hdr['HIERARCH SLICE WIDTH'] = (other_info[2], 'Width of each slice')
    # hdr['BIAS FLUX'] = (other_info[3], 'Flux for bias')

    empty_hdr = fits.PrimaryHDU(header=hdr)

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
            hdul.writeto('output/files_from_py/{}_{}_PR_img.fits'.format(name, img[3]), overwrite=True)
        else:
            hdul.writeto('output/files_from_py/{}_{}_4MOST_img.fits'.format(name, img[3]), overwrite=True)
    else:
        hdul.writeto('output/files_from_py/{}_img.fits'.format(name), overwrite=True)

# --------------------------------------------

def get_edges(allpix):
    """
    Class to check existence of overlapping slices.
    
    """
    
    edges = []

    for i in range(4100):

        y_edges = []

        y_array = allpix[i,:]
        
        # Loop the whole column
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

    control_edges = get_edges(control_pix)
    edges = get_edges(allpix)

    min_sep = 1000
    overlap_detected = False

    for i in range(len(edges)):

        if any(edges[i]):

            for j in range(len(edges[i])-1):

                gap = edges[i][j+1] - edges[i][j]

                if len(edges[i]) != control_edges[i]:

                    print('overlapping slices!')
                    overlap_detected = True
                    
                    break 

                else:
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

    uplist = np.concatenate((imglist[0], imglist[1]), axis=1)
    downlist = np.concatenate((imglist[2], imglist[3]), axis=1)

    mergelist = np.concatenate((uplist, downlist), axis=0)

    return mergelist

# --------------------------------------------

def make_realistic_noise(size, ovsc, dict, array=None, no_bias=False, just_bias=False):

    half_x = int(size[0]/2)
    half_y = int(size[1]/2)

    biasimg = []
    rawimg = []

    for key, l in dict.items():

        bias = np.random.normal(loc=l[4], scale=l[5], size=[half_y+(ovsc*2),
                                                            half_x+(ovsc*2)])
        
        if just_bias:
            biasimg.append(bias)
        else:
            quardx = array[l[0]:l[1],l[2]:l[3]].copy()
            raw = np.zeros((half_y+(ovsc*2), half_x+(ovsc*2)))

            raw[ovsc:half_y+ovsc, ovsc:half_x+ovsc] = quardx

            if no_bias:
                rawimg.append(raw)
            else:
                raw = raw + bias
                rawimg.append(raw)
    
    if just_bias:
        bias_img = combine_images(biasimg)
        return bias_img
    else:
        raw_img = combine_images(rawimg)
        return raw_img

# ------------------------------------------------------------

class generate_images():
    """
    Class to generate simulated BlueMUSE images.
    
    """

    def __init__(self, pix_disp, size, overscan, FWHM, scale_amp, gaus_width, 
                 flat_flux, bias_flux, arc_lines=None, line_name=None, spec=False, 
                 full_spec=None, source=None, noise=False):

        self.pix_disp = pix_disp                # Get the dispersion dataframe
        self.arc_lines = arc_lines              # Arc line lists
        self.line_name = line_name              # Name of element for arc lines
        self.source = source                    # Source spectra

        self.size = size                        # A tuple of matrix size: size_x and size_y
        self.half_x = int(self.size[0]/2)
        self.half_y = int(self.size[1]/2)
        # self.slice_width = slice_width        # Slice width
        self.overscan = overscan                # Overscan width

        self.FWHM = FWHM                        # Determined FWHM flux for each gaussian flux for arc lines. In angstrom
        self.scale_amp = scale_amp              # Scaling amplitude. Arc line lists has normalized relative amplitude
        self.gaus_width = gaus_width            # Width of gaussian line for arc lines

        self.flat_flux = flat_flux              # Flux for slices in flat images
        self.bias_flux = bias_flux              # Flux for bias images

        self.trace_degree = 8                   # Polynomial degree for slice tracing
        self.wavecal_degree_1 = 2                # Polynomial degree for wavelength calibration
        self.wavecal_degree_2 = 11

        self.std = self.FWHM / 2.355            # Standard deviation derived from FWHM, in angstrom
        # self.half = int(self.slice_width / 2)   # Half of slice width

        self.noise = noise
        self.spec = spec

        if self.spec:
            self.full_spec = full_spec          # Full spectra, if want to convert the spectra
            self.full_spec = self.full_spec.sort_values(by='lambda', ascending=True)    # Sort spectra by lambda
            self.full_spec['flux'] = self.full_spec['flux'] / max(self.full_spec['flux'])

            self.specpix = np.zeros((self.size[1], self.size[0]), dtype=np.float32)     # Array to store

        # For flat image, initially an array of zeros with a size of size_x (self.size[0]) and size_y (self.size[1])
        self.allpix = np.zeros((self.size[1], self.size[0]), dtype=np.float64)
        self.wvlpix = np.empty((self.size[1], self.size[0]), dtype=np.float64) + np.nan
        self.arcpix = np.zeros((self.size[1], self.size[0]), dtype=np.float64) 
        self.biaspix = np.empty((self.size[1], self.size[0]), dtype=np.float64) + 1000

        # Create dictionary of image types and where they're being saved
        other_info_arc = [self.trace_degree, self.wavecal_degree_1, self.wavecal_degree_2,
                      self.bias_flux, self.flat_flux, self.source]
        
        other_info = [self.trace_degree, self.wavecal_degree_1, self.wavecal_degree_2,
                      self.bias_flux, self.flat_flux, 'BlueMUSE']
        
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
        """

        # Transform to pix
        self.pix_disp = transform_to_pix(self.pix_disp)

        self.trace_coef_list = []  # 48 x 3
        self.wavecal_coef_list = [] # 48

        for j in range(48):

            # To indicate slice number
            n = j + 1

            for m in range(3):

                # For one slice and FIE (0 -> center, 1 -> slice right, 2 -> slice left)
                pix_disp_fie_in = self.pix_disp[(self.pix_disp['CONF'] == n) & 
                                        (self.pix_disp['FIE'] == m+1)].copy().dropna()

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

        for j in range(48):
            
            n = j + 1

            # Create integers arrays of pixels from Y values.
            # For example, if the slice goes from y_pix = 25.6 to 78.1, I make an array of 
            # y pixels from 26 to 78. This will be the length of each slice.
            
            # yarray = np.arange(np.round(min(Y)), np.round(max(Y)))
            yarray = np.arange(0, self.size[1])

            xright = get_poly_1d(yarray, self.trace_coef_list[(j*3)+1])
            xleft = get_poly_1d(yarray, self.trace_coef_list[(j*3)+2])

            # # Determine wavelength for every y position           
            # wavel = get_poly_1d(yarray, self.wavecal_coef_list[j])	        

            # Convert to integers
            yarray = yarray.astype(int)
            xright = xright.astype(int)
            xleft = xleft.astype(int)

            for i in range(len(yarray)):

                # For each y position, determine the range of slice
                x_right = xright[i]
                x_left = xleft[i]

                # To check whether the x position indeed returns an array with size slice_width
                xxx = np.arange(x_left, x_right)
                # if len(xxx) != slice_width:
                #     raise ValueError("Array doesn't match the selected slice width")

                for x in xxx:

                    # Flat image, replaces values in slice range with flat_flux
                    self.allpix[yarray[i], x] = 10000

                    # Wavemap, replaces values in slice range with wavelength, wvl = f(x_center, y_center)
                    # Each slice strip for one row has the same wavelength value
                    self.wvlpix[yarray[i], x] = np.polynomial.polynomial.polyval2d(x, yarray[i], 
                                                                                 self.wavecal_coef_list[j])

                if self.noise:

                    # Add photon noise
                    for k in range(x_right-x_left):

                        # Get array position
                        m = k + x_left
                        old_value = self.allpix[yarray[i], m]

                        # Random generator with gaussian N(old_value, sqrt(old_value))
                        mu, sigma = old_value, np.sqrt(old_value)
                        new_value = np.random.normal(mu, sigma, 1)

                        self.allpix[yarray[i], m] = new_value[0]
    
            print('Slice {} done'.format(n),end='\r')
        
        if (self.noise == False) and FITS:

            self.allpix = self.allpix.astype('int16')
            self.allpix = self.biaspix.astype('int16')

            write_FITS(self.allpix + self.bias_flux, self.img_dict['flat'], 'flat', self.spec)
            print('FITS for flat done')

            write_FITS(self.wvlpix, self.img_dict['wavemap'], 'wavemap', self.spec)
            print('FITS for wavemap done')

            write_FITS(self.biaspix, self.img_dict['bias'], 'bias', self.spec)
            print('FITS for bias done')
        
        if self.noise:

            self.allpix = make_realistic_noise(self.size, self.overscan, self.quards, 
                                               self.allpix)
            # self.wvlpix = make_realistic_noise(self.size, self.overscan, self.quards, 
            #                                    self.wvlpix, no_bias=True)
            self.biaspix = make_realistic_noise(self.size, self.overscan, self.quards, 
                                                just_bias=True)
            
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
        Class to make all images: flat, bias, wavemap, and arc lines. 

        """

        for j in range(48):

            n = j + 1
            
            # yarray = np.arange(np.round(min(Y)), np.round(max(Y)))
            yarray = np.arange(0, self.size[1])
            xcenter = get_poly_1d(yarray, self.trace_coef_list[(j*3)])
            xright = get_poly_1d(yarray, self.trace_coef_list[(j*3)+1])
            xleft = get_poly_1d(yarray, self.trace_coef_list[(j*3)+2])

            av_right = int(np.max(xright))
            av_left = int(np.max(xleft))

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

                if (lam >= min(WVL)) and (lam <= max(WVL)):

                    for x in xxx:
                
                        # Get the x and y solution (sol) for each wavelength.
                        # Also the resolution, if res = True. 
                        sol, res = solve_for_y_wvl(self.wavecal_coef_list[j], wvl_target=lam, x_fixed=x,
                                                initial_guess=yarray[15], reso=True)

                        # Get y_center for the wavelength
                        y_wvl = sol[0]
                        xr_wvl = int(get_poly_1d(y_wvl, self.trace_coef_list[(j*3)+1]))
                        xl_wvl = int(get_poly_1d(y_wvl, self.trace_coef_list[(j*3)+2]))

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

                                self.arcpix[y_down:y_up,x] = gaussian_values

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

        if (self.noise == False) and FITS:

            self.arcpix = self.arcpix.astype('int16')
            self.bias_flux = self.bias_flux.astype('int16')

            write_FITS(self.arcpix + self.bias_flux, self.img_dict['arc'], '{}'.format(self.line_name), self.spec)
            print('FITS for arc done')
        
        if self.noise:

            self.arcpix = make_realistic_noise(self.size, self.overscan, self.quards, 
                                               self.arcpix)

            if FITS:

                self.arcpix = self.arcpix.astype('int16')

                write_FITS(self.arcpix, self.img_dict['arc'], '{}'.format(self.line_name), self.spec)
                print('FITS for arc done')
    
    # --------------------------------------------
    
    # def make_arc_from_full_spec(self, FITS=False):

    #     """
    #     Class to make all images: flat, bias, wavemap, and arc lines. 

    #     """

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

    # Load files
    # disp = pd.read_csv('files/dispersion.dat', sep=" ")

    # arc_lines = pd.read_csv('files/fp_list.csv')
    # line_name = 'fabry_perot'
    
    # Load parameters from ASCII
    param = read_ascii_config('parameters_input_new.ascii')

    # Access parameters
    size_x = int(param['size_x'])
    size_y = int(param['size_y'])
    flat_flux = int(param['flat_flux'])
    bias_flux = int(param['bias_flux'])
    # slice_width = int(param['slice_width'])
    # slice_width = 76
    FWHM = float(param['FWHM'])
    scale_amp = int(param['scale_amp'])
    gaus_width = int(param['gaus_width'])
    overscan = 32
    noise = True

    # line_dict = {
            #  'ThAr_4MOST':[5, 'ThAr', '4MOST'],
            #  'HgNe_4MOST': [5, 'HgNe', '4MOST'],
            #  'Cd_PR': [5, 'Cd', 'G1'],
            #  'HgAr_PR': [5, 'HgAr', 'G1'],
            #  'HgNe_PR':[5, 'HgNe', 'G1'],
            #  'Kr_PR':[5, 'Kr', 'G1'],
            #  'Xe_PR':[5, 'Xe', 'G1'],
            #  'Zn_PR':[5, 'Zn', 'G1'],
            # #  'Ar_PR':[5, 'Ar', 'G2'],
            #  'Ne_PR':[5, 'Ne', 'G2'],
            #  'Fabry-Perot_4MOST':[2, 'Fabry-Perot', None]}

    # -----------------------------------------------

    # line_4most = {'Cd_4MOST': [5, 'Cd', '4MOST'],
    #               'Cs_4MOST': [5, 'Cs', '4MOST'],
    #               'He_4MOST': [5, 'He', '4MOST'],
    #               'Hg_4MOST': [5, 'Hg', '4MOST'],
    #               'Zn_4MOST': [5, 'Zn', '4MOST']
    #             }
    
    # for key, l in line_4most.items():

    #     line_name = l[1]

    #     print('generating images for {}'.format(key))

    #     disp = pd.read_csv('files/new_dispersion.dat', sep=" ")

    #     # full_spec_line = pd.read_csv('files/4most-new/spectra/spectra_{}_blue.csv'.format(l[1]), sep=",")
    #     spec = False
    #     full_spec_line = False
    #     arc_lines = pd.read_csv('files/4most-new/{}_list_new.csv'.format(l[1]))
    #     source = '4MOST'

    #     gen = generate_images(pix_disp=disp, size=[size_x, size_y], overscan=overscan, 
    #                      FWHM=FWHM, scale_amp=scale_amp, 
    #                     gaus_width=int(l[0]), flat_flux=flat_flux, bias_flux=bias_flux,
    #                     arc_lines=arc_lines, line_name=line_name, spec=spec,
    #                     full_spec=full_spec_line, source=source, noise=noise)

    #     gen.generate_pix_disp()

    #     if l[1] == 'Cd':
    #         gen.make_flat_wavemap_bias(FITS=True)
        
    #     gen.make_flat_wavemap_bias()
    #     gen.make_arc(FITS=True)
        
    #     print('all images for {} done'.format(key))
    #     print('----')
        
    # -----------------------------------------------
    
    # for key, l in line_dict.items():

    #     line_name = l[1]

    #     print('generating images for {}'.format(key))

    #     disp = pd.read_csv('files/new_dispersion.dat', sep=" ")

    #     if l[2] != None:

    #         spec = True
    #         full_spec_name = 'Pen-Ray-spectra/{}_{}.asc'.format(l[1], l[2])

    #         if (l[2] == 'G1') or (l[2] == 'G2'):

    #             full_spec_line = pd.read_csv(full_spec_name, sep=",", names=['lambda', 'flux'])
    #             full_spec_line['lambda'] = full_spec_line['lambda'] * 10
    #             arc_lines = pd.read_csv('files/{}_list.csv'.format(l[1]))
    #             source = 'Pen-Ray'

    #         else:

                # full_spec_line = pd.read_csv(full_spec_name, sep=",")
                # arc_lines = pd.read_csv('files/{}_4MOST_list.csv'.format(l[1]))
        #       # source = '4MOST'

        # else:
        #     spec = False
        #     full_spec_line = None
        # #     arc_lines = pd.read_csv('files/{}_list.csv'.format(l[1]))
        #     source = '4MOST'
        
        # gen = generate_images(pix_disp=disp, size=[size_x, size_y], overscan=overscan, 
        #                       slice_width=slice_width, FWHM=FWHM, scale_amp=scale_amp, 
        #                       gaus_width=int(l[0]), flat_flux=flat_flux, bias_flux=bias_flux,
        #                       arc_lines=arc_lines, line_name=line_name, spec=spec,
        #                       full_spec=full_spec_line, source=source, noise=noise)
        
        # gen.generate_pix_disp()

        # if l[1] == 'ThAr':
        #     gen.make_flat_wavemap_bias(FITS=True)
        
        # gen.make_flat_wavemap_bias()
        # gen.make_arc(FITS=True)

        # if l[2] != None:
        #     gen.make_arc_from_full_spec(FITS=True)
        
        # print('all images for {} done'.format(key))
        # print('----')

    # -----------------------------------------------
    
    # gen.make_flat_wavemap_bias(FITS=True)
    # gen.make_arc(FITS=True)
    # gen.make_arc_from_full_spec(FITS=True)

    # print(gen.pix_disp)

    # print(gen.pix_disp)
    # print(gen.arcpix)







