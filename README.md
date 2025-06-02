# wavecal_for_bluemuse
Codes for wavelength calibration with 4MOST for BlueMUSE simulation images

- data/

  - lines_for_img/

    - `*.csv`
      (list of wavelength and normalized amplitude for each arc line)
  - `BMUS_IFU-0236-1.53_disp.dat`
    - BlueMUSE dispersion file
  - `parameters_input_new.ascii`
    - input parameters file 

- func/

  - `extra_spectra_from_raw.py`
    - script to extract spectra from the raw 4MOST spectra
  - `fabry_perot_calibration.py`
    - script to do wavelength calibration for Fabry-Perot
  - `generate_line_list_and_table.py`
    - script to generate line list to generate simulation images and FITS table
    - input: line_lists
    - output: output/file_from_py
              data/lines_for_img
  - `generate_simulation_images.py`
    - script to simulate all calibration images
    - input: data
    - output: output/file_from_py

- line_lists/

  - compiled_master/
    - `master_list_*_.csv`
       - line list of each arc lines from NIST and 4MOST
  - gaussian_peaks/
    - `peaks_*_*_.csv`
       - gaussian peaks for each arc lines from 4MOST for different filters
  - NIST/
    - `*_lines.txt`
      - line lists from NIST

  - `master_peaks_blue.csv`
    - all gaussian peaks and identified wavelength from all arc lines in blue filter
  - `master_peaks_fp.csv`
    - all identified wavelength peaks from `fabry_perot_calibration.py`
  - `master_peaks_green.csv`
    - all gaussian peaks and identified wavelength from all arc lines in green filter

- notebooks_on_list/
  - the notebooks on this folder is not fully documented. it cannot be run with the data that
    is provided here, but it gives an idea what I did to identify and rearrange the lists if
    the situation requires.

  - `01_line_identify.ipynb`
    - notebook to identify line
  - `02_rearranging_line_lists.ipynb`
    - notebook to rearrange line list

- output/

  - files_from_py/

    - `*img.fits`
      - output from `generate_simulation_images.py
        - `arc*.fits`
          - simulated arc lines
        - `bias_img.fits`
          - simulated bias image
        - `flat_img.fits`
          - simulated flat image
        - `wavemap_img.fits`
          - simulated wavemap

    - `*line_tab_*.fits`
      - output from `generate_line_list_and_table.py` 

- spectra/
  - `spectra_*.csv`
    - spectra of all arc lines for different filters 