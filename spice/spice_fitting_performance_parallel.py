import sys
import time

import numpy as np
from astropy.io import fits
import asdf

import astropy.units as u

from sunraster.instr.spice import read_spice_l2_fits
from sospice.calibrate import spice_error

from astropy.modeling import fitting
from astropy.modeling.fitting import parallel_fit_dask

if __name__ == "__main__":

    filename = "solo_L2_spice-n-ras_20230415T120519_V02_184549780-000.fits.gz"
    window = 'N IV 765 ... Ne VIII 770 (Merged)'

    spice = read_spice_l2_fits(filename)
    spice = spice[window]
    spice.mask |= (spice.data <= 0)
    include = ~np.all(spice.mask, axis=(0, 2, 3))

    hdulist = fits.open(filename)
    for h in hdulist:
        if h.name == window:
            hdu = h
            break
    av_cojstant_noise_level, sigmadict = spice_error(hdu)
    sigma = sigmadict["Total"].value
    spice.mask = spice.mask | np.isnan(sigma) | (sigma <= 0)
    # drop leading length 1 dimension
    spice = spice[0, :, :, 10:110]

    # We were given a model to fit, I assume it's "emprical" starting parameters
    with asdf.open("spice-model.asdf") as af:
        initial_model = af["spice-model"]

    wave = spice.axis_world_coords("em.wl")[0].to_value(u.AA)
    fitter = fitting.TRFLSQFitter()

    n_spectra = np.prod(spice.data.shape[1:])
    print(f"Fitting {n_spectra} spectra in parallel")

    start = time.time()
    spice_model_fit = parallel_fit_dask(
        data=spice.data,
        world=(wave,),
        mask=spice.mask,
        model=initial_model,
        fitter=fitting.TRFLSQFitter(),
        fitting_axes=0,
        fitter_kwargs={"filter_non_finite": True}, # Filter out non-finite values,
    )
    end = time.time()

    t_elasped = end - start
    print(f"Fitting {n_spectra} spectra in parallel took {t_elasped}s")
    print(f"Or {t_elasped / n_spectra}s per spectra")
