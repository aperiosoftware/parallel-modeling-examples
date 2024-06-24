import numpy as np
from astropy.io import fits
# import matplotlib.pyplot as plt
import asdf
from tqdm.dask import TqdmCallback

import astropy.units as u
# from astropy.visualization import quantity_support
# quantity_support()

from sunraster.instr.spice import read_spice_l2_fits
from sospice.calibrate import spice_error

from astropy.modeling.fitting_parallel import parallel_fit_model_nd
from astropy.modeling import fitting

if __name__ == "__main__":

    filename = "solo_L2_spice-n-ras_20230415T120519_V02_184549780-000.fits.gz"
    window='N IV 765 ... Ne VIII 770 (Merged)'

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
    spice = spice[0]

    # We were given a model to fit, I assume it's "emprical" starting parameters
    with asdf.open("spice-model.asdf") as af:
        initial_model = af["spice-model"]

    interesting_looking_pixel = np.s_[:, 284, 116]

    # fig, ax = plt.subplots()
    wave = spice.axis_world_coords("em.wl")[0]
    # ax.set_title("Intitial guess")
    # ax.plot(wave.to(u.nm), spice.data[interesting_looking_pixel], "o-", label="Data")
    # ax.plot(wave, initial_model(wave.to_value(u.AA)), "--", label="Initial Guess")
    # plt.legend()


    with TqdmCallback(desc="fitting"):

        spice_model_fit = parallel_fit_model_nd(
            model=initial_model,
            fitter=fitting.TRFLSQFitter(),
            data=spice.data,
            fitting_axes=0,
            world = {0: wave},
            diagnostics="failed",
            diagnostics_path="diag",
            fitter_kwargs={"filter_non_finite": True},
            chunk_n_max=5000,
        )

    # plt.show()
