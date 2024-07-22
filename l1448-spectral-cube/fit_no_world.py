from astropy.io import fits

import matplotlib.pyplot as plt

from astropy.modeling.models import Gaussian1D
from astropy.modeling.fitting import LMLSQFitter
from astropy.modeling.fitting_parallel import parallel_fit_dask

if __name__ == "__main__":

    data = fits.getdata("l1448_13co.fits")

    g_init = Gaussian1D(mean=25, stddev=10, amplitude=1)
    g_init.mean.bounds = (0, 53)

    g_fit = parallel_fit_dask(
        model=g_init,
        fitter=LMLSQFitter(),
        data=data,
        fitting_axes=0,
    )

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(g_fit.amplitude.value, vmin=0, vmax=5, origin="lower")
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(g_fit.mean.value, vmin=0, vmax=50, origin="lower")
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(g_fit.stddev.value, vmin=0, vmax=20, origin="lower")
    fig.savefig("results_no_world.png", bbox_inches="tight")
