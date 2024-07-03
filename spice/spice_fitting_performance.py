import time

import numpy as np
from astropy.io import fits
import asdf

import astropy.units as u

from sunraster.instr.spice import read_spice_l2_fits
from sospice.calibrate import spice_error

from astropy.modeling import fitting


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
spice = spice[0, :, :, 10]
spice

# We were given a model to fit, I assume it's "emprical" starting parameters
with asdf.open("spice-model.asdf") as af:
    initial_model = af["spice-model"]

wave = spice.axis_world_coords("em.wl")[0].to_value(u.AA)
fitter = fitting.TRFLSQFitter()
fits = []

start = time.time()
for slit_pos in range(spice.data.shape[-1]):
    data = spice[:, slit_pos].data
    if np.isnan(data).all():
        print(f"All is NaN Skipping {slit_pos=}")
        continue
    try:
        fits.append(fitter(initial_model, wave, data, filter_non_finite=True))
    except Exception:
        print(f"Failed {slit_pos=}")

end = time.time()

print(f"Fitting {spice.data.shape[-1]} spectra took {end - start}s")
