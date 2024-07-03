import dkist
import dkist.net
from sunpy.net import Fido, attrs as a

import astropy.modeling.models as m
import astropy.units as u

import matplotlib.pyplot as plt
from astropy.visualization import quantity_support

from astropy.modeling.fitting import TRFLSQFitter
import numpy as np

from pyinstrument import Profiler

# asdf_file = "/dkist/prod/pid_2_114/ALDLJ/VISP_L1_20231016T220247_ALDLJ.asdf"
asdf_file = (
    "/home/tom/Data/DKIST/prod/pid_2_114/ALDLJ/VISP_L1_20231016T220247_ALDLJ.asdf"
)

visp = dkist.load_dataset(asdf_file)

wave = visp[0, :, 1000].axis_world_coords("em.wl")[0]


line_1 = m.Lorentz1D(amplitude=-0.6, fwhm=0.1, x_0=854.3)
line_1_constrained = line_1.copy()
line_1_constrained.x_0.min = 854.25
line_1_constrained.x_0.max = 854.35

line_2 = m.Lorentz1D(amplitude=-0.25, fwhm=0.01, x_0=853.98)
line_2_constrained = line_2.copy()
line_2_constrained.x_0.min = 853.95
line_2_constrained.x_0.max = 854.00

line_3 = m.Lorentz1D(amplitude=-0.15, fwhm=0.01, x_0=854.08)
line_3_constrained = line_3.copy()
line_3_constrained.x_0.min = 854.05
line_3_constrained.x_0.max = 854.13

model_constrained = (
    m.Const1D(1) + line_1_constrained + line_2_constrained + line_3_constrained
)

model = m.Const1D(1) + line_1 + line_2 + line_3

x = wave.to_value("nm")
y = visp[0, :, 1000].data.compute()

n_iter = 100

profiler = Profiler()
profiler.start()

for iter in range(n_iter):
    fit = TRFLSQFitter()(model, x, y)

profiler.stop()

profiler.write_html("profiling_unconstrained.html", show_all=True)

profiler = Profiler()
profiler.start()

for iter in range(n_iter):
    fit_constrained = TRFLSQFitter()(model_constrained, x, y)

profiler.stop()

profiler.write_html("profiling_constrained.html", show_all=True)

fig, ax = plt.subplots()
ax.set_title("VISP")
ax.plot(x, y, label="slit average")
ax.plot(x, model(x), label="initial guess")
ax.plot(x, fit(x), label="fit (unconstrained)")
ax.plot(wave, fit_constrained(wave), label="fit (constrained)")
plt.legend()
plt.savefig("simple.png")
