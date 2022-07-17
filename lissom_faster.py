import matplotlib

# matplotlib.use('Agg')
from rate_model import *
from projections import *
import imagen
import imagen.random
from imagen.transferfn import DivisiveNormalizeL1
import numpy
import pylab
import numbergen
from analysis import *
from visualization import *
from analysis import *
import time
import sys

# parameters
# aff_ratio, inh_ratio, lr, threshold


# Sheets
retina = InputSheet("Retina", 2.4, 25, None)
lgn_on = NoTimeConstantSheet("LGN_ON", 1.6, 25, None)
lgn_off = NoTimeConstantSheet("LGN_OFF", 1.6, 25, None)
V1 = HomeostaticSheet(
    "V1", 1.0, 74, 0.002, init_threshold=0.04, mu=0.002, alpha=0.1, smoothing=0.91
)


# Projections
# DoG weights for the LGN
centerg = imagen.Gaussian(
    size=0.07385, aspect_ratio=1.0, output_fns=[DivisiveNormalizeL1()]
)
surroundg = imagen.Gaussian(
    size=0.29540, aspect_ratio=1.0, output_fns=[DivisiveNormalizeL1()]
)
on_weights = imagen.Composite(generators=[centerg, surroundg], operator=numpy.subtract)
off_weights = imagen.Composite(generators=[surroundg, centerg], operator=numpy.subtract)


retina_to_lgn_on = FastConnectionFieldProjection(
    "RetinaToLgnOn", retina, lgn_on, 1.0, 0.001, 0.375, on_weights
)
retina_to_lgn_off = FastConnectionFieldProjection(
    "RetinaToLgnOff", retina, lgn_off, 1.0, 0.001, 0.375, off_weights
)

lgn_on_to_V1 = FastConnectionFieldProjection(
    "LGNOnToV1",
    lgn_on,
    V1,
    0.5,
    0.001,
    0.27083,
    imagen.random.GaussianCloud(gaussian_size=2 * 0.27083),
)
lgn_off_to_V1 = FastConnectionFieldProjection(
    "LGNOffToV1",
    lgn_off,
    V1,
    0.5,
    0.001,
    0.27083,
    imagen.random.GaussianCloud(gaussian_size=2 * 0.27083),
)

center_lat = imagen.Gaussian(
    size=0.05,
    aspect_ratio=1.0,
    output_fns=[DivisiveNormalizeL1()],
    xdensity=V1.density + 1,
    ydensity=V1.density + 1,
    bounds=BoundingBox(radius=V1.size / 2.0),
)()[14:-14, 14:-14]
surround_lat = imagen.Gaussian(
    size=0.05 * numpy.sqrt(2),
    aspect_ratio=1.0,
    output_fns=[DivisiveNormalizeL1()],
    xdensity=V1.density + 1,
    ydensity=V1.density + 1,
    bounds=BoundingBox(radius=V1.size / 2.0),
)()[14:-14, 14:-14]

V1_lat_exc = ConvolutionalProjection("LateralExc", V1, V1, 0.5 * 7.0, 0.001, center_lat)
V1_lat_inh = ConvolutionalProjection(
    "LateralInh", V1, V1, -0.5 * 7.0 * 0.84, 0.001, surround_lat
)

# Model initialization and execution
lissom = Model([retina, lgn_on, lgn_off, V1], 0.001)

g1 = imagen.Gaussian(
    xdensity=retina.unit_diameter,
    ydensity=retina.unit_diameter,
    x=numbergen.UniformRandom(lbound=-0.3, ubound=0.3, seed=342),
    y=numbergen.UniformRandom(lbound=-0.3, ubound=0.3, seed=343),
    orientation=numbergen.UniformRandom(lbound=-numpy.pi, ubound=numpy.pi, seed=333),
    size=0.7 * 0.048388,
    aspect_ratio=1.2 * 4.66667,
    scale=1.0,
)

g2 = imagen.Gaussian(
    xdensity=retina.unit_diameter,
    ydensity=retina.unit_diameter,
    x=numbergen.UniformRandom(lbound=-0.3, ubound=0.3, seed=312),
    y=numbergen.UniformRandom(lbound=-0.3, ubound=0.3, seed=313),
    orientation=numbergen.UniformRandom(lbound=-numpy.pi, ubound=numpy.pi, seed=322),
    size=0.7 * 0.048388,
    aspect_ratio=1.2 * 4.66667,
    scale=1.0,
)
print(g1)
print(g2)
m = numpy.maximum(g1(), g2())
print(m)

run_for = 10000
t = time.time()
for i in range(run_for):
    print(g1)
    print(g2)
    retina.set_activity(numpy.maximum(g1(), g2()))
    lissom.run(0.05)

    lgn_on_to_V1.apply_hebbian_learning_step(0.0003)
    lgn_off_to_V1.apply_hebbian_learning_step(0.0003)
    V1.apply_homeostatic_thresh()
    print(i, ":", "Expected time to run: ", ((time.time() - t) / (i + 1)) * (
        run_for - i
    ), "s")

    if i == run_for - 1:
        pylab.figure()
        display_model_state(lissom, filename="activity.png")
    lissom.reset()

if False:
    import pickle

    f = open("lissom_long.pickle", "wb")
    pickle.dump(lissom, f)
    f.close()

pylab.figure()
plot_projection(lgn_on_to_V1, filename="onProjection.png", downsample=0.5)

pylab.figure()
plot_projection(lgn_off_to_V1, filename="offProjection.png", downsample=0.5)

pylab.figure()
fullfield_sine_grating_orientation_tuning_protocol(
    lissom,
    retina,
    sheets=[V1],
    num_orientation=8,
    num_phase=10,
    duration=0.02,
    frequency=4.0,
    filename="freq=4",
    reset=True,
    plot=True,
    load=False,
)
pylab.figure()
fullfield_sine_grating_orientation_tuning_protocol(
    lissom,
    retina,
    sheets=[V1],
    num_orientation=8,
    num_phase=10,
    duration=0.02,
    frequency=5.0,
    filename="freq=5",
    reset=True,
    plot=True,
    load=False,
)

pylab.show()

if True:
   pylab.figure()
   plot_projection(lgn_on_to_V1)
   pylab.figure()
   plot_projection(lgn_off_to_V1)
   pylab.show()